import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data as GData
from torch.nn.utils.rnn import pad_sequence

from generate.features import GraphFeature

class GraphDataset(Dataset):
    """
    Representation of a protein graph.

    The data contains describes nodes, connectivity (adjacency), edges, and the backbone structure (special case of adjacency).

    Attributes
    ----------
    root : str
        The root directory for the dataset.
    proteins : list of Bio.PDB.Structure
        The proteins from which to compute the features.
    length : int
        The length of the dataset in terms of valid proteins.
    """

    def __init__(self, root, proteins, pdb_dir, stride_dir,
                 dist_thr=12, max_size=30, x_type=torch.FloatTensor, bb_type=torch.LongTensor, adj_type=torch.LongTensor, edge_type=torch.FloatTensor,
                 mode='sparse', probabilistic=False, load=False):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The root directory where the data is stored.
        proteins : list of Bio.PDB.Structure
            The proteins from which to compute the dataset.
        pdb_dir : str
            The directory holding the PDB files.
        stride_dir : str
            The directory holding the Stride tool.
        dist_thr : float, optional
            The distance threshold below which two residues are interacting. The default is 12.
        max_size : int, optional
            The maximum number of residues in a fragment. The default is 30.
        x_type : torch.type, optional
            The type of the node features. The default is torch.FloatTensor.
        bb_type : torch.type, optional
            The type of the backbone features. The default is torch.LongTensor.
        adj_type : torch.type, optional
            The type of the adjacency matrix. The default is torch.LongTensor.
        edge_type : torch.type, optional
            The type of the edge features. The default is torch.FloatTensor.
        mode : str, optional
            How the data should be. Valid options are ['sparse', 'dense']. The default is 'sparse'.
        probabilistic : bool, optional
            Whether the adjacency matrix should contain 1s on the diagonal, indicating the existence of a node. The default is False.
        load : bool, optional
            Whether the data should be computed or loaded (if it has already been computed). The default is False.
        """
        self.root = root
        self.proteins = proteins
        self.length = 0
        self._x_type = x_type
        self._bb_type = bb_type
        self._adj_type = adj_type
        self._edge_type = edge_type
        self._mode = mode
        self._probabilistic = probabilistic
        if load:
            self.load()
        else:
            self._data = self._compute_features(pdb_dir, stride_dir, dist_thr, max_size)

    def __getitem__(self, index):
        """
        Return the selected entry.

        Parameters
        ----------
        index : int
            The index of the item to select.

        Returns
        -------
        list of Any
            The data.
        """
        return self._data[index]

    def __len__(self):
        """
        Return the size of the data.

        Returns
        -------
        int
            The size of the data.
        """
        return self.length

    def _compute_features(self, pdb_dir, stride_dir, dist_thr, max_size):
        dense_data_list = []
        sparse_data_list = []
        for i in range(len(self.proteins)):
            # Get features
            bb, adj, edge, weight, x = GraphFeature(self.proteins[i], pdb_dir, stride_dir, dist_thr=dist_thr, mode=self._mode).get_features()
            if x is None:
                continue
            if x.shape[0] == 0:
                continue
            # Pad
            delta = max_size - x.shape[0]
            if self._mode == 'dense':
                bb = np.pad(bb, (0,delta))
                adj = np.pad(adj, (0,delta))
                edge = np.pad(edge, ((0,delta),(0,delta),(0,0)))
                weight = np.pad(weight, (0,delta))
                x = np.pad(x, ((0,delta),(0,0)))
                if self._probabilistic:
                    adj = adj + torch.eye(adj.shape[0])
            if self._mode == 'sparse':
                bb = torch.from_numpy(adj).type(self._bb_type)
                adj = torch.from_numpy(adj).type(self._adj_type)
                edge = torch.from_numpy(edge).type(self._edge_type)
                weight = torch.from_numpy(weight).type(torch.FloatTensor)
                x = torch.from_numpy(x).type(self._x_type)
            # Node mask
            nm = torch.ones(max_size).type(torch.BoolTensor)
            nm[-delta:] = False
            # Assign features
            if self._mode == 'dense':
                dense_data_list.append({'x': torch.tensor(x).type(self._x_type),
                                        'bb': torch.tensor(bb).type(self._bb_type),
                                        'adj': torch.tensor(adj).type(self._adj_type),
                                        'edge': torch.tensor(edge).type(self._edge_type),
                                        'weight': torch.tensor(weight).type(torch.FloatTensor),
                                        'mask': nm,
                                        'len': x.shape[0],
                                        'pdb_id': self.proteins[i].get_id()})
            elif self._mode == 'sparse':
                gdata = GData(x=x, edge_index=adj.t().contiguous(), edge_attr=edge)
                gdata.weight = weight
                gdata.node_mask = nm
                gdata.x_len = x.shape[0]
                gdata.edge_len = adj.shape[0]
                gdata.pdb_id = self.proteins[i].get_id()
                sparse_data_list.append(gdata)
        if self._mode == 'dense':
            self.length = len(dense_data_list)
            return dense_data_list
        elif self._mode == 'sparse':
            self.length = len(sparse_data_list)
            return sparse_data_list

    def save(self):
        """
        Save the data.

        Returns
        -------
        None
        """
        torch.save(self._data, self.root + 'graph_' + self._mode + '.pt')

    def load(self):
        """
        Load the data.

        Returns
        -------
        None
        """
        self._data = torch.load(self.root + 'graph_' + self._mode + '.pt')

    def get_data(self):
        """
        Return the data.

        Returns
        -------
        list of Any
            The data.
        """
        return self._data

class RNNDataset_Feat(Dataset):
    """
    Representation of a protein graph as a sequence.

    Each node is represented as its own features and its edges to other nodes, fashioned as a lower triangular matrix.

    [x_1, ..., x_m]
    [x_1, ..., x_m, e_10]
    [x_1, ..., x_m, e_20, e_21]
    ...
    [x_1, ..., x_m, e_n0, e_n1, ..., e_nn]

    Attributes
    ----------
    root : str
        The root directory for the dataset.
    proteins : list of Bio.PDB.Structure
        The proteins from which to compute the features.
    """

    def __init__(self, root, proteins, pdb_dir, stride_dir, dist_thr=12, max_size=30, load=False):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The root directory where the data is stored.
        proteins : list of Bio.PDB.Structure
            The proteins from which to compute the dataset.
        pdb_dir : str
            The directory holding the PDB files.
        stride_dir : str
            The directory holding the Stride tool.
        dist_thr : float, optional
            The distance threshold below which two residues are interacting. The default is 12.
        max_size : int, optional
            The maximum number of residues in a fragment. The default is 30.
        load : bool, optional
            Whether the data should be computed or loaded (if it has already been computed). The default is False.
        """
        super(RNNDataset_Feat, self).__init__()
        self.root = root
        self.proteins = proteins
        self._max_size = max_size
        if load:
            self.load()
        else:
            self._data = self._compute_features(pdb_dir, stride_dir, dist_thr)

    def __getitem__(self, index):
        """
        Return the selected entry.

        Parameters
        ----------
        index : int
            The index of the item to select.

        Returns
        -------
        list of Any
            The data.
        """
        return self._data[index]

    def __len__(self):
        """
        Return the size of the data.

        Returns
        -------
        int
            The size of the data.
        """
        return len(self._data)

    def _encode_data(self, adj, edge, x):
        n = adj.shape[0]
        s_pi = []
        for i in range(n):
            s_i = []
            for k in range(x.shape[1]):
                s_i.append(x[i,k])
            for j in range(i):
                if adj[i,j] > 0:
                    s_i.append(edge[i,j,1]+1)
                else:
                    s_i.append(0)
            s_i = torch.tensor(s_i)
            s_pi.append(s_i)
        return s_pi

    def _edge_mapping(self, adj, edge):
        n = adj.shape[0]
        e_pi = []
        for i in range(n):
            e_i = []
            for j in range(i):
                if adj[i,j] > 0:
                    e_i.append(edge[i,j,1]+1)
                else:
                    e_i.append(0)
            e_i = torch.tensor(e_i)
            e_pi.append(e_i)
        return e_pi

    def _compute_features(self, pdb_dir, stride_dir, dist_thr):
        data = []
        for i in range(len(self.proteins)):
            # Get features
            adj, edge, x = GraphFeature(self.proteins[i], pdb_dir, stride_dir, dist_thr=dist_thr, mode='dense', weighted=False).get_features()
            if x is None:
                continue
            s_pi = self._encode_data(adj, edge, x)
            e_pi = self._edge_mapping(adj, edge)
            n = len(s_pi)
            s_pi = pad_sequence(s_pi, batch_first=True)
            e_pi = pad_sequence(e_pi, batch_first=True)
            xt = torch.zeros(self._max_size+1,n+x.shape[1]-1) # minus 1 b/c the first node has no edge
            y_edge = torch.zeros(self._max_size,n-1) # minus 1 b/c the first node has no edge
            xt[0,:] = 1
            xt[1:n+1,:] = s_pi
            y_edge[0:n-1,:] = e_pi[1:n]
            delta_x = (self._max_size + x.shape[1] - 1) - (n + x.shape[1] - 1)
            delta_y_edge = self._max_size - (n - 1)
            delta_y_feat = self._max_size - n
            xt = F.pad(xt, (0,delta_x))
            y_edge = F.pad(y_edge, (0,delta_y_edge))
            y_feat = F.pad(torch.tensor(x), (0,0,0,delta_y_feat)) # plus 1 to make them even with xt
            data.append({'x': xt, 'y_edge': y_edge, 'y_feat': y_feat, 'len': n})
        return data

    def save(self):
        """
        Save the data.

        Returns
        -------
        None
        """
        torch.save(self._data, self.root + 'rnn_' + self._mode + '.pt')

    def load(self):
        """
        Load the data.

        Returns
        -------
        None
        """
        self._data = torch.load(self.root + 'rnn_' + self._mode + '.pt')