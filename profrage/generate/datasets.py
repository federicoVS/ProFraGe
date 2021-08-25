import copy
from itertools import compress

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

    def __init__(self, root, id, split, proteins, pdb_dir, stride_dir, dist_thr=12, max_size=30, mode='sparse', load=False):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The root directory where the data is stored.
        id : int
            The ID of the dataset.
        split : str
            To which split in train/val/test the dataset belongs to. Valid options are ['train','val','test'].
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
        mode : str, optional
            How the data should be. Valid options are ['sparse', 'dense']. The default is 'sparse'.
        load : bool, optional
            Whether the data should be computed or loaded (if it has already been computed). The default is False.
        """
        self.root = root
        self.id = id
        self.split = split
        self.proteins = proteins
        self.length = 0
        self._mode = mode
        self._load = load
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
            adj, w_adj, edge, x = GraphFeature(self.proteins[i], pdb_dir, stride_dir, dist_thr=dist_thr, mode=self._mode).get_features()
            if x is None:
                continue
            if x.shape[0] == 0:
                continue
            # Pad
            delta = max_size - x.shape[0]
            if self._mode == 'dense':
                adj = np.pad(adj, (0,delta))
                w_adj = np.pad(w_adj, (0,delta))
                edge = np.pad(edge, ((0,delta),(0,delta),(0,0)))
                x = np.pad(x, ((0,delta),(0,0)))
            if self._mode == 'sparse':
                adj = torch.from_numpy(adj).type(torch.LongTensor)
                w_adj = torch.from_numpy(w_adj).type(torch.FloatTensor)
                edge = torch.from_numpy(edge).type(torch.FloatTensor)
                x = torch.from_numpy(x).type(torch.FloatTensor)
            # Node mask
            nm = torch.ones(max_size).type(torch.BoolTensor)
            nm[-delta:] = False
            # Assign features
            if self._mode == 'dense':
                dense_data_list.append({'x': torch.tensor(x).type(torch.FloatTensor),
                                        'adj': torch.tensor(adj).type(torch.LongTensor),
                                        'w_adj': torch.tensor(w_adj).type(torch.FloatTensor),
                                        'edge': torch.tensor(edge).type(torch.FloatTensor),
                                        'mask': nm,
                                        'len': x.shape[0],
                                        'pdb_id': self.proteins[i].get_id()})
            elif self._mode == 'sparse':
                gdata = GData(x=x, edge_index=adj.t().contiguous(), edge_attr=edge)
                gdata.w_adj = w_adj
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

        Note that the data is only saved if the data was computed from scratch.

        Returns
        -------
        None
        """
        if not self._load:
            file_name = 'graph_' + self._mode + '_' + str(self.id) + '_' + self.split + '.pt'
            torch.save(self._data, self.root + file_name)

    def load(self):
        """
        Load the data.

        Returns
        -------
        None
        """
        file_name = 'graph_' + self._mode + '_' + str(self.id) + '_' + self.split + '.pt'
        self._data = torch.load(self.root + file_name)
        self.length = len(self._data)

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

    def __init__(self, root, id, split, proteins, pdb_dir, stride_dir, dist_thr=12, max_size=30, load=False):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The root directory where the data is stored.
        id : int
            The ID of the dataset.
        split : str
            To which split in train/val/test the dataset belongs to. Valid options are ['train','val','test'].
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
        split : str, optional
            To which split in train/val/test the dataset belongs to. The default is 'train'.
        load : bool, optional
            Whether the data should be computed or loaded (if it has already been computed). The default is False.
        """
        super(RNNDataset_Feat, self).__init__()
        self.root = root
        self.id = id
        self.split = split
        self.proteins = proteins
        self._max_size = max_size
        self._load = load
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

    def _encode_data(self, w_adj, x):
        n = w_adj.shape[0]
        s_pi = []
        for i in range(n):
            s_i = []
            for k in range(x.shape[1]):
                s_i.append(x[i,k])
            for j in range(i):
                if w_adj[i,j] > 0:
                    s_i.append(w_adj[i,j])
                else:
                    s_i.append(0)
            s_i = torch.tensor(s_i)
            s_pi.append(s_i)
        return s_pi

    def _w_adj_mapping(self, w_adj):
        n = w_adj.shape[0]
        e_pi = []
        for i in range(n):
            e_i = []
            for j in range(i):
                if w_adj[i,j] > 0:
                    e_i.append(w_adj[i,j])
                else:
                    e_i.append(0)
            e_i = torch.tensor(e_i)
            e_pi.append(e_i)
        return e_pi

    def _compute_features(self, pdb_dir, stride_dir, dist_thr):
        data = []
        for i in range(len(self.proteins)):
            # Get features
            _, w_adj, _, x = GraphFeature(self.proteins[i], pdb_dir, stride_dir, dist_thr=dist_thr, mode='dense').get_features()
            if x is None:
                continue
            s_pi = self._encode_data(w_adj, x)
            e_pi = self._w_adj_mapping(w_adj)
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

    def sample(self):
        """
        Sample a part of data and return a new dataset based on said data.

        Returns
        -------
        dataset_copy : RNNDataset_Feat
            A new RNNDataset_Feat based on sampled data from self.
        """
        idx = np.random.choice(a=[True,False], size=len(self._data)).tolist()
        sampled = list(compress(self._data, idx))
        dataset_copy = copy.deepcopy(self)
        dataset_copy._data = sampled
        return dataset_copy

    def save(self):
        """
        Save the data.

        Note that the data is only saved if the data was computed from scratch.

        Returns
        -------
        None
        """
        if not self._load:
            file_name = 'rnn_' + str(self.id) + '_' + self.split + '.pt'
            torch.save(self._data, self.root + file_name)

    def load(self):
        """
        Load the data.

        Returns
        -------
        None
        """
        file_name = 'rnn_' + str(self.id) + '_' + self.split + '.pt'
        self._data = torch.load(self.root + file_name)