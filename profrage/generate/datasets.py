import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data as GData
from torch.nn.utils.rnn import pad_sequence

from generate.features import ProteinFeature, SequenceFeature, StructuralFeature, GraphFeature

class ProteinDataSet(Dataset):

    def __init__(self, proteins, pdb_dir, stride_dir, max_size=30):
        self.proteins = proteins
        self.data = torch.from_numpy(self._compute_features(pdb_dir, stride_dir, max_size)).type(torch.FloatTensor)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def get_n_features(self):
        return self.data.shape[1]

    def _compute_features(self, pdb_dir, stride_dir, max_size):
        features = []
        for protein in self.proteins:
            features.append(ProteinFeature(protein, pdb_dir, stride_dir).flatten())
        for i in range(len(features)):
            if features[i].shape[0] < max_size:
                delta = max_size - features[i].shape[0]
                features[i] = np.pad(features[i], (0,delta), 'constant')
        features = np.array(features)
        return features

class StructSeqDataset(Dataset):

    def __init__(self, proteins, pdb_dir, stride_dir, amplitude=1000):
        self.proteins = proteins
        self.x = torch.from_numpy(self._compute_structural_features(pdb_dir, stride_dir, amplitude)).type(torch.LongTensor)
        self.y = torch.from_numpy(self._compute_sequence_features()).type(torch.LongTensor)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

    def _compute_sequence_features(self):
        features = []
        max_size = 0
        for protein in self.proteins:
            fs = SequenceFeature(protein).get_features()
            if fs.shape[0] > max_size:
                max_size = fs.shape[0]
            features.append(fs)
        for i in range(len(features)):
            if features[i].shape[0] < max_size:
                delta = max_size - features[i].shape[0]
                features[i] = np.pad(features[i], (0,delta), 'constant')
        features = np.array(features)
        return features

    def _compute_structural_features(self, pdb_dir, stride_dir, amplitude):
        features = []
        max_size = 0
        for protein in self.proteins:
            fs = StructuralFeature(protein, pdb_dir, stride_dir, amplitude=amplitude).flatten()
            if fs.shape[0] > max_size:
                max_size = fs.shape[0]
            features.append(fs)
        for i in range(len(features)):
            if features[i].shape[0] < max_size:
                delta = max_size - features[i].shape[0]
                features[i] = np.pad(features[i], (0,delta), 'constant')
        features = np.array(features)
        return features

class GraphDataset(Dataset):

    def __init__(self, proteins, pdb_dir, stride_dir,
                 dist_thr=12, max_size=30, x_type=torch.FloatTensor, bb_type=torch.LongTensor, adj_type=torch.LongTensor, edge_type=torch.FloatTensor,
                 mode='sparse', weighted=False):
        self.proteins = proteins
        self.length = 0
        self._x_type = x_type
        self._bb_type = bb_type
        self._adj_type = adj_type
        self._edge_type = edge_type
        self._mode = mode
        self._weighted = weighted
        self._data = self._compute_features(pdb_dir, stride_dir, dist_thr, max_size)

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return self.length

    def _compute_features(self, pdb_dir, stride_dir, dist_thr, max_size):
        dense_data_list = []
        sparse_data_list = []
        for i in range(len(self.proteins)):
            # Get features
            bb, adj, edge, x = GraphFeature(self.proteins[i], pdb_dir, stride_dir, dist_thr=dist_thr, mode=self._mode, weighted=self._weighted).get_features()
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
                x = np.pad(x, ((0,delta),(0,0)))
            if self._mode == 'sparse':
                bb = torch.from_numpy(adj).type(self._bb_type)
                adj = torch.from_numpy(adj).type(self._adj_type)
                edge = torch.from_numpy(edge).type(self._edge_type)
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
                                        'mask': nm,
                                        'len': x.shape[0]})
            elif self._mode == 'sparse':
                gdata = GData(x=x, edge_index=adj.t().contiguous(), edge_attr=edge)
                gdata.node_mask = nm
                gdata.batch_len = x.shape[0]
                sparse_data_list.append(gdata)
        if self._mode == 'dense':
            self.length = len(dense_data_list)
            return dense_data_list
        elif self._mode == 'sparse':
            self.length = len(sparse_data_list)
            return sparse_data_list

    def get_sparse_data(self):
        return self._data

class RNNDataset_Seq(Dataset):

    def __init__(self, proteins, pdb_dir, stride_dir, dist_thr=12, max_size=30):
        super(RNNDataset_Seq, self).__init__()
        self.proteins = proteins
        self.max_size = max_size
        self._data = self._compute_features(pdb_dir, stride_dir, dist_thr)

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def _encode_adj(self, adj):
        n = adj.shape[0]
        s_pi = []
        for i in range(n):
            s_i = []
            for j in range(i):
                s_i.append(adj[i,j])
            s_i = torch.LongTensor(s_i)
            s_pi.append(s_i)
        return s_pi

    def _compute_features(self, pdb_dir, stride_dir, dist_thr):
        data = []
        for i in range(len(self.proteins)):
            # Get features
            adj, _, x = GraphFeature(self.proteins[i], pdb_dir, stride_dir, dist_thr=dist_thr, mode='dense', weighted=False).get_features()
            if x is None:
                continue
            s_pi = self._encode_adj(adj)
            n = len(s_pi)
            s_pi = pad_sequence(s_pi[1:n], batch_first=True)
            x, y = torch.zeros(self.max_size,n-1), torch.zeros(self.max_size,n-1)
            x[0,:] = 1
            x[1:n,:] = s_pi
            y[0:n-1,:] = s_pi
            delta = self.max_size - (n - 1)
            x, y = F.pad(x, (0,delta)), F.pad(y, (0,delta))
            data.append({'x': x, 'y': y, 'len': n})
        return data

class RNNDataset_Feat(Dataset):

    def __init__(self, proteins, pdb_dir, stride_dir, dist_thr=12, aa_dim=20, max_size=30, ignore_idx=-100):
        super(RNNDataset_Feat, self).__init__()
        self.proteins = proteins
        self.aa_dim = aa_dim
        self.max_size = max_size
        self.ignore_idx = ignore_idx
        self._data = self._compute_features(pdb_dir, stride_dir, dist_thr)

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
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
            xt = torch.zeros(self.max_size+1,n+x.shape[1]-1) # minus 1 b/c the first node has no edge
            y_edge = torch.zeros(self.max_size,n-1) # minus 1 b/c the first node has no edge
            xt[0,:] = 1
            xt[1:n+1,:] = s_pi
            y_edge[0:n-1,:] = e_pi[1:n]
            delta_x = (self.max_size + x.shape[1] - 1) - (n + x.shape[1] - 1)
            delta_y_edge = self.max_size - (n - 1)
            delta_y_feat = self.max_size - n
            xt = F.pad(xt, (0,delta_x))
            y_edge = F.pad(y_edge, (0,delta_y_edge))
            y_feat = F.pad(torch.tensor(x), (0,0,0,delta_y_feat)) # plus 1 to make them even with xt
            data.append({'x': xt, 'y_edge': y_edge, 'y_feat': y_feat, 'len': n})
        return data