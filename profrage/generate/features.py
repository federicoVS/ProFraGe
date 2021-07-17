import numpy as np

from generate.utils import circular_mean
from utils.structure import AA_TO_INT, structure_length
from utils.stride import SS_CODE_TO_INT, single_stride
from utils.io import from_pdb

class ProteinFeature:

    def __init__(self, protein, pdb_dir, stride_dir):
        self.protein = protein
        self._features = None
        self._compute_features(pdb_dir, stride_dir)

    def _compute_features(self, pdb_dir, stride_dir):
        pdb_id = self.protein.get_id()
        stride_entries = single_stride(stride_dir, pdb_dir + pdb_id + '.pdb')
        self._features = np.zeros(shape=(len(stride_entries), ResidueFeature.get_n_features()))
        for i in range(len(stride_entries)):
            self._features[i,:] = ResidueFeature(stride_entries[i]).get_features()

    def flatten(self):
        return self._features.reshape(self._features.shape[0]*self._features.shape[1],)

    def get_features(self):
        return self._features

class ResidueFeature:

    def __init__(self, residue, stride_entry):
        self._features = [0 for _ in range(ResidueFeature.get_n_features())]
        self._features[0] = float(AA_TO_INT[stride_entry[0]])
        for i in range(1, len(stride_entry)):
            self._features[i] = stride_entry[i]
        self._features[1] = float(SS_CODE_TO_INT[stride_entry[1]]+1) # add one as to not have zeros, which are used for padding
        self._features[2] = circular_mean([self._features[2]])
        self._features[3] = circular_mean([self._features[3]])
        ca_atom = residue['CA']
        coords = ca_atom.get_coord()
        self._features[5] = coords[0]
        self._features[6] = coords[1]
        self._features[7] = coords[2]
        self._features[8] = ca_atom.get_occupancy()
        self._features[9] = ca_atom.get_bfactor()
        self._features = np.array(self._features)

    @staticmethod
    def get_n_features():
        return 10

    def get_features(self):
        return self._features

class SequenceFeature:

    def __init__(self, protein):
        self.protein = protein
        self._features = np.zeros(shape=(structure_length(protein),))
        self._compute_features()

    def _compute_features(self):
        residues = []
        for residue in self.protein.get_residues():
            r_id = residue.get_id()
            residues.append((residue.get_resname(), r_id[1]))
        residues = sorted(residues, key=lambda x: x[1])
        for i in range(len(residues)):
            self._features[i:,] = float(AA_TO_INT[residues[i][0]])

    def get_features(self):
        return self._features

class StructuralFeature:

    def __init__(self, protein, pdb_dir, stride_dir, amplitude=1000):
        self.protein = protein
        self._features = None
        self._compute_features(pdb_dir, stride_dir, amplitude)

    def _compute_features(self, pdb_dir, stride_dir, amplitude):
        pdb_id = self.protein.get_id()
        stride_entries = single_stride(stride_dir, pdb_dir + pdb_id + '.pdb')
        refined_entries = []
        for se in stride_entries:
            _, code, phi, psi, area = se
            refined_entries.append((code, phi, psi, area))
        self._features = np.zeros(shape=(len(refined_entries), 4))
        for i in range(len(refined_entries)):
            code, phi, psi, area = refined_entries[i]
            f = np.array([float(SS_CODE_TO_INT[code]), phi, psi, area])
            self._features[i,:] = (amplitude*np.cos(f)) + amplitude

    def flatten(self):
        return self._features.reshape(self._features.shape[0]*self._features.shape[1],)

    def get_features(self):
        return self._features

class GraphFeature:

    def __init__(self, protein, pdb_dir, stride_dir, dist_thr=12, mode='dense', weighted=False):
        self.protein = protein
        self.pdb_dir = pdb_dir
        self.stride_dir = stride_dir
        self.dist_thr = dist_thr
        self.mode = mode
        self.weighted = weighted
        self._weights_cache = {}


    def _get_backbone(self):
        if self.mode == 'sparse':
            return None
        n = structure_length(self.protein)
        bb = np.zeros(shape=(n,n))
        residues = []
        for residue in self.protein.get_residues():
            residues.append(residue)
        for i in range(n-1):
            for j in range(i+1, n):
                if abs(i-j) == 1:
                    bb[i,j] = bb[j,i] = 1
        return bb

    def _get_adjacency(self):
        n = structure_length(self.protein)
        if self.mode == 'dense':
            adj = np.zeros(shape=(n,n))
        elif self.mode == 'sparse':
            adj = []
        residues = []
        for residue in self.protein.get_residues():
            residues.append(residue)
        for i in range(n-1):
            for j in range(i+1, n):
                if 'CA' in residues[i] and 'CA' in residues[j]:
                    ca_i, ca_j = residues[i]['CA'], residues[j]['CA']
                    ca_dist = np.linalg.norm(ca_i.get_vector()-ca_j.get_vector())
                    if ca_dist < self.dist_thr:
                        if self.mode == 'dense':
                            if self.weighted:
                                adj[i,j] = adj[j,i] = 1/ca_dist
                            else:
                                adj[i,j] = adj[j,i] = 1
                        elif self.mode == 'sparse':
                            adj.append([i,j])
                            adj.append([j,i])
                        self._weights_cache[(i,j)] = self._weights_cache[(j,i)] = ca_dist
        if self.mode == 'sparse':
            adj = np.array(adj)
        return adj

    def _get_edge_feature(self):
        n = structure_length(self.protein)
        if self.mode == 'dense':
            edge = np.zeros(shape=(n,n,2))
        elif self.mode == 'sparse':
            edge = []
        for i in range(n-1):
            for j in range(i+1, n):
                if (i,j) in self._weights_cache:
                    w = self._weights_cache[(i,j)]
                    bb = 1 if abs(i-j) == 1 else 0 # 1 is backbone, 0 is contact
                    if self.mode == 'dense':
                        edge[i,j] = edge[j,i] = np.array([w, bb])
                    elif self.mode == 'sparse':
                        edge.append([w, bb])
                        edge.append([w, bb])
        if self.mode == 'sparse':
            edge = np.array(edge)
        return edge

    def _get_node_feature(self):
        pdb_id = self.protein.get_id()
        stride_entries = single_stride(self.stride_dir, self.pdb_dir + pdb_id + '.pdb')
        if stride_entries is None:
            return None
        structure = from_pdb(pdb_id, self.pdb_dir + pdb_id + '.pdb', quiet=True)
        residues = []
        for residue in structure.get_residues():
            residues.append(residue)
        x = np.zeros(shape=(len(stride_entries),ResidueFeature.get_n_features()))
        for i in range(len(stride_entries)):
            x[i,:] = ResidueFeature(residues[i], stride_entries[i]).get_features()
        return x

    def get_features(self):
        bb, adj, edge, x = self._get_backbone(), self._get_adjacency(), self._get_edge_feature(), self._get_node_feature()
        return bb, adj, edge, x