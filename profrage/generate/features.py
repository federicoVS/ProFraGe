import numpy as np

from utils.structure import AA_TO_INT, structure_length
from utils.stride import SS_CODE_TO_INT, single_stride
from utils.io import from_pdb

class ProteinFeature:
    """
    Modelling of a single protein features.

    Each feature represents residues features.

    Attributes
    ----------
    protein : Bio.PDB.Structure
        The protein from which to extract the features.
    """

    def __init__(self, protein, pdb_dir, stride_dir):
        """
        Initialize the class.

        Parameters
        ----------
        protein : Bio.PDB.Structure
            The protein from which to extract the features.
        pdb_dir : str
            The directory holding the PDB files.
        stride_dir : str
            The directory holding the Stride tool.
        """
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
        """
        Flatten the data

        Returns
        -------
        numpy.ndarray
            The flattened data.
        """
        return self._features.reshape(self._features.shape[0]*self._features.shape[1],)

    def get_features(self):
        """
        Return the features.

        Returns
        -------
        numpy.ndarray
            The features.
        """
        return self._features

class ResidueFeature:
    """
    Modelling of a single residue features.

    A residue is described by 10 features: amino acid code (e.g. ALA), secondary structure ID (from Stride), Psi and Phi angle (converted to circular
    mean), solubility area, atomic coordinates, occupancy, and temperature factor.
    """

    def __init__(self, residue, stride_entry):
        self._features = [0 for _ in range(ResidueFeature.get_n_features())]
        all_features = [0 for _ in range(len(stride_entry))]
        self._features[0] = float(AA_TO_INT[stride_entry[0]])
        for i in range(1, len(stride_entry)):
            all_features[i] = stride_entry[i]
        self._features[1] = float(SS_CODE_TO_INT[stride_entry[1]]+1) # add one here, so different from stride that starts from 0
        # self._features[1] = circular_mean([all_features[2]])
        # self._features[2] = circular_mean([all_features[3]])
        # ca_atom = residue['CA']
        # coords = ca_atom.get_coord()
        # self._features[5] = coords[0]
        # self._features[6] = coords[1]
        # self._features[7] = coords[2]
        # self._features[8] = ca_atom.get_occupancy()
        # self._features[9] = ca_atom.get_bfactor()
        self._features = np.array(self._features)

    @staticmethod
    def get_n_features():
        """
        Return the size of a single feature.

        Returns
        -------
        int
            The size of a feature vector.
        """
        return 2

    def get_features(self):
        """
        Return the features.

        Returns
        -------
        numpy.ndarray
            The features.
        """
        return self._features

class GraphFeature:
    """
    Modelling of graph features.

    A protein graph is composed by residues (nodes) either connected (backbone) or interacting.

    How the features are modelled depends on the mode. If the mode is dense, then the connections are represented as a
    full adjacency matrix. If sparse, then it is represented as a list of connections. The same goes for edge features.

    This class makes use of the `ResidueFeature`.

    Attributes
    ----------
    protein : Bio.PDB.Structure
        The protein from which to extract the features.
    pdb_dir : str
        The directory holding the PDB files.
    stride_dir : str
        The directory holding the Stride tool.
    """

    def __init__(self, protein, pdb_dir, stride_dir, dist_thr=12, mode='sparse'):
        """
        Initialize the class.

        Parameters
        ----------
        protein : Bio.PDB.Structure
            The protein from which to extract the features.
        pdb_dir : str
            The directory holding the PDB files.
        stride_dir : str
            The directory holding the Stride tool.
        dist_thr : float, optional
            The distance threshold below which two residues are interacting. The default is 12.
        mode : str, optional
            How the data should be. Valid options are ['sparse', 'dense']. The default is 'sparse'.
        """
        self.protein = protein
        self.pdb_dir = pdb_dir
        self.stride_dir = stride_dir
        self._dist_thr = dist_thr
        self._mode = mode
        self._weights_cache = {}

    def _get_adjacency(self):
        n = structure_length(self.protein)
        if self._mode == 'dense':
            adj, w_adj = np.zeros(shape=(n,n)), np.zeros(shape=(n,n))
        elif self._mode == 'sparse':
            adj, w_adj = [], []
        residues = []
        for residue in self.protein.get_residues():
            residues.append(residue)
        for i in range(n-1):
            for j in range(i+1, n):
                if 'CA' in residues[i] and 'CA' in residues[j]:
                    ca_i, ca_j = residues[i]['CA'], residues[j]['CA']
                    ca_dist = np.linalg.norm(ca_i.get_vector()-ca_j.get_vector())
                    if ca_dist < self._dist_thr:
                        if self._mode == 'dense':
                            adj[i,j] = adj[j,i] = 1
                            w_adj[i,j] = w_adj[j,i] = 1/ca_dist
                        elif self._mode == 'sparse':
                            adj.append([i,j])
                            adj.append([j,i])
                            w_adj.append(1/ca_dist)
                            w_adj.append(1/ca_dist)
                        self._weights_cache[(i,j)] = self._weights_cache[(j,i)] = ca_dist
        if self._mode == 'sparse':
            adj, w_adj = np.array(adj), np.array(w_adj)
        return adj, w_adj

    def _get_edge_feature(self):
        n = structure_length(self.protein)
        if self._mode == 'dense':
            edge = np.zeros(shape=(n,n,2))
        elif self._mode == 'sparse':
            edge = []
        for i in range(n-1):
            for j in range(i+1, n):
                if (i,j) in self._weights_cache:
                    w = self._weights_cache[(i,j)]
                    bb = 1 if abs(i-j) == 1 else 0 # 1 is backbone, 0 is contact
                    if self._mode == 'dense':
                        edge[i,j] = edge[j,i] = np.array([w, bb])
                    elif self._mode == 'sparse':
                        edge.append([w, bb])
                        edge.append([w, bb])
        if self._mode == 'sparse':
            edge = np.array(edge)
        return edge

    def _get_node_feature(self):
        pdb_id = self.protein.get_id()
        stride_entries = single_stride(self.stride_dir, self.pdb_dir + pdb_id + '.pdb')
        if stride_entries is None or structure_length(self.protein) != len(stride_entries):
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
        """
        Compute and return the features.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            The features.
        """
        adj, w_adj = self._get_adjacency() # remember should always be called before _get_edge_features
        edge = self._get_edge_feature()
        x = self._get_node_feature()
        return adj, w_adj, edge, x