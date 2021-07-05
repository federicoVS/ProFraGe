from operator import le, ge

import numpy as np

from cluster.Cluster import Cluster
from structure.representation import USR, FullStride
from structure.similarity import QCPSimilarity
from utils.structure import lengths_within
from utils.tm_align import run_tm_align
from utils.ProgressBar import ProgressBar

class GreedyCluster(Cluster):
    """
    The abstract greedy cluster class.

    It contains the skeleton for greedy clustering.

    Attributes
    ----------
    score_thr : float
        The score threshold.
    compare : function
        The operator to apply to compare two scores. It should be a function defined in `operator`.
    """
    
    def __init__(self, structures, score_thr, compare, verbose=False):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        score_thr : float
            The score threshold.
        compare : function
            The comparing function.
        verbose : bool, optional
            Whether to print progress information. The default is False.
        """
        super(GreedyCluster, self).__init__(structures, verbose)
        self.score_thr = score_thr
        self.compare = compare
        
    def condition(self, i, j):
        """
        Return whether two structures are suitable to be compared. This method is meant to be overridden by subclasses.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        None.
        """
        pass
    
    def score_function(self, i, j):
        """
        Return the score for two structures. This method is meant to be overridden by subclasses.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        None.
        """
        pass

    def greedy(self):
        """
        Perform the full greedy clustering. This method is meant to  be called by subclasses in their `cluster` method.

        Returns
        -------
        None.
        """
        # Define n for convenience
        n = len(self.structures)
        # Data structures for clustering
        cluster_id = 0
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if len(self.clusters) == 0:
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                cluster_id += 1
            else:
                assigned = False
                for idx in range(cluster_id):
                    scores = []
                    for j in self.clusters[idx]:
                        scores.append(self.score_function(i, j))
                    score = np.mean(np.array(scores))
                    if self.compare(score, self.score_thr):
                        self.clusters[idx].append(i)
                        assigned = True
                        break
                if not assigned:
                    self.clusters[cluster_id] = []
                    self.clusters[cluster_id].append(i)
                    cluster_id += 1
        if self.verbose:
            progress_bar.end()
    
    def optimal(self):
        """
        Perform the optimal greedy clustering. This method is meant to  be called by subclasses in their `cluster` method.

        Returns
        -------
        None.
        """
        # Define n for convenience
        n = len(self.structures)
        # Data structures for clustering
        cluster_id = 0
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if len(self.clusters) == 0:
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                cluster_id += 1
            else:
                best_idx, best_score = -1, 0
                for idx in range(cluster_id):
                    scores = []
                    for j in self.clusters[idx]:
                        scores.append(self.score_function(i, j))
                    score = np.mean(np.array(scores))
                    if self.compare(score, self.score_thr) and (best_idx == -1 or self.compare(score, best_score)):
                        best_score = score
                        best_idx = idx
                if best_idx != -1:
                    self.clusters[best_idx].append(i)
                else:
                    self.clusters[cluster_id] = []
                    self.clusters[cluster_id].append(i)
                    cluster_id += 1
        if self.verbose:
            progress_bar.end()

    def cluster(self, mode):
        """
        Perform the specified clustering algorithm.

        Parameters
        ----------
        mode : str
            The clustering algorithm mode.

        Returns
        -------
        None.
        """
        if mode == 'greedy':
            self.greedy()
        elif mode == 'optim':
            self.optimal()
        else:
            return
    
class AtomicSuperImpose(GreedyCluster):
    """
    Perform clustering of the structures based on their superimposition.
    
    Two structures are then considered similar and clustered together if the resulting RMSD is lower than the specified threshold.
    
    Attributes
    ----------
    length_pct : float in [0,1]
        The percentage of length two structures should share to be considered similar.
    """
    
    def __init__(self, structures, rmsd_thr=2.0, length_pct=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        rmsd_thr : float, optional
            The RMSD threshold below which two fragments are considered similar. The default is 2.
        length_pct : float in [0,1], optional
            The length percentage threshold. The default is 0.5.
        verbose : TYPE, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(AtomicSuperImpose, self).__init__(structures, rmsd_thr, le, verbose)
        self.length_pct = length_pct
        
    def condition(self, i, j):
        """
        Check whether the two structures have comparable length.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        bool
            Whether the two structures have comparable length.
        """
        return lengths_within(self.structures[i], self.structures[j], self.length_pct)
    
    def score_function(self, i, j):
        """
        Return the super-imposition RMSD score between the two structures.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        float
            The RMSD score.
        """
        qcps = QCPSimilarity(self.structures[i], self.structures[j])
        return qcps.compare()
            
class TMAlign(GreedyCluster):
    """
    Perform clustering based on TM-Align score similarity computed using the MICAN tool.
    
    Attributes
    ----------
    tm_align_dir : str
        The directory holding the TM-Align tool.
    pdb_dir : str
        The directory holding the PDB files.
    length_pct : float in [0,1]
        The percentage length two structures must share measure in their number of amino acids.
    score_thr : float in [0,1]
        The TM-Align score threshold above which two structures are considered to be similar.
    """
    
    def __init__(self, structures, tm_align_dir, pdb_dir, score_thr=0.5, length_pct=0.8, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures to cluster.
        tm_align_dir : str
            The directory holding the TM-Align tool.
        pdb_dir : str
            The directory holding the PDB files.
        score_thr : float in [0,1], optional
            The TM-Align score threshold. The default is 0.5.
        length_pct : float in [0,1], optional
            The percentage length two structures must share. The default is 0.8
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(TMAlign, self).__init__(structures, score_thr, ge, verbose)
        self.tm_align_dir = tm_align_dir
        self.pdb_dir = pdb_dir
        self.score_thr = score_thr
        self.length_pct = length_pct
        
    def condition(self, i, j):
        """
        Check whether the length of the two structures is comparable.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        bool
            Whether the length of the two structures is comparable.
        """
        return lengths_within(self.structures[i], self.structures[j], self.length_pct)
    
    def score_function(self, i, j):
        """
        Return the TM-Align score of the two structures.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        float in [0,1]
            The TM-Align score.
        """
        pdb_i = self.pdb_dir + self.structures[i].get_id() + '.pdb'
        pdb_j = self.pdb_dir + self.structures[j].get_id() + '.pdb'
        return run_tm_align(self.tm_align_dir, pdb_i, pdb_j)
            
class USRCluster(GreedyCluster):
    """
    Perform clustering of the structures based on their USR score.
    
    Two structures are considered to be similar if the USR-similarity between their USR features is
    above the specified threshold.
    
    Attributes
    ----------
    score_thr : float in [0,1]
        The similarity score threshold above which two structures are considered to be similar. The higher
        the tighter.
    bb_atoms : bool
        Whether to only use the backbone atoms to compute the USR.
    _features : numpy.ndarray
        A matrix holding the USR features for each structure.
    """
    
    def __init__(self, structures, score_thr=0.5, bb_atoms=False, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to be clustered.
        score_thr : float in [0,1], optional
            The similarity score threshold. The default is 0.5.
        bb_atoms : bool, optional
            Whether to only use the backbone atoms to compute the USR. The default is False.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(USRCluster, self).__init__(structures, score_thr, ge, verbose)
        self.score_thr = score_thr
        self.bb_atoms = bb_atoms
        self._features = None
        
    def condition(self, i, j):
        """
        Do nothing.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        None.
        """
        pass
    
    def score_function(self, i, j):
        """
        Return the USR similarity score of the two structures.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        float in [0,1]
            The similarity score.
        """
        return USR.get_similarity_score(self._features[i], self._features[j])
        
    def cluster(self, mode):
        """
        Perform the clustering.

        Parameters
        ----------
        mode : str
            The clustering algorithm mode.

        Returns
        -------
        None.
        """
        # Define n for convenience
        n = len(self.structures)
        # Define feature matrices
        self._features = np.zeros(shape=(n, USR.get_n_features()))
        # Compute USR for each structure
        for i in range(n):
            usr = USR(self.structures[i], bb_atoms=self.bb_atoms)
            self._features[i,:] = usr.get_features()
        # Cluster
        super().cluster(mode)
            
class StrideCluster(GreedyCluster):
    """
    Perform clustering of the structures based on their Stride score.
    
    Two structures are considered to be similar if the cosine distance between their Stride features is
    above the specified threshold.
    
    Attributes
    ----------
    stride_dir : str
        The directory holding the Stride tool.
    pdb_dir : str
        The directory holding the PDB files to cluster.
    score_thr : float in [0,1]
        The cosine score threshold above which two structures are considered to be similar. The higher
        the tighter.
    _features : numpy.ndarray
        A matrix holding the USR features for each structure.
    """
    
    def __init__(self, structures, stride_dir, pdb_dir, score_thr=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to be clustered.
        stride_dir : str
            The directory holding the Stride tool.
        pdb_dir : str
            The directory holding the PDB files to cluster.
        score_thr : float in [0,1], optional
            The cosine score threshold. The default is 0.5.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(StrideCluster, self).__init__(structures, score_thr, ge, verbose)
        self.stride_dir = stride_dir
        self.pdb_dir = pdb_dir
        self.score_thr = score_thr
        self._features = None
        
    def condition(self, i, j):
        """
        Do nothing.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        None.
        """
        pass
    
    def score_function(self, i, j):
        """
        Return the Stride similarity score between the two structures.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        float in [0,1]
            The similarity score.
        """
        return FullStride.get_similarity_score(self._features[i], self._features[j])
        
    def cluster(self, mode):
        """
        Perform the clustering.

        Parameters
        ----------
        mode : str
            The clustering algorithm mode.

        Returns
        -------
        None.
        """
        # Define n for convenience
        n = len(self.structures)
        # Keep for valid structures
        valid_structures = []
        # Define feature matrices
        self._features = [] #np.zeros(shape=(n, FullStride.get_n_features()))
        # Compute Stride for each structure
        for i in range(n):
            pdb = self.pdb_dir + self.structures[i].get_id() + '.pdb'
            feats = FullStride(self.stride_dir, pdb).get_features()
            if feats is not None:
                self._features.append(feats)
                valid_structures.append(self.structures[i])
        # Set structures to valid ones
        self.structures = valid_structures
        # Numpify features
        self._features = np.array(self._features)
        # Cluster
        super().cluster(mode)