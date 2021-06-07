# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:11:01 2021

@author: Federico van Swaaij
"""

from operator import le, ge

import numpy as np

from Bio.PDB.QCPSuperimposer import QCPSuperimposer

from cluster.Cluster import Cluster
from structure.representation import USR, FullStride
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
    
    def greedy_cluster(self):
        """
        Perform the greedy clustering. This method is meant to  be called by subclasses in their `cluster` method.

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
    
class AtomicSuperImpose(GreedyCluster):
    """
    Perform clustering of the structures based on their superimposition.
    
    Two structuress are then considered similar and clustered together if the resulting RMSD is higher
    than the specified threshold.
    
    Attributes
    ----------
    score_thr : float
        The RMSD threshold for two structures to be considered similar.
    length_pct : float in [0,1]
        The percentage of length two structures should share to be considered similar.
    """
    
    def __init__(self, structures, score_thr=2, length_pct=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        score_thr : float, optional
            The RMSD threshold below which two fragments are considered similar. The default is 2.
        length_pct : float in [0,1], optional
            The length percentage threshold. The default is 0.5.
        verbose : TYPE, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(AtomicSuperImpose, self).__init__(structures, score_thr, le, verbose)
        self.score_thr = score_thr
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
        return self._compare(i, j)
    
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        self.greedy_cluster()
        
    def _get_atoms(self, index):
        """
        Get the atoms from the specified structure.

        Parameters
        ----------
        index : int
            The index of the structure.

        Returns
        -------
        ca_atoms : list of Bio.PDB.Atom
            The atoms of the structure.
        """
        structure = self.structures[index]
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        atoms.append(residue['CA'])
        return atoms
    
    def _get_shifted_atoms(self, atoms_long, atoms_short):
        """
        Compute all subsets of contiguous atoms for the longer chain.
        
        This may be necessary because superimposition requires structures of the same length.

        Parameters
        ----------
        atoms_long : list of Bio.PDB.Atoms 
            The list of the atoms of the larger structure.
        atoms_short : list of Bio.PDB.Atoms
            The list of the atoms of the shorter structure..

        Returns
        -------
        shifted_atoms : list of list of Bio.PDB.Atoms
            The list of shifted atoms.
        """
        shifted_atoms, i = [], 0
        while i + len(atoms_short) <= len(atoms_long):
            shifted_atoms.append(atoms_long[i:i+len(atoms_short)])
            i += 1
        return shifted_atoms
    
    def _superimpose(self, index, atoms_fixed, atoms_moving):
        """
        Perform superimposition between the specified structures.

        Parameters
        ----------
        index : int
            The index of the reference structure.
        atoms_fixed : list of Bio.PDB.Atoms
            The list of the fixed (reference) atoms.
        atoms_moving : numpy.ndarray
            The list of the moving atoms.

        Returns
        -------
        float
            The RMSD. In case of errors numpy.inf is returned.
        """
        # Prepare the coordinates
        fixed, moving = np.zeros(shape=(len(atoms_fixed),3)), np.zeros(shape=(len(atoms_fixed),3))
        for i in range(len(atoms_fixed)):
            fixed[i,:], moving[i,:] = atoms_fixed[i].get_coord(), atoms_moving[i].get_coord()
        # Superimpose
        qcpsi = QCPSuperimposer()
        qcpsi.set(fixed, moving)
        qcpsi.run()
        return qcpsi.get_rms()
        
    def _compare(self, index_1, index_2):
        """
        Compare the specified structures by superimposing them.

        Parameters
        ----------
        index_1 : int
            The index of the first structure.
        index_2 : int
            The index of the second structure.

        Returns
        -------
        float
            The minimum RMSD value across the computed super-impositions.
        """
        atoms_1 = self._get_atoms(index_1)
        atoms_2 = self._get_atoms(index_2)
        if len(atoms_1) == len(atoms_2):
            return self._superimpose(index_1, atoms_1, atoms_2)
        elif len(atoms_1) > len(atoms_2):
            shifted_atoms = self._get_shifted_atoms(atoms_1, atoms_2)
            rmsds = []
            for sas in shifted_atoms:
                rmsds.append(self._superimpose(index_1, sas, atoms_2))
            rmsds = np.array(rmsds)
            return np.min(rmsds)
        else:
            shifted_atoms = self._get_shifted_atoms(atoms_2, atoms_1)
            rmsds = []
            for sas in shifted_atoms:
                rmsds.append(self._superimpose(index_1, atoms_1, sas))
            rmsds = np.array(rmsds)
            return np.min(rmsds)
            
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
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        self.greedy_cluster()
            
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
    ca_atoms : bool
        Whether to only use C-alpha atoms to compute the USR.
    _features : numpy.ndarray
        A matrix holding the USR features for each structure.
    """
    
    def __init__(self, structures, score_thr=0.5, ca_atoms=False, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to be clustered.
        score_thr : float in [0,1], optional
            The similarity score threshold. The default is 0.5.
        ca_atoms : bool, optional
            Whether to only use C-alpha atoms to compute the USR. The default is False.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(USRCluster, self).__init__(structures, score_thr, ge, verbose)
        self.score_thr = score_thr
        self.ca_atoms = ca_atoms
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
        
    def cluster(self):
        """
        Perform the clustering.

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
            usr = USR(self.structures[i], ca_atoms=self.ca_atoms)
            self._features[i,:] = usr.get_features()
        # Cluster
        self.greedy_cluster()
            
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
        Return the Stride similaroty score between the two structures.

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
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        # Define n for convenience
        n = len(self.structures)
        # Define feature matrices
        self._features = np.zeros(shape=(n, FullStride.get_n_features()))
        # Compute USR for each structure
        for i in range(n):
            pdb = self.pdb_dir + self.structures[i].get_id() + '.pdb'
            self._features[i,:] = FullStride(self.stride_dir, pdb).get_features()
        # Cluster
        self.greedy_cluster()
            
class USRStrideCluster(GreedyCluster):
    """
    Perform clustering of the structures combining their USR and Stride scores.
    
    Two structures are considered to be similar if the sum of the USR and Stride scores are above the
    specified threshold.
    
    Attributes
    ----------
    stride_dir : str
        The directory holding the Stride tool.
    pdb_dir : str
        The directory holding the PDB files to cluster.
    full_thr : float in [0,2]
        The threshold above which two structures are considered to be equal. The higher the tigher.
    ca_atoms : bool
        Whether to only use C-alpha atoms to compute the USR.
    _usr_features : numpy.ndarray
        A matrix holding the USR features for each structure.
    _stride_features : numpy.ndarray
        A matrix holding the Stride features for each structure.
    """
    
    def __init__(self, structures, stride_dir, pdb_dir, full_thr=1.0, ca_atoms=False, verbose=False, **params):
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
        full_thr : float in [0,2], optional
            The threshold above which two structures are considered to be equal. The default is 1.0
        ca_atoms : bool, optional
            Whether to only use C-alpha atoms to compute the USR. The default is False.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(USRStrideCluster, self).__init__(structures, full_thr, ge, verbose)
        self.stride_dir = stride_dir
        self.pdb_dir = pdb_dir
        self.full_thr = full_thr
        self.ca_atoms = ca_atoms
        self._usr_features = None
        self._stride_features = None
        
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
        Return the sum of the USR similarity score and the Stride similarity score.

        Parameters
        ----------
        i : int
            The index of the first structure.
        j : int
            The index of the second structure.

        Returns
        -------
        float in [0,2]
            The similarity score.
        """
        usr = USR.get_similarity_score(self._usr_features[i], self._usr_features[j])
        cosine = FullStride.get_similarity_score(self._stride_features[i], self._stride_features[j])
        return usr + cosine
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        # Define n for convenience
        n = len(self.structures)
        # Define feature matrices
        self._usr_features = np.zeros(shape=(n, USR.get_n_features()))
        self._stride_features = np.zeros(shape=(n, FullStride.get_n_features()))
        # Compute USR for each structure
        for i in range(n):
            pdb = self.pdb_dir + self.structures[i].get_id() + '.pdb'
            self._usr_features[i,:] = USR(self.structures[i], ca_atoms=self.ca_atoms).get_features()
            self._stride_features[i,:] = FullStride(self.stride_dir, pdb).get_features()
        # Cluster
        self.greedy_cluster()
    
    