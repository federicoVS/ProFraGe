# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:11:01 2021

@author: Federico van Swaaij
"""

import numpy as np

from Bio import pairwise2
from Bio.PDB.Polypeptide import CaPPBuilder
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.QCPSuperimposer import QCPSuperimposer

from cluster.Cluster import Cluster
from structure.representation import USR, FullStride
from utils.structure import lengths_within
from utils.mican import run_mican
from utils.ProgressBar import ProgressBar

class SeqAlign(Cluster):
    """
    Perform clustering of the structures based on the their sequence alignement.
    
    Each structure is matched against the others, with the ones falling within the score
    threshold being added to its cluster.
    The clustering is greedy in that once a structure is assigned to a cluster, it cannot
    be assigned to another one.
    
    Attributes
    ----------
    seq_score_thr : int
        The threshold score for two sequences to be considered similar.
    length_pct_thr : float
        The percentage of length two structures have to share to be considered similar.
    """
    
    def __init__(self, structures, seq_score_thr=10, length_pct_thr=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        seq_score_thr : int, optional
            The threshold score above which two sequences to be considered similar. The default is 10
        length_pct_thr : float in [0,1], optional
            The mininal percentage of length two structures have share to be considered. The default is 0.5.
            similar.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(SeqAlign, self).__init__(structures, verbose)
        self.seq_score_thr = seq_score_thr
        self.length_pct_thr = length_pct_thr
        self.verbose = verbose
        
    def _get_sequences(self, index):
        """
        Return the sequences of the specified structure.
        
        The sequence is that of its polypeptides, which are computed based on the
        distances of its Ca atoms.

        Parameters
        ----------
        index : int
            The index of the structure.

        Returns
        -------
        sequences : list of Bio.Seq.Seq
            The list of sequences of the structure.
        """
        sequences = []
        ppb = CaPPBuilder()
        for pp in ppb.build_peptides(self.structures[index]):
            sequences.append(pp.get_sequence())
        return sequences
    
    def _compare(self, index_1, index_2):
        """
        Compare the specified structures by aligning them.

        Parameters
        ----------
        index_1 : int
            The index of the first structure.
        index_2 : TYPE
            The second of the first structure..

        Returns
        -------
        int
            The best score, i.e. the largest one among all sequences.
        """
        seq_1 = self._get_sequences(index_1)
        seq_2 = self._get_sequences(index_2)
        score = 0
        for s_1 in seq_1:
            for s_2 in seq_2:
                score += pairwise2.align.globalxx(s_1, s_2, score_only=True)
        return score
    
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        cluster_id = 0
        placed = [False for i in range(len(self.structures))]
        score_cache = {}
        progress_bar = ProgressBar(len(self.structures))
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(len(self.structures)):
            if self.verbose:
                progress_bar.step()
            # Check if structure i already belongs to a cluster
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(len(self.structures)):
                    if j != i and lengths_within(self.structures[i], self.structures[j], self.length_pct_thr) and not placed[j]:
                        score = 0
                        if (i,j) not in score_cache:
                            score = self._compare(i, j)
                            score_cache[(i,j)] = score
                        else:
                            score = score_cache[(i,j)]
                        if score >= self.seq_score_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
    
class AtomicSuperImpose(Cluster):
    """
    Perform clustering of the structures based on their superimposition.
    
    Two structuress are then considered similar and clustered together if the resulting RMSD is higher
    than the specified threshold.
    
    Attributes
    ----------
    rmsd_thr : float
        The RMSD threshold for two structures to be considered similar.
    length_pct : float in [0,1]
        The percentage of length two structures should share to be considered similar.
    """
    
    def __init__(self, structures, rmsd_thr=10, length_pct=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        rmsd_thr : float, optional
            The RMSD threshold below which two fragments are considered similar. The default is 10.
        length_pct : float in [0,1], optional
            The length percentage threshold. The default is 0.5.
        verbose : TYPE, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(AtomicSuperImpose, self).__init__(structures, verbose)
        self.rmsd_thr = rmsd_thr
        self.length_pct = length_pct
        
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
            The minimum RMSD value across the computed superimpositions .
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
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        cluster_id = 0
        placed = [False for i in range(len(self.structures))]
        rmsd_cache = {}
        progress_bar = ProgressBar(len(self.structures))
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(len(self.structures)):
            if self.verbose:
                progress_bar.step()
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(len(self.structures)):
                    if j != i and lengths_within(self.structures[i], self.structures[j], self.length_pct) and not placed[j]:
                        rmsd = 0
                        if (i,j) not in rmsd_cache:
                            rmsd = self._compare(i, j)
                            rmsd_cache[(i,j)] = rmsd_cache[(j,i)] = rmsd
                        else:
                            rmsd = rmsd_cache[(i,j)]
                        if rmsd < self.rmsd_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
            
class Mican(Cluster):
    """
    Perform clustering based on TM-Align score similarity computed using the MICAN tool.
    
    Attributes
    ----------
    mican_dir : str
        The directory holding the MICAN tool.
    pdb_dir : str
        The directory holding the PDB files.
    length_pct : float in [0,1]
        The percentage length two structures must share measure in their number of amino acids.
    score_thr : float in [0,1]
        The TM-Align score threshold above which two structures are considered to be similar.
    """
    
    def __init__(self, structures, mican_dir, pdb_dir, score_thr=0.5, length_pct=0.8, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures to cluster.
        mican_dir : str
            The directory holding the MICAN tool.
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
        super(Mican, self).__init__(structures, verbose)
        self.mican_dir = mican_dir
        self.pdb_dir = pdb_dir
        self.score_thr = score_thr
        self.length_pct = length_pct
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        n = len(self.structures)
        cluster_id = 0
        placed = [False for i in range(n)]
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(n):
                    if i != j and not placed[j]:
                        s_i, s_j = self.structures[i], self.structures[j]
                        if not lengths_within(s_i, s_j, self.length_pct):
                            continue
                        pdb_i = self.pdb_dir + s_i.get_id() + '.pdb'
                        pdb_j = self.pdb_dir + s_j.get_id() + '.pdb'
                        score = run_mican(self.mican_dir, pdb_i, pdb_j)
                        if score > self.score_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
            
class USRCluster(Cluster):
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
        super(USRCluster, self).__init__(structures, verbose)
        self.score_thr = score_thr
        self.ca_atoms = ca_atoms
        self._features = None
        
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
        # Data structures for clustering
        cluster_id = 0
        placed = [False for i in range(n)]
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(n):
                    if j != i and not placed[j]:
                        score = USR.get_similarity_score(self._features[i], self._features[j])
                        if score > self.score_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
            
class StrideCluster(Cluster):
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
    cosine_thr : float in [0,1]
        The cosine score threshold above which two structures are considered to be similar. The higher
        the tighter.
    _features : numpy.ndarray
        A matrix holding the USR features for each structure.
    """
    
    def __init__(self, structures, stride_dir, pdb_dir, cosine_thr=0.5, verbose=False, **params):
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
        cosine_thr : float in [0,1], optional
            The cosine score threshold. The default is 0.5.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(StrideCluster, self).__init__(structures, verbose)
        self.stride_dir = stride_dir
        self.pdb_dir = pdb_dir
        self.cosine_thr = cosine_thr
        self._features = None
        
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
        # Data structures for clustering
        cluster_id = 0
        placed = [False for i in range(n)]
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(n):
                    if j != i and not placed[j]:
                        cosine = FullStride.get_similarity_score(self._features[i], self._features[j])
                        if cosine > self.cosine_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
            
class USRStrideCluster(Cluster):
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
        super(USRStrideCluster, self).__init__(structures, verbose)
        self.stride_dir = stride_dir
        self.pdb_dir = pdb_dir
        self.full_thr = full_thr
        self.ca_atoms = ca_atoms
        self._usr_features = None
        self._stride_features = None
        
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
        # Data structures for clustering
        cluster_id = 0
        placed = [False for i in range(n)]
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(n):
                    if j != i and not placed[j]:
                        usr = USR.get_similarity_score(self._usr_features[i], self._usr_features[j])
                        cosine = FullStride.get_similarity_score(self._stride_features[i], self._stride_features[j])
                        score = usr + cosine
                        if score > self.full_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
    
    