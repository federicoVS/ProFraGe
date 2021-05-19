# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:11:01 2021

@author: Federico van Swaaij
"""

import numpy as np
from Bio import pairwise2
from Bio.PDB.Polypeptide import CaPPBuilder
from Bio.PDB.Superimposer import Superimposer

from cluster.Cluster import Cluster
from structure.representation import USR
from utils.ProgressBar import ProgressBar
from utils.structure import lengths_within

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
    
class CASuperImpose(Cluster):
    """
    Perform clustering of the structures based on the superimposition of their alpha-carbon (CA) atoms.
    
    Two structuress are then considered similar and clustered together if the resulting RMSD is higher
    than the specified threshold.
    
    Attributes
    ----------
    rmsd_thr : float
        The RMSD threshold for two structures to be considered similar.
    length_pct_thr : float in [0,1]
        The percentage of length two structures should share to be considered similar.
    """
    
    def __init__(self, structures, rmsd_thr=10, length_pct_thr=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        rmsd_thr : float, optional
            The RMSD threshold below which two fragments are considered similar. The default is 10.
        length_pct_thr : float in [0,1], optional
            The length percentage threshold. The default is 0.5.
        verbose : TYPE, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(CASuperImpose, self).__init__(structures, verbose)
        self.rmsd_thr = rmsd_thr
        self.length_pct_thr = length_pct_thr
        
    def _get_ca_atoms(self, index):
        """
        Get the CA atoms from the specified structure.

        Parameters
        ----------
        index : int
            The index of the structure.

        Returns
        -------
        ca_atoms : list of Bio.PDB.Atoms
            The CA atoms of the structure.
        """
        structure = self.structures[index]
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    ca_atoms.append(residue['CA'].copy())
        return ca_atoms
    
    def _get_shifted_atoms_subsets(self, ca_atoms_long, ca_atoms_short):
        """
        Compute all subsets of contiguous CA atoms for the longer chain.
        
        This may be necessary because superimposition requires structures of the same length.

        Parameters
        ----------
        ca_atoms_long : list of Bio.PDB.Atoms 
            The list of the CA atoms of the larger structure.
        ca_atoms_short : list of Bio.PDB.Atoms
            The list of the CA atoms of the shorter structure..

        Returns
        -------
        shifted_atoms : list of list of Bio.PDB.Atoms
            The list of shifted CA atoms.
        """
        shifted_atoms = []
        for i in range(len(ca_atoms_long) - len(ca_atoms_short)):
            s_atoms = []
            for j in range(i, i+len(ca_atoms_short)):
                s_atoms.append(ca_atoms_long[j])
            shifted_atoms.append(s_atoms)
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
        si = Superimposer()
        try:
            np.seterr(all='ignore')
            si.set_atoms(atoms_fixed, atoms_moving)
            si.apply(self.structures[index])
            return si.rms
        except np.linalg.LinAlgError:
            return np.inf
        
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
        ca_atoms_1 = self._get_ca_atoms(index_1)
        ca_atoms_2 = self._get_ca_atoms(index_2)
        if len(ca_atoms_1) == len(ca_atoms_2):
            return self._superimpose(index_1, ca_atoms_1, ca_atoms_2)
        elif len(ca_atoms_1) > len(ca_atoms_2):
            shifted_atoms = self._get_shifted_atoms_subsets(ca_atoms_1, ca_atoms_2)
            rmsds = []
            for sas in shifted_atoms:
                rmsds.append(self._superimpose(index_1, sas, ca_atoms_2))
            rmsds = np.array(rmsds)
            return np.min(rmsds)
        else:
            shifted_atoms = self._get_shifted_atoms_subsets(ca_atoms_2, ca_atoms_1)
            rmsds = []
            for sas in shifted_atoms:
                rmsds.append(self._superimpose(index_1, ca_atoms_1, sas))
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
                    if j != i and lengths_within(self.structures[i], self.structures[j], self.length_pct_thr) and not placed[j]:
                        rmsd = 0
                        if (i,j) not in rmsd_cache:
                            rmsd = self._compare(i, j)
                            rmsd_cache[(i,j)] = rmsd
                        else:
                            rmsd = rmsd_cache[(i,j)]
                        if rmsd <= self.rmsd_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
            
class USRCluster(Cluster):
    """
    Perform clustering of the structures based on their USR score.
    
    Two structures are considered to be similar if the cosine distance between their USR features is
    below a specified threshold.
    
    Attributes
    ----------
    score_thr : float in [0,1]
        The similarity score threshold above which two structures are considered to be similar. The higher
        the tighter.
    _features : numpy.ndarray
        A matrix holding the USR features for each structure.
    """
    
    def __init__(self, structures, score_thr=0.5, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to be clustered.
        score_thr : float in [0,1], optional
            The similarity score threshold. The default is 0.5.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(USRCluster, self).__init__(structures, verbose)
        self.score_thr = score_thr
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
            usr = USR(self.structures[i])
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
    
    