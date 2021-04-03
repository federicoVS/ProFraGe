# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:11:01 2021

@author: FVS
"""

import numpy as np
from Bio import pairwise2
from Bio.PDB.Polypeptide import CaPPBuilder
from Bio.PDB.Superimposer import Superimposer
from cluster.Cluster import Cluster
from utils.ProgressBar import ProgressBar
from utils.misc import structure_length

class SeqAlign(Cluster):
    '''
    Performs clustering of the structures based on the their sequence alignement.
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
    '''
    
    def __init__(self, structures, seq_score_thr, length_pct_thr, verbose=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures to cluster.
        seq_score_thr : int
            The threshold score for two sequences to be considered similar.
        length_pct_thr : float in [0,1]
            The mininal percentage of length two structures have share to be considered
            similar.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        '''
        super(SeqAlign, self).__init__(structures, verbose)
        self.seq_score_thr = seq_score_thr
        self.length_pct_thr = length_pct_thr
        self.verbose = verbose
        
    def _compare(self, index_1, index_2):
        '''
        Compares the specified structures by aligning them.

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
        '''
        seq_1 = self._get_sequences(index_1)
        seq_2 = self._get_sequences(index_2)
        score = 0
        for s_1 in seq_1:
            for s_2 in seq_2:
                score += pairwise2.align.globalxx(s_1, s_2, score_only=True)
        return score
            
    def cluster(self):
        '''
        Performs the clustering.

        Returns
        -------
        None.
        '''
        # Initialize
        self.clusters = {}
        cluster_id = 1
        placed = [False for i in range(len(self.structures))]
        score_cache = {}
        progress_bar = ProgressBar()
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(len(self.structures)):
            if self.verbose:
                progress_bar.step(i+1, len(self.structures))
            # Check if structure i already belongs to a cluster
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(len(self.structures)):
                    if j != i and self._structure_lengths_within(i, j) and not placed[j]:
                        score = 0
                        if (i,j) not in score_cache:
                            score = self.compare(i, j)
                            score_cache[(i,j)] = score
                        else:
                            score = score_cache[(i,j)]
                        if score >= self.seq_score_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
            
    def _structure_lengths_within(self, index_1, index_2):
        '''
        Returns whether the specified structures have comparable length.

        Parameters
        ----------
        index_1 : int
            The index of the first structure.
        index_2 : int
            The index of the first structure.

        Returns
        -------
        bool
            Whether the structures have comparable length.
        '''
        small = structure_length(self.structures[index_1])
        large = structure_length(self.structures[index_2])
        if small > large:
            temp = large
            large = small
            small = temp
        return (small/large) >= self.length_pct_thr
        
    def _get_sequences(self, index):
        '''
        Returns the sequences of the specified structure.
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
        '''
        sequences = []
        ppb = CaPPBuilder()
        for pp in ppb.build_peptides(self.structures[index]):
            sequences.append(pp.get_sequence())
        return sequences
    
class CASuperImpose(Cluster):
    
    def __init__(self, structures, rmsd_thr, length_pct_thr, verbose=False):
        super(CASuperImpose, self).__init__(structures, verbose)
        self.rmsd_thr = rmsd_thr
        self.length_pct_thr = length_pct_thr
        
    def compare(self, index_1, index_2):
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
        self.clusters = {}
        cluster_id = 1
        placed = [False for i in range(len(self.structures))]
        rmsd_cache = {}
        progress_bar = ProgressBar()
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(len(self.structures)):
            if self.verbose:
                progress_bar.step(i, len(self.structures))
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(len(self.structures)):
                    if j != i and self._length_within(i, j) and not placed[j]:
                        rmsd = 0
                        if (i,j) not in rmsd_cache:
                            rmsd = self.compare(i, j)
                            rmsd_cache[(i,j)] = rmsd
                        else:
                            rmsd = rmsd_cache[(i,j)]
                        if rmsd <= self.rmsd_thr:
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
                        
    def _length_within(self, index_1, index_2):
        '''
        Returns whether the specified structures have comparable length.

        Parameters
        ----------
        index_1 : int
            The index of the first structure.
        index_2 : int
            The index of the first structure.

        Returns
        -------
        bool
            Whether the structures have comparable length.
        '''
        small = structure_length(self.structures[index_1])
        large = structure_length(self.structures[index_2])
        if small > large:
            temp = large
            large = small
            small = temp
        return (small/large) >= self.length_pct_thr
        
    def _get_ca_atoms(self, index):
        structure = self.structures[index]
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    ca_atoms.append(residue['CA'].copy())
        return ca_atoms
    
    def _get_shifted_atoms_subsets(self, ca_atoms_long, ca_atoms_short):
        shifted_atoms = []
        for i in range(len(ca_atoms_long) - len(ca_atoms_short)):
            s_atoms = []
            for j in range(i, i+len(ca_atoms_short)):
                s_atoms.append(ca_atoms_long[j])
            shifted_atoms.append(s_atoms)
        return shifted_atoms
    
    def _superimpose(self, index, atoms_fixed, atoms_moving):
        si = Superimposer()
        # print(atoms_fixed)
        try:
            np.seterr(all='ignore')
            si.set_atoms(atoms_fixed, atoms_moving)
            si.apply(self.structures[index])
            return si.rms
        except np.linalg.LinAlgError:
            return np.inf
    
    