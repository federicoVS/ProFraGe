#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:11:01 2021

@author: FVS
"""

import numpy as np
import matplotlib.pyplot as plt
from Bio import pairwise2
from Bio.PDB.Polypeptide import CaPPBuilder
from utils.ProgressBar import ProgressBar
from utils.misc import structure_length

class SACluster:
    '''
    Performs clustering of the structures based on the their sequence alignement.
    Each structure is matched against the others, with the ones falling within the score
    threshold being added to its cluster.
    
    Attributes
    ----------
    structures : list of Bio.PDB.Structure
        The list of structures.
    seq_score_thr : int
        The threshold score for two sequences to be considered similar.
    str_length_pct : float
        The percentage of length two structures have to share to be considered similar.
    clusters : dict of int -> list of int
        The representation of the clusters as a dictionary mapping a cluster ID to the list
        of indices pointing to the similar structures. Such structures are similar to the
        extent of the score threshold.
    typicals : dict of int -> Bio.PDB.Structure
        A dictionary where the cluster ID points to the exemplary structure within the
        cluster.
    score_matrix : numpy.ndarray
        The matrix holding the pairwise score
    verbose : bool
        Whether to print progress information.
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
        self.structures = structures
        self.seq_score_thr = seq_score_thr
        self.length_pct_thr = length_pct_thr
        self.verbose = verbose
        self.clusters = {}
        self.typicals = {}
        self.score_matrix = self._compute_alignments()
        
    def get_clusters(self):
        '''
        Returns the generated clusters.

        Returns
        -------
        dict of int -> list of int
            The clusters.
        '''
        return self.clusters
    
    def get_typicals(self):
        '''
        Returns the typicals.

        Returns
        -------
        dict of int -> Bio.PDB.Structure
            The typicals.

        '''
        return self.typicals
    
    def show_clusters(self):
        '''
        Plots the size of each cluster as an histogram. The plot should indicate whether the
        clusters are balanced.

        Returns
        -------
        None.
        '''
        freqs = {}
        for cluster_id in self.clusters:
            freqs[cluster_id] = len(self.clusters[cluster_id])
            print(f'Cluster {cluster_id} has size {len(self.clusters[cluster_id])}')
        # Print the frequencies
        plt.bar(freqs.keys(), freqs.values(), 1.0, color='b')
        plt.show()
            
    def cluster(self):
        # Initialize
        self.clusters = {}
        self.typicals = {}
        cluster_id = 1
        placed = [False for i in range(len(self.structures))]
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
                    if j != i:
                        if self.score_matrix[i,j] >= self.seq_score_thr:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
            progress_bar.end()
        # Find typicals
        progress_bar = ProgressBar()
        if self.verbose:
            print('Finding typicals...')
            progress_bar.start()
        count = 1
        for cluster_id in self.clusters:
            if self.verbose:
                progress_bar.step(count, len(self.clusters))
                count += 1
            self._compute_typical(cluster_id)
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
        # print(len(sequences))
        return sequences
        
    def _align(self, index_1, index_2):
        '''
        Aligns and computes the best scores of the two specified structures.

        Parameters
        ----------
        index_1 : int
            The index of the first structure.
        index_2 : TYPE
            The second of the first structure..

        Returns
        -------
        int
            The best score, i.e. the largest one.
        '''
        seq_1 = self._get_sequences(index_1)
        seq_2 = self._get_sequences(index_2)
        score = 0
        for s_1 in seq_1:
            for s_2 in seq_2:
                score += pairwise2.align.globalxx(s_1, s_2, score_only=True)
        return score
    
    def _compute_alignments(self):
        '''
        Computes the score matrix.

        Returns
        -------
        score_matrix : numpy.ndarray
            The score matrix
        '''
        score_matrix = np.zeros(shape=(len(self.structures), len(self.structures)))
        for i in range(len(self.structures)-1):
            score_matrix[i,i] = structure_length(self.structures[i])
            for j in range(i+1, len(self.structures)):
                if self._structure_lengths_within(i, j):
                    score = self._align(i, j)
                    score_matrix[i,j] = score
                    score_matrix[j,i] = score
                else:
                    score_matrix[i,j] = -1
                    score_matrix[j,i] = -1
        return score_matrix
    
    def _compute_typical(self, cluster_id):
        '''
        Computes the typical (i.e. exemplary structure) in the cluster specified by the ID.
        The structure with the highest overall score is selected.

        Parameters
        ----------
        cluster_id : int
            The ID of the cluster of which to compute the typical.

        Returns
        -------
        None.
        '''
        scores = [-1 for i in range(len(self.structures))]
        indices = self.clusters[cluster_id]
        for i in indices:
            score = 0
            for j in indices:
                if j != i and self.score_matrix[i,j] >= 0:
                    score += self.score_matrix[i,j]
            scores[i] = score
        scores = np.array(scores)
        self.typicals[cluster_id] = self.structures[np.argmax(scores)]
    
    