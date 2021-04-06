#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:55:07 2021

@author: FVS
"""

import matplotlib.pyplot as plt

class Cluster:
    '''
    A generic cluster class.
    
    Attributes
    ----------
    clusters : dict of int -> list of int
        The representation of the clusters as a mapping from a cluster ID to the index of the
        structures associeted with it.
    structures : list of Bio.PDB.Structure
        The list of structures.
    verbose : bool
        Whether to print progress information.
    '''
    
    def __init__(self, structures, verbose=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures to cluster.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        '''
        self.clusters = {}
        self.structures = structures
        self.verbose = verbose
        
    def __len__(self):
        '''
        Returns the number of clusters.

        Returns
        -------
        int
            The number of clusters.
        '''
        return len(self.clusters)
    
    def get_clustered_structure(self, cluster_id, index):
        '''
        Returns the specified structure from the specified cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster ID.
        index : int
            The index of the structure within the cluster.

        Returns
        -------
        structure : Bio.PDB.Structure
            The structure.
        '''
        s_id = self.clusters[cluster_id][index]
        structure = self.structures[s_id]
        if self.verbose:
            print(f'Cluster {cluster_id} has representative {structure.get_full_id()[0]}')
        return structure
    
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
        plt.bar(freqs.keys(), freqs.values(), 1.0, color='b')
        plt.show()
        
    def cluster(self):
        '''
        Performs the clustering.

        Returns
        -------
        None.
        '''
        pass