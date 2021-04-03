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
    clusters : dict of int -> list of Bio.PDB.Structure
        The representation of the clusters as a mapping from a cluster ID to the structures
        associated with it.
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
        
    def get_clusters(self):
        '''
        Returns the generated clusters.

        Returns
        -------
        dict of int -> list of Bio.PDB.Structure
            The clusters.
        '''
        return self.clusters
    
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
        Bio.PDB.Structure
            The structure.
        '''
        return self.structures[self.clusters[cluster_id][index]]
    
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