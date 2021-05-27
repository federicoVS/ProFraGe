# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:55:07 2021

@author: Federico van Swaaij
"""

import numpy as np
import matplotlib.pyplot as plt

from structure.representation import USR

class Cluster:
    """
    The abstract cluster class.
    
    Attributes
    ----------
    clusters : dict of int -> list of int
        The representation of the clusters as a mapping from a cluster ID to the index of the
        structures associeted with it.
    structures : list of Bio.PDB.Structure
        The list of structures.
    verbose : bool
        Whether to print progress information.
    """
    
    def __init__(self, structures, verbose=False):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures to cluster.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        self.clusters = {}
        self.structures = structures
        self.verbose = verbose
        
    def __len__(self):
        """
        Return the number of clusters.

        Returns
        -------
        int
            The number of clusters.
        """
        return len(self.clusters)
    
    def get_clustered_structure(self, cluster_id, index):
        """
        Return the specified structure from the specified cluster.

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
        """
        s_id = self.clusters[cluster_id][index]
        structure = self.structures[s_id]
        return structure
    
    def print_clusters_statistics(self):
        """
        Print clusters statistics.

        Returns
        -------
        None.
        """
        freqs = {}
        for cluster_id in self.clusters:
            freqs[cluster_id] = len(self.clusters[cluster_id])
            s_id = self.best_representative(cluster_id).get_id()
            print(f'Cluster {cluster_id} has representative {s_id} and size {len(self.clusters[cluster_id])}')
    
    def show_clusters(self):
        """
        Plot the size of each cluster as an histogram.
        
        The plot should indicate whether the clusters are balanced.

        Returns
        -------
        None.
        """
        freqs = {}
        for cluster_id in self.clusters:
            freqs[cluster_id] = len(self.clusters[cluster_id])
        plt.bar(freqs.keys(), freqs.values(), 1.0, color='b')
        plt.show()
        
    def best_representative(self, cluster_id):
        """
        Select the best representative from the specified cluster based on USR similarity score.

        Parameters
        ----------
        cluster_id : int
            The cluster ID.

        Returns
        -------
        structure : Bio.PDB.Structure
            The best representative structure for the cluster.
        """
        # Check if there is only one structure in the cluster
        if len(self.clusters[cluster_id]) == 1:
            s_id = self.clusters[cluster_id][0]
            return self.structures[s_id]
        # Compute the scores
        features = np.zeros(shape=(len(self.clusters[cluster_id]),USR.get_n_features()))
        for i in range(len(self.clusters[cluster_id])):
            i_idx = self.clusters[cluster_id][i]
            usr = USR(self.structures[i_idx])
            features[i,:] = usr.get_features()
        # Get the best representative
        best_idx, best_score = -1, -1
        for i in range(len(self.clusters[cluster_id])):
            median_score = []
            for j in range(len(self.clusters[cluster_id])):
                if j != i:
                    median_score.append(USR.get_similarity_score(features[i], features[j]))
            median_score = np.median(median_score)
            if median_score > best_score:
                best_idx = i
                best_score = median_score
        s_id = self.clusters[cluster_id][best_idx]
        return self.structures[s_id]
        
    def cluster(self):
        """
        Perform the clustering. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        """
        pass