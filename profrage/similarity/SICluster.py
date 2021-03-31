# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:58:01 2021

@author: FVS
"""

import numpy as np
from similarity.SIComparator import SIComparator
from utils.misc import structure_length
from utils.ProgressBar import ProgressBar

class SICluster:
    '''
    Performs clustering of the structures based on the their super imposition.
    Each structure is matched against the others, with the ones falling within the RMSD
    threshold being added to its cluster.
    
    Attributes
    ----------
    structures : list of Bio.PDB.Structure
        The list of structures.
    rmsd_thr : float
        The RMSD threshold for which two proteins are considered to be similar.
    clusters : dict of int -> list of Bio.PDB.Structure
        The representation of the clusters as a dictionary mapping a cluster ID to a list
        of similar structures. Such structures are similar to the extent of the RMSD
        threshold.
    rmsd_matrix : numpy.ndarray
        The matrix holding the pairwise RMSD
    typicals : dict of int -> Bio.PDB.Structure
        A dictionary where the cluster ID points to the exemplary structure within the
        cluster.
    verbose : bool
        Whether to print progress information.
    '''
    
    def __init__(self, structures, rmsd_thr, verbose=False):
        '''
        Initializes the class with the specified structures.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures.
        rmsd_thr : float
            The RMSD threshold for which two proteins are considered to be similar.
        verbose : bool, optional
        Whether to print progress information. The default is False.

        Returns
        -------
        None.
        '''
        self.structures = structures
        self.rmsd_thr = rmsd_thr
        self.clusters = {}
        self.rmsd_matrix = None
        self.typicals = {}
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
    
    def get_typicals(self):
        '''
        Returns the typicals.

        Returns
        -------
        dict of int -> Bio.PDB.Structure
            The typicals.

        '''
        return self.typicals
        
    def cluster(self):
        '''
        Performs the clustering algorithm.
        
        Returns
        -------
        None.
        '''
        # Initialize/reset the clusters dictionary
        self.clusters = {}
        self.typicals = {}
        cluster_id = 1
        placed = [False for i in range(len(self.structures))]
        # Compute the RMSD matrix
        self._compute_rmsd_matrix()
        # Print progress (if verbose)
        progress_bar = ProgressBar()
        if self.verbose:
            progress_bar.start()
        # Iterate over structures to create the clusters
        for i in range(len(self.structures)):
            if self.verbose:
                progress_bar.start(i, len(self.structures))
            # Check if structure i already belongs to a cluster
            if not placed[i]:
                placed[i] = True
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
                for j in range(len(self.structures)):
                    if j != i:
                        if self.rmsd_matrix[i,j] <= self.rmsd_thr and self.rmsd_matrix[i,j] >= 0:
                            placed[j] = True
                            self.clusters[cluster_id].append(j)
                cluster_id += 1
        if self.verbose:
                progress_bar.end()
        # Select a typical out of each cluster, based on th overall RMSD score
        for cluster_id in self.clusters:
            self._compute_typical(cluster_id)
                        
    def _compute_rmsd_matrix(self):
        '''
        Computes the RMSD matrix for the structures.

        Returns
        -------
        None.
        '''
        # Reset the matrix
        self.rmsd_matrix = np.zeros(shape=(len(self.structures), len(self.structures)))
        # Fill matrix
        for i in range(len(self.structures)-1):
            for j in range(i+1, len(self.structures)):
                if structure_length(self.structures[i]) == structure_length(self.structures[j]):
                    sic = SIComparator(self.structures[i], self.structures[j])
                    sic.compare()
                    rmsd = sic.get_rmsd(self.structures[j])
                    self.rmsd_matrix[i,j] = rmsd
                    self.rmsd_matrix[j,i] = rmsd
                else:
                    self.rmsd_matrix[i,j] = -1
                    self.rmsd_matrix[j,i] = -1
                
    def _compute_typical(self, cluster_id):
        '''
        Computes the typical (i.e. exemplary structure) in the cluster specified by the ID.

        Parameters
        ----------
        cluster_id : int
            The ID of the cluster of which to compute the typical.

        Returns
        -------
        None.
        '''
        scores = np.zeros(len(self.structures))
        indices = self.clusters[cluster_id]
        for i in indices:
            scores[i] = self.rmsd_matrix[i,[indices]] - self.rmsd_matrix[i,i]
        self.typicals[cluster_id] = self.structures[np.argmin(scores)]
        