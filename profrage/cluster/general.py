# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:42:01 2021

@author: FVS
"""

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.cluster import SpectralClustering
from cluster.Cluster import Cluster
from utils.ProgressBar import ProgressBar

class Spectral(Cluster):
    '''
    Performs spectral clustering based on the specified distance matrix. The clustering is
    then based on either the Laplacian matrix - representing a weighted graph - or on the similarity
    matrix - built using the Gaussian kernel.
    
    Attributes
    ----------
    k : int
        The number of clusters.
    dist_matrix : numpy.ndarray
        The (symmetric) distance matrix.
    dist_thr : float in [0,1]
        The threshold above which two structures are not considered similar. In terms of graphs, it
        means there is no edge between structure i and structure j.
    m_type : list of str in ['laplace', 'similarity']
        The type of the affinity used for the clustering.
    g_delta : int
        The width of the Gaussian kernel.
    '''
    
    def __init__(self, structures, k, dist_matrix,  dist_thr, m_type='laplace', g_delta=16, verbose=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structures
            The list of structures to cluster.
        k : int
            The number of cluster.
        dist_matrix : numpy.ndarray
            The distance matrix.
        dist_thr : float in [0,1]
            he threshold above which two structures are not considered similar.
        m_type : str, optional
            The affinity matrix to compute. The default is 'laplace'.
        g_delta : TYPE, optional
            The width of thr Gaussian kernel. It is only used if m_type is 'similarity'. The default is 16.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        '''
        super(Spectral, self).__init__(structures, verbose)
        self.k = k
        self.dist_matrix = dist_matrix
        self.dist_thr = dist_thr
        self.m_type = m_type
        self.g_delta = g_delta
        
    def get_laplacian(self):
        '''
        Computes the Laplacian matrix based on the similarity one. In this setting, the matrix
        represents a weighted graph.

        Returns
        -------
        scipy.sparse.csr_matrix
            The Laplacian matrix.
        '''
        # Apply cutoff
        cutoff_inds = self.dist_matrix > self.dist_thr
        self.dist_matrix[cutoff_inds] = 0
        # Compute Laplacian
        A = np.reciprocal(self.dist_matrix)
        D = np.diag(A.sum(axis=1))
        L = D - A
        return csr_matrix(L)
    
    def get_similarity(self):
        '''
        Computes the similarity matrix based on the distance matrix.
        
        Source
        ------
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

        Returns
        -------
         scipy.sparse.csr_matrix
            The similarity matrix.
        '''
        return csr_matrix(np.exp(-self.dist_matrix**2/(2*self.g_delta**2)))

    def cluster(self):
        '''
        Performs the clustering.

        Returns
        -------
        None.
        '''
        matrix = None
        if self.m_type == 'laplace':
            matrix = self.get_laplacian()
        elif self.m_type == 'similarity':
            matrix = self.get_similarity(self.g_delta)
        else:
            return
        sc = SpectralClustering(n_clusters=self.k, random_state=0, affinity='precomputed', assign_labels='discretize', n_jobs=4, verbose=self.verbose)
        sc.fit(matrix)
        # Retrieve clusters
        for i in range(sc.labels_.shape[0]):
            cluster_id = sc.labels_[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(self.structures[i])
            else:
                self.clusters[cluster_id].append(self.structures[i])

class KNN(Cluster):
    '''
    Performs KNN clustering.
    
    Attributes
    ----------
    k : int
        The number of clusters.
    dist_matrix : numpy.ndarray
        The (symmetric) distance matrix.
    '''
    
    def __init__(self, structures, k, dist_matrix, verbose=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structures
            The list of structures to cluster.
        k : int
            The number of clusters.
        dist_matrix : numpy.ndarray
            The distance matrix.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        '''
        super(KNN, self).__init__(structures, verbose)
        self.k = k
        self.dist_matrix = dist_matrix
    
    def cluster(self):
        n = len(self.structures)
        cluster_id = 1
        progress_bar = ProgressBar()
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step(i, n)
        # Retrieve clusters
        for i in range(n):
            a=1
        