# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:42:01 2021

@author: FVS
"""

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from cluster.Cluster import Cluster

class Spectral(Cluster):
    """
    Perform spectral clustering based on the specified distance matrix.
    
    The clustering is then based on either the Laplacian matrix - representing a weighted graph - or on
    the similarity matrix - built using the Gaussian kernel.
    
    Attributes
    ----------
    dist_matrix : numpy.ndarray
        The (symmetric) distance matrix.
    k : int
        The number of clusters.
    g_delta : int
        The width of the Gaussian kernel.
    to_invert : bool
        Whether to print progress information.
    """
    
    def __init__(self, structures, dist_matrix, k, g_delta=16, to_invert=False, verbose=False):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structures
            The list of structures to cluster.
        dist_matrix : numpy.ndarray
            The distance matrix.
        k : int
            The number of clusters.
        g_delta : TYPE, optional
            The width of thr Gaussian kernel. It is only used if m_type is 'similarity'. The default is 16.
        to_invert : bool, optional
            Whether to invert the values of the distance matrix. The default is False.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(Spectral, self).__init__(structures, verbose)
        self.dist_matrix = dist_matrix.astype(np.float64)
        self.k = k
        self.g_delta = g_delta
        self.to_invert = to_invert
    
    def _get_similarity(self):
        """
        Compute the similarity matrix based on the distance matrix.
        
        Source
        ------
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

        Returns
        -------
        scipy.sparse.csr_matrix
            The similarity matrix.
        """
        matrix = self.dist_matrix
        if self.to_invert:
            matrix[matrix==0] = 1e-09
            matrix = np.reciprocal(matrix)
        return csr_matrix(np.exp(-matrix**2/(2*self.g_delta**2)))

    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        if self.verbose:
            print('Clustering...')
        matrix = self._get_similarity()
        sc = SpectralClustering(n_clusters=self.k, random_state=0, affinity='precomputed', assign_labels='discretize', n_jobs=4)
        sc.fit(matrix)
        # Retrieve clusters
        for i in range(sc.labels_.shape[0]):
            cluster_id = sc.labels_[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
            else:
                self.clusters[cluster_id].append(i)

class KMean(Cluster):
    """
    Perform KMeans clustering based on the given coordinates matrix.
    
    The sklearn.cluster.KMeans algorithm is used.
    
    Attributes
    ----------
    coords_matrix : numpy.ndarray
        The matrix representing the coordinates.
    k : int
        The number of clusters.
    """
    
    def __init__(self, structures, coords_matrix, k, verbose=False):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The list of structures to cluster.
        coords_matrix : numpy.ndarray
            The coordinates matrix.
        k : int
            The number of clusters.
        verbose : bool, optional
            Whether to pring progress information. The default is False.

        Returns
        -------
        None.
        """
        super(KMean, self).__init__(structures, verbose)
        self.coords_matrix = coords_matrix
        self.k = k
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        if self.verbose:
            print('Clustering...')
        km = KMeans(n_clusters=self.k, verbose=self.verbose)
        km.fit(self.coords_matrix)
        # Retrieve clusters
        for i in range(km.labels_.shape[0]):
            cluster_id = km.labels_[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
            else:
                self.clusters[cluster_id].append(i)
                
class GMM(Cluster):
    """
    Perform clustering using Gaussian mixture models.
    
    The sklearn.mixture.GaussianMixture algorithm is used.
    
    Attributes
    ----------
    features : numpy.ndarray
        The matrix of features. It should have shape n_samples, n_features.
    k : int
        The number of clusters.
    n_init : int
        The number of initializations.
    """
    
    def __init__(self, structures, features, k, reg_covar=1e-6, n_init=1, verbose=False):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        features : numpy.ndarray
            The matrix of features. It should have shape n_samples, n_features.
        k : int
            The number of clusters.
        reg_covar : float, optional
            Regularization added to the covariance matrix to ensure its its positiveness.
        n_init : int, optional
            The number of initializations. The default is 1.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(GMM, self).__init__(structures, verbose)
        self.features = features
        self.k = k
        self.reg_covar = reg_covar
        self.n_init = n_init
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        if self.verbose:
            print('Clustering...')
        gm = GaussianMixture(n_components=self.k, reg_covar=self.reg_covar, n_init=self.n_init, verbose=self.verbose)
        gm.fit(self.features)
        labels = gm.predict(self.features)
        # Retrieve clusters
        for i in range(labels.shape[0]):
            cluster_id = labels[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)
            else:
                self.clusters[cluster_id].append(i)

