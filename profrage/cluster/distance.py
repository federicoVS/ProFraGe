# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:42:01 2021

@author: Federico van Swaaij
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from cluster.Cluster import Cluster

from structure.representation import USR
from utils.ProgressBar import ProgressBar

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
    
    def __init__(self, structures, dist_matrix, k=100, g_delta=16, to_invert=False, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structures
            The list of structures to cluster.
        dist_matrix : numpy.ndarray
            The distance matrix.
        k : int, optional
            The number of clusters. The default is 100.
        g_delta : int, optional
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
    tol : float
        The tolerance.
    reg_covar : float
        Non-negative regularization.
    max_iter : int
        The maximum number of iterations.
    n_init : int
        The number of initializations.
    """
    
    def __init__(self, structures, features, k=100, tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=10, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
            The structures to cluster.
        features : numpy.ndarray
            The matrix of features. It should have shape n_samples, n_features.
        k : int, optional
            The number of clusters. The default is 100.
        tol : float, optional
            The tolerance. The default is 1e-3.
        reg_covar : float, optional
            Regularization added to the covariance matrix to ensure its its positiveness.
        max_iter : int, optional
            The maximum number of iterations. The default is 100.
        n_init : int, optional
            The number of initializations. The default is 10.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(GMM, self).__init__(structures, verbose)
        self.features = features
        self.k = k
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
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
        gm = GaussianMixture(n_components=self.k, tol=self.tol, reg_covar=self.reg_covar, max_iter=self.max_iter, n_init=self.n_init, verbose=self.verbose)
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
                
class Agglomerative(Cluster):
    """
    Performs hierarchical clustering of the given structures.
    
    With its default parameters, no fixed number of clusters is assumed.
    
    Attributes
    ----------
    features : numpy.ndarray
        The matrix of features.
    k : int
        The number of clusters.
    linkage: str
        The linkage criterion to use
    distance_threshold : float
        The linkage distance threshold above which clusters will not be merged
    """
    
    def __init__(self, structures, features, k=None, linkage='ward', distance_threshold=10, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structures
            The structures to cluster.
        features : numpy.ndarray
            The matrix of features.
        k : int, optional
            The number of clusters. The default is None.
        linkage : str, optional
            The linkage criterion to use. The default is 'ward'.
        distance_threshold : float, optional
            The linkage distance threshold above which clusters will not be merged
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(Agglomerative, self).__init__(structures, verbose)
        self.features = features
        self.k = k
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        if self.verbose:
            print('Clustering...')
        # Cluster using the hierarchical algorithm
        aggcl = AgglomerativeClustering(n_clusters=self.k, linkage=self.linkage, compute_full_tree=True, distance_threshold=self.distance_threshold)
        aggcl.fit(self.features)
        # Retrieve the clusters
        for i in range(len(aggcl.labels_)):
            cluster_id = aggcl.labels_[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)
                
class KUSR(Cluster):
    """
    A static version of K-Means clustering using USR score.
    
    A specified k structures are selected to be centers of the clusters they represent, which is based
    on the median USR score similarity, based on all-against-all comparison. Once the centers are computed,
    the other strucutures are assigned to the cluster with the smallest USR cosine distance to them.
    
    Attributes
    ----------
    k : int
        The number of clusters.
    _centers : list of int
        The list of the cluster centers. Entry i maps to the center of cluster i.
    _features : numpy.ndarray
        The USR-feature matrix.
    """
    
    def __init__(self, structures, k=100, verbose=False, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structures : Bio.PDB.Structure
            The structures to cluster.
        k : int, optional
            The number of clusters. The default is 100.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(KUSR, self).__init__(structures, verbose)
        self.k = k
        self._centers = [-1 for i in range(k)]
        self._features = None
        
    def _get_best_center(self, idx):
        """
        Get the best center for the specified index to belong to.

        Parameters
        ----------
        idx : int
            The index of the structure to place.

        Returns
        -------
        best_center_idx : int
            The index of the best center.
        """
        best_center_idx, best_score = -1, -1
        for i in range(self.k):
            center_idx = self._centers[i]
            score = USR.get_similarity_score(self._features[idx], self._features[center_idx])
            if score > best_score:
                best_center_idx = i
                best_score = score
        return best_center_idx
        
    def _compute_features(self):
        """
        Compute the USR-feature matrix.

        Returns
        -------
        None.
        """
        n = len(self.structures)
        self._features = np.zeros(shape=(n, USR.get_n_features()))
        for i in range(n):
            usr = USR(self.structures[i])
            self._features[i,:] = usr.get_features()
        
    def _compute_centers(self):
        """
        Compute and define the centers of the clusters.

        Returns
        -------
        None.
        """
        # Define data structures
        n = len(self.structures)
        scores = [[i, 0] for i in range(n)]
        # Compute the scores for each structure
        for i in range(n):
            for j in range(n):
                if i != j:
                    score = USR.get_similarity_score(self._features[i], self._features[j])
                    scores[i][0] = i
                    scores[i][1] += score
        # Sort the scores and get the best n_centers scores
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for i in range(self.k):
            self._centers[i] = scores[i][0]
            
    def best_representative(self, cluster_id):
        """
        Return the best representative for the specified cluster.
        
        In this case, the best representative corresponds to the cluster center.

        Parameters
        ----------
        cluster_id : int
            The cluster ID.

        Returns
        -------
        Bio.PDB.Structure
            The representative structure of the specified cluster.
        """
        s_id = self._centers[cluster_id]
        return self.structures[s_id]
    
    def cluster(self):
        """
        Perform the clustering.

        Returns
        -------
        None.
        """
        # Compute the features
        self._compute_features()
        # Compute the centers
        self._compute_centers()
        # Initialize the clusters
        for i in range(self.k):
            self.clusters[i] = []
            self.clusters[i].append(self._centers[i])
        n = len(self.structures)
        progress_bar = ProgressBar(n)
        if self.verbose:
            print('Clustering...')
            progress_bar.start()
        for i in range(n):
            if self.verbose:
                progress_bar.step()
            if i not in self._centers:
                cluster_id = self._get_best_center(i)
                self.clusters[cluster_id].append(i)
        if self.verbose:
            progress_bar.end()
            
