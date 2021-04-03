# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:42:01 2021

@author: FVS
"""

import numpy as np
from cluster.Cluster import Cluster
from utils.ProgressBar import ProgressBar

class KMeans(Cluster):
    
    def __init__(self, structures, k, dist_matrix, verbose=False):
        super(KMeans, self).__init__(structures, verbose)
        self.k = k
        self.dist_matrix = dist_matrix
        
    def cluster(self):
        # Initialize
        self.clusters = {}
        progress_bar = ProgressBar()
        if self.verbose:
            print('Clustering...')
            progress_bar.start()