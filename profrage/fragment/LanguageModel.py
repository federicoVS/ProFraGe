# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:37:33 2021

@author: Federico van Swaaij
"""

import numpy as np

from structure.representation import USR

class LanguageModel:
    """
    Implement a language model where protein fragments act as words.
    
    The frequencies for each word extracted from a training set is counted and compared to those of
    the testing set.
    
    The training and testing set are represented with clusters. All elements of a single cluster
    represent single words. This is because no two fragments are exactly equal, so clustering is introduced
    based on USR-similarity.
    
    Attributes
    ----------
    sentence_probs : dict of str -> float in [0,1]
        The probability that a protein is made of its fragments.
    word_probs : dict of int -> float in [0,1]
        The probabilities of single words. These are based on the training set.
    cluster : cluster.Cluster.Cluster
        The cluster object holding the fragments.
    score_thr : float in [0,1]
        The USR score threshold above which two words (fragments) are considered to be similar.
    """
    
    def __init__(self, cluster, score_thr):
        """
        Initialize the class.

        Parameters
        ----------
        cluster : cluster.Cluster.Cluster
            The cluster object holding the fragments.
        score_thr : float in [0,1]
            The USR score threshold.

        Returns
        -------
        None.
        """
        self.sentence_probs = {}
        self.word_probs = {}
        self.cluster = cluster
        self.score_thr = score_thr
        
    def compute_sentence_probs(self, pdb_id, fragments, ep=1e-3):
        """
        Compute the probability that a protein is made of the specified fragments.

        Parameters
        ----------
        pdb_id : str
            The ID of the protein.
        fragments : list of Bio.PDB.Structure
            The fragments composing the structure. Note that these fragments should be unclustered.
        ep : float in [0,1], optional
            The probability to assign to fragments that are not found in the training data. The default is 1e-3.

        Returns
        -------
        None.
        """
        probs = []
        for fragment in fragments:
            f_momenta = USR(fragment).get_features()
            best_score, best_cluster_id = -1, -1
            for cluster_id in len(self.cluster):
                rep = self.cluster.best_representative(cluster_id)
                r_momenta = USR(rep).get_features()
                score = USR.get_similarity_score(f_momenta, r_momenta)
                if score > self.score_thr and score > best_score:
                    best_cluster_id = cluster_id
                    best_score = score
            if best_cluster_id == -1:
                probs.append(ep)
            else:
                probs.append(self.word_probs[best_cluster_id])
        self.sentence_probs[pdb_id] = 1
        for p in probs:
            self.sentence_probs[pdb_id] *= p
    
    def get_word_probs(self):
        """
        Compute the probabilities for words to appear in the dataset.
        
        Note that for this operation should be performed on the training set.

        Returns
        -------
        None.
        """
        total = len(self.cluster.structures)
        for cluster_id in len(self.cluster):
            self.word_probs[cluster_id] = len(self.cluster.clusters[cluster_id])/total
            
    def get_avg_plausubility(self):
        """
        Return the average plausibility for the language model.

        Returns
        -------
        float in [0,1]
            The average probability.
        """
        probs = []
        for pdb_id in self.sentence_probs:
            probs.append(self.sentence_probs[pdb_id])
        return np.mean(np.array(probs))
            