# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:22:23 2021

@author: Federico van Swaaij
"""
import os
import itertools
import numpy as np

from cluster.distance import Agglomerative
from fragment.mine import LeidenMiner
from fragment.filtering import in_range, is_spherical, is_compact, is_connected
from fragment.LanguageModel import LanguageModel
from structure.representation import USR
from utils.io import get_files, from_pdb

def leiden_hierarchy_grid(train_set_dir, test_set_dir, cmap_dir, params, lm_score_thr):
    search_space = (dict(zip(params, x)) for x in itertools.product(*params.values()))
    best_params = []
    for param_config in search_space:
        # Train, filter, and cluster the model
        pdbs = get_files(train_set_dir, ext='.pdb')
        fragments = []
        for pdbf in pdbs:
            pdb_id = os.path.basename(pdbf)[:-4]
            cmapf = cmap_dir + pdb_id + '.cmap'
            structure = from_pdb(pdb_id, pdbf, quiet=True)
            model = LeidenMiner(structure, cmapf, **param_config)
            model.mine()
            frags = model.get_fragments()
            for f in frags:
                fragments.append(f)
        filtered = []
        for frag in fragments:
            if in_range(frag) and is_spherical(frag) and is_compact(frag) and is_connected(frag):
                filtered.append(frag)
        momenta = np.zeros(shape=(len(filtered),USR.get_n_features()))
        for i in range(len(filtered)):
            usr = USR(filtered[i])
            momenta[i,:] = usr.get_features()
        cluster = Agglomerative(filtered, momenta)
        cluster.cluster()
        # Mine fragments on the test set
        pdbs = get_files(test_set_dir, ext='.pdb')
        fragments = {}
        for pdbf in pdbs:
            pdb_id = os.path.basename(pdbf)[:-4]
            cmapf = cmap_dir + pdb_id + '.cmap'
            structure = from_pdb(pdb_id, pdbf, quiet=True)
            model = LeidenMiner(structure, cmapf, **param_config)
            model.mine()
            fragments[pdb_id] = model.get_fragments()
        # Define language model
        lm = LanguageModel(cluster, lm_score_thr)
        lm.get_word_probs()
        for pdb_id in fragments:
            lm.compute_sentence_probs(pdb_id, fragments[pdb_id])
        best_params.append((lm.get_avg_plausubility(), param_config))
    best_configs = sorted(best_params, key=lambda x: x[0], reverse=True)[0:5]
    for best_config in best_configs:
        print(f'Probability: {best_config[0]}, Parameters: {best_config[1]}')
    