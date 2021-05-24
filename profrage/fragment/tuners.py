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
from utils.stride import single_stride, get_composition
from utils.io import get_files, from_pdb, to_pdb
from utils.ProgressBar import ProgressBar

def leiden_agglomerative_gridsearch(train_set_dir, test_set_dir, cmap_train_dir, cmap_test_dir, stride_dir, params, lm_score_thr=0.6, to_show=3, verbose=False):
    """
    Perform GridSearch based on the Leiden mining algorithm coupled with Agglomerative clustering.

    Parameters
    ----------
    train_set_dir : str
        The directory holding the PDB files for the training phase.
    test_set_dir : str
        The directory holding the PDB files for the testing phase.
    cmap_train_dir : str
        The directory holding the CMAP files for the training set.
    cmap_test_dir : str
        The directory holding the CMAP files for the test set.
    stride_dir : str
        The directory holding the Stride tool.
    params : dict of str -> Any
        The parameters to try for the Leiden algorithm.
    lm_score_thr : float in [0,1], optional
        The USR threshold score to be used in the language model. The default is 0.6.
    to_show : int, optional
        The best configurations to show. The default is 3.
    verbose : bool optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    total_len = 1
    for p in params:
        total_len *= len(params[p])
    search_space = (dict(zip(params, x)) for x in itertools.product(*params.values()))
    progress_bar = ProgressBar(total_len)
    if verbose:
        progress_bar.start()
    best_params = []
    for param_config in search_space:
        if verbose:
            progress_bar.step()
        # Train, filter, and cluster the model
        pdbs = get_files(train_set_dir, ext='.pdb')
        fragments = []
        for pdbf in pdbs:
            pdb_id = os.path.basename(pdbf)[:-4]
            cmapf = cmap_train_dir + pdb_id + '.cmap'
            structure = from_pdb(pdb_id, pdbf, quiet=True)
            model = LeidenMiner(structure, cmapf, **param_config)
            model.mine()
            frags = model.get_fragments()
            for f in frags:
                fragments.append(f)
        filtered = []
        if os.path.exists('lhg-tmp/'):
            os.rmdir('lhg-tmp/')
        if not os.path.exists('lhg-tmp/'):
            os.makedirs('lhg-tmp/')
        for frag in fragments:
            if in_range(frag) and is_spherical(frag) and is_compact(frag) and is_connected(frag):
                filtered.append(frag)
                to_pdb(frag, frag.get_id(), out_dir='lhg-tmp/')
        pdbs = get_files('lhg-tmp/', ext='.pkl')
        assignements = {}
        pre_clusters = {}
        representatives = []
        for pdb in pdbs:
            pdb_id = os.path.basename(pdb)[:-4]
            code_dict = single_stride(stride_dir, pdb)
            keys = get_composition(code_dict)
            assignements[pdb_id] = keys
        for pdb in pdbs:
            pdb_id = os.path.basename(pdb)[:-4]
            keys = assignements[pdb_id]
            if keys is None:
                continue
            if keys not in pre_clusters:
                pre_clusters[keys] = []
            pre_clusters[keys].append(from_pdb(pdb_id, pdb))
        for keys in pre_clusters:
            structures = pre_clusters[keys]
            features = np.zeros(shape=(len(structures),USR.get_n_features()))
            if features.shape[0] < 2:
                continue
            for i in range(len(structures)):
                usr = USR(structures[i], ca_atoms=True)
                momenta = usr.get_features()
                features[i,:] = momenta
            aggl = Agglomerative(structures, features)
            aggl.cluster()
            for cluster_id in range(len(aggl)):
                rep = aggl.best_representative(cluster_id)
                freq = len(aggl.clusters[cluster_id])
                representatives.append((keys, rep, freq))
        # Mine fragments on the test set
        pdbs = get_files(test_set_dir, ext='.pdb')
        fragments = {}
        for pdbf in pdbs:
            pdb_id = os.path.basename(pdbf)[:-4]
            cmapf = cmap_test_dir + pdb_id + '.cmap'
            structure = from_pdb(pdb_id, pdbf, quiet=True)
            model = LeidenMiner(structure, cmapf, **param_config)
            model.mine()
            frags = model.get_fragments()
            fragments[pdb_id] = []
            for frag in frags:
                if in_range(frag) and is_spherical(frag) and is_compact(frag) and is_connected(frag):
                    to_pdb(frag, 'tmp-pdb')
                    code_dict = single_stride(stride_dir, 'tmp-pdb.pdb')
                    keys = get_composition(code_dict)
                    if keys is None:
                        continue
                    fragments[pdb_id].append((keys, frag))
                    os.remove('tmp-pdb.pdb')
        # Define language model
        lm = LanguageModel(representatives, lm_score_thr)
        lm.get_word_probs()
        for pdb_id in fragments:
            lm.compute_sentence_probs(pdb_id, fragments[pdb_id])
        best_params.append((lm.get_avg_plausibility(), param_config))
        if os.path.exists('lhg-tmp/'):
            os.rmdir('lhg-tmp/')
    best_configs = sorted(best_params, key=lambda x: x[0], reverse=True)[0:to_show]
    if verbose:
        progress_bar.end()
    for best_config in best_configs:
        print(f'Probability: {best_config[0]}, Parameters: {best_config[1]}')
    