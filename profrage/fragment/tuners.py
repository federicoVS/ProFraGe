# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:22:23 2021

@author: Federico van Swaaij
"""
import os
import itertools
import numpy as np

from cluster.distance import Agglomerative
from cluster.greedy import USRCluster, StrideCluster, USRStrideCluster
from fragment.mine import LeidenMiner
from fragment.filtering import in_range, is_spherical, is_compact, is_connected
from fragment.LanguageModel import LanguageModel
from structure.representation import FullStride
from utils.stride import single_stride, get_composition, get_stride_frequencies
from utils.io import get_files, from_pdb, to_pdb
from utils.ProgressBar import ProgressBar

def leiden_agglomerative_gridsearch(train_set_dir, test_set_dir, cmap_train_dir, cmap_test_dir, stride_dir, cluster_name, miner_params, pre_cluster_params, cluster_params, range_params, spherical_params, compact_params, connected_params, lm_score_thr=0.6, to_show=3, verbose=False):
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
    cluster_name : str
        The name of the clustering algorithm to use.
        Valid names are: 'aggl', 'usrc', 'stridec', 'usrstridec'
    miner_params : dict of str -> Any
        The parameters to try for the Leiden algorithm.
    pre_cluster_params : dict of str -> Any
        The parameters for the pre-clustering (using Stride).
    cluster_params : dict of str -> Any
        The parameters for the Agglomerative clustering.
    range_params : dict of str -> Any
        The parameters for the range filtering. Note that just the `lower` key is needed.
    spherical_params : dict of str -> Any
        The parameters for the spherical filtering.
    compact_params : dict of str -> Any
        The parameters for the compact filtering.
    connected_params : dict of str -> Any
        The parameters for the connected components filtering.
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
    # Compute total number of permutations
    total_len, counter = 1, 1
    for p in miner_params:
        total_len *= len(miner_params[p])
    search_space = (dict(zip(miner_params, x)) for x in itertools.product(*miner_params.values()))
    best_params = []
    # Start the grid search
    for param_config in search_space:
        if verbose:
            print(f'Configuration {counter}/{total_len}')
            counter += 1
        # Train, filter, and cluster the model
        pdbs = get_files(train_set_dir, ext='.pdb')
        fragments = []
        progress_bar = ProgressBar(len(pdbs))
        if verbose:
            print('Generating fragments for training...')
            progress_bar.start()
        for pdbf in pdbs:
            if verbose:
                progress_bar.step()
            pdb_id = os.path.basename(pdbf)[:-4]
            cmapf = cmap_train_dir + pdb_id + '.cmap'
            structure = from_pdb(pdb_id, pdbf, quiet=True)
            model = LeidenMiner(structure, cmapf, **param_config)
            model.mine()
            frags = model.get_fragments()
            for f in frags:
                fragments.append(f)
        if verbose:
            progress_bar.end()
        if os.path.exists('lhg-tmp/'):
            os.rmdir('lhg-tmp/')
        if not os.path.exists('lhg-tmp/'):
            os.makedirs('lhg-tmp/')
        if verbose:
            print('Filtering training fragments...')
        # Modify the range parameters
        range_params['upper'] = param_config['max_size']
        # Filter the fragments
        for frag in fragments:
            if in_range(frag, **range_params) and is_spherical(frag, **spherical_params) and is_compact(frag, **compact_params) and is_connected(frag, **connected_params):
                to_pdb(frag, frag.get_id(), out_dir='lhg-tmp/')
        pdbs = get_files('lhg-tmp/')
        assignements = {}
        pre_clusters = {}
        representatives = []
        if verbose:
            print('Clustering training fragments...')
        for pdb in pdbs:
            pdb_id = os.path.basename(pdb)[:-4]
            stride_desc = single_stride(stride_dir, pdb)
            code_dict = get_stride_frequencies(stride_desc)
            keys = get_composition(code_dict, **pre_cluster_params)
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
            features = np.zeros(shape=(len(structures),FullStride.get_n_features()))
            if features.shape[0] < 2:
                continue
            for i in range(len(structures)):
                feat = FullStride(stride_dir, 'lhg-tmp/'+structures[i].get_id()+'.pdb').get_features()
                features[i,:] = feat
            clualg = None
            if cluster_name == 'aggl':
                clualg = Agglomerative(structures, features, **cluster_params)
            elif cluster_name == 'usrc':
                clualg = USRCluster(structures, **cluster_params)
            elif cluster_name == 'stridec':
                clualg = StrideCluster(structures, stride_dir, 'lhg-tmp/', **cluster_params)
            elif cluster_name == 'usrstridec':
                clualg = USRStrideCluster(structures, stride_dir, 'lhg-tmp/', **cluster_params)
            clualg.cluster()
            for cluster_id in range(len(clualg)):
                rep = clualg.best_representative(cluster_id)
                freq = len(clualg.clusters[cluster_id])
                representatives.append((keys, rep, freq))
        # Mine fragments on the test set
        pdbs = get_files(test_set_dir, ext='.pdb')
        progress_bar = ProgressBar(len(pdbs))
        if verbose:
            print('Generating fragments for validation...')
            progress_bar.start()
        fragments = {}
        for pdbf in pdbs:
            if verbose:
                progress_bar.step()
            pdb_id = os.path.basename(pdbf)[:-4]
            cmapf = cmap_test_dir + pdb_id + '.cmap'
            structure = from_pdb(pdb_id, pdbf, quiet=True)
            model = LeidenMiner(structure, cmapf, **param_config)
            model.mine()
            frags = model.get_fragments()
            fragments[pdb_id] = []
            if verbose:
                print('Filtering and clustering validation fragments...')
            for frag in frags:
                if in_range(frag) and is_spherical(frag) and is_compact(frag) and is_connected(frag):
                    to_pdb(frag, 'tmp-pdb')
                    stride_desc = single_stride(stride_dir, 'tmp-pdb.pdb')
                    code_dict = get_stride_frequencies(stride_desc)
                    keys = get_composition(code_dict, **pre_cluster_params)
                    if keys is None:
                        continue
                    fragments[pdb_id].append((keys, frag))
                    os.remove('tmp-pdb.pdb')
        if verbose:
            progress_bar.end()
        # Define language model
        lm = LanguageModel(representatives, lm_score_thr)
        lm.get_word_probs()
        progress_bar = ProgressBar(len(fragments))
        if verbose:
            print('Computing words plausibility...')
            progress_bar.start()
        for pdb_id in fragments:
            if verbose:
                progress_bar.step()
            lm.compute_sentence_probs(pdb_id, fragments[pdb_id])
        if verbose:
            progress_bar.end()
        best_params.append((lm.get_avg_plausibility(), param_config))
        if os.path.exists('lhg-tmp/'):
            os.rmdir('lhg-tmp/')
    best_configs = sorted(best_params, key=lambda x: x[0], reverse=True)[0:to_show]
    for best_config in best_configs:
        print(f'Probability: {best_config[0]}, Parameters: {best_config[1]}')
    