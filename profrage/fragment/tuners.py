import os

import shutil

import itertools

from cluster.greedy import USRCluster, StrideCluster, AtomicSuperImpose
from fragment.mine import LeidenMiner
from fragment.filtering import in_range
from fragment.LanguageModel import LanguageModel
from utils.io import get_files, from_pdb, to_pdb
from utils.ProgressBar import ProgressBar

def leiden_gridsearch(train_set_dir, test_set_dir, cmap_train_dir, cmap_test_dir, stride_dir, leiden_params, first_cluster_params, second_cluster_params, third_cluster_params, range_params, lm_rmsd_thr=2.5, train_size=500, to_show=3, verbose=False, write_stats=False):
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
    leiden_params : dict of str -> Any
        The parameters to try for the Leiden algorithm.
    first_cluster_params : dict of str -> Any
        The parameters for the first level clustering (Stride).
    second_cluster_params : dict of str -> Any
        The parameters for the second level clustering (USR).
    third_cluster_params : dict of str -> Any
        The parameters for the third level clustering (Super Imposition).
    range_params : dict of str -> Any
        The parameters for the range filtering. Note that just the `lower` key is needed.
    lm_rmsd_thr : float, optional
        The RMSD threshold score to be used in the language model. The default is 2.5.
    train_size : int, optional
        The number of training sample to use. The reason for the limit is the fact that the full pipeline can be quite time-consuming.
        The default is 500.
    to_show : int, optional
        The best configurations to show. The default is 3.
    verbose : bool optional
        Whether to print progress information. The default is False.
    write_stats : bool, optional
        Whether to write cluster statistics on a file. The default is False.

    Returns
    -------
    None.
    """
    # Compute total number of permutations
    total_len, counter = 1, 1
    for p in leiden_params:
        total_len *= len(leiden_params[p])
    search_space = (dict(zip(leiden_params, x)) for x in itertools.product(*leiden_params.values()))
    best_params = []
    # Start the grid search
    for param_config in search_space:
        if verbose:
            print(f'Configuration {counter}/{total_len}')
            counter += 1
        # Create the output directory
        if not os.path.exists('lhg-tmp/'):
            os.makedirs('lhg-tmp/')
        # Train, filter, and cluster the model
        pdbs = sorted(get_files(train_set_dir, ext='.pdb'))[0:train_size]
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
            for frag in frags:
                if in_range(frag, **range_params):
                    to_pdb(frag, frag.get_id(), out_dir='lhg-tmp/')
        if verbose:
            progress_bar.end()
        all_structures = []
        pre_clusters = {}
        pdbs = get_files('lhg-tmp/', ext='.pdb')
        if len(pdbs) == 0:
            shutil.rmtree('lhg-tmp/')
            continue # no fragments have been generated
        for pdb in pdbs:
            pdb_id = os.path.basename(pdb)[:-4]
            all_structures.append(from_pdb(pdb_id, pdb, quiet=True))
        representatives = []
        if verbose:
            print('Clustering training fragments...')
        # First level clustering
        first_clualg = StrideCluster(all_structures, stride_dir, 'lhg-tmp/', **first_cluster_params)
        first_clualg.cluster('greedy')
        for first_cluster_id in range(len(first_clualg)):
            pre_clusters[first_cluster_id] = []
            for idx in first_clualg.clusters[first_cluster_id]:
                s = first_clualg.structures[idx]
                pre_clusters[first_cluster_id].append(s)
        # Second level clustering
        for keys in pre_clusters:
            pre_structures = pre_clusters[keys]
            second_clualg = USRCluster(pre_structures, **second_cluster_params)
            second_clualg.cluster('greedy')
            for second_cluster_id in range(len(second_clualg)):
                structures = []
                for second_idx in second_clualg.clusters[second_cluster_id]:
                    structures.append(second_clualg.structures[second_idx])
                # Third level clutering
                third_clualg = AtomicSuperImpose(structures, **third_cluster_params)
                third_clualg.cluster('optim')
                for third_cluster_id in range(len(third_clualg)):
                    structure = third_clualg.best_representative(third_cluster_id)
                    representatives.append((structure, len(third_clualg.clusters[third_cluster_id])))
                if write_stats:
                    write_cluster_stats(str(keys)+str(second_cluster_id), third_clualg)
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
            for frag in frags:
                if in_range(frag, **range_params):
                    fragments[pdb_id].append(frag)
        if verbose:
            progress_bar.end()
        # Define language model
        lm = LanguageModel(representatives, lm_rmsd_thr)
        lm.get_word_probs()
        progress_bar = ProgressBar(len(fragments))
        if verbose:
            print('Computing words plausibility...')
            progress_bar.start()
        for pdb_id in fragments:
            if verbose:
                progress_bar.step()
            lm.compute_sentence_probs(pdb_id, fragments[pdb_id], ep=1/(10*len(get_files('lhg-tmp/', ext='.pdb'))))
        if verbose:
            progress_bar.end()
        avg_plausibility, var_plausibility, median_plausibility = lm.get_full_plausibility()
        best_params.append((median_plausibility, param_config))
        if verbose:
            print(f'Configuration {param_config} has Avg: {avg_plausibility}, Var: {var_plausibility}, Median: {median_plausibility}')
        if os.path.exists('lhg-tmp/'):
            shutil.rmtree('lhg-tmp/')
    best_configs = sorted(best_params, key=lambda x: x[0], reverse=True)[0:to_show]
    for best_config in best_configs:
        print(f'Median Probability: {best_config[0]}, Parameters: {best_config[1]}')

def write_cluster_stats(full_id, clualg):
    file = open('lhg-cluster-log', 'a')
    for cluster_id in range(len(clualg)):
        total = len(clualg.clusters[cluster_id])
        file.write(str(full_id) + str(cluster_id) + ": " + str(total) + "\n")
    file.close()