# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:15:24 2021

@author: FVS
"""

import os
import shutil
import argparse
import numpy as np
from fragments.USR import USR
from cluster.distance import Spectral, KMean, GMM
from cluster.greedy import SeqAlign, CASuperImpose
from utils.io import get_files, to_pdb, from_mmtf
from utils.tm_align import tm_align
from utils.ProgressBar import ProgressBar

def pipeline(method, data_get, fragments_dir, out_dir, tm_align_dir, dist_matrix_file, k, rmsd_thr=0.5, score_thr=10, length_pct_thr=0.5, verbose=False, show=False):
    '''
    This pipeline takes filtered fragments from the specified directory and clusters them using the
    specified algorithm.

    Parameters
    ----------
    method : str
        The clustering method to apply to the structures. The options are [spectral, kmeans, gmm, seq_align, super_imp].
    data_get : str
        The data to use for the clustering. The options are [structural, tm_align, usr]. Option `structural`
        should be used with clustering methods `seq_align` and `super_imp`. Option `tm_align` should
        be used with clustering method `spectral`. Option `usr` should be used with clustering method
        `gmm`.
    fragments_dir : str
        The directory containing the fragments. The fragments are assumed to be stored in MMTF format.
    out_dir : str
        The directory where the clustered fragments will be saved.
    tm_align_dir : str
        The directory where the TM-align tool is to be found.
    dist_matrix_file : str
        The file storing the TM-score matrix. If the file does not exist, it will be computed here.
    k : int
        The number of clusters.
    rmsd_thr : float in [0,1], optional
        The RMSD threshold under which two structures are considered to be equal. The default is 0.5.
    score_thr : int, int
        The alignment score above which two structures are considered similar. The default is 10.
    length_pct_thr : float in [0,1], optional
        The percentage of length two structures must share to be considered similar. The default is 0.5.
    verbose : bool, optional
        Whether to print progress information. The default is False.
    show : bool, optional
        Whether to show a plot of the distribution of the clusters. The default is False.

    Returns
    -------
    None.
    '''
    mmtfs = get_files(fragments_dir, ext='.mmtf')
    structures = [None for i in range(len(mmtfs))]
    for i in range(len(mmtfs)):
        structures[i] = from_mmtf(mmtfs[i])
    dist_matrix = None
    coords_matrix = None
    if data_get == 'tm_align':
        if os.path.isfile(dist_matrix_file):
            dist_matrix = np.load(dist_matrix_file)
        else:
            dist_matrix = tm_align(structures, tm_align_dir, 'tm_align', out_dir='../pdb/ids/', verbose=verbose)
    elif data_get == 'usr':
        coords_matrix = np.zeros(shape=(len(structures),12))
        progress_bar = ProgressBar()
        if verbose:
            print('Computing USR matrix...')
            progress_bar.start()
        for i in range(len(structures)):
            if verbose:
                progress_bar.step(i+1, len(structures))
            usr = USR(structures[i])
            usr.compute_all()
            coords_matrix[i,:] = usr.momenta
        if verbose:
            progress_bar.end()
    else:
        print('Invalid data.')
        return
    cluster = None
    if method == 'spectral':
        cluster = Spectral(structures, dist_matrix, k, to_invert=True, verbose=verbose)
    elif method == 'kmeans':
        cluster = KMean(structures, coords_matrix, k, verbose=verbose)
    elif method == 'gmm':
        cluster = GMM(structures, coords_matrix, k, n_init=5, verbose=verbose)
    elif method == 'seq_align':
        cluster = SeqAlign(structures, score_thr, length_pct_thr)
    elif method == 'super_imp':
        cluster = CASuperImpose(structures, rmsd_thr, length_pct_thr)
    else:
        print('Invalid clustering method.')
        return
    cluster.cluster()
    # Iterate over the clusters and get representative
    for cluster_id in range(len(cluster)):
        structure = cluster.get_clustered_structure(cluster_id, 0)
        shutil.copy(fragments_dir + structure.get_full_id()[0] + '.mmtf', out_dir + structure.get_full_id()[0] + '.mmtf')
        to_pdb(structure, structure.get_full_id()[0], out_dir='lol/')
    if verbose:
        cluster.print_clusters_statistics()
    if show:
        cluster.show_clusters()
    

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Clustering of fragments.')
    arg_parser.add_argument('method', type=str, help='The clustering method to apply to the fragments. The options are [spectral, kmeans, gmm, seq_align, super_imp].')
    arg_parser.add_argument('data_get', type=str, help='Which data to use for the clustering. The options are [structural, tm_align, usr].')
    arg_parser.add_argument('fragments_dir', type=str, help='The directory where the fragments are held in MMTF format.')
    arg_parser.add_argument('out_dir', type=str, help='The directory where the clustered fragments will be saved.')
    arg_parser.add_argument('--tm_align_dir', type=str, default=None, help='The directory where the TM-align tool is located.')
    arg_parser.add_argument('--dist_matrix_file', type=str, default='../pdb/ids/tm_align.npy', help='The directory where the TM-align distance matrix is located. The default is ../pdb/ids/tm_align.npy')
    arg_parser.add_argument('--k', type=int, default=30, help='The number of clusters, should kNN or spectral clustering be chosen.')
    arg_parser.add_argument('--rmsd_thr', type=float, default=0.5, help='The RMSD threshold under which two fragments are considered similar. The default is 0.5.')
    arg_parser.add_argument('--score_thr', type=int, default=10, help='The alignment score threshold above which two fragments are considered similar. The default is 10.')
    arg_parser.add_argument('--length_pct_thr', type=float, default=0.5, help='The percentage of length two fragments must share in order to be considered similar. The default is 0.5.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    arg_parser.add_argument('--show', type=bool, default=False, help='Whether to show a plot of the distribution of the clusters. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    pipeline(args.method, args.data_get, args.fragments_dir, args.out_dir, args.tm_align_dir, args.dist_matrix_file, args.k, rmsd_thr=args.rmsd_thr, score_thr=args.score_thr, length_pct_thr=args.length_pct_thr, verbose=args.verbose, show=args.show)
    