# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:15:24 2021

@author: FVS
"""

import os
import shutil
import argparse
import numpy as np
from fragments.Fragment import Fragment
from cluster.general import Spectral, KNN
from cluster.greedy import SeqAlign, CASuperImpose
from utils.misc import get_files
from utils.tm_align import tm_align

def pipeline(fragments_dir, out_dir, tm_align_dir, dist_matrix_file, method, rmsd_thr, tm_thr, score_thr, length_pct_thr, k, verbose=False):
    mmtfs = get_files(fragments_dir, ext='.mmtf')
    fragments = [None for i in range(len(mmtfs))]
    structures = [None for i in range(len(mmtfs))]
    mmtf_dict = {}
    for i in range(len(mmtfs)):
        frag = Fragment(mmtfs[i])
        struct = frag.get_fragment()
        fragments[i] = frag
        structures[i] = struct
        mmtf_dict[struct] = mmtfs[i]
    dist_matrix = None
    if method == 'spectral' or method == 'knn':
        if os.path.isfile(dist_matrix_file):
            dist_matrix = np.load(dist_matrix_file)
        else:
            dist_matrix = tm_align(fragments, tm_align_dir, 'tm_align', out_dir='../pdb/ids/', verbose=verbose)
    cluster = None
    if method == 'spectral':
        cluster = Spectral(structures, k, dist_matrix, tm_thr, verbose=verbose)
    elif method == 'knn':
        cluster = KNN(structures, k, dist_matrix, verbose=verbose)
    elif method == 'seq_align':
        cluster = SeqAlign(structures, score_thr, length_pct_thr)
    elif method == 'super_imp':
        cluster = CASuperImpose(structures, rmsd_thr, length_pct_thr)
    cluster.cluster()
    # Iterate over the clusters and get representative
    for cluster_id in range(len(cluster)):
        structure = cluster.get_clustered_structure(cluster_id, 0)
        shutil.copy(mmtf_dict[structure], out_dir + os.path.basename(mmtf_dict[structure])[:-5])
    

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Clustering of fragments')
    arg_parser.add_argument('--fragments_dir', type=str, default='../pdb/fragments/terms-filtered/', help='The directory where the fragments are held. The default is ../pdb/fragments/terms/')
    arg_parser.add_argument('--out_dir', type=str, default='../pdb/fragments/terms-clustered/', help='The directory where the clustered fragments will be saved. The default is ../pdb/fragments/terms-clustered/')
    arg_parser.add_argument('--tm_align_dir', type=str, default='../pfg-tm-align/', help='The directory where the TM-align tool is located. The default is ../pfg-tm-align/')
    arg_parser.add_argument('--dist_matrix_file', type=str, default='../pdb/ids/tm_align.npy', help='The directory where the TM-align distance matrix is located. The default is ../pdb/ids/tm_align.npy')
    arg_parser.add_argument('--method', type=str, default='spectral', help='The clustering method to apply to the fragments. The options are [spectral, knn, seq_align, super_imp]. The default is spectral.')
    arg_parser.add_argument('--rmsd_thr', type=float, default=0.5, help='The RMSD threshold under which two fragments are considered similar. The default is 0.5.')
    arg_parser.add_argument('--tm_thr', type=float, default=0.5, help='The TM-align threshold under which two fragments are considered similar. The default is 0.5.')
    arg_parser.add_argument('--score_thr', type=int, default=10, help='The alignment score threshold above which two fragments are considered similar. The default is 10.')
    arg_parser.add_argument('--length_pct_thr', type=float, default=0.5, help='The percentage of length two fragments must share in order to be considered similar. The default is 0.5.')
    arg_parser.add_argument('--k', type=int, default=40, help='The number of clusters, should kNN or spectral clustering be chosen. The default is 40.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Perform the pipeline
    pipeline(args.fragments_dir, args.out_dir, args.tm_align_dir, args.dist_matrix_file, args.method, args.rmsd_thr, args.tm_thr, args.score_thr, args.length_pct_thr, args.k, verbose=args.verbose)
    