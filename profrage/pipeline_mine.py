import os

import argparse

from cluster.greedy import USRCluster, StrideCluster, AtomicSuperImpose
from fragment.mine import LeidenMiner
from fragment.filtering import in_range
from utils.structure import is_complete
from utils.io import get_files, from_pdb, to_pdb
from utils.ProgressBar import ProgressBar

def _mine(pdb_dir, cmap_dir, stride_dir, filter_dir, cluster_dir, max_size, contacts, bb_strength, n_iters, f_thr, lower_size,
          first_score_thr, second_score_thr, second_bb_atoms, third_rmsd_thr, third_length_pct, write_stats=False, verbose=False):
    pdbs = get_files(pdb_dir, ext='.pdb')
    progress_bar = ProgressBar(len(pdbs))
    if verbose:
        print('Mining and filtering...')
        progress_bar.start()
    for pdbf in pdbs:
        if verbose:
            progress_bar.step()
        pdb_id = os.path.basename(pdbf)[:-4]
        structure = from_pdb(pdb_id, pdbf, quiet=True)
        cmapf = cmap_dir + pdb_id + '.cmap'
        miner = LeidenMiner(structure, cmapf, contacts=contacts, bb_strenght=bb_strength, n_iters=n_iters, max_size=max_size, f_thr=f_thr)
        miner.mine()
        frags = miner.get_fragments()
        for frag in frags:
            if in_range(frag, lower=lower_size, upper=max_size) and is_complete(frag):
                to_pdb(frag, frag.get_id(), out_dir=filter_dir)
    if verbose:
        progress_bar.end()
    all_structures = []
    pre_clusters = {}
    inter_clusters = {}
    pdbs = get_files(filter_dir, ext='.pdb')
    if len(pdbs) == 0:
        return # no fragment has been generated
    for pdbf in pdbs:
        pdb_id = os.path.basename(pdbf)[:-4]
        all_structures.append(from_pdb(pdb_id, pdbf, quiet=True))
    # First level clustering
    if verbose:
        print('First level clustering...')
    first_clualg = StrideCluster(all_structures, stride_dir, filter_dir, score_thr=first_score_thr)
    first_clualg.cluster('greedy')
    for first_cluster_id in range(len(first_clualg)):
        pre_clusters[first_cluster_id] = []
        for idx in first_clualg.clusters[first_cluster_id]:
            s = first_clualg.structures[idx]
            pre_clusters[first_cluster_id].append(s)
    # Second level clustering
    if verbose:
        print('Second level clustering...')
    for keys in pre_clusters:
        pre_structures = pre_clusters[keys]
        second_clualg = USRCluster(pre_structures, score_thr=second_score_thr, bb_atoms=second_bb_atoms)
        second_clualg.cluster('greedy')
        for second_cluster_id in range(len(second_clualg)):
            inter_clusters[str(keys) + '-' + str(second_cluster_id)] = []
            for second_idx in second_clualg.clusters[second_cluster_id]:
                s = second_clualg.structures[second_idx]
                inter_clusters[str(keys) + '-' + str(second_cluster_id)].append(s)
    # Third level clustering
    if verbose:
        print('Third level clustering...')
    for keys in inter_clusters:
        inter_structures = inter_clusters[keys]
        third_clualg = AtomicSuperImpose(inter_structures, rmsd_thr=third_rmsd_thr, length_pct=third_length_pct)
        third_clualg.cluster('optim')
        for third_cluster_id in range(len(third_clualg)):
            s = third_clualg.best_representative(third_cluster_id)
            to_pdb(s, s.get_id(), cluster_dir)
        if write_stats:
            _write_cluster_stats(keys, third_clualg)

def _write_cluster_stats(full_id, clualg):
    file = open('lhg-cluster-log', 'a')
    for cluster_id in range(len(clualg)):
        total = len(clualg.clusters[cluster_id])
        file.write(str(full_id) + str(cluster_id) + ": " + str(total) + "\n")
    file.close()

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full mining pipeline.')
    arg_parser.add_argument('pdb_dir', type=str, help='The directory holding the PDB files from which to mine.')
    arg_parser.add_argument('cmap_dir', type=str, help='The directory holding the CMAP files.')
    arg_parser.add_argument('stride_dir', type=str, help='The directory holding the Stride tool.')
    arg_parser.add_argument('filter_dir', type=str, help='The directory to hold the filtered fragments.')
    arg_parser.add_argument('cluster_dir', type=str, help='The directory to hold the clustered files. Note that only cluster representatives will be saved.')
    arg_parser.add_argument('--max_size', type=int, default=30, help='The maximum number of amino acids in a fragment. The default is 30.')
    arg_parser.add_argument('--contacts', type=str, default='dist', help='How to measure whether two amino acids are in contact. Valid values are [cmap, dist]. The default is `dist`.')
    arg_parser.add_argument('--bb_strength', type=float, default=0.55, help='The offset to add to backbone connections to the Leiden algorithm. The default is 0.55.')
    arg_parser.add_argument('--n_iters', type=int, default=2, help='The number of iterations in the Leiden algorithm. The default is 22.')
    arg_parser.add_argument('--f_thr', type=float, default=0.1, help='The CMAP threshold. The default is 0.1.')
    arg_parser.add_argument('--lower_size', type=int, default=12, help='The minimum number of amino acids in a fragment. The default is 12.')
    arg_parser.add_argument('--first_score_thr', type=float, default=0.425, help='The score threshold for the Stride-level clustering. Valid range is [0,1]. The default is 0.425.')
    arg_parser.add_argument('--second_score_thr', type=float, default=0.425, help='The score threshold for the USR-level clustering. Valid range is [0,1]. The default is 0.425.')
    arg_parser.add_argument('--second_bb_atoms', type=bool, default=True, help='Whether to only use backbone atoms in the USR-level clustering. The default is True.')
    arg_parser.add_argument('--third_score_thr', type=float, default=2.0, help='The RMSD threshold for the AtomicSuperimposing-level clustering. The default is 2.0.')
    arg_parser.add_argument('--third_length_pct', type=float, default=0.6, help='The length threshold two structures must share in the AtomicSuperimposing-level clustering. Valid range is [0,1]. The default is 0.6.')
    arg_parser.add_argument('--write_stats', type=bool, default=False, help='Whether to write cluster stats. The default is False.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    _mine(args.pdb_dir, args.cmap_dir, args.stride_dir, args.filter_dir, args.cluster_dir,
          args.max_size, args.contacts, args.bb_strength, args.n_iters, args.f_thr,
          args.lower_size, args.first_score_thr, args.second_score_thr, args.second_bb_atoms, args.third_score_thr, args.third_length_pct,
          write_stats=args.write_stats, verbose=args.verbose)