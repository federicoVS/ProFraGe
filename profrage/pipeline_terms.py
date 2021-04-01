# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:48:57 2021

@author: FVS
"""

import os
import shutil
import argparse
import pickle
from fragments.TERMFragment import TERMFragment
from similarity.SACluster import SACluster
from utils.ProgressBar import ProgressBar
from utils.misc import get_files

def pipeline(fragments_dir, pdb_ch_ids, method, radius, grade, seq_score_thr, length_pct_thr, out_dir='./', verbose=False):
    '''
    This pipeline takes a directory of TERMs and filters them in order to retain ones which
    are composed of multiple segments.

    Parameters
    ----------
    fragments_dir : str
        The directory containing the TERMs.
    pdb_ch_ids : str
        The file holding the IDs of proteins and chains.
    method : str
        The filtering method to apply to the fragments.
    radius : float
        The minimal radius to consider two redidues belonging to different segments.
    grade : int
        The minimal number of residues for a fragment to be complex.
    seq_score_thr : int
        The threshold score for two sequences to be considered similar.
    length_pct_thr : float in [0,1]
        The minimal percentage of length that two structures must share in order to be equal
    out_dir : str, optional
        The directory where to save the results. The default is './' (current directory).
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    bool
        Returns True upon success, False upon failure
    '''
    count = 1
    progress_bar = ProgressBar()
    if verbose:
        print('Filtering the fragments...')
        progress_bar.start()
    # Fragments matching the first layer of filtering
    fragments = {}
    # Iterate over the protein, chain IDs
    for pdb_ch_id in pdb_ch_ids:
        if verbose:
            progress_bar.step(count, len(pdb_ch_ids))
            count += 1
        p_id, c_id = pdb_ch_id
        dir_name = fragments_dir + p_id + '_' + c_id + '/fragments/'
        # Check if directory for the TERM indeed exists
        if os.path.isdir(dir_name):
            # Get all MMTF files holding the TERMs
            mmtfs = get_files(dir_name, ext='.mmtf')
            # Iterate through each MMTF file
            for mmtf in mmtfs:
                term = TERMFragment(mmtf)
                if method == 'connected':
                    if not term.is_connected(radius=radius):
                        fragments[term.get_fragment()] = mmtf
                elif method == 'complex':
                    if term.is_complex(grade=grade):
                        fragments[term.get_fragment()] = mmtf
                else:
                    print(f'Filtering method {method} is not valid.')
                    return False
    if verbose:
        progress_bar.end()
    # Cluster the generated fragments
    frags_list = []
    for frag in fragments.keys():
        frags_list.append(frag)
    if verbose:
        print(f'Resulting fragments: {len(frags_list)}.')
    # print(len(frags_list))
    sac = SACluster(frags_list, seq_score_thr, length_pct_thr, verbose=verbose)
    sac.cluster()
    typicals = sac.get_typicals()
    for index in typicals:
        frag = typicals[index]
        shutil.copy(fragments[frag], out_dir + os.path.basename(fragments[frag]))
        term = TERMFragment(fragments[frag])
        term.to_pdb(out_dir='lol/')
    # Show clusters
    sac.show_clusters()
    # Return success
    return True


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Merging of fragments')
    arg_parser.add_argument('--fragments_dir', type=str, default='../pdb/fragments/terms/', help='The directory where the fragments are held. The default is ../pdb/fragments/all/')
    arg_parser.add_argument('--pdb_ch_ids', type=str, default='../pdb/ids/m_pdb_ch_ids', help='The file holding the IDs of the proteins and their chains. The default is ../pdb/ids/m_pdb_ch_ids')
    arg_parser.add_argument('--out_dir', type=str, default='../pdb/fragments/terms-filtered/', help='The directory where the super-fragments will be saved. The default is ../pdb/fragments/merged/')
    arg_parser.add_argument('--method', type=str, default='connected', help='The filtering method to apply to the fragments. The options are [connected, complex]. The default is connected.')
    arg_parser.add_argument('--radius', type=float, default=5, help='The minimal radius to consider two residues belonging to different segments. The default is 5.')
    arg_parser.add_argument('--grade', type=float, default=12, help='The minimal number of residues for a fragment to be complex. The default is 12.')
    arg_parser.add_argument('--seq_score_thr', type=int, default=10, help='The score alignment threshold for two fragments to be considered similar. The default is 150.')
    arg_parser.add_argument('--length_pct_thr', type=float, default=0.5, help='The percentage of length two fragments need to share to be considered similar. The default is 0.8, with acceptable value belonging to the interval [0,1].')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Read the protein, chain IDs
    pdb_ch_ids = None
    with open(args.pdb_ch_ids, 'rb') as f:
        pdb_ch_ids = pickle.load(f)
    pipeline(args.fragments_dir, pdb_ch_ids, args.method, args.radius, args.grade, args.seq_score_thr, args.length_pct_thr, out_dir=args.out_dir, verbose=args.verbose)
    