# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:48:57 2021

@author: FVS
"""

import os
import shutil
import argparse
import pickle
from fragments.Fragment import Fragment
from utils.ProgressBar import ProgressBar
from utils.misc import get_files

def pipeline(fragments_dir, pdb_ch_ids, method, radius, grade, out_dir='./', verbose=False):
    '''
    This pipeline takes a directory of fragments and filters them in order to retain ones which
    are composed of multiple segments.

    Parameters
    ----------
    fragments_dir : str
        The directory containing the fragments.
    pdb_ch_ids : str
        The file holding the IDs of proteins and chains.
    method : str
        The filtering method to apply to the fragments.
    radius : float
        The minimal radius to consider two redidues belonging to different segments.
    grade : int
        The minimal number of residues for a fragment to be complex.
    out_dir : str, optional
        The directory where to save the results. The default is './' (current directory).
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    count = 1
    progress_bar = ProgressBar()
    if verbose:
        print('Filtering the fragments...')
        progress_bar.start()
    n_fragments = 0
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
                term = Fragment(mmtf)
                if method == 'connected':
                    if not term.is_connected(radius=radius):
                        n_fragments += 1
                        shutil.copy(mmtf, out_dir + os.path.basename(mmtf)[:-5] + '.mmtf')
                elif method == 'complex':
                    if term.is_complex(grade=grade):
                        n_fragments += 1
                        shutil.copy(mmtf, out_dir + os.path.basename(mmtf)[:-5] + '.mmtf')
                else:
                    if verbose:
                        progress_bar.end()
                    return
    if verbose:
        progress_bar.end()
        print(f'Generated {n_fragments} fragments.')


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Merging of fragments')
    arg_parser.add_argument('--fragments_dir', type=str, default='../pdb/fragments/terms/', help='The directory where the fragments are held. The default is ../pdb/fragments/terms/')
    arg_parser.add_argument('--pdb_ch_ids', type=str, default='../pdb/ids/m_pdb_ch_ids', help='The file holding the IDs of the proteins and their chains. The default is ../pdb/ids/m_pdb_ch_ids')
    arg_parser.add_argument('--out_dir', type=str, default='../pdb/fragments/terms-filtered/', help='The directory where the filtered fragments will be saved. The default is ../pdb/fragments/terms-filtered/')
    arg_parser.add_argument('--method', type=str, default='connected', help='The filtering method to apply to the fragments. The options are [connected, complex]. The default is connected.')
    arg_parser.add_argument('--radius', type=float, default=5, help='The minimal radius in Angstroms to consider two residues belonging to different segments. The default is 5A.')
    arg_parser.add_argument('--grade', type=float, default=12, help='The minimal number of residues for a fragment to be complex. The default is 12.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Read the protein, chain IDs
    pdb_ch_ids = None
    with open(args.pdb_ch_ids, 'rb') as f:
        pdb_ch_ids = pickle.load(f)
    pipeline(args.fragments_dir, pdb_ch_ids, args.method, args.radius, args.grade, out_dir=args.out_dir, verbose=args.verbose)
    