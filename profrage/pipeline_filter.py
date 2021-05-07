# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:48:57 2021

@author: Federico van Swaaij
"""

import os
import shutil
import argparse
import pickle

from fragment.filtering import in_range, is_connected
from utils.ProgressBar import ProgressBar
from utils.io import get_files, from_mmtf

def pipeline(method, base_dir, pdb_ch_ids, filtering_method, radius=5, verbose=False):
    """
    Take a directory of fragments and filters them in order to retain ones which are composed of multiple segments.

    Parameters
    ----------
    method : str
        The method used to generate the fragments.
    base_dir : str
        The base directory for input and output.
    pdb_ch_ids : str
        The file holding the IDs of proteins and chains.
    filtering_method : str
        The filtering method to apply to the fragments.
    radius : float, optional
        The minimal radius to consider two redidues belonging to different segments.The default is 5 A.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    # Define proper output directory
    out_name = method + '-filtered/'
    count = 1
    progress_bar = ProgressBar()
    if verbose:
        print('Filtering the fragments...')
        progress_bar.start()
    n_fragments = 0
    if method == 'terms' or method == 'seq' or method == 'ccg':
        # Build proper output directory
        if not os.path.exists(base_dir+out_name):
            os.makedirs(base_dir+out_name)
        # Iterate over the protein, chain IDs
        for pdb_ch_id in pdb_ch_ids:
            if verbose:
                progress_bar.step(count, len(pdb_ch_ids))
                count += 1
            p_id, c_id = pdb_ch_id
            dir_name = base_dir + method + '/' + p_id + '_' + c_id + '/fragments/'
            # Check if directory for the TERM indeed exists
            if os.path.isdir(dir_name):
                # Get all MMTF files holding the TERMs
                mmtfs = get_files(dir_name, ext='.mmtf')
                # Iterate through each MMTF file
                for mmtf in mmtfs:
                    frag = from_mmtf(mmtf)
                    if filtering_method == 'connected':
                        if not is_connected(frag, radius=radius):
                            n_fragments += 1
                            shutil.copy(mmtf, base_dir + out_name + os.path.basename(mmtf)[:-5] + '.mmtf')
                    elif filtering_method == 'complex':
                        if in_range(frag):
                            n_fragments += 1
                            shutil.copy(mmtf, base_dir + out_name + os.path.basename(mmtf)[:-5] + '.mmtf')
                    else:
                        print('Invalid filtering method.')
                        return
    elif method == 'fuzzle':
        for root, _, files in os.walk(base_dir + method + '/'):
            if verbose:
                progress_bar.step(count, len(pdb_ch_ids))
                count += 1
            for name in files:
                if name[:-4] == 'mmtf':
                    mmtf = os.path.join(root, name)
                    frag = from_mmtf(mmtf)
                    if filtering_method == 'connected':
                        if not is_connected(frag, radius=radius):
                            n_fragments += 1
                            shutil.copy(mmtf, base_dir + out_name + os.path.basename(mmtf)[:-5] + '.mmtf')
                    elif filtering_method == 'complex':
                        if in_range(frag):
                            n_fragments += 1
                            shutil.copy(mmtf, base_dir + out_name + os.path.basename(mmtf)[:-5] + '.mmtf')
                    else:
                        print('Invalid filtering method.')
                        return
    if verbose:
        progress_bar.end()
        print(f'Generated {n_fragments} fragments.')


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Filtering the fragments.')
    arg_parser.add_argument('method', type=str, help='The method used to generate the fragments.')
    arg_parser.add_argument('--base_dir', type=str, default='../pdb/fragments/', help='The base directory for input and output. The default is ../pdb/fragments/.')
    arg_parser.add_argument('--pdb_ch_ids', type=str, default='../pdb/ids/m_pdb_ch_ids', help='The file holding the IDs of the proteins and their chains. The default is ../pdb/ids/m_pdb_ch_ids')
    arg_parser.add_argument('--filtering_method', type=str, default='complex', help='The filtering method to apply to the fragments. The options are [connected, complex]. The default is complex.')
    arg_parser.add_argument('--radius', type=float, default=5, help='The minimal radius in Angstroms to consider two residues belonging to different segments. The default is 5A.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Read the protein, chain IDs
    pdb_ch_ids = None
    with open(args.pdb_ch_ids, 'rb') as f:
        pdb_ch_ids = pickle.load(f)
    # Begin pipeline
    pipeline(args.method, args.base_dir, pdb_ch_ids, args.filtering_method, args.radius, verbose=args.verbose)
    