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
from utils import progress_bar
from utils.misc import get_files

def pipeline(fragments_dir, pdb_ch_ids, radius, out_dir='./', verbose=False):
    '''
    This pipeline takes a directory of TERMs and filters them in order to retain ones which
    are composed of multiple segments.

    Parameters
    ----------
    fragments_dir : str
        The directory containing the TERMs.
    pdb_ch_ids : str
        The file holding the IDs of proteins and chains.
    radius : float
        The minimal radius to consider two redidues belonging to different segments.
    out_dir : str, optional
        The directory where to save the results. The default is './' (current directory).
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    count = 1
    latest_bar = 1
    if verbose:
        print('Filtering the fragments...')
        progress_bar.start()
    # Iterate over the protein, chain IDs
    for pdb_ch_id in pdb_ch_ids:
        if verbose:
            latest_bar = progress_bar.progress(count, len(pdb_ch_ids), latest_bar)
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
                if not term.is_connected(radius=radius):
                    file_name = out_dir + os.path.basename(mmtf)
                    shutil.copy(mmtf, file_name)
    if verbose:
        progress_bar.end()
                    
    
    

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Merging of fragments')
    arg_parser.add_argument('--fragments_dir', type=str, default='../pdb/fragments/terms/', help='The directory where the fragments are held. The default is ../pdb/fragments/all/')
    arg_parser.add_argument('--pdb_ch_ids', type=str, default='../pdb/ids/m_pdb_ch_ids', help='The file holding the IDs of the proteins and their chains. The default is ../pdb/ids/m_pdb_ch_ids')
    arg_parser.add_argument('--out_dir', type=str, default='../pdb/fragments/terms-filtered/', help='The directory where the super-fragments will be saved. The default is ../pdb/fragments/merged/')
    arg_parser.add_argument('--radius', type=float, default=5, help='The minimal radius to consider two redidues belonging to different segments. The default is 5.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Read the protein, chain IDs
    pdb_ch_ids = None
    with open(args.pdb_ch_ids, 'rb') as f:
        pdb_ch_ids = pickle.load(f)
    pipeline(args.fragments_dir, pdb_ch_ids=pdb_ch_ids, out_dir=args.out_dir, radius=args.radius, verbose=args.verbose)
    