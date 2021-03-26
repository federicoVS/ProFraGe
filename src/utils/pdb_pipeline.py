# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:20:13 2021

@author: FVS
"""

import argparse
from pdb_file import get_files, read_pdb_ids_file, fetch_pdb
from sequence_cluster import match_clusters, get_clusters
import progress_bar

def pipeline(pdb_ids, out_dir, sqid=30, loaded=False, verbose=False):
    '''
    Performs a full PDB data pipeline.

    Parameters
    ----------
    pdb_ids_list : list<string>
        A list of the PDB IDs to be downloaded.
    out_dir : str
        The directory where to save the the PDB files.
    sqid : int, optional
        The sequence identity percentage for the sequence clustering. The default is 30.
    loaded : bool, optional
        Whether the clusters have already been loaded. The default is False.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    # Get clusters
    clusters = get_clusters(sqid=sqid, loaded=loaded, verbose=verbose)
    # Match the PDB IDs with the clusters
    m_pdb_ch_ids = match_clusters(pdb_ids, clusters=clusters, sqid=sqid, verbose=verbose)
    # Fetch the PDB files
    count = 1
    latest_bar = 1
    if verbose:
        print('Fetching PDBs...')
        progress_bar.start()
    for m_pdb_ch_id in m_pdb_ch_ids:
        if verbose:
            latest_bar = progress_bar.progress(count, len(m_pdb_ch_ids), latest_bar)
            count += 1
        fetch_pdb(m_pdb_ch_id, out_dir=out_dir)
    if verbose:
        progress_bar.end()
            

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full PDB dataset pipeline')
    arg_parser.add_argument('--ids_dir', type=str, default='../../pdb/ids/', help='The directory containing the .txt files holding the IDs for the PDB.')
    arg_parser.add_argument('--out_dir', type=str, default='../../pdb/data/', help='The directory where to store the PDB files.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    arg_parser.add_argument('--sqid', type=int, default=30, help='The sequence identity percentage for the sequence clustering. The default is 30.')
    arg_parser.add_argument('--loaded', type=bool, default=False, help='Whether the sequence clusters have already been loaded. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Get PDB IDs
    pdb_ids = []
    files = get_files(args.ids_dir, ext='.txt')
    for file in files:
        pdb_ids += read_pdb_ids_file(file)
    # Begin pipeline
    pipeline(pdb_ids, args.out_dir, sqid=args.sqid, loaded=args.loaded, verbose=args.verbose)