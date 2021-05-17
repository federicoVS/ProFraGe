# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:20:13 2021

@author: Federico van Swaaij
"""

import argparse

from utils.pdb_file import read_pdb_ids_file, fetch_pdb
from utils.io import get_files
from utils.sequence_cluster import match_clusters, get_clusters
from utils.ProgressBar import ProgressBar

def pipeline(pdb_ids, pdb_gz_dir, out_dir='./', m_pdb_ch_file=None, sqid=30, loaded=False, remove_pdb_gz=False, verbose=False):
    """
    Perform a full PDB data pipeline.
    
    The dataset is build based on the BLAST clustering. After all proteins have been structured based on
    the specified sequence identity percentage, the first representative of each cluster is chosen.

    Parameters
    ----------
    pdb_ids_list : list of str
        A list of the PDB IDs to be downloaded.
    pdb_gz_dir : str
        The directory where to find the PDB files in .pdb.gz format.
    out_dir : str, optional
        The directory where to save the the PDB files. The default is './' (current directory).
    m_pdb_ch_file : str, optional
        The file holding the list of the protein chains matching with the clusters. The default is None
    sqid : int, optional
        The sequence identity percentage for the sequence clustering. The default is 30.
    loaded : bool, optional
        Whether the clusters have already been loaded. The default is False.
    remove_pdb_gz : bool, optional
        Whether to remove the original compressed PDB file. The default is False.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    # Get clusters
    clusters = get_clusters(sqid=sqid, loaded=loaded, verbose=verbose)
    # Match the PDB IDs with the clusters
    m_pdb_ch_ids = match_clusters(pdb_ids, clusters=clusters, sqid=sqid, out_file=m_pdb_ch_file, verbose=verbose)
    # Fetch the PDB files
    progress_bar = ProgressBar(len(m_pdb_ch_ids))
    if verbose:
        print('Fetching PDBs...')
        progress_bar.start()
    for m_pdb_ch_id in m_pdb_ch_ids:
        if verbose:
            progress_bar.step()
        fetch_pdb(m_pdb_ch_id, pdb_gz_dir=pdb_gz_dir, out_dir=out_dir, remove_pdb_gz=remove_pdb_gz)
    if verbose:
        progress_bar.end()


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full PDB dataset pipeline.')
    arg_parser.add_argument('--ids_dir', type=str, default='../../pdb/ids/', help='The directory containing the .txt files holding the IDs for the PDB. The default is ../../pdb/ids/.')
    arg_parser.add_argument('--pdb_gz_dir', type=str, default='../../pdb/gz/', help='The directory containing the downloaded compressed PDB files. The default is ../../pdb/gz/.')
    arg_parser.add_argument('--out_dir', type=str, default='../../pdb/data/', help='The directory where to store the PDB files. The default is ../../pdb/data/.')
    arg_parser.add_argument('--m_pdb_ch_file', type=str, default='../../pdb/ids/m_pdb_ch_ids', help='The file holding the list of the protein chains matching with the clusters. The default is ../../pdb/ids/m_pdb_ch_ids.')
    arg_parser.add_argument('--sqid', type=int, default=30, help='The sequence identity percentage for the BLAST sequence clustering. The default is 30.')
    arg_parser.add_argument('--loaded', type=bool, default=False, help='Whether the sequence clusters have already been loaded. The default is False.')
    arg_parser.add_argument('--remove_pdb_gz', type=bool, default=False, help='Whether to remove the original compressed PDB file. The default is False.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Get PDB IDs
    pdb_ids = []
    files = get_files(args.ids_dir, ext='.txt')
    for file in files:
        pdb_ids += read_pdb_ids_file(file)
    # Begin pipeline
    pipeline(pdb_ids, args.pdb_gz_dir, args.out_dir, m_pdb_ch_file=args.m_pdb_ch_file, sqid=args.sqid, loaded=args.loaded, remove_pdb_gz=args.remove_pdb_gz, verbose=args.verbose)