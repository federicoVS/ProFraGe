# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:20:13 2021

@author: FVS
"""

import argparse
from pdb_file import get_files, read_pdb_ids_file, fetch_pdbs
from sequence_cluster import match_clusters, get_clusters

def pipeline(pdb_ids_list, out_dir, sqid=30, verbose=False):
    '''
    Performs a full PDB data pipeline.

    Parameters
    ----------
    pdb_ids_list : list<list<string>>
        A list containing a list of the PDB IDs to be downloaded. It is structured this way because for more
        than 25000 IDs, the PDB website generates multiple .txt files of a maximum 25000 IDs each.
    out_dir : string
        The directory where to save the the PDB files.
    sqid : int, optional
        The sequence identity percentage for the sequence clustering. The default is 30.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    # Get clusters
    clusters = get_clusters(verbose=verbose)
    # Match the PDB IDs with the clusters
    m_clusters = []
    for pdb_ids in pdb_ids_list:
        m_clusters.append(match_clusters(pdb_ids, clusters=clusters, sqid=sqid, verbose=verbose))
    # Fetch the PDB files
    count = 1
    for m_cluster in m_clusters:
        if verbose:
            print(f'Processing cluster {count}/{len(m_clusters)}')
            count += 1
        fetch_pdbs(m_cluster, out_dir=out_dir, verbose=verbose)
            

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full PDB dataset pipeline')
    arg_parser.add_argumennt('--ids_dir', type=str, default='../../pdb/ids/', help='The directory containing the .txt files holding the IDs for the PDB.')
    arg_parser.add_argumennt('--out_dir', type=str, default='../../pdb/data/', help='The directory where to store the PDB files.')
    arg_parser.add_argumennt('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    arg_parser.add_argumennt('--sqid', type=int, default=30, help='The sequence identity percentage for the sequence clustering. The default is 30.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Get list of PDBs
    pdb_ids_list = []
    files = get_files(args.pdb_ids_dir, ext='.txt')
    for file in files:
        pdb_ids_list.append(read_pdb_ids_file(file))
    # Begin pipeline
    pipeline(pdb_ids_list, args.out_dir, sqid=args.sqid, verbose=args.verbose)