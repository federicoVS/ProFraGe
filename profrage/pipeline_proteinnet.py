# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:17:17 2021

@author: Federico van Swaaij
"""

import os
import argparse
from Bio.PDB.PDBList import PDBList

from utils.structure import build_protein_net_structures
from utils.io import get_files, parse_protein_net, from_pdb, to_pdb
from utils.ProgressBar import ProgressBar

def pipeline(casp_dir, pdb_dir, sqid=30, out_dir='protein-net/', verbose=False):
    """
    Perform a full ProteinNet pipeline based on the given existing PDB dataset.
    
    All ProteinNet entries are parsed, and if they are not already contained by the existing PDB
    dataset, are added as well.
    
    The pipeline also takes care in handling multiple chains. If a structure has multiple models, only the
    first one is selected. If multiple chains are detected, all are selected.

    Parameters
    ----------
    casp_dir : str
        The directory holding the ProteinNet splits.
    pdb_dir : str
        The directory holding the existing PDB dataset.
    sqid : int, optional
        the sequence identity percentage, which should be the same as the PDB dataset. The default is 30.
    out_dir : str, optional
        The output directory. The default is 'protein-net/'.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    pdb_files = get_files(pdb_dir)
    pdb_ids = [os.path.basename(pdbf)[:-4] for pdbf in pdb_files]
    pdbs_dict = {}
    pdbl = PDBList()
    for pdb_id in pdb_ids:
        pdbs_dict[pdb_id] = True
    splits = ['testing', 'validation', 'training_' + str(sqid)]
    progress_bar = ProgressBar()
    count = 1
    if verbose:
        print('Parsing ProteinNet...')
        progress_bar.start()
    for split in splits:
        if verbose:
            progress_bar.step(count, len(splits))
            count += 1
        casp_file = casp_dir + split
        casp_dict = parse_protein_net(casp_file)
        for key in casp_dict:
            # Get the IDs from the entry
            _, p_id, _, ch_id = casp_dict[key]['ID']
            pdb_id = p_id + '_' + ch_id
            if pdb_id not in pdbs_dict:
                # Download the file
                file_name = pdbl.retrieve_pdb_file(p_id, file_format='pdb', pdir=out_dir)
                if os.path.isfile(file_name):
                    # Change the extension of the file
                    pre, ext = os.path.splitext(file_name[3+len(out_dir):]) # first 3 is for the 'pdb' added by the PDBList
                    new_file_name = out_dir + pre.upper() + '.pdb'
                    os.rename(file_name, new_file_name)
                    # Read the structure
                    structure = from_pdb(pre, new_file_name, quiet=True)
                    # Compute the refined structures
                    refined_structures = build_protein_net_structures(casp_dict[key], structure)
                    # Remove the old structure
                    os.remove(new_file_name)
                    # Write the refined structures
                    for refined_structure in refined_structures:
                        to_pdb(refined_structure, pre, out_dir=out_dir)
    if verbose:
        progress_bar.end()
        
if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full ProteinNet dataset pipeline with respect to the existing PDB dataset.')
    arg_parser.add_argument('casp_dir', type=str, help='The directory containing the ProteinNet splits.')
    arg_parser.add_argument('pdb_dir', type=str, help='The directory containing the existing PDB dataset.')
    arg_parser.add_argument('--sqid', type=int, default=30, help='The sequence identity percentage. The default is 30.')
    arg_parser.add_argument('--out_dir', type=str, default='protein-net', help='The output directory. The default is protein-net/.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    pipeline(args.casp_dir, args.pdb_dir, sqid=args.sqid, out_dir=args.out_dir, verbose=args.verbose)
    