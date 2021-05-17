# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:05:59 2021

@author: Federico van Swaaij
"""

import os
from Bio.PDB.PDBList import PDBList

from utils.structure import break_chains, filter_chains, contains_dna_rna, build_structure
from utils.io import get_files, from_pdb, to_pdb
from utils.ProgressBar import ProgressBar


def process_testing_set(testing_dir, pnet_dir, verbose=False):
    """
    Process the ProteinNet testing set.
    
    It breaks each protein in the testing set into its chains.

    Parameters
    ----------
    testing_dir : str
        The directory where the raw PDB files are located.
    pnet_dir : str
        The super-directory for ProteinNet output.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    # Create output directory
    out_dir = pnet_dir + 'testing/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Get PDB files
    pdbfs = get_files(testing_dir)
    progress_bar = ProgressBar(len(pdbfs))
    if verbose:
        print('Processing testing set...')
        progress_bar.start()
    # Process each PDB file by breaking it into its chains
    for pdbf in pdbfs:
        if verbose:
            progress_bar.step()
        p_id = os.path.basename(pdbf)[:-4].upper()
        structure = from_pdb(p_id, pdbf, quiet=True)
        if contains_dna_rna(structure):
            continue # contains DNA/RNA, not usable
        structures_dict = break_chains(structure)
        chains = []
        for ch_id in structures_dict:
            pdb_id = p_id + '_' + ch_id
            chains.append(build_structure(pdb_id, structures_dict[ch_id], structure.header))
        uniques = filter_chains(chains)
        for unique in uniques:
            to_pdb(unique, unique.get_id(), out_dir=out_dir)
    if verbose:
        progress_bar.end()

def process_validation_set(pn_dict, pnet_dir, sqid, verbose=False):
    """
    Process the ProteinNet validation set.
    
    It checks for each entry whether its sequence identity matches with the given one. If so, it downloads
    the PDB structure and, if necessary, breaks it into its chains.

    Parameters
    ----------
    pn_dict : dict of str -> dict of str -> Any
        The dictionary representing the ProteinNet validation set.
    pnet_dir : str
        The super-directory for ProteinNet output.
    sqid : int
        The sequence identity.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    # Create output directory
    out_dir = pnet_dir + 'validation/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pdbl = PDBList(verbose=False)
    progress_bar = ProgressBar(len(pn_dict))
    if verbose:
        print('Processing validation set...')
        progress_bar.start()
    # Iterate over each ProteinNet entry
    for key in pn_dict:
        if verbose:
            progress_bar.step()
        sq_id, p_id, m_id, c_id = pn_dict[key]['ID']
        if int(sq_id) == sqid:
            _process_helper(pdbl, p_id, c_id, out_dir)
    if verbose:
        progress_bar.end()

def process_training_set(pn_dict, pnet_dir, verbose=False):
    """
    Process the ProteinNet training set.
    
    It downloads the PDB structures and, if necessary, breaks them into separate chains.
    
    Because this process can take some time, it also checks whether the desired file has already been
    downloaded.

    Parameters
    ----------
    pn_dict : dict of str -> dict of str -> Any
        The dictionary representing the ProteinNet validation set.
    pnet_dir : str
        The super-directory for ProteinNet output.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    # Create output directory
    out_dir = pnet_dir + 'training/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Get already existing files
    pdbfs = get_files(out_dir)
    downloaded_pdbs = {}
    for pdbf in pdbfs:
        p_id = os.path.basename(pdbf)[:-4].split('_')[0]
        downloaded_pdbs[p_id] = True
    pdbl = PDBList(verbose=False)
    progress_bar = ProgressBar(len(pn_dict))
    if verbose:
        print('Processing training set...')
        progress_bar.start()
    # Iterate over each ProteinNet entry
    for key in pn_dict:
        if verbose:
            progress_bar.step()
        sq_id, p_id, m_id, c_id = pn_dict[key]['ID']
        # Check if PDB file has already been downloaded
        if p_id in downloaded_pdbs:
            continue
        _process_helper(pdbl, p_id, c_id, out_dir)
    if verbose:
        progress_bar.end()
                    
def _process_helper(pdbl, p_id, c_id, out_dir):
    # Retrieve PDB file, if it does not exist then skip iteration
    file_name = pdbl.retrieve_pdb_file(p_id, file_format='pdb', pdir='./')
    if not os.path.isfile(file_name):
        return
    structure = from_pdb(p_id, file_name, quiet=True)
    if contains_dna_rna(structure):
        os.remove(file_name) # don't need this
        return # contain DNA/RNA, not usable
    # Chain ID is not specified
    if c_id == '-1' :
        # Break structure into chains
        structures_dict = break_chains(structure)
        # Build structures and to filter them based on 90% similarity
        chains = []
        for ch_id in structures_dict:
            pdb_id = p_id.upper() + '_' + ch_id
            if ch_id not in structures_dict:
                continue
            chains.append(build_structure(pdb_id, structures_dict[ch_id], structure.header))
        uniques = filter_chains(chains)
        for unique in uniques:
            to_pdb(unique, unique.get_id(), out_dir=out_dir)
    # Chain ID is specified
    elif c_id != '-1':
        # Break structure into chains
        structures_dict = break_chains(structure)
        pdb_id = p_id.upper() + '_' + c_id.upper()
        if c_id.upper() not in structures_dict:
            return
        new_structure = build_structure(pdb_id, structures_dict[c_id.upper()], structure.header)
        to_pdb(new_structure, pdb_id, out_dir=out_dir)
    os.remove(file_name) # don't need this
