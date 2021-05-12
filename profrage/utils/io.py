# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:46:37 2021

@author: Federico van Swaaij
"""

import os
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmtf import MMTFParser, MMTFIO

from utils.structure import AA_DICT

def get_files(data_dir, ext='.pdb'):
    """
    Return a list of files with the desired extension from the specified directory.
    
    Parameters
    ----------
    data_dir : str
        The name of the directory.
    ext : str, optional
        The file extension. The default is '.pdb'
    
    Returns
    -------
    files : list of str
        The list containing the files.
    """
    files = []
    for file in os.listdir(data_dir):
        if file.endswith(ext):
            files.append(data_dir + file)
    return files

def parse_cmap(cmap_file):
    """
    Perform parsing of the specified CMAP file holding interaction information and return its entries.

    Parameters
    ----------
    cmap_file : str
        The CMAP file holding the interaction informations among residues of the same structure.

    Returns
    -------
    entries : list of (str, str, int, int, float)
        The list of entries. Each entry is a tuple of the form
        (chain_id_1, chain_id_2, residue_id_1, residue_id_2, f), where `f` belongs to [0,1] and is a
        measure of whether residues 1 and 2 are interacting. Usually value of `f` larger than 0.1
        indicate interaction.
    """
    entries = []
    cmap = open(cmap_file)
    for line in cmap:
        fields = line.split()
        if fields[0] == 'contact':
            fs_1_2_1, fs_1_2_2 =  fields[1].split(','), fields[2].split(',')
            if len(fs_1_2_1) != 2 or len(fs_1_2_2) != 2 or not fs_1_2_1[1].isdigit() or not fs_1_2_2[1].isdigit():
                return
            chain_id_1, res_idx_1 = fs_1_2_1[0], int(fs_1_2_1[1])
            chain_id_2, res_idx_2 = fs_1_2_2[0], int(fs_1_2_2[1])
            f = float(fields[3])
            entries.append((chain_id_1, chain_id_2, res_idx_1, res_idx_2, f))
    return entries

def parse_protein_net(file_name):
    """
    Parse a ProteinNet file and returns its contents as a dictionary.

    Parameters
    ----------
    file_name : str
        The ProteinNet file.

    Returns
    -------
    pn_dict : dict of str -> dict of str -> list of Any
        The dictionary describing the ProteinNet file.
    """
    # Read the file contents
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    # Define data structures
    pn_dict = {}
    pdb_id = ''
    state = ''
    # Iterate over the lines
    for line in lines:
        # Detect state
        if line[0:4] == '[ID]':
            state = 'ID'
        elif line[0:9] == '[PRIMARY]':
            state = 'PRIMARY'
        elif line[0:14] == '[EVOLUTIONARY]':
            state = 'EVOLUTIONARY'
        elif line[0:11] == '[SECONDARY]':
            state = 'SECONDARY'
        elif line[0:10] == '[TERTIARY]':
            state = 'TERTIARY'
        elif line[0:6] == '[MASK]':
            state = 'MASK'
        else:
            # Based on the state, update the dictionary
            nt = None
            if state == 'ID':
                pdb_id = line.split()[0]
                pn_dict[pdb_id] = {}
                full_id = pdb_id.split('_')
                seqp = full_id[0].split('#')
                if len(seqp) == 2:
                    seq_id, p_id = seqp[0], seqp[1] # sequencing ID, protein ID
                else:
                    seq_id, p_id = '0', seqp[0] # sequencing ID (does not matter here), protein ID
                if len(full_id) == 3:
                    m_id, ch_id = full_id[1], full_id[2] # model ID, chain ID
                elif len(full_id) == 2 and len(full_id[1]) == 1:
                    m_id, ch_id = full_id[1], '-1' # model ID, chain ID
                else:
                    m_id, ch_id = '-1', '-1' # model ID, chain ID
                pn_dict[pdb_id]['ID'] = [seq_id, p_id, m_id, ch_id]
                continue
            elif state == 'PRIMARY':
                def f(x):
                    chars = [c for c in x]
                    return [AA_DICT[c] for c in chars]
                nt = f
            elif state == 'TERTIARY':
                nt = float
            elif state == 'MASK':
                def f(x):
                    chars = [c for c in x]
                    return [1 if c=='+' else 0 for c in chars]
                nt = f
            else:
                continue # not interested in anything else
            splits = line.split()
            if splits == []:
                continue # empty line
            array = [nt(splits[i]) for i in range(len(splits))]
            pn_dict[pdb_id][state] = array
    return pn_dict

def to_pdb(structure, name, out_dir='./'):
    """
    Write the fragment into a PDB file.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to convert into PDB format.
    name : str
        The name of the file. It should not contain the '.pdb' extension.
    out_dir : str, optional
        The directory where to save the PDB file. The default is './' (current directory).

    Returns
    -------
    None.
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_dir + name + '.pdb')
    
def to_mmtf(structure, name, out_dir='./'):
    """
    Write the fragment into a MMTF file.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to convert into MMTF format.
    name : str
        The name of the file. It should not contain the '.mmtf' extension.
    out_dir : str, optional
        The directory where to save the MMTF file. The default is './' (current directory).

    Returns
    -------
    None.
    """
    io = MMTFIO()
    io.set_structure(structure)
    io.save(out_dir + name + '.mmtf')
    
def from_mmtf(mmtf):
    """
    Read the structure from a MMTF file.

    Parameters
    ----------
    mmtf : str
        The MMTF file to read from.

    Returns
    -------
    structure : Bio.PDB.Structure
        The structure.
    """
    parser = MMTFParser()
    structure = parser.get_structure(mmtf)
    return structure

def from_pdb(name, pdb, quiet=False):
    """
    Read the structure from a PDB file.

    Parameters
    ----------
    name : str
        The name of the structure.
    pdb : str
        The PDB file to read from.
    quiet : bool, optional
        Whether not to pring warnings. The default is False.

    Returns
    -------
    structure : Bio.PDB.Structure
        The structure.
    """
    parser = PDBParser(QUIET=quiet)
    structure = parser.get_structure(name, pdb)
    return structure
    
    
    