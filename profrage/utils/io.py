# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:46:37 2021

@author: FVS
"""

import os
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmtf import MMTFParser

def get_files(data_dir, ext='.pdb'):
    '''
    Returns a list of files with the desired extension from the specified directory.
    
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
    '''
    files = []
    for file in os.listdir(data_dir):
        if file.endswith(ext):
            files.append(data_dir + file)
    return files

def to_pdb(structure, name, out_dir='./'):
    '''
    Writes the fragment into a PDB file. This can be useful for human analysis and
    visualization.

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
    '''
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_dir + name + '.pdb')
    
def from_mmtf(mmtf):
    '''
    Reads the structure from a MMTF file.

    Parameters
    ----------
    mmtf : str
        The MMTF to read from.

    Returns
    -------
    structure : Bio.PDB.Structure
        The structure.
    '''
    parser = MMTFParser()
    structure = parser.get_structure(mmtf)
    return structure
    
    
    