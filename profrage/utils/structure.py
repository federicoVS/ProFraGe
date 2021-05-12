# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:17:57 2021

@author: Federico van Swaaij
"""

import numpy as np
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

AA_DICT = {'A': 'ALA',
           'R': 'ARG',
           'N': 'ASN',
           'D': 'ASP',
           'C': 'CYS',
           'Q': 'GLN',
           'G': 'GLY',
           'E': 'GLU',
           'H': 'HIS',
           'I': 'ILE',
           'L': 'LEU',
           'K': 'LYS',
           'M': 'MET',
           'F': 'PHE',
           'P': 'PRO',
           'S': 'SER',
           'T': 'THR',
           'W': 'TRP',
           'Y': 'TYR',
           'V': 'VAL'}

def structure_length(structure):
    """
    Return the length of the specified structure in terms of its number of residues.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to compute the length of.

    Returns
    -------
    count : int
        The length of the structure measured in number of residues.
    """
    count = 0
    for residue in structure.get_residues():
        count += 1
    return count

def get_atoms_coords(structure):
    """
    Return the coordinates of the atoms of the given structure.
    
    The output has shape Nx3, where N is the total number of atoms.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to get the atoms coordinates.

    Returns
    -------
    numpy.ndarray
        The atoms coordinates.
    """
    atoms_coords = []
    count = 0
    for atom in structure.get_atoms():
        coords = atom.get_coord()
        atoms_coords.append([coords[0], coords[1], coords[2]])
        count += 1
    return np.array(atoms_coords).reshape((count,3))

def get_residue_center(residue):
    """
    Compute the coordinates of the center of the given residue.

    Parameters
    ----------
    residue : Bio.PDB.Residue
        The residue.

    Returns
    -------
    numpy.ndarray
        The ccoordinates of the center of the residue.
    """
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    coords = np.array(coords)
    return np.mean(coords, axis=0)

def lengths_within(structure_1, structure_2, ptc_thr):
    """
    Return whether the specified structures have comparable length.

    Parameters
    ----------
    structure_1 : Bio.PDB.Structure
        The first structure.
    structure_2 : Bio.PDB.Structure
    The first structure.

    Returns
    -------
    bool
        Whether the structures have comparable length.
    """
    small = structure_length(structure_1)
    large = structure_length(structure_2)
    if small > large:
        temp = large
        large = small
        small = temp
    return (small/large) >= ptc_thr

def build_protein_net_structures(pn_entry_dict, structure):
    """
    Build refined structures based on the provided ProteinNet entry.
    
    Depending on whether the model and chain IDs are provided, one or more structures are returned.

    Parameters
    ----------
    pn_entry_dict : dict of str -> list of Any
        A singke entry in the ProteinNet dataset.
    structure : Bio.PDB.Structure:
        The structure associated with the dictionary entry.

    Returns
    -------
    list of Bio.PDB.Structure
        The list of refined structures described by the ProteinNet entry.
    """
    _, p_id, m_id, ch_id = pn_entry_dict['ID']
    if m_id != '-1' and ch_id != '-1':
        residues = []
        s_id = p_id + '_' + ch_id
        for model in structure:
            if model.get_id() == m_id:
                for chain in model:
                    if chain.get_id() == ch_id:
                        for residue in chain:
                            residues.append(residue)
        refined_structure = build_structure(s_id, residues, structure.header)
        return [refined_structure]
    else:
        refined_structures = []
        residues = {}
        for model in structure:
            if model.get_id() == 0:
                for chain in model:
                    key = chain.get_id()
                    if key not in residues:
                        residues[key] = []
                    for residue in chain:
                        residues[key].append(residue)
        for key in residues:
            s_id = p_id + '_' + key
            refined_structures.append(build_structure(s_id, residues[key], structure.header))
        return refined_structures

def build_structure(s_id, residues, header):
    """
    Generate a structure based on a collection of residues.

    Parameters
    ----------
    s_id : str
        The ID of the structure to build.
    residues : list of Bio.PDB.Residue
        The list of residue from which to build the structure.
    header : dict of str -> Any
        The header of the structure. See the documentation of Bio.PDB.Structure for more information.

    Returns
    -------
    structure : Bio.PDB.Structure
        The generate structure.
    """
    structure = Structure(s_id)
    # Sort residues to ensure correct sequence order
    residues = sorted(residues, key=lambda x: x.get_id()[1])
    for residue in residues:
        r_full_id = residue.get_full_id()
        # Check if residue model exists, if not add it
        if not structure.has_id(r_full_id[1]):
            structure.add(Model(r_full_id[1]))
        # Get correct model for the residue
        for model in structure:
            been_added = False
            if model.get_id() == r_full_id[1]:
                # Check if model has the chain, if not add it
                if not model.has_id(r_full_id[2]):
                    model.add(Chain(r_full_id[2]))
                for chain in model:
                    if chain.get_id() == r_full_id[2]:
                        r_id = r_full_id[3]
                        if not chain.has_id(r_id):
                            r = Residue(r_id, residue.get_resname(), residue.get_segid())
                            for atom in residue:
                                r.add(atom)
                            chain.add(r)
                            been_added = True
                            break
                # If residue has been added then we can exit the loop
                if been_added:
                    break
    # Add stucture header
    structure.header = header
    # Return created structure
    return structure
