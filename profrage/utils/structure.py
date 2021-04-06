# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:17:57 2021

@author: FVS
"""

import numpy as np

def structure_length(structure):
    '''
    Returns the length of the specified structure in terms of its number of residues.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to compute the length of.

    Returns
    -------
    count : int
        The length of the structure measured in number of residues.
    '''
    count = 1
    for residue in structure.get_residues():
        count += 1
    return count

def get_atoms_coords(structure):
    '''
    Returns the coordinates of the atoms of the given structure. The output has shape Nx3, where N is
    the total number of atoms.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to get the atoms coordinates.

    Returns
    -------
    numpy.ndarray
        The atoms coordinates.
    '''
    atoms_coords = []
    count = 0
    for atom in structure.get_atoms():
        coords = atom.get_coord()
        atoms_coords.append([coords[0], coords[1], coords[2]])
        count += 1
    return np.array(atoms_coords).reshape((count,3))

def get_residue_center(self, residue):
    '''
    Computes the coordinates of the center of the given residue.

    Parameters
    ----------
    residue : Bio.PDB.Residue
        The residue.

    Returns
    -------
    numpy.ndarray
        The ccoordinates of the center of the residue.
    '''
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    coords = np.array(coords)
    return np.mean(coords, axis=0)

def lengths_within(structure_1, structure_2, ptc_thr):
    '''
    Returns whether the specified structures have comparable length.

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
    '''
    small = structure_length(structure_1)
    large = structure_length(structure_2)
    if small > large:
        temp = large
        large = small
        small = temp
    return (small/large) >= ptc_thr

