# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 02:10:15 2021

@author: FVS
"""

from Bio.PDB import NeighborSearch, Selection
from fragments.graphs import UUGraph
from utils.structure import structure_length, get_residue_center

def is_complex(structure, grade=12):
    '''
    Checks whether the given structure is complex, according to its complexity grade, that is,
    how many residues compose it.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to compute the complexity.
    grade : int, optional
        The minimal number of residues for the fragment to be considered complex. The default is 12.

    Returns
    -------
    bool
        Whether the fragment is complex. True if |residues| >= grade, False otherwise.
    '''
    return structure_length(structure) >= grade

def is_connected(structure, radius=5):
    '''
    Checks whether the fragment is connected. The search is conducted at the residual level.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to compute the complexity.
    radius : float, optional
        Search radius in Angstroms. The default is 5.

    Returns
    -------
    bool
        Whether the fragment is connected.
    '''
    # Create dictionary which encodes residues as integers
    index = 0
    vertex_dict = {}
    for residue in structure.get_residues():
        vertex_dict[residue] = index
        index += 1
    # Create graph
    graph = UUGraph(index)
    # Iterate over the residues
    for target_residue in structure.get_residues():
        center_coord = get_residue_center(target_residue)
        atoms = Selection.unfold_entities(structure, 'A')
        ns = NeighborSearch(atoms)
        close_residues = ns.search(center_coord, radius, level='R')
        # Remove the target protein itself
        if target_residue in close_residues:
            close_residues.remove(target_residue)
        for cr in close_residues:
            graph.add_edge(vertex_dict[target_residue], vertex_dict[cr])
    # Compute the connected components
    graph.compute_connected_components()
    return len(graph.connected_components) == 1

