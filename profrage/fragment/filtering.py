# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 02:10:15 2021

@author: Federico van Swaaij
"""
import numpy as np
from sklearn.preprocessing import normalize
from Bio.PDB import NeighborSearch, Selection

from fragment.graphs import UUGraph
from utils.structure import structure_length, get_residue_center, get_atoms_coords

def has_hetatoms(structure, pct=1):
    """
    Check whether the percentage of hetatoms within the structure is higher of the allowed one.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to check its percentage composition of hetatoms.
    pct : float in [0,1], optional
        The maximum allowed percentage of hetatoms within the structure. The default is 1.

    Returns
    -------
    bool
         Whether the percentage of hetatoms is higher than the allowed one.
    """
    count = 0
    n = structure_length(structure)
    for residue in structure.get_residues():
        r_id = residue.get_id()
        if r_id[0] != ' ' and r_id[0] != '':
            count += 1
    return float(count/n) > pct

def is_complex(structure, grade=12):
    """
    Check whether the given structure is complex, according to how many residues compose it.

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
    """
    return structure_length(structure) >= grade

def is_connected(structure, radius=5):
    """
    Check whether the fragment is connected.
    
    The search is conducted at the residual level.

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
    """
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

def is_compact(structure, thr=1):
    """
    Check whether the fragment is compact.
    
    The check is based on the variance of the squared distances from the centre of the fragment.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to compute the compactedness.
    thr : float, optional
        The variance threshold under which a fragment is considered to be compact. The default is 1.

    Returns
    -------
    bool
        Whether the fragment is compact.
    """
    coords = get_atoms_coords(structure)
    centre = np.mean(coords, axis=0)
    squared_dist = np.sum((coords-centre)**2, axis=1)
    squared_dist = normalize(squared_dist)
    var = np.var(squared_dist)
    return var < thr

