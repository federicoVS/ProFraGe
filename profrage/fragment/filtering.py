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

def in_range(structure, lower=15, upper=40):
    """
    Check whether the given structure length is in range with respect to the given bounds.
    
    The length of the structure is expressed in the number of residues composing it.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure.
    lower : int, optional
        The lower bound. The default is 15.
    upper : int, optional
        The upper bound. The default is 40.

    Returns
    -------
    bool
        Whether the structure length is withing the bounds.
    """
    return structure_length(structure) >= lower and structure_length(structure) <= upper

def is_connected(structure, radius=5):
    """
    Check whether the structure is connected.
    
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
        Whether the structure is connected.
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

def is_compact(structure, pct_thr=0.65):
    """
    Check whether the structure is compact.
    
    The check is based on the percentage between the median distance from the center of the structure
    and the farthest one.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to compute the compactedness.
    pct_thr : float in [0,1], optional
        The percentage threshold above which a fragment is considered to be compact. It refers to the
        ratio between the averge distance from the center of the structure and distance between the center
        and the farthest point from it. The default is 0.65.

    Returns
    -------
    bool
        Whether the structure is compact.
    """
    coords = get_atoms_coords(structure)
    center = np.mean(coords, axis=0)
    norms = []
    for i in range(coords.shape[0]):
        norms.append(np.linalg.norm(coords[i,:]-center))
    norms = np.array(norms)
    farthest = norms[np.argmax(norms)]
    med = np.median(norms)
    pct = med/farthest
    return pct > pct_thr

def is_spherical(structure, var_thr=0.05):
    """
    Check whether the structure has a spherical shape.
    
    The check is based on whether the variance of the (normalized) distances from the center of the
    structure is below the threshold value.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to compute the sphericaledness.
    var_thr : float, optional
        The variance threshold. The default is 0.05.

    Returns
    -------
    bool
        Whether the structure is spherical.
    """
    coords = get_atoms_coords(structure)
    coords = normalize(coords)
    center = np.mean(coords, axis=0)
    norms = []
    for i in range(coords.shape[0]):
        norms.append(np.linalg.norm(coords[i,:]-center))
    norms = np.array(norms)
    variance = np.var(norms)
    return variance < var_thr

