#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 00:45:54 2021

@author: Federico van Swaaij
"""

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

class Neighborhood:
    """
    Model a neighborhood.
    
    A neighborhood is defined as a centroid along with k subsequent residues on the left and on the right,
    thus creating a neighborhood of 2k+1 residues.
    
    Attributes
    ----------
    idx : int
        The ID of the neighborhood.
    residues : list of Bio.PDB.Residue
        The list of residues composing the neighborhood.
    max_inters : int
        The maximum number of interacting neighborhoods to be considered when creating a fragment out
        of the neighborhood. The interactions are selected based on their score, thus keeping the best
        `max_inters`.
    features : numpy.ndarray
        The features representing the subset of residues.
    interactions : list of (fragments.data.Neighborhood, float)
        The list of interactions. Each entry is a tuple, the first element being the interacting
        neighborhood, and the second being the interaction score.
    """
    
    def __init__(self, idx, residues, max_inters=3):
        """
        Initialize the class.

        Parameters
        ----------
        idx : int
            The ID of the neighborhood.
        residues : list of Bio.PDB.Residue
            The list of residues composing the neighborhood.
        max_inters : int, optional
            The maximum number of interacting neighborhoods. The default is 3.

        Returns
        -------
        None.
        """
        self.idx = idx
        self.residues = residues
        self.max_inters = max_inters
        self.features = None
        self.interactions = []
        
    def add_interaction(self, neighborhood, f_sum):
        """
        Add an interaction to the neighborhood.

        Parameters
        ----------
        neighborhood : fragments.data.Neighborhood
            The neighborhood to add.
        f_sum : float
            The sum of interaction scores.

        Returns
        -------
        None.
        """
        self.interactions.append((neighborhood, f_sum))
    
    def filter_interactions(self):
        """
        Filter and retain only the top `max_inters` interactions.

        Returns
        -------
        None.
        """
        self.interactions = sorted(self.interactions, key=lambda x: x[1], reverse=True)
        self.interactions = self.interactions[0:self.max_inters]
        
    def compute_features(self, Rep):
        """
        Compute the features of the structure using the specified representation.

        Parameters
        ----------
        Rep : structure.representation.Representation
            The class of structure representation.

        Returns
        -------
        None.
        """
        c_structure = Structure('S')
        model = Model(0)
        chain = Chain('A')
        for residue in self.residues:
            r = Residue(residue.get_id(), residue.get_resname(), residue.get_segid())
            for atom in residue:
                r.add(atom)
            chain.add(r)
        model.add(chain)
        c_structure.add(model)
        self.features = Rep(c_structure).get_features()