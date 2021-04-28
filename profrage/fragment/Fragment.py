#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:04:21 2021

@author: Federico van Swaaij
"""
from scipy.spatial import distance

from structure.representation import USR
from utils.structure import generate_structure

class Fragment:
    """
    Representation of a fragment.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure the fragment belongs to.
    f_id : str
        The ID of the fragment.
    residues : list of Bio.PDB.Residue
        The list of residues composing the fragment.
    """
    
    def __init__(self, structure, f_id):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure the fragment belongs to.
        f_id : str
            The ID of the fragment.

        Returns
        -------
        None.
        """
        self.structure = structure
        self.f_id = f_id
        self.residues = []
        
    def __len__(self):
        """
        Return the length of the fragment in residues.

        Returns
        -------
        int
            The lenght in residues.
        """
        return len(self.residues)
        
    def add_residue(self, r_candidate):
        """
        Add a residue to the fragment, if it is not already part of.

        Parameters
        ----------
        r_candidate : Bio.PDB.Residue
            The candidate residue to add.

        Returns
        -------
        None.
        """
        for residue in self.residues:
            if r_candidate.get_id() == residue.get_id():
                return
        self.residues.append(r_candidate)
        
    def match_at(self, structure, start):
        """
        Compute the match score of the fragment at the specified position in the specified structure.
        
        The score is computed as the cosine similarity between the fragment USR and the sub-structure USR
        starting at the specified position.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure within which the fragment is to be matched.
        start : int
            The starting position at which the fragment is to be matched.

        Returns
        -------
        float in [0,1]
            The cosine score.
        """
        # Get structure residues
        s_residues = []
        for residue in structure.get_residues():
            s_residues.append(residue)
        # Sort structure residues
        s_residues = sorted(s_residues, key=lambda x: x.get_id()[1])
        # Get fragment instance
        fragment = generate_structure(self.f_id, self.residues, self.structure.header)
        # Compute USR of the fragment
        usr = USR(fragment)
        usr.compute_all()
        f_momenta = usr.momenta
        # Get sizes of the fragment and the structure
        f_size, s_size = len(self.residues), len(s_residues)
        # Check if starting position is legal
        if start + f_size > s_size:
            return
        s_structure = generate_structure('S', s_residues[start:start+f_size], structure.header)
        usr = USR(s_structure)
        usr.compute_all()
        s_momenta = usr.momenta
        # Compute and return the score
        return distance.cosine(f_momenta, s_momenta)
        
    def best_match(self, structure):
        """
        Compute the best match of the fragment with respect to the specified structure.
        
        It it done by translating the fragment on the given structure, and for each iteration it computes
        the similarity between the fragment USR and the sub-structure USR, done via the cosine similarity.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure within which the fragment is to be matched.

        Returns
        -------
        best_idx : int
            The starting (residue) position in the structure where the fragment fits the best.
        best_cosine : float in [0,1]
            The best cosine score associated with `best_idx`.
        """
        # Get structure residues
        s_residues = []
        for residue in structure.get_residues():
            s_residues.append(s_residues)
        # Sort structure residues
        s_residues = sorted(s_residues, key=lambda x: x.get_id()[1])
        # Get fragment instance
        fragment = generate_structure(self.f_id, self.residues, self.structure.header)
        # Compute USR of the fragment
        usr = USR(fragment)
        usr.compute_all()
        f_momenta = usr.momenta
        # Get sizes of the fragment and the structure
        f_size, s_size = len(self.residues), len(s_residues)
        # Keep track of best cosine score and its index
        best_idx = 0
        best_cosine = 1
        # Iterate over the structure residues starting from the desired one
        for i in range(s_size):
            if i + f_size > s_size:
                break
            sub_structure = generate_structure('S', s_residues[i:i+f_size], structure.header)
            usr = USR(sub_structure)
            usr.compute_all()
            sub_s_momenta = usr.momenta
            cosine = distance.cosine(f_momenta, sub_s_momenta)
            if cosine < best_cosine:
                best_cosine = cosine
                best_idx = i
        return (best_idx, best_cosine)