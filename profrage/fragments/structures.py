# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:48:33 2021

@author: Federico van Swaaij
"""
from scipy.spatial import distance

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from fragments.representation import USR
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
    
class Neighborhoods:
    """
    Model the representation of a structure as a set of overlapping neighborhoods.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure to represent
    neighborhoods : dict of int -> fragments.structures.Neighborhood
        The dictionary mapping an ID to its neighborhood
    k : int
        The number of residues to the left and to the right of the centroid. The maximal number of residues
        per neighborhood is 2k+1.
    max_inters : int
        The maximum number of interacting neighborhoods a neighborhood can have.
    """
    
    def __init__(self, structure, k, max_inters=3):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure to represent.
        k : int
            The number of residues to the left and to the right of the centroid.
        max_inters : int, optional
            The maximum number of interacting neighborhoods a neighborhood can have. The default is 3.

        Returns
        -------
        None.
        """
        self.structure = structure
        self.neighborhoods = {}
        self.k = k
        self.max_inters = max_inters
        
    def __len__(self):
        """
        Return the number of neighborhoods.

        Returns
        -------
        int
            The number of neighborhoods.
        """
        return len(self.neighborhoods)
    
    def __getitem__(self, idx):
        """
        Return the neighborhood at the specified index.

        Parameters
        ----------
        idx : int
            The index.

        Returns
        -------
        fragments.structures.Neighborhood
            The neighborhood at the specified index.
        """
        return self.neighborhoods[idx]
        
    def __iter__(self):
        """
        Return an iterable object over the neighborhoods.

        Returns
        -------
        list_iterator
            The neighborhoods iterator.
        """
        return iter(self.neighborhoods)
        
    def generate(self):
        """
        Generate the neighborhoods.

        Returns
        -------
        None.
        """
        # Get residues and define data structures
        residues = []
        for residue in self.structure.get_residues():
            r_id = residue.get_id()
            if r_id[0] == ' ':
                residues.append(residue)
        # Sort to ensure the sequence is correct
        residues = sorted(residues, key=lambda x: x.get_id()[1])
        n = len(residues)
        # Compute neighborhoods
        for i in range(n):
            min_idx = max(0, i-self.k)
            max_idx = min(i+self.k+1, n) # add one b/c later int range the upper index is exclusive
            self.neighborhoods[i] = Neighborhood(i, residues[min_idx:max_idx], max_inters=self.max_inters) # residue i is also added here
        # For each neighborhood compute its USR momenta
        for i in self.neighborhoods:
            self.neighborhoods[i].compute_momenta()
    
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
    usr_momenta : numpy.ndarray
        The USR momenta vector.
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
        self.usr_momenta = None
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
        
    def compute_momenta(self):
        """
        Compute the USR momenta for the neighborhood.

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
        usr = USR(c_structure)
        usr.compute_all()
        self.usr_momenta = usr.momenta
    
    