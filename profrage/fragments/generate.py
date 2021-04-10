# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:59:26 2021

@author: FVS
"""

import os
from Bio.PDB import NeighborSearch, Selection
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from fragments.graphs import UUGraph
from utils.structure import get_residue_center
from utils.io import to_mmtf

class Generator:
    '''
    The abstract fragment-generator class.
    
    Attributes
    ----------
    fragments : dict of int -> list of Bio.PDB.Residue
        The dictionary of fragments. Each fragment is identified by an index, which points to a list
        of the residues belonging to it.
    structure : Bio.PDB.Structure
        The structure of which to generate fragments.
    '''
    
    def __init__(self, structure):
        '''
        Initializes the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.

        Returns
        -------
        None.
        '''
        self.fragments = {}
        self.structure = structure
        self._remove_hetero_water()
        
    def _remove_hetero_water(self):
        residues = []
        for residue in self.structure.get_residues():
            residues.append(residue)
        for residue in residues:
            r_id = residue.get_id()
            if r_id[0] != ' ':
                r_id_full = residue.get_full_id()
                for model in self.structure:
                    found = False
                    if model.get_id() == r_id_full[1]:
                        for chain in model:
                            if chain.get_id() == r_id_full[2]:
                                chain.detach_child(r_id)
                                found = True
                                break
                    if found:
                        break
        
    def create_fragment(self, frag_id):
        '''
        Creates a fragment based on a list of residues

        Parameters
        ----------
        frag_id : int
            The index referring to the list of residues to use.

        Returns
        -------
        fragment : Bio.PDB.Structure
            The fragment.
        '''
        # Get residues
        residues = self.fragments[frag_id]
        # Create the fragment ID (<pdb_id>_<chain_id>_<chain_id><frag_id>)
        s_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
        fragment = Structure(s_id)
        seq_id = 1
        for residue in residues:
            r_full_id = residue.get_full_id()
            # Check if residue model exists, if not add it
            if not fragment.has_id(r_full_id[1]):
                fragment.add(Model(r_full_id[1]))
            # Get correct model for the residue
            for model in fragment:
                been_added = False
                if model.get_id() == r_full_id[1]:
                    # Check if model has the chain, if not add it
                    if not model.has_id(r_full_id[2]):
                        model.add(Chain(r_full_id[2]))
                    # Get correct chain and add residue
                    for chain in model:
                        if chain.get_id() == r_full_id[2]:
                            r_id = r_full_id[3]
                            r = Residue((r_id[0], seq_id, r_id[2]), residue.get_resname(), residue.get_segid())
                            for atom in residue:
                                r.add(atom)
                            chain.add(r)
                            seq_id += 1
                            been_added = True
                            break
                    # If residue has been added then we can exit the loop
                    if been_added:
                        break
        # Add stucture header
        fragment.header = self.structure.header
        return fragment
         
    def save(self, out_dir):
        '''
        Saves the fragments in MMTF format in the specified directory.

        Parameters
        ----------
        out_dir : str
            The directory where to save the fragments.

        Returns
        -------
        None.
        '''
        # Build proper output directory
        out_dir += self.structure.get_id() + '/fragments/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # print(len(self.fragments))
        for frag_id in self.fragments:
            # Generate fragment
            fragment = self.create_fragment(frag_id)
            # Save to MMTF format
            to_mmtf(fragment, fragment.get_id(), out_dir=out_dir)
        
    def generate(self):
        '''
        Generate the fragments. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        '''
        pass
        
class CCG(Generator):
    '''
    Generates fragment from a given structure based on connected components.
    The structure is encoded as a graph, where each residue represents as node. The edges represent
    connections between residues, where a connection results from a distance cutoff between the
    residues: this results in an unweighted graph.
    The procedure results in non-overlapping sets of fragments, each corresponding to a connected
    component of the graph.
    
    Attributes
    ----------
    radius : float
        The maximum distance between two residues, over which the cutoff is carried out.
    '''
    
    def __init__(self, structure, radius):
        '''
        Initializes the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        radius : float
            The maximum cutoff radius.

        Returns
        -------
        None.
        '''
        super(CCG, self).__init__(structure)
        self.radius = radius
        
    def generate(self):
        '''
        Generates the fragments by applying the connected components algorithms over the structure
        graph.

        Returns
        -------
        None.
        '''
        # Create dictionary which maps residues tp integers, and viceversa
        index = 0
        vertex_dict = {}
        index_dict = {}
        for residue in self.structure.get_residues():
            vertex_dict[residue] = index
            index_dict[index] = residue
            index += 1
        # Create graph
        graph = UUGraph(index)
        # Iterate over the residues
        for target_residue in self.structure.get_residues():
            center_coord = get_residue_center(target_residue)
            atoms = Selection.unfold_entities(self.structure, 'A')
            ns = NeighborSearch(atoms)
            close_residues = ns.search(center_coord, self.radius, level='R')
            # Remove the target protein itself
            if target_residue in close_residues:
                close_residues.remove(target_residue)
            for cr in close_residues:
                graph.add_edge(vertex_dict[target_residue], vertex_dict[cr])
        # Compute the connected components
        graph.compute_connected_components()
        # Retrieve the residues
        frag_id = 0
        for cc in graph.connected_components:
            self.fragments[frag_id] = []
            for vertex in cc:
                self.fragments[frag_id].append(index_dict[vertex])
            frag_id += 1
            
class SeqG(Generator):
    '''
    Generates fragments as contiguous subsets of k residues. For example, if the protein has sequence
    SPQR and k=2, then the generated fragments are SP, PQ, and QR.
    
    Attributes
    ----------
    k : int
        The number of consecutive residues forming a fragment.
    '''
    
    def __init__(self, structure, k):
        '''
        Initializes the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        k : int
            The number of residues forming a fragment.

        Returns
        -------
        None.
        '''
        super(SeqG, self).__init__(structure)
        self.k = k
        
    def generate(self):
        '''
        Generates the fragments by finding contiguous subsets of residues.

        Returns
        -------
        None.
        '''
        # Store the residues for convenience
        residues = []
        for residue in self.structure.get_residues():
            residues.append(residue)
        frag_id = 1
        i = 0
        # Find the fragments
        while i + self.k < len(residues):
            self.fragments[frag_id] = []
            for j in range(i, i+self.k):
                self.fragments[frag_id].append(residues[j])
            frag_id += 1
            i += 1
            
        
        
        
        