# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:59:26 2021

@author: FVS
"""

import os
import json
from Bio.PDB import NeighborSearch, Selection
from Bio.PDB.mmtf import MMTFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from fragments.graphs import UUGraph
from utils.structure import get_residue_center
from utils.io import to_mmtf
from utils.ProgressBar import ProgressBar

class Generator:
    '''
    The abstract fragment-generator class.
    
    Attributes
    ----------
    fragments : dict of str -> (Bio.PDB.Structure, dict of int -> (str, list of Bio.PDB.Residue))
        The dictionary of fragments. Each fragment is identified by the PDB ID of the structure is belongs
        to. This points to a tuple, where the first element points to the structure instance, and the
        second is a dictionary, where each fragment ID points to a tuple, where the first element is its
        name and the second is the list of residues composing it.
    '''
    
    def __init__(self):
        '''
        Initializes the class.

        Returns
        -------
        None.
        '''
        self.fragments = {}
        
    def create_fragment(self, pdb_id, frag_id):
        '''
        Creates a fragment based on a list of residues

        Parameters
        ----------
        pdb_id : str
            The ID of the structure the fragments belong to.
        frag_id : int
            The fragment ID.

        Returns
        -------
        fragment : Bio.PDB.Structure
            The fragment.
        '''
        # Get fragment ID and residues
        structure, frag_dict = self.fragments[pdb_id]
        s_id, residues = frag_dict[frag_id]
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
        fragment.header = structure.header
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
        for pdb_id in self.fragments:
            _, frag_dict = self.fragments[pdb_id]
            # Build proper output directory
            splitted = pdb_id.split('_')
            p_id = splitted[0]
            dir_name = ''
            if len(splitted) > 1 and splitted[1].isalpha():
                c_id = splitted[1]
                dir_name += p_id + '_' + c_id + '/fragments/'
            else:
                dir_name += p_id + '/fragments/'
            if not os.path.exists(out_dir+dir_name):
                os.makedirs(out_dir+dir_name)
            for frag_id in frag_dict:
                # Generate fragment
                fragment = self.create_fragment(pdb_id, frag_id)
                # Save to MMTF format
                to_mmtf(fragment, self.fragments[pdb_id][1][frag_id][0], out_dir=out_dir+dir_name)
        
    def generate(self):
        '''
        Generate the fragments. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        '''
        pass

class SingleGenerator(Generator):
    '''
    The abstract single-fragment-generator class. It is single in the sense that it is based on the
    on fragments based on a single fragments.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure of which to generate fragments.
    '''
    
    def __init__(self, structure, remove_hw=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        remove_hw : bool, optional
            Whether to remove heterogeneous and water atoms. The default is False.

        Returns
        -------
        None.
        '''
        super(SingleGenerator, self).__init__()
        self.structure = structure
        if remove_hw:
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
         
class CCGen(SingleGenerator):
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
    
    def __init__(self, structure, radius, remove_hw=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        radius : float
            The maximum cutoff radius.
        remove_hw : bool, optional
            Whether to remove heterogeneous and water atoms. The default is False.

        Returns
        -------
        None.
        '''
        super(CCGen, self).__init__(structure, remove_hw=remove_hw)
        self.radius = radius
        
    def generate(self):
        '''
        Generates the fragments by applying the connected components algorithms over the structure
        graph.

        Returns
        -------
        None.
        '''
        # Initialize the fragment dictionary
        self.fragments[self.structure.get_id()] = (self.structure, {})
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
        frag_id = 1
        for cc in graph.connected_components:
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            s_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            self.fragments[self.structure.get_id()][1][frag_id] = (s_id, [])
            for vertex in cc:
                self.fragments[self.structure.get_id()][1][frag_id][1].append(index_dict[vertex])
            frag_id += 1
            
class SeqGen(SingleGenerator):
    '''
    Generates fragments as contiguous subsets of k residues. For example, if the protein has sequence
    SPQR and k=2, then the generated fragments are SP, PQ, and QR.
    
    Attributes
    ----------
    k : int
        The number of consecutive residues forming a fragment.
    '''
    
    def __init__(self, structure, k, remove_hw=False):
        '''
        Initializes the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        k : int
            The number of residues forming a fragment.
        remove_hw : bool, optional
            Whether to remove heterogeneous and water atoms. The default is False.

        Returns
        -------
        None.
        '''
        super(SeqGen, self).__init__(structure, remove_hw=remove_hw)
        self.k = k
        
    def generate(self):
        '''
        Generates the fragments by finding contiguous subsets of residues.

        Returns
        -------
        None.
        '''
        # Initialize the fragment dictionary
        self.fragments[self.structure.get_id()] = (self.structure, {})
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
                # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
                s_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
                self.fragments[self.structure.get_id()][1][frag_id] = (s_id, [])
                self.fragments[self.structure.get_id()][1][frag_id][1].append(residues[j])
            frag_id += 1
            i += 1
            
class FuzzleGen(Generator):
    '''
    Generates fragments based on the Fuzzle fragment database.
    
    Attributes
    ----------
    data : dict
        A dictionary representing the JSON Fuzzle fragment network.
    verbose : bool
        Whether to print progress information.
    '''
    
    def __init__(self, fuzzle_json, verbose=False):
        '''
        Initializes the class by reading the input JSON file.

        Parameters
        ----------
        fuzzle_json : str
            The JSON file describing the Fuzzle fragment network.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        '''
        super(FuzzleGen, self).__init__()
        with open(fuzzle_json) as fj:
            self.data = json.load(fj)
        self.verbose = verbose
        
    def generate(self):
        '''
        Generates the fragments by getting the residues from each specified fragment. 

        Returns
        -------
        None.
        '''
        parser = MMTFParser()
        frag_id = 1
        count = 1
        progress_bar = ProgressBar()
        if self.verbose:
            print('Generating fragments...')
            progress_bar.start()
        for node in self.data['nodes']:
            if self.verbose:
                progress_bar.step(count, len(self.data['nodes']))
                count += 1
            pdb_id = node['domain'][1:5].upper()
            chain_id = node['domain'][5].upper()
            structure = parser.get_structure_from_url(pdb_id)
            if structure is not None:
                if pdb_id not in self.fragments:
                    self.fragments[pdb_id] = (structure, {})
                s_id = pdb_id + '_' + str(node['start']) + '_' + str(node['end'])
                self.fragments[pdb_id][1][frag_id] = (s_id, [])
                for residue in structure.get_residues():
                    c_id = residue.get_full_id()[2]
                    r_id = residue.get_id()
                    if chain_id != '.' and chain_id != '_' and chain_id == c_id:
                        if r_id[1] >= node['start'] and r_id[1] <= node['end']:
                            self.fragments[pdb_id][1][frag_id][1].append(residue)
                    elif chain_id == '.' or chain_id == '_':
                        if r_id[1] >= node['start'] and r_id[1] <= node['end']:
                            self.fragments[pdb_id][1][frag_id][1].append(residue)
                frag_id += 1
        if self.verbose:
            progress_bar.end()
                    
        
        
        
        