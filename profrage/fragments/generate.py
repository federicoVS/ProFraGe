# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:59:26 2021

@author: Federico van Swaaij
"""

import os
import itertools
import json
from scipy.spatial import distance
from Bio.PDB import NeighborSearch, Selection
from Bio.PDB.mmtf import MMTFParser
from fragments.data import Fragment, Neighborhood
from fragments.graphs import UUGraph
from utils.structure import get_residue_center, structure_length, generate_structure
from utils.io import to_mmtf, to_pdb, parse_cmap
from utils.ProgressBar import ProgressBar

class Generator:
    """
    The abstract fragment-generator class.
    
    Attributes
    ----------
    fragments : dict of str -> list of fragments.Fragment
        The dictionary of fragments. Each fragment is identified by the PDB ID of the structure is belongs
        to. This points a Fragment instance.
    """
    
    def __init__(self):
        """
        Initialize the class.

        Returns
        -------
        None.
        """
        self.fragments = {}
        
    def save(self, out_dir, ext='.mmtf'):
        """
        Save the fragments in the specified format in the specified directory.

        Parameters
        ----------
        out_dir : str
            The directory where to save the fragments.
        ext : str, optional
            The file format to save the structure. The default is '.mmtf'.

        Returns
        -------
        None.
        """
        for pdb_id in self.fragments:
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
            for fragment in self.fragments[pdb_id]:
                # Generate fragment
                s_frag = generate_structure(fragment.f_id, fragment.residues, fragment.structure.header)
                # Save structure to specified format
                if ext == '.mmtf':
                    to_mmtf(s_frag, fragment.f_id, out_dir=out_dir+dir_name)
                elif ext == '.pdb':
                    to_pdb(s_frag, fragment.f_id, out_dir=out_dir+dir_name)
        
    def generate(self):
        """
        Generate the fragments. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        """
        pass

class SingleGenerator(Generator):
    """
    The abstract single-fragment-generator class.
    
    It is single in the sense that it is based on the on fragments based on a single fragments.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure of which to generate fragments.
    """
    
    def __init__(self, structure):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.

        Returns
        -------
        None.
        """
        super(SingleGenerator, self).__init__()
        self.structure = structure
         
class CCGen(SingleGenerator):
    """
    Generate fragment from a given structure based on connected components.
    
    The structure is encoded as a graph, where each residue represents as node. The edges represent
    connections between residues, where a connection results from a distance cutoff between the
    residues: this results in an unweighted graph.
    The procedure results in non-overlapping sets of fragments, each corresponding to a connected
    component of the graph.
    
    Attributes
    ----------
    radius : float
        The maximum distance between two residues, over which the cutoff is carried out.
    """
    
    def __init__(self, structure, radius):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        radius : float
            The maximum cutoff radius.

        Returns
        -------
        None.
        """
        super(CCGen, self).__init__(structure)
        self.radius = radius
        
    def generate(self):
        """
        Generate the fragments by applying the connected components algorithms over the structure graph.

        Returns
        -------
        None.
        """
        # Initialize the fragment dictionary
        self.fragments[self.structure.get_id()] = []
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
                graph.add_edge(vertex_dict[cr], vertex_dict[target_residue])
        # Compute the connected components
        graph.compute_connected_components()
        # Retrieve the residues
        frag_id = 1
        for cc in graph.connected_components:
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            fragment = Fragment(self.structure, f_id)
            for vertex in cc:
                fragment.add_residue(index_dict[vertex])
            self.fragments[self.structure.get_id()].append(fragment)
            frag_id += 1
            
class KSeqGen(SingleGenerator):
    """
    Generates fragments by clustering similar neighborhoods, each of which composed by 2k+1 residues.
    
    Each residue is paired with its k neighbors to the left and right (if any), resulting in a substructure.
    This structure is then analyzed using Ultrafast Shape Recognition and compared to other neighborhoods.
    If similar, they are clustered in contiguously to form a fragment.
    
    Attributes
    ----------
    neighborhoods : dict of int -> Neighborhood
        A dictionary mapping the index of the centroid residue to a Neighborhood instance, representing
        the sequential neighborhood.
    k : int
        The number of neighbors for each residue.
    usr_thr : float in [0,1]
        The similarity threshold under which two neighborhoods are considered to be similar.
    """
    
    def __init__(self, structure, k, usr_thr):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        k : int
            The number of residues for each neighbor.
        usr_thr : float in [0,1]
            The similarity threshold between two neighborhoods. The smaller the better.

        Returns
        -------
        None.
        """
        super(KSeqGen, self).__init__(structure)
        self.neighborhoods = {}
        self.k = k
        self.usr_thr = usr_thr
    
    def all_similarity(self, fragment, candidate):
        """
        Check if the candidate neighborhood is similar to all other neighborhoods in the cluster.

        Parameters
        ----------
        fragment : list of fragments.data.Neighborhood
            The list of neighborhoods, which represents a fragment.
        candidate : fragments.data.Neighborhood
            The candidate neighborhood.

        Returns
        -------
        bool
            Whether the candidate neighborhood is to be added to the fragment.
        """
        for n_id in fragment:
            momenta_n = self.neighborhoods[n_id].usr_momenta
            momenta_c = candidate.usr_momenta
            cosine = distance.cosine(momenta_n, momenta_c)
            if cosine > self.usr_thr:
                return False
        return True
    
    def compute_neighborhoods(self):
        """
        Compute the neighborhoods for each residue.

        Returns
        -------
        None.
        """
        # Get residues and define data structures
        n = structure_length(self.structure)
        residues = []
        for residue in self.structure.get_residues():
            residues.append(residue)
        # Sort to ensure the sequence is correct
        residues = sorted(residues, key=lambda x: x.get_id()[1])
        # Compute neighborhoods
        for i in range(n):
            min_idx = max(0, i-self.k)
            max_idx = min(i+self.k+1, n) # add one b/c later int range the upper index is exclusive
            self.neighborhoods[i] = Neighborhood(i, residues[min_idx:max_idx]) # residue i is also added here
        # For each centroid, define a dummy Structure object and compute its USR
        for n_id in self.neighborhoods:
            self.neighborhoods[n_id].compute_momenta()
        
    def generate(self):
        """
        Generate the fragments using USR similarities.

        Returns
        -------
        None.
        """
        # Compute neighborhoods
        self.compute_neighborhoods()
        # Define fragment dictionary which points to the indices of the neighborhood
        frag_dict = {} # int -> list of int
        frag_id = 1
        # Iterate over each neighbor of neighborhoods
        for i in range(len(self.neighborhoods)):
            frag_dict[frag_id] = []
            frag_dict[frag_id].append(i)
            for j in range(i+1, len(self.neighborhoods)):
                if self.all_similarity(frag_dict[frag_id], self.neighborhoods[j]):
                    frag_dict[frag_id].append(j)
                else:
                    break
            for j in range(i-1, -1, -1):
                if self.all_similarity(frag_dict[frag_id], self.neighborhoods[j]):
                    frag_dict[frag_id].append(j)
                else:
                    break
            frag_id += 1
        # Build the fragments
        self.fragments[self.structure.get_id()] = []
        for frag_id in frag_dict:
            n_ids = frag_dict[frag_id]
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            fragment = Fragment(self.structure, f_id)
            for n_id in n_ids:
                neighbor = self.neighborhoods[n_id]
                for residue in neighbor.residues:
                    fragment.add_residue(residue)
            self.fragments[self.structure.get_id()].append(fragment)
            
class KSeqTerGen(SingleGenerator):
    """
    Generates fragments by clustering similar neighborhoods, each of which composed by 2k+1 residues.
    
    Differently from `KSeqGen`, this version considers tertiary structure information in the form
    of interactions between residues belonging to the different residues.
    
    Attributes
    ----------
    neighborhoods : dict of int -> Neighborhood
        A dictionary mapping the index of the centroid residue to a Neighborhood instance, representing
        the sequential neighborhood.
    k : int
        The number of residues for each neighbor.
    usr_thr : float in [0,1]
        The similarity threshold between two neighborhoods. A small USR score means two neighborhoods are
        very similar.
    cmap_file : str
        The CMAP file containing interaction information.
    f_thr : float in [0,1]
        The interaction threshold between two residues. A high interaction score means two residues have
        a very high interaction.
    max_size : int
        The maximum number of neighborhoods per fragment.
    """
    
    def __init__(self, structure, k, usr_thr, cmap_file, f_thr=0.1, max_size=5):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to compute the fragments.
        k : int
            The number of residues for each neighbor.
        usr_thr : float in [0,1]
            The similarity threshold between two neighborhoods.
        cmap_file : str
            The contact map file.
        f_thr : float in [0,1], optional
            The interaction threshold between two residues. The default is 0.1.
        max_size : int, optional
            The maximum number of neighborhoods per fragment. The default is 5.

        Returns
        -------
        None.
        """
        super(KSeqTerGen, self).__init__(structure)
        self.neighborhoods = {}
        self.k = k
        self.usr_thr = usr_thr
        self.cmap_file = cmap_file
        self.f_thr = f_thr
        self.max_size = max_size
        
    def all_similarity(self, fragment, candidate):
        """
        Check if the candidate neighborhood is similar to all other neighborhoods in the cluster.

        Parameters
        ----------
        fragment : list of fragments.data.Neighborhood
            The list of neighborhoods, which represents a fragment.
        candidate : fragments.data.Neighborhood
            The candidate neighborhood.

        Returns
        -------
        bool
            Whether the candidate neighborhood is to be added to the fragment..
        float
            The sum of the cosine score. A lower sum indicates stronger similarity.
        """
        cosine_sum = 0
        for n_id in fragment:
            momenta_n = self.neighborhoods[n_id].usr_momenta
            momenta_c = candidate.usr_momenta
            cosine = distance.cosine(momenta_n, momenta_c)
            if cosine > self.usr_thr:
                return False, -1
            cosine_sum += cosine
        return True, cosine_sum
        
    def detect_interactions(self, entries, neigh_1, neigh_2):
        """
        Detect interaction between residues belonging to the specified neighborhoods.

        Parameters
        ----------
        entries : list of (str, str, int, int, float)
            The list containing the entries in the CMAP.
        neigh_1 : fragments.data.Neighborhood
            The first neighborhood.
        neigh_2 : fragments.data.Neighborhood
            The second neighborhood.

        Returns
        -------
        interact : bool
            Whether there is interaction between the two neighborhoods.
        best_f : float
            The sum of the interaction scores. A higher sum indicates stronger interaction.
        """
        interact, f_sum = False, 0
        r_pairs = list(itertools.product(neigh_1.residues, neigh_2.residues))
        for entry in entries:
            _, _, r_idx_1, r_idx_2, f = entry
            for res_1, res_2 in r_pairs:
                r_1_id_1, r_2_id_1 = res_1.get_id()[1], res_2.get_id()[1]
                r_idx_1_match = r_1_id_1 == r_idx_1 or r_2_id_1 == r_idx_1
                r_idx_2_match = r_1_id_1 == r_idx_2 or r_2_id_1 == r_idx_2
                if r_1_id_1 != r_2_id_1 and r_idx_1_match and r_idx_2_match and f > self.f_thr:
                    interact = True
                    f_sum += f
        return interact, f_sum
        
    def compute_neighborhoods(self, entries):
        """
        Generate the neighborhoods by grouping 2k+1 sequential residues and keeping track of interacting neighborhoods.

        Parameters
        ----------
        entries : list of (str, str, int, int, float)
            The list containing the entries in the CMAP.

        Returns
        -------
        None.
        """
        # Get residues and define data structures
        n = structure_length(self.structure)
        residues = []
        for residue in self.structure.get_residues():
            residues.append(residue)
        # Sort to ensure the sequence is correct
        residues = sorted(residues, key=lambda x: x.get_id()[1])
        # Compute neighborhoods
        for i in range(n):
            min_idx = max(0, i-self.k)
            max_idx = min(i+self.k+1, n) # add one b/c later int range the upper index is exclusive
            self.neighborhoods[i] = Neighborhood(i, residues[min_idx:max_idx]) # residue i is also added here
        # For each neighborhood compute its USR momenta
        for n_id in self.neighborhoods:
            self.neighborhoods[n_id].compute_momenta()
        # For each neighborhood, check if it has interaction with any residue of any other neighborhood
        for i in range(len(self.neighborhoods)-1):
            for j in range(i+1, len(self.neighborhoods)):
                neigh_1, neigh_2 = self.neighborhoods[i], self.neighborhoods[j]
                interact, f_sum = self.detect_interactions(entries, neigh_1, neigh_2)
                if interact:
                    self.neighborhoods[i].add_interaction(self.neighborhoods[j], f_sum)
                    self.neighborhoods[j].add_interaction(self.neighborhoods[i], f_sum)
        for i in range(len(self.neighborhoods)):
            self.neighborhoods[i].filter_interactions()
    
    def generate(self):
        """
        Generate the fragments.
        
        Returns in case of errors in reading the contact map.

        Returns
        -------
        None.
        """
        # Parse the CMAP file and get all the entries
        entries = parse_cmap(self.cmap_file)
        # Check if the entries are valid, if not just exit
        if entries is None:
            return
        # Compute the neighborhoods
        self.compute_neighborhoods(entries)
        # Define fragment dictionary which points to the indices of the neighborhood
        frag_dict = {} # int -> list of int
        frag_id = 1
        n = len(self.neighborhoods)
        # Iterate over each neighbor of neighborhoods
        for i in range(n):
            frag_dict[frag_id] = []
            frag_dict[frag_id].append(i)
            upper_size = int(self.max_size/(len(self.neighborhoods[i].interactions)+1))
            l_idx, u_idx = 1, 1
            # Iterate over possible combinations
            for j in range(1, upper_size):
                lower = i - l_idx
                upper = i + u_idx
                sl, su = False, False
                if lower < 0 and upper < n:
                    su, c_sum_u = self.all_similarity(frag_dict[frag_id], self.neighborhoods[upper])
                elif lower >= 0 and upper >= n:
                    sl, c_sum_l = self.all_similarity(frag_dict[frag_id], self.neighborhoods[lower])
                elif lower >= 0 and upper < n:
                    sl, c_sum_l = self.all_similarity(frag_dict[frag_id], self.neighborhoods[lower])
                    su, c_sum_u = self.all_similarity(frag_dict[frag_id], self.neighborhoods[upper])
                # Find the best match
                if sl and su:
                    if c_sum_l < c_sum_u:
                        frag_dict[frag_id].append(lower)
                        l_idx += 1
                    else:
                        frag_dict[frag_id].append(upper)
                        u_idx += 1
                elif sl and not su:
                    frag_dict[frag_id].append(lower)
                    l_idx += 1
                elif not sl and su:
                    frag_dict[frag_id].append(upper)
                    u_idx += 1
                else:
                    break # not suitable candidate found
            frag_id += 1
        # Retrieve fragments
        self.fragments[self.structure.get_id()] = []
        for frag_id in frag_dict:
            n_ids = frag_dict[frag_id]
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            fragment = Fragment(self.structure, f_id)
            for n_id in n_ids:
                neighbor = self.neighborhoods[n_id]
                for residue in neighbor.residues:
                    fragment.add_residue(residue)
                for inter in neighbor.interactions:
                    for residue in inter[0].residues:
                        fragment.add_residue(residue)
            self.fragments[self.structure.get_id()].append(fragment)
            
class ConFindGen(SingleGenerator):
    """
    Generate fragments using interaction information between the residues.
    
    The contact information is retrieved using the ConFind tool. The fragments are then computed using
    the connected components algorithm.
    
    Attrubutes
    ----------
    cmap_file : str
        The file holding the contact information.
    dcut : float in [0,1]
        The threshold above which two residue are considered to be interacting.
    """
    
    def __init__(self, structure, cmap_file, dcut=0.1):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to generate fragments.
        cmap_file : str
            The file holding the contact information.
        dcut : float in [0,1], optional
            The threshold above which two residue are considered to be interacting. The default is 0.1.

        Returns
        -------
        None.
        """
        super(ConFindGen, self).__init__(structure)
        self.cmap_file = cmap_file
        self.dcut = dcut
    
    def generate(self):
        """
        Generate the fragments using the connected components algorithm.

        Returns
        -------
        None.
        """
        # Initialize the fragment dictionary
        self.fragments[self.structure.get_id()] = []
        # Get and index residues
        index = 0
        residue_dict = {}
        vertex_dict = {}
        for residue in self.structure.get_residues():
            r_id = residue.get_id()
            residue_dict[r_id[1]] = index
            vertex_dict[index] = residue
            index += 1
        # Create graph
        n = structure_length(self.structure)
        graph = UUGraph(n)
        entries = parse_cmap(self.cmap_file)
        for entry in entries:
            _, _, res_idx_1, res_idx_2, dist = entry
            if dist > self.dcut:
                graph.add_edge(residue_dict[res_idx_1], residue_dict[res_idx_2])
                graph.add_edge(residue_dict[res_idx_2], residue_dict[res_idx_1])
        # Compute the connected components
        graph.compute_connected_components()
        # Retrieve the residues
        frag_id = 1
        for cc in graph.connected_components:
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            fragment = Fragment(self.structure, f_id)
            for vertex in cc:
                fragment.add_residue(vertex_dict[vertex])
            self.fragments[self.structure.get_id()].append(fragment)
            frag_id += 1
            
class FuzzleGen(Generator):
    """
    Generate fragments based on the Fuzzle fragment database.
    
    Attributes
    ----------
    data : dict
        A dictionary representing the JSON Fuzzle fragment network.
    verbose : bool
        Whether to print progress information.
    """
    
    def __init__(self, fuzzle_json, verbose=False):
        """
        Initialize the class by reading the input JSON file.

        Parameters
        ----------
        fuzzle_json : str
            The JSON file describing the Fuzzle fragment network.
        verbose : bool, optional
            Whether to print progress information. The default is False.

        Returns
        -------
        None.
        """
        super(FuzzleGen, self).__init__()
        with open(fuzzle_json) as fj:
            self.data = json.load(fj)
        self.verbose = verbose
        
    def generate(self):
        """
        Generate the fragments by getting the residues from each specified fragment.

        Returns
        -------
        None.
        """
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
            if count > 50:
                return
            pdb_id = node['domain'][1:5].upper()
            chain_id = node['domain'][5].upper()
            structure = parser.get_structure_from_url(pdb_id)
            if structure is not None:
                if pdb_id not in self.fragments:
                    self.fragments[pdb_id] = []
                f_id = pdb_id + '_' + str(node['start']) + '_' + str(node['end'])
                fragment = Fragment(structure, f_id)
                for residue in structure.get_residues():
                    c_id = residue.get_full_id()[2]
                    r_id = residue.get_id()
                    if chain_id != '.' and chain_id != '_' and chain_id == c_id:
                        if r_id[1] >= node['start'] and r_id[1] <= node['end']:
                            fragment.add_residue(residue)
                    elif chain_id == '.' or chain_id == '_':
                        if r_id[1] >= node['start'] and r_id[1] <= node['end']:
                            fragment.add_residue(residue)
                self.fragments[pdb_id].append(fragment)
                frag_id += 1
        if self.verbose:
            progress_bar.end()
                    
        
        
        
        