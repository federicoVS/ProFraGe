# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:59:26 2021

@author: Federico van Swaaij
"""

import os
import json
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from Bio.PDB.mmtf import MMTFParser

from fragment.Fragment import Fragment
from fragment.builders import Neighborhoods
from utils.structure import generate_structure
from utils.io import to_mmtf, to_pdb
from utils.ProgressBar import ProgressBar

class Miner:
    """
    The abstract fragment-miner class.
    
    Attributes
    ----------
    fragments : dict of str -> list of fragment.Fragment
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
        
    def mine(self):
        """
        Mine the fragments from the structure. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        """
        pass

class SingleMiner(Miner):
    """
    The abstract single-fragment-miner class.
    
    It is single in the sense that it is based on the on fragments based on a single fragments.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure from which to mine fragments.
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
        super(SingleMiner, self).__init__()
        self.structure = structure
            
class KSeqMine(SingleMiner):
    """
    Mine fragments by clustering similar neighborhoods, each of which composed by 2k+1 residues.
    
    Each residue is paired with its k neighbors to the left and right (if any), resulting in a substructure.
    This structure is then analyzed using Ultrafast Shape Recognition and compared to other neighborhoods.
    If similar, they are clustered in contiguously to form a fragment.
    
    Attributes
    ----------
    neighborhoods : fragment.builders.Neighborhoods
        An object represeting the neighborhoods.
    k : int
        The number of neighbors for each residue.
    cosine_thr : float in [0,1]
        The similarity threshold under which two neighborhoods are considered to be similar.
    """
    
    def __init__(self, structure, Rep, k, cosine_thr, f_thr=0.1, max_size=5):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to compute the fragments.
        Rep : structure.representation.Representation
            The class of structure representation.
        k : int
            The number of residues to the left and to the right of the centroid.
        cosine_thr : float in [0,1]
            The similarity threshold between two neighborhoods. The lower the tighter.
        f_thr : float in [0,1], optional
            The interaction threshold between two residues. The default is 0.1.

        Returns
        -------
        None.
        """
        super(KSeqMine, self).__init__(structure)
        self.neighborhoods = Neighborhoods(structure, Rep, None, k, f_thr=f_thr)
        self.k = k
        self.cosine_thr = cosine_thr
    
    def all_similarity(self, fragment, candidate):
        """
        Check if the candidate neighborhood is similar to all other neighborhoods in the cluster.

        Parameters
        ----------
        fragment : list of structure.Neighborhood.Neighborhood
            The list of neighborhoods, which represents a fragment.
        candidate : structure.Neighborhood.Neighborhood
            The candidate neighborhood.

        Returns
        -------
        bool
            Whether the candidate neighborhood is to be added to the fragment.
        """
        for n_id in fragment:
            features_n = self.neighborhoods[n_id].features
            features_c = candidate.features
            cosine = distance.cosine(features_n, features_c)
            if cosine > self.cosine_thr:
                return False
        return True
        
    def mine(self):
        """
        Mine the fragments using USR similarities.

        Returns
        -------
        None.
        """
        # Compute neighborhoods
        self.neighborhoods.generate()
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
            
class KSeqTerMine(SingleMiner):
    """
    Mine fragments by clustering similar neighborhoods, each of which composed by 2k+1 residues.
    
    Differently from `KSeqGen`, this version considers tertiary structure information in the form
    of interactions between residues belonging to the different residues.
    
    Attributes
    ----------
    neighborhoods : fragments.builders.Neighborhoods
        An object represeting the neighborhoods. Note that this parameter must be a class,
        not an instance of a class.
    cosine_thr : float in [0,1]
        The similarity threshold between two neighborhoods. A small score means two neighborhoods are
        very similar.
    max_size : int
        The maximum number of neighborhoods per fragment.
    """
    
    def __init__(self, structure, Rep, k, cosine_thr, cmap, f_thr=0.1, max_size=4):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to compute the fragments.
        Rep : structure.representation.Representation
            The class of structure representation.
        k : int
            The number of residues to the left and to the right of the centroid.
        cosine_thr : float in [0,1]
            The similarity threshold between two neighborhoods. The lower the tighter.
        cmap : str
            The CMAP file.
        f_thr : float in [0,1], optional
            The interaction threshold between two residues. The default is 0.1.
        max_size : int, optional
            The maximum number of neighborhoods per fragment. The default is 4.

        Returns
        -------
        None.
        """
        super(KSeqTerMine, self).__init__(structure)
        self.neighborhoods = Neighborhoods(structure, Rep, cmap, k, f_thr=f_thr)
        self.cosine_thr = cosine_thr
        self.max_size = max_size
        
    def all_similarity(self, fragment, candidate):
        """
        Check if the candidate neighborhood is similar to all other neighborhoods in the cluster.

        Parameters
        ----------
        fragment : list of structure.Neighborhood.Neighborhood
            The list of neighborhoods, which represents a fragment.
        candidate : structure.Neighborhood.Neighborhood
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
            features_n = self.neighborhoods[n_id].features
            features_c = candidate.features
            cosine = distance.cosine(features_n, features_c)
            if cosine > self.cosine_thr:
                return False, -1
            cosine_sum += cosine
        return True, cosine_sum
    
    def mine(self):
        """
        Generate the fragments.
        
        Returns in case of errors in reading the contact map.

        Returns
        -------
        None.
        """
        # Compute the neighborhoods
        self.neighborhoods.generate()
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
            for j in range(0, upper_size):
                lower = i - l_idx
                upper = i + u_idx
                sl, su = False, False
                c_sum_l, c_sum_u = 0, 0
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
            
class HierarchyMine(SingleMiner):
    """
    Implement hierarchical clustering for fragment mining.
    
    It employs the sklearn.cluster.AgglomerativeClustering algorithm.
    
    Attributes
    ----------
    builder : fragments.builder.Builder
        The object building the fragments.
    n_clusters : int
        The number of clusters.
    connectivity : bool
        Whether to use the adjacency matrix in the clustering.
    linkage : str
        The linkage criterion to use.
    """
    
    def __init__(self, structure, builder, n_clusters=2, connectivity=False, linkage='ward'):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure from which to generate the fragments.
        builder : fragments.builder.Builder
            The object building the fragments.
        n_clusters : int, optional
            The number of clusters. The default is 2.
        connectivity : bool, optional
            Whether to use the adjacency matrix. The default is False.
        linkage : str, optional
            The linkage criterion to use. The default is 'ward'.

        Returns
        -------
        None.
        """
        super(HierarchyMine, self).__init__(structure)
        self.builder = builder
        self.n_clusters = n_clusters
        self.connectivity = connectivity
        self.linkage = linkage
        
    def mine(self):
        """
        Mine the fragments.

        Returns
        -------
        None.
        """
        # Generate features and check if they are valid
        self.builder.generate()
        X = self.builder.get_features()
        if X.shape[0] < 2:
            return
        # Generate adjacency matrix (if requested)
        if self.connectivity:
            A = self.builder.get_adjacency()
        else:
            A = None
        # Initialize clustering algorithm
        aggcl = AgglomerativeClustering(n_clusters=self.n_clusters, connectivity=A, linkage=self.linkage)
        aggcl.fit(X)
        # Initialize the fragment dictionary with the number of clusters
        self.fragments[self.structure.get_id()] = [None for i in range(self.n_clusters)]
        # Retrieve clusters to form fragments
        for c_id in aggcl.labels_:
            # Define fragment ID
            frag_id = c_id + 1
            # Check if the fragment already
            if self.fragments[self.structure.get_id()][c_id] is None:
                # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
                f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
                self.fragments[self.structure.get_id()][c_id] = Fragment(self.structure, f_id)
            residues = self.builder.get_residues_at(c_id)
            for residue in residues:
                self.fragments[self.structure.get_id()][c_id].add_residue(residue)
            
class FuzzleMine(Miner):
    """
    Mine fragments based on the Fuzzle fragment database.
    
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
        super(FuzzleMine, self).__init__()
        with open(fuzzle_json) as fj:
            self.data = json.load(fj)
        self.verbose = verbose
        
    def mine(self):
        """
        Mine the fragments by getting the residues from each specified fragment.

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
    