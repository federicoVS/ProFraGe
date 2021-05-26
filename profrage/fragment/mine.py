# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:59:26 2021

@author: Federico van Swaaij
"""

import os
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from Bio.PDB.mmtf import MMTFParser
import leidenalg
import igraph as ig

from fragment.Fragment import Fragment
from fragment.builders import Neighborhoods
from structure.representation import USR
from utils.structure import build_structure
from utils.io import to_mmtf, to_pdb, parse_cmap
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
        
    def get_fragments(self):
        """
        Return the fragments.

        Returns
        -------
        f_structures : list of Bio.PDB.Structure
            The list of generated fragments.
        """
        f_structures = []
        for pdb_id in self.fragments:
            for fragment in self.fragments[pdb_id]:
                f_structures.append(build_structure(fragment.f_id, fragment.residues, fragment.structure.header))
        return f_structures
        
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
                s_frag = build_structure(fragment.f_id, fragment.residues, fragment.structure.header)
                # Save structure to specified format
                if ext == '.mmtf':
                    to_mmtf(s_frag, fragment.f_id, out_dir=out_dir+dir_name)
                elif ext == '.pdb':
                    to_pdb(s_frag, fragment.f_id, out_dir=out_dir+dir_name)
                    
    def pre_filter(self, jump_thr=20, min_size=12):
        """
        Pre-filters the fragments. This method is meant to be overridden by its subclasses.

        Parameters
        ----------
        jump_thr : int, optional
            The threshold above which two segments in the fragment are considered separated in the pre-filtering. The default is 20.
        min_size : int, optional
            The minimum size for a segment to be considered part of the main fragment structure in the pre-filtering. The default is 12.

        Returns
        -------
        None.
        """
        pass
        
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
    do_pre_filter : bool
        Whether to pre-filter the fragment before saving it.
    jump_thr : int
        The threshold above which two segments in the fragment are considered separated in the pre-filtering.
    min_size : int
        The minimum size for a segment to be considered part of the main fragment structure in the pre-filtering.
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
        
    def pre_filter(self, jump_thr=20, min_size=12):
        """
        Pre-filter the fragments.

        Parameters
        ----------
        jump_thr : int, optional
            The threshold above which two segments in the fragment are considered separated in the pre-filtering. The default is 20.
        min_size : int, optional
            The minimum size for a segment to be considered part of the main fragment structure in the pre-filtering. The default is 12.

        Returns
        -------
        None.
        """
        old_fragments = self.fragments[self.structure.get_id()]
        self.fragments[self.structure.get_id()] = []
        for fragment in old_fragments:
            new_fragment = Fragment(self.structure, fragment.f_id) # the new, filtered fragment
            temp_res = [] # temporary residues to add to the newly formed
            residues = fragment.residues
            if len(residues) <= 0:
                continue
            current = residues[0].get_id()[1]
            for i in range(1, len(residues)):
                residue = residues[i]
                r_id = residue.get_id()
                temp_res.append(residue)
                if r_id[1] - current > jump_thr:
                    if len(temp_res) < min_size:
                        temp_res = []
                        current = r_id[1]
                    else:
                        for tr in temp_res:
                            new_fragment.add_residue(tr)
                        temp_res = []
                        current = r_id[1]
            self.fragments[self.structure.get_id()].append(new_fragment)
            
class KSeqMiner(SingleMiner):
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
    score_thr : float in [0,1]
        The similarity threshold above which two neighborhoods are considered to be similar.
    """
    
    def __init__(self, structure, Rep, k=3, score_thr=0.4, max_inters=3, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to compute the fragments.
        Rep : structure.representation.Representation
            The class of structure representation.
        k : int, optional
            The number of residues to the left and to the right of the centroid. The default is 3.
        score_thr : float in [0,1]
            The similarity threshold between two neighborhoods, the higher the tighter. The default is 0.4.
        max_inters : int, optional
            The maximum number of interactions a neighborhood can have. The default is 3.

        Returns
        -------
        None.
        """
        super(KSeqMiner, self).__init__(structure)
        self.neighborhoods = Neighborhoods(structure, Rep, None, k, max_inters=max_inters)
        self.k = k
        self.score_thr = score_thr
    
    def _all_similarity(self, fragment, candidate):
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
            score = USR.get_similarity_score(features_n, features_c)
            if score < self.score_thr:
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
                if self._all_similarity(frag_dict[frag_id], self.neighborhoods[j]):
                    frag_dict[frag_id].append(j)
                else:
                    break
            for j in range(i-1, -1, -1):
                if self._all_similarity(frag_dict[frag_id], self.neighborhoods[j]):
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
            
class KSeqTerMiner(SingleMiner):
    """
    Mine fragments by clustering similar neighborhoods, each of which composed by 2k+1 residues.
    
    Differently from `KSeqGen`, this version considers tertiary structure information in the form
    of interactions between residues belonging to the different residues.
    
    Attributes
    ----------
    neighborhoods : fragments.builders.Neighborhoods
        An object represeting the neighborhoods. Note that this parameter must be a class,
        not an instance of a class.
    score_thr : float in [0,1]
        The similarity threshold between two neighborhoods. A high score means two neighborhoods are
        very similar.
    max_size : int
        The maximum number of neighborhoods per fragment.
    """
    
    def __init__(self, structure, Rep, cmap, k=3, score_thr=0.4, f_thr=0.1, max_inters=1, max_size=4, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to compute the fragments.
        Rep : structure.representation.Representation
            The class of structure representation.
        cmap : str
            The CMAP file.
        k : int, optional
            The number of residues to the left and to the right of the centroid. The default is 3.
        score_thr : float in [0,1], optional
            The similarity threshold between two neighborhoods, the higher the tighter. The default is 0.4.
        f_thr : float in [0,1], optional
            The interaction threshold between two residues. The default is 0.1.
        max_inters : int, optional
            The maximum number of interactions a neighborhood can have. The default is 1.
        max_size : int, optional
            The maximum number of neighborhoods per fragment. The default is 4.

        Returns
        -------
        None.
        """
        super(KSeqTerMiner, self).__init__(structure)
        self.neighborhoods = Neighborhoods(structure, Rep, cmap, k, f_thr=f_thr, max_inters=max_inters)
        self.score_thr = score_thr
        self.max_size = max_size
        
    def _all_similarity(self, fragment, candidate):
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
            The sum of the cosine score. A higher sum indicates stronger similarity.
        """
        score_sum = 0
        for n_id in fragment:
            features_n = self.neighborhoods[n_id].features
            features_c = candidate.features
            score = USR.get_similarity_score(features_n, features_c)
            if score < self.score_thr:
                return False, -1
            score_sum += score
        return True, score_sum
    
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
                    su, c_sum_u = self._all_similarity(frag_dict[frag_id], self.neighborhoods[upper])
                elif lower >= 0 and upper >= n:
                    sl, c_sum_l = self._all_similarity(frag_dict[frag_id], self.neighborhoods[lower])
                elif lower >= 0 and upper < n:
                    sl, c_sum_l = self._all_similarity(frag_dict[frag_id], self.neighborhoods[lower])
                    su, c_sum_u = self._all_similarity(frag_dict[frag_id], self.neighborhoods[upper])
                # Find the best match
                if sl and su:
                    if c_sum_l > c_sum_u:
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
            
class KTerCloseMiner(SingleMiner):
    """
    Mine fragments by considering close neighborhoods.
    
    Each neighborhood is paired with a maximum of m interacting neighborhoods. Then, each of the
    neighborhoods checks whether it can expand to its left of right, bringing forth a candidate and its
    associated score. The best of these candidates is selected. This procedure continues until either no
    further viable candidates are found or the maximum number of residues is reached.
    
    Attributes
    ----------
    neighborhoods : fragment.builders.Neighborhoods
        An object represeting the neighborhoods.
    k : int
        The number of neighbors for each residue.
    score_thr : float in [0,1]
        The similarity threshold above which two neighborhoods are considered to be similar.
    max_residues : int
        The maximum number of residues accepted in a fragment.
    """
    
    def __init__(self, structure, Rep, cmap, k=3, score_thr=0.4, f_thr=0.1, max_inters=1, max_residues=40, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which to compute the fragments.
        Rep : structure.representation.Representation
            The class of structure representation.
        cmap : str
            The CMAP file.
        k : int, optional
            The number of residues to the left and to the right of the centroid. The default is 3.
        cosine_thr : float in [0,1], optional
            The similarity threshold between two neighborhoods, the higher the tighter. The default is 0.4.
        f_thr : float in [0,1], optional
            The interaction threshold between two residues. The default is 0.1.
        max_inters : int, optional
            The maximum number of interactions a neighborhood can have. The default is 1.
        max_residues : int, optional
            The maximum number of residues accepted in a fragment. The default is 40.

        Returns
        -------
        None.
        """
        super(KTerCloseMiner, self).__init__(structure)
        self.neighborhoods = Neighborhoods(structure, Rep, cmap, k, f_thr=f_thr, max_inters=max_inters)
        self.k = k
        self.score_thr = score_thr
        self.max_residues = max_residues
        
    def _all_similarity(self, fragment, candidate):
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
        score_sum = 0
        for n_id in fragment:
            features_n = self.neighborhoods[n_id].features
            features_c = candidate.features
            score = USR.get_similarity_score(features_n, features_c)
            if score < self.score_thr:
                return False, -1
            score_sum += score
        return True, score_sum
    
    def _best_neigh_match(self, frag_dict, offsets_dict, idx):
        """
        Return the best expansion candidate to either the left or the right of the given fragment.
        
        In case no viable candidate is found, a dummy entry if returned.

        Parameters
        ----------
        frag_dict : dict of int -> int
            The temporary fragment dictionary.
        offsets_dict : dict of int -> [int, int]
            The dictionary indicating for each fragment the offset to the left and to the right.
        idx : int
            The index of the fragment.

        Returns
        -------
        float
            The similarity score. In case of failure, it is set to 1e9.
        int
            The index of the neighborhood to be added to the fragment. In case of failure, it is set to -1.
        int
            The index of the fragment. Returning this is useful for the later sorting. In case of failure,
            it is set to -1.
        int
            Set to 0 if the neighborhood to the left has been added, 1 if it is the neighborhood to the
            right. In case of failure, it is set to -1.
        """
        n = len(self.neighborhoods)
        offsets = offsets_dict[idx]
        lower = idx - offsets[0]
        upper = idx + offsets[1]
        sl, su = False, False
        c_sum_l, c_sum_u = 0, 0
        if lower < 0 and upper < n:
            su, c_sum_u = self._all_similarity(frag_dict[idx], self.neighborhoods[upper])
        elif lower >= 0 and upper >= n:
            sl, c_sum_l = self._all_similarity(frag_dict[idx], self.neighborhoods[lower])
        elif lower >= 0 and upper < n:
            sl, c_sum_l = self._all_similarity(frag_dict[idx], self.neighborhoods[lower])
            su, c_sum_u = self._all_similarity(frag_dict[idx], self.neighborhoods[upper])
        if sl and su:
            if c_sum_l > c_sum_u:
                return c_sum_l, lower, idx, 0
            else:
                return c_sum_u, upper, idx, 1
        elif sl and not su:
            return c_sum_l, lower, idx, 0
        elif not sl and su:
            return c_sum_u, upper, idx, 1
        else:
            return -1, -1, -1, -1
    
    def mine(self):
        """
        Mine the fragments.

        Returns
        -------
        None.
        """
        # Compute the neighborhoods (including selecting the best interaction)
        self.neighborhoods.generate()
        # Define fragment dictionary which points to the indices of the neighborhood
        frag_dict = {} # int -> list of int
        full_frag_dict = {} # int -> list of int
        offsets_dict = {}
        n = len(self.neighborhoods)
        # Iterate over neighborhoods to get fragments
        for i in range(n):
            # Initialize dictionaries
            for j in range(n):
                frag_dict[j] = []
                frag_dict[j].append(j)
                offsets_dict[j] = [1,1]
            n_residues = len(self.neighborhoods[i])
            for inter in self.neighborhoods[i].interactions:
                n_residues += len(inter[0])
            while n_residues <= self.max_residues:
                candidates = []
                candidates.append(self._best_neigh_match(frag_dict, offsets_dict, i))
                for inter in self.neighborhoods[i].interactions:
                    candidates.append(self._best_neigh_match(frag_dict, offsets_dict, inter[0].idx))
                best = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
                if best[1] == -1:
                    break # no match found
                else:
                    frag_dict[best[2]].append(best[1])
                    offsets_dict[best[2]][best[3]] += 1
                    n_residues += 1
            # Compose full fragment
            full_frag_dict[i+1] = []
            for frag in frag_dict[i]:
                full_frag_dict[i+1].append(frag)
            for inter in self.neighborhoods[i].interactions:
                idx = inter[0].idx
                for frag in frag_dict[idx]:
                    if frag not in full_frag_dict[i+1]:
                        full_frag_dict[i+1].append(frag)
            # print(full_frag_dict[i+1])
        # Retrieve fragments
        self.fragments[self.structure.get_id()] = []
        for frag_id in full_frag_dict:
            n_ids = full_frag_dict[frag_id]
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            fragment = Fragment(self.structure, f_id)
            for n_id in n_ids:
                neighbor = self.neighborhoods[n_id]
                for residue in neighbor.residues:
                    fragment.add_residue(residue)
            self.fragments[self.structure.get_id()].append(fragment)
            
class HierarchyMiner(SingleMiner):
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
    
    def __init__(self, structure, builder, n_clusters=2, connectivity=False, linkage='ward', **params):
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
        super(HierarchyMiner, self).__init__(structure)
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
                
class LeidenMiner(SingleMiner):
    """
    Implement the Leiden community-finding algorithm for detecting fragments.
    
    Attributes
    ----------
    adjacency : list of (int, int)
        The adjacency list of the form (node, node). Note it is undirected and unweighted.
    weights : list of float
        The list of weights. It is built in such a way that its order matches the edges of the adjacency list.
    cmap : str
        The file holding the CMAP.
    partition : ??
        The partition of apply. See the Leiden documentation for more information.
    bb_strength : float
        The offset strength to add to the weights of the backbone.
    f_thr : float in [0,1]
        The threshold above which two residues are interacting. It is also used as the weight assigned to
        the edges.
    n_iters : int
        The number of iterations of the Leiden algorithm.
    max_size : int
        The maximum size of a community.
    _res_dict : dict of int -> Bio.PDB.Residue
        A dictionary mapping the segment ID of the residue to the residue itself.
    """
    
    def __init__(self, structure, cmap, partition=leidenalg.ModularityVertexPartition, bb_strength=1, f_thr=0.1, n_iters=5, max_size=40, **params):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Stucture
            The structure from which to generate the fragments.
        cmap : str
            The file holding the CMAP.
        partition : ??, optional
            The partition of apply. The default is leidenalg.ModularityVertexPartition.
        bb_strength : float, optional
            The offset strength to add to the weights of the backbone. The default is 1.
        f_thr : float in [0,1], optional
            The threshold above which two residues are interacting. It is also used as the weight
            assigned to the edges. The default is 0.1.
        n_iters : int, optional
            The number of iterations of the Leiden algorithm. The default is 5.
        max_size : int, optional
            The maximum size of a community. The default is 40.

        Returns
        -------
        None.
        """
        super(LeidenMiner, self).__init__(structure)
        self.adjacency = []
        self.weights = []
        self.cmap = cmap
        self.partition = partition
        self.bb_strength = bb_strength
        self.f_thr = f_thr
        self.n_iters = n_iters
        self.max_size = max_size
        self._res_dict = {}
        
    def _get_residues(self):
        """
        Fill the residue dictionary with the residues.
        
        Residues belonging to W-/HET-ATOM are discarded.

        Returns
        -------
        None.
        """
        for residue in self.structure.get_residues():
            r_id = residue.get_id()
            if r_id[0] == ' ' and r_id[1] >= 0:
                self._res_dict[r_id[1]] = residue
        
    def _compute_adjacency(self, entries):
        """
        Compute the adjacency matrix.
        
        Backbone and interactions are taken into consideration. The weights are also built here.

        Parameters
        ----------
        entries : list of (str, str, int, int, float)
            The list of entries of the CMAP.

        Returns
        -------
        None.
        """
        # Backbone connections
        keys = self._res_dict.keys()
        keys = sorted(keys, key=lambda x: x) # should already be sorted but just to be sure
        for i in range(len(keys)-1):
            self.adjacency.append((keys[i],keys[i+1]))
            res_1 = self._res_dict[keys[i]]
            res_2 = self._res_dict[keys[i+1]]
            ca_dist_inv = 0.2612
            if 'CA' in res_1 and 'CA' in res_2:
                ca_1 = res_1['CA']
                ca_2 = res_2['CA']
                ca_dist = np.linalg.norm(ca_1.get_vector()-ca_2.get_vector())
                ca_dist_inv = self.bb_strength + 1/ca_dist
            self.weights.append(ca_dist_inv)
        # Interaction connections
        for entry in entries:
            _, _, res_1, res_2, f = entry
            if res_1 in self._res_dict and res_2 in self._res_dict and f > self.f_thr:
                self.adjacency.append((res_1, res_2))
                ca_dist_inv = 0.0421
                if 'CA' in self._res_dict[res_1] and 'CA' in self._res_dict[res_2]:
                    ca_1 = self._res_dict[res_1]['CA']
                    ca_2 = self._res_dict[res_2]['CA']
                    ca_dist = np.linalg.norm(ca_1.get_vector()-ca_2.get_vector())
                    ca_dist_inv = 1/ca_dist
                self.weights.append(ca_dist_inv)
                
    def mine(self):
        """
        Mine the fragments.

        Returns
        -------
        None.
        """
        # Get the residues
        self._get_residues()
        # Get the entries
        entries = parse_cmap(self.cmap)
        if entries is None:
            return
        # Compute the adjacency
        self._compute_adjacency(entries)
        # Define IGraph
        G = ig.Graph(self.adjacency)
        # Compute the partitions using the Leiden algorithm
        partitions = leidenalg.find_partition(G, self.partition, weights=self.weights, n_iterations=self.n_iters, max_comm_size=self.max_size)
        # Retrieve fragments
        self.fragments[self.structure.get_id()] = []
        frag_id = 1
        for partition in partitions:
            # The structure of the structure is (<pdb_id>_<chain_id>_<chain_id><frag_id>)
            f_id = self.structure.get_id() + '_' + self.structure.get_id()[-1] + str(frag_id)
            fragment = Fragment(self.structure, f_id)
            for node in partition:
                if node in self._res_dict:
                    fragment.add_residue(self._res_dict[node])
            self.fragments[self.structure.get_id()].append(fragment)
            frag_id += 1
            
class FuzzleMiner(Miner):
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
        super(FuzzleMiner, self).__init__()
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
        progress_bar = ProgressBar(len(self.data['nodes']))
        if self.verbose:
            print('Generating fragments...')
            progress_bar.start()
        for node in self.data['nodes']:
            if self.verbose:
                progress_bar.step()
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
    