# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:30:55 2021

@author: FVS
"""

import os
import numpy as np
from Bio.PDB import NeighborSearch, Selection
from Bio.PDB.mmtf import MMTFParser
from utils.misc import structure_length

class Fragment:
    '''
    The representation of a fragment. The generation can be whatever method.
    This class is mainly concerned with finding whether the generated fragment is composed
    of a single segment, or by more than one.
    
    Attributes
    ----------
    mmtf : str
        The file holding the fragment in MMTF format.
    fragment : Bio.PDB.Structure
        The fragment generated.
    '''
    
    def __init__(self, mmtf):
        '''
        Initialized the class. The input MMTF is parsed into a Structure instance.

        Parameters
        ----------
        mmtf : str
            The MMTF file containing the generated fragment.

        Returns
        -------
        None.
        '''
        self.mmtf = mmtf
        parser = MMTFParser()
        self.fragment = parser.get_structure(mmtf)
        
    def get_fragment(self):
        '''
        Returns the Structure instance of the fragment.

        Returns
        -------
        Bio.PDB.Structure
            The Structure instance of the fragment.
        '''
        return self.fragment
    
    def get_name(self):
        '''
        Returns the name of the fragment.

        Returns
        -------
        str
            The name of the fragment.
        '''
        return os.path.basename(self.mmtf)[:-5]
    
    def get_mmtf(self):
        '''
        Returns the MMTF file holding the fragment.

        Returns
        -------
        str
            The MMTF file holding the fragment.
        '''
        return self.mmtf
        
    def residue_center(self, residue):
        '''
        Computes the center of the given residue.

        Parameters
        ----------
        residues : list of Bio.PDB.Residue
            The list of residues.

        Returns
        -------
        numpy.ndarray
            The center of the array
        '''
        coords = []
        for atom in residue:
            coords.append(atom.get_coord())
        coords = np.array(coords)
        return np.mean(coords, axis=0)
    
    def is_complex(self, grade=12):
        '''
        Checks whether the fragment is complex, according to its complexity grade, that is,
        how many residues compose it.

        Parameters
        ----------
        grade : int, optional
            The minimal number of residues for the fragment to be considered complex. The default is 12.

        Returns
        -------
        bool
            Whether the fragment is complex.
            True if |residues| >= grade, False otherwise.
        '''
        return structure_length(self.fragment) >= grade

    def is_connected(self, radius=5):
        '''
        Checks whether the fragment is connected. The search is conducted at the residual
        level.

        Parameters
        ----------
        radius : float, optional
            Search radius in Angstroms. The default is 5.

        Returns
        -------
        bool
            Whether the fragment is connected.
            True if the length connected components if 1, False otherwise.
        '''
        # Create dictionary which encodes residues as integers
        index = 0
        vertex_dict = {}
        for residue in self.fragment.get_residues():
            vertex_dict[residue] = index
            index += 1
        # Create graph
        graph = _ComponentsGraph(index)
        # Iterate over the residues
        for target_residue in self.fragment.get_residues():
            center_coord = self.residue_center(target_residue)
            atoms = Selection.unfold_entities(self.fragment, 'A')
            ns = NeighborSearch(atoms)
            close_residues = ns.search(center_coord, radius, level='R')
            # Remove the target protein itself
            if target_residue in close_residues:
                close_residues.remove(target_residue)
            for cr in close_residues:
                graph.add_edge(vertex_dict[target_residue], vertex_dict[cr])
        # Compute the connected components
        connected_components = graph.connected_components()
        return len(connected_components) == 1
            
class _ComponentsGraph:
    '''
    A private class which represents a fragment as a graph. The goal of this class is
    to compute the connected components of the fragment to determine whether it is composed
    of multiple fragments.
    
    Source
    ------
    https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
    https://www.geeksforgeeks.org/iterative-depth-first-traversal/
    
    Attributes
    ----------
    N : int
        The number of vetices.
    adj : list of list of int
        The adjacency matrix. Because a a fragment has a median of 12 residues and a
        maximum of 41, it is feasible to use a list of lists.
    '''
    
    def __init__(self, N):
        '''
        Initializes the graph by creating an empty adjacency list.

        Parameters
        ----------
        N : int
            The number of vertices.

        Returns
        -------
        None.
        '''
        self.N = N
        self.adj = [[] for i in range(N)]
    
    def add_edge(self, v, w):
        '''
        Adds an edge between the specified vertices. Note that the addition goes only one
        way, i.e. v->w, and not the other way around.

        Parameters
        ----------
        v : int
            The first vertex to add.
        w : int
            The second vertex to add.

        Returns
        -------
        None.
        '''
        self.adj[v].append(w)
        
    def connected_components(self):
        '''
        Performs the connected component algorithm.

        Returns
        -------
        connected_components : list of list of int
            The connected components for the graph.
        '''
        # Initialize visited
        visited = []
        for i in range(self.N):
            visited.append(False)
        connected_components = []
        # Iterate over nodes, find for each its components
        for v in range(self.N):
            if not visited[v]:
                v_component, visited = self._dfs(v, visited)
                connected_components.append(v_component)
        return connected_components
        
    def _dfs(self, v, visited):
        '''
        Performs DFS (iteratively) starting from the specified node. The procedure is only
        called for nodes not already visited.

        Parameters
        ----------
        v : int
            The vertex from which to start DFS.
        visited : list of bool
            The list holding whether node i has been visited, i.e. visited[i] == True.

        Returns
        -------
        v_component, visited : (list of int, list of bool)
            The connected component for node v, the updated list of visited nodes.
        '''
        # Create component for v
        v_component = []
        # Update visited list
        visited[v] = True
        # DFS stack
        stack = []
        # Push v
        stack.append(v)
        while len(stack) != 0:
            # Pop from the stack
            s = stack.pop()
            # Check is s has been visited
            if not visited[s]:
                visited[s] = True
                v_component.append(s)
            # Iterate through s neighbors
            for t in self.adj[s]:
                if not visited[t]:
                    stack.append(t)
        return v_component, visited
    
    