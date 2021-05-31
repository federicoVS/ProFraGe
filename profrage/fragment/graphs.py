# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:37:42 2021

@author: Federico van Swaaij
"""

import numpy as np

import igraph as ig

import graphkernels.kernels as gk

class UUGraph:
    """
    A generic undirected, unweighted graph class.
    
    Source
    ------
    https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
    https://www.geeksforgeeks.org/iterative-depth-first-traversal/
    
    Attributes
    ----------
    connected_components : list of list of int
        The list where each entry correspond to a connected component, and where each of such entries
        holds the indexes corresponding to the nodes in said component.
    N : int
        The number of vetices.
    adj : list of list of int
        The adjacency matrix represented as a list of lists.
    """
    
    def __init__(self, N):
        """
        Initialize the graph by creating an empty adjacency list.

        Parameters
        ----------
        N : int
            The number of vertices.

        Returns
        -------
        None.
        """
        self.connected_components = []
        self.N = N
        self.adj = [[] for i in range(N)]
    
    def add_edge(self, v, w):
        """
        Add an edge between the specified vertices.
        
        Note that the addition goes only one way, i.e. v->w, and not the other way around.

        Parameters
        ----------
        v : int
            The first vertex to add.
        w : int
            The second vertex to add.

        Returns
        -------
        None.
        """
        self.adj[v].append(w)
        
    def compute_connected_components(self):
        """
        Compute the connected components.

        Returns
        -------
        None.
        """
        # Initialize visited
        visited = []
        for i in range(self.N):
            visited.append(False)
        # Iterate over nodes, find for each its components
        for v in range(self.N):
            if not visited[v]:
                v_component, visited = self.dfs(v, visited)
                self.connected_components.append(v_component)
        
    def dfs(self, v, visited):
        """
        Perform DFS (iteratively) starting from the specified node.
        
        The procedure is only called for nodes not already visited.

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
        """
        # Create component for v
        v_component = []
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
    
class GraphKernel:
    """
    A wrapper class for the GraphKernels library.
    
    Source
    ------
    https://github.com/eghisu/GraphKernels
    
    Attributes
    ----------
    structures : list of Bio.PDB.Structure
        The list of structures of which to compute the kernel.
    ca_dist_thr : float
        The minimum distance in Angstrom for two residues to be interacting.
    h : int
        The number of iterations to perform.
    """
    
    def __init__(self, structures, ca_dist_thr=5, h=10):
        """
        Initialize the class.

        Parameters
        ----------
        structures : list of Bio.PDB.Structure
        The list of structures of which to compute the kernel.
        ca_dist_thr : float, optional
            The minimum distance in Angstrom for two residues to be interactin. The default is 5.
        h : int, optional
            The number of iterations. The default is 10.

        Returns
        -------
        None.
        """
        self.structures = structures
        self.ca_dist_thr = ca_dist_thr
        self.h = h
    
    def _get_adjacency(self, idx):
        # Define residue and adjacency lists
        adjacency, residues = [], []
        # Get the residues
        for residue in self.structures[idx].get_residues():
            r_id = residue.get_id()
            if r_id[0] == ' ' and r_id[1] >= 0:
                residues.append(residue)
        # Get connections
        for i in range(len(residues)):
            for j in range(len(residues)):
                if i != j:
                    res_i, res_j = residues[i], residues[j]
                    if 'CA' in res_i and 'CA' in res_j:
                        ca_i = res_i['CA']
                        ca_j = res_j['CA']
                        ca_dist = np.linalg.norm(ca_i.get_vector()-ca_j.get_vector())
                        if ca_dist <= self.ca_dist_thr:
                            adjacency.append((i,j))
        return adjacency
    
    def get_kernel(self):
        """
        Return the kernel describing the structures.

        Returns
        -------
        kernel : numpy.ndarray
            The kernel for the graphs.
        """
        # Define n for convenience
        n = len(self.structures)
        # Define the graphs list
        graphs = []
        # Build the graphs
        for i in range(n):
            # Get the adjacency
            adjacency = self._get_adjacency(i)
            # Define the graph
            G = ig.Graph(adjacency)
            # Add the graph to the list of graphs
            graphs.append(G)
        # Compute kernel
        kernel = gk.CalculateWLKernel(graphs, par=self.h)
        # Return the kernel
        return kernel