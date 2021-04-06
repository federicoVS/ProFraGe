# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:37:42 2021

@author: FVS
"""

class ComponentsGraph:
    '''
    A class which represents a fragment as a graph. The goal of this class is
    to compute the connected components of the fragment to determine whether it is composed
    of multiple fragments.
    
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
        self.connected_components = []
        self.N = N
        self.adj = [[] for i in range(N)]
        
    def get_components(self):
        '''
        Returns the connected components

        Returns
        -------
        list of list of int
            The connected components.
        '''
        return self.connected_components
    
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
        
    def compute_connected_components(self):
        '''
        Computes the connected components.

        Returns
        -------
        None.
        '''
        # Initialize visited
        visited = []
        for i in range(self.N):
            visited.append(False)
        # Iterate over nodes, find for each its components
        for v in range(self.N):
            if not visited[v]:
                v_component, visited = self._dfs(v, visited)
                self.connected_components.append(v_component)
        
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