import itertools

import numpy as np

from structure.representation import MITResidue
from structure.Neighborhood import Neighborhood
from utils.search import binary_search
from utils.io import parse_cmap
    
class Builder:
    """An abstract class which builds the representation of a given structure."""
    
    def __init__(self):
        """
        Initialize the class.

        Returns
        -------
        None.
        """
        pass
    
    def get_residues_at(self, idx):
        """
        Return the residues at the specified index. This method is meant to be overridden by subclasses.

        Parameters
        ----------
        idx : int
            The index.

        Returns
        -------
        None.
        """
        pass
    
    def get_adjacency(self):
        """
        Return the adjacency matrix, if any. This method is meant to be overridden by subclasses.
        
        The adjacency matrix is computed according to the structure representation.

        Returns
        -------
        None.
        """
        pass
    
    def get_features(self):
        """
        Return the features. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        """
        pass
    
    def generate(self):
        """
        Generate the representation. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        """
        pass
    
class Neighborhoods(Builder):
    """
    Model the representation of a structure as a set of overlapping neighborhoods.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure to represent.
    Rep : structure.representation.Representation
        The class of structure representation.
    elements : dict of int -> structure.Neighborhood.Neighborhood
        The dictionary mapping an ID to its neighborhood.
    cmap_file : str
        The CMAP file containing interaction information.
    k : int
        The number of residues to the left and to the right of the centroid. The maximal number of residues
        per neighborhood is 2k+1.
    f_thr : float in [0,1]
        The interaction threshold between two residues. A high interaction score means two residues have
        a very high interaction.
    max_inters : int
        The maximum number of interacting neighborhoods a neighborhood can have.
    _intr_cache : dict of int -> float
        A cache storing whether a certain residue-residue interaction has already been tested. The mapped
        value represents the contact score, which is set to -1 in case there is no residue-residue interaction.
    """
    
    def __init__(self, structure, Rep, cmap, k=3, f_thr=0.1, max_inters=1):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure to represent.
        Rep : structure.representation.Representation
            The class to be used to represent the structure..
        cmap_file : str
            The contact map file.
        k : int, optional
            The number of residues to the left and to the right of the centroid. The default is 3.
        f_thr : float in [0,1]
            The interaction threshold between two residues. A high interaction score means two residues have
            a very high interaction.
        max_inters : int, optional
            The maximum number of interacting neighborhoods a neighborhood can have. The default is 1.

        Returns
        -------
        None.
        """
        super(Neighborhoods, self).__init__()
        self.structure = structure
        self.Rep = Rep
        self.elements = {}
        self.cmap = cmap
        self.k = k
        self.f_thr = f_thr
        self.max_inters = max_inters
        self._intr_cache = {}
        
    def __len__(self):
        """
        Return the number of neighborhoods.

        Returns
        -------
        int
            The number of neighborhoods.
        """
        return len(self.elements)
    
    def __getitem__(self, key):
        """
        Return the neighborhood mapped by the specified key.

        Parameters
        ----------
        key : int
            The key.

        Returns
        -------
        structure.Neighborhood.Neighborhood
            The neighborhood mapped by the specified key.
        """
        return self.elements[key]
        
    def __iter__(self):
        """
        Return an iterable object over the neighborhoods.

        Returns
        -------
        dict_keyiterator
            The iterator object.
        """
        return iter(self.elements)
    
    def _detect_interactions(self, entries, neigh_1, neigh_2):
        """
        Detect interaction between residues belonging to the specified neighborhoods.

        Parameters
        ----------
        entries : list of (int, float)
            The list containing the processed entries of the CMAP.
        neigh_1 : structure.Neighborhood.Neighborhood
            The first neighborhood.
        neigh_2 : structure.Neighborhood.Neighborhood
            The second neighborhood.

        Returns
        -------
        interact : bool
            Whether there is interaction between the two neighborhoods.
        f_sum : float
            The sum of the interaction scores. A higher sum indicates stronger interaction.
        """
        # Define residues to search and entries to be searched
        interact, f_sum = False, 0
        r_pairs = list(itertools.product(neigh_1.residues, neigh_2.residues))
        searcheable_list = [x[0] for x in entries]
        seen = {} # as to not count the same thing twice
        # Iterate over neighborhoods residues
        for res_1, res_2 in r_pairs:
            r_id_1, r_id_2 = res_1.get_id(), res_2.get_id()
            # Continue if it is the same residue
            if r_id_1[1] == r_id_2[1]:
                continue
            r_target_1 = int(str(r_id_1[1])+'0'+str(r_id_2[1]))
            r_target_2 = int(str(r_id_2[1])+'0'+str(r_id_1[1]))
            # Check if their contribution has already been counted
            if r_target_1 in seen or r_target_2 in seen:
                continue
            # Check if first target is in the cache
            if r_target_1 in self._intr_cache:
                if self._intr_cache[r_target_1] > self.f_thr:
                    f_sum += self._intr_cache[r_target_1]
                seen[r_target_1] = seen[r_target_2] =  True
                continue
            # Check if second target is in the cache
            elif r_target_2 in self._intr_cache:
                if self._intr_cache[r_target_2] > self.f_thr:
                    f_sum += self._intr_cache[r_target_2]
                seen[r_target_2] = seen[r_target_1] = True
                continue 
            # If targets are not in the cache, perform binary search and insert results in the cache
            found_1, idx_1 = binary_search(r_target_1, searcheable_list)
            found_2, idx_2 = binary_search(r_target_2, searcheable_list)
            seen[r_target_1] = seen[r_target_2] = True # target been seen this iteration
            if found_1:
                f = entries[idx_1][1]
                if f > self.f_thr:
                    interact = True
                    f_sum += f
                self._intr_cache[r_target_1] = self._intr_cache[r_target_2] = f
            elif found_2:
                f = entries[idx_2][1]
                if f > self.f_thr:
                    interact = True
                    f_sum += f
                self._intr_cache[r_target_2] = self._intr_cache[r_target_1] = f
            else:
                self._intr_cache[r_target_1] = self._intr_cache[r_target_2] = -1
        return interact, f_sum
    
    def _filter_interactions(self):
        """
        Filter the interactions of every neighborhood, retaining only the better ones.

        Returns
        -------
        None.
        """
        for i in range(len(self.elements)):
            self.elements[i].filter_interactions()
            
    def get_residues_at(self, idx):
        """
        Return the residue at the specified neighborhood, including the ones from the latter interactions.

        Parameters
        ----------
        idx : int
            The index.

        Returns
        -------
        residues : list of Bio.PDB.Residue
            The list of residues belonging to the neighborhood.
        """
        residues = []
        neighbor = self.elements[idx]
        for residue in neighbor.residues:
            residues.append(residue)
            for inter in neighbor.interactions:
                for residue in inter[0].residues:
                    residues.append(residue)
        return residues

    def get_adjacency(self):
        """
        Return the adjacency matrix based on the contact map information.

        Returns
        -------
        A : numpy.ndarray
            The adjacency matrix.
        """
        n = len(self.elements)
        A = np.zeros(shape=(n,n))
        # Backbone adjacency
        A[0,1] = A[n-1,n-2] = 1 
        for i in range(1,n-1):
            A[i-1,i] = A[i,i+1] = 1
        # Contact map adjancency
        for i in range(n-1):
            current = self.elements[i]
            for inter in current.interactions:
                A[i,inter.idx] = A[inter.idx,i] = 1
        return A
    
    def get_features(self):
        """
        Return the feature matrix.

        Returns
        -------
        X : numpy.ndarray
            The features.
        """
        # Get shape of features
        n = len(self.elements)
        X = np.zeros(shape=(n,self.Rep.get_n_features()))
        # Add the features
        for i in range(n):
            if len(self.elements[0].features.shape) == 2:
                X[i,:] = np.mean(self.elements[i].features, axis=0)
            else:
                X[i,:] = self.elements[i].features
        return X
        
    def generate(self):
        """
        Generate the neighborhoods.
        
        If the CMAP file is not None, it will check for interacting neighborhoods.

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
            self.elements[i] = Neighborhood(i, residues[min_idx:max_idx], max_inters=self.max_inters) # residue i is also added here
        # For each neighborhood compute its USR momenta
        for i in range(len(self.elements)):
            self.elements[i].compute_features(self.Rep)
        # Check if the CMAP is None. If so, no need to proceed
        if self.cmap is not None:
            # Get entries
            entries = parse_cmap(self.cmap)
            if entries is None:
                return
            # Prepare the entries for binary search
            entries = [(int(str(x[2])+'0'+str(x[3])), x[4]) for x in entries]
            entries = sorted(entries, key=lambda x: x[0])
            # For each neighborhood, check if it has interaction with any residue of any other neighborhood
            for i in range(len(self.elements)-1):
                for j in range(i+1, len(self.elements)):
                    neigh_1, neigh_2 = self.elements[i], self.elements[j]
                    interact, f_sum = self._detect_interactions(entries, neigh_1, neigh_2)
                    if interact:
                        self.elements[i].add_interaction(self.elements[j], f_sum)
                        self.elements[j].add_interaction(self.elements[i], f_sum)
            self._filter_interactions()
        
            
class MITStructure(Builder):
    """
    A dummy class with no real purpose in life other than being a convenient wrap-class.
    
    Because this class has very low self-esteem, we are going to keep lying to it so that it will go on
    believing to have a purpose in its vain life and thus avoiding screwing up the whole code base.
    
    Attributes
    ----------
    structure : Bio.PDB.Structure
        The structure to represent.
    """
    
    def __init__(self, structure):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure.

        Returns
        -------
        None.
        """
        super(MITStructure, self).__init__()
        self.structure = structure
        
    def get_residues_at(self, idx):
        """
        Return the residue at the given index.

        Parameters
        ----------
        idx : int
            The index.

        Returns
        -------
        list of Bio.PDB.Residue
            The residue at the specified index. It is wrapped in a list so it can be iterated over.
        """
        return [self.mitr.residues[idx]]
    
    def get_adjacency(self):
        """
        Return the adjacency matrix computed according to distance between carbon alpha of the residues.

        Returns
        -------
        A : numpy.ndarray
            The adjacency matrix.
        """
        n = len(self.mitr.residues)
        A = np.zeros(shape=(n,n))
        for i in range(1, n-1):
            A[i-1,i] = A[i,i+1] = 1
        A[0,1] = 1
        A[n-1,n-2] = 1
        for i in range(n-1):
            for j in range(i+1, n):
                if self.mitr.contact_map[i,j] < 8:
                    A[i,j] = A[j,i] = 1
        return A
        
    def get_features(self):
        """
        Return the computed features.

        Returns
        -------
        numpy.ndarray
            The features.
        """
        return self.mitr.embeddings
    
    def generate(self):
        """
        Generate the MIT representation for each residue in the structure.

        Returns
        -------
        None.
        """
        self.mitr = MITResidue(self.structure)
        self.mitr.compute_representation()
        self.mitr.compute_contact_map()
    
    