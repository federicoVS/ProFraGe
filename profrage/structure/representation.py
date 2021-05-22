# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:55:58 2021

@author: Federico van Swaaij
"""

import numpy as np
from scipy.stats import skew
from Bio.PDB.vectors import calc_dihedral

from utils.structure import get_atoms_coords, get_ca_atoms_coords

class Representation:
    """An abstract class to implement structure representation."""
    
    def __init__(self):
        """
        Initialize the class.

        Returns
        -------
        None.
        """
        pass

    @staticmethod
    def get_n_features():
        """
        Return the number of a single feature-vector. This method is meant to be overridden by subclasses.

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

class USR(Representation):
    """
    Implements Ultrafast Shape Recognition (USR).
    
    Source
    ------
    Ultrafast shape recognition for similarity search in molecular databases
    (Pedro J. Ballester, W. Graham Richards)
    
    Attributes
    ----------
    ctd : numpy.ndarray
        The coordinates of the centroid of the structure. Note that it is the average of the atoms, so
        very likely it does not correspond to a real atom.
    cst : numpy.ndarray
        The coordinates of the closest atom to the centroid.
    fct : numpy.ndarray
        The coordinates of the farthest atom to the centroid.
    ftf : numpy.ndarray
        The coordinates of the farthest atom to `fct`.
    momenta : numpy.ndarray
        The array containing the momenta for the previous four features. The array is organized in four
        blocks of (mean, varriance, skewness)_f, where each element in the tuple is computed relative to
        f, with f being `ctd`, `cst`, `fct`, and `ftf` respecitvely.
    _coords : numpy.ndarray
        The matrix of coordinates
    """
    
    def __init__(self, structure, ca_atoms=False):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure to represent.
        ca_atoms : str, optional
            Whether to use only C-alpha atoms to compute the USR. The default is False.

        Returns
        -------
        None.
        """
        super(USR, self).__init__()
        if ca_atoms:
            self._coords = get_ca_atoms_coords(structure)
        else:
            self._coords = get_atoms_coords(structure)
        self.ctd = None
        self.cst = None
        self.fct = None
        self.ftf = None
        self.momenta = None
        
    @staticmethod
    def get_similarity_score(momenta_1, momenta_2, size=12):
        """
        Compute the similarity score between two momenta, as described in the source paper.

        Parameters
        ----------
        momenta_1 : numpy.ndarray
            The USR momenta of the first structure.
        momenta_2 : TYPE
            The USR momenta of the second structure.
        size : int, optional
            The size of the momenta vector. The default is 12

        Returns
        -------
        float in (0,1)
            The similarity score.
        """
        score = 0
        for i in range(size):
            score += abs(momenta_1[i]-momenta_2[i])
        score /= size
        score += 1
        return 1/score
        
    @staticmethod
    def get_n_features():
        """
        Return the length of USR vector.

        Returns
        -------
        int
            The length of USR vector.
        """
        return 12
        
    def _get_ctd(self):
        """
        Compute the coordinates of `ctd`.

        Returns
        -------
        None.
        """
        self.ctd = np.mean(self._coords, axis=0)
        
    def _get_cst(self):
        """
        Compute the coordinates of `cst`.

        Returns
        -------
        None.
        """
        squared_dist = np.sqrt(np.sum((self._coords-self.ctd)**2, axis=1))
        self.cst = self._coords[np.argmin(squared_dist),:]
    
    def _get_fct(self):
        """
        Compute the coordinates of `fct`.

        Returns
        -------
        None.
        """
        squared_dist = np.sqrt(np.sum((self._coords-self.ctd)**2, axis=1))
        self.fct = self._coords[np.argmax(squared_dist),:]
    
    def _get_ftf(self):
        """
        Compute the coordinates of `ftf`.

        Returns
        -------
        None.
        """
        squared_dist = np.sqrt(np.sum((self._coords-self.fct)**2, axis=1))
        self.ftf = self._coords[np.argmax(squared_dist),:]
        
    def _compute_momenta(self):
        """
        Compute the momenta.

        Returns
        -------
        None.
        """
        # Initialize momenta
        self.momenta = np.zeros(shape=(12,))
        # Compute distances
        dist_ctd = np.sqrt(np.sum((self._coords-self.ctd)**2, axis=1))
        dist_cst = np.sqrt(np.sum((self._coords-self.cst)**2, axis=1))
        dist_fct = np.sqrt(np.sum((self._coords-self.fct)**2, axis=1))
        dist_ftf = np.sqrt(np.sum((self._coords-self.ftf)**2, axis=1))
        # Mean
        self.momenta[0] = np.mean(dist_ctd) # ctd
        self.momenta[3] = np.mean(dist_cst) # cst
        self.momenta[6] = np.mean(dist_fct) # fct
        self.momenta[9] = np.mean(dist_ftf) # ftf
        # Variance
        self.momenta[1] = np.var(dist_ctd) # ctd
        self.momenta[4] = np.var(dist_cst) # cst
        self.momenta[7] = np.var(dist_fct) # fct
        self.momenta[10] = np.var(dist_ftf) # ftf
        # Skewness
        self.momenta[2] = skew(dist_ctd) # ctd
        self.momenta[5] = skew(dist_cst) # cst
        self.momenta[8] = skew(dist_fct) # fct
        self.momenta[11] = skew(dist_ftf) # ftf
        
    def _compute_all(self):
        """
        Compute the features and their momenta.

        Returns
        -------
        None.
        """
        self._get_ctd()
        self._get_cst()
        self._get_fct()
        self._get_ftf()
        self._compute_momenta()
        
    def get_features(self):
        """
        Return the USR-momenta.

        Returns
        -------
        numpy.ndarray
            The USR-momenta.
        """
        if self.momenta is None:
            self._compute_all()
        return self.momenta
    
class USRpp(Representation):
    """
    Representation of a structure using a combination of shape information and secondary structure information.
    
    The former is obtained with USR, while the latter is obtained with Stride.
    
    Attributes
    ----------
    features : numpy.ndarray
        The full feature-vector. The first 12 places hold the USR-momenta, while the last 7 hold secondary
        information.
    stride_dict : dict of str -> int
        The dictonary build from Stride holding secondary structure information.
    """
    
    def __init__(self, structure, stride_dict, ca_atoms=False):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure to represent.
        stride_dict : dict of str -> int
            The Stride dictionary.
        ca_atoms : bool, optional
            Whether to use only C-alpha atoms to compute the USR. The default is False.

        Returns
        -------
        None.
        """
        super(USRpp, self).__init__()
        self.features = np.zeros(shape=(USR.get_n_features()+7,))
        self.features[0:12] = USR(structure, ca_atoms=ca_atoms).get_features()
        self.stride_dict = stride_dict
        
    @staticmethod
    def get_n_features():
        """
        Return the number of a single feature-vector.

        Returns
        -------
        int
            The length of the feature-vector.
        """
        return USR.get_n_features() + 7
    
    @staticmethod
    def get_similarity_score(features_1, features_2):
        """
        Compute the similarity score between two feature vectors.
        
        The similarity is computed using the USR-defined similarity function on the entire feature-vectors,
        in order to enforce a unique comparison strategy.

        Parameters
        ----------
        features_1 : numpy.ndarray
            The first feature-vector.
        features_2 : numpy.ndarray
            The second feature-vector.

        Returns
        -------
        float in (0,1)
            The similarity score.
        """
        return USR.get_similarity_score(features_1, features_2, size=USR.get_n_features()+7)
    
    def _compute_secondary(self):
        total = 0
        for key in self.stride_dict:
            total += self.stride_dict[key]
        if 'H' in self.stride_dict:
            self.features[12] = self.stride_dict['H']/total
        else:
            self.features[12] = 0
        if 'G' in self.stride_dict:
            self.features[13] = self.stride_dict['G']/total
        else:
            self.features[13] = 0
        if 'I' in self.stride_dict:
            self.features[14] = self.stride_dict['I']/total
        else:
            self.features[14] = 0
        if 'E' in self.stride_dict:
            self.features[15] = self.stride_dict['E']/total
        else:
            self.features[15] = 0
        if 'B' in self.stride_dict:
            self.features[16] = self.stride_dict['B']/total
        else:
            self.features[16] = 0
        if 'b' in self.stride_dict:
            self.features[16] = self.stride_dict['b']/total
        else:
            self.features[16] = 0
        if 'T' in self.stride_dict:
            self.features[17] = self.stride_dict['T']/total
        else:
            self.features[17] = 0
        if 'C' in self.stride_dict:
            self.features[18] = self.stride_dict['C']/total
        else:
            self.features[18] = 0
    
    def get_features(self):
        """
        Return the features.

        Returns
        -------
        numpy.ndarray
            The features.
        """
        self._compute_secondary()
        return self.features
        
        
class MITResidue(Representation):
    """
    Representation of individual residues belonging to a structure.
    
    Source
    ------
    Generative models for graph-based protein design
    (John Ingraham, Vikas K. Garg, Regina Barzilay, Tommi Jaakkola)
    
    Attributes
    ----------
    embeddings : numpy.ndarray
        The embeddings for each residue. Each embedding is a feature-vector with 8 entries
        (Psi, Phi, Omega, Ca_i-Ca_(i-1), Ca_(i+1)-Ca_i, ID_(i-1), ID_(i+1)). If any entry cannot be
        computed, it is set to 0.
    contact_map : numpy.ndarray
        A contact map representing distances between the alpha-Carbons of two residues.
    residues : list of Bio.PDB.Residue
        The list of residues in the structure, except water- and het-atoms.
    """
    
    def __init__(self, structure):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure.
            The structure to represent.

        Returns
        -------
        None.
        """
        super(MITResidue, self).__init__()
        self.residues = []
        for residue in structure.get_residues():
            r_id = residue.get_id()
            if r_id[0] == ' ':
                self.residues.append(residue)
        self.residues = sorted(self.residues, key=lambda x: x.get_id()[1])
        self.embeddings = None
        self.contact_map = np.zeros(shape=(len(self.residues),len(self.residues)))
        
    @staticmethod
    def get_n_features():
        """
        Return the length of a single residue feature-vector.

        Returns
        -------
        int
            The length of a single feature-vector.
        """
        return 7
    
    def compute_contact_map(self):
        """
        Compute the contact map.

        Returns
        -------
        None.
        """
        n = len(self.residues)
        for i in range(n-1):
            ca_i = self.residues[i]['CA'].get_coord()
            for j in range(i+1, n):
                ca_j = self.residues[j]['CA'].get_coord()
                self.contact_map[i,j] = self.contact_map[j,i] = np.linalg.norm(ca_i-ca_j)
        
    def compute_representation(self):
        """
        Compute the representation of each residue belonging to the structure.

        Returns
        -------
        None.
        """
        # Initilize embeddings
        self.embeddings = np.zeros(shape=(len(self.residues),7))
        n = len(self.residues)
        # Residues [1,n-1]
        for i in range(1, n-1):
            # Representing i with i, i-1 with h, i+1 with j
            r_i, r_h, r_j = self.residues[i], self.residues[i-1], self.residues[i+1]
            # Atoms for residue i
            c_i = r_i['C'].get_vector()
            ca_i = r_i['CA'].get_vector()
            n_i = r_i['N'].get_vector()
            # Atoms for residue i-1
            c_h = r_h['C'].get_vector()
            ca_h = r_h['CA'].get_vector()
            # Atoms for residue i+1
            ca_j = r_j['CA'].get_vector()
            n_j = r_j['N'].get_vector()
            # Compute Phi
            phi = calc_dihedral(c_h, n_i, ca_i, c_i)
            # Compute Psi
            psi = calc_dihedral(n_i, ca_i, c_i, n_j)
            # Compute Omega
            omega = calc_dihedral(ca_i, c_i, n_j, ca_j)
            # Compute Ca distances
            dist_hi = np.linalg.norm(ca_i-ca_h)
            dist_ij = np.linalg.norm(ca_j-ca_i)
            # Compute neighbors codes
            neigh_h = int(''.join([str(ord(k)) for k in r_h.get_resname()]))
            neigh_j = int(''.join([str(ord(k)) for k in r_j.get_resname()]))
            # Set embeddings
            self.embeddings[i,:] = np.array([phi, psi, omega, dist_hi, dist_ij, neigh_h, neigh_j])
        # Residue 0
        if 'C' not in self.residues[0] or 'C' not in self.residues[0] or 'N' not in self.residues[0]:
            self.embeddings[0,:] = np.array([0, 0, 0, 0, 0, 0, 0])
        else:
            c_0 = self.residues[0]['C'].get_vector()
            ca_0 = self.residues[0]['CA'].get_vector()
            n_0 = self.residues[0]['N'].get_vector()
            ca_1 = self.residues[1]['CA'].get_vector()
            n_1 = self.residues[1]['N'].get_vector()
            psi = calc_dihedral(n_0, ca_0, c_0, n_1)
            omega = calc_dihedral(ca_0, c_0, n_1, ca_1)
            dist_01 = np.linalg.norm(ca_1-ca_0)
            neigh_1 = int(''.join([str(ord(k)) for k in self.residues[1].get_resname()]))
            self.embeddings[0,:] = np.array([0, psi, omega, 0, dist_01, 0, neigh_1])
        # Residue n
        if 'C' not in self.residues[n-1] or 'C' not in self.residues[n-1] or 'N' not in self.residues[n-1]:
            self.embeddings[n-1,:] = np.array([0, 0, 0, 0, 0, 0, 0])
        else:
            c_n = self.residues[n-1]['C'].get_vector()
            ca_n = self.residues[n-1]['CA'].get_vector()
            n_n = self.residues[n-1]['N'].get_vector()
            c_nm1 = self.residues[n-2]['C'].get_vector()
            ca_nm1 = self.residues[n-2]['CA'].get_vector()
            phi = calc_dihedral(c_nm1, n_n, ca_n, c_n)
            dist_nm1n = np.linalg.norm(ca_n-ca_nm1)
            neigh_nm1 = int(''.join([str(ord(k)) for k in self.residues[n-2].get_resname()]))
            self.embeddings[n-1,:] = np.array([phi, 0, 0, dist_nm1n, 0, neigh_nm1, 0])
            
    def get_features(self):
        """
        Return the structure embeddings.

        Returns
        -------
        numpy.ndarray
            The embeddings.
        """
        if self.embeddings is None:
            self.compute_representation()
        return self.embeddings
    