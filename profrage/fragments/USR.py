# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:55:58 2021

@author: FVS
"""

import numpy as np
from scipy.stats import skew
from utils.structure import get_atoms_coords

class USR:
    """
    Implements Ultrafast Shape Recognition (USR).
    
    Source
    ------
    Ultrafast shape recognition for similarity search in molecular databases
    (Pedro J. Ballester, W. Graham Richards)
    
    Attributes
    ----------
    coords : numpy.ndarray
        The matrix of coordinates
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
    """
    
    def __init__(self, structure):
        """
        Initialize the class.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure onto which to perform USR.

        Returns
        -------
        None.
        """
        self.coords = get_atoms_coords(structure)
        self.ctd = None
        self.cst = None
        self.fct = None
        self.ftf = None
        self.momenta = np.zeros(shape=(12,))
        
    def get_ctd(self):
        """
        Compute the coordinates of `ctd`.

        Returns
        -------
        None.
        """
        self.ctd = np.mean(self.coords, axis=0)
        
    def get_cst(self):
        """
        Compute the coordinates of `cst`.

        Returns
        -------
        None.
        """
        squared_dist = np.sum((self.coords-self.ctd)**2, axis=1)
        self.cst = self.coords[np.argmin(squared_dist),:]
    
    def get_fct(self):
        """
        Compute the coordinates of `fct`.

        Returns
        -------
        None.
        """
        squared_dist = np.sum((self.coords-self.ctd)**2, axis=1)
        self.fct = self.coords[np.argmax(squared_dist),:]
    
    def get_ftf(self):
        """
        Compute the coordinates of `ftf`.

        Returns
        -------
        None.
        """
        squared_dist = np.sum((self.coords-self.fct)**2, axis=1)
        self.ftf = self.coords[np.argmax(squared_dist),:]
        
    def compute_momenta(self):
        """
        Compute the momenta.

        Returns
        -------
        None.
        """
        dist_ctd = np.sum((self.coords-self.ctd)**2, axis=1)
        dist_cst = np.sum((self.coords-self.cst)**2, axis=1)
        dist_fct = np.sum((self.coords-self.fct)**2, axis=1)
        dist_ftf = np.sum((self.coords-self.ftf)**2, axis=1)
        # Mean
        self.momenta[0] = np.mean(dist_ctd) # ctd
        self.momenta[3] = np.mean(dist_cst) #cst
        self.momenta[6] = np.mean(dist_fct) # fct
        self.momenta[9] = np.mean(dist_ftf) # ftf
        # Variance
        self.momenta[1] = np.var(dist_ctd) # ctd
        self.momenta[4] = np.var(dist_cst) #cst
        self.momenta[7] = np.var(dist_fct) # fct
        self.momenta[10] = np.var(dist_ftf) # ftf
        # Skewness
        self.momenta[2] = skew(dist_ctd) # ctd
        self.momenta[5] = skew(dist_cst) #cst
        self.momenta[8] = skew(dist_fct) # fct
        self.momenta[11] = skew(dist_ftf) # ftf
        
    def compute_all(self):
        """
        Compute the features and their momenta.

        Returns
        -------
        None.
        """
        self.get_ctd()
        self.get_cst()
        self.get_fct()
        self.get_ftf()
        self.compute_momenta()
    
    