# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:20:02 2021

@author: FVS
"""

from Bio.PDB.Superimposer import Superimposer

class SIComparator:
    '''
    This class compares protein structures based on superimposition.
    
    Attributes
    ----------
    target : Bio.PDB.Structure
        The target protein.
    structures : list of Bio.PDB.Structure
        The structures which will be compared with the target structured.
    rmsds : dict of Bio.PDB.Structure -> float
        The dictionary holding the resulting RMSDs for each structure.
    '''
    
    def __init__(self, target, structures):
        '''
        Initializes the class.

        Parameters
        ----------
        target : Bio.PDB.Structure
            The target protein.
        structures : list of Bio.PDB.Structure
            The list of structures to be compared

        Returns
        -------
        None.
        '''
        self.target = target
        self.structures = structures
        self.rmsds = {}
        
    def get_rmsd(self, structure):
        '''
        Returns the RMSD for the specified protein.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            The structure of which the RMSD is to be returned.

        Returns
        -------
        float
            The RMSD associated with the structure. Note that if the structure has not yet
            been matched against the target structure, None is returned.
        '''
        if structure in self.rmsds:
            return self.rmsds[structure]
        else:
            return None
        
    def compare(self):
        '''
        Performs comparison via superimposition.

        Returns
        -------
        None.
        '''
        # Initialize/reset the RMSD dictionary
        self.rmsds = {}
        fixed = []
        for atom in self.target.get_atoms():
            fixed.append(atom)
        for structure in self.structures:
            moving = []
            for atom in structure.get_atoms():
                moving.append(atom.copy()) # should not modify the actual structure
            si = Superimposer()
            si.set_atoms(fixed, moving)
            si.apply()
            self.rmsds[structure] = si.rms
        