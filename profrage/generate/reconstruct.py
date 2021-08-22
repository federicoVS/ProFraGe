import numpy as np

import torch

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from utils.structure import INT_TO_AA
from utils.io import to_pdb

class GramReconstruction:
    """
    Reconstruction of coordinate matrix X based on distance matrix D using Gram matrix.

    Attributes
    ----------
    device : str
        The device where to put the data.
    """

    def __init__(self, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        self.device = device

    def reconstruct(self, D):
        """
        Reconstructs the coordinates based on the given distance matrix.

        Parameters
        ----------
        D : torch.Tensor
            The distance matrix.

        Returns
        -------
        X : numpy.ndarray
            The coordinate matrix.
        """
        n = D.shape[0]
        M = torch.zeros_like(D).to(self.device)
        for i in range(n):
            for j in range(n):
                M[i,j] = (D[0,j]**2 + D[0,i]**2 + - D[i,j]**2)/2
        M = M.detach().cpu().numpy()
        u, s, vh = np.linalg.svd(M)
        X = np.zeros(shape=(n,3))
        for i in range(n):
            for j in range(3):
                X[i,j] = u[i,j]*np.sqrt(s[j])
        return X

class FragmentBuilder:

    def __init__(self, f_id, x_pred, coords):
        self.f_id = f_id
        self.x_pred = x_pred
        self.coords = coords

    def build(self, out_dir='./'):
        structure = Structure(self.f_id)
        model = Model(0)
        chain = Chain("A")
        n = self.x_pred.shape[0]
        for i in range(n):
            atom = Atom("CA", np.round(self.coords[i,:], 3), 50.00, 1.00, " ", " CA ", "C")
            residue = Residue((" ",i+1," "), INT_TO_AA[self.x_pred[i,0].int().item()], i+1)
            residue.add(atom)
            chain.add(residue)
        model.add(chain)
        structure.add(model)
        to_pdb(structure, self.f_id, out_dir=out_dir)
