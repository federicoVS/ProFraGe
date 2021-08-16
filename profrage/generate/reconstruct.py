import numpy as np
import torch

class Gram:
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
                M[i,j] = (D[0,j]**2 + D[i,0]**2 + - D[i,j]**2)/2
        M = M.numpy()
        w, v = np.linalg.eig(M)
        X = np.matmul(w, np.sqrt(v))[:,0:3]
        return X