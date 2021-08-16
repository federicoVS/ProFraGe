import numpy as np
import torch

class Gram:

    def __init__(self, device='cpu'):
        self.device = device

    def reconstruct(self, D):
        n = D.shape[0]
        M = torch.zeros_like(D).to(self.device)
        for i in range(n):
            for j in range(n):
                M[i,j] = (D[0,j]**2 + D[i,0]**2 + - D[i,j]**2)/2
        M = M.numpy()
        w, v = np.linalg.eig(M)
        X = np.matmul(w, np.sqrt(v))[:,0:3]
        return X