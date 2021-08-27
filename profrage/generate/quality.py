import numpy as np

import torch

from scipy.stats import wasserstein_distance

from generate.metrics import ca_metrics, amino_acid_metrics, secondary_sequence_metrics

class MMD:
    """
    Compute the Maximum Mean Discrepancy (MMD) between the predicted graph and a set of target ones.

    Graph statistics include amino acid sequence information and secondary structure sequence information.

    Source
    ------
    https://torchdrift.org/notebooks/note_on_mmd.html

    Attributes
    ----------
    pred_graph : (torch.Tensor, torch.Tensor)
        The predicted graph consisting of the predicted node features and the predicted distance matrix.
    target_graphs : list of (torch.Tensor, torch.Tensor)
        The target graphs, where each entry consists of the node features and the distance matrix.
    median_subset : int
        The size of the subset to use to compute `sigma`.
    """

    def __init__(self, pred_graph, target_graphs, median_subset=100):
        """
        Initialize the class.

        Parameters
        ----------
        pred_graph : (torch.Tensor, torch.Tensor)
            The predicted graph consisting of the predicted node features and the predicted distance matrix.
        target_graphs : list of (torch.Tensor, torch.Tensor)
            The target graphs, where each entry consists of the node features and the distance matrix.
        median_subset : int, optional
            The size of the subset to use to compute `sigma`. The default is 100.
        """
        self.pred_graph = pred_graph
        self.target_graphs = target_graphs
        self.median_subset = median_subset

    def _wasserstein_kernel(self, x, y, sigma):
        x_flat, y_flat = x.detach().view(-1), y.detach().view(-1)
        return torch.exp(wasserstein_distance(x_flat, y_flat)/2*(sigma**2))

    def _mmd(self, x, y):
        # Get number of samples
        n, m = x.shape[0], y.shape[0]
        # Compute sigma
        dists = torch.pdist(torch.cat([x.detach(), y.detach()], dim=0)[:,None])
        sigma = dists[:self.median_subset].median()/2
        # Compute the mmd
        xx, yy, xy = 0, 0, 0
        for i in range(n):
            for j in range(n):
                xx += self._wasserstein_kernel(x[i], x[j], sigma)
        for i in range(m):
            for j in range(m):
                yy += self._wasserstein_kernel(y[i], y[j], sigma)
        for i in range(n):
            for j in range(m):
                xy += self._wasserstein_kernel(x[i], y[j], sigma)
        return xx + yy - 2*xy

    def compare_graphs(self):
        """
        Compare the graphs.

        Returns
        -------
        scores : numpy.ndarray
            The comparison scores.
        """
        scores = []
        # Compute MMD score of the predicted graph
        x_aa = amino_acid_metrics(self.pred_graph[0])
        x_ss = secondary_sequence_metrics(self.pred_graph[0])
        x = torch.cat((x_aa,x_ss))
        for target_graph in self.target_graphs:
            y_aa = amino_acid_metrics(target_graph[0])
            y_ss = secondary_sequence_metrics(target_graph[0])
            y = torch.cat((y_aa,y_ss))
            # Compare
            scores.append(self._mmd(x, y))
        scores = np.array(scores)
        return scores

class QCP:
    """
    Compute the superimposition RMSD of the predicted atoms against validation ones.

    The idea of this control is that if the overall RMSD is low, then the generated protein is likely to be
    realistic.

    Attributes
    ----------
    pred_coords : numpy.ndarray
        The coordinates of the generated C-alpha atoms.
    target_proteins : list of Bio.PDB.Structure
        The list of target proteins.
    """

    def __init__(self, pred_coords, target_proteins):
        """
        Initialize the class.

        Parameters
        ----------
        pred_coords : torch.tensor
            The coordinates of the generated C-alpha atoms. The input should be a PyTorch tensor, which it then
            converted to a Numpy array.
        target_proteins : list of Bio.PDB.Structure
            The list of target proteins.
        """
        self.pred_coords = pred_coords
        self.target_proteins = target_proteins

    def _get_target_ca_atoms_coords(self):
        target_coords_list = []
        for protein in self.target_proteins:
            coords = []
            for model in protein:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords.append(residue['CA'].get_coord())
            target_coords_list.append(np.array(coords))
        return target_coords_list

    def _get_shifted_coords(self, coords_long, coords_short):
        shifted_coords, i = [], 0
        while i + len(coords_short) <= len(coords_long):
            shifted_coords.append(coords_long[i:i+len(coords_short)])
            i += 1
        return shifted_coords

    def superimpose(self):
        """
        Compute the superimposition.

        Returns
        -------
        numpy.ndarray
            The array of RMSD scores (each i-th entry corresponds to the comparison of the generated atoms with the
            i-th protein)
        """
        scores = []
        target_coords_list = self._get_target_ca_atoms_coords()
        for target_coords in target_coords_list:
            if len(target_coords) == len(self.pred_coords):
                scores.append(ca_metrics(target_coords, self.pred_coords))
            elif len(target_coords) > len(self.pred_coords):
                shifted_coords = self._get_shifted_coords(target_coords, self.pred_coords)
                rmsds = []
                for scs in shifted_coords:
                    rmsds.append(ca_metrics(scs, self.pred_coords))
                scores.append(np.min(np.array(rmsds)))
            else:
                shifted_coords = self._get_shifted_coords(self.pred_coords, target_coords)
                rmsds = []
                for scs in shifted_coords:
                    rmsds.append(ca_metrics(target_coords, scs))
                scores.append(np.min(np.array(rmsds)))
        scores = np.array(scores)
        return scores