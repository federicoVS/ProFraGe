import numpy as np

import torch

from Bio.PDB.QCPSuperimposer import QCPSuperimposer

def graph_metrics(adj):
    """
    Compute metrics associated with the generated adjacency matrix.

    The median backbone score assigns penalties to backbone connections not in the backbone. The illegal edges score assigns
    penalties for self-loops. The contacts degree count the median degree for interactions (not including backbones).

    Parameters
    ----------
    adj : torch.tensor
        The symmetric adjacency metric of which to compute the metrics.

    Returns
    -------
    scores : torch.tensor
        The scores.
    """
    # Get number of node
    n = adj.shape[0]
    # Initialize scores
    backbone_score, illegal_edges, contacts_deg = [], [], []
    # Compute the scores
    for i in range(n-1):
        bb_score, c_deg = 0, 0
        for j in range(i+1,n):
            edge = int(adj[i,j])
            if i == j:
                if edge != 0:
                    illegal_edges.append(1)
            else:
                illegal_edges.append(0)
                if edge == 1:
                    c_deg += 1
                if edge == 2 and abs(i-j) == 1:
                    bb_score += 1
                if edge == 2 and abs(i-j) != 1:
                    bb_score -= 1
        backbone_score.append(bb_score)
        contacts_deg.append(c_deg)
    backbone_score = torch.tensor(backbone_score)
    illegal_edges = torch.tensor(illegal_edges)
    contacts_deg = torch.tensor(contacts_deg)
    scores = torch.tensor([torch.median(backbone_score), torch.median(illegal_edges), torch.median(contacts_deg)])
    return scores

def amino_acid_metrics(x, aa_num=20):
    """
    Compute metrics associated with with amino acids.

    It counts how many times an amino acid appears in the proteins, and the median length of a sequence of
    amino acids, e.g. ALA, ALA, ..., ALA.

    Parameters
    ----------
    x : torch.tensor:
        The node features.
    aa_num : int, optional
        The number of amino acids. The default is 20.

    Returns
    -------
    scores : torch.tensor
        The score having the following format: [AA_1_count, ..., AA_aa_num_count, median_seq_len]
    """
    # Get number of amino acids
    n = x.shape[0]
    # Initialize scores
    aa_counts, seq_len = [0]*aa_num, []
    current_seq_len, current_aa = 0, -1
    # Compute the scores
    for i in range(n):
        aa = int(x[i,0])
        aa_counts[aa-1] += 1
        if current_aa == -1:
            current_seq_len += 1
            current_aa = aa
        elif aa != current_aa:
            seq_len.append(current_seq_len)
            current_seq_len = 1
            current_aa = aa
        else:
            current_seq_len += 1
    seq_len.append(current_seq_len) # last one to add
    aa_counts = [x/n for x in aa_counts]
    seq_len = float(torch.median(torch.tensor(seq_len)))
    scores = torch.tensor(aa_counts + [seq_len])
    return scores

def secondary_sequence_metrics(x, ss_num=7):
    """
    Compute metrics associated with with secondary structure.

    It counts how many times a secondary structure appears in the proteins, and the median length of a sequence of
    secondary structure, e.g. H, H, ..., H.

    Parameters
    ----------
    x : torch.tensor:
        The node features.
    ss_num : int, optional
        The number of secondary structures. The default is 7.

    Returns
    -------
    scores : torch.tensor
        The score having the following format: [SS_1_count, ..., SS_ss_num_count, median_seq_len]
    """
    # Get number of amino acids
    n = x.shape[0]
    # Initialize scores
    ss_counts, seq_len = [0]*ss_num, []
    current_seq_len, current_ss = 0, -1
    # Compute the scores
    for i in range(n):
        ss = int(x[i,1])
        ss_counts[ss-1] += 1
        if current_ss == -1:
            current_seq_len += 1
            current_ss = ss
        elif ss != current_ss:
            seq_len.append(current_seq_len)
            current_seq_len = 1
            current_ss = ss
        else:
            current_seq_len += 1
    seq_len.append(current_seq_len) # last one to add
    ss_counts = [x/n for x in ss_counts]
    seq_len = float(torch.median(torch.tensor(seq_len)))
    scores = torch.tensor(ss_counts + [seq_len])
    return scores

def ca_metrics(coords_fixed, coords_moving):
    """
    Compute the superimposing of a two sets of C-alpha atoms.

    Parameters
    ----------
    coords_fixed : list of torch.tensor
        The list of target C-alpha coordinates.
    coords_moving : list of torch.tensor
        The list of input C-alpha coordinates.

    Returns
    -------
    float
        The RMSD score of the superimposition.
    """
    # Prepare the coordinates
    fixed, moving = np.zeros(shape=(len(coords_fixed),3)), np.zeros(shape=(len(coords_moving),3))
    for i in range(len(coords_fixed)):
        fixed[i,:], moving[i,:] = coords_fixed[i], coords_moving[i]
    # Superimpose
    qcpsi = QCPSuperimposer()
    qcpsi.set(fixed, moving)
    qcpsi.run()
    return qcpsi.get_rms()