import torch

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
    aa_counts = aa_counts/n
    seq_len = torch.median(torch.tensor(seq_len))
    scores = torch.tensor(aa_counts + seq_len)
    return scores

def secondary_sequence_metrics(x, ss_num=7):
    # Get number of amino acids
    n = x.shape[0]
    # Initialize scores
    ss_counts, seq_len = [0]*ss_num, []
    current_seq_len, current_ss = 0, -1
    # Compute the scores
    for i in range(n):
        ss = int(x[i,1])
        ss_counts[ss] += 1
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
    ss_counts = ss_counts/n
    seq_len = torch.median(torch.tensor(seq_len))
    scores = torch.tensor(ss_counts + seq_len)
    return scores