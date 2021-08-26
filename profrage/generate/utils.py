import torch

def reparametrize(mu, log_var, device):
    """
    Reparametrize based on input mean and log variance

    Parameters
    ----------
    mu : torch.tensor
        The mean.
    log_var : torch.tensor
        The log variance.
    device : str
        The device on which to put the data.

    Returns
    -------
    z : torch.tensor
        The reparametrized value.
    """
    sigma = torch.exp(0.5*log_var)
    epsilon = torch.rand_like(sigma)
    z = mu + epsilon*sigma
    return z.to(device)

def adj_to_seq(adj, device='cpu'):
    """
    Convert a dense adjacency matrix into a sequence.

    Parameters
    ----------
    adj : torch.Tensor
        The dense adjacency tensor.
    device : str, optional
        The device onto which to put the data. The default is 'cpu'.

    Returns
    -------
    adj_seq : torch.Tensor
        The sequence representing the input adjacency tensor.
    """
    B, N = adj.shape[0], adj.shape[1]
    adj_seq = torch.zeros(B,int(((N-1)*N)/2)).to(device)
    for b in range(B):
        for i in range(1,N):
            for j in range(i):
                adj_seq[b,i+j] = adj[b,i,j]
    return adj_seq

def seq_to_adj(adj_seq, device='cpu'):
    """
    Convert an adjacency sequence to its corresponding dense representation.

    Parameters
    ----------
    adj_seq : torch.Tensor
        The sequence adjacency.
    device : str, optional
        The device onto which to put the data. The default is 'cpu'.

    Returns
    -------
    adj : torch.Tensor
        The dense representation of the input sequence.
    """
    B, n = adj_seq.shape[0], adj_seq.shape[1]
    adj = torch.zeros(B,n,n).to(device)
    for b in range(B):
        for i in range(n):
            for j in range(n):
                adj[b,i,j] = adj[b,j,i] = adj_seq[b,i,j]
    return adj

def clipping_dist(delta):
    """
    Returns the average distance between residues i and j, based on experimental data.

    Parameters
    ----------
    delta : int
        The delta between residue indexes i and j.

    Returns
    -------
    float
        The average distance.
    """
    if delta == 1:
        return 4
    elif delta == 2:
        return 6
    elif delta == 3:
        return 7.5
    elif delta == 4:
        return 8.5
    elif delta == 5:
        return 10
    elif delta == 6:
        return 10.5
    elif delta == 7:
        return 11
    elif delta == 8:
        return 12
    else:
        return 12.5