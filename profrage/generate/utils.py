import numpy as np

import torch
from torch.autograd import Variable

def circular_mean(alphas, is_deg=True):
    """
    Compute the circular mean of the given angles.

    Parameters
    ----------
    alphas : list of float
        The angles.
    is_deg : bool, optional
        Whether the angles are expressed in degrees. The default is True.

    Returns
    -------
    float
        The circular mean.
    """
    n = len(alphas)
    sum_sin, sum_cos = 0, 0
    for i in range(n):
        alpha = alphas[i]
        if is_deg:
            alpha = np.deg2rad(alpha)
        sum_sin += np.sin(alpha)
        sum_cos += np.cos(alpha)
    sum_sin, sum_cos = sum_sin/n, sum_cos/n
    return np.arctan2(sum_sin, sum_cos)

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, device='cpu'):
    """
    Sample from the given set of values.

    Source
    ------
    https://github.com/snap-stanford/GraphRNN

    Parameters
    ----------
    y : torch.tensor
        The tensor from which to sample.
    sample : bool
        Whether to sample.
    thresh : float
        The threshold.
    sample_time : int, optional
        The sample time. The default is 2.
    device : str, optional
        The device on where to put the data. The default is 'cpu'.

    Returns
    -------
    y_result : torch.tensor
        The sampled result.
    """
    y = torch.sigmoid(y)
    if sample:
        if sample_time > 1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).to(device)
            for i in range(y_result.size(0)):
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).to(device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).to(device)
            y_result = torch.gt(y, y_thresh).float()
    else:
        y_thresh = Variable(torch.ones(y.size(0),y.size(1),y.size(2))*thresh).to(device)
        y_result = torch.gt(y,y_thresh).float()
    return y_result

def nan_to_num(x, nan=0, posinf=1e10, neginf=1e-10):
    """
    Replaces non-numerical values in the input tensor with the specified ones.

    The tensor is modified in-place.

    Parameters
    ----------
    x : torch.tensor
        The input tensor.
    nan : float, optional
        The value with which to replace NaN entries. The default is 0.
    posinf : float, optional
        The value with which to replace positive infinite entries. The default is 1e10.
    neginf : float, optional
        The value with which to replace negative infinite entries. The default is 1e10.

    Returns
    -------
    x : torch.tensor
        The modified tensor.
    """
    x[x!=x] = nan
    x[x==float('Inf')] = posinf
    x[x==-float('Inf')] = neginf
    return x

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

def node_feature_target_classes(x, device, aa_dim=20, ss_dim=7, ignore_idx=-100):
    """
    Map the node amino acid and secondary structure to a single class (as a target).

    Parameters
    ----------
    x : torch.tensor
        The sparse node features.
    device : str
        The device on which to put the data.
    aa_dim : int, optional
        The number of amino acids. The default is 20
    ss_dim : int, optional
        The number of secondary structure types. The default is 7.
    ignore_idx : int, optional
        The indexes to ignore. The default is -100.

    Returns
    -------
    x_classes : torch.tensor
        The target tensor.
    """
    b_dim, n = x.shape[0], x.shape[1]
    x_classes = torch.zeros(b_dim,1, dtype=torch.long).to(device)
    for b in range(b_dim):
        a, s = x[b,0] - 1, x[b,1] - 1 # subtract one because classes begin with 1
        mapping = s*aa_dim + a
        if mapping < 0 or mapping >= aa_dim*ss_dim:
            x_classes[b,0] = ignore_idx
        else:
            x_classes[b,0] = mapping
    x_classes = x_classes.view(b_dim)
    return x_classes

def edge_features_input(adj_edge, x_len, max_size, edge_dim, edge_class_dim, device):
    """
    Map the predicted edge features tensor to be the input.

    Parameters
    ----------
    adj_edge : torch.tensor
        The predicted tensor of combining the adjacency and edge features.
    x_len : list of int
        The number of nodes for each graph in the batch.
    max_size : int
        The maximum number of nodes.
    edge_dim : int
        The number of edge features.
    edge_class_dim : int
        The number of classes in the edge features.
    device : str
        The device on which to put the data.

    Returns
    -------
    ae_dense : torch.tensor
        The input tensor.
    """
    ae_dense = torch.zeros(len(x_len),max_size,max_size,edge_dim+edge_class_dim-1).to(device)
    prev = 0
    for i in range(len(x_len)):
        xl = x_len[i]
        ae_dense[i,0:xl,:,:] = adj_edge[prev:prev+xl]
        prev += xl
    return ae_dense

def edge_features_target(adj_sparse, edge_sparse, x_len, edge_len, max_size, edge_dim, device):
    """
    Target edge features.

    Parameters
    ----------
    adj_sparse : torch.tensor
        The adjacency tensor. It should be sparse.
    edge_sparse : torch.tensor
        the edge features. It should be sparse.
    x_len : list of int
        The number of nodes for each graph in the batch.
    edge_len : list of int
        The number of edge for each graph in the batch.
    max_size : int
        The maximum number of nodes.
    edge_dim : int
        The number of edge features.
    device : str
        The device on which to put the data.

    Returns
    -------
    ae_dense : torch.tensor
        The target tensor.
    """
    ae_dense = torch.zeros(len(edge_len),max_size,max_size,edge_dim).to(device)
    prev_i, prev_x = 0, 0
    for i in range(len(edge_len)):
        el = edge_len[i]
        i_idx, j_idx = adj_sparse[:,prev_i:prev_i+el][0] - prev_x, adj_sparse[:,prev_i:prev_i+el][1] - prev_x
        edge_type, edge_dist = edge_sparse[prev_i:prev_i+el][:,1], edge_sparse[prev_i:prev_i+el][:,0]
        ae_dense[i,i_idx,j_idx,0] = edge_dist
        ae_dense[i,i_idx,j_idx,1] = edge_type + 1 # now 0 means no connections
        prev_i += el
        prev_x += x_len[i]
    return ae_dense

def sparse_adj_target(adj_sparse, N, device):
    M = adj_sparse.shape[1]
    adj_targets = torch.zeros(N*N).to(device) # int((N*(N+1))/2)-N
    for m in range(M):
        # Skip the second edge since it is the same as the first
        if m % 2 == 0:
            i, j = adj_sparse[:,m][0], adj_sparse[:,m][1]
            mapping = N*i + j
            adj_targets[mapping] = mapping
    return adj_targets

def sparse_edge_target(adj_sparse, edge_sparse, N, edge_dim, device):
    M = edge_sparse.shape[0]
    edge_targets = torch.zeros(N*N,edge_dim).to(device)
    for m in range(M):
        # Skip the second edge since it is the same as the first
        if m % 2 == 0:
            i, j = adj_sparse[:,m][0], adj_sparse[:,m][1]
            mapping = N*i + j
            edge_targets[mapping] = edge_sparse[m,:]
    return edge_targets