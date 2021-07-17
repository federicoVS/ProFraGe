import numpy as np

import torch
from torch.autograd import Variable

def circular_mean(alphas, is_deg=True):
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

def dense_to_sparse(x_dense, adj_dense, edge_dense):
    bs, n = x_dense.shape[0], x_dense.shape[1]
    x_sparse, adj_sparse, edge_sparse = torch.zeros(bs*n,x_dense.shape[2]), [], []
    for b in range(bs):
        x_sparse[b*n:(b+1)*n,:] = x_dense[b,:,:]
        for i in range(n-1):
            for j in range(i+1,n):
                if adj_dense[b,i,j] > 0:
                    adj_sparse.append([i,j])
                    adj_sparse.append([j,i])
                    if edge_dense is not None:
                        edge_sparse.append([edge_dense[b,i,j,0],edge_dense[b,i,j,1]])
                        edge_sparse.append([edge_dense[b,j,i,0],edge_dense[b,j,i,0]])
    adj_sparse = torch.LongTensor(adj_sparse).t().contiguous()
    edge_sparse = torch.FloatTensor(edge_sparse)
    return x_sparse, adj_sparse, edge_sparse

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, device='cpu'):
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

def sample_softmax(y):
    y = torch.softmax(y, dim=2)
    return torch.argmax(y)
