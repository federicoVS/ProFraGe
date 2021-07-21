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
