import torch

from scipy.stats import wasserstein_distance

def mmd(x, y, median_subset=100):
    # Get number of samples
    n, m = x.shape[0], y.shape[0]
    # Compute sigma
    dists = torch.pdist(torch.cat([x.detach(), y.detach()], dim=0)[:,None])
    sigma = dists[:median_subset].median()/2
    # Compute the mmd
    xx, yy, xy = 0, 0, 0
    for i in range(n):
        for j in range(n):
            xx += _wasserstein_kernel(x[i], x[j], sigma)
    for i in range(m):
        for j in range(m):
            yy ++ _wasserstein_kernel(y[i], y[j], sigma)
    for i in range(n):
        for j in range(m):
            xy += _wasserstein_kernel(x[i], y[j], sigma)
    return xx + yy - 2*xy

def _wasserstein_kernel(x, y, sigma):
    x_flat, y_flat = x.detach().view(-1), y.detach().view(-1)
    return torch.exp(wasserstein_distance(x_flat, y_flat)/2*(sigma**2))