import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from generate.layers import MLPLayer, DGCLayer
from generate.utils import reparametrize

class ProVAE(nn.Module):
    """
    VAE model to generate a graph.
    """

    def __init__(self, root, hidden_dim, latent_dim, gcn_dims, mlp_dims,
                 max_size=30, aa_dim=20, ss_dim=7, dropout=0.1, weight_init=5e-5, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The directory where to save the model state.
        hidden_dim : int
            The hidden dimension.
        latent_dim : int
            The dimension of the latent space.
        gcn_dims : list of int
            The dimensions for the graph convolutional layers.
        mlp_dims : list of int
            The dimensions for the MLP layers.
        max_size : int, optional
            The maximum number of amino acids in a protein. The default is 30.
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structures. The default is 7.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1.
        weight_init : float, optional
            The weight initialization bounds. The default is 5e-5.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(ProVAE, self).__init__()
        self.root = root
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.max_size = max_size
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.weight_init = weight_init
        self.device = device

        self.X_DIM = 2

        # Encoding
        self.enc = DGCLayer([self.X_DIM] + gcn_dims + [hidden_dim])
        # Sampling
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoding
        self.dec_x_aa = MLPLayer([latent_dim] + mlp_dims + [aa_dim+1])
        self.dec_x_ss = MLPLayer([latent_dim] + mlp_dims + [ss_dim+1])
        self.dec_w_adj = MLPLayer([latent_dim] + mlp_dims + [max_size])

        # Weights initialization
        self._init_weights(self.latent_mu)
        self._init_weights(self.latent_log_var)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-self.weight_init, b=self.weight_init)

    def _vae_loss(self, x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, mu, log_var, l_kld):
        # Node classification/regression
        ce_loss_aa = F.cross_entropy(out_x_aa.permute(0,2,1), x[:,:,0].long())
        ce_loss_ss = F.cross_entropy(out_x_ss.permute(0,2,1), x[:,:,1].long())
        # Weight regression
        diag_idx = torch.eye(self.max_size,self.max_size).bool().to(self.device)
        mse_loss_edge = F.mse_loss(out_w_adj.masked_fill_(diag_idx, 0), w_adj.squeeze(2).masked_fill_(diag_idx, 0))
        # Node existence classification
        existence = torch.diagonal(out_w_adj, dim1=1, dim2=2)
        ce_loss_exist = F.binary_cross_entropy_with_logits(existence, mask.float())
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return ce_loss_aa + ce_loss_ss + mse_loss_edge + ce_loss_exist + l_kld*kl_loss

    def encode(self, x, w_adj, mask):
        """
        Compute the encoding.

        Parameters
        ----------
        x : torch.Tensor
            The node features tensor.
        w_adj : torch.Tensor
            The weighted adjacency tensor.
        mask : torch.Tensor
            The tensor indicating whether a particular node exists.

        Returns
        -------
        (out, mu, log_var) : (torch.Tensor, torch.Tensor, torch.Tensor)
            The encoded value, the mean and variance logarithm.
        """
        out = self.enc(x, w_adj, mask=mask)
        mu, log_var = self.latent_mu(out), self.latent_log_var(out)
        return out, mu, log_var

    def decode(self, z):
        """
        Decode the sample space.

        Parameters
        ----------
        z : torch.Tensor
            The sample space.

        Returns
        -------
        (out_x_aa, out_x_ss, out_adj_w) : (torch.Tensor, torch.Tensor, torch.Tensor)
            The decoded node amino acid, secondary structure, and weighted adjacency.
        """
        out_x_aa = F.dropout(self.dec_x_aa(z), p=self.dropout)
        out_x_ss = F.dropout(self.dec_x_ss(z), p=self.dropout)
        out_w_adj = F.dropout(self.dec_w_adj(z), p=self.dropout)
        out_w_adj = out_w_adj.view(-1,self.max_size,self.max_size)
        return out_x_aa, out_x_ss, out_w_adj

    def checkpoint(self, epoch, optimizer, loss):
        """
        Create a checkpoint saving the results on the ongoing optimization.

        The checkpoint is saved at ROOT/checkpoint_<epoch>.

        Parameters
        ----------
        epoch : int
            The current epoch.
        optimizer : torch.optim.Optimizer
            The optimizer.
        loss : float
            The current loss.

        Returns
        -------
        None
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, self.root + 'checkpoint_' + str(epoch))

    def fit(self, loader, n_epochs, lr=1e-3, l_kld=1e-3, betas=(0.9, 0.999), checkpoint=500, verbose=False):
        """
        Train the model.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader or torch_geometric.data.DataLoader
            The data loader.
        n_epochs : int
            The number of epochs to perform.
        lr : float, optional
            The learning rate. The default is 1e-3.
        l_kld : float, optional
            The penalty to apply to the Kullback-Leibler loss, for stability reasons. The default is 1e-3.
        betas : (float,float), optional
            Coefficients used to compute averages of the gradient. The default is (0.9, 0.999).
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print loss information. The default is False.

        Returns
        -------
        None
        """
        optimizer = Adam(self.parameters(), lr=lr, betas=betas)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                # Get the data
                x, w_adj, mask = data['x'], data['w_adj'], data['mask']
                # Put the data on the device
                x, w_adj, mask = x.to(self.device), w_adj.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x, w_adj, mask)
                z = reparametrize(mu, log_var, self.device)
                out_x_aa, out_x_ss, out_w_adj = self.decode(z)
                loss = self._vae_loss(x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, mu, log_var, l_kld)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), w_norm)
                optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, optimizer, loss)
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x, w_adj, mask, l_kld):
        """
        Compute the evaluation loss of the model.

        This function is called during cross-validation.

        Parameters
        ----------
        x : torch.Tensor
            The node features.
        w_adj : torch.Tensor
            The weighted adjacency tensor.
        mask : torch.Tensor
            The tensor indicating whether a particular node exists.
        l_kld : float in [0,1]
            The penalty to apply to the Kullback-Leibler loss, for stability reasons.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Forward pass
        _, mu, log_var = self.encode(x, w_adj, mask)
        z = reparametrize(mu, log_var, self.device)
        out_x_aa, out_x_ss, out_w_adj = self.decode(z)
        # Compute the loss
        loss = self._vae_loss(x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, mu, log_var, l_kld)
        return {'Loss': loss}

    def generate(self, verbose=False):
        """
        Evaluate the model by generating new data.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print generation information. The default is False.

        Returns
        -------
        (x_pred, dist_pred) : (torch.Tensor, torch.Tensor)
            The predicted node features and distance matrix.
        """
        # Get the generated data
        z = torch.randn((1,self.max_size,self.latent_dim)).to(self.device)
        gen_x_aa, gen_x_ss, gen_w_adj = self.decode(z)
        # Softmax on classes
        gen_x_aa = torch.softmax(gen_x_aa, dim=2)
        gen_x_ss = torch.softmax(gen_x_ss, dim=2)
        # Get the nodes to generate and define the total number of nodes
        nodes, N = [0 for _ in range(self.max_size)], 0
        for i in range(self.max_size):
            nodes[i] = round(gen_w_adj[0,i,i].item())
            N += nodes[i]
        if verbose:
            print(f'The generated graph has {N} nodes.')
        # Define data to be returned
        x_pred, dist_pred = torch.zeros(N,self.X_DIM).to(self.device), torch.zeros(N,N).to(self.device)
        # Fill the node prediction
        idx = 0
        for i in range(self.max_size):
            if nodes[i] == 1:
                x_pred[idx,0] = torch.argmax(gen_x_aa[0,i,:])
                x_pred[idx,1] = torch.argmax(gen_x_ss[0,i,:])
                idx += 1
        # Fill the distance matrix prediction
        idx_i = 0
        for i in range(self.max_size):
            if nodes[i] == 1:
                idx_j = 0
                for j in range(self.max_size):
                    if nodes[j] == 1:
                        if i != j:
                            dist_pred[idx_i,idx_j] = 1/gen_w_adj[0,i,j]
                        idx_j += 1
                idx_i += 1
        return x_pred.long(), dist_pred.float()