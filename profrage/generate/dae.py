import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from generate.layers import MLPLayer, DGCLayer
from generate.utils import reparametrize

class ProDAAE(nn.Module):
    """
    Protein graph generation model based on denoising adversarial autoencoders (DAAEs).

    Source
    ------
    Paper: => Educating Text Autoencoders: Latent Representation Guidance via Denoising
              Tianxiao Shen, Jonas Mueller, Regina Barzilay, Tommi Jaakkola
    Code   => https://github.com/shentianxiao/text-autoencoders
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
            The latent dimension.
        gcn_dims : list of int
            The dimensions for the ECC layers.
        mlp_dims : list of int
            The dimensions for the MLP layers.
        max_size : int, optional
            The maximum number of nodes in a graph. The default is 30.
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structure types. The default is 7.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1
        weight_init : float, optional
            The weight initialization bounds. The default is 5e-5.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(ProDAAE, self).__init__()
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
        # Discriminator
        self.discriminator = MLPLayer([latent_dim] + mlp_dims + [1])
        # Decoder (Generator)
        self.dec_x_aa = MLPLayer([latent_dim] + mlp_dims + [aa_dim+1])
        self.dec_x_ss = MLPLayer([latent_dim] + mlp_dims + [ss_dim+1])
        self.dec_w_adj = MLPLayer([latent_dim] + mlp_dims + [max_size])

        # Weights initialization
        self._init_weights(self.latent_mu)
        self._init_weights(self.latent_log_var)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-self.weight_init, b=self.weight_init)

    def _noise_x(self, x, p=0.95):
        x_noised = x.clone()
        B, N, F = x.shape[0], x.shape[1], x.shape[2]
        noised = torch.rand(B,N,F) > p
        non_noised = ~noised
        aa_noised = (x[:,:,0]*non_noised[:,:,0].int()).unsqueeze(2) + torch.add(torch.randint(self.aa_dim, (B,N,1)),1)
        ss_noised = (x[:,:,0]*non_noised[:,:,0].int()).unsqueeze(2) + torch.add(torch.randint(self.ss_dim, (B,N,1)),1)
        x_noised += torch.cat((aa_noised,ss_noised), dim=2)
        return x_noised.to(self.device)

    def _noise_w_adj(self, w_adj, p=0.95):
        w_adj_noised = w_adj.clone()
        B, N = w_adj.shape[0], w_adj.shape[1]
        noised = torch.rand(B,N) > p
        for b in range(B):
            for n in range(N):
                if noised[b,n]:
                    eps = torch.randn(N)
                    w_adj_noised[b,n,0:N] += eps
                    w_adj_noised[b,0:N,n] += eps
                w_adj_noised[b,n,n] = 0
        return w_adj_noised.to(self.device)

    def _loss(self, x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, z, l_adv):
        # Node classification/regression
        ce_loss_aa = F.cross_entropy(out_x_aa.permute(0,2,1), x[:,:,0].long())
        ce_loss_ss = F.cross_entropy(out_x_ss.permute(0,2,1), x[:,:,1].long())
        # Weight regression
        diag_idx = torch.eye(self.max_size,self.max_size).bool().to(self.device)
        mse_loss_edge = F.mse_loss(out_w_adj.masked_fill_(diag_idx, 0), w_adj.squeeze(2).masked_fill_(diag_idx, 0))
        # Node existence classification
        existence = torch.diagonal(out_w_adj, dim1=1, dim2=2)
        ce_loss_exist = F.binary_cross_entropy_with_logits(existence, mask.float())
        # Discriminator loss (https://github.com/shentianxiao/text-autoencoders/blob/master/model.py, lines 155-160)
        zn = torch.randn_like(z)
        zeros = torch.zeros(z.shape[0],z.shape[1],1, device=self.device)
        ones = torch.ones(z.shape[0],z.shape[1],1, device=self.device)
        d_z = torch.sigmoid(self.discriminator(z.detach()))
        d_zn = torch.sigmoid(self.discriminator(zn))
        d_loss = F.binary_cross_entropy(d_z, zeros) + F.binary_cross_entropy(d_zn, ones)
        # Return full loss
        return (ce_loss_aa + ce_loss_ss + mse_loss_edge + ce_loss_exist) - l_adv*d_loss

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
            The mask tensor indicating whether a paritcular node exists.

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
        z : torch.tensor
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

    def fit(self, loader, n_epochs, lr=1e-3, l_adv=1, betas=(0.9, 0.999), checkpoint=500, verbose=False):
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
        l_adv : float, optional
            The multiplier to apply to the adversarial loss. The default is 1.
        betas : (float,float), optional
            Coefficients used to compute averages of the gradient. The default is (0.9, 0.999).
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print loss information. The default is False.

        Returns
        -------

        """
        optimizer = Adam(self.parameters(), lr=lr, betas=betas)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                x, w_adj, mask = data['x'], data['w_adj'], data['mask']
                # Add noise (and put on the device in the process)
                x_noised = self._noise_x(x)
                w_adj_noised = self._noise_w_adj(w_adj)
                # Put (original) data on device
                x, w_adj, mask = x.to(self.device), w_adj.to(self.device), mask.to(self.device)
                # Start process
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x_noised, w_adj_noised, mask)
                z = reparametrize(mu, log_var, self.device)
                out_x_aa, out_x_ss, out_adj_w = self.decode(z)
                loss = self._loss(x, w_adj, mask, out_x_aa, out_x_ss, out_adj_w, z, l_adv)
                loss.backward()
                optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, optimizer, loss)
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x, w_adj, mask, l_adv):
        """
        Compute the evaluation loss of the model.

        Parameters
        ----------
        x : torch.Tensor
            The node features.
        w_adj : torch.Tensor
            The weighted adjacency tensor.
        mask : torch.Tensor
            The tensor indicating whether a particular node exists.
        l_adv : float
            The multiplier to apply to the adversarial loss

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
        loss = self._loss(x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, z, l_adv)
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