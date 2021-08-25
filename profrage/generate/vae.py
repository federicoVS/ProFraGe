import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from generate.layers import MLPLayer, DGCLayer
from generate.utils import reparametrize

class ProVAE(nn.Module):
    """
    VAE model to generate a graph.
    """

    def __init__(self, root, hidden_dim, latent_dim, gcn_dims, mlp_dims,
                 max_size=30, aa_dim=20, ss_dim=7, dropout=0.1, device='cpu'):
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
        self.device = device

        self.X_DIM = 2

        # Encoding
        self.enc_gnn = DGCLayer([self.X_DIM] + gcn_dims + [hidden_dim])
        # Sampling
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoding
        self.dec_x_aa = MLPLayer([latent_dim] + mlp_dims + [aa_dim+1])
        self.dec_x_ss = MLPLayer([latent_dim] + mlp_dims + [ss_dim+1])
        self.dec_w_adj = MLPLayer([latent_dim] + mlp_dims + [max_size])
        self.dec_mask = MLPLayer([latent_dim] + mlp_dims + [2])

        # Weights initialization
        nn.init.xavier_uniform_(self.latent_mu.weight.data)
        self.latent_mu.bias.data.zero_()
        nn.init.xavier_uniform_(self.latent_log_var.weight.data)
        self.latent_log_var.bias.data.zero_()

    def _vae_loss(self, x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, out_mask, mu, log_var, l_kld):
        # Node classification/regression
        ce_loss_aa = F.cross_entropy(out_x_aa.permute(0,2,1), x[:,:,0].long())
        ce_loss_ss = F.cross_entropy(out_x_ss.permute(0,2,1), x[:,:,1].long())
        # Weight regression
        adj_seq_tgt = w_adj - torch.triu(w_adj)
        adj_seq_out = out_w_adj - torch.triu(out_w_adj)
        mse_loss_edge = F.mse_loss(adj_seq_out, adj_seq_tgt)
        # Node existence classification
        ce_loss_exist = F.cross_entropy(out_mask.permute(0,2,1), mask.long())
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return {'AA loss': ce_loss_aa,
                'SS loss': ce_loss_ss,
                'A_w loss': mse_loss_edge,
                'Mask loss': ce_loss_exist,
                'KL': l_kld*kl_loss,
                'Full loss': ce_loss_aa + ce_loss_ss + mse_loss_edge + ce_loss_exist + l_kld*kl_loss}

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
        out = self.enc_gnn(x, w_adj.clone(), mask=mask.clone())
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
        (out_x_aa, out_x_ss, out_adj_w, out_mask) : (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            The decoded node amino acid, secondary structure, weighted adjacency, and node mask.
        """
        out_x_aa = F.dropout(self.dec_x_aa(z), p=self.dropout)
        out_x_ss = F.dropout(self.dec_x_ss(z), p=self.dropout)
        out_w_adj = F.dropout(self.dec_w_adj(z, activation=F.relu), p=self.dropout)
        out_mask = F.dropout(self.dec_mask(z), p=self.dropout)
        out_w_adj = out_w_adj.view(-1,self.max_size,self.max_size)
        out_mask = out_mask.squeeze(2)
        return out_x_aa, out_x_ss, out_w_adj, out_mask

    def checkpoint(self, epoch, optimizers, schedulers, loss):
        """
        Create a checkpoint saving the results on the ongoing optimization.

        The checkpoint is saved at ROOT/checkpoint_<epoch>.

        Parameters
        ----------
        epoch : int
            The current epoch.
        optimizers : list of torch.optim.Optimizer
            The optimizers.
        schedulers : list of torch.optim.lr_scheduler.Scheduler
            The schedulers.
        loss : float
            The current loss.

        Returns
        -------
        None
        """
        state = {}
        state['epoch'] = epoch
        state['model_state_dict'] = self.state_dict()
        for i in range(len(optimizers)):
            state['optimizer_state_dict_'+str(i)] = optimizers[i].state_dict()
        for i in range(len(schedulers)):
            state['scheduler_state_dict_'+str(i)] = schedulers[i].state_dict()
        state['loss'] = loss
        torch.save(state, self.root + 'checkpoint_' + str(epoch))

    def fit(self, loader, n_epochs, lr=1e-3, l_kld=1e-3, betas=(0.9, 0.999), decay_milestones=[400,1000], decay=0.1, checkpoint=500, verbose=False):
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
        decay_milestones : list of int, optional
            The milestones at which to aply weight decay. The default is [400,1000].
        decay : float in [0,1], optional
            The weight decay. The default is 0.1.
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print loss information. The default is False.

        Returns
        -------
        None
        """
        optimizer = Adam(self.parameters(), lr=lr, betas=betas)
        scheduler = MultiStepLR(optimizer, milestones=decay_milestones, gamma=decay)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                # Get the data
                x, w_adj, mask = data['x'], data['w_adj'], data['mask']
                # Put the data on the device
                x, w_adj, mask = x.to(self.device), w_adj.to(self.device), mask.to(self.device)
                _, mu, log_var = self.encode(x, w_adj, mask)
                z = reparametrize(mu, log_var, self.device)
                out_x_aa, out_x_ss, out_w_adj, out_mask = self.decode(z)
                losses = self._vae_loss(x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, out_mask, mu, log_var, l_kld)
                loss = losses['Full loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, [optimizer], [scheduler], loss)
            if verbose:
                progress = 'epochs: ' + str(epoch+1) + '/' + str(n_epochs) + ', '
                for key in losses:
                    progress += key + ': ' + str(losses[key].item()) + ', '
                print(progress)

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
        # Put onto device
        x, w_adj, mask = x.to(self.device), w_adj.to(self.device), mask.to(self.device)
        # Forward pass
        _, mu, log_var = self.encode(x, w_adj, mask)
        z = reparametrize(mu, log_var, self.device)
        out_x_aa, out_x_ss, out_w_adj, out_mask = self.decode(z)
        # Compute the loss
        losses = self._vae_loss(x, w_adj, mask, out_x_aa, out_x_ss, out_w_adj, out_mask, mu, log_var, l_kld)
        return {'Loss': losses['Full loss']}

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
        gen_x_aa, gen_x_ss, gen_w_adj, gen_mask = self.decode(z)
        # Softmax on classes
        gen_x_aa = torch.softmax(gen_x_aa, dim=2)
        gen_x_ss = torch.softmax(gen_x_ss, dim=2)
        gen_mask = torch.softmax(gen_mask, dim=2)
        # Reshape and reformat adjacency
        gen_w_adj = gen_w_adj.view(gen_w_adj.shape[1],gen_w_adj.shape[2])
        # gen_w_adj = gen_w_adj - torch.triu(gen_w_adj)
        # gen_w_adj = gen_w_adj + torch.transpose(gen_w_adj, 0,1)
        # Get the nodes to generate and define the total number of nodes
        nodes, N = [0 for _ in range(self.max_size)], 0
        for i in range(self.max_size):
            nodes[i] = torch.argmax(gen_mask[0,i])
            N += nodes[i]
        if verbose:
            print(f'The generated graph has {N} nodes.')
        # Define data to be returned
        x_pred, dist_pred = torch.zeros(N,self.X_DIM).to(self.device), torch.zeros(N,N).to(self.device)
        # Fill the node prediction
        idx = 0
        for i in range(self.max_size):
            if nodes[i] == 1:
                x_pred[idx,0] = torch.argmax(gen_x_aa[0,i,1:])+1
                x_pred[idx,1] = torch.argmax(gen_x_ss[0,i,1:])+1
                idx += 1
        # Fill the distance matrix prediction
        idx_i = 0
        for i in range(self.max_size):
            if nodes[i] == 1:
                idx_j = 0
                for j in range(self.max_size):
                    if nodes[j] == 1:
                        if i != j:
                            dist_pred[idx_i,idx_j] = dist_pred[idx_j,idx_i] = min(1/gen_w_adj[i,j], 12)
                        idx_j += 1
                idx_i += 1
        return x_pred.long(), dist_pred.float()