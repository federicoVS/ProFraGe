import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.optim import Adam

from generate.layers import MLPLayer, ECCLayer
from generate.utils import reparametrize, node_feature_target_classes, edge_features_input, edge_features_target, nan_to_num

class GraphDAAE(nn.Module):
    """
    Protein graph generation model based on denoising adversarial autoencoders (DAAEs).

    Source
    ------
    Paper: => Educating Text Autoencoders: Latent Representation Guidance via Denoising
              Tianxiao Shen, Jonas Mueller, Regina Barzilay, Tommi Jaakkola
    Code   => https://github.com/shentianxiao/text-autoencoders
    """

    def __init__(self, root, x_dim, edge_dim, hidden_dim, latent_dim, ecc_dims, mlp_dims, dropout=0.1,
                 x_class_dim=2, edge_class_dim=3, max_size=30, aa_dim=20, ss_dim=7, ignore_idx=-100, weight_init=5e-5, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The directory where to save the model state.
        x_dim : int
            The dimension of the node features.
        edge_dim : int
            The dimension of the edge features.
        hidden_dim : int
            The hidden dimension.
        latent_dim : int
            The latent dimension.
        ecc_dims : list of int
            The dimensions for the ECC layers.
        mlp_dims : list of int
            The dimensions for the MLP layers.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1
        x_class_dim : int, optional
            The number of node features that represent classes. The default is 2.
        edge_class_dim : int, optional
            The number of edge features that represent classes. The default is 3.
        max_size : int, optional
            The maximum number of nodes in a graph. The default is 30.
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structure types. The default is 7.
        ignore_idx : int, optional
            The classes to ignore in the cross-entropy loss. The default is -100.
        weight_init : float, optional
            The weight initialization bounds. The default is 5e-5.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(GraphDAAE, self).__init__()
        self.root = root
        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.max_size = max_size
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.x_class_dim = x_class_dim
        self.edge_class_dim = edge_class_dim
        self.ignore_idx = ignore_idx
        self.weight_init = weight_init
        self.device = device

        # Encoding
        self.enc = ECCLayer([x_dim] + ecc_dims + [hidden_dim], [8,16,8], edge_dim)
        # Sampling
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        # Discriminator
        self.discriminator = MLPLayer([latent_dim] + mlp_dims + [1])
        # Decoder (Generator)
        self.dec_x = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        self.dec_edge = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        # Output
        self.fc_out_x = nn.Linear(hidden_dim,aa_dim*ss_dim+x_dim-x_class_dim)
        self.fc_out_edge = nn.Linear(hidden_dim,max_size*(edge_dim+edge_class_dim-1))

        # Weights initialization
        self._init_weights(self.latent_mu)
        self._init_weights(self.latent_log_var)
        self.fc_out_x.apply(self._init_weights)
        self.fc_out_edge.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-self.weight_init, b=self.weight_init)

    def _noise_x(self, x, p=0.95):
        x_noised = x.clone()
        n = x.shape[0]
        noised = torch.rand(n) > p
        means, stds = x.mean(0), x.std(0) # mean, std across all batches
        x_noised[noised,0] = torch.FloatTensor([torch.randint(self.aa_dim, (1,1))[0,0] + 1])
        x_noised[noised,1] = torch.FloatTensor([torch.randint(self.ss_dim, (1,1))[0,0] + 1])
        x_noised[noised,1:] = torch.normal(mean=means[1:], std=stds[1:])
        return x_noised.to(self.device)

    def _noise_adj_edge(self, adj, edge, edge_len, p=0.95):
        adj_noised, edge_noised, edge_len_noised = adj.clone(), edge.clone(), []
        n = edge.shape[0]
        noised = torch.rand(n) > p
        # Add noise to data tensors
        adj_noised[:,noised], edge_noised[noised,:] = -1, -1
        adj_noised, edge_noised = adj_noised[adj_noised!=-1].view(2,-1), edge_noised[edge_noised!=-1].view(-1,2)
        # Add noise to edge mappings
        offset = 0
        for i in range(len(edge_len)):
            num_edges = 0
            for j in range(offset,offset+edge_len[i]):
                if not noised[j]:
                    num_edges += 1
            edge_len_noised.append(num_edges)
            offset += edge_len[i]
        return adj_noised.to(self.device), edge_noised.to(self.device), edge_len_noised

    def _loss(self, x, adj, edge, out_x, out_edge, z, x_len, edge_len, l_adv):
        # Get target classes
        x_classes  = node_feature_target_classes(x, self.device)
        adj_edge_classes = edge_features_target(adj, edge, x_len, edge_len, self.max_size, self.edge_dim, self.device)
        # Get input classes
        edge_input = edge_features_input(out_edge, x_len, self.max_size, self.edge_dim, self.edge_class_dim, self.device)
        # Get lower triangular indexes
        tril_idx = torch.tril_indices(edge_input.shape[1],edge_input.shape[1])
        # Node classification and regression
        ce_loss_x = F.cross_entropy(out_x[:,0:self.aa_dim*self.ss_dim], x_classes)
        mse_loss_x = F.mse_loss(out_x[:,self.aa_dim*self.ss_dim:], x[:,self.x_class_dim:])
        # Adjacency/edge classification
        ce_loss_edge = F.cross_entropy(torch.transpose(edge_input[:,:,:,0:self.edge_class_dim], 1,3), adj_edge_classes[:,:,:,1].long(), reduction='none')
        ce_loss_edge = ce_loss_edge[:,tril_idx]
        ce_loss_edge = torch.mean(ce_loss_edge)
        # Edge regression
        mse_loss_edge = F.mse_loss(edge_input[:,:,:,self.edge_class_dim:].squeeze(3), adj_edge_classes[:,:,:,0], reduction='none')
        mse_loss_edge = mse_loss_edge[:,tril_idx]
        mse_loss_edge = torch.mean(mse_loss_edge)
        # Beta parameters for stability
        beta_x = torch.abs(ce_loss_x.detach()/mse_loss_x.detach()).detach()
        beta_edge = torch.abs(ce_loss_edge.detach()/mse_loss_edge.detach()).detach()
        # Discriminator loss (https://github.com/shentianxiao/text-autoencoders/blob/master/model.py, lines 155-160)
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=self.device)
        ones = torch.ones(len(z), 1, device=self.device)
        d_z = torch.sigmoid(self.discriminator(z.detach()))
        d_zn = torch.sigmoid(self.discriminator(zn))
        d_loss = F.binary_cross_entropy(d_z, zeros) + F.binary_cross_entropy(d_zn, ones)
        # Return full loss
        return (ce_loss_x + beta_x*mse_loss_x + ce_loss_edge + beta_edge*mse_loss_edge) - l_adv*d_loss

    def encode(self, x, adj, edge):
        """
        Compute the encoding.

        Parameters
        ----------
        x : torch.tensor
            The node features tensor.
        adj : torch.tensor
            The adjacency tensor.
        edge : torch.tensor
            The edge features tensor.

        Returns
        -------
        (out, mu, log_var) : (torch.tensor, torch.tensor, torch.tensor)
            The encoded value, the mean and variance logarithm.
        """
        out = self.enc(x, adj, edge)
        out = F.relu(out)
        mu, log_var = self.latent_mu(out), self.latent_log_var(out)
        mu, log_var = torch.clip(mu, min=-1, max=1), torch.clip(log_var, min=-1, max=1)
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
        (out_x, out_edge) : (torch.tensor, torch.tensor, torch.tensor)
            The decoded node features classes and and adjacency/edge feature matrix.
        """
        # Decode from z
        dec_x, dec_edge = self.dec_x(z), self.dec_edge(z)
        out_x = self.fc_out_x(dec_x)
        out_edge = self.fc_out_edge(dec_edge)
        out_edge = out_edge.view(out_edge.shape[0],self.max_size,self.edge_dim+self.edge_class_dim-1)
        return out_x, out_edge

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

    def fit(self, loader, n_epochs, lr=1e-3, l_adv=1, checkpoint=500, verbose=False):
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
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print loss information. The default is False.

        Returns
        -------

        """
        optimizer = Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                x, adj, edge, x_len, edge_len = data.x, data.edge_index, data.edge_attr, data.x_len, data.edge_len
                # Add noise (and put on the device in the process)
                x_noised = self._noise_x(x)
                adj_noised, edge_noised, edge_len_noised = self._noise_adj_edge(adj, edge, edge_len)
                # Put (original) data on device
                x, adj, edge = x.to(self.device), adj.to(self.device), edge.to(self.device)
                # Start process
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x_noised, adj_noised, edge_noised)
                z = reparametrize(mu, log_var, self.device)
                out_x, out_edge = self.decode(z)
                loss = self._loss(x, adj, edge, out_x, out_edge, z, x_len, edge_len, l_adv)
                loss.backward()
                optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, optimizer, loss)
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x, adj, edge, x_len, edge_len, l_adv):
        """
        Compute the evaluation loss of the model.

        Parameters
        ----------
        x : torch.tensor
            The node features.
        adj : torch.tensor
            The adjacency matrix.
        edge : torch.tensor
            The edge features.
        x_len : list of int
            The number of nodes in each batch.
        edge_len : list of int
            The number of edges in each batch.
        l_adv : float
            The multiplier to apply to the adversarial loss

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Forward pass
        _, mu, log_var = self.encode(x, adj, edge)
        z = reparametrize(mu, log_var, self.device)
        out_x, out_edge = self.decode(z)
        # Compute the loss
        loss = self._loss(x, adj, edge, out_x, out_edge, z, x_len, edge_len, l_adv)
        return {'Loss': loss}

    def generate(self, max_num_nodes):
        """
        Evaluate the model by generating new data.

        Parameters
        ----------
        max_num_nodes : int
            The maximum number of nodes to generate.

        Returns
        -------
        (x_pred, adj_edge_pred) : (torch.tensor, torch.tensor)
            The predicted node features and adjacency/edge features.
        """
        # Get the generated data
        gaussian = torch.distributions.Normal(torch.zeros(max_num_nodes,self.latent_dim), torch.ones(max_num_nodes,self.latent_dim))
        z = gaussian.sample()
        gen_x, gen_edge = self.decode(z)
        # Softmax on classes
        gen_x[:,0:self.aa_dim*self.ss_dim] = torch.softmax(gen_x[:,0:self.aa_dim*self.ss_dim], dim=1)
        gen_edge[:,0:self.edge_class_dim] = torch.softmax(gen_edge[:,0:self.edge_class_dim], dim=1)
        # Refine the generated data
        x_pred, adj_edge_pred = torch.zeros(max_num_nodes,self.x_dim), torch.zeros(max_num_nodes,max_num_nodes,self.edge_dim)
        for i in range(x_pred.shape[0]):
            idx = torch.argmax(gen_x[i,0:self.aa_dim*self.ss_dim])
            a_tensor, s_tensor = idx - (idx//self.aa_dim)*self.aa_dim + 1, idx//self.aa_dim
            a, s = int(a_tensor.item()), int(s_tensor.item())
            x_pred[i,0:2] = torch.LongTensor([a, s])
            x_pred[i,2:] = gen_x[i,self.aa_dim*self.ss_dim:]
        for i in range(adj_edge_pred.shape[0]):
            idx = torch.argmax(gen_edge[i,0:max_num_nodes,0:self.edge_class_dim], dim=1)
            adj_edge_pred[i,:,0] = idx
            adj_edge_pred[i,:,1] = gen_edge[i,0:max_num_nodes,self.edge_class_dim]
        return x_pred, adj_edge_pred