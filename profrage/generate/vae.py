import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from generate.layers import MLPLayer, ECCLayer
from generate.utils import reparametrize, node_feature_target_classes, edge_features_input, edge_features_target

class GraphVAE(nn.Module):
    """
    A VAE model to generate a graph.

    It takes as input a graph in a sparse format: this means that the number of nodes is fixed, and thus it does not
    compute the probability of nodes being in the graph.

    The encoder is a EC graph convolutional layer, which results in a graph embedding based on its node features,
    adjacency, and edge features.
    The decoder consists of two MLPs: a node feature MLP and an adjacency/edge MLP, which itself is divided
    into a classifier and regressor.
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
            The dimension of the latent space.
        ecc_dims : list of int
            The dimensions for the ECC layers.
        mlp_dims : list of int
            The dimensions for the MLP layers.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1.
        x_class_dim : int, optional
            The number of node features which represent classes. The default is 2.
        edge_class_dim : int, optional
            The number of edge features which represent classes. The default is 3.
        max_size : int, optional
            The maximum number of amino acids in a protein. The default is 30.
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structures. The default is 7.
        ignore_idx : int, optional
            The classes to ignore in the cross-entropy loss. The default is -100.
        weight_init : float, optional
            The weight initialization bounds. The default is 5e-5.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(GraphVAE, self).__init__()
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
        # Decoding
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

    def _vae_loss(self, x, adj, edge, dec_x, dec_edge, mu, log_var, x_len, edge_len, l_kld):
        # Get target classes
        x_classes = node_feature_target_classes(x, self.device)
        adj_edge_classes = edge_features_target(adj, edge, x_len, edge_len, self.max_size, self.edge_dim, self.device)
        # Get input classes
        adj_edge_input = edge_features_input(dec_edge, x_len, self.max_size, self.edge_dim, self.edge_class_dim, self.device)
        # Get lower triangular indexes
        tril_idx = torch.tril_indices(adj_edge_input.shape[1],adj_edge_input.shape[1])
        # Node classification and regression
        ce_loss_x = F.cross_entropy(dec_x[:,0:self.aa_dim*self.ss_dim], x_classes)
        mse_loss_x_reg = F.mse_loss(dec_x[:,self.aa_dim*self.ss_dim:], x[:,self.x_class_dim:])
        # Adjacency/edge classification
        ce_loss_adj_edge = F.cross_entropy(torch.transpose(adj_edge_input[:,:,:,0:self.edge_class_dim], 1,3), adj_edge_classes[:,:,:,1].long(), reduction='none')
        ce_loss_adj_edge = ce_loss_adj_edge[:,tril_idx]
        ce_loss_adj_edge = torch.mean(ce_loss_adj_edge)
        # Edge regression
        mse_loss_edge = F.mse_loss(adj_edge_input[:,:,:,self.edge_class_dim:].squeeze(3), adj_edge_classes[:,:,:,0], reduction='none')
        mse_loss_edge = mse_loss_edge[:,tril_idx]
        mse_loss_edge = torch.mean(mse_loss_edge)
        # Beta parameters for stability
        beta_x = torch.abs(ce_loss_x.detach()/mse_loss_x_reg.detach()).detach()
        beta_adj_edge = torch.abs(ce_loss_adj_edge.detach()/mse_loss_edge.detach()).detach()
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return ce_loss_x + beta_x*mse_loss_x_reg + ce_loss_adj_edge + beta_adj_edge*mse_loss_edge + l_kld*kl_loss

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
        out = F.relu_(out)
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

    def fit(self, loader, n_epochs, lr=1e-3, l_kld=1e-3, checkpoint=500, verbose=False):
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
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print loss information. The default is False.

        Returns
        -------
        None
        """
        optimizer = Adam(self.parameters(), lr=lr) # TODO try to optmize them separately
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                # Get the data
                x, adj, edge, x_len, edge_len = data.x, data.edge_index, data.edge_attr, data.x_len, data.edge_len
                # Put the data on the device
                x, adj, edge = x.to(self.device), adj.to(self.device), edge.to(self.device)
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x, adj, edge)
                z = reparametrize(mu, log_var, self.device)
                out_x, out_edge = self.decode(z)
                loss = self._vae_loss(x, adj, edge, out_x, out_edge, mu, log_var, x_len, edge_len, l_kld)
                loss.backward()
                optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, optimizer, loss)
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x, adj, edge, x_len, edge_len, l_kld):
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
        l_kld : float in [0,1]
            The penalty to apply to the Kullback-Leibler loss, for stability reasons.

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
        loss = self._vae_loss(x, adj, edge, out_x, out_edge, mu, log_var, x_len, edge_len, l_kld)
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
        z = torch.randn((max_num_nodes,self.latent_dim))
        gen_x, gen_edge = self.decode(z)
        # Softmax on classes
        gen_x[:,0:self.aa_dim*self.ss_dim] = torch.softmax(gen_x[:,0:self.aa_dim*self.ss_dim], dim=1)
        gen_edge[:,0:self.edge_class_dim] = torch.softmax(gen_edge[:,0:self.edge_class_dim], dim=1)
        # Define data to be returned
        x_pred, adj_edge_pred = torch.zeros(max_num_nodes,self.x_dim), torch.zeros(max_num_nodes,max_num_nodes,self.edge_dim)
        # Refine node predictions
        for i in range(x_pred.shape[0]):
            idx = torch.argmax(gen_x[i,0:self.aa_dim*self.ss_dim])
            a_tensor, s_tensor = idx - (idx//self.aa_dim)*self.aa_dim + 1, idx//self.aa_dim + 1
            a, s = int(a_tensor.item()), int(s_tensor.item())
            x_pred[i,0:2] = torch.LongTensor([a, s])
            x_pred[i,2:] = gen_x[i,self.aa_dim*self.ss_dim:]
        # Refine adjacency prediction
        for i in range(adj_edge_pred.shape[0]):
            idx = torch.argmax(gen_edge[i,0:i+1,0:self.edge_class_dim], dim=1)
            adj_edge_pred[i,0:i+1,0] = idx
            adj_edge_pred[i,0:i+1,1] = gen_edge[i,0:i+1,self.edge_class_dim]
        return x_pred, adj_edge_pred

class GraphVAE_Seq(nn.Module):
    """
    A VAE model to generate the node features belonging to a graph. Such node features should be sequential.
    Similarly to the `GraphVAE` model, the number of nodes is fixed.

    The encoder consists in a EC graph convolutional layer, much like the `GraphVAE` model.
    The decoder however only has one MLP for the node features.
    """

    def __init__(self, root, x_dim, edge_dim, hidden_dim, latent_dim, ecc_dims, mlp_dims, dropout=0.1,
                 x_class_dim=2, aa_dim=20, ss_dim=7, ignore_idx=-100, weight_init=5e-5, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            Where to save the data.
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
            The dropout probability. The default is 0.1.
        x_class_dim : int, optional
            The number of node features which represent classes. The default is 2.
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structures. The default is 7.
        ignore_idx : int, optional
            The classes to ignore in the cross-entropy loss. The default is -100.
        weight_init : float, optional
            The weight initialization bounds. The default is 5e-5.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(GraphVAE_Seq, self).__init__()
        self.root = root
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.x_class_dim = x_class_dim
        self.ignore_idx = ignore_idx
        self.weight_init = weight_init
        self.device = device

        # Encoding
        self.enc = ECCLayer([x_dim] + ecc_dims + [hidden_dim], [8,16,8], edge_dim)
        # Sampling
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoding
        self.dec_x = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        # Output
        self.fc_out_x = nn.Linear(hidden_dim,aa_dim*ss_dim+x_dim-x_class_dim)

        # Weights initialization
        self._init_weights(self.latent_mu)
        self._init_weights(self.latent_log_var)
        self.fc_out_x.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-self.weight_init, b=self.weight_init)

    def _vae_loss(self, x, dec_x, mu, log_var, l_kld):
        # Get target classes
        x_classes = node_feature_target_classes(x, self.device)
        # Node classification and regression
        ce_loss_x = F.cross_entropy(dec_x[:,0:self.aa_dim*self.ss_dim], x_classes)
        mse_loss_x = F.mse_loss(dec_x[:,self.aa_dim*self.ss_dim:], x[:,self.x_class_dim:])
        # Beta parameter for stability
        beta = torch.abs(ce_loss_x.detach()/mse_loss_x.detach()).detach()
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return ce_loss_x + beta*mse_loss_x + l_kld*kl_loss

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
        out_x : torch.tensor
            The decoded node features.
        """
        dec_x = self.dec_x(z)
        out_x = self.fc_out_x(dec_x)
        return out_x

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

    def fit(self, loader, n_epochs, lr=1e-3, l_kld=1e-3, checkpoint=500, verbose=False):
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
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print loss information. The default is False.

        Returns
        -------
        None
        """
        optimizer = Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                # Get the data
                x, adj, edge = data.x, data.edge_index, data.edge_attr
                # Put the data on the device
                x, adj, edge = x.to(self.device), adj.to(self.device), edge.to(self.device)
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x, adj, edge)
                z = reparametrize(mu, log_var, self.device)
                out_x = self.decode(z)
                loss = self._vae_loss(x, out_x, mu, log_var, l_kld)
                loss.backward()
                optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, optimizer, loss)
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x, adj, edge, l_kld):
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
        l_kld : float
            The penalty to apply to the Kullback-Leibler loss, for stability reasons.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Forward pass
        _, mu, log_var = self.encode(x, adj, edge)
        z = reparametrize(mu, log_var, self.device)
        out_x = self.decode(z)
        # Compute the loss
        loss = self._vae_loss(x, out_x, mu, log_var, l_kld)
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
        x_pred : torch.tensor
            The predicted node features.
        """
        # Get the generated data
        z = torch.randn((max_num_nodes,self.latent_dim)).to(self.device)
        gen_x = self.decode(z)
        # Softmax on classes
        gen_x[:,0:self.aa_dim*self.ss_dim] = torch.softmax(gen_x[:,0:self.aa_dim*self.ss_dim], dim=1)
        # Refine the generated data
        x_pred = torch.zeros(max_num_nodes,self.x_dim)
        for i in range(x_pred.shape[0]):
            idx = torch.argmax(gen_x[i,0:self.aa_dim*self.ss_dim])
            a_tensor, s_tensor = idx - (idx//self.aa_dim)*self.aa_dim + 1, idx//self.aa_dim
            a, s = int(a_tensor.item()), int(s_tensor.item())
            x_pred[i,0:2] = torch.LongTensor([a, s])
            x_pred[i,2:] = gen_x[i,self.aa_dim*self.ss_dim:]
        return x_pred