import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.optim import Adam

from generate.layers import MLPLayer, CNN1Layer

class GraphVAE(nn.Module):

    def __init__(self, node_dim, hidden_dim, latent_dim, enc_dims, x_mlp_dims, adj_mlp_dims, adj_cnn_dims=None,
                 kernel_size=2, dropout=0.1, max_size=30, aa_dim=20, ss_dim=7, atom_dim=5, l_kld=1e-3, ignore_idx=-100,
                 use_cnn=False, device='cpu'):
        super(GraphVAE, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.max_size = max_size
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.atom_dim = atom_dim
        self.l_kld = l_kld
        self.ignore_idx = ignore_idx
        self.device = device

        self.enc_sage = gnn.DenseSAGEConv(node_dim, hidden_dim)
        self.enc_dropout = nn.Dropout(dropout)
        self.enc_mlp = MLPLayer([hidden_dim] + enc_dims + [hidden_dim])
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        self.dec_mlp_x = MLPLayer([latent_dim] + x_mlp_dims + [hidden_dim])
        self.dec_mlp_x_atomic = MLPLayer([latent_dim] + x_mlp_dims + [hidden_dim])
        if use_cnn:
            self.dec_adj = CNN1Layer([max_size] + adj_cnn_dims + [max_size], kernel_size)
        else:
            self.dec_adj = MLPLayer([latent_dim] + adj_mlp_dims + [hidden_dim])
        self.dec_dropout_x = nn.Dropout(dropout)
        self.dec_dropout_x_atomic = nn.Dropout(dropout)
        self.dec_dropout_adj = nn.Dropout(dropout)
        self.fc_out_x = nn.Linear(hidden_dim,aa_dim*ss_dim)
        self.fc_out_x_atomic = nn.Linear(hidden_dim,atom_dim)
        if use_cnn:
            self.fc_out_adj = nn.Linear(self.gen_adj.out_dim(latent_dim),2*max_size)
        else:
            self.fc_out_adj = nn.Linear(hidden_dim,2*max_size)
        # Weights initialization
        self.enc_mlp.apply(self._init_weights)
        self.dec_mlp_x.apply(self._init_weights)
        self.dec_adj.apply(self._init_weights)
        self._init_weights(self.latent_mu, mode='unif')
        self._init_weights(self.latent_log_var, mode='unif')

    def _init_weights(self, m, mode='he'):
        if isinstance(m, nn.Linear):
            if mode == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif mode == 'unif':
                nn.init.uniform_(m.weight, a=-0.005, b=0.005)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def _binary_adjacency(self, adj, node_mask):
        adj_bin = torch.zeros_like(adj)
        num_batch, n = adj.shape[0], adj.shape[1]
        for b in range(num_batch):
            for i in range(n):
                if node_mask[b,i]:
                    for j in range(n):
                        if i == j and node_mask[b,j]:
                            adj_bin[b,i,j] = 1 # aka probability of one for nodes (i,j) to exist
                        elif node_mask[b,j]:
                            if adj[b,i,j] > 0:
                                adj_bin[b,i,j] = 1 # contact
        return adj_bin

    def _input_classes(self, x):
        b_dim, n = x.shape[0], x.shape[1]
        x_classes = torch.zeros(b_dim,n, dtype=torch.long)
        for b in range(b_dim):
            for i in range(n):
                a, s = x[b,i,0] - 1, x[b,i,1] - 1 # subtract one because classes begin with 1
                mapping = s*self.aa_dim + a
                if mapping < 0:
                    x_classes[b,i] = self.ignore_idx
                else:
                    x_classes[b,i] = mapping
        return x_classes

    def _reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.rand_like(sigma)
        z = mu + epsilon*sigma
        return z

    def _vae_loss(self, x, adj, node_mask, dec_x, dec_x_atom, dec_adj, mu, log_var):
        x_classes = self._input_classes(x)
        adj_bin = self._binary_adjacency(adj, node_mask)
        ce_x, ce_adj = nn.CrossEntropyLoss(reduction='none'), nn.CrossEntropyLoss(reduction='none')
        mse_x_atom = nn.MSELoss(reduction='none')
        # Node classification and atomic regression
        ce_loss_x = ce_x(torch.transpose(dec_x, 1,2), x_classes) # transpose to comply with cross entropy loss function
        ce_loss_x[~node_mask] = 0 # ignore padding
        ce_loss_x = torch.mean(ce_loss_x)
        mse_loss_x_atom = mse_x_atom(dec_x_atom, x[:,:,self.atom_dim:])
        mse_loss_x_atom[~node_mask] = 0 # ignore padding
        mse_loss_x_atom = torch.mean(mse_loss_x_atom)
        # Adjacency probability
        ce_loss_adj = ce_adj(torch.transpose(dec_adj, 1,3), adj_bin.type(torch.LongTensor))
        ce_loss_adj[~node_mask] = 0 # ignore padding
        ce_loss_adj = torch.mean(ce_loss_adj)
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Balance the losses
        beta = torch.abs((ce_loss_x.detach()+ce_loss_adj.detach())/mse_loss_x_atom.detach()).detach()
        return ce_loss_x + beta*mse_loss_x_atom + ce_loss_adj + self.l_kld*kl_loss

    def encode(self, x, adj, node_mask):
        out = self.enc_sage(x, adj, mask=node_mask)
        out = F.relu(out)
        out = self.enc_dropout(out)
        out = self.enc_mlp(out)
        mu, log_var = self.latent_mu(out), self.latent_log_var(out)
        return out, mu, log_var

    def decode(self, z):
        dec_x, dec_x_atom, dec_adj = self.dec_mlp_x(z), self.dec_mlp_x_atomic(z), self.dec_adj(z)
        dec_x, dex_x_atom, dec_adj = self.dec_dropout_x(dec_x), self.dec_dropout_x_atomic(dec_x_atom), self.dec_dropout_adj(dec_adj)
        dec_x_class, dec_x_atom = self.fc_out_x(dec_x), self.fc_out_x_atomic(dex_x_atom)
        dec_adj = self.fc_out_adj(dec_adj)
        dec_adj = dec_adj.view(dec_adj.shape[0],self.max_size,self.max_size,2)
        return dec_x_class, dec_x_atom, dec_adj

    def train(self, loader, n_epochs, lr=1e-3, verbose=False):
        optimizer = Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                x, adj, edge, node_mask = data['x'], data['adj'], data['edge'], data['mask']
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x, adj, node_mask)
                z = self._reparametrize(mu, log_var)
                out_x, out_x_atom, out_adj = self.decode(z)
                loss = self._vae_loss(x, adj, node_mask, out_x, out_x_atom, out_adj, mu, log_var)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval(self, verbose=False):
        # Get the generated data
        gaussian = torch.distributions.Normal(torch.zeros(self.max_size,self.latent_dim), torch.ones(self.max_size,self.latent_dim))
        z = gaussian.sample()
        gen_x, gen_x_atom, gen_adj = self.decode(z.unsqueeze(0))
        gen_adj = F.softmax(gen_adj, dim=3)
        # Refine the generated data
        exists, n_exist = [0]*self.max_size, 0
        for i in range(gen_adj.shape[1]):
            if torch.argmax(gen_adj[0,i,i]) == 1:
                exists[i] = True
                n_exist += 1
        if verbose:
            print(f'The generated graph has {n_exist} nodes.')
        x_pred, adj_pred = torch.zeros(n_exist,2), torch.zeros(n_exist,n_exist)
        x_pred_atom = gen_x_atom
        i_pred = 0
        for i in range(gen_x.shape[1]):
            if exists[i]:
                idx = torch.argmax(gen_x[0,i,:])
                a_tensor, s_tensor = idx - (idx//self.aa_dim)*self.aa_dim + 1, idx//self.aa_dim
                a, s = int(a_tensor.item()), int(s_tensor.item())
                x_pred[i_pred] = torch.LongTensor([a, s])
                i_pred += 1
        i_pred = 0
        for i in range(gen_adj.shape[1]):
            if exists[i]:
                j_pred = 0
                for j in range(gen_adj.shape[1]):
                    if exists[j]:
                        adj_pred[i_pred,j_pred] = torch.argmax(gen_adj[0,i,j])
                        j_pred += 1
                i_pred += 1
        return x_pred, x_pred_atom, adj_pred

class GraphVAEEdge(GraphVAE):

    def __init__(self, node_dim, edge_dim, hidden_dim, latent_dim, enc_dims, x_mlp_dims, adj_mlp_dims, edge_mlp_dims, adj_cnn_dims=None,
                 kernel_size=2, dropout=0.1, max_size=30, aa_dim=20, ss_dim=7, atom_dim=5, l_kld=1e-3, ignore_idx=-100,
                 use_cnn=False, device='cpu'):
        super(GraphVAEEdge, self).__init__(node_dim, hidden_dim, latent_dim, enc_dims, x_mlp_dims, adj_mlp_dims,
                                           adj_cnn_dims=adj_cnn_dims, kernel_size=kernel_size, dropout=dropout, max_size=max_size,
                                           aa_dim=aa_dim, ss_dim=ss_dim, atom_dim=atom_dim, l_kld=l_kld, ignore_idx=ignore_idx,
                                           use_cnn=use_cnn, device=device)
        self.edge_dim = edge_dim
        self.dec_mlp_edge = MLPLayer([latent_dim] + edge_mlp_dims + [hidden_dim])
        self.dec_dropout_edge = nn.Dropout(dropout)
        self.fc_out_edge = nn.Linear(hidden_dim,edge_dim*max_size)
        # Weights initialization
        self.dec_mlp_edge.apply(self._init_weights)

    def _init_weights(self, m, mode='he'):
        if isinstance(m, nn.Linear):
            if mode == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif mode == 'unif':
                nn.init.uniform_(m.weight, a=-0.005, b=0.005)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def _edge_clases(self, edge, node_mask):
        edge_classes = torch.zeros(edge.shape[0],edge.shape[1],edge.shape[2])
        num_batch, n = edge.shape[0], edge.shape[1]
        for b in range(num_batch):
            for i in range(n):
                if node_mask[b,i]:
                    for j in range(n):
                        if node_mask[b,j]:
                            edge_classes[b,i,j] = edge[b,i,j,1]
                        else:
                            edge_classes[b,i,j] = self.ignore_idx
                else:
                    for j in range(n):
                        edge_classes[b,i,j] = self.ignore_idx
        return edge_classes

    def _vae_loss(self, x, adj, edge, node_mask, dec_x, dec_x_atom, dec_adj, dec_edge, mu, log_var):
        x_classes = self._input_classes(x)
        adj_bin = self._binary_adjacency(adj, node_mask)
        edge_classes = self._edge_clases(edge, node_mask)
        ce_x, ce_adj, ce_edge = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
        mse_x_atom = nn.MSELoss()
        # Node classification and atomic regression
        ce_loss_x = ce_x(torch.transpose(dec_x, 1,2), x_classes) # transpose to comply with cross entropy loss function
        mse_loss_x_atom = mse_x_atom(dec_x_atom, x[:,:,self.atom_dim:])
        # Adjacency probability
        ce_loss_adj = ce_adj(torch.transpose(dec_adj, 1,3), adj_bin.type(torch.LongTensor))
        # Edge classification loss
        ce_loss_edge = ce_edge(torch.transpose(dec_edge, 1,3), edge_classes.type(torch.LongTensor))
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Balance the loss
        beta = torch.abs((ce_loss_x.detach()+ce_loss_edge.detach())/mse_loss_x_atom.detach()).detach()
        return ce_loss_x + beta*mse_loss_x_atom + ce_loss_adj + ce_loss_edge + self.l_kld*kl_loss

    def decode(self, z):
        dec_x_class, dec_x_atom, dec_adj = super().decode(z)
        dec_edge = self.dec_mlp_edge(z)
        dec_edge = self.dec_dropout_edge(dec_edge)
        dec_edge = self.fc_out_edge(dec_edge)
        dec_edge = dec_edge.view(dec_edge.shape[0],self.max_size,self.max_size,self.edge_dim)
        return dec_x_class, dec_x_atom, dec_adj, dec_edge

    def train(self, loader, n_epochs, lr=1e-3, verbose=False):
        optimizer = Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                optimizer.zero_grad()
                x, adj, edge, node_mask = data['x'], data['bb'], data['edge'], data['mask']
                _, mu, log_var = self.encode(x, adj, node_mask)
                z = self._reparametrize(mu, log_var)
                out_x, out_x_atom, out_adj, out_edge = self.decode(z)
                loss = self._vae_loss(x, adj, edge, node_mask, out_x, out_x_atom, out_adj, out_edge, mu, log_var)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval(self, verbose=False):
        # Get the generated data
        gaussian = torch.distributions.Normal(torch.zeros(self.max_size,self.latent_dim), torch.ones(self.max_size,self.latent_dim))
        z = gaussian.sample()
        gen_x, gen_x_atom, gen_adj, gen_edge = self.decode(z.unsqueeze(0))
        gen_adj, gen_edge = F.softmax(gen_adj, dim=3), F.softmax(gen_edge, dim=3)
        # Refine the generated data
        exists, n_exist = [0]*self.max_size, 0
        for i in range(gen_adj.shape[1]):
            if torch.argmax(gen_adj[0,i,i]) == 1:
                exists[i] = True
                n_exist += 1
        if verbose:
            print(f'The generated graph has {n_exist} nodes.')
        x_pred, adj_pred = torch.zeros(n_exist,2), torch.zeros(n_exist,n_exist)
        x_pred_atom = gen_x_atom
        i_pred = 0
        for i in range(gen_x.shape[1]):
            if exists[i]:
                idx = torch.argmax(gen_x[0,i,:])
                a_tensor, s_tensor = idx - (idx//self.aa_dim)*self.aa_dim + 1, idx//self.aa_dim
                a, s = int(a_tensor.item()), int(s_tensor.item())
                x_pred[i_pred] = torch.LongTensor([a, s])
                i_pred += 1
        i_pred = 0
        for i in range(gen_adj.shape[1]):
            if exists[i]:
                j_pred = 0
                for j in range(gen_adj.shape[1]):
                    if exists[j]:
                        ee = torch.argmax(gen_adj[0,i,j])
                        if ee == 1:
                            et = torch.argmax(gen_edge[0,i,j])
                            adj_pred[i_pred,j_pred] = ee + et
                            j_pred += 1
                        else:
                            adj_pred[i_pred,j_pred] = ee
                            j_pred += 1
                i_pred += 1
        return x_pred, x_pred_atom, adj_pred