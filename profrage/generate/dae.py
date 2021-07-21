import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.optim import Adam

from generate.layers import MLPLayer

class GraphDAE(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, latent_dim, mlp_dims, dropout=0.1,
                 x_class_dim=2, edge_class_dim=3, max_size=30, aa_dim=20, ss_dim=7, l_kld=1e-3, ignore_idx=-100, weight_init=5e-5, device='cpu'):
        super(GraphDAE, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.max_size = max_size
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.x_class_dim = x_class_dim
        self.edge_class_dim = edge_class_dim
        self.l_kld = l_kld
        self.ignore_idx = ignore_idx
        self.weight_init = weight_init
        self.device = device

        # Encoding
        self.enc = gnn.ECConv(node_dim, hidden_dim, nn.Sequential(nn.Linear(edge_dim,node_dim*hidden_dim)))
        # Sampling
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoding
        self.dec_x_class = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        self.dec_x_reg = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        self.dec_adj_edge = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        # Output
        self.fc_out_x_class = nn.Linear(hidden_dim,aa_dim*ss_dim)
        self.fc_out_x_reg = nn.Linear(hidden_dim,node_dim-x_class_dim)
        self.fc_out_adj_edge = nn.Linear(hidden_dim,max_size*(edge_dim+edge_class_dim-1))

        # Weights initialization
        self._init_weights(self.latent_mu)
        self._init_weights(self.latent_log_var)
        self.dec_x_class.apply(self._init_weights)
        self.dec_x_reg.apply(self._init_weights)
        self.dec_adj_edge.apply(self._init_weights)
        self.fc_out_x_class.apply(self._init_weights)
        self.fc_out_x_reg.apply(self._init_weights)
        self.fc_out_adj_edge.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-self.weight_init, b=self.weight_init)

    def _input_classes(self, x):
        b_dim, n = x.shape[0], x.shape[1]
        x_classes = torch.zeros(b_dim,1, dtype=torch.long)
        for b in range(b_dim):
            a, s = x[b,0] - 1, x[b,1] - 1 # subtract one because classes begin with 1
            mapping = s*self.aa_dim + a
            if mapping < 0:
                x_classes[b,0] = self.ignore_idx
            else:
                x_classes[b,0] = mapping
        x_classes = x_classes.view(b_dim)
        return x_classes

    def _adjacency_classes(self, adj_sparse, edge_sparse, x_len, edge_len):
        ae_dense = torch.zeros(len(edge_len),self.max_size,self.max_size,self.edge_dim)
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

    def _adjacency_input(self, adj_edge, x_len):
        ae_dense = torch.zeros(len(x_len),self.max_size,self.max_size,self.edge_dim+self.edge_class_dim-1)
        prev = 0
        for i in range(len(x_len)):
            xl = x_len[i]
            ae_dense[i,0:xl,:,:] = adj_edge[prev:prev+xl]
            prev += xl
        return ae_dense

    def _reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.rand_like(sigma)
        z = mu + epsilon*sigma
        return z

    def _noise_x(self, x, p=0.9):
        x_noised = x.clone()
        n = x.shape[0]
        noised = torch.rand(n) > p
        means, stds = x.mean(0), x.std(0) # mean, std across all batches
        x_noised[noised,0] = torch.FloatTensor([torch.randint(self.aa_dim, (1,1))[0,0] + 1])
        x_noised[noised,1] = torch.FloatTensor([torch.randint(self.ss_dim, (1,1))[0,0] + 1])
        x_noised[noised,1:] = torch.normal(mean=means[1:], std=stds[1:])
        return x_noised

    def _vae_loss(self, x, adj, edge, dec_x_class, dec_x_reg, dec_adj_edge, mu, log_var, x_len, edge_len):
        # Get target classes
        x_classes, adj_edge_classes = self._input_classes(x), torch.tril(self._adjacency_classes(adj, edge, x_len, edge_len))
        # Get input classes
        adj_edge_input = torch.tril(self._adjacency_input(dec_adj_edge, x_len))
        # Node classification and regression
        ce_loss_x = F.cross_entropy(dec_x_class, x_classes) # transpose to comply with cross entropy loss function
        mse_loss_x_reg = F.mse_loss(dec_x_reg, x[:,self.x_class_dim:])
        # Adjacency/edge classification
        ce_loss_adj_edge = F.cross_entropy(torch.transpose(adj_edge_input[:,:,:,0:self.edge_class_dim], 1,3), adj_edge_classes[:,:,:,1].long())
        # Edge regression
        mse_loss_edge = F.mse_loss(adj_edge_input[:,:,:,self.edge_class_dim:].squeeze(3), adj_edge_classes[:,:,:,0])
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return ce_loss_x + mse_loss_x_reg + ce_loss_adj_edge + mse_loss_edge + self.l_kld*kl_loss

    def encode(self, x, adj, edge):
        x_noised = self._noise_x(x)
        out = self.enc(x_noised, adj, edge_attr=edge)
        out = F.relu(out)
        mu, log_var = self.latent_mu(out), self.latent_log_var(out)
        return out, mu, log_var

    def decode(self, z):
        dec_x_class, dec_x_reg, dec_adj_edge = self.dec_x_class(z), self.dec_x_reg(z), self.dec_adj_edge(z)
        out_x_class = self.fc_out_x_class(dec_x_class)
        out_x_reg = self.fc_out_x_reg(dec_x_reg)
        out_adj_edge = self.fc_out_adj_edge(dec_adj_edge)
        out_adj_edge = out_adj_edge.view(out_adj_edge.shape[0],self.max_size,self.edge_dim+self.edge_class_dim-1)
        return out_x_class, out_x_reg, out_adj_edge

    def train(self, loader, n_epochs, lr=1e-3, verbose=False):
        optimizer = Adam(self.parameters(), lr=lr) # TODO try to optmize them separately
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                x, adj, edge, x_len, edge_len = data.x, data.edge_index, data.edge_attr, data.x_len, data.edge_len
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x, adj, edge)
                z = self._reparametrize(mu, log_var)
                out_x_class, out_x_reg, out_adj_edge = self.decode(z)
                loss = self._vae_loss(x, adj, edge, out_x_class, out_x_reg, out_adj_edge, mu, log_var, x_len, edge_len)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval(self, max_num_nodes, verbose=False):
        # Get the generated data
        gaussian = torch.distributions.Normal(torch.zeros(max_num_nodes,self.latent_dim), torch.ones(max_num_nodes,self.latent_dim))
        z = gaussian.sample()
        gen_x_class, gen_x_reg, gen_adj_edge = self.decode(z)
        # Softmax on classes
        gen_x_class = torch.softmax(gen_x_class, dim=1)
        gen_adj_edge[:,0:self.edge_class_dim] = torch.softmax(gen_adj_edge[:,0:self.edge_class_dim], dim=1)
        # Refine the generated data
        x_pred, adj_edge_pred = torch.zeros(max_num_nodes,self.node_dim), torch.zeros(max_num_nodes,max_num_nodes,self.edge_dim)
        for i in range(x_pred.shape[0]):
            idx = torch.argmax(gen_x_class[i,:])
            a_tensor, s_tensor = idx - (idx//self.aa_dim)*self.aa_dim + 1, idx//self.aa_dim
            a, s = int(a_tensor.item()), int(s_tensor.item())
            x_pred[i,0:2] = torch.LongTensor([a, s])
            x_pred[i,2:] = gen_x_reg[i,:]
        for i in range(adj_edge_pred.shape[0]):
            idx = torch.argmax(gen_adj_edge[i,0:max_num_nodes,0:self.edge_class_dim], dim=1)
            adj_edge_pred[i,:,0] = idx
            adj_edge_pred[i,:,1] = gen_adj_edge[i,0:max_num_nodes,self.edge_class_dim]
        return x_pred, adj_edge_pred