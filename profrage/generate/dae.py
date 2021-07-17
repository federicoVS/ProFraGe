import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.optim import Adam

from generate.layers import MLPLayer, CNN1Layer

class GraphDAE(nn.Module):

    def __init__(self, node_dim, hidden_dim, latent_dim, mlp_dims, cnn_dims,
                 kernel_size=2, dropout=0.1, max_size=30, aa_dim=20, ss_dim=7, atom_dim=5, l_kld=1e-3, ignore_idx=-100, device='cpu'):
        super(GraphDAE, self).__init__()
        self.node_dim = node_dim
        self.latent_dim = latent_dim
        self.max_size = max_size
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.atom_dim = atom_dim
        self.l_kld = l_kld
        self.ignore_idx = ignore_idx
        self.device = device

        self.enc_sage = gnn.DenseSAGEConv(node_dim, hidden_dim)
        self.enc_dropout = nn.Dropout(dropout)
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)
        self.gen_x = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        self.gen_x_atomic = MLPLayer([latent_dim] + mlp_dims + [hidden_dim])
        self.gen_adj = CNN1Layer([max_size] + cnn_dims + [max_size], kernel_size)
        self.gen_dropout_x = nn.Dropout(dropout)
        self.gen_dropout_x_atomic = nn.Dropout(dropout)
        self.gen_dropout_adj = nn.Dropout(dropout)
        self.fc_out_x = nn.Linear(hidden_dim,aa_dim*ss_dim)
        self.fc_out_x_atomic = nn.Linear(hidden_dim,atom_dim)
        self.fc_out_adj = nn.Linear(self.gen_adj.out_dim(latent_dim),max_size)
        # Weights initialization
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

    def _cnn_out_dim(self, m):
        l_in = self.latent_dim
        for child in m.children():
            if isinstance(child, nn.Conv1d) or isinstance(child, nn.Conv2d):
                ks, s, p, d = child.kernel_size[0], child.stride[0], child.padding[0], child.dilation[0]
                l_in = int((l_in + 2*p - d*(ks - 1) - 1)/s + 1)
        return l_in

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

    def _noise_x(self, x, p=0.9):
        x_noised = x.clone()
        n = x.shape[1]
        noised = torch.rand(n) > p
        means, stds = x.mean(1).mean(0), x.std(1).std(0) # mean, std across all batches
        x_noised[:,noised,0] = torch.FloatTensor([torch.randint(self.aa_dim, (1,1))[0,0] + 1])
        x_noised[:,noised,1] = torch.FloatTensor([torch.randint(self.ss_dim, (1,1))[0,0] + 1])
        x_noised[:,noised,1:] = torch.normal(mean=means[1:], std=stds[1:])
        return x_noised

    def _noise_adj(self, adj, p=0.9):
        adj_noised = adj.clone()
        n = adj.shape[1]
        noised = torch.rand(n,n) > p
        adj_noised[:,noised] = torch.FloatTensor([torch.randint(2, (1,1))[0,0]])
        return adj_noised

    def _vae_loss(self, x, adj, node_mask, dec_x, dec_x_atom, dec_adj, mu, log_var):
        x_classes = self._input_classes(x)
        adj_bin = self._binary_adjacency(adj, node_mask)
        ce_x, bce_adj = nn.CrossEntropyLoss(reduction='none'), nn.BCEWithLogitsLoss(reduction='none')
        mse_x_atom = nn.MSELoss(reduction='none')
        # Node classification and atomic regression
        ce_loss_x = ce_x(torch.transpose(dec_x, 1,2), x_classes) # transpose to comply with cross entropy loss function
        # ce_loss_x[~node_mask] = 0 # ignore padded indexes
        ce_loss_x = torch.mean(ce_loss_x)
        mse_loss_x_atom = mse_x_atom(dec_x_atom, x[:,:,self.atom_dim:])
        # mse_loss_x_atom[~node_mask] = 0 # ignore padded indexes
        mse_loss_x_atom = torch.mean(mse_loss_x_atom)
        # Adjacency classification
        bce_loss_adj = bce_adj(dec_adj, adj_bin)
        # bce_loss_adj[~node_mask] = 0 # ignore padded indexes
        bce_loss_adj = torch.mean(bce_loss_adj)
        # Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Balance the losses
        beta = torch.abs((ce_loss_x.detach()+bce_loss_adj.detach())/mse_loss_x_atom.detach()).detach()
        return ce_loss_x + beta*mse_loss_x_atom + bce_loss_adj + self.l_kld*kl_loss

    def encode(self, x, adj, node_mask):
        x_noised, adj_noised = self._noise_x(x), self._noise_adj(adj)
        out = self.enc_sage(x_noised, adj_noised, mask=node_mask)
        out = F.relu(out)
        out = self.enc_dropout(out)
        mu, log_var = self.latent_mu(out), self.latent_log_var(out)
        return out, mu, log_var

    def decode(self, z):
        out_x, out_x_atom, out_adj = self.gen_x(z), self.gen_x_atomic(z), self.gen_adj(z, activation=nn.ReLU())
        out_x, out_x_atom, out_adj = self.gen_dropout_x(out_x), self.gen_dropout_x_atomic(out_x_atom), self.gen_dropout_adj(out_adj)
        out_x_class, out_x_atom, out_adj = self.fc_out_x(out_x), self.fc_out_x_atomic(out_x), self.fc_out_adj(out_adj)
        out_adj = out_adj.view(out_adj.shape[0],self.max_size,self.max_size)
        return out_x_class, out_x_atom, out_adj

    def train(self, loader, n_epochs, lr=1e-3, verbose=False):
        optimizer = Adam(self.parameters(), lr=lr) # TODO try to optmize them separately
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                x, adj, node_mask = data['x'], data['adj'], data['mask']
                optimizer.zero_grad()
                _, mu, log_var = self.encode(x, adj, node_mask)
                z = self._reparametrize(mu, log_var)
                out_x_class, out_x_atom, out_adj = self.decode(z)
                loss = self._vae_loss(x, adj, node_mask, out_x_class, out_x_atom, out_adj, mu, log_var)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

    def eval(self, verbose=False):
        # Get the generated data
        gaussian = torch.distributions.Normal(torch.zeros(self.max_size,self.latent_dim), torch.ones(self.max_size,self.latent_dim))
        z = gaussian.sample()
        gen_x, gen_x_atom, gen_adj = self.decode(z.unsqueeze(0))
        gen_adj = torch.sigmoid(gen_adj)
        # Refine the generated data
        exists, n_exist = [0]*self.max_size, 0
        for i in range(gen_adj.shape[1]):
            if gen_adj[0,i,i] > 0.5:
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
                        if gen_adj[0,i,j] > 0.5:
                            adj_pred[i_pred,j_pred] = 1
                            j_pred += 1
                i_pred += 1
        return x_pred, x_pred_atom, adj_pred


