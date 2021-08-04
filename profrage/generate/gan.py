import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import RMSprop

from generate.layers import GANGenerator, GANDiscriminator

class ProGAN(nn.Module):
    """
    Protein graph generation based on the `MolGAN` model.

    Source
    ------
    Paper => MolGAN: An implicit generative model for small molecular graphs
             Nicola De Cao, Thomas Kipf
    Code  => https://github.com/yongqyu/MolGAN-pytorch, https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch
    """

    def __init__(self, root, max_num_nodes, x_dim, edge_dim, z_dim, conv_out_dim, agg_dim, g_mlp_dims, d_mlp_dims, dropout=0.1, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The directory where the model is stored.
        max_num_nodes : int
            The maximum number of nodes.
        x_dim : int
            The dimension of the node features.
        edge_dim : int
            The dimension of the edge features.
        z_dim : int
            The dimension of the sample space.
        conv_out_dim : int
            The dimension of the output of the graph convolutional layer.
        agg_dim : int
            The dimension of the graph aggregation layer.
        g_mlp_dims : list of int
            The dimensions of the MLP layers of the generator.
        d_mlp_dims : list of int
            The dimensions of the MLP layers of the discriminator/reward network.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(ProGAN, self).__init__()
        self.root = root
        self.max_num_nodes = max_num_nodes
        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.z_dim = z_dim
        self.device = device

        self.generator = GANGenerator(max_num_nodes, x_dim, edge_dim, z_dim, g_mlp_dims, dropout=dropout)
        self.discriminator = GANDiscriminator(x_dim, edge_dim, conv_out_dim, agg_dim, d_mlp_dims, dropout=dropout) # FIXME
        self.reward = GANDiscriminator(x_dim, edge_dim, conv_out_dim, agg_dim, d_mlp_dims, dropout=dropout) # FIXME

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.r_optimizer.zero_grad()

    def _sample_z(self, N):
        gaussian = torch.distributions.Normal(torch.zeros(N,self.z_dim), torch.ones(N,self.z_dim))
        z = gaussian.sample()
        return z

    def _gradient_penalty(self, x, adj, edge, x_gen, adj_gen, edge_gen):
        pass

    def _reward_fun(self, adj, edge, batch_mask=None):
        bs, n = adj.shape[0], adj.shape[1]
        rewards = []
        for b in range(bs):
            reward = 0
            if batch_mask is not None:
                n = batch_mask[b]
            if int(adj[b,0,0]) == 0 and int(edge[b,0,0,1]) == 0:
                reward += 1
            if int(adj[b,0,1]) > 0 and int(edge[b,0,1,1]) > 0:
                reward += 1
            for i in range(1,n-1):
                if int(adj[b,i,i]) == 0 and int(edge[b,i-1,i,1]) == 0:
                    reward += 1
                if int(adj[b,i,i-1]) > 0 and int(edge[b,i,i-1,1]) > 0:
                    reward += 1
                if int(adj[b,i,i+1]) > 0 and int(edge[b,i,i+1,1]) > 0:
                    reward += 1
            if int(adj[b,n-1,n-1]) == 0 and int(edge[b,n-1,n-1,1]) == 0:
                reward += 1
            if int(adj[b,n-1,n-2]) > 0 and int(edge[b,n-1,n-2,1]) > 0:
                reward += 1
            rewards.append(reward)
        return torch.FloatTensor(rewards)

    def _dense_to_sparse(self, x_dense, adj_dense, edge_dense, batch_len, target=True):
        bs, n = x_dense.shape[0], x_dense.shape[1]
        if target:
            x_sparse, adj_sparse, edge_sparse = torch.zeros(sum(batch_len),x_dense.shape[2]), [], []
        else:
            x_sparse, adj_sparse, edge_sparse = torch.zeros(bs*n,x_dense.shape[2]), [], []
        prev = 0
        for b in range(bs):
            bl = batch_len[b]
            if target:
                x_sparse[prev:prev+bl,:] = x_dense[b,0:bl,:]
            else:
                x_sparse[prev:prev+n,:] = x_dense[b,:,:]
            for i in range(n-1):
                for j in range(i+1,n):
                    if adj_dense[b,i,j] > 0:
                        adj_sparse.append([i+prev,j+prev])
                        adj_sparse.append([j+prev,i+prev])
                        if edge_dense is not None:
                            l_ij, l_ji = [], []
                            for k in range(edge_dense.shape[3]):
                                l_ij.append(edge_dense[b,i,j,k])
                                l_ji.append(edge_dense[b,j,i,k])
                            edge_sparse.append(l_ij)
                            edge_sparse.append(l_ji)
            if target:
                prev += bl
            else:
                prev += n
        adj_sparse = torch.LongTensor(adj_sparse).t().contiguous()
        edge_sparse = torch.FloatTensor(edge_sparse)
        return x_sparse, adj_sparse, edge_sparse

    def checkpoint(self, epoch, optimizers, losses):
        """
        Create a checkpoint saving the results on the ongoing optimization.

        The checkpoint is saved at ROOT/checkpoint_<epoch>.

        Parameters
        ----------
        epoch : int
            The current epoch.
        optimizers : list of torch.optim.Optimizer
            The optimizers of the model. The first one should be the for the discriminator, the second one for the
            generator, and the third one for the reward network.
        losses : list of float
            The list of losses. The first one should be of the discriminator, the second one of the
            generator, and the third one of the reward network.

        Returns
        -------
        None
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'd_optimizer_state_dict': optimizers[0].state_dict(),
                    'g_optimizer_state_dict': optimizers[1].state_dict(),
                    'r_optimizer_state_dict': optimizers[2].state_dict(),
                    'loss_d': losses[0],
                    'loss_grl': losses[1],
                    'loss_r': losses[2]}, self.root + 'checkpoint_' + str(epoch))

    def fit(self, loader, n_epochs, n_critic=5, w_clip=0.01, l_wrl=0.6, lr_g=5e-5, lr_d=5e-5, lr_r=5e-5, checkpoint=500, verbose=False):
        """
        Train the model.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader or torch_geometric.data.DataLoader
            The data loader.
        n_epochs : int
            The number of epochs.
        n_critic : int, optional
            The number of critical iterations (to train the discriminator).
        w_clip : float, optional
            The clipping to apply to the discriminator weights. The default is 0.01.
        l_wrl : float in [0,1], optional
            The trade-off between the generator loss and the reward network loss. The default is 0.6.
        lr_g : float, optional
            The learning rate of the generator. The default is 5e-5.
        lr_d : float, optional
            The learning rate of the discriminator. The default is 5e-5.
        lr_r : float, optional
            The learning rate of the reward network. the default is 5e-5.
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print the loss. The default is False.

        Returns
        -------
        None
        """
        self.g_optimizer = RMSprop(self.generator.parameters(), lr=lr_g)
        self.d_optimizer = RMSprop(self.discriminator.parameters(), lr=lr_d)
        self.r_optimizer = RMSprop(self.reward.parameters(), lr=lr_r)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                # Get the data
                x, adj, edge, batch_len = data['x'], data['adj'], data['edge'], data['len']
                # Put the data on the device
                x, adj, edge = x.to(self.device), adj.to(self.device), edge.to(self.device)
                # Sparsify the data
                x_sparse, adj_sparse, edge_sparse = self._dense_to_sparse(x, adj, edge, batch_len, target=True)
                ### 1. Train the discriminator ###
                for _ in range(n_critic):
                    # Sample z
                    z = self._sample_z(x.shape[0])
                    # Compute generated graph
                    x_gen, adj_gen, edge_gen = self.generator(z)
                    # Sparsify the generated data
                    x_gen_sparse, adj_gen_sparse, edge_gen_sparse = self._dense_to_sparse(x_gen, adj_gen, edge_gen, batch_len, target=False)
                    # Logits from generated data
                    logits_gen = self.discriminator(x_gen_sparse, adj_gen_sparse, edge_gen_sparse, activation=None)
                    # Logits from real data
                    logits_true = self.discriminator(x_sparse, adj_sparse, edge_sparse, activation=None)
                    # Compute loss
                    loss_d = -(torch.mean(logits_true) - torch.mean(logits_gen))
                    # Optimize
                    self.reset_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), w_clip)
                    self.d_optimizer.step()
                ### 2. Train the generator ###
                # Sample z
                z = self._sample_z(x.shape[0])
                # Compute generated graph
                x_gen, adj_gen, edge_gen = self.generator(z)
                # Sparsify the generated data
                x_gen_sparse, adj_gen_sparse, edge_gen_sparse = self._dense_to_sparse(x_gen, adj_gen, edge_gen, batch_len, target=False)
                # Compute logits for generated graph
                logits_gen = self.discriminator(x_gen_sparse, adj_gen_sparse, edge_gen_sparse, activation=None)
                # Reward losses
                logits_rew_true = self.reward(x_sparse, adj_sparse, edge_sparse, activation=None)
                logits_rew_gen = self.reward(x_gen_sparse, adj_gen_sparse, edge_gen_sparse, activation=None)
                reward_true = self._reward_fun(adj, edge, batch_mask=batch_len)
                reward_gen = self._reward_fun(adj_gen, edge_gen)
                # Define losses
                loss_g = -torch.mean(logits_gen)
                loss_rl = -torch.mean(logits_rew_gen)
                loss_r = torch.mean((logits_rew_true - reward_true)**2 + (logits_rew_gen - reward_gen)**2)
                beta = torch.abs(loss_g.detach()/loss_rl.detach()).detach() # as to make it balanced
                loss_grl = l_wrl*loss_g + (1 - l_wrl)*beta*loss_rl
                # Optimize
                self.reset_grad()
                loss_grl.backward(retain_graph=True)
                loss_r.backward()
                self.g_optimizer.step()
                self.r_optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, [self.d_optimizer, self.g_optimizer, self.r_optimizer], [loss_d, loss_grl, loss_r])
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss_D={loss_d.item():.4}, loss_G={loss_g.item():.4}, loss_R={loss_r.item()}')

    def eval_loss(self, x, adj, edge, x_len, l_wrl):
        """
        Compute the evaluation loss of the model.

        A forward pass is performed, and the loss of the discriminator, generator, and reward network are computed.

        Parameters
        ----------
        x : torch.tensor
            The node features.
        adj : torch.tensor
            The adjacency matrix.
        edge : torch.tensor
            The edge features.
        l_wrl : float in [0,1]
            The trade-off between the generator loss and the reward network loss.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Sparsify the data
        x_sparse, adj_sparse, edge_sparse = self._dense_to_sparse(x, adj, edge, x_len, target=True)
        # Sample z
        z = self._sample_z(1)
        # Compute generated graph
        x_gen, adj_gen, edge_gen = self.generator(z)
        # Sparsify the generated data
        x_gen_sparse, adj_gen_sparse, edge_gen_sparse = self._dense_to_sparse(x_gen, adj_gen, edge_gen, x_len, target=False)
        # Logits from generated data
        logits_gen = self.discriminator(x_gen_sparse, adj_gen_sparse, edge_gen_sparse, activation=None)
        # Logits from real data
        logits_true = self.discriminator(x_sparse, adj_sparse, edge_sparse, activation=None)
        # Logits reward
        logits_rew_true = self.reward(x_sparse, adj_sparse, edge_sparse, activation=None)
        logits_rew_gen = self.reward(x_gen_sparse, adj_gen_sparse, edge_gen_sparse, activation=None)
        reward_true = self._reward_fun(adj, edge, batch_mask=x_len)
        reward_gen = self._reward_fun(adj_gen, edge_gen)
        # Compute the loss
        loss_d = -(torch.mean(logits_true) - torch.mean(logits_gen))
        loss_g = -torch.mean(logits_gen)
        loss_rl = -torch.mean(logits_rew_gen)
        loss_r = torch.mean((logits_rew_true - reward_true)**2 + (logits_rew_gen - reward_gen)**2)
        beta = torch.abs(loss_g.detach()/loss_rl.detach()).detach() # as to make it balanced
        loss_grl = l_wrl*loss_g + (1 - l_wrl)*beta*loss_rl
        return {'Discriminator': loss_d, 'Generator': loss_grl, 'Reward': loss_r}

    def generate(self, test_batch_size=1, aa_min=1, aa_max=20, ss_min=1, ss_max=7, verbose=False):
        """
        Evaluate the model by generating new data.

        Parameters
        ----------
        test_batch_size : int, optional
            The number of samples to generate. The default is 1.
        aa_min : int, optional
            The minimum amino acid code. The default is 1.
        aa_max : int, optional
            The maximum amino acid code. The default is 20.
        ss_min : int, optional
            The minimum secondary structure code. The default is 1.
        ss_max : int, optional
            The maximum secondary structure code. The default is 7.

        Returns
        -------
        (x_gen, adj_gen, edge_gen) : (torch.tensor, torch.tensor, torch.tensor)
            The generated node features, adjacency matrix, and edge features.
        """
        # Sample z
        z = self._sample_z(test_batch_size)
        # Compute generated graph
        x_gen, adj_gen, edge_gen = self.generator(z)
        gen_adj = F.softmax(adj_gen, dim=3)
        # Refine the generated data
        exists, n_exist = [0]*self.max_size, 0
        for i in range(gen_adj.shape[1]):
            if torch.argmax(gen_adj[0,i,i]) == 1:
                exists[i] = True
                n_exist += 1
        if verbose:
            print(f'The generated graph has {n_exist} nodes.')
        x_pred, adj_edge_pred = torch.zeros(n_exist,self.x_dim), torch.zeros(n_exist,n_exist)
        i_pred = 0
        for i in range(x_gen.shape[1]):
            if exists[i]:
                x_pred[i_pred] = x_gen[0,i,:]
                x_pred[i_pred,0] = torch.clip(x_pred[i_pred,0], min=aa_min, max=aa_max)
                x_pred[i_pred,1] = torch.clip(x_pred[i_pred,1], min=ss_min, max=ss_max)
                i_pred += 1
        i_pred = 0
        for i in range(gen_adj.shape[1]):
            if exists[i]:
                j_pred = 0
                for j in range(gen_adj.shape[1]):
                    if exists[j]:
                        adj_edge_pred[i_pred,j_pred] = torch.argmax(gen_adj[0,i,j]) + edge_gen[0,i,j,1]
                        j_pred += 1
                i_pred += 1
        return x_pred, adj_edge_pred