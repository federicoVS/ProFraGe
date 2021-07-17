import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from generate.layers import GANGenerator, GANDiscriminator

class ProGAN(nn.Module):

    def __init__(self, max_num_nodes, node_dim, edge_dim, z_dim, conv_out_dim, agg_dim, g_mlp_dims, d_mlp_dims, dropout=0.1, device='cpu'):
        super(ProGAN, self).__init__()
        self.max_num_nodes = max_num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.z_dim = z_dim
        self.device = device
        self.generator = GANGenerator(max_num_nodes, node_dim, edge_dim, z_dim, g_mlp_dims, dropout=dropout)
        self.discriminator = GANDiscriminator(node_dim, edge_dim, conv_out_dim, agg_dim, d_mlp_dims, dropout=dropout) # FIXME
        self.reward = GANDiscriminator(node_dim, edge_dim, conv_out_dim, agg_dim, d_mlp_dims, dropout=dropout) # FIXME

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

    def train(self, loader, n_epochs, n_critic=5, w_clip=0.01, l_wrl=0.6, lr_g=5e-5, lr_d=5e-5, lr_r=5e-5, alpha=10, use_gp=False, verbose=False):
        self.g_optimizer = RMSprop(self.generator.parameters(), lr=lr_g)
        self.d_optimizer = RMSprop(self.discriminator.parameters(), lr=lr_d)
        self.r_optimizer = RMSprop(self.reward.parameters(), lr=lr_r)
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                x, adj, edge, batch_len = data['x'], data['adj'], data['edge'], data['len']
                ### 1. Train the discriminator ###
                for _ in range(n_critic):
                    # Sample z
                    z = self._sample_z(x.shape[0])
                    # Compute generated graph
                    x_gen, adj_gen, edge_gen = self.generator(z)
                    # x_tilde, adj_tilde, edge_tilde = F.gumbel_softmax(x_gen), F.gumbel_softmax(adj_gen), F.gumbel_softmax(edge_gen)
                    x_tilde, adj_tilde, edge_tilde = x_gen, adj_gen, edge_gen
                    # Logits from generated data
                    logits_gen = self.discriminator(x_tilde, adj_tilde, edge_tilde, activation=None)
                    # Logits from real data
                    logits_true = self.discriminator(x, adj, edge, activation=None)
                    # Compute loss
                    loss_d = -(torch.mean(logits_true) - torch.mean(logits_gen))
                    # Optimize
                    self.reset_grad()
                    loss_d.backward()
                    self.d_optimizer.step()
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-w_clip, w_clip)
                ### 2. Train the generator ###
                # Sample z
                z = self._sample_z(x.shape[0])
                # Compute generated graph
                x_gen, adj_gen, edge_gen = self.generator(z)
                # x_tilde, adj_tilde, edge_tilde = F.gumbel_softmax(x_gen), F.gumbel_softmax(adj_gen), F.gumbel_softmax(edge_gen)
                x_tilde, adj_tilde, edge_tilde = x_gen, adj_gen, edge_gen
                # Compute logits for real and generated graph
                logits_gen = self.discriminator(x_tilde, adj_tilde, edge_tilde, activation=None)
                # Reward losses
                logits_rew_true = self.reward(x, adj, edge, activation=None)
                logits_rew_gen = self.reward(x_tilde, adj_tilde, edge_tilde, activation=None)
                reward_true = self._reward_fun(adj, edge, batch_mask=batch_len)
                reward_gen = self._reward_fun(adj_tilde, edge_tilde)
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
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss_D={loss_d.item():.4}, loss_G={loss_g.item():.4}, loss_R={loss_r.item()}')

    def eval(self, test_batch_size=1, aa_min=1, aa_max=20, ss_min=0, ss_max=6):
        # Sample z
        z = self._sample_z(test_batch_size)
        # Compute generated graph
        x_gen, adj_gen, edge_gen = self.generator(z)
        # TODO check how to interpret adj matrix (I think it should be in a probabilistic sense but maybe I have forgotten)
        return x_gen, adj_gen, edge_gen