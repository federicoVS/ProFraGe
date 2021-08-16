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

    def __init__(self, root, max_size, z_dim, gcn_dims, agg_dim, g_mlp_dims, dropout=0.1, rl=True, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            The directory where the model is stored.
        max_size : int
            The maximum number of nodes.
        z_dim : int
            The dimension of the sample space.
        agg_dim : int
            The dimension of the graph aggregation layer.
        g_mlp_dims : list of int
            The dimensions of the MLP layers of the generator.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1.
        rl : bool, optional
            Whether to use reinforcement learning. The default is True.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(ProGAN, self).__init__()
        self.root = root
        self.max_size = max_size
        self.z_dim = z_dim
        self.rl = rl
        self.device = device

        self.X_DIM = 2

        self.generator = GANGenerator(max_size, self.X_DIM, z_dim, g_mlp_dims, dropout=dropout)
        self.discriminator = GANDiscriminator(self.X_DIM, gcn_dims, agg_dim, dropout=dropout)
        self.reward = GANDiscriminator(self.X_DIM, gcn_dims, agg_dim, dropout=dropout)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.r_optimizer.zero_grad()

    def _zero_diagonal(self, w_adj):
        diag_idx = torch.eye(self.max_size,self.max_size).bool()
        return w_adj.masked_fill_(diag_idx, 0)

    def _generated_mask_index(self, mask_gen):
        B, N = mask_gen.shape[0], mask_gen.shape[1]
        mask_gen_idx = torch.zeros(B,N).int().to(self.device)
        for b in range(B):
            for n in range(N):
                mask_gen_idx[b,n] = round(mask_gen[b,n].item())
        return mask_gen_idx.bool()

    def _reward_fun(self, w_adj, mask, eps=1e-3):
        B, N = w_adj.shape[0], w_adj.shape[1]
        reward = 0
        for b in range(B):
            seen_zero, beginning = False, True
            for n in range(N):
                if mask[b,n] == 1 and seen_zero:
                    break
                elif mask[b,n] == 0 and beginning:
                    break
                else:
                    reward += 1
                if beginning:
                    beginning = False
                if mask[b,n] == 0:
                    seen_zero = True
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if abs(w_adj[b,i,j]-w_adj[b,j,i]) < eps:
                        reward += 1
        reward /= B*N*N
        return reward

    def _sample_z(self, B):
        gaussian = torch.distributions.Normal(torch.zeros(B,self.max_size,self.z_dim), torch.ones(B,self.max_size,self.z_dim))
        z = gaussian.sample()
        return z

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

    def fit(self, loader, n_epochs, n_critic=5, lr=1e-3, l_wrl=0.6, w_clip=0.01, checkpoint=500, verbose=False):
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
        lr : float, optional
            The learning rate. The default is 1e-3.
        l_wrl : float in [0,1], optional
            The trade-off between the generator loss and the reward network loss. The default is 0.6.
        w_clip : float, optional
            The clipping to apply to the discriminator weights. The default is 0.01.
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print the loss. The default is False.

        Returns
        -------
        None
        """
        # Optimizers
        self.g_optimizer = RMSprop(self.generator.parameters(), lr=lr)
        self.d_optimizer = RMSprop(self.discriminator.parameters(), lr=lr)
        self.r_optimizer = RMSprop(self.reward.parameters(), lr=lr)
        # Cache for reward function
        reward_fun_cache = {}
        for epoch in range(n_epochs):
            for i, data in enumerate(loader):
                # Get the data
                x, w_adj, mask = data['x'], data['w_adj'], data['mask']
                # Put the data on the device
                x, w_adj, mask = x.to(self.device), w_adj.to(self.device), mask.to(self.device)
                ### 1. Train the discriminator ###
                for _ in range(n_critic):
                    # Sample z
                    z = self._sample_z(x.shape[0])
                    # Compute generated graph
                    x_gen, w_adj_gen = self.generator(z)
                    # Logits from generated data
                    logits_gen = self.discriminator(x_gen, self._zero_diagonal(w_adj_gen), self._generated_mask_index(torch.diagonal(w_adj_gen, dim1=1, dim2=2)), activation=None)
                    # Logits from real data
                    logits_true = self.discriminator(x, w_adj, mask, activation=None)
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
                x_gen, w_adj_gen = self.generator(z)
                # Compute logits for generated graph
                logits_gen = self.discriminator(x_gen, self._zero_diagonal(w_adj_gen), self._generated_mask_index(torch.diagonal(w_adj_gen, dim1=1, dim2=2)), activation=None)
                # Check whether RL should be used
                if self.rl:
                    # Reward losses
                    logits_rew_true = self.reward(x, w_adj, mask, activation=torch.sigmoid)
                    logits_rew_gen = self.reward(x_gen, self._zero_diagonal(w_adj_gen), self._generated_mask_index(torch.diagonal(w_adj_gen, dim1=1, dim2=2)), activation=torch.sigmoid)
                    if i not in reward_fun_cache:
                        reward_true = self._reward_fun(w_adj, mask.int())
                        reward_fun_cache[i] = reward_true
                    else:
                        reward_true = reward_fun_cache[i]
                    reward_gen = self._reward_fun(w_adj_gen, self._generated_mask_index(torch.diagonal(w_adj_gen, dim1=1, dim2=2)).int())
                    # Define losses
                    loss_r = torch.mean((logits_rew_true - reward_true)**2 + (logits_rew_gen - reward_gen)**2)
                    loss_grl = -(l_wrl*torch.mean(logits_gen) + (1 - l_wrl)*torch.mean(logits_rew_gen))
                else:
                    loss_grl = -torch.mean(logits_gen)
                # Optimize
                self.reset_grad()
                if self.rl:
                    loss_grl.backward(retain_graph=True)
                    loss_r.backward()
                    self.g_optimizer.step()
                    self.r_optimizer.step()
                else:
                    loss_grl.backward()
                    self.g_optimizer.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                if self.rl:
                    self.checkpoint(epoch, [self.d_optimizer, self.g_optimizer, self.r_optimizer], [loss_d, loss_grl, loss_r])
                else:
                    self.checkpoint(epoch, [self.d_optimizer, self.g_optimizer, self.r_optimizer], [loss_d, loss_grl, -1])
            if verbose:
                if self.rl:
                    print(f'epoch {epoch+1}/{n_epochs}, loss_D={loss_d.item():.4}, loss_G={loss_grl.item():.4}, loss_R={loss_r.item():.4}')
                else:
                    print(f'epoch {epoch+1}/{n_epochs}, loss_D={loss_d.item():.4}, loss_G={loss_grl.item():.4}')

    def eval_loss(self, x, w_adj, mask, l_wrl):
        """
        Compute the evaluation loss of the model.

        A forward pass is performed, and the loss of the discriminator, generator, and reward network are computed.

        Parameters
        ----------
        x : torch.Tensor
            The node features.
        w_adj : torch.tensor
            The weighted adjacency tensor.
        mask : torch.tensor
            The mask indicating whether a particular node exists.
        l_wrl : float in [0,1]
            The trade-off between the generator loss and the reward network loss.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Sample z
        z = self._sample_z(1)
        # Compute generated graph
        x_gen, w_adj_gen = self.generator(z)
        # Logits from generated data
        logits_gen = self.discriminator(x_gen, w_adj_gen, torch.diagonal(w_adj_gen, dim1=1, dim2=2), activation=None)
        # Logits from real data
        logits_true = self.discriminator(x, w_adj, mask, activation=None)
        # Check whether RL should be used
        if self.rl:
            # Logits reward
            logits_rew_true = self.reward(x, w_adj, mask, activation=None)
            logits_rew_gen = self.reward(x_gen, w_adj_gen, torch.diagonal(w_adj_gen, dim1=1, dim2=2), activation=None)
            reward_true = self._reward_fun(x, w_adj, mask.int())
            reward_gen = self._reward_fun(x_gen, w_adj_gen, self._generated_mask_index(torch.diagonal(w_adj_gen, dim1=1, dim2=2)).int())
            loss_r = torch.mean((logits_rew_true - reward_true)**2 + (logits_rew_gen - reward_gen)**2)
            loss_grl = -(l_wrl*torch.mean(logits_gen) + (1 - l_wrl)*torch.mean(logits_rew_gen))
        else:
            loss_grl = -torch.mean(logits_gen)
        # Compute the discriminator loss
        loss_d = -(torch.mean(logits_true) - torch.mean(logits_gen))
        if self.rl:
            return {'Discriminator': loss_d, 'Generator': loss_grl, 'Reward': loss_r}
        else:
            return {'Discriminator': loss_d, 'Generator': loss_grl, 'Reward': -1}

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
        verbose : bool, optional
            Whether to print generation information. The default is False.

        Returns
        -------
        (x_pred, dist_pred) : (torch.Tensor, torch.Tensor)
            The predicted node features and distance matrix.
        """
        # Sample z
        z = self._sample_z(test_batch_size)
        # Compute generated graph
        gen_x, gen_w_adj = self.generator(z)
        # Refine the generated data
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
                x_pred[idx,0] = torch.clip(round(gen_x[0,i,0]), min=aa_min, max=aa_max)
                x_pred[idx,1] = torch.clip(round(gen_x[0,i,1]), min=ss_min, max=ss_max)
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