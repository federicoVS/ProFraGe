import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from generate.layers import GRULayer, MLPLayer
from generate.utils import seq_to_adj, clipping_dist

class ProRNN(nn.Module):
    """
    An augmented version of the `GraphRNN` model.

    A GRU layer (f_trans) initially computes the state of the graph. From there, an MLP (f_out_x) computes the node
    features while another GRU (f_out_edge) layer computes the adjacency/edge features matrix.

    In this model for the edge features only the class (i.e. type of connections) are taken into account.

    Source
    ------
    Paper => GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
             Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec
    Code  => https://github.com/snap-stanford/GraphRNN
    """

    def __init__(self, root, max_prev_node,
                 t_hidden_dim=64, o_hidden_dim=16, t_embed_dim=32, o_embed_dim=8, num_layers=4, mlp_dims=[1024],
                 aa_dim=20, ss_dim=7, dropout=0.1, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            Where the model data is saved.
        max_prev_node : int
            The maximum number of nodes.
        t_hidden_dim : int, optional
            The hidden dimension for f_trans. The default is 64.
        o_hidden_dim : int, optional
            The output dimension for f_trans. The default is 16.
        t_embed_dim : int, optional
            The embedding dimension of f_trans. The default is 32.
        o_embed_dim : int, optional
            The embedding dimension for f_out_x and f_out_edge. The default is 8.
        num_layers : int, optional
            The number of layers for f_trans and f_out_edge. The default is 4.
        mlp_dims : list of int, optional
            The dimensions of the MLPs. The default is [1024].
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structures. The default is 7.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(ProRNN, self).__init__()
        self.root = root
        self.max_prev_node = max_prev_node
        self.num_layers = num_layers
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.device = device

        self.X_DIM = 2

        self.f_trans = GRULayer(self.X_DIM+max_prev_node-1, t_hidden_dim, t_embed_dim, num_layers, has_output=True, out_dim=o_hidden_dim, dropout=dropout, device=device)
        self.f_out_x_aa = MLPLayer([t_hidden_dim] + mlp_dims + [aa_dim+1])
        self.f_out_x_ss = MLPLayer([t_hidden_dim] + mlp_dims + [ss_dim+1])
        self.f_out_w_adj = GRULayer(1, o_hidden_dim, o_embed_dim, num_layers, has_output=True, out_dim=1, dropout=dropout, device=device)

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

    def fit(self, loader, num_epochs, lr=1e-3, betas=(0.9, 0.999), decay_milestones=[400,1000], decay=0.1, checkpoint=500, verbose=False):
        """
        Train the model.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader or torch_geometric.data.DataLoader
            The data loader.
        num_epochs : int
            The number of epochs to perform.
        lr : float, optional
            The learning rate. The default is 1e-3.
        betas : (float,float), optional
            Coefficients used to compute averages of the gradient. The default is (0.9, 0.999).
        milestones : list of int, optional
            The list of milestones at which to decay the learning rate. The default is [400,1000].
        decay : float in [0,1], optional
            The decay of to apply to the learning rate. The default is 0.3.
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        verbose : bool, optional
            Whether to print the loss. The default is False.

        Returns
        -------
        None
        """
        optimizer_graph = Adam(self.f_trans.parameters(), lr=lr, betas=betas)
        optimizer_node = Adam(list(self.f_out_x_aa.parameters()) + list(self.f_out_x_ss.parameters()), lr=lr, betas=betas)
        optimizer_edge = Adam(self.f_out_w_adj.parameters(), lr=lr, betas=betas)
        scheduler_graph = MultiStepLR(optimizer_graph, milestones=decay_milestones, gamma=decay)
        scheduler_node = MultiStepLR(optimizer_node, milestones=decay_milestones, gamma=decay)
        scheduler_edge = MultiStepLR(optimizer_edge, milestones=decay_milestones, gamma=decay)
        for epoch in range(num_epochs):
            for i, (data) in enumerate(loader):
                self.f_trans.zero_grad()
                self.f_out_x_aa.zero_grad()
                self.f_out_x_ss.zero_grad()
                self.f_out_w_adj.zero_grad()
                # Get data
                x_unsorted = data['x']
                y_unsorted = data['y_edge']
                y_feat_unsorted = data['y_feat']
                y_len_unsorted = data['len']
                # Prepare data
                y_len_max = max(y_len_unsorted)
                x_unsorted = x_unsorted[:,0:y_len_max,:]
                y_unsorted = y_unsorted[:,0:y_len_max,:]
                y_feat_unsorted = y_feat_unsorted[:,0:y_len_max,:]
                self.f_trans.hidden = self.f_trans.init_hidden(batch_size=x_unsorted.shape[0])
                # Sort input
                y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
                y_len = y_len.numpy().tolist()
                x = torch.index_select(x_unsorted,0,sort_index)
                y = torch.index_select(y_unsorted,0,sort_index)
                y_feat = torch.index_select(y_feat_unsorted,0,sort_index)
                y_feat = y_feat.float()
                # Input, output for output RNN
                y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
                idx = [j for j in range(y_reshape.size(0)-1, -1, -1)] # reverse as to keep sorted lengths (also same size as y_edge_reshape)
                idx = torch.LongTensor(idx)
                y_reshape = y_reshape.index_select(0, idx)
                y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)
                output_x = torch.cat((torch.ones(y_reshape.size(0),1,1), y_reshape[:,0:-1,0:1]), dim=1)
                output_y = y_reshape
                output_y_len = []
                output_y_len_bin = np.bincount(np.array(y_len))
                for j in range(len(output_y_len_bin)-1,0,-1):
                    count_temp = np.sum(output_y_len_bin[j:]) # count how many y_len is above j
                    output_y_len.extend([min(j,self.max_prev_node)]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
                x = Variable(x).to(self.device)
                y_feat = Variable(y_feat).to(self.device)
                output_x = Variable(output_x).to(self.device)
                output_y = Variable(output_y).to(self.device) #[:,self.X_DIM:,]
                # Predict state of graph
                h_raw, h = self.f_trans(x, pack=True, input_len=y_len)
                h = pack_padded_sequence(h, y_len, batch_first=True).data # get packed hidden vector
                # Reverse h
                idx = [j for j in range(h.size(0) - 1, -1, -1)]
                idx = Variable(torch.LongTensor(idx)).to(self.device)
                h = h.index_select(0, idx)
                hidden_null = Variable(torch.zeros(self.num_layers-1, h.size(0), h.size(1))).to(self.device)
                self.f_out_w_adj.hidden = torch.cat((h.view(1,h.size(0),h.size(1)), hidden_null),dim=0) # num_layers, batch_size, hidden_size
                # Predict the weighted adjacency
                _, w_adj_pred = self.f_out_w_adj(output_x, pack=True, input_len=output_y_len)
                # Predict the node features
                x_pred_aa = self.f_out_x_aa(h_raw)
                x_pred_ss = self.f_out_x_ss(h_raw)
                x_pred_aa = x_pred_aa.float()
                x_pred_ss = x_pred_ss.float()
                # Clean
                w_adj_pred = pack_padded_sequence(w_adj_pred, output_y_len, batch_first=True)
                w_adj_pred = pad_packed_sequence(w_adj_pred, batch_first=True)[0]
                output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
                output_y = pad_packed_sequence(output_y, batch_first=True)[0]
                output_y = output_y.view(output_y.size(0),output_y.size(1))
                # Losses
                ce_loss_aa = F.cross_entropy(x_pred_aa.permute(0,2,1), y_feat[:,:,0].long())
                ce_loss_ss = F.cross_entropy(x_pred_ss.permute(0,2,1), y_feat[:,:,1].long())
                mse_loss_w_adj = F.mse_loss(w_adj_pred.squeeze(2), output_y)
                # Loss and optimization step
                loss = ce_loss_aa + ce_loss_ss + mse_loss_w_adj
                loss.backward()
                optimizer_graph.step()
                optimizer_node.step()
                optimizer_edge.step()
            # Weight decay
            scheduler_graph.step()
            scheduler_node.step()
            scheduler_edge.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, [optimizer_graph,optimizer_node,optimizer_edge], [scheduler_graph,scheduler_node,scheduler_edge], loss)
            if verbose:
                print(f'epoch {epoch+1}/{num_epochs},'
                      f'AA loss: {ce_loss_aa.item():.4},'
                      f'SS loss: {ce_loss_ss.item():.4},'
                      f'A_w loss: {mse_loss_w_adj.item():.4},'
                      f'Full loss = {loss.item():.4}')

    def eval_loss(self, x_unsorted, y_unsorted, y_feat_unsorted, y_len_unsorted):
        """
        Compute the evaluation loss of the model.

        A forward pass is performed, and the loss is computed.

        Parameters
        ----------
        x_unsorted : torch.Tensor
            The unsorted node features.
        y_unsorted : torch.Tensor
            The unsorted binary adjacency matrix.
        y_feat_unsorted : torch.Tensor
            The unsorted edge features.
        y_len_unsorted : list of int
            The unsorted number of nodes in each batch.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Prepare data
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:,0:y_len_max,:]
        y_unsorted = y_unsorted[:,0:y_len_max,:]
        y_feat_unsorted = y_feat_unsorted[:,0:y_len_max,:]
        self.f_trans.hidden = self.f_trans.init_hidden(batch_size=x_unsorted.shape[0])
        # Sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        y_feat = torch.index_select(y_feat_unsorted,0,sort_index)
        y_feat = y_feat.float()
        # Input, output for output RNN
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        idx = [j for j in range(y_reshape.size(0)-1, -1, -1)] # reverse as to keep sorted lengths (also same size as y_edge_reshape)
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)
        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1), y_reshape[:,0:-1,0:1]), dim=1)
        output_x = output_x
        output_y = y_reshape
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for j in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[j:]) # count how many y_len is above j
            output_y_len.extend([min(j,self.max_prev_node)]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        x = Variable(x).to(self.device)
        y_feat = Variable(y_feat).to(self.device)
        output_x = Variable(output_x).to(self.device)
        output_y = Variable(output_y).to(self.device) #[:,self.X_DIM:,]
        # Predict state of graph
        h_raw, h = self.f_trans(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h, y_len, batch_first=True).data # get packed hidden vector
        # Reverse h
        idx = [j for j in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(self.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(self.num_layers-1, h.size(0), h.size(1))).to(self.device)
        self.f_out_w_adj.hidden = torch.cat((h.view(1,h.size(0),h.size(1)), hidden_null),dim=0) # num_layers, batch_size, hidden_size
        # Predict the weighted adjacency
        _, w_adj_pred = self.f_out_w_adj(output_x, pack=True, input_len=output_y_len)
        # Predict the node features
        x_pred_aa = self.f_out_x_aa(h_raw)
        x_pred_ss = self.f_out_x_ss(h_raw)
        x_pred_aa = x_pred_aa.float()
        x_pred_ss = x_pred_ss.float()
        # Clean
        w_adj_pred = pack_padded_sequence(w_adj_pred, output_y_len, batch_first=True)
        w_adj_pred = pad_packed_sequence(w_adj_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        output_y = output_y.view(output_y.size(0),output_y.size(1))
        # Losses
        ce_loss_aa = F.cross_entropy(x_pred_aa.permute(0,2,1), y_feat[:,:,0].long())
        ce_loss_ss = F.cross_entropy(x_pred_ss.permute(0,2,1), y_feat[:,:,1].long())
        mse_loss_w_adj = F.mse_loss(w_adj_pred.squeeze(2), output_y)
        # Loss and optimization step
        loss = ce_loss_aa + ce_loss_ss + mse_loss_w_adj
        return {'Loss': loss}

    def generate(self, test_batch_size, max_num_nodes):
        """
        Evaluate the model by generating new data.

        Parameters
        ----------
        test_batch_size : int
            The number of graphs to generate.
        max_num_nodes : int
            The number of nodes to generate.

        Returns
        -------
        (x_pred, dist_pred) : (torch.Tensor, torch.Tensor)
            The predicted node features and distance matrix.
        """
        self.f_trans.hidden = self.f_trans.init_hidden(test_batch_size)
        node_pred_long = Variable(torch.zeros(test_batch_size,max_num_nodes,self.X_DIM)).to(self.device)
        w_adj_pred_float = Variable(torch.zeros(test_batch_size,max_num_nodes,self.max_prev_node-1)).to(self.device)
        x_step = Variable(torch.ones(test_batch_size,1,self.max_prev_node+self.X_DIM-1)).to(self.device)
        for i in range(max_num_nodes):
            h_raw, h = self.f_trans(x_step)
            hidden_null = Variable(torch.zeros(self.num_layers-1,h.size(0),h.size(2))).to(self.device)
            self.f_out_w_adj.hidden = torch.cat((h.permute(1,0,2),hidden_null), dim=0)  # num_layers, batch_size, hidden_size
            x_step = Variable(torch.zeros(test_batch_size,1,self.max_prev_node+self.X_DIM-1)).to(self.device)
            output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(self.device)
            pred_aa = torch.argmax(F.softmax(self.f_out_x_aa(h_raw), dim=2)[0,0,:])
            pred_ss = torch.argmax(F.softmax(self.f_out_x_ss(h_raw), dim=2)[0,0,:])
            node_pred_long[:,i,:] = torch.LongTensor([pred_aa, pred_ss]) # insert prediction
            x_step[:,:,0:self.X_DIM] = torch.LongTensor([pred_aa, pred_ss]) # update step
            for j in range(min(self.max_prev_node,i+1)):
                _, output_y_pred_step = self.f_out_w_adj(output_x_step)
                output_x_step = output_y_pred_step
                x_step[:,:,self.X_DIM+j:self.X_DIM+j+1] = output_x_step
                self.f_out_w_adj.hidden = Variable(self.f_out_w_adj.hidden.data).to(self.device) # update hidden state
            w_adj_pred_float[:,i:i+1,:] = x_step[:,:,self.X_DIM:]
            self.f_trans.hidden = Variable(self.f_trans.hidden.data).to(self.device)
        node_pred_float_data = node_pred_long.data.long()
        w_adj_pred_float = w_adj_pred_float.data.float()
        x_pred = node_pred_float_data
        w_adj_pred = seq_to_adj(w_adj_pred_float)
        # Define and fill the distance matrix prediction
        dist_pred = torch.zeros_like(w_adj_pred).to(self.device)
        for b in range(test_batch_size):
            for i in range(max_num_nodes):
                for j in range(max_num_nodes):
                    if i != j:
                        dist_pred[b,i,j] = min(1/w_adj_pred[b,i,j], clipping_dist(abs(i-j)))
        return x_pred, dist_pred