import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from generate.layers import GRULayer, GruMLPLayer, MLPLayer, ECCLayer

class GraphRNN_A(nn.Module):
    """
    An augmented (A) version of the `GraphRNN` model.

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
                 t_hidden_dim=64, o_hidden_dim=16, t_embed_dim=32, o_embed_dim=8, num_layers=4, x_dim=10, edge_class_dim=3, dropout=0, device='cpu'):
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
        x_dim : int, optional
            The dimension of the node features. The default is 10.
        edge_class_dim : int, optional
            The number of the edge classes. The default is 3.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.1.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(GraphRNN_A, self).__init__()
        self.root = root
        self.max_prev_node = max_prev_node
        self.num_layers = num_layers
        self.x_dim = x_dim
        self.edge_class_dim = edge_class_dim
        self.device = device

        self.f_trans = GRULayer(x_dim+max_prev_node-1, t_hidden_dim, t_embed_dim, num_layers, has_output=True, out_dim=o_hidden_dim, dropout=dropout, device=device)
        self.f_out_x = GruMLPLayer(t_hidden_dim, o_embed_dim, x_dim)
        self.f_out_edge = GRULayer(1, o_hidden_dim, o_embed_dim, num_layers, has_output=True, out_dim=edge_class_dim, dropout=dropout, device=device)

    def _decode_adj(self, adj_seq, b=0):
        n = adj_seq.shape[1]
        adj = torch.zeros(n,n)
        for i in range(n):
            for j in range(n):
                adj[i,j] = adj[j,i] = adj_seq[b,i,j]
        return adj

    def checkpoint(self, epoch, optimizers, schedulers, loss):
        """
        Create a checkpoint saving the results on the ongoing optimization.

        The checkpoint is saved at ROOT/checkpoint_<epoch>.

        Parameters
        ----------
        epoch : int
            The current epoch.
        optimizers : list of torch.optim.Optimizer
            The list of optimizers. The first one should be the associated with f_trans, the second with f_out_x, and
            the third for f_out_edge.
        schedulers : list of torch.optim.lr_scheduler.Scheduler
            The list of scheduler. The first one should be the associated with optimizer_trans_trans, the second with
            optimizer_trans_out_x, and the third for optimizer_trans_out_edge.
        loss : float
            The current loss.

        Returns
        -------
        None
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'trans_optimizer_state_dict': optimizers[0].state_dict(),
                    'x_optimizer_state_dict': optimizers[1].state_dict(),
                    'edge_optimizer_state_dict': optimizers[2].state_dict(),
                    'trans_scheduler_state_dict': schedulers[0].state_dict(),
                    'x_scheduler_state_dict': schedulers[1].state_dict(),
                    'edge_scheduler_state_dict': schedulers[2].state_dict(),
                    'loss': loss}, self.root + 'checkpoint_' + str(epoch))

    def fit(self, loader, num_epochs, lr_trans=3e-3, lr_out_x=3e-3, lr_out_edge=3e-3, checkpoint=500, milestones=[400,1000], decay=0.3, verbose=False):
        """
        Train the model.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader or torch_geometric.data.DataLoader
            The data loader.
        num_epochs : int
            The number of epochs to perform.
        lr_trans : float, optional
            The learning rate for f_trans. The default is 3e-3.
        lr_out_x : float, optional
            The learning rate for f_out_x. The default is 3e-3.
        lr_out_edge : float, optional
            The learning rate for f_out_edge. The default is 3e-3.
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        milestones : list of int, optional
            The list of milestones at which to decay the learning rate. The default is [400,1000].
        decay : float in [0,1], optional
            The decay of to apply to the learning rate. The default is 0.3.
        verbose : bool, optional
            Whether to print the loss. The default is False.

        Returns
        -------
        None
        """
        optimizer_trans = Adam(self.f_trans.parameters(), lr=lr_trans)
        optimizer_out_x = Adam(self.f_out_x.parameters(), lr=lr_out_x)
        optimizer_out_edge = Adam(self.f_out_edge.parameters(), lr=lr_out_edge)
        scheduler_trans = MultiStepLR(optimizer_trans, milestones=milestones, gamma=decay)
        scheduler_out_x = MultiStepLR(optimizer_out_x, milestones=milestones, gamma=decay)
        scheduler_out_edge = MultiStepLR(optimizer_out_edge, milestones=milestones, gamma=decay)
        for epoch in range(num_epochs):
            for i, (data) in enumerate(loader):
                self.f_trans.zero_grad()
                self.f_out_x.zero_grad()
                self.f_out_edge.zero_grad()
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
                output_y = Variable(output_y).to(self.device) #[:,self.x_dim:,]
                h_raw, h = self.f_trans(x, pack=True, input_len=y_len)
                h = pack_padded_sequence(h, y_len, batch_first=True).data # get packed hidden vector
                # Reverse h
                idx = [j for j in range(h.size(0) - 1, -1, -1)]
                idx = Variable(torch.LongTensor(idx)).to(self.device)
                h = h.index_select(0, idx)
                hidden_null = Variable(torch.zeros(self.num_layers-1, h.size(0), h.size(1))).to(self.device)
                self.f_out_edge.hidden = torch.cat((h.view(1,h.size(0),h.size(1)), hidden_null),dim=0) # num_layers, batch_size, hidden_size
                # Predict the edges
                _, edge_pred = self.f_out_edge(output_x, pack=True, input_len=output_y_len)
                edge_pred = torch.softmax(edge_pred, dim=2)
                # Predict the node features
                x_pred = self.f_out_x(h_raw)
                x_pred = x_pred.float()
                # Clean
                edge_pred = pack_padded_sequence(edge_pred, output_y_len, batch_first=True)
                edge_pred = pad_packed_sequence(edge_pred, batch_first=True)[0]
                edge_pred = edge_pred.permute(0,2,1)
                output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
                output_y = pad_packed_sequence(output_y, batch_first=True)[0]
                output_y = output_y.view(output_y.size(0),output_y.size(1))
                output_y = output_y.long() # target must have long type
                # Losses
                ce_loss = F.cross_entropy(edge_pred, output_y)
                mse_loss = F.mse_loss(x_pred, y_feat)
                # Beta parameter for stability
                beta = torch.abs(ce_loss.detach()/mse_loss.detach()).detach()
                # Loss and optimization step
                loss = ce_loss + beta*mse_loss
                loss.backward()
                optimizer_trans.step()
                optimizer_out_x.step()
                optimizer_out_edge.step()
                scheduler_trans.step()
                scheduler_out_x.step()
                scheduler_out_edge.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, [optimizer_trans,optimizer_out_x,optimizer_out_edge], [scheduler_trans,scheduler_out_x,scheduler_out_edge], loss)
            if verbose:
                print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x_unsorted, y_unsorted, y_feat_unsorted, y_len_unsorted):
        """
        Compute the evaluation loss of the model.

        A forward pass is performed, and the loss is computed.

        Parameters
        ----------
        x_unsorted : torch.tensor
            The unsorted node features.
        y_unsorted : torch.tensor
            The unsorted binary adjacency matrix.
        y_feat_unsorted : torch.tensor
            The unsorted edge features.
        y_len_unsorted : list of int
            The unsorted number of nodes in each batch.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Forward pass
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
        output_y = Variable(output_y).to(self.device) #[:,self.x_dim:,]
        h_raw, h = self.f_trans(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h, y_len, batch_first=True).data # get packed hidden vector
        # Reverse h
        idx = [j for j in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(self.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(self.num_layers-1, h.size(0), h.size(1))).to(self.device)
        self.f_out_edge.hidden = torch.cat((h.view(1,h.size(0),h.size(1)), hidden_null),dim=0) # num_layers, batch_size, hidden_size
        # Predict the edges
        _, edge_pred = self.f_out_edge(output_x, pack=True, input_len=output_y_len)
        edge_pred = torch.softmax(edge_pred, dim=2)
        # Predict the node features
        x_pred = self.f_out_x(h_raw)
        x_pred = x_pred.float()
        # Clean
        edge_pred = pack_padded_sequence(edge_pred, output_y_len, batch_first=True)
        edge_pred = pad_packed_sequence(edge_pred, batch_first=True)[0]
        edge_pred = edge_pred.permute(0,2,1)
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        output_y = output_y.view(output_y.size(0),output_y.size(1))
        output_y = output_y.long() # target must have long type
        # Losses
        ce_loss = F.cross_entropy(edge_pred, output_y)
        mse_loss = F.mse_loss(x_pred, y_feat)
        # Beta parameter for stability
        beta = torch.abs(ce_loss.detach()/mse_loss.detach()).detach()
        # Loss and optimization step
        loss = ce_loss + beta*mse_loss
        return {'Loss': loss}

    def generate(self, max_num_nodes, test_batch_size=1, aa_min=1, aa_max=20, ss_min=1, ss_max=7):
        """
        Evaluate the model by generating new data.

        Parameters
        ----------
        max_num_nodes : int
            The number of nodes to generate.
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
        (x_pred, adj_pred) : (torch.tensor, torch.tensor)
            The predicted node features and adjacency matrix.
        """
        self.f_trans.hidden = self.f_trans.init_hidden(test_batch_size)
        node_pred_float = Variable(torch.zeros(test_batch_size,max_num_nodes,self.x_dim)).to(self.device)
        edge_pred_long = Variable(torch.zeros(test_batch_size,max_num_nodes,self.max_prev_node-1)).to(self.device)
        x_step = Variable(torch.ones(test_batch_size,1,self.max_prev_node+self.x_dim-1)).to(self.device)
        for i in range(max_num_nodes):
            h_raw, h = self.f_trans(x_step)
            hidden_null = Variable(torch.zeros(self.num_layers-1,h.size(0),h.size(2))).to(self.device)
            self.f_out_edge.hidden = torch.cat((h.permute(1,0,2),hidden_null), dim=0)  # num_layers, batch_size, hidden_size
            x_step = Variable(torch.zeros(test_batch_size,1,self.max_prev_node+self.x_dim-1)).to(self.device)
            output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(self.device)
            node_pred = self.f_out_x(h_raw)
            node_pred_float[:,i,:] = node_pred # insert prediction
            x_step[:,:,0:self.x_dim] = node_pred # update step
            for j in range(min(self.max_prev_node,i+1)):
                _, output_y_pred_step = self.f_out_edge(output_x_step)
                output_x_step = torch.argmax(torch.softmax(output_y_pred_step, dim=2))
                x_step[:,:,self.x_dim+j:self.x_dim+j+1] = output_x_step
                output_x_step = torch.FloatTensor([[[output_x_step]]]) # convert and reshape prediction
                self.f_out_edge.hidden = Variable(self.f_out_edge.hidden.data).to(self.device) # update hidden state
            edge_pred_long[:,i:i+1,:] = x_step[:,:,self.x_dim:]
            self.f_trans.hidden = Variable(self.f_trans.hidden.data).to(self.device)
        node_pred_float_data = node_pred_float.data.float()
        edge_pred_long_data = edge_pred_long.data.long()
        x_pred = node_pred_float_data
        adj_pred = self._decode_adj(edge_pred_long_data)
        x_pred = x_pred.view(x_pred.size(1),x_pred.size(2))
        for i in range(max_num_nodes):
            x_pred[i,0] = torch.clip(x_pred[i,0].round(), min=aa_min, max=aa_max)
            x_pred[i,1] = torch.clip(x_pred[i,1].round(), min=ss_min, max=ss_max)
        return x_pred, adj_pred

class GraphRNN_G(nn.Module):
    """
    A graph (G) version of the `GraphRNN` model.

    Instead of representing the graph as a sequence, an EC graph convolutional layer is used to get an embedding
    of the graph. This sequence is then fed to a GRU layer to be processed.
    Last, predictions (node classes and values) are carried out by two distinct MLPs.
    """

    def __init__(self, root, max_prev_node, x_dim, edge_dim, hidden_dim, g_latent_dim, embed_dim, ecc_dims, mlp_dims, ecc_inner_dims=[], num_layers=4,
                 dropout=0, class_dim=2, aa_dim=20, ss_dim=7, ignore_idx=-100, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        root : str
            Where to save the model data.
        max_prev_node : int
            The maximum number of nodes.
        x_dim : int
            The dimension of the node features.
        edge_dim : int
            The dimension of the edge features.
        hidden_dim : int
            The hidden dimension.
        g_latent_dim : int
            The latent dimension in the EC graph convolution.
        embed_dim : int
            The embedding dimension in the GRU layer.
        ecc_dims : list of int
            The dimensions of the ECC layers.
        mlp_dims : list of int
            The dimensions of the MLP layers.
        ecc_inner_dims : list of int, optional
            The dimensions for the `h` in ECC. The default is [8].
        num_layers : int, optional
            The number of GRU cells.
        dropout : float in [0,1], optional
            The dropout probability. The default is 0.
        class_dim : int, optional
            The number of classes in the node features. The default is 2.
        aa_dim : int, optional
            The number of amino acids. The default is 20.
        ss_dim : int, optional
            The number of secondary structures. The default is 7.
        ignore_idx : int, optional
            The index of classes to be ignored by the cross-entropy loss. The default is -100.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(GraphRNN_G, self).__init__()
        self.root = root
        self.max_prev_node = max_prev_node
        self.x_dim = x_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.class_dim = class_dim
        self.aa_dim = aa_dim
        self.ss_dim = ss_dim
        self.ignore_idx = ignore_idx
        self.device = device

        self.enc = ECCLayer([x_dim] + ecc_dims + [g_latent_dim], ecc_inner_dims, edge_dim)
        self.lin = nn.Linear(g_latent_dim, x_dim)
        self.gru = GRULayer(x_dim, hidden_dim, embed_dim, num_layers, dropout=dropout, device=device)
        self.fc_out = MLPLayer([hidden_dim] + mlp_dims + [aa_dim*ss_dim+x_dim-class_dim])

        self._init_weights(self.lin)
        self.fc_out.apply(self._init_weights)

    def _init_weights(self, m, mode='he'):
        if isinstance(m, nn.Linear):
            if mode == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif mode == 'unif':
                nn.init.uniform_(m.weight, a=-0.005, b=0.005)

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

    def _encode_rnn_data(self, x_true, x_gen, batch_len):
        x_true_seq, x_gen_seq = [], []
        prev = 0
        for i in range(len(batch_len)):
            x_true_seq.append(x_true[prev:prev+batch_len[i]].clone().detach())
            x_gen_seq.append(x_gen[prev:prev+batch_len[i]].clone().detach())
            prev += batch_len[i]
        x_true_seq = pad_sequence(x_true_seq, batch_first=True)
        x_gen_seq = pad_sequence(x_gen_seq, batch_first=True)
        n, m = len(x_true_seq), max(batch_len)
        xt, yt = torch.zeros(n,m,self.x_dim), torch.zeros(n,m,self.x_dim)
        xt[:,0:m,:] = x_gen_seq
        yt[:,0:m,:] = x_true_seq
        return xt, yt

    def checkpoint(self, epoch, optimizer, scheduler, loss):
        """
        Create a checkpoint saving the results on the ongoing optimization.

        The checkpoint is saved at ROOT/checkpoint_<epoch>.

        Parameters
        ----------
        epoch : int
            The current epoch.
        optimizer : torch.optim.Optimizer
            The optimizer.
        scheduler : torch.optim.lr_scheduler.Scheduler
            The scheduler.
        loss : float
            The current loss.

        Returns
        -------
        None
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss}, self.root + 'checkpoint_' + str(epoch))

    def fit(self, loader, num_epochs, lr=3e-3, checkpoint=500, milestones=[400,1000], decay=0.3, verbose=False):
        """
        Train the model.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader or torch_geometric.data.DataLoader
            The data loader.
        lr : float, optional
            The learning rate. The default is 3e-3.
        checkpoint : int, optional
            The epoch interval at which a checkpoint is created. The default is 500.
        milestones : list of int, optional
            The list of milestones at which to decay the learning rate. The default is [400,1000].
        decay : float, optional
            The decay to apply to the learning rate. The default is 0.3.
        verbose : bool, optional
            Whether to print the loss. The default is False.

        Returns
        -------
        None
        """
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=decay)
        for epoch in range(num_epochs):
            for i, (data) in enumerate(loader):
                optimizer.zero_grad()
                # Get data
                x, adj, edge, batch_len = data.x, data.edge_index, data.edge_attr, data.x_len
                # Put the data on device
                x, adj, edge = x.to(self.device), adj.to(self.device), edge.to(self.device)
                # Encode the graph
                graph = self.enc(x, adj, edge)
                graph = self.lin(graph)
                # Get RNN data and sort it
                xt, yt = self._encode_rnn_data(x, graph, batch_len)
                # Sort the input
                y_len, sort_index = torch.sort(batch_len, 0, descending=True)
                y_len = y_len.numpy().tolist()
                xt, yt = torch.index_select(xt, 0, sort_index), torch.index_select(yt, 0, sort_index)
                xt = torch.cat((torch.ones(xt.shape[0],1,self.x_dim), xt), dim=1)
                xt = Variable(xt).to(self.device)
                yt = Variable(yt).to(self.device)
                # Initialize hidden state
                self.gru.hidden = self.gru.init_hidden(batch_size=xt.shape[0])
                out, _ = self.gru(xt, pack=True, input_len=y_len)
                out_x = self.fc_out(out)
                # Clean
                y_pred = pack_padded_sequence(out_x, y_len, batch_first=True)
                y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
                # Losses
                ce_loss = F.cross_entropy(torch.transpose(y_pred[:,:,0:self.aa_dim*self.ss_dim], 1,2), self._input_classes(yt[:,:,0:self.class_dim]))
                mse_loss = F.mse_loss(y_pred[:,:,self.aa_dim*self.ss_dim:], yt[:,:,self.class_dim:])
                # Beta parameter for stability
                beta = torch.abs(ce_loss.detach()/mse_loss.detach()).detach()
                loss = ce_loss + beta*mse_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
            if checkpoint is not None and epoch != 0 and epoch % checkpoint == 0:
                self.checkpoint(epoch, optimizer, scheduler, loss)
            if verbose:
                print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4}')

    def eval_loss(self, x, adj, edge, batch_len):
        """
        Compute the evaluation loss of the model.

        A forward pass is performed, and the loss is computed.

        Parameters
        ----------
        x : torch.tensor
            The node features.
        adj : torch.tensor
            The adjacency matrix.
        edge : torch.tensor
            The edge features.
        batch_len : list of int
            The number of nodes in each batch.

        Returns
        -------
        dict of str -> float
            The losses.
        """
        # Forward pass
        # Encode the graph
        graph = self.enc(x, adj, edge_attr=edge)
        graph = self.lin(graph)
        # Get RNN data and sort it
        xt, yt = self._encode_rnn_data(x, graph, batch_len)
        # Sort the input
        y_len, sort_index = torch.sort(batch_len, 0, descending=True)
        y_len = y_len.numpy().tolist()
        xt, yt = torch.index_select(xt, 0, sort_index), torch.index_select(yt, 0, sort_index)
        xt = torch.cat((torch.ones(xt.shape[0],1,self.x_dim), xt), dim=1)
        xt = Variable(xt).to(self.device)
        yt = Variable(yt).to(self.device)
        # Initialize hidden state
        self.gru.hidden = self.gru.init_hidden(batch_size=xt.shape[0])
        out, _ = self.gru(xt, pack=True, input_len=y_len)
        out_x = self.fc_out(out)
        # Clean
        y_pred = pack_padded_sequence(out_x, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # Losses
        ce_loss = F.cross_entropy(torch.transpose(y_pred[:,:,0:self.aa_dim*self.ss_dim], 1,2), self._input_classes(yt[:,:,0:self.class_dim]))
        mse_loss = F.mse_loss(y_pred[:,:,self.aa_dim*self.ss_dim:], yt[:,:,self.class_dim:])
        # Beta parameter for stability
        beta = torch.abs(ce_loss.detach()/mse_loss.detach()).detach()
        loss = ce_loss + beta*mse_loss
        return {'Loss': loss}

    def generate(self, max_num_nodes):
        """
        Evaluate the model by generating new data.

        Parameters
        ----------
        max_num_nodes : int
            The number of nodes to generate.

        Returns
        -------
        x_pred : torch.tensor
            The predicted node features.
        """
        self.gru.hidden = self.gru.init_hidden(batch_size=1)
        x_pred = Variable(torch.zeros(1,max_num_nodes,self.x_dim)).to(self.device)
        x_step = Variable(torch.ones(1,1,self.x_dim)).to(self.device)
        for i in range(max_num_nodes):
            h_raw, _ = self.gru(x_step)
            x_class_idx = torch.argmax(self.fc_out_class(h_raw)[:,:,])
            a_tensor, s_tensor = x_class_idx - (x_class_idx//self.aa_dim)*self.aa_dim + 1, x_class_idx//self.aa_dim + 1
            x_step_class = torch.tensor([[[a_tensor, s_tensor]]])
            x_step_reg = self.fc_out_reg(h_raw)
            x_step = torch.cat((x_step_class, x_step_reg), dim=2)
            x_pred[:,i,:] = x_step
            self.gru.hidden = Variable(self.gru.hidden.data).to(self.device)
        x_pred = x_pred.data.float()
        x_pred = x_pred.view(x_pred.shape[1],x_pred.shape[2])
        return x_pred
