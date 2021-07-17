import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from generate.layers import GRULayer, GruMLPLayer, MLPLayer
from generate.utils import sample_softmax

class GraphRNN_A(nn.Module):

    def __init__(self, max_prev_node,
                 t_hidden_dim=64, o_hidden_dim=16, t_embed_dim=32, o_embed_dim=8, num_layers=4, node_dim=5, edge_dim=3, dropout=0, device='cpu'):
        super(GraphRNN_A, self).__init__()
        self.max_prev_node = max_prev_node
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = device

        self.f_trans = GRULayer(max_prev_node+node_dim-1, t_hidden_dim, t_embed_dim, num_layers, has_output=True, out_dim=o_hidden_dim, dropout=dropout, device=device)
        self.f_out_x = GruMLPLayer(t_hidden_dim, o_embed_dim, node_dim)
        self.f_out_edge = GRULayer(1, o_hidden_dim, o_embed_dim, num_layers, has_output=True, out_dim=edge_dim, dropout=dropout, device=device)

    def _decode_adj(self, adj_seq, b=0):
        n = adj_seq.shape[1]
        adj = torch.zeros(n,n)
        for i in range(n):
            for j in range(n):
                adj[i,j] = adj[j,i] = adj_seq[b,i,j]
        return adj

    def train(self, loader, num_epochs, batch_size, lr_trans=3e-3, lr_out_x=3e-3, lr_out_edge=3e-3, elw=10, milstones=[400,1000], decay=0.3, verbose=False):
        optimizer_trans = Adam(self.f_trans.parameters(), lr=lr_trans)
        optimizer_out_x = Adam(self.f_out_x.parameters(), lr=lr_out_x)
        optimizer_out_edge = Adam(self.f_out_edge.parameters(), lr=lr_out_edge)
        scheduler_trans = MultiStepLR(optimizer_trans, milestones=milstones, gamma=decay)
        scheduler_out_x = MultiStepLR(optimizer_out_x, milestones=milstones, gamma=decay)
        scheduler_out_edge = MultiStepLR(optimizer_out_edge, milestones=milstones, gamma=decay)
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
                output_y = Variable(output_y).to(self.device) #[:,self.node_dim:,]
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
                output_y = output_y.view(output_y.size(0),output_y.size(1)) # suitable shape for cross entropy
                output_y = output_y.long() # target must have long type
                # Loss and optimization step
                loss = elw*F.cross_entropy(edge_pred, output_y) + F.mse_loss(x_pred, y_feat)
                loss.backward()
                optimizer_trans.step()
                optimizer_out_x.step()
                optimizer_out_edge.step()
                scheduler_trans.step()
                scheduler_out_x.step()
                scheduler_out_edge.step()
            if verbose:
                print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4}')

    def eval(self, max_num_nodes, test_batch_size=1, aa_min=1, aa_max=20, ss_min=0, ss_max=6):
        self.f_trans.hidden = self.f_trans.init_hidden(test_batch_size)
        node_pred_float = Variable(torch.zeros(test_batch_size,max_num_nodes,self.node_dim)).to(self.device)
        edge_pred_long = Variable(torch.zeros(test_batch_size,max_num_nodes,self.max_prev_node-1)).to(self.device)
        x_step = Variable(torch.ones(test_batch_size,1,self.max_prev_node+self.node_dim-1)).to(self.device)
        for i in range(max_num_nodes):
            h_raw, h = self.f_trans(x_step)
            hidden_null = Variable(torch.zeros(self.num_layers-1,h.size(0),h.size(2))).to(self.device)
            self.f_out_edge.hidden = torch.cat((h.permute(1,0,2),hidden_null), dim=0)  # num_layers, batch_size, hidden_size
            x_step = Variable(torch.zeros(test_batch_size,1,self.max_prev_node+self.node_dim-1)).to(self.device)
            output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(self.device)
            node_pred = self.f_out_x(h_raw)
            node_pred_float[:,i,:] = node_pred # insert prediction
            x_step[:,:,0:self.node_dim] = node_pred # update step
            for j in range(min(self.max_prev_node,i+1)):
                _, output_y_pred_step = self.f_out_edge(output_x_step)
                output_x_step = sample_softmax(output_y_pred_step) #sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
                x_step[:,:,self.node_dim+j:self.node_dim+j+1] = output_x_step
                output_x_step = torch.FloatTensor([[[output_x_step]]]) # convert and reshape prediction
                self.f_out_edge.hidden = Variable(self.f_out_edge.hidden.data).to(self.device) # update hidden state
            edge_pred_long[:,i:i+1,:] = x_step[:,:,self.node_dim:]
            self.f_trans.hidden = Variable(self.f_trans.hidden.data).to(self.device)
        node_pred_float_data = node_pred_float.data.float()
        edge_pred_long_data = edge_pred_long.data.long()
        x_pred = node_pred_float_data
        adj_pred = self._decode_adj(edge_pred_long_data)
        x_pred = x_pred.view(x_pred.size(1),x_pred.size(2))
        for i in range(max_num_nodes):
            x_pred[i,0] = torch.clip(x_pred[i,0], min=aa_min, max=aa_max)
            x_pred[i,1] = torch.clip(x_pred[i,1], min=ss_min, max=ss_max)
        return x_pred, adj_pred

class GraphRNN_G(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, embed_dim, mlp_dims, num_layers=4, dropout=0, device='cpu'):
        super(GraphRNN_G, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.device = device

        self.ecc = gnn.ECConv(node_dim, node_dim, nn.Sequential(nn.Linear(edge_dim,node_dim*node_dim)))
        self.norm = nn.LayerNorm(node_dim)
        self.gru = GRULayer(node_dim, hidden_dim, embed_dim, num_layers, dropout=dropout, device=device)
        self.fc_out = MLPLayer([hidden_dim] + mlp_dims + [node_dim])

    def train(self, loader, num_epochs, batch_size, lr=3e-3, l_kld=1e-5, milstones=[400,1000], decay=0.3, verbose=False):
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milstones, gamma=decay)
        for epoch in range(num_epochs):
            for i, (data) in enumerate(loader):
                optimizer.zero_grad()
                # Get data
                x, adj, edge, batch_len = data.x, data.edge_index, data.edge_attr, data.batch_len
                # Encode the graph
                graph = self.ecc(x, adj, edge_attr=edge)
                graph = self.norm(graph)
                # Reshape graph emebdding
                graph_reshape = graph.view(graph.shape[0],1,graph.shape[1])
                graph_input = torch.cat((torch.ones(1,1,self.node_dim), graph_reshape))
                x_output = torch.cat((x, torch.zeros(1,x.shape[1])))
                graph_input = Variable(graph_input).to(self.device)
                x_output = Variable(x_output).to(self.device)
                # Initialize hidden state
                self.gru.hidden = self.gru.init_hidden(batch_size=graph_input.shape[0])
                out, _ = self.gru(graph_input)
                out = out.view(out.shape[0],out.shape[2])
                out = self.fc_out(out)
                loss = F.mse_loss(out, x_output)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if verbose:
                print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4}')

    def eval(self, max_num_nodes, aa_min=1, aa_max=20, ss_min=0, ss_max=6):
        self.gru.hidden = self.gru.init_hidden(batch_size=1)
        x_pred = Variable(torch.zeros(max_num_nodes,self.node_dim)).to(self.device)
        x_step = Variable(torch.ones(1,self.node_dim)).to(self.device)
        for i in range(max_num_nodes):
            x_step = x_step.view(x_step.shape[0],1,x_step.shape[1])
            h_raw, _ = self.gru(x_step)
            h_raw = h_raw.view(h_raw.shape[0],h_raw.shape[2])
            x_step = self.fc_out(h_raw)
            x_pred[i] = x_step
            self.gru.hidden = Variable(self.gru.hidden.data).to(self.device)
        x_pred = x_pred.data.float()
        for i in range(max_num_nodes):
            x_pred[i,0] = torch.clip(x_pred[i,0], min=aa_min, max=aa_max)
            x_pred[i,1] = torch.clip(x_pred[i,1], min=ss_min, max=ss_max)
        return x_pred
