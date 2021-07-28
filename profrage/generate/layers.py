import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.values = nn.Linear(self.head_dim, self.head_dim)
        self.keys = nn.Linear(self.head_dim, self.head_dim)
        self.queries = nn.Linear(self.head_dim, self.head_dim)
        self.fc_out = nn.Linear(num_heads*self.head_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split embedding into num_heads partitions
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.values(keys)
        queries = self.values(query)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            # print(energy.shape, mask.shape)
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy/(self.embed_dim**(0.5)), dim=3)
        out = torch.einsum('nhql,nlhd->nqhd', [attention,values]).reshape(N, query_len, self.num_heads*self.head_dim) # key length and values length match
        out = self.fc_out(out)
        return out

class GANGenerator(nn.Module):
    """
    The generator used in the `ProGAN` model.

    Source
    ------
    https://github.com/yongqyu/MolGAN-pytorch, https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch
    """

    def __init__(self, max_num_nodes, node_dim, edge_dim, z_dim, mlp_dims, dropout=0.1):
        """
        Initialize the class.

        Parameters
        ----------
        max_num_nodes : int
            The maximum number of nodes.
        node_dim : int
            The size of the node features.
        edge_dim : int
            The size of the edge features.
        z_dim : int
            The size of the sample space.
        mlp_dims : list of int
            The sizes of the layers in the MLP.
        dropout : float in [0,1], optional
            The probability of dropout. The default is 0.1.
        """
        super(GANGenerator, self).__init__()
        self.max_num_nodes = max_num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        layers = []
        for in_dim, out_dim in zip([z_dim]+mlp_dims[:-1], mlp_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.hidden_layers = nn.Sequential(*layers)
        self.x_layer = nn.Linear(mlp_dims[-1], max_num_nodes*node_dim)
        self.adj_layer = nn.Linear(mlp_dims[-1], max_num_nodes*max_num_nodes)
        self.edge_layer = nn.Linear(mlp_dims[-1], max_num_nodes*max_num_nodes*edge_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, z):
        """
        Compute the forward pass.

        Parameters
        ----------
        z : torch.tensor
            The sampled space/

        Returns
        -------
        (x_logits, adj_logits, edge_logits) : (torch.tensor, torch.tensor, torch.tensor)
            The generated node features, adjacency matrix, and edge features.
        """
        out = self.hidden_layers(z)
        x_logits = self.dropout_layer(self.x_layer(out).view(-1, self.max_num_nodes, self.node_dim))
        adj_logits = self.dropout_layer(self.adj_layer(out).view(-1, self.max_num_nodes, self.max_num_nodes))
        edge_logits = self.dropout_layer(self.edge_layer(out).view(-1, self.max_num_nodes, self.max_num_nodes, self.edge_dim))
        return x_logits, adj_logits, edge_logits

class GANDiscriminator(nn.Module):
    """
    The discriminator used in the `ProGAN` model.

    Source
    ------
    https://github.com/yongqyu/MolGAN-pytorch, https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch
    """

    def __init__(self, node_dim, edge_dim, conv_out_dim, agg_dim, mlp_dims, dropout=0.1):
        super(GANDiscriminator, self).__init__()

        self.conv_layer = gnn.ECConv(node_dim, conv_out_dim, nn.Sequential(nn.Linear(edge_dim, node_dim*conv_out_dim)))
        self.agg_layer = GraphAggregation(conv_out_dim, agg_dim)
        layers = []
        for in_dim, out_dim in zip([1]+mlp_dims[:-1], mlp_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(mlp_dims[-1], 1)

    def forward(self, x, adj, edge, activation=None):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : torch.tensor
            The node feature tensor.
        adj : torch.tensor
            The adjacency tensor.
        edge : torch.tensor
            The edge features tensor.
        activation : callable, optional
            The activation function. The default is None.

        Returns
        -------
        out : torch.tensor
            The output.
        """
        out = self.conv_layer(x, adj, edge_attr=edge)
        out = self.agg_layer(out)
        out = self.mlp_layers(out)
        out = self.fc_out(out)
        out = activation(out) if activation is not None else out
        return out

class GraphAggregation(nn.Module):
    """
    Perform graph aggregation in the `GANDiscriminator` module.

    Source
    ------
    https://github.com/yongqyu/MolGAN-pytorch, https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch
    """

    def __init__(self, input_dim, output_dim):
        super(GraphAggregation, self).__init__()

        self.sigmoid_layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                           nn.Sigmoid())
        self.tanh_layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                        nn.Tanh())

    def forward(self, x):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : torch.tensor
            The tensor representing the state of the graph.

        Returns
        -------
        out : torch.tensor
            The aggregated output.
        """
        i, j = self.sigmoid_layer(x), self.tanh_layer(x)
        out = torch.sum(torch.mul(i,j), 1).view(-1, 1)
        return out

class GRULayer(nn.Module):
    """
    A GRU layer composed of several GRU cells.

    Source
    ------
    https://github.com/snap-stanford/GraphRNN
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, has_input=True, has_output=False, out_dim=None, dropout=0, device='cpu'):
        """
        Initialize the class.

        Parameters
        ----------
        input_dim : int
            The dimension of the input features.
        hidden_dim : int
            The hidden dimension.
        embed_dim : int
            The embedding dimension.
        num_layers : int
            The number of GRU cells.
        has_input : bool, optional
            Whether the layer has an output. The default is True.
        has_output : bool, optional
            Whether the layer has an output. The default is False.
        out_dim : int, optional
            The dimension of the output features. The default is None.
        dropout : float in [0,1], optional
            The dropout probability.
        device : str, optional
            The device where to put the data. The default is 'cpu'.
        """
        super(GRULayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.has_input = has_input
        self.has_output = has_output
        self.device = device

        if has_input:
            self.in_layer = nn.Linear(input_dim, embed_dim)
            self.rnn_layer = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            self.rnn_layer = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        if has_output:
            self.out_layer = nn.Sequential(nn.Linear(hidden_dim, embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(embed_dim, out_dim))

        self.relu = nn.ReLU()
        self.hidden = None # initialize before run

        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        torch.autograd.Variable
            The hidden state.
        """
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)).to(self.device)

    def forward(self, input_raw, pack=False, input_len=None):
        """
        Compute the forward pass.

        Parameters
        ----------
        input_raw : torch.tensor
            The input representing a sequence.
        pack : bool, optional
            Whether to pack the sequence. The default is False.
        input_len : list of int, optional
            The lengths of single sequences within the batch. The default is None.

        Returns
        -------
        (output_raw, output) : (torch.tensor, torch.tensor)
            The raw output and the output after being passed into an MLP.
        """
        if self.has_input:
            input = self.in_layer(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn_layer(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        output = None
        if self.has_output:
            output = self.out_layer(output_raw)
        return output_raw, output

class GruMLPLayer(nn.Module):
    """
    An MLP layer associated with the GRU model.
    """

    def __init__(self, hidden_dim, embed_dim, y_dim):
        """
        Initialize the class.

        Parameters
        ----------
        hidden_dim : int
            The dimension of the input features..
        embed_dim : int
            The hidden dimension.
        y_dim : int
            The dimension of the output features.
        """
        super(GruMLPLayer, self).__init__()

        self.lin_layer = nn.Sequential(nn.Linear(hidden_dim, embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(embed_dim, y_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        """
        Compute the forward pass.

        Parameters
        ----------
        h : torch.tensor
            The input.

        Returns
        -------
        y : torch.tensor
            The output.
        """
        y = self.lin_layer(h)
        return y

class MLPLayer(nn.Module):
    """
    A generic MLP layer.
    """

    def __init__(self, dims):
        """
        Initialize the class.

        Parameters
        ----------
        dims : list of int
            The dimensions of the MLP layers.
        """
        super(MLPLayer, self).__init__()

        layers = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : torch.tensor
            The input.

        Returns
        -------
        out : torch.tensor
            The output.
        """
        out = self.mlp_layers(x)
        return out

class ECCLayer(nn.Module):
    """
    A layer for multiple ECC layers.
    """

    def __init__(self, dims, inner_dims, edge_dim):
        """
        Initialize the class.

        Parameters
        ----------
        dims : list of int
            The dimensions of the ECC layers.
        inner_dims : list of int
            The dimensions of the edge features mapping.
        edge_dim : int
            The dimension of the edge features.
        """
        super(ECCLayer, self).__init__()

        self.ecc_layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims, dims[1:]):
            inner_layers = []
            for inner_in_dim, inner_out_dim in zip([edge_dim] + inner_dims, inner_dims + [in_dim*out_dim]):
                inner_layers.append(nn.Linear(inner_in_dim,inner_out_dim))
            inner_sequential = nn.Sequential(*inner_layers)
            self.ecc_layers.append(gnn.ECConv(in_dim, out_dim, inner_sequential))

    def forward(self, x, adj, edge):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : torch.tensor
            The node features.
        adj : torch.tensor
            The adjacency matrix.
        edge : torch.tensor:
            The edge features.

        Returns
        -------
        x : torch.tensor
            The output.
        """
        for ecc_layer in self.ecc_layers:
            x = ecc_layer(x, adj, edge_attr=edge)
            x = F.relu(x)
        return x

class CNN1Layer(nn.Module):
    """
    A generic 1-dimensional convolutional layer.
    """

    def __init__(self, channels, kernel_size):
        """
        Initialize the class.

        Parameters
        ----------
        channels : list of int
            The channels to apply.
        kernel_size : int
            The size of the kernel.
        """
        super(CNN1Layer, self).__init__()

        layers = []
        for in_dim, out_dim in zip(channels, channels[1:]):
            layers.append(nn.Conv1d(in_dim,out_dim, kernel_size))
            layers.append(nn.BatchNorm1d(out_dim))
        self.cnn_layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data = nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def out_dim(self, l_in):
        """
        Compute the dimension of the output features.

        Parameters
        ----------
        l_in : int
            The dimension of the input features.

        Returns
        -------
        l_in : int
            The dimension of the output features.
        """
        for child in self.cnn_layers.children():
            if isinstance(child, nn.Conv1d) or isinstance(child, nn.Conv2d):
                ks, s, p, d = child.kernel_size[0], child.stride[0], child.padding[0], child.dilation[0]
                l_in = int((l_in + 2*p - d*(ks - 1) - 1)/s + 1)
        return l_in

    def forward(self, x, activation=None):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : torch.tensor
            The input tensor.
        activation : callable, None
            The activation function. The default is None.

        Returns
        -------
        out : torch.tensor
            The output.
        """
        out = self.cnn_layers(x)
        if activation is not None:
            out = activation(out)
        return out