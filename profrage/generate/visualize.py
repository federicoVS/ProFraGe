from pyvis.network import Network

from utils.structure import INT_TO_AA

class ProVisAdj:
    """
    Visualizer class using the pyvis tool.

    The output is `.html` file depicting a graph with nodes and edges.

    Attributes
    ----------
    adj : torch.tensor
        The adjacency matrix. Note it should be symmetric.
    file_name : str
        The name of the output file. Note it should have no extension.
    path : str
        The path where to store the file.
    """

    def __init__(self, adj, file_name, path='./'):
        """
        Initialize the class.

        Parameters
        ----------
        adj : torch.tensor
            The adjacency matrix. Note it should be symmetric.
        file_name : str
            The name of the output file. Note it should have no extension.
        path : str, optional
            The path where to store the file. The default is './'.
        """
        self.adj = adj
        self.file_name = file_name
        self.path = path

    def visify(self):
        """
        Compute the visualization.

        Returns
        -------
        None
        """
        n = self.adj.shape[0]
        net = Network()
        for i in range(n):
            net.add_node(i+1)
        for i in range(n-1):
            for j in range(i+1,n):
                if self.adj[i,j] > 0:
                    net.add_edge(i+1, j+1)
        net.show_buttons(filter_=['nodes','edges'])
        net.show(self.path + self.file_name + '.html')

class ProViz:
    """
    Visualizer class using the GraphViz tool.

    The output is a `.dot` file depicting a graph with node, node features, edges, edge types, and edge features.

    Attributes
    ----------
    x : torch.tensor
        The node features.
    adj : torch.tensor
        The adjacency matrix. In case the graph has any edge feature, it should also contain the edge features.
    file_name : str
        The name of the output file. Note it should have no extension.
    bb_idx : int
        The ID to assign to edges representing backbone connections.
    inter_idx : int
        The ID to assign to edges representing interactions.
    path : str
        The path where to store the file.
    """

    def __init__(self, x, adj=None, file_name='generated', bb_idx=2, inter_idx=1, path='./'):
        """
        Initialize the class.

        Parameters
        ----------
        x : torch.tensor
            The node features.
        adj : torch.tensor
            The adjacency matrix. In case the graph has any edge feature, it should also contain the edge features.
        file_name : str, optional
            The name of the output file. Note it should have no extension. The default is 'generated'.
        bb_idx : int, optional
            The ID to assign to edges representing backbone connections. The default is 2.
        inter_idx : int
            The ID to assign to edges representing interactions. The default is 1.
        path : str, optional
            The path where to store the file. The default is './'.
        """
        self.x = x
        self.adj = adj
        if adj is not None and len(adj.size()) == 2:
            self.adj = adj.unsqueeze(2)
        self.file_name = file_name
        self.bb_idx = bb_idx
        self.inter_idx = inter_idx
        self.path = path
        self._ss_colors = ["blue", "fuchsia", "crimson", "cyan", "darkgreen", "gold", "orange"]

    def vizify(self):
        """
        Compute the visualization

        Returns
        -------
        None
        """
        # Build graph
        edge_map = {}
        n = self.x.shape[0]
        graph = 'graph {' + '\n' + 'rankdir="LR";' + '\n'
        for i in range(n):
            aa_num = int(self.x[i,0])
            ss_num = int(self.x[i,1])
            graph += str(i) + ' [label="' + INT_TO_AA[aa_num] + '", color=' + self._ss_colors[ss_num-1] + '];' + '\n'
        if self.adj is None:
            for i in range(n-1):
                graph += str(i) + '--' + str(i+1) + '\n'
        else:
            for i in range(n):
                for j in range(i):
                    edge_type = int(self.adj[i,j,0])
                    if self.adj.shape[2] == 2:
                        edge_dist = float(self.adj[i,j,1])
                    if edge_type == self.inter_idx and (i,j) not in edge_map:
                        graph += str(i) + '--' + str(j) + ' [style="dotted"]' + '\n'
                        edge_map[(i,j)] = edge_map[(j,i)] = self.inter_idx
                    elif edge_type == self.bb_idx and self.adj.shape[2] == 2 and (i,j) not in edge_map:
                        graph += str(i) + '--' + str(j) + ' [label="' + str(edge_dist) + '"]' + '\n'
                        edge_map[(i,j)] = edge_map[(j,i)] = self.bb_idx
                    elif edge_type == self.bb_idx and (i,j) not in edge_map:
                        graph += str(i) + '--' + str(j) + '\n'
                        edge_map[(i,j)] = edge_map[(j,i)] = self.bb_idx
        graph += '}'
        # Write to file
        file = open(self.path + self.file_name + '.dot', 'w')
        file.write(graph)
        file.close()