from pyvis.network import Network

from utils.structure import INT_TO_AA

class ProVisAdj:

    def __init__(self, adj, file_name, path='./'):
        self.adj = adj
        self.file_name = file_name
        self.path = path

    def visify(self):
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

    def __init__(self, x, adj=None, file_name='generated', bb_idx=2, inter_idx=1, path='./'):
        self.x = x
        self.adj = adj
        self.file_name = file_name
        self.bb_idx = bb_idx
        self.inter_idx = inter_idx
        self.path = path
        self._ss_colors = ["blue", "fuchsia", "crimson", "cyan", "darkgreen", "gold", "orange"]

    def vizify(self):
        # Build graph
        edge_map = {}
        n = self.x.shape[0]
        graph = 'graph {' + '\n' + 'rankdir="LR";' + '\n'
        for i in range(n):
            aa_num = int(self.x[i,0])
            ss_num = int(self.x[i,1])
            graph += str(i) + ' [label="' + INT_TO_AA[aa_num] + '", color=' + self._ss_colors[ss_num] + '];' + '\n'
        if self.adj is None:
            for i in range(n-1):
                graph += str(i) + '--' + str(i+1) + '\n'
        else:
            for i in range(n-1):
                for j in range(i+1,n):
                    edge_type = int(self.adj[i,j])
                    if edge_type == self.inter_idx and (i,j) not in edge_map:
                        graph += str(i) + '--' + str(j) + ' [style="dotted"]' + '\n'
                        edge_map[(i,j)] = edge_map[(j,i)] = self.inter_idx
                    elif edge_type == self.bb_idx and (i,j) not in edge_map:
                        graph += str(i) + '--' + str(j) + '\n'
                        edge_map[(i,j)] = edge_map[(j,i)] = self.bb_idx
        graph += '}'
        # Write to file
        file = open(self.path + self.file_name + '.dot', 'w')
        file.write(graph)
        file.close()