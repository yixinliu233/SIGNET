import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool
from torch_geometric.utils import softmax

scalar = 20
eps = 1e-10

def DHT(edge_index, batch, add_loops=True):
    num_edge = edge_index.size(1)
    device = edge_index.device

    ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
    edge_to_node_index = torch.arange(0, num_edge, 1, device=device).repeat_interleave(2).view(1, -1)
    hyperedge_index = edge_index.T.reshape(1, -1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()

    ### Transform batch of nodes to batch of edges
    edge_batch = hyperedge_index[1, :].reshape(-1, 2)[:, 0]
    edge_batch = torch.index_select(batch, 0, edge_batch)

    ### Add self-loops to each node in the dual hypergraph
    if add_loops:
        bincount = hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]] != 1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([torch.arange(0, num_edge, 1, device=device).view(1, -1),
                           torch.arange(max_edge + 1, max_edge + num_edge + 1, 1, device=device).view(1, -1)],
                          dim=0)

        hyperedge_index = torch.cat([hyperedge_index[:, mask], loops], dim=1)

    return hyperedge_index, edge_batch


class Explainer_MLP(torch.nn.Module):
    def __init__(self, num_features, dim, n_layers):
        super(Explainer_MLP, self).__init__()

        self.n_layers = n_layers
        self.mlps = torch.nn.ModuleList()

        for i in range(n_layers):
            if i:
                nn = Sequential(Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim))
            self.mlps.append(nn)

        self.final_mlp = Linear(dim, 1)


    def forward(self, x, edge_index, batch):

        for i in range(self.n_layers):
            x = self.mlps[i](x)
            x = F.relu(x)

        node_prob = self.final_mlp(x)
        node_prob = softmax(node_prob, batch)
        return node_prob


class Explainer_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, readout):
        super(Explainer_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.readout = readout

        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)

        if self.readout == 'concat':
            self.mlp = Linear(dim * num_gc_layers, 1)
        else:
            self.mlp = Linear(dim, 1)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            if i != self.num_gc_layers - 1:
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
            else:
                x = self.convs[i](x, edge_index)
            xs.append(x)

        if self.readout == 'last':
            node_prob = xs[-1]
        elif self.readout == 'concat':
            node_prob = torch.cat([x for x in xs], 1)
        elif self.readout == 'add':
            node_prob = 0
            for x in xs:
                node_prob += x

        node_prob = self.mlp(node_prob)
        node_prob = softmax(node_prob, batch)
        return node_prob


class Explainer_HGNN(torch.nn.Module):
    def __init__(self, input_dim, input_dim_edge, hidden_dim, num_gc_layers):
        super(Explainer_HGNN, self).__init__()

        self.num_node_features = input_dim
        if input_dim_edge:
            self.num_edge_features = input_dim_edge
            self.use_edge_attr = True
        else:
            self.num_edge_features = input_dim
            self.use_edge_attr = False
        self.nhid = hidden_dim
        self.num_convs = num_gc_layers
        self.convs = self.get_convs()

        self.mlp = Linear(hidden_dim*num_gc_layers, 1)

    def get_convs(self):

        convs = torch.nn.ModuleList()

        for i in range(self.num_convs):

            if i == 0:
                conv = HypergraphConv(self.num_edge_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)

            convs.append(conv)

        return convs


    def forward(self, x, edge_index, edge_attr, batch):

        if not self.use_edge_attr:
            a_, b_ = x[edge_index[0]], x[edge_index[1]]
            edge_attr = (a_ + b_) / 2
        hyperedge_index, edge_batch = DHT(edge_index, batch)

        xs = []
        # Message Passing
        for _ in range(self.num_convs):
            edge_attr = F.relu( self.convs[_](edge_attr, hyperedge_index))
            xs.append(edge_attr)

        edge_prob = self.mlp(torch.cat(xs, 1))
        edge_prob = softmax(edge_prob, edge_batch)

        return edge_prob


class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling, readout):
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout

        self.convs = torch.nn.ModuleList()
        self.dim = dim
        self.pool = self.get_pool()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)

            self.convs.append(conv)

    def forward(self, x, edge_index, batch, node_imp):

        if node_imp is not None:
            out, _ = torch_scatter.scatter_max(torch.reshape(node_imp.detach(), (1, -1)), batch)
            out = out.reshape(-1, 1)
            out = out[batch]
            node_imp /= out + eps
            node_imp = (2 * node_imp - 1)/(2 * scalar) + 1
            x = x * node_imp

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))

            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)

        return graph_emb, torch.cat(xs, 1)

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool


class HyperGNN(torch.nn.Module):

    def __init__(self, input_dim, input_dim_edge, hidden_dim, num_gc_layers, pooling, readout):

        super(HyperGNN, self).__init__()

        self.num_node_features = input_dim
        if input_dim_edge:
            self.num_edge_features = input_dim_edge
            self.use_edge_attr = True
        else:
            self.num_edge_features = input_dim
            self.use_edge_attr = False
        self.nhid = hidden_dim
        self.enhid = hidden_dim
        self.num_convs = num_gc_layers
        self.pooling = pooling
        self.readout = readout
        self.convs = self.get_convs()
        self.pool = self.get_pool()


    def forward(self, x, edge_index, edge_attr, batch, edge_imp):

        if not self.use_edge_attr:
            a_, b_ = x[edge_index[0]], x[edge_index[1]]
            edge_attr = (a_ + b_) / 2

        hyperedge_index, edge_batch = DHT(edge_index, batch)

        if edge_imp is not None:
            out, _ = torch_scatter.scatter_max(torch.reshape(edge_imp, (1, -1)), edge_batch)
            out = out.reshape(-1, 1)
            out = out[edge_batch]
            edge_imp /= out + eps
            edge_imp = (2 * edge_imp - 1)/(2 * scalar) + 1
            edge_attr = edge_attr * edge_imp

        xs = []

        for _ in range(self.num_convs):
            edge_attr = F.relu( self.convs[_](edge_attr, hyperedge_index))
            xs.append(edge_attr)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], edge_batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, edge_batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, edge_batch)

        return graph_emb, None

    def get_convs(self):
        convs = torch.nn.ModuleList()
        for i in range(self.num_convs):
            if i == 0:
                conv = HypergraphConv(self.num_edge_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)
            convs.append(conv)

        return convs

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))

        return pool
