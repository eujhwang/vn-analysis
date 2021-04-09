from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Conv2d, BatchNorm1d, LeakyReLU, Softplus, ELU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn import AGNNConv
from torch_geometric.nn import APPNP


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, normalize=True, cached=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=normalize, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=normalize, cached=cached))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=normalize, cached=cached))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, K):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, hidden_channels, K=K)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels*heads, heads=heads))
        self.convs.append(GATConv(hidden_channels*heads, out_channels, heads=1))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, adj_t))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(
            GINConv(
                Sequential(
                    Linear(in_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels)
                )
            )
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers-1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels)
                    )
                )
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            if isinstance(conv, Linear):
                conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = self.batch_norms[i](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class VirtualNode(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_virtual_nodes, model,
                 rand_num=0, aggregation="sum", activation="relu", JK="last", normalize=True, cached=False):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if model == "gcn-v":
            self.convs.append(
                GCNConv(in_channels, hidden_channels, normalize=normalize, cached=cached))
            self.batch_norms.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, normalize=normalize, cached=cached))
                self.batch_norms.append(BatchNorm1d(hidden_channels))
            self.convs.append(
                GCNConv(hidden_channels, out_channels, normalize=normalize, cached=cached))
            self.batch_norms.append(BatchNorm1d(out_channels))

        elif model == "sage-v":
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.batch_norms.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.batch_norms.append(BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
            self.batch_norms.append(BatchNorm1d(out_channels))

        self.num_virtual_nodes = num_virtual_nodes
        self.virtual_node = torch.nn.Embedding(self.num_virtual_nodes, in_channels)
        torch.nn.init.constant_(self.virtual_node.weight.data, 0) # set the initial virtual node embedding to 0.

        self.rand_num = rand_num
        if self.rand_num > 0:
            assert self.rand_num < num_virtual_nodes
            self.rand_indices = torch.randperm(num_virtual_nodes)[:self.rand_num]
            self.num_virtual_nodes = self.rand_num

        if activation == "relu":
            activation_layer = ReLU()
        elif activation == "leaky":
            activation_layer = LeakyReLU()
        elif activation == "elu":
            activation_layer = ELU()
        else:
            raise ValueError(f"{activation} is unsupported at this time!")

        self.virtual_node_mlp = torch.nn.ModuleList()
        for i in range(self.num_virtual_nodes):
            self.virtual_node_mlp.append(
                Sequential(
                    Linear(in_channels, 2 * hidden_channels),
                    activation_layer,
                    torch.nn.LayerNorm(2 * hidden_channels),
                    Linear(2 * hidden_channels, hidden_channels),
                    activation_layer,
                    torch.nn.LayerNorm(hidden_channels),
                )
            )
        for layer in range(num_layers-2):
            for i in range(self.num_virtual_nodes):
                self.virtual_node_mlp.append(
                    Sequential(
                        Linear(hidden_channels, 2*hidden_channels),
                        activation_layer,
                        torch.nn.LayerNorm(2*hidden_channels),
                        Linear(2*hidden_channels, hidden_channels),
                        activation_layer,
                        torch.nn.LayerNorm(hidden_channels),
                    )
                )
        self.aggregation = aggregation
        self.JK = JK
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        """
        x:              [# of nodes, # of features]
        adj_t:          [# of nodes, # of nodes]
        virtual_node:   [# of virtual nodes, # of features]
        """
        # initialize virtual node to zero
        virtual_node = self.virtual_node(torch.zeros(self.num_virtual_nodes).to(torch.long).to(x.device))
        if self.rand_num > 0:
            virtual_node = virtual_node[self.rand_indices]

        embs = [x]
        for layer in range(self.num_layers):
            if self.aggregation == "sum":
                aggregated_virtual_node = virtual_node.sum(dim=0, keepdim=True)
            elif self.aggregation == "mean":
                aggregated_virtual_node = virtual_node.mean(dim=0, keepdim=True)
            elif self.aggregation == "max":
                aggregated_virtual_node = torch.max(virtual_node, dim=0, keepdim=True).values

            new_x = embs[layer] + aggregated_virtual_node      # add message from virtual node
            new_x = self.convs[layer](new_x, adj_t) # GCN layer
            new_x = self.batch_norms[layer](new_x)
            new_x = F.relu(new_x)
            new_x = F.dropout(new_x, p=self.dropout, training=self.training)

            embs.append(new_x)
            # update virtual node
            if layer < self.num_layers-1:
                # create a node that contains all graph nodes information
                # global_add_pool: [1, # of features]
                # virtual_node_tmp: [# of virtual nodes, # of features], virtual_node: [# of virtual nodes, # of features]
                virtual_node_tmp = global_add_pool(embs[layer], torch.zeros(1, dtype=torch.int64, device=x.device)) + virtual_node
                # mlp layer for each virtual node
                virtual_node_list = []
                for v in range(self.num_virtual_nodes):
                    virtual_node_mlp = self.virtual_node_mlp[v+layer*self.num_virtual_nodes](virtual_node_tmp[v].unsqueeze(0))
                    virtual_node_list.append(virtual_node_mlp)
                virtual_node = F.dropout(torch.cat(virtual_node_list, dim=0), self.dropout, training=self.training)


        if self.JK == "last":
            emb = embs[-1]
        elif self.JK == "sum":
            emb = 0
            for layer in range(1, self.num_layers):
                emb += embs[layer]
        return emb


class APPNP_Net(torch.nn.Module):
    # def __init__(self, num_features , num_classes):
    #     super(AGNNConv_Net, self).__init__()
    #     self.lin1 = torch.nn.Linear(num_features, 16)
    #     self.prop1 = AGNNConv(requires_grad=False)
    #     self.prop2 = AGNNConv(requires_grad=True)
    #     self.lin2 = torch.nn.Linear(16,num_classes)
    #     self.convs = torch.nn.ModuleList()
    #
    #
    # def forward(self, x, adj_t):
    #     x = F.dropout(x, training=self.training)
    #     x = F.relu(self.lin1(x))
    #     x = self.prop1(x, adj_t)
    #     x = self.prop2(x, adj_t)
    #     x = F.dropout(x, training=self.training)
    #     x = self.lin2(x)
    #     return F.log_softmax(x, dim=1)
    #
    # def reset_parameters(self):
    #     for conv in self.convs:
    #         conv.reset_parameters()
    def __init__(self, num_features , num_classes,args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(num_features, args.hidden)
        self.lin2 = Linear(args.hidden, num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.args=args
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj_t):
        x, edge_index = x, adj_t
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

class SGC_Net(torch.nn.Module):
    def __init__(self,num_features , num_classes):
        super(SGC_Net, self).__init__()
        self.conv1 = SGConv(
            num_features, num_classes, K=2, cached=True)
        self.convs = torch.nn.ModuleList()

    def forward(self, x, adj_t):
        x, edge_index = x, adj_t
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class SDC_Net(torch.nn.Module):
    def __init__(self,num_features , num_classes):
        super(SGC_Net, self).__init__()
        self.conv1 = SGConv(
            num_features, num_classes, K=2, cached=True)
        self.convs = torch.nn.ModuleList()

    def forward(self, x, adj_t):
        x, edge_index = x, adj_t
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class GDC_Net(torch.nn.Module):
    def __init__(self,num_features , num_classes,args,edge_weight):
        super(GDC_Net, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, num_classes, cached=True,
                             normalize=not args.use_gdc)
        self.args=args
        self.edge_weight=edge_weight
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, adj_t,edge_weight):
        x, edge_index, edge_weight = x, adj_t, self.edge_weight
        x = F.relu(self.conv1(x, edge_index, self.edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, self.edge_weight)
        return F.log_softmax(x, dim=1)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()