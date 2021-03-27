from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Conv2d, BatchNorm1d, LeakyReLU, Softplus, ELU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv, GINConv, global_mean_pool, global_add_pool


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


class GCN_Virtual(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, activation="relu", JK="last", normalize=True, cached=False):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
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

        ### set the initial virtual node embedding to 0.
        self.virtual_node = torch.nn.Embedding(1, in_channels)
        torch.nn.init.constant_(self.virtual_node.weight.data, 0)

        self.virtual_node_mlp = torch.nn.ModuleList()
        if activation == "relu":
            activation_layer = ReLU()
        elif activation == "leaky":
            activation_layer = LeakyReLU()
        elif activation == "softplus":
            activation_layer = Softplus()
        elif activation == "elu":
            activation_layer = ELU()
        else:
            raise ValueError(f"{activation} is unsupported at this time!")

        self.virtual_node_mlp.append(
            Sequential(
                Linear(in_channels, 2 * hidden_channels),
                activation_layer,
                Linear(2 * hidden_channels, hidden_channels),
                activation_layer,
            )
        )
        for layer in range(num_layers-2):
            self.virtual_node_mlp.append(
                Sequential(
                    Linear(hidden_channels, 2*hidden_channels),
                    activation_layer,
                    Linear(2*hidden_channels, hidden_channels),
                    activation_layer,
                )
            )

        self.JK = JK
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        """
        x:              [# of nodes, # of features]
        adj_t:          [# of nodes, # of nodes]
        virtual_node:   [1, # of features]
        """
        # initialize virtual node to zero
        virtual_node = self.virtual_node(torch.zeros(1).to(torch.long).to(x.device))

        embs = [x]
        for layer in range(self.num_layers):
            new_x = embs[layer] + virtual_node      # add message from virtual node
            new_x = self.convs[layer](new_x, adj_t) # GCN layer
            new_x = self.batch_norms[layer](new_x)
            new_x = F.relu(new_x)
            new_x = F.dropout(new_x, p=self.dropout, training=self.training)

            embs.append(new_x)
            # update virtual node
            if layer < self.num_layers-1:
                # create a node that contains all graph nodes information
                # virtual_node_tmp: [1, # of features], virtual_node: [1, # of features]
                virtual_node_tmp = global_add_pool(embs[layer], torch.zeros(1, dtype=torch.int64, device=x.device)) + virtual_node
                virtual_node = self.virtual_node_mlp[layer](virtual_node_tmp)   # mlp layer
                virtual_node = F.dropout(virtual_node, self.dropout, training=self.training)

        if self.JK == "last":
            emb = embs[-1]
        elif self.JK == "sum":
            emb = 0
            for layer in range(1, self.num_layers):
                emb += embs[layer]
        return emb
