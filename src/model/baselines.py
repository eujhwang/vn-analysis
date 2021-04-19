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

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
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

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
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

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = self.batch_norms[i](x)
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

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, adj_t))
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

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class APPNP_Net(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, K, alpha, dropout):  #, normalize=True, cached=False):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K, alpha)  #, normalize=normalize, cached=cached)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
