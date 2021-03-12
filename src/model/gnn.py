from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Conv2d
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv, GINConv, global_mean_pool, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

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
        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConv(in_channels, hidden_channels, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(SGConv(hidden_channels, hidden_channels, K=K))
        self.convs.append(SGConv(hidden_channels, out_channels, K=K))
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


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels*heads, heads=heads))
        self.convs.append(GATConv(hidden_channels*heads, out_channels, heads=1))
        self.dropout = dropout
        print(self.convs)

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
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, pool_type="add"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(
                Sequential(
                    Linear(in_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels)
                )
            )
        )

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
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        self.pool_type = pool_type

    def reset_parameters(self):
        for conv in self.convs:
            if isinstance(conv, Linear):
                conv.reset_parameters()

    def forward(self, x, adj_t, batch=None):
        batch = torch.arange(0, x.shape[0]).to(x.device)
        for conv in self.convs:
            x = conv(x, adj_t)
            x = self.bn(x)
        if self.pool_type == "add":
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return x