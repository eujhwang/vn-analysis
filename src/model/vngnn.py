from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Conv2d, BatchNorm1d, LeakyReLU, Softplus, ELU
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, global_add_pool
from torch_cluster import graclus_cluster


def get_conv_layer(name, in_channels, hidden_channels, out_channels, gcn_normalize=False, gcn_cached=False):
    if name.startswith("gcn"):
        return GCNConv(in_channels, out_channels, normalize=gcn_normalize, cached=gcn_cached)
    elif name.startswith("sage"):
        return SAGEConv(in_channels, out_channels)
    elif name.startswith("gin"):
        return GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_channels)
            )
        )
    else:
        raise ValueError(f"{name} is not supported at this time!")


def get_activation(name):
    if name == "relu":
        return ReLU()
    elif name == "leaky":
        return LeakyReLU()
    elif name == "elu":
        return ELU()
    else:
        raise ValueError(f"{name} is unsupported at this time!")


def get_vn_index(name, num_ns, num_vns, num_vns_conn, edge_index):
    idx = None
    if name == "full":
        idx = torch.ones(num_vns, num_ns)
    elif name == "random":
        idx = torch.zeros(num_vns, num_ns)
        for i in range(num_ns):
            rand_indices = torch.randperm(num_vns)[:num_vns_conn]
            idx[rand_indices, i] = 1
    # elif name == "graclus":
    #     idx = torch.zeros(num_vns, num_ns)
    #     cl = graclus_cluster(edge_index[0], edge_index[1], num_nodes=2)  #num_vns)
    #     for i in range(num_ns):
    #         rand_indices = torch.randperm(num_vns)[:num_vns_conn]
    #         idx[rand_indices, i] = 1
    else:
        raise ValueError(f"{name} is unsupported at this time!")

    return idx == 1


class VNGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_nodes, edge_index,
                 model, num_vns=1, num_vns_conn=1, vn_idx="full",  # maybe choose a better name for this parameter...
                 aggregation="sum", activation="relu", JK="last", gcn_normalize=True, gcn_cached=False):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(get_conv_layer(model, in_channels, hidden_channels, out_channels, gcn_normalize=gcn_normalize, gcn_cached=gcn_cached))
        else:
            self.convs.append(get_conv_layer(model, in_channels, hidden_channels, hidden_channels, gcn_normalize=gcn_normalize, gcn_cached=gcn_cached))
            self.batch_norms.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    get_conv_layer(model, in_channels, hidden_channels, hidden_channels, gcn_normalize=gcn_normalize, gcn_cached=gcn_cached))
                self.batch_norms.append(BatchNorm1d(hidden_channels))
            self.convs.append(
                get_conv_layer(model, in_channels, hidden_channels, out_channels, gcn_normalize=gcn_normalize, gcn_cached=gcn_cached))
        self.batch_norms.append(BatchNorm1d(out_channels))

        self.num_virtual_nodes = num_vns
        self.virtual_node = torch.nn.Embedding(self.num_virtual_nodes, in_channels)
        torch.nn.init.constant_(self.virtual_node.weight.data, 0)  # set the initial virtual node embedding to 0.
        # index[i] specifies which nodes are connected to VN i
        self.vn_index = get_vn_index(vn_idx, num_nodes, num_vns, num_vns_conn, edge_index)

        activation_layer = get_activation(activation)
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
        for layer in range(num_layers - 2):
            for i in range(self.num_virtual_nodes):
                self.virtual_node_mlp.append(
                    Sequential(
                        Linear(hidden_channels, 2 * hidden_channels),
                        activation_layer,
                        torch.nn.LayerNorm(2 * hidden_channels),
                        Linear(2 * hidden_channels, hidden_channels),
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

        embs = [x]
        for layer in range(self.num_layers):
            if self.aggregation == "sum":
                aggregated_virtual_node = virtual_node.sum(dim=0, keepdim=True)
            elif self.aggregation == "mean":
                aggregated_virtual_node = virtual_node.mean(dim=0, keepdim=True)
            elif self.aggregation == "max":
                aggregated_virtual_node = torch.max(virtual_node, dim=0, keepdim=True).values

            new_x = embs[layer] + aggregated_virtual_node  # add message from virtual node
            new_x = self.convs[layer](new_x, adj_t)  # GCN layer
            new_x = self.batch_norms[layer](new_x)
            new_x = F.relu(new_x)
            new_x = F.dropout(new_x, p=self.dropout, training=self.training)

            embs.append(new_x)
            # update virtual node
            if layer < self.num_layers - 1 and self.num_virtual_nodes > 0:
                # create a node that contains all graph nodes information
                # global_add_pool: [1, # of features]
                # virtual_node_tmp: [# of virtual nodes, # of features], virtual_node: [# of virtual nodes, # of features]
                # embs[layer]: [# of nodes, # of features] -> [# of nodes, hid_dim] -> [# of nodes, hid_dim]
                virtual_node_tmp_list = []
                for v in range(self.num_virtual_nodes):
                    # [1, # of features] -> [1, hid_dim]
                    # select only related nodes using vn_index == 1
                    virtual_node_tmp = global_add_pool(embs[layer][self.vn_index[v]].squeeze(),
                                                       torch.zeros(1, dtype=torch.int64, device=x.device))
                    virtual_node_tmp_list.append(virtual_node_tmp)
                virtual_node_tmp = torch.cat(virtual_node_tmp_list, dim=0) + virtual_node

                # mlp layer for each virtual node
                virtual_node_list = []
                for v in range(self.num_virtual_nodes):
                    virtual_node_mlp = self.virtual_node_mlp[v + layer * self.num_virtual_nodes](
                        virtual_node_tmp[v].unsqueeze(0))
                    virtual_node_list.append(virtual_node_mlp)
                virtual_node = F.dropout(torch.cat(virtual_node_list, dim=0), self.dropout, training=self.training)

        if self.JK == "last":
            emb = embs[-1]
        elif self.JK == "sum":
            emb = 0
            for layer in range(1, self.num_layers):
                emb += embs[layer]
        return emb

