import pickle
from typing import Optional
import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU, Conv2d, BatchNorm1d, LeakyReLU, Softplus, ELU
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, global_add_pool, global_max_pool, \
    GlobalAttention
from torch_geometric.data import ClusterData, Data
from torch_cluster import graclus_cluster
from tqdm import tqdm

import time

from model.diff_pool_iterative import iterative_diff_pool

def get_conv_layer(name, in_channels, hidden_channels, out_channels, gcn_normalize, gcn_cached):
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


def get_graph_pooling(name):
    if name == "sum":
        pool = global_add_pool
    elif name == "mean":
        pool = global_mean_pool
    elif name == "max":
        pool = global_max_pool
    else:
        raise ValueError(f"graph pooling - {name} is unsupported at this time!")
    return pool


def iterative_graclus(num_cl, edge_index):
    # to test
    # num_cl = 1
    # edge_index = torch.LongTensor([[0,0,0,0,5,5,8,8],[1,2,3,4,6,7,9,10]])
    num_cl1 = max(max(edge_index[0]), max(edge_index[1])).item() + 1  # approx node num
    num_cl0 = num_cl1 + 1  # dummy
    edge_index1 = edge_index
    print("clusters originally:", num_cl1)

    n2cl = torch.LongTensor(list(range(num_cl1)))
    while num_cl1 > num_cl and num_cl1 < num_cl0:
        cts = torch.zeros(len(n2cl))
        # print("*"*20)
        # print("hoi", num_cl1)
        # print("ei:")
        # print(edge_index1[0].tolist())
        # print(edge_index1[1].tolist())
        n2cl1 = graclus_cluster(edge_index1[0], edge_index1[1])
        # update
        edge_index2 = edge_index1.clone()
        # print("ass:")
        # print([i for i in range(len(n2cl1))])
        # print([i.item() for i in n2cl1], "(learned)")
        # print(n2cl.tolist(), "(old)")
        for i in range(len(n2cl1)):
            # print(i, n2cl[n2cl == i])
            if n2cl[i].item() == i:
                n2cl[n2cl == i] = n2cl1[i].item()
            cts[n2cl[i]] += 1
            edge_index2[edge_index1 == i] = n2cl1[i].item()
        # print(n2cl.tolist(),"(new)")
        edge_index1 = edge_index2
        num_cl0 = num_cl1
        num_cl1 = sum(cts > 0).item()
        print("cluster progress from {} to {}".format(num_cl0, num_cl1))
    print("clusters finally:", num_cl1)
    print("requested were:", num_cl)
    # print(n2cl.tolist())
    # now assign ids from 0 to num_cluster
    c = 0  # first correct cluster id
    for i in range(len(n2cl1)):
        if i in n2cl:
            if i > c:
                n2cl[n2cl == i] = c
                c += 1
            elif i == c:
                c += 1
    with open('graclus_cluster.pickle', 'wb') as handle:
        pickle.dump(n2cl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return n2cl


def get_vn_index(name, num_ns, num_vns, num_vns_conn, edge_index):
    if name == "full":
        idx = torch.ones(num_vns, num_ns)
    elif name == "random":
        idx = torch.zeros(num_vns, num_ns)
        for i in range(num_ns):
            rand_indices = torch.randperm(num_vns)[:num_vns_conn]
            idx[rand_indices, i] = 1
    elif name == "random-f" or name == "diffpool" or name == "random-e":
        return None
    elif name == "graclus":
        n2cl = iterative_graclus(num_vns, edge_index)
        diff = num_ns - len(n2cl)
        if diff > 0:  # if there are some nodes with ids > max id in edge_index, we don't have cluster ids for them yet
            startid = max(n2cl).item() + 1
            n2cl = torch.cat([n2cl, torch.LongTensor([i for i in range(startid, startid+diff)])], dim=0)
        num_vns = max(n2cl).item() + 1
        idx = torch.zeros(num_vns, num_ns)
        for i in range(num_vns):
            idx[i][n2cl == i] = 1
    elif "metis" in name:
        clu = ClusterData(Data(edge_index=edge_index), num_parts=num_vns)
        idx = torch.zeros(num_vns, num_ns)
        for i in range(num_vns):
            idx[i][clu.perm[clu.partptr[i]:clu.partptr[i+1]]] = 1
    else:
        raise ValueError(f"{name} is unsupported at this time!")

    return idx == 1


class VNGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_nodes, edge_index,
                 model, num_vns=1, vn_idx="full",
                 aggregation="sum", graph_pool="sum", activation="relu", JK="last", gcn_normalize=True, gcn_cached=False,
                 use_only_last=False, num_clusters=0):
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
                    get_conv_layer(model, hidden_channels, hidden_channels, hidden_channels, gcn_normalize=gcn_normalize, gcn_cached=gcn_cached))
                self.batch_norms.append(BatchNorm1d(hidden_channels))
            self.convs.append(
                get_conv_layer(model, hidden_channels, hidden_channels, out_channels, gcn_normalize=gcn_normalize, gcn_cached=gcn_cached))
        self.batch_norms.append(BatchNorm1d(out_channels))

        self.num_nodes = num_nodes
        self.num_vns_conn = 1
        self.num_clusters = num_clusters
        self.vn_index_type = vn_idx
        # index[i] specifies which nodes are connected to VN i
        self.vn_index = get_vn_index(vn_idx, num_nodes, num_clusters if num_clusters > 0 else num_vns, 1, edge_index)
        self.cl_index = self.vn_index # this will be different later only for metis+
        self.num_virtual_nodes = self.vn_index.shape[0] if self.vn_index is not None and vn_idx != "metis+" else num_vns  # might be > as num_vns in case we cannot split into less with graclus...
        self.virtual_node = torch.nn.Embedding(self.num_virtual_nodes, in_channels)
        torch.nn.init.constant_(self.virtual_node.weight.data, 0)  # set the initial virtual node embedding to 0.

        self.graph_pooling_layer = get_graph_pooling(graph_pool)

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
        self.use_only_last = use_only_last

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def init_epoch(self):
        if self.vn_index_type == "random-e":
            self.vn_index = get_vn_index("random", self.num_nodes, self.num_virtual_nodes, self.num_vns_conn, None)
        elif self.vn_index_type == "metis+":
            # random assignment of clusters to VNs
            vn2cl = get_vn_index("random", self.num_clusters, self.num_virtual_nodes, self.num_vns_conn, None)
            # now propagate to node level
            self.vn_index = torch.zeros(self.num_virtual_nodes, self.num_nodes)
            for c in range(self.num_clusters):
                self.vn_index[vn2cl[:, c].nonzero(), self.cl_index[c]] = 1
                # somehow this did not work, see https://discuss.pytorch.org/t/slicing-and-assign/44707
                # self.vn_index[vn2cl[:, c]][:, self.cl_index[c]] = 1
            #     to test
            # for n in range(self.num_nodes):
            #     for c in range(self.num_clusters):
            #         for v in range(self.num_virtual_nodes):
            #             # print(self.cl_index[c, n], vn2cl[v, c], self.vn_index[v, n])
            #             if (int(self.cl_index[c, n]) * int(vn2cl[v, c])):
            #                 assert self.vn_index[v, n]

            self.vn_index = self.vn_index == 1

    def forward(self, data):
        """
        x:              [# of nodes, # of features]
        adj_t:          [# of nodes, # of nodes]
        virtual_node:   [# of virtual nodes, # of features]
        """
        x, adj_t = data.x, data.adj_t
        adj = data.adj if hasattr(data, "adj") else data.adj_t.t()

        # initialize virtual node to zero
        if self.num_virtual_nodes == 0:
            virtual_node = torch.zeros(1).to(x.device)
        else:
            virtual_node = self.virtual_node(torch.zeros(self.num_virtual_nodes).to(torch.long).to(x.device))

        if self.vn_index_type == "random-f":
            self.vn_index = get_vn_index("random", self.num_nodes, self.num_virtual_nodes, self.num_vns_conn, None)
        elif self.vn_index == None and self.vn_index_type == "diffpool":  # currengtlhy the second condition is always true
            self.vn_index = iterative_diff_pool(self.num_virtual_nodes, self.num_vns_conn, x, adj)

            # vn_index: [# of vns, # of nodes], vn_index.T: [# of nodes, # of vns]
            # vn_index.T.nonzero(): [# of nodes * vns_conn, 2]; [:, 0]: graph node index [:, 1]: virtual node index
        vn_indices = torch.nonzero(self.vn_index.T).to(x.device)

        embs = [x]
        for layer in range(self.num_layers):
            # select corresponding virtual node vector using vn_indices[:, 1]
            selected_vns = torch.index_select(virtual_node, 0, vn_indices[:, 1].to(torch.long))
            new_x = embs[layer] + selected_vns
            new_x = self.convs[layer](new_x, adj_t)  # GCN layer
            new_x = self.batch_norms[layer](new_x)
            new_x = F.relu(new_x)
            new_x = F.dropout(new_x, p=self.dropout, training=self.training)

            embs.append(new_x)
            # update virtual node
            if self.use_only_last:
                if layer == self.num_layers - 2 and self.num_virtual_nodes > 0:
                    # create a node that contains all graph nodes information
                    # global_add_pool: [1, # of features]
                    # virtual_node_tmp: [# of virtual nodes, # of features], virtual_node: [# of virtual nodes, # of features]
                    # embs[layer]: [# of nodes, # of features] -> [# of nodes, hid_dim] -> [# of nodes, hid_dim]
                    virtual_node_tmp_list = []
                    for v in range(self.num_virtual_nodes):
                        # [1, # of features] -> [1, hid_dim]
                        # select only related nodes using vn_index == 1
                        virtual_node_tmp = self.graph_pooling_layer(embs[layer][self.vn_index[v]],
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
            else:
                if layer < self.num_layers - 1 and self.num_virtual_nodes > 0:
                    # create a node that contains all graph nodes information
                    # global_add_pool: [1, # of features]
                    # virtual_node_tmp: [# of virtual nodes, # of features], virtual_node: [# of virtual nodes, # of features]
                    # embs[layer]: [# of nodes, # of features] -> [# of nodes, hid_dim] -> [# of nodes, hid_dim]
                    virtual_node_tmp_list = []
                    for v in range(self.num_virtual_nodes):
                        # [1, # of features] -> [1, hid_dim]
                        # select only related nodes using vn_index == 1
                        virtual_node_tmp = self.graph_pooling_layer(embs[layer][self.vn_index[v]],
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

        if self.JK == "last" or self.num_layers <= 1:
            emb = embs[-1]
        elif self.JK == "sum":
            emb = 0
            for layer in range(1, self.num_layers):
                emb += embs[layer]
        return emb

