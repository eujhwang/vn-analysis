import os.path as osp
import pickle
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv

from model.diff_pool import dense_diff_pool

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


def iterative_diff_pool(num_clusters, numcl_p_n, x, adj, mask=None):

#     return
# 
# 
# class Net(torch.nn.Module):
#     def __init__(self, num_features, num_classes, max_nodes):
#         super(Net, self).__init__()
    num_features = x.shape[-1]
    num_nodes = min(num_clusters*4, x.shape[0])  #ceil(0.25 * max_nodes)
    gnn1_pool = GNN(num_features, 64, num_nodes).to(x.device)
    gnn1_embed = GNN(num_features, 64, 64, lin=False).to(x.device)

    num_nodes = num_clusters
    gnn2_pool = GNN(3 * 64, 64, num_nodes).to(x.device)
    # gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

    # gnn3_embed = GNN(3 * 64, 64, 64, lin=False)
    #
    # lin1 = torch.nn.Linear(3 * 64, 64)
    # lin2 = torch.nn.Linear(64, num_classes)

# def forward(self, x, adj, mask=None):
    s = gnn1_pool(x, adj, mask)
    x = gnn1_embed(x, adj, mask)

    x1, adj1, s1 = dense_diff_pool(x, adj, s, mask)

    s2 = gnn2_pool(x1, adj1)
    # don't need the diff pool in the last step
    # x = gnn2_embed(x, adj)
    # x, adj, s2 = dense_diff_pool(x, adj, s)
    s2 = torch.softmax(s2, dim=-1)

    n2cl = torch.mm(s1.squeeze(0), s2.squeeze(0))
    with open('diffpool_cluster_'+str(num_features)+'.pickle', 'wb') as handle:
        pickle.dump(n2cl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    topk, indices = torch.topk(n2cl, numcl_p_n, dim=-1, largest=True, sorted=True)

    idx = torch.zeros(num_clusters, x.shape[-2])
    for i in range(x.shape[-2]):
        idx[indices[i], i] = 1
    # x = gnn3_embed(x, adj)
    #
    # x = x.mean(dim=1)
    # x = F.relu(lin1(x))
    # x = lin2(x)
    return idx == 1  #F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
