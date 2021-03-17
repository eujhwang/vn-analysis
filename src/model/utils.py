import torch
from model.gnn import GCN, SAGE
from model.mlp import MLP


def init_model(args, data, dataset_id, outdim=None):
    model = None
    if outdim is None:
        outdim = args.hid_dim

    if args.model == "mlp":
        model = MLP(data.num_features, args.hid_dim, outdim, args.layers,
                    args.dropout)
        if args.use_node_embedding:
            embedding = torch.load("model/embedding_{}.pt".format(dataset_id)).to(data.device)  # , map_location='cpu')
            data.x = torch.cat([data.x, embedding], dim=-1)
    elif args.model == "sage":
        model = SAGE(data.num_features, args.hid_dim, outdim,
                     args.layers, args.dropout)
    elif args.model == "gcn":
        model = GCN(data.num_features, args.hid_dim, outdim,
                    args.layers, args.dropout)
        precompute_norm(data)
    return model


def precompute_norm(data):
    # Pre-compute GCN normalization.
    adj = data.adj_t.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

