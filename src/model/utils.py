import torch
from model.gnn import GCN, SAGE, GAT, SGC, GIN, VirtualNode, APPNP_Net,SGC_Net,GDC_Net

from model.mlp import MLP


def init_model(args, data, dataset_id, outdim=None):
    model = None
    input_dim = data.num_features

    if outdim is None:
        outdim = args.hid_dim

    if dataset_id == "ogbl-ddi":
        input_dim = args.hid_dim

    if args.model == "mlp":
        model = MLP(input_dim, args.hid_dim, outdim, args.layers, args.dropout)
        if args.use_node_embedding:
            embedding = torch.load("model/embedding_{}.pt".format(dataset_id)).to(data.device)  # , map_location='cpu')
            data.x = torch.cat([data.x, embedding], dim=-1)
    elif args.model == "sage":
        model = SAGE(input_dim, args.hid_dim, outdim, args.layers, args.dropout)
    elif args.model == "gcn":
        if dataset_id == "ogbl-ppa":
            model = GCN(input_dim, args.hid_dim, outdim, args.layers, args.dropout, normalize=False, cached=False)
            precompute_norm(data)
        elif dataset_id == "ogbl-collab" or dataset_id == "ogbl-ddi":
            model = GCN(input_dim, args.hid_dim, outdim, args.layers, args.dropout, normalize=True, cached=True)
    elif args.model == "gat":
        model = GAT(input_dim, args.hid_dim, outdim, args.layers, args.heads, args.dropout)
    elif args.model == "sgc":
        model = SGC(input_dim, args.hid_dim, outdim, args.layers, args.dropout, args.K)
    elif args.model == "gin":
        model = GIN(input_dim, args.hid_dim, args.layers, args.dropout)
    elif args.model == "gcn-v" or args.model == "sage-v":
        model = VirtualNode(input_dim, args.hid_dim, outdim, args.layers, args.dropout, args.num_virtual_nodes, args.model,
                            rand_num=args.rand_num, aggregation=args.aggregation, activation=args.activation, JK=args.JK,
                            normalize=False, cached=False)
    elif args.model == "appnp":
        model = APPNP_Net(input_dim, args.hid_dim, args)
    elif args.model == "gdc":
        model = GDC_Net(input_dim, args.hid_dim,args,data.edge_weight)

    return model


def precompute_norm(data):
    # Pre-compute GCN normalization.
    adj = data.adj_t.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

