import argparse
import random
import torch


def build_parser(ident):
    parser = argparse.ArgumentParser(description='our-system-name')
    register_default_args(parser, ident)
    return parser


def build_args(ident):
    parser = build_parser(ident)
    return parser.parse_args()


def register_default_args(parser, ident):
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=1)  # LEAVE 1 for now since logger not adapted for others yet
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500 if torch.cuda.is_available() else 30)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--log_batch_size', type=int, default=14)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--model', type=str, default="gcn",
                        choices=["mlp", "gcn", "sage", "mlp", "gat", "sgc", "gin",
                                 "appnp", "gcn-gdc", "sage-gdc", "gin-gdc", "pgnn",
                                 "gcn-vn", "sage-vn", "gin-vn"])
    parser.add_argument('--clusters', type=int, default=0)  # metis+
    parser.add_argument('--vns', type=int, default=0)
    parser.add_argument('--vns_conn', type=int, default=3)  # is ignored (always considered 1) with graclus currently
    parser.add_argument('--vn_idx', type=str, default="full", choices=["full", "random", "random-f", "random-e", "graclus", "diffpool", "metis", "metis+"])
    parser.add_argument('--K', type=int, default=10)  # different meanings in different models
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--approximate', dest='approximate', default=-1, type=int,
                        help='PGNN: k-hop shortest path distance. -1 means exact shortest path')  # -1, 2
    parser.add_argument('--anchors', type=int, default=64)  # PGNN

    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")

    parser.add_argument('--train_idx', type=str, default="train10",
                        help="train_idx files for ogbl-ppa. train50 : 50% of train data")

    # virtual nodes
    parser.add_argument('--JK', type=str, default="last", choices=["last", "sum"], help="how to combine nodes at the end")
    parser.add_argument('--activation', type=str, default="relu", choices=["relu", "leaky", "elu"],
                        help="activation layer for gnn-v relu: ReLU, leaky: LeakyReLU, elu: ELU")
    parser.add_argument('--aggregation', type=str, default="sum", choices=["sum", "mean", "max"],
                        help="aggregation for virtual nodes")
    parser.add_argument('--graph_pooling', type=str, default="sum", choices=["sum", "mean", "max"],
                        help="aggregation for graph nodes")

    # LinkPredictor uses lp_layers
    parser.add_argument('--lp_layers', type=int, default=3)

    # ogbl-ppa uses node_embedding
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--heads', type=int, default=5)

    # ogbl-collab uses node_embedding
    parser.add_argument('--use_valedges_as_input', type=int, default=0, help="0: false, 1: true")

    parser.add_argument('--gcn_normalize', type=int, default=1, help="0: false, 1: true")
    parser.add_argument('--gcn_cached', type=int, default=0, help="0: false, 1: true")
    parser.add_argument('--use_only_last', type=int, default=0, help="0: false, 1: true")

    parser.add_argument('--cross_valid', type=int, default=0, help="0: false, 1: true")
    parser.add_argument('--runs', type=int, default=1, help="# of runs")