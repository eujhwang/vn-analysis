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
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=2000 if torch.cuda.is_available() else 10)  # ogbn-pro default
    parser.add_argument('--patience', default=100, type=int)  # ogbn-pro default
    # parser.add_argument('--batch_size', type=int, default=16 * 1024)  # currently ogbl-ppa only
    parser.add_argument('--log_batch_size', type=int, default=16)  #ogbl-ppa: 14, ddi:16 only
    parser.add_argument('--data_dir', type=str, default="data")

    parser.add_argument('--model', type=str, default="gcn", choices=["gcn", "sage", "mlp"])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")

    parser.add_argument('--train_idx', type=str, default="", help="use train_idx files for ogbl-ppa")

    # LinkPredictor uses lp_layers
    parser.add_argument('--lp_layers', type=int, default=3)

    # ogbl-ppa uses node_embedding
    parser.add_argument('--use_node_embedding', action='store_true')

    # ogbl-collab uses node_embedding
    parser.add_argument('--use_valedges_as_input', action='store_true')



    # parser.add_argument('--use_node_embedding', type=int, default=0, choices=[0, 1])  # for mlp
    # parser.add_argument('--runs', type=int, default=10 if torch.cuda.is_available() else 3)
    # parser.add_argument('--dir_results', type=str, default="../r_{}".format(ident))
    # parser.add_argument('--dir_save', default="../s_{}".format(ident))
    # parser.add_argument('--filename', type=str, default="test")
    # parser.add_argument('--checkpointing', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--checkpoint', type=str, default="")