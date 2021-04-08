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
    parser.add_argument('--epochs', type=int, default=500 if torch.cuda.is_available() else 30)  # ogbn-pro default
    parser.add_argument('--patience', default=50, type=int)  # ogbn-pro default
    parser.add_argument('--log_batch_size', type=int, default=14)  #ogbl-ppa: 14, ddi:16 only
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--model', type=str, default="appnp",
                        choices=["mlp", "gcn", "sage", "mlp", "gat", "sgc", "gin", "gcn-v", "sage-v"])
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")

    parser.add_argument('--train_idx', type=str, default="train20",
                        help="train_idx files for ogbl-ppa. train50 : 50% of train data")
    # ogbl-ppa-SGC
    parser.add_argument('--K', type=int, default=1, help="K hop for SGC")

    # virtual nodes
    parser.add_argument('--JK', type=str, default="last", help="JK aggregation for gcn w/ virtual node")
    parser.add_argument('--activation', type=str, default="relu", choices=["relu","leaky", "elu"],
                        help="activation layer for gnn-v relu: ReLU, leaky: LeakyReLU, elu: ELU")
    parser.add_argument('--aggregation', type=str, default="sum", choices=["sum", "mean", "max"],
                        help="aggregation rule for virtual nodes")
    parser.add_argument('--num_virtual_nodes', type=int, default=1, help="number of virtual node for virtual node model")

    # LinkPredictor uses lp_layers
    parser.add_argument('--lp_layers', type=int, default=3)

    # ogbl-ppa uses node_embedding
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--heads', type=int, default=5)

    # ogbl-collab uses node_embedding
    parser.add_argument('--use_valedges_as_input', type=int, default=0, help="0: false, 1: true")


    # parser.add_argument('--use_node_embedding', type=int, default=0, choices=[0, 1])  # for mlp
    # parser.add_argument('--runs', type=int, default=10 if torch.cuda.is_available() else 3)
    # parser.add_argument('--dir_results', type=str, default="../r_{}".format(ident))
    # parser.add_argument('--dir_save', default="../s_{}".format(ident))
    # parser.add_argument('--filename', type=str, default="test")
    # parser.add_argument('--checkpointing', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--checkpoint', type=str, default="")

    #appnp
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.1)


def kg_parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data_dir', type=str, default='data', help='data directory path')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")
    parser.add_argument('--max_steps', default=100000, type=int)

    parser.add_argument('--dataset', type=str, default='ogbl-biokg', help='dataset name, default to biokg')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-double_entity', '--double_entity_embedding', action='store_true')
    parser.add_argument('-double_relation', '--double_relation_embedding', action='store_true')

    parser.add_argument('--negative_sample_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=500, type=int)
    parser.add_argument('--gamma', default=12.0, type=float)
    parser.add_argument('--negative_adversarial_sampling', action='store_true')
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('--log_train_batch_size', default=10, type=int) # 1024
    parser.add_argument('--reg', default=0.0, type=float, help="regularization")
    parser.add_argument('--log_valid_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000,
                        help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500,
                        help='number of negative samples when evaluating training triples')
    parser.add_argument('--patience', default=50, type=int)  # ogbn-pro default
    parser.add_argument('--eval_steps', default=1, type=int)  # ogbn-pro default

    return parser.parse_args()