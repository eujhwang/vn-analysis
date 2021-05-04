import logging
import os
import random
import time
import warnings
from datetime import datetime

import torch
from pathlib import Path
from typing import *
from ogb.linkproppred import PygLinkPropPredDataset, LinkPropPredDataset
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import pandas as pd

from utils.to_dense import ToDense
from utils.to_sparse_tensor import ToSparseTensor
from model.pgnn_utils import PGNN_Transform
from model.kgnn_transform import TwoMalkin, ConnectedThreeMalkin

def save_args(args, fn):
    with open(fn, 'w') as f:
        for k, v in args.__dict__.items():
            f.write("{},{}\n".format(k, v))
    print("saved args:", args)


def create_checkpoint(checkpoint_fn, epoch, model, optimizer, results):
    checkpoint = {"epoch": epoch,
                  "model": model.state_dict() if model is not None else None,
                  "optimizer": optimizer.state_dict(),
                  "results": results}
    torch.save(checkpoint, checkpoint_fn)


def remove_checkpoint(checkpoint_fn):
    os.remove(checkpoint_fn)


def load_checkpoint(checkpoint_fn, model, optimizer):
    checkpoint = torch.load(checkpoint_fn)
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['results'], checkpoint['epoch'], model, optimizer


def load_checkpoint_results(checkpoint_fn):
    checkpoint = torch.load(checkpoint_fn)
    return checkpoint['results']


# helper method to test locally
def get_nodes_small(nodes):
    return nodes[(nodes < 3000)]


# helper method to test locally
# edge is in pyg's edge_index format
def get_edges_small_index(edge):
    return ((edge[0, :] < 3000) & (edge[1, :] < 3000))


# helper method to test locally
# edge is a sequence of edge pairs
def get_edge_pairs_small(edge):
    return edge[(edge[:, 0] < 3000) & (edge[:, 1] < 3000), :]


# helper method to test locally
def get_edges_small(edge):
    return edge[get_edges_small_index(edge)]


def cuda_if_available(device) -> torch.device:
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.benchmark = True


def create_dataset(args, dataset_id: str, data_dir: Union[Path, str]):

    epoch_transform = None
    if args.model.endswith("gdc"):
        # need to do this in between because it changes the edge_index which
        # we change using trainidx... and use to create adj_t in the sparse transform
        transform = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=args.alpha),
                    sparsification_kwargs=dict(method='topk', k=args.K,
                                               dim=0), exact=True)
    elif args.model.endswith("-vn") and args.vn_idx == "diffpool":
        # precompute attribute "adj"
        transform = ToDense(remove_edge_index=False)
    elif args.model == "pgnn":
        # precompute anchors, distances
        transform = PGNN_Transform(args.layers, args.anchors, args.approximate)
        epoch_transform = transform  # need to call this during training too
    elif args.model == "123gnn":
        # this just adds attributes to data based on edge_index
        transform = T.Compose([TwoMalkin(), ConnectedThreeMalkin()])
    else:  # do nothing
        transform = lambda x: x

    if dataset_id == "ogbl-ppa":
        dataset = PygLinkPropPredDataset(name=dataset_id, root=data_dir)
        # x: 576289 rows, each 58 dimension, edge_index: [191305, 261775] tensor 가 42463862 개
        data = dataset[0] # Data(edge_index=[2, 42463862], x=[576289, 58])
        data_edge_dict = dataset.get_edge_split()

        if args.train_idx:
            print(f"Using train_idx_{args.train_idx}")
            train_idx = pd.read_csv(os.path.join(data_dir, "{}_idx".format(dataset_id), args.train_idx + ".csv.gz"),
                                    compression="gzip", header=None).values.T[0]
            data_edge_dict['train']['edge'] = data_edge_dict['train']['edge'][train_idx]
            train_idx1 = [i * 2 for i in train_idx] + [(i * 2) + 1 for i in train_idx]
            data.edge_index = data.edge_index[:, train_idx1]

        data.x = data.x.to(torch.float)
        data = transform(data)
        data = ToSparseTensor(remove_edge_index=False)(data, data.x.shape[0])
    elif dataset_id == "ogbl-collab":
        dataset = PygLinkPropPredDataset(name='ogbl-collab')
        data = dataset[0]
        data = transform(data)
        edge_index = data.edge_index  # TODO VT not sure about this conceptually since we use diffusion merged with valid below
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
        data_edge_dict = dataset.get_edge_split()

        data.edge_index = edge_index # for graclus
        if args.use_valedges_as_input:
            val_edge_index = data_edge_dict["valid"]["edge"].t()
            data.full_edge_index = to_undirected(torch.cat([edge_index, val_edge_index], dim=-1))
            data.full_adj_t = SparseTensor.from_edge_index(data.full_edge_index).t()
            # data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_edge_index = data.edge_index
            data.full_adj_t = data.adj_t

    elif dataset_id == "ogbl-biokg":  # TODO add gdc later if needed
        dataset = LinkPropPredDataset(name='ogbl-biokg')
        data = dataset[0]
        data_edge_dict = dataset.get_edge_split()
    elif dataset_id == "ogbl-ddi":
        dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                         transform=T.Compose([transform, T.ToSparseTensor(remove_edge_index=False)]))
        data = dataset[0]

        device = cuda_if_available(args.device)
        data.emb = torch.nn.Embedding(data.num_nodes, args.hid_dim).to(device)
        torch.nn.init.xavier_uniform_(data.emb.weight)
        data.x = data.emb.weight  # VT needs to be tested, not fully sure if works like this
        data_edge_dict = dataset.get_edge_split()

    return data, data_edge_dict, epoch_transform


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def set_logger(dataset_id: str, wandb_id: str):
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    log_dir = "./log/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging_path = log_dir + f"{dataset_id}_{timestamp}_{wandb_id.split('/')[-1]}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("log file is saved at: %s" % os.path.abspath(logging_path))
    return logger

