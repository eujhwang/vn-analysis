import logging
import os
import os.path as osp
import random
import time
import warnings
from datetime import datetime

import torch
import numpy as np
from pathlib import Path
from typing import *
from ogb.linkproppred import PygLinkPropPredDataset, LinkPropPredDataset
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, train_test_split_edges, add_self_loops, negative_sampling
from torch_geometric.datasets import PPI, Planetoid
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def do_edge_split(data):
    """
    split edges into pos and neg, 50% for each
    Args:
        data: Data(x=..., y=..., edge_index=[...], num_nodes=...)
    Returns:
        Tensor: pos_edge_index, neg_edge_index
    """
    num_nodes = data.num_nodes
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]

    # positive edges 50% of edges -> positive
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    pos_edge_index = torch.stack([row, col], dim=0)

    # negative edges the rest 50% of edges -> negative
    neg_edge_index = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=row.size(0))

    return pos_edge_index, neg_edge_index


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
        data.num_nodes = data.adj_t.to_dense().shape[0]
        device = cuda_if_available(args.device)
        data.emb = torch.nn.Embedding(data.num_nodes, args.hid_dim).to(device)
        torch.nn.init.xavier_uniform_(data.emb.weight)
        data.x = data.emb.weight  # VT needs to be tested, not fully sure if works like this
        data_edge_dict = dataset.get_edge_split()

    elif dataset_id == "ogbl-ppi":
        device = cuda_if_available(args.device)

        train_data = PPI(root=args.data_dir, split="train", transform=T.ToSparseTensor(remove_edge_index=False))
        valid_data = PPI(root=args.data_dir, split="val", transform=T.ToSparseTensor(remove_edge_index=False))
        test_data = PPI(root=args.data_dir, split="test", transform=T.ToSparseTensor(remove_edge_index=False))

        data = Data(
            edge_index=torch.cat([train_data.data.edge_index, valid_data.data.edge_index, test_data.data.edge_index], dim=1),
            x=torch.cat([train_data.data.x, valid_data.data.x, test_data.data.x], dim=0),
            y=torch.cat([train_data.data.y, valid_data.data.y, test_data.data.y], dim=0)
        )

        # adding adj_t by converting it to SparseTensor
        data = T.ToSparseTensor(remove_edge_index=False)(data).to(device)

        train_pos_edge_index = train_data.data.edge_index
        valid_pos_edge_index, valid_neg_edge_index = do_edge_split(valid_data.data)
        test_pos_edge_index, test_neg_edge_index = do_edge_split(test_data.data)
        data_edge_dict = {
            "train": {"edge": train_pos_edge_index.t().to(device)},
            "valid": {"edge": valid_pos_edge_index.t().to(device), "edge_neg": valid_neg_edge_index.t().to(device)},
            "test": {"edge": test_pos_edge_index.t().to(device), "edge_neg": test_neg_edge_index.t().to(device)}
        }

    elif dataset_id == "ogbl-cora":
        device = cuda_if_available(args.device)
        dataset = Planetoid(args.data_dir, "Cora")
        data = T.ToSparseTensor(remove_edge_index=False)(dataset[0])
        splitted_data = train_test_split_edges(dataset[0], 0.05, 0.1)
        data_edge_dict = {
            "train": {"edge": splitted_data.train_pos_edge_index.t().to(device)},
            "valid": {"edge": splitted_data.val_pos_edge_index.t().to(device), "edge_neg": splitted_data.val_neg_edge_index.t().to(device)},
            "test": {"edge": splitted_data.test_pos_edge_index.t().to(device), "edge_neg": splitted_data.test_neg_edge_index.t().to(device)}
        }
    elif dataset_id == "ogbl-pubmed":
        device = cuda_if_available(args.device)
        dataset = Planetoid(args.data_dir, "PubMed")
        data = T.ToSparseTensor(remove_edge_index=False)(dataset[0])
        splitted_data = train_test_split_edges(dataset[0], 0.05, 0.1)
        data_edge_dict = {
            "train": {"edge": splitted_data.train_pos_edge_index.t().to(device)},
            "valid": {"edge": splitted_data.val_pos_edge_index.t().to(device), "edge_neg": splitted_data.val_neg_edge_index.t().to(device)},
            "test": {"edge": splitted_data.test_pos_edge_index.t().to(device), "edge_neg": splitted_data.test_neg_edge_index.t().to(device)}
        }

    return data, data_edge_dict, epoch_transform


def set_logger_(dataset_id: str):
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    log_dir = "./log/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging_path = log_dir + f"{dataset_id}_{timestamp}.log"

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("log file is saved at: %s" % os.path.abspath(logging_path))
    return logger


def set_logger(dataset_id: str, wandb_id: str):
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    log_dir = "./log/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging_path = log_dir + f"{dataset_id}_{timestamp}_{wandb_id.split('/')[-1]}.log"

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("log file is saved at: %s" % os.path.abspath(logging_path))
    return logger

