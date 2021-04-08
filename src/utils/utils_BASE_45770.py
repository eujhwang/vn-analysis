import logging
import os
import random
import warnings
from pathlib import Path
from typing import *

import torch
from ogb.linkproppred import PygLinkPropPredDataset, LinkPropPredDataset
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import pandas as pd

class ToSparseTensor(object):
    r"""Converts the :obj:`edge_index` attribute of a data object into a
    (transposed) :class:`torch_sparse.SparseTensor` type with key
    :obj:`adj_.t`.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
    """
    def __init__(self, remove_edge_index: bool = True):
        self.remove_edge_index = remove_edge_index

    def __call__(self, data, num_nodes):
        assert data.edge_index is not None

        (row, col), N, E = data.edge_index, num_nodes, data.num_edges
        perm = (col * N + row).argsort()
        row, col = row[perm], col[perm]

        if self.remove_edge_index:
            data.edge_index = None

        value = None
        for key in ['edge_weight', 'edge_attr', 'edge_type']:
            if data[key] is not None:
                value = data[key][perm]
                if self.remove_edge_index:
                    data[key] = None
                break

        for key, item in data:
            if item.size(0) == E:
                data[key] = item[perm]

        data.adj_t = SparseTensor(row=col, col=row, value=value,
                                  sparse_sizes=(N, N), is_sorted=True)

        # Pre-process some important attributes.
        data.adj_t.storage.rowptr()
        data.adj_t.storage.csr2csc()

        return data


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
        data = ToSparseTensor()(data, data.x.shape[0])
    elif dataset_id == "ogbl-collab":
        dataset = PygLinkPropPredDataset(name='ogbl-collab')
        data = dataset[0]
        edge_index = data.edge_index
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
        data_edge_dict = dataset.get_edge_split()

        if args.use_valedges_as_input:
            val_edge_index = data_edge_dict["valid"]["edge"].t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t

    return data, data_edge_dict


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


class EarlyStoppingException(Exception):
    """Max Value Exceeded"""

    def __init__(self, condition: str):
        self.condition = condition

    def __str__(self):
        return f"EarlyStopping: {self.condition}"


class EarlyStopping:
    """
    Stop looping if a value is stagnant.
    name: str = "EarlyStopping value"
    patience: int = 10
    """
    def __init__(self, name: str, patience: int):
        self.name = name
        self.patience = patience
        self.count = 0
        self.value = 0

    def __call__(self, value):
        if value == self.value:
            self.count += 1
            if self.count >= self.patience:
                raise EarlyStoppingException(
                    f"{self.name} has not changed in {self.patience} steps."
                )
        else:
            self.value = value
            self.count = 0
