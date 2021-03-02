import os
import random
import warnings
from pathlib import Path
from typing import *

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
import torch_geometric.transforms as T


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
        data = dataset[0] # Data(edge_index=[2, 42463862], x=[576289, 58])

        data.x = data.x.to(torch.float)
        data = ToSparseTensor()(data, data.x.shape[0])
        data_edge_dict = dataset.get_edge_split()
    elif dataset_id == "ogbl-ddi":
        dataset = PygLinkPropPredDataset(name=dataset_id, root=data_dir, transform=T.ToSparseTensor())
        data = dataset[0] # Data(edge_index=[2, 42463862], x=[576289, 58])
        emb = torch.nn.Embedding(data.num_nodes, args.hid_dim)
        torch.nn.init.xavier_uniform_(emb.weight)
        data.x = emb.weight

        data_edge_dict = dataset.get_edge_split()
        idx = torch.randperm(data_edge_dict['train']['edge'].size(0))
        idx = idx[:data_edge_dict['valid']['edge'].size(0)]
        data_edge_dict['eval_train'] = {'edge': data_edge_dict['train']['edge'][idx]}

    return data, data_edge_dict


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
