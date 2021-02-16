import random
import wandb
from typing import Dict, Any

import pandas as pd
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch import Tensor
from torch.nn import Module

from utils.parser import build_args
from utils.utils import *
from utils.logger import Logger
from model.utils import init_model

# necessary to flush on some nodes, setting it globally here
import functools
print = functools.partial(print, flush=True)


def training(
        args: Dict[str, Any], data_split_idx: Dict[str, Tensor], train_idx: Tensor, data: Any, model: Module, optimizer: torch.optim.Optimizer,
        evaluator: Module, early_stopping: Module
    ):
    print("Training start..")
    start_epoch = 1
    prev_best = 0.0
    for epoch in range(start_epoch, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer)
        wandb.log(
            {
                "[Train] Epoch": epoch,
                "[Train] Loss": loss,
            },
            commit=False,
        )
        if epoch % args.eval_steps == 0:
            result = evaluation(model, data, data_split_idx, evaluator)
            train_rocauc, valid_rocauc, test_rocauc = result

            if prev_best < valid_rocauc:
                prev_best = valid_rocauc

            wandb.log(
                {
                    "[Train] ROC AUC": train_rocauc,
                    "[Valid] ROC AUC": valid_rocauc,
                    "[Valid] Best ROC AUC": prev_best,
                },
                commit=False,
            )
            early_stopping(valid_rocauc)
        wandb.log({})
    print("done!")


def train(model, data, train_idx, optimizer):
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()

    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()

    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluation(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def _save_results(args):
    # save detailed results (per epoch)
    train_file = os.path.join(args.dir_results, "d_" + args.filename + '.csv')
    if not os.path.exists(train_file):
        with open(train_file, 'w') as f:
            f.write("fold,epoch,loss,train-rocauc,valid-rocauc,test-rocauc\n")

    # save results (per run/fold)
    res_file = os.path.join(args.dir_results, args.filename + '.csv')
    if not os.path.exists(res_file):
        with open(res_file, 'w') as f:
            f.write("fold,loss,bepoch,train-rocauc,valid-rocauc,test-rocauc\n")
    return train_file, res_file


def _create_dataset(args: Dict[str, Any], dataset_id: str):
    dataset = PygNodePropPredDataset(name=dataset_id, root="../data" if not args.dir_data else args.dir_data)
    data = dataset[0]

    data_split_idx = dataset.get_idx_split() # dict["train", "valid", "test"]
    if args.train_idx:
        print("train_idx:", args.train_idx)
        print("Using", args.train_idx)
        train_idx = pd.read_csv(os.path.join("../data", "{}_idx".format(dataset_id), args.train_idx + ".csv.gz"),
                                compression="gzip", header=None).values.T[0]
        data_split_idx["train"] = data_split_idx["train"][train_idx]

    if not torch.cuda.is_available():  # for local test runs only use a subset of nodes
        idx = get_edges_small_index(data.edge_index)
        data.edge_index = data.edge_index[:, idx]
        data.edge_attr = data.edge_attr[idx]
        data.node_species = data.node_species[:3000]
        data.y = data.y[:3000]
        for k in ['train', 'valid', 'test']:
            data_split_idx[k] = get_nodes_small(data_split_idx[k])
            if data_split_idx[k].shape[0] == 0:
                data_split_idx[k] = [1, 2, 3]

    train_idx = data_split_idx['train']
    return data, data_split_idx, train_idx


def _edge_to_node(data: Any):
    # Move edge features to node features.
    # need to get adj_t for that already now, to init x
    # copy is not needed since seems to only do a sort of the edge index
    data = ToSparseTensor()(data, data.num_nodes)  # we do the sparse transformation manually here since we above may change the edge_index using our train_idx
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    return data


def _set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.benchmark = True


def setup(args):
    device = cuda_if_available(args.device)

    # os.makedirs(args.dir_results, exist_ok=True)
    # os.makedirs(args.dir_save, exist_ok=True)
    #
    # save_args(args, os.path.join(args.dir_results, "a_" + args.filename + '.csv'))  # save arguments
    # train_file, res_file = _save_results(args)

    dataset_id = "ogbn-proteins"
    data, data_split_idx, train_idx = _create_dataset(args, dataset_id)
    train_idx = train_idx.to(device)

    data = _edge_to_node(data)
    data = data.to(device)

    model = init_model(args, data, dataset_id, outdim=112)
    model = model.to(device)

    wandb.watch(model)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    evaluator = Evaluator(name=dataset_id)
    early_stopping = EarlyStopping("Accuracy", patience=args.patience)
    # logger = Logger(args.runs, args)
    return train_idx, data, model, optimizer, evaluator, early_stopping, data_split_idx


def main():
    args = build_args("pro")
    assert args.model  # must not be empty for node property prediction
    _set_seed(args.seed)
    wandb.init(project="ogb-revisited", entity="hwang7520")
    wandb.config.update(args, allow_val_change=True)
    args = wandb.config
    train_idx, data, model, optimizer, evaluator, early_stopping, data_split_idx = setup(args)
    training(args, data_split_idx, train_idx, data, model, optimizer, evaluator, early_stopping)


if __name__ == "__main__":
    main()
