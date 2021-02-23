import random
import time

import wandb
from typing import Dict, Any

import pandas as pd
import os.path as osp
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
import torch_geometric.transforms as T

from tqdm import tqdm
from model.mlp import LinkPredictor
from utils.parser import build_args
from utils.utils import *
from utils.logger import Logger
from model.utils import init_model

# necessary to flush on some nodes, setting it globally here
import functools
print = functools.partial(print, flush=True)


def training(
        args: Dict[str, Any], data: Any, data_edge_dict: Dict[str, Tensor], model: Module, predictor: Module,
        optimizer: torch.optim.Optimizer, evaluator: Module, early_stopping: Module
    ):
    print("Training start..")

    model.reset_parameters()
    predictor.reset_parameters()

    start_epoch = 1
    prev_best = 0.0
    full_start_time = time.time()
    for epoch in range(start_epoch, 1 + args.epochs):
        epoch_start_time = time.time()

        loss, pos_train_pred = train(model, predictor, data, data_edge_dict, optimizer, args.batch_size)
        wandb.log(
            {
                "[Train] Epoch": epoch,
                "[Train] Loss": loss,
                "[Train] Elapsed Time:": (time.time() - epoch_start_time)
            },
            commit=False,
        )

        if epoch % args.eval_steps == 0:
            valid_result = evaluation("valid", model, predictor, data, data_edge_dict, evaluator, args.batch_size, pos_train_pred)
            test_result = evaluation("test", model, predictor, data, data_edge_dict, evaluator, args.batch_size, pos_train_pred)

            if prev_best < valid_result["[Valid] Hits@50"]:
                prev_best = valid_result["[Valid] Hits@50"]
            wandb.log(valid_result, commit=False)
            early_stopping(prev_best)
        wandb.log({})
    print("done!")


def train(model, predictor, data, data_edge_dict, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = data_edge_dict["train"]["edge"].to(data.x.device)
    # if not torch.cuda.is_available():
    #     pos_train_edge = get_edge_pairs_small(pos_train_edge)

    train_dataloader = DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)
    total_loss = 0
    pos_train_preds = []
    for i, perm in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])

        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])

        loss = loss_func(pos_out, neg_out)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        pos_train_preds += [pos_out.squeeze().cpu()]

    loss /= len(train_dataloader)
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    return loss.item(), pos_train_pred


def loss_func(pos_score: Tensor, neg_score: Tensor) -> float:
    pos_loss = -torch.log(pos_score + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
    loss = pos_loss + neg_loss
    return loss


@torch.no_grad()
def evaluation(type, model, predictor, data, data_edge_dict, evaluator, batch_size, pos_train_pred):
    model.eval()
    predictor.eval()
    eval_start_time = time.time()

    h = model(data.x, data.adj_t)

    pos_edge = data_edge_dict[type]["edge"].to(h.device)
    neg_edge = data_edge_dict[type]["edge_neg"].to(h.device)

    # if not torch.cuda.is_available():
    #     pos_edge = get_edge_pairs_small(pos_edge)
    #     neg_edge = get_edge_pairs_small(neg_edge)

    pos_dataloader = DataLoader(range(pos_edge.size(0)), batch_size)
    neg_dataloader = DataLoader(range(neg_edge.size(0)), batch_size)

    pos_preds = []
    for perm in tqdm(pos_dataloader):
        edge = pos_edge[perm].t()
        pos_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for perm in tqdm(neg_dataloader):
        edge = neg_edge[perm].t()
        neg_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_pred = torch.cat(neg_preds, dim=0)

    metrics_all = {}
    type_name = type.capitalize()
    for K in [10, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            "y_pred_pos": pos_pred,
            "y_pred_neg": neg_pred,
        })[f"hits@{K}"]
        metrics_all[f"[{type_name}] Hits@{K}"] = hits

    metrics_all[f"[{type_name}] Elapsed Time"] = time.time() - eval_start_time
    return metrics_all


def _create_dataset(args: Dict[str, Any], dataset_id: str):
    dataset = PygLinkPropPredDataset(name=dataset_id,
                                     root="../data" if not args.dir_data else args.dir_data)
    data = dataset[0] # Data(edge_index=[2, 42463862], x=[576289, 58])

    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = ToSparseTensor()(data, data.x.shape[0])

    data_edge_dict = dataset.get_edge_split()

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        val_edge_index = data_edge_dict['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    return data, data_edge_dict


def _set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.benchmark = True


def setup(args):
    device = cuda_if_available(args.device)

    dataset_id = "ogbl-collab"
    _create_dataset(args, dataset_id)
    data, data_edge_dict = _create_dataset(args, dataset_id)
    data = data.to(device)

    model = init_model(args, data, dataset_id, outdim=None)
    model = model.to(device)

    wandb.watch(model)

    predictor = LinkPredictor(args.hid_dim, args.hid_dim, 1, args.lp_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    evaluator = Evaluator(name=dataset_id)
    early_stopping = EarlyStopping("Accuracy", patience=args.patience)

    return data, data_edge_dict, model, predictor, optimizer, evaluator, early_stopping


def main():
    args = build_args("collab")
    assert args.model  # must not be empty for node property prediction
    _set_seed(args.seed)
    wandb.init(project="ogb-revisited", entity="hwang7520")
    wandb.config.update(args, allow_val_change=True)
    args = wandb.config
    setup(args)
    data, data_edge_dict, model, predictor, optimizer, evaluator, early_stopping = setup(args)
    training(args, data, data_edge_dict, model, predictor, optimizer, evaluator, early_stopping)


if __name__ == "__main__":
    main()