from __future__ import annotations

import datetime
from pathlib import Path
from typing import *

import torch
import time
import wandb
from ogb.linkproppred import Evaluator
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from loss import loss_func


class Trainer:
    def __init__(
        self,
        dataset_id: str,
        data: Any,
        data_edge_dict: Dict[str, Tensor],
        model: Module,
        model_type: str,
        predictor: Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs:int,
        eval_steps:int,
        evaluation: Evaluation,
        early_stopping: Module,
        device: torch.device,
        wandb_id: str,
    ):

        self.dataset_id = dataset_id
        self.data = data
        self.data_edge_dict = data_edge_dict
        self.model = model
        self.model_type = model_type
        self.predictor = predictor
        self.train_dataloader = train_dataloader
        self.opt = optimizer
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.early_stopping = early_stopping
        self.evaluation = evaluation
        self.best_score = 0.0
        self.device = device
        self.best_epoch = -1

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        self.model_save_dir = "./saved_model/"
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        self.model_save_path = self.model_save_dir + f"{dataset_id}_{timestamp}_{wandb_id}.pt"

    def update_save_best_score(self, score: float, epoch: int):
        if self.best_score < score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.model, self.model_save_path)
            print("model is saved here: %s, best epoch: %s, best f1 score: %f"
                  % (self.model_save_path, self.best_epoch, self.best_score))

    def train(self):
        print("Training start..")

        self.model.reset_parameters()
        self.predictor.reset_parameters()
        pos_train_edge = self.data_edge_dict["train"]["edge"].to(self.device) # shape: [21231931, 2]

        for epoch in range(1, 1 + self.epochs):
            print("\n===================== Epoch (%d / %d) =====================" % (epoch, self.epochs))
            epoch_start_time = time.time()
            self.model.train()
            self.predictor.train()

            total_loss = []
            pos_train_preds = []
            for i, perm in enumerate(tqdm(self.train_dataloader)):
                # perm: [batch_size]; [16384]
                self.opt.zero_grad()
                h = self.model(self.data.x, self.data.adj_t)

                edge = pos_train_edge[perm].t()
                pos_out = self.predictor(h[edge[0]], h[edge[1]])

                edge = torch.randint(0, self.data.num_nodes, edge.size(), dtype=torch.long, device=h.device)
                neg_out = self.predictor(h[edge[0]], h[edge[1]])

                loss = loss_func(pos_out, neg_out)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

                self.opt.step()

                total_loss.append(loss.item())
                pos_train_preds += [pos_out.squeeze().cpu()]

            final_loss = sum(total_loss) / len(total_loss)
            pos_train_pred = torch.cat(pos_train_preds, dim=0)
            wandb.log(
                {
                    "[Train] Epoch": epoch,
                    "[Train] Loss": final_loss,
                    "[Train] Elapsed Time:": (time.time() - epoch_start_time)
                },
                commit=False,
            )

            if epoch % self.eval_steps == 0:
                metrics = self.evaluation.evaluate(pos_train_pred)
                self.update_save_best_score(metrics["[Valid] Hits@100"], epoch)
                metrics["[Valid] Best Hits@100"] = self.best_score
                print("metrics", metrics)
                wandb.log(metrics, commit=False)
                self.early_stopping(self.best_score)
            wandb.log({})

        metrics = self.evaluation.evaluate(pos_train_pred)
        self.update_save_best_score(metrics["[Valid] Hits@100"], epoch)
        metrics["[Valid] Best Hits@100"] = self.best_score
        print("metrics", metrics)
        wandb.log(metrics, commit=False)
        wandb.log({})
        print("done!")


class Evaluation:
    def __init__(
        self,
        dataset_id: str,
        model: Module,
        predictor: Module,
        data: Any,
        data_edge_dict: Dict[str, Tensor],
        valid_pos_dataloader: DataLoader,
        valid_neg_dataloader: DataLoader,
        test_pos_dataloader: DataLoader,
        test_neg_dataloader: DataLoader
    ):
        self.dataset_id = dataset_id
        self.model = model
        self.predictor = predictor
        self.data = data
        self.data_edge_dict = data_edge_dict
        self.valid_pos_dataloader = valid_pos_dataloader
        self.valid_neg_dataloader = valid_neg_dataloader
        self.test_pos_dataloader = test_pos_dataloader
        self.test_neg_dataloader = test_neg_dataloader

        self._evaulator = Evaluator(name=dataset_id)

    @torch.no_grad()
    def evaluate(self, pos_train_pred=None):
        print("Evaluation start...")
        eval_start_time = time.time()
        self.model.eval()
        self.predictor.eval()

        h = self.model(self.data.x, self.data.adj_t)

        pos_valid_edge = self.data_edge_dict['valid']['edge'].to(h.device)
        neg_valid_edge = self.data_edge_dict['valid']['edge_neg'].to(h.device)
        pos_test_edge = self.data_edge_dict['test']['edge'].to(h.device)
        neg_test_edge = self.data_edge_dict['test']['edge_neg'].to(h.device)

        pos_valid_preds = []
        for perm in self.valid_pos_dataloader:
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in self.valid_neg_dataloader:
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in self.test_pos_dataloader:
            edge = pos_test_edge[perm].t()
            pos_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in self.test_neg_dataloader:
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        results = {}
        for K in [10, 50, 100]:
            self._evaulator.K = K
            # dummy train, using valid
            train_hits = self._evaulator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            results[f"[Train] Hits@{K}"] = train_hits
            valid_hits = self._evaulator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            results[f"[Valid] Hits@{K}"] = valid_hits
            test_hits = self._evaulator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
            results[f"[Test] Hits@{K}"] = test_hits
            # results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
        results[f"[Eval] Elapsed Time"] = time.time() - eval_start_time
        return results

