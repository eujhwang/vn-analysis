from __future__ import annotations

import datetime
import os
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
from model.vngnn import VNGNN

class Trainer:
    def __init__(
        self,
        dataset_id: str,
        data: Any,
        data_edge_dict: Dict[str, Tensor],
        epoch_transform: Module,
        model: Module,
        model_type: str,
        predictor: Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs:int,
        eval_steps:int,
        evaluation: Evaluation,
        device: torch.device,
        wandb_id: str,
        patience: int,
    ):

        self.dataset_id = dataset_id
        self.data = data
        self.data_edge_dict = data_edge_dict
        self.epoch_transform = epoch_transform
        self.model = model
        self.model_type = model_type
        self.predictor = predictor
        self.train_dataloader = train_dataloader
        self.opt = optimizer
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.evaluation = evaluation
        self.best_valid_score = -1.0
        self.best_test_score = 0.0
        self.device = device
        self.best_epoch = -1
        self.patience = patience

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        self.model_save_dir = "./saved_model/"
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        self.model_save_path = self.model_save_dir + f"{dataset_id}_model_{timestamp}_{wandb_id}.pt"
        self.predictor_save_path = self.model_save_dir + f"{dataset_id}_predictor_{timestamp}_{wandb_id}.pt"

        self.node_info_save_dir = f"./saved_embeddings/{dataset_id}/"
        Path(self.node_info_save_dir).mkdir(parents=True, exist_ok=True)
        self.node_info_save_path = self.node_info_save_dir  + f"emb_{timestamp}_{wandb_id}.pkl"

    def update_save_best_score(self, valid_score: float, test_score: float, epoch: int, emb=None, vn_emb=None, vn_index=None):
        if self.best_valid_score < valid_score:
            self.best_valid_score = valid_score
            self.best_test_score = test_score
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.model_save_path)
            torch.save(self.predictor.state_dict(), self.predictor_save_path)
            print("model is saved here: %s, predictor saved path: %s, best epoch: %s, best valid f1 score: %f, best test f1 score: %f"
                  % (os.path.abspath(self.model_save_path), os.path.abspath(self.predictor_save_path), self.best_epoch, self.best_valid_score, self.best_test_score))
            # if emb is not None and vn_emb is not None and vn_index is not None:
            #     torch.save({"emb": emb, "vn_emb": vn_emb, "vn_index": vn_index}, self.node_info_save_path)
            #     print("emb will be saved here:", os.path.abspath(self.node_info_save_path))

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

            # if self.epoch_transform is not None:
            #     self.data = self.epoch_transform(self.data)

            if isinstance(self.model, VNGNN):
                self.model.init_epoch()

            total_loss = []
            pos_train_preds = []
            for i, perm in enumerate(tqdm(self.train_dataloader)):

                if self.epoch_transform is not None:  #TODO rename is no epoch transform anymore
                    self.data = self.epoch_transform(self.data)
                # perm: [batch_size]; [16384]
                self.opt.zero_grad()
                if isinstance(self.model, VNGNN):
                    h, vn_emb, vn_index = self.model(self.data)
                else:
                    h = self.model(self.data)

                edge = pos_train_edge[perm].t()
                pos_out = self.predictor(h[edge[0]], h[edge[1]])

                if self.dataset_id == "ogbl-ddi":
                    edge = negative_sampling(self.data.edge_index, num_nodes=self.data.x.size(0),
                                             num_neg_samples=perm.size(0), method="dense")
                else:
                    edge = torch.randint(0, self.data.num_nodes, edge.size(), dtype=torch.long, device=h.device)
                neg_out = self.predictor(h[edge[0]], h[edge[1]])

                loss = loss_func(pos_out, neg_out)
                loss.backward()

                if hasattr(self.data, "emb"):
                    torch.nn.utils.clip_grad_norm_(self.data.emb.parameters(), 1.0)
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

                if self.dataset_id == "ogbl-ppa":
                    self.update_save_best_score(metrics["[Valid] Hits@100"], metrics["[Test] Hits@100"], epoch)
                    metrics["[Valid] Best Hits@100"] = self.best_valid_score
                    metrics["[Test] Best Hits@100"] = self.best_test_score
                elif self.dataset_id == "ogbl-collab":
                    self.update_save_best_score(metrics["[Valid] Hits@50"], metrics["[Test] Hits@50"], epoch)
                    metrics["[Valid] Best Hits@50"] = self.best_valid_score
                    metrics["[Test] Best Hits@50"] = self.best_test_score
                elif self.dataset_id == "ogbl-ddi":
                    self.update_save_best_score(metrics["[Valid] Hits@20"], metrics["[Test] Hits@20"], epoch)
                    metrics["[Valid] Best Hits@20"] = self.best_valid_score
                    metrics["[Test] Best Hits@20"] = self.best_test_score
                print("metrics", metrics)
                wandb.log(metrics, commit=False)
                if (epoch - self.best_epoch) >= self.patience:
                    print(f"\nAccuracy has not changed in {self.patience} steps! Stopping the run after final evaluation...")
                    break
            wandb.log({})

        metrics = self.evaluation.evaluate(pos_train_pred)
        print("metrics", metrics)
        wandb.log(metrics, commit=False)
        wandb.log({})
        print("done!")

        return {"best_valid": self.best_valid_score, "best_test": self.best_test_score, "best_epoch": self.best_epoch}


class Evaluation:
    def __init__(
        self,
        dataset_id: str,
        model: Module,
        predictor: Module,
        epoch_transform: Module,
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
        self.epoch_transform = epoch_transform
        self._evaluator = Evaluator(name=dataset_id)

    @torch.no_grad()
    def evaluate(self, pos_train_pred=None):
        print("Evaluation start...")
        eval_start_time = time.time()
        self.model.eval()
        self.predictor.eval()

        if self.epoch_transform is not None:
            self.data = self.epoch_transform(self.data)

        if isinstance(self.model, VNGNN):
            self.model.init_epoch()

        if isinstance(self.model, VNGNN):
            h, _, _ = self.model(self.data)
        else:
            h = self.model(self.data)

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

        if self.dataset_id == "ogbl-collab":
            ei_saved = self.data.edge_index
            adj_saved = self.data.adj_t
            self.data.edge_index = self.data.full_edge_index
            self.data.adj_t = self.data.full_adj_t
            if isinstance(self.model, VNGNN):
                h, _, _ = self.model(self.data)
            else:
                h = self.model(self.data)
            self.data.edge_index = ei_saved
            self.data.adj_t = adj_saved

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
        for K in [10, 20, 30, 50, 100]:
            self._evaluator.K = K
            # dummy train, using valid
            if pos_train_pred is not None:
                train_hits = self._evaluator.eval({
                    'y_pred_pos': pos_train_pred,
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                results[f"[Train] Hits@{K}"] = train_hits
            valid_hits = self._evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            results[f"[Valid] Hits@{K}"] = valid_hits
            test_hits = self._evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
            results[f"[Test] Hits@{K}"] = test_hits
            # results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
        results[f"[Eval] Elapsed Time"] = time.time() - eval_start_time
        return results

