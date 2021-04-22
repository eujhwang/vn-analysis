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


class KGTrainer:
    def __init__(
        self,
        dataset_id: str,
        model: Module,
        train_iterator: DataLoader,
        optimizer: torch.optim.Optimizer,
        eval_steps:int,
        max_steps: int,
        learning_rate: float,
        evaluation: Evaluation,
        early_stopping: Module,
        device: torch.device,
        wandb_id: str,
        negative_adversarial_sampling: Optional[bool] = False,
        uni_weight: Optional[bool] = False,
        regularization: Optional[float] = 0.0,
        adversarial_temperature: Optional[float] = 0.0,
    ):

        self.dataset_id = dataset_id
        self.model = model
        self.train_iterator = train_iterator
        self.opt = optimizer
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.evaluation = evaluation
        self.best_score = 0.0
        self.device = device
        self.best_epoch = -1

        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.uni_weight = uni_weight
        self.regularization = regularization
        self.adversarial_temperature = adversarial_temperature

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        self.model_save_dir = "./saved_model/"
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        self.model_save_path = self.model_save_dir + f"{dataset_id}_{timestamp}_{wandb_id}.pt"

    def update_save_best_score(self, score: float, epoch: int):
        if self.best_score < score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.model_save_path)
            print("model is saved here: %s, best epoch: %s, best f1 score: %f"
                  % (self.model_save_path, self.best_epoch, self.best_score))

    def train(self):
        print("Training start..")
        warm_up_steps = self.max_steps // 2
        for step in tqdm(range(0, self.max_steps)):
            step_start_time = time.time()
            loss = self.model.train_step(self.model, self.opt, self.train_iterator, self.device, self.negative_adversarial_sampling,
                                  self.uni_weight, self.regularization, self.adversarial_temperature)

            if step >= warm_up_steps:
                self.learning_rate = self.learning_rate / 10
                print('Change learning_rate to %f at step %d' % (self.learning_rate, step))
                self.opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % self.eval_steps == 0:
                self.evaluation.evaluate()

            wandb.log(
                {
                    "[Train] Step": step,
                    "[Train] Loss": loss,
                    "[Train] Elapsed Time:": (time.time() - step_start_time)
                },
                commit=False,
            )
            wandb.log({})
        self.evaluation.evaluate()
        wandb.log({})
        print("done!")



class Evaluation:
    def __init__(
        self,
        dataset_id: str,
        model: Module,
        valid_dataset_list: List[DataLoader],
        test_dataset_list: List[DataLoader],
        valid_triples: Dict[str, Any],
        test_triples: Dict[str, Any],
        entity_dict: Dict[str, int],
        device: torch.device,
    ):
        self.dataset_id = dataset_id
        self.model = model
        self.valid_dataset_list = valid_dataset_list
        self.test_dataset_list = test_dataset_list
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.entity_dict = entity_dict
        self.device = device

    @torch.no_grad()
    def evaluate(self):
        print("Evaluation start...")
        eval_start_time = time.time()
        valid_metrics = self.model.test_step(self.model, self.valid_dataset_list, self.device)
        print("valid_metrics", valid_metrics)
        test_metrics = self.model.test_step(self.model, self.test_dataset_list, self.device)
        print("test_metrics", test_metrics)

        wandb.log({
            "[Valid] Hits@1": valid_metrics["hits@1_list"],
            "[Valid] Hits@3": valid_metrics["hits@3_list"],
            "[Valid] Hits@10": valid_metrics["hits@10_list"],
            "[Valid] MRR": valid_metrics["mrr_list"],
        }, commit=False)
        wandb.log({
            "[Test] Hits@1": test_metrics["hits@1_list"],
            "[Test] Hits@3": test_metrics["hits@3_list"],
            "[Test] Hits@10": test_metrics["hits@10_list"],
            "[Test] MRR": test_metrics["mrr_list"],
        }, commit=False)
        wandb.log({
            "[Eval] Elapsed Time": time.time() - eval_start_time,
        }, commit=False)
        print("done!")
        return valid_metrics, test_metrics


