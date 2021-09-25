import time
from typing import *

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from model.vngnn import VNGNN


class CustomEvaluation:
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

    def _eval_hits(self, y_pred_pos, y_pred_neg, K):
        '''
            compute Hits@K
            For each positive target node, the negative target nodes are the same.

            y_pred_neg is an array.
            rank y_pred_pos[i] against y_pred_neg for each i
        '''
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        return {'hits@{}'.format(K): hitsK}

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
        for K in [5, 10, 20, 30, 50, 100]:
            # dummy train, using valid
            if pos_train_pred is not None:
                train_hits = self._eval_hits(pos_train_pred, neg_valid_pred, K)[f'hits@{K}']
                results[f"[Train] Hits@{K}"] = train_hits
            valid_hits = self._eval_hits(pos_valid_pred, neg_valid_pred, K)[f'hits@{K}']
            results[f"[Valid] Hits@{K}"] = valid_hits
            test_hits = self._eval_hits(pos_test_pred, neg_test_pred, K)[f'hits@{K}']
            results[f"[Test] Hits@{K}"] = test_hits
        results[f"[Eval] Elapsed Time"] = time.time() - eval_start_time
        print("results:", results)
        return results