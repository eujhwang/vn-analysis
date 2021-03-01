import torch
from torch import Tensor


def loss_func(pos_score: Tensor, neg_score: Tensor) -> float:
    pos_loss = -torch.log(pos_score + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
    loss = pos_loss + neg_loss
    return loss