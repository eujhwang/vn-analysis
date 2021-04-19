import wandb

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
from model.mlp import LinkPredictor
from train import Trainer, Evaluation
from utils.parser import build_args
from utils.utils import *
from model.utils import init_model

# necessary to flush on some nodes, setting it globally here
import functools
print = functools.partial(print, flush=True)


def create_dataloader(data_edge_dict: Dict[str, Tensor], log_batch_size: int):
    pos_train_edge = data_edge_dict["train"]["edge"]
    train_dataloader = DataLoader(range(pos_train_edge.size(0)), 2 ** log_batch_size, shuffle=True)

    pos_valid_edge = data_edge_dict["valid"]["edge"]
    neg_valid_edge = data_edge_dict["valid"]["edge_neg"]
    valid_pos_dataloader = DataLoader(range(pos_valid_edge.size(0)), 2 ** log_batch_size)
    valid_neg_dataloader = DataLoader(range(neg_valid_edge.size(0)), 2 ** log_batch_size)

    pos_test_edge = data_edge_dict["test"]["edge"]
    neg_test_edge = data_edge_dict["test"]["edge_neg"]
    test_pos_dataloader = DataLoader(range(pos_test_edge.size(0)), 2 ** log_batch_size)
    test_neg_dataloader = DataLoader(range(neg_test_edge.size(0)), 2 ** log_batch_size)

    return train_dataloader, valid_pos_dataloader, valid_neg_dataloader, test_pos_dataloader, test_neg_dataloader


def setup(args):
    device = cuda_if_available(args.device)
    dataset_id = "ogbl-ppa"
    data_dir = Path(args.data_dir).expanduser()
    data, data_edge_dict = create_dataset(args, dataset_id, data_dir)
    print("data:", data)

    data = data.to(device)

    train_dataloader, valid_pos_dataloader, valid_neg_dataloader, test_pos_dataloader, test_neg_dataloader = create_dataloader(data_edge_dict, args.log_batch_size)

    model, predictor = init_model(args, data, dataset_id, outdim=None)
    model = model.to(device)
    predictor = predictor.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    early_stopping = EarlyStopping("Accuracy", patience=args.patience)

    evaluation = Evaluation(
        dataset_id=dataset_id,
        model=model,
        predictor=predictor,
        data=data,
        data_edge_dict=data_edge_dict,
        valid_pos_dataloader=valid_pos_dataloader,
        valid_neg_dataloader=valid_neg_dataloader,
        test_pos_dataloader=test_pos_dataloader,
        test_neg_dataloader=test_neg_dataloader,
    )

    trainer = Trainer(
        dataset_id=dataset_id,
        data=data,
        data_edge_dict=data_edge_dict,
        model=model,
        model_type=args.model,
        predictor=predictor,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        evaluation=evaluation,
        early_stopping=early_stopping,
        epochs=args.epochs,
        eval_steps=args.eval_steps,
        device=device,
        wandb_id=wandb.run.id,
    )

    return trainer


def main():
    args = build_args("ppa")
    print("args:", args)
    assert args.model  # must not be empty for node property prediction
    set_seed(args.seed)
    wandb.init()
    wandb.config.update(args, allow_val_change=True)
    args = wandb.config
    trainer = setup(args)
    trainer.train()


if __name__ == "__main__":
    main()