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


def setup(args, model_state_dict=None, predictor_state_dict=None):
    device = cuda_if_available(args.device)
    dataset_id = "ogbl-ppa"
    data_dir = Path(args.data_dir).expanduser()
    data, data_edge_dict, epoch_transform = create_dataset(args, dataset_id, data_dir)
    print("data:", data)

    data = data.to(device)

    train_dataloader, valid_pos_dataloader, valid_neg_dataloader, test_pos_dataloader, test_neg_dataloader = create_dataloader(data_edge_dict, args.log_batch_size)

    model, predictor = init_model(args, data, dataset_id, outdim=None)

    if model_state_dict and predictor_state_dict:
        print("load_state_dict model and predictor")
        model.load_state_dict(model_state_dict)
        predictor.load_state_dict(predictor_state_dict)
    model = model.to(device)
    predictor = predictor.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

    evaluation = Evaluation(
        dataset_id=dataset_id,
        model=model,
        predictor=predictor,
        epoch_transform=epoch_transform,
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
        epoch_transform=epoch_transform,
        model=model,
        model_type=args.model,
        predictor=predictor,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        evaluation=evaluation,
        epochs=args.epochs,
        eval_steps=args.eval_steps,
        device=device,
        wandb_id=wandb.run.id,
        patience=args.patience,
    )

    return trainer, evaluation


def main():
    args = build_args("ppa")
    assert args.model  # must not be empty for node property prediction

    if args.cross_valid:
        assert args.wandb_id != ""
        logger = set_logger("ogbl-ppa", args.wandb_id)
        wandb.init(reinit=True)
        seed = args.seed
        api = wandb.Api()
        run = api.run(args.wandb_id)

        cross_fold_num = args.runs
        wandb.config.update(run.config, allow_val_change=True)
        wandb.config.update({"seed": seed}, allow_val_change=True)
        if "use_only_last" not in run.config.keys():
            wandb.config.update({"use_only_last": 0}, allow_val_change=True)
        if "clusters" not in run.config.keys():
            wandb.config.update({"clusters": 0}, allow_val_change=True)
        args = wandb.config

        logger.info(f"args: {args}")
        best_valid_scores, best_test_scores = [], []

        for i in range(cross_fold_num):
            logger.info("run: %d, seed: %d" % (i, args.seed))
            set_seed(args.seed)
            trainer, evaluation = setup(args)
            best_metrics = trainer.train()
            best_valid_scores.append(best_metrics["best_valid"])
            best_test_scores.append(best_metrics["best_test"])
            logger.info("best_valid_score: %f, test_score: %f, best_epoch: %d"
                        % (best_metrics["best_valid"], best_metrics["best_test"], best_metrics["best_epoch"]))

        best_valid_score_tensor = torch.tensor(best_valid_scores)
        best_test_score_tensor = torch.tensor(best_test_scores)
        logger.info(f"Best Valid: {best_valid_score_tensor.mean():.2f} ± {best_valid_score_tensor.std():.2f}")
        logger.info(f"Final Test: {best_test_score_tensor.mean():.2f} ± {best_test_score_tensor.std():.2f}")
    elif args.load_model:
        assert args.saved_model != ""
        assert args.wandb_id != ""
        logger = set_logger("ogbl-ppa", args.wandb_id)
        model_state_dict_path = Path(args.saved_model).expanduser()
        predictor_state_dict_path = Path(args.saved_predictor).expanduser()
        print("Loading model state dict...", end="", flush=True)
        model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
        print("done!")

        print("Loading predictor state dict...", end="", flush=True)
        predictor_state_dict = torch.load(predictor_state_dict_path, map_location="cpu")
        print("done!")
        wandb.init()

        api = wandb.Api()
        run = api.run(args.wandb_id)
        wandb.config.update(run.config, allow_val_change=True)
        wandb.config.update({"wandb_id": args.wandb_id}, allow_val_change=True)
        for key,value in sorted(vars(args).items()):
            if key not in wandb.config.keys():
                wandb.config.update({key: value}, allow_val_change=True)

        args = wandb.config
        logger.info(f"args: {args}")
        trainer, evaluation = setup(args, model_state_dict, predictor_state_dict)
        results = evaluation.evaluate()
        logger.info(results)
        wandb.log(results)
        trainer.train()

    else:
        wandb.init()
        wandb.config.update(args, allow_val_change=True)
        args = wandb.config
        set_seed(args.seed)
        trainer, evaluation = setup(args)
        trainer.train()


if __name__ == "__main__":
    main()