import wandb
import logging

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


def init_logger(dataset_id, wandb_id):
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    log_dir = "./log/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging_path = log_dir + f"{dataset_id}_{timestamp}_{wandb_id.split('/')[-1]}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )

    logging.info("log file is saved at: %s" % os.path.abspath(logging_path))


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
    dataset_id = "ogbl-ddi"
    data_dir = Path(args.data_dir).expanduser()
    data, data_edge_dict, epoch_transform = create_dataset(args, dataset_id, data_dir)

    data = data.to(device)

    train_dataloader, valid_pos_dataloader, valid_neg_dataloader, test_pos_dataloader, test_neg_dataloader = create_dataloader(data_edge_dict, args.log_batch_size)

    model, predictor = init_model(args, data, dataset_id, outdim=None)
    model = model.to(device)
    predictor = predictor.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()) + (list(data.emb.parameters()) if hasattr(data, "emb") else []), lr=args.lr)

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

    return trainer


def main():
    args = build_args("ddi")
    assert args.model  # must not be empty for node property prediction
    if args.cross_valid:
        assert args.wandb_id != ""
        init_logger("ogbl-ddi", args.wandb_id)
        logger = logging.getLogger(__name__)

        api = wandb.Api()
        run = api.run(args.wandb_id)

        cross_fold_num = args.runs
        wandb.init()
        wandb.config.update(run.config, allow_val_change=True)
        args = wandb.config

        logger.warning(f"args: {args}")
        best_valid_scores, best_test_scores = [], []
        seed = args.seed  # initial seed

        for i in range(cross_fold_num):
            logger.warning("run: %d, seed: %d" % (i, seed))
            set_seed(seed)
            trainer = setup(args)
            best_metrics = trainer.train()
            seed = random.randint(0, 2 ** 32)
            best_valid_scores.append(best_metrics["best_valid"])
            best_test_scores.append(best_metrics["best_test"])
            logger.warning("best_valid_score: %f, test_score: %f, best_epoch: %d"
                        % (best_metrics["best_valid"], best_metrics["best_test"], best_metrics["best_epoch"]))

        best_valid_score_tensor = torch.tensor(best_valid_scores)
        best_test_score_tensor = torch.tensor(best_test_scores)
        logger.warning(f"Best Valid: {best_valid_score_tensor.mean():.2f} ± {best_valid_score_tensor.std():.2f}")
        logger.warning(f"Final Test: {best_test_score_tensor.mean():.2f} ± {best_test_score_tensor.std():.2f}")
    else:
        wandb.init()
        wandb.config.update(args, allow_val_change=True)
        args = wandb.config
        set_seed(args.seed)
        trainer = setup(args)
        trainer.train()


if __name__ == "__main__":
    main()