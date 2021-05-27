from torch import Tensor
from torch.utils.data import DataLoader
from train import Trainer, Evaluation
from utils.parser import build_args
from utils.utils import *
from model.utils import init_model

# necessary to flush on some nodes, setting it globally here
import functools
print = functools.partial(print, flush=True)
logger = set_logger("ogbl-ddi")

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

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()) + (list(data.emb.parameters()) if hasattr(data, "emb") else []), lr=args.lr)

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
        patience=args.patience,
    )

    return trainer


def main():
    args = build_args("ddi")
    assert args.model  # must not be empty for node property prediction
    logger.info("args: %s" % args)
    set_seed(args.seed)
    trainer = setup(args)
    trainer.train()


if __name__ == "__main__":
    main()