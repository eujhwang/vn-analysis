from collections import defaultdict
from datetime import datetime
from pathlib import Path

import wandb
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from kg_train import KGTrainer, Evaluation
from model.kg_model import KGEModel
from utils.parser import kg_parse_args
from utils.utils import *
from dataloader import TrainDataset, TestDataset
from dataloader import BidirectionalOneShotIterator
from ogb.linkproppred import LinkPropPredDataset, Evaluator


def create_dataloader(args, data, data_edge_dict):
    """
    data["num_nodes_dict"]: dictionary. key = entity_type & value = # of entities
    ex)
        data["num_nodes_dict"] = {'disease': 10687, 'drug': 10533, 'function': 45085 ...}
    train_triples: 'head_type', 'head', 'relation', 'tail_type', 'tail'
    valid_triples: 'head_type', 'head', 'head_neg', 'relation', 'tail_type', 'tail', 'tail_neg'
    test_triples: 'head_type', 'head', 'head_neg', 'relation', 'tail_type', 'tail', 'tail_neg'
    ex)
        {'head_type': ['disease'], 'head': [3935], 'head_neg': [[3851, 3068, 1491]], 'relation': [0],
        'tail_type': ['protein'], 'tail': [1987], 'tail_neg': [[2143, 7551, 14670]]}
    """
    train_triples = data_edge_dict["train"]
    valid_triples = data_edge_dict["valid"]
    test_triples = data_edge_dict["test"]

    entity_dict = {}
    cur_idx = 0
    for key in data["num_nodes_dict"]:
        entity_dict[key] = (cur_idx, cur_idx + data["num_nodes_dict"][key])
        cur_idx += data["num_nodes_dict"][key]
    nentity = sum(data["num_nodes_dict"].values())
    nrelation = int(max(train_triples["relation"]))+1

    args.nentity = nentity
    args.nrelation = nrelation

    train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    for i in tqdm(range(len(train_triples["head"]))):
        head, relation, tail = train_triples["head"][i], train_triples["relation"][i], train_triples["tail"][i]
        head_type, tail_type = train_triples["head_type"][i], train_triples["tail_type"][i]
        train_count[(head, relation, head_type)] += 1
        train_count[(tail, -relation - 1, tail_type)] += 1
        train_true_head[(relation, tail)].append(head)
        train_true_tail[(head, relation)].append(tail)

    # Set training dataloader iterator
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, nentity, nrelation,
                     args.negative_sample_size, 'head-batch',
                     train_count, train_true_head, train_true_tail,
                     entity_dict),
        batch_size=2 ** args.log_train_batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples, nentity, nrelation,
                     args.negative_sample_size, 'tail-batch',
                     train_count, train_true_head, train_true_tail,
                     entity_dict),
        batch_size=2 ** args.log_train_batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    logging.info('#entity: %d, #relation: %d' % (nentity, nrelation))
    logging.info('#train: %d, #valid: %d, #test: %d' % (len(train_triples['head']), len(valid_triples['head']), len(test_triples['head'])))

    # Prepare dataloader for evaluation
    valid_dataloader_head = DataLoader(
        TestDataset(triples=valid_triples, args=args, mode='head-batch', random_sampling=False, entity_dict=entity_dict),
        batch_size=2**args.log_valid_batch_size,
        collate_fn=TestDataset.collate_fn
    )
    valid_dataloader_tail = DataLoader(
        TestDataset(triples=valid_triples, args=args, mode='tail-batch', random_sampling=False, entity_dict=entity_dict),
        batch_size=2**args.log_valid_batch_size,
        collate_fn=TestDataset.collate_fn
    )
    valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]

    test_dataloader_head = DataLoader(
        TestDataset(triples=test_triples, args=args, mode='head-batch', random_sampling=False, entity_dict=entity_dict),
        batch_size=2**args.log_valid_batch_size,
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_tail = DataLoader(
        TestDataset(triples=test_triples, args=args, mode='tail-batch', random_sampling=False, entity_dict=entity_dict),
        batch_size=2**args.log_valid_batch_size,
        collate_fn=TestDataset.collate_fn
    )
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    return train_iterator, valid_dataset_list, test_dataset_list, valid_triples, test_triples, entity_dict, nentity, nrelation


def create_model(args, evaluator, nentity, nrelation):
    model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        evaluator=evaluator
    )
    return model


def setup(args):
    device = cuda_if_available(args.device)
    dataset_id = args.dataset
    data_dir = Path(args.data_dir).expanduser()

    # make save folder
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    save_path = 'log/%s/%s/%s-%s/%s' % (dataset_id, args.model, args.hidden_dim, wandb.run.id, timestamp)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # # Write logs to checkpoint and console
    # set_logger(args)

    data, data_edge_dict, epoch_transform = create_dataset(args, dataset_id, data_dir)
    train_iterator, valid_dataset_list, test_dataset_list, valid_triples, test_triples, entity_dict, nentity, nrelation = create_dataloader(args, data, data_edge_dict)

    evaluator = Evaluator(name=args.dataset)
    model = create_model(args, evaluator, nentity, nrelation).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    early_stopping = EarlyStopping("Accuracy", patience=args.patience)

    evaluation = Evaluation(
        dataset_id=dataset_id,
        model=model,
        valid_dataset_list=valid_dataset_list,
        test_dataset_list=test_dataset_list,
        valid_triples=valid_triples,
        test_triples=test_triples,
        entity_dict=entity_dict,
        device=device,
    )

    trainer = KGTrainer(
        dataset_id=dataset_id,
        model=model,
        train_iterator=train_iterator,
        optimizer=optimizer,
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        evaluation=evaluation,
        early_stopping=early_stopping,
        device=device,
        wandb_id=wandb.run.id,
        negative_adversarial_sampling=args.negative_adversarial_sampling,
        uni_weight=args.uni_weight,
        regularization=args.reg,
        adversarial_temperature=args.adversarial_temperature
    )

    return trainer


def main():
    args = kg_parse_args()
    set_seed(args.seed)
    wandb.init()
    wandb.config.update(args, allow_val_change=True)
    args = wandb.config
    print(args)
    trainer = setup(args)
    trainer.train()

if __name__ == '__main__':
    main()