import random
import pandas as pd
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils.parser import build_args
from utils.utils import *
from utils.logger import Logger
from model.utils import init_model

# necessary to flush on some nodes, setting it globally here
import functools
print = functools.partial(print, flush=True)


def train(model, data, train_idx, optimizer):
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()

    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()

    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    args = build_args("pro")  # id to create results and save directory

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    os.makedirs(args.dir_results, exist_ok=True)
    os.makedirs(args.dir_save, exist_ok=True)

    # save arguments
    save_args(args, os.path.join(args.dir_results, "a_" + args.filename + '.csv'))

    # save detailed results (per epoch)
    train_file = os.path.join(args.dir_results, "d_" + args.filename + '.csv')
    if not os.path.exists(train_file):
        with open(train_file, 'w') as f:
            f.write("fold,epoch,loss,train-rocauc,valid-rocauc,test-rocauc\n")

    # save results (per run/fold)
    res_file = os.path.join(args.dir_results, args.filename + '.csv')
    if not os.path.exists(res_file):
        with open(res_file, 'w') as f:
            f.write("fold,loss,bepoch,train-rocauc,valid-rocauc,test-rocauc\n")

    ##################################################################
    dataset_id = 'ogbn-proteins'
    dataset = PygNodePropPredDataset(name=dataset_id, root="../data" if not args.dir_data else args.dir_data)

    data = dataset[0]

    assert args.model  # must not be empty for node property prediction

    split_idx = dataset.get_idx_split()
    if args.train_idx:  # select a subset of edges to train on
        print("Using", args.train_idx)
        train_idx = pd.read_csv(os.path.join("../data", "{}_idx".format(dataset_id), args.train_idx + ".csv.gz"),
                                compression="gzip", header=None).values.T[0]
        split_idx['train'] = split_idx['train'][train_idx]

    if not torch.cuda.is_available():  # for local test runs only use a subset of nodes
        idx = get_edges_small_index(data.edge_index)
        data.edge_index = data.edge_index[:, idx]
        data.edge_attr = data.edge_attr[idx]
        data.node_species = data.node_species[:3000]
        data.y = data.y[:3000]
        for k in ['train', 'valid', 'test']:
            split_idx[k] = get_nodes_small(split_idx[k])
            if split_idx[k].shape[0] == 0:
                split_idx[k] = [1, 2, 3]

    train_idx = split_idx['train'].to(device)

    ##################################################################
    # Move edge features to node features.
    # need to get adj_t for that already now, to init x
    # copy is not needed since seems to only do a sort of the edge index
    data = ToSparseTensor()(data, data.num_nodes)  # we do the sparse transformation manually here since we above may change the edge_index using our train_idx
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    data = data.to(device)
    ##################################################################

    model = init_model(args, data, dataset_id, 112).to(device)

    evaluator = Evaluator(name=dataset_id)
    logger = Logger(args.runs, args)

    ##################################################################
    start_run = 0
    checkpoint_fn = ""

    if args.checkpointing and args.checkpoint:
        s = args.checkpoint[:-3].split("_")
        start_run = int(s[-2])
        start_epoch = int(s[-1]) + 1

        checkpoint_fn = os.path.join(args.dir_save, args.checkpoint)  # need to remove it in any case

        if start_epoch > args.epochs:  # DISCARD checkpoint's model (ie not results), need a new model!
            args.checkpoint = ""
            start_run += 1
            logger = load_checkpoint_results(checkpoint_fn)

    ##################################################################

    for run in range(start_run, args.runs):

        torch.manual_seed(run)
        random.seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
            torch.backends.cudnn.benchmark = True

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        start_epoch = 1

        # overwrite some settings
        if args.checkpointing and args.checkpoint:
            # signal that it has been used
            args.checkpoint = ""
            loggers, start_epoch, model, optimizer = load_checkpoint(checkpoint_fn, model, optimizer)
            start_epoch += 1

        for epoch in range(start_epoch, 1 + args.epochs):
            # print(epoch)
            loss = train(model, data, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                train_rocauc, valid_rocauc, test_rocauc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_rocauc:.2f}%, '
                      f'Valid: {100 * valid_rocauc:.2f}% '
                      f'Test: {100 * test_rocauc:.2f}%')

                s = ",{:.2f},{:.2f},{:.2f}".format(train_rocauc * 100, valid_rocauc * 100, test_rocauc * 100)
                with open(train_file, 'a') as f:
                    f.write("{},{},{:.4f}".format(run, epoch, loss) + s + "\n")

                if args.checkpointing:
                    old_checkpoint_fn = checkpoint_fn
                    checkpoint_fn = '%s.pt' % os.path.join(args.dir_save,
                                                           args.filename + "_" + str(run) + "_" + str(epoch))
                    create_checkpoint(checkpoint_fn, epoch, model, optimizer, logger)
                    if osp.exists(old_checkpoint_fn):
                        remove_checkpoint(old_checkpoint_fn)

                best_val_epoch = logger.get_best_epoch(run)
                if args.patience > 0 and best_val_epoch + args.patience < epoch:
                    print("Early stopping!")
                    break

        epoch, train_rocauc, valid_rocauc, test_rocauc = logger.print_statistics(run)

        s = ",{},{:.2f},{:.2f},{:.2f}".format(epoch, train_rocauc, valid_rocauc, test_rocauc)
        with open(res_file, 'a') as f:
            f.write("{},{:.4f}".format(run, loss) + s + "\n")

    _, (train_rocauc, s1, valid_rocauc, s2, test_rocauc, s3) = logger.print_statistics()

    s = ",{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(train_rocauc, s1, valid_rocauc, s2, test_rocauc, s3)
    with open(res_file, 'a') as f:
        f.write(s[1:] + "\n")


if __name__ == "__main__":
    main()
