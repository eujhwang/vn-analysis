import random
import pandas as pd
import os.path as osp
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from utils.utils import *
from utils.parser import build_args
from utils.logger import Logger
from model.utils import init_model
from model.mlp import LinkPredictor

# necessary to flush on some nodes, setting it globally here
import functools
print = functools.partial(print, flush=True)


def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    if not torch.cuda.is_available():
        pos_train_edge = get_edge_pairs_small(pos_train_edge)

    total_loss = total_examples = 0
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if model is not None and list(model.parameters()):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        pos_train_preds += [pos_out.squeeze().cpu()]

    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    return total_loss / total_examples, pos_train_pred


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, pos_train_pred):
    model.eval()

    h = model(data.x, data.adj_t)

    # pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)
    if not torch.cuda.is_available():
        pos_valid_edge = get_edge_pairs_small(pos_valid_edge)
        neg_valid_edge = get_edge_pairs_small(neg_valid_edge)
        pos_test_edge = get_edge_pairs_small(pos_test_edge)
        neg_test_edge = get_edge_pairs_small(neg_test_edge)

    # this consumes too much time...
    # pos_train_preds = []
    # for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
    #     edge = pos_train_edge[perm].t()
    #     pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    # pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        # dummy train, using valid
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    args = build_args("ppa")

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    os.makedirs(args.dir_results, exist_ok=True)
    os.makedirs(args.dir_save, exist_ok=True)

    save_args(args, os.path.join(args.dir_results, "a_" + args.filename + '.csv'))

    train_file = os.path.join(args.dir_results, "d_" + args.filename + '.csv')
    if not os.path.exists(train_file):
        with open(train_file, 'w') as f:
            f.write("fold,epoch,loss,train10,valid10,test10,train50,valid50,test50,train100,valid100,test100\n")

    res_file = os.path.join(args.dir_results, args.filename + '.csv')
    if not os.path.exists(res_file):
        with open(res_file, 'w') as f:
            f.write("fold,loss,bepoch,train10,valid10,test10,train50,valid50,test50,train100,valid100,test100\n")

    ##################################################################
    dataset_id = 'ogbl-ppa'
    dataset = PygLinkPropPredDataset(name=dataset_id, root="../data" if not args.dir_data else args.dir_data)

    data = dataset[0]
    data.x = data.x.to(torch.float)

    split_edge = dataset.get_edge_split()

    if args.train_idx and torch.cuda.is_available():
        print("Using", args.train_idx)
        train_idx = pd.read_csv(os.path.join("../data", "{}_idx".format(dataset_id), args.train_idx + ".csv.gz"),
                                compression="gzip", header=None).values.T[0]
        split_edge['train']['edge'] = split_edge['train']['edge'][train_idx]

        train_idx1 = [i*2 for i in train_idx] + [(i*2)+1 for i in train_idx]
        data.edge_index = data.edge_index[:, train_idx1]

    ##################################################################

    data = data.to(device)
    data = ToSparseTensor()(data, data.x.shape[0])   # we do the sparse transformation manually here since we above may change the edge_index using our train_idx

    ##################################################################

    model = init_model(args, data, dataset_id)
    model = model.to(device)

    predictor = LinkPredictor(args.hid_dim, args.hid_dim, 1, args.lp_layers, args.dropout).to(device)

    evaluator = Evaluator(name=dataset_id)

    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

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
            loggers = load_checkpoint_results(checkpoint_fn)

    ##################################################################

    for run in range(start_run, args.runs):

        torch.manual_seed(run)
        random.seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
            torch.backends.cudnn.benchmark = True

        model.reset_parameters()
        predictor.reset_parameters()

        optimizer = torch.optim.Adam((list(model.parameters()) if model is not None else []) + list(predictor.parameters()), lr=args.lr)

        start_epoch = 1

        # overwrite some settings
        if args.checkpointing and args.checkpoint:
            # signal that it has been used
            args.checkpoint = ""
            loggers, start_epoch, model, optimizer = load_checkpoint(checkpoint_fn, model, optimizer)
            start_epoch += 1

        for epoch in range(start_epoch, 1 + args.epochs):
            print("epoch: %d", epoch)
            old_checkpoint_fn = checkpoint_fn
            checkpoint_fn = '%s.pt' % os.path.join(args.dir_save, args.filename + "_" + str(run) + "_" + str(epoch))

            loss, pos_train_pred = train(model, predictor, data, split_edge, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator, args.batch_size, pos_train_pred)

                for key, result in results.items():
                    loggers[key].add_result(run, result)

                s = ""
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    s += ",{:.2f},{:.2f},{:.2f}".format(train_hits * 100, valid_hits * 100, test_hits * 100)

                    print(key + ("" if key[-2] == "0" else " ") + f' Run: {run + 1:02d}, '
                                                                  f'Epoch: {epoch:02d}, '
                                                                  f'Loss: {loss:.4f}, '
                                                                  f'Train: {100 * train_hits:.2f}%, '
                                                                  f'Valid: {100 * valid_hits:.2f}%, '
                                                                  f'Test: {100 * test_hits:.2f}%')

                with open(train_file, 'a') as f:
                    f.write("{},{},{:.4f}".format(run, epoch, loss) + s + "\n")

                if args.checkpointing:
                    create_checkpoint(checkpoint_fn, epoch, model, optimizer, loggers)
                    if osp.exists(old_checkpoint_fn):
                        remove_checkpoint(old_checkpoint_fn)

                best_val_epoch = loggers['Hits@100'].get_best_epoch(run)
                if args.patience > 0 and best_val_epoch + args.patience < epoch:
                    print("Early stopping!")
                    break

        s = ""
        for key in loggers.keys():
            print(key)
            _, train_hits, valid_hits, test_hits = loggers[key].print_statistics(run, epoch=best_val_epoch)
            s += ",{:.2f},{:.2f},{:.2f}".format(train_hits, valid_hits, test_hits)

        with open(res_file, 'a') as f:
            f.write("{},{:.4f},{}".format(run, loss, best_val_epoch) + s + "\n")

    s = ""
    es, (train_hits, s1, valid_hits, s2, test_hits, s3) = loggers['Hits@100'].print_statistics()
    s += "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(train_hits, s1, valid_hits, s2, test_hits, s3)
    for key in ['Hits@50', 'Hits@10']:
        print(key)
        _, (train_hits, s1, valid_hits, s2, test_hits, s3) = loggers[key].print_statistics(epoch=es)
        s = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},".format(train_hits, s1, valid_hits, s2, test_hits, s3) + s

    with open(res_file, 'a') as f:
        f.write(s[1:] + "\n")


if __name__ == "__main__":
    main()
