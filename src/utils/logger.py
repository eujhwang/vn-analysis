import torch


class Logger(object):
    def __init__(self, runs, info=None, max=True):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.maximize=max

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_best_epoch(self, run, type=1):
        result = 100 * torch.tensor(self.results[run])
        _, argmax = torch.max(result[:, type], 0) if self.maximize else torch.min(result[:, type], 0)
        return argmax.item() + 1

    def print_statistics(self, run=None, epoch=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = (result[:, 1].argmax().item() if self.maximize else result[:, 1].argmin().item()) if epoch is None else epoch-1
            print(f'Run {run + 1:02d}:')
            print(f'Best Train: {result[:, 0].max():.2f}')
            print(f'Best Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')

            return argmax + 1, result[argmax, 0], result[argmax, 1], result[argmax, 2]

        else:  # changed ogb code because of patience, we have runs of different length
            best_results = []
            argmaxs = []
            for i, result in enumerate(self.results):  # for every run
                r = 100 * torch.tensor(result)

                train1 = r[:, 0].max().item() if self.maximize else r[:, 0].min().item()
                argmax = (r[:, 1].argmax().item() if self.maximize else r[:, 1].argmin().item()) if epoch is None else epoch[i]-1
                valid = r[argmax, 1].item()
                train2 = r[argmax, 0].item()
                test = r[argmax, 2].item()
                best_results.append((train1, valid, train2, test))

                argmaxs += [argmax]

            best_result = torch.tensor(best_results)

            print(f'All runs:') #, best_result.tolist())
            rs = []
            r = best_result[:, 0]
            print(f'Best Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            rs += [r.mean(), r.std()]
            print(f'Best Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            rs = [r.mean(), r.std()] + rs
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            rs += [r.mean(), r.std()]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return argmaxs, tuple(rs)
