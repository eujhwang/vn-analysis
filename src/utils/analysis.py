import torch
import statistics
import sys
import random
from torch_geometric.utils import to_undirected
from ogb.linkproppred import PygLinkPropPredDataset
from src.model.vngnn import get_vn_index, iterative_diff_pool
from src.utils.to_dense import ToDense


parts = ["train", "valid", "test"]


def get_clustering(name, num_ns, num_vns, edge_index, x=None, adj=None, metisplus_cl_index=None):
    if name == "diffpool":
        return iterative_diff_pool(num_vns, 1, x, adj)
    if name == "metis+" and metisplus_cl_index is not None:  # after initial one
            # random assignment of clusters to VNs
            vn2cl = get_vn_index("random", metisplus_cl_index.shape[0], num_vns, 1, None)
            # now propagate to node level
            vn_index = torch.zeros(num_vns, num_ns)
            for c in range(metisplus_cl_index.shape[0]):
                vn_index[vn2cl[:, c].nonzero(), metisplus_cl_index[c]] = 1

            vn_index = vn_index == 1
            return vn_index

    return get_vn_index(name, num_ns, num_vns, 1, edge_index)


def main(name='ogbl-ddi', num_vns=[32], cluster_algo="metis+", runs=3, num_clusters=[], check=False):  #vn="metis"):  #

    print("*" * 50)
    print(name, cluster_algo, num_vns, num_clusters)
    print("*" * 50)
    dataset = PygLinkPropPredDataset(name=name, root="../dataset",
                                     transform=ToDense(remove_edge_index=False) if "diffpool" == cluster_algo else None)
    split_edge = dataset.get_edge_split()
    num_ns = dataset.data.num_nodes[0]

    if "collab" in name:
        val_edge_index = split_edge["valid"]["edge"].t()
        dataset.data.edge_index = to_undirected(torch.cat([dataset.data.edge_index, val_edge_index], dim=-1))

    print("# N", num_ns)
    print("# E (Graph, norm)", dataset.data.edge_index.shape[1]/2)
    print("# E", sum([split_edge[p]['edge'].size(0) for p in parts]))

    # if check:
    #     num_ns_ei = max(max(dataset.data.edge_index[0]),max(dataset.data.edge_index[1])).item() +1
    #     assert num_ns_ei == num_ns

    num_clusters = [0] if not num_clusters else num_clusters
    for ncl in num_clusters:
        print("+" * 10)
        print("NCL : ", ncl)
        cl_index = get_clustering(cluster_algo, num_ns, ncl, dataset.data.edge_index) if ncl else None

        for nvn in num_vns:
            print("-" * 10)
            print("NVN : ", nvn)
            results = {p: [] for p in parts}
            for r in range(runs):
                torch.manual_seed(r)
                random.seed(r)
                vn_idx = get_clustering(cluster_algo, num_ns, nvn, dataset.data.edge_index,
                                        adj=dataset.data.adj if hasattr(dataset.data, "adj") else None,  # adj yet w/o valid for collab
                                        metisplus_cl_index=cl_index)

                for p in parts:
                    idx = split_edge[p]['edge']

                    intra = 0
                    for e in idx:
                        intra += is_intra_cluster(*e, vn_idx)
                    results[p] += [intra]

            s0, s1, s2 = "", "", ""
            for p in parts:
                print("-"*10)
                idx = split_edge[p]['edge']
                s0 += p + " / "
                s1 += "{}".format(idx.size(0)) + " / "
                print(["{:.4f}".format(i) for i in results[p]])

                val_list = [i/idx.size(0) for i in results[p]]
                print(["{:.4f}".format(i) for i in val_list])

                if runs > 1:
                    s2 += "{:.4f} \pm {:.4f}".format(sum(val_list) / len(val_list), statistics.stdev(val_list)) + " / "
                    # print("AVG : {:.4f}, {:.4f}".format(sum(val_list) / len(val_list), statistics.stdev(val_list)))
                else:
                    s2 += "{:.4f}".format(val_list[0]) + " / "
                    # print("RES : {:.4f}".format(val_list[0]))

            print(s0)
            print("NUM : ", s1)
            print("RES : ", s2)
            sys.stdout.flush()


def is_intra_cluster(src, trg, vn_idx):
    i = (vn_idx[:, src] == 1).nonzero()[0].item()
    j = (vn_idx[:, trg] == 1).nonzero()[0].item()
    return i == j


# STARTED ddi 12:58
if __name__ == "__main__":

    orig_stdout = sys.stdout
    f = open('../analysis_ddi_cm+2.txt', 'a')
    sys.stdout = f

    algos = ["metis+"]  #"random", "metis", "metis+"]
    algos2 = ["diffpool"] #"graclus",
    cls = [100,150,500]  #, 500, 1000]
    vns = [ 4, 8, 16, 32, 64]  #, 128, 256, 512] 4, 8,
    runs = {
        "random": 10, "metis": 1, "metis+": 10, "graclus": 1, "diffpool": 1
    }

    for algo in algos:
        for name in ["ogbl-ddi"]:  #["ogbl-ddi", "ogbl-ppa", "ogbl-collab"]:  #, "ogbl-biokg"]:
            main(name, num_vns=vns, cluster_algo=algo, runs=runs[algo], num_clusters=cls if algo=="metis+" else [])

    sys.stdout = orig_stdout
    f.close()
