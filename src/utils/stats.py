import torch
import statistics
import sys
import random
import snap
import pandas as pd
import numpy as np
from torch_geometric.utils import to_undirected
from ogb.linkproppred import PygLinkPropPredDataset
from src.model.vngnn import get_vn_index, iterative_diff_pool
from src.utils.to_dense import ToDense


parts = ["train", "valid", "test"]


def main():  #name='ogbl-ddi', num_vns=[32], cluster_algo="metis+", runs=3, num_clusters=[], check=False):  #vn="metis"):  #

    # print("*" * 50)
    # print(name, cluster_algo, num_vns, num_clusters)
    # print("*" * 50)
    # dataset = PygLinkPropPredDataset(name=name, root="../dataset",
    #                                  transform=ToDense(remove_edge_index=False) if "diffpool" == cluster_algo else None)
    # split_edge = dataset.get_edge_split()
    # num_ns = dataset.data.num_nodes[0]
    #
    # if "collab" in name:
    #     val_edge_index = split_edge["valid"]["edge"].t()
    #     dataset.data.edge_index = to_undirected(torch.cat([dataset.data.edge_index, val_edge_index], dim=-1))
    #
    # print("# N", num_ns)
    # print("# E (Graph, norm)", dataset.data.edge_index.shape[1]/2)
    # print("# E", sum([split_edge[p]['edge'].size(0) for p in parts]))

    # if check:
    #     num_ns_ei = max(max(dataset.data.edge_index[0]),max(dataset.data.edge_index[1])).item() +1
    #     assert num_ns_ei == num_ns

    name = "ogbl-ppa"
    dataset = PygLinkPropPredDataset(name=name, root="../dataset")
    split_edge = dataset.get_edge_split()

    train_idx = pd.read_csv("../dataset/train10.csv.gz", compression="gzip", header=None).values.T[0]
    split_edge['train']['edge'] = split_edge['train']['edge'][train_idx]
    # train_idx1 = [i * 2 for i in train_idx] + [(i * 2) + 1 for i in train_idx]
    # data.edge_index = data.edge_index[:, train_idx1]

    all = sum([split_edge[p]["edge"].size(0) for p in parts])
    splitratio = "{}/{}/{}".format(split_edge["train"]["edge"].size(0)/all,split_edge["valid"]["edge"].size(0)/all,split_edge["test"]["edge"].size(0)/all)

    p = "/Users/vth/Desktop/git/ogb-revisited/src/dataset/{}/raw/edge10.csv".format(name.replace("-","_"))
    np.savetxt(p, split_edge['train']['edge'].numpy().astype(int), fmt='%i', delimiter=",")

    G = snap.LoadEdgeList(snap.TUNGraph, p, 0, 1, ',')
    print("#N", len(list(G.Nodes())))
    print("#E", all) #, len(list(G.Edges())))
    maxscc_ratio = len(list(G.GetMxScc().Nodes())) / len(list(G.Nodes()))

    degcount = G.GetDegCnt()
    avg_nodes_deg = sum([item.GetVal2()*item.GetVal1() for item in degcount]) / sum([item.GetVal2() for item in degcount])
    del degcount
    avg_clust_coeff = G.GetClustCf(-1)

    print("splitratio", splitratio)

    print("maxscc_ratio", maxscc_ratio)
    print("avg_nodes_deg", avg_nodes_deg)
    print("avg_clust_coeff", avg_clust_coeff)

    sys.stdout.flush()

    diameter = G.GetBfsFullDiam(1000, False)
    print("diameter", diameter)

if __name__ == "__main__":

    orig_stdout = sys.stdout
    f = open('../ppi10stats.txt', 'a')
    sys.stdout = f
    main()
    sys.stdout = orig_stdout
    f.close()
