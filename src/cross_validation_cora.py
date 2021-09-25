import logging
import os
import random
import torch

def calculate(scores, name):
    score_tensor = torch.tensor(scores)
    print("{} \t mean: {:.4f} $\pm$ {:.4f}".format(name, score_tensor.mean().item(), score_tensor.std().item()))

def main():
    # cora-gcn-cm-@5
    scores = [0.9412, 0.797, 0.9279, 0.6907, 0.871, 0.6888, 0.7097, 0.759, 0.8748, 0.8634]
    calculate(scores, "cora-gcn-cm")

    # cora-sage-cm-@5
    scores = [0.9393, 0.9298, 0.8292, 0.5863, 0.8178, 0.8767, 0.8805, 0.9715, 0.907, 0.9146]
    calculate(scores, "cora-sage-cm")

    # cora-gin-cm-@5
    scores = [0.888, 0.5465, 0.888, 0.7362, 0.6338, 0.5863, 0.7799, 0.7552, 0.6774, 0.6148]
    calculate(scores, "cora-gin-cm")

    # cora-gcn-@5
    scores = [0.8918, 0.907, 0.9298, 0.8539, 0.8539, 0.9051, 0.8027, 0.8425, 0.9241, 0.7856]
    calculate(scores, "cora-gcn")

    # cora-sage-@5
    scores = [0.8349, 0.9222, 0.871, 0.5674, 0.7514, 0.5769, 0.8368, 0.5522, 0.8767, 0.8994]
    calculate(scores, "cora-sage")

    # cora-gin-@5
    scores = [0.8065, 0.8653, 0.5731, 0.8843, 0.7533, 0.7647, 0.6926, 0.7476, 0.5939, 0.6262]
    calculate(scores, "cora-gin")


    # cora-gcn-cm-@10
    scores = [0.9412, 0.797, 0.9279, 0.6907, 0.871, 0.6888, 0.7097, 0.759, 0.8748, 0.8634]
    calculate(scores, "cora-gcn-cm")

    # cora-sage-cm-@10
    scores = [0.9393, 0.9298, 0.8292, 0.5863, 0.8178, 0.8767, 0.8805, 0.9715, 0.907, 0.9146]
    calculate(scores, "cora-sage-cm")

    # cora-gin-cm-@10
    scores = [0.888, 0.5465, 0.888, 0.7362, 0.6338, 0.5863, 0.7799, 0.7552, 0.6774, 0.6148]
    calculate(scores, "cora-gin-cm")

    # cora-gcn-@10
    scores = [0.8918, 0.907, 0.9298, 0.8539, 0.8539, 0.9051, 0.8027, 0.8425, 0.9241, 0.7856]
    calculate(scores, "cora-gcn")

    # cora-sage-@10
    scores = [0.8349, 0.9222, 0.871, 0.5674, 0.7514, 0.5769, 0.8368, 0.5522, 0.8767, 0.8994]
    calculate(scores, "cora-sage")

    # cora-gin-@10
    scores = [0.8065, 0.8653, 0.5731, 0.8843, 0.7533, 0.7647, 0.6926, 0.7476, 0.5939, 0.6262]
    calculate(scores, "cora-gin")




if __name__ == '__main__':
    main()
