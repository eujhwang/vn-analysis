import logging
import os
import random
import torch

def calculate(scores, name):
    score_tensor = torch.tensor(scores)
    print("{} \t mean: {:.4f} $\pm$ {:.4f}".format(name, score_tensor.mean().item(), score_tensor.std().item()))

def main():
    # # cora-gcn-cm-@5
    # scores = [0.9412, 0.797, 0.9279, 0.6907, 0.871, 0.6888, 0.7097, 0.759, 0.8748, 0.8634]
    # calculate(scores, "cora-gcn-cm")
    #
    # # cora-sage-cm-@5
    # scores = [0.9393, 0.9298, 0.8292, 0.5863, 0.8178, 0.8767, 0.8805, 0.9715, 0.907, 0.9146]
    # calculate(scores, "cora-sage-cm")
    #
    # # cora-gin-cm-@5
    # scores = [0.888, 0.5465, 0.888, 0.7362, 0.6338, 0.5863, 0.7799, 0.7552, 0.6774, 0.6148]
    # calculate(scores, "cora-gin-cm")
    #
    # # cora-gcn-@5
    # scores = [0.8918, 0.907, 0.9298, 0.8539, 0.8539, 0.9051, 0.8027, 0.8425, 0.9241, 0.7856]
    # calculate(scores, "cora-gcn")
    #
    # # cora-sage-@5
    # scores = [0.8349, 0.9222, 0.871, 0.5674, 0.7514, 0.5769, 0.8368, 0.5522, 0.8767, 0.8994]
    # calculate(scores, "cora-sage")
    #
    # # cora-gin-@5
    # scores = [0.8065, 0.8653, 0.5731, 0.8843, 0.7533, 0.7647, 0.6926, 0.7476, 0.5939, 0.6262]
    # calculate(scores, "cora-gin")


    # cora-gcn-cm-@10
    scores = [0.9791, 0.8994, 0.8824, 0.962, 0.9051, 0.8672, 0.8159, 0.9184, 0.9374, 0.9336]
    calculate(scores, "cora-gcn-cm")

    # cora-gcn-cm-@10
    scores = [0.9734, 0.9279, 0.7894, 0.9298, 0.8937, 0.8748, 0.8653, 0.9393, 0.8254, 0.8956]
    calculate(scores, "cora-gcn-cm")

    # cora-gcn-cm-@10
    scores = [0.9734, 0.8956, 0.9032, 0.8235, 0.9336, 0.8254, 0.871, 0.8824, 0.9355, 0.9089]
    calculate(scores, "cora-gcn-cm")

    # cora-sage-cm-@10
    scores = [0.9639, 0.9336, 0.8368, 0.9336, 0.8615, 0.9526, 0.945, 0.8729, 0.9829, 0.9507]
    calculate(scores, "cora-sage-cm")

    # cora-gin-cm-@10
    scores = [0.9564, 0.9146, 0.8065, 0.8634, 0.7306, 0.759, 0.8065, 0.8843, 0.888, 0.8634]
    calculate(scores, "cora-gin-cm")

    # cora-gin-cm-@10
    scores = [0.8937, 0.8861, 0.9431, 0.8899, 0.8273, 0.9089, 0.8463, 0.8786, 0.852, 0.8216]
    calculate(scores, "cora-gin-cm")

    # cora-gcn-@10
    scores = [0.9658, 0.9203, 0.9905, 0.9734, 0.962, 0.9488, 0.9602, 0.8843, 0.9696, 0.9317]
    calculate(scores, "cora-gcn")

    # cora-sage-@10
    scores = [0.9374, 0.9032, 0.9715, 0.888, 0.8501, 0.7249, 0.9526, 0.8178, 0.8975, 0.9677]
    calculate(scores, "cora-sage")

    # cora-gin-@10
    scores = [0.9602, 0.8539, 0.9241, 0.797, 0.8197, 0.8729, 0.7078, 0.9241, 0.7343, 0.8786]
    calculate(scores, "cora-gin")


    # cora-gcn-vn-@10
    scores = [0.8861, 0.8994, 0.8824, 0.962, 0.9051, 0.8672, 0.8159, 0.9184, 0.9374, 0.9336]
    calculate(scores, "cora-gcn-vn")

    scores = [0.8861, 0.9431, 0.9412, 0.8767, 0.8748, 0.9241, 0.9127, 0.962, 0.9317, 0.9583]
    calculate(scores, "cora-gcn-vn")

    scores = [0.9734, 0.9279, 0.7894, 0.9298, 0.8937, 0.8748, 0.8653, 0.9393, 0.8254, 0.8956]
    calculate(scores, "cora-gcn-vn")

    # cora-sage-vn-@10
    scores = [0.74, 0.9203, 0.9412, 0.9715, 0.9412, 0.9469, 0.9146, 0.871, 0.9753, 0.945]
    calculate(scores, "cora-sage-vn")

    # cora-gin-vn-@10
    scores = [0.8463, 0.9013, 0.8577, 0.8691, 0.8292, 0.7192, 0.8102, 0.8197, 0.871, 0.8501]
    calculate(scores, "cora-gin-vn")

    # cora-gin-vn-@10
    scores = [0.7837, 0.8235, 0.7799, 0.8311, 0.7913, 0.8065, 0.8254, 0.8615, 0.8254, 0.8918]
    calculate(scores, "cora-gin-vn")

    # cora-gin-vn-@10
    scores = [0.8349, 0.8672, 0.852, 0.8805, 0.8121, 0.7875, 0.7514, 0.7818, 0.8577, 0.8558]
    calculate(scores, "cora-gin-vn")

    # cora-gin-vn-@10
    scores = [0.8918, 0.8311, 0.9317, 0.8937, 0.8463, 0.9051, 0.9146, 0.8767, 0.9374, 0.9108]
    calculate(scores, "cora-gin-vn")

    # cora-gcn-rm
    scores = [0.8729, 0.9526, 0.9108, 0.8861, 0.8254, 0.8824, 0.8406, 0.8046, 0.9127, 0.9298]
    calculate(scores, "cora-gcn-rm")

    # cora-sage-rm
    scores = [0.8596, 0.8937, 0.8311, 0.8861, 0.8046, 0.7932, 0.9203, 0.8861, 0.8767, 0.9203]
    calculate(scores, "cora-sage-rm")

    # cora-gin-rm
    scores = [0.8254, 0.7419, 0.8065, 0.6641, 0.6167, 0.8235, 0.7628, 0.8178, 0.8102, 0.8748]
    calculate(scores, "cora-gin-rm")

    # cora-gcn-rmf
    scores = [0.7837, 0.9564, 0.8767, 0.8178, 0.8653, 0.8805, 0.9412, 0.8216, 0.9431, 0.9715]
    calculate(scores, "cora-gcn-rmf")

    # cora-sage-rmf
    scores = [0.8975, 0.9696, 0.9165, 0.9032, 0.7116, 0.8824, 0.9658, 0.9564, 0.9507, 0.962]
    calculate(scores, "cora-sage-rmf")

    # cora-gin-rmf
    scores = [0.8994, 0.9279, 0.926, 0.7609, 0.6565, 0.9032, 0.8482, 0.8539, 0.8653, 0.8634]
    calculate(scores, "cora-gin-rmf")

    # cora-gcn-cm+
    scores = [0.8805, 0.797, 0.8634, 0.5066, 0.907, 0.7419, 0.8975, 0.8918, 0.8216, 0.9317]
    calculate(scores, "cora-gcn-cm+")

    # cora-sage-cm+
    scores = [0.7742, 0.8065, 0.833, 0.7951, 0.6964, 0.7875, 0.7381, 0.8729, 0.8691, 0.8843]
    calculate(scores, "cora-sage-cm+")

    # cora-gin-cm+
    scores = [0.8235, 0.7438, 0.871, 0.8254, 0.7799, 0.9032, 0.7894, 0.926, 0.8539, 0.9032]
    calculate(scores, "cora-gin-cm+")

if __name__ == '__main__':
    main()
