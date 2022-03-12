import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    def binary_cross_entropy(predict, target):
        return -(target * torch.log(predict + 1e-12) + (1 - target) * torch.log(1 - predict + 1e-12))

    a = torch.Tensor([.1, .3])
    b = torch.Tensor([.2, .2])
    bce = binary_cross_entropy(a, b)
    print(bce)