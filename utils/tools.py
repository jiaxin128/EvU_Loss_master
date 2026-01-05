import torch


def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes)
    return y[labels]