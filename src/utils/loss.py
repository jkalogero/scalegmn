import torch
import torch.nn as nn
from nfn.common import WeightSpaceFeatures
import math
from src.data.base_datasets import Batch


def select_criterion(criterion: str, criterion_args: dict) -> nn.Module:
    _map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(**criterion_args),
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss()
    }
    if criterion not in _map.keys():
        raise NotImplementedError
    else:
        return _map[criterion]


def L2_distance(x, x_hat, batch_size=1):
    """
    Compute L2 Loss between the inputs.
    """

    if isinstance(x, torch.Tensor) and isinstance(x_hat, torch.Tensor):
        loss = torch.square(x - x_hat).sum() / batch_size

    elif isinstance(x, dict) and isinstance(x_hat, dict):
        loss = 0
        for key in x:
            loss += torch.square(x[key] - x_hat[key]).sum()

        loss = loss / batch_size

    elif isinstance(x, WeightSpaceFeatures):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])
        loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, Batch):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])
        loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, tuple):
        diff_weights = sum([torch.sum(torch.square(w1 - w2), (1,2,3)) for w1, w2 in zip(x_hat[0], x[0])])
        diff_biases = sum([torch.sum(torch.square(b1 - b2), (1,2)) for b1, b2 in zip(x_hat[1], x[1])])
        loss = (diff_weights + diff_biases) / batch_size
    else:
        raise NotImplemented

    return loss
