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


def L2_distance(x, x_hat, batch_size=1, dim_size=None, normalize=False):
    """
    Compute L2 Loss between the inputs.
    This function never normalizes on the total size
    of the input.
    :param x:
    :param x_hat:
    :param batch_size:
    :param normalize:
    :return:
    """

    if isinstance(x, torch.Tensor) and isinstance(x_hat, torch.Tensor):

        if normalize:
            dim_size = dim_size if dim_size is not None else x.shape[1]  # full
            loss = torch.square(x - x_hat).sum() / (batch_size * dim_size)
        else:
            loss = torch.square(x - x_hat).sum() / batch_size  # no norm

    elif isinstance(x, dict) and isinstance(x_hat, dict):
        loss = 0
        dim_size = 0
        for key in x:
            if normalize:
                dim_size += math.prod([x[key].shape[i] for i in range(1,len(x[key].shape))])
            loss += torch.square(x[key] - x_hat[key]).sum()
        if normalize:
            loss = loss / (batch_size * dim_size)
        else:
            loss = loss / batch_size

    elif isinstance(x, WeightSpaceFeatures):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])

        if normalize:
            loss = (diff_weights + diff_biases) / (batch_size * dim_size)
        else:
            loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, Batch):  # same as nfn
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])

        if normalize:
            loss = (diff_weights + diff_biases) / (batch_size * dim_size)
        else:
            loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, tuple):  # same as nfn
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat[0], x[0])])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat[1], x[1])])

        if normalize:
            loss = (diff_weights + diff_biases) / (batch_size * dim_size)
        else:
            loss = (diff_weights + diff_biases) / batch_size
    else:
        raise NotImplemented

    return loss
