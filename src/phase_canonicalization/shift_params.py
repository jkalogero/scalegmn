import torch
import sys
import math
from test_inr import test_inr
from copy import deepcopy


def iterate_l(**kwargs):
    for k, v in kwargs.items():
        for i, el in enumerate(v):
            print(f"{k}[{i}].shape: {el.shape}")


def create_signature_matrices(shape, mask):
    q = torch.eye(shape)
    q[mask, :] *= -1
    return q


def init_q(shape, mask):
    q = torch.ones(shape)
    q[mask] *= -1
    return q


def first_step(w, b):
    q = []
    for l in range(len(w)):
        w[l] = w[l].squeeze(0, -1)
        b[l] = b[l].squeeze(0, -1)
        mask = b[l] < 0
        b[l][mask] *= -1
        s_mask = mask.squeeze()
        w[l][s_mask, :] *= -1
        q.append(init_q(w[l].shape[0], s_mask))
    return w, b, q


def second_step(b):
    for l in range(len(b)):
        threshold = 2*math.pi
        mask = b[l] > threshold  # all biases seem to be less than that
        b[l][mask] = b[l][mask] % threshold
    return b


def third_step(b, q):
    for l in range(len(b)):
        mask1 = b[l] > math.pi
        mask2 = b[l] <= 2*math.pi
        mask = mask1 * mask2
        b[l][mask] -= - math.pi
        q[l][mask] *= -1
    return b, q


def final_step(b, q):
    for l in range(len(b)):
        mask1 = b[l] > math.pi / 2
        mask2 = b[l] <= math.pi
        mask = mask1 * mask2
        b[l][mask] -= math.pi
        q[l][mask] *= -1
    return b, q


def shift_params(batch, rec_inr=False):
    init_w, init_b = deepcopy(batch[0]), deepcopy(batch[1])
    w1, b1, q1 = first_step(init_w, init_b)
    b2 = second_step(b1)
    b3, q3 = third_step(b2, q1)
    bf, qf = final_step(b3, q3)
    final_w = [(qf[l].unsqueeze(-1) * w1[l]).unsqueeze(0) for l in range(len(w1))]
    final_b = [(qf[l] * bf[l]).reshape(1,-1) for l in range(len(bf))]

    # test INR
    if rec_inr:
        iterate_l(final_w=final_w, final_b=final_b)
        test_inr(final_w, final_b)
    return final_w, final_b