import torch
import torch_geometric
import numpy as np
from torch_geometric.data import Data


def fast_nn_to_edge_index(layer_layout, device, dtype=torch.long):
    cum_layer_sizes = np.cumsum([0] + layer_layout)
    layer_indices = [
        torch.arange(cum_layer_sizes[i], cum_layer_sizes[i + 1], dtype=dtype)
        for i in range(len(cum_layer_sizes) - 1)
    ]
    edge_index = torch.cat(
        [
            torch.cartesian_prod(layer_indices[i], layer_indices[i + 1])
            for i in range(len(layer_indices) - 1)
        ],
        dim=0
    ).to(device).t()
    return edge_index


def graph_to_wb(
    edge_features,
    node_features,
    weights,
    biases
):
    new_weights = []
    new_biases = []
    cnt1, cnt2 = 0, weights[0].shape[1]
    for i, w in enumerate(weights):
        new_weights.append(edge_features[:, cnt1: cnt1+w.shape[1], cnt2: cnt2+w.shape[2]])
        cnt1 += w.shape[1]
        cnt2 += w.shape[2]
    cnt1 = weights[0].shape[1]
    for i, b in enumerate(biases):
        new_biases.append(node_features[:, cnt1: cnt1 + b.shape[1]])
        cnt1 += b.shape[1]
    return new_weights, new_biases


# replace the below with get_node_layer()
def get_nodes_at_layer(x, layer_idx, ptr, layer='hidden'):
    """
    TODO: this assumes fixed architecture for the input models.
    Modify this to handle different architectures. Maybe create and pass the l(i), l: V -> [L].
    Get hidden nodes from node_features
    """
    if layer == 'hidden':
        selected_nodes = range(layer_idx[1], layer_idx[-2])
    elif layer == 'first':
        selected_nodes = range(layer_idx[0], layer_idx[1])
    else:
        raise ValueError('Invalid layer type')

    selected_nodes = torch.tensor([m + p for p in ptr[:-1] for m in selected_nodes])
    mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    mask[selected_nodes] = 1
    return mask.unsqueeze(1)
