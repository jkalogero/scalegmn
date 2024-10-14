import torch
import torch_geometric
import numpy as np
from torch_geometric.data import Data


def nn_to_edge_index(layer_layout, device, dtype=torch.long):
    edge_index = []

    node_offset = 0
    nodes_per_layer = []
    for n in layer_layout:
        nodes_per_layer.append(list(range(node_offset, node_offset + n)))
        node_offset += n

    for i in range(1, len(layer_layout)):
        for j in nodes_per_layer[i - 1]:
            for k in nodes_per_layer[i]:
                edge_index.append([j, k])

    return torch.tensor(edge_index, device=device, dtype=dtype).T



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


def get_node_types(nodes_per_layer):
    node_types = []
    type = 0
    for i, el in enumerate(nodes_per_layer):
        if i == 0:  # first layer
            for _ in range(el):
                node_types.append(type)
                type += 1
        elif i > 0 and i < len(nodes_per_layer) - 1:  #  hidden layers
            for _ in range(el):
                node_types.append(type)
            type += 1
        elif i == len(nodes_per_layer) - 1:  # last layer
            for _ in range(el):
                node_types.append(type)
                type += 1
    return torch.tensor(node_types)


def get_edge_types(nodes_per_layer):
    edge_types = []
    type = 0
    for i, el in enumerate(nodes_per_layer[:-1]):
        if i == 0:  # first layer
            for _ in range(el):
                for neighbour in range(nodes_per_layer[i+1]):
                    edge_types.append(type)
                type += 1
        elif i > 0 and i < len(nodes_per_layer) - 2:  #  hidden layers
            for _ in range(el):
                for neighbour in range(nodes_per_layer[i+1]):
                    edge_types.append(type)
            type += 1
        elif i == len(nodes_per_layer) - 2:  # last layer
            for neighbour in range(nodes_per_layer[i + 1]):
                for _ in range(el):
                    edge_types.append(type)
                type += 1

    # from collections import Counter
    # print(Counter(edge_types))
    return torch.tensor(edge_types)


def to_pyg_batch(node_features,
                 edge_features,
                 edge_index,
                 node2type=None,
                 edge2type=None,
                 direction='forward',
                 rev_node_features=None,
                 rev_edge_features=None):
    # edge_features = edge_features.flatten(1, 2)
    if direction in ['forward', 'backward']:
        edge_features = edge_features if direction == 'forward' else edge_features.transpose(-2, -3)
        edge_index = edge_index if direction == 'forward' else torch.flip(edge_index, [0])
        data_list = [
            torch_geometric.data.Data(
                x=node_features[i],
                edge_index=edge_index,
                edge_attr=edge_features[i, edge_index[0], edge_index[1]],
                node2type=node2type.to(node_features.device) if node2type is not None else None,
                edge2type=edge2type.to(node_features.device) if edge2type is not None else None
            )
            for i in range(node_features.shape[0])
        ]
        batch = torch_geometric.data.Batch.from_data_list(data_list)

        rev_batch = None
        return batch, rev_batch

    elif direction == 'bidirectional':
        data_list = []
        for i in range(node_features.shape[0]):
            data = torch_geometric.data.HeteroData()
            data['fw_neuron'].x = node_features[i]
            data['bw_neuron'].x = node_features[i]
            data['fw_neuron', 'fw_edge', 'fw_neuron'].edge_index = edge_index
            data['bw_neuron', 'bw_edge', 'bw_neuron'].edge_index = torch.flip(edge_index, [0])
            data['fw_neuron', 'fw_edge', 'fw_neuron'].edge_attr = edge_features[i, edge_index[0], edge_index[1]]
            data['bw_neuron', 'bw_edge', 'bw_neuron'].edge_attr = edge_features.transpose(-2, -3)[i, edge_index[0], edge_index[1]]
            data['node2type'] = node2type.to(node_features.device) if node2type is not None else None
            data['edge2type'] = edge2type.to(node_features.device) if edge2type is not None else None
            data_list.append(data)

        batch = torch_geometric.data.Batch.from_data_list(data_list)
        bw_batch = None
        return batch, bw_batch



def reverse_batch(batch):
    "Given a torch geometric data object, create a new graph with reversed edges"
    rev_batch = batch.clone()
    # edge_index = batch.pop('edge_index')
    rev_batch.edge_index = torch.flip(rev_batch.edge_index, [0])
    # set dynamically every other attr as the same
    # return torch_geometric.data.Data(edge_index=rev_edge_index, **batch)
    return rev_batch


def is_reversed(edge_feat1, edge_feat2):
    bs, n1, n2, _ = edge_feat1.shape
    for b in range(bs):
        for i in range(n1):
            for j in range(n2):
                if not (edge_feat1[b, i, j] == edge_feat2[b, j,i]).all():
                    return False
    return True

def graph_to_wb(
    edge_features,
    node_features,
    weights,
    biases
):
    new_weights = []
    new_biases = []

    start = 0
    for i, w in enumerate(weights):
        size = torch.prod(torch.tensor(w.shape[1:]))
        new_weights.append(edge_features[:, start : start + size].view(w.shape))
        start += size

    start = 0
    for i, b in enumerate(biases):
        size = torch.prod(torch.tensor(b.shape[1:]))
        new_biases.append(node_features[:, start : start + size].view(b.shape))
        start += size

    return new_weights, new_biases


def graph_to_wb_old(edge_features, node_features, weights, biases):
    new_weights = []
    new_biases = []

    start = 0
    for w in weights:
        size = torch.prod(torch.tensor(w.shape[1:-1]))
        new_weights.append(edge_features[:, start:start + size].view(*w.shape[:-1], edge_features.shape[-1]))
        start += size

    start = 0
    for b in biases:
        size = torch.prod(torch.tensor(b.shape[1:-1]))
        new_biases.append(node_features[:, start:start + size].view(*b.shape[:-1], node_features.shape[-1]))
        start += size

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
