import torch
import torch_geometric


def to_pyg_batch(node_features,
                 edge_features,
                 edge_index,
                 node2type=None,
                 edge2type=None,
                 direction='forward',
                 label=None,
                 hidden_nodes=None,
                 first_layer_nodes=None
                 ):
    if direction in ['forward', 'backward']:
        edge_features = edge_features if direction == 'forward' else edge_features.transpose(-2, -3)
        edge_index = edge_index if direction == 'forward' else torch.flip(edge_index, [0])
        data = torch_geometric.data.Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features[edge_index[0], edge_index[1]],
                node2type=node2type if node2type is not None else None,
                edge2type=edge2type if edge2type is not None else None,
                label=label,
                mask_hidden=hidden_nodes,
                mask_first_layer=first_layer_nodes
            )
        return data, None

    elif direction == 'bidirectional':
        data = torch_geometric.data.Data(
            x=node_features,
            edge_index=edge_index,
            bw_edge_index=torch.flip(edge_index, [0]),
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            node2type=node2type if node2type is not None else None,
            edge2type=edge2type if edge2type is not None else None,
            label=label,
            mask_hidden=hidden_nodes,
            mask_first_layer=first_layer_nodes
        )
        return data, None


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
