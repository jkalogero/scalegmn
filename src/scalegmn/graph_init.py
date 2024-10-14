import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_to_graphs(
    weights,
    biases,
    weights_norm=None,
    biases_norm=None,
    input_emb=None
):
    device = weights[0].device
    bsz = weights[0].shape[0]
    num_nodes = weights[0].shape[1] + sum(w.shape[2] for w in weights)

    node_features = torch.zeros(bsz, num_nodes, biases[0].shape[-1], device=device)
    edge_features = torch.zeros(
        bsz, num_nodes, num_nodes, weights[0].shape[-1], device=device
    )

    row_offset = 0
    col_offset = weights[0].shape[1]  # no edge to input nodes
    for i, w in enumerate(weights):
        _, num_in, num_out, _ = w.shape
        w_norm = weights_norm[i] if weights_norm is not None else 1
        edge_features[
        :, row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ] = w / w_norm
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[1]  # no bias in input nodes

    if input_emb is not None:
        node_features[:, 0: row_offset] = input_emb
    else:
        node_features[:, 0: row_offset] = torch.tensor([1])  # set input node state to 1.

    for i, b in enumerate(biases):
        _, num_out, _ = b.shape
        b_norm = biases_norm[i] if biases_norm is not None else 1
        node_features[:, row_offset : row_offset + num_out] = b / b_norm
        row_offset += num_out

    return node_features, edge_features


class GraphInit(nn.Module):
    """
    Initialize node and edge features.
    Simply apply linear layers to the biases and weights respectively.
    """
    def __init__(
        self,
        d_in_v,
        d_in_e,
        d_node,
        d_edge,
        project_node_feats,
        project_edge_feats,
    ):
        super().__init__()
        self.d_in = d_in_v
        self.d_node = d_node

        self.project_node_feats = project_node_feats
        self.project_edge_feats = project_edge_feats
        if self.project_node_feats:
            self.proj_bias = nn.Linear(d_in_v, d_node, bias=False)
        if self.project_edge_feats:
            self.proj_weight = nn.Linear(d_in_e, d_edge, bias=False)

    def forward(self, batch):
        if self.project_node_feats:
            batch.x = self.proj_bias(batch.x) if self.project_node_feats else batch.x
        if self.project_edge_feats:
            batch.edge_attr = self.proj_weight(batch.edge_attr) if self.project_edge_feats else batch.edge_attr
        return batch