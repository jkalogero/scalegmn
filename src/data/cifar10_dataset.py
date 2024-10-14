import json
import math
import os
import random
from pathlib import Path
from typing import NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .data_utils import get_node_types, get_edge_types


class CNNBatch(NamedTuple):
    weights: Tuple
    biases: Tuple
    y: Union[torch.Tensor, float]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            y=self.y.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class CNNDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir,
            splits_path,
            split="train",
            normalize=False,
            augmentation=False,
            statistics_path="dataset/statistics.pth",
            noise_scale=1e-1,
            drop_rate=1e-2,
            max_kernel_size=(3, 3),
            linear_as_conv=False,
            flattening_method="repeat_nodes",
            max_num_hidden_layers=3,
    ):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            (Path(dataset_dir) / Path(p)).as_posix() for p in self.dataset["path"]
        ]

        self.augmentation = augmentation
        self.normalize = normalize
        if self.normalize:
            statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.noise_scale = noise_scale
        self.drop_rate = drop_rate

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

    def __len__(self):
        return len(self.dataset["score"])

    @staticmethod
    def _transform_weights_biases(w, max_kernel_size, linear_as_conv=False):
        """
        Convolutional weights are 4D, and they are stored in the following
        order: [out_channels, in_channels, height, width]
        Linear weights are 2D, and they are stored in the following order:
        [out_features, in_features]

        1. We transpose the in_channels and out_channels dimensions in
        convolutions, and the in_features and out_features dimensions in linear
        layers
        2. We have a maximum HxW value, and pad the convolutional kernel with
        0s if necessary
        3. We flatten the height and width dimensions of the convolutional
        weights
        4. We unsqueeze the last dimension of weights and biases
        """
        if w.ndim == 1:
            w = w.unsqueeze(-1)
            return w

        w = w.transpose(0, 1)

        if linear_as_conv:
            if w.ndim == 2:
                w = w.unsqueeze(-1).unsqueeze(-1)
            w = pad_and_flatten_kernel(w, max_kernel_size)
        else:
            w = (
                pad_and_flatten_kernel(w, max_kernel_size)
                if w.ndim == 4
                else w.unsqueeze(-1)
            )

        return w

    @staticmethod
    def _cnn_to_mlp_repeat_nodes(weights, biases, conv_mask):
        final_conv_layer = max([i for i, w in enumerate(conv_mask) if w])
        final_feature_map_size = (
                weights[final_conv_layer + 1].shape[0] // weights[final_conv_layer].shape[1]
        )
        weights[final_conv_layer] = weights[final_conv_layer].repeat(
            1, final_feature_map_size, 1
        )
        biases[final_conv_layer] = biases[final_conv_layer].repeat(
            final_feature_map_size, 1
        )
        return weights, biases, final_feature_map_size

    @staticmethod
    def _cnn_to_mlp_extra_layer(weights, biases, conv_mask, max_kernel_size):
        final_conv_layer = max([i for i, w in enumerate(conv_mask) if w])
        final_feature_map_size = (
                weights[final_conv_layer + 1].shape[0] // weights[final_conv_layer].shape[1]
        )
        dtype = weights[final_conv_layer].dtype
        # NOTE: We assume that the final feature map is square
        spatial_resolution = int(math.sqrt(final_feature_map_size))
        new_weights = (
            torch.eye(weights[final_conv_layer + 1].shape[0])
            .unflatten(0, (weights[final_conv_layer].shape[1], final_feature_map_size))
            .transpose(1, 2)
            .unflatten(-1, (spatial_resolution, spatial_resolution))
        )
        new_weights = pad_and_flatten_kernel(new_weights, max_kernel_size)

        new_biases = torch.zeros(
            (weights[final_conv_layer + 1].shape[0], 1),
            dtype=dtype,
        )
        weights = (
                weights[: final_conv_layer + 1]
                + [new_weights]
                + weights[final_conv_layer + 1:]
        )
        biases = (
                biases[: final_conv_layer + 1]
                + [new_biases]
                + biases[final_conv_layer + 1:]
        )
        return weights, biases, final_feature_map_size

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        # Create a mask to denote which layers are convolutional and which are linear
        conv_mask = [
            1 if w.ndim == 4 else 0 for k, w in state_dict.items() if "weight" in k
        ]

        layer_layout = [list(state_dict.values())[0].shape[1]] + [
            v.shape[0] for k, v in state_dict.items() if "bias" in k
        ]

        weights = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "weight" in k
        ]
        biases = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "bias" in k
        ]
        score = float(self.dataset["score"][item])

        # NOTE: We assume that the architecture includes linear layers and
        # convolutional layers
        if self.flattening_method == "repeat_nodes":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_repeat_nodes(
                weights, biases, conv_mask
            )
        elif self.flattening_method == "extra_layer":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_extra_layer(
                weights, biases, conv_mask, self.max_kernel_size
            )
        elif self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
        )
        return data


class NFNZooDataset(CNNDataset):
    """
    Adapted from NFN and neural-graphs source code.
    """

    def __init__(
            self,
            dataset,
            dataset_path,
            data_path,
            metrics_path,
            layout_path,
            split,
            activation_function,
            debug=False,
            idcs_file=None,
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            layer_layout=None,
            direction='forward',
            max_kernel_size=(3, 3),
            linear_as_conv=False,
            flattening_method=None,
            max_num_hidden_layers=3,
            data_format="graph",
    ):

        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.layer_layout = layer_layout
        self.direction = direction
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask

        data = np.load(data_path)
        # Hardcoded shuffle order for consistent test set.
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        # metrics = pd.read_csv(os.path.join(metrics_path))
        metrics = pd.read_csv(metrics_path, compression="gzip")
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(layout_path)
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[split]
        data = data[idcs]
        if activation_function is not None:
            metrics = metrics.iloc[idcs]
            mask = metrics['config.activation'] == activation_function
            self.metrics = metrics[mask]
            data = data[mask]
        else:
            self.metrics = metrics.iloc[idcs]

        if debug:
            data = data[:16]
            self.metrics = self.metrics[:16]
        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

        if self.node_pos_embed:
            self.node2type = get_node_types(self.layer_layout)
        if self.edge_pos_embed:
            self.edge2type = get_edge_types(self.layer_layout)

        # Since the current datasets have the same architecture for every datapoint, we can
        # create the below masks on initialization, rather than on __getitem__.
        if self.equiv_on_hidden:
            self.hidden_nodes = self.mark_hidden_nodes()
        if self.get_first_layer_mask:
            self.first_layer_nodes = self.mark_input_nodes()

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def get_layer_layout(self):
        return self.layer_layout

    def mark_hidden_nodes(self) -> torch.Tensor:
        hidden_nodes = torch.tensor(
                [False for _ in range(self.layer_layout[0])] +
                [True for _ in range(sum(self.layer_layout[1:-1]))] +
                [False for _ in range(self.layer_layout[-1])]).unsqueeze(-1)
        return hidden_nodes

    def mark_input_nodes(self) -> torch.Tensor:
        input_nodes = torch.tensor(
            [True for _ in range(self.layer_layout[0])] +
            [False for _ in range(sum(self.layer_layout[1:]))]).unsqueeze(-1)
        return input_nodes

    def __getitem__(self, idx):
        weights = [torch.from_numpy(w[idx]) for w in self.weights]
        biases = [torch.from_numpy(b[idx]) for b in self.biases]
        score = self.metrics.iloc[idx].test_accuracy.item()
        activation_function = self.metrics.iloc[idx]['config.activation']

        if self.data_format == "nfn":
            return CNNBatch(weights=weights, biases=biases, y=score)

        # Create a mask to denote which layers are convolutional and which are
        # linear
        conv_mask = [1 if w.ndim == 4 else 0 for w in weights]

        layer_layout = [weights[0].shape[1]] + [v.shape[0] for v in biases]

        weights = [
            self._transform_weights_biases(w, self.max_kernel_size,
                                           linear_as_conv=self.linear_as_conv)
            for w in weights
        ]
        biases = [
            self._transform_weights_biases(b, self.max_kernel_size,
                                           linear_as_conv=self.linear_as_conv)
            for b in biases
        ]

        if self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            self.direction,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            mask_hidden=self.hidden_nodes if self.equiv_on_hidden else None,
            mask_first_layer=self.first_layer_nodes if self.get_first_layer_mask else None,
            sign_mask=activation_function == 'tanh')
        return data


def cnn_to_graph(
        weights,
        biases,
        weights_mean=None,
        weights_std=None,
        biases_mean=None,
        biases_std=None,
):
    weights_mean = weights_mean if weights_mean is not None else [0.0] * len(weights)
    weights_std = weights_std if weights_std is not None else [1.0] * len(weights)
    biases_mean = biases_mean if biases_mean is not None else [0.0] * len(biases)
    biases_std = biases_std if biases_std is not None else [1.0] * len(biases)

    # The graph will have as many nodes as the total number of channels in the
    # CNN, plus the number of output dimensions for each linear layer
    device = weights[0].device
    num_input_nodes = weights[0].shape[0]
    num_nodes = num_input_nodes + sum(b.shape[0] for b in biases)

    edge_features = torch.zeros(
        num_nodes, num_nodes, weights[0].shape[-1], device=device
    )

    edge_feature_masks = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)

    row_offset = 0
    col_offset = num_input_nodes  # no edge to input nodes
    for i, w in enumerate(weights):
        num_in, num_out = w.shape[:2]
        edge_features[
        row_offset:row_offset + num_in, col_offset:col_offset + num_out, :w.shape[-1]
        ] = (w - weights_mean[i]) / weights_std[i]
        edge_feature_masks[row_offset:row_offset + num_in, col_offset:col_offset + num_out] = w.shape[-1] == 1
        adjacency_matrix[row_offset:row_offset + num_in, col_offset:col_offset + num_out] = True
        row_offset += num_in
        col_offset += num_out

    node_features = torch.cat(
        [
            torch.zeros((num_input_nodes, 1), device=device, dtype=biases[0].dtype),
            *[(b - biases_mean[i]) / biases_std[i] for i, b in enumerate(biases)]
        ]
    )

    return node_features, edge_features, edge_feature_masks, adjacency_matrix


def cnn_to_tg_data(
        weights,
        biases,
        conv_mask,
        direction,
        weights_mean=None,
        weights_std=None,
        biases_mean=None,
        biases_std=None,
        **kwargs,
):
    node_features, edge_features, edge_feature_masks, adjacency_matrix = cnn_to_graph(
        weights, biases, weights_mean, weights_std, biases_mean, biases_std)
    edge_index = adjacency_matrix.nonzero().t()

    num_input_nodes = weights[0].shape[0]
    cnn_sizes = [w.shape[1] for i, w in enumerate(weights) if conv_mask[i]]
    num_cnn_nodes = num_input_nodes + sum(cnn_sizes)
    send_nodes = num_input_nodes + sum(cnn_sizes[:-1])
    spatial_embed_mask = torch.zeros_like(node_features[:, 0], dtype=torch.bool)
    spatial_embed_mask[send_nodes:num_cnn_nodes] = True
    node_types = torch.cat([
        torch.zeros(num_cnn_nodes, dtype=torch.long),
        torch.ones(node_features.shape[0] - num_cnn_nodes, dtype=torch.long)
    ])

    if direction == 'forward':
        data = Data(
            x=node_features,
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            edge_index=edge_index,
            mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
            spatial_embed_mask=spatial_embed_mask,
            node_types=node_types,
            conv_mask=conv_mask,
            **kwargs,
        )
    else:
        data = Data(
            x=node_features,
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            edge_index=edge_index,
            bw_edge_index=torch.flip(edge_index, [0]),
            bw_edge_attr=torch.reciprocal(edge_features[edge_index[0], edge_index[1]]),
            mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
            spatial_embed_mask=spatial_embed_mask,
            node_types=node_types,
            conv_mask=conv_mask,
            **kwargs,
        )

    return data


def pad_and_flatten_kernel(kernel, max_kernel_size):
    full_padding = (
        max_kernel_size[0] - kernel.shape[2],
        max_kernel_size[1] - kernel.shape[3],
    )
    padding = (
        full_padding[0] // 2,
        full_padding[0] - full_padding[0] // 2,
        full_padding[1] // 2,
        full_padding[1] - full_padding[1] // 2,
    )
    return F.pad(kernel, padding).flatten(2, 3)


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    # label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            # label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])
