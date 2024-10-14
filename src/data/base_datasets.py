from abc import ABC, abstractmethod
import json
from typing import NamedTuple, Tuple, Union
import torch
from nfn.common import state_dict_to_tensors
from .data_utils import to_pyg_batch, get_node_types, get_edge_types, nn_to_edge_index
from pathlib import Path


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(
            self,
            dataset,
            dataset_path,
            split_path,
            split="train",
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            image_size=(28, 28),
            layer_layout=None,
            direction='forward',
            return_path=False,
            data_format="graph",
            switch_to_canon=True
    ):
        super().__init__()
        self.dataset_name = dataset
        self.split = split
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(split_path)
        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.layer_layout = layer_layout
        self.direction = direction
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask
        self.data_format = data_format

        self.image_size = image_size
        self.return_path = return_path
        self.return_wb = self.dataset_name == 'mnist_inr_edit'
        self.switch_to_canon = switch_to_canon

        if self.switch_to_canon:
            path = Path(self.dataset_path)
            canon = '/'.join(path.parts[:2]) + '_canon/'
            if Path(canon).exists():
                print("Found canonicalized dataset.")
                self.dataset_path = canon
            else:
                print("Canonicalized dataset not found. Using original dataset.")

        self.edge_index = nn_to_edge_index(self.layer_layout, "cpu", dtype=torch.long)

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, aux = self.get_path(index)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = self.get_label(index, state_dict, aux)

        if not self.return_path:
            weights = tuple([v.permute(1, 0) for w, v in state_dict.items() if "weight" in w])
            weights = tuple([w.unsqueeze(-1) for w in weights])

            biases = tuple([v for w, v in state_dict.items() if "bias" in w])
            biases = tuple([b.unsqueeze(-1) for b in biases])

        else:
            weights = tuple([v for w, v in state_dict.items() if "weight" in w])
            biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        if self.data_format == "wb":
            if self.return_path:
                # reserved for the canonicalization script.
                return Batch(weights=weights, biases=biases, label=label), path
            else:
                return Batch(weights=weights, biases=biases, label=label)

        # convert to graph
        node_features, edge_features = self.batch_to_graphs(weights, biases)
        # convert to pyg data object
        batch, _ = to_pyg_batch(
            node_features,
            edge_features,
            self.edge_index,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            direction=self.direction,
            label=label,
            hidden_nodes=self.hidden_nodes if self.equiv_on_hidden else None,
            first_layer_nodes=self.first_layer_nodes if self.get_first_layer_mask else None
        )

        if self.return_wb:
            w_b = Batch(weights=weights, biases=biases, label=label)
            return batch, w_b
        else:
            return batch


    def load_dataset(self, split_path):
        return json.load(open(split_path, "r"))[self.split]

    @abstractmethod
    def get_path(self, index):
        pass

    @abstractmethod
    def get_label(self, index, state_dict, aux):
        pass

    def get_layer_layout(self):
        return self.layer_layout

    def get_sd_keys(self):
        return self.state_dict_keys

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

    def batch_to_graphs(
            self,
            weights,
            biases,
            input_emb=None,
            **kwargs
    ):
        num_nodes = weights[0].shape[0] + sum(w.shape[1] for w in weights)

        node_features = torch.zeros(num_nodes, biases[0].shape[-1])
        edge_features = torch.zeros(num_nodes, num_nodes, weights[0].shape[-1])

        row_offset = 0
        col_offset = weights[0].shape[0]  # no edge to input nodes
        for i, w in enumerate(weights):
            num_in, num_out, _ = w.shape
            edge_features[row_offset : row_offset + num_in, col_offset : col_offset + num_out] = w
            row_offset += num_in
            col_offset += num_out

        row_offset = weights[0].shape[0]  # no bias in input nodes

        if input_emb is not None:
            node_features[:, 0: row_offset] = input_emb
        else:
            node_features[:, 0: row_offset] = torch.tensor([1])  # set input node state to 1.

        for i, b in enumerate(biases):
            num_out, _ = b.shape
            node_features[row_offset : row_offset + num_out] = b
            row_offset += num_out

        return node_features, edge_features