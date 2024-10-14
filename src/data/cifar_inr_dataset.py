import os
import glob
import re
from typing import Tuple
from .base_datasets import BaseDataset


class CifarINRDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            dataset_path,
            split_path="",
            debug=False,
            split="train",
            split_points: Tuple[int, int] = None,
            prefix="randinit_smaller",
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            image_size=(28, 28),
            direction='forward',
            layer_layout=None,
            return_path=False,
            data_format="graph",
            switch_to_canon=True
    ):
        self.idx_to_path = {}
        self.idx_to_label = {}
        self.prefix = prefix
        self.split_points = split_points

        super().__init__(
            dataset,
            dataset_path,
            split_path,
            split,
            node_pos_embed,
            edge_pos_embed,
            equiv_on_hidden,
            get_first_layer_mask,
            image_size,
            layer_layout,
            direction,
            return_path,
            data_format,
            switch_to_canon)

        if debug:
            self.dataset = self.dataset[:16]

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, split_path):
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"_(\d)s"

        for siren_path in glob.glob(os.path.join(self.dataset_path, f"{self.prefix}_[0-9]s/*.pth")):
            idx = int(re.search(idx_pattern, siren_path).group(1))
            self.idx_to_path[idx] = siren_path
            label = int(re.search(label_pattern, siren_path).group(1))
            self.idx_to_label[idx] = label
        if self.split == "all":
            dataset = list(range(len(self.idx_to_path)))
        else:
            val_point, test_point = self.split_points
            dataset = {
                "train": list(range(val_point)),
                "val": list(range(val_point, test_point)),
                "test": list(range(test_point, len(self.idx_to_path))),
            }[self.split]
        return dataset

    def get_path(self, index):
        data_idx = self.dataset[index]
        path = self.idx_to_path[data_idx]
        return path, data_idx

    def get_label(self, index, state_dict, data_idx):
        label = self.idx_to_label[data_idx]
        return label