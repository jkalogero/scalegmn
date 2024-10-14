from .base_datasets import BaseDataset


class LabeledFashionMnistINRDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            dataset_path,
            split_path,
            debug=False,
            split="train",
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

    def get_path(self, index):
        return self.dataset_path + '/'.join(self.dataset[index].split("/")[-2:]), None

    def get_label(self, index, state_dict, aux):
        return int(state_dict.pop("label"))