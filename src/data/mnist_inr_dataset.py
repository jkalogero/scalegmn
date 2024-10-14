from .base_datasets import BaseDataset
from torchvision import transforms
import numpy as np
from src.utils.image_processing import style_edit
import torchvision
from pathlib import Path


class LabeledINRDataset(BaseDataset):
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
            image_size=(28,28),
            layer_layout=None,
            direction='forward',
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
            self.dataset = {k: v[:16] for k,v in self.dataset.items()}

    def __len__(self):
        return len(self.dataset["label"])

    def get_path(self, index):
        return self.dataset["path"][index], None
    
    def get_label(self, index, state_dict, aux):
        return int(self.dataset["label"][index])


class MNISTINRImageDataset(LabeledINRDataset):
    def __init__(self, img_ds, style_function, **kwargs):
        super().__init__(**kwargs)
        self.dataset["path"] = [Path(kwargs['dataset_path'] + '/'.join(p.split("/")[2:])) for p in self.dataset["path"]]
        self.img_ds = getattr(torchvision.datasets, 'MNIST')(
            root=kwargs['dataset_path'],
            train=(self.split in ['train', 'val']),
            download=True,
            transform=None)

        transformation = style_edit(style_function)
        self.img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                transformation,
                transforms.ToTensor(),
                # transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ]
        )
        self.input_img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, item):
        batch, w_b = super().__getitem__(item)
        img_id = int(self.dataset["path"][item].parts[-3].split("_")[-1])
        img, _ = self.img_ds[img_id]
        transformed_img = self.img_transform(img)
        input_img = self.input_img_transform(img)
        return batch, w_b, transformed_img, input_img
