from .mnist_inr_dataset import LabeledINRDataset, MNISTINRImageDataset
from .fmnist_inr_dataset import LabeledFashionMnistINRDataset
from .cifar10_dataset import NFNZooDataset, CNNDataset
from .cifar_inr_dataset import CifarINRDataset


def dataset(dataset_config, **kwargs):
    _map = {
        'labeled_mnist_inr': LabeledINRDataset,
        'labeled_fashion_mnist_inr': LabeledFashionMnistINRDataset,
        'cifar_inr': CifarINRDataset,
        'cifar10': NFNZooDataset,
        'svhn': NFNZooDataset,
        'mnist_inr_edit': MNISTINRImageDataset
    }
    return _map[dataset_config["dataset"]](**dataset_config, **kwargs)