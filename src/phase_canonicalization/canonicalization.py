from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
import yaml
import torch
import sys
from shift_params import shift_params, iterate_l
from tqdm import tqdm
import json
from src.data import dataset
from src.utils.loss import L2_distance


def get_path(path: Path, suffix: str) -> Path:
    path_parts = list(path.parts)
    path_parts[1] = path_parts[1] + suffix
    return Path('/'.join(path_parts))


def store_canonicalized_params(w, b, paths, label, dataset_name):
    sd_keys = ['seq.0.weight', 'seq.0.bias', 'seq.1.weight', 'seq.1.bias', 'seq.2.weight', 'seq.2.bias']
    batch_size = w[0].shape[0]
    for datapoint in range(batch_size):
        dict_of_tensors = OrderedDict({k: [] for k in sd_keys})
        for layer_idx in range(len(w)):
            w_k = f'seq.{layer_idx}.weight'
            b_k = f'seq.{layer_idx}.bias'
            dict_of_tensors[w_k] = w[layer_idx][datapoint]
            dict_of_tensors[b_k] = b[layer_idx][datapoint]

        if dataset_name != 'labeled_mnist_inr':
            dict_of_tensors['label'] = int(label[datapoint]) if dataset_name == 'labeled_fashion_mnist_inr' else label[datapoint]
        new_path = get_path(Path(paths[datapoint]), '_canon')
        target_dir = Path(new_path).parent
        target_dir.mkdir(exist_ok=True, parents=True)
        torch.save(dict_of_tensors, new_path)


def canonicalization(conf, batch_size: int = 1):
    """
    Canonicalize the dataset specified from the conf file.
    The canonicalization regards the phase symmetry of INRs.
    """
    # out_path = Path(conf['save_output']['output_dir'])
    # out_path.mkdir(exist_ok=True, parents=True)

    splits = ['train', 'test', 'val']
    conf['data']['debug'] = False
    extra_aug = conf['data'].pop('extra_aug') if 'extra_aug' in conf['data'] else 0
    if extra_aug> 0 and conf['data']['dataset'] == 'cifar_inr':
        splits = ['train']
    total_distance = 0
    for _split in splits:
        if extra_aug > 0 and conf['data']['dataset'] == 'cifar_inr':
            aug_dsets = []
            for i in range(extra_aug):
                aug_dsets.append(
                    dataset(conf['data'], split='train', prefix=f"randinit_smaller_aug{i}"))
            my_dataset = torch.utils.data.ConcatDataset(aug_dsets)
        else:
            my_dataset = dataset(conf['data'], split=_split)
        data_loader = torch.utils.data.DataLoader(
            dataset=my_dataset, batch_size=batch_size, shuffle=False, num_workers=16
        )
        print(f'Length of {_split} dataset: {len(my_dataset)}')
        for i, batch in enumerate(tqdm(data_loader)):
            data, paths = batch
            new_W, new_b = shift_params(data, args.reconstruct_inr)
            d = L2_distance((data.weights, data.biases), (new_W, new_b))
            total_distance += d
            if d > 0:
                print(f'L2 distance between original and new for path: {paths}: {d}')
            store_canonicalized_params(new_W, new_b, paths, data.label, conf['data']['dataset'])
    print('Total L2 distance between original and new parameters: ', total_distance)
    print(f'Saved canonicalized dataset.')


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Canonicalization script.")
    arg_parser.add_argument('--conf', type=str, required=True, default='canonicalization.yml')
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--reconstruct_inr', type=bool, default=False, help='Reconstruct new INR, for testing purposes.')
    arg_parser.add_argument('--extra_aug', type=int, default=0)
    args = arg_parser.parse_args()

    conf = yaml.safe_load(open(args.conf))
    if conf['data']['dataset'] == 'cifar_inr':
        conf['data']['extra_aug'] = args.extra_aug
    print(yaml.dump(conf, default_flow_style=False))
    canonicalization(conf, batch_size=args.batch_size)

