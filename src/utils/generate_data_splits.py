import json
import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

# from experiments.utils import common_parser, set_logger


def generate_splits(data_path, save_path, name="mnist_splits.json", val_size=5000):
    save_path = Path(save_path) / name
    inr_path = Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for p in list(inr_path.glob("mnist_png_*/**/*.pth")):
        if "train" in p.as_posix():
            s = "train"
        else:
            s = "test"
        data_split[s]["path"].append(p.as_posix())
        data_split[s]["label"].append(p.parent.parent.stem.split("_")[-2])

    # val split
    val_size = val_size
    train_indices, val_indices = train_test_split(
        range(len(data_split["train"]["path"])), test_size=val_size
    )
    data_split["val"]["path"] = [data_split["train"]["path"][v] for v in val_indices]
    data_split["val"]["label"] = [data_split["train"]["label"][v] for v in val_indices]

    data_split["train"]["path"] = [data_split["train"]["path"][v] for v in train_indices]
    data_split["train"]["label"] = [data_split["train"]["label"][v] for v in train_indices]

    with open(save_path, "w") as file:
        json.dump(data_split, file)


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate data splits")
    parser.add_argument('--data_path', type=str, default='data/mnist-inrs')
    parser.add_argument('--save_path', type=str, default='data/mnist-inrs')
    parser.add_argument(
        "--name", type=str, default="mnist_splits.json", help="json file name"
    )
    parser.add_argument(
        "--val-size", type=int, default=5000, help="number of validation examples"
    )
    args = parser.parse_args()

    generate_splits(
        data_path=args.data_path,
        save_path=args.save_path,
        name=args.name,
        val_size=args.val_size,
    )
