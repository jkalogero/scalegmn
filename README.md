# Scale Equivariant Graph Metanetworks
Official repository for the NeurIPS'24 paper "Scale Equivariant Graph Metanetworks" 
by Ioannis Kalogeropoulos*, Giorgos Bouritsas* and Yannis Panagakis.

[[arXiv](https://arxiv.org/pdf/2406.10685)]
## Abstract
_This paper pertains to an emerging machine learning paradigm: learning higher-order functions, 
i.e. functions whose inputs are functions themselves, _particularly
when these inputs are Neural Networks (NNs)_. With the growing interest in architectures
that process NNs, a recurring design principle has permeated the field:
adhering to the permutation symmetries arising from the connectionist structure of
NNs. _However, are these the sole symmetries present in NN parameterizations?_
Zooming into most practical activation functions (e.g. sine, ReLU, tanh) answers
this question negatively and gives rise to intriguing new symmetries, which we
collectively refer to as _scaling symmetries_, that is, non-zero scalar multiplications
and divisions of weights and biases. In this work, we propose Scale Equivariant
Graph MetaNetworks - ScaleGMNs, a framework that adapts the Graph Metanetwork (message-passing) paradigm by incorporating scaling symmetries and thus
rendering neuron and edge representations equivariant to valid scalings. We introduce novel building blocks, of independent technical interest, that allow for
equivariance or invariance with respect to individual scalar multipliers or their
product and use them in all components of ScaleGMN. Furthermore, we prove
that, under certain expressivity conditions, ScaleGMN can simulate the forward
and backward pass of any input feedforward neural network. Experimental results
demonstrate that our method advances the state-of-the-art performance for several
datasets and activation functions, highlighting the power of scaling symmetries as
an inductive bias for NN processing._

  * [Setup](#setup)
  * [Data](#data)
  * [Experiments](#experiments)
  * [Citation](#citation)



## Setup

To create a clean virtual environment and install the necessary dependencies execute:
```bash
git clone git@github.com:jkalogero/scalegmn.git
cd scalegmn/
conda env create -n scalegmn --file environment.yml
conda activate scalegmn
```


## Data
First, create the `data/` directory in the root of the repository:
```bash
mkdir data
````
Alternatively, you can specify a different directory for the data by changing
the corresponding fields in the config file.

### INR Classification and Editing
For the INR datasets, we use the data provided by [DWS](https://github.com/AvivNavon/DWSNets) and [NFN](https://github.com/AllanYangZhou/nfn/).
The datasets can be downloaded from the following links: 

- [MNIST-INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=mnist-inrs.zip) - ([Navon et al. 2023](https://arxiv.org/abs/2301.12780))
- [FMNIST-INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=fmnist_inrs.zip) - ([Navon et al. 2023](https://arxiv.org/abs/2301.12780))
- [CIFAR10-INRs](https://drive.google.com/file/d/14RUV3eN6-lSOr9XuwyKFQFVcqKl0L2bw/view?usp=drive_link) - ([Zhou et al. 2023](https://arxiv.org/abs/2302.14040))

Download the datasets and extract them in the directory `data/`. For example, you can run the following to download
and extract the MNIST-INR dataset and generate the splits:
```bash
DATA_DIR=./data
wget "https://www.dropbox.com/sh/56pakaxe58z29mq/AABrctdu2U65jGYr2WQRzmMna/mnist-inrs.zip?dl=0" -O "$DATA_DIR/mnist-inrs.zip"
unzip -q "$DATA_DIR/mnist-inrs.zip" -d "$DATA_DIR"
rm "$DATA_DIR/mnist-inrs.zip" # remove the zip file
# generate the splits
python src/utils/generate_data_splits.py --data_path $DATA_DIR/mnist-inrs --save_path $DATA_DIR/mnist-inrs
```

Generating the splits is necessary only for the MNIST-INR dataset.

#### Phase canonicalization
For the INR datasets, we preprocess each datapoint to canonicalize the phase symmetry (see [Algorithm 1](https://arxiv.org/pdf/2406.10685v1#algocf.1) in the appendix).
To run the phase canonicalization script, run the following command:
```bash
python src/phase_canonicalization/canonicalization.py --conf src/phase_canonicalization/<dataset>.yml
```
where `<dataset>` can be one of `mnist`, `fmnist`, `cifar`.

To apply the canonicalization to the augmented CIFAR10-INR dataset, also run:
```bash 
python src/phase_canonicalization/canonicalization.py --conf src/phase_canonicalization/cifar.yml --extra_aug 20
```

The above script will store the canonicalized dataset in a new directory `data/<dataset>_canon/`. The training scripts will automatically use the canonicalized dataset, if it exists.
To use the dataset specified in the config file (and not search for `data/<dataset>_canon/`), set the `data.switch_to_canon` field of the config to `False` or simply use the CLI argument `--data.switch_to_canon False`. 

### Generalization prediction
We follow the experiments from [NFN](https://github.com/AllanYangZhou/nfn/) and use the datasets provided by [Unterthiner et al,
2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy). The datasets can be downloaded from the following links:
- [CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz)
- [SVHN](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/svhn_cropped.tar.xz)


Similarly, extract the dataset in the directory `data/` and execute:

For the CIFAR10 dataset:
```bash
tar -xvf cifar10.tar.xz
# download cifar10 splits
wget https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits/cifar10_split.csv -O data/cifar10/cifar10_split.csv
```
For the SVHN dataset:
```bash
tar -xvf svhn_cropped.tar.xz
# download svhn splits
wget https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits/svhn_split.csv -O data/svhn_cropped/svhn_split.csv
```

 

## Experiments
For every experiment, we provide the corresponding configuration file in the `config/` directory.
Each config contains the selected hyperparameters for the experiment, as well as the paths to the dataset.
To enable wandb logging, use the CLI argument `--wandb True`. For more useful CLI arguments, check the [src/utils/setup_arg_parser.py](src/utils/setup_arg_parser.py) file.

**Note:** To employ a GMN accounting only for the permutation symmetries, simply set 
`--scalegmn_args.symmetry=permutation`.

### INR Classification
To train and evaluate ScaleGMN on the INR classification task, 
select any config file under [configs/mnist_cls](configs/mnist_cls)
, [configs/fmnist_cls](configs/fmnist_cls) or 
[configs/cifar_inr_cls](configs/cifar_inr_cls). For example, to 
train ScaleGMN on the FMNIST-INR dataset, execute the following:
```bash
python inr_classification.py --conf configs/fmnist_cls/scalegmn.yml
```

### INR Editing
To train and evaluate ScaleGMN on the INR editing task, use the configs under
[configs/mnist_editing](configs/mnist_editing) directory and execute:

```bash
python inr_editing.py --conf configs/mnist_editing/scalegmn_bidir.yml
```

### Generalization prediction
To train and evaluate ScaleGMN on the INR classification task, 
select any config file under [configs/cifar10](configs/cifar10)
or [configs/svhn](configs/svhn). For example, to 
train ScaleGMN on the CIFAR10 dataset on heterogeneous activation functions,
execute the following:

```bash
python predicting_generalization.py --conf configs/cifar10/scalegmn_hetero.yml
```

# Citation

```bib
@article{kalogeropoulos2024scale,
    title={Scale Equivariant Graph Metanetworks},
    author={Kalogeropoulos, Ioannis and Bouritsas, Giorgos and Panagakis, Yannis},
    journal={Advances in Neural Information Processing Systems},
    year={2024}
}
```