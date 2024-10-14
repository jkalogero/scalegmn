from typing import List, Tuple, Union, Dict
import numpy as np
import torch
import torch.nn as nn
import random
from src.utils.setup_arg_parser import none_or_str
import os
from torch.distributed import init_process_group


def overwrite_conf(config: dict, cli_args: dict) -> dict:
    """
    Function that iterates `args` and overwrites the corresponding `config` value.
    Replaces config[section][hyper_param] with args[section.hyper_param]
    e.g.: config['encoder']['out_features'] -> args['encoder.out_features']
    For arbitrary length of section
    """
    for arg in dict(cli_args):
        sec_hparam = arg.split(".")
        # nested parameters
        if len(sec_hparam) >= 2:
            sec = sec_hparam[0]
            hparam = sec_hparam[-1]
            if sec in config:
                nested_dict = config[sec]
                for key in sec_hparam[1:-1]:
                    if key in nested_dict:
                        nested_dict = nested_dict[key]
                    else:
                        break
                if hparam in nested_dict and cli_args[arg] is not None:
                    nested_dict[hparam] = none_or_str(cli_args[arg])
        else:
            if cli_args[arg] is not None:
                config[arg] = cli_args[arg]

    config = update_common_args(config)
    return config


def update_common_args(config: dict) -> dict:
    """
    Function that updates common args in config.
    """
    config['scalegmn_args']['graph_init']['d_node'] = config['scalegmn_args']['d_hid']
    config['scalegmn_args']['graph_init']['d_edge'] = config['scalegmn_args']['d_hid']
    config['scalegmn_args']['gnn_args']['d_hid'] = config['scalegmn_args']['d_hid']
    config['data']['node_pos_embed'] = config['scalegmn_args']['node_pos_embed']
    config['data']['edge_pos_embed'] = config['scalegmn_args']['edge_pos_embed']

    if 'readout_args' in config['scalegmn_args']:
        config['scalegmn_args']['readout_args']['d_rho'] = config['scalegmn_args']['d_hid']

    # dropout_all
    if config['scalegmn_args']['gnn_args']['dropout_all']:
        config['scalegmn_args']['mlp_args']['dropout'] = config['scalegmn_args']['gnn_args']['dropout']

    # dk
    for layer_idx in range(len(config['scalegmn_args']['mlp_args']['d_k'])):
        config['scalegmn_args']['mlp_args']['d_k'][layer_idx] = config['scalegmn_args']['d_hid']

    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # torch_geometric.seed(conf['train_args']['seed'])
    print('[info] Setting all random seeds {}'.format(seed))


def count_parameters(**kwargs):
    def _count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    c = 0
    for n, t in kwargs.items():
        cnt = _count(t)
        print(f"{n} params: {cnt:,}")
        c += cnt
    return c


def count_named_parameters(model: nn.Module):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(f"{name} params: {params:,}")
        total_params += params
    print(f"Total Trainable Params: {total_params:,}")
    return total_params


def mask_input(conf: Dict) -> bool:
    return_mask = conf['scalegmn_args']['gnn_args']['msg_num_mlps'] == 3 \
                           or conf['scalegmn_args']['gnn_args']['upd_num_mlps'] == 3
    return return_mask


def mask_hidden(conf: Dict) -> bool:
    return_mask = conf['scalegmn_args']['gnn_args']['msg_equiv_on_hidden'] \
                    or conf['scalegmn_args']['gnn_args']['upd_equiv_on_hidden'] \
                    or conf['scalegmn_args']['gnn_args']['layer_msg_equiv_on_hidden'] \
                    or conf['scalegmn_args']['gnn_args']['layer_upd_equiv_on_hidden']
    return return_mask


def assert_symms(conf: dict) -> None:
    assert (
            (conf["data"]["activation_function"] == 'relu' and conf["scalegmn_args"]["symmetry"] == 'scale') or
            (conf["data"]["activation_function"] == 'tanh' and conf["scalegmn_args"]["symmetry"] == 'sign') or
            (conf["data"]["activation_function"] is None and conf["scalegmn_args"]["symmetry"] == 'hetero') or
            conf["scalegmn_args"]["symmetry"] == 'permutation')
