import argparse


def setup_arg_parser():
    parser = argparse.ArgumentParser()

    #############################################################################################
    # Basic CLI arguments
    #############################################################################################
    # set path to config file
    parser.add_argument('--conf', type=str)
    # set seeds to ensure reproducibility
    parser.add_argument('--train_args.seed', type=int, default=0)
    # specify data loader parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_threads', type=int, default=1)
    # specify training parameters
    parser.add_argument("--debug", type=str2bool, help="Use a subset of a few datapoints, meant for debugging.", default=False)
    parser.add_argument("--validate", type=str2bool, help="Whether to validate on val split after every epoch.", default=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=0, help="List of ids of GPUs to use.")
    # wandb
    parser.add_argument("--wandb", type=str2bool, help="Log results to wandb.", default=False)

    #############################################################################################
    # Dataset arguments
    #############################################################################################
    parser.add_argument('--data.dataset_path', type=str)
    parser.add_argument('--data.split_path', type=str)
    parser.add_argument('--data.extra_aug', type=int)
    parser.add_argument('--data.switch_to_canon', type=str2bool)

    #############################################################################################
    # Optimization arguments
    #############################################################################################
    parser.add_argument('--optimization.optimizer_name', type=str)
    parser.add_argument('--optimization.clip_grad', type=str2bool)
    parser.add_argument('--optimization.clip_grad_max_norm', type=float)
    parser.add_argument('--optimization.optimizer_args.lr', type=float)
    parser.add_argument('--optimization.optimizer_args.weight_decay', type=float)
    parser.add_argument('--optimization.scheduler_args.scheduler', type=str)
    parser.add_argument('--optimization.scheduler_args.scheduler_mode', type=str)
    parser.add_argument('--optimization.scheduler_args.decay_rate', type=float)
    parser.add_argument('--optimization.scheduler_args.decay_steps', type=float)
    parser.add_argument('--optimization.scheduler_args.patience', type=float)
    parser.add_argument('--optimization.scheduler_args.min_lr', type=float)

    parser.add_argument('--train_args.num_epochs', type=int)
    parser.add_argument('--train_args.val_acc_threshold', type=float)
    parser.add_argument('--train_args.patience', type=int)
    parser.add_argument('--train_args.loss', type=str)

    #############################################################################################
    # ScaleNet arguments
    #############################################################################################
    parser.add_argument('--scalegmn_args.d_in_v', type=int)
    parser.add_argument('--scalegmn_args.d_in_e', type=int)
    parser.add_argument('--scalegmn_args.d_hid', type=int)
    parser.add_argument('--scalegmn_args.num_layers', type=int)
    parser.add_argument('--scalegmn_args.direction', type=str)
    parser.add_argument('--scalegmn_args.symmetry', type=str)
    parser.add_argument('--scalegmn_args.jit', type=str2bool)
    parser.add_argument('--scalegmn_args.compile', type=str2bool)
    parser.add_argument('--scalegmn_args.readout_range', type=str, choices=['last_layer', 'full_graph'])
    parser.add_argument('--scalegmn_args.gnn_skip_connections', type=str2bool)
    parser.add_argument('--scalegmn_args.concat_mlp_directions', type=str2bool)
    parser.add_argument('--scalegmn_args.reciprocal', type=str2bool)
    parser.add_argument('--scalegmn_args.node_pos_embed', type=str2bool)
    parser.add_argument('--scalegmn_args.edge_pos_embed', type=str2bool)

    # graph_init
    parser.add_argument('--scalegmn_args.graph_init.project_node_feats', type=str2bool)
    parser.add_argument('--scalegmn_args.graph_init.project_edge_feats', type=str2bool)
    parser.add_argument('--scalegmn_args.graph_init.input_layers', type=int)

    # positional encodings
    parser.add_argument('--scalegmn_args.positional_encodings.final_linear_pos_embed', type=str2bool)
    parser.add_argument('--scalegmn_args.positional_encodings.sum_pos_enc', type=str2bool)
    parser.add_argument('--scalegmn_args.positional_encodings.po_as_different_linear', type=str2bool)
    parser.add_argument('--scalegmn_args.positional_encodings.equiv_net', type=str2bool)
    parser.add_argument('--scalegmn_args.positional_encodings.sum_on_io', type=str2bool)
    parser.add_argument('--scalegmn_args.positional_encodings.equiv_on_hidden', type=str2bool)
    parser.add_argument('--scalegmn_args.positional_encodings.num_mlps', type=int)
    parser.add_argument('--scalegmn_args.positional_encodings.layer_equiv_on_hidden', type=str2bool)

    # gnn_args
    parser.add_argument('--scalegmn_args.gnn_args.message_fn_layers', type=int)
    parser.add_argument('--scalegmn_args.gnn_args.message_fn_skip_connections', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.update_node_feats_fn_layers', type=int)
    parser.add_argument('--scalegmn_args.gnn_args.update_node_feats_fn_skip_connections', type=str2bool)

    parser.add_argument('--scalegmn_args.gnn_args.update_edge_attr', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.dropout', type=float)
    parser.add_argument('--scalegmn_args.gnn_args.dropout_all', type=str2bool)

    parser.add_argument('--scalegmn_args.gnn_args.msg_equiv_on_hidden', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.layer_msg_equiv_on_hidden', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.upd_equiv_on_hidden', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.layer_upd_equiv_on_hidden', type=str2bool)

    parser.add_argument('--scalegmn_args.gnn_args.mlp_on_io', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.msg_num_mlps', type=int)
    parser.add_argument('--scalegmn_args.gnn_args.upd_num_mlps', type=int)

    parser.add_argument('--scalegmn_args.gnn_args.pos_embed_msg', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.pos_embed_upd', type=str2bool)

    parser.add_argument('--scalegmn_args.gnn_args.update_as_act', type=str2bool)
    parser.add_argument('--scalegmn_args.gnn_args.update_as_act_arg', type=str)
    parser.add_argument('--scalegmn_args.gnn_args.aggregator', type=str)
    parser.add_argument('--scalegmn_args.gnn_args.sign_symmetrization', type=str2bool)

    # mlp_args
    parser.add_argument('--scalegmn_args.mlp_args.d_k', type=str2list2int)
    parser.add_argument('--scalegmn_args.mlp_args.activation', type=str)
    parser.add_argument('--scalegmn_args.mlp_args.dropout', type=float)
    parser.add_argument('--scalegmn_args.mlp_args.final_activation', type=str)
    parser.add_argument('--scalegmn_args.mlp_args.batch_norm', type=str2bool)
    parser.add_argument('--scalegmn_args.mlp_args.layer_norm', type=str2bool)
    parser.add_argument('--scalegmn_args.mlp_args.bias', type=str2bool)
    parser.add_argument('--scalegmn_args.mlp_args.skip', type=str2bool)

    # readout_args
    parser.add_argument('--scalegmn_args.readout_args.d_rho', type=int)
    parser.add_argument('--scalegmn_args.readout_args.d_out', type=int)

    #############################################################################################
    # Wandb arguments
    #############################################################################################
    parser.add_argument('--wandb_args.project', type=str)
    parser.add_argument('--wandb_args.config_file', type=str)
    parser.add_argument('--wandb_args.entity', type=str)
    parser.add_argument('--wandb_args.name', type=str)
    parser.add_argument('--wandb_args.group', type=str)
    parser.add_argument('--wandb_args.tags', type=str)

    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    return [c for c in v.split(',')]


def str2list2int(v):
    return [int(c) for c in v.split(',')]


def none_or_str(value):
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return value