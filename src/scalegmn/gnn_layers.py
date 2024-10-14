import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from src.scalegmn.layers import MLPNet, SineUpdate, EquivariantNet, EdgeUpdate


class base_GNN_layer(MessagePassing):
    def __init__(self,
                 d_in_v,
                 d_in_e,
                 d_hid,
                 layer_layout,
                 update_edge_attr,
                 symmetry,
                 last_layer,
                 sign_symmetrization,
                 direction,
                 **kwargs):
        aggr = kwargs['aggregator']
        super().__init__(aggr=aggr)
        self.aggr = aggr
        self.nodes_per_layer = layer_layout
        self.update_edge_attr = update_edge_attr
        self.symmetry = symmetry
        self.num_nodes = sum(self.nodes_per_layer)
        self.layer_idx = kwargs['layer_idx']

        self.update_as_act = kwargs['update_as_act']
        self.update_as_act_arg = kwargs['update_as_act_arg']

        self.pos_embed_upd = kwargs['pos_embed_upd']
        self.pos_embed_msg = kwargs['pos_embed_msg']

        self.last_layer = last_layer
        self.equivariant_gnn = kwargs['equivariant']
        self.direction = direction

        if self.update_edge_attr and (not last_layer or self.equivariant_gnn) and direction == 'forward':
            self.update_edge_attr_fn = EdgeUpdate(
                d_in_v,
                1 if self.equivariant_gnn and self.last_layer else d_hid,
                kwargs['mlp_args'],
                symmetry=self.symmetry,
                sign_symmetrization=sign_symmetrization)

    def reset_parameters(self):
        self.message_fn.reset_parameters()
        if hasattr(self, 'update_node_feats_fn'):
            self.update_node_feats_fn.reset_parameters()
        if self.update_edge_attr:
            self.update_edge_attr_fn.reset_parameters()
        if hasattr(self, 'w_v'):
            self.w_v.reset_parameters()
        if hasattr(self, 'w_e'):
            self.w_e.reset_parameters()
        if hasattr(self, 'w_msg'):
            self.w_msg.reset_parameters()
        if hasattr(self, 'w_h'):
            self.w_h.reset_parameters()

    def forward(self, x, edge_index, edge_attr, mask_hidden=None, mask_first_layer=None, pos_embed=None, sign_mask=None):
        aggregated = self.propagate(edge_index=edge_index,
                                    x=x,
                                    edge_attr=edge_attr,
                                    mask_hidden=mask_hidden,
                                    mask_first_layer=mask_first_layer,
                                    pos_embed=pos_embed,
                                    sign_mask=sign_mask)

        if self.direction == 'bidirectional':
            return aggregated

        if self.update_as_act:
            update_arg = torch.cat((x, aggregated), dim=-1) if self.update_as_act_arg == 'cat' else self.w_h(x)+self.w_msg(aggregated)
            out_v = self.update_node_feats_fn(update_arg)

        else:
            out_v = self.update_node_feats_fn(
                torch.cat((x, aggregated), dim=-1),
                extra_features=pos_embed if self.pos_embed_upd else None,
                mask_hidden=mask_hidden,
                mask_first_layer=mask_first_layer,
                sign_mask=sign_mask)

        if self.update_edge_attr and (not self.last_layer or self.equivariant_gnn):
            out_e = self.update_edge_attr_fn(x, edge_index, edge_attr, sign_mask=sign_mask)
        else:
            out_e = edge_attr
        return out_v, out_e

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(\n' + \
                    f'aggregator={self.aggr},\n' + \
                    f'message_fn={self.message_fn},\n' + \
                    f'w_v={self.w_v if hasattr(self, "w_v") else hasattr(self, "w_v")},\n' + \
                    f'w_e={self.w_e if hasattr(self, "w_e") else hasattr(self, "w_e")}, \n' + \
                    f'w_h={self.w_h if hasattr(self, "w_h") else hasattr(self, "w_h")}, \n' + \
                    f'w_msg={self.w_msg if hasattr(self, "w_msg") else hasattr(self, "w_msg")}, \n'
        if self.direction == 'forward':
            repr_str += f'update_node_feats_fn={self.update_node_feats_fn},\n' + \
                        f'update_edge_attr_fn={self.update_edge_attr_fn if hasattr(self, "update_edge_attr_fn") else False})'
        return repr_str


class ScaleEq_GNN_layer(base_GNN_layer):
    """
    Scale Equivariant GNN layer. MSG and UPD functions are Scale Equivariant.
    """
    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden, upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden, sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, sign_symmetrization, direction='forward', **kwargs)

        self.w_v = nn.Linear(d_in_v, d_hid, bias=False)
        self.w_e = nn.Linear(d_in_e, d_hid, bias=False)

        # Scale equivariant message function
        self.message_fn = EquivariantNet(
            kwargs['message_fn_layers'],
            d_hid,
            d_hid,
            kwargs['mlp_args'],
            d_extra=d_hid * self.pos_embed_msg,
            symmetry=symmetry,
            equiv_on_hidden=msg_equiv_on_hidden,
            layer_equiv_on_hidden=layer_msg_equiv_on_hidden,
            mlp_on_io=kwargs['mlp_on_io'],
            num_mlps=msg_num_mlps,
            skip_connections=kwargs['message_fn_skip_connections'],
            sign_symmetrization=sign_symmetrization)

        if self.update_as_act:
            if self.update_as_act_arg == 'cat':
                sine_input_dim = d_hid + d_in_v
            else:
                sine_input_dim = d_in_v
                self.w_msg = nn.Linear(d_hid, sine_input_dim, bias=False)
                self.w_h = nn.Linear(d_in_v, sine_input_dim, bias=False)
            self.update_node_feats_fn = SineUpdate(sine_input_dim, d_hid)  # Sine update function

        else:
            # Scale equivariant update function
            self.update_node_feats_fn = EquivariantNet(
                kwargs['update_node_feats_fn_layers'],
                d_hid + d_in_v,
                1 if self.equivariant_gnn and self.last_layer else d_hid,
                kwargs['mlp_args'],
                d_extra=d_hid * self.pos_embed_upd,
                symmetry=symmetry,
                equiv_on_hidden=upd_equiv_on_hidden,
                layer_equiv_on_hidden=layer_upd_equiv_on_hidden,
                mlp_on_io=kwargs['mlp_on_io'],
                num_mlps=upd_num_mlps,
                skip_connections=kwargs['update_node_feats_fn_skip_connections'],
                sign_symmetrization=sign_symmetrization)

    def message(self,
                edge_index,
                x_i,
                x_j,
                edge_attr,
                mask_hidden=None,
                mask_first_layer=None,
                pos_embed=None,
                sign_mask=None):

        x_j = self.w_v(x_j)
        edge_attr = self.w_e(edge_attr)

        msg_j = self.message_fn(
            edge_attr * x_j,
            extra_features=pos_embed[edge_index[1]] if self.pos_embed_msg else None,
            mask_hidden=mask_hidden[edge_index[1]] if mask_hidden is not None else None,
            mask_first_layer=mask_first_layer[edge_index[1]] if mask_first_layer is not None else None,
            sign_mask=sign_mask[edge_index[1]] if sign_mask is not None else None)
        return msg_j


class GNN_layer(base_GNN_layer):
    """
    Plain GNN layer. MSG and UPD functions are MLPs.
    """
    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden, upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden, sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, sign_symmetrization, direction='forward', **kwargs)

        self.message_fn = MLPNet(
            2*d_hid,
            d_hid,
            kwargs['mlp_args'],
        )
        self.update_node_feats_fn = MLPNet(
            d_hid + d_in_v,
            1 if self.equivariant_gnn and self.last_layer else d_hid,
            kwargs['mlp_args'],
        )

    def message(self, edge_index, x_i, x_j, edge_attr, mask_hidden=None, mask_first_layer=None, pos_embed=None, sign_mask=None):
        msg_j = self.message_fn(
            torch.cat((edge_attr, x_j), dim=-1),
            extra_features=pos_embed[edge_index[1]] if self.pos_embed_msg else None,
            mask_hidden=mask_hidden[edge_index[1]] if mask_hidden is not None else None,
            mask_first_layer=mask_first_layer[edge_index[1]] if mask_first_layer is not None else None)

        return msg_j


class ScaleGMN_GNN_layer_aggr(base_GNN_layer):
    """
    Scale Equivariant GNN layer for the bidirectional variant. MSG function is Scale Equivariant.
    This layer only aggregates the messages from the neighbors. The UPD function is applied later.
    """
    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden, upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden, sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, sign_symmetrization, direction='bidirectional', **kwargs)

        self.w_v = nn.Linear(d_in_v, d_hid, bias=False)
        self.w_e = nn.Linear(d_in_e, d_hid, bias=False)

        self.message_fn = EquivariantNet(
            kwargs['message_fn_layers'],
            d_hid,
            d_hid,
            kwargs['mlp_args'],
            d_extra=d_hid*self.pos_embed_msg,
            symmetry=symmetry,
            equiv_on_hidden=msg_equiv_on_hidden,
            layer_equiv_on_hidden=layer_msg_equiv_on_hidden,
            mlp_on_io=kwargs['mlp_on_io'],
            num_mlps=msg_num_mlps,
            skip_connections=kwargs['message_fn_skip_connections'],
            sign_symmetrization=sign_symmetrization)

    def message(self, edge_index, x_j, edge_attr, mask_hidden=None, mask_first_layer=None, pos_embed=None, sign_mask=None):
        x_j = self.w_v(x_j)
        edge_attr = self.w_e(edge_attr)

        msg_j = self.message_fn(
            edge_attr * x_j,
            extra_features=pos_embed[edge_index[0]] if self.pos_embed_msg else None,
            mask_hidden=mask_hidden[edge_index[1]] if mask_hidden is not None else None,
            mask_first_layer=mask_first_layer[edge_index[1]] if mask_first_layer is not None else None,
            sign_mask=sign_mask[edge_index[1]] if sign_mask is not None else None)
        return msg_j



class GNN_layer_aggr(base_GNN_layer):
    """
    Plain GNN layer for the bidirectional variant. MSG function is MLP.
    """
    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, msg_equiv_on_hidden,
                 msg_num_mlps, upd_equiv_on_hidden, upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden,
                 sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry, last_layer, sign_symmetrization,
                         direction='bidirectional', **kwargs)

        self.message_fn = MLPNet(
            2*d_hid,
            d_hid,
            kwargs['mlp_args'])

    def message(self, edge_index, x_j, edge_attr, mask_hidden=None, mask_first_layer=None, pos_embed=None):
        msg_j = self.message_fn(
            torch.cat((edge_attr, x_j), dim=-1),
            extra_features=pos_embed[edge_index[0]] if self.pos_embed_msg else None,
            mask_hidden=mask_hidden[edge_index[1]] if mask_hidden is not None else None,
            mask_first_layer=mask_first_layer[edge_index[1]] if mask_first_layer is not None else None)

        return msg_j