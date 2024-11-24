import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from src.scalegmn.graph_init import GraphInit
from src.scalegmn.utils import graph_to_wb
from src.scalegmn.layers import DeepSet, PermScaleInvariantReadout, MLPNet, PositionalEncoding, EquivariantNet, EdgeUpdate
from src.scalegmn.gnn_layers import ScaleEq_GNN_layer, GNN_layer, ScaleGMN_GNN_layer_aggr, GNN_layer_aggr
from src.scalegmn.mlp import mlp
from src.data.data_utils import get_node_types, get_edge_types


class BaseScaleGMN(nn.Module):
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.nodes_per_layer = model_args['layer_layout']
        self.num_nodes = sum(self.nodes_per_layer)
        self.layer_idx = torch.cumsum(torch.tensor([0] + model_args['layer_layout']), dim=0)

        self.node_pos_embed = model_args['node_pos_embed']
        self.edge_use_pos_embed = model_args['edge_pos_embed']

        self.direction = model_args['direction']

        self.model = {
            'forward': ScaleGMN_GNN_fw,
            'bidirectional': ScaleGMN_GNN_bidir
        }

        # if convolution
        if '_max_kernel_height' in model_args:
            model_args['graph_init']['d_in_e'] = model_args['_max_kernel_height'] * model_args['_max_kernel_width']

        self.construct_graph = GraphInit(
            **model_args['graph_init'])

        if self.node_pos_embed:
            self.node2type = get_node_types(self.nodes_per_layer)
            self.positional_embeddings = PositionalEncoding(**self.set_pe_args(model_args, 'node'))

        if self.edge_use_pos_embed:
            self.edge2type = get_edge_types(self.nodes_per_layer)
            self.positional_embeddings_edge = PositionalEncoding(**self.set_pe_args(model_args, 'edge'))

        self.gnn = self.model[self.direction](model_args, layer_idx=self.layer_idx)

    def forward(self, batch, w=None, b=None):
        batch = self.construct_graph(batch)
        # apply positional embeddings
        pos_embed, edge_pos_embed = None, None
        if self.node_pos_embed:
            batch, pos_embed = self.positional_embeddings(batch)

        if self.edge_use_pos_embed:
            batch, edge_pos_embed = self.positional_embeddings_edge(batch)

        return batch, pos_embed, edge_pos_embed

    def set_pe_args(self, model_args, param) -> dict:
        pe_args = {
            'd_in': model_args['graph_init'][f'd_{param}'] if model_args['graph_init'][f'project_{param}_feats'] else model_args['graph_init']['d_in'],
            'd_hid': model_args['graph_init'][f'd_{param}'],
            'num_param_types': max(getattr(self, f"{param}2type")) + 1,
            'param': param,
            **model_args['positional_encodings'],
            'num_layers': model_args['gnn_args']['message_fn_layers'],
            'mlp_args': model_args['mlp_args'],
            'symmetry': model_args['symmetry'],
            'sign_symmetrization': model_args['gnn_args']['sign_symmetrization'],
            'mlp_on_io': model_args['gnn_args']['mlp_on_io'],
        }
        return pe_args


class ScaleGMN(BaseScaleGMN):
    """
    Invariant ScaleGMN model
    """
    def __init__(self, model_args, **kwargs):
        super().__init__(model_args, **kwargs)

    def forward(self, batch, w=None, b=None):
        batch, pos_embed, edge_pos_embed = super().forward(batch)
        graph_features = self.gnn(batch, self.num_nodes, pos_embed)
        return graph_features


class ScaleGMN_equiv(BaseScaleGMN):
    """
    Equivariant ScaleGMN model
    """
    def __init__(self, model_args, **kwargs):
        super().__init__(model_args, **kwargs)

        self.weight_scale = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(model_args['out_scale']))
                for _ in range(len(model_args['layer_layout']) - 1)
            ]
        )
        self.bias_scale = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(model_args['out_scale']))
                for _ in range(len(model_args['layer_layout']) - 1)
            ]
        )

    def forward(self, batch, w=None, b=None):
        batch, pos_embed, edge_pos_embed = super().forward(batch)
        x, edge_attr = self.gnn(batch, self.num_nodes, pos_embed)
        node_features, m = to_dense_batch(x, batch.batch)
        edge_features = to_dense_adj(batch.edge_index, batch.batch, edge_attr)

        weights, biases = graph_to_wb(
            edge_features=edge_features,
            node_features=node_features,
            weights=w,
            biases=b
        )

        weights = [_w * s for _w, s in zip(weights, self.weight_scale)]
        biases = [_b * s for _b, s in zip(biases, self.bias_scale)]

        return weights, biases


class ScaleGMN_GNN(nn.Module):
    """
    Base class for Scale Equivariant Graph Neural Network
    """
    def __init__(self, model_args, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.direction = model_args['direction']
        self.dropout = model_args['gnn_args']['dropout']
        self.equivariant = model_args['equivariant']
        self.gnn_skip_connections = model_args['gnn_skip_connections']
        self.symmetry = model_args['symmetry']
        self.d_hid = model_args['gnn_args']['d_hid']

        if not self.equivariant:
            self.readout_range = model_args['readout_range']
            self.only_last_layer = self.readout_range == 'last_layer'

        self.gnn_layer = {
            'forward': ScaleEq_GNN_layer if self.symmetry != 'permutation' else GNN_layer,
            'bidirectional': ScaleGMN_GNN_layer_aggr if self.symmetry != 'permutation' else GNN_layer_aggr
        }

        fw_layers = []
        for i in range(model_args['num_layers']):
            if i == 0:
                d_in_v = model_args['graph_init']['d_node']
                d_in_e = model_args['graph_init']['d_edge']

            else:
                d_in_v = self.d_hid
                if model_args['gnn_args']['update_edge_attr']:
                    d_in_e = self.d_hid
                else:
                    d_in_e = model_args['graph_init']['d_edge']

            model_args['gnn_args']['update_edge_attr'] = model_args['gnn_args']['update_edge_attr'] and i < model_args['num_layers']

            fw_layers.append(self.gnn_layer[self.direction](
                d_in_v=d_in_v,
                d_in_e=d_in_e,
                layer_idx=layer_idx,
                layer_layout=model_args['layer_layout'],
                symmetry=self.symmetry,
                last_layer=i == model_args['num_layers']-1,
                equivariant=model_args['equivariant'],
                mlp_args=model_args['mlp_args'],
                **model_args['gnn_args']
            ))

        self.fw_layers = nn.ModuleList(fw_layers)

        if self.gnn_skip_connections:
            n_skips = model_args['num_layers'] if not self.equivariant else model_args['num_layers'] - 1
            self.skip_layers = nn.ModuleList(
                [nn.Linear(self.d_hid, self.d_hid, bias=False) for _ in
                 range(n_skips)])

        if not self.equivariant:
            readout = self.get_readout()

            self.last_layer_nodes = self.layer_idx[-1] - self.layer_idx[-2]
            self.first_layer_nodes = self.layer_idx[1] - self.layer_idx[0]

            d_in_map = {
                'forward': {
                    True: self.last_layer_nodes * self.d_hid,  # only last layer
                    False: self.d_hid},
                'bidirectional': {
                    True: 2 * self.d_hid,
                    False: self.d_hid},
            }
            self.d_in_readout = d_in_map[self.direction][self.only_last_layer]

            self.readout = readout(
                d_in=self.d_in_readout,
                sym=self.symmetry,
                sign_symmetrization=model_args['gnn_args']['sign_symmetrization'],
                mlp_args=model_args['mlp_args'],
                **model_args['readout_args'],
                num_io_nodes=(self.last_layer_nodes + self.first_layer_nodes))

    def forward(self, batch, num_nodes=None, pos_embed=None, pos_embed_edge=None):
        pass

    def get_readout(self):
        if self.only_last_layer:
            readout_net = MLPNet
        else:
            if self.symmetry != 'permutation':
                readout_net = PermScaleInvariantReadout
            else:
                readout_net = DeepSet

        return readout_net


class ScaleGMN_GNN_fw(ScaleGMN_GNN):
    """
    Scale Equivariant Graph Neural Network for the forward variant
    """
    def __init__(self, model_args, layer_idx):
        super().__init__(model_args, layer_idx)

    def forward(self, batch, num_nodes=None, pos_embed=None, pos_embed_edge=None):
        for i, layer in enumerate(self.fw_layers):
            if not i:
                x = batch.x
                edge_attr = batch.edge_attr

            x_tilde, edge_attr = layer(x=x,
                                       edge_index=batch.edge_index,
                                       edge_attr=edge_attr,
                                       mask_hidden=batch.mask_hidden if hasattr(batch, 'mask_hidden') else None,
                                       mask_first_layer=batch.mask_first_layer if hasattr(batch, 'mask_first_layer') else None,
                                       pos_embed=pos_embed if pos_embed is not None else None,
                                       sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None,
                                       )

            apply_skip = self.gnn_skip_connections and i < len(self.skip_layers)
            x = self.skip_layers[i](x) + x_tilde if apply_skip else x_tilde

            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.equivariant:
            return x, edge_attr

        else:
            if not self.only_last_layer:
                graph_features = self.readout(x,
                                              batch=batch.batch,
                                              mask_hidden=batch.mask_hidden if hasattr(batch, 'mask_hidden') else None,
                                              sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None
                                              )
            else:
                node_features = x.reshape(batch.num_graphs, num_nodes, x.shape[-1])  # change if processing varying architectures
                node_features = node_features[:, -self.last_layer_nodes:].flatten(1, 2)
                if pos_embed is not None:
                    pos_embed = pos_embed.reshape(batch.num_graphs, num_nodes, pos_embed.shape[-1])
                    pos_embed = pos_embed[:, -self.last_layer_nodes:].flatten(1, 2)

                graph_features = self.readout(node_features, pos_embed)

            return graph_features

    def __repr__(self):
        parent_repr = super().__repr__()

        return parent_repr + f'\ndropout between layers={self.dropout}'


class ScaleGMN_GNN_bidir(ScaleGMN_GNN):
    """
    Scale Equivariant Graph Neural Network for the backward variant.
    """
    def __init__(self, model_args, layer_idx):
        super().__init__(model_args, layer_idx)
        bw_layers = []
        upd_layers = []

        self.concat_mlp_directions = model_args['concat_mlp_directions'] if 'concat_mlp_directions' in model_args else False

        self.pos_embed_upd = model_args['gnn_args']['pos_embed_upd']
        self.update_edge_attr = model_args['gnn_args']['update_edge_attr']
        self.reciprocal = model_args['reciprocal'] and not (model_args['symmetry'] == 'permutation')

        if self.update_edge_attr:
            fw_upd_edge_layers = []
            bw_upd_edge_layers = []

        for i in range(model_args['num_layers']):
            last_layer = i == model_args['num_layers'] - 1
            if i == 0:
                d_in_v = model_args['graph_init']['d_node']
                d_in_e = model_args['graph_init']['d_edge']

            else:
                d_in_v = self.d_hid
                if model_args['gnn_args']['update_edge_attr']:
                    d_in_e = self.d_hid
                else:
                    d_in_e = model_args['graph_init']['d_edge']

            model_args['gnn_args']['update_edge_attr'] = model_args['gnn_args']['update_edge_attr'] and i < model_args['num_layers']

            bw_layers.append(
                self.gnn_layer[self.direction](
                    d_in_v=d_in_v,
                    d_in_e=d_in_e,
                    layer_idx=layer_idx,
                    layer_layout=model_args['layer_layout'],
                    symmetry=model_args['symmetry'],
                    equivariant=model_args['equivariant'],
                    last_layer=last_layer,
                    mlp_args=model_args['mlp_args'],
                    **model_args['gnn_args'])
            )

            if model_args['symmetry'] != 'permutation':
                upd_layers.append(EquivariantNet(
                            model_args['gnn_args']['update_node_feats_fn_layers'],
                            3*self.d_hid,
                            1 if self.equivariant and last_layer else self.d_hid,
                            model_args['mlp_args'],
                            d_extra=self.d_hid if self.pos_embed_upd else 0,
                            symmetry=model_args['symmetry'],
                            equiv_on_hidden=model_args['gnn_args']['upd_equiv_on_hidden'],
                            layer_equiv_on_hidden=model_args['gnn_args']['layer_upd_equiv_on_hidden'],
                            mlp_on_io=model_args['gnn_args']['mlp_on_io'],
                            num_mlps=model_args['gnn_args']['upd_num_mlps'],
                            skip_connections=model_args['gnn_args']['update_node_feats_fn_skip_connections'],
                            sign_symmetrization=model_args['gnn_args']['sign_symmetrization']))
            else:
                upd_layers.append(MLPNet(
                    # model_args['gnn_args']['d_hid_v'] + 2 * model_args['gnn_args']['d_msg'],
                    3 * self.d_hid,
                    1 if self.equivariant and last_layer else self.d_hid,
                    model_args['mlp_args'],
                ))

            if self.update_edge_attr and (i < model_args['num_layers'] - 1 or self.equivariant):
                if model_args['symmetry'] != 'permutation':
                    fw_upd_edge_layers.append(EdgeUpdate(
                        d_in_v,
                        1 if self.equivariant and last_layer else d_in_e,
                        model_args['mlp_args'],
                        model_args['symmetry'],
                        sign_symmetrization=model_args['gnn_args']['sign_symmetrization']))
                    if model_args['symmetry'] != 'sign':
                        bw_upd_edge_layers.append(EdgeUpdate(
                            d_in_v,
                            1 if self.equivariant and last_layer else d_in_e,
                            model_args['mlp_args'],
                            model_args['symmetry'],
                            sign_symmetrization=model_args['gnn_args']['sign_symmetrization']))
                else:
                    fw_upd_edge_layers.append(EdgeUpdate(
                        d_in_v,
                        1 if self.equivariant and last_layer else d_in_e,
                        model_args['mlp_args'],
                        model_args['symmetry'],
                        sign_symmetrization=model_args['gnn_args']['sign_symmetrization']))
                    if model_args['symmetry'] == 'scale':
                        bw_upd_edge_layers.append(EdgeUpdate(
                            d_in_v,
                            1 if self.equivariant and last_layer else d_in_e,
                            model_args['mlp_args'],
                            model_args['symmetry'],
                            sign_symmetrization=model_args['gnn_args']['sign_symmetrization']))

        self.bw_layers = nn.ModuleList(bw_layers)
        self.update_node_feats_fn = nn.ModuleList(upd_layers)
        if self.update_edge_attr:
            self.fw_update_edge_attr_fn = nn.ModuleList(fw_upd_edge_layers)
            if model_args['symmetry'] == 'scale':
                self.bw_update_edge_attr_fn = nn.ModuleList(bw_upd_edge_layers)

        if not self.equivariant:
            if self.only_last_layer:
                if self.concat_mlp_directions:
                    self.project_first = mlp(in_features=self.first_layer_nodes * model_args['gnn_args']['d_hid_v'],
                                             out_features=model_args['gnn_args']['d_hid_v'],
                                             **model_args['mlp_args'])
                    self.project_last = mlp(in_features=self.last_layer_nodes * model_args['gnn_args']['d_hid_v'],
                                            out_features=model_args['gnn_args']['d_hid_v'],
                                            **model_args['mlp_args'])
                else:
                    self.project_first = nn.Linear(self.first_layer_nodes * self.d_hid,
                                                   self.d_hid)
                    self.project_last = nn.Linear(self.last_layer_nodes * self.d_hid,
                                                  self.d_hid)

    def forward(self, batch, num_nodes=None, pos_embed=None, pos_embed_edge=None):
        for i in range(len(self.fw_layers)):
            if not i:
                x = batch.x
                edge_attr = batch.edge_attr
                bw_edge_attr = batch.edge_attr

            fw_aggr = self.fw_layers[i](
                x=x,
                edge_index=batch.edge_index,
                edge_attr=edge_attr,
                mask_hidden=batch.mask_hidden if hasattr(batch, 'mask_hidden') else None,
                mask_first_layer=batch.mask_first_layer if hasattr(batch, 'mask_first_layer') else None,
                pos_embed=pos_embed if pos_embed is not None else None,
                sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None,
            )

            x_feats = x
            bw_aggr = self.bw_layers[i](x=x_feats,
                                        edge_index=batch.bw_edge_index,
                                        edge_attr=bw_edge_attr,
                                        mask_hidden=batch.mask_hidden if hasattr(batch, 'mask_hidden') else None,
                                        mask_first_layer=batch.mask_first_layer if hasattr(batch, 'mask_first_layer') else None,
                                        pos_embed=pos_embed if pos_embed is not None else None,
                                        sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None,
            )

            if self.update_edge_attr and i < len(self.fw_update_edge_attr_fn):
                edge_attr = self.fw_update_edge_attr_fn[i](x, batch.edge_index, edge_attr, sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None)
                if self.symmetry == 'scale':
                    bw_edge_attr = self.bw_update_edge_attr_fn[i](x, batch.bw_edge_index, bw_edge_attr, sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None)

            # update node embeddings
            feats = bw_aggr
            x_tilde = self.update_node_feats_fn[i](
                torch.cat((x, fw_aggr, feats), dim=-1),
                extra_features=pos_embed if self.pos_embed_upd else None,
                mask_hidden=batch.mask_hidden if hasattr(batch, 'mask_hidden') else None,
                mask_first_layer=batch.mask_first_layer if hasattr(batch, 'mask_first_layer') else None,
                sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None,)

            apply_skip = self.gnn_skip_connections and i < len(self.skip_layers)
            x = self.skip_layers[i](x) + x_tilde if apply_skip else x_tilde  # TODO: check if this is correct in every scenario

            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.equivariant:
            return x, edge_attr
        else:
            if not self.only_last_layer:
                graph_features = self.readout(x,
                                              batch.batch,
                                              mask_hidden=batch.mask_hidden if hasattr(batch, 'mask_hidden') else None,
                                              sign_mask=batch.sign_mask[batch.batch].unsqueeze(-1) if hasattr(batch, 'sign_mask') else None
                                              )
            else:
                node_features = x.reshape(batch.num_graphs, num_nodes, x.shape[-1])  # change if processing varying architectures
                last_layer_node_features = self.project_last(node_features[:, -self.last_layer_nodes:].flatten(1, 2))
                first_layer_node_features = self.project_first(node_features[:, :self.first_layer_nodes].flatten(1, 2))
                node_features = torch.cat([last_layer_node_features, first_layer_node_features], dim=1)
                graph_features = self.readout(node_features)

            return graph_features

    def __repr__(self):
        parent_repr = super().__repr__()
        return parent_repr + f'\ndropout between layers={self.dropout}'
