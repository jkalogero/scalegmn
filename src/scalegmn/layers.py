import torch
import torch.nn as nn
import math
from torch_geometric.nn import DeepSetsAggregation
from torch_geometric.utils import to_dense_batch
from src.scalegmn.mlp import mlp
from src.scalegmn.inr import Sine


class PositionalEncoding(nn.Module):
    """
    Positional Embedding for ScaleGMN
    Options:
        1. sum_on_io: sum on input and output nodes, element-wise product otherwise
        2. sum_on_io_edge: sum on input and output edges, element-wise product otherwise
        3. equiv_net: apply a scale equivariant network.
        4. sum_pos_enc: sum the positional embeddings. *Ruins scale equivariance*
    When considering only the permutation symmetries, sum_pos_enc is set to True.
    """
    def __init__(self,
                 d_in,
                 d_hid,
                 num_param_types,
                 param=None,
                 final_linear_pos_embed=False,
                 sum_pos_enc=False,
                 po_as_different_linear=False,
                 equiv_net=False,
                 sum_on_io=True,
                 num_layers=1,
                 mlp_args=None,
                 symmetry=None,
                 sign_symmetrization=False,
                 mlp_on_io=False,
                 num_mlps=2,
                 equiv_on_hidden=True,
                 layer_equiv_on_hidden=True):

        super().__init__()
        self.param = param
        self.final_linear_pos_embed = final_linear_pos_embed
        self.num_param_types = num_param_types
        self.sum_pos_enc = sum_pos_enc
        self.different_linear = po_as_different_linear
        self.equiv_net = equiv_net
        self.sum_on_io = sum_on_io
        self.sum_on_io_edge = False

        if symmetry == 'permutation':
            self.sum_pos_enc = True
            self.apply_embed = True
            self.equiv_net = False
            self.different_linear = False
            self.final_linear_pos_embed = False

        if self.different_linear:
            self.linears = nn.Parameter(torch.randn(self.num_param_types, d_hid, d_hid if self.equiv_net else d_in))
            nn.init.kaiming_uniform_(self.linears, a=math.sqrt(5))

        self.pos_embed = nn.Parameter(torch.randn(self.num_param_types, d_hid))

        if self.equiv_net:
            self.init_fn = EquivariantNet(
                num_layers,
                d_in,
                d_hid,
                mlp_args,
                d_extra=d_hid,
                symmetry=symmetry,
                equiv_on_hidden=equiv_on_hidden,
                layer_equiv_on_hidden=layer_equiv_on_hidden,
                mlp_on_io=mlp_on_io,
                num_mlps=num_mlps,
                sign_symmetrization=sign_symmetrization,
            )

        if self.final_linear_pos_embed:
            self.final_linear = nn.Linear(d_hid, d_hid, bias=False)

    def forward(self, data):
        if self.equiv_net:
            data, pos_embed = self.apply_eq(data)
        else:
            data, pos_embed = self.apply_pos_embed(data)
        if self.different_linear:
            data, pos_embed = self.apply_diff_layer(data)

        return data, pos_embed

    def apply_diff_layer(self, data):
        if self.param == 'node':
            data.x = torch.bmm(data.x.unsqueeze(1), self.linears[data.node2type,:,:].transpose(-2, -1)).squeeze(1)
            pos_embed = self.pos_embed[data.node2type]
        else:
            data.edge_attr = torch.bmm(data.edge_attr.unsqueeze(1), self.linears[data.edge2type, :, :].transpose(-2, -1)).squeeze(1)
            pos_embed = self.pos_embed[data.edge2type]
        return data, pos_embed

    def transform(self, x, node2type, op='prod'):
        if op == 'prod':
            x = x * self.pos_embed[node2type]
        elif op == 'sum':
            x = x + self.pos_embed[node2type]
        return x

    def apply_pos_embed(self, data):
        if self.param == 'node':
            if self.sum_pos_enc:
                data.x = data.x + self.pos_embed[data.node2type]
            else:
                if self.sum_on_io:
                    data.x = data.mask_hidden * self.transform(data.x, data.node2type, 'prod') + (~data.mask_hidden) * self.transform(data.x, data.node2type, 'sum')
                else:
                    data.x = data.x * self.pos_embed[data.node2type]

            if self.final_linear_pos_embed:
                data.x = self.final_linear(data.x)

            return data, self.pos_embed[data.node2type]

        elif self.param == 'edge':
            if self.sum_pos_enc:
                data.edge_attr = data.edge_attr + self.pos_embed[data.edge2type]
            else:
                if self.sum_on_io and self.sum_on_io_edge:

                    edge_mask = (data.mask_hidden[data.edge_index[0]] * ~data.mask_hidden[data.edge_index[1]]) + (data.mask_hidden[data.edge_index[1]] * ~data.mask_hidden[data.edge_index[0]])

                    data.edge_attr = (edge_mask * self.transform(data.edge_attr, data.edge2type, 'prod') +
                                      (~edge_mask) * self.transform(data.edge_attr, data.edge2type, 'sum'))
                else:
                    data.edge_attr = data.edge_attr * self.pos_embed[data.edge2type]
            if self.final_linear_pos_embed:
                data.edge_attr = self.final_linear(data.edge_attr)

            return data, self.pos_embed[data.edge2type]

    def apply_eq(self, data):
        if self.param == 'node':
            if self.sum_on_io:
                data.x = (data.mask_hidden *  self.init_fn(
                                                data.x,
                                                extra_features=self.pos_embed[data.node2type],
                                                mask_hidden=data.mask_hidden if hasattr(data, 'mask_hidden') else None,
                                                mask_first_layer=data.mask_first_layer if hasattr(data, 'mask_first_layer') else None)
                          + (~data.mask_hidden) * self.transform(data.x, data.node2type, 'sum'))
            else:
                data.x = self.init_fn(
                    data.x,
                    extra_features=self.pos_embed[data.node2type],
                    mask_hidden=data.mask_hidden if hasattr(data, 'mask_hidden') else None,
                    mask_first_layer=data.mask_first_layer if hasattr(data, 'mask_first_layer') else None)
            return data, self.pos_embed[data.node2type]
        else:
            data.edge_attr = self.init_fn(
                data.edge_attr,
                extra_features=self.pos_embed[data.edge2type],
                mask_hidden=data.mask_hidden[data.edge_index[1]] if hasattr(data, 'mask_hidden') else None,
                mask_first_layer=data.mask_first_layer[data.edge_index[1]] if hasattr(data, 'mask_first_layer') else None)
            return data, self.pos_embed[data.edge2type]

    def __repr__(self):

        return (f'PositionalEmbedding(\n'
                f' positional_embeddings={self.pos_embed.shape},\n'
                f' sum_pos_enc={self.sum_pos_enc},\n'
                f' equiv_net={self.equiv_net},\n'
                f' sum_on_io={self.sum_on_io},\n'
                f' sum_on_io_edge={self.sum_on_io_edge},\n'
                f' po_as_different_linear={self.different_linear},\n'
                f' Linears: {self.linears.shape if self.different_linear else str(False)},\n'
                f' final_linear_pos_embed={self.final_linear if self.final_linear_pos_embed else str(False)},\n'
                f' init_fn={self.init_fn if self.equiv_net else str(False)},\n'
                ')')


class InvariantLayer(nn.Module):
    """
    Layer to apply canonicalization or symmetrization to the input data.
    - sign symmetry: we apply symmetrization f(x) = rho(x) + rho(-x) or canonicalization f(x) = |x|
    - scale symmetry: we apply canonicalization f(x) = x / (||x|| + eps)
    If extra_features are provided, we concatenate them to the canonicalized/symmetrized data, as in AugScaleInv (A.1.2. Supplementary)
    """
    def __init__(self, in_features, out_features, mlp_args, sym, sign_symmetrization=True, d_extra=0, final_mlp=True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.symmetry = sym
        self.sign_symmetrization = sign_symmetrization
        self.final_mlp = final_mlp
        if final_mlp:
            self.phi = mlp(in_features=in_features+d_extra, out_features=out_features, **mlp_args)
        if sym == 'sign' and sign_symmetrization:
            self.rho = mlp(in_features=in_features, out_features=in_features, **mlp_args)

    def forward(self, x: torch.Tensor, extra_features=None):

        if self.symmetry == 'sign':
            if self.sign_symmetrization:
                r = self.rho(x) + self.rho(-x)
            else:
                r = torch.abs(x)
        elif self.symmetry == 'scale':
            eps = 1e-10
            norms = torch.norm(x, p=2, dim=-1, keepdim=True)
            r = x / (norms + eps)
        else:
            raise NotImplementedError
        if extra_features is not None:
            r = torch.cat((r, extra_features), dim=-1)
        return self.phi(r) if self.final_mlp else r


class InvariantNet(nn.Module):
    """
    The Invariant module of a Scale Equivariant Layer.
    """
    def __init__(self, d_in, d_out, mlp_args, sym, d_extra=0, equiv_on_hidden=False, num_mlps=2, sign_symmetrization=True):
        """
        Args:
            d_in: input dimension
            d_out: output dimension
            mlp_args: arguments for the MLP
            sym: symmetry
            d_extra: extra features dimension (if using AugScaleInv)
            equiv_on_hidden: apply just an MLP to the I/O nodes if true
            num_mlps: use different MLPs for the input and the output nodes
            sign_symmetrization: apply canonicalization or symmetrization to the input data
        """
        super().__init__()

        self.symmetry = sym
        self.d_extra = d_extra
        self.equiv_on_hidden = equiv_on_hidden
        self.num_mlps = num_mlps
        self.sign_symmetrization = sign_symmetrization
        if self.equiv_on_hidden:
            if num_mlps in [2, 3]:
                self.phi_in = mlp(in_features=d_in+d_extra, out_features=d_out, **mlp_args)
            if num_mlps == 3:
                self.phi_last = mlp(in_features=d_in+d_extra, out_features=d_out, **mlp_args)

        if self.symmetry != 'hetero':
            self.g1 = InvariantLayer(d_in, d_out, mlp_args, sym, sign_symmetrization, d_extra=d_extra)
        else:
            self.g_sign = InvariantLayer(d_in, d_out, mlp_args, 'sign', sign_symmetrization, d_extra=d_extra)
            self.g_scale = InvariantLayer(d_in, d_out, mlp_args, 'scale', d_extra=d_extra)

    def forward(self, x: torch.Tensor, extra_features=None, mask_hidden=None, mask_first_layer=None, sign_mask=None):
        if self.symmetry == 'hetero':
            assert sign_mask is not None
            x_canon = sign_mask * self.g_sign(x) + (~sign_mask) * self.g_scale(x)
        else:
            x_canon = self.g1(x, extra_features)
        x_p = torch.cat((x, extra_features), dim=-1) if extra_features is not None else x
        if self.equiv_on_hidden:
            if self.num_mlps == 1:
                x = mask_hidden * x_canon + (~mask_hidden) * self.phi(x_p)
            elif self.num_mlps == 2:
                x = mask_hidden * x_canon + (~mask_hidden) * self.phi_in(x_p)
            elif self.num_mlps == 3:
                x = mask_hidden * x_canon + mask_first_layer * self.phi_in(x_p) + ((~mask_hidden) & (~mask_first_layer)) * self.phi_last(x_p)
        else:
            x = x_canon
        return x

    def __repr__(self):
        parent_repr = super().__repr__()

        return parent_repr + f'\nsymmetry={self.symmetry}' + f'\nsign_symmetrization={self.sign_symmetrization}' + f'\nequiv_on_hidden={self.equiv_on_hidden}'


class MLPNet(nn.Module):
    """
    Class for a simple MLP.
    """
    def __init__(self, d_in, d_out, mlp_args, pos_embed_dim=0,  **kwargs):
        super().__init__()
        self.pos_embed_dim = pos_embed_dim
        self.rho = mlp(in_features=d_in+pos_embed_dim, out_features=d_out, **mlp_args)

    def forward(self, x: torch.Tensor, extra_features=None, *args, **kwargs):
        r = self.rho(x) if not self.pos_embed_dim else self.rho(torch.cat((x, extra_features), dim=1))
        return r


class PermScaleInvariantReadout(nn.Module):
    """
    Permutation and Scale Invariant Readout.
    """
    def __init__(self, d_in, d_rho, d_out, mlp_args, sym, num_io_nodes, sign_symmetrization=True):
        """
        Args:
            d_in: input dimension
            d_rho: intermediate dimension
            d_out: output dimension
            mlp_args: arguments for the MLP
            sym: symmetry
            num_io_nodes: number of I/O nodes - must be computed for every datapoint if processing varying architectures
            sign_symmetrization: apply canonicalization or symmetrization to the input data
        """
        super().__init__()
        self.symmetry = sym
        self.sign_symmetrization = sign_symmetrization

        self.phi = mlp(in_features=d_in, out_features=d_rho, **mlp_args)

        if self.symmetry != 'hetero':
            self.g1 = InvariantLayer(d_in, d_rho, mlp_args, sym, sign_symmetrization)
        else:
            self.g_sign = InvariantLayer(d_in, d_rho, mlp_args, 'sign', sign_symmetrization, final_mlp=True)
            self.g_scale = InvariantLayer(d_in, d_rho, mlp_args, 'scale', final_mlp=True)

        self.aggr = DeepSetsAggregation(None, None)
        self.global_mlp = mlp(in_features=d_rho+d_rho*num_io_nodes, out_features=d_out, **mlp_args)

    def forward(self, x, batch, mask_hidden=None, sign_mask=None):
        if self.symmetry == 'hetero':
            x_canon = sign_mask * self.g_sign(x) + (~sign_mask) * self.g_scale(x)
        else:
            x_canon = self.g1(x)
        sum_hidden = self.aggr(mask_hidden * x_canon, index=batch)
        i_o_nodes = to_dense_batch(x[~mask_hidden.squeeze(1)], batch[~mask_hidden.squeeze(1)])[0]
        i_o_nodes = i_o_nodes.reshape(i_o_nodes.shape[0], -1)
        graph_emb = torch.cat((sum_hidden, i_o_nodes), dim=-1)
        graph_emb = self.global_mlp(graph_emb)
        return graph_emb

    def __repr__(self):
        global_str = f'\nglobal_mlp={self.global_mlp}'
        return f'\nDeepsets={self.aggr}' + global_str + f'\nsymmetry={self.symmetry}' + f'\nsign_symmetrization={self.sign_symmetrization}'


class DeepSet(nn.Module):
    """
    Readout when accounting only for permutation symmetries.
    Args:
        d_in: input dimension
        d_rho: intermediate dimension
        d_out: output dimension
        mlp_args: arguments for the MLPs
    """
    def __init__(self, d_in, d_rho, d_out, mlp_args, **kwargs):
        super().__init__()
        self.local_mlp = mlp(in_features=d_in, out_features=d_rho, **mlp_args)
        self.global_mlp = mlp(in_features=d_rho, out_features=d_out, **mlp_args)
        self.aggr = DeepSetsAggregation(self.local_mlp, self.global_mlp)

    def forward(self, x, batch, **kwargs):
        r = self.aggr(x, index=batch)
        return r

    def __repr__(self):
        return f'\nDeepsets={self.aggr}'


class EquivariantNet(nn.Module):
    """
    Scale Equivariant module.
    """
    def __init__(self,
                 n_layers,
                 d_in,
                 d_out,
                 mlp_args,
                 d_extra=0,
                 symmetry='sign',
                 equiv_on_hidden=False,
                 layer_equiv_on_hidden=False,
                 mlp_on_io=False,
                 num_mlps=2,
                 sign_symmetrization=False,
                 skip_connections=False):
        """
        Args
            n_layers: number of scale equivariant layers
            d_in: input dimension
            d_out: output dimension
            mlp_args: arguments for the MLPs
            d_extra: extra features dimension (if using AugScaleInv)
            symmetry: symmetry
            equiv_on_hidden: apply just an MLP to the I/O nodes if true
            layer_equiv_on_hidden: apply equivariant layer *only* to the hidden nodes if true
            mlp_on_io: apply an MLP to the I/O nodes if true
            num_mlps: use different MLPs for the input and the output nodes
            sign_symmetrization: apply canonicalization or symmetrization to the input data
            skip_connections: use skip connections between scale equivariant layers
        """
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.skip_connections = skip_connections
        self.equiv_on_hidden = equiv_on_hidden
        self.layer_equiv_on_hidden = layer_equiv_on_hidden
        self.num_mlps = num_mlps
        self.mlp_on_io = mlp_on_io
        self.d_extra = d_extra

        layers = []
        for i in range(n_layers):
            layers.append(EquivariantLayer(d_out if i else d_in,
                                           d_out,
                                           mlp_args,
                                           d_extra,
                                           symmetry=symmetry,
                                           equiv_on_hidden=equiv_on_hidden and not (layer_equiv_on_hidden or mlp_on_io),
                                           num_mlps=num_mlps,
                                           sign_symmetrization=sign_symmetrization))

        self.eq_layers = nn.ModuleList(layers)

        self.eq_layers_in, self.eq_layers_last = None, None
        if self.mlp_on_io:
            # instead of using List of Equivariant layers, just use an MLP
            self.eq_layers_in = nn.ModuleList([MLPNet(d_out if i else d_in, d_out, mlp_args, d_extra) for i in range(n_layers)])
            if num_mlps == 3:
                self.eq_layers_last = nn.ModuleList([MLPNet(d_out if i else d_in, d_out, mlp_args, d_extra) for i in range(n_layers)])

        elif self.layer_equiv_on_hidden:
            if num_mlps in [2,3]:
                self.eq_layers_in = nn.ModuleList(
                    [EquivariantLayer(d_out if i else d_in,
                                      d_out,
                                      mlp_args,
                                      d_extra,
                                      symmetry=symmetry,
                                      equiv_on_hidden=False,
                                      num_mlps=num_mlps,
                                      sign_symmetrization=sign_symmetrization) for i in range(n_layers)
                     ]
                )
            if num_mlps == 3:
                self.eq_layers_last = nn.ModuleList(
                    [EquivariantLayer(d_out if i else d_in,
                                      d_out,
                                      mlp_args,
                                      d_extra,
                                      symmetry=symmetry,
                                      equiv_on_hidden=False,
                                      num_mlps=num_mlps,
                                      sign_symmetrization=sign_symmetrization) for i in range(n_layers)
                     ]
                )

        if self.skip_connections:
            self.skip_layers = nn.ModuleList(
                [nn.Linear(d_out if i else d_in, d_out, bias=False) for i in range(n_layers)])

    def forward(self, x: torch.Tensor, extra_features=None, mask_hidden=None, mask_first_layer=None, sign_mask=None):
        for i, layer in enumerate(self.eq_layers):
            x_tilde_hid = layer(x, extra_features, mask_hidden, mask_first_layer, sign_mask=sign_mask)
            if self.layer_equiv_on_hidden or self.mlp_on_io:
                if self.num_mlps == 1:
                    x_tilde = mask_hidden * x_tilde_hid + (~mask_hidden) * layer(x, extra_features=extra_features)
                elif self.num_mlps == 2:
                    x_tilde = mask_hidden * x_tilde_hid + (~mask_hidden) * self.eq_layers_in[i](x, extra_features=extra_features)
                elif self.num_mlps == 3:
                    x_tilde = mask_hidden * x_tilde_hid + mask_first_layer * self.eq_layers_in[i](x, extra_features=extra_features) + \
                            ((~mask_hidden) & (~mask_first_layer)) * self.eq_layers_last[i](x, extra_features=extra_features)
            else:
                x_tilde = x_tilde_hid
            x = self.skip_layers[i](x) + x_tilde if self.skip_connections else x_tilde

        return x

    def __repr__(self):
        # Get the parent representation and remove '\n)' from the end
        parent_repr = super().__repr__()[:-2]
        layer_equiv_on_hidden_str = f',\n  layer_equiv_on_hidden={self.layer_equiv_on_hidden}'
        num_mlps_str = f',\n  num_layer_types={self.num_mlps}' if self.layer_equiv_on_hidden else ''
        mlp_on_io_str = f',\n  mlp_on_io={self.mlp_on_io}'
        d_extra_str = f'\n  d_extra={self.d_extra}'
        return parent_repr + mlp_on_io_str + layer_equiv_on_hidden_str + num_mlps_str + d_extra_str + '\n)'


class EquivariantLayer(nn.Module):
    """
    Scale Equivariant Layer.
    """
    def __init__(self, d_in, d_out, mlp_args, d_extra=0, symmetry='sign', equiv_on_hidden=False, num_mlps=2, sign_symmetrization=False, hetero_sym=False):
        """
        Args:
            d_in: input dimension
            d_out: output dimension
            mlp_args: arguments for the MLP
            d_extra: extra features dimension (if using AugScaleInv)
            symmetry: symmetry
            equiv_on_hidden: apply just an MLP to the I/O nodes if true
            num_mlps: use different MLPs for the input and the output nodes
            sign_symmetrization: apply canonicalization or symmetrization to the input data
            hetero_sym: use heterogeneous symmetries
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_extra = d_extra
        self.hetero_sym = hetero_sym
        self.W = nn.Linear(d_in, d_out, bias=False)

        inv_net = InvariantNet if symmetry != 'permutation' else MLPNet

        self.g = inv_net(d_in,
                         d_out,
                         mlp_args,
                         symmetry,
                         d_extra,
                         equiv_on_hidden=equiv_on_hidden,
                         num_mlps=num_mlps,
                         sign_symmetrization=sign_symmetrization)

    def forward(self, x: torch.Tensor, extra_features=None, mask_hidden=None, mask_first_layer=None, sign_mask=None):
        linearly_transformed = self.W(x)
        r = linearly_transformed * self.g(x, extra_features, mask_hidden, mask_first_layer, sign_mask=sign_mask)
        return r


class EdgeUpdate(nn.Module):
    """
    Scale Equivariant Edge Update function.
    """
    def __init__(self, d_in, d_out, mlp_args, symmetry, sign_symmetrization):
        """
        Args:
            d_in: input dimension
            d_out: output dimension
            mlp_args: arguments for the MLP
            symmetry: symmetry
            sign_symmetrization: apply canonicalization or symmetrization to the input data
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.symmetry = symmetry
        self.sign_symmetrization = sign_symmetrization
        self.phi = mlp(in_features=(2+(symmetry=='permutation'))*d_in, out_features=d_out, **mlp_args)
        if symmetry != 'permutation':
            self.W = nn.Linear(d_in, d_out, bias=False)

        if self.symmetry != 'hetero':
            self.g1 = InvariantLayer(d_in, d_out, mlp_args, symmetry, sign_symmetrization, final_mlp=False)
        else:
            self.g_sign = InvariantLayer(d_in, d_out, mlp_args, 'sign', sign_symmetrization, final_mlp=False)
            self.g_scale = InvariantLayer(d_in, d_out, mlp_args, 'scale', final_mlp=False)

    def forward(self, x, edge_index, edge_attr, sign_mask=None):
        if self.symmetry == 'permutation':
            h_e = self.phi(torch.cat((edge_attr, x[edge_index[0]], x[edge_index[1]]), dim=-1))
        else:
            if self.symmetry == 'hetero':
                x_canon_0 = sign_mask[edge_index[0]] * self.g_sign(x[edge_index[0]]) + (~sign_mask[edge_index[0]]) * self.g_scale(x[edge_index[0]])
                x_canon_1 = sign_mask[edge_index[1]] * self.g_sign(x[edge_index[1]]) + (~sign_mask[edge_index[1]]) * self.g_scale(x[edge_index[1]])

            else:
                x_canon_0 = self.g1(x[edge_index[0]])
                x_canon_1 = self.g1(x[edge_index[1]])
            h_e = self.W(edge_attr) * self.phi(torch.cat((x_canon_0, x_canon_1), dim=-1))
        return h_e

    def __repr__(self):
        parent_repr = super().__repr__()

        return parent_repr + f'\nsymmetry={self.symmetry}' + f'\nsign_symmetrization={self.sign_symmetrization}'


class SineUpdate(nn.Module):
    """
    Module that applies a linear layer W to the input and then calls Sine layer
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.sine = Sine(w0=30.0)

    def forward(self, x):
        return self.sine(self.W(x))
