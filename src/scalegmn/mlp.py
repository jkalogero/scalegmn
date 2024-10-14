import torch
import torch.nn as nn
from src.scalegmn.inr import Sine


def choose_activation(activation, **kwargs):
    activations = {
        'elu': nn.ELU(),
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(),
        'sine': Sine(kwargs['w0']),
        'identity': lambda x: x
    }

    if activation in activations:
        return activations[activation]
    else:
        raise NotImplementedError


class mlp(nn.Module):
    """
    Generic implementation of an MLP with optional batch normalization, layer normalization, skip connections, and dropout.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 d_k,
                 activation='elu',
                 batch_norm=False,
                 layer_norm=False,
                 bias=True,
                 skip=False,
                 dropout=0.0,
                 final_activation='identity',
                 weight_init=None,
                 weight_init_args=None,
                 w0=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.d_k = d_k
        self.activation_name = activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.skip = skip
        d_in = [in_features]
        d_k = d_k + [out_features]

        self.fc = nn.ModuleList()
        if batch_norm:
            self.bn = nn.ModuleList()
        if layer_norm:
            self.ln = nn.ModuleList()

        for i in range(0, len(d_k)):
            self.fc.append(nn.Linear(d_in[i], d_k[i], bias=bias))
            d_in = d_in + [d_k[i]]

            if self.batch_norm and i != len(d_k) - 1:
                self.bn.append(nn.BatchNorm1d((d_k[i])))
            if self.layer_norm and i != len(d_k) - 1:
                self.ln.append(nn.LayerNorm(d_k[i]))

        self.dropout = nn.Dropout(dropout)
        self.activation = choose_activation(activation, w0=w0)
        self.final_activation = choose_activation(final_activation, w0=w0)

        if weight_init is not None:
            with torch.no_grad():  # with no_grad anyway
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        getattr(torch.nn.init, weight_init)(m.weight, **weight_init_args)
                        if m.bias is not None:
                            m.bias.data.zero_()

    def forward(self, x):
        hidden = []
        for i in range(len(self.fc)):
            z = self.fc[i](x)
            if i == len(self.fc) - 1:
                x = z
            else:
                if self.layer_norm:
                    z = self.ln[i](z)
                if self.batch_norm:
                    z = self.bn[i](z)
                x = self.dropout(self.activation(z))
            if self.skip and i > 0:
                x = x + hidden[-1]
            hidden.append(x)
        x = self.final_activation(x)
        return x
