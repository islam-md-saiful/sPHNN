"""
Fully Input Convex Neural Networks (FICNNs), from
Amos, Xu, Kolter (2017) https://arxiv.org/abs/1609.07152

Recurrence (matches the paper / the original dynax implementation):
    z_1     = sigma_0(W_0 x + b_0)
    z_{i+1} = sigma_i(U_i z_i + W_i x + b_i),   i = 1, ..., k-1
    f(x)    = z_k

If every U_i has non-negative entries and every sigma_i is convex and
non-decreasing, f is convex in x (composition rules for convex functions).
We enforce U_i >= 0 via a softplus reparameterization of the raw weight,
which is smooth and keeps gradients well-behaved (equivalent in spirit to
equinox's `NonNegative` constraint used in the original JAX code).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FICNNLayer(nn.Module):
    """One "z" layer of the FICNN: z_out = softplus(W_z) @ z_in + W_y @ x + b."""

    def __init__(self, y_size: int, z_in_size: int, z_out_size: int):
        super().__init__()
        # raw (unconstrained) parameter; softplus(weight_z_raw) >= 0 always
        self.weight_z_raw = nn.Parameter(torch.empty(z_out_size, z_in_size))
        self.weight_y = nn.Parameter(torch.empty(z_out_size, y_size))
        self.bias = nn.Parameter(torch.empty(z_out_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_z_raw)
        nn.init.xavier_uniform_(self.weight_y)
        nn.init.zeros_(self.bias)

    @property
    def weight_z(self) -> torch.Tensor:
        # softplus reparameterization guarantees weight_z >= 0 at all times,
        # which is the structural requirement for input-convexity.
        return F.softplus(self.weight_z_raw)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return z @ self.weight_z.T + y @ self.weight_y.T + self.bias


class FICNN(nn.Module):
    """
    Fully input-convex neural network f: R^in_size -> R^out_size (typically
    out_size = 1, used as the "energy" backbone of the Hamiltonian).

    Batched: forward accepts x of shape (..., in_size) and returns
    (..., out_size) (or (...,) if out_size == 1 and you pass squeeze=True).
    """

    def __init__(
        self,
        in_size: int,
        out_size: int = 1,
        width: int = 16,
        depth: int = 2,
        activation=F.softplus,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        layers = nn.ModuleList()
        if depth == 0:
            layers.append(nn.Linear(in_size, out_size))
        else:
            layers.append(nn.Linear(in_size, width))
            for _ in range(depth - 1):
                layers.append(_FICNNLayer(in_size, width, width))
            layers.append(_FICNNLayer(in_size, width, out_size))
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layers[0](x)
        z = self.activation(z)
        for layer in self.layers[1:-1]:
            z = layer(z, x)
            z = self.activation(z)
        z = self.layers[-1](z, x)
        if self.out_size == 1:
            z = z.squeeze(-1)
        return z
