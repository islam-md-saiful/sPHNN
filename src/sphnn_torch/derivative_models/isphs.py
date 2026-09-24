"""
Learnable Input-State Port-Hamiltonian Systems (ISPHS).

    xdot = [J(x) - R(x)] grad_x H(x) + G(x) u(t)

This is a 1:1 port of dynax/derivative_models/_isphs.py. `hamiltonian`,
`poisson_matrix`, `resistive_matrix`, `input_matrix` are themselves
nn.Modules (e.g. LyapunovNN, SkewSymmetricMatrix, SPDMatrix, MatrixFunction)
supplied by the caller -- ISPHS just wires them together into Eq. (3) of
Roth et al. (2025).

Batching: x has shape (..., n); the gradient of H w.r.t. x is computed via
autograd.grad on H(x).sum() -- valid because H is applied independently to
each batch element (no batch-mixing ops such as BatchNorm are used
anywhere in this codebase).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ISPHS(nn.Module):
    def __init__(
        self,
        hamiltonian: nn.Module,
        poisson_matrix: nn.Module,
        resistive_matrix: nn.Module | None = None,
        input_matrix: nn.Module | None = None,
    ):
        super().__init__()
        self.hamiltonian = hamiltonian
        self.poisson_matrix = poisson_matrix
        self.resistive_matrix = resistive_matrix
        self.input_matrix = input_matrix

    def _grad_H(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True) if not x.requires_grad else x
        H = self.hamiltonian(x)
        (grad_H,) = torch.autograd.grad(H.sum(), x, create_graph=True)
        return grad_H

    def forward(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        S = self.poisson_matrix(x)
        if self.resistive_matrix is not None:
            S = S - self.resistive_matrix(x)

        grad_H = self._grad_H(x)
        x_t = torch.einsum("...ij,...j->...i", S, grad_H)

        if u is not None and self.input_matrix is not None:
            g = self.input_matrix(x)
            x_t = x_t + torch.einsum("...ij,...j->...i", g, u)
        return x_t

    def get_conservative_dynamics(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        S = self.poisson_matrix(x)
        grad_H = self._grad_H(x)
        x_t = torch.einsum("...ij,...j->...i", S, grad_H)
        if u is not None and self.input_matrix is not None:
            g = self.input_matrix(x)
            x_t = x_t + torch.einsum("...ij,...j->...i", g, u)
        return x_t

    def get_dissipative_dynamics(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        if self.resistive_matrix is not None:
            R = self.resistive_matrix(x)
        else:
            n = x.shape[-1]
            R = torch.zeros(*x.shape[:-1], n, n, dtype=x.dtype, device=x.device)
        grad_H = self._grad_H(x)
        x_t = -torch.einsum("...ij,...j->...i", R, grad_H)
        if u is not None and self.input_matrix is not None:
            g = self.input_matrix(x)
            x_t = x_t + torch.einsum("...ij,...j->...i", g, u)
        return x_t
