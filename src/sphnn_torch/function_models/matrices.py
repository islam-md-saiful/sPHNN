"""
Neural-network models for matrix-valued functions, used to parameterize
the structure matrix J(x) (skew-symmetric), the dissipation matrix R(x)
(symmetric PSD/PD), and generic matrices (e.g. the input matrix G(x)).

All "learned" variants work by having a small MLP output a compact vector
of the independent matrix entries, then scattering that vector into the
appropriate matrix structure via a fixed (non-learnable) basis tensor.
This guarantees the structural constraint (skew-symmetry / PSD-ness) holds
exactly, for any MLP output -- the constraint lives in the architecture,
not in the loss.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_size: int, out_size: int, width: int = 16, depth: int = 2,
         activation=F.softplus) -> nn.Module:
    layers = []
    sizes = [in_size] + [width] * depth + [out_size]
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(_Act(activation))
    return nn.Sequential(*layers)


class _Act(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ------------------------------------------------------------------ #
# Arbitrary (unconstrained) matrices, e.g. for the input matrix G(x)  #
# ------------------------------------------------------------------ #

class ConstantMatrix(nn.Module):
    """A constant, learnable matrix, ignoring the input (used e.g. for
    state-independent G)."""

    def __init__(self, shape: tuple[int, ...], initialize: Literal["zero", "random"] = "random"):
        super().__init__()
        A = torch.zeros(shape) if initialize == "zero" else torch.empty(shape)
        if initialize == "random":
            nn.init.xavier_uniform_(A) if A.dim() == 2 else nn.init.normal_(A, std=0.1)
        self.A = nn.Parameter(A)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.A


class MatrixFunction(nn.Module):
    """f: R^in_size -> R^(shape), via an MLP whose output is reshaped."""

    def __init__(self, in_size: int, shape: tuple[int, ...] | None = None,
                 width: int = 16, depth: int = 2, activation=F.softplus):
        super().__init__()
        self.shape = shape if shape is not None else (in_size, in_size)
        k = int(np.prod(self.shape))
        self.mlp = _mlp(in_size, k, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(x)
        return a.reshape(*a.shape[:-1], *self.shape)


# ------------------------------------------------------------------ #
# Skew-symmetric matrices, for the structure/Poisson matrix J(x)      #
# ------------------------------------------------------------------ #

def _skew_basis(n: int) -> torch.Tensor:
    """Basis tensor of shape (n, n, k), k = n(n-1)/2, mapping the k
    independent entries of a skew-symmetric n x n matrix to full matrices."""
    tensors = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = np.zeros((n, n), dtype=np.float32)
            s[i, j] = 1.0
            s[j, i] = -1.0
            tensors.append(s)
    return torch.from_numpy(np.stack(tensors, axis=-1))  # (n, n, k)


class SkewSymmetricMatrix(nn.Module):
    """State-dependent skew-symmetric matrix J(x), parameterized by an MLP
    R^in_size -> R^k (k = n(n-1)/2) scattered into R^(n x n) via a fixed
    basis tensor. J = -J^T holds exactly for any MLP output."""

    def __init__(self, in_size: int, matrix_size: int | None = None,
                 width: int = 16, depth: int = 2, activation=F.softplus):
        super().__init__()
        n = matrix_size if matrix_size is not None else in_size
        k = n * (n - 1) // 2
        self.register_buffer("tensor", _skew_basis(n))  # (n, n, k)
        self.mlp = _mlp(in_size, k, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(x)  # (..., k)
        return torch.einsum("ijk,...k->...ij", self.tensor, a)


class ConstantSkewSymmetricMatrix(nn.Module):
    """A constant, learnable skew-symmetric matrix (state-independent J)."""

    def __init__(self, size: int, initialize: Literal["symplectic", "random"] = "random"):
        super().__init__()
        if initialize == "symplectic":
            assert size % 2 == 0, "symplectic initialization needs an even size"
            n_dof = size // 2
            I, O = torch.eye(n_dof), torch.zeros(n_dof, n_dof)
            A = 0.5 * torch.cat([torch.cat([O, I], -1), torch.cat([-I, O], -1)], 0)
        else:
            A = torch.empty(size, size)
            nn.init.xavier_uniform_(A)
        A = A - A.T  # force skew-symmetry of the initial value
        self.A = nn.Parameter(A)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.A - self.A.T  # re-project every call so grad steps can't break skew-symmetry


class SymplecticMatrix(nn.Module):
    """Fixed (non-learnable) canonical symplectic matrix J = [[0, I], [-I, 0]]."""

    def __init__(self, size: int):
        super().__init__()
        assert size % 2 == 0, "size must be even"
        n_dof = size // 2
        I, O = torch.eye(n_dof), torch.zeros(n_dof, n_dof)
        J = torch.cat([torch.cat([O, I], -1), torch.cat([-I, O], -1)], 0)
        self.register_buffer("J", J)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.J


# ------------------------------------------------------------------ #
# Symmetric PSD / PD matrices, for the dissipation matrix R(x)        #
# ------------------------------------------------------------------ #

def _tril_basis(n: int) -> torch.Tensor:
    """Basis tensor of shape (n, n, k), k = n(n+1)/2, mapping the k
    independent entries of a lower-triangular n x n matrix to full matrices."""
    tensors = []
    for i in range(n):
        for j in range(i + 1):
            s = np.zeros((n, n), dtype=np.float32)
            s[i, j] = 1.0
            tensors.append(s)
    return torch.from_numpy(np.stack(tensors, axis=-1))  # (n, n, k)


def _cholesky_like(L: torch.Tensor, diag_fn) -> torch.Tensor:
    """Given a raw lower-triangular matrix (..., n, n), fix the diagonal via
    `diag_fn` to make it non-negative (SPSD, diag_fn=abs) or strictly
    positive (SPD, diag_fn=softplus), then return L @ L^T."""
    n = L.shape[-1]
    eye = torch.eye(n, dtype=L.dtype, device=L.device)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    L_lower = torch.tril(L, diagonal=-1)
    L_fixed = L_lower + torch.diag_embed(diag_fn(diag))
    return L_fixed @ L_fixed.transpose(-1, -2)


class SPSDMatrix(nn.Module):
    """State-dependent symmetric positive semi-definite matrix R(x) = L(x)L(x)^T."""

    def __init__(self, in_size: int, matrix_size: int | None = None,
                 width: int = 16, depth: int = 2, activation=F.softplus):
        super().__init__()
        n = matrix_size if matrix_size is not None else in_size
        k = n * (n + 1) // 2
        self.register_buffer("tensor", _tril_basis(n))
        self.mlp = _mlp(in_size, k, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(x)
        L = torch.einsum("ijk,...k->...ij", self.tensor, a)
        return _cholesky_like(L, torch.abs)  # diag >= 0 -> PSD


class ConstantSPSDMatrix(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        L = torch.empty(size, size)
        nn.init.xavier_uniform_(L)
        self.L = nn.Parameter(L)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        return _cholesky_like(self.L, torch.abs)


class SPDMatrix(nn.Module):
    """State-dependent symmetric positive DEFINITE matrix R(x) = L(x)L(x)^T,
    diagonal passed through softplus for strict positivity (R(x) >> 0)."""

    def __init__(self, in_size: int, matrix_size: int | None = None,
                 width: int = 16, depth: int = 2, activation=F.softplus):
        super().__init__()
        n = matrix_size if matrix_size is not None else in_size
        k = n * (n + 1) // 2
        self.register_buffer("tensor", _tril_basis(n))
        self.mlp = _mlp(in_size, k, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(x)
        L = torch.einsum("ijk,...k->...ij", self.tensor, a)
        return _cholesky_like(L, F.softplus)  # diag > 0 -> strictly PD


class ConstantSPDMatrix(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        L = torch.empty(size, size)
        nn.init.xavier_uniform_(L)
        self.L = nn.Parameter(L)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        return _cholesky_like(self.L, F.softplus)
