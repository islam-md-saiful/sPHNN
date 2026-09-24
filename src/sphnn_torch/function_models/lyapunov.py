"""
LyapunovNN: normalizes a convex scalar network (typically a FICNN) into a
valid Lyapunov function / port-Hamiltonian Hamiltonian, satisfying
Theorem 3.1 of Roth et al. (2025):

    H(x) = f(x) - f(x*) - grad_f(x*)^T (x - x*) [+ eps * ||x - x*||^2]

By construction: H(x*) = 0, grad H(x*) = 0, and (if f is strictly convex
near x*, optionally aided by the eps term) the Hessian at x* is positive
definite. Combined with global convexity of f, H is then a valid global
Lyapunov function: positive definite everywhere and radially unbounded.

x* ("minimum") can be a fixed, known equilibrium or a learnable parameter
(sPHNN-LM variant in the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ficnn import FICNN


class LyapunovNN(nn.Module):
    def __init__(
        self,
        ficnn: FICNN,
        minimum: torch.Tensor | None = None,
        state_size: int | None = None,
        minimum_learnable: bool | None = None,
        eps: float = 0.0,
    ):
        """
        Args:
            ficnn: a convex scalar network R^n -> R (e.g. FICNN instance).
            minimum: fixed location x* of the Hamiltonian's minimum. If
                None, x* is initialized at zero and treated as learnable
                (mirrors sPHNN-LM), unless minimum_learnable=False.
            state_size: needed to initialize x*=0 if `minimum` is not given
                and `ficnn` doesn't expose `.in_size`.
            minimum_learnable: overrides the default learnability behaviour.
            eps: strength of the optional quadratic regularizer
                eps * ||x - x*||^2 (Theorem 3.1's f_reg term). Often
                unnecessary in practice, as noted in the paper -- the FICNN
                usually already provides local strict convexity.
        """
        super().__init__()
        self.ficnn = ficnn
        self.eps = eps

        if minimum is None:
            n = state_size if state_size is not None else getattr(ficnn, "in_size", None)
            if n is None:
                raise ValueError("Provide `minimum` or `state_size` (or use a FICNN with `.in_size`).")
            init = torch.zeros(n)
            self.minimum_learnable = True if minimum_learnable is None else minimum_learnable
        else:
            init = minimum.clone().detach()
            self.minimum_learnable = False if minimum_learnable is None else minimum_learnable

        if self.minimum_learnable:
            self.minimum = nn.Parameter(init)
        else:
            self.register_buffer("minimum", init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.minimum
        # f0, grad_f0 at the (possibly learnable) minimum. create_graph=True
        # lets gradients w.r.t. ficnn's/minimum's parameters flow through
        # this second-order term during training.
        x0_ = x0.detach().requires_grad_(True) if not x0.requires_grad else x0
        f0 = self.ficnn(x0_)
        (grad_f0,) = torch.autograd.grad(f0, x0_, create_graph=self.training or x0.requires_grad)

        f = self.ficnn(x)
        delta = x - x0
        f_norm = f - (f0 + torch.sum(delta * grad_f0, dim=-1))

        if self.eps > 0:
            f_norm = f_norm + self.eps * torch.sum(delta * delta, dim=-1)
        return f_norm
