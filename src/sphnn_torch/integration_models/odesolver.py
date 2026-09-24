"""
Trajectory integration for ISPHS (and any nn.Module with signature
forward(t, x, u) -> xdot), via torchdiffeq -- the torch analogue of the
diffrax-based integrator used in the original JAX codebase.

Supports optional time-dependent inputs u(t) given as samples on a grid,
linearly interpolated at the solver's internal (adaptive) time steps --
mirrors how the paper feeds known input signals (e.g. pump voltage, oven
temperature) through the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint


class _InputInterpolator:
    """Piecewise-linear interpolation of u(t) from samples (ts, us)."""

    def __init__(self, ts: torch.Tensor, us: torch.Tensor):
        self.ts = ts
        self.us = us

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp(self.ts[0], self.ts[-1])
        idx = torch.searchsorted(self.ts, t.detach(), right=True).clamp(1, len(self.ts) - 1)
        t0, t1 = self.ts[idx - 1], self.ts[idx]
        u0, u1 = self.us[idx - 1], self.us[idx]
        w = ((t - t0) / (t1 - t0).clamp_min(1e-12)).unsqueeze(-1)
        return u0 + w * (u1 - u0)


class _WrappedODEFunc(nn.Module):
    """Adapts a `forward(t, x, u)` model to torchdiffeq's `forward(t, x)`."""

    def __init__(self, model: nn.Module, u_fn=None):
        super().__init__()
        self.model = model
        self.u_fn = u_fn

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        u = self.u_fn(t) if self.u_fn is not None else None
        return self.model(t, x, u)


class ODESolver(nn.Module):
    """
    Usage:
        solver = ODESolver(model, method="dopri5", rtol=1e-6, atol=1e-8)
        xs = solver(x0, ts)                      # autonomous / no input
        xs = solver(x0, ts, u_ts=ts_u, u=u_vals)  # with a sampled input u(t)

    `model` must implement forward(t, x, u) -> xdot (e.g. an ISPHS
    instance). x0 has shape (..., n); the returned trajectory has shape
    (len(ts), ..., n), matching torchdiffeq's convention.
    """

    def __init__(self, model: nn.Module, method: str = "dopri5",
                 rtol: float = 1e-6, atol: float = 1e-8):
        super().__init__()
        self.model = model
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x0: torch.Tensor, ts: torch.Tensor,
                u_ts: torch.Tensor | None = None, u: torch.Tensor | None = None) -> torch.Tensor:
        u_fn = _InputInterpolator(u_ts, u) if u is not None else None
        func = _WrappedODEFunc(self.model, u_fn)
        return odeint(func, x0, ts, method=self.method, rtol=self.rtol, atol=self.atol)
