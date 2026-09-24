"""
Training loops for ISPHS (and NODE-style) models. Torch port of the two
fitting strategies discussed in Roth et al. (2025), Sec. 3.3:

  * derivative fitting -- compares model(t, x, u) directly against measured
    (x, xdot) pairs. Cheap, but requires derivative data (or finite-diff
    approximations of it).
  * trajectory fitting -- integrates the model over time (via ODESolver)
    and compares the resulting trajectory against measured x(t). Requires
    backprop through the ODE solve, but is the only option when only
    state (not derivative) measurements are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from .losses import mse
from .integration_models.odesolver import ODESolver


@dataclass
class TrainHistory:
    loss: list = field(default_factory=list)


def train_derivative_fit(
    model: nn.Module,
    x: torch.Tensor,
    xdot_target: torch.Tensor,
    u: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    num_steps: int = 5000,
    lr: float = 1e-3,
    verbose_every: int = 500,
) -> TrainHistory:
    """Fits model(t, x, u) -> xdot directly against (x, xdot) pairs."""
    opt = optim.Adam(model.parameters(), lr=lr)
    t = torch.zeros(x.shape[0]) if t is None else t
    hist = TrainHistory()

    for step in range(num_steps):
        opt.zero_grad()
        xdot_pred = model(t, x, u)
        loss = mse(xdot_pred, xdot_target)
        loss.backward()
        opt.step()
        hist.loss.append(loss.item())
        if verbose_every and step % verbose_every == 0:
            print(f"[derivative fit] step {step:6d}  loss {loss.item():.6e}")
    return hist


def train_trajectory_fit(
    model: nn.Module,
    x0: torch.Tensor,
    ts: torch.Tensor,
    x_target: torch.Tensor,
    u_ts: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    num_steps: int = 2000,
    lr: float = 1e-3,
    method: str = "dopri5",
    verbose_every: int = 100,
) -> TrainHistory:
    """Integrates model over `ts` starting at x0 and fits the resulting
    trajectory against x_target (shape (len(ts), ..., n)) via backprop
    through the ODE solve (adjoint not used here for simplicity; swap in
    torchdiffeq.odeint_adjoint for very long/high-dim trajectories)."""
    solver = ODESolver(model, method=method)
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = TrainHistory()

    for step in range(num_steps):
        opt.zero_grad()
        x_pred = solver(x0, ts, u_ts=u_ts, u=u)
        loss = mse(x_pred, x_target)
        loss.backward()
        opt.step()
        hist.loss.append(loss.item())
        if verbose_every and step % verbose_every == 0:
            print(f"[trajectory fit] step {step:6d}  loss {loss.item():.6e}")
    return hist
