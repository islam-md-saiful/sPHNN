# sphnn-torch

A **PyTorch** reimplementation of the `dynax` library from
*"Stable Port-Hamiltonian Neural Networks"* (Roth, Klein, Kannapinn, Peters,
Weeger; [arXiv:2502.02480](https://arxiv.org/abs/2502.02480)).

Original JAX/`equinox`/`diffrax` codebase:
https://github.com/CPShub/sphnn-publication

This port mirrors the original module layout so the two codebases map
1:1 onto each other:

| Original (`dynax`, JAX)                            | This port (`sphnn_torch`, PyTorch)                          |
|------------------------------------------------------|-----------------------------------------------------------|
| `dynax/function_models/_FICNN.py`                    | `sphnn_torch/function_models/ficnn.py`                     |
| `dynax/function_models/_LyapunovNN.py`               | `sphnn_torch/function_models/lyapunov.py`                  |
| `dynax/function_models/matrices.py`                  | `sphnn_torch/function_models/matrices.py`                  |
| `dynax/derivative_models/_isphs.py`                  | `sphnn_torch/derivative_models/isphs.py`                   |
| `dynax/integration_models/_odesolver.py` (diffrax)   | `sphnn_torch/integration_models/odesolver.py` (torchdiffeq)|
| `dynax/sphnn_tools.py`                               | `sphnn_torch/sphnn_tools.py`                                |
| `dynax/losses.py`, `dynax/training.py`               | `sphnn_torch/losses.py`, `sphnn_torch/training.py`          |

**Key implementation-level differences from the original (all functionally
equivalent):**
- `equinox.Module` -> plain `torch.nn.Module`; parameters are ordinary
  `nn.Parameter`s instead of PyTree leaves.
- `jax.grad` inside the PHS right-hand side -> `torch.autograd.grad(..., create_graph=True)`,
  computed via `H(x).sum()` since batches are processed independently
  (no batch-mixing layers anywhere in the model).
- FICNN's non-negative weight constraint (`equinox`'s `NonNegative` field
  constraint) -> a `softplus` reparameterization of a raw weight tensor
  (`weight_z = softplus(weight_z_raw)`), smooth and always non-negative.
- `diffrax.diffeqsolve` -> `torchdiffeq.odeint`, wrapped in `ODESolver` with
  the same `(x0, ts, u_ts, u)` calling convention, including
  piecewise-linear interpolation of a sampled input signal `u(t)`.

## Installation

This is a [`uv`](https://docs.astral.sh/uv/) project.

```bash
uv sync
```

or with plain pip, once you have a `requirements.txt` exported (`uv export > requirements.txt`):

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchdiffeq`, `numpy`, `scipy`, `matplotlib`.

## Quick start

```python
import torch
from sphnn_torch import FICNN, LyapunovNN, SkewSymmetricMatrix, SPDMatrix, ISPHS, ODESolver

n = 3  # state dimension

# H(x): convex FICNN, normalized into a valid Lyapunov function/Hamiltonian
H = LyapunovNN(FICNN(in_size=n, width=16, depth=2), state_size=n, minimum_learnable=False)  # x* = 0

# J(x): state-dependent skew-symmetric structure matrix
J = SkewSymmetricMatrix(in_size=n, matrix_size=n)

# R(x): state-dependent, strictly positive-definite dissipation matrix
R = SPDMatrix(in_size=n, matrix_size=n)

model = ISPHS(hamiltonian=H, poisson_matrix=J, resistive_matrix=R, input_matrix=None)

# xdot = [J(x) - R(x)] grad H(x)
x = torch.randn(5, n)  # batch of 5 states
xdot = model(torch.tensor(0.0), x, u=None)

# integrate a trajectory
solver = ODESolver(model, method="dopri5")
ts = torch.linspace(0, 10, 100)
traj = solver(torch.tensor([1.0, -0.5, 0.8]), ts)  # -> converges to 0

# verify Theorem 3.1's hypotheses hold numerically
from sphnn_torch import is_zero_gas_guarantee_valid
assert is_zero_gas_guarantee_valid(model)
```

**Important:** never wrap a forward/integration call in `torch.no_grad()` --
the PHS right-hand side needs autograd internally to compute `grad_x H(x)`
at every evaluation, even at inference time. Just don't call `.backward()`
afterwards, and `.detach()` the result if you need a plain tensor.

## Experiments

- `experiments/smoke_test.py` -- structural checks (J skew-symmetric, R
  positive-definite, 0-GAS guarantee) plus a numerical check that H(t) is
  non-increasing along a trajectory of a randomly-initialized model
  (Theorem 3.1's core physical claim, before any training).
- `experiments/spinning_rigid_body.py` -- full reproduction of Sec. 4.1 /
  Sec. 2.2 of the paper: generates data from Euler's damped rotation
  equations, trains an sPHNN via derivative fitting, and checks the
  learned Hamiltonian tracks the true kinetic energy on a held-out
  trajectory. No external data required.

Run with:
```bash
uv run experiments/smoke_test.py
uv run experiments/spinning_rigid_body.py
```

## Not yet ported

The `cascaded_tanks`, `thermal_food_processing_surrogate`, and
`additive_manufacturing_surrogate` experiments from the original repo
(Secs. 4.2-4.4) depend on external datasets and POD-based dimensionality
reduction utilities that weren't ported here -- only the core `dynax`
library and the self-contained spinning-rigid-body experiment are
included. The library itself (`FICNN`, `LyapunovNN`, matrix
parameterizations, `ISPHS`, `ODESolver`, `training.py`) is complete and
sufficient to rebuild any of them.

## License

Apache-2.0, matching the original repository.
