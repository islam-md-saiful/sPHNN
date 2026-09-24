"""
Post-hoc verification that a trained ISPHS satisfies Theorem 3.1's
hypotheses, i.e. is 0-GAS (globally asymptotically stable at the origin /
at the Hamiltonian's minimum). Torch port of dynax/sphnn_tools.py.

This does NOT re-derive the guarantee -- it numerically checks that the
Hessian of H at its minimum is positive definite and that R is positive
definite, which are exactly the two extra facts (beyond H's convexity,
which is guaranteed architecturally by the FICNN) needed to invoke
Theorem 3.1.
"""

from __future__ import annotations

import torch

from .function_models.lyapunov import LyapunovNN
from .function_models.ficnn import FICNN
from .function_models.matrices import (
    SkewSymmetricMatrix, ConstantSkewSymmetricMatrix, SymplecticMatrix,
    SPSDMatrix, ConstantSPSDMatrix, SPDMatrix, ConstantSPDMatrix,
)
from .derivative_models.isphs import ISPHS

_VALID_STRUCTURE_MATRICES = (SkewSymmetricMatrix, ConstantSkewSymmetricMatrix, SymplecticMatrix)
_VALID_SPSD_MATRICES = (SPSDMatrix, ConstantSPSDMatrix)
_VALID_SPD_MATRICES = (SPDMatrix, ConstantSPDMatrix)


def is_positive_definite(matrix: torch.Tensor, epsilon: float = 1e-6) -> bool:
    eigvals = torch.linalg.eigvals(matrix).real
    return bool(torch.all(eigvals > epsilon))


def _check_valid_isphs(isphs: ISPHS):
    assert isinstance(isphs, ISPHS), "model must be an ISPHS instance"
    assert isinstance(isphs.hamiltonian, LyapunovNN), "Hamiltonian is not a LyapunovNN"
    assert isinstance(isphs.hamiltonian.ficnn, FICNN), "LyapunovNN's inner network is not a FICNN"
    assert isinstance(isphs.poisson_matrix, _VALID_STRUCTURE_MATRICES), "J is not of a known structure-preserving type"
    assert isinstance(isphs.resistive_matrix, _VALID_SPD_MATRICES + _VALID_SPSD_MATRICES), "R is not of a known PSD/PD type"


def is_zero_gas_guarantee_valid(isphs: ISPHS, epsilon: float = 1e-6) -> bool:
    """Checks Theorem 3.1's remaining hypotheses numerically:
    (1) Hessian of H at the minimum is positive definite,
    (2) R is (globally, if state-dependent) positive definite.
    Convexity of H itself is guaranteed architecturally by the FICNN."""
    _check_valid_isphs(isphs)

    x_star = isphs.hamiltonian.minimum.detach().clone().requires_grad_(True)
    H_hessian = torch.autograd.functional.hessian(lambda x: isphs.hamiltonian(x), x_star)
    hessian_pd = is_positive_definite(H_hessian, epsilon)

    R_globally_pd = False
    if isinstance(isphs.resistive_matrix, _VALID_SPD_MATRICES):
        # SPDMatrix / ConstantSPDMatrix guarantee R >> 0 everywhere by
        # construction (softplus'd Cholesky diagonal).
        R_globally_pd = True

    return hessian_pd and R_globally_pd


def get_eigenvals(isphs: ISPHS):
    _check_valid_isphs(isphs)
    x_star = isphs.hamiltonian.minimum.detach().clone().requires_grad_(True)
    H_hessian = torch.autograd.functional.hessian(lambda x: isphs.hamiltonian(x), x_star)
    H_eigvals = torch.linalg.eigvals(H_hessian).real

    R_eigvals = None
    if isinstance(isphs.resistive_matrix, (ConstantSPDMatrix, ConstantSPSDMatrix)):
        R = isphs.resistive_matrix(None)
        R_eigvals = torch.linalg.eigvals(R).real

    return H_eigvals, R_eigvals
