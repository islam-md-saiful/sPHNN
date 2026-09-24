"""
sphnn_torch
===========
A PyTorch reimplementation of the `dynax` library used in
"Stable Port-Hamiltonian Neural Networks" (Roth, Klein, Kannapinn, Peters,
Weeger; arXiv:2502.02480). Original JAX/equinox/diffrax code:
https://github.com/CPShub/sphnn-publication

This port mirrors the original module layout 1:1 so the mapping between
the two codebases stays obvious:

    dynax/function_models/_FICNN.py       -> sphnn_torch/function_models/ficnn.py
    dynax/function_models/_LyapunovNN.py  -> sphnn_torch/function_models/lyapunov.py
    dynax/function_models/matrices.py     -> sphnn_torch/function_models/matrices.py
    dynax/derivative_models/_isphs.py     -> sphnn_torch/derivative_models/isphs.py
    dynax/integration_models/_odesolver.py-> sphnn_torch/integration_models/odesolver.py
    dynax/sphnn_tools.py                  -> sphnn_torch/sphnn_tools.py
"""

from .function_models.ficnn import FICNN
from .function_models.lyapunov import LyapunovNN
from .function_models.matrices import (
    ConstantMatrix,
    MatrixFunction,
    SkewSymmetricMatrix,
    ConstantSkewSymmetricMatrix,
    SymplecticMatrix,
    SPSDMatrix,
    ConstantSPSDMatrix,
    SPDMatrix,
    ConstantSPDMatrix,
)
from .derivative_models.isphs import ISPHS
from .integration_models.odesolver import ODESolver
from .sphnn_tools import is_zero_gas_guarantee_valid, get_eigenvals

__all__ = [
    "FICNN",
    "LyapunovNN",
    "ConstantMatrix",
    "MatrixFunction",
    "SkewSymmetricMatrix",
    "ConstantSkewSymmetricMatrix",
    "SymplecticMatrix",
    "SPSDMatrix",
    "ConstantSPSDMatrix",
    "SPDMatrix",
    "ConstantSPDMatrix",
    "ISPHS",
    "ODESolver",
    "is_zero_gas_guarantee_valid",
    "get_eigenvals",
]
