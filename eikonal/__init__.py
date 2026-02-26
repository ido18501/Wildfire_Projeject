"""
Differentiable Randers-Finsler Eikonal Solver
==============================================

A framework for solving and differentiating through Randers-Finsler eikonal
equations, enabling gradient-based learning of spatially-varying propagation
parameters from arrival time observations.

Core components:
    - Forward solvers: PyTorch (reference), Numba CPU, CUDA batched
    - Backward pass: Implicit differentiation via adjoint system
    - Neural network: Covariate encoder and regularization

Reference:
    Gahtan, Shpund, Bronstein. "Wildfire Simulation with Differentiable
    Randers-Finsler Eikonal Solvers." SGP 2025.
"""

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve, create_test_problem
from eikonal.backward.implicit_diff import (
    backward_pass, identify_active_stencils, compute_parameter_gradients,
    solve_adjoint_direct, verify_gradients_finite_difference,
    UpdateType, StencilInfo, NEIGHBOR_OFFSETS, get_stencil_pair,
)
from eikonal.nn.encoder import FlexibleCovariateEncoder, project_G_eigenvalue, project_B_norm
from eikonal.nn.regularization import compute_tv_regularization
from eikonal.nn.config import DEVICE, TVMetricType

__version__ = "1.0.0"
