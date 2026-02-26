"""
Backward pass implementations for implicit differentiation through
the converged eikonal solution.

Implementations:
    implicit_diff    - Core backward pass with adjoint system solve
    hybrid           - Stencil-boundary-aware hybrid (implicit + finite diff)
    adjoint_numba    - Numba JIT-compiled adjoint solve (~3700x speedup)
    adjoint_sparse   - Sparse adjoint for band-decomposed training
    adjoint_batched  - Batched backward pass for multi-fire training
"""

from eikonal.backward.implicit_diff import (
    backward_pass, identify_active_stencils, compute_parameter_gradients,
    solve_adjoint_direct, UpdateType, StencilInfo,
)
