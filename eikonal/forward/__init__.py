"""
Forward solvers for the Randers-Finsler eikonal equation.

Implementations:
    pytorch     - Reference PyTorch implementation (CPU, differentiable-friendly)
    numba_cpu   - Numba JIT-compiled solver (~83x speedup over PyTorch)
    cuda_batched - Batched CUDA solver for parallel multi-fire solving
"""

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve, create_test_problem
