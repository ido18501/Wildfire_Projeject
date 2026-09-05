"""Verified diagonal 3D eikonal reference. Zero drift; no general SPD support."""
from .analytic import analytical_arrival
from .solver import SolveResult, local_update, solve_diagonal as _cpu_solve_diagonal
from .autograd import SharedSpeedModel, solve_eikonal


def solve_diagonal(shape_zyx, source_zyx, speeds_xyz, spacing_xyz=(.8, .8, .8),
                   tol=1e-10, residual_tol=1e-8, max_cycles=64, backend="auto"):
    if backend == "cuda":
        from .cuda import solve_diagonal_cuda
        return solve_diagonal_cuda(shape_zyx, source_zyx, speeds_xyz, spacing_xyz,
                                   tol, residual_tol, max_cycles)
    return _cpu_solve_diagonal(shape_zyx, source_zyx, speeds_xyz, spacing_xyz,
                               tol, residual_tol, max_cycles, backend)

__all__ = ["analytical_arrival", "SolveResult", "local_update", "solve_diagonal",
           "SharedSpeedModel", "solve_eikonal"]
