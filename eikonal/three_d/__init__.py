"""Verified diagonal 3D eikonal reference. Zero drift; no general SPD support."""
from .analytic import analytical_arrival
from .solver import SolveResult, local_update, solve_diagonal
from .autograd import SharedSpeedModel, solve_eikonal

__all__ = ["analytical_arrival", "SolveResult", "local_update", "solve_diagonal",
           "SharedSpeedModel", "solve_eikonal"]
