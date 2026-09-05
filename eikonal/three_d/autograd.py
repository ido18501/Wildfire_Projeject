"""First-order implicit differentiation of shared diagonal CPU speeds."""
import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .solver import solve_diagonal, njit


def _adjoint_core(arrival, donors, speeds, spacing, upstream):
    """Reverse causal traversal of the verified converged local equations.

    R=sum_i w_i*(T-T_donor_i)^2-1; D=sum_i w_i*(T-T_donor_i).
    dT/dT_donor_i=w_i*delta_i/D, dT/dc_i=-c_i*delta_i^2/(h_i^2*D).
    No sweeping history or analytical solution is differentiated.
    """
    adjoint = upstream.copy()
    gradient = np.zeros(3)
    weights = (speeds / spacing) ** 2
    order = np.argsort(arrival)
    for k in range(order.size - 1, -1, -1):
        idx = order[k]
        denominator = 0.0
        for axis in range(3):
            donor = donors[idx, axis]
            if donor >= 0:
                denominator += weights[axis] * (arrival[idx] - arrival[donor])
        if denominator == 0.0:  # Fixed source; no dependence on speeds.
            continue
        for axis in range(3):
            donor = donors[idx, axis]
            if donor >= 0:
                delta = arrival[idx] - arrival[donor]
                coefficient = adjoint[idx] / denominator
                adjoint[donor] += coefficient * weights[axis] * delta
                gradient[axis] -= coefficient * speeds[axis] * delta**2 / spacing[axis]**2
    return gradient


_adjoint_compiled = njit(cache=True)(_adjoint_core) if njit is not None else None


class _DiagonalArrival(torch.autograd.Function):
    @staticmethod
    def forward(ctx, speeds_xyz, shape_zyx, source_zyx, spacing_xyz, backend):
        if speeds_xyz.device.type != "cpu" or speeds_xyz.dtype not in (torch.float32, torch.float64):
            raise ValueError("The 3D reference requires CPU float32/float64 speeds")
        if speeds_xyz.shape != (3,):
            raise ValueError("Autograd input speeds_xyz must have shape (3,); expand isotropy first")
        speeds = speeds_xyz.detach().double().numpy().copy()
        result = solve_diagonal(shape_zyx, source_zyx, speeds, spacing_xyz, backend=backend)
        ctx.result = result
        ctx.speeds = speeds
        ctx.spacing = np.asarray(spacing_xyz, dtype=np.float64).copy()
        ctx.input_dtype = speeds_xyz.dtype
        return torch.from_numpy(result.arrival.copy()).to(dtype=speeds_xyz.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        upstream = grad_output.detach().double().contiguous().numpy().ravel()
        if not np.isfinite(upstream).all():
            raise FloatingPointError("Nonfinite loss gradient entering 3D solver")
        result = ctx.result
        adjoint = _adjoint_core if result.backend == "python" else _adjoint_compiled
        gradient = adjoint(result.arrival.ravel(), result.donors, ctx.speeds, ctx.spacing, upstream)
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite 3D speed gradient")
        return torch.from_numpy(gradient).to(dtype=ctx.input_dtype), None, None, None, None


def solve_eikonal(speeds_xyz, shape_zyx, source_zyx, spacing_xyz=(0.8, 0.8, 0.8), backend="auto"):
    """Return a differentiable [Z,Y,X] arrival tensor, one known source at T=0.

    Only first derivatives of three shared speeds are supported. At exact
    opposite-neighbor min ties, use the forward's deterministic branch
    derivative. Sources, spacing and grid geometry are fixed metadata.
    """
    if backend == "cuda":
        from .cuda import solve_eikonal_cuda
        return solve_eikonal_cuda(speeds_xyz, shape_zyx, source_zyx, spacing_xyz)
    return _DiagonalArrival.apply(speeds_xyz, shape_zyx, source_zyx, spacing_xyz, backend)


class SharedSpeedModel(torch.nn.Module):
    """Model A: one log-speed; Model B: three shared axis-aligned log-speeds."""
    def __init__(self, initial_speeds=0.5, isotropic=True):
        super().__init__()
        values = torch.as_tensor(initial_speeds, dtype=torch.float64).reshape(-1)
        if isotropic:
            if values.numel() != 1:
                raise ValueError("Isotropic model requires one initial speed")
        elif values.numel() == 1:
            values = values.repeat(3)
        elif values.numel() != 3:
            raise ValueError("Anisotropic model requires one or three initial speeds")
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Initial speeds must be finite and positive")
        self.log_speeds = torch.nn.Parameter(values.log())
        self.isotropic = bool(isotropic)

    @property
    def speeds_xyz(self):
        speeds = self.log_speeds.exp()
        return speeds.expand(3) if self.isotropic else speeds

    def forward(self, shape_zyx, source_zyx, spacing_xyz=(0.8, 0.8, 0.8), backend="auto"):
        return solve_eikonal(self.speeds_xyz, shape_zyx, source_zyx, spacing_xyz, backend)
