"""PyTorch CUDA, shared homogeneous diagonal speeds only; structured 3D, b=0.

For this restricted problem, arrival is monotone away from the source along
each axis. Inward donors therefore form Manhattan-distance wavefronts. Each
front applies the reference Godunov equation, without an iterative history.
Backward propagates three parameter sensitivities over the same causal DAG,
then contracts with the upstream gradient. No CPU numerical solve is used.
"""
from time import perf_counter

import numpy as np
import torch
from torch.autograd.function import once_differentiable


def _geometry(shape, source, device):
    shape, source = np.asarray(shape), np.asarray(source)
    if shape.shape != (3,) or not np.issubdtype(shape.dtype, np.integer) or (shape < 1).any():
        raise ValueError("Expected three positive integer grid dimensions")
    if (source.shape != (3,) or not np.issubdtype(source.dtype, np.integer)
            or (source < 0).any() or (source >= shape).any()):
        raise ValueError("Expected three in-bounds integer source indices")
    # Geometry only is prepared on CPU; arrival and derivative arithmetic is
    # entirely on the selected tensor device. No labels/analytical times seed it.
    coordinates = np.indices(tuple(shape)).reshape(3, -1).T
    offsets = coordinates - source
    levels = np.abs(offsets).sum(axis=1)
    flat = np.arange(len(coordinates))
    donors = np.full((len(flat), 3), -1, dtype=np.int64)
    for axis, stride in enumerate((1, int(shape[2]), int(shape[1] * shape[2]))):
        offset = offsets[:, 2 - axis]
        mask = offset != 0
        donors[mask, axis] = flat[mask] - np.sign(offset[mask]).astype(np.int64) * stride
    fronts = [torch.as_tensor(np.flatnonzero(levels == level), device=device)
              for level in range(1, int(levels.max()) + 1)]
    source_flat = int(np.ravel_multi_index(tuple(source), tuple(shape)))
    return tuple(int(n) for n in shape), source_flat, torch.as_tensor(donors, device=device), fronts


def _local(times, weights):
    """Vectorized reference rationalized positive-root/prefix update."""
    ordered, permutation = times.sort(dim=1, stable=True)
    w = weights.expand_as(times).gather(1, permutation)
    root = ordered[:, 0] + w[:, 0].rsqrt()
    root = torch.where(root > ordered[:, 0], root, float("nan"))
    for count in (2, 3):
        needed = root > ordered[:, count - 1]
        q = torch.where(needed, ordered[:, count - 1], 0.)
        delta = torch.where(needed[:, None], q[:, None] - ordered[:, :count], 0.)
        a = w[:, :count].sum(dim=1)
        b = (w[:, :count] * delta).sum(dim=1)
        remaining = 1. - (w[:, :count] * delta.square()).sum(dim=1)
        increment = remaining / ((b.square() + a * remaining).clamp_min(0.).sqrt() + b)
        candidate = q + increment
        # Match the reference's rejection of noncausal/unrepresentable roots.
        candidate = torch.where((remaining > 0) & (candidate > q), candidate, float("nan"))
        root = torch.where(needed, candidate, root)
    return root


@torch.no_grad()
def _forward(speeds, shape, source, spacing, residual_tol=1e-8):
    # CPU tensors are accepted privately for hardware-independent equation
    # tests. Public CUDA entry points never fall back to CPU.
    if speeds.shape != (3,) or speeds.dtype != torch.float64:
        raise ValueError("Internal solve expects three float64 speeds")
    spacing = torch.as_tensor(spacing, dtype=torch.float64, device=speeds.device)
    if (spacing.shape != (3,) or not torch.isfinite(spacing).all() or (spacing <= 0).any()
            or not torch.isfinite(speeds).all() or (speeds <= 0).any()):
        raise ValueError("Speeds and spacing must be finite positive XYZ triples")
    shape, source_flat, donors, fronts = _geometry(shape, source, speeds.device)
    weights = (speeds / spacing).square()
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Rescale speeds/spacing to representable weights")
    arrival = torch.full((donors.shape[0],), float("inf"), dtype=torch.float64, device=speeds.device)
    arrival[source_flat] = 0.
    for front in fronts:
        ids = donors[front]
        times = torch.where(ids >= 0, arrival[ids.clamp_min(0)], float("inf"))
        arrival[front] = _local(times, weights)
    times = arrival[donors.clamp_min(0)]
    active = (donors >= 0) & (times < arrival[:, None])
    donors = torch.where(active, donors, -1)
    delta = torch.where(active, arrival[:, None] - times, 0.)
    residual = (weights * delta.square()).sum(dim=1) - 1.
    residual[source_flat] = 0.
    max_residual = float(residual.abs().max())
    if not torch.isfinite(arrival).all() or not np.isfinite(max_residual) or max_residual > residual_tol:
        raise RuntimeError(f"CUDA 3D branch verification failed: residual={max_residual}")
    return arrival.reshape(shape), donors, fronts, spacing, max_residual


@torch.no_grad()
def _speed_vjp(arrival, donors, fronts, speeds, spacing, upstream):
    flat = arrival.flatten()
    ids = donors.clamp_min(0)
    delta = torch.where(donors >= 0, flat[:, None] - flat[ids], 0.)
    weights = (speeds / spacing).square()
    denominator = (weights * delta).sum(dim=1)
    denominator = torch.where(denominator > 0, denominator, 1.)  # fixed source
    coefficients = weights * delta / denominator[:, None]
    direct = -(speeds / spacing.square()) * delta.square() / denominator[:, None]
    sensitivity = torch.zeros_like(direct)
    for front in fronts:
        sensitivity[front] = (coefficients[front, :, None] * sensitivity[ids[front]]).sum(dim=1) + direct[front]
    gradient = (upstream.double().reshape(-1, 1) * sensitivity).sum(dim=0)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite CUDA shared-speed gradient")
    return gradient


class _TensorArrival(torch.autograd.Function):
    @staticmethod
    def forward(ctx, speeds, shape, source, spacing):
        values = speeds.detach().double()
        arrival, donors, fronts, spacing, _ = _forward(values, shape, source, spacing)
        ctx.save_for_backward(arrival, donors, values, spacing)
        ctx.fronts, ctx.input_dtype = fronts, speeds.dtype
        return arrival.to(dtype=speeds.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        arrival, donors, speeds, spacing = ctx.saved_tensors
        gradient = _speed_vjp(arrival, donors, ctx.fronts, speeds, spacing, upstream)
        return gradient.to(ctx.input_dtype), None, None, None


def solve_eikonal_cuda(speeds, shape, source, spacing=(.8, .8, .8)):
    if speeds.device.type != "cuda":
        raise ValueError("backend=cuda requires CUDA speed tensors; move the 3D model to cuda")
    if speeds.dtype not in (torch.float32, torch.float64):
        raise ValueError("CUDA 3D speeds must be float32 or float64")
    return _TensorArrival.apply(speeds, shape, source, spacing)


def solve_diagonal_cuda(shape_zyx, source_zyx, speeds_xyz, spacing_xyz=(.8, .8, .8),
                        tol=1e-10, residual_tol=1e-8, max_cycles=64):
    """Existing NumPy result boundary; all solve arithmetic executes on CUDA.

    Wavefront solve is direct: cycles=1, max_change/root error=0 by construction.
    The positive CPU iteration/tolerance options are accepted for API parity.
    """
    from .solver import SolveResult
    if not torch.cuda.is_available():
        raise RuntimeError("backend=cuda requested but CUDA is unavailable")
    if (not np.isfinite(tol) or tol <= 0 or not np.isfinite(residual_tol) or residual_tol <= 0
            or not isinstance(max_cycles, (int, np.integer)) or max_cycles < 1):
        raise ValueError("Positive tolerances and max_cycles >= 1 required")
    torch.cuda.synchronize()
    began = perf_counter()
    speeds = torch.as_tensor(speeds_xyz, dtype=torch.float64, device="cuda")
    if speeds.ndim == 0:
        speeds = speeds.expand(3)
    arrival, donors, _, _, residual = _forward(speeds, shape_zyx, source_zyx, spacing_xyz, residual_tol)
    values, branches = arrival.cpu().numpy(), donors.cpu().numpy()
    return SolveResult(values, branches, 1, 0., residual, 0., "cuda", perf_counter() - began)
