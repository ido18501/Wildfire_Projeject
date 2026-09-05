"""CPU reference for the 3D diagonal eikonal equation, with zero drift.

Arrays are [z,y,x], speeds and spacing are (x,y,z). Cardiac units are
mm/ms, mm, and ms. This is a first-order Godunov discretization, not a
general SPD/Randers stencil. Both Python and optional Numba run the same
serial sweep/update source code; no analytic travel times seed the solver.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

try:
    from numba import njit
    from numba.extending import register_jitable
except ImportError:  # Correctness does not require the optional accelerator.
    njit = None

    def register_jitable(function):
        return function


@register_jitable
def _local_update_core(times, weights):
    """Solve sum weights * max(t-times,0)**2 = 1 without root clamps.

Enter larger prefixes only if the previous root exceeds the next donor.
For prefix maximum q, solve A*s*s+2*B*s+E=1, t=q+s. The rationalized
positive increment avoids subtracting nearby quadratic roots. Invalid
larger prefixes are never accepted; an admissible face/edge is used.
"""
    order = np.argsort(times)
    active = np.zeros(3, dtype=np.bool_)
    if not np.isfinite(times[order[0]]):
        return np.inf, active
    root = np.inf
    for count in range(1, 4):
        q = times[order[count - 1]]
        if not np.isfinite(q):
            break
        a, b, e = 0.0, 0.0, 0.0
        for j in range(count):
            axis = order[j]
            d = q - times[axis]
            a += weights[axis]
            b += weights[axis] * d
            e += weights[axis] * d * d
        remaining = 1.0 - e
        if remaining <= 0.0:
            # A prefix reached in exact arithmetic always has E<1. Fail
            # explicitly if floating-point resolution destroys this fact.
            raise ArithmeticError("Noncausal local prefix: rescale speeds/spacing")
        increment = remaining / (np.sqrt(b * b + a * remaining) + b)
        root = q + increment
        if not np.isfinite(root) or root <= q:
            raise ArithmeticError("Local time increment is not representable")
        if count == 3 or root <= times[order[count]]:
            break
    for axis in range(3):
        active[axis] = times[axis] < root
    return root, active


def local_update(donor_times_xyz, weights_xyz):
    """Public local reference; +inf denotes an unavailable axis donor."""
    times = np.asarray(donor_times_xyz, dtype=np.float64)
    weights = np.asarray(weights_xyz, dtype=np.float64)
    if times.shape != (3,) or weights.shape != (3,):
        raise ValueError("Expected three XYZ donor times and weights")
    if np.isnan(times).any() or np.isneginf(times).any():
        raise ValueError("Donor times must be finite or +inf")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be finite and positive")
    return _local_update_core(times, weights)


@register_jitable
def _node_update(arrival, z, y, x, weights):
    nz, ny, nx = arrival.shape
    times = np.full(3, np.inf)
    donors = np.full(3, -1, dtype=np.int64)
    # Negative-side donor wins exact opposite-neighbor ties, consistently.
    if x > 0:
        times[0] = arrival[z, y, x - 1]
        donors[0] = (z * ny + y) * nx + x - 1
    if x + 1 < nx and arrival[z, y, x + 1] < times[0]:
        times[0] = arrival[z, y, x + 1]
        donors[0] = (z * ny + y) * nx + x + 1
    if y > 0:
        times[1] = arrival[z, y - 1, x]
        donors[1] = (z * ny + y - 1) * nx + x
    if y + 1 < ny and arrival[z, y + 1, x] < times[1]:
        times[1] = arrival[z, y + 1, x]
        donors[1] = (z * ny + y + 1) * nx + x
    if z > 0:
        times[2] = arrival[z - 1, y, x]
        donors[2] = ((z - 1) * ny + y) * nx + x
    if z + 1 < nz and arrival[z + 1, y, x] < times[2]:
        times[2] = arrival[z + 1, y, x]
        donors[2] = ((z + 1) * ny + y) * nx + x
    root, active = _local_update_core(times, weights)
    for axis in range(3):
        if not active[axis]:
            donors[axis] = -1
    return root, donors


def _sweep_core(arrival, source, weights, tol, max_cycles):
    nz, ny, nx = arrival.shape
    max_change = np.inf
    for cycle in range(max_cycles):
        max_change = 0.0
        for ordering in range(8):
            dz = 1 if (ordering & 4) == 0 else -1
            dy = 1 if (ordering & 2) == 0 else -1
            dx = 1 if (ordering & 1) == 0 else -1
            for z in range(0 if dz == 1 else nz - 1, nz if dz == 1 else -1, dz):
                for y in range(0 if dy == 1 else ny - 1, ny if dy == 1 else -1, dy):
                    for x in range(0 if dx == 1 else nx - 1, nx if dx == 1 else -1, dx):
                        if z == source[0] and y == source[1] and x == source[2]:
                            continue
                        candidate, _ = _node_update(arrival, z, y, x, weights)
                        old = arrival[z, y, x]
                        if candidate < old:
                            # Do not suppress small positive decreases.
                            max_change = max(max_change, old - candidate)
                            arrival[z, y, x] = candidate
        if max_change <= tol:
            return cycle + 1, max_change
    return max_cycles, max_change


def _reconstruct_core(arrival, source, weights):
    """Reapply the SAME update to final values; retain verified causal donors."""
    nz, ny, nx = arrival.shape
    flat = arrival.ravel()
    donors = np.full((flat.size, 3), -1, dtype=np.int64)
    max_residual, max_error = 0.0, 0.0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if z == source[0] and y == source[1] and x == source[2]:
                    continue
                idx = (z * ny + y) * nx + x
                root, ids = _node_update(arrival, z, y, x, weights)
                residual = -1.0
                count = 0
                for axis in range(3):
                    donor = ids[axis]
                    if donor >= 0:
                        if flat[donor] >= flat[idx]:
                            raise ArithmeticError("Final donor is not strictly causal")
                        count += 1
                        donors[idx, axis] = donor
                        delta = flat[idx] - flat[donor]
                        residual += weights[axis] * delta * delta
                if count == 0:
                    raise ArithmeticError("Nonsource has no causal donor")
                max_residual = max(max_residual, abs(residual))
                max_error = max(max_error, abs(root - flat[idx]))
    return donors, max_residual, max_error


_sweep_compiled = njit(cache=True)(_sweep_core) if njit is not None else None
_reconstruct_compiled = njit(cache=True)(_reconstruct_core) if njit is not None else None


@dataclass(frozen=True)
class SolveResult:
    arrival: np.ndarray
    donors: np.ndarray
    cycles: int
    max_change: float
    max_residual: float
    max_fixed_point_error: float
    backend: str
    elapsed_seconds: float


def solve_diagonal(shape_zyx, source_zyx, speeds_xyz, spacing_xyz=(0.8, 0.8, 0.8),
                   tol=1e-10, residual_tol=1e-8, max_cycles=64, backend="auto"):
    """Converged single-source diagonal solve in float64 on all volume planes.

    Grid-aligned sources only. ``speeds_xyz`` is a scalar or three shared
    positive speeds. Outer faces use the available one-sided donors; they
    are included in the domain, not held at an unreachable sentinel.
    Raises on nonconvergence or inconsistent final branches. ``cycles``
    counts complete sets of eight sweep orderings, not individual sweeps.
    """
    started = perf_counter()
    shape_values = np.asarray(shape_zyx)
    source_values = np.asarray(source_zyx)
    if (shape_values.shape != (3,) or not np.issubdtype(shape_values.dtype, np.integer)
            or (shape_values < 1).any()):
        raise ValueError("shape_zyx must contain three positive integers")
    if (source_values.shape != (3,) or not np.issubdtype(source_values.dtype, np.integer)
            or (source_values < 0).any() or (source_values >= shape_values).any()):
        raise ValueError("source_zyx must contain three in-bounds integer voxel indices")
    speeds = np.asarray(speeds_xyz, dtype=np.float64)
    if speeds.ndim == 0:
        speeds = np.repeat(speeds, 3)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if speeds.shape != (3,) or spacing.shape != (3,):
        raise ValueError("speeds_xyz and spacing_xyz must be scalar/3 and 3 respectively")
    if (not np.isfinite(speeds).all() or (speeds <= 0).any()
            or not np.isfinite(spacing).all() or (spacing <= 0).any()):
        raise ValueError("Speeds and spacing must be finite and positive")
    if (not np.isfinite(tol) or tol <= 0 or not np.isfinite(residual_tol)
            or residual_tol <= 0 or not isinstance(max_cycles, (int, np.integer))
            or max_cycles < 1):
        raise ValueError("Positive tolerances and integer max_cycles >= 1 required")
    if backend == "auto":
        backend = "numba" if njit is not None else "python"
    if backend not in ("python", "numba"):
        raise ValueError("backend must be auto, python, or numba")
    if backend == "numba" and njit is None:
        raise ImportError("Numba backend requested; install the numba extra or use python")
    with np.errstate(over="ignore", under="ignore"):
        weights = (speeds / spacing) ** 2
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Speeds/spacing cannot be represented safely; rescale units")
    source = np.asarray(source_values, dtype=np.int64)
    arrival = np.full(tuple(shape_values), np.inf, dtype=np.float64)
    arrival[tuple(source)] = 0.0
    sweep = _sweep_core if backend == "python" else _sweep_compiled
    reconstruct = _reconstruct_core if backend == "python" else _reconstruct_compiled
    cycles, change = sweep(arrival, source, weights, tol, max_cycles)
    if not np.isfinite(arrival).all() or change > tol:
        raise RuntimeError(f"3D sweeping did not converge in {cycles} cycles (change={change})")
    donors, residual, error = reconstruct(arrival, source, weights)
    if not np.isfinite(residual) or residual > residual_tol or error > tol:
        raise RuntimeError(f"3D branch verification failed: residual={residual}, root error={error}")
    return SolveResult(arrival, donors, cycles, float(change), float(residual),
                       float(error), backend, perf_counter() - started)
