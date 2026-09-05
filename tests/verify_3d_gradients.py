"""Centered finite differences of shared XYZ speeds through the 3D solve.

The scalar losses are fixed dense weights, an off-axis probe, and exact x/y/z
axis probes. Float64 and two relative perturbation sizes separate numerical
accuracy from finite-difference truncation. Run ``python -m tests.verify_3d_gradients``.
"""

import argparse
import json
import time

import numpy as np
import torch

from eikonal.three_d.autograd import solve_eikonal
from eikonal.three_d.solver import solve_diagonal


def _losses(shape, source, seed, include_probes):
    rng = np.random.default_rng(seed)
    dense = rng.uniform(0.2, 1.8, size=shape)
    dense /= dense.sum()
    losses = [("dense", dense, (0, 1, 2))]
    if include_probes:
        probe = np.zeros(shape)
        probe[-1, 1, -1] = 1.0
        losses.append(("off_axis_probe", probe, (0, 1, 2)))
        for axis_xyz, axis_zyx in enumerate((2, 1, 0)):
            point = list(source)
            point[axis_zyx] = shape[axis_zyx] - 1
            weight = np.zeros(shape)
            weight[tuple(point)] = 1.0
            losses.append(("axis_" + "xyz"[axis_xyz], weight, (axis_xyz,)))
    return losses


def run_verification(backend="auto", verbose=True):
    started = time.perf_counter()
    records = []
    problems = [
        ("interior", (5, 7, 9), (1, 4, 2), (0.57, 0.39, 0.28), (0.7, 0.9, 1.1), True),
        ("boundary", (6, 5, 8), (5, 0, 7), (0.42, 0.63, 0.31), (1.3, 0.65, 0.9), False),
    ]
    for problem, shape, source, values, spacing, include_probes in problems:
        for loss_name, weights, axes in _losses(shape, source, 903, include_probes):
            speeds = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            arrival = solve_eikonal(speeds, shape, source, spacing, backend=backend)
            loss = (arrival * torch.from_numpy(weights)).sum()
            automatic, = torch.autograd.grad(loss, speeds)
            automatic = automatic.detach().numpy()
            assert np.isfinite(automatic).all(), (problem, loss_name, automatic)
            # This exact homogeneity identity is independent of finite differences.
            scaling_error = abs(float(np.dot(values, automatic)) + float(loss.detach()))
            assert scaling_error < 1e-8, (problem, loss_name, "scaling", scaling_error)
            if loss_name.startswith("axis_"):
                for axis in set(range(3)) - set(axes):
                    assert abs(automatic[axis]) < 1e-10, (loss_name, axis, automatic)
            for axis in axes:
                assert abs(automatic[axis]) > 1e-8, (problem, loss_name, axis, "untested zero component")
                for relative_step in (1e-4, 1e-5):
                    step = relative_step * values[axis]
                    plus, minus = np.array(values), np.array(values)
                    plus[axis] += step
                    minus[axis] -= step
                    upper = solve_diagonal(shape, source, plus, spacing, backend=backend).arrival
                    lower = solve_diagonal(shape, source, minus, spacing, backend=backend).arrival
                    finite_difference = float(np.sum(weights * (upper - lower)) / (2 * step))
                    absolute_error = abs(float(automatic[axis]) - finite_difference)
                    relative_error = absolute_error / max(abs(float(automatic[axis])), abs(finite_difference), 1e-12)
                    record = {
                        "problem": problem,
                        "loss": loss_name,
                        "axis": "xyz"[axis],
                        "relative_step": relative_step,
                        "step_mm_per_ms": step,
                        "autograd": float(automatic[axis]),
                        "finite_difference": finite_difference,
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                    }
                    records.append(record)
                    if verbose:
                        print("GRAD {}/{} {} step_rel={:.0e} AD={:.10e} FD={:.10e} abs={:.3e} rel={:.3e}".format(
                            problem, loss_name, record["axis"], relative_step, record["autograd"], finite_difference, absolute_error, relative_error
                        ))
                    assert absolute_error < 1e-8 or relative_error < 2e-5, record
    # Two independent sources must share tissue parameters while retaining
    # independent arrival solutions and autograd contexts.
    shape, sources = (5, 7, 9), [(1, 4, 2), (4, 0, 8)]
    values, spacing = (0.57, 0.39, 0.28), (0.7, 0.9, 1.1)
    weights = [_losses(shape, source, 903 + i, False)[0][1] for i, source in enumerate(sources)]
    speeds = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss = sum((solve_eikonal(speeds, shape, source, spacing, backend=backend) * torch.from_numpy(weight)).sum()
               for source, weight in zip(sources, weights))
    automatic, = torch.autograd.grad(loss, speeds)
    automatic = automatic.detach().numpy()
    assert np.isfinite(automatic).all() and np.all(np.abs(automatic) > 1e-8)
    for axis in range(3):
        for relative_step in (1e-4, 1e-5):
            step = relative_step * values[axis]
            plus, minus = np.array(values), np.array(values)
            plus[axis] += step
            minus[axis] -= step
            difference = 0.0
            for source, weight in zip(sources, weights):
                upper = solve_diagonal(shape, source, plus, spacing, backend=backend).arrival
                lower = solve_diagonal(shape, source, minus, spacing, backend=backend).arrival
                difference += float(np.sum(weight * (upper - lower)))
            finite_difference = difference / (2 * step)
            absolute_error = abs(float(automatic[axis]) - finite_difference)
            relative_error = absolute_error / max(abs(float(automatic[axis])), abs(finite_difference), 1e-12)
            record = {
                "problem": "two_sources", "loss": "shared_dense", "axis": "xyz"[axis],
                "relative_step": relative_step, "step_mm_per_ms": step,
                "autograd": float(automatic[axis]), "finite_difference": finite_difference,
                "absolute_error": absolute_error, "relative_error": relative_error,
            }
            records.append(record)
            if verbose:
                print("GRAD two_sources/shared_dense {} step_rel={:.0e} AD={:.10e} FD={:.10e} abs={:.3e} rel={:.3e}".format(
                    record["axis"], relative_step, record["autograd"], finite_difference, absolute_error, relative_error
                ))
            assert absolute_error < 1e-8 or relative_error < 2e-5, record
    elapsed = time.perf_counter() - started
    if verbose:
        print("PASS: {} centered finite differences; max_abs={:.3e}, max_rel={:.3e}; {:.3f} s".format(
            len(records), max(r["absolute_error"] for r in records), max(r["relative_error"] for r in records), elapsed
        ))
    return {"passed": True, "records": records, "elapsed_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "python", "numba"), default="auto")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args()
    result = run_verification(args.backend, verbose=not args.json)
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
