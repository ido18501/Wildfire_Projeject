"""Executable documentation of two existing 2D contracts; no solver changes.

These assertions intentionally preserve observed historical behavior, including
the forward/backward mismatch. A passing test documents the concern; it does
not certify the old branch as mathematically consistent.

Run: python -X utf8 -m tests.verify_2d_contract_audit
"""

import time

import torch

from eikonal.backward.implicit_diff import (
    UpdateType,
    compute_stencil_residuals_and_jacobians,
)
from eikonal.forward.pytorch import (
    compute_one_point_update,
    compute_two_point_update,
    stencil_update,
)


def test_drift_uses_current_to_donor_offset():
    """A donor to the left gives cost 1-b_x for rightward propagation."""
    scalar = lambda value: torch.tensor(value, dtype=torch.float64)
    arrival = compute_one_point_update(
        *map(scalar, (0, 1, 0, 1, 0.2, 0)), m=(-1, 0)
    )
    assert abs(arrival.item() - 0.8) < 1e-12
    # For the displayed (grad T-b)^T G^-1(grad T-b)=1 the corresponding
    # homogeneous rightward characteristic instead has time 1+b_x=1.2.
    assert abs(arrival.item() - 1.2) > 0.3
    print("PASS observed 2D drift convention: rightward edge time=0.8; displayed PDE gives 1.2")


def test_clamped_update_is_not_reconstructed_residual():
    """A strictly SPD, zero-drift stencil accepts a non-root after clamping."""
    scalar = lambda value: torch.tensor(value, dtype=torch.float64)
    m1, m2 = (-1, -1), (-1, 0)
    g11, g12, g22, b1, b2 = map(scalar, (2, -3.5, 7, 0, 0))
    assert (g11 * g22 - g12.square()).item() > 0
    t1, t2, old = map(scalar, (1, 0, 100))
    candidate, valid = compute_two_point_update(t1, t2, g11, g12, g22, b1, b2, m1, m2)
    accepted = stencil_update(t1, t2, old, g11, g12, g22, b1, b2, m1, m2)
    assert valid.item()
    assert candidate.item() == accepted.item() == 1.0
    edge1 = compute_one_point_update(t1, g11, g12, g22, b1, b2, m1)
    edge2 = compute_one_point_update(t2, g11, g12, g22, b1, b2, m2)
    assert accepted < edge1 and accepted < edge2
    # E=M^T G M=[[2,-1.5],[-1.5,2]], Q=E^-1.
    e = torch.tensor([[2, -1.5], [-1.5, 2]], dtype=torch.float64)
    q = torch.linalg.inv(e)
    u = torch.stack((t1 - accepted, t2 - accepted))
    planar_residual = (u @ q @ u - 1).item()
    assert abs(planar_residual - 1 / 7) < 1e-12
    t = torch.full((1, 3, 3), 100.0, dtype=torch.float64)
    t[0, 1, 1], t[0, 0, 0], t[0, 1, 0] = accepted, t1, t2
    g = torch.tensor([2, -3.5, 7], dtype=torch.float64).view(1, 3, 1, 1).expand(1, 3, 3, 3)
    b = torch.zeros((1, 2, 3, 3), dtype=torch.float64)
    reconstructed = compute_stencil_residuals_and_jacobians(t, g, b, m1, m2)
    assert reconstructed["update_type"].item() == UpdateType.EDGE_M2
    assert abs(reconstructed["abs_residual"].item() - 0.5) < 1e-12
    print("PASS observed 2D clamp mismatch: accepted=1; planar residual=1/7; backward selects edge with |R|=0.5")


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    test_drift_uses_current_to_donor_offset()
    test_clamped_update_is_not_reconstructed_residual()
    print(f"2 historical contract regressions passed in {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
