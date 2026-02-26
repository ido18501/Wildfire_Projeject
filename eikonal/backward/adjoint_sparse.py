"""
Sparse Adjoint Solve for Band-Decomposed Training
===================================================

When band training uses teacher-forced sources, the forward solve
produces arrival times where most of the grid is trivial:

  - Source pixels (T = 0):  50-70% of grid (teacher-forced burned region)
  - Active band pixels (0 < T < band_width): 10-20% of grid
  - Unreachable pixels (T = 1e6):  10-30% of grid

The standard adjoint solve processes ALL ~327K pixels on a 568x579 grid.
This sparse variant pre-filters to active pixels only, giving 5-10x
speedup on the adjoint solve.

CORRECTNESS ARGUMENT
--------------------

Only active pixels (0 < T < 1e5) need to be processed:

1. Source pixels (T=0) are processed last in decreasing-T order, so
   their lambda does not feed into any other pixel's computation.
   Their parameter gradient is 0 (SOURCE update type has dR/dG = 0).

2. Unreachable pixels (T >= 1e5) have lambda = 0 (initialized).
   Their Jacobian entries (dR_dT1, dR_dT2) are 0 because they were
   never updated by any stencil. Parameter gradient contribution is 0.

3. Active pixels are processed in correct decreasing-T order among
   themselves. Off-diagonal contributions from source/unreachable
   neighbors are 0 (their Jacobian entries and lambda are both 0).

This is a drop-in replacement for solve_adjoint_numba — same inputs,
same outputs, same interface.  The only difference is internal: it
pre-filters pixels before the sequential walk.

Author: Barak Gahtan
"""

import numpy as np
import torch
from torch import Tensor
import time

try:
    from numba import jit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# Sparse adjoint kernels
# ============================================================================

@jit(nopython=True, cache=True)
def _solve_adjoint_sparse_kernel(
    T_interior,             # (H, W) float32
    g,                      # (H, W) float64/32
    dR_dT0,                 # (H, W) float64/32
    dR_dT1,                 # (H, W) float64/32
    dR_dT2,                 # (H, W) float64/32
    m1_x, m1_y,            # (H, W) int
    m2_x, m2_y,            # (H, W) int
    active_sorted_indices,  # (N_active,) int64 — sorted by decreasing T
):
    """
    Adjoint solve visiting only active pixels.

    Identical logic to _solve_adjoint_numba_kernel but iterates over
    a pre-filtered, pre-sorted index array containing only pixels
    where 0 < T < 1e5.
    """
    H, W = T_interior.shape
    lam = np.zeros((H, W), dtype=np.float64)

    for idx_pos in range(len(active_sorted_indices)):
        idx = active_sorted_indices[idx_pos]
        i = idx // W
        j = idx % W

        # Accumulate off-diagonal: neighbors that use (i,j) as a donor
        off_diag = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    m1_nx = int(m1_x[ni, nj])
                    m1_ny = int(m1_y[ni, nj])
                    m2_nx = int(m2_x[ni, nj])
                    m2_ny = int(m2_y[ni, nj])
                    if (ni + m1_ny == i) and (nj + m1_nx == j):
                        off_diag += dR_dT1[ni, nj] * lam[ni, nj]
                    if (ni + m2_ny == i) and (nj + m2_nx == j):
                        off_diag += dR_dT2[ni, nj] * lam[ni, nj]

        dR_dT0_val = dR_dT0[i, j]
        g_val = g[i, j]
        if abs(dR_dT0_val) > 1e-12:
            lam[i, j] = (g_val - off_diag) / dR_dT0_val
        else:
            lam[i, j] = g_val

    return lam


@jit(nopython=True, parallel=True, cache=True)
def _solve_adjoint_sparse_batch(
    T_interior,   # (K, H, W)
    g,            # (K, H, W)
    dR_dT0,       # (K, H, W)
    dR_dT1,       # (K, H, W)
    dR_dT2,       # (K, H, W)
    m1_x, m1_y,  # (K, H, W)
    m2_x, m2_y,  # (K, H, W)
):
    """
    Batch sparse adjoint with prange over K bands.

    For each band, pre-filters to active pixels (T < 1e5),
    sorts only those, then calls the sparse kernel.
    """
    K, H, W = T_interior.shape
    n_pixels = H * W
    lam = np.zeros((K, H, W), dtype=np.float64)

    for k in prange(K):
        # --- Pre-filter to active pixels only ---
        T_flat = T_interior[k].ravel()

        # Count active pixels (include source T=0, exclude unreachable)
        n_active = 0
        for p in range(n_pixels):
            if T_flat[p] < 1e5:
                n_active += 1

        if n_active == 0:
            continue

        # Collect active indices and their T values
        active_indices = np.empty(n_active, dtype=np.int64)
        active_T = np.empty(n_active, dtype=np.float32)
        pos = 0
        for p in range(n_pixels):
            if T_flat[p] < 1e5:
                active_indices[pos] = p
                active_T[pos] = T_flat[p]
                pos += 1

        # Sort active indices by decreasing T
        sort_order = np.argsort(-active_T)
        active_sorted = np.empty(n_active, dtype=np.int64)
        for s in range(n_active):
            active_sorted[s] = active_indices[sort_order[s]]

        lam[k] = _solve_adjoint_sparse_kernel(
            T_interior[k], g[k],
            dR_dT0[k], dR_dT1[k], dR_dT2[k],
            m1_x[k], m1_y[k], m2_x[k], m2_y[k],
            active_sorted
        )

    return lam


# ============================================================================
# Public API (drop-in replacement for solve_adjoint_numba)
# ============================================================================

def solve_adjoint_numba_sparse(
    stencil_info,      # StencilInfo
    dL_dT: Tensor,     # (K, M, N)
    problem,           # EikonalProblem
    T: Tensor,         # (K, M, N)
    verbose: bool = False,
) -> Tensor:
    """
    Sparse adjoint solve — drop-in replacement for solve_adjoint_numba.

    Only processes pixels where T < 1e5, skipping unreachable (T >= 1e5)
    regions. For band-decomposed training this typically processes 15-50%
    of the grid, giving 2-5x speedup on the adjoint solve.
    """
    K, M, N = dL_dT.shape
    device = dL_dT.device

    T_interior = T[:, 1:-1, 1:-1].cpu().numpy()
    g = dL_dT[:, 1:-1, 1:-1].cpu().numpy()

    dR_dT0 = stencil_info.dR_dT0.cpu().numpy()
    dR_dT1 = stencil_info.dR_dT1.cpu().numpy()
    dR_dT2 = stencil_info.dR_dT2.cpu().numpy()

    m1_x = stencil_info.m1[:, 0].cpu().numpy()
    m1_y = stencil_info.m1[:, 1].cpu().numpy()
    m2_x = stencil_info.m2[:, 0].cpu().numpy()
    m2_y = stencil_info.m2[:, 1].cpu().numpy()

    if verbose:
        total = T_interior.size
        active = np.sum(T_interior < 1e5)
        print(f"  Sparse adjoint: {active}/{total} active pixels "
              f"({100*active/total:.1f}%)")

    t0 = time.time()

    if K == 1:
        # Single fire
        T_flat = T_interior[0].ravel()
        mask = T_flat < 1e5
        active_indices = np.where(mask)[0].astype(np.int64)
        active_T = T_flat[active_indices]
        sort_order = np.argsort(-active_T)
        active_sorted = active_indices[sort_order]

        lam_interior = _solve_adjoint_sparse_kernel(
            T_interior[0], g[0],
            dR_dT0[0], dR_dT1[0], dR_dT2[0],
            m1_x[0], m1_y[0], m2_x[0], m2_y[0],
            active_sorted
        )
        lam_interior = lam_interior[np.newaxis, :, :]
    else:
        # Batch (prange over K)
        lam_interior = _solve_adjoint_sparse_batch(
            T_interior, g,
            dR_dT0, dR_dT1, dR_dT2,
            m1_x, m1_y, m2_x, m2_y
        )

    lam = torch.zeros(K, M, N, device=device, dtype=dL_dT.dtype)
    lam[:, 1:-1, 1:-1] = torch.from_numpy(lam_interior).to(device)

    if verbose:
        dt = time.time() - t0
        print(f"  Sparse adjoint completed in {dt:.3f}s")
        print(f"  lambda range: [{lam.min().item():.4e}, "
              f"{lam.max().item():.4e}]")

    return lam


# ============================================================================
# Warmup (JIT compilation on tiny problem)
# ============================================================================

def warmup_sparse_adjoint():
    """Compile the sparse kernels with a tiny dummy problem."""
    H, W, K = 20, 20, 2
    T = np.ones((K, H, W), dtype=np.float32) * 5.0
    T[:, 10, 10] = 0.0  # source
    T[:, 0, :] = 1e6    # unreachable
    g = np.zeros((K, H, W), dtype=np.float64)
    g[:, 15, 15] = 1.0
    dR_dT0 = np.ones((K, H, W), dtype=np.float64) * -2.0
    dR_dT1 = np.ones((K, H, W), dtype=np.float64) * 1.0
    dR_dT2 = np.ones((K, H, W), dtype=np.float64) * 1.0
    m1_x = np.ones((K, H, W), dtype=np.int64)
    m1_y = np.zeros((K, H, W), dtype=np.int64)
    m2_x = np.zeros((K, H, W), dtype=np.int64)
    m2_y = np.ones((K, H, W), dtype=np.int64)
    _solve_adjoint_sparse_batch(
        T, g, dR_dT0, dR_dT1, dR_dT2,
        m1_x, m1_y, m2_x, m2_y
    )
