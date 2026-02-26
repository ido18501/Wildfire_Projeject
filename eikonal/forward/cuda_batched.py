"""
Batched Multi-Fire CUDA Solver v3 - Source Mask Support

Extension of v2 to support source masks instead of single source points.
This ensures exact match with the sequential solver which uses 3x3 source regions.

Key differences from v2:
- source_masks: (n_fires, M, N) boolean array instead of list of (row, col)
- T is initialized to 0 where source_mask is True, INF elsewhere
"""

import numpy as np
from numba import cuda
import math
import torch
from typing import Tuple, List, Union
import time


@cuda.jit(device=True)
def _stencil_update_device(
    T1, T2, T0,
    g11, g12, g22,
    b1, b2,
    m1x, m1y,
    m2x, m2y,
):
    """Device function: compute stencil update for a single pixel."""
    eps = 1e-12
    INF = 1e10
    
    e11 = g11 * (m1x * m1x) + 2.0 * g12 * (m1x * m1y) + g22 * (m1y * m1y)
    e22 = g11 * (m2x * m2x) + 2.0 * g12 * (m2x * m2y) + g22 * (m2y * m2y)
    e12 = g11 * (m1x * m2x) + g12 * (m1x * m2y + m1y * m2x) + g22 * (m1y * m2y)

    detE = e11 * e22 - e12 * e12
    det_safe = max(detE, eps)
    q11 = e22 / det_safe
    q12 = -e12 / det_safe
    q22 = e11 / det_safe

    m1_dot_b = m1x * b1 + m1y * b2
    m2_dot_b = m2x * b1 + m2y * b2
    S1 = T1 + m1_dot_b
    S2 = T2 + m2_dot_b

    alpha = q11 + 2.0 * q12 + q22
    beta = (q11 + q12) * S1 + (q12 + q22) * S2
    gamma = q11 * S1 * S1 + 2.0 * q12 * S1 * S2 + q22 * S2 * S2 - 1.0

    discriminant = beta * beta - alpha * gamma
    disc_safe = max(discriminant, 0.0)
    alpha_safe = max(alpha, eps)
    
    t0_2pt = (beta + math.sqrt(disc_safe)) / alpha_safe
    t0_2pt = max(t0_2pt, max(T1, T2))
    
    lam1 = q11 * (t0_2pt - S1) + q12 * (t0_2pt - S2)
    lam2 = q12 * (t0_2pt - S1) + q22 * (t0_2pt - S2)
    
    if lam1 < 0.0 or lam2 < 0.0:
        t0_1pt_1 = T1 + math.sqrt(max(e11, eps)) + m1_dot_b
        t0_1pt_2 = T2 + math.sqrt(max(e22, eps)) + m2_dot_b
        t0_cand = min(t0_1pt_1, t0_1pt_2)
    else:
        t0_cand = t0_2pt
    
    return min(T0, t0_cand)


@cuda.jit
def _batched_sweep_kernel_source_mask(
    T_batch,       # (n_fires, M, N)
    G_batch,       # (n_fires, 3, M, N)
    B_batch,       # (n_fires, 2, M, N)
    source_masks,  # (n_fires, M, N) - boolean/int8 array
    M, N, n_fires, max_iter
):
    """
    CUDA kernel for batched fast sweeping with source mask support.
    
    Each block handles one fire. Threads within block process rows/columns in parallel.
    Source mask allows arbitrary source regions (not just single points).
    """
    fire_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if fire_idx >= n_fires:
        return
    
    # Initialize T for this fire using source mask
    for idx in range(thread_idx, M * N, block_size):
        i = idx // N
        j = idx % N
        if source_masks[fire_idx, i, j]:
            T_batch[fire_idx, i, j] = 0.0
        else:
            T_batch[fire_idx, i, j] = 1e6
    
    cuda.syncthreads()
    
    # Iterate
    for iteration in range(max_iter):
        # Sweep 1: columns left to right
        p = -1.0
        for j in range(1, N - 1):
            j_nbr = j - 1
            for i in range(1 + thread_idx, M - 1, block_size):
                # Skip source pixels - they stay at 0
                if source_masks[fire_idx, i, j]:
                    continue
                    
                g11 = G_batch[fire_idx, 0, i, j]
                g12 = G_batch[fire_idx, 1, i, j]
                g22 = G_batch[fire_idx, 2, i, j]
                b1 = B_batch[fire_idx, 0, i, j]
                b2 = B_batch[fire_idx, 1, i, j]
                T0 = T_batch[fire_idx, i, j]
                
                T1_up = T_batch[fire_idx, i - 1, j_nbr]
                T2_up = T_batch[fire_idx, i, j_nbr]
                T_updated = _stencil_update_device(
                    T1_up, T2_up, T0, g11, g12, g22, b1, b2,
                    p, -1.0, p, 0.0
                )
                
                T1_dn = T_batch[fire_idx, i + 1, j_nbr]
                T2_dn = T_batch[fire_idx, i, j_nbr]
                T_updated = _stencil_update_device(
                    T1_dn, T2_dn, T_updated, g11, g12, g22, b1, b2,
                    p, 1.0, p, 0.0
                )
                
                T_batch[fire_idx, i, j] = T_updated
            cuda.syncthreads()
        
        # Sweep 2: rows top to bottom
        p = -1.0
        for i in range(1, M - 1):
            i_nbr = i - 1
            for j in range(1 + thread_idx, N - 1, block_size):
                if source_masks[fire_idx, i, j]:
                    continue
                    
                g11 = G_batch[fire_idx, 0, i, j]
                g12 = G_batch[fire_idx, 1, i, j]
                g22 = G_batch[fire_idx, 2, i, j]
                b1 = B_batch[fire_idx, 0, i, j]
                b2 = B_batch[fire_idx, 1, i, j]
                T0 = T_batch[fire_idx, i, j]
                
                T1_left = T_batch[fire_idx, i_nbr, j - 1]
                T2_left = T_batch[fire_idx, i_nbr, j]
                T_updated = _stencil_update_device(
                    T1_left, T2_left, T0, g11, g12, g22, b1, b2,
                    -1.0, p, 0.0, p
                )
                
                T1_right = T_batch[fire_idx, i_nbr, j + 1]
                T2_right = T_batch[fire_idx, i_nbr, j]
                T_updated = _stencil_update_device(
                    T1_right, T2_right, T_updated, g11, g12, g22, b1, b2,
                    1.0, p, 0.0, p
                )
                
                T_batch[fire_idx, i, j] = T_updated
            cuda.syncthreads()
        
        # Sweep 3: columns right to left
        p = 1.0
        for j in range(N - 2, 0, -1):
            j_nbr = j + 1
            for i in range(1 + thread_idx, M - 1, block_size):
                if source_masks[fire_idx, i, j]:
                    continue
                    
                g11 = G_batch[fire_idx, 0, i, j]
                g12 = G_batch[fire_idx, 1, i, j]
                g22 = G_batch[fire_idx, 2, i, j]
                b1 = B_batch[fire_idx, 0, i, j]
                b2 = B_batch[fire_idx, 1, i, j]
                T0 = T_batch[fire_idx, i, j]
                
                T1_up = T_batch[fire_idx, i - 1, j_nbr]
                T2_up = T_batch[fire_idx, i, j_nbr]
                T_updated = _stencil_update_device(
                    T1_up, T2_up, T0, g11, g12, g22, b1, b2,
                    p, -1.0, p, 0.0
                )
                
                T1_dn = T_batch[fire_idx, i + 1, j_nbr]
                T2_dn = T_batch[fire_idx, i, j_nbr]
                T_updated = _stencil_update_device(
                    T1_dn, T2_dn, T_updated, g11, g12, g22, b1, b2,
                    p, 1.0, p, 0.0
                )
                
                T_batch[fire_idx, i, j] = T_updated
            cuda.syncthreads()
        
        # Sweep 4: rows bottom to top
        p = 1.0
        for i in range(M - 2, 0, -1):
            i_nbr = i + 1
            for j in range(1 + thread_idx, N - 1, block_size):
                if source_masks[fire_idx, i, j]:
                    continue
                    
                g11 = G_batch[fire_idx, 0, i, j]
                g12 = G_batch[fire_idx, 1, i, j]
                g22 = G_batch[fire_idx, 2, i, j]
                b1 = B_batch[fire_idx, 0, i, j]
                b2 = B_batch[fire_idx, 1, i, j]
                T0 = T_batch[fire_idx, i, j]
                
                T1_left = T_batch[fire_idx, i_nbr, j - 1]
                T2_left = T_batch[fire_idx, i_nbr, j]
                T_updated = _stencil_update_device(
                    T1_left, T2_left, T0, g11, g12, g22, b1, b2,
                    -1.0, p, 0.0, p
                )
                
                T1_right = T_batch[fire_idx, i_nbr, j + 1]
                T2_right = T_batch[fire_idx, i_nbr, j]
                T_updated = _stencil_update_device(
                    T1_right, T2_right, T_updated, g11, g12, g22, b1, b2,
                    1.0, p, 0.0, p
                )
                
                T_batch[fire_idx, i, j] = T_updated
            cuda.syncthreads()


def fast_sweeping_solve_batched_v3(
    G_batch: np.ndarray,  # (n_fires, 3, M, N)
    B_batch: np.ndarray,  # (n_fires, 2, M, N)
    source_masks: np.ndarray,  # (n_fires, M, N) boolean
    max_iter: int = 50,
    threads_per_block: int = 128,
) -> np.ndarray:
    """
    Solve multiple fires in parallel with per-fire G, B, and source masks.
    
    This version accepts source masks instead of single points, ensuring
    exact match with the sequential solver which uses 3x3 source regions.
    
    Args:
        G_batch: Metric tensor for each fire, shape (n_fires, 3, M, N)
        B_batch: Drift field for each fire, shape (n_fires, 2, M, N)
        source_masks: Boolean source masks, shape (n_fires, M, N)
        max_iter: Number of sweeping iterations
        threads_per_block: CUDA threads per block
        
    Returns:
        T_batch: Arrival times, shape (n_fires, M, N)
    """
    n_fires = source_masks.shape[0]
    assert G_batch.shape[0] == n_fires, f"G_batch has {G_batch.shape[0]} fires, expected {n_fires}"
    assert B_batch.shape[0] == n_fires, f"B_batch has {B_batch.shape[0]} fires, expected {n_fires}"
    
    _, _, M, N = G_batch.shape
    assert source_masks.shape == (n_fires, M, N), f"source_masks shape mismatch"
    
    # Ensure contiguous arrays with correct dtypes
    G_batch = np.ascontiguousarray(G_batch, dtype=np.float32)
    B_batch = np.ascontiguousarray(B_batch, dtype=np.float32)
    source_masks = np.ascontiguousarray(source_masks, dtype=np.int8)  # Numba needs int, not bool
    
    # Allocate output
    T_batch = np.zeros((n_fires, M, N), dtype=np.float32)
    
    # Transfer to GPU
    d_T = cuda.to_device(T_batch)
    d_G = cuda.to_device(G_batch)
    d_B = cuda.to_device(B_batch)
    d_source = cuda.to_device(source_masks)
    
    # Launch kernel: one block per fire
    blocks = n_fires
    threads = threads_per_block
    
    _batched_sweep_kernel_source_mask[blocks, threads](
        d_T, d_G, d_B, d_source,
        M, N, n_fires, max_iter
    )
    
    # Copy back
    d_T.copy_to_host(T_batch)
    
    return T_batch


# Convenience function that accepts either source_masks or source_points
def fast_sweeping_solve_batched(
    G_batch: np.ndarray,
    B_batch: np.ndarray,
    sources: Union[np.ndarray, List[Tuple[int, int]]],
    max_iter: int = 50,
    threads_per_block: int = 128,
) -> np.ndarray:
    """
    Unified interface: accepts either source masks or source points.
    
    Args:
        G_batch: (n_fires, 3, M, N)
        B_batch: (n_fires, 2, M, N)
        sources: Either:
            - np.ndarray of shape (n_fires, M, N) - source masks
            - List of (row, col) tuples - source points (converted to masks)
        max_iter: Number of iterations
        threads_per_block: CUDA config
        
    Returns:
        T_batch: (n_fires, M, N)
    """
    n_fires = G_batch.shape[0]
    M, N = G_batch.shape[2], G_batch.shape[3]
    
    # Convert source_points to source_masks if needed
    if isinstance(sources, list):
        source_masks = np.zeros((n_fires, M, N), dtype=np.int8)
        for k, (si, sj) in enumerate(sources):
            source_masks[k, si, sj] = 1
    else:
        source_masks = sources
    
    return fast_sweeping_solve_batched_v3(
        G_batch, B_batch, source_masks, max_iter, threads_per_block
    )


def warmup_v3():
    """Warmup the JIT compiler for v3 kernels."""
    G = np.ones((2, 3, 32, 32), dtype=np.float32) * 0.35
    G[:, 1] = 0
    B = np.zeros((2, 2, 32, 32), dtype=np.float32)
    source_masks = np.zeros((2, 32, 32), dtype=np.int8)
    source_masks[0, 15:18, 15:18] = 1  # 3x3 region
    source_masks[1, 15:18, 15:18] = 1
    fast_sweeping_solve_batched_v3(G, B, source_masks, max_iter=2)


# =============================================================================
# Verification
# =============================================================================

def verify_v3_correctness():
    """Verify v3 matches sequential solver with source masks."""
    import sys
    
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    from eikonal.forward.pytorch import EikonalProblem
    
    print("="*70)
    print("Verifying v3 (source mask) correctness")
    print("="*70)
    
    # Warmup
    warmup_v3()
    
    # Test with realistic anisotropic G and 3x3 source masks
    K = 4
    M, N = 64, 64
    
    np.random.seed(42)
    
    # Create anisotropic G fields
    G_batch = np.zeros((K, 3, M, N), dtype=np.float32)
    B_batch = np.zeros((K, 2, M, N), dtype=np.float32)
    source_masks = np.zeros((K, M, N), dtype=np.int8)
    
    for k in range(K):
        # Anisotropic metric
        angle = np.random.rand() * np.pi
        c, s = np.cos(angle), np.sin(angle)
        base = 0.3 + 0.3 * np.random.rand()
        ratio = 1.0 + 0.5 * np.random.rand()
        
        G_batch[k, 0] = (c**2 * ratio + s**2) * base  # g11
        G_batch[k, 1] = (c * s * (ratio - 1)) * base   # g12
        G_batch[k, 2] = (s**2 * ratio + c**2) * base   # g22
        
        # Small drift
        B_batch[k, 0] = 0.05 * np.random.randn()
        B_batch[k, 1] = 0.05 * np.random.randn()
        
        # 3x3 source mask (like your prepare_scenario)
        si, sj = 20 + k * 5, 20 + k * 5
        source_masks[k, max(0,si-1):min(M,si+2), max(0,sj-1):min(N,sj+2)] = 1
    
    print(f"\nConfiguration: K={K}, grid={M}x{N}, 3x3 source masks")
    
    # Sequential reference
    print("\n1. Running sequential Numba solver...")
    T_seq = []
    t0 = time.perf_counter()
    for k in range(K):
        source_mask_torch = torch.from_numpy(source_masks[k].astype(bool)).unsqueeze(0)
        G_torch = torch.from_numpy(G_batch[k:k+1])
        B_torch = torch.from_numpy(B_batch[k:k+1])
        problem = EikonalProblem(G_torch, B_torch, source_mask_torch)
        T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
        T_seq.append(T[0].numpy())
    t_seq = time.perf_counter() - t0
    T_seq = np.stack(T_seq)
    print(f"   Time: {t_seq*1000:.1f} ms")
    
    # Batched v3
    print("\n2. Running batched v3 solver...")
    t0 = time.perf_counter()
    T_batched = fast_sweeping_solve_batched_v3(G_batch, B_batch, source_masks, max_iter=50)
    t_batched = time.perf_counter() - t0
    print(f"   Time: {t_batched*1000:.1f} ms")
    
    # Compare
    print("\n3. Comparing results...")
    all_pass = True
    for k in range(K):
        valid = (T_seq[k] < 1e5) & (T_batched[k] < 1e5)
        if valid.sum() == 0:
            print(f"   Fire {k}: No valid pixels!")
            continue
        
        diff = np.abs(T_seq[k] - T_batched[k])
        max_diff = diff[valid].max()
        mean_diff = diff[valid].mean()
        
        # Correlation
        corr = np.corrcoef(T_seq[k][valid].flatten(), T_batched[k][valid].flatten())[0, 1]
        
        status = "✓" if max_diff < 1e-5 else "✗"
        if max_diff >= 1e-5:
            all_pass = False
        
        print(f"   Fire {k}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, corr={corr:.6f} {status}")
    
    print(f"\n   Speedup: {t_seq/t_batched:.2f}x")
    print(f"\n{'='*70}")
    print(f"RESULT: {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*70}")
    
    return all_pass


if __name__ == "__main__":
    verify_v3_correctness()
