"""
Batched Backward Pass for Training

This module provides a complete batched backward pass that:
1. Processes multiple fires in parallel
2. Computes gradients w.r.t. G and B for each fire
3. Supports accumulating gradients to neural network parameters

Key insight: The existing backward pass code is already vectorized over K,
so we mainly need a clean interface that:
- Takes batched G, B, T, and loss gradients
- Returns batched gradients dL/dG, dL/dB
- Handles the EikonalProblem construction internally
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, List, NamedTuple
from dataclasses import dataclass
import time

# Import from existing modules
import sys

from eikonal.forward.pytorch import EikonalProblem
from eikonal.backward.implicit_diff import (
    identify_active_stencils, 
    compute_parameter_gradients,
    StencilInfo,
    UpdateType
)
from eikonal.backward.adjoint_numba import solve_adjoint_numba


@dataclass
class BatchedProblem:
    """
    Represents a batch of eikonal problems with potentially different G, B.
    
    This is a lightweight wrapper that provides the interface expected by
    backward_pass functions while supporting per-fire G/B tensors.
    """
    G: Tensor  # (K, 3, M, N)
    B: Tensor  # (K, 2, M, N)
    source_mask: Tensor  # (K, M, N)
    
    @property
    def K(self) -> int:
        return self.G.shape[0]
    
    @property
    def M(self) -> int:
        return self.G.shape[2]
    
    @property
    def N(self) -> int:
        return self.G.shape[3]


def batched_backward_pass(
    T_batch: Tensor,      # (K, M, N) converged arrival times
    dL_dT_batch: Tensor,  # (K, M, N) loss gradients
    G_batch: Tensor,      # (K, 3, M, N) metric tensors
    B_batch: Tensor,      # (K, 2, M, N) drift fields
    source_masks: Tensor,  # (K, M, N) boolean source masks
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Complete batched backward pass for K fires.
    
    Computes dL/dG and dL/dB for each fire in the batch, enabling
    gradient computation for batched training.
    
    Args:
        T_batch: Arrival times from forward solve, shape (K, M, N)
        dL_dT_batch: Gradient of loss w.r.t. T, shape (K, M, N)
        G_batch: Metric tensor for each fire, shape (K, 3, M, N)
        B_batch: Drift field for each fire, shape (K, 2, M, N)
        source_masks: Boolean source masks, shape (K, M, N)
        verbose: Print progress information
        
    Returns:
        dL_dG_batch: Gradient w.r.t. G, shape (K, 3, M, N)
        dL_dB_batch: Gradient w.r.t. B, shape (K, 2, M, N)
    """
    K, M, N = T_batch.shape
    device = T_batch.device
    
    # Convert source_masks to tensor if needed
    if isinstance(source_masks, np.ndarray):
        source_mask = torch.from_numpy(source_masks.astype(bool)).to(device)
    else:
        source_mask = source_masks.to(device)
    
    # Create batched problem
    problem = BatchedProblem(G_batch, B_batch, source_mask)
    
    if verbose:
        print(f"Batched backward pass: K={K}, M={M}, N={N}")
    
    # Step 1: Identify active stencils for all fires
    if verbose:
        print("  Step 1: Identifying active stencils...")
    t0 = time.perf_counter()
    stencil_info = identify_active_stencils(T_batch, problem)
    t_stencil = time.perf_counter() - t0
    
    if verbose:
        for ut in UpdateType:
            count = (stencil_info.update_type == ut).sum().item()
            print(f"    {ut.name}: {count} nodes")
        print(f"    Time: {t_stencil*1000:.1f} ms")
    
    # Step 2: Solve adjoint system for all fires (parallel via Numba)
    if verbose:
        print("  Step 2: Solving adjoint system...")
    t0 = time.perf_counter()
    lam = solve_adjoint_numba(stencil_info, dL_dT_batch, problem, T_batch, verbose=False)
    t_adjoint = time.perf_counter() - t0
    
    if verbose:
        print(f"    λ range: [{lam.min().item():.4e}, {lam.max().item():.4e}]")
        print(f"    Time: {t_adjoint*1000:.1f} ms")
    
    # Step 3: Compute parameter gradients for all fires
    if verbose:
        print("  Step 3: Computing parameter gradients...")
    t0 = time.perf_counter()
    dL_dG_batch, dL_dB_batch = compute_parameter_gradients(stencil_info, lam, problem)
    t_grad = time.perf_counter() - t0
    
    if verbose:
        print(f"    dL/dG range: [{dL_dG_batch.min().item():.4e}, {dL_dG_batch.max().item():.4e}]")
        print(f"    dL/dB range: [{dL_dB_batch.min().item():.4e}, {dL_dB_batch.max().item():.4e}]")
        print(f"    Time: {t_grad*1000:.1f} ms")
        print(f"  Total backward time: {(t_stencil + t_adjoint + t_grad)*1000:.1f} ms")
    
    return dL_dG_batch, dL_dB_batch


def verify_batched_backward():
    """Verify batched backward pass against sequential single-fire backward."""
    from eikonal.forward.pytorch import create_test_problem, fast_sweeping_solve
    from eikonal.backward.implicit_diff import backward_pass
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    
    print("="*70)
    print("Verifying Batched Backward Pass")
    print("="*70)
    
    # Create multiple test problems
    K = 4  # Batch size
    M, N = 64, 64
    
    print(f"\nTest configuration: K={K}, M={M}, N={N}")
    
    # Generate random G and B for each fire
    torch.manual_seed(42)
    G_batch = torch.ones(K, 3, M, N) * 0.35
    G_batch[:, 1] = 0  # g12 = 0
    
    # Add some variation
    for k in range(K):
        scale = 0.2 + 0.3 * (k / K)
        G_batch[k] *= scale
    
    B_batch = torch.zeros(K, 2, M, N)
    for k in range(K):
        B_batch[k, 0] = 0.05 * (k - K/2)
    
    # Random source points
    np.random.seed(42)
    source_points = [(np.random.randint(15, M-15), np.random.randint(15, N-15)) 
                     for _ in range(K)]
    
    # Forward solve each fire
    print("\n1. Forward solving each fire...")
    T_batch = torch.zeros(K, M, N)
    
    for k in range(K):
        source_mask = torch.zeros(1, M, N, dtype=torch.bool)
        source_mask[0, source_points[k][0], source_points[k][1]] = True
        
        problem = EikonalProblem(G_batch[k:k+1], B_batch[k:k+1], source_mask)
        T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
        T_batch[k] = T[0]
    
    print(f"   T range: [{T_batch[T_batch < 1e5].min():.2f}, {T_batch[T_batch < 1e5].max():.2f}]")
    
    # Create loss gradients (pretend loss is MSE at observed points)
    print("\n2. Creating loss gradients...")
    dL_dT_batch = torch.zeros(K, M, N)
    for k in range(K):
        # Random "observed" points
        obs_mask = torch.rand(M, N) < 0.3
        obs_mask &= (T_batch[k] < 1e5)
        
        # Pretend target is T + noise
        T_target = T_batch[k] + torch.randn(M, N) * 2
        
        # dL/dT = T - T_target at observed points
        dL_dT_batch[k][obs_mask] = (T_batch[k][obs_mask] - T_target[obs_mask])
    
    print(f"   dL/dT range: [{dL_dT_batch.min():.4f}, {dL_dT_batch.max():.4f}]")
    
    # Sequential backward pass (reference)
    print("\n3. Running sequential backward pass (reference)...")
    t0 = time.perf_counter()
    dL_dG_seq = torch.zeros(K, 3, M, N)
    dL_dB_seq = torch.zeros(K, 2, M, N)
    
    for k in range(K):
        source_mask = torch.zeros(1, M, N, dtype=torch.bool)
        source_mask[0, source_points[k][0], source_points[k][1]] = True
        
        problem = EikonalProblem(G_batch[k:k+1], B_batch[k:k+1], source_mask)
        dL_dG_k, dL_dB_k = backward_pass(
            T_batch[k:k+1], 
            dL_dT_batch[k:k+1], 
            problem,
            verbose=False
        )
        dL_dG_seq[k] = dL_dG_k[0]
        dL_dB_seq[k] = dL_dB_k[0]
    
    t_seq = time.perf_counter() - t0
    print(f"   Time: {t_seq*1000:.1f} ms")
    print(f"   dL/dG range: [{dL_dG_seq.min():.4e}, {dL_dG_seq.max():.4e}]")
    print(f"   dL/dB range: [{dL_dB_seq.min():.4e}, {dL_dB_seq.max():.4e}]")
    
    # Batched backward pass
    print("\n4. Running batched backward pass...")
    t0 = time.perf_counter()
    dL_dG_batch, dL_dB_batch = batched_backward_pass(
        T_batch, dL_dT_batch, G_batch, B_batch, source_points,
        verbose=True
    )
    t_batch = time.perf_counter() - t0
    print(f"   Total time: {t_batch*1000:.1f} ms")
    
    # Compare
    print("\n5. Comparing results...")
    print("-"*70)
    
    for k in range(K):
        # dL/dG comparison
        diff_G = (dL_dG_batch[k] - dL_dG_seq[k]).abs()
        max_diff_G = diff_G.max().item()
        
        # dL/dB comparison
        diff_B = (dL_dB_batch[k] - dL_dB_seq[k]).abs()
        max_diff_B = diff_B.max().item()
        
        status_G = "✓" if max_diff_G < 1e-5 else "✗"
        status_B = "✓" if max_diff_B < 1e-5 else "✗"
        
        print(f"   Fire {k}: dL/dG max_diff={max_diff_G:.2e} {status_G}, "
              f"dL/dB max_diff={max_diff_B:.2e} {status_B}")
    
    # Overall comparison
    total_diff_G = (dL_dG_batch - dL_dG_seq).abs().max().item()
    total_diff_B = (dL_dB_batch - dL_dB_seq).abs().max().item()
    
    print("-"*70)
    print(f"Overall max diff: dL/dG={total_diff_G:.2e}, dL/dB={total_diff_B:.2e}")
    
    if total_diff_G < 1e-5 and total_diff_B < 1e-5:
        print("\n✓ Batched backward pass matches sequential!")
    else:
        print("\n✗ Results differ - check implementation!")
    
    # Speedup
    print(f"\nSpeedup: {t_seq/t_batch:.2f}x ({t_seq*1000:.1f}ms → {t_batch*1000:.1f}ms)")
    
    return dL_dG_batch, dL_dB_batch


def benchmark_batched_backward():
    """Benchmark batched backward pass with varying batch sizes."""
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    
    print("="*70)
    print("Benchmarking Batched Backward Pass")
    print("="*70)
    
    M, N = 128, 128
    
    print(f"\nGrid size: {M}×{N}")
    print("-"*70)
    print(f"{'Batch K':>10} | {'Stencil ID':>12} | {'Adjoint':>12} | {'Param Grad':>12} | {'Total':>12}")
    print("-"*70)
    
    for K in [1, 4, 8, 16, 32]:
        # Create test data
        torch.manual_seed(42)
        G_batch = torch.ones(K, 3, M, N) * 0.35
        G_batch[:, 1] = 0
        B_batch = torch.zeros(K, 2, M, N)
        
        np.random.seed(42)
        source_points = [(np.random.randint(20, M-20), np.random.randint(20, N-20)) 
                         for _ in range(K)]
        
        # Forward solve
        T_batch = torch.zeros(K, M, N)
        for k in range(K):
            source_mask = torch.zeros(1, M, N, dtype=torch.bool)
            source_mask[0, source_points[k][0], source_points[k][1]] = True
            problem = EikonalProblem(G_batch[k:k+1], B_batch[k:k+1], source_mask)
            T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
            T_batch[k] = T[0]
        
        # Loss gradient
        dL_dT_batch = torch.randn(K, M, N) * 0.01
        dL_dT_batch[T_batch >= 1e5] = 0
        
        # Warmup
        batched_backward_pass(T_batch, dL_dT_batch, G_batch, B_batch, source_points)
        
        # Benchmark
        n_runs = 5
        times = {'stencil': [], 'adjoint': [], 'param': []}
        
        for _ in range(n_runs):
            source_mask = torch.zeros(K, M, N, dtype=torch.bool)
            for k, (si, sj) in enumerate(source_points):
                source_mask[k, si, sj] = True
            problem = BatchedProblem(G_batch, B_batch, source_mask)
            
            t0 = time.perf_counter()
            stencil_info = identify_active_stencils(T_batch, problem)
            times['stencil'].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            lam = solve_adjoint_numba(stencil_info, dL_dT_batch, problem, T_batch)
            times['adjoint'].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
            times['param'].append(time.perf_counter() - t0)
        
        t_stencil = np.mean(times['stencil']) * 1000
        t_adjoint = np.mean(times['adjoint']) * 1000
        t_param = np.mean(times['param']) * 1000
        t_total = t_stencil + t_adjoint + t_param
        
        print(f"{K:>10} | {t_stencil:>10.2f}ms | {t_adjoint:>10.2f}ms | "
              f"{t_param:>10.2f}ms | {t_total:>10.2f}ms")
    
    print("-"*70)


if __name__ == "__main__":
    # Run verification
    verify_batched_backward()
    
    print("\n")
    
    # Run benchmark
    benchmark_batched_backward()
