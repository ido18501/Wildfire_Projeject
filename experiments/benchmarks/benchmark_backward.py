#!/usr/bin/env python3
"""
Benchmark: Backward Pass Performance Comparison
================================================

Compares backward pass performance across implementations:
1. Original Python adjoint solve (solve_adjoint_direct)
2. Numba JIT adjoint solve (solve_adjoint_numba)
3. Numba sparse adjoint solve (solve_adjoint_numba_sparse)

Also benchmarks the complete backward pass (stencil ID + adjoint + param gradients).

Reports timing for various grid sizes.
Generates log-log plot similar to forward solver benchmark.

Usage:
    python benchmark_backward_solvers.py [--output figures/backward_benchmark.pdf]

Author: Barak Gahtan
Date: February 2025
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add project path (script should be placed in /path/to/project/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import forward solver
from eikonal.forward.pytorch import EikonalProblem
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_forward

# Import backward pass components
from eikonal.backward.implicit_diff import (
    identify_active_stencils,
    solve_adjoint_direct,
    compute_parameter_gradients,
    backward_pass,
)

# Import optimized backward solvers
from eikonal.backward.adjoint_numba import solve_adjoint_numba
from eikonal.backward.adjoint_sparse import (
    solve_adjoint_numba_sparse,
    warmup_sparse_adjoint,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    grid_sizes: List[int]           # NxN grid sizes to test
    batch_sizes: List[int]          # Batch sizes to test (K fires)
    max_iter: int                   # Forward sweeping iterations
    n_warmup: int                   # Warmup runs (not timed)
    n_trials: int                   # Timed trials for averaging
    skip_original: bool             # Skip slow original Python implementation


DEFAULT_CONFIG = BenchmarkConfig(
    grid_sizes=[50, 100, 150, 200, 250, 300],
    batch_sizes=[1, 8, 16, 32, 64],
    max_iter=30,
    n_warmup=2,
    n_trials=5,
    skip_original=False,  # Include original Python for all grids
)


# =============================================================================
# Test Problem Creation
# =============================================================================

def create_test_problem(N: int):
    """
    Create a test problem with anisotropic metric and small drift.
    
    Returns:
        problem: EikonalProblem
        T: Converged arrival times
        dL_dT: Loss gradient (MSE-style)
    """
    # Anisotropic metric (rotated ellipse)
    angle = np.pi / 6  # 30 degrees
    c, s = np.cos(angle), np.sin(angle)
    base_speed = 0.35
    ratio = 1.8  # Anisotropy
    
    G = torch.zeros(1, 3, N, N)
    G[0, 0] = (c**2 * ratio + s**2) * base_speed  # g11
    G[0, 1] = (c * s * (ratio - 1)) * base_speed   # g12
    G[0, 2] = (s**2 * ratio + c**2) * base_speed   # g22
    
    # Small drift
    B = torch.zeros(1, 2, N, N)
    B[0, 0] = 0.03
    B[0, 1] = 0.02
    
    # 3x3 source at center
    source = torch.zeros(1, N, N, dtype=torch.bool)
    ci, cj = N // 2, N // 2
    source[0, ci-1:ci+2, cj-1:cj+2] = True
    
    problem = EikonalProblem(G, B, source)
    
    # Forward solve
    T, _ = fast_sweeping_solve_numba(problem, max_iter=30)
    
    # Create loss gradient (pretend MSE loss at random observation points)
    torch.manual_seed(42)
    obs_mask = torch.rand(N, N) < 0.3
    obs_mask &= (T[0] < 1e5)
    obs_mask &= ~source[0]  # Exclude source
    
    T_target = T[0] + torch.randn(N, N) * 2
    
    dL_dT = torch.zeros_like(T)
    dL_dT[0, obs_mask] = (T[0, obs_mask] - T_target[obs_mask]) * 0.01
    
    return problem, T, dL_dT


def create_batched_test_problem(N: int, K: int):
    """
    Create a batched test problem with K fires.
    
    Returns:
        problem: EikonalProblem with K fires
        T: Converged arrival times (K, N, N)
        dL_dT: Loss gradients (K, N, N)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    G_batch = torch.zeros(K, 3, N, N)
    B_batch = torch.zeros(K, 2, N, N)
    source_batch = torch.zeros(K, N, N, dtype=torch.bool)
    
    for k in range(K):
        # Varying anisotropic metric
        angle = (k / K) * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        base_speed = 0.3 + 0.1 * (k / K)
        ratio = 1.5 + 0.5 * (k / K)
        
        G_batch[k, 0] = (c**2 * ratio + s**2) * base_speed
        G_batch[k, 1] = (c * s * (ratio - 1)) * base_speed
        G_batch[k, 2] = (s**2 * ratio + c**2) * base_speed
        
        # Small varying drift
        B_batch[k, 0] = 0.02 * np.cos(angle)
        B_batch[k, 1] = 0.02 * np.sin(angle)
        
        # 3x3 source at center
        ci, cj = N // 2, N // 2
        source_batch[k, ci-1:ci+2, cj-1:cj+2] = True
    
    problem = EikonalProblem(G_batch, B_batch, source_batch)
    
    # Forward solve each fire
    T_batch = torch.zeros(K, N, N)
    for k in range(K):
        prob_k = EikonalProblem(G_batch[k:k+1], B_batch[k:k+1], source_batch[k:k+1])
        T_k, _ = fast_sweeping_solve_numba(prob_k, max_iter=30)
        T_batch[k] = T_k[0]
    
    # Create loss gradients
    dL_dT_batch = torch.zeros(K, N, N)
    for k in range(K):
        obs_mask = torch.rand(N, N) < 0.3
        obs_mask &= (T_batch[k] < 1e5)
        obs_mask &= ~source_batch[k]
        
        T_target = T_batch[k] + torch.randn(N, N) * 2
        dL_dT_batch[k, obs_mask] = (T_batch[k, obs_mask] - T_target[obs_mask]) * 0.01
    
    return problem, T_batch, dL_dT_batch


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_adjoint_original(stencil_info, dL_dT, problem, T, config: BenchmarkConfig) -> float:
    """Benchmark original Python adjoint solve."""
    # Warmup
    for _ in range(config.n_warmup):
        solve_adjoint_direct(stencil_info, dL_dT, problem, T)
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        t0 = time.perf_counter()
        solve_adjoint_direct(stencil_info, dL_dT, problem, T)
        times.append(time.perf_counter() - t0)
    
    return np.median(times)


def benchmark_adjoint_numba(stencil_info, dL_dT, problem, T, config: BenchmarkConfig) -> float:
    """Benchmark Numba JIT adjoint solve."""
    # Warmup
    for _ in range(config.n_warmup):
        solve_adjoint_numba(stencil_info, dL_dT, problem, T)
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        t0 = time.perf_counter()
        solve_adjoint_numba(stencil_info, dL_dT, problem, T)
        times.append(time.perf_counter() - t0)
    
    return np.median(times)


def benchmark_adjoint_numba_sparse(stencil_info, dL_dT, problem, T, config: BenchmarkConfig) -> float:
    """Benchmark Numba sparse adjoint solve."""
    # Warmup
    for _ in range(config.n_warmup):
        solve_adjoint_numba_sparse(stencil_info, dL_dT, problem, T)
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        t0 = time.perf_counter()
        solve_adjoint_numba_sparse(stencil_info, dL_dT, problem, T)
        times.append(time.perf_counter() - t0)
    
    return np.median(times)


def benchmark_full_backward(problem, T, dL_dT, adjoint_fn, config: BenchmarkConfig) -> Tuple[float, float, float]:
    """
    Benchmark full backward pass with specified adjoint solver.
    
    Returns:
        (t_stencil, t_adjoint, t_param) - time for each step in seconds
    """
    times = {'stencil': [], 'adjoint': [], 'param': []}
    
    for _ in range(config.n_trials):
        # Step 1: Identify stencils
        t0 = time.perf_counter()
        stencil_info = identify_active_stencils(T, problem)
        times['stencil'].append(time.perf_counter() - t0)
        
        # Step 4: Adjoint solve
        t0 = time.perf_counter()
        lam = adjoint_fn(stencil_info, dL_dT, problem, T)
        times['adjoint'].append(time.perf_counter() - t0)
        
        # Step 5: Parameter gradients
        t0 = time.perf_counter()
        dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
        times['param'].append(time.perf_counter() - t0)
    
    return (np.median(times['stencil']), 
            np.median(times['adjoint']), 
            np.median(times['param']))


def benchmark_batched_adjoint(N: int, K: int, config: BenchmarkConfig) -> float:
    """
    Benchmark batched adjoint solve for K fires of size NxN.
    
    Returns:
        Time per fire in seconds (total_time / K)
    """
    # Create batched problem
    problem, T_batch, dL_dT_batch = create_batched_test_problem(N, K)
    
    # Warmup
    stencil_info = identify_active_stencils(T_batch, problem)
    for _ in range(config.n_warmup):
        solve_adjoint_numba_sparse(stencil_info, dL_dT_batch, problem, T_batch)
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        stencil_info = identify_active_stencils(T_batch, problem)
        t0 = time.perf_counter()
        solve_adjoint_numba_sparse(stencil_info, dL_dT_batch, problem, T_batch)
        times.append(time.perf_counter() - t0)
    
    # Return per-fire time
    return np.median(times) / K


def benchmark_batched_full_backward(N: int, K: int, config: BenchmarkConfig) -> float:
    """
    Benchmark full batched backward pass for K fires of size NxN.
    
    Returns:
        Time per fire in seconds (total_time / K)
    """
    # Create batched problem
    problem, T_batch, dL_dT_batch = create_batched_test_problem(N, K)
    
    # Warmup
    for _ in range(config.n_warmup):
        stencil_info = identify_active_stencils(T_batch, problem)
        lam = solve_adjoint_numba_sparse(stencil_info, dL_dT_batch, problem, T_batch)
        dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        t0 = time.perf_counter()
        stencil_info = identify_active_stencils(T_batch, problem)
        lam = solve_adjoint_numba_sparse(stencil_info, dL_dT_batch, problem, T_batch)
        dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
        times.append(time.perf_counter() - t0)
    
    # Return per-fire time
    return np.median(times) / K


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(config: BenchmarkConfig = None) -> Dict:
    """
    Run full benchmark suite.
    
    Returns:
        Dictionary with timing results
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("=" * 70)
    print("Backward Pass Benchmark: Adjoint Solve Implementations")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid sizes: {config.grid_sizes}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Trials per test: {config.n_trials}")
    print(f"  Skip original Python: {config.skip_original}")
    if not config.skip_original:
        print(f"\n  WARNING: Original Python is ~4000x slower than Numba.")
        print(f"           For N=300, expect ~30-60 seconds per trial.")
    print()
    
    # Warmup compilers
    print("Warming up JIT compilers...")
    warmup_forward()
    warmup_sparse_adjoint()
    
    # Warmup numba adjoint with small problem
    print("Warming up Numba adjoint...")
    problem_warmup, T_warmup, dL_dT_warmup = create_test_problem(32)
    stencil_warmup = identify_active_stencils(T_warmup, problem_warmup)
    solve_adjoint_numba(stencil_warmup, dL_dT_warmup, problem_warmup, T_warmup)
    solve_adjoint_numba_sparse(stencil_warmup, dL_dT_warmup, problem_warmup, T_warmup)
    print("Done.\n")
    
    # Results storage
    results = {
        'grid_sizes': config.grid_sizes,
        'batch_sizes': config.batch_sizes,
        'adjoint_original': [],
        'adjoint_numba': [],
        'adjoint_sparse': [],
        'full_stencil': [],
        'full_adjoint': [],
        'full_param': [],
        'batched_adjoint': {bs: [] for bs in config.batch_sizes},
        'batched_full': {bs: [] for bs in config.batch_sizes},
    }
    
    # Header for adjoint comparison (single fire)
    print("=" * 70)
    print("ADJOINT SOLVE COMPARISON (Single Fire, K=1)")
    print("=" * 70)
    if config.skip_original:
        header = f"{'N':>6} | {'Numba':>12} | {'Sparse':>12} | {'Speedup':>10}"
    else:
        header = f"{'N':>6} | {'Original':>12} | {'Numba':>12} | {'Sparse':>12} | {'Speedup':>10}"
    print(header)
    print("-" * len(header))
    
    for N in config.grid_sizes:
        # Create test problem
        problem, T, dL_dT = create_test_problem(N)
        stencil_info = identify_active_stencils(T, problem)
        
        row = f"{N:>6}"
        
        # Original Python
        if config.skip_original:
            t_orig = np.nan
            results['adjoint_original'].append(np.nan)
        else:
            t_orig = benchmark_adjoint_original(stencil_info, dL_dT, problem, T, config)
            results['adjoint_original'].append(t_orig)
            row += f" | {t_orig*1000:>10.1f}ms"
        
        # Numba
        t_numba = benchmark_adjoint_numba(stencil_info, dL_dT, problem, T, config)
        results['adjoint_numba'].append(t_numba)
        row += f" | {t_numba*1000:>10.2f}ms"
        
        # Sparse
        t_sparse = benchmark_adjoint_numba_sparse(stencil_info, dL_dT, problem, T, config)
        results['adjoint_sparse'].append(t_sparse)
        row += f" | {t_sparse*1000:>10.2f}ms"
        
        # Speedup (Original vs Numba)
        if not np.isnan(t_orig):
            speedup = t_orig / t_numba
            row += f" | {speedup:>8.0f}x"
        else:
            speedup = t_numba / t_sparse
            row += f" | {speedup:>8.1f}x (N/S)"
        
        print(row)
    
    # Batched backward pass (normalized per fire)
    print("\n" + "=" * 70)
    print("BATCHED BACKWARD PASS (Time per fire, using Numba Sparse)")
    print("=" * 70)
    header = f"{'N':>6}"
    for bs in config.batch_sizes:
        header += f" | {'K='+str(bs):>10}"
    print(header)
    print("-" * len(header))
    
    for N in config.grid_sizes:
        row = f"{N:>6}"
        for bs in config.batch_sizes:
            t_per_fire = benchmark_batched_full_backward(N, bs, config)
            results['batched_full'][bs].append(t_per_fire)
            row += f" | {t_per_fire*1000:>8.2f}ms"
        print(row)
    
    # Full backward pass breakdown (single fire)
    print("\n" + "=" * 70)
    print("FULL BACKWARD PASS BREAKDOWN (K=1, using Numba sparse)")
    print("=" * 70)
    header = f"{'N':>6} | {'Stencil ID':>12} | {'Adjoint':>12} | {'Param Grad':>12} | {'Total':>12}"
    print(header)
    print("-" * len(header))
    
    for N in config.grid_sizes:
        problem, T, dL_dT = create_test_problem(N)
        
        t_stencil, t_adjoint, t_param = benchmark_full_backward(
            problem, T, dL_dT, solve_adjoint_numba_sparse, config
        )
        
        results['full_stencil'].append(t_stencil)
        results['full_adjoint'].append(t_adjoint)
        results['full_param'].append(t_param)
        
        t_total = t_stencil + t_adjoint + t_param
        
        print(f"{N:>6} | {t_stencil*1000:>10.2f}ms | {t_adjoint*1000:>10.2f}ms | "
              f"{t_param*1000:>10.2f}ms | {t_total*1000:>10.2f}ms")
    
    print()
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_benchmark_results(results: Dict, output_path: str = None, show: bool = True):
    """
    Create publication-quality plot of benchmark results.
    
    Three panels:
    - Left: Adjoint solve comparison (log-log)
    - Middle: Batched backward pass per-fire time
    - Right: Full backward pass breakdown (stacked bar)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    batch_sizes = results['batch_sizes']
    
    # =================================
    # Left panel: Adjoint solve comparison
    # =================================
    ax1 = axes[0]
    
    # Original Python (if available)
    t_orig = np.array(results['adjoint_original']) * 1000
    valid_orig = ~np.isnan(t_orig)
    if valid_orig.any():
        ax1.loglog(grid_sizes[valid_orig], t_orig[valid_orig], 'o-', 
                  color='#E74C3C', linewidth=2, markersize=8, 
                  label='Original Python')
    
    # Numba
    t_numba = np.array(results['adjoint_numba']) * 1000
    ax1.loglog(grid_sizes, t_numba, 's-', color='#3498DB', 
              linewidth=2, markersize=8, label='Numba JIT')
    
    # Sparse
    t_sparse = np.array(results['adjoint_sparse']) * 1000
    ax1.loglog(grid_sizes, t_sparse, '^-', color='#2ECC71', 
              linewidth=2, markersize=8, label='Numba Sparse')
    
    # O(N²) reference
    N_ref = grid_sizes
    scale = t_numba[len(t_numba)//2] / (grid_sizes[len(grid_sizes)//2]**2)
    ax1.loglog(N_ref, scale * N_ref**2, ':', color='gray', linewidth=1.5, 
               alpha=0.7, label='$O(N^2)$')
    
    ax1.set_xlabel('Grid Size $N$', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Adjoint Solve (K=1)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)
    
    # =================================
    # Middle panel: Batched backward (per-fire time)
    # =================================
    ax2 = axes[1]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(batch_sizes)))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, bs in enumerate(batch_sizes):
        t_batched = np.array(results['batched_full'][bs]) * 1000
        ax2.loglog(grid_sizes, t_batched, marker=markers[i % len(markers)],
                  linestyle='-', color=colors[i], linewidth=2, markersize=7,
                  label=f'K={bs}')
    
    ax2.set_xlabel('Grid Size $N$', fontsize=12)
    ax2.set_ylabel('Time per Fire (ms)', fontsize=12)
    ax2.set_title('Batched Backward Pass', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)
    
    # =================================
    # Right panel: Full backward breakdown
    # =================================
    ax3 = axes[2]
    
    t_stencil = np.array(results['full_stencil']) * 1000
    t_adjoint = np.array(results['full_adjoint']) * 1000
    t_param = np.array(results['full_param']) * 1000
    
    x = np.arange(len(grid_sizes))
    width = 0.6
    
    ax3.bar(x, t_stencil, width, label='Stencil ID', color='#3498DB')
    ax3.bar(x, t_adjoint, width, bottom=t_stencil, label='Adjoint Solve', color='#E74C3C')
    ax3.bar(x, t_param, width, bottom=t_stencil + t_adjoint, 
           label='Param Gradients', color='#2ECC71')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(n) for n in grid_sizes])
    ax3.set_xlabel('Grid Size $N$', fontsize=12)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Backward Pass Breakdown (K=1)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"\nFigure saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_single_panel(results: Dict, output_path: str = None):
    """
    Create single-panel log-log runtime plot showing batched backward pass.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    batch_sizes = results['batch_sizes']
    
    # Colors for different batch sizes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, bs in enumerate(batch_sizes):
        t_batched = np.array(results['batched_full'][bs]) * 1000
        ax.loglog(grid_sizes, t_batched, marker=markers[i % len(markers)],
                 linestyle='-', color=colors[i % len(colors)], 
                 linewidth=2.5, markersize=8, label=f'K={bs}')
    
    # O(N²) reference
    t_ref = np.array(results['batched_full'][1]) * 1000
    N_mid = grid_sizes[len(grid_sizes)//2]
    t_mid = t_ref[len(t_ref)//2]
    scale = t_mid / (N_mid**2)
    ax.loglog(grid_sizes, scale * grid_sizes**2, ':', color='gray', 
              linewidth=2, alpha=0.7, label='$O(N^2)$')
    
    ax.set_xlabel('Grid Size $N$', fontsize=13)
    ax.set_ylabel('Time per Fire (ms)', fontsize=13)
    ax.set_title('Backward Pass: Batched Runtime', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"\nFigure saved to: {output_path}")
    
    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark backward pass solvers')
    parser.add_argument('--output', type=str, default='backward_solver_benchmark.pdf',
                       help='Output figure path')
    parser.add_argument('--grid-sizes', type=int, nargs='+', 
                       default=[50, 100, 150, 200, 250, 300],
                       help='Grid sizes to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[1, 8, 16, 32, 64],
                       help='Batch sizes for batched backward pass')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of timed trials')
    parser.add_argument('--skip-original', action='store_true',
                       help='Skip slow original Python implementation')
    parser.add_argument('--single-panel', action='store_true',
                       help='Create single panel plot')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        grid_sizes=args.grid_sizes,
        batch_sizes=args.batch_sizes,
        max_iter=30,
        n_warmup=2,
        n_trials=args.trials,
        skip_original=args.skip_original,
    )
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Plot
    if args.single_panel:
        plot_single_panel(results, args.output)
    else:
        plot_benchmark_results(results, args.output, show=not args.no_show)
    
    # Save raw results
    results_path = args.output.rsplit('.', 1)[0] + '_data.npz'
    save_dict = {
        'grid_sizes': results['grid_sizes'],
        'batch_sizes': results['batch_sizes'],
        'adjoint_original': results['adjoint_original'],
        'adjoint_numba': results['adjoint_numba'],
        'adjoint_sparse': results['adjoint_sparse'],
        'full_stencil': results['full_stencil'],
        'full_adjoint': results['full_adjoint'],
        'full_param': results['full_param'],
    }
    # Add batched results
    for bs in results['batch_sizes']:
        save_dict[f'batched_full_K{bs}'] = results['batched_full'][bs]
    
    np.savez(results_path, **save_dict)
    print(f"Raw data saved to: {results_path}")


if __name__ == '__main__':
    main()
