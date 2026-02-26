#!/usr/bin/env python3
"""
Benchmark: Forward Solver Performance Comparison
=================================================

Compares forward pass performance across implementations:
1. CPU (Numba JIT-compiled) - single problem at a time
2. GPU (CUDA batched) - multiple problems in parallel, normalized per problem

Reports timing for various grid sizes and batch sizes (8, 16, 32, 64).
Generates log-log plot similar to SGP paper Figure 38.

Usage:
    python benchmark_forward_solvers.py [--output figures/forward_benchmark.pdf]

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

# Import solvers
from eikonal.forward.pytorch import EikonalProblem
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba

# Check if CUDA is available
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        from eikonal.forward.cuda_batched import (
            fast_sweeping_solve_batched_v3, 
            warmup_v3 as warmup_gpu
        )
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA not available. GPU benchmarks will be skipped.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    grid_sizes: List[int]           # NxN grid sizes to test
    batch_sizes: List[int]          # Batch sizes for GPU
    max_iter: int                   # Sweeping iterations
    n_warmup: int                   # Warmup runs (not timed)
    n_trials: int                   # Timed trials for averaging
    threads_per_block: int          # CUDA threads


DEFAULT_CONFIG = BenchmarkConfig(
    grid_sizes=[50, 100, 150, 200, 250, 300, 400],
    batch_sizes=[8, 16, 32, 64],
    max_iter=30,
    n_warmup=2,
    n_trials=5,
    threads_per_block=128,
)


# =============================================================================
# Test Problem Creation
# =============================================================================

def create_test_problems(N: int, batch_size: int = 1, device: str = 'cpu'):
    """
    Create test problems with anisotropic metric and small drift.
    
    Returns numpy arrays for GPU solver and torch tensors for CPU solver.
    """
    np.random.seed(42)  # Reproducibility
    
    # Create batch of problems with varying anisotropy
    G_batch = np.zeros((batch_size, 3, N, N), dtype=np.float32)
    B_batch = np.zeros((batch_size, 2, N, N), dtype=np.float32)
    source_masks = np.zeros((batch_size, N, N), dtype=np.int8)
    
    for k in range(batch_size):
        # Anisotropic metric (rotated ellipse)
        angle = (k / batch_size) * np.pi / 2  # Varying rotation
        c, s = np.cos(angle), np.sin(angle)
        base_speed = 0.35  # Base propagation speed
        ratio = 1.5 + 0.5 * (k / batch_size)  # Anisotropy ratio
        
        # G = R^T * diag(a, b) * R
        G_batch[k, 0] = (c**2 * ratio + s**2) * base_speed  # g11
        G_batch[k, 1] = (c * s * (ratio - 1)) * base_speed   # g12
        G_batch[k, 2] = (s**2 * ratio + c**2) * base_speed   # g22
        
        # Small drift (wind effect)
        B_batch[k, 0] = 0.02 * np.cos(angle)
        B_batch[k, 1] = 0.02 * np.sin(angle)
        
        # 3x3 source mask at center
        ci, cj = N // 2, N // 2
        source_masks[k, ci-1:ci+2, cj-1:cj+2] = 1
    
    return G_batch, B_batch, source_masks


def create_eikonal_problem(G_np: np.ndarray, B_np: np.ndarray, 
                           source_mask_np: np.ndarray) -> EikonalProblem:
    """Convert numpy arrays to EikonalProblem for CPU solver."""
    G = torch.from_numpy(G_np).unsqueeze(0)  # (1, 3, N, N)
    B = torch.from_numpy(B_np).unsqueeze(0)  # (1, 2, N, N)
    source = torch.from_numpy(source_mask_np.astype(bool)).unsqueeze(0)  # (1, N, N)
    return EikonalProblem(G, B, source)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_cpu_numba(N: int, config: BenchmarkConfig) -> float:
    """
    Benchmark CPU Numba solver for a single NxN problem.
    
    Returns:
        Average time per problem in seconds
    """
    # Create single problem
    G_batch, B_batch, source_masks = create_test_problems(N, batch_size=1)
    problem = create_eikonal_problem(G_batch[0], B_batch[0], source_masks[0])
    
    # Warmup
    for _ in range(config.n_warmup):
        fast_sweeping_solve_numba(problem, max_iter=config.max_iter)
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        fast_sweeping_solve_numba(problem, max_iter=config.max_iter)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
    
    return np.median(times)


def benchmark_gpu_batched(N: int, batch_size: int, config: BenchmarkConfig) -> float:
    """
    Benchmark GPU batched solver for batch_size problems of size NxN.
    
    Returns:
        Average time per problem in seconds (total time / batch_size)
    """
    if not CUDA_AVAILABLE:
        return np.nan
    
    # Create batch of problems
    G_batch, B_batch, source_masks = create_test_problems(N, batch_size=batch_size)
    
    # Warmup
    for _ in range(config.n_warmup):
        fast_sweeping_solve_batched_v3(G_batch, B_batch, source_masks, 
                                       max_iter=config.max_iter,
                                       threads_per_block=config.threads_per_block)
    cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(config.n_trials):
        cuda.synchronize()
        t0 = time.perf_counter()
        fast_sweeping_solve_batched_v3(G_batch, B_batch, source_masks,
                                       max_iter=config.max_iter,
                                       threads_per_block=config.threads_per_block)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    # Return per-problem time
    return np.median(times) / batch_size


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
    print("Forward Solver Benchmark: CPU (Numba) vs GPU (Batched)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid sizes: {config.grid_sizes}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Max iterations: {config.max_iter}")
    print(f"  Trials per test: {config.n_trials}")
    print(f"  CUDA available: {CUDA_AVAILABLE}")
    print()
    
    # Warmup compilers
    print("Warming up JIT compilers...")
    warmup_numba()
    if CUDA_AVAILABLE:
        warmup_gpu()
    print("Done.\n")
    
    # Results storage
    results = {
        'grid_sizes': config.grid_sizes,
        'batch_sizes': config.batch_sizes,
        'cpu_numba': [],
        'gpu_batched': {bs: [] for bs in config.batch_sizes},
    }
    
    # Header
    header = f"{'N':>6} | {'CPU (Numba)':>12}"
    for bs in config.batch_sizes:
        header += f" | {'GPU B='+str(bs):>12}"
    print(header)
    print("-" * len(header))
    
    for N in config.grid_sizes:
        row = f"{N:>6}"
        
        # CPU Numba
        t_cpu = benchmark_cpu_numba(N, config)
        results['cpu_numba'].append(t_cpu)
        row += f" | {t_cpu*1000:>10.2f}ms"
        
        # GPU batched with different batch sizes
        for bs in config.batch_sizes:
            t_gpu = benchmark_gpu_batched(N, bs, config)
            results['gpu_batched'][bs].append(t_gpu)
            if np.isnan(t_gpu):
                row += f" | {'N/A':>12}"
            else:
                row += f" | {t_gpu*1000:>10.3f}ms"
        
        print(row)
    
    print()
    
    # Compute speedups
    if CUDA_AVAILABLE:
        print("\nSpeedups (CPU / GPU):")
        header = f"{'N':>6}"
        for bs in config.batch_sizes:
            header += f" | {'B='+str(bs):>10}"
        print(header)
        print("-" * len(header))
        
        for i, N in enumerate(config.grid_sizes):
            row = f"{N:>6}"
            for bs in config.batch_sizes:
                t_cpu = results['cpu_numba'][i]
                t_gpu = results['gpu_batched'][bs][i]
                if not np.isnan(t_gpu) and t_gpu > 0:
                    speedup = t_cpu / t_gpu
                    row += f" | {speedup:>9.1f}x"
                else:
                    row += f" | {'N/A':>10}"
            print(row)
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_benchmark_results(results: Dict, output_path: str = None, show: bool = True):
    """
    Create publication-quality log-log plot of benchmark results.
    
    Two panels:
    - Left: Runtime comparison (log-log scale)
    - Right: Speedup vs grid size
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    cpu_times = np.array(results['cpu_numba']) * 1000  # Convert to ms
    
    # Colors for different batch sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results['batch_sizes'])))
    
    # =================================
    # Left panel: Runtime comparison
    # =================================
    ax1 = axes[0]
    
    # CPU line
    ax1.loglog(grid_sizes, cpu_times, 'o-', color='#E74C3C', 
               linewidth=2, markersize=8, label='CPU (Numba)')
    
    # GPU lines for each batch size
    for bs, color in zip(results['batch_sizes'], colors):
        gpu_times = np.array(results['gpu_batched'][bs]) * 1000
        valid = ~np.isnan(gpu_times)
        if valid.any():
            ax1.loglog(grid_sizes[valid], gpu_times[valid], 's--', 
                      color=color, linewidth=2, markersize=7,
                      label=f'GPU (B={bs})')
    
    # Reference O(N²) line
    N_ref = grid_sizes
    scale = cpu_times[len(cpu_times)//2] / (grid_sizes[len(grid_sizes)//2]**2)
    ax1.loglog(N_ref, scale * N_ref**2, ':', color='gray', linewidth=1.5, 
               alpha=0.7, label='$O(N^2)$')
    
    ax1.set_xlabel('Grid Size $N$', fontsize=12)
    ax1.set_ylabel('Time per Problem (ms)', fontsize=12)
    ax1.set_title('Runtime Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='-')
    ax1.set_xlim(grid_sizes.min() * 0.8, grid_sizes.max() * 1.2)
    
    # =================================
    # Right panel: Speedup
    # =================================
    ax2 = axes[1]
    
    for bs, color in zip(results['batch_sizes'], colors):
        gpu_times = np.array(results['gpu_batched'][bs])
        valid = ~np.isnan(gpu_times) & (gpu_times > 0)
        if valid.any():
            speedup = cpu_times[valid] / (gpu_times[valid] * 1000)
            ax2.bar(np.arange(len(grid_sizes))[valid] + 0.15 * (results['batch_sizes'].index(bs) - 1.5),
                   speedup, width=0.15, color=color, edgecolor='black', 
                   linewidth=0.5, label=f'B={bs}')
    
    ax2.set_xticks(np.arange(len(grid_sizes)))
    ax2.set_xticklabels([str(n) for n in grid_sizes])
    ax2.set_xlabel('Grid Size $N$', fontsize=12)
    ax2.set_ylabel('Speedup (CPU / GPU)', fontsize=12)
    ax2.set_title('GPU Speedup', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
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
    Create single-panel log-log runtime plot (matching the reference image style).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    cpu_times = np.array(results['cpu_numba']) * 1000  # ms
    
    # CPU line (blue with circles like the reference)
    ax.loglog(grid_sizes, cpu_times, 'o-', color='#1f77b4', 
              linewidth=2.5, markersize=9, label='CPU (Numba)', zorder=3)
    
    # GPU lines (orange/yellow tones)
    gpu_colors = ['#ff7f0e', '#d62728', '#9467bd', '#2ca02c']
    markers = ['s', '^', 'D', 'v']
    
    for i, bs in enumerate(results['batch_sizes']):
        gpu_times = np.array(results['gpu_batched'][bs]) * 1000
        valid = ~np.isnan(gpu_times)
        if valid.any():
            ax.loglog(grid_sizes[valid], gpu_times[valid], 
                     marker=markers[i % len(markers)], linestyle='--',
                     color=gpu_colors[i % len(gpu_colors)], 
                     linewidth=2, markersize=7,
                     label=f'GPU (B={bs})')
    
    # O(N²) reference
    N_mid = grid_sizes[len(grid_sizes)//2]
    t_mid = cpu_times[len(cpu_times)//2]
    scale = t_mid / (N_mid**2)
    ax.loglog(grid_sizes, scale * grid_sizes**2, ':', color='gray', 
              linewidth=2, alpha=0.7, label='$O(N^2)$')
    
    ax.set_xlabel('Grid Size $N$', fontsize=13)
    ax.set_ylabel('Time (ms)', fontsize=13)
    ax.set_title('Forward Solver: Runtime Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3)
    
    # Nice tick formatting
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
    parser = argparse.ArgumentParser(description='Benchmark forward solvers')
    parser.add_argument('--output', type=str, default='forward_solver_benchmark.png',
                       help='Output figure path')
    parser.add_argument('--grid-sizes', type=int, nargs='+', 
                       default=[50, 100, 150, 200, 250, 300, 400],
                       help='Grid sizes to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[8, 16, 32, 64],
                       help='Batch sizes for GPU')
    parser.add_argument('--max-iter', type=int, default=30,
                       help='Sweeping iterations')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of timed trials')
    parser.add_argument('--single-panel', action='store_true',
                       help='Create single panel plot (like reference)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        grid_sizes=args.grid_sizes,
        batch_sizes=args.batch_sizes,
        max_iter=args.max_iter,
        n_warmup=2,
        n_trials=args.trials,
        threads_per_block=128,
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
    np.savez(results_path,
             grid_sizes=results['grid_sizes'],
             batch_sizes=results['batch_sizes'],
             cpu_numba=results['cpu_numba'],
             **{f'gpu_b{bs}': results['gpu_batched'][bs] for bs in results['batch_sizes']})
    print(f"Raw data saved to: {results_path}")


if __name__ == '__main__':
    main()
