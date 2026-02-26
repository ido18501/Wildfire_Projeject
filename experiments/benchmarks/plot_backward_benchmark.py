#!/usr/bin/env python3
"""
Plot Backward Benchmark Results from Saved Data
================================================

Loads existing benchmark data from .npz file and generates publication-quality plots.

Usage:
    python plot_backward_benchmark.py --input backward_solver_benchmark_data.npz --output figures/backward_benchmark.pdf
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_benchmark_data(npz_path: str) -> dict:
    """Load benchmark results from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    
    results = {
        'grid_sizes': data['grid_sizes'],
        'adjoint_original': data['adjoint_original'],
        'adjoint_numba': data['adjoint_numba'],
        'adjoint_sparse': data['adjoint_sparse'],
        'full_stencil': data['full_stencil'],
        'full_adjoint': data['full_adjoint'],
        'full_param': data['full_param'],
    }
    
    # Load batch sizes if available
    if 'batch_sizes' in data:
        results['batch_sizes'] = data['batch_sizes']
        results['batched_full'] = {}
        for bs in results['batch_sizes']:
            key = f'batched_full_B{bs}'
            if key not in data:
                key = f'batched_full_K{bs}'  # Old naming convention
            if key in data:
                results['batched_full'][bs] = data[key]
    
    return results


def plot_benchmark_results(results: dict, output_path: str = None, show: bool = True):
    """
    Create publication-quality plot of benchmark results.
    
    Three panels:
    - Left: Adjoint solve comparison (log-log)
    - Middle: Batched backward pass per-fire time
    - Right: Full backward pass breakdown (stacked bar)
    """
    has_batched = 'batched_full' in results and len(results.get('batched_full', {})) > 0
    
    if has_batched:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    
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
    ax1.set_ylabel('Time per Problem (ms)', fontsize=12)
    # ax1.set_title('Adjoint Solve (B=1)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)
    
    # =================================
    # Middle panel: Batched backward (per-fire time)
    # =================================
    if has_batched:
        ax2 = axes[1]
        batch_sizes = list(results['batched_full'].keys())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(batch_sizes)))
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, bs in enumerate(batch_sizes):
            t_batched = np.array(results['batched_full'][bs]) * 1000
            ax2.loglog(grid_sizes, t_batched, marker=markers[i % len(markers)],
                      linestyle='-', color=colors[i], linewidth=2, markersize=7,
                      label=f'B={bs}')
        
        ax2.set_xlabel('Grid Size $N$', fontsize=12)
        ax2.set_ylabel('Time per Problem (ms)', fontsize=12)
        # ax2.set_title('Batched Backward Pass', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, which='both', alpha=0.3)
        
        ax3 = axes[2]
    else:
        ax3 = axes[1]
    
    # =================================
    # Right panel: Full backward breakdown
    # =================================
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
    ax3.set_xticklabels([str(int(n)) for n in grid_sizes])
    ax3.set_xlabel('Grid Size $N$', fontsize=12)
    ax3.set_ylabel('Time per Problem (ms)', fontsize=12)
    # ax3.set_title('Backward Pass Breakdown (B=1)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_single_panel(results: dict, output_path: str = None, show: bool = True):
    """
    Create single-panel log-log runtime plot showing batched backward pass.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    
    if 'batched_full' in results and len(results['batched_full']) > 0:
        batch_sizes = list(results['batched_full'].keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, bs in enumerate(batch_sizes):
            t_batched = np.array(results['batched_full'][bs]) * 1000
            ax.loglog(grid_sizes, t_batched, marker=markers[i % len(markers)],
                     linestyle='-', color=colors[i % len(colors)], 
                     linewidth=2.5, markersize=8, label=f'B={bs}')
        
        # O(N²) reference
        t_ref = np.array(results['batched_full'][batch_sizes[0]]) * 1000
    else:
        # Fall back to adjoint solve comparison
        t_numba = np.array(results['adjoint_numba']) * 1000
        ax.loglog(grid_sizes, t_numba, 's-', color='#1f77b4', 
                 linewidth=2.5, markersize=8, label='Numba JIT')
        
        t_sparse = np.array(results['adjoint_sparse']) * 1000
        ax.loglog(grid_sizes, t_sparse, '^-', color='#2ca02c', 
                 linewidth=2.5, markersize=8, label='Numba Sparse')
        
        t_ref = t_numba
    
    # O(N²) reference
    N_mid = grid_sizes[len(grid_sizes)//2]
    t_mid = t_ref[len(t_ref)//2]
    scale = t_mid / (N_mid**2)
    ax.loglog(grid_sizes, scale * grid_sizes**2, ':', color='gray', 
              linewidth=2, alpha=0.7, label='$O(N^2)$')
    
    ax.set_xlabel('Grid Size $N$', fontsize=13)
    ax.set_ylabel('Time per Problem (ms)', fontsize=13)
    ax.set_title('Backward Pass: Runtime Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def print_summary(results: dict):
    """Print summary of benchmark results."""
    grid_sizes = results['grid_sizes']
    
    print("=" * 70)
    print("Backward Benchmark Summary (from saved data)")
    print("=" * 70)
    
    print(f"\nGrid sizes: {list(grid_sizes)}")
    
    print("\nAdjoint Solve Times (ms):")
    print(f"{'N':>6} | {'Original':>12} | {'Numba':>12} | {'Sparse':>12} | {'Speedup':>10}")
    print("-" * 60)
    
    for i, N in enumerate(grid_sizes):
        t_orig = results['adjoint_original'][i] * 1000
        t_numba = results['adjoint_numba'][i] * 1000
        t_sparse = results['adjoint_sparse'][i] * 1000
        
        if not np.isnan(t_orig):
            speedup = t_orig / t_numba
            print(f"{int(N):>6} | {t_orig:>10.1f}ms | {t_numba:>10.2f}ms | {t_sparse:>10.2f}ms | {speedup:>8.0f}x")
        else:
            print(f"{int(N):>6} | {'N/A':>12} | {t_numba:>10.2f}ms | {t_sparse:>10.2f}ms |")
    
    if 'batched_full' in results and len(results['batched_full']) > 0:
        print("\nBatched Backward Pass (ms per fire):")
        batch_sizes = list(results['batched_full'].keys())
        header = f"{'N':>6}"
        for bs in batch_sizes:
            header += f" | {'B='+str(bs):>10}"
        print(header)
        print("-" * len(header))
        
        for i, N in enumerate(grid_sizes):
            row = f"{int(N):>6}"
            for bs in batch_sizes:
                t = results['batched_full'][bs][i] * 1000
                row += f" | {t:>8.2f}ms"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Plot backward benchmark from saved data')
    parser.add_argument('--input', type=str, default='backward_solver_benchmark_data.npz',
                       help='Input .npz file with benchmark data')
    parser.add_argument('--output', type=str, default='backward_solver_benchmark.pdf',
                       help='Output figure path')
    parser.add_argument('--single-panel', action='store_true',
                       help='Create single panel plot')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary of results')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    results = load_benchmark_data(args.input)
    
    if args.summary:
        print_summary(results)
    
    # Plot
    if args.single_panel:
        plot_single_panel(results, args.output, show=not args.no_show)
    else:
        plot_benchmark_results(results, args.output, show=not args.no_show)


if __name__ == '__main__':
    main()
