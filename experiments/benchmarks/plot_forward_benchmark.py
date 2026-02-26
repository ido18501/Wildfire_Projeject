#!/usr/bin/env python3
"""
Plot Forward Benchmark Results from Saved Data
===============================================

Loads existing benchmark data from .npz file and generates publication-quality plots.

Usage:
    python plot_forward_benchmark.py --input forward_solver_benchmark_data.npz --output figures/forward_benchmark.pdf
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
        'cpu_numba': data['cpu_numba'],
    }
    
    # Load batch sizes
    if 'batch_sizes' in data:
        results['batch_sizes'] = list(data['batch_sizes'])
    else:
        # Try to infer from keys
        results['batch_sizes'] = []
        for key in data.files:
            if key.startswith('gpu_b') or key.startswith('gpu_B'):
                bs = int(key.split('b')[-1].split('B')[-1])
                results['batch_sizes'].append(bs)
        results['batch_sizes'] = sorted(results['batch_sizes'])
    
    results['gpu_batched'] = {}
    for bs in results['batch_sizes']:
        for key_pattern in [f'gpu_b{bs}', f'gpu_B{bs}']:
            if key_pattern in data:
                results['gpu_batched'][bs] = data[key_pattern]
                break
    
    return results


def plot_benchmark_results(results: dict, output_path: str = None, show: bool = True):
    """
    Create publication-quality log-log plot of benchmark results.
    
    Two panels:
    - Left: Runtime comparison (log-log scale)
    - Right: Speedup vs grid size
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    cpu_times = np.array(results['cpu_numba']) * 1000  # Convert to ms
    batch_sizes = results['batch_sizes']
    
    # Colors for different batch sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(batch_sizes)))
    
    # =================================
    # Left panel: Runtime comparison
    # =================================
    ax1 = axes[0]
    
    # CPU line
    ax1.loglog(grid_sizes, cpu_times, 'o-', color='#E74C3C', 
               linewidth=2, markersize=8, label='CPU (Numba)')
    
    # GPU lines for each batch size
    markers = ['s', '^', 'D', 'v', 'p']
    for i, bs in enumerate(batch_sizes):
        if bs in results['gpu_batched']:
            gpu_times = np.array(results['gpu_batched'][bs]) * 1000
            valid = ~np.isnan(gpu_times)
            if valid.any():
                ax1.loglog(grid_sizes[valid], gpu_times[valid], 
                          marker=markers[i % len(markers)], linestyle='--',
                          color=colors[i], linewidth=2, markersize=7,
                          label=f'GPU (B={bs})')
    
    # Reference O(N²) line
    N_ref = grid_sizes
    scale = cpu_times[len(cpu_times)//2] / (grid_sizes[len(grid_sizes)//2]**2)
    ax1.loglog(N_ref, scale * N_ref**2, ':', color='gray', linewidth=1.5, 
               alpha=0.7, label='$O(N^2)$')
    
    ax1.set_xlabel('Grid Size $N$', fontsize=12)
    ax1.set_ylabel('Time per Problem (ms)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='-')
    ax1.set_xlim(grid_sizes.min() * 0.8, grid_sizes.max() * 1.2)
    
    # =================================
    # Right panel: Speedup
    # =================================
    ax2 = axes[1]
    
    bar_width = 0.8 / len(batch_sizes)
    
    for i, bs in enumerate(batch_sizes):
        if bs in results['gpu_batched']:
            gpu_times = np.array(results['gpu_batched'][bs])
            valid = ~np.isnan(gpu_times) & (gpu_times > 0)
            if valid.any():
                speedup = (np.array(results['cpu_numba'])[valid]) / gpu_times[valid]
                x_pos = np.arange(len(grid_sizes))[valid] + bar_width * (i - len(batch_sizes)/2 + 0.5)
                ax2.bar(x_pos, speedup, width=bar_width, color=colors[i], 
                       edgecolor='black', linewidth=0.5, label=f'B={bs}')
    
    ax2.set_xticks(np.arange(len(grid_sizes)))
    ax2.set_xticklabels([str(int(n)) for n in grid_sizes])
    ax2.set_xlabel('Grid Size $N$', fontsize=12)
    ax2.set_ylabel('Speedup (CPU / GPU)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
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
    Create single-panel log-log runtime plot (matching the reference image style).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    grid_sizes = np.array(results['grid_sizes'])
    cpu_times = np.array(results['cpu_numba']) * 1000  # ms
    batch_sizes = results['batch_sizes']
    
    # CPU line (blue with circles like the reference)
    ax.loglog(grid_sizes, cpu_times, 'o-', color='#1f77b4', 
              linewidth=2.5, markersize=9, label='CPU (Numba)', zorder=3)
    
    # GPU lines (orange/yellow tones)
    gpu_colors = ['#ff7f0e', '#d62728', '#9467bd', '#2ca02c']
    markers = ['s', '^', 'D', 'v']
    
    for i, bs in enumerate(batch_sizes):
        if bs in results['gpu_batched']:
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
    ax.set_ylabel('Time per Problem (ms)', fontsize=13)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3)
    
    # Nice tick formatting
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
    batch_sizes = results['batch_sizes']
    
    print("=" * 70)
    print("Forward Benchmark Summary (from saved data)")
    print("=" * 70)
    
    print(f"\nGrid sizes: {list(grid_sizes)}")
    print(f"Batch sizes: {batch_sizes}")
    
    print("\nRuntime (ms):")
    header = f"{'N':>6} | {'CPU':>12}"
    for bs in batch_sizes:
        header += f" | {'GPU B='+str(bs):>12}"
    print(header)
    print("-" * len(header))
    
    for i, N in enumerate(grid_sizes):
        t_cpu = results['cpu_numba'][i] * 1000
        row = f"{int(N):>6} | {t_cpu:>10.2f}ms"
        for bs in batch_sizes:
            if bs in results['gpu_batched']:
                t_gpu = results['gpu_batched'][bs][i] * 1000
                row += f" | {t_gpu:>10.3f}ms"
            else:
                row += f" | {'N/A':>12}"
        print(row)
    
    print("\nSpeedup (CPU / GPU):")
    header = f"{'N':>6}"
    for bs in batch_sizes:
        header += f" | {'B='+str(bs):>10}"
    print(header)
    print("-" * len(header))
    
    for i, N in enumerate(grid_sizes):
        t_cpu = results['cpu_numba'][i]
        row = f"{int(N):>6}"
        for bs in batch_sizes:
            if bs in results['gpu_batched']:
                t_gpu = results['gpu_batched'][bs][i]
                if t_gpu > 0 and not np.isnan(t_gpu):
                    speedup = t_cpu / t_gpu
                    row += f" | {speedup:>9.1f}x"
                else:
                    row += f" | {'N/A':>10}"
            else:
                row += f" | {'N/A':>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Plot forward benchmark from saved data')
    parser.add_argument('--input', type=str, default='forward_solver_benchmark_data.npz',
                       help='Input .npz file with benchmark data')
    parser.add_argument('--output', type=str, default='forward_solver_benchmark.pdf',
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
