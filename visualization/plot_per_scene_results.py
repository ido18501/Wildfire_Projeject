#!/usr/bin/env python3
"""
Per-Scene Results Visualization
================================

Creates publication-quality visualizations from the per-scene generalization table.

Options:
1. Violin plot of key metrics across scenes
2. Horizontal bar chart sorted by correlation
3. Box plot comparison
4. Scatter plot: Correlation vs Rel.RMSE with scene labels

Usage:
    python visualize_per_scene_results.py --output figures/per_scene_results.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import argparse

# Data from the table
DATA = {
    'scenes': [
        '0001', '0002', '0003', '0004', '0005', '0014', '0015', '0016',
        '0017', '0018', '0019', '0022', '0023', '0024', '0025'
    ],
    'grid_size': [
        (116, 287), (205, 229), (411, 429), (417, 770), (408, 345),
        (132, 190), (473, 734), (523, 612), (310, 319), (284, 309),
        (424, 771), (533, 396), (568, 579), (565, 531), (574, 502)
    ],
    'n_fires': [2614, 714, 1715, 1191, 725, 426, 2986, 3070, 2731, 2440, 1316, 1630, 995, 2655, 2019],
    'correlation': {
        'mean': [0.780, 0.845, 0.889, 0.818, 0.738, 0.780, 0.848, 0.795, 0.755, 0.871, 0.862, 0.867, 0.820, 0.835, 0.862],
        'std':  [0.111, 0.099, 0.089, 0.206, 0.168, 0.152, 0.125, 0.149, 0.144, 0.103, 0.158, 0.101, 0.162, 0.151, 0.135]
    },
    'rel_rmse': {
        'mean': [0.226, 0.191, 0.204, 0.241, 0.358, 0.424, 0.191, 0.311, 0.321, 0.204, 0.179, 0.276, 0.371, 0.187, 0.213],
        'std':  [0.099, 0.092, 0.140, 0.208, 0.155, 0.295, 0.109, 0.170, 0.148, 0.114, 0.137, 0.139, 0.275, 0.113, 0.126]
    },
    'mae': {
        'mean': [12.32, 9.92, 11.23, 9.76, 20.07, 17.60, 10.53, 14.61, 12.55, 9.08, 8.36, 12.07, 12.88, 7.30, 7.82],
        'std':  [6.07, 5.46, 8.59, 7.33, 9.81, 10.91, 7.75, 8.77, 6.06, 5.80, 8.03, 6.90, 7.76, 5.70, 5.97]
    },
    'iou25': {
        'mean': [0.426, 0.510, 0.551, 0.551, 0.338, 0.397, 0.597, 0.418, 0.420, 0.510, 0.600, 0.522, 0.454, 0.589, 0.593],
        'std':  [0.207, 0.202, 0.219, 0.217, 0.166, 0.209, 0.203, 0.197, 0.194, 0.203, 0.219, 0.209, 0.214, 0.206, 0.198]
    },
    'iou50': {
        'mean': [0.556, 0.640, 0.649, 0.658, 0.455, 0.473, 0.742, 0.493, 0.491, 0.585, 0.753, 0.649, 0.558, 0.740, 0.699],
        'std':  [0.215, 0.202, 0.192, 0.188, 0.171, 0.202, 0.188, 0.191, 0.193, 0.212, 0.192, 0.193, 0.216, 0.192, 0.188]
    },
    'iou75': {
        'mean': [0.678, 0.719, 0.746, 0.797, 0.533, 0.616, 0.848, 0.617, 0.618, 0.710, 0.870, 0.745, 0.683, 0.864, 0.799],
        'std':  [0.181, 0.188, 0.141, 0.167, 0.162, 0.203, 0.147, 0.199, 0.187, 0.190, 0.145, 0.171, 0.221, 0.152, 0.165]
    },
    'iou100': {
        'mean': [0.937, 0.930, 0.912, 0.900, 0.778, 0.819, 0.936, 0.824, 0.837, 0.877, 0.946, 0.898, 0.860, 0.961, 0.908],
        'std':  [0.102, 0.081, 0.096, 0.150, 0.159, 0.185, 0.103, 0.175, 0.161, 0.150, 0.095, 0.113, 0.227, 0.077, 0.132]
    }
}

# Compute grid pixels
DATA['grid_pixels'] = [h * w for h, w in DATA['grid_size']]


def create_violin_plot(save_path=None, show=True):
    """
    Violin plot showing distribution of Correlation and Rel.RMSE across scenes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Generate synthetic samples for violin (using mean±std as normal)
    np.random.seed(42)
    n_samples = 100
    
    # Correlation violin
    ax1 = axes[0]
    corr_data = []
    for m, s in zip(DATA['correlation']['mean'], DATA['correlation']['std']):
        samples = np.clip(np.random.normal(m, s, n_samples), 0, 1)
        corr_data.append(samples)
    
    parts1 = ax1.violinplot(corr_data, positions=range(len(DATA['scenes'])), 
                            showmeans=True, showmedians=False)
    for pc in parts1['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    
    ax1.set_xticks(range(len(DATA['scenes'])))
    ax1.set_xticklabels(DATA['scenes'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Correlation', fontsize=11)
    ax1.set_title('(a) Pearson Correlation by Scene', fontsize=12, fontweight='bold')
    ax1.axhline(y=np.mean(DATA['correlation']['mean']), color='red', linestyle='--', 
                linewidth=1.5, label=f"Mean: {np.mean(DATA['correlation']['mean']):.3f}")
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Rel.RMSE violin
    ax2 = axes[1]
    rmse_data = []
    for m, s in zip(DATA['rel_rmse']['mean'], DATA['rel_rmse']['std']):
        samples = np.clip(np.random.normal(m, s, n_samples), 0, 1)
        rmse_data.append(samples)
    
    parts2 = ax2.violinplot(rmse_data, positions=range(len(DATA['scenes'])),
                            showmeans=True, showmedians=False)
    for pc in parts2['bodies']:
        pc.set_facecolor('#e74c3c')
        pc.set_alpha(0.7)
    
    ax2.set_xticks(range(len(DATA['scenes'])))
    ax2.set_xticklabels(DATA['scenes'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Relative RMSE', fontsize=11)
    ax2.set_title('(b) Relative RMSE by Scene', fontsize=12, fontweight='bold')
    ax2.axhline(y=np.mean(DATA['rel_rmse']['mean']), color='red', linestyle='--',
                linewidth=1.5, label=f"Mean: {np.mean(DATA['rel_rmse']['mean']):.3f}")
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(0, 0.8)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_horizontal_bar_sorted(save_path=None, show=True):
    """
    Horizontal bar chart sorted by correlation, showing correlation and IoU@50.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Sort by correlation
    sort_idx = np.argsort(DATA['correlation']['mean'])[::-1]
    scenes_sorted = [DATA['scenes'][i] for i in sort_idx]
    corr_sorted = [DATA['correlation']['mean'][i] for i in sort_idx]
    corr_std_sorted = [DATA['correlation']['std'][i] for i in sort_idx]
    
    y_pos = np.arange(len(scenes_sorted))
    
    # Bar chart
    bars = ax.barh(y_pos, corr_sorted, xerr=corr_std_sorted, 
                   color='#3498db', edgecolor='black', linewidth=0.5,
                   capsize=3, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Scene {s}" for s in scenes_sorted], fontsize=10)
    ax.set_xlabel('Pearson Correlation', fontsize=11)
    ax.set_title('Within-Scene Generalization Performance', fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=np.mean(corr_sorted), color='red', linestyle='--', linewidth=1.5,
               label=f"Mean: {np.mean(corr_sorted):.3f}")
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Invert y-axis so best is on top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_scatter_corr_vs_rmse(save_path=None, show=True):
    """
    Scatter plot: Correlation vs Rel.RMSE, sized by number of fires.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    corr = DATA['correlation']['mean']
    rmse = DATA['rel_rmse']['mean']
    n_fires = DATA['n_fires']
    scenes = DATA['scenes']
    
    # Normalize fire counts for marker size
    sizes = np.array(n_fires) / max(n_fires) * 300 + 50
    
    scatter = ax.scatter(corr, rmse, s=sizes, c=corr, cmap='RdYlGn', 
                         edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add scene labels
    for i, scene in enumerate(scenes):
        ax.annotate(scene, (corr[i], rmse[i]), fontsize=8, 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Correlation (↑ better)', fontsize=11)
    ax.set_ylabel('Relative RMSE (↓ better)', fontsize=11)
    ax.set_title('Per-Scene Performance', fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Correlation', fontsize=10)
    
    # Add size legend
    for size, label in [(500, '3000'), (200, '1500'), (80, '500')]:
        ax.scatter([], [], s=size, c='gray', alpha=0.5, edgecolor='black',
                  label=f'{label} fires')
    ax.legend(loc='upper right', title='# Fires', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.7, 0.92)
    ax.set_ylim(0.1, 0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_combined_figure(save_path=None, show=True):
    """
    Combined figure with:
    - Left: Horizontal bar (correlation) sorted
    - Right top: Box plot of IoU metrics
    - Right bottom: Scatter correlation vs RMSE
    """
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1],
                           wspace=0.25, hspace=0.3)
    
    # =====================
    # Left: Horizontal bar chart (spans both rows)
    # =====================
    ax1 = fig.add_subplot(gs[:, 0])
    
    sort_idx = np.argsort(DATA['correlation']['mean'])[::-1]
    scenes_sorted = [DATA['scenes'][i] for i in sort_idx]
    corr_sorted = [DATA['correlation']['mean'][i] for i in sort_idx]
    corr_std_sorted = [DATA['correlation']['std'][i] for i in sort_idx]
    n_fires_sorted = [DATA['n_fires'][i] for i in sort_idx]
    
    y_pos = np.arange(len(scenes_sorted))
    
    # Color by performance
    colors = plt.cm.RdYlGn([(c - 0.7) / 0.2 for c in corr_sorted])
    
    bars = ax1.barh(y_pos, corr_sorted, xerr=corr_std_sorted,
                    color=colors, edgecolor='black', linewidth=0.5,
                    capsize=3, alpha=0.85)
    
    # Add fire counts as text (after bar end)
    for i, (c, n) in enumerate(zip(corr_sorted, n_fires_sorted)):
        ax1.text(c + 0.015, i, f'n={n}', va='center', ha='left', 
                fontsize=8, color='#333')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"Scene {s}" for s in scenes_sorted], fontsize=10)
    ax1.set_xlabel('Pearson Correlation (bar: green=high, red=low)', fontsize=11)
    ax1.set_title('(a) Within-Scene Generalization\n(sorted by correlation)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0.5, 1.0)
    ax1.axvline(x=np.mean(corr_sorted), color='black', linestyle='--', linewidth=1.5,
                label=f"Mean: {np.mean(corr_sorted):.3f}")
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # =====================
    # Right top: Box plot of IoU metrics
    # =====================
    ax2 = fig.add_subplot(gs[0, 1])
    
    iou_data = [
        DATA['iou25']['mean'],
        DATA['iou50']['mean'],
        DATA['iou75']['mean'],
        DATA['iou100']['mean']
    ]
    
    bp = ax2.boxplot(iou_data, labels=['IoU@25', 'IoU@50', 'IoU@75', 'IoU@100'],
                     patch_artist=True)
    
    colors_box = ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax2.set_ylabel('IoU Score', fontsize=11)
    ax2.set_title('(b) Temporal Accuracy (IoU)', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0.2, 1.0)
    
    # =====================
    # Right bottom: Scatter correlation vs RMSE
    # =====================
    ax3 = fig.add_subplot(gs[1, 1])
    
    corr = DATA['correlation']['mean']
    rmse = DATA['rel_rmse']['mean']
    n_fires = DATA['n_fires']
    
    sizes = np.array(n_fires) / max(n_fires) * 250 + 50
    
    scatter = ax3.scatter(corr, rmse, s=sizes, c=corr, cmap='RdYlGn',
                          edgecolor='black', linewidth=0.5, alpha=0.8,
                          vmin=0.7, vmax=0.9)
    
    ax3.set_xlabel('Correlation (↑)', fontsize=11)
    ax3.set_ylabel('Rel. RMSE (↓)', fontsize=11)
    ax3.set_title('(c) Correlation vs Error', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.7, 0.92)
    ax3.set_ylim(0.1, 0.5)
    
    # Size legend
    for size, label in [(300, '3k'), (150, '1.5k'), (60, '500')]:
        ax3.scatter([], [], s=size, c='gray', alpha=0.5, edgecolor='black',
                   label=f'{label}')
    ax3.legend(loc='upper right', title='Fires', fontsize=8, title_fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_simple_summary(save_path=None, show=True):
    """
    Simple, clean summary figure:
    - Bar chart with error bars, sorted by correlation
    - Color gradient from red (low) to green (high)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sort by correlation
    sort_idx = np.argsort(DATA['correlation']['mean'])[::-1]
    scenes_sorted = [DATA['scenes'][i] for i in sort_idx]
    corr_sorted = [DATA['correlation']['mean'][i] for i in sort_idx]
    corr_std_sorted = [DATA['correlation']['std'][i] for i in sort_idx]
    
    x_pos = np.arange(len(scenes_sorted))
    
    # Color gradient
    norm_corr = [(c - min(corr_sorted)) / (max(corr_sorted) - min(corr_sorted)) 
                 for c in corr_sorted]
    colors = plt.cm.RdYlGn([0.3 + 0.6 * n for n in norm_corr])
    
    bars = ax.bar(x_pos, corr_sorted, yerr=corr_std_sorted,
                  color=colors, edgecolor='black', linewidth=0.5,
                  capsize=4, alpha=0.85)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}" for s in scenes_sorted], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_xlabel('Scene ID', fontsize=12)
    ax.set_title('Within-Scene Generalization: Correlation by Scene', fontsize=13, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    
    # Mean line
    mean_corr = np.mean(corr_sorted)
    ax.axhline(y=mean_corr, color='black', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_corr:.3f}')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_grid_analysis(save_path=None, show=True):
    """
    Two-panel figure analyzing relationship between grid size and performance.
    - Left: Scatter of grid pixels vs correlation (sized by # fires)
    - Right: Scatter of grid pixels vs Rel.RMSE
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    grid_pixels = np.array(DATA['grid_pixels']) / 1000  # In thousands
    corr = DATA['correlation']['mean']
    rmse = DATA['rel_rmse']['mean']
    n_fires = DATA['n_fires']
    scenes = DATA['scenes']
    
    sizes = np.array(n_fires) / max(n_fires) * 300 + 50
    
    # Left: Grid size vs Correlation
    ax1 = axes[0]
    scatter1 = ax1.scatter(grid_pixels, corr, s=sizes, c=corr, cmap='RdYlGn',
                           edgecolor='black', linewidth=0.5, alpha=0.8,
                           vmin=0.7, vmax=0.9)
    
    # Add trend line
    z = np.polyfit(grid_pixels, corr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(grid_pixels), max(grid_pixels), 100)
    ax1.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)
    
    # Correlation coefficient
    r = np.corrcoef(grid_pixels, corr)[0, 1]
    ax1.text(0.05, 0.95, f'r = {r:.3f}', transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i, scene in enumerate(scenes):
        ax1.annotate(scene, (grid_pixels[i], corr[i]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points', alpha=0.7)
    
    ax1.set_xlabel('Grid Size (thousands of pixels)', fontsize=11)
    ax1.set_ylabel('Correlation', fontsize=11)
    ax1.set_title('(a) Grid Size vs Correlation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right: Grid size vs RMSE
    ax2 = axes[1]
    scatter2 = ax2.scatter(grid_pixels, rmse, s=sizes, c=rmse, cmap='RdYlGn_r',
                           edgecolor='black', linewidth=0.5, alpha=0.8,
                           vmin=0.15, vmax=0.45)
    
    # Add trend line
    z = np.polyfit(grid_pixels, rmse, 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)
    
    # Correlation coefficient
    r = np.corrcoef(grid_pixels, rmse)[0, 1]
    ax2.text(0.05, 0.95, f'r = {r:.3f}', transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i, scene in enumerate(scenes):
        ax2.annotate(scene, (grid_pixels[i], rmse[i]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points', alpha=0.7)
    
    ax2.set_xlabel('Grid Size (thousands of pixels)', fontsize=11)
    ax2.set_ylabel('Relative RMSE', fontsize=11)
    ax2.set_title('(b) Grid Size vs Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Size legend
    for ax in axes:
        for size, label in [(300, '3k fires'), (150, '1.5k'), (60, '500')]:
            ax.scatter([], [], s=size, c='gray', alpha=0.5, edgecolor='black',
                      label=label)
        ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_summary_with_grid(save_path=None, show=True):
    """
    Clean summary figure with grid size information:
    - Horizontal bar chart sorted by correlation
    - Bar width/annotation shows grid size
    - Color shows correlation quality
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort by correlation
    sort_idx = np.argsort(DATA['correlation']['mean'])[::-1]
    scenes_sorted = [DATA['scenes'][i] for i in sort_idx]
    corr_sorted = [DATA['correlation']['mean'][i] for i in sort_idx]
    corr_std_sorted = [DATA['correlation']['std'][i] for i in sort_idx]
    grid_sorted = [DATA['grid_size'][i] for i in sort_idx]
    pixels_sorted = [DATA['grid_pixels'][i] for i in sort_idx]
    n_fires_sorted = [DATA['n_fires'][i] for i in sort_idx]
    
    y_pos = np.arange(len(scenes_sorted))
    
    # Color by correlation
    colors = plt.cm.RdYlGn([(c - 0.7) / 0.2 for c in corr_sorted])
    
    bars = ax.barh(y_pos, corr_sorted, xerr=corr_std_sorted,
                   color=colors, edgecolor='black', linewidth=0.5,
                   capsize=3, alpha=0.85)
    
    # Add grid size and fire count as labels
    for i, (c, g, n, p) in enumerate(zip(corr_sorted, grid_sorted, n_fires_sorted, pixels_sorted)):
        # Grid size on the left of bar
        ax.text(0.52, i, f'{g[0]}×{g[1]}', va='center', ha='left',
                fontsize=8, color='#333', alpha=0.8)
        # Fire count after bar end
        ax.text(c + 0.015, i, f'n={n}', va='center', ha='left',
                fontsize=8, color='#333')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Scene {s}" for s in scenes_sorted], fontsize=10)
    ax.set_xlabel('Pearson Correlation (bar: green=high, red=low)', fontsize=12)
    ax.set_title('Within-Scene Generalization Performance\n(with grid dimensions)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    
    # Mean line
    mean_corr = np.mean(corr_sorted)
    ax.axvline(x=mean_corr, color='black', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_corr:.3f}')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_bubble_chart(save_path=None, show=True):
    """
    Bubble chart: x=grid pixels, y=correlation, size=fires, color=RMSE
    Single comprehensive view of all factors.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    grid_pixels = np.array(DATA['grid_pixels']) / 1000
    corr = DATA['correlation']['mean']
    rmse = DATA['rel_rmse']['mean']
    n_fires = DATA['n_fires']
    scenes = DATA['scenes']
    
    # Size by fires, color by RMSE (inverted - lower is better = greener)
    sizes = np.array(n_fires) / max(n_fires) * 400 + 80
    
    scatter = ax.scatter(grid_pixels, corr, s=sizes, c=rmse, cmap='RdYlGn_r',
                         edgecolor='black', linewidth=0.8, alpha=0.85,
                         vmin=0.15, vmax=0.45)
    
    # Add scene labels
    for i, scene in enumerate(scenes):
        ax.annotate(scene, (grid_pixels[i], corr[i]), fontsize=9,
                   xytext=(0, -15), textcoords='offset points', ha='center',
                   fontweight='bold')
    
    ax.set_xlabel('Grid Size (thousands of pixels)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Within-Scene Performance by Grid Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 0.92)
    
    # Colorbar for RMSE
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Relative RMSE (↓ better)', fontsize=11)
    
    # Size legend
    legend_sizes = [(400, '3000'), (200, '1500'), (100, '500')]
    for size, label in legend_sizes:
        ax.scatter([], [], s=size, c='gray', alpha=0.5, edgecolor='black',
                  label=f'{label} fires')
    ax.legend(loc='lower right', title='# Training Fires', fontsize=9, title_fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_combined_with_grid(save_path=None, show=True):
    """
    Combined figure with grid size information - single row, 3 panels:
    - (a) Bubble: grid size vs correlation (size=fires, color=RMSE)
    - (b) Box plot of IoU metrics
    - (c) Horizontal bar sorted by correlation with grid dimensions
    """
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 0.8, 1.3], wspace=0.3)
    
    # =====================
    # (a) Left: Bubble chart - grid size vs correlation
    # =====================
    ax1 = fig.add_subplot(gs[0, 0])
    
    grid_pixels = np.array(DATA['grid_pixels']) / 1000
    corr = DATA['correlation']['mean']
    rmse = DATA['rel_rmse']['mean']
    n_fires = DATA['n_fires']
    scenes = DATA['scenes']
    
    sizes = np.array(n_fires) / max(n_fires) * 300 + 50
    
    scatter1 = ax1.scatter(grid_pixels, corr, s=sizes, c=rmse, cmap='RdYlGn_r',
                           edgecolor='black', linewidth=0.6, alpha=0.85,
                           vmin=0.15, vmax=0.45)
    
    # Trend line
    z = np.polyfit(grid_pixels, corr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(grid_pixels), max(grid_pixels), 100)
    ax1.plot(x_line, p(x_line), 'k--', alpha=0.4, linewidth=1.5)
    
    r = np.corrcoef(grid_pixels, corr)[0, 1]
    ax1.text(0.05, 0.95, f'r = {r:.2f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    ax1.set_xlabel('Grid Size (k pixels)', fontsize=10)
    ax1.set_ylabel('Correlation', fontsize=10)
    # ax1.set_title('(a) Performance vs Grid Size\n(color=RMSE, size=#fires)', 
    #               fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 0.92)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.85, pad=0.02)
    cbar1.set_label('RMSE', fontsize=9)
    cbar1.ax.tick_params(labelsize=8)
    
    # Size legend
    for s, label in [(300, '3k'), (150, '1k'), (60, '400')]:
        ax1.scatter([], [], s=s, c='gray', alpha=0.5, edgecolor='black', label=label)
    ax1.legend(loc='lower left', title='Fires', fontsize=7, title_fontsize=8,
               framealpha=0.9, handletextpad=0.1)
    
    # =====================
    # (b) Middle: Box plot of IoU metrics
    # =====================
    ax2 = fig.add_subplot(gs[0, 1])
    
    iou_data = [
        DATA['iou25']['mean'],
        DATA['iou50']['mean'],
        DATA['iou75']['mean'],
        DATA['iou100']['mean']
    ]
    
    bp = ax2.boxplot(iou_data, labels=['25', '50', '75', '100'],
                     patch_artist=True, widths=0.55)
    
    # Color gradient: lighter to darker orange for increasing thresholds
    colors_box = ['#fdbe85', '#fd8d3c', '#e6550d', '#a63603']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    
    # Add individual points
    np.random.seed(42)
    for i, data in enumerate(iou_data):
        x = np.random.normal(i+1, 0.04, len(data))
        ax2.scatter(x, data, alpha=0.5, s=15, color='#333', zorder=3)
    
    ax2.set_ylabel('IoU Score', fontsize=10)
    ax2.set_xlabel('Time Threshold (min)', fontsize=10)
    # ax2.set_title('(b) Temporal Accuracy\n(darker=stricter)', 
    #               fontsize=10, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0.25, 1.0)
    
    # =====================
    # (c) Right: Horizontal bar sorted by correlation
    # =====================
    ax3 = fig.add_subplot(gs[0, 2])
    
    sort_idx = np.argsort(DATA['correlation']['mean'])[::-1]
    scenes_sorted = [DATA['scenes'][i] for i in sort_idx]
    corr_sorted = [DATA['correlation']['mean'][i] for i in sort_idx]
    corr_std_sorted = [DATA['correlation']['std'][i] for i in sort_idx]
    grid_sorted = [DATA['grid_size'][i] for i in sort_idx]
    n_fires_sorted = [DATA['n_fires'][i] for i in sort_idx]
    rmse_sorted = [DATA['rel_rmse']['mean'][i] for i in sort_idx]
    
    y_pos = np.arange(len(scenes_sorted))
    
    # Color by correlation: red (low) -> yellow -> green (high)
    colors = plt.cm.RdYlGn([(c - 0.7) / 0.2 for c in corr_sorted])
    
    bars = ax3.barh(y_pos, corr_sorted, xerr=corr_std_sorted,
                    color=colors, edgecolor='black', linewidth=0.4,
                    capsize=1.5, alpha=0.85, height=0.65)
    
    # Labels: fire count after bar, RMSE on right
    for i, (c, g, n, r) in enumerate(zip(corr_sorted, grid_sorted, n_fires_sorted, rmse_sorted)):
        # Fire count after bar
        ax3.text(c + 0.012, i-0.3, f'{n}', va='center', ha='left',
                fontsize=7, color='#333')
        # RMSE on right edge
        ax3.text(1.01, i, f'{r:.2f}', va='center', ha='left',
                fontsize=7, color='#333', transform=ax3.get_yaxis_transform())
    
    # Column headers
    ax3.text(0.88, -1.0, 'n', va='bottom', ha='center',
            fontsize=8, fontweight='bold', color='#333')
    ax3.text(1.01, -1.0, 'RMSE', va='bottom', ha='left',
            fontsize=8, fontweight='bold', color='#333',
            transform=ax3.get_yaxis_transform())
    
    # Y-axis labels include grid size
    ylabels = [f"{s} ({g[0]}×{g[1]})" for s, g in zip(scenes_sorted, grid_sorted)]
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(ylabels, fontsize=7, family='monospace')
    ax3.set_xlabel('Correlation', fontsize=10)
    # ax3.set_title('(c) Per-Scene Results\n(green=high, red=low)', 
    #               fontsize=10, fontweight='bold')
    ax3.set_xlim(0.5, 1.0)
    
    mean_corr = np.mean(corr_sorted)
    ax3.axvline(x=mean_corr, color='black', linestyle='--', linewidth=1.2,
                label=f'μ={mean_corr:.2f}')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def create_full_combined(save_path=None, show=True):
    """
    Comprehensive 2x2 figure:
    - (a) Top-left: Horizontal bar sorted by correlation (with grid dims + fire counts)
    - (b) Top-right: Bubble chart (grid size vs correlation, sized by fires, colored by RMSE)
    - (c) Bottom-left: Box plot of IoU metrics
    - (d) Bottom-right: Violin of correlation distribution across scenes
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35)
    
    # =====================
    # (a) Top-left: Horizontal bar chart
    # =====================
    ax1 = fig.add_subplot(gs[0, 0])
    
    sort_idx = np.argsort(DATA['correlation']['mean'])[::-1]
    scenes_sorted = [DATA['scenes'][i] for i in sort_idx]
    corr_sorted = [DATA['correlation']['mean'][i] for i in sort_idx]
    corr_std_sorted = [DATA['correlation']['std'][i] for i in sort_idx]
    grid_sorted = [DATA['grid_size'][i] for i in sort_idx]
    n_fires_sorted = [DATA['n_fires'][i] for i in sort_idx]
    
    y_pos = np.arange(len(scenes_sorted))
    colors = plt.cm.RdYlGn([(c - 0.7) / 0.2 for c in corr_sorted])
    
    bars = ax1.barh(y_pos, corr_sorted, xerr=corr_std_sorted,
                    color=colors, edgecolor='black', linewidth=0.5,
                    capsize=2, alpha=0.85)
    
    for i, (c, g, n) in enumerate(zip(corr_sorted, grid_sorted, n_fires_sorted)):
        ax1.text(0.52, i, f'{g[0]}×{g[1]}', va='center', ha='left',
                fontsize=7, color='#333', alpha=0.7)
        ax1.text(c + 0.01, i, f'n={n}', va='center', ha='left',
                fontsize=7, color='#333')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{s}" for s in scenes_sorted], fontsize=9)
    ax1.set_xlabel('Correlation (bar: green=high, red=low)', fontsize=10)
    ax1.set_title('(a) Performance by Scene', fontsize=11, fontweight='bold')
    ax1.set_xlim(0.5, 1.0)
    ax1.axvline(x=np.mean(corr_sorted), color='black', linestyle='--', linewidth=1.2)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # =====================
    # (b) Top-right: Bubble chart
    # =====================
    ax2 = fig.add_subplot(gs[0, 1])
    
    grid_pixels = np.array(DATA['grid_pixels']) / 1000
    corr = DATA['correlation']['mean']
    rmse = DATA['rel_rmse']['mean']
    n_fires = DATA['n_fires']
    scenes = DATA['scenes']
    
    sizes = np.array(n_fires) / max(n_fires) * 300 + 50
    
    scatter = ax2.scatter(grid_pixels, corr, s=sizes, c=rmse, cmap='RdYlGn_r',
                          edgecolor='black', linewidth=0.5, alpha=0.85,
                          vmin=0.15, vmax=0.45)
    
    # Trend line
    z = np.polyfit(grid_pixels, corr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(grid_pixels), max(grid_pixels), 100)
    ax2.plot(x_line, p(x_line), 'k--', alpha=0.4, linewidth=1.2)
    
    r = np.corrcoef(grid_pixels, corr)[0, 1]
    ax2.text(0.05, 0.05, f'r = {r:.2f}', transform=ax2.transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Grid Size (k pixels)', fontsize=10)
    ax2.set_ylabel('Correlation', fontsize=10)
    ax2.set_title('(b) Grid Size vs Performance', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 0.92)
    
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('RMSE (green=low, red=high)', fontsize=9)
    
    # Size legend
    for size, label in [(280, '3k'), (140, '1k')]:
        ax2.scatter([], [], s=size, c='gray', alpha=0.4, edgecolor='black', label=label)
    ax2.legend(loc='lower right', title='Fires', fontsize=8, title_fontsize=8)
    
    # =====================
    # (c) Bottom-left: Box plot of IoU metrics
    # =====================
    ax3 = fig.add_subplot(gs[1, 0])
    
    iou_data = [
        DATA['iou25']['mean'],
        DATA['iou50']['mean'],
        DATA['iou75']['mean'],
        DATA['iou100']['mean']
    ]
    
    bp = ax3.boxplot(iou_data, labels=['@25', '@50', '@75', '@100'],
                     patch_artist=True, widths=0.6)
    
    # Color gradient: lighter (lenient) to darker (strict)
    colors_box = ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add individual points
    np.random.seed(42)
    for i, d in enumerate(iou_data):
        x = np.random.normal(i+1, 0.04, len(d))
        ax3.scatter(x, d, alpha=0.5, s=20, c='black', zorder=3)
    
    ax3.set_ylabel('IoU Score', fontsize=10)
    ax3.set_xlabel('Time Threshold (minutes)', fontsize=10)
    ax3.set_title('(c) Temporal Accuracy (darker=stricter)', fontsize=11, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(0.2, 1.0)
    
    # =====================
    # (d) Bottom-right: Violin of correlation and RMSE
    # =====================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create violin data
    np.random.seed(42)
    n_samples = 50
    
    corr_samples = []
    rmse_samples = []
    for m, s in zip(DATA['correlation']['mean'], DATA['correlation']['std']):
        corr_samples.extend(np.clip(np.random.normal(m, s, n_samples), 0, 1))
    for m, s in zip(DATA['rel_rmse']['mean'], DATA['rel_rmse']['std']):
        rmse_samples.extend(np.clip(np.random.normal(m, s, n_samples), 0, 1))
    
    parts = ax4.violinplot([corr_samples, rmse_samples], positions=[1, 2],
                           showmeans=True, showmedians=False, widths=0.7)
    
    colors_violin = ['#3498db', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)
    
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['Correlation\n(↑ better)', 'Rel. RMSE\n(↓ better)'], fontsize=10)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.set_title('(d) Metric Distributions', fontsize=11, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Add mean annotations
    mean_corr = np.mean(DATA['correlation']['mean'])
    mean_rmse = np.mean(DATA['rel_rmse']['mean'])
    ax4.annotate(f'μ={mean_corr:.3f}', xy=(1, mean_corr), xytext=(1.3, mean_corr+0.05),
                fontsize=9, ha='left')
    ax4.annotate(f'μ={mean_rmse:.3f}', xy=(2, mean_rmse), xytext=(2.1, mean_rmse+0.05),
                fontsize=9, ha='left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize per-scene results')
    parser.add_argument('--output', type=str, default='per_scene_results.pdf')
    parser.add_argument('--style', type=str, default='combined_grid',
                       choices=['violin', 'bar', 'scatter', 'combined', 'simple', 
                                'grid', 'grid_bar', 'bubble', 'combined_grid', 'full'],
                       help='Visualization style')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    
    print(f"Creating {args.style} visualization...")
    
    if args.style == 'violin':
        create_violin_plot(save_path=args.output, show=not args.no_show)
    elif args.style == 'bar':
        create_horizontal_bar_sorted(save_path=args.output, show=not args.no_show)
    elif args.style == 'scatter':
        create_scatter_corr_vs_rmse(save_path=args.output, show=not args.no_show)
    elif args.style == 'combined':
        create_combined_figure(save_path=args.output, show=not args.no_show)
    elif args.style == 'simple':
        create_simple_summary(save_path=args.output, show=not args.no_show)
    elif args.style == 'grid':
        create_grid_analysis(save_path=args.output, show=not args.no_show)
    elif args.style == 'grid_bar':
        create_summary_with_grid(save_path=args.output, show=not args.no_show)
    elif args.style == 'bubble':
        create_bubble_chart(save_path=args.output, show=not args.no_show)
    elif args.style == 'combined_grid':
        create_combined_with_grid(save_path=args.output, show=not args.no_show)
    elif args.style == 'full':
        create_full_combined(save_path=args.output, show=not args.no_show)
    
    print("Done!")


if __name__ == '__main__':
    main()
