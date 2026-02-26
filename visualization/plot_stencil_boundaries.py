#!/usr/bin/env python3
"""
Stencil Boundary Visualization (Improved B7)
=============================================

Visualizes where the active stencil changes between adjacent pixels,
forming measure-zero curves that trace characteristics from the source.

Improvements over original B7:
1. Non-trivial anisotropic metric with drift (wind effect)
2. Discrete categorical colormap with proper stencil labels
3. Publication-quality figure

The 8 triangular stencils correspond to:
  0: NW quadrant, upper triangle
  1: NW quadrant, lower triangle  
  2: NE quadrant, upper triangle
  3: NE quadrant, lower triangle
  4: SE quadrant, upper triangle
  5: SE quadrant, lower triangle
  6: SW quadrant, upper triangle
  7: SW quadrant, lower triangle

Usage:
    python visualize_stencil_boundaries.py --output figures/stencil_boundaries.pdf

Author: Barak Gahtan
Date: February 2025
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Ellipse

# Add project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.backward.implicit_diff import identify_active_stencils


# =============================================================================
# Stencil definitions
# =============================================================================

STENCIL_LABELS = [
    '0: NW-upper',
    '1: NW-lower', 
    '2: NE-upper',
    '3: NE-lower',
    '4: SE-upper',
    '5: SE-lower',
    '6: SW-upper',
    '7: SW-lower',
]

# Stencil directions (m1, m2) for reference
# Each stencil uses two neighboring pixels to compute the update
STENCIL_DIRECTIONS = {
    0: ((-1, -1), (-1, 0)),   # NW quadrant, upper
    1: ((-1, -1), (0, -1)),   # NW quadrant, lower
    2: ((-1, +1), (-1, 0)),   # NE quadrant, upper
    3: ((-1, +1), (0, +1)),   # NE quadrant, lower
    4: ((+1, +1), (+1, 0)),   # SE quadrant, upper
    5: ((+1, +1), (0, +1)),   # SE quadrant, lower
    6: ((+1, -1), (+1, 0)),   # SW quadrant, upper
    7: ((+1, -1), (0, -1)),   # SW quadrant, lower
}


def create_anisotropic_problem(N: int, angle_deg: float = 30.0, 
                                anisotropy: float = 2.0,
                                drift_strength: float = 0.1,
                                drift_angle_deg: float = 45.0):
    """
    Create a non-trivial anisotropic problem with drift.
    
    Args:
        N: Grid size
        angle_deg: Rotation angle of the metric ellipse (degrees)
        anisotropy: Ratio of major/minor axis (>1 means faster along major axis)
        drift_strength: Magnitude of drift vector (wind effect)
        drift_angle_deg: Direction of drift (degrees)
    
    Returns:
        problem: EikonalProblem
        T: Solved arrival times
    """
    # Metric tensor G (rotated ellipse)
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    
    # Speed is sqrt(1/eigenvalue), so larger eigenvalue = slower
    # For anisotropy ratio r: λ1 = 1, λ2 = r
    # This means faster propagation along eigenvector 1 (angle direction)
    base = 0.35
    
    G = torch.zeros(1, 3, N, N)
    G[0, 0] = (c**2 * anisotropy + s**2) * base      # g11
    G[0, 1] = c * s * (anisotropy - 1) * base        # g12
    G[0, 2] = (s**2 * anisotropy + c**2) * base      # g22
    
    # Drift vector B (wind effect)
    drift_angle = np.deg2rad(drift_angle_deg)
    B = torch.zeros(1, 2, N, N)
    B[0, 0] = drift_strength * np.cos(drift_angle)  # b_x
    B[0, 1] = drift_strength * np.sin(drift_angle)  # b_y
    
    # Source at center
    source = torch.zeros(1, N, N, dtype=torch.bool)
    ci, cj = N // 2, N // 2
    source[0, ci-1:ci+2, cj-1:cj+2] = True  # 3x3 source
    
    problem = EikonalProblem(G, B, source)
    T, n_iter = fast_sweeping_solve(problem, max_iter=50, verbose=False)
    
    return problem, T


def create_complex_metric_problem(N: int, complexity: str = 'radial'):
    """
    Create a spatially-varying metric for complex stencil patterns.
    
    Args:
        N: Grid size
        complexity: Type of spatial variation
            - 'radial': Anisotropy direction rotates around center
            - 'vortex': Circular flow pattern with varying speed
            - 'gradient': Anisotropy varies from left to right
            - 'checkerboard': Alternating high/low anisotropy regions
            - 'wave': Smooth sinusoidal variation (like reference image)
    
    Returns:
        problem: EikonalProblem
        T: Solved arrival times
    """
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    x = x.float()
    y = y.float()
    
    # Center coordinates
    cx, cy = N / 2, N / 2
    dx = x - cx
    dy = y - cy
    
    G = torch.zeros(1, 3, N, N)
    B = torch.zeros(1, 2, N, N)
    
    if complexity == 'radial':
        # Anisotropy direction rotates around center (like spokes of a wheel)
        # Fast direction is radial (pointing away from center)
        angle = torch.atan2(dy, dx)  # Angle from center
        
        base_speed = 0.35
        anisotropy = 2.5  # Strong anisotropy
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        # Metric with radial fast direction
        G[0, 0] = (c**2 * anisotropy + s**2) * base_speed
        G[0, 1] = c * s * (anisotropy - 1) * base_speed
        G[0, 2] = (s**2 * anisotropy + c**2) * base_speed
        
        # Small tangential drift (creates spiral effect)
        r = torch.sqrt(dx**2 + dy**2).clamp(min=1)
        B[0, 0] = -0.05 * dy / r  # Tangential direction
        B[0, 1] = 0.05 * dx / r
        
    elif complexity == 'vortex':
        # Circular flow with varying anisotropy strength
        angle = torch.atan2(dy, dx)
        r = torch.sqrt(dx**2 + dy**2).clamp(min=1)
        
        # Tangential fast direction (perpendicular to radial)
        tang_angle = angle + np.pi / 2
        c = torch.cos(tang_angle)
        s = torch.sin(tang_angle)
        
        # Anisotropy increases with radius
        base_speed = 0.3
        anisotropy = 1.5 + 1.5 * (r / (N/2)).clamp(max=1)
        
        G[0, 0] = (c**2 * anisotropy + s**2) * base_speed
        G[0, 1] = c * s * (anisotropy - 1) * base_speed
        G[0, 2] = (s**2 * anisotropy + c**2) * base_speed
        
        # Outward drift
        B[0, 0] = 0.08 * dx / r
        B[0, 1] = 0.08 * dy / r
        
    elif complexity == 'gradient':
        # Anisotropy direction varies from horizontal (left) to vertical (right)
        angle = (x / N) * (np.pi / 2)  # 0 to 90 degrees
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        base_speed = 0.35
        anisotropy = 2.0
        
        G[0, 0] = (c**2 * anisotropy + s**2) * base_speed
        G[0, 1] = c * s * (anisotropy - 1) * base_speed
        G[0, 2] = (s**2 * anisotropy + c**2) * base_speed
        
        # Diagonal drift
        B[0, 0] = 0.06
        B[0, 1] = 0.04
        
    elif complexity == 'checkerboard':
        # Alternating regions of different anisotropy directions
        block_size = N // 4
        block_x = (x // block_size).long() % 2
        block_y = (y // block_size).long() % 2
        checker = (block_x + block_y) % 2
        
        # Two different angles based on checker pattern
        angle = checker.float() * (np.pi / 3)  # 0 or 60 degrees
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        base_speed = 0.35
        anisotropy = 2.2
        
        G[0, 0] = (c**2 * anisotropy + s**2) * base_speed
        G[0, 1] = c * s * (anisotropy - 1) * base_speed
        G[0, 2] = (s**2 * anisotropy + c**2) * base_speed
        
        # Varying drift based on position
        B[0, 0] = 0.05 * torch.sin(2 * np.pi * x / N)
        B[0, 1] = 0.05 * torch.cos(2 * np.pi * y / N)
    
    elif complexity == 'wave':
        # Smooth sinusoidal variation - like reference image with rotating ellipses
        # Angle varies smoothly across the domain
        freq_x = 2.0  # Spatial frequency
        freq_y = 1.5
        
        angle = (np.pi / 3) * torch.sin(2 * np.pi * freq_x * x / N) * \
                             torch.cos(2 * np.pi * freq_y * y / N)
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        base_speed = 0.35
        # Anisotropy also varies spatially
        anisotropy = 2.0 + 0.8 * torch.sin(2 * np.pi * x / N + np.pi * y / N)
        
        G[0, 0] = (c**2 * anisotropy + s**2) * base_speed
        G[0, 1] = c * s * (anisotropy - 1) * base_speed
        G[0, 2] = (s**2 * anisotropy + c**2) * base_speed
        
        # Small varying drift
        B[0, 0] = 0.04 * torch.sin(np.pi * x / N)
        B[0, 1] = 0.04 * torch.cos(np.pi * y / N)
    
    else:
        raise ValueError(f"Unknown complexity type: {complexity}")
    
    # Source at center
    source = torch.zeros(1, N, N, dtype=torch.bool)
    ci, cj = N // 2, N // 2
    source[0, ci-1:ci+2, cj-1:cj+2] = True
    
    problem = EikonalProblem(G, B, source)
    T, n_iter = fast_sweeping_solve(problem, max_iter=50, verbose=False)
    
    return problem, T


def find_stencil_boundaries(stencil_idx: np.ndarray) -> np.ndarray:
    """
    Find pixels where the active stencil differs from at least one neighbor.
    
    These boundaries form measure-zero curves tracing characteristics.
    """
    H, W = stencil_idx.shape
    boundaries = np.zeros_like(stencil_idx, dtype=bool)
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            center = stencil_idx[i, j]
            # 4-connectivity neighbors
            neighbors = [
                stencil_idx[i-1, j],  # top
                stencil_idx[i+1, j],  # bottom
                stencil_idx[i, j-1],  # left
                stencil_idx[i, j+1],  # right
            ]
            if any(n != center for n in neighbors):
                boundaries[i, j] = True
    
    return boundaries


def visualize_stencils(stencil_idx: np.ndarray, boundaries: np.ndarray,
                       T: np.ndarray, G: np.ndarray = None, B: np.ndarray = None,
                       title: str = None,
                       save_path: str = None, show: bool = True):
    """
    Create publication-quality two-panel stencil visualization.
    
    Left panel: Arrival time with metric ellipses and drift quiver
    Right panel: Stencil map with legend (boundaries visible at color transitions)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.05)
    
    H, W = T.shape
    step = max(H, W) // 12
    
    # =================================
    # Left panel: Arrival time + metric field
    # =================================
    ax0 = axes[0]
    
    T_plot = T.copy()
    T_plot[T_plot > 1e5] = np.nan
    
    im0 = ax0.imshow(T_plot, origin='upper', cmap='viridis')
    
    # Black contours
    T_max = np.nanmax(T_plot)
    levels = np.arange(0, T_max + 10, 10)
    ax0.contour(T_plot, levels=levels, colors='black', linewidths=1.0, alpha=0.7)
    
    # Metric ellipses and drift arrows at same positions
    if G is not None:
        scale = step * 0.4
        # Positions for ellipses (and arrows)
        positions_i = list(range(step//2, H, step))
        positions_j = list(range(step//2, W, step))
        
        for i in positions_i:
            for j in positions_j:
                g = G[:, i, j]
                G_mat = np.array([[g[0], g[1]], [g[1], g[2]]])
                try:
                    G_inv = np.linalg.inv(G_mat)
                    vals, vecs = np.linalg.eigh(G_inv)
                    r = np.sqrt(np.maximum(vals, 1e-10))
                    rx, ry = scale * r[1], scale * r[0]
                    major_vec = vecs[:, 1] if r[1] >= r[0] else vecs[:, 0]
                    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
                    ellipse = Ellipse((j, i), width=2*rx, height=2*ry, angle=angle,
                                      fill=False, edgecolor='red', linewidth=1.0, alpha=0.8)
                    ax0.add_patch(ellipse)
                    
                    # Draw drift arrow at center of ellipse
                    if B is not None:
                        bx, by = B[0, i, j], B[1, i, j]
                        arrow_scale = step * 3  # Scale factor for arrow length
                        ax0.arrow(j, i, bx * arrow_scale, by * arrow_scale,
                                 head_width=1.5, head_length=0.8, fc='red', ec='red', alpha=0.8)
                except:
                    pass
    
    # Remove ticks
    ax0.set_xticks([])
    ax0.set_yticks([])
    
    # =================================
    # Right panel: Stencil map (legend only, no colorbar)
    # =================================
    ax1 = axes[1]
    
    colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#ffff33', '#a65628', '#f781bf',
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 8.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Mask source region (T~0) so it doesn't show as white
    # Note: stencil_idx may be smaller than T (excludes boundaries)
    stencil_plot = stencil_idx.astype(float)
    H_s, W_s = stencil_idx.shape
    T_cropped = T[1:H_s+1, 1:W_s+1] if stencil_idx.shape != T.shape else T
    stencil_plot[T_cropped < 1e-6] = np.nan
    
    ax1.imshow(stencil_plot, origin='upper', cmap=cmap, norm=norm)
    
    # Remove ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Legend for stencil labels (no colorbar needed)
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', linewidth=0.5,
                             label=STENCIL_LABELS[i]) 
                       for i in range(8)]
    fig.legend(handles=legend_elements, loc='lower center', 
               ncol=4, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# Keep alias for backward compatibility
visualize_stencils_with_arrival = visualize_stencils


def main():
    parser = argparse.ArgumentParser(description='Visualize stencil boundaries')
    parser.add_argument('--output', type=str, default='stencil_boundaries.pdf',
                       help='Output figure path')
    parser.add_argument('--N', type=int, default=100,
                       help='Grid size')
    parser.add_argument('--complexity', type=str, default=None,
                       choices=['radial', 'vortex', 'gradient', 'checkerboard', 'wave'],
                       help='Use spatially-varying metric: radial, vortex, gradient, checkerboard, or wave')
    parser.add_argument('--angle', type=float, default=30.0,
                       help='Metric rotation angle (degrees) - for uniform metric')
    parser.add_argument('--anisotropy', type=float, default=2.0,
                       help='Anisotropy ratio - for uniform metric')
    parser.add_argument('--drift', type=float, default=0.1,
                       help='Drift strength - for uniform metric')
    parser.add_argument('--drift-angle', type=float, default=45.0,
                       help='Drift direction (degrees) - for uniform metric')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stencil Boundary Visualization")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Grid size: {args.N} × {args.N}")
    
    if args.complexity:
        print(f"  Metric type: {args.complexity} (spatially-varying)")
    else:
        print(f"  Metric type: uniform")
        print(f"  Metric angle: {args.angle}°")
        print(f"  Anisotropy ratio: {args.anisotropy}")
        print(f"  Drift strength: {args.drift}")
        print(f"  Drift angle: {args.drift_angle}°")
    print()
    
    # Create problem and solve
    print("Solving forward problem...")
    if args.complexity:
        problem, T = create_complex_metric_problem(N=args.N, complexity=args.complexity)
        title = f"Stencil Boundaries ({args.complexity} metric, N={args.N})"
    else:
        problem, T = create_anisotropic_problem(
            N=args.N,
            angle_deg=args.angle,
            anisotropy=args.anisotropy,
            drift_strength=args.drift,
            drift_angle_deg=args.drift_angle,
        )
        title = f"Stencil Boundaries (anisotropy={args.anisotropy}, drift={args.drift})"
    
    # Identify stencils
    print("Identifying active stencils...")
    stencil_info = identify_active_stencils(T, problem)
    stencil_idx = stencil_info.stencil_idx[0].cpu().numpy()
    
    # Find boundaries
    print("Finding stencil boundaries...")
    boundaries = find_stencil_boundaries(stencil_idx)
    
    pct_boundary = boundaries.sum() / boundaries.size * 100
    print(f"  Boundary pixels: {boundaries.sum()} ({pct_boundary:.2f}%)")
    
    # Print stencil distribution
    print("\nStencil distribution:")
    for i in range(8):
        count = (stencil_idx == i).sum()
        pct = count / stencil_idx.size * 100
        print(f"  {STENCIL_LABELS[i]}: {count} ({pct:.1f}%)")
    
    # Visualize
    T_np = T[0].cpu().numpy()
    G_np = problem.G[0].cpu().numpy()
    B_np = problem.B[0].cpu().numpy()
    
    visualize_stencils(
        stencil_idx, boundaries, T_np, G=G_np, B=B_np,
        title=title,
        save_path=args.output,
        show=not args.no_show
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()