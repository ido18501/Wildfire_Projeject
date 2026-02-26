#!/usr/bin/env python3
"""
Visualize G and B metric components.

Usage:
    python plot_metric_components.py \
        --G_path G_fire170.npy \
        --B_path B_fire170.npy \
        --elev_path elev_fire170.npy \
        --npz results-scences/scene_0001_02614/test_predictions.npz \
        --fire_idx 170 \
        --output metric_components.jpg
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib
matplotlib.use('Agg')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--G_path', required=True)
    p.add_argument('--B_path', required=True)
    p.add_argument('--elev_path', default=None)
    p.add_argument('--npz', default=None)
    p.add_argument('--fire_idx', type=int, default=0)
    p.add_argument('--output', default='metric_components.jpg')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    G = np.load(args.G_path)   # (3, H, W)
    B = np.load(args.B_path)   # (2, H, W)

    g11, g12, g22 = G[0], G[1], G[2]
    b1, b2 = B[0], B[1]

    # Derived quantities
    tr = g11 + g22
    diff = g11 - g22
    disc = np.sqrt(diff**2 + 4*g12**2 + 1e-12)
    lmax = (tr + disc) / 2
    lmin = np.maximum((tr - disc) / 2, 1e-10)
    aniso = lmax / lmin
    bmag = np.sqrt(b1**2 + b2**2)
    bangle = np.degrees(np.arctan2(b2, b1))

    # Eigenvector angle (orientation of major axis)
    theta = 0.5 * np.degrees(np.arctan2(2*g12, g11 - g22))

    # Valid mask
    mask = None
    if args.npz:
        npz = np.load(args.npz, allow_pickle=True)
        vm = npz['valid_mask']
        mask = vm[args.fire_idx].astype(bool) if vm.ndim == 3 else vm.astype(bool)

    # Elevation
    elev = None
    if args.elev_path:
        elev = np.load(args.elev_path)

    # ── Figure: 4 rows × 3 cols ──
    # Row 1: g11, g12, g22
    # Row 2: λ_max, λ_min, anisotropy
    # Row 3: b1, b2, |B|
    # Row 4: eigenvector angle, drift angle, elevation

    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    fig.suptitle('Learned Metric Components — Scene 0001_02614, Fire 170',
                 fontsize=14, fontweight='bold', y=0.98)

    def show(ax, data, title, cmap='viridis', center_zero=False, mask_invalid=True):
        d = data.copy().astype(float)
        if mask is not None and mask_invalid:
            d[~mask] = np.nan

        if center_zero and np.nanmin(d) < 0 and np.nanmax(d) > 0:
            vmax = max(abs(np.nanmin(d)), abs(np.nanmax(d)))
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            im = ax.imshow(d, cmap=cmap, norm=norm, interpolation='nearest')
        else:
            im = ax.imshow(d, cmap=cmap, interpolation='nearest')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=7)

        # Stats
        vals = d[~np.isnan(d)] if np.any(~np.isnan(d)) else d.ravel()
        ax.text(0.02, 0.02, f'μ={np.nanmean(vals):.3f}\nσ={np.nanstd(vals):.3f}',
                transform=ax.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # Row 1: Raw G components
    show(axes[0, 0], g11, '$g_{11}$', cmap='magma')
    show(axes[0, 1], g12, '$g_{12}$', cmap='RdBu_r', center_zero=True)
    show(axes[0, 2], g22, '$g_{22}$', cmap='magma')

    # Row 2: Eigenvalues and anisotropy
    show(axes[1, 0], lmax, '$\\lambda_{\\max}$', cmap='inferno')
    show(axes[1, 1], lmin, '$\\lambda_{\\min}$', cmap='inferno')
    show(axes[1, 2], aniso, 'Anisotropy ($\\lambda_{\\max}/\\lambda_{\\min}$)', cmap='YlOrRd')

    # Row 3: Drift components
    show(axes[2, 0], b1, '$\\beta_1$ (row drift)', cmap='RdBu_r', center_zero=True)
    show(axes[2, 1], b2, '$\\beta_2$ (col drift)', cmap='RdBu_r', center_zero=True)
    show(axes[2, 2], bmag, '$|\\beta|$ (drift magnitude)', cmap='Reds')

    # Row 4: Angles and elevation
    show(axes[3, 0], theta, 'Eigenvector angle (°)', cmap='twilight')
    show(axes[3, 1], bangle, 'Drift angle (°)', cmap='twilight')
    if elev is not None:
        show(axes[3, 2], elev, 'Elevation (m)', cmap='terrain', mask_invalid=False)
    else:
        axes[3, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {args.output}")

    # ── Print summary ──
    vals = mask if mask is not None else np.ones_like(g11, dtype=bool)
    print(f"\nMetric summary (over valid region):")
    print(f"  g11:   {g11[vals].mean():.4f} ± {g11[vals].std():.4f}  [{g11[vals].min():.4f}, {g11[vals].max():.4f}]")
    print(f"  g12:   {g12[vals].mean():.4f} ± {g12[vals].std():.4f}  [{g12[vals].min():.4f}, {g12[vals].max():.4f}]")
    print(f"  g22:   {g22[vals].mean():.4f} ± {g22[vals].std():.4f}  [{g22[vals].min():.4f}, {g22[vals].max():.4f}]")
    print(f"  λ_max: {lmax[vals].mean():.4f} ± {lmax[vals].std():.4f}")
    print(f"  λ_min: {lmin[vals].mean():.4f} ± {lmin[vals].std():.4f}")
    print(f"  aniso: {aniso[vals].mean():.3f} ± {aniso[vals].std():.3f}  max={aniso[vals].max():.3f}")
    print(f"  |B|:   {bmag[vals].mean():.4f} ± {bmag[vals].std():.4f}")
    print(f"  B dir: {np.degrees(np.arctan2(b2[vals].mean(), b1[vals].mean())):.1f}°")


if __name__ == '__main__':
    main()
