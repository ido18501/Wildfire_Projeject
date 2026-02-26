#!/usr/bin/env python3
"""
Visualize covariates → learned metric mapping + colored ellipse field.

Row 1: Spatial covariates (elevation, fuel, vegetation)
Row 2: Learned metric quantities (λ_max, anisotropy, |β|)
Row 3: Colored ellipse field with drift arrows on terrain

Usage:
    python plot_covariates_vs_metric.py \
        --npz_data fire_data_0001_02544.npz \
        --output covariates_vs_metric.jpg

    # Or with separate files:
    python plot_covariates_vs_metric.py \
        --G_path G_fire170.npy \
        --B_path B_fire170.npy \
        --elev_path elev_fire170.npy \
        --npz results-scences/scene_0001_02614/test_predictions.npz \
        --fire_idx 170 \
        --data_root /path/to/sim2real_data \
        --scene_id 0001_02614 \
        --event_id 0001_02544 \
        --output covariates_vs_metric.jpg
"""

import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
matplotlib.use('Agg')


def load_from_package(npz_path):
    """Load everything from a single package npz."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        'G': d['G'],
        'B': d['B'],
        'elevation': d['elevation'],
        'true_T': d['true_T'],
        'pred_T': d['pred_T'],
        'valid_mask': d['valid_mask'].astype(bool),
        'fuel': d.get('fuel_fbfm13', d.get('fuel_FBFM13', None)),
        'vegetation': d.get('veg_existing_vegetation_type',
                           d.get('veg_Existing_Vegetation_Type', None)),
        'slope': d.get('slope', None),
    }


def load_from_separate(args):
    """Load from separate files + data_root."""
    G = np.load(args.G_path)
    B = np.load(args.B_path)
    elev = np.load(args.elev_path)

    npz = np.load(args.npz, allow_pickle=True)
    vm = npz['valid_mask']
    mask = vm[args.fire_idx].astype(bool) if vm.ndim == 3 else vm.astype(bool)
    true_T = npz['true_T'][args.fire_idx] if npz['true_T'].ndim == 3 else npz['true_T']
    pred_T = npz['pred_T'][args.fire_idx] if npz['pred_T'].ndim == 3 else npz['pred_T']

    # Load raw covariates
    fuel, veg, slope = None, None, None
    if args.data_root and args.scene_id and args.event_id:
        from experiments.wildfire.sim2real_loader import Sim2RealFireLoader
        loader = Sim2RealFireLoader(args.data_root)
        raw = loader.load_scenario(args.scene_id, args.event_id)
        fuel_key = 'fbfm13' if 'fbfm13' in raw['fuel'] else list(raw['fuel'].keys())[0]
        fuel = raw['fuel'][fuel_key].astype(np.float32)
        veg_key = ('existing_vegetation_type' if 'existing_vegetation_type' in raw['vegetation']
                   else list(raw['vegetation'].keys())[0])
        veg = raw['vegetation'][veg_key].astype(np.float32)
        slope = raw['topography'].get('slope', None)
        if slope is not None:
            slope = slope.astype(np.float32)

    return {
        'G': G, 'B': B, 'elevation': elev,
        'true_T': true_T, 'pred_T': pred_T, 'valid_mask': mask,
        'fuel': fuel, 'vegetation': veg, 'slope': slope,
    }


def compute_metric_fields(G, B, mask):
    """Derive eigenvalues, anisotropy, drift from G and B."""
    g11, g12, g22 = G[0], G[1], G[2]
    b1, b2 = B[0], B[1]

    tr = g11 + g22
    diff = g11 - g22
    disc = np.sqrt(diff**2 + 4*g12**2 + 1e-12)
    lmax = (tr + disc) / 2
    lmin = np.maximum((tr - disc) / 2, 1e-10)
    aniso = lmax / lmin
    bmag = np.sqrt(b1**2 + b2**2)

    # Eigenvector of lambda_max
    theta = 0.5 * np.arctan2(2*g12, g11 - g22)

    return {
        'lmax': lmax, 'lmin': lmin, 'aniso': aniso,
        'bmag': bmag, 'theta': theta,
        'b1': b1, 'b2': b2,
    }


def crop_to_valid(mask, pad=5):
    """Find bounding box of valid region with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return slice(None), slice(None)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W = mask.shape
    r0 = max(0, r0 - pad)
    r1 = min(H - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(W - 1, c1 + pad)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def draw_scalar_panel(ax, data, mask, title, cmap='viridis', crop=None):
    """Draw a scalar field with invalid pixels grayed out."""
    d = data.copy().astype(float)
    if crop:
        d = d[crop[0], crop[1]]
        m = mask[crop[0], crop[1]]
    else:
        m = mask

    d[~m] = np.nan
    im = ax.imshow(d, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([])

    # Colorbar
    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, aspect=20)
    cb.ax.tick_params(labelsize=7)

    # Stats annotation
    vals = d[~np.isnan(d)]
    if len(vals) > 0:
        ax.text(0.03, 0.04,
                f'μ={np.mean(vals):.2f}\nσ={np.std(vals):.2f}',
                transform=ax.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='gray', lw=0.5))
    return im


def draw_ellipse_field(ax, data, metric, mask, crop):
    """Draw colored ellipses + drift arrows on terrain."""
    elev = data['elevation'].copy().astype(float)
    elev_c = elev[crop[0], crop[1]]
    mask_c = mask[crop[0], crop[1]]
    M, N = mask_c.shape

    # Hillshade background
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=35)
    hillshade = ls.hillshade(elev_c, vert_exag=3.0)

    # Terrain + hillshade blend
    elev_display = elev_c.copy()
    elev_display[~mask_c] = np.nan
    norm_e = Normalize(vmin=np.nanmin(elev_display), vmax=np.nanmax(elev_display))
    terrain_rgb = plt.cm.YlGn(norm_e(elev_display))[:, :, :3]
    blended = np.zeros_like(terrain_rgb)
    for i in range(3):
        blended[:, :, i] = terrain_rgb[:, :, i] * (0.5 + 0.5 * hillshade)
    blended[~mask_c] = 0.85
    ax.imshow(blended, interpolation='nearest')

    # Crop metric fields
    aniso_c = metric['aniso'][crop[0], crop[1]]
    lmax_c = metric['lmax'][crop[0], crop[1]]
    lmin_c = metric['lmin'][crop[0], crop[1]]
    theta_c = metric['theta'][crop[0], crop[1]]
    b1_c = metric['b1'][crop[0], crop[1]]
    b2_c = metric['b2'][crop[0], crop[1]]
    bmag_c = metric['bmag'][crop[0], crop[1]]

    # Ellipse grid
    step = max(4, min(M, N) // 14)
    aniso_valid = aniso_c[mask_c]
    norm_a = Normalize(vmin=max(1.0, aniso_valid.min()),
                       vmax=min(aniso_valid.max(), np.percentile(aniso_valid, 98)))
    cmap_a = plt.cm.YlOrRd

    scale = 0.42
    max_r = 0.45 * step

    for i in range(step // 2, M, step):
        for j in range(step // 2, N, step):
            if not mask_c[i, j]:
                continue
            # Check neighborhood
            r0 = max(0, i - step // 3)
            r1 = min(M, i + step // 3)
            c0 = max(0, j - step // 3)
            c1 = min(N, j + step // 3)
            if mask_c[r0:r1, c0:c1].mean() < 0.5:
                continue

            lmx = lmax_c[i, j]
            lmn = lmin_c[i, j]
            th = theta_c[i, j]
            an = aniso_c[i, j]

            r_major = scale * step / np.sqrt(lmn + 1e-10)
            r_minor = scale * step / np.sqrt(lmx + 1e-10)
            r_major = min(r_major, max_r)
            r_minor = min(r_minor, max_r)

            color = cmap_a(norm_a(an))
            ell = Ellipse((j, i), width=2*r_major, height=2*r_minor,
                          angle=-np.degrees(th),
                          facecolor=color, edgecolor='k',
                          linewidth=0.6, alpha=0.8, zorder=2)
            ax.add_patch(ell)

    # Drift arrows
    arrow_step = int(1.5 * step)
    for i in range(arrow_step // 2, M, arrow_step):
        for j in range(arrow_step // 2, N, arrow_step):
            if not mask_c[i, j]:
                continue
            r0 = max(0, i - step // 2)
            r1 = min(M, i + step // 2)
            c0 = max(0, j - step // 2)
            c1 = min(N, j + step // 2)
            if mask_c[r0:r1, c0:c1].mean() < 0.5:
                continue

            bm = bmag_c[i, j]
            if bm < 0.01:
                continue
            length = 1.0 * step
            dx = -b2_c[i, j] / (bm + 1e-10) * length
            dy = -b1_c[i, j] / (bm + 1e-10) * length
            ax.annotate('', xy=(j + dx, i + dy), xytext=(j, i),
                        arrowprops=dict(arrowstyle='->', color='#C62828',
                                        lw=1.8, mutation_scale=12),
                        zorder=3)

    ax.set_xlim(0, N); ax.set_ylim(M, 0)
    ax.set_xticks([]); ax.set_yticks([])

    # Colorbar for anisotropy
    sm = plt.cm.ScalarMappable(norm=norm_a, cmap=cmap_a)
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cb.set_label('Anisotropy', fontsize=9)
    cb.ax.tick_params(labelsize=7)

    # Legend
    ax.annotate('→ drift (wind)', xy=(0.02, 0.04), xycoords='axes fraction',
                fontsize=8, color='#C62828', fontweight='bold',
                bbox=dict(fc='white', alpha=0.85, ec='gray', lw=0.5, pad=2))


def main():
    p = argparse.ArgumentParser()
    # Single-file mode
    p.add_argument('--npz_data', default=None, help='Package npz from package_data.py')
    # Separate-file mode
    p.add_argument('--G_path', default=None)
    p.add_argument('--B_path', default=None)
    p.add_argument('--elev_path', default=None)
    p.add_argument('--npz', default=None)
    p.add_argument('--fire_idx', type=int, default=0)
    p.add_argument('--data_root', default=None)
    p.add_argument('--scene_id', default=None)
    p.add_argument('--event_id', default=None)
    # Output
    p.add_argument('--output', default='covariates_vs_metric.jpg')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    # Load data
    if args.npz_data:
        data = load_from_package(args.npz_data)
    else:
        data = load_from_separate(args)

    G, B = data['G'], data['B']
    mask = data['valid_mask']
    metric = compute_metric_fields(G, B, mask)

    # Crop to valid region
    rs, cs = crop_to_valid(mask, pad=8)
    crop = (rs, cs)

    # ── Build figure ──
    fig = plt.figure(figsize=(15, 16))
    fig.suptitle('Learned Finsler Geometry from Landscape Covariates',
                 fontsize=14, fontweight='bold', y=0.98)

    # GridSpec: 3 rows — top two are 3-column, bottom is full width
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.4],
                          hspace=0.25, wspace=0.25,
                          left=0.05, right=0.95, top=0.94, bottom=0.03)

    # ── Row 1: Covariates ──
    ax_elev = fig.add_subplot(gs[0, 0])
    draw_scalar_panel(ax_elev, data['elevation'], mask, 'Elevation (m)',
                      cmap='terrain', crop=crop)

    ax_fuel = fig.add_subplot(gs[0, 1])
    if data['fuel'] is not None:
        draw_scalar_panel(ax_fuel, data['fuel'], mask, 'Fuel Model (FBFM13)',
                          cmap='YlOrBr', crop=crop)
    else:
        ax_fuel.text(0.5, 0.5, 'Fuel data\nnot available',
                     ha='center', va='center', transform=ax_fuel.transAxes)
        ax_fuel.set_title('Fuel Model (FBFM13)', fontsize=11, fontweight='bold')

    ax_veg = fig.add_subplot(gs[0, 2])
    if data['vegetation'] is not None:
        draw_scalar_panel(ax_veg, data['vegetation'], mask, 'Vegetation Type',
                          cmap='Greens', crop=crop)
    else:
        ax_veg.text(0.5, 0.5, 'Vegetation data\nnot available',
                    ha='center', va='center', transform=ax_veg.transAxes)
        ax_veg.set_title('Vegetation Type', fontsize=11, fontweight='bold')

    # Row label
    fig.text(0.02, 0.82, 'Spatial\nCovariates', fontsize=11, fontweight='bold',
             va='center', ha='center', rotation=90, color='#333333')

    # ── Row 2: Learned metric quantities ──
    ax_lmax = fig.add_subplot(gs[1, 0])
    draw_scalar_panel(ax_lmax, metric['lmax'], mask,
                      '$\\lambda_{\\max}$ (max propagation cost)',
                      cmap='inferno', crop=crop)

    ax_aniso = fig.add_subplot(gs[1, 1])
    draw_scalar_panel(ax_aniso, metric['aniso'], mask,
                      'Anisotropy ($\\lambda_{\\max} / \\lambda_{\\min}$)',
                      cmap='YlOrRd', crop=crop)

    ax_bmag = fig.add_subplot(gs[1, 2])
    draw_scalar_panel(ax_bmag, metric['bmag'], mask,
                      'Drift magnitude $|\\beta|$',
                      cmap='Reds', crop=crop)

    # Row label
    fig.text(0.02, 0.52, 'Learned\nMetric', fontsize=11, fontweight='bold',
             va='center', ha='center', rotation=90, color='#333333')

    # ── Row 3: Ellipse field ──
    ax_ell = fig.add_subplot(gs[2, :])
    draw_ellipse_field(ax_ell, data, metric, mask, crop)
    ax_ell.set_title('Randers-Finsler Metric Field (ellipses = $G$, arrows = $\\beta$, color = anisotropy)',
                     fontsize=12, fontweight='bold', pad=8)

    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
