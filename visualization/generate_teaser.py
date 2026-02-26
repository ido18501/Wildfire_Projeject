#!/usr/bin/env python3
"""
Generate SGP Representative Image (Teaser Figure)
===================================================

Three-panel composite (≥1500×1200 JPEG):
  (a) Observed fire spread — ground truth arrival times
  (b) Learned Randers-Finsler geometry — ellipses + drift over terrain
  (c) Predicted fire spread — model output with observed contours overlaid

Usage:
    python generate_teaser.py \
        --npz .../test_predictions.npz --fire_idx 184 \
        --G_path ./G_fire184.npy --B_path ./B_fire184.npy \
        --elev_path ./elev_fire184.npy \
        --output teaser.jpg --dpi 200

Author: Barak Gahtan
Date: February 2025
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LightSource
from typing import Tuple, Optional, Dict


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fire_data(npz_path, fire_idx=0, G_path=None, B_path=None, elev_path=None):
    data = np.load(npz_path, allow_pickle=True)
    result = {}

    # T_obs
    for k in ['true_T', 'T_obs', 't_obs', 'T_true']:
        if k in data:
            arr = data[k]
            result['T_obs'] = arr[fire_idx] if arr.ndim == 3 else arr
            break

    # T_pred
    for k in ['pred_T', 'T_pred', 't_pred']:
        if k in data:
            arr = data[k]
            result['T_pred'] = arr[fire_idx] if arr.ndim == 3 else arr
            break

    # valid_mask
    for k in ['valid_mask', 'mask']:
        if k in data:
            arr = data[k]
            result['valid_mask'] = (arr[fire_idx] if arr.ndim == 3 else arr).astype(bool)
            break
    else:
        result['valid_mask'] = np.isfinite(result['T_obs']) & (result['T_obs'] < 1e5)

    # G, B, elevation from external files
    result['G'] = np.load(G_path) if G_path and os.path.exists(G_path) else None
    result['B'] = np.load(B_path) if B_path and os.path.exists(B_path) else None
    result['elev'] = np.load(elev_path) if elev_path and os.path.exists(elev_path) else None

    # fire_id
    for k in ['event_ids', 'fire_ids']:
        if k in data and fire_idx < len(data[k]):
            result['fire_id'] = str(data[k][fire_idx]); break
    else:
        result['fire_id'] = f'fire_{fire_idx}'

    return result


# =============================================================================
# GEOMETRY
# =============================================================================

def metric_eigen(G):
    """From G (3, M, N), return (lambda_max, lambda_min, anisotropy, theta)."""
    g11, g12, g22 = G[0], G[1], G[2]
    trace = g11 + g22
    diff = g11 - g22
    disc = np.sqrt(diff**2 + 4*g12**2 + 1e-12)
    lmax = (trace + disc) / 2
    lmin = np.maximum((trace - disc) / 2, 1e-10)
    theta = 0.5 * np.arctan2(2*g12, diff + 1e-12)
    return lmax, lmin, lmax/lmin, theta


def bbox(mask, pad=15):
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    if not rows.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1 = np.where(rows)[0][[0,-1]]
    c0, c1 = np.where(cols)[0][[0,-1]]
    return (max(0, r0-pad), min(mask.shape[0], r1+pad+1),
            max(0, c0-pad), min(mask.shape[1], c1+pad+1))


# =============================================================================
# PANEL (b): METRIC VISUALIZATION
# =============================================================================

def draw_metric_panel(ax, G_c, B_c, valid_c, elev_c=None,
                      ellipse_step=None, ellipse_scale=0.32):
    """
    Panel (b): learned Randers-Finsler geometry over terrain.

    - Background: shaded-relief elevation (terrain context)
    - Ellipses: fixed cream color, shape shows metric anisotropy
    - Arrows: red, showing drift (wind) direction and magnitude
    - Inset: single enlarged Randers indicatrix
    """
    M, N = valid_c.shape

    # --- Terrain background ---
    if elev_c is not None:
        ls = LightSource(azdeg=315, altdeg=45)
        e = elev_c.copy().astype(float)
        e[~valid_c] = np.nan
        e_filled = np.nan_to_num(e, nan=np.nanmean(e))
        vlo = np.nanpercentile(e[valid_c], 1)
        vhi = np.nanpercentile(e[valid_c], 99)
        # Pure hillshade in grayscale, then tint with a muted colormap
        hillshade = ls.hillshade(e_filled, vert_exag=3.0)
        # Normalize elevation to [0,1] for coloring
        e_norm = np.clip((e_filled - vlo) / (vhi - vlo + 1e-8), 0, 1)
        # Use YlGn (yellow-green) for land elevation — no blues or reds
        cmap_elev = plt.cm.YlGn
        elev_rgb = cmap_elev(e_norm)[..., :3]
        # Blend hillshade with color: darken by hillshade
        hs3 = hillshade[..., np.newaxis]
        rgb = elev_rgb * (0.4 + 0.6 * hs3)
        # Fade invalid areas
        alpha = np.where(valid_c, 1.0, 0.12)[..., np.newaxis]
        rgb = rgb * alpha + 0.94 * (1 - alpha)
        ax.imshow(rgb, origin='lower', interpolation='bilinear')
    else:
        bg = np.ma.masked_where(~valid_c, np.ones_like(valid_c, dtype=float))
        ax.imshow(bg, cmap='Greys', origin='lower', alpha=0.15, vmin=0, vmax=2)

    # --- Ellipse spacing ---
    if ellipse_step is None:
        ellipse_step = max(10, min(M, N) // 22)

    # --- Metric ellipses ---
    lmax, lmin, aniso, theta = metric_eigen(G_c)
    ellipses = []
    offset = ellipse_step // 2
    margin = ellipse_step // 3  # smaller margin for boundary check
    for i in range(offset, M, ellipse_step):
        for j in range(offset, N, ellipse_step):
            if not valid_c[i, j]:
                continue
            # Skip if near invalid boundary (avoids half-ellipses at edges)
            r0c = max(0, i - margin)
            r1c = min(M, i + margin + 1)
            c0c = max(0, j - margin)
            c1c = min(N, j + margin + 1)
            if valid_c[r0c:r1c, c0c:c1c].mean() < 0.5:
                continue
            # Semi-axes: 1/sqrt(eigenvalue) — fast direction = long axis
            a = ellipse_scale * ellipse_step / np.sqrt(lmin[i, j])
            b = ellipse_scale * ellipse_step / np.sqrt(lmax[i, j])
            maxr = ellipse_step * 0.42
            minr = ellipse_step * 0.05
            a = np.clip(a, minr, maxr)
            b = np.clip(b, minr, maxr)
            # Width = major axis (lambda_min dir), so rotate by theta+90
            angle_deg = np.degrees(theta[i, j]) + 90.0
            ell = Ellipse((j, i), 2*a, 2*b, angle=angle_deg)
            ellipses.append(ell)

    if ellipses:
        pc = PatchCollection(ellipses, facecolor='#FFFDE7', edgecolor='#222222',
                             linewidths=0.8, alpha=0.85, zorder=3)
        ax.add_collection(pc)

    # --- Drift arrows (denser than before) ---
    if B_c is not None:
        arrow_step = max(ellipse_step, int(ellipse_step * 1.5))
        Xs, Ys, Us, Vs = [], [], [], []
        for i in range(arrow_step//2, M, arrow_step):
            for j in range(arrow_step//2, N, arrow_step):
                if not valid_c[i, j]:
                    continue
                # Boundary check for arrows too
                r0c = max(0, i - arrow_step//3)
                r1c = min(M, i + arrow_step//3 + 1)
                c0c = max(0, j - arrow_step//3)
                c1c = min(N, j + arrow_step//3 + 1)
                if valid_c[r0c:r1c, c0c:c1c].mean() < 0.4:
                    continue
                b1, b2 = B_c[0, i, j], B_c[1, i, j]
                mag = np.sqrt(b1**2 + b2**2)
                if mag < 1e-4:
                    continue
                Xs.append(j); Ys.append(i)
                Us.append(b2); Vs.append(b1)  # b2=x, b1=y for quiver

        if Xs:
            Us, Vs = np.array(Us), np.array(Vs)
            mags = np.sqrt(Us**2 + Vs**2)
            arrow_len = ellipse_step * 1.2
            s = arrow_len / (mags.max() + 1e-8)
            ax.quiver(Xs, Ys, Us*s, Vs*s,
                      color='#C62828', alpha=0.9,
                      angles='xy', scale_units='xy', scale=1.0,
                      width=0.005, headwidth=3.5,
                      headlength=4.0, headaxislength=3.0,
                      zorder=5)

    # --- Randers indicatrix inset ---
    _draw_indicatrix_inset(ax, G_c, B_c, valid_c)

    ax.set_xlim(0, N); ax.set_ylim(0, M)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

    # Legend entries
    ax.annotate('→ drift (wind)', xy=(0.02, 0.04), xycoords='axes fraction',
                fontsize=7.5, color='#C62828', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='gray', lw=0.5))
    # Label for indicatrix inset (placed above the inset box)
    ax.annotate('Indicatrix', xy=(0.79, 0.43), xycoords='axes fraction',
                fontsize=8, fontweight='bold', color='#333333', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))


def _draw_indicatrix_inset(ax, G_c, B_c, valid_c):
    """
    Inset: Randers indicatrix F(x,v) = 1 at a representative point.
    The Finsler unit ball is an ellipse offset by the drift vector B.
    """
    M, N = valid_c.shape
    ci, cj = M//2, N//2
    if not valid_c[ci, cj]:
        ys, xs = np.where(valid_c)
        if len(ys) == 0: return
        ci, cj = int(np.median(ys)), int(np.median(xs))

    g11, g12, g22 = G_c[0, ci, cj], G_c[1, ci, cj], G_c[2, ci, cj]
    b1 = B_c[0, ci, cj] if B_c is not None else 0
    b2 = B_c[1, ci, cj] if B_c is not None else 0

    # Larger inset in lower-right
    inset = ax.inset_axes([0.55, 0.02, 0.44, 0.50])
    inset.set_facecolor('white')
    inset.patch.set_alpha(0.93)

    # Eigendecompose
    t_arr = np.linspace(0, 2*np.pi, 200)
    tr = g11 + g22
    diff = g11 - g22
    disc = np.sqrt(diff**2 + 4*g12**2 + 1e-12)
    l1 = (tr + disc) / 2
    l2 = max((tr - disc) / 2, 1e-10)

    if abs(g12) > 1e-10:
        vx, vy = g12, l1 - g11
        n = np.sqrt(vx**2 + vy**2 + 1e-10)
        vx, vy = vx/n, vy/n
    else:
        vx, vy = 1.0, 0.0

    r1 = 1.0 / np.sqrt(l1)
    r2 = 1.0 / np.sqrt(l2)
    ex = r1 * np.cos(t_arr) * vx + r2 * np.sin(t_arr) * (-vy)
    ey = r1 * np.cos(t_arr) * vy + r2 * np.sin(t_arr) * vx

    # Normalize to fit nicely
    max_ext = max(abs(ex).max(), abs(ey).max()) * 1.6
    if max_ext < 1e-6: max_ext = 1.0
    ex_n = ex / max_ext
    ey_n = ey / max_ext

    # Randers shift (amplified for visibility)
    drift_mag = np.sqrt(b1**2 + b2**2)
    if drift_mag > 1e-6:
        shift_x = -b2 / drift_mag * 0.35
        shift_y = -b1 / drift_mag * 0.35
    else:
        shift_x, shift_y = 0, 0

    # Draw Riemannian (centered at origin)
    inset.plot(ex_n, ey_n, '--', color='#666666', lw=1.5, zorder=2)
    inset.fill(ex_n, ey_n, color='#E0E0E0', alpha=0.25)

    # Draw Randers (shifted)
    rx = ex_n + shift_x
    ry = ey_n + shift_y
    inset.plot(rx, ry, '-', color='#C62828', lw=2.0, zorder=3)
    inset.fill(rx, ry, color='#FFCDD2', alpha=0.25)

    # Drift arrow
    if abs(shift_x) + abs(shift_y) > 0.01:
        inset.annotate('', xy=(shift_x, shift_y), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.8),
                       zorder=4)
        inset.annotate('$\\beta$',
                       xy=(shift_x*0.5 + 0.08, shift_y*0.5 + 0.06),
                       fontsize=9, color='#C62828', fontweight='bold')

    # Origin
    inset.plot(0, 0, 'k+', ms=7, mew=1.2, zorder=5)

    # Inline labels (no legend box)
    # Place "Riemannian" at the top of the dashed ellipse
    top_idx = np.argmax(ey_n)
    inset.annotate('Riemannian', xy=(ex_n[top_idx], ey_n[top_idx] + 0.06),
                   fontsize=7, color='#444444', ha='center', fontstyle='italic')
    # Place "Randers" at the top of the red ellipse
    top_idx_r = np.argmax(ry)
    inset.annotate('Randers', xy=(rx[top_idx_r] + 0.05, ry[top_idx_r] + 0.06),
                   fontsize=7, color='#C62828', ha='center', fontweight='bold')

    inset.set_xlim(-1.0, 1.0); inset.set_ylim(-1.0, 1.0)
    inset.set_aspect('equal')
    inset.set_xticks([]); inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_linewidth(0.8); sp.set_color('#555555')


# =============================================================================
# MAIN FIGURE
# =============================================================================

def create_teaser(T_obs, T_pred, valid_mask, G=None, B=None, elev=None,
                  fire_id='', correlation=0.0, output_path='teaser.jpg',
                  padding=15, dpi=200, title='',
                  ellipse_step=None, ellipse_scale=0.32):

    has_metric = G is not None
    r0, r1, c0, c1 = bbox(valid_mask, padding)

    T_obs_c = T_obs[r0:r1, c0:c1]
    T_pred_c = T_pred[r0:r1, c0:c1]
    valid_c = valid_mask[r0:r1, c0:c1]
    M_c, N_c = valid_c.shape

    G_c = G[:, r0:r1, c0:c1] if has_metric else None
    B_c = B[:, r0:r1, c0:c1] if B is not None else None
    elev_c = elev[r0:r1, c0:c1] if elev is not None else None

    # Shared arrival time range
    valid_obs = T_obs_c[valid_c]
    valid_pred = T_pred_c[valid_c]
    finite_obs = valid_obs[(valid_obs > -1e5) & (valid_obs < 1e5)]
    finite_pred = valid_pred[(valid_pred > -1e5) & (valid_pred < 1e5)]
    vmin_t = min(finite_obs.min(), finite_pred.min())
    vmax_t = max(finite_obs.max(), finite_pred.max())

    # Ignition location
    T_ign = T_obs_c.copy(); T_ign[~valid_c] = 1e10
    ign_r, ign_c = np.unravel_index(T_ign.argmin(), T_ign.shape)

    # --- Figure layout ---
    n_panels = 3 if has_metric else 2
    aspect = M_c / N_c
    pw = 5.0
    ph = pw * aspect
    fig_w = pw * n_panels + 1.8
    fig_h = ph + 1.2
    fig_w = max(fig_w, 1500/dpi)
    fig_h = max(fig_h, 1200/dpi)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    ratios = [1, 1.08, 1] if has_metric else [1, 1]
    gs = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.28,
                           width_ratios=ratios,
                           left=0.03, right=0.97, bottom=0.05, top=0.89)

    cmap_arr = 'inferno'
    bg_col = '#f0f0f0'

    # =================================================================
    # Panel (a): Observed Fire Spread
    # =================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_col)
    T_obs_m = np.ma.masked_where(~valid_c, T_obs_c)
    im1 = ax1.imshow(T_obs_m, cmap=cmap_arr, origin='lower',
                     vmin=vmin_t, vmax=vmax_t, interpolation='bilinear')
    ax1.plot(ign_c, ign_r, 'w*', ms=14, mec='black', mew=0.8, zorder=5)
    ax1.set_title('(a) Observed Arrival Times', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_linewidth(0.5)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('Arrival Time (h)', fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    # =================================================================
    # Panel (b): Learned Randers-Finsler Geometry
    # =================================================================
    if has_metric:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(bg_col)
        draw_metric_panel(ax2, G_c, B_c, valid_c, elev_c,
                          ellipse_step=ellipse_step, ellipse_scale=ellipse_scale)
        ax2.set_title('(b) Learned Finsler Geometry', fontsize=13,
                       fontweight='bold', pad=10)
        pred_idx = 2
    else:
        pred_idx = 1

    # =================================================================
    # Panel (c): Predicted Fire Spread + Observed Contours
    # =================================================================
    ax3 = fig.add_subplot(gs[0, pred_idx])
    ax3.set_facecolor(bg_col)
    T_pred_m = np.ma.masked_where(~valid_c, T_pred_c)
    im3 = ax3.imshow(T_pred_m, cmap=cmap_arr, origin='lower',
                     vmin=vmin_t, vmax=vmax_t, interpolation='bilinear')

    # Observed contours
    n_cont = 8
    levels = np.linspace(vmin_t + 0.05*(vmax_t-vmin_t),
                         vmax_t - 0.05*(vmax_t-vmin_t), n_cont)
    T_cont = T_obs_c.copy(); T_cont[~valid_c] = np.nan
    try:
        ax3.contour(T_cont, levels=levels, colors='white',
                    linewidths=1.2, linestyles='dashed',
                    origin='lower', alpha=0.85)
    except Exception:
        pass

    ax3.plot(ign_c, ign_r, 'w*', ms=14, mec='black', mew=0.8, zorder=5)

    lbl = '(c)' if has_metric else '(b)'
    ttl = f'{lbl} Predicted Arrival Times'
    if correlation > 0:
        ttl += f'  ($\\rho$ = {correlation:.3f})'
    ax3.set_title(ttl, fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values(): sp.set_linewidth(0.5)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label('Arrival Time (h)', fontsize=9)
    cb3.ax.tick_params(labelsize=8)

    ax3.annotate('--- observed contours', xy=(0.03, 0.03),
                 xycoords='axes fraction', fontsize=7.5, color='white',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='black',
                           alpha=0.5, ec='none'))

    # =================================================================
    # Suptitle
    # =================================================================
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
    else:
        fig.suptitle('Differentiable Randers-Finsler Eikonal Solver '
                     'for Wildfire Propagation',
                     fontsize=14, fontweight='bold', y=0.96)

    # =================================================================
    # Save
    # =================================================================
    kw = dict(dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        kw['pil_kwargs'] = {'quality': 95}
    fig.savefig(output_path, **kw)
    plt.close()
    print(f"\nSaved: {output_path}")

    try:
        from PIL import Image
        w, h = Image.open(output_path).size
        print(f"Size: {w}×{h}px")
        print("✓ SGP OK" if w >= 1500 and h >= 1200 else
              f"✗ Below 1500×1200. Try --dpi {max(dpi+50, int(1500/fig_w)+10)}")
    except ImportError:
        pass


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description='Generate SGP teaser figure')
    p.add_argument('--npz', required=True)
    p.add_argument('--fire_idx', type=int, default=0)
    p.add_argument('--output', default='teaser.jpg')
    p.add_argument('--G_path', default=None)
    p.add_argument('--B_path', default=None)
    p.add_argument('--elev_path', default=None)
    p.add_argument('--padding', type=int, default=15)
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--title', default='')
    p.add_argument('--ellipse_step', type=int, default=None)
    p.add_argument('--ellipse_scale', type=float, default=0.32)
    p.add_argument('--correlation', type=float, default=None)
    p.add_argument('--inspect', action='store_true')
    args = p.parse_args()

    if args.inspect:
        data = np.load(args.npz, allow_pickle=True)
        for k in sorted(data.files):
            a = data[k]
            print(f"  {k:25s}  shape={str(getattr(a,'shape','?')):20s}  dtype={getattr(a,'dtype','?')}")
        return

    print(f"Loading fire_idx={args.fire_idx} from {args.npz}")
    fd = load_fire_data(args.npz, args.fire_idx, args.G_path, args.B_path, args.elev_path)

    T_obs, T_pred, vm = fd['T_obs'], fd['T_pred'], fd['valid_mask']
    G, B, elev = fd['G'], fd['B'], fd['elev']
    fid = fd['fire_id']

    print(f"  T_obs: {T_obs.shape},  T_pred: {T_pred.shape}")
    print(f"  valid: {vm.sum()}/{vm.size}")
    print(f"  G: {'yes '+str(G.shape) if G is not None else 'no'}")
    print(f"  B: {'yes '+str(B.shape) if B is not None else 'no'}")
    print(f"  elev: {'yes '+str(elev.shape) if elev is not None else 'no'}")

    # Correlation
    corr = args.correlation
    if corr is None:
        ov = T_obs[vm]; pv = T_pred[vm]
        both = (ov > 0) & (ov < 1e5) & (pv > 0) & (pv < 1e5)
        corr = np.corrcoef(ov[both], pv[both])[0,1] if both.sum() > 10 else 0
    print(f"  correlation: {corr:.4f}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    create_teaser(T_obs, T_pred, vm, G, B, elev, fid, corr, args.output,
                  args.padding, args.dpi, args.title,
                  args.ellipse_step, args.ellipse_scale)


if __name__ == '__main__':
    main()
