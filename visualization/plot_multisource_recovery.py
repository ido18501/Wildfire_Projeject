#!/usr/bin/env python3
"""
Multi-Source Metric Recovery Visualization
===========================================

Demonstrates how multiple ignition sources improve metric recovery:
- Few sources (5) → high error
- Many sources (50) → low error

Uses the actual differentiable eikonal solver and backward pass.

For SGP paper main text figure.

Usage:
    python visualize_multisource_recovery.py --output figures/multisource_recovery.pdf
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

# Add project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Imports handled via installed eikonal package

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.backward.implicit_diff import backward_pass

# Try numba solvers for speed
try:
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print("Numba solver not available, using pure Python")


def solve_forward(G, B, source_mask, max_iter=50):
    """Forward solve with best available solver."""
    problem = EikonalProblem(G, B, source_mask)
    if USE_NUMBA:
        T, _ = fast_sweeping_solve_numba(problem, max_iter=max_iter, verbose=False)
    else:
        T, _ = fast_sweeping_solve(problem, max_iter=max_iter, verbose=False)
    return T, problem


def compute_tv_regularization(G, B, lambda_G, lambda_B):
    """Total Variation regularization."""
    reg = 0.0
    if lambda_G > 0 and G is not None:
        for c in range(3):
            dx = G[:, c, :, 1:] - G[:, c, :, :-1]
            dy = G[:, c, 1:, :] - G[:, c, :-1, :]
            reg = reg + lambda_G * (dx.abs().sum() + dy.abs().sum())
    if lambda_B > 0 and B is not None:
        for c in range(2):
            dx = B[:, c, :, 1:] - B[:, c, :, :-1]
            dy = B[:, c, 1:, :] - B[:, c, :-1, :]
            reg = reg + lambda_B * (dx.abs().sum() + dy.abs().sum())
    return reg


class MetricRecoveryOptimizer:
    """Gradient descent optimizer using differentiable eikonal solver."""
    
    def __init__(self, M, N, recover_G=True, recover_B=True,
                 lambda_G=0.01, lambda_B=0.01):
        self.M, self.N = M, N
        self.recover_G, self.recover_B = recover_G, recover_B
        self.lambda_G, self.lambda_B = lambda_G, lambda_B
        
        # Initialize with uniform metric
        self.G = torch.ones(1, 3, M, N)
        self.G[:, 1, :, :] = 0
        self.B = torch.zeros(1, 2, M, N)
        
        self.history = {'loss': [], 'loss_data': [], 'loss_reg': []}
    
    def project_G(self):
        """Project G to SPD domain."""
        with torch.no_grad():
            self.G[:, 0] = torch.clamp(self.G[:, 0], min=0.1, max=10.0)
            self.G[:, 2] = torch.clamp(self.G[:, 2], min=0.1, max=10.0)
            g11, g12, g22 = self.G[:, 0], self.G[:, 1], self.G[:, 2]
            max_g12 = torch.sqrt(g11 * g22) * 0.95
            self.G[:, 1] = torch.clamp(g12, min=-max_g12, max=max_g12)
    
    def project_B(self):
        """Project B to causal domain."""
        with torch.no_grad():
            B_norm = torch.sqrt(self.B[:, 0]**2 + self.B[:, 1]**2)
            mask = B_norm > 0.9
            if mask.any():
                self.B[:, 0][mask] *= 0.9 / B_norm[mask]
                self.B[:, 1][mask] *= 0.9 / B_norm[mask]
    
    def fit_multi_source(self, sources, T_obs_list, obs_mask_list,
                         n_iter=500, lr=0.1, verbose=True):
        """
        Fit parameters using observations from multiple sources.
        Uses actual backward_pass with implicit differentiation.
        """
        N = self.N
        
        for it in range(n_iter):
            total_loss_data = 0.0
            total_dL_dG = torch.zeros_like(self.G)
            total_dL_dB = torch.zeros_like(self.B)
            
            # Accumulate gradients from all sources
            for (si, sj), T_obs, obs_mask in zip(sources, T_obs_list, obs_mask_list):
                # Source mask
                source_mask = torch.zeros(1, N, N, dtype=torch.bool)
                source_mask[0, max(0,si-1):si+2, max(0,sj-1):sj+2] = True
                
                # Forward solve
                T, problem = solve_forward(self.G, self.B, source_mask)
                
                # Data loss
                residuals = T[obs_mask] - T_obs[obs_mask]
                loss_data = 0.5 * (residuals**2).sum()
                total_loss_data += loss_data.item()
                
                # Backward pass (implicit differentiation)
                dL_dT = torch.zeros_like(T)
                dL_dT[obs_mask] = residuals
                dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
                
                total_dL_dG += dL_dG
                total_dL_dB += dL_dB
            
            # Regularization
            loss_reg = compute_tv_regularization(
                self.G, self.B if self.recover_B else None,
                self.lambda_G, self.lambda_B
            )
            
            total_loss = total_loss_data + loss_reg.item()
            self.history['loss'].append(total_loss)
            self.history['loss_data'].append(total_loss_data)
            self.history['loss_reg'].append(loss_reg.item())
            
            if verbose and (it % 50 == 0 or it == n_iter - 1):
                print(f"    Iter {it}: loss={total_loss:.2f} "
                      f"(data={total_loss_data:.2f}, reg={loss_reg.item():.2f})")
            
            # Add TV gradient (finite diff)
            if self.lambda_G > 0:
                eps = 1e-6
                for c in range(3):
                    G_p = self.G.clone(); G_p[:, c] += eps
                    G_m = self.G.clone(); G_m[:, c] -= eps
                    reg_p = compute_tv_regularization(G_p, None, self.lambda_G, 0)
                    reg_m = compute_tv_regularization(G_m, None, self.lambda_G, 0)
                    total_dL_dG[:, c] += (reg_p - reg_m) / (2 * eps)
            
            # Update
            with torch.no_grad():
                if self.recover_G:
                    self.G -= lr * total_dL_dG
                if self.recover_B:
                    self.B -= lr * total_dL_dB
                self.project_G()
                if self.recover_B:
                    self.project_B()
            
            # LR decay
            if it > 0 and it % 100 == 0:
                lr *= 0.8
        
        return {'final_loss': total_loss}


def create_gt_metric(N):
    """Create spatially-varying anisotropic metric with drift."""
    y, x = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    x = x.float() / N
    y = y.float() / N
    
    # Spatially-varying angle
    angle = np.pi * (0.3 * torch.sin(2 * np.pi * x) * torch.cos(np.pi * y) + 
                     0.2 * torch.cos(3 * np.pi * x * y))
    c, s = torch.cos(angle), torch.sin(angle)
    
    # Spatially-varying anisotropy
    base_speed = 0.3 + 0.1 * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    anisotropy = 1.8 + 0.6 * torch.cos(2 * np.pi * x) * torch.sin(2 * np.pi * y)
    
    G_gt = torch.zeros(1, 3, N, N)
    G_gt[0, 0] = (c**2 * anisotropy + s**2) * base_speed
    G_gt[0, 1] = c * s * (anisotropy - 1) * base_speed
    G_gt[0, 2] = (s**2 * anisotropy + c**2) * base_speed
    
    # Drift
    B_gt = torch.zeros(1, 2, N, N)
    B_gt[0, 0] = 0.06 * torch.sin(np.pi * x) * torch.cos(0.5 * np.pi * y)
    B_gt[0, 1] = 0.04 * torch.cos(0.5 * np.pi * x) * torch.sin(np.pi * y)
    
    return G_gt, B_gt


def generate_sources(N, n_sources, margin=None, seed=42):
    """Generate random source locations."""
    if margin is None:
        margin = N // 6
    np.random.seed(seed)
    return [(np.random.randint(margin, N-margin), 
             np.random.randint(margin, N-margin)) for _ in range(n_sources)]


def generate_observations(G, B, sources, obs_density=0.07, seed=42):
    """Generate arrival times from multiple sources."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, N, _ = G.shape
    
    T_list, obs_list = [], []
    for idx, (si, sj) in enumerate(sources):
        source_mask = torch.zeros(1, N, N, dtype=torch.bool)
        source_mask[0, max(0,si-1):si+2, max(0,sj-1):sj+2] = True
        
        T, _ = solve_forward(G, B, source_mask)
        T_list.append(T.clone())
        
        torch.manual_seed(seed + idx)
        obs_mask = (torch.rand(1, N, N) < obs_density) & (T < 1e5) & ~source_mask
        obs_list.append(obs_mask)
    
    return T_list, obs_list


def compute_errors(G_gt, B_gt, G_rec, B_rec):
    """Compute component-wise errors."""
    errors = {}
    G_diff = (G_rec - G_gt).abs()
    errors['G_total'] = (G_diff.sum() / (G_gt.abs().sum() + 1e-8)).item() * 100
    
    for i, name in enumerate(['g11', 'g12', 'g22']):
        diff = (G_rec[0, i] - G_gt[0, i]).abs()
        errors[name] = (diff.mean() / (G_gt[0, i].abs().mean() + 1e-8)).item() * 100
    
    B_diff = (B_rec - B_gt).abs()
    errors['B_total'] = (B_diff.sum() / (B_gt.abs().sum() + 1e-8)).item() * 100
    
    for i, name in enumerate(['b1', 'b2']):
        diff = (B_rec[0, i] - B_gt[0, i]).abs()
        errors[name] = (diff.mean() / (B_gt[0, i].abs().mean() + 1e-8)).item() * 100
    
    return errors


def draw_ellipse(ax, G_np, i, j, scale, color, linestyle='-', alpha=0.8):
    """Draw single metric ellipse."""
    g = G_np[:, i, j]
    G_mat = np.array([[g[0], g[1]], [g[1], g[2]]])
    try:
        G_inv = np.linalg.inv(G_mat)
        vals, vecs = np.linalg.eigh(G_inv)
        r = np.sqrt(np.maximum(vals, 1e-10))
        rx, ry = scale * r[1], scale * r[0]
        major_vec = vecs[:, 1] if r[1] >= r[0] else vecs[:, 0]
        angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
        ellipse = Ellipse((j, i), width=2*rx, height=2*ry, angle=angle,
                          fill=False, edgecolor=color, linewidth=1.0, 
                          alpha=alpha, linestyle=linestyle)
        ax.add_patch(ellipse)
    except:
        pass


def plot_overlay(ax, G_gt, G_rec, title, step=10):
    """Overlay GT (red) and reconstructed (blue) ellipses."""
    G_gt_np = G_gt[0].numpy()
    G_rec_np = G_rec[0].numpy()
    _, H, W = G_gt_np.shape
    
    # Error background
    G_diff = np.sqrt((G_gt_np[0] - G_rec_np[0])**2 + 
                     (G_gt_np[1] - G_rec_np[1])**2 + 
                     (G_gt_np[2] - G_rec_np[2])**2)
    ax.imshow(G_diff, origin='upper', cmap='Reds', alpha=0.5)
    
    scale = step * 0.35
    for i in range(step//2, H, step):
        for j in range(step//2, W, step):
            draw_ellipse(ax, G_gt_np, i, j, scale, 'red', '-')
            draw_ellipse(ax, G_rec_np, i, j, scale, 'blue', '--')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


def create_figure(G_gt, B_gt, G_5, B_5, err_5, src_5, T_5,
                  G_50, B_50, err_50, src_50, T_50, save_path=None, show=True):
    """Create main figure."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.3)
    
    N = G_gt.shape[2]
    step = max(8, N // 10)
    
    # (a) Ground truth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f5f5f5')
    G_gt_np, B_gt_np = G_gt[0].numpy(), B_gt[0].numpy()
    H, W = N, N
    scale = step * 0.35
    for i in range(step//2, H, step):
        for j in range(step//2, W, step):
            draw_ellipse(ax1, G_gt_np, i, j, scale, 'darkred')
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ax1.quiver(X[::step, ::step], Y[::step, ::step],
               B_gt_np[0, ::step, ::step], B_gt_np[1, ::step, ::step],
               color='darkblue', alpha=0.7, scale=2.0)
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
    ax1.set_title('(a) Ground Truth', fontsize=11, fontweight='bold')
    
    # (b) 5 sources
    ax2 = fig.add_subplot(gs[0, 1])
    T_plot = T_5[0][0].numpy().copy()
    T_plot[T_plot > 1e5] = np.nan
    ax2.imshow(T_plot, origin='upper', cmap='viridis', alpha=0.7)
    for si, sj in src_5:
        ax2.plot(sj, si, 'r*', markersize=12, markeredgecolor='white')
    for idx in range(min(3, len(T_5))):
        T_src = T_5[idx][0].numpy().copy()
        T_src[T_src > 1e5] = np.nan
        levels = np.linspace(np.nanmin(T_src), np.nanmax(T_src), 8)
        ax2.contour(T_src, levels=levels, colors='black', linewidths=0.6, alpha=0.6)
    ax2.set_title('(b) 5 Sources', fontsize=11, fontweight='bold')
    
    # (c) 50 sources
    ax3 = fig.add_subplot(gs[0, 2])
    T_plot = T_50[0][0].numpy().copy()
    T_plot[T_plot > 1e5] = np.nan
    ax3.imshow(T_plot, origin='upper', cmap='viridis', alpha=0.7)
    for si, sj in src_50:
        ax3.plot(sj, si, 'r*', markersize=4, markeredgecolor='white', markeredgewidth=0.2)
    for idx in range(min(5, len(T_50))):
        T_src = T_50[idx][0].numpy().copy()
        T_src[T_src > 1e5] = np.nan
        levels = np.linspace(np.nanmin(T_src), np.nanmax(T_src), 6)
        ax3.contour(T_src, levels=levels, colors='black', linewidths=0.4, alpha=0.4)
    ax3.set_title('(c) 50 Sources', fontsize=11, fontweight='bold')
    
    # (d) 5 sources overlay
    ax4 = fig.add_subplot(gs[1, 0])
    plot_overlay(ax4, G_gt, G_5, f"(d) 5 Sources: Error {err_5['G_total']:.1f}%", step)
    
    # (e) 50 sources overlay
    ax5 = fig.add_subplot(gs[1, 1])
    plot_overlay(ax5, G_gt, G_50, f"(e) 50 Sources: Error {err_50['G_total']:.1f}%", step)
    
    # (f) Error bar chart
    ax6 = fig.add_subplot(gs[1, 2])
    comps = ['$g_{11}$', '$g_{12}$', '$g_{22}$', '$b_1$', '$b_2$', 'Total']
    keys = ['g11', 'g12', 'g22', 'b1', 'b2', 'G_total']
    e5 = [err_5[k] for k in keys]
    e50 = [err_50[k] for k in keys]
    x = np.arange(len(comps))
    w = 0.35
    ax6.bar(x - w/2, e5, w, label='5 Sources', color='#e74c3c', edgecolor='black')
    ax6.bar(x + w/2, e50, w, label='50 Sources', color='#3498db', edgecolor='black')
    ax6.set_xticks(x); ax6.set_xticklabels(comps)
    ax6.set_ylabel('Relative Error (%)')
    ax6.set_title('(f) Error by Component', fontsize=11, fontweight='bold')
    ax6.legend(); ax6.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='multisource_recovery.pdf')
    parser.add_argument('--N', type=int, default=60)
    parser.add_argument('--n-sources-few', type=int, default=5)
    parser.add_argument('--n-sources-many', type=int, default=50)
    parser.add_argument('--n-iter', type=int, default=200)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Source Recovery (Differentiable Solver)")
    print("=" * 60)
    print(f"Grid: {args.N}x{args.N}, Few: {args.n_sources_few}, Many: {args.n_sources_many}")
    print(f"Iterations: {args.n_iter}, Numba: {USE_NUMBA}\n")
    
    # Ground truth
    print("Creating ground truth...")
    G_gt, B_gt = create_gt_metric(args.N)
    
    # === 5 SOURCES ===
    print(f"\n--- {args.n_sources_few} SOURCES ---")
    src_5 = generate_sources(args.N, args.n_sources_few, seed=42)
    T_5, obs_5 = generate_observations(G_gt, B_gt, src_5, seed=42)
    print(f"  Observations: {sum(m.sum().item() for m in obs_5):.0f}")
    
    print("  Optimizing...")
    opt_5 = MetricRecoveryOptimizer(args.N, args.N, lambda_G=0.02, lambda_B=0.02)
    opt_5.fit_multi_source(src_5, T_5, obs_5, n_iter=args.n_iter, lr=0.05)
    err_5 = compute_errors(G_gt, B_gt, opt_5.G, opt_5.B)
    print(f"  Error: G={err_5['G_total']:.1f}%, B={err_5['B_total']:.1f}%")
    
    # === 50 SOURCES ===
    print(f"\n--- {args.n_sources_many} SOURCES ---")
    src_50 = generate_sources(args.N, args.n_sources_many, seed=123)
    T_50, obs_50 = generate_observations(G_gt, B_gt, src_50, seed=123)
    print(f"  Observations: {sum(m.sum().item() for m in obs_50):.0f}")
    
    print("  Optimizing...")
    opt_50 = MetricRecoveryOptimizer(args.N, args.N, lambda_G=0.02, lambda_B=0.02)
    opt_50.fit_multi_source(src_50, T_50, obs_50, n_iter=args.n_iter, lr=0.05)
    err_50 = compute_errors(G_gt, B_gt, opt_50.G, opt_50.B)
    print(f"  Error: G={err_50['G_total']:.1f}%, B={err_50['B_total']:.1f}%")
    
    # Figure
    print("\nCreating figure...")
    create_figure(G_gt, B_gt, opt_5.G, opt_5.B, err_5, src_5, T_5,
                  opt_50.G, opt_50.B, err_50, src_50, T_50,
                  save_path=args.output, show=not args.no_show)
    
    # Summary
    print("\n" + "=" * 55)
    print(f"{'Component':<10} | {'5 Src':>10} | {'50 Src':>10} | {'Improv':>10}")
    print("-" * 55)
    for comp, k in [('g11','g11'), ('g12','g12'), ('g22','g22'), 
                    ('b1','b1'), ('b2','b2'), ('Total G','G_total')]:
        imp = (err_5[k] - err_50[k]) / (err_5[k] + 1e-8) * 100
        print(f"{comp:<10} | {err_5[k]:>9.1f}% | {err_50[k]:>9.1f}% | {imp:>9.1f}%")
    print("Done!")


if __name__ == '__main__':
    main()
