"""
Category B: Gradient Verification
=================================

9 experiments verifying the backward pass:

B1: FD vs Implicit (Isotropic) - Basic gradient check
B2: FD vs Implicit (Anisotropic) - All 3 G components
B3: FD vs Implicit (Drift) - B gradients
B4: Ring Loss Pattern - Adjoint propagation visualization
B5: Gradient Magnitude Decay - Spatial gradient behavior
B6: Stencil Map Visualization - Show stencil structure
B7: Stencil Boundary Visualization - Where is ∇ undefined?
B8: Random Direction Agreement - Measure-zero argument
B9: Gradient Under Perturbation - Stability check
"""

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import sys
import os

from experiment_base import (
    Experiment, ExperimentResult, register_experiment, DEVICE,
    create_metric_from_eigenvalues, finite_difference_gradient,
    plot_arrival_time, plot_stencil_map, plot_error_map
)

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.backward.implicit_diff import backward_pass, identify_active_stencils, UpdateType


# =============================================================================
# B1: FD VS IMPLICIT (ISOTROPIC)
# =============================================================================

@register_experiment
class B1_FD_Isotropic(Experiment):
    """Verify gradients on isotropic problem via finite differences."""
    
    name = "B1_fd_vs_implicit_isotropic"
    category = "B_gradient_verification"
    description = "Verify backward pass gradients against FD (isotropic)"
    
    def __init__(self, N: int = 50, n_test_points: int = 20, eps: float = 1e-5, device: str = None):
        super().__init__(device)
        self.N, self.n_test_points, self.eps = N, n_test_points, eps
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        # Target point loss
        target_i, target_j = N//2 + 15, N//2 + 10
        dL_dT = torch.zeros_like(T)
        dL_dT[0, target_i, target_j] = 1.0
        
        dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
        
        # Select test points
        np.random.seed(42)
        test_points = []
        while len(test_points) < self.n_test_points:
            i, j = np.random.randint(5, N-5, 2)
            if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                test_points.append((i, j))
        
        results_G, results_B = [], []
        
        for i, j in test_points:
            for c in [0, 2]:  # g11, g22 for isotropic
                fd = finite_difference_gradient(problem, dL_dT, 'G', (c, i, j), self.eps)
                impl = dL_dG[0, c, i, j].item()
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results_G.append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
            
            for c in [0, 1]:
                fd = finite_difference_gradient(problem, dL_dT, 'B', (c, i, j), self.eps)
                impl = dL_dB[0, c, i, j].item()
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results_B.append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
        
        # Stats
        G_errors = [r['rel_err'] for r in results_G if r['rel_err'] < 10]
        B_errors = [r['rel_err'] for r in results_B if r['rel_err'] < 10]
        
        pct_good_G = sum(1 for e in G_errors if e < 0.1) / len(G_errors) * 100 if G_errors else 0
        pct_good_B = sum(1 for e in B_errors if e < 0.1) / len(B_errors) * 100 if B_errors else 0
        
        print(f"  G: {pct_good_G:.1f}% < 10% err, B: {pct_good_B:.1f}% < 10% err")
        
        self.results_G, self.results_B = results_G, results_B
        self.T, self.dL_dG, self.dL_dB = T, dL_dG, dL_dB
        self.target = (target_i, target_j)
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=(pct_good_G > 80 and pct_good_B > 80),
            metrics={'pct_good_G': pct_good_G, 'pct_good_B': pct_good_B,
                    'mean_err_G': np.mean(G_errors), 'mean_err_B': np.mean(B_errors)},
            arrays={'G_fd': np.array([r['fd'] for r in results_G]),
                   'G_impl': np.array([r['impl'] for r in results_G]),
                   'B_fd': np.array([r['fd'] for r in results_B]),
                   'B_impl': np.array([r['impl'] for r in results_B])},
            metadata={'N': N, 'n_test': self.n_test_points}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plots
        for ax, results, name in [(axes[0,0], self.results_G, 'G'), (axes[0,1], self.results_B, 'B')]:
            fd = np.array([r['fd'] for r in results])
            impl = np.array([r['impl'] for r in results])
            ax.scatter(fd, impl, alpha=0.6)
            lim = max(abs(fd).max(), abs(impl).max()) * 1.1
            if lim > 0:
                ax.plot([-lim, lim], [-lim, lim], 'r--')
            ax.set_xlabel('Finite Difference')
            ax.set_ylabel('Implicit')
            ax.set_title(f'{name} gradients')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Gradient magnitude maps
        B_mag = torch.sqrt(self.dL_dB[0,0]**2 + self.dL_dB[0,1]**2).cpu().numpy()
        G_mag = torch.sqrt(self.dL_dG[0,0]**2 + self.dL_dG[0,2]**2).cpu().numpy()
        
        for ax, mag, name in [(axes[1,0], B_mag, 'B'), (axes[1,1], G_mag, 'G')]:
            im = ax.imshow(mag, origin='upper', cmap='hot')
            plt.colorbar(im, ax=ax)
            ax.plot(self.target[1], self.target[0], 'c*', markersize=15)
            ax.plot(self.N//2, self.N//2, 'g*', markersize=15)
            ax.set_title(f'|∇_{name} L|')
        
        fig.suptitle('B1: Gradient Verification (Isotropic)', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B2: FD VS IMPLICIT (ANISOTROPIC)
# =============================================================================

@register_experiment
class B2_FD_Anisotropic(Experiment):
    """Verify gradients for anisotropic metric (all 3 components)."""
    
    name = "B2_fd_vs_implicit_anisotropic"
    category = "B_gradient_verification"
    description = "Verify gradients for anisotropic metric"
    
    def __init__(self, N: int = 50, n_test_points: int = 15, device: str = None):
        super().__init__(device)
        self.N, self.n_test_points = N, n_test_points
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = create_metric_from_eigenvalues(N, N, 2.0, 0.5, np.deg2rad(30), self.device)
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        dL_dT = torch.zeros_like(T)
        dL_dT[0, N//2 + 12, N//2 + 8] = 1.0
        
        dL_dG, _ = backward_pass(T, dL_dT, problem, verbose=False)
        
        np.random.seed(43)
        test_points = []
        while len(test_points) < self.n_test_points:
            i, j = np.random.randint(5, N-5, 2)
            if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                test_points.append((i, j))
        
        results = {0: [], 1: [], 2: []}
        for i, j in test_points:
            for c in range(3):
                fd = finite_difference_gradient(problem, dL_dT, 'G', (c, i, j), 1e-5)
                impl = dL_dG[0, c, i, j].item()
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results[c].append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
        
        metrics = {}
        for c, name in [(0, 'g11'), (1, 'g12'), (2, 'g22')]:
            errs = [r['rel_err'] for r in results[c] if r['rel_err'] < 10]
            if errs:
                metrics[f'pct_good_{name}'] = sum(1 for e in errs if e < 0.1) / len(errs) * 100
                print(f"  {name}: {metrics[f'pct_good_{name}']:.1f}% < 10% err")
        
        self.results = results
        pct_good = {c: 100 * sum(1 for r in results[c] if r['rel_err'] < 0.1) / len(results[c])
                    for c in [0, 1, 2]}
        return ExperimentResult(
            name=self.name, category=self.category,
            success=all(pct > 70 for pct in pct_good.values()),
            metrics={**{f'pct_good_{n}': pct_good[c] for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]},
                     **{f'mean_err_{n}': np.mean([r['rel_err'] for r in results[c]])
                        for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]}},
            arrays={**{f'{n}_fd': np.array([r['fd'] for r in results[c]])
                       for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]},
                    **{f'{n}_impl': np.array([r['impl'] for r in results[c]])
                       for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]}},
            metadata={'N': self.N, 'n_test_points': self.n_test_points}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (c, name) in zip(axes, [(0, 'g₁₁'), (1, 'g₁₂'), (2, 'g₂₂')]):
            fd = np.array([r['fd'] for r in self.results[c]])
            impl = np.array([r['impl'] for r in self.results[c]])
            ax.scatter(fd, impl, alpha=0.6, s=50)
            lim = max(abs(fd).max(), abs(impl).max()) * 1.1
            if lim > 0:
                ax.plot([-lim, lim], [-lim, lim], 'r--')
            ax.set_xlabel('FD')
            ax.set_ylabel('Implicit')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        fig.suptitle('B2: Anisotropic Gradient Verification', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B3: FD VS IMPLICIT (DRIFT)
# =============================================================================

@register_experiment
class B3_FD_Drift(Experiment):
    """Verify B gradients with nonzero drift."""
    
    name = "B3_fd_vs_implicit_drift"
    category = "B_gradient_verification"
    description = "Verify drift gradients"
    
    def __init__(self, N: int = 50, n_test_points: int = 15, device: str = None):
        super().__init__(device)
        self.N, self.n_test_points = N, n_test_points
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        B[0, 0, :, :] = 0.15
        B[0, 1, :, :] = 0.08
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        dL_dT = torch.zeros_like(T)
        dL_dT[0, N//2 + 10, N//2 + 10] = 1.0
        
        _, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
        
        np.random.seed(44)
        test_points = []
        while len(test_points) < self.n_test_points:
            i, j = np.random.randint(5, N-5, 2)
            if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                test_points.append((i, j))
        
        results = {0: [], 1: []}
        for i, j in test_points:
            for c in range(2):
                fd = finite_difference_gradient(problem, dL_dT, 'B', (c, i, j), 1e-5)
                impl = dL_dB[0, c, i, j].item()
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results[c].append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
        
        metrics = {}
        for c, name in [(0, 'b1'), (1, 'b2')]:
            errs = [r['rel_err'] for r in results[c] if r['rel_err'] < 10]
            if errs:
                metrics[f'pct_good_{name}'] = sum(1 for e in errs if e < 0.1) / len(errs) * 100
        
        self.results = results
        pct_good = {c: 100 * sum(1 for r in results[c] if r['rel_err'] < 0.1) / len(results[c])
                    for c in [0, 1]}
        return ExperimentResult(
            name=self.name, category=self.category,
            success=all(pct > 70 for pct in pct_good.values()),
            metrics={**{f'pct_good_{n}': pct_good[c] for c, n in [(0, 'b1'), (1, 'b2')]},
                     **{f'mean_err_{n}': np.mean([r['rel_err'] for r in results[c]])
                        for c, n in [(0, 'b1'), (1, 'b2')]}},
            arrays={**{f'{n}_fd': np.array([r['fd'] for r in results[c]])
                       for c, n in [(0, 'b1'), (1, 'b2')]},
                    **{f'{n}_impl': np.array([r['impl'] for r in results[c]])
                       for c, n in [(0, 'b1'), (1, 'b2')]}},
            metadata={'N': self.N, 'n_test_points': self.n_test_points}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (c, name) in zip(axes, [(0, 'b₁'), (1, 'b₂')]):
            fd = np.array([r['fd'] for r in self.results[c]])
            impl = np.array([r['impl'] for r in self.results[c]])
            ax.scatter(fd, impl, alpha=0.6, s=50)
            lim = max(abs(fd).max(), abs(impl).max()) * 1.1
            if lim > 0:
                ax.plot([-lim, lim], [-lim, lim], 'r--')
            ax.set_xlabel('FD')
            ax.set_ylabel('Implicit')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        fig.suptitle('B3: Drift Gradient Verification', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B4: RING LOSS PATTERN
# =============================================================================

@register_experiment
class B4_RingLoss(Experiment):
    """Visualize adjoint propagation pattern for ring loss."""
    
    name = "B4_ring_loss_pattern"
    category = "B_gradient_verification"
    description = "Adjoint propagation with ring loss"
    
    def __init__(self, N: int = 100, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        # Ring loss: observations on a ring around source
        dL_dT = torch.zeros_like(T)
        radius = 25
        for theta in np.linspace(0, 2*np.pi, 32, endpoint=False):
            i = int(N//2 + radius * np.sin(theta))
            j = int(N//2 + radius * np.cos(theta))
            if 0 <= i < N and 0 <= j < N:
                dL_dT[0, i, j] = 1.0
        
        dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
        
        G_mag = torch.sqrt(dL_dG[0,0]**2 + dL_dG[0,2]**2)
        B_mag = torch.sqrt(dL_dB[0,0]**2 + dL_dB[0,1]**2)
        
        self.T, self.dL_dT, self.G_mag, self.B_mag = T, dL_dT, G_mag, B_mag
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'G_max': G_mag.max().item(), 'B_max': B_mag.max().item()},
            arrays={'T': T.cpu().numpy(), 'G_mag': G_mag.cpu().numpy(), 'B_mag': B_mag.cpu().numpy()},
            metadata={'N': N, 'radius': radius}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        plot_arrival_time(self.T, ax=axes[0], title='T')
        
        axes[1].imshow(self.dL_dT[0].cpu().numpy(), origin='upper', cmap='Reds')
        axes[1].set_title('dL/dT (ring)')
        
        im = axes[2].imshow(self.G_mag.cpu().numpy(), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[2])
        axes[2].set_title('|∇_G L|')
        
        im = axes[3].imshow(self.B_mag.cpu().numpy(), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[3])
        axes[3].set_title('|∇_B L|')
        
        fig.suptitle('B4: Ring Loss Adjoint Pattern', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B5: GRADIENT MAGNITUDE DECAY
# =============================================================================

@register_experiment
class B5_GradientDecay(Experiment):
    """Analyze gradient magnitude decay with distance from target."""
    
    name = "B5_gradient_decay"
    category = "B_gradient_verification"
    description = "Gradient magnitude vs distance from target"
    
    def __init__(self, N: int = 100, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        target_i, target_j = N//2 + 20, N//2 + 20
        dL_dT = torch.zeros_like(T)
        dL_dT[0, target_i, target_j] = 1.0
        
        dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
        
        G_mag = torch.sqrt(dL_dG[0,0]**2 + dL_dG[0,2]**2)
        
        # Compute distance from target for each point
        i_coords = torch.arange(N, device=self.device) - target_i
        j_coords = torch.arange(N, device=self.device) - target_j
        I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')
        dist = torch.sqrt(I.float()**2 + J.float()**2)
        
        # Bin by distance
        distances, magnitudes = [], []
        for d in range(1, 40):
            mask = (dist >= d - 0.5) & (dist < d + 0.5)
            if mask.sum() > 0:
                distances.append(d)
                magnitudes.append(G_mag[mask].mean().item())
        
        self.distances, self.magnitudes = distances, magnitudes
        self.G_mag = G_mag
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'max_grad': G_mag.max().item()},
            arrays={'distances': np.array(distances), 'magnitudes': np.array(magnitudes)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im = axes[0].imshow(self.G_mag.cpu().numpy(), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title('|∇_G L|')
        
        axes[1].semilogy(self.distances, self.magnitudes, 'o-')
        axes[1].set_xlabel('Distance from target')
        axes[1].set_ylabel('Mean |∇_G L|')
        axes[1].set_title('Gradient Decay')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('B5: Gradient Magnitude Decay', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B6: STENCIL MAP
# =============================================================================

@register_experiment
class B6_StencilMap(Experiment):
    """Visualize which stencil is active at each node."""
    
    name = "B6_stencil_map"
    category = "B_gradient_verification"
    description = "Visualize active stencil structure"
    
    def __init__(self, N: int = 100, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        stencil_info = identify_active_stencils(T, problem)
        
        # Count types
        counts = {}
        for i in range(8):
            counts[f'stencil_{i}'] = int((stencil_info.stencil_idx == i).sum().item())
        
        for ut in UpdateType:
            counts[ut.name] = int((stencil_info.update_type == ut.value).sum().item())
        
        self.T, self.stencil_info = T, stencil_info
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics=counts,
            arrays={'stencil_idx': stencil_info.stencil_idx.cpu().numpy(),
                   'update_type': stencil_info.update_type.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_arrival_time(self.T, ax=axes[0], title='T')
        plot_stencil_map(self.stencil_info.stencil_idx, ax=axes[1], title='Stencil Index')
        
        ut = self.stencil_info.update_type.squeeze().cpu().numpy()
        cmap = plt.cm.get_cmap('Set1', 4)
        im = axes[2].imshow(ut, origin='upper', cmap=cmap, vmin=-0.5, vmax=3.5)
        cbar = plt.colorbar(im, ax=axes[2], ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['PLANAR', 'EDGE_M1', 'EDGE_M2', 'SOURCE'])
        axes[2].set_title('Update Type')
        
        fig.suptitle('B6: Stencil Structure', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B7: STENCIL BOUNDARIES
# =============================================================================

@register_experiment
class B7_StencilBoundaries(Experiment):
    """Visualize stencil boundaries (where gradient undefined)."""
    
    name = "B7_stencil_boundaries"
    category = "B_gradient_verification"
    description = "Visualize stencil boundaries (measure zero)"
    
    def __init__(self, N: int = 100, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        stencil_info = identify_active_stencils(T, problem)
        stencil_idx = stencil_info.stencil_idx[0].cpu().numpy()
        
        # Find boundaries
        H, W = stencil_idx.shape
        boundaries = np.zeros_like(stencil_idx, dtype=bool)
        for i in range(1, H-1):
            for j in range(1, W-1):
                center = stencil_idx[i, j]
                neighbors = [stencil_idx[i-1,j], stencil_idx[i+1,j], 
                            stencil_idx[i,j-1], stencil_idx[i,j+1]]
                if any(n != center for n in neighbors):
                    boundaries[i, j] = True
        
        pct_boundary = boundaries.sum() / boundaries.size * 100
        print(f"  Boundary pixels: {pct_boundary:.2f}%")
        
        self.stencil_idx, self.boundaries = stencil_idx, boundaries
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'pct_boundary': pct_boundary, 'n_boundary': int(boundaries.sum())},
            arrays={'stencil_idx': stencil_idx, 'boundaries': boundaries},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im = axes[0].imshow(self.stencil_idx, origin='upper', cmap='tab10', alpha=0.7)
        plt.colorbar(im, ax=axes[0])
        by, bx = np.where(self.boundaries)
        axes[0].scatter(bx, by, c='red', s=1, alpha=0.5)
        axes[0].set_title('Stencils + Boundaries (red)')
        
        axes[1].imshow(self.boundaries, origin='upper', cmap='gray_r')
        axes[1].set_title(f'Boundaries ({self.result.metrics["pct_boundary"]:.2f}%)')
        
        fig.suptitle('B7: Stencil Switching Boundaries', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B8: RANDOM DIRECTIONS
# =============================================================================

@register_experiment
class B8_RandomDirections(Experiment):
    """Test gradient agreement along random directions."""
    
    name = "B8_random_directions"
    category = "B_gradient_verification"
    description = "Verify gradient correct a.e. via random directions"
    
    def __init__(self, N: int = 50, n_directions: int = 100, device: str = None):
        super().__init__(device)
        self.N, self.n_directions = N, n_directions
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        dL_dT = torch.zeros_like(T)
        dL_dT[0, N//2 + 10, N//2 + 10] = 1.0
        
        dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
        
        eps = 1e-5
        np.random.seed(45)
        
        errors = []
        for _ in range(self.n_directions):
            dB = torch.zeros_like(B)
            dB[0, :, 5:-5, 5:-5] = torch.randn(2, N-10, N-10, device=self.device)
            dB = dB / dB.norm()
            
            impl_dir = (dL_dB * dB).sum().item()
            
            B_plus = problem.B + eps * dB
            B_minus = problem.B - eps * dB
            
            T_plus, _ = fast_sweeping_solve(EikonalProblem(problem.G, B_plus, source_mask), verbose=False)
            T_minus, _ = fast_sweeping_solve(EikonalProblem(problem.G, B_minus, source_mask), verbose=False)
            
            fd_dir = ((T_plus - T_minus) * dL_dT).sum().item() / (2 * eps)
            
            rel_err = abs(fd_dir - impl_dir) / abs(fd_dir) if abs(fd_dir) > 1e-10 else abs(impl_dir)
            errors.append(rel_err)
        
        errors = np.array(errors)
        pct_good = (errors < 0.1).sum() / len(errors) * 100
        print(f"  {pct_good:.1f}% directions < 10% error")
        
        self.errors = errors
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=pct_good > 10,
            metrics={'pct_good': pct_good, 'median_err': float(np.median(errors))},
            arrays={'errors': errors},
            metadata={'N': N, 'n_directions': self.n_directions}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(self.errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(0.1, color='r', linestyle='--', label='10%')
        axes[0].set_xlabel('Relative Error')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{self.result.metrics["pct_good"]:.1f}% < 10%')
        axes[0].legend()
        
        sorted_err = np.sort(self.errors)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
        axes[1].plot(sorted_err, cdf)
        axes[1].axvline(0.1, color='r', linestyle='--')
        axes[1].axhline(0.9, color='g', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Error')
        axes[1].set_ylabel('CDF')
        axes[1].set_title('CDF')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('B8: Random Direction Verification', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B9: GRADIENT UNDER PERTURBATION
# =============================================================================

@register_experiment
class B9_GradientStability(Experiment):
    """Test gradient stability under small parameter perturbations."""
    
    name = "B9_gradient_stability"
    category = "B_gradient_verification"
    description = "Gradient stability under perturbations"
    
    def __init__(self, N: int = 50, n_perturbations: int = 20, device: str = None):
        super().__init__(device)
        self.N, self.n_perturbations = N, n_perturbations
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        dL_dT = torch.zeros(1, N, N, device=self.device)
        dL_dT[0, N//2 + 10, N//2 + 10] = 1.0
        
        # Base gradient
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        dL_dG_base, dL_dB_base = backward_pass(T, dL_dT, problem, verbose=False)
        
        # Perturbed gradients
        np.random.seed(46)
        variations = []
        
        for _ in range(self.n_perturbations):
            G_pert = G + 0.01 * torch.randn_like(G)
            G_pert[:, 0, :, :] = torch.clamp(G_pert[:, 0, :, :], min=0.5)
            G_pert[:, 2, :, :] = torch.clamp(G_pert[:, 2, :, :], min=0.5)
            
            problem_pert = EikonalProblem(G_pert, B, source_mask)
            T_pert, _ = fast_sweeping_solve(problem_pert, verbose=False)
            dL_dG_pert, _ = backward_pass(T_pert, dL_dT, problem_pert, verbose=False)
            
            rel_diff = (dL_dG_pert - dL_dG_base).abs().mean().item() / dL_dG_base.abs().mean().item()
            variations.append(rel_diff)
        
        variations = np.array(variations)
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=np.mean(variations) < 0.5,
            metrics={'mean_variation': float(np.mean(variations)), 
                    'max_variation': float(np.max(variations))},
            arrays={'variations': variations},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(self.result.arrays['variations'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.result.arrays['variations']), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(self.result.arrays["variations"]):.3f}')
        ax.set_xlabel('Relative Gradient Variation')
        ax.set_ylabel('Count')
        ax.set_title('B9: Gradient Stability')
        ax.legend()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    B1_FD_Isotropic,
    B2_FD_Anisotropic,
    B3_FD_Drift,
    B4_RingLoss,
    B5_GradientDecay,
    B6_StencilMap,
    B7_StencilBoundaries,
    B8_RandomDirections,
    B9_GradientStability,
]

def run_all(save=True, visualize=True):
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY B SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results


if __name__ == "__main__":
    run_all()
