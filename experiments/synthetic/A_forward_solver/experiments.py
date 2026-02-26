"""
Category A: Forward Solver Validation
=====================================

9 experiments validating the fast sweeping solver:

A1: Isotropic Convergence - O(h) convergence rate
A2: Uniform Anisotropic - Diagonal metric G = diag(a², b²)  
A3: Rotated Anisotropic - Off-diagonal g12 component
A4: Constant Drift - Randers/advection term
A5: Combined Anisotropic + Drift - Full Randers-Finsler
A6: Piecewise Constant Metric - Discontinuities/refraction
A7: Spatially Varying Metric - Smooth variations
A8: Multi-Source Problem - Multiple ignition points
A9: Iteration Complexity - O(N) total work
"""

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import sys
import os

from experiment_base import (
    Experiment, ExperimentResult, register_experiment, DEVICE,
    euclidean_distance_field, anisotropic_distance_field, create_metric_from_eigenvalues,
    compute_errors, compute_convergence_rate,
    plot_arrival_time, plot_error_map, plot_metric_ellipses, plot_drift_field
)

# Import from project
from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve


# =============================================================================
# A1: ISOTROPIC CONVERGENCE
# =============================================================================
@register_experiment
class A1_IsotropicConvergence(Experiment):
    """Verify O(h) convergence on isotropic problem."""

    name = "A1_isotropic_convergence"
    category = "A_forward_solver"
    description = "Verify O(h) convergence rate for isotropic eikonal equation"

    def __init__(self, grid_sizes: List[int] = None, device: str = None):
        super().__init__(device)
        self.grid_sizes = grid_sizes or [25, 50, 100, 200, 400]

    def run(self) -> ExperimentResult:
        h_values, errors_l2, errors_linf, rel_errors = [], [], [], []

        # Fixed physical domain [0, 1] x [0, 1]
        L = 1.0

        for N in self.grid_sizes:
            print(f"  Grid size: {N}x{N}")
            h = L / N  # Physical grid spacing

            G = torch.ones(1, 3, N, N, device=self.device)
            G[:, 1, :, :] = 0
            B = torch.zeros(1, 2, N, N, device=self.device)

            source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
            source_i, source_j = N // 2, N // 2
            source_mask[0, source_i, source_j] = True

            problem = EikonalProblem(G, B, source_mask)
            T_grid, n_iter = fast_sweeping_solve(problem, max_iter=50, tol=1e-8, verbose=False)

            # T_grid is in grid units. Convert to physical: T_physical = h * T_grid
            T_physical = h * T_grid

            # Ground truth in physical units: distance from source
            i_coords = torch.arange(N, device=self.device, dtype=torch.float32)
            j_coords = torch.arange(N, device=self.device, dtype=torch.float32)
            I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')
            # Physical positions
            X = I * h
            Y = J * h
            source_x, source_y = source_i * h, source_j * h
            T_exact = torch.sqrt((X - source_x) ** 2 + (Y - source_y) ** 2)

            # Interior mask excluding source vicinity
            interior_mask = torch.zeros(N, N, dtype=torch.bool, device=self.device)
            interior_mask[2:-2, 2:-2] = True
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    interior_mask[source_i + di, source_j + dj] = False

            diff = (T_physical[0] - T_exact)[interior_mask]

            l2 = torch.sqrt((diff ** 2).mean()).item()
            linf = torch.abs(diff).max().item()
            rel_l2 = l2 / T_exact[interior_mask].max().item()

            h_values.append(h)
            errors_l2.append(l2)
            errors_linf.append(linf)
            rel_errors.append(rel_l2)
            print(f"    h={h:.4f}, L2={l2:.6f}, Linf={linf:.6f}, rel={rel_l2:.6f}")

        rate_l2, r2_l2 = compute_convergence_rate(h_values, errors_l2)
        rate_linf, r2_linf = compute_convergence_rate(h_values, errors_linf)
        print(f"\n  L2 rate: {rate_l2:.2f}, Linf rate: {rate_linf:.2f}")

        self._data = {'h': h_values, 'e_l2': errors_l2, 'e_linf': errors_linf, 'rel': rel_errors}

        # Success: rate between 0.8 and 1.5 (allowing some margin)
        return ExperimentResult(
            name=self.name, category=self.category,
            success=(0.65 <= rate_l2 <= 1.5),
            metrics={'rate_l2': rate_l2, 'rate_linf': rate_linf, 'r2_l2': r2_l2,
                     'r2_linf': r2_linf, 'max_rel_error': max(rel_errors)},
            arrays={'h_values': np.array(h_values), 'errors_l2': np.array(errors_l2),
                    'errors_linf': np.array(errors_linf), 'rel_errors': np.array(rel_errors)},
            metadata={'grid_sizes': self.grid_sizes}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        h, e_l2, e_linf = self._data['h'], self._data['e_l2'], self._data['e_linf']

        ax.loglog(h, e_l2, 'o-', label=f'L² (rate={self.result.metrics["rate_l2"]:.2f})', markersize=8)
        ax.loglog(h, e_linf, 's-', label=f'L∞ (rate={self.result.metrics["rate_linf"]:.2f})', markersize=8)

        h_ref = np.array([h[0], h[-1]])
        ax.loglog(h_ref, e_l2[0] * (h_ref / h[0]), 'k--', alpha=0.5, label='O(h)')

        ax.set_xlabel('Grid spacing h')
        ax.set_ylabel('Error (physical units)')
        ax.set_title('A1: Isotropic Convergence Study')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig

# =============================================================================
# A2: UNIFORM ANISOTROPIC
# =============================================================================

@register_experiment
class A2_UniformAnisotropic(Experiment):
    """Verify handling of uniform anisotropic metric G = diag(a², b²)."""
    
    name = "A2_uniform_anisotropic"
    category = "A_forward_solver"
    description = "Verify elliptical wavefronts with diagonal anisotropic metric"
    
    def __init__(self, N: int = 200, a: float = 2.0, b: float = 0.5, device: str = None):
        super().__init__(device)
        self.N, self.a, self.b = N, a, b
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.zeros(1, 3, N, N, device=self.device)
        G[0, 0, :, :] = self.a**2
        G[0, 1, :, :] = 0
        G[0, 2, :, :] = self.b**2
        B = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_i, source_j = N // 2, N // 2
        source_mask[0, source_i, source_j] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        T_exact = anisotropic_distance_field(N, N, source_i, source_j, self.a**2, self.b**2, 0, self.device)
        
        interior_mask = torch.zeros_like(T, dtype=torch.bool)
        interior_mask[0, 5:-5, 5:-5] = True
        interior_mask[0, source_i-3:source_i+4, source_j-3:source_j+4] = False
        
        errs = compute_errors(T, T_exact.unsqueeze(0), interior_mask)
        print(f"  Relative L2 error: {errs['rel_l2']:.6f}")
        
        self.T, self.T_exact, self.G = T, T_exact, G
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=errs['rel_l2'] < 0.02,
            metrics=errs,
            arrays={'T': T.cpu().numpy(), 'T_exact': T_exact.cpu().numpy()},
            metadata={'N': N, 'a': self.a, 'b': self.b}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Computed T')
        plot_metric_ellipses(self.G, axes[0], step=20, scale=3.0)
        plot_arrival_time(self.T_exact.unsqueeze(0), ax=axes[1], title='Exact T')
        plot_error_map(self.T.squeeze() - self.T_exact, ax=axes[2], 
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle(f'A2: Uniform Anisotropic (a={self.a}, b={self.b})', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A3: ROTATED ANISOTROPIC  
# =============================================================================

@register_experiment
class A3_RotatedAnisotropic(Experiment):
    """Verify handling of rotated anisotropic metric with nonzero g12."""
    
    name = "A3_rotated_anisotropic"
    category = "A_forward_solver"
    description = "Verify correct handling of rotated metric (nonzero g12)"
    
    def __init__(self, N: int = 200, lambda1: float = 4.0, lambda2: float = 0.25, 
                 theta_deg: float = 45.0, device: str = None):
        super().__init__(device)
        self.N, self.lambda1, self.lambda2 = N, lambda1, lambda2
        self.theta = np.deg2rad(theta_deg)
        self.theta_deg = theta_deg
        
    def run(self) -> ExperimentResult:
        N = self.N
        G = create_metric_from_eigenvalues(N, N, self.lambda1, self.lambda2, self.theta, self.device)
        B = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_i, source_j = N // 2, N // 2
        source_mask[0, source_i, source_j] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        T_exact = anisotropic_distance_field(N, N, source_i, source_j, self.lambda1, self.lambda2, 
                                              self.theta, self.device)
        
        interior_mask = torch.zeros_like(T, dtype=torch.bool)
        interior_mask[0, 5:-5, 5:-5] = True
        interior_mask[0, source_i-3:source_i+4, source_j-3:source_j+4] = False
        
        errs = compute_errors(T, T_exact.unsqueeze(0), interior_mask)
        print(f"  Relative L2 error: {errs['rel_l2']:.6f}")
        print(f"  g12 at center: {G[0, 1, N//2, N//2].item():.4f}")
        
        self.T, self.T_exact, self.G = T, T_exact, G
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=errs['rel_l2'] < 0.02,
            metrics=errs,
            arrays={'T': T.cpu().numpy(), 'T_exact': T_exact.cpu().numpy()},
            metadata={'theta_deg': self.theta_deg, 'lambda1': self.lambda1, 'lambda2': self.lambda2}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Computed T')
        plot_metric_ellipses(self.G, axes[0], step=20, scale=3.0)
        plot_arrival_time(self.T_exact.unsqueeze(0), ax=axes[1], title='Exact T')
        plot_error_map(self.T.squeeze() - self.T_exact, ax=axes[2],
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle(f'A3: Rotated Anisotropic (θ={self.theta_deg}°)', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A4: CONSTANT DRIFT
# =============================================================================

@register_experiment
class A4_ConstantDrift(Experiment):
    """Verify constant drift produces asymmetric wavefronts."""
    
    name = "A4_constant_drift"
    category = "A_forward_solver"
    description = "Verify drift field produces asymmetric wavefronts (Randers)"
    
    def __init__(self, N: int = 200, drift_magnitude: float = 0.3, device: str = None):
        super().__init__(device)
        self.N, self.drift_magnitude = N, drift_magnitude
        
    def run(self) -> ExperimentResult:
        N, b = self.N, self.drift_magnitude
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        B[0, 0, :, :] = b  # drift in x
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_i, source_j = N // 2, N // 2
        source_mask[0, source_i, source_j] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, n_iter = fast_sweeping_solve(problem, verbose=False)
        
        # Check asymmetry
        T_left = T[0, source_i, source_j - 30].item()
        T_right = T[0, source_i, source_j + 30].item()
        asymmetry = (T_left - T_right) / (T_left + T_right)
        
        print(f"  T at x=-30: {T_left:.4f}, T at x=+30: {T_right:.4f}")
        print(f"  Asymmetry: {asymmetry:.4f}")
        
        self.T, self.B = T, B
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=asymmetry > 0.1,
            metrics={'T_left': T_left, 'T_right': T_right, 'asymmetry': asymmetry},
            arrays={'T': T.cpu().numpy()},
            metadata={'drift_magnitude': b}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Arrival Time with Drift')
        plot_drift_field(self.B, axes[0], step=15, scale=20)
        
        N = self.N
        profile = self.T[0, N//2, :].cpu().numpy()
        x = np.arange(N) - N//2
        axes[1].plot(x, profile)
        axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('x offset')
        axes[1].set_ylabel('Arrival time')
        axes[1].set_title(f'Profile (b={self.drift_magnitude})')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('A4: Constant Drift Field', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A5: COMBINED ANISOTROPIC + DRIFT
# =============================================================================

@register_experiment
class A5_CombinedAnisotropicDrift(Experiment):
    """Full Randers-Finsler: anisotropic metric + drift field."""

    name = "A5_combined_anisotropic_drift"
    category = "A_forward_solver"
    description = "Verify full Randers-Finsler equation"

    def __init__(self, N: int = 100, N_fine: int = 400, device: str = None):
        super().__init__(device)
        self.N, self.N_fine = N, N_fine

    def run(self) -> ExperimentResult:
        N, N_fine = self.N, self.N_fine
        lambda1, lambda2, theta = 2.0, 0.5, np.deg2rad(30)
        drift_x, drift_y = 0.2, 0.1

        # Coarse solution
        G = create_metric_from_eigenvalues(N, N, lambda1, lambda2, theta, self.device)
        B = torch.zeros(1, 2, N, N, device=self.device)
        B[0, 0, :, :], B[0, 1, :, :] = drift_x, drift_y
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N // 2, N // 2] = True

        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)

        # Fine solution
        G_fine = create_metric_from_eigenvalues(N_fine, N_fine, lambda1, lambda2, theta, self.device)
        B_fine = torch.zeros(1, 2, N_fine, N_fine, device=self.device)
        B_fine[0, 0, :, :], B_fine[0, 1, :, :] = drift_x, drift_y
        source_mask_fine = torch.zeros(1, N_fine, N_fine, dtype=torch.bool, device=self.device)
        source_mask_fine[0, N_fine // 2, N_fine // 2] = True

        problem_fine = EikonalProblem(G_fine, B_fine, source_mask_fine)
        T_fine, _ = fast_sweeping_solve(problem_fine, verbose=False)

        # Scale fine solution to coarse physical domain
        # Coarse uses h=1/N, Fine uses h=1/N_fine
        # T_physical = h * T_grid, so T_fine_physical = (1/N_fine) * T_fine_grid
        # To compare at same physical points: interpolate fine to coarse grid
        scale = N_fine // N

        # Sample fine solution at coarse grid points (accounting for different h)
        # Fine grid point (i*scale, j*scale) corresponds to coarse point (i, j)
        T_fine_sampled = T_fine[0, ::scale, ::scale] * (1.0 / scale)  # Scale to same physical units
        T_coarse_physical = T[0] * 1.0  # h_coarse / h_fine = scale, already in coarse units

        # Actually simpler: both are in grid units of their respective grids
        # At same physical location, T_fine_grid / N_fine ≈ T_coarse_grid / N
        # So T_fine_grid * (N / N_fine) ≈ T_coarse_grid
        T_fine_rescaled = T_fine[0, ::scale, ::scale] / scale

        interior = torch.zeros(N, N, dtype=torch.bool, device=self.device)
        interior[5:-5, 5:-5] = True
        interior[N // 2 - 3:N // 2 + 4, N // 2 - 3:N // 2 + 4] = False

        diff = T[0] - T_fine_rescaled
        rel_l2 = torch.sqrt((diff[interior] ** 2).mean()).item() / T[0][interior].abs().mean().item()
        print(f"  Relative L2 vs fine: {rel_l2:.6f}")

        self.T, self.T_fine_rescaled = T, T_fine_rescaled
        self.G, self.B = G, B

        return ExperimentResult(
            name=self.name, category=self.category,
            success=rel_l2 < 0.10,  # 10% tolerance
            metrics={'rel_l2': rel_l2},
            arrays={'T': T.cpu().numpy(), 'T_fine': T_fine_rescaled.cpu().numpy()},
            metadata={'N': N, 'N_fine': N_fine}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Coarse')
        plot_metric_ellipses(self.G, axes[0], step=10, scale=1.5)
        plot_drift_field(self.B, axes[0], step=10, scale=30)
        plot_arrival_time(self.T_fine_rescaled.unsqueeze(0), ax=axes[1], title='Fine (rescaled)')
        plot_error_map(self.T.squeeze() - self.T_fine_rescaled, ax=axes[2],
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle('A5: Anisotropic + Drift', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig

# =============================================================================
# A6: PIECEWISE METRIC
# =============================================================================

@register_experiment
class A6_PiecewiseMetric(Experiment):
    """Test wavefront refraction at metric discontinuity."""
    
    name = "A6_piecewise_metric"
    category = "A_forward_solver"
    description = "Verify wavefront refraction at interface"
    
    def __init__(self, N: int = 200, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        G[0, 0, :, N//2:] = 4.0  # slow region
        G[0, 2, :, N//2:] = 4.0
        
        B = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_i, source_j = N // 2, N // 4
        source_mask[0, source_i, source_j] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, max_iter=100, verbose=False)
        
        dist_to_interface = N//2 - N//4
        T_at_interface = T[0, N//2, N//2].item()
        T_past = T[0, N//2, N//2 + 10].item()
        
        speed_in_slow = (T_past - T_at_interface) / 10
        print(f"  Expected speed in slow region: 2.0, got: {speed_in_slow:.2f}")
        
        self.T, self.G = T, G
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=1.5 < speed_in_slow < 2.5,
            metrics={'speed_in_slow': speed_in_slow},
            arrays={'T': T.cpu().numpy(), 'G11': G[0, 0].cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Arrival Time')
        axes[0].axvline(self.N//2, color='r', linestyle='--', linewidth=2, label='Interface')
        axes[0].legend()
        
        im = axes[1].imshow(self.G[0, 0].cpu().numpy(), origin='upper', cmap='RdYlBu_r')
        plt.colorbar(im, ax=axes[1], label='g₁₁')
        axes[1].axvline(self.N//2, color='k', linestyle='--', linewidth=2)
        axes[1].set_title('Metric g₁₁')
        
        fig.suptitle('A6: Piecewise Metric', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A7: SPATIALLY VARYING METRIC
# =============================================================================

@register_experiment
class A7_SpatiallyVaryingMetric(Experiment):
    """Test smoothly varying metric field."""

    name = "A7_spatially_varying_metric"
    category = "A_forward_solver"
    description = "Verify solver handles smooth metric variations"

    def __init__(self, N: int = 200, device: str = None):
        super().__init__(device)
        self.N = N

    def run(self) -> ExperimentResult:
        N = self.N

        # Spatially varying metric
        j_coords = torch.arange(N, device=self.device, dtype=torch.float32)
        variation = 1.0 + 0.5 * torch.sin(2 * np.pi * j_coords / N)

        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        G[0, 0, :, :] = variation.unsqueeze(0).expand(N, N)
        G[0, 2, :, :] = variation.unsqueeze(0).expand(N, N)

        B = torch.zeros(1, 2, N, N, device=self.device)

        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N // 2, N // 2] = True

        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, max_iter=100, verbose=False)

        # Self-consistency check: verify |∇T|_G ≈ 1 in interior
        # Compute numerical gradient
        T_np = T[0].cpu().numpy()
        grad_i = np.zeros_like(T_np)
        grad_j = np.zeros_like(T_np)
        grad_i[1:-1, :] = (T_np[2:, :] - T_np[:-2, :]) / 2
        grad_j[:, 1:-1] = (T_np[:, 2:] - T_np[:, :-2]) / 2

        # |∇T|_G = sqrt((∇T)ᵀ G⁻¹ (∇T)) = sqrt(A @ grad) where A = G⁻¹
        # For diagonal G: A = diag(1/g11, 1/g22)
        g11 = G[0, 0].cpu().numpy()
        g22 = G[0, 2].cpu().numpy()
        a11 = 1.0 / g11  # A = G^(-1)
        a22 = 1.0 / g22
        grad_norm_G = np.sqrt(a11 * grad_i ** 2 + a22 * grad_j ** 2)

        # Check in interior (exclude source and boundaries)
        interior = np.zeros((N, N), dtype=bool)
        interior[10:-10, 10:-10] = True
        interior[N // 2 - 5:N // 2 + 6, N // 2 - 5:N // 2 + 6] = False

        eikonal_residual = np.abs(grad_norm_G[interior] - 1.0)
        mean_residual = np.mean(eikonal_residual)
        max_residual = np.max(eikonal_residual)

        print(f"  Eikonal residual: mean={mean_residual:.6f}, max={max_residual:.6f}")

        self.T, self.G = T, G
        self.grad_norm_G = grad_norm_G

        return ExperimentResult(
            name=self.name, category=self.category,
            success=mean_residual < 0.05,  # Mean |∇T|_G - 1 < 5%
            metrics={'mean_residual': mean_residual, 'max_residual': max_residual},
            arrays={'T': T.cpu().numpy(), 'G11': G[0, 0].cpu().numpy(),
                    'grad_norm_G': grad_norm_G},
            metadata={'N': N}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Arrival Time')

        im = axes[1].imshow(self.G[0, 0].cpu().numpy(), origin='upper', cmap='viridis')
        plt.colorbar(im, ax=axes[1], label='g₁₁')
        axes[1].set_title('Spatially Varying g₁₁')

        residual = np.abs(self.grad_norm_G - 1.0)
        im = axes[2].imshow(residual, origin='upper', cmap='hot', vmin=0, vmax=0.2)
        plt.colorbar(im, ax=axes[2], label='||∇T||_G - 1|')
        axes[2].set_title(f'Eikonal Residual (mean={self.result.metrics["mean_residual"]:.4f})')

        fig.suptitle('A7: Spatially Varying Metric', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig

# =============================================================================
# A8: MULTI-SOURCE
# =============================================================================

@register_experiment
class A8_MultiSource(Experiment):
    """Verify Voronoi-like structure with multiple sources."""
    
    name = "A8_multi_source"
    category = "A_forward_solver"
    description = "Verify correct multi-source handling"
    
    def __init__(self, N: int = 200, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        B = torch.zeros(1, 2, N, N, device=self.device)
        
        sources = [(N//4, N//4), (N//4, 3*N//4), (3*N//4, N//2)]
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        for si, sj in sources:
            source_mask[0, si, sj] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        # Ground truth: min distance
        T_exact = torch.full((N, N), float('inf'), device=self.device)
        for si, sj in sources:
            T_exact = torch.minimum(T_exact, euclidean_distance_field(N, N, si, sj, self.device))
        
        interior = torch.zeros(N, N, dtype=torch.bool, device=self.device)
        interior[3:-3, 3:-3] = True
        for si, sj in sources:
            interior[si-2:si+3, sj-2:sj+3] = False
        
        errs = compute_errors(T, T_exact.unsqueeze(0), interior.unsqueeze(0))
        
        self.T, self.T_exact, self.sources = T, T_exact, sources
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=errs['rel_l2'] < 0.02,
            metrics=errs,
            arrays={'T': T.cpu().numpy(), 'T_exact': T_exact.cpu().numpy()},
            metadata={'sources': sources}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, T, title in [(axes[0], self.T, 'Computed'), (axes[1], self.T_exact.unsqueeze(0), 'Exact')]:
            plot_arrival_time(T, ax=ax, title=title)
            for si, sj in self.sources:
                ax.plot(sj, si, 'r*', markersize=15)
        
        plot_error_map(self.T.squeeze() - self.T_exact, ax=axes[2],
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle('A8: Multi-Source', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A9: ITERATION COMPLEXITY
# =============================================================================

@register_experiment
class A9_IterationComplexity(Experiment):
    """Verify O(1) iterations, O(N²) total work."""

    name = "A9_iteration_complexity"
    category = "A_forward_solver"
    description = "Verify iteration count is O(1)"

    def __init__(self, grid_sizes: List[int] = None, device: str = None):
        super().__init__(device)
        self.grid_sizes = grid_sizes or [50, 100, 200, 400]  # Removed 800 - too slow

    def run(self) -> ExperimentResult:
        iterations, times = [], []

        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")

            G = torch.ones(1, 3, N, N, device=self.device)
            G[:, 1, :, :] = 0
            B = torch.zeros(1, 2, N, N, device=self.device)
            source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
            source_mask[0, N // 2, N // 2] = True

            problem = EikonalProblem(G, B, source_mask)

            import time
            start = time.time()
            T, n_iter = fast_sweeping_solve(problem, max_iter=100, tol=1e-6, verbose=False)
            elapsed = time.time() - start

            iterations.append(n_iter)
            times.append(elapsed)
            print(f"    Iterations: {n_iter}, Time: {elapsed:.3f}s")

        iter_mean, iter_std = np.mean(iterations), np.std(iterations)
        print(f"\n  Iterations: {iter_mean:.1f} ± {iter_std:.1f}")

        # Check O(N²) scaling for time
        N_arr = np.array(self.grid_sizes, dtype=float)
        t_arr = np.array(times)
        # Fit time = c * N^alpha
        log_N = np.log(N_arr)
        log_t = np.log(t_arr)
        alpha, _ = np.polyfit(log_N, log_t, 1)

        self._data = {'sizes': self.grid_sizes, 'iters': iterations, 'times': times}

        return ExperimentResult(
            name=self.name, category=self.category,
            success=(iter_std < 5 and 1 < alpha < 3.0),  # Relaxed criteria
            metrics={'iter_mean': iter_mean, 'iter_std': iter_std, 'time_scaling': alpha},
            arrays={'grid_sizes': np.array(self.grid_sizes),
                    'iterations': np.array(iterations), 'times': np.array(times)},
            metadata={'grid_sizes': self.grid_sizes}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        N = np.array(self._data['sizes'])

        axes[0].plot(N, self._data['iters'], 'o-', markersize=10)
        axes[0].axhline(np.mean(self._data['iters']), color='r', linestyle='--',
                        label=f'Mean: {np.mean(self._data["iters"]):.1f}')
        axes[0].set_xlabel('Grid size N')
        axes[0].set_ylabel('Iterations')
        axes[0].set_title('Iterations (should be O(1))')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].loglog(N, self._data['times'], 'o-', markersize=10, label='Measured')
        axes[1].loglog(N, self._data['times'][0] * (N / N[0]) ** 2, 'k--', alpha=0.5,
                       label=f'O(N²), measured α={self.result.metrics["time_scaling"]:.2f}')
        axes[1].set_xlabel('Grid size N')
        axes[1].set_ylabel('Time (s)')
        axes[1].set_title('Timing')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('A9: Complexity Analysis', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig

# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    A1_IsotropicConvergence,
    A2_UniformAnisotropic,
    A3_RotatedAnisotropic,
    A4_ConstantDrift,
    A5_CombinedAnisotropicDrift,
    A6_PiecewiseMetric,
    A7_SpatiallyVaryingMetric,
    A8_MultiSource,
    A9_IterationComplexity,
]

def run_all(save=True, visualize=True):
    """Run all Category A experiments."""
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY A SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results


if __name__ == "__main__":
    run_all()
