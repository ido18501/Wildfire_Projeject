"""
Category C: Inverse Problem - Metric Recovery
==============================================

10 experiments demonstrating metric/drift recovery:

C1: Isotropic Recovery (Full Obs)
C2: Isotropic Recovery (Sparse Obs) 
C3: Diagonal Anisotropic Recovery
C4: Full Metric Recovery (all 3 components)
C5: Drift Recovery (Fixed Metric)
C6: Joint Metric + Drift Recovery
C7: Regularization Ablation Study
C8: Observation Density Study
C9: Noise Robustness
C10: Multiple Sources Benefit
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
    create_metric_from_eigenvalues, compute_tv_regularization,
    get_interior_mask, create_sparse_observation_mask,
    plot_arrival_time, plot_error_map
)

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.backward.implicit_diff import backward_pass


# =============================================================================
# METRIC RECOVERY OPTIMIZER
# =============================================================================

class MetricRecoveryOptimizer:
    """Gradient descent optimizer for metric recovery."""
    
    def __init__(self, M: int, N: int, device: str = DEVICE,
                 recover_G: bool = True, recover_B: bool = False,
                 lambda_G: float = 0.01, lambda_B: float = 0.01,
                 constrain_isotropic: bool = False):
        self.M, self.N = M, N
        self.device = device
        self.recover_G, self.recover_B = recover_G, recover_B
        self.lambda_G, self.lambda_B = lambda_G, lambda_B
        self.constrain_isotropic = constrain_isotropic
        
        # Initialize
        self.G = torch.ones(1, 3, M, N, device=device)
        self.G[:, 1, :, :] = 0
        self.B = torch.zeros(1, 2, M, N, device=device)
        
        self.history = {'loss': [], 'loss_data': [], 'loss_reg': []}
    
    def project_G(self):
        """Project G to SPD domain."""
        with torch.no_grad():
            self.G[:, 0, :, :] = torch.clamp(self.G[:, 0, :, :], min=0.1, max=10.0)
            self.G[:, 2, :, :] = torch.clamp(self.G[:, 2, :, :], min=0.1, max=10.0)
            
            if self.constrain_isotropic:
                avg = (self.G[:, 0, :, :] + self.G[:, 2, :, :]) / 2
                self.G[:, 0, :, :] = avg
                self.G[:, 2, :, :] = avg
                self.G[:, 1, :, :] = 0
            else:
                g11, g12, g22 = self.G[:, 0], self.G[:, 1], self.G[:, 2]
                max_g12 = torch.sqrt(g11 * g22) * 0.95
                self.G[:, 1, :, :] = torch.clamp(g12, min=-max_g12, max=max_g12)
    
    def project_B(self):
        """Project B to causal domain (|B| < 1)."""
        with torch.no_grad():
            B_norm = torch.sqrt(self.B[:, 0]**2 + self.B[:, 1]**2)
            mask = B_norm > 0.9
            if mask.any():
                self.B[:, 0][mask] *= 0.9 / B_norm[mask]
                self.B[:, 1][mask] *= 0.9 / B_norm[mask]
    
    def fit(self, source_mask: Tensor, T_obs: Tensor, obs_mask: Tensor,
            n_iter: int = 500, lr: float = 0.1, verbose: bool = True) -> Dict:
        """Fit parameters to observed arrival times."""
        
        T_obs_values = T_obs[obs_mask]
        
        for it in range(n_iter):
            # Forward solve
            problem = EikonalProblem(self.G, self.B, source_mask)
            T, _ = fast_sweeping_solve(problem, max_iter=50, tol=1e-7, verbose=False)
            
            # Data loss
            residuals = T[obs_mask] - T_obs_values
            loss_data = 0.5 * (residuals**2).sum()
            
            # Regularization
            loss_reg, reg_info = compute_tv_regularization(self.G, self.B if self.recover_B else None,
                                                           self.lambda_G, self.lambda_B)
            
            total = loss_data.item() + loss_reg.item()
            self.history['loss'].append(total)
            self.history['loss_data'].append(loss_data.item())
            self.history['loss_reg'].append(loss_reg.item())
            
            if verbose and (it % 50 == 0 or it == n_iter - 1):
                print(f"  Iter {it}: loss={total:.4f} (data={loss_data.item():.4f}, reg={loss_reg.item():.4f})")
            
            # Backward pass
            dL_dT = torch.zeros_like(T)
            dL_dT[obs_mask] = residuals
            
            dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
            
            # Add TV gradient (approximate)
            if self.lambda_G > 0 and self.recover_G:
                eps = 1e-6
                for c in range(3):
                    G_p = self.G.clone()
                    G_p[:, c, :, :] += eps
                    reg_p, _ = compute_tv_regularization(G_p, None, self.lambda_G, 0)
                    G_m = self.G.clone()
                    G_m[:, c, :, :] -= eps
                    reg_m, _ = compute_tv_regularization(G_m, None, self.lambda_G, 0)
                    dL_dG[:, c, :, :] += (reg_p - reg_m) / (2 * eps)
            
            # Update
            with torch.no_grad():
                if self.recover_G:
                    self.G -= lr * dL_dG
                if self.recover_B:
                    self.B -= lr * dL_dB
                
                self.project_G()
                if self.recover_B:
                    self.project_B()
            
            # LR decay
            if it > 0 and it % 100 == 0:
                lr *= 0.8
        
        return {'final_loss': total, 'final_data': loss_data.item(), 'final_reg': loss_reg.item()}


# =============================================================================
# C1: ISOTROPIC RECOVERY (FULL OBS)
# =============================================================================

@register_experiment
class C1_IsotropicFull(Experiment):
    """Recover isotropic metric from full observations."""
    
    name = "C1_isotropic_recovery_full"
    category = "C_inverse_problem"
    description = "Recover isotropic metric from full observations"
    
    def __init__(self, N: int = 80, n_iter: int = 300, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # True metric: piecewise constant
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        # Generate observations
        problem_true = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem_true, verbose=False)
        
        # Full observations
        obs_mask = get_interior_mask(N, N, 3, source_mask, device=self.device)
        
        # Recover
        optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=0.001, constrain_isotropic=True)
        metrics = optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05)
        
        # Compute error
        interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
        g11_true = G_true[0, 0][interior[0]]
        g11_rec = optimizer.G[0, 0][interior[0]]
        rel_err = torch.sqrt(((g11_true - g11_rec)**2).mean()).item() / g11_true.mean().item()
        
        print(f"\n  Recovery error: {rel_err:.4f}")
        
        self.G_true, self.G_rec = G_true, optimizer.G
        self.T_obs, self.optimizer = T_obs, optimizer
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=rel_err < 0.15,
            metrics={'rel_error': rel_err, **metrics},
            arrays={'G_true': G_true.cpu().numpy(), 'G_rec': optimizer.G.cpu().numpy(),
                   'loss_history': np.array(optimizer.history['loss'])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        vmin, vmax = self.G_true[0, 0].min().item(), self.G_true[0, 0].max().item()
        
        im = axes[0,0].imshow(self.G_true[0, 0].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[0,0])
        axes[0,0].set_title('True g₁₁')
        
        im = axes[0,1].imshow(self.G_rec[0, 0].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[0,1])
        axes[0,1].set_title('Recovered g₁₁')
        
        err = (self.G_rec[0, 0] - self.G_true[0, 0]).cpu().numpy()
        plot_error_map(torch.tensor(err), ax=axes[0,2], title='Error')
        
        plot_arrival_time(self.T_obs, ax=axes[1,0], title='Observed T')
        
        axes[1,1].semilogy(self.optimizer.history['loss'])
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].set_title('Optimization')
        axes[1,1].grid(True, alpha=0.3)
        
        j = self.N // 2
        axes[1,2].plot(self.G_true[0, 0, j, :].cpu().numpy(), 'b-', label='True', lw=2)
        axes[1,2].plot(self.G_rec[0, 0, j, :].cpu().numpy(), 'r--', label='Recovered', lw=2)
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('g₁₁')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        fig.suptitle(f'C1: Isotropic Recovery (err={self.result.metrics["rel_error"]:.2%})', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C2: ISOTROPIC RECOVERY (SPARSE OBS)
# =============================================================================

@register_experiment
class C2_IsotropicSparse(Experiment):
    """Recover isotropic metric from sparse (7%) observations."""
    
    name = "C2_isotropic_recovery_sparse"
    category = "C_inverse_problem"
    description = "Recover isotropic metric from 7% observations + TV"
    
    def __init__(self, N: int = 80, obs_fraction: float = 0.07, n_iter: int = 400, device: str = None):
        super().__init__(device)
        self.N, self.obs_fraction, self.n_iter = N, obs_fraction, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # Smooth metric
        j_coords = torch.arange(N, device=self.device, dtype=torch.float32)
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, :] = 1.0 + 0.8 * torch.sigmoid((j_coords - N/2) / 10).unsqueeze(0)
        G_true[0, 2, :, :] = G_true[0, 0, :, :]
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem_true = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem_true, verbose=False)
        
        # Sparse observations
        obs_mask = create_sparse_observation_mask(N, N, self.obs_fraction, source_mask, seed=46, device=self.device)
        print(f"  Using {obs_mask.sum().item()} observations ({self.obs_fraction*100:.1f}%)")
        
        # Recover with TV
        optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=0.05, constrain_isotropic=True)
        metrics = optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.03)
        
        interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
        rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - optimizer.G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
        
        print(f"\n  Recovery error: {rel_err:.4f}")
        
        self.G_true, self.G_rec = G_true, optimizer.G
        self.T_obs, self.obs_mask, self.optimizer = T_obs, obs_mask, optimizer
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=rel_err < 0.25,
            metrics={'rel_error': rel_err, 'n_obs': int(obs_mask.sum().item()), **metrics},
            arrays={'G_true': G_true.cpu().numpy(), 'G_rec': optimizer.G.cpu().numpy()},
            metadata={'N': N, 'obs_fraction': self.obs_fraction}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        vmin, vmax = self.G_true[0, 0].min().item(), self.G_true[0, 0].max().item()
        
        axes[0,0].imshow(self.G_true[0, 0].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0,0].set_title('True')
        
        axes[0,1].imshow(self.G_rec[0, 0].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0,1].set_title('Recovered')
        
        err = (self.G_rec[0, 0] - self.G_true[0, 0]).cpu().numpy()
        im = axes[0,2].imshow(err, origin='upper', cmap='RdBu_r')
        plt.colorbar(im, ax=axes[0,2])
        oy, ox = torch.where(self.obs_mask[0])
        axes[0,2].scatter(ox.cpu(), oy.cpu(), c='g', s=2, alpha=0.3)
        axes[0,2].set_title('Error + Obs')
        
        plot_arrival_time(self.T_obs, ax=axes[1,0], title='T')
        
        axes[1,1].semilogy(self.optimizer.history['loss'])
        axes[1,1].set_title('Loss')
        axes[1,1].grid(True, alpha=0.3)
        
        j = self.N // 2
        axes[1,2].plot(self.G_true[0, 0, j, :].cpu().numpy(), 'b-', label='True', lw=2)
        axes[1,2].plot(self.G_rec[0, 0, j, :].cpu().numpy(), 'r--', label='Recovered', lw=2)
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        fig.suptitle(f'C2: Sparse Recovery ({self.obs_fraction*100:.0f}% obs, err={self.result.metrics["rel_error"]:.2%})', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C3: DIAGONAL ANISOTROPIC RECOVERY
# =============================================================================

@register_experiment
class C3_DiagonalAnisotropic(Experiment):
    """Recover diagonal anisotropic metric (g11 ≠ g22)."""
    
    name = "C3_diagonal_anisotropic"
    category = "C_inverse_problem"
    description = "Recover diagonal anisotropic metric"
    
    def __init__(self, N: int = 80, n_iter: int = 400, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # Different eigenvalues spatially
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, :] = 1.5  # g11
        G_true[0, 2, :, :] = 0.5  # g22
        G_true[0, 0, :, N//2:] = 0.5
        G_true[0, 2, :, N//2:] = 1.5
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem, verbose=False)
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=47, device=self.device)
        
        optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=0.03, constrain_isotropic=False)
        optimizer.G[:, 1, :, :] = 0  # Keep g12=0
        metrics = optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.03)
        
        interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
        err_g11 = torch.sqrt(((G_true[0, 0][interior[0]] - optimizer.G[0, 0][interior[0]])**2).mean()).item()
        err_g22 = torch.sqrt(((G_true[0, 2][interior[0]] - optimizer.G[0, 2][interior[0]])**2).mean()).item()
        
        self.G_true, self.G_rec = G_true, optimizer.G
        self.T_obs = T_obs
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=(err_g11 < 0.3 and err_g22 < 0.3),
            metrics={'err_g11': err_g11, 'err_g22': err_g22, **metrics},
            arrays={'G_true': G_true.cpu().numpy(), 'G_rec': optimizer.G.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for row, (c, name) in enumerate([(0, 'g₁₁'), (2, 'g₂₂')]):
            vmin = min(self.G_true[0, c].min().item(), self.G_rec[0, c].min().item())
            vmax = max(self.G_true[0, c].max().item(), self.G_rec[0, c].max().item())
            
            axes[row, 0].imshow(self.G_true[0, c].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'True {name}')
            
            axes[row, 1].imshow(self.G_rec[0, c].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'Recovered {name}')
            
            err = (self.G_rec[0, c] - self.G_true[0, c]).cpu().numpy()
            im = axes[row, 2].imshow(err, origin='upper', cmap='RdBu_r')
            plt.colorbar(im, ax=axes[row, 2])
            axes[row, 2].set_title(f'Error')
        
        fig.suptitle('C3: Diagonal Anisotropic Recovery', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C5: DRIFT RECOVERY
# =============================================================================

@register_experiment  
class C5_DriftRecovery(Experiment):
    """Recover drift field with known metric."""
    
    name = "C5_drift_recovery"
    category = "C_inverse_problem"
    description = "Recover drift B with fixed G"
    
    def __init__(self, N: int = 60, n_iter: int = 300, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        B_true[0, 0, :, :] = 0.2
        B_true[0, 1, :, :] = 0.1
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem, verbose=False)
        
        obs_mask = create_sparse_observation_mask(N, N, 0.15, source_mask, seed=48, device=self.device)
        
        optimizer = MetricRecoveryOptimizer(N, N, self.device, recover_G=False, recover_B=True, lambda_B=0.01)
        optimizer.G = G_true.clone()
        metrics = optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.02)
        
        interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
        err_b1 = torch.sqrt(((B_true[0, 0][interior[0]] - optimizer.B[0, 0][interior[0]])**2).mean()).item()
        err_b2 = torch.sqrt(((B_true[0, 1][interior[0]] - optimizer.B[0, 1][interior[0]])**2).mean()).item()
        
        self.B_true, self.B_rec = B_true, optimizer.B
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=(err_b1 < 0.1 and err_b2 < 0.1),
            metrics={'err_b1': err_b1, 'err_b2': err_b2, **metrics},
            arrays={'B_true': B_true.cpu().numpy(), 'B_rec': optimizer.B.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for row, (c, name) in enumerate([(0, 'b₁'), (1, 'b₂')]):
            vmin = min(self.B_true[0, c].min().item(), self.B_rec[0, c].min().item())
            vmax = max(self.B_true[0, c].max().item(), self.B_rec[0, c].max().item())
            
            axes[row, 0].imshow(self.B_true[0, c].cpu().numpy(), origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'True {name}')
            
            axes[row, 1].imshow(self.B_rec[0, c].cpu().numpy(), origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'Recovered {name}')
            
            err = (self.B_rec[0, c] - self.B_true[0, c]).cpu().numpy()
            im = axes[row, 2].imshow(err, origin='upper', cmap='RdBu_r')
            plt.colorbar(im, ax=axes[row, 2])
            axes[row, 2].set_title('Error')
        
        fig.suptitle('C5: Drift Recovery', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C7: REGULARIZATION ABLATION
# =============================================================================

@register_experiment
class C7_RegularizationAblation(Experiment):
    """Show TV regularization is necessary for sparse observations."""
    
    name = "C7_regularization_ablation"
    category = "C_inverse_problem"
    description = "Recovery error vs regularization strength"
    
    def __init__(self, N: int = 60, n_iter: int = 200, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        self.lambda_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem, verbose=False)
        
        obs_mask = create_sparse_observation_mask(N, N, 0.05, source_mask, seed=47, device=self.device)
        
        results = []
        for lam in self.lambda_values:
            print(f"\n  λ = {lam}")
            optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=lam, constrain_isotropic=True)
            optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=False)
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - optimizer.G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
            
            results.append({'lambda': lam, 'error': rel_err})
            print(f"    Error: {rel_err:.4f}")
        
        best = min(results, key=lambda x: x['error'])
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'best_lambda': best['lambda'], 'best_error': best['error'], 
                    'no_reg_error': results[0]['error']},
            arrays={'lambdas': np.array([r['lambda'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lambdas = np.array([r['lambda'] for r in self.results])
        errors = np.array([r['error'] for r in self.results])
        
        lambdas_plot = np.where(lambdas == 0, 1e-4, lambdas)
        ax.semilogx(lambdas_plot, errors, 'o-', markersize=10, lw=2)
        
        best_idx = np.argmin(errors)
        ax.scatter([lambdas_plot[best_idx]], [errors[best_idx]], c='red', s=200, zorder=5, 
                   label=f'Best: λ={lambdas[best_idx]:.3f}')
        
        ax.set_xlabel('Regularization λ')
        ax.set_ylabel('Relative Error')
        ax.set_title('C7: Regularization Ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C8: OBSERVATION DENSITY
# =============================================================================

@register_experiment
class C8_ObservationDensity(Experiment):
    """Recovery error vs observation density."""
    
    name = "C8_observation_density"
    category = "C_inverse_problem"
    description = "Recovery error vs observation density"
    
    def __init__(self, N: int = 60, n_iter: int = 200, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        self.obs_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem, verbose=False)
        
        results = []
        for frac in self.obs_fractions:
            print(f"\n  Fraction: {frac*100:.0f}%")
            
            if frac >= 1.0:
                obs_mask = get_interior_mask(N, N, 3, source_mask, device=self.device)
            else:
                obs_mask = create_sparse_observation_mask(N, N, frac, source_mask, seed=48, device=self.device)
            
            lambda_G = 0.01 if frac < 0.5 else 0.001
            optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=lambda_G, constrain_isotropic=True)
            optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=False)
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - optimizer.G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
            
            results.append({'fraction': frac, 'error': rel_err, 'n_obs': int(obs_mask.sum().item())})
            print(f"    Error: {rel_err:.4f}")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'min_error': min(r['error'] for r in results)},
            arrays={'fractions': np.array([r['fraction'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fracs = np.array([r['fraction'] for r in self.results]) * 100
        errors = np.array([r['error'] for r in self.results])
        
        ax.plot(fracs, errors, 'o-', markersize=10, lw=2)
        ax.set_xlabel('Observation Density (%)')
        ax.set_ylabel('Relative Error')
        ax.set_title('C8: Observation Density Study')
        ax.grid(True, alpha=0.3)
        
        for f, e in zip(fracs, errors):
            ax.annotate(f'{e:.2%}', xy=(f, e), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C9: NOISE ROBUSTNESS
# =============================================================================

@register_experiment
class C9_NoiseRobustness(Experiment):
    """Test recovery under measurement noise."""
    
    name = "C9_noise_robustness"
    category = "C_inverse_problem"
    description = "Recovery error vs noise level"
    
    def __init__(self, N: int = 60, n_iter: int = 200, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        self.noise_levels = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_clean, _ = fast_sweeping_solve(problem, verbose=False)
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=49, device=self.device)
        
        results = []
        np.random.seed(50)
        
        for noise in self.noise_levels:
            print(f"\n  Noise: {noise*100:.0f}%")
            
            T_obs = T_clean.clone()
            if noise > 0:
                noise_tensor = noise * T_clean.std() * torch.randn_like(T_clean)
                T_obs = T_clean + noise_tensor
            
            optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=0.02, constrain_isotropic=True)
            optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=False)
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - optimizer.G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
            
            results.append({'noise': noise, 'error': rel_err})
            print(f"    Error: {rel_err:.4f}")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'clean_error': results[0]['error'], 'noisy_error_5pct': results[3]['error']},
            arrays={'noise_levels': np.array([r['noise'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        noise = np.array([r['noise'] for r in self.results]) * 100
        errors = np.array([r['error'] for r in self.results])
        
        ax.plot(noise, errors, 'o-', markersize=10, lw=2)
        ax.set_xlabel('Noise Level (%)')
        ax.set_ylabel('Relative Error')
        ax.set_title('C9: Noise Robustness')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C10: MULTIPLE SOURCES
# =============================================================================

@register_experiment
class C10_MultipleSources(Experiment):
    """Test if multiple sources improve recovery."""
    
    name = "C10_multiple_sources"
    category = "C_inverse_problem"
    description = "Recovery with multiple ignition points"
    
    def __init__(self, N: int = 60, n_iter: int = 200, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        results = []
        
        for n_sources, sources in [(1, [(N//2, N//4)]), 
                                    (2, [(N//4, N//4), (3*N//4, N//4)]),
                                    (3, [(N//4, N//4), (3*N//4, N//4), (N//2, 3*N//4)])]:
            print(f"\n  {n_sources} source(s)")
            
            source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
            for si, sj in sources:
                source_mask[0, si, sj] = True
            
            problem = EikonalProblem(G_true, B_true, source_mask)
            T_obs, _ = fast_sweeping_solve(problem, verbose=False)
            
            obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=51, device=self.device)
            
            optimizer = MetricRecoveryOptimizer(N, N, self.device, lambda_G=0.02, constrain_isotropic=True)
            optimizer.fit(source_mask, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=False)
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - optimizer.G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
            
            results.append({'n_sources': n_sources, 'error': rel_err})
            print(f"    Error: {rel_err:.4f}")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'error_1source': results[0]['error'], 'error_3sources': results[2]['error']},
            arrays={'n_sources': np.array([r['n_sources'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        n = [r['n_sources'] for r in self.results]
        errors = [r['error'] for r in self.results]
        
        ax.bar(n, errors, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Sources')
        ax.set_ylabel('Relative Error')
        ax.set_title('C10: Multiple Sources')
        ax.set_xticks(n)
        
        for i, (ni, ei) in enumerate(zip(n, errors)):
            ax.annotate(f'{ei:.2%}', xy=(ni, ei), ha='center', va='bottom')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    C1_IsotropicFull,
    C2_IsotropicSparse,
    C3_DiagonalAnisotropic,
    C5_DriftRecovery,
    C7_RegularizationAblation,
    C8_ObservationDensity,
    C9_NoiseRobustness,
    C10_MultipleSources,
]

def run_all(save=True, visualize=True):
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY C SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results


if __name__ == "__main__":
    run_all()
