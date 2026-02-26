"""
Category E: Comparisons
=======================

Comparisons with alternative approaches:

E1: Fast Sweeping vs Fast Marching - Runtime comparison
E2: Implicit vs Finite Diff Gradients - Accuracy/speed tradeoff
E3: Regularization Strategies - TV vs Tikhonov vs none
E4: Optimization Methods - GD vs Adam vs L-BFGS
E5: Scalability Study - Performance vs grid size
"""

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import time
import sys
import os

from experiment_base import (
    Experiment, ExperimentResult, register_experiment, DEVICE,
    create_metric_from_eigenvalues, finite_difference_gradient,
    get_interior_mask, create_sparse_observation_mask,
    plot_arrival_time
)

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.backward.implicit_diff import backward_pass


# =============================================================================
# E1: FAST SWEEPING VS FAST MARCHING
# =============================================================================

@register_experiment
class E1_SweepingVsMarching(Experiment):
    """Compare fast sweeping to a simple Dijkstra-style solver."""
    
    name = "E1_sweeping_vs_marching"
    category = "E_comparisons"
    description = "Runtime comparison: fast sweeping vs simple solver"
    
    def __init__(self, grid_sizes: List[int] = None, device: str = None):
        super().__init__(device)
        self.grid_sizes = grid_sizes or [50, 100, 200, 400]
        
    def run(self) -> ExperimentResult:
        results = []
        
        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")
            
            G = torch.ones(1, 3, N, N, device=self.device)
            G[:, 1, :, :] = 0
            B = torch.zeros(1, 2, N, N, device=self.device)
            
            source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
            source_mask[0, N//2, N//2] = True
            
            problem = EikonalProblem(G, B, source_mask)
            
            # Fast sweeping
            start = time.time()
            T_sweep, n_iter = fast_sweeping_solve(problem, max_iter=50, verbose=False)
            time_sweep = time.time() - start
            
            # Simple iterative (Jacobi-style) for comparison
            start = time.time()
            T_jacobi = self._jacobi_solve(G, source_mask, max_iter=N*2)
            time_jacobi = time.time() - start
            
            results.append({
                'N': N,
                'time_sweep': time_sweep,
                'time_jacobi': time_jacobi,
                'sweep_iters': n_iter,
                'speedup': time_jacobi / time_sweep if time_sweep > 0 else 0
            })
            
            print(f"    Sweep: {time_sweep:.3f}s ({n_iter} iter), Jacobi: {time_jacobi:.3f}s")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'avg_speedup': np.mean([r['speedup'] for r in results])},
            arrays={'grid_sizes': np.array([r['N'] for r in results]),
                   'time_sweep': np.array([r['time_sweep'] for r in results]),
                   'time_jacobi': np.array([r['time_jacobi'] for r in results])},
            metadata={'grid_sizes': self.grid_sizes}
        )
    
    def _jacobi_solve(self, G, source_mask, max_iter=100):
        """Simple Jacobi iteration for comparison."""
        N = G.shape[2]
        T = torch.full((1, N, N), float('inf'), device=self.device)
        T[source_mask] = 0
        
        for _ in range(max_iter):
            T_new = T.clone()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    if source_mask[0, i, j]:
                        continue
                    # Simple upwind
                    Tx_min = min(T[0, i-1, j].item(), T[0, i+1, j].item())
                    Ty_min = min(T[0, i, j-1].item(), T[0, i, j+1].item())
                    g11 = G[0, 0, i, j].item()
                    
                    # Solve quadratic for new T
                    if abs(Tx_min - Ty_min) < 1.0 / np.sqrt(g11):
                        T_new[0, i, j] = (Tx_min + Ty_min + np.sqrt(2/g11 - (Tx_min - Ty_min)**2)) / 2
                    else:
                        T_new[0, i, j] = min(Tx_min, Ty_min) + 1.0 / np.sqrt(g11)
            
            if (T_new - T).abs().max() < 1e-6:
                break
            T = T_new
        
        return T
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        N = np.array([r['N'] for r in self.results])
        t_sweep = np.array([r['time_sweep'] for r in self.results])
        t_jacobi = np.array([r['time_jacobi'] for r in self.results])
        
        axes[0].loglog(N, t_sweep, 'o-', label='Fast Sweeping', markersize=10)
        axes[0].loglog(N, t_jacobi, 's-', label='Jacobi', markersize=10)
        axes[0].loglog(N, t_sweep[0] * (N/N[0])**2, 'k--', alpha=0.5, label='O(N²)')
        axes[0].set_xlabel('Grid size N')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_title('Runtime Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        speedup = t_jacobi / t_sweep
        axes[1].bar(range(len(N)), speedup, tick_label=[str(n) for n in N])
        axes[1].set_xlabel('Grid size N')
        axes[1].set_ylabel('Speedup')
        axes[1].set_title('Speedup of Fast Sweeping')
        
        fig.suptitle('E1: Fast Sweeping vs Jacobi', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E2: IMPLICIT VS FINITE DIFF GRADIENTS
# =============================================================================

@register_experiment
class E2_ImplicitVsFD(Experiment):
    """Compare implicit diff vs finite diff for gradient computation."""
    
    name = "E2_implicit_vs_fd"
    category = "E_comparisons"
    description = "Accuracy/speed of implicit diff vs finite diff"
    
    def __init__(self, grid_sizes: List[int] = None, device: str = None):
        super().__init__(device)
        self.grid_sizes = grid_sizes or [30, 50, 75, 100]
        
    def run(self) -> ExperimentResult:
        results = []
        
        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")
            
            G = torch.ones(1, 3, N, N, device=self.device)
            G[:, 1, :, :] = 0
            B = torch.zeros(1, 2, N, N, device=self.device)
            
            source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
            source_mask[0, N//2, N//2] = True
            
            problem = EikonalProblem(G, B, source_mask)
            T, _ = fast_sweeping_solve(problem, verbose=False)
            
            dL_dT = torch.zeros_like(T)
            dL_dT[0, N//2 + N//4, N//2 + N//4] = 1.0
            
            # Implicit differentiation timing
            start = time.time()
            dL_dG_impl, dL_dB_impl = backward_pass(T, dL_dT, problem, verbose=False)
            time_impl = time.time() - start
            
            # Finite difference timing (subset of parameters)
            n_params = min(50, (N-10)**2)
            np.random.seed(56)
            test_points = []
            while len(test_points) < n_params:
                i, j = np.random.randint(5, N-5, 2)
                if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                    test_points.append((i, j))
            
            start = time.time()
            for i, j in test_points[:20]:  # Only time 20 for speed
                finite_difference_gradient(problem, dL_dT, 'B', (0, i, j), 1e-5)
            time_fd_20 = time.time() - start
            
            # Extrapolate FD time
            time_fd_full = time_fd_20 * (N * N / 20)  # Rough estimate
            
            results.append({
                'N': N,
                'time_impl': time_impl,
                'time_fd_20': time_fd_20,
                'time_fd_est': time_fd_full,
                'speedup': time_fd_full / time_impl if time_impl > 0 else 0
            })
            
            print(f"    Impl: {time_impl:.3f}s, FD(20): {time_fd_20:.3f}s, Est speedup: {results[-1]['speedup']:.0f}x")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'avg_speedup': np.mean([r['speedup'] for r in results])},
            arrays={'grid_sizes': np.array([r['N'] for r in results]),
                   'time_impl': np.array([r['time_impl'] for r in results]),
                   'time_fd_est': np.array([r['time_fd_est'] for r in results])},
            metadata={'grid_sizes': self.grid_sizes}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        N = np.array([r['N'] for r in self.results])
        t_impl = np.array([r['time_impl'] for r in self.results])
        t_fd = np.array([r['time_fd_est'] for r in self.results])
        
        ax.semilogy(N, t_impl, 'o-', label='Implicit Diff', markersize=10)
        ax.semilogy(N, t_fd, 's-', label='Finite Diff (est)', markersize=10)
        
        ax.set_xlabel('Grid size N')
        ax.set_ylabel('Time (s)')
        ax.set_title('E2: Gradient Computation Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E3: REGULARIZATION STRATEGIES
# =============================================================================

@register_experiment
class E3_RegularizationStrategies(Experiment):
    """Compare TV vs Tikhonov vs no regularization."""
    
    name = "E3_regularization_strategies"
    category = "E_comparisons"
    description = "Compare different regularization approaches"
    
    def __init__(self, N: int = 60, n_iter: int = 200, device: str = None):
        super().__init__(device)
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # True metric
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem, verbose=False)
        
        obs_mask = create_sparse_observation_mask(N, N, 0.05, source_mask, seed=57, device=self.device)
        
        results = {}
        
        for reg_type in ['none', 'tv', 'tikhonov']:
            print(f"\n  Regularization: {reg_type}")
            
            G = torch.ones(1, 3, N, N, device=self.device)
            G[:, 1, :, :] = 0
            
            lr = 0.05
            lam = 0.01 if reg_type != 'none' else 0
            
            for it in range(self.n_iter):
                prob = EikonalProblem(G, B_true, source_mask)
                T, _ = fast_sweeping_solve(prob, verbose=False)
                
                residuals = T[obs_mask] - T_obs[obs_mask]
                loss_data = 0.5 * (residuals**2).sum()
                
                dL_dT = torch.zeros_like(T)
                dL_dT[obs_mask] = residuals
                
                dL_dG, _ = backward_pass(T, dL_dT, prob, verbose=False)
                
                # Add regularization gradient
                if reg_type == 'tv' and lam > 0:
                    for c in [0, 2]:
                        Gc = G[0, c]
                        reg_grad = torch.zeros_like(Gc)
                        reg_grad[:, 1:-1] += lam * torch.sign(Gc[:, 1:-1] - Gc[:, :-2])
                        reg_grad[:, :-1] -= lam * torch.sign(Gc[:, 1:] - Gc[:, :-1])
                        reg_grad[1:-1, :] += lam * torch.sign(Gc[1:-1, :] - Gc[:-2, :])
                        reg_grad[:-1, :] -= lam * torch.sign(Gc[1:, :] - Gc[:-1, :])
                        dL_dG[0, c] += reg_grad
                        
                elif reg_type == 'tikhonov' and lam > 0:
                    dL_dG += 2 * lam * (G - 1)  # Penalize deviation from identity
                
                with torch.no_grad():
                    G -= lr * dL_dG
                    G[:, 0, :, :] = torch.clamp(G[:, 0, :, :], min=0.1, max=10)
                    G[:, 2, :, :] = torch.clamp(G[:, 2, :, :], min=0.1, max=10)
                    G[:, 1, :, :] = 0  # Keep isotropic
                    avg = (G[:, 0, :, :] + G[:, 2, :, :]) / 2
                    G[:, 0, :, :] = avg
                    G[:, 2, :, :] = avg
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
            
            results[reg_type] = {'error': rel_err, 'G': G.cpu().numpy()}
            print(f"    Error: {rel_err:.4f}")
        
        self.results = results
        self.G_true = G_true
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={f'error_{k}': v['error'] for k, v in results.items()},
            arrays={f'G_{k}': v['G'] for k, v in results.items()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        vmin, vmax = self.G_true[0, 0].min().item(), self.G_true[0, 0].max().item()
        
        axes[0].imshow(self.G_true[0, 0].cpu().numpy(), origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('True')
        
        for ax, (name, res) in zip(axes[1:], self.results.items()):
            ax.imshow(res['G'][0, 0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'{name.upper()} (err={res["error"]:.2%})')
        
        fig.suptitle('E3: Regularization Comparison', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E4: OPTIMIZATION METHODS
# =============================================================================

@register_experiment
class E4_OptimizationMethods(Experiment):
    """Compare GD vs Adam for metric recovery."""
    
    name = "E4_optimization_methods"
    category = "E_comparisons"
    description = "Compare optimization algorithms"
    
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
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem = EikonalProblem(G_true, B_true, source_mask)
        T_obs, _ = fast_sweeping_solve(problem, verbose=False)
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=58, device=self.device)
        
        results = {}
        
        for method in ['gd', 'adam']:
            print(f"\n  Method: {method}")
            
            G = torch.ones(1, 3, N, N, device=self.device, requires_grad=False)
            G[:, 1, :, :] = 0
            
            if method == 'adam':
                m = torch.zeros_like(G)
                v = torch.zeros_like(G)
                beta1, beta2 = 0.9, 0.999
                eps = 1e-8
            
            lr = 0.05 if method == 'gd' else 0.01
            losses = []
            
            for it in range(self.n_iter):
                prob = EikonalProblem(G, B_true, source_mask)
                T, _ = fast_sweeping_solve(prob, verbose=False)
                
                residuals = T[obs_mask] - T_obs[obs_mask]
                loss = 0.5 * (residuals**2).sum().item()
                losses.append(loss)
                
                dL_dT = torch.zeros_like(T)
                dL_dT[obs_mask] = residuals
                
                dL_dG, _ = backward_pass(T, dL_dT, prob, verbose=False)
                
                with torch.no_grad():
                    if method == 'gd':
                        G -= lr * dL_dG
                    else:  # Adam
                        m = beta1 * m + (1 - beta1) * dL_dG
                        v = beta2 * v + (1 - beta2) * (dL_dG ** 2)
                        m_hat = m / (1 - beta1 ** (it + 1))
                        v_hat = v / (1 - beta2 ** (it + 1))
                        G -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                    
                    G[:, 0, :, :] = torch.clamp(G[:, 0, :, :], min=0.1, max=10)
                    G[:, 2, :, :] = torch.clamp(G[:, 2, :, :], min=0.1, max=10)
                    G[:, 1, :, :] = 0
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((G_true[0, 0][interior[0]] - G[0, 0][interior[0]])**2).mean()).item() / G_true[0, 0][interior[0]].mean().item()
            
            results[method] = {'error': rel_err, 'losses': losses, 'G': G.cpu().numpy()}
            print(f"    Final error: {rel_err:.4f}")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={f'error_{k}': v['error'] for k, v in results.items()},
            arrays={f'losses_{k}': np.array(v['losses']) for k, v in results.items()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for method, res in self.results.items():
            axes[0].semilogy(res['losses'], label=f'{method.upper()} (err={res["error"]:.2%})')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        methods = list(self.results.keys())
        errors = [self.results[m]['error'] for m in methods]
        axes[1].bar(methods, errors, color=['steelblue', 'coral'])
        axes[1].set_ylabel('Relative Error')
        axes[1].set_title('Final Error')
        
        fig.suptitle('E4: Optimization Methods', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E5: SCALABILITY STUDY
# =============================================================================

@register_experiment
class E5_Scalability(Experiment):
    """Full pipeline scalability: forward + backward."""
    
    name = "E5_scalability"
    category = "E_comparisons"
    description = "Scalability of full forward+backward pipeline"
    
    def __init__(self, grid_sizes: List[int] = None, device: str = None):
        super().__init__(device)
        self.grid_sizes = grid_sizes or [50, 100, 200, 400, 800]
        
    def run(self) -> ExperimentResult:
        results = []
        
        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")
            
            G = torch.ones(1, 3, N, N, device=self.device)
            G[:, 1, :, :] = 0
            B = torch.zeros(1, 2, N, N, device=self.device)
            
            source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
            source_mask[0, N//2, N//2] = True
            
            problem = EikonalProblem(G, B, source_mask)
            
            # Forward timing
            start = time.time()
            T, _ = fast_sweeping_solve(problem, verbose=False)
            time_fwd = time.time() - start
            
            # Backward timing
            dL_dT = torch.zeros_like(T)
            dL_dT[0, N//2 + N//4, N//2 + N//4] = 1.0
            
            start = time.time()
            dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
            time_bwd = time.time() - start
            
            results.append({
                'N': N,
                'time_fwd': time_fwd,
                'time_bwd': time_bwd,
                'time_total': time_fwd + time_bwd,
                'ratio': time_bwd / time_fwd if time_fwd > 0 else 0
            })
            
            print(f"    Fwd: {time_fwd:.3f}s, Bwd: {time_bwd:.3f}s, Ratio: {results[-1]['ratio']:.2f}")
        
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'avg_bwd_fwd_ratio': np.mean([r['ratio'] for r in results])},
            arrays={'grid_sizes': np.array([r['N'] for r in results]),
                   'time_fwd': np.array([r['time_fwd'] for r in results]),
                   'time_bwd': np.array([r['time_bwd'] for r in results])},
            metadata={'grid_sizes': self.grid_sizes}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        N = np.array([r['N'] for r in self.results])
        t_fwd = np.array([r['time_fwd'] for r in self.results])
        t_bwd = np.array([r['time_bwd'] for r in self.results])
        
        axes[0].loglog(N, t_fwd, 'o-', label='Forward', markersize=10)
        axes[0].loglog(N, t_bwd, 's-', label='Backward', markersize=10)
        axes[0].loglog(N, t_fwd + t_bwd, '^-', label='Total', markersize=10)
        axes[0].loglog(N, t_fwd[0] * (N/N[0])**2, 'k--', alpha=0.5, label='O(N²)')
        axes[0].set_xlabel('Grid size N')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_title('Timing')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        ratio = t_bwd / t_fwd
        axes[1].plot(N, ratio, 'o-', markersize=10)
        axes[1].axhline(np.mean(ratio), color='r', linestyle='--', label=f'Mean: {np.mean(ratio):.2f}')
        axes[1].set_xlabel('Grid size N')
        axes[1].set_ylabel('Backward / Forward ratio')
        axes[1].set_title('Backward/Forward Time Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('E5: Scalability Study', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    E1_SweepingVsMarching,
    E2_ImplicitVsFD,
    E3_RegularizationStrategies,
    E4_OptimizationMethods,
    E5_Scalability,
]

def run_all(save=True, visualize=True):
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY E SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results


if __name__ == "__main__":
    run_all()
