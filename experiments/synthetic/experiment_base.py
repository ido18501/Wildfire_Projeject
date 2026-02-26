"""
SGP Paper Experiments - Base Infrastructure
===========================================

Comprehensive utilities for all experiments:
- Experiment base class with logging, timing, saving
- Ground truth computation helpers  
- Visualization utilities
- Metric computation (errors, convergence rates)
- Finite difference gradient verification

Author: Barak
Paper: Differentiable Randers-Finsler Eikonal Solver
"""

import os
import sys
import json
import time
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =============================================================================
# PATH CONFIGURATION - ADJUST FOR YOUR LOCAL SETUP
# =============================================================================

# Set this environment variable to your project path, or modify the default
# Project imports handled via installed eikonal package

# Import from your project
try:
    from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve, create_test_problem
    from eikonal.backward.implicit_diff import backward_pass, identify_active_stencils, UpdateType, StencilInfo
    IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Could not import from project: {e}")
    print(f"Make sure the eikonal package is installed: pip install -e .")
    IMPORTS_OK = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Paper-quality figure settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# EXPERIMENT RESULT CONTAINER
# =============================================================================

@dataclass
class ExperimentResult:
    """Container for experiment results."""
    name: str
    category: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def save(self, base_dir: str = None):
        """Save results to disk."""
        if base_dir is None:
            base_dir = RESULTS_DIR
            
        exp_dir = os.path.join(base_dir, self.category, self.name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save metrics and metadata as JSON
        summary = {
            'name': self.name,
            'category': self.category,
            'success': self.success,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in self.metrics.items()},
            'metadata': self.metadata,
            'runtime_seconds': self.runtime_seconds,
            'timestamp': self.timestamp,
        }
        with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save arrays as npz
        if self.arrays:
            np.savez_compressed(os.path.join(exp_dir, 'arrays.npz'), **self.arrays)
        
        return exp_dir
    
    @classmethod
    def load(cls, exp_dir: str) -> 'ExperimentResult':
        """Load results from disk."""
        with open(os.path.join(exp_dir, 'summary.json'), 'r') as f:
            summary = json.load(f)
        
        arrays = {}
        arrays_path = os.path.join(exp_dir, 'arrays.npz')
        if os.path.exists(arrays_path):
            with np.load(arrays_path) as data:
                arrays = dict(data)
        
        return cls(
            name=summary['name'],
            category=summary['category'],
            success=summary['success'],
            metrics=summary['metrics'],
            arrays=arrays,
            metadata=summary.get('metadata', {}),
            runtime_seconds=summary.get('runtime_seconds', 0),
            timestamp=summary.get('timestamp', ''),
        )


# =============================================================================
# EXPERIMENT BASE CLASS
# =============================================================================

class Experiment(ABC):
    """Base class for all experiments."""
    
    name: str = "base_experiment"
    category: str = "uncategorized"
    description: str = "Base experiment class"
    
    def __init__(self, device: str = None):
        self.device = device or DEVICE
        self.result = None
        
    def setup(self) -> None:
        """Setup experiment (override if needed)."""
        pass
    
    @abstractmethod
    def run(self) -> ExperimentResult:
        """Run the experiment (must override)."""
        raise NotImplementedError
    
    @abstractmethod  
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization (must override)."""
        raise NotImplementedError
    
    def execute(self, save: bool = True, visualize: bool = True, 
                verbose: bool = True) -> ExperimentResult:
        """Full execution: setup, run, save, visualize."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: {self.name}")
            print(f"Category: {self.category}")
            print(f"{'='*60}")
        
        self.setup()
        
        start_time = time.time()
        self.result = self.run()
        self.result.runtime_seconds = time.time() - start_time
        
        if verbose:
            print(f"\nCompleted in {self.result.runtime_seconds:.2f}s")
            print(f"Success: {self.result.success}")
            
            if self.result.metrics:
                print("\nMetrics:")
                for k, v in self.result.metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.6f}")
                    else:
                        print(f"  {k}: {v}")
        
        if save:
            exp_dir = self.result.save()
            if verbose:
                print(f"\nResults saved to: {exp_dir}")
        
        if visualize:
            fig_dir = os.path.join(FIGURES_DIR, self.category)
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{self.name}.pdf")
            try:
                fig = self.visualize(save_path=fig_path)
                if fig is not None:
                    plt.close(fig)
                    if verbose:
                        print(f"Figure saved to: {fig_path}")
            except Exception as e:
                print(f"Visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        return self.result


# =============================================================================
# GROUND TRUTH COMPUTATION
# =============================================================================

def euclidean_distance_field(M: int, N: int, source_i: int, source_j: int, 
                              device: str = DEVICE) -> Tensor:
    """Compute exact Euclidean distance from source point."""
    i_coords = torch.arange(M, device=device, dtype=torch.float32)
    j_coords = torch.arange(N, device=device, dtype=torch.float32)
    I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')
    return torch.sqrt((I - source_i)**2 + (J - source_j)**2)


def anisotropic_distance_field(M: int, N: int, source_i: int, source_j: int,
                                lambda1: float, lambda2: float, theta: float = 0.0,
                                device: str = DEVICE) -> Tensor:
    """
    Compute exact distance for anisotropic metric G = R^T diag(λ1, λ2) R.
    T = sqrt(λ1 * x_rot² + λ2 * y_rot²)
    """
    i_coords = torch.arange(M, device=device, dtype=torch.float32) - source_i
    j_coords = torch.arange(N, device=device, dtype=torch.float32) - source_j
    Y, X = torch.meshgrid(i_coords, j_coords, indexing='ij')
    
    c, s = np.cos(theta), np.sin(theta)
    X_rot = c * X + s * Y
    Y_rot = -s * X + c * Y
    
    return torch.sqrt(lambda1 * X_rot**2 + lambda2 * Y_rot**2)


def create_metric_from_eigenvalues(M: int, N: int, lambda1: float, lambda2: float,
                                    theta: float = 0.0, device: str = DEVICE) -> Tensor:
    """Create metric tensor G = R^T diag(λ1, λ2) R."""
    c, s = np.cos(theta), np.sin(theta)
    g11 = lambda1 * c**2 + lambda2 * s**2
    g12 = (lambda1 - lambda2) * c * s
    g22 = lambda1 * s**2 + lambda2 * c**2
    
    G = torch.zeros(1, 3, M, N, device=device)
    G[0, 0, :, :] = g11
    G[0, 1, :, :] = g12
    G[0, 2, :, :] = g22
    return G


# =============================================================================
# ERROR METRICS
# =============================================================================

def compute_errors(T_computed: Tensor, T_exact: Tensor, 
                   mask: Optional[Tensor] = None) -> Dict[str, float]:
    """Compute error metrics between computed and exact solutions."""
    if mask is None:
        mask = torch.ones_like(T_computed, dtype=torch.bool)
    if mask.dim() != T_computed.dim():
        mask = mask.unsqueeze(0)
    
    valid = mask & torch.isfinite(T_computed) & torch.isfinite(T_exact)
    if valid.sum() == 0:
        return {'l1': float('nan'), 'l2': float('nan'), 'linf': float('nan'), 
                'rel_l2': float('nan'), 'rel_linf': float('nan')}
    
    diff = (T_computed - T_exact)[valid]
    T_ex = T_exact[valid]
    
    l1 = torch.abs(diff).mean().item()
    l2 = torch.sqrt((diff**2).mean()).item()
    linf = torch.abs(diff).max().item()
    
    T_scale = torch.abs(T_ex).max().item()
    rel_l2 = l2 / T_scale if T_scale > 1e-10 else float('nan')
    rel_linf = linf / T_scale if T_scale > 1e-10 else float('nan')
    
    return {'l1': l1, 'l2': l2, 'linf': linf, 'rel_l2': rel_l2, 'rel_linf': rel_linf}


def compute_convergence_rate(h_values: List[float], errors: List[float]) -> Tuple[float, float]:
    """Compute convergence rate via log-log linear regression."""
    log_h = np.log(np.array(h_values))
    log_e = np.log(np.array(errors))
    
    A = np.vstack([log_h, np.ones_like(log_h)]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    rate, const = result[0]
    
    residuals = log_e - (rate * log_h + const)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_e - np.mean(log_e))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return float(rate), float(r_squared)


# =============================================================================
# FINITE DIFFERENCE GRADIENT VERIFICATION
# =============================================================================

def finite_difference_gradient(problem, dL_dT: Tensor, param: str, 
                                indices: Tuple[int, ...], eps: float = 1e-5) -> float:
    """Compute gradient via central finite differences."""
    def compute_loss(G, B):
        prob = EikonalProblem(G, B, problem.source_mask)
        T, _ = fast_sweeping_solve(prob, verbose=False)
        return (T * dL_dT).sum().item()
    
    c, i, j = indices
    
    if param == 'G':
        G_plus, G_minus = problem.G.clone(), problem.G.clone()
        G_plus[0, c, i, j] += eps
        G_minus[0, c, i, j] -= eps
        L_plus = compute_loss(G_plus, problem.B)
        L_minus = compute_loss(G_minus, problem.B)
    else:  # 'B'
        B_plus, B_minus = problem.B.clone(), problem.B.clone()
        B_plus[0, c, i, j] += eps
        B_minus[0, c, i, j] -= eps
        L_plus = compute_loss(problem.G, B_plus)
        L_minus = compute_loss(problem.G, B_minus)
    
    return (L_plus - L_minus) / (2 * eps)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_arrival_time(T: Tensor, ax: plt.Axes = None, title: str = "Arrival Time",
                      contour_levels: int = 20, cmap: str = 'viridis') -> plt.Axes:
    """Plot arrival time field with contours."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    T_np = T.squeeze().cpu().numpy()
    T_np = np.where(np.isinf(T_np), np.nan, T_np)
    
    im = ax.imshow(T_np, origin='upper', cmap=cmap)
    plt.colorbar(im, ax=ax, label='T(x)')
    
    if np.nanmax(T_np) > np.nanmin(T_np):
        levels = np.linspace(np.nanmin(T_np), np.nanmax(T_np), contour_levels)
        ax.contour(T_np, levels=levels, colors='white', linewidths=0.5, alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    return ax


def plot_error_map(error: Tensor, ax: plt.Axes = None, title: str = "Error",
                   cmap: str = 'RdBu_r', symmetric: bool = True) -> plt.Axes:
    """Plot error field."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    err_np = error.squeeze().cpu().numpy()
    vmax = np.nanmax(np.abs(err_np)) if symmetric else np.nanmax(err_np)
    vmin = -vmax if symmetric else np.nanmin(err_np)
    
    im = ax.imshow(err_np, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    return ax


def plot_metric_ellipses(G: Tensor, ax: plt.Axes, step: int = 10, 
                         scale: float = 0.4, color: str = 'blue') -> None:
    """Overlay metric tensor ellipses."""
    G_np = G.squeeze().cpu().numpy()
    if G_np.ndim == 2:
        return
    M, N = G_np.shape[1], G_np.shape[2]
    
    for i in range(step//2, M, step):
        for j in range(step//2, N, step):
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
                                  fill=False, edgecolor=color, linewidth=0.5, alpha=0.6)
                ax.add_patch(ellipse)
            except:
                pass


def plot_drift_field(B: Tensor, ax: plt.Axes, step: int = 10, 
                     color: str = 'red', scale: float = None) -> None:
    """Overlay drift vectors."""
    B_np = B.squeeze().cpu().numpy()
    M, N = B_np.shape[1], B_np.shape[2]
    X, Y = np.meshgrid(np.arange(N), np.arange(M))
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              B_np[0, ::step, ::step], B_np[1, ::step, ::step],
              color=color, alpha=0.7, scale=scale)


def plot_stencil_map(stencil_idx: Tensor, ax: plt.Axes = None,
                     title: str = "Active Stencils") -> plt.Axes:
    """Plot active stencil at each node."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    s_np = stencil_idx.squeeze().cpu().numpy()
    cmap = plt.cm.get_cmap('tab10', 8)
    im = ax.imshow(s_np, origin='upper', cmap=cmap, vmin=-0.5, vmax=7.5)
    plt.colorbar(im, ax=ax, ticks=range(8), label='Stencil Index')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    return ax


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_interior_mask(M: int, N: int, boundary_width: int = 3, 
                      source_mask: Tensor = None, source_radius: int = 3,
                      device: str = DEVICE) -> Tensor:
    """Create mask for interior points."""
    mask = torch.zeros(1, M, N, dtype=torch.bool, device=device)
    mask[0, boundary_width:-boundary_width, boundary_width:-boundary_width] = True
    
    if source_mask is not None:
        source_locs = torch.where(source_mask[0])
        for si, sj in zip(source_locs[0], source_locs[1]):
            for di in range(-source_radius, source_radius + 1):
                for dj in range(-source_radius, source_radius + 1):
                    ni, nj = si + di, sj + dj
                    if 0 <= ni < M and 0 <= nj < N:
                        mask[0, ni, nj] = False
    return mask


def create_sparse_observation_mask(M: int, N: int, fraction: float,
                                    source_mask: Tensor = None,
                                    seed: int = None, device: str = DEVICE) -> Tensor:
    """Create sparse random observation mask."""
    if seed is not None:
        np.random.seed(seed)
    
    interior = get_interior_mask(M, N, 3, source_mask, device=device)
    valid_indices = torch.where(interior[0])
    n_valid = len(valid_indices[0])
    n_obs = int(n_valid * fraction)
    selected = np.random.choice(n_valid, size=n_obs, replace=False)
    
    obs_mask = torch.zeros(1, M, N, dtype=torch.bool, device=device)
    for idx in selected:
        i, j = valid_indices[0][idx].item(), valid_indices[1][idx].item()
        obs_mask[0, i, j] = True
    return obs_mask


# =============================================================================
# REGULARIZATION
# =============================================================================

def compute_tv_regularization(G: Tensor, B: Tensor = None, 
                               lambda_G: float = 0.01, lambda_B: float = 0.01) -> Tuple[Tensor, Dict]:
    """Compute Total Variation regularization."""
    device = G.device
    
    tv_G = torch.tensor(0.0, device=device)
    for c in range(3):
        Gc = G[0, c]
        tv_G += torch.abs(Gc[:, 1:] - Gc[:, :-1]).sum()
        tv_G += torch.abs(Gc[1:, :] - Gc[:-1, :]).sum()
    
    tv_B = torch.tensor(0.0, device=device)
    if B is not None:
        for c in range(2):
            Bc = B[0, c]
            tv_B += torch.abs(Bc[:, 1:] - Bc[:, :-1]).sum()
            tv_B += torch.abs(Bc[1:, :] - Bc[:-1, :]).sum()
    
    total = lambda_G * tv_G + lambda_B * tv_B
    return total, {'tv_G': tv_G.item(), 'tv_B': tv_B.item()}


# =============================================================================
# EXPERIMENT REGISTRY
# =============================================================================

EXPERIMENT_REGISTRY: Dict[str, type] = {}

def register_experiment(cls):
    """Decorator to register an experiment class."""
    key = f"{cls.category}_{cls.name}"
    EXPERIMENT_REGISTRY[key] = cls
    return cls


if __name__ == "__main__":
    print("SGP Experiments - Base Infrastructure")
    print(f"Device: {DEVICE}")
    print(f"Imports OK: {IMPORTS_OK}")
