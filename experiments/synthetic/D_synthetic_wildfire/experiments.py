"""
Category D: Synthetic Wildfire Scenarios
========================================

Realistic test cases mimicking wildfire propagation:

D1: Terrain-Driven Propagation - Slope affects speed
D2: Wind-Driven Propagation - Drift from wind
D3: Fuel Heterogeneity - Varying fuel loads
D4: Combined Terrain + Wind - Full realistic scenario
D5: Fire Line Reconstruction - Recover arrival times from sparse obs
D6: Parameter Sensitivity - How errors in G/B affect predictions
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
    compute_errors, get_interior_mask, create_sparse_observation_mask,
    plot_arrival_time, plot_error_map, plot_metric_ellipses, plot_drift_field
)

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve


# =============================================================================
# D1: TERRAIN-DRIVEN PROPAGATION
# =============================================================================

@register_experiment
class D1_TerrainDriven(Experiment):
    """Fire spread influenced by terrain slope."""
    
    name = "D1_terrain_driven"
    category = "D_synthetic_wildfire"
    description = "Terrain slope affects fire spread speed"
    
    def __init__(self, N: int = 150, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # Create terrain: ridge in the middle
        i_coords = torch.arange(N, device=self.device, dtype=torch.float32)
        j_coords = torch.arange(N, device=self.device, dtype=torch.float32)
        I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # Terrain elevation: ridge along diagonal
        elevation = 50 * torch.exp(-((I - N/2)**2 + (J - N/2)**2) / (2 * (N/4)**2))
        
        # Compute slope magnitude (affects speed)
        # Fire spreads faster uphill, slower downhill
        grad_i = torch.zeros_like(elevation)
        grad_j = torch.zeros_like(elevation)
        grad_i[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / 2
        grad_j[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / 2
        
        slope_mag = torch.sqrt(grad_i**2 + grad_j**2)
        
        # Metric: faster on steeper terrain (simplified)
        # G = (1 + slope)^2 * I means slower travel time on steep terrain
        speed_factor = 1.0 / (1.0 + 0.5 * slope_mag)
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[0, 0, :, :] = 1.0 / speed_factor**2
        G[0, 1, :, :] = 0
        G[0, 2, :, :] = 1.0 / speed_factor**2
        
        B = torch.zeros(1, 2, N, N, device=self.device)
        
        # Ignition at corner
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//4, N//4] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, max_iter=100, verbose=False)
        
        self.T, self.G, self.elevation = T, G, elevation
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': T.max().item(), 'slope_max': slope_mag.max().item()},
            arrays={'T': T.cpu().numpy(), 'elevation': elevation.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im = axes[0].imshow(self.elevation.cpu().numpy(), origin='upper', cmap='terrain')
        plt.colorbar(im, ax=axes[0], label='Elevation')
        axes[0].set_title('Terrain')
        
        plot_arrival_time(self.T, ax=axes[1], title='Fire Arrival Time')
        
        im = axes[2].imshow(self.G[0, 0].cpu().numpy(), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[2], label='g₁₁')
        axes[2].set_title('Metric (from slope)')
        
        fig.suptitle('D1: Terrain-Driven Fire Spread', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D2: WIND-DRIVEN PROPAGATION
# =============================================================================

@register_experiment
class D2_WindDriven(Experiment):
    """Fire spread influenced by wind (drift field)."""
    
    name = "D2_wind_driven"
    category = "D_synthetic_wildfire"
    description = "Wind creates asymmetric fire spread"
    
    def __init__(self, N: int = 150, wind_speed: float = 0.25, wind_dir_deg: float = 45, device: str = None):
        super().__init__(device)
        self.N = N
        self.wind_speed = wind_speed
        self.wind_dir = np.deg2rad(wind_dir_deg)
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        
        # Wind drift
        B = torch.zeros(1, 2, N, N, device=self.device)
        B[0, 0, :, :] = self.wind_speed * np.cos(self.wind_dir)
        B[0, 1, :, :] = self.wind_speed * np.sin(self.wind_dir)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, verbose=False)
        
        # Also solve without wind for comparison
        problem_nowind = EikonalProblem(G, torch.zeros_like(B), source_mask)
        T_nowind, _ = fast_sweeping_solve(problem_nowind, verbose=False)
        
        self.T, self.T_nowind, self.B = T, T_nowind, B
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': T.max().item(), 'T_nowind_max': T_nowind.max().item()},
            arrays={'T': T.cpu().numpy(), 'T_nowind': T_nowind.cpu().numpy()},
            metadata={'wind_speed': self.wind_speed, 'wind_dir_deg': np.rad2deg(self.wind_dir)}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_arrival_time(self.T_nowind, ax=axes[0], title='No Wind')
        
        plot_arrival_time(self.T, ax=axes[1], title='With Wind')
        plot_drift_field(self.B, axes[1], step=15, scale=15, color='white')
        
        diff = self.T - self.T_nowind
        plot_error_map(diff, ax=axes[2], title='Difference (wind effect)')
        
        fig.suptitle('D2: Wind-Driven Fire Spread', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D3: FUEL HETEROGENEITY
# =============================================================================

@register_experiment
class D3_FuelHeterogeneity(Experiment):
    """Varying fuel loads create speed variations."""
    
    name = "D3_fuel_heterogeneity"
    category = "D_synthetic_wildfire"
    description = "Fuel load variations affect spread rate"
    
    def __init__(self, N: int = 150, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # Create patchy fuel distribution
        np.random.seed(52)
        
        # Base fuel load
        fuel = torch.ones(N, N, device=self.device)
        
        # Add random patches of high/low fuel
        for _ in range(20):
            ci, cj = np.random.randint(10, N-10, 2)
            radius = np.random.randint(10, 30)
            intensity = np.random.uniform(0.5, 2.0)
            
            i_coords = torch.arange(N, device=self.device)
            j_coords = torch.arange(N, device=self.device)
            I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')
            
            dist = torch.sqrt((I - ci).float()**2 + (J - cj).float()**2)
            mask = dist < radius
            fuel[mask] = intensity
        
        # Smooth fuel
        kernel_size = 5
        fuel_np = fuel.cpu().numpy()
        from scipy.ndimage import gaussian_filter
        fuel_smooth = torch.tensor(gaussian_filter(fuel_np, sigma=2), device=self.device)
        
        # Metric from fuel: higher fuel = faster spread
        # G = 1/fuel^2 means T ~ distance/fuel, so high fuel = fast spread
        G = torch.ones(1, 3, N, N, device=self.device)
        G[0, 0, :, :] = 1.0 / (fuel_smooth**2 + 0.1)
        G[0, 1, :, :] = 0
        G[0, 2, :, :] = G[0, 0, :, :]
        
        B = torch.zeros(1, 2, N, N, device=self.device)
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//4, N//4] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, max_iter=100, verbose=False)
        
        self.T, self.fuel = T, fuel_smooth
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': T.max().item(), 'fuel_range': (fuel_smooth.min().item(), fuel_smooth.max().item())},
            arrays={'T': T.cpu().numpy(), 'fuel': fuel_smooth.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im = axes[0].imshow(self.fuel.cpu().numpy(), origin='upper', cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0], label='Fuel Load')
        axes[0].set_title('Fuel Distribution')
        
        plot_arrival_time(self.T, ax=axes[1], title='Fire Arrival Time')
        
        fig.suptitle('D3: Fuel Heterogeneity', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D4: COMBINED TERRAIN + WIND
# =============================================================================

@register_experiment
class D4_CombinedScenario(Experiment):
    """Full realistic scenario with terrain and wind."""
    
    name = "D4_combined_scenario"
    category = "D_synthetic_wildfire"
    description = "Combined terrain, fuel, and wind effects"
    
    def __init__(self, N: int = 200, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        i_coords = torch.arange(N, device=self.device, dtype=torch.float32)
        j_coords = torch.arange(N, device=self.device, dtype=torch.float32)
        I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # Terrain: valley with ridge
        elevation = (30 * torch.sin(2 * np.pi * I / N) * torch.cos(np.pi * J / N) + 
                    20 * torch.exp(-((I - N/2)**2 + (J - N/3)**2) / (2 * (N/6)**2)))
        
        # Slope-based speed modification
        grad_i = torch.zeros_like(elevation)
        grad_j = torch.zeros_like(elevation)
        grad_i[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / 2
        grad_j[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / 2
        slope_mag = torch.sqrt(grad_i**2 + grad_j**2)
        
        # Fuel: patchy
        np.random.seed(53)
        fuel = torch.ones(N, N, device=self.device)
        for _ in range(15):
            ci, cj = np.random.randint(20, N-20, 2)
            radius = np.random.randint(15, 40)
            intensity = np.random.uniform(0.6, 1.5)
            dist = torch.sqrt((I - ci)**2 + (J - cj)**2)
            fuel[dist < radius] = intensity
        
        # Combined metric
        speed = fuel / (1.0 + 0.3 * slope_mag)
        
        G = torch.ones(1, 3, N, N, device=self.device)
        G[0, 0, :, :] = 1.0 / (speed**2 + 0.1)
        G[0, 1, :, :] = 0
        G[0, 2, :, :] = G[0, 0, :, :]
        
        # Wind
        B = torch.zeros(1, 2, N, N, device=self.device)
        B[0, 0, :, :] = 0.15
        B[0, 1, :, :] = 0.1
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//4, N//4] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T, _ = fast_sweeping_solve(problem, max_iter=100, verbose=False)
        
        self.T, self.G, self.B = T, G, B
        self.elevation, self.fuel = elevation, fuel
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': T.max().item()},
            arrays={'T': T.cpu().numpy(), 'elevation': elevation.cpu().numpy(), 'fuel': fuel.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        im = axes[0,0].imshow(self.elevation.cpu().numpy(), origin='upper', cmap='terrain')
        plt.colorbar(im, ax=axes[0,0], label='Elevation')
        axes[0,0].set_title('Terrain')
        
        im = axes[0,1].imshow(self.fuel.cpu().numpy(), origin='upper', cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0,1], label='Fuel')
        axes[0,1].set_title('Fuel')
        
        plot_arrival_time(self.T, ax=axes[1,0], title='Arrival Time')
        plot_drift_field(self.B, axes[1,0], step=20, scale=20, color='white')
        
        im = axes[1,1].imshow(self.G[0, 0].cpu().numpy(), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[1,1], label='g₁₁')
        axes[1,1].set_title('Effective Metric')
        
        fig.suptitle('D4: Combined Wildfire Scenario', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D5: FIRE LINE RECONSTRUCTION
# =============================================================================

@register_experiment
class D5_FireLineReconstruction(Experiment):
    """Reconstruct fire evolution from sparse observations."""
    
    name = "D5_fire_line_reconstruction"
    category = "D_synthetic_wildfire"
    description = "Reconstruct fire lines from sparse arrival time obs"
    
    def __init__(self, N: int = 100, device: str = None):
        super().__init__(device)
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # Simple heterogeneous scenario
        G = torch.ones(1, 3, N, N, device=self.device)
        G[:, 1, :, :] = 0
        G[0, 0, :, N//2:] = 2.0
        G[0, 2, :, N//2:] = 2.0
        
        B = torch.zeros(1, 2, N, N, device=self.device)
        B[0, 0, :, :] = 0.1
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//4] = True
        
        problem = EikonalProblem(G, B, source_mask)
        T_true, _ = fast_sweeping_solve(problem, verbose=False)
        
        # Sparse observations (10%)
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=54, device=self.device)
        T_obs = T_true.clone()
        T_obs[~obs_mask] = float('nan')
        
        # Interpolate to reconstruct full field (simple nearest neighbor for demo)
        # In practice, would use the inverse problem
        from scipy.interpolate import griddata
        
        obs_points = torch.where(obs_mask[0])
        obs_values = T_true[0, obs_points[0], obs_points[1]].cpu().numpy()
        obs_coords = np.stack([obs_points[0].cpu().numpy(), obs_points[1].cpu().numpy()], axis=1)
        
        grid_i, grid_j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        grid_points = np.stack([grid_i.ravel(), grid_j.ravel()], axis=1)
        
        T_interp = griddata(obs_coords, obs_values, grid_points, method='linear')
        T_interp = T_interp.reshape(N, N)
        T_interp = torch.tensor(T_interp, device=self.device)
        
        # Compute reconstruction error
        valid = torch.isfinite(T_interp)
        err = torch.abs(T_true[0][valid] - T_interp[valid]).mean().item()
        
        self.T_true, self.T_interp, self.obs_mask = T_true, T_interp, obs_mask
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=err < 5.0,
            metrics={'reconstruction_error': err, 'n_obs': int(obs_mask.sum().item())},
            arrays={'T_true': T_true.cpu().numpy(), 'T_interp': T_interp.cpu().numpy()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_arrival_time(self.T_true, ax=axes[0], title='True Arrival Time')
        oy, ox = torch.where(self.obs_mask[0])
        axes[0].scatter(ox.cpu(), oy.cpu(), c='red', s=5, alpha=0.5)
        
        T_interp_t = self.T_interp.unsqueeze(0)
        plot_arrival_time(T_interp_t, ax=axes[1], title='Interpolated')
        
        err = torch.abs(self.T_true[0] - self.T_interp)
        plot_error_map(err, ax=axes[2], title='Reconstruction Error', symmetric=False)
        
        fig.suptitle('D5: Fire Line Reconstruction', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D6: PARAMETER SENSITIVITY
# =============================================================================

@register_experiment
class D6_ParameterSensitivity(Experiment):
    """How do errors in G/B affect arrival time predictions?"""
    
    name = "D6_parameter_sensitivity"
    category = "D_synthetic_wildfire"
    description = "Sensitivity of predictions to parameter errors"
    
    def __init__(self, N: int = 100, device: str = None):
        super().__init__(device)
        self.N = N
        self.perturbation_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        # True parameters
        G_true = torch.ones(1, 3, N, N, device=self.device)
        G_true[:, 1, :, :] = 0
        G_true[0, 0, :, N//2:] = 2.0
        G_true[0, 2, :, N//2:] = 2.0
        
        B_true = torch.zeros(1, 2, N, N, device=self.device)
        B_true[0, 0, :, :] = 0.15
        
        source_mask = torch.zeros(1, N, N, dtype=torch.bool, device=self.device)
        source_mask[0, N//2, N//2] = True
        
        problem_true = EikonalProblem(G_true, B_true, source_mask)
        T_true, _ = fast_sweeping_solve(problem_true, verbose=False)
        
        results = []
        np.random.seed(55)
        
        for pert in self.perturbation_levels:
            # Perturb G
            G_pert = G_true * (1 + pert * torch.randn_like(G_true))
            G_pert[:, 0, :, :] = torch.clamp(G_pert[:, 0, :, :], min=0.1)
            G_pert[:, 2, :, :] = torch.clamp(G_pert[:, 2, :, :], min=0.1)
            
            problem_pert = EikonalProblem(G_pert, B_true, source_mask)
            T_pert, _ = fast_sweeping_solve(problem_pert, verbose=False)
            
            interior = get_interior_mask(N, N, 5, source_mask, device=self.device)
            rel_err = torch.sqrt(((T_true[interior] - T_pert[interior])**2).mean()).item() / T_true[interior].abs().max().item()
            
            results.append({'perturbation': pert, 'T_error': rel_err})
            print(f"  Pert={pert*100:.0f}%: T error={rel_err:.4f}")
        
        self.results = results
        self.T_true = T_true
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'error_at_10pct': results[2]['T_error'], 'error_at_30pct': results[4]['T_error']},
            arrays={'perturbations': np.array([r['perturbation'] for r in results]),
                   'errors': np.array([r['T_error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pert = np.array([r['perturbation'] for r in self.results]) * 100
        err = np.array([r['T_error'] for r in self.results]) * 100
        
        ax.plot(pert, err, 'o-', markersize=10, lw=2)
        ax.set_xlabel('Parameter Perturbation (%)')
        ax.set_ylabel('Arrival Time Error (%)')
        ax.set_title('D6: Parameter Sensitivity Analysis')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    D1_TerrainDriven,
    D2_WindDriven,
    D3_FuelHeterogeneity,
    D4_CombinedScenario,
    D5_FireLineReconstruction,
    D6_ParameterSensitivity,
]

def run_all(save=True, visualize=True):
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY D SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results


if __name__ == "__main__":
    run_all()
