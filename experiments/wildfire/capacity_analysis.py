#!/usr/bin/env python3
"""
Sim2Real Model Overfitting Diagnostic (Paper-Quality)
======================================================

Measures the MINIMUM MODELING ERROR by training on individual fires 
with ZERO regularization, using the EXACT same workflow as the main
experiment_suite_parallel.py.

This diagnostic answers: "What is the best our model can possibly do?"

Methodology:
- Uses the exact same SceneTrainer class, prepare_scenario, train_step
- Sets λ_G = λ_B = 0 (zero regularization)  
- Trains on single fire, evaluates on THE SAME fire
- Samples fires from ALL scenes for geographic coverage

Usage:
    # Standard: 3 fires × 19 scenes = 57 fires
    python sim2real_overfitting_diagnostic.py
    
    # Quick test: 1 fire per scene
    python sim2real_overfitting_diagnostic.py --fires_per_scene 1

Author: Barak Gahtan
Date: January 2025
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import csv

# Project imports - use exact same imports as experiment_suite_parallel.py

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba
from eikonal.backward.implicit_diff import backward_pass

try:
    from eikonal.backward.adjoint_numba import solve_adjoint_numba, HAVE_NUMBA
    from eikonal.backward.implicit_diff import identify_active_stencils, compute_parameter_gradients
except ImportError:
    HAVE_NUMBA = False

from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader,
    CovariateAdapter,
    extract_arrival_times,
    find_ignition_point
)

from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.regularization import compute_tv_regularization
from eikonal.nn.config import DEVICE, TVMetricType


# =============================================================================
# Configuration - Same as experiment_suite_parallel.py but with ZERO regularization
# =============================================================================

@dataclass
class DiagnosticConfig:
    """Configuration matching ExperimentConfig but with zero regularization."""
    # Model architecture (SAME as experiments)
    hidden_channels: int = 64
    num_cnn_layers: int = 5
    mlp_hidden: int = 128
    mlp_num_layers: int = 2
    eps: float = 0.1
    max_drift: float = 0.8
    max_eigenvalue: float = 10.0
    local_scale: float = 0.1
    initial_eigenvalue: float = 0.35  # SAME as experiments
    
    # Training
    n_epochs: int = 1000           # More epochs for convergence
    learning_rate: float = 1e-3
    patience: int = 100           # Longer patience for single-fire overfitting
    grad_clip: float = 1.0
    
    # ZERO REGULARIZATION - this is the key difference
    lambda_G: float = 0.0
    lambda_B: float = 0.0
    
    # Solver (SAME as experiments)
    max_iter: int = 50
    
    # Sampling
    fires_per_scene: int = 3
    min_pixels: int = 500  # Minimum burned pixels to include a fire
    
    # Hardware
    device: str = "cuda:0"
    
    # Output
    output_dir: str = "./overfitting_diagnostic"
    seed: int = 42


# =============================================================================
# Metrics - SAME as experiment_suite_parallel.py
# =============================================================================

def compute_all_metrics(pred_T: np.ndarray, true_T: np.ndarray,
                        valid_mask: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for a single fire. SAME as experiment_suite_parallel.py"""
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    
    if valid.sum() < 10:
        return {
            'relative_rmse': float('nan'),
            'rmse': float('nan'),
            'correlation': float('nan'),
            'mae': float('nan'),
            'n_valid': int(valid.sum()),
        }
    
    pred_valid = pred_T[valid]
    true_valid = true_T[valid]
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
    
    # Relative RMSE
    max_time = true_valid.max()
    relative_rmse = rmse / max_time if max_time > 1e-6 else float('nan')
    
    # Correlation
    if np.std(pred_valid) > 1e-8 and np.std(true_valid) > 1e-8:
        correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
    else:
        correlation = float('nan')
    
    # MAE
    mae = np.mean(np.abs(pred_valid - true_valid))
    
    # IoU at different time thresholds
    metrics = {
        'relative_rmse': float(relative_rmse),
        'rmse': float(rmse),
        'correlation': float(correlation),
        'mae': float(mae),
        'max_time_gt': float(max_time),
        'n_valid': int(valid.sum()),
    }
    
    # IoU at 25%, 50%, 75%, 100% of max time
    for frac in [0.25, 0.5, 0.75, 1.0]:
        t_threshold = max_time * frac
        pred_burned = pred_valid <= t_threshold
        true_burned = true_valid <= t_threshold
        
        intersection = (pred_burned & true_burned).sum()
        union = (pred_burned | true_burned).sum()
        
        iou = intersection / union if union > 0 else 0.0
        metrics[f'iou_{int(frac*100)}'] = float(iou)
    
    return metrics


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, 
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    values = [v for v in values if not np.isnan(v)]
    if len(values) < 3:
        return np.nan, np.nan, np.nan
    
    values = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    
    means = np.array(means)
    alpha = (1 - ci) / 2
    lower = np.percentile(means, 100 * alpha)
    upper = np.percentile(means, 100 * (1 - alpha))
    
    return np.mean(values), lower, upper


# =============================================================================
# Single Fire Trainer - Uses EXACT same logic as SceneTrainer
# =============================================================================

class SingleFireTrainer:
    """
    Trainer for a single fire scenario.
    
    Uses the EXACT same logic as SceneTrainer from experiment_suite_parallel.py:
    - Same prepare_scenario method
    - Same train_step method  
    - Same evaluate method
    
    Only difference: lambda_G = lambda_B = 0
    """
    
    def __init__(self, config: DiagnosticConfig, loader: Sim2RealFireLoader):
        self.config = config
        self.loader = loader
        self.device = torch.device(config.device)
        self.adapter = CovariateAdapter()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def setup_model(self, H: int, W: int, n_spatial_features: int = 6):
        """Initialize model - SAME as SceneTrainer.setup_model"""
        self.H, self.W = H, W
        
        self.model = FlexibleCovariateEncoder(
            n_spatial_features=n_spatial_features,
            n_global_features=5,
            hidden_channels=self.config.hidden_channels,
            num_cnn_layers=self.config.num_cnn_layers,
            mlp_hidden=self.config.mlp_hidden,
            mlp_num_layers=self.config.mlp_num_layers,
            eps=self.config.eps,
            max_drift=self.config.max_drift,
            max_eigenvalue=self.config.max_eigenvalue,
            local_scale=self.config.local_scale,
        ).to(self.device)
        
        # Initialize to reasonable values - SAME as SceneTrainer
        with torch.no_grad():
            ev = self.config.initial_eigenvalue
            self.model.global_mlp[-1].bias.copy_(
                torch.tensor([ev, 0.0, ev, 0.0, 0.0], device=self.device)
            )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )
    
    def prepare_scenario(self, scene_id: str, event_id: str) -> Optional[Dict]:
        """Load and prepare a single fire scenario - SAME as SceneTrainer.prepare_scenario"""
        try:
            data = self.loader.load_scenario(scene_id, event_id)
            
            # Get spatial and global features
            spatial_feat, global_feat = self.adapter(data)
            
            # Extract arrival times
            arrival = extract_arrival_times(data['masks'], data['timestamps'])
            
            # Find ignition point
            ignition = find_ignition_point(data['masks'])
            if ignition is None:
                return None
            
            H, W = arrival.shape
            si, sj = ignition
            
            # Create source mask - 3x3 region around ignition (SAME as SceneTrainer)
            source_mask = np.zeros((H, W), dtype=bool)
            source_mask[max(0, si-1):min(H, si+2), max(0, sj-1):min(W, sj+2)] = True
            
            # Valid mask
            valid_mask = np.isfinite(arrival)
            
            # Check minimum valid pixels (use config value)
            if valid_mask.sum() < self.config.min_pixels:
                return None
            
            return {
                'spatial_feat': spatial_feat,
                'global_feat': global_feat,
                'arrival_times': arrival,
                'source_mask': source_mask,
                'valid_mask': valid_mask,
                'ignition': (si, sj),
                'scene_id': scene_id,
                'event_id': event_id,
                'n_pixels': int(valid_mask.sum()),
                'grid_shape': (H, W),
            }
        except Exception as e:
            return None
    
    def train_step(self, scenario: Dict) -> Optional[Dict]:
        """Single training step - SAME as SceneTrainer.train_step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
        global_f = scenario['global_feat'].unsqueeze(0).to(self.device)
        
        # Forward through CNN
        G, B, G_local, B_local = self.model(spatial, global_f)
        
        # Check for NaN in model output
        if torch.isnan(G).any() or torch.isnan(B).any():
            return None
        
        # Move to CPU for solver - MUST DETACH!
        G_cpu = G.detach().cpu()
        B_cpu = B.detach().cpu()
        source = torch.from_numpy(scenario['source_mask']).unsqueeze(0)
        true_T = torch.from_numpy(scenario['arrival_times']).float()
        valid = torch.from_numpy(scenario['valid_mask'])
        
        # Forward solve
        problem = EikonalProblem(G_cpu, B_cpu, source)
        T_pred, n_iter = fast_sweeping_solve_numba(problem, max_iter=self.config.max_iter)
        
        if torch.isnan(T_pred).any():
            return None
        
        # Compute loss - use MEAN (SAME as SceneTrainer)
        obs_mask = valid & (T_pred[0] < 1e5)
        n_obs = obs_mask.sum().item()
        
        if n_obs < 10:
            return None
        
        residuals = T_pred[0][obs_mask] - true_T[obs_mask]
        loss_data = 0.5 * (residuals ** 2).mean()
        
        if loss_data.item() > 1e6 or torch.isnan(loss_data):
            return None
        
        # Backward pass - compute dL/dT (SAME as SceneTrainer)
        dL_dT = torch.zeros_like(T_pred)
        dL_dT[0][obs_mask] = residuals / n_obs
        
        if HAVE_NUMBA:
            stencil_info = identify_active_stencils(T_pred, problem)
            lam = solve_adjoint_numba(stencil_info, dL_dT, problem, T_pred)
            dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
        else:
            dL_dG, dL_dB = backward_pass(T_pred, dL_dT, problem)
        
        if torch.isnan(dL_dG).any() or torch.isnan(dL_dB).any():
            return None
        
        # Mask gradients to valid region (SAME as SceneTrainer)
        valid_tensor = torch.from_numpy(scenario['valid_mask']).float()[None, None, :, :]
        dL_dG = dL_dG * valid_tensor.expand(1, 3, -1, -1)
        dL_dB = dL_dB * valid_tensor.expand(1, 2, -1, -1)
        
        # Regularization (with lambda=0, this returns 0 but keeps same code structure)
        loss_reg = compute_tv_regularization(
            G, B,
            self.config.lambda_G,  # = 0
            self.config.lambda_B,  # = 0
            TVMetricType.FROBENIUS
        )
        
        if torch.isnan(loss_reg):
            return None
        
        # Backward: regularization via autograd (even if zero)
        if loss_reg.requires_grad:
            loss_reg.backward(retain_graph=True)
        
        # Backward: data loss via implicit differentiation
        dL_dG = dL_dG.to(self.device)
        dL_dB = dL_dB.to(self.device)
        
        G2, B2, _, _ = self.model(spatial, global_f)
        G2.backward(dL_dG, retain_graph=True)
        B2.backward(dL_dB)
        
        # Clip and update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        return {
            'loss': loss_data.item() + loss_reg.item(),
            'data_loss': loss_data.item(),
            'reg_loss': loss_reg.item(),
        }
    
    @torch.no_grad()
    def evaluate(self, scenario: Dict) -> Dict:
        """Evaluate model on a scenario - SAME logic as SceneTrainer.evaluate"""
        self.model.eval()
        
        spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
        global_f = scenario['global_feat'].unsqueeze(0).to(self.device)
        
        G, B, _, _ = self.model(spatial, global_f)
        
        G_cpu = G.cpu()
        B_cpu = B.cpu()
        source = torch.from_numpy(scenario['source_mask']).unsqueeze(0)
        
        problem = EikonalProblem(G_cpu, B_cpu, source)
        T_pred, _ = fast_sweeping_solve_numba(problem, max_iter=self.config.max_iter)
        
        pred_T = T_pred[0].numpy()
        true_T = scenario['arrival_times']
        valid_mask = scenario['valid_mask']
        
        return compute_all_metrics(pred_T, true_T, valid_mask)
    
    def train_on_fire(self, scenario: Dict, verbose: bool = False) -> Dict:
        """
        Train model on a single fire until convergence.
        
        For overfitting diagnostic: train and evaluate on the SAME fire.
        """
        best_rmse = float('inf')
        best_state = None
        patience_counter = 0
        
        history = {'loss': [], 'relative_rmse': [], 'correlation': []}
        
        for epoch in range(self.config.n_epochs):
            # Train step
            result = self.train_step(scenario)
            
            if result is None:
                continue
            
            # Evaluate on same fire (overfitting diagnostic)
            metrics = self.evaluate(scenario)
            
            history['loss'].append(result['data_loss'])
            history['relative_rmse'].append(metrics['relative_rmse'])
            history['correlation'].append(metrics['correlation'])
            
            # Update scheduler based on correlation (same as SceneTrainer)
            self.scheduler.step(metrics['correlation'])
            
            # Track best
            if metrics['relative_rmse'] < best_rmse and not np.isnan(metrics['relative_rmse']):
                best_rmse = metrics['relative_rmse']
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"      Epoch {epoch+1}: loss={result['data_loss']:.2f}, "
                      f"relRMSE={100*metrics['relative_rmse']:.2f}%, "
                      f"corr={metrics['correlation']:.4f}")
            
            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"      Converged at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Final evaluation
        final_metrics = self.evaluate(scenario)
        final_metrics['n_epochs'] = epoch + 1
        final_metrics['history'] = history
        
        return final_metrics


# =============================================================================
# Fire Selection
# =============================================================================

def select_fires_from_all_scenes(loader: Sim2RealFireLoader, 
                                  config: DiagnosticConfig) -> List[Tuple[str, str, Dict]]:
    """
    Select fires from ALL scenes, stratified by size.
    
    Returns list of (scene_id, event_id, scenario_dict) tuples.
    """
    all_fires = []
    
    print(f"\nSelecting {config.fires_per_scene} fires from each scene (min {config.min_pixels} pixels)...")
    
    for scene_id in sorted(loader.scenes.keys()):
        events = loader.scenes[scene_id]['events']
        print(f"  Scene {scene_id}: {len(events)} events...")
        
        # Get first event to determine grid size
        try:
            first_data = loader.load_scenario(scene_id, events[0])
            elev = first_data['topography'].get('elevation')
            H, W = elev.shape
        except:
            print(f"    WARNING: Could not load scene {scene_id}")
            continue
        
        # Create temporary trainer to prepare scenarios
        temp_trainer = SingleFireTrainer(config, loader)
        temp_trainer.setup_model(H, W)
        
        # Try to prepare valid fires (search through more events since many may be filtered)
        valid_fires = []
        for event_id in events:
            scenario = temp_trainer.prepare_scenario(scene_id, event_id)
            if scenario is not None and scenario['n_pixels'] >= config.min_pixels:
                valid_fires.append((scene_id, event_id, scenario))
            
            # Search through enough events to find candidates
            if len(valid_fires) >= config.fires_per_scene * 10:
                break
        
        if not valid_fires:
            print(f"    WARNING: No valid fires in scene {scene_id}")
            continue
        
        # Sort by size and select stratified sample
        valid_fires.sort(key=lambda x: x[2]['n_pixels'])
        
        if len(valid_fires) <= config.fires_per_scene:
            selected = valid_fires
        else:
            indices = np.linspace(0, len(valid_fires)-1, config.fires_per_scene).astype(int)
            selected = [valid_fires[i] for i in indices]
        
        all_fires.extend(selected)
        sizes = [f[2]['n_pixels'] for f in selected]
        print(f"    Selected {len(selected)} fires (sizes: {sizes})")
    
    return all_fires


# =============================================================================
# Main Diagnostic
# =============================================================================

def run_diagnostic(config: DiagnosticConfig) -> Dict:
    """Run the full overfitting diagnostic."""
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Warmup
    print("Warming up Numba JIT...")
    warmup_numba()
    
    # Load data
    data_root = '/home/shirb/sim2real_fire_data/simulation_data'
    print(f"\nLoading Sim2Real data from: {data_root}")
    loader = Sim2RealFireLoader(data_root)
    
    n_scenes = len(loader.scenes)
    total_events = sum(len(info['events']) for info in loader.scenes.values())
    print(f"Found {n_scenes} scenes with {total_events:,} total fire events")
    
    # Select fires
    all_fires = select_fires_from_all_scenes(loader, config)
    print(f"\nTotal fires to test: {len(all_fires)}")
    
    # Run diagnostic for each fire
    all_results = []
    scene_results = {}
    
    for i, (scene_id, event_id, scenario) in enumerate(all_fires):
        print(f"\n[{i+1}/{len(all_fires)}] {scene_id}/{event_id} "
              f"({scenario['n_pixels']:,} pixels, {scenario['grid_shape']})")
        
        try:
            # Create fresh trainer for each fire
            H, W = scenario['grid_shape']
            trainer = SingleFireTrainer(config, loader)
            trainer.setup_model(H, W)
            
            # Train on this single fire
            metrics = trainer.train_on_fire(scenario, verbose=True)
            
            result = {
                'scene_id': scene_id,
                'event_id': event_id,
                'n_pixels': scenario['n_pixels'],
                'grid_shape': list(scenario['grid_shape']),
                **{k: v for k, v in metrics.items() if k != 'history'}
            }
            
            all_results.append(result)
            
            if scene_id not in scene_results:
                scene_results[scene_id] = []
            scene_results[scene_id].append(result)
            
            print(f"    → RelRMSE: {100*metrics['relative_rmse']:.2f}%, "
                  f"Corr: {metrics['correlation']:.4f}, "
                  f"IoU@50: {metrics.get('iou_50', 0):.3f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==========================================================================
    # Compute Statistics
    # ==========================================================================
    
    rel_rmses = [r['relative_rmse'] for r in all_results if not np.isnan(r['relative_rmse'])]
    correlations = [r['correlation'] for r in all_results if not np.isnan(r['correlation'])]
    iou50s = [r.get('iou_50', float('nan')) for r in all_results if not np.isnan(r.get('iou_50', float('nan')))]
    
    # Bootstrap CI
    mean_rmse, ci_low, ci_high = bootstrap_ci(rel_rmses)
    
    overall_stats = {
        'n_fires': len(all_results),
        'n_valid': len(rel_rmses),
        'relative_rmse': {
            'mean': float(np.mean(rel_rmses)) if rel_rmses else float('nan'),
            'std': float(np.std(rel_rmses)) if rel_rmses else float('nan'),
            'median': float(np.median(rel_rmses)) if rel_rmses else float('nan'),
            'min': float(np.min(rel_rmses)) if rel_rmses else float('nan'),
            'max': float(np.max(rel_rmses)) if rel_rmses else float('nan'),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
        },
        'correlation': {
            'mean': float(np.mean(correlations)) if correlations else float('nan'),
            'std': float(np.std(correlations)) if correlations else float('nan'),
        },
        'iou_50': {
            'mean': float(np.mean(iou50s)) if iou50s else float('nan'),
            'std': float(np.std(iou50s)) if iou50s else float('nan'),
        }
    }
    
    # Per-scene stats
    per_scene_stats = {}
    for scene_id, results in scene_results.items():
        scene_rmses = [r['relative_rmse'] for r in results if not np.isnan(r['relative_rmse'])]
        scene_corrs = [r['correlation'] for r in results if not np.isnan(r['correlation'])]
        
        if scene_rmses:
            per_scene_stats[scene_id] = {
                'n_fires': len(results),
                'mean_relative_rmse': float(np.mean(scene_rmses)),
                'std_relative_rmse': float(np.std(scene_rmses)) if len(scene_rmses) > 1 else 0.0,
                'mean_correlation': float(np.mean(scene_corrs)) if scene_corrs else float('nan'),
            }
    
    # ==========================================================================
    # Print Summary
    # ==========================================================================
    
    print(f"\n{'='*70}")
    print("OVERFITTING DIAGNOSTIC RESULTS")
    print(f"{'='*70}")
    print(f"\nConfiguration: λ_G = {config.lambda_G}, λ_B = {config.lambda_B} (ZERO regularization)")
    print(f"Tested {len(all_results)} fires across {len(scene_results)} scenes")
    
    print(f"\n--- OVERALL STATISTICS ---")
    print(f"Relative RMSE:")
    print(f"  Mean ± Std:  {100*overall_stats['relative_rmse']['mean']:.2f}% ± {100*overall_stats['relative_rmse']['std']:.2f}%")
    print(f"  95% CI:      [{100*ci_low:.2f}%, {100*ci_high:.2f}%]")
    print(f"  Median:      {100*overall_stats['relative_rmse']['median']:.2f}%")
    print(f"  Range:       [{100*overall_stats['relative_rmse']['min']:.2f}%, {100*overall_stats['relative_rmse']['max']:.2f}%]")
    
    print(f"\nCorrelation:")
    print(f"  Mean ± Std:  {overall_stats['correlation']['mean']:.4f} ± {overall_stats['correlation']['std']:.4f}")
    
    print(f"\nIoU@50%:")
    print(f"  Mean ± Std:  {overall_stats['iou_50']['mean']:.3f} ± {overall_stats['iou_50']['std']:.3f}")
    
    print(f"\n--- PER-SCENE STATISTICS ---")
    print(f"{'Scene':<12} {'N':>3} {'RelRMSE Mean':>12} {'Corr Mean':>10}")
    print("-" * 45)
    for scene_id in sorted(per_scene_stats.keys()):
        s = per_scene_stats[scene_id]
        print(f"{scene_id:<12} {s['n_fires']:>3} {100*s['mean_relative_rmse']:>11.2f}% {s['mean_correlation']:>10.4f}")
    
    # Interpretation
    print(f"\n--- INTERPRETATION ---")
    mean_rmse_pct = 100 * overall_stats['relative_rmse']['mean']
    if mean_rmse_pct < 5:
        print(f"Excellent! Mean relative RMSE = {mean_rmse_pct:.1f}% < 5%")
        print("Model can nearly perfectly fit individual fires.")
    elif mean_rmse_pct < 10:
        print(f"Good! Mean relative RMSE = {mean_rmse_pct:.1f}% < 10%")
        print("Model expressiveness is sufficient for most fires.")
    elif mean_rmse_pct < 15:
        print(f"Moderate. Mean relative RMSE = {mean_rmse_pct:.1f}%")
        print("Some modeling error present.")
    else:
        print(f"High modeling error: {mean_rmse_pct:.1f}%")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    full_results = {
        'config': asdict(config),
        'overall_stats': overall_stats,
        'per_scene_stats': per_scene_stats,
        'per_fire_results': all_results,
        'metadata': {
            'n_scenes': n_scenes,
            'total_events_in_dataset': total_events,
            'timestamp': datetime.now().isoformat(),
        }
    }
    
    with open(output_path / 'overfitting_diagnostic_full.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # CSV for per-fire results
    csv_path = output_path / 'overfitting_results_per_fire.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['scene_id', 'event_id', 'n_pixels', 'relative_rmse', 
                      'correlation', 'rmse', 'mae', 'iou_50', 'iou_100']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    # CSV for per-scene results
    scene_csv_path = output_path / 'overfitting_results_per_scene.csv'
    with open(scene_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'scene_id', 'n_fires', 'mean_relative_rmse', 'std_relative_rmse', 'mean_correlation'
        ])
        writer.writeheader()
        for scene_id in sorted(per_scene_stats.keys()):
            writer.writerow({'scene_id': scene_id, **per_scene_stats[scene_id]})
    
    print(f"\n--- FILES SAVED ---")
    print(f"  {output_path / 'overfitting_diagnostic_full.json'}")
    print(f"  {csv_path}")
    print(f"  {scene_csv_path}")
    
    return full_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sim2Real Model Overfitting Diagnostic (Paper-Quality)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This diagnostic measures the MINIMUM achievable modeling error by training
on individual fires with ZERO regularization, using the exact same workflow
as experiment_suite_parallel.py.

Examples:
    # Standard: 3 fires per scene
    python sim2real_overfitting_diagnostic.py

    # Quick test: 1 fire per scene  
    python sim2real_overfitting_diagnostic.py --fires_per_scene 1

    # More fires for robustness
    python sim2real_overfitting_diagnostic.py --fires_per_scene 5
        """
    )
    
    parser.add_argument('--fires_per_scene', type=int, default=3,
                        help='Number of fires per scene (default: 3)')
    parser.add_argument('--min_pixels', type=int, default=500,
                        help='Minimum burned pixels to include fire (default: 500)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max epochs per fire (default: 500)')
    parser.add_argument('--output', type=str, default='./overfitting_diagnostic',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config = DiagnosticConfig(
        fires_per_scene=args.fires_per_scene,
        min_pixels=args.min_pixels,
        n_epochs=args.epochs,
        output_dir=args.output,
        device=f'cuda:{args.gpu}',
        seed=args.seed,
        lambda_G=0.0,
        lambda_B=0.0,
    )
    
    print("="*70)
    print("SIM2REAL OVERFITTING DIAGNOSTIC")
    print("="*70)
    print(f"Fires per scene:  {config.fires_per_scene}")
    print(f"Min pixels:       {config.min_pixels}")
    print(f"Max epochs:       {config.n_epochs}")
    print(f"Regularization:   λ_G={config.lambda_G}, λ_B={config.lambda_B} (ZERO)")
    print(f"Device:           {config.device}")
    print("="*70)
    
    run_diagnostic(config)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
