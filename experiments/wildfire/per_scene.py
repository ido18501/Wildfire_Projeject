#!/usr/bin/env python3
"""
Sim2Real-Fire Experiment Suite (Parallel GPU Version)
=====================================================

Comprehensive training and evaluation across all 19 scenes.
Now with parallel execution across multiple GPUs!

Experiments:
1. Per-scene training: Train/val/test within each scene independently
2. Multi-scene training: Train on N scenes, test on held-out scenes

Usage:
    # Per-scene experiments (all 19 scenes) - PARALLEL across 4 GPUs
    python experiment_suite_parallel.py --mode per_scene --gpus 0,1,2,3 --epochs 100
    
    # Multi-scene experiments
    python experiment_suite_parallel.py --mode multi_scene --n_train_scenes 5 --gpus 0,1,2,3
    
    # Single scene test (sequential)
    python experiment_suite_parallel.py --mode per_scene --scenes 0014_00426 --gpus 0

Author: Barak Gahtan
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import logging

# Project imports

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba
from eikonal.backward.implicit_diff import backward_pass

try:
    from eikonal.backward.adjoint_numba import solve_adjoint_numba, HAVE_NUMBA
    from eikonal.backward.implicit_diff import identify_active_stencils, compute_parameter_gradients
except ImportError:
    HAVE_NUMBA = False

# Import batched CUDA forward solver for fast batch processing
try:
    from eikonal.forward.cuda_batched import fast_sweeping_solve_batched as fast_sweeping_solve_batched_cuda
    HAVE_BATCHED_CUDA = True
except ImportError:
    HAVE_BATCHED_CUDA = False
    fast_sweeping_solve_batched_cuda = None

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
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Experiment info
    experiment_name: str = ""
    experiment_type: str = "per_scene"  # or "multi_scene"
    scenes: List[str] = None
    test_scenes: List[str] = None  # For multi-scene: held-out scenes
    
    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model architecture
    hidden_channels: int = 64
    num_cnn_layers: int = 5
    mlp_hidden: int = 128
    mlp_num_layers: int = 2
    eps: float = 0.1
    max_drift: float = 0.8
    max_eigenvalue: float = 10.0
    local_scale: float = 0.1
    initial_eigenvalue: float = 0.35
    
    # Training
    n_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    patience: int = 20
    grad_clip: float = 1.0
    
    # Regularization
    lambda_G: float = 0.005
    lambda_B: float = 0.005
    
    # Solver
    max_iter: int = 50
    
    # Hardware
    device: str = "cuda:0"
    n_workers: int = 8
    
    # Output
    output_dir: str = "./experiments"
    seed: int = 42


# =============================================================================
# Metrics
# =============================================================================

def compute_relative_rmse(pred_T: np.ndarray, true_T: np.ndarray, 
                          valid_mask: np.ndarray) -> float:
    """
    Compute relative RMSE: RMSE / max_time_in_ground_truth
    
    This normalizes the error by the fire duration, making it comparable
    across fires of different sizes/durations.
    """
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    
    if valid.sum() < 10:
        return float('nan')
    
    pred_valid = pred_T[valid]
    true_valid = true_T[valid]
    
    rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
    max_time = true_valid.max()
    
    if max_time < 1e-6:
        return float('nan')
    
    return rmse / max_time


def compute_all_metrics(pred_T: np.ndarray, true_T: np.ndarray,
                        valid_mask: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for a single fire."""
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


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """Aggregate metrics across multiple fires."""
    if not metrics_list:
        return {}
    
    # Filter out invalid metrics - check correlation is a valid number
    valid_metrics = []
    for m in metrics_list:
        corr = m.get('correlation')
        if isinstance(corr, (int, float)) and not np.isnan(corr):
            valid_metrics.append(m)
    
    if not valid_metrics:
        return {'n_samples': 0}
    
    result = {'n_samples': len(valid_metrics)}
    
    # Compute mean and std for each numeric metric
    for key in valid_metrics[0].keys():
        # Skip non-numeric fields
        if key in ['n_valid', 'scene_id', 'event_id']:
            continue
        
        # Collect numeric values only
        values = []
        for m in valid_metrics:
            val = m.get(key)
            if isinstance(val, (int, float)) and not np.isnan(val):
                values.append(val)
        
        if values:
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_std'] = float(np.std(values))
    
    return result


# =============================================================================
# Scene Info
# =============================================================================

def get_scene_info(loader) -> Dict[str, Dict]:
    """Get information about all scenes."""
    scene_info = {}
    
    for scene_id, info in loader.scenes.items():
        # Load one fire to get grid size
        events = info['events']
        if not events:
            continue
        
        try:
            data = loader.load_scenario(scene_id, events[0])
            elev = data['topography'].get('elevation')
            H, W = elev.shape
            
            scene_info[scene_id] = {
                'n_fires': len(events),
                'grid_size': (H, W),
                'path': info['path'],
            }
        except Exception as e:
            print(f"Warning: Could not load scene {scene_id}: {e}")
    
    return scene_info


# =============================================================================
# Single Scene Trainer
# =============================================================================

class SceneTrainer:
    """Trainer for a single scene."""
    
    def __init__(self, config: ExperimentConfig, loader):
        self.config = config
        self.loader = loader
        self.device = torch.device(config.device)
        self.adapter = CovariateAdapter()
        
        # Will be set when we know grid size
        self.model = None
        self.optimizer = None
        self.H = None
        self.W = None
    
    def setup_model(self, H: int, W: int, n_spatial_features: int = 6):
        """Initialize model for given grid size."""
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
        
        # Initialize to reasonable values
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
        """Load and prepare a single fire scenario."""
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
            
            # Create source mask
            source_mask = np.zeros((H, W), dtype=bool)
            source_mask[max(0, si-1):min(H, si+2), max(0, sj-1):min(W, sj+2)] = True
            
            # Valid mask
            valid_mask = np.isfinite(arrival)
            
            return {
                'spatial_feat': spatial_feat,
                'global_feat': global_feat,
                'arrival_times': arrival,
                'source_mask': source_mask,
                'valid_mask': valid_mask,
                'ignition': (si, sj),
                'scene_id': scene_id,
                'event_id': event_id,
            }
        except Exception as e:
            return None
    
    def train_step(self, scenario: Dict) -> Optional[Dict]:
        """Single training step - matches original trainer.py logic."""
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
        
        # Compute loss - use SUM like original trainer.py
        obs_mask = valid & (T_pred[0] < 1e5)
        n_obs = obs_mask.sum().item()
        
        if n_obs < 10:
            return None
        
        residuals = T_pred[0][obs_mask] - true_T[obs_mask]
        loss_data = 0.5 * (residuals ** 2).mean()  # MEAN - consistent gradient across fires of different sizes
        
        if loss_data.item() > 1e6 or torch.isnan(loss_data):
            return None
        
        # Backward pass - compute dL/dT
        dL_dT = torch.zeros_like(T_pred)
        dL_dT[0][obs_mask] = residuals / n_obs  # Divide by n_obs when using mean()
        
        if HAVE_NUMBA:
            stencil_info = identify_active_stencils(T_pred, problem)
            lam = solve_adjoint_numba(stencil_info, dL_dT, problem, T_pred)
            dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
        else:
            dL_dG, dL_dB = backward_pass(T_pred, dL_dT, problem)
        
        if torch.isnan(dL_dG).any() or torch.isnan(dL_dB).any():
            return None
        
        # Mask gradients to valid region (like original trainer.py)
        valid_tensor = torch.from_numpy(scenario['valid_mask']).float()[None, None, :, :]
        dL_dG = dL_dG * valid_tensor.expand(1, 3, -1, -1)
        dL_dB = dL_dB * valid_tensor.expand(1, 2, -1, -1)
        
        # Regularization on local components
        loss_reg = compute_tv_regularization(
            G, B,
            self.config.lambda_G,
            self.config.lambda_B,
            TVMetricType.FROBENIUS
        )
        
        if torch.isnan(loss_reg):
            return None
        
        # Backward: regularization via autograd
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
    
    def evaluate(self, scenarios: List[Dict], save_predictions: bool = False) -> Tuple[Dict, List[Dict], Optional[Dict]]:
        """
        Evaluate model on a set of scenarios.
        
        Args:
            scenarios: List of scenario dicts
            save_predictions: If True, return predictions for later analysis
            
        Returns:
            aggregate_metrics: Aggregated metrics dict
            individual_metrics: List of per-fire metrics
            predictions: Dict with pred_T, true_T, valid_mask arrays (if save_predictions)
        """
        self.model.eval()
        metrics_list = []
        
        # For saving predictions
        all_pred_T = []
        all_true_T = []
        all_valid_mask = []
        all_event_ids = []
        
        with torch.no_grad():
            for scenario in scenarios:
                spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
                global_f = scenario['global_feat'].unsqueeze(0).to(self.device)
                
                G, B, _, _ = self.model(spatial, global_f)
                
                G_cpu = G.cpu()  # No detach needed - already in no_grad context
                B_cpu = B.cpu()
                source = torch.from_numpy(scenario['source_mask']).unsqueeze(0)
                
                problem = EikonalProblem(G_cpu, B_cpu, source)
                T_pred, _ = fast_sweeping_solve_numba(problem, max_iter=self.config.max_iter)
                
                pred_T = T_pred[0].numpy()
                true_T = scenario['arrival_times']
                valid_mask = scenario['valid_mask']
                
                metrics = compute_all_metrics(pred_T, true_T, valid_mask)
                metrics['scene_id'] = scenario['scene_id']
                metrics['event_id'] = scenario['event_id']
                metrics_list.append(metrics)
                
                if save_predictions:
                    all_pred_T.append(pred_T)
                    all_true_T.append(true_T)
                    all_valid_mask.append(valid_mask)
                    all_event_ids.append(scenario['event_id'])
        
        predictions = None
        if save_predictions:
            predictions = {
                'pred_T': np.array(all_pred_T),
                'true_T': np.array(all_true_T),
                'valid_mask': np.array(all_valid_mask),
                'event_ids': np.array(all_event_ids),
            }
        
        return aggregate_metrics(metrics_list), metrics_list, predictions
    
    def train(self, train_scenarios: List[Dict], val_scenarios: List[Dict],
              verbose: bool = True, scene_prefix: str = "") -> Dict:
        """Full training loop with NaN recovery (matches original trainer.py)."""
        history = {
            'train_loss': [], 'train_data_loss': [], 'train_reg_loss': [],
            'val_correlation': [], 'val_relative_rmse': [],
            'best_epoch': 0, 'best_val_corr': 0, 'learning_rate': [],
        }
        
        best_val_corr = -1
        best_state = None
        patience_counter = 0
        
        # NaN recovery settings (from original trainer.py)
        nan_streak = 0
        max_nan_streak = 5
        recovery_count = 0
        max_recoveries = 10
        
        for epoch in range(self.config.n_epochs):
            # Training
            epoch_losses = []
            epoch_data_losses = []
            epoch_reg_losses = []
            np.random.shuffle(train_scenarios)
            
            for scenario in train_scenarios:
                result = self.train_step(scenario)
                if result is not None:
                    epoch_losses.append(result['loss'])
                    epoch_data_losses.append(result['data_loss'])
                    epoch_reg_losses.append(result['reg_loss'])
            
            # Check for epoch failure
            if not epoch_losses:
                nan_streak += 1
                if nan_streak >= max_nan_streak and best_state is not None:
                    recovery_count += 1
                    if recovery_count > max_recoveries:
                        if verbose:
                            print(f"{scene_prefix}  Too many recoveries ({recovery_count}), stopping early")
                        break
                    
                    if verbose:
                        print(f"{scene_prefix}  Recovery #{recovery_count}: Restoring best model")
                    self.model.load_state_dict(best_state)
                    
                    # Reset optimizer with reduced learning rate
                    new_lr = self.config.learning_rate * (0.5 ** recovery_count)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=new_lr)
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
                    )
                    nan_streak = 0
                continue
            
            nan_streak = 0
            
            avg_loss = np.mean(epoch_losses)
            avg_data_loss = np.mean(epoch_data_losses)
            avg_reg_loss = np.mean(epoch_reg_losses)
            
            history['train_loss'].append(avg_loss)
            history['train_data_loss'].append(avg_data_loss)
            history['train_reg_loss'].append(avg_reg_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_agg, _, _ = self.evaluate(val_scenarios, save_predictions=False)
                val_corr = val_agg.get('correlation_mean', 0)
                val_rmse = val_agg.get('relative_rmse_mean', float('inf'))
                
                history['val_correlation'].append(val_corr)
                history['val_relative_rmse'].append(val_rmse)
                
                self.scheduler.step(val_corr)
                
                if val_corr > best_val_corr:
                    best_val_corr = val_corr
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    history['best_epoch'] = epoch + 1
                    history['best_val_corr'] = best_val_corr
                    patience_counter = 0
                else:
                    patience_counter += 5
                
                if verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"{scene_prefix}  Epoch {epoch+1:3d}: loss={avg_loss:.1f} "
                          f"(data={avg_data_loss:.1f}, reg={avg_reg_loss:.3f}), "
                          f"val_corr={val_corr:.3f}, val_rel_rmse={val_rmse:.3f}, lr={lr:.1e}")
                
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"{scene_prefix}  Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        if recovery_count > 0 and verbose:
            print(f"{scene_prefix}  Required {recovery_count} recovery attempts")
        
        return history


# =============================================================================
# Per-Scene Experiment (Single Scene)
# =============================================================================

def run_single_scene_experiment(
    scene_id: str,
    config: ExperimentConfig,
    data_root: str,
    output_dir: Path,
    scene_prefix: str = "",
) -> Dict:
    """Run experiment on a single scene."""
    
    print(f"\n{scene_prefix}{'='*60}")
    print(f"{scene_prefix}Scene: {scene_id}")
    print(f"{scene_prefix}{'='*60}")
    
    start_time = time.time()
    
    # Setup
    loader = Sim2RealFireLoader(data_root)
    warmup_numba()
    
    # Get scene info
    scene_info = loader.scenes.get(scene_id)
    if scene_info is None:
        return {'error': f'Scene {scene_id} not found'}
    
    events = scene_info['events']
    print(f"{scene_prefix}Total fires in scene: {len(events)}")
    
    # Load first fire to get grid size
    first_data = loader.load_scenario(scene_id, events[0])
    elev = first_data['topography'].get('elevation')
    H, W = elev.shape
    print(f"{scene_prefix}Grid size: {H} x {W}")
    
    # Split data
    np.random.seed(config.seed)
    np.random.shuffle(events)
    
    n_train = int(len(events) * config.train_ratio)
    n_val = int(len(events) * config.val_ratio)
    
    train_events = events[:n_train]
    val_events = events[n_train:n_train + n_val]
    test_events = events[n_train + n_val:]
    
    print(f"{scene_prefix}Split: {len(train_events)} train, {len(val_events)} val, {len(test_events)} test")
    
    # Prepare scenarios
    print(f"{scene_prefix}Loading scenarios...")
    trainer = SceneTrainer(config, loader)
    trainer.setup_model(H, W)
    
    train_scenarios = []
    for event_id in train_events:
        scenario = trainer.prepare_scenario(scene_id, event_id)
        if scenario is not None:
            train_scenarios.append(scenario)
    
    val_scenarios = []
    for event_id in val_events:
        scenario = trainer.prepare_scenario(scene_id, event_id)
        if scenario is not None:
            val_scenarios.append(scenario)
    
    test_scenarios = []
    for event_id in test_events:
        scenario = trainer.prepare_scenario(scene_id, event_id)
        if scenario is not None:
            test_scenarios.append(scenario)
    
    print(f"{scene_prefix}Loaded: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test")
    
    # Train
    print(f"{scene_prefix}\nTraining...")
    history = trainer.train(train_scenarios, val_scenarios, verbose=True, scene_prefix=scene_prefix)
    
    # Test evaluation with saved predictions
    print(f"{scene_prefix}\nEvaluating on test set...")
    test_agg, test_details, test_predictions = trainer.evaluate(test_scenarios, save_predictions=True)
    
    elapsed = time.time() - start_time
    
    # Compute additional statistics
    all_correlations = [m['correlation'] for m in test_details if not np.isnan(m.get('correlation', float('nan')))]
    all_rel_rmses = [m['relative_rmse'] for m in test_details if not np.isnan(m.get('relative_rmse', float('nan')))]
    
    # Results with comprehensive metadata
    results = {
        'scene_id': scene_id,
        'grid_size': [H, W],
        'n_fires': len(events),
        'n_train': len(train_scenarios),
        'n_val': len(val_scenarios),
        'n_test': len(test_scenarios),
        'train_events': train_events,
        'val_events': val_events,
        'test_events': test_events,
        'history': history,
        'test_metrics': test_agg,
        'test_details': test_details,
        'elapsed_seconds': elapsed,
        'config': asdict(config),
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'device': config.device,
            'seed': config.seed,
        }
    }
    
    # Save
    scene_output_dir = output_dir / f'scene_{scene_id}'
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': asdict(config),
        'grid_size': (H, W),
        'best_epoch': history.get('best_epoch', 0),
        'best_val_corr': history.get('best_val_corr', 0),
    }, scene_output_dir / 'model.pt')
    
    # Save predictions for later analysis
    if test_predictions is not None:
        np.savez_compressed(
            scene_output_dir / 'test_predictions.npz',
            pred_T=test_predictions['pred_T'],
            true_T=test_predictions['true_T'],
            valid_mask=test_predictions['valid_mask'],
            event_ids=test_predictions['event_ids'],
            metadata={
                'scene_id': scene_id,
                'grid_size': [H, W],
                'n_test': len(test_scenarios),
                'timestamp': datetime.now().isoformat(),
            }
        )
    
    # Save results JSON
    with open(scene_output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save splits
    with open(scene_output_dir / 'train_fires.txt', 'w') as f:
        f.write('\n'.join(train_events))
    with open(scene_output_dir / 'val_fires.txt', 'w') as f:
        f.write('\n'.join(val_events))
    with open(scene_output_dir / 'test_fires.txt', 'w') as f:
        f.write('\n'.join(test_events))
    
    # Print summary
    print(f"\n{scene_prefix}{'='*60}")
    print(f"{scene_prefix}Scene {scene_id} Results:")
    print(f"{scene_prefix}  Test correlation: {test_agg.get('correlation_mean', 0):.3f} ± {test_agg.get('correlation_std', 0):.3f}")
    print(f"{scene_prefix}  Test relative RMSE: {test_agg.get('relative_rmse_mean', 0):.3f} ± {test_agg.get('relative_rmse_std', 0):.3f}")
    print(f"{scene_prefix}  Test IoU@50%: {test_agg.get('iou_50_mean', 0):.3f}")
    print(f"{scene_prefix}  Time: {elapsed:.1f}s")
    print(f"{scene_prefix}  Saved to: {scene_output_dir}")
    
    return results


# =============================================================================
# Parallel Scene Worker (for multiprocessing)
# =============================================================================

def _run_scene_worker(args):
    """
    Worker function for parallel scene processing.
    
    Must be a top-level function for multiprocessing pickling.
    Each worker runs on a dedicated GPU.
    """
    scene_id, config_dict, data_root, output_dir, gpu_id, scene_idx, total_scenes, threads_per_worker = args
    
    # Limit CPU threads for this worker to avoid oversubscription
    import os
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    os.environ['NUMBA_NUM_THREADS'] = str(threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
    
    # Also set torch threads
    torch.set_num_threads(threads_per_worker)
    
    # Reconstruct config from dict
    config = ExperimentConfig(**config_dict)
    config.device = f'cuda:{gpu_id}'
    config.experiment_name = f'per_scene_{scene_id}'
    
    # Set CUDA device for this process
    torch.cuda.set_device(gpu_id)
    
    # Create scene prefix for log messages
    scene_prefix = f"[GPU{gpu_id}|{scene_id}] "
    
    print(f"\n{scene_prefix}Starting scene {scene_idx+1}/{total_scenes}")
    
    try:
        results = run_single_scene_experiment(
            scene_id, config, data_root, Path(output_dir) / 'per_scene',
            scene_prefix=scene_prefix
        )
        return scene_id, results, None
    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"{scene_prefix}ERROR: {e}")
        return scene_id, {'error': str(e)}, error_msg


def run_per_scene_experiments_parallel(
    data_root: str,
    output_dir: Path,
    config: ExperimentConfig,
    scenes: List[str] = None,
    gpu_ids: List[int] = [0, 1, 2, 3],
    total_cores: int = 47,  # Leave 1 core idle from 48
):
    """
    Run experiments on all scenes in parallel across GPUs.
    
    Each GPU processes one scene at a time. When a GPU finishes,
    it picks up the next unprocessed scene from the queue.
    
    Args:
        data_root: Path to Sim2Real-Fire data
        output_dir: Output directory for results
        config: Experiment configuration
        scenes: List of scene IDs (None = all scenes)
        gpu_ids: List of GPU IDs to use
        total_cores: Total CPU cores to use (default 47, leaving 1 idle)
    """
    loader = Sim2RealFireLoader(data_root)
    all_scenes = list(loader.scenes.keys())
    
    if scenes is not None:
        all_scenes = [s for s in all_scenes if s in scenes]
    
    n_gpus = len(gpu_ids)
    threads_per_worker = total_cores // n_gpus  # Divide cores among workers
    
    print(f"Running per-scene experiments on {len(all_scenes)} scenes")
    print(f"Using {n_gpus} GPUs in PARALLEL: {gpu_ids}")
    print(f"CPU cores: {total_cores} total, {threads_per_worker} per worker")
    print(f"Scenes: {all_scenes}")
    
    # If only 1 scene or 1 GPU, run sequentially
    if len(all_scenes) == 1 or n_gpus == 1:
        print("Running sequentially (single scene or single GPU)")
        return run_per_scene_experiments_sequential(
            data_root, output_dir, config, scenes, gpu_ids[0]
        )
    
    # Prepare arguments for each scene
    config_dict = asdict(config)
    work_items = []
    for i, scene_id in enumerate(all_scenes):
        gpu_id = gpu_ids[i % n_gpus]
        work_items.append((
            scene_id, config_dict, data_root, str(output_dir), 
            gpu_id, i, len(all_scenes), threads_per_worker
        ))
    
    all_results = {}
    start_time = time.time()
    
    # Run in parallel with ProcessPoolExecutor
    # max_workers = n_gpus ensures we don't oversubscribe GPUs
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        # Submit all jobs
        futures = {executor.submit(_run_scene_worker, item): item[0] 
                   for item in work_items}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            scene_id = futures[future]
            completed += 1
            try:
                scene_id_result, results, error = future.result()
                all_results[scene_id_result] = results
                
                if error is None:
                    corr = results['test_metrics'].get('correlation_mean', 0)
                    rmse = results['test_metrics'].get('relative_rmse_mean', 0)
                    elapsed = results.get('elapsed_seconds', 0)
                    print(f"\n✓ [{completed}/{len(all_scenes)}] Completed {scene_id_result}: "
                          f"corr={corr:.3f}, rel_rmse={rmse:.3f}, time={elapsed:.0f}s")
                else:
                    print(f"\n✗ [{completed}/{len(all_scenes)}] Failed {scene_id_result}")
                    
            except Exception as e:
                print(f"\n✗ [{completed}/{len(all_scenes)}] Exception for {scene_id}: {e}")
                all_results[scene_id] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    # Save summary
    summary = {
        'experiment_type': 'per_scene_parallel',
        'n_scenes': len(all_scenes),
        'n_gpus': n_gpus,
        'gpu_ids': gpu_ids,
        'total_time_seconds': total_time,
        'scenes': all_scenes,
        'timestamp': datetime.now().isoformat(),
        'results_summary': {},
    }
    
    for scene_id, results in all_results.items():
        if 'error' not in results:
            summary['results_summary'][scene_id] = {
                'correlation': results['test_metrics'].get('correlation_mean', 0),
                'relative_rmse': results['test_metrics'].get('relative_rmse_mean', 0),
                'iou_50': results['test_metrics'].get('iou_50_mean', 0),
                'n_test': results['n_test'],
                'elapsed_seconds': results.get('elapsed_seconds', 0),
            }
    
    with open(output_dir / 'per_scene_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print overall summary
    print("\n" + "="*70)
    print("PER-SCENE EXPERIMENT SUMMARY (PARALLEL)")
    print("="*70)
    
    corrs = []
    rmses = []
    for scene_id, results in sorted(all_results.items()):
        if 'error' not in results:
            corr = results['test_metrics'].get('correlation_mean', 0)
            rmse = results['test_metrics'].get('relative_rmse_mean', 0)
            corrs.append(corr)
            rmses.append(rmse)
            print(f"  {scene_id}: corr={corr:.3f}, rel_rmse={rmse:.3f}")
        else:
            print(f"  {scene_id}: FAILED - {results.get('error', 'unknown')[:50]}")
    
    print("-"*70)
    if corrs:
        print(f"  Average correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
        print(f"  Average relative RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
        print(f"  Successful: {len(corrs)}/{len(all_scenes)} scenes")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Average time per scene: {total_time/len(all_scenes):.1f}s")
    
    return all_results


def run_per_scene_experiments_sequential(
    data_root: str,
    output_dir: Path,
    config: ExperimentConfig,
    scenes: List[str] = None,
    gpu_id: int = 0,
):
    """Run experiments on all scenes sequentially (fallback for single GPU)."""
    
    loader = Sim2RealFireLoader(data_root)
    all_scenes = list(loader.scenes.keys())
    
    if scenes is not None:
        all_scenes = [s for s in all_scenes if s in scenes]
    
    print(f"Running per-scene experiments on {len(all_scenes)} scenes SEQUENTIALLY")
    print(f"Using GPU: {gpu_id}")
    print(f"Scenes: {all_scenes}")
    
    all_results = {}
    
    for i, scene_id in enumerate(all_scenes):
        print(f"\n[{i+1}/{len(all_scenes)}] Processing scene {scene_id}")
        
        scene_config = ExperimentConfig(**asdict(config))
        scene_config.device = f'cuda:{gpu_id}'
        scene_config.experiment_name = f'per_scene_{scene_id}'
        
        try:
            results = run_single_scene_experiment(
                scene_id, scene_config, data_root, output_dir / 'per_scene'
            )
            all_results[scene_id] = results
        except Exception as e:
            print(f"ERROR on scene {scene_id}: {e}")
            traceback.print_exc()
            all_results[scene_id] = {'error': str(e)}
    
    # Save summary
    summary = {
        'experiment_type': 'per_scene',
        'n_scenes': len(all_scenes),
        'scenes': all_scenes,
        'timestamp': datetime.now().isoformat(),
        'results_summary': {},
    }
    
    for scene_id, results in all_results.items():
        if 'error' not in results:
            summary['results_summary'][scene_id] = {
                'correlation': results['test_metrics'].get('correlation_mean', 0),
                'relative_rmse': results['test_metrics'].get('relative_rmse_mean', 0),
                'iou_50': results['test_metrics'].get('iou_50_mean', 0),
                'n_test': results['n_test'],
            }
    
    with open(output_dir / 'per_scene_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print overall summary
    print("\n" + "="*70)
    print("PER-SCENE EXPERIMENT SUMMARY")
    print("="*70)
    
    corrs = []
    rmses = []
    for scene_id, results in all_results.items():
        if 'error' not in results:
            corr = results['test_metrics'].get('correlation_mean', 0)
            rmse = results['test_metrics'].get('relative_rmse_mean', 0)
            corrs.append(corr)
            rmses.append(rmse)
            print(f"  {scene_id}: corr={corr:.3f}, rel_rmse={rmse:.3f}")
    
    print("-"*70)
    print(f"  Average correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
    print(f"  Average relative RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    
    return all_results


# =============================================================================
# Multi-Scene Experiment (with padding)
# =============================================================================

def pad_to_size(arr: np.ndarray, target_H: int, target_W: int, 
                pad_value: float = 0) -> np.ndarray:
    """Pad array to target size."""
    if arr.ndim == 2:
        H, W = arr.shape
        padded = np.full((target_H, target_W), pad_value, dtype=arr.dtype)
        padded[:H, :W] = arr
        return padded
    elif arr.ndim == 3:
        C, H, W = arr.shape
        padded = np.full((C, target_H, target_W), pad_value, dtype=arr.dtype)
        padded[:, :H, :W] = arr
        return padded
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


def run_multi_scene_experiment(
    train_scenes: List[str],
    test_scenes: List[str],
    data_root: str,
    output_dir: Path,
    config: ExperimentConfig,
):
    """
    Train on multiple scenes, test on held-out scenes.
    Pads all scenes to the maximum grid size.
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-SCENE EXPERIMENT")
    print(f"Train scenes: {train_scenes}")
    print(f"Test scenes: {test_scenes}")
    print(f"{'='*70}")
    
    loader = Sim2RealFireLoader(data_root)
    warmup_numba()
    
    # Find max grid size across all scenes
    max_H, max_W = 0, 0
    scene_grids = {}
    
    for scene_id in train_scenes + test_scenes:
        if scene_id not in loader.scenes:
            print(f"Warning: Scene {scene_id} not found")
            continue
        
        events = loader.scenes[scene_id]['events']
        data = loader.load_scenario(scene_id, events[0])
        elev = data['topography'].get('elevation')
        H, W = elev.shape
        scene_grids[scene_id] = (H, W)
        max_H = max(max_H, H)
        max_W = max(max_W, W)
    
    print(f"Max grid size: {max_H} x {max_W}")
    print(f"Scene grid sizes: {scene_grids}")
    
    # Setup trainer with max grid size
    trainer = SceneTrainer(config, loader)
    trainer.setup_model(max_H, max_W)
    
    # Load and pad all training scenarios
    print("\nLoading training scenarios...")
    train_scenarios = []
    
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            continue
        
        events = loader.scenes[scene_id]['events']
        H, W = scene_grids[scene_id]
        
        # Use 70% for training
        np.random.seed(config.seed)
        np.random.shuffle(events)
        n_train = int(len(events) * 0.7)
        train_events = events[:n_train]
        
        for event_id in train_events[:200]:  # Limit per scene
            scenario = trainer.prepare_scenario(scene_id, event_id)
            if scenario is not None:
                # Pad to max size
                scenario['spatial_feat'] = torch.from_numpy(
                    pad_to_size(scenario['spatial_feat'].numpy(), max_H, max_W)
                )
                scenario['arrival_times'] = pad_to_size(
                    scenario['arrival_times'], max_H, max_W, pad_value=np.inf
                )
                scenario['source_mask'] = pad_to_size(
                    scenario['source_mask'].astype(np.float32), max_H, max_W
                ).astype(bool)
                scenario['valid_mask'] = pad_to_size(
                    scenario['valid_mask'].astype(np.float32), max_H, max_W
                ).astype(bool)
                
                train_scenarios.append(scenario)
    
    print(f"Loaded {len(train_scenarios)} training scenarios")
    
    # Load validation (from training scenes)
    val_scenarios = []
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            continue
        
        events = loader.scenes[scene_id]['events']
        H, W = scene_grids[scene_id]
        
        np.random.seed(config.seed)
        np.random.shuffle(events)
        n_train = int(len(events) * 0.7)
        n_val = int(len(events) * 0.15)
        val_events = events[n_train:n_train + n_val]
        
        for event_id in val_events[:50]:
            scenario = trainer.prepare_scenario(scene_id, event_id)
            if scenario is not None:
                scenario['spatial_feat'] = torch.from_numpy(
                    pad_to_size(scenario['spatial_feat'].numpy(), max_H, max_W)
                )
                scenario['arrival_times'] = pad_to_size(
                    scenario['arrival_times'], max_H, max_W, pad_value=np.inf
                )
                scenario['source_mask'] = pad_to_size(
                    scenario['source_mask'].astype(np.float32), max_H, max_W
                ).astype(bool)
                scenario['valid_mask'] = pad_to_size(
                    scenario['valid_mask'].astype(np.float32), max_H, max_W
                ).astype(bool)
                
                val_scenarios.append(scenario)
    
    print(f"Loaded {len(val_scenarios)} validation scenarios")
    
    # Load test scenarios (from held-out scenes)
    test_scenarios = []
    for scene_id in test_scenes:
        if scene_id not in loader.scenes:
            continue
        
        events = loader.scenes[scene_id]['events']
        H, W = scene_grids.get(scene_id, (max_H, max_W))
        
        for event_id in events[:100]:  # Test on up to 100 per scene
            scenario = trainer.prepare_scenario(scene_id, event_id)
            if scenario is not None:
                scenario['spatial_feat'] = torch.from_numpy(
                    pad_to_size(scenario['spatial_feat'].numpy(), max_H, max_W)
                )
                scenario['arrival_times'] = pad_to_size(
                    scenario['arrival_times'], max_H, max_W, pad_value=np.inf
                )
                scenario['source_mask'] = pad_to_size(
                    scenario['source_mask'].astype(np.float32), max_H, max_W
                ).astype(bool)
                scenario['valid_mask'] = pad_to_size(
                    scenario['valid_mask'].astype(np.float32), max_H, max_W
                ).astype(bool)
                
                test_scenarios.append(scenario)
    
    print(f"Loaded {len(test_scenarios)} test scenarios from held-out scenes")
    
    # Train
    print("\nTraining...")
    history = trainer.train(train_scenarios, val_scenarios, verbose=True)
    
    # Test evaluation with saved predictions
    print("\nEvaluating on held-out test scenes...")
    test_agg, test_details, test_predictions = trainer.evaluate(test_scenarios, save_predictions=True)
    
    # Results
    exp_name = f"{len(train_scenes)}_scenes"
    results = {
        'experiment_name': exp_name,
        'train_scenes': train_scenes,
        'test_scenes': test_scenes,
        'max_grid_size': [max_H, max_W],
        'scene_grid_sizes': {k: list(v) for k, v in scene_grids.items()},
        'n_train': len(train_scenarios),
        'n_val': len(val_scenarios),
        'n_test': len(test_scenarios),
        'history': history,
        'test_metrics': test_agg,
        'test_details': test_details,
        'config': asdict(config),
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
        }
    }
    
    # Save
    exp_output_dir = output_dir / 'multi_scene' / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model with metadata
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': asdict(config),
        'max_grid_size': (max_H, max_W),
        'train_scenes': train_scenes,
        'test_scenes': test_scenes,
    }, exp_output_dir / 'model.pt')
    
    # Save predictions
    if test_predictions is not None:
        np.savez_compressed(
            exp_output_dir / 'test_predictions.npz',
            pred_T=test_predictions['pred_T'],
            true_T=test_predictions['true_T'],
            valid_mask=test_predictions['valid_mask'],
            event_ids=test_predictions['event_ids'],
            metadata={
                'train_scenes': train_scenes,
                'test_scenes': test_scenes,
                'max_grid_size': [max_H, max_W],
                'n_test': len(test_scenarios),
                'timestamp': datetime.now().isoformat(),
            }
        )
    
    with open(exp_output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Multi-Scene Results ({len(train_scenes)} train scenes → {len(test_scenes)} test scenes):")
    print(f"  Test correlation: {test_agg.get('correlation_mean', 0):.3f} ± {test_agg.get('correlation_std', 0):.3f}")
    print(f"  Test relative RMSE: {test_agg.get('relative_rmse_mean', 0):.3f} ± {test_agg.get('relative_rmse_std', 0):.3f}")
    print(f"  Test IoU@50%: {test_agg.get('iou_50_mean', 0):.3f}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    # IMPORTANT: Set multiprocessing start method BEFORE any CUDA operations
    # This must be done in the main block, before any child processes are created
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description='Sim2Real-Fire Experiment Suite (Parallel)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='per_scene',
                       choices=['per_scene', 'multi_scene', 'all'],
                       help='Experiment mode')
    
    # Data
    parser.add_argument('--data_root', type=str,
                       default=None,
                        help='Path to Sim2Real-Fire data root (required)')
    parser.add_argument('--output', type=str, default='./experiments')
    
    # Scene selection
    parser.add_argument('--scenes', type=str, default=None,
                       help='Comma-separated scene IDs (default: all)')
    parser.add_argument('--n_train_scenes', type=int, default=5,
                       help='Number of training scenes for multi-scene mode')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    
    # Hardware
    parser.add_argument('--gpus', type=str, default='0',
                       help='Comma-separated GPU IDs for parallel execution')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--cores', type=int, default=47,
                       help='Total CPU cores to use (default 47, leaving 1 idle from 48)')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential execution (disable parallel)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Parse GPUs
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    n_gpus = len(gpu_ids)
    
    # Config
    config = ExperimentConfig(
        n_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        n_workers=args.n_workers,
        seed=args.seed,
        device=f'cuda:{gpu_ids[0]}',
    )
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse scenes
    scenes = None
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(',')]
    
    print("="*70)
    print("SIM2REAL-FIRE EXPERIMENT SUITE (PARALLEL)")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Data root: {args.data_root}")
    print(f"Output: {output_dir}")
    print(f"GPUs: {gpu_ids} ({'PARALLEL' if n_gpus > 1 and not args.sequential else 'SEQUENTIAL'})")
    print(f"CPU cores: {args.cores} (leaving {48 - args.cores} idle)")
    print(f"Scenes: {scenes if scenes else 'all'}")
    print()
    
    if args.mode == 'per_scene' or args.mode == 'all':
        if n_gpus > 1 and not args.sequential:
            # PARALLEL execution
            run_per_scene_experiments_parallel(
                args.data_root, output_dir, config,
                scenes=scenes, gpu_ids=gpu_ids, total_cores=args.cores
            )
        else:
            # SEQUENTIAL execution
            run_per_scene_experiments_sequential(
                args.data_root, output_dir, config,
                scenes=scenes, gpu_id=gpu_ids[0]
            )
    
    if args.mode == 'multi_scene' or args.mode == 'all':
        # Get all scenes
        loader = Sim2RealFireLoader(args.data_root)
        all_scenes = sorted(loader.scenes.keys())
        
        # Run experiments with 5, 10, 15 training scenes
        for n_train in [5, 10, 15]:
            if n_train >= len(all_scenes):
                continue
            
            np.random.seed(args.seed)
            np.random.shuffle(all_scenes)
            
            train_scenes = all_scenes[:n_train]
            test_scenes = all_scenes[n_train:]
            
            run_multi_scene_experiment(
                train_scenes, test_scenes,
                args.data_root, output_dir, config
            )
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
