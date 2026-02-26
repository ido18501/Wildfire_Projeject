"""
Optimized Patched Cross-Scene Training for Wildfire Model

Multi-GPU, high batch size, parallel data loading version.

Key optimizations:
- Multi-GPU support (DataParallel)
- Parallel CPU workers for data loading
- Large batch sizes (64+)
- No artificial limits on fires per scene
- Parallel forward solving using joblib

Author: Based on discussion with Barak
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Any
from scipy import ndimage
from joblib import Parallel, delayed
import time
import json
from pathlib import Path
from dataclasses import dataclass
import argparse
import os

# Import existing modules
# Import existing modules
import sys

from experiments.wildfire.sim2real_loader import Sim2RealFireLoader, CovariateAdapter, extract_arrival_times, find_ignition_point
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.regularization import compute_tv_regularization
from eikonal.nn.config import TVMetricType
from eikonal.forward.pytorch import EikonalProblem
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
from eikonal.forward.cuda_batched import fast_sweeping_solve_batched_v3, warmup_v3
from eikonal.backward.adjoint_batched import batched_backward_pass

# For metrics
from scipy.stats import spearmanr, kendalltau


# =============================================================================
# Patch Extraction (same as before)
# =============================================================================

def find_sources_in_patch(arrival_times: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Find source points for disconnected fire components."""
    H, W = arrival_times.shape
    source_mask = np.zeros((H, W), dtype=bool)
    
    if not valid_mask.any():
        return source_mask
    
    labeled, num_components = ndimage.label(valid_mask)
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id)
        comp_arrivals = np.where(comp_mask, arrival_times, np.inf)
        min_idx = np.unravel_index(np.argmin(comp_arrivals), comp_arrivals.shape)
        
        si, sj = min_idx
        source_mask[max(0, si-1):min(H, si+2), max(0, sj-1):min(W, sj+2)] = True
    
    source_mask = source_mask & valid_mask
    
    if not source_mask.any() and valid_mask.any():
        valid_arrivals = np.where(valid_mask, arrival_times, np.inf)
        min_idx = np.unravel_index(np.argmin(valid_arrivals), valid_arrivals.shape)
        si, sj = min_idx
        source_mask[max(0, si-1):min(H, si+2), max(0, sj-1):min(W, sj+2)] = True
        source_mask = source_mask & valid_mask
    
    return source_mask


def extract_patch(
    spatial_feat: np.ndarray,
    global_feat: np.ndarray,
    arrival_times: np.ndarray,
    i_start: int, j_start: int,
    patch_size: int,
    scene_id: str = '',
    event_id: str = '',
    patch_id: int = 0,
    min_burned_pixels: int = 100,
) -> Optional[Dict]:
    """Extract a single patch from a fire scenario."""
    C, H_full, W_full = spatial_feat.shape
    
    i_end = min(i_start + patch_size, H_full)
    j_end = min(j_start + patch_size, W_full)
    
    spatial_patch = spatial_feat[:, i_start:i_end, j_start:j_end]
    arrival_patch = arrival_times[i_start:i_end, j_start:j_end]
    valid_mask = np.isfinite(arrival_patch)
    
    if valid_mask.sum() < min_burned_pixels:
        return None
    
    actual_H, actual_W = arrival_patch.shape
    if actual_H < patch_size or actual_W < patch_size:
        spatial_padded = np.zeros((C, patch_size, patch_size), dtype=spatial_feat.dtype)
        spatial_padded[:, :actual_H, :actual_W] = spatial_patch
        spatial_patch = spatial_padded
        
        arrival_padded = np.full((patch_size, patch_size), np.inf, dtype=arrival_patch.dtype)
        arrival_padded[:actual_H, :actual_W] = arrival_patch
        arrival_patch = arrival_padded
        
        valid_padded = np.zeros((patch_size, patch_size), dtype=bool)
        valid_padded[:actual_H, :actual_W] = valid_mask
        valid_mask = valid_padded
    
    source_mask = find_sources_in_patch(arrival_patch, valid_mask)
    
    if not source_mask.any():
        return None
    
    source_time = arrival_patch[source_mask].min()
    arrival_normalized = arrival_patch - source_time
    arrival_normalized[~valid_mask] = np.inf
    
    return {
        'spatial_feat': torch.from_numpy(spatial_patch.copy()).float(),
        'global_feat': torch.from_numpy(global_feat.copy()).float(),
        'arrival_times': arrival_normalized.astype(np.float32),
        'source_mask': source_mask,
        'valid_mask': valid_mask,
        'scene_id': scene_id,
        'event_id': event_id,
        'patch_id': patch_id,
    }


def extract_patches_grid(
    spatial_feat: np.ndarray,
    global_feat: np.ndarray,
    arrival_times: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    scene_id: str = '',
    event_id: str = '',
    min_burned_pixels: int = 100,
) -> List[Dict]:
    """Extract patches using a regular grid with overlap."""
    C, H, W = spatial_feat.shape
    patches = []
    patch_id = 0
    
    for i_start in range(0, H, stride):
        for j_start in range(0, W, stride):
            patch = extract_patch(
                spatial_feat, global_feat, arrival_times,
                i_start, j_start, patch_size,
                scene_id, event_id, patch_id, min_burned_pixels
            )
            if patch is not None:
                patches.append(patch)
                patch_id += 1
    
    return patches


# =============================================================================
# Dataset with Prefetching
# =============================================================================

class PatchDataset(Dataset):
    """Dataset of patches for training."""
    
    def __init__(self, patches: List[Dict]):
        self.patches = patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx) -> Dict:
        p = self.patches[idx]
        return {
            'spatial_feat': p['spatial_feat'],
            'global_feat': p['global_feat'],
            'arrival_times': torch.from_numpy(p['arrival_times']).float(),
            'source_mask': torch.from_numpy(p['source_mask'].astype(np.int8)),
            'valid_mask': torch.from_numpy(p['valid_mask']),
        }


def collate_patches(batch: List[Dict]) -> Dict:
    """Collate function for patch dataloader."""
    return {
        'spatial_feat': torch.stack([b['spatial_feat'] for b in batch]),
        'global_feat': torch.stack([b['global_feat'] for b in batch]),
        'arrival_times': torch.stack([b['arrival_times'] for b in batch]),
        'source_mask': torch.stack([b['source_mask'] for b in batch]),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
    }


# =============================================================================
# Parallel Forward Solver
# =============================================================================

def solve_single_fire(G_k, B_k, source_mask_k, max_iter=50):
    """Solve eikonal for a single fire (for parallel execution)."""
    source_mask_torch = torch.from_numpy(source_mask_k.astype(bool)).unsqueeze(0)
    G_torch = torch.from_numpy(G_k[np.newaxis, ...])
    B_torch = torch.from_numpy(B_k[np.newaxis, ...])
    problem = EikonalProblem(G_torch, B_torch, source_mask_torch)
    T, _ = fast_sweeping_solve_numba(problem, max_iter=max_iter)
    return T[0].numpy()


def parallel_forward_solve(G_np, B_np, source_masks, max_iter=50, n_jobs=8):
    """Parallel forward solve using joblib."""
    K = G_np.shape[0]
    
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(solve_single_fire)(G_np[k], B_np[k], source_masks[k], max_iter)
        for k in range(K)
    )
    
    return np.stack(results, axis=0)


# =============================================================================
# Multi-GPU Trainer
# =============================================================================

class OptimizedTrainer:
    """
    Optimized trainer with multi-GPU and parallel processing.
    Uses EXACT same loss, gradients, and metrics as original trainer.py
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 5e-4,
        lambda_G: float = 0.01,
        lambda_B: float = 0.01,
        batch_size: int = 64,
        num_workers: int = 8,
        n_gpus: int = None,
        n_solver_jobs: int = 16,
        metric_type: TVMetricType = TVMetricType.LOG_EUCLIDEAN,
    ):
        self.batch_size = batch_size
        self.lambda_G = lambda_G
        self.lambda_B = lambda_B
        self.num_workers = num_workers
        self.n_solver_jobs = n_solver_jobs
        self.metric_type = metric_type
        
        # Multi-GPU setup
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
        
        self.n_gpus = n_gpus
        self.device = device
        
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs with DataParallel")
            self.model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
            self.model = self.model.to(device)
            self.base_model = model  # Keep reference for saving
        else:
            self.model = model.to(device)
            self.base_model = model
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Warmup solvers
        warmup_v3()
    
    def compute_tv_loss(self, G_local: Tensor, B_local: Tensor) -> Tensor:
        """Total Variation regularization - EXACT match to original trainer.py"""
        return compute_tv_regularization(
            G_local, B_local,
            self.lambda_G, self.lambda_B,
            self.metric_type
        )
    
    def forward_solve_batch(
        self,
        G_batch: Tensor,
        B_batch: Tensor,
        source_masks: np.ndarray,
        max_iter: int = 50,
    ) -> Tensor:
        """Parallel forward solve using multiple CPU cores."""
        G_np = G_batch.detach().cpu().numpy()
        B_np = B_batch.detach().cpu().numpy()
        
        T_batch = parallel_forward_solve(
            G_np, B_np, source_masks, 
            max_iter=max_iter, 
            n_jobs=self.n_solver_jobs
        )
        
        return torch.from_numpy(T_batch).to(self.device)
    
    def compute_loss_and_grad(
        self,
        T_pred: Tensor,
        T_obs: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute MSE loss and gradient using MEAN (not sum).
        
        Using mean ensures consistent gradient magnitude across fires of different sizes.
        This is important for multi-fire training where fires have varying numbers of
        observed pixels.
        
        Loss = (1/N) * sum((pred - obs)^2)
        dL/dT = (2/N) * (pred - obs) for observed pixels
        """
        K, H, W = T_pred.shape
        
        dL_dT = torch.zeros_like(T_pred)
        total_loss = 0.0
        n_fires_with_data = 0
        
        for k in range(K):
            # obs_mask: valid, not source, finite prediction and observation
            obs_mask = (
                valid_mask[k] & 
                ~source_mask[k].bool() & 
                (T_pred[k] < 1e5) & 
                torch.isfinite(T_obs[k])
            )
            
            n_obs = obs_mask.sum().item()
            if n_obs == 0:
                continue
            
            n_fires_with_data += 1
            
            # Residuals
            residuals = T_pred[k][obs_mask] - T_obs[k][obs_mask]
            
            # Mean loss (consistent gradients across fire sizes)
            loss_k = (residuals ** 2).mean()
            total_loss += loss_k.item()
            
            # Gradient: dL/dT = (2/N) * residuals = 2 * residuals / n_obs
            # But we use 0.5 * MSE so gradient is just residuals / n_obs
            dL_dT[k][obs_mask] = residuals / n_obs
        
        # Average loss across fires in batch
        loss = torch.tensor(total_loss / max(n_fires_with_data, 1), device=self.device)
        return loss, dL_dT
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step - EXACT match to original trainer.py
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        spatial = batch['spatial_feat'].to(self.device)
        global_feat = batch['global_feat'].to(self.device)
        T_obs = batch['arrival_times'].to(self.device)
        source_masks = batch['source_mask'].numpy()
        valid_mask = batch['valid_mask'].to(self.device)
        source_mask_tensor = batch['source_mask'].to(self.device)
        
        # Forward through network (multi-GPU)
        G, B, G_local, B_local = self.model(spatial, global_feat)
        
        # TV loss (using original compute_tv_regularization)
        tv_loss = self.compute_tv_loss(G_local, B_local)
        tv_loss.backward(retain_graph=True)
        
        # Parallel forward solve
        with torch.no_grad():
            T_pred = self.forward_solve_batch(G, B, source_masks)
        
        # Data loss (EXACT match to original)
        with torch.no_grad():
            data_loss, dL_dT = self.compute_loss_and_grad(
                T_pred, T_obs, valid_mask, source_mask_tensor
            )
        
        # Backward pass (batched Numba)
        with torch.no_grad():
            dL_dG, dL_dB = batched_backward_pass(
                T_pred.cpu(), dL_dT.cpu(),
                G.detach().cpu(), B.detach().cpu(),
                source_masks
            )
        
        # IMPORTANT: Mask gradients to valid region (EXACT match to original)
        # Original: dL_dG_cpu = dL_dG_cpu * valid_tensor.expand(1, 3, -1, -1)
        valid_tensor = valid_mask.float().unsqueeze(1)  # (K, 1, H, W)
        dL_dG = dL_dG.to(self.device) * valid_tensor.expand(-1, 3, -1, -1)
        dL_dB = dL_dB.to(self.device) * valid_tensor.expand(-1, 2, -1, -1)
        
        # Backprop
        G.backward(dL_dG, retain_graph=True)
        B.backward(dL_dB)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'data_loss': data_loss.item(),
            'tv_loss': tv_loss.item(),
            'total_loss': data_loss.item() + tv_loss.item(),
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        losses = {'data_loss': [], 'tv_loss': [], 'total_loss': []}
        
        for batch in dataloader:
            step_losses = self.train_step(batch)
            for k, v in step_losses.items():
                losses[k].append(v)
        
        # Mean across batches (consistent with mean loss per fire)
        return {k: np.mean(v) for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate_full_scene(self, scenario: Dict, return_predictions: bool = False) -> Dict:
        """
        Evaluate on a full scene with comprehensive metrics.
        
        Metrics:
        - Relative RMSE: RMSE / max_time_gt
        - RMSE, MAE
        - Correlations: Pearson, Spearman, Kendall
        - IoU at 25%, 50%, 75%, 100% thresholds
        
        Args:
            scenario: Dict with spatial_feat, global_feat, arrival_times, etc.
            return_predictions: If True, include T_pred, T_obs, G, B in output
        """
        self.model.eval()
        
        spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
        global_feat = scenario['global_feat'].unsqueeze(0).to(self.device)
        
        # Handle DataParallel
        if isinstance(self.model, nn.DataParallel):
            G, B, _, _ = self.model.module(spatial, global_feat)
        else:
            G, B, _, _ = self.model(spatial, global_feat)
        
        source_mask = scenario['source_mask'][np.newaxis, ...]
        source_mask_torch = torch.from_numpy(source_mask.astype(bool))
        
        problem = EikonalProblem(G.cpu(), B.cpu(), source_mask_torch)
        T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
        T_pred = T[0].numpy()
        
        # Get ground truth
        T_obs = scenario['arrival_times']
        valid_mask = scenario['valid_mask']
        
        # Valid pixels: burned in GT, finite prediction
        valid = valid_mask & np.isfinite(T_obs) & (T_pred < 1e5) & np.isfinite(T_pred)
        
        # Skip if too few valid pixels
        n_valid = valid.sum()
        if n_valid < 10:
            result = {
                'rel_rmse': np.nan, 'rmse': np.nan, 'mae': np.nan,
                'correlation': np.nan, 'spearman': np.nan, 'kendall': np.nan,
                'iou_t25': np.nan, 'iou_t50': np.nan, 'iou_t75': np.nan, 'iou_t100': np.nan,
                'scene_id': scenario.get('scene_id', ''),
                'event_id': scenario.get('event_id', ''),
                'n_valid': n_valid,
            }
            if return_predictions:
                result['T_pred'] = T_pred
                result['T_obs'] = T_obs
                result['valid_mask'] = valid_mask
                result['source_mask'] = scenario['source_mask']
                result['G'] = G[0].cpu().numpy()  # (3, H, W)
                result['B'] = B[0].cpu().numpy()  # (2, H, W)
            return result
        
        pred_valid = T_pred[valid]
        obs_valid = T_obs[valid]
        
        # Max time in ground truth (for normalization and IoU thresholds)
        max_time_gt = obs_valid.max()
        
        # RMSE and Relative RMSE
        rmse = np.sqrt(np.mean((pred_valid - obs_valid) ** 2))
        rel_rmse = rmse / max_time_gt if max_time_gt > 0 else np.nan
        
        # MAE
        mae = np.mean(np.abs(pred_valid - obs_valid))
        
        # Pearson correlation
        corr = np.corrcoef(pred_valid, obs_valid)[0, 1] if len(pred_valid) > 1 else np.nan
        
        # Spearman and Kendall correlations
        spearman_corr, _ = spearmanr(pred_valid, obs_valid)
        kendall_corr, _ = kendalltau(pred_valid, obs_valid)
        
        # IoU at different time thresholds (25%, 50%, 75%, 100% of max_time_gt)
        iou_results = {}
        for frac, key in [(0.25, 'iou_t25'), (0.50, 'iou_t50'), (0.75, 'iou_t75'), (1.0, 'iou_t100')]:
            t_threshold = max_time_gt * frac
            
            # Burned at time t: arrival_time <= t AND finite
            pred_burned = (T_pred <= t_threshold) & np.isfinite(T_pred) & (T_pred < 1e5)
            true_burned = (T_obs <= t_threshold) & np.isfinite(T_obs)
            
            intersection = (pred_burned & true_burned).sum()
            union = (pred_burned | true_burned).sum()
            
            iou_results[key] = intersection / (union + 1e-8)
        
        result = {
            'rel_rmse': rel_rmse,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            **iou_results,
            'max_time_gt': max_time_gt,
            'scene_id': scenario.get('scene_id', ''),
            'event_id': scenario.get('event_id', ''),
            'n_valid': n_valid,
        }
        
        if return_predictions:
            result['T_pred'] = T_pred
            result['T_obs'] = T_obs
            result['valid_mask'] = valid_mask
            result['source_mask'] = scenario['source_mask']
            result['G'] = G[0].cpu().numpy()  # (3, H, W) - g11, g12, g22
            result['B'] = B[0].cpu().numpy()  # (2, H, W) - b1, b2
        
        return result
    
    def evaluate(self, scenarios: List[Dict], verbose: bool = False, 
                 return_predictions: bool = False) -> Tuple[Dict, List[Dict]]:
        """
        Evaluate on full-scene scenarios with comprehensive metrics.
        
        Args:
            scenarios: List of scenario dicts
            verbose: Print summary metrics
            return_predictions: Include T_pred, T_obs, G, B in each result
        """
        metrics_list = []
        
        for s in scenarios:
            m = self.evaluate_full_scene(s, return_predictions=return_predictions)
            metrics_list.append(m)
        
        # Aggregate all metrics
        def safe_mean(values):
            valid = [v for v in values if np.isfinite(v)]
            return np.mean(valid) if valid else np.nan
        
        def safe_std(values):
            valid = [v for v in values if np.isfinite(v)]
            return np.std(valid) if len(valid) > 1 else np.nan
        
        agg = {
            # Relative RMSE (main metric for comparison)
            'rel_rmse': safe_mean([m['rel_rmse'] for m in metrics_list]),
            'rel_rmse_std': safe_std([m['rel_rmse'] for m in metrics_list]),
            
            # RMSE and MAE
            'rmse': safe_mean([m['rmse'] for m in metrics_list]),
            'mae': safe_mean([m['mae'] for m in metrics_list]),
            
            # Correlations
            'correlation': safe_mean([m['correlation'] for m in metrics_list]),
            'correlation_std': safe_std([m['correlation'] for m in metrics_list]),
            'spearman': safe_mean([m['spearman'] for m in metrics_list]),
            'kendall': safe_mean([m['kendall'] for m in metrics_list]),
            
            # IoU at different thresholds
            'iou_t25': safe_mean([m['iou_t25'] for m in metrics_list]),
            'iou_t50': safe_mean([m['iou_t50'] for m in metrics_list]),
            'iou_t75': safe_mean([m['iou_t75'] for m in metrics_list]),
            'iou_t100': safe_mean([m['iou_t100'] for m in metrics_list]),
            
            'n_fires': len(metrics_list),
        }
        
        if verbose:
            print(f"  Rel RMSE: {agg['rel_rmse']:.3f} ± {agg['rel_rmse_std']:.3f}")
            print(f"  Correlation: {agg['correlation']:.3f} ± {agg['correlation_std']:.3f}")
            print(f"  Spearman: {agg['spearman']:.3f}, Kendall: {agg['kendall']:.3f}")
            print(f"  IoU@25%: {agg['iou_t25']:.3f}, IoU@50%: {agg['iou_t50']:.3f}, "
                  f"IoU@75%: {agg['iou_t75']:.3f}, IoU@100%: {agg['iou_t100']:.3f}")
            print(f"  RMSE: {agg['rmse']:.3f}, MAE: {agg['mae']:.3f}")
        
        return agg, metrics_list
    
    def train(
        self,
        train_patches: List[Dict],
        val_scenarios: List[Dict],
        n_epochs: int = 100,
        early_stop_patience: int = 20,
        save_path: Optional[Path] = None,
        val_every: int = 1,
    ) -> Dict:
        """Full training loop - matches original trainer.py structure."""
        
        train_dataset = PatchDataset(train_patches)
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'collate_fn': collate_patches,
            'num_workers': self.num_workers,
            'pin_memory': True if self.num_workers > 0 else False,
        }
        if self.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 2
        
        train_loader = DataLoader(train_dataset, **loader_kwargs)
        
        history = {
            'train_loss': [], 'train_loss_data': [], 'train_loss_reg': [],
            'val_corr': [], 'val_rel_rmse': [], 'val_iou_t50': [],
            'lr': [],
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"  Patches: {len(train_patches)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Batches/epoch: {len(train_loader)}")
        print(f"  GPUs: {self.n_gpus}")
        print(f"  Data workers: {self.num_workers}")
        print(f"  Solver jobs: {self.n_solver_jobs}")
        print(f"  Validation scenes: {len(val_scenarios)}")
        print(f"  TV Metric: {self.metric_type.value}")
        print(f"  lambda_G: {self.lambda_G}, lambda_B: {self.lambda_B}")
        print(f"  Loss: MEAN (for consistent gradient magnitude)")
        print(f"{'='*70}\n")
        
        for epoch in range(n_epochs):
            t0 = time.perf_counter()
            
            train_metrics = self.train_epoch(train_loader)
            total_loss = train_metrics['total_loss']
            
            # Scheduler on total loss (matches original)
            self.scheduler.step(total_loss)
            
            # Validate periodically
            if epoch % val_every == 0:
                val_agg, _ = self.evaluate(val_scenarios)
                history['val_corr'].append(val_agg['correlation'])
                history['val_rel_rmse'].append(val_agg['rel_rmse'])
                history['val_iou_t50'].append(val_agg['iou_t50'])
                val_corr = val_agg['correlation']
                val_rel_rmse = val_agg['rel_rmse']
            else:
                val_corr = history['val_corr'][-1] if history['val_corr'] else 0
                val_rel_rmse = history['val_rel_rmse'][-1] if history['val_rel_rmse'] else np.nan
            
            history['train_loss'].append(total_loss)
            history['train_loss_data'].append(train_metrics['data_loss'])
            history['train_loss_reg'].append(train_metrics['tv_loss'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            t_epoch = time.perf_counter() - t0
            
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Loss: {total_loss:.4f} (data={train_metrics['data_loss']:.4f}, reg={train_metrics['tv_loss']:.4f}) | "
                  f"Val: corr={val_corr:.3f}, relRMSE={val_rel_rmse:.3f} | "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {t_epoch:.1f}s")
            
            # Save best model based on loss (matches original)
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.base_model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history


# =============================================================================
# Data Loading with Parallel Processing
# =============================================================================

def load_full_scene_scenario(
    loader: Sim2RealFireLoader,
    adapter: CovariateAdapter,
    scene_id: str,
    event_id: str,
) -> Optional[Dict]:
    """Load a complete fire scenario at full resolution."""
    try:
        data = loader.load_scenario(scene_id, event_id)
        spatial_feat, global_feat = adapter(data)
        
        masks = data['masks']
        timestamps = data['timestamps']
        arrival = extract_arrival_times(masks, timestamps)
        ignition = find_ignition_point(masks)
        
        H, W = arrival.shape
        si, sj = ignition if ignition else (H//2, W//2)
        
        source_mask = np.zeros((H, W), dtype=bool)
        source_mask[max(0, si-1):min(H, si+2), max(0, sj-1):min(W, sj+2)] = True
        
        valid_mask = np.isfinite(arrival)
        
        return {
            'spatial_feat': spatial_feat,
            'global_feat': global_feat,
            'arrival_times': arrival,
            'source_mask': source_mask,
            'valid_mask': valid_mask,
            'scene_id': scene_id,
            'event_id': event_id,
            'grid_size': (H, W),
        }
    except Exception as e:
        return None


def load_scene_patches_parallel(
    loader: Sim2RealFireLoader,
    adapter: CovariateAdapter,
    scene_id: str,
    event_ids: List[str],
    patch_size: int,
    patch_stride: int,
    n_jobs: int = 8,
) -> List[Dict]:
    """Load and extract patches from a scene in parallel."""
    
    def process_event(event_id):
        scenario = load_full_scene_scenario(loader, adapter, scene_id, event_id)
        if scenario is None:
            return []
        
        return extract_patches_grid(
            scenario['spatial_feat'].numpy(),
            scenario['global_feat'].numpy(),
            scenario['arrival_times'],
            patch_size=patch_size,
            stride=patch_stride,
            scene_id=scene_id,
            event_id=event_id,
        )
    
    # Parallel processing of events
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_event)(event_id) for event_id in event_ids
    )
    
    # Flatten results
    patches = []
    for patch_list in results:
        patches.extend(patch_list)
    
    return patches


# =============================================================================
# Main Experiment
# =============================================================================

def run_optimized_experiment(
    train_scenes: List[str],
    test_scenes: List[str],
    data_root: str,
    output_dir: Path,
    patch_size: int = 256,
    patch_stride: int = 64,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    lambda_G: float = 0.01,
    lambda_B: float = 0.01,
    num_workers: int = 8,
    n_gpus: int = None,
    n_solver_jobs: int = 16,
    seed: int = 42,
    train_split: float = 0.7,
    val_split: float = 0.15,
    metric_type: TVMetricType = TVMetricType.LOG_EUCLIDEAN,
    max_fires_per_scene: int = None,
):
    """Run optimized cross-scene experiment."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect GPUs
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    
    print("=" * 70)
    print("OPTIMIZED CROSS-SCENE EXPERIMENT")
    print("=" * 70)
    print(f"Train scenes: {train_scenes}")
    print(f"Test scenes: {test_scenes}")
    print(f"Patch size: {patch_size}, Stride: {patch_stride}")
    print(f"Batch size: {batch_size}")
    print(f"GPUs: {n_gpus}, Workers: {num_workers}, Solver jobs: {n_solver_jobs}")
    print(f"lambda_G: {lambda_G}, lambda_B: {lambda_B}, TV metric: {metric_type.value}")
    if max_fires_per_scene:
        print(f"Max fires per scene: {max_fires_per_scene} (LIMITED)")
    print("=" * 70)
    
    loader = Sim2RealFireLoader(data_root)
    adapter = CovariateAdapter()
    
    # ==========================================================================
    # Extract training patches
    # ==========================================================================
    if max_fires_per_scene:
        print(f"\nExtracting training patches (LIMITED to {max_fires_per_scene} fires/scene)...")
    else:
        print("\nExtracting training patches (using ALL fires)...")
    train_patches = []
    
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            print(f"  Warning: Scene {scene_id} not found")
            continue
        
        events = loader.scenes[scene_id]['events'].copy()
        np.random.shuffle(events)
        n_train = int(len(events) * train_split)
        train_events = events[:n_train]
        
        # Apply limit if specified
        if max_fires_per_scene:
            train_events = train_events[:max_fires_per_scene]
        
        t0 = time.perf_counter()
        scene_patches = load_scene_patches_parallel(
            loader, adapter, scene_id, train_events,
            patch_size, patch_stride, n_jobs=max(1, num_workers)
        )
        train_patches.extend(scene_patches)
        
        t_scene = time.perf_counter() - t0
        print(f"  {scene_id}: {len(scene_patches)} patches from {len(train_events)} fires ({t_scene:.1f}s)")
    
    print(f"\nTotal training patches: {len(train_patches)}")
    
    # ==========================================================================
    # Load validation scenarios
    # ==========================================================================
    print("\nLoading validation scenarios...")
    val_scenarios = []
    
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            continue
        
        events = loader.scenes[scene_id]['events'].copy()
        np.random.seed(seed)
        np.random.shuffle(events)
        n_train = int(len(events) * train_split)
        n_val = int(len(events) * val_split)
        val_events = events[n_train:n_train + n_val]
        
        for event_id in val_events:
            scenario = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if scenario is not None:
                val_scenarios.append(scenario)
    
    print(f"Validation scenarios: {len(val_scenarios)}")
    
    # ==========================================================================
    # Load test scenarios
    # ==========================================================================
    print("\nLoading test scenarios (held-out scenes)...")
    test_scenarios = []
    
    for scene_id in test_scenes:
        if scene_id not in loader.scenes:
            print(f"  Warning: Scene {scene_id} not found")
            continue
        
        events = loader.scenes[scene_id]['events']
        
        for event_id in events:
            scenario = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if scenario is not None:
                test_scenarios.append(scenario)
        
        scene_count = len([s for s in test_scenarios if s['scene_id'] == scene_id])
        print(f"  {scene_id}: {scene_count} scenarios")
    
    print(f"Total test scenarios: {len(test_scenarios)}")
    
    # ==========================================================================
    # Create model and trainer
    # ==========================================================================
    n_spatial = train_patches[0]['spatial_feat'].shape[0]
    n_global = train_patches[0]['global_feat'].shape[0]
    
    model = FlexibleCovariateEncoder(
        n_spatial_features=n_spatial,
        n_global_features=n_global,
        hidden_channels=64,
        num_cnn_layers=5,
        mlp_hidden=128,
        mlp_num_layers=2,
    )
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    trainer = OptimizedTrainer(
        model=model,
        device='cuda',
        lr=lr,
        lambda_G=lambda_G,
        lambda_B=lambda_B,
        batch_size=batch_size,
        num_workers=num_workers,
        n_gpus=n_gpus,
        n_solver_jobs=n_solver_jobs,
        metric_type=metric_type,
    )
    
    # ==========================================================================
    # Train
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    save_path = output_dir / "best_model.pt"
    history = trainer.train(
        train_patches,
        val_scenarios,
        n_epochs=n_epochs,
        early_stop_patience=20,
        save_path=save_path,
    )
    
    # ==========================================================================
    # Test with prediction saving
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST EVALUATION (Held-out scenes)")
    print("=" * 70)
    
    if save_path.exists():
        model.load_state_dict(torch.load(save_path))
        trainer.base_model = model
        if trainer.n_gpus > 1:
            trainer.model = nn.DataParallel(model, device_ids=list(range(trainer.n_gpus)))
        else:
            trainer.model = model
        trainer.model = trainer.model.to('cuda')
    
    # Evaluate WITH predictions for later analysis
    test_agg, test_details = trainer.evaluate(test_scenarios, verbose=True, return_predictions=True)
    
    print("\nPer-scene test results:")
    for scene_id in test_scenes:
        scene_metrics = [m for m in test_details if m['scene_id'] == scene_id]
        if scene_metrics:
            corrs = [m['correlation'] for m in scene_metrics if np.isfinite(m['correlation'])]
            rel_rmses = [m['rel_rmse'] for m in scene_metrics if np.isfinite(m['rel_rmse'])]
            iou50s = [m['iou_t50'] for m in scene_metrics if np.isfinite(m['iou_t50'])]
            if corrs:
                print(f"  {scene_id}: Corr={np.mean(corrs):.3f}±{np.std(corrs):.3f}, "
                      f"RelRMSE={np.mean(rel_rmses):.3f}, IoU@50={np.mean(iou50s):.3f} (n={len(scene_metrics)})")
    
    # ==========================================================================
    # Save results (metrics only - JSON)
    # ==========================================================================
    
    def to_json_serializable(v):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (int, float)):
            return v if np.isfinite(v) else None
        elif isinstance(v, str):
            return v
        else:
            return None
    
    results = {
        'config': {
            'train_scenes': train_scenes,
            'test_scenes': test_scenes,
            'patch_size': patch_size,
            'patch_stride': patch_stride,
            'batch_size': batch_size,
            'n_gpus': n_gpus,
            'lambda_G': lambda_G,
            'lambda_B': lambda_B,
            'lr': lr,
            'n_train_patches': len(train_patches),
            'n_val_scenarios': len(val_scenarios),
            'n_test_scenarios': len(test_scenarios),
        },
        'history': {k: [to_json_serializable(v) for v in vals] 
                    for k, vals in history.items()},
        'test_metrics': {k: to_json_serializable(v) for k, v in test_agg.items()},
        'per_scene_test': {},
        'per_fire_metrics': [],  # All metrics per fire (without arrays)
    }
    
    for scene_id in test_scenes:
        scene_metrics = [m for m in test_details if m['scene_id'] == scene_id]
        if scene_metrics:
            corrs = [m['correlation'] for m in scene_metrics if np.isfinite(m['correlation'])]
            rel_rmses = [m['rel_rmse'] for m in scene_metrics if np.isfinite(m['rel_rmse'])]
            iou50s = [m['iou_t50'] for m in scene_metrics if np.isfinite(m['iou_t50'])]
            results['per_scene_test'][scene_id] = {
                'correlation': float(np.mean(corrs)) if corrs else None,
                'correlation_std': float(np.std(corrs)) if corrs else None,
                'rel_rmse': float(np.mean(rel_rmses)) if rel_rmses else None,
                'rel_rmse_std': float(np.std(rel_rmses)) if rel_rmses else None,
                'iou_t50': float(np.mean(iou50s)) if iou50s else None,
                'n_fires': len(scene_metrics),
            }
    
    # Save per-fire metrics (without large arrays)
    for m in test_details:
        fire_metrics = {k: to_json_serializable(v)
                       for k, v in m.items() 
                       if k not in ['T_pred', 'T_obs', 'valid_mask', 'source_mask', 'G', 'B']}
        results['per_fire_metrics'].append(fire_metrics)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # ==========================================================================
    # Save predictions (numpy arrays - separate file)
    # ==========================================================================
    print("\nSaving predictions...")
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    
    for m in test_details:
        scene_id = m['scene_id']
        event_id = m['event_id']
        
        # Save as compressed npz
        np.savez_compressed(
            predictions_dir / f"{scene_id}_{event_id}.npz",
            T_pred=m['T_pred'],
            T_obs=m['T_obs'],
            valid_mask=m['valid_mask'],
            source_mask=m['source_mask'],
            G=m['G'],  # (3, H, W) - g11, g12, g22
            B=m['B'],  # (2, H, W) - b1, b2
        )
    
    print(f"  Saved {len(test_details)} prediction files to {predictions_dir}")
    print(f"\nResults saved to {output_dir}")
    
    return history, test_agg, test_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Cross-Scene Training")
    parser.add_argument("--data_root", type=str,
                        default="/path/to/sim2real_data")
    parser.add_argument("--output_dir", type=str, default="./optimized_cross_scene_results")
    parser.add_argument("--train_scenes", nargs="+",
                        default=["0001_02614", "0002_00714", "0003_01715", "0004_01191",
                                 "0015_02986", "0016_03070", "0017_02731", "0018_02440",
                                 "0019_01316", "0022_01630", "0025_02019"])
    parser.add_argument("--test_scenes", nargs="+",
                        default=["0005_00725", "0014_00426", "0023_00995", "0024_02655"])
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patch_stride", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_G", type=float, default=0.01)
    parser.add_argument("--lambda_B", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_gpus", type=int, default=None)
    parser.add_argument("--n_solver_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_fires_per_scene", type=int, default=None,
                        help="Limit fires per scene for testing (default: no limit)")
    
    args = parser.parse_args()
    
    run_optimized_experiment(
        train_scenes=args.train_scenes,
        test_scenes=args.test_scenes,
        data_root=args.data_root,
        output_dir=Path(args.output_dir),
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_G=args.lambda_G,
        lambda_B=args.lambda_B,
        num_workers=args.num_workers,
        n_gpus=args.n_gpus,
        n_solver_jobs=args.n_solver_jobs,
        seed=args.seed,
        max_fires_per_scene=args.max_fires_per_scene,
    )
