#!/usr/bin/env python3
"""
Real Fire Data Loader and Training for Sim2Real Dataset
========================================================

This module handles real wildfire data from the Sim2Real-Fire dataset.
Real fires differ from simulated fires in that arrival times are derived
from satellite imagery with coarse temporal resolution (days vs hours).

Key differences from simulation data:
- Arrival times derived from dated satellite masks (2-8 snapshots per fire)
- Coarse temporal resolution (days between observations)
- Different folder structure (double-nested: fire_id/fire_id/)
- Weather data is per-observation-date, not hourly

Training approach:
- Use IoU at observation dates as primary metric (not continuous RMSE)
- Compute pseudo arrival times from dated masks
- Train with same forward/backward pipeline as simulation

Usage:
    python real_fire_training.py --data_root /path/to/real_fires --epochs 100

Author: Barak Gahtan
Date: January 2026
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
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available, using PIL fallback for TIF loading")

# Project imports - adjust path as needed

# Import from the package structure shown in the directory
try:
    from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba
    SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import solver modules: {e}")
    SOLVER_AVAILABLE = False

try:
    from eikonal.backward.adjoint_numba import solve_adjoint_numba
    from eikonal.backward.implicit_diff import identify_active_stencils, compute_parameter_gradients, backward_pass
    HAVE_NUMBA = True
except ImportError:
    try:
        from eikonal.backward.implicit_diff import backward_pass, identify_active_stencils, compute_parameter_gradients
        HAVE_NUMBA = False
    except ImportError:
        print("Warning: backward_pass not found")
        HAVE_NUMBA = False
        backward_pass = None

try:
    from eikonal.nn.encoder import FlexibleCovariateEncoder
    from eikonal.nn.regularization import compute_tv_regularization
    from eikonal.nn.config import DEVICE, TVMetricType
except ImportError:
    print("Warning: fire_training modules not found, using fallback device")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TVMetricType = None


# =============================================================================
# Real Fire Data Loader
# =============================================================================

class RealFireLoader:
    """
    Load real fire data from Sim2Real-Fire dataset.
    
    Structure (double-nested):
        real_fires/
        └── 0001_00001/
            └── 0001_00001/
                ├── Weather_Data/
                │   └── 0001_00001.txt
                ├── Vegetation_Map/
                │   └── *.tif
                ├── Topography_Map/
                │   └── Elevation.tif, Slope.tif, Aspect.tif
                ├── Fuel_Map/
                │   └── FBFM40.tif, Canopy_*.tif, ...
                ├── Satellite_Images_Mask/
                │   └── 0001_00001/
                │       └── YYYYMMDD.png (dated masks)
                └── Satellite_Images/
    """
    
    # Which TIF files to use
    TOPO_FILES = ['Elevation.tif', 'Slope.tif', 'Aspect.tif']
    FUEL_FILES = ['FBFM40.tif', 'FBFM13.tif', 'Canopy_Cover.tif', 'Canopy_Height.tif']
    VEG_FILES = ['Existing_Vegetation_Type.tif', 'Existing_Vegetation_Cover.tif', 
                 'Existing_Vegetation_Height.tif']
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to real_fires directory
        """
        self.data_root = Path(data_root)
        self.fires = self._discover_fires()
        
    def _discover_fires(self) -> Dict:
        """Find all real fires and their data."""
        fires = {}
        
        for entry in sorted(self.data_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            
            # Handle double-nested structure: fire_id/fire_id/
            fire_dir = entry / entry.name
            if not fire_dir.exists():
                # Try direct structure
                fire_dir = entry
            
            # Check for required subfolders
            mask_dir = fire_dir / "Satellite_Images_Mask"
            topo_dir = fire_dir / "Topography_Map"
            
            if mask_dir.exists() and topo_dir.exists():
                # Find mask subfolder
                mask_subdir = None
                for subdir in mask_dir.iterdir():
                    if subdir.is_dir():
                        mask_subdir = subdir
                        break
                
                if mask_subdir is None:
                    continue
                
                # Find dated mask files
                mask_files = sorted(mask_subdir.glob("*.png"))
                if not mask_files:
                    continue
                
                # Parse dates from filenames
                dates = []
                for f in mask_files:
                    try:
                        date = datetime.strptime(f.stem, "%Y%m%d")
                        dates.append((date, f))
                    except ValueError:
                        continue
                
                if len(dates) < 2:  # Need at least 2 dates
                    continue
                
                dates.sort(key=lambda x: x[0])
                
                fires[entry.name] = {
                    'path': fire_dir,
                    'mask_dir': mask_subdir,
                    'dates': [(d.strftime("%Y-%m-%d"), str(f)) for d, f in dates],
                    'n_dates': len(dates),
                }
        
        return fires
    
    def load_fire(self, fire_id: str) -> Dict:
        """
        Load a complete real fire scenario.
        
        Returns:
            dict with keys:
                'topography': dict of numpy arrays
                'fuel': dict of numpy arrays
                'vegetation': dict of numpy arrays  
                'weather': numpy array (per-date weather)
                'masks': list of (date_str, mask_array) tuples
                'arrival_times': [H, W] array (days from first observation)
                'valid_mask': [H, W] boolean array
                'fire_id': str
                'grid_size': (H, W)
        """
        fire_info = self.fires[fire_id]
        fire_dir = fire_info['path']
        
        # Load static covariates
        topo = self._load_tif_folder(fire_dir / 'Topography_Map', self.TOPO_FILES)
        fuel = self._load_tif_folder(fire_dir / 'Fuel_Map', self.FUEL_FILES)
        veg = self._load_tif_folder(fire_dir / 'Vegetation_Map', self.VEG_FILES)
        
        # Get grid size from elevation
        elev = topo.get('elevation')
        if elev is None:
            # Try slope
            elev = topo.get('slope')
        if elev is None:
            raise ValueError(f"No topography data found for {fire_id}")
        H, W = elev.shape
        
        # Load dated masks
        masks = []
        for date_str, mask_path in fire_info['dates']:
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Use first channel
            mask = (mask > 0).astype(bool)
            masks.append((date_str, mask))
        
        # Compute arrival times from dated masks
        arrival_times = self._compute_arrival_times(masks)
        
        # Valid mask = pixels that burned at some point
        valid_mask = np.isfinite(arrival_times)
        
        # Load weather (if available)
        weather = self._load_weather(fire_dir, fire_id)
        
        return {
            'topography': topo,
            'fuel': fuel,
            'vegetation': veg,
            'weather': weather,
            'masks': masks,
            'arrival_times': arrival_times,
            'valid_mask': valid_mask,
            'fire_id': fire_id,
            'grid_size': (H, W),
            'n_dates': len(masks),
            'date_range': (masks[0][0], masks[-1][0]),
        }
    
    def _load_tif_folder(self, folder: Path, file_list: List[str]) -> Dict:
        """Load specified TIF files from a folder."""
        data = {}
        for filename in file_list:
            filepath = folder / filename
            if filepath.exists():
                key = filename.replace('.tif', '').lower()
                data[key] = self._load_tif(filepath)
        return data
    
    def _load_tif(self, filepath: Path) -> np.ndarray:
        """Load a GeoTIFF file."""
        if HAS_RASTERIO:
            with rasterio.open(filepath) as src:
                data = src.read(1).astype(np.float32)
                # Handle nodata
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                return data
        else:
            img = Image.open(filepath)
            return np.array(img, dtype=np.float32)
    
    def _compute_arrival_times(self, masks: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Compute arrival time field from dated masks.
        
        Arrival time = first observation date when pixel appears burned.
        Times are in days from first observation.
        """
        if not masks:
            raise ValueError("No masks provided")
        
        first_date = datetime.strptime(masks[0][0], "%Y-%m-%d")
        H, W = masks[0][1].shape
        
        arrival = np.full((H, W), np.inf, dtype=np.float32)
        
        for date_str, mask in masks:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            days = (date - first_date).days
            
            # Pixels burned in this mask but not yet assigned
            newly_burned = mask & np.isinf(arrival)
            arrival[newly_burned] = float(days)
        
        return arrival
    
    def _load_weather(self, fire_dir: Path, fire_id: str) -> np.ndarray:
        """
        Load weather data file.
        
        Format: Year Month Day HHMM Temp Humidity Precip WindSpeed WindDir Extra
        
        Returns:
            [N_dates, 5] array: temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos
        """
        weather_dir = fire_dir / 'Weather_Data'
        weather_file = weather_dir / f"{fire_id}.txt"
        
        if not weather_file.exists():
            # Return default weather
            return np.array([[70.0, 40.0, 10.0, 0.0, 1.0]], dtype=np.float32)
        
        data = []
        with open(weather_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    temp = float(parts[4])
                    humidity = float(parts[5])
                    wind_speed = float(parts[7])
                    wind_dir = float(parts[8])
                    
                    wind_dir_rad = np.deg2rad(wind_dir)
                    wind_dir_sin = np.sin(wind_dir_rad)
                    wind_dir_cos = np.cos(wind_dir_rad)
                    
                    data.append([temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos])
        
        if not data:
            return np.array([[70.0, 40.0, 10.0, 0.0, 1.0]], dtype=np.float32)
        
        return np.array(data, dtype=np.float32)


# =============================================================================
# Real Fire Covariate Adapter
# =============================================================================

class RealFireCovariateAdapter:
    """
    Convert real fire covariates to model input format.
    Uses robust normalization matching simulation data pipeline.
    """
    
    def __init__(self, spatial_channels: int = 6, global_channels: int = 5):
        self.spatial_channels = spatial_channels
        self.global_channels = global_channels
    
    def __call__(self, fire_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert fire data to (spatial_features, global_features).
        
        Returns:
            spatial_feat: [C_spatial, H, W] tensor
            global_feat: [C_global] tensor
        """
        topo = fire_data['topography']
        fuel = fire_data['fuel']
        
        # Get grid size
        elev = topo.get('elevation', topo.get('slope'))
        H, W = elev.shape
        
        # Build spatial features
        spatial = np.zeros((self.spatial_channels, H, W), dtype=np.float32)
        
        # Channel 0: Elevation
        spatial[0] = self._robust_normalize(topo.get('elevation', np.zeros((H, W))))
        
        # Channel 1: Slope
        spatial[1] = self._robust_normalize(topo.get('slope', np.zeros((H, W))))
        
        # Channel 2-3: Aspect as sin/cos
        aspect = topo.get('aspect', np.zeros((H, W)))
        aspect_rad = np.deg2rad(aspect)
        spatial[2] = np.sin(aspect_rad)
        spatial[3] = np.cos(aspect_rad)
        
        # Channel 4: Fuel model
        fuel_model = fuel.get('fbfm40', fuel.get('fbfm13', np.ones((H, W))))
        spatial[4] = self._robust_normalize(fuel_model)
        
        # Channel 5: Canopy cover
        canopy = fuel.get('canopy_cover', np.zeros((H, W)))
        spatial[5] = self._robust_normalize(canopy)
        
        # Build global features (time-averaged weather)
        weather = fire_data['weather']
        global_feat = weather.mean(axis=0)  # Average over observations
        
        # Normalize weather features
        global_feat[0] = (global_feat[0] - 75) / 15  # Temperature
        global_feat[1] = (global_feat[1] - 50) / 25  # Humidity
        global_feat[2] = (global_feat[2] - 10) / 8   # Wind speed
        # Wind direction sin/cos already in [-1, 1]
        
        global_feat = np.clip(global_feat, -10, 10)
        global_feat = np.nan_to_num(global_feat, nan=0.0)
        
        return torch.tensor(spatial), torch.tensor(global_feat)
    
    def _robust_normalize(self, arr: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """Robustly normalize using median/IQR."""
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        if arr.std() < 1e-10:
            return np.zeros_like(arr)
        
        median = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        
        if iqr > 1e-10:
            normalized = (arr - median) / (iqr * 0.7413 + 1e-10)
        else:
            mad = np.median(np.abs(arr - median))
            if mad > 1e-10:
                normalized = (arr - median) / (mad * 1.4826 + 1e-10)
            else:
                normalized = arr - median
        
        return np.clip(normalized, -clip_range, clip_range).astype(np.float32)


# =============================================================================
# Metrics for Real Fire Evaluation
# =============================================================================

def compute_iou_at_dates(pred_T: np.ndarray, masks: List[Tuple[str, np.ndarray]], 
                         valid_mask: np.ndarray) -> Dict:
    """
    Compute IoU at each observation date.
    
    This is the appropriate metric for real fires where we only know
    burned/unburned status at specific satellite observation times.
    
    Args:
        pred_T: Predicted arrival times [H, W]
        masks: List of (date_str, burned_mask) tuples
        valid_mask: Valid region mask [H, W]
    
    Returns:
        Dictionary with IoU at each date and average IoU
    """
    if not masks:
        return {'mean_iou': 0.0}
    
    first_date = datetime.strptime(masks[0][0], "%Y-%m-%d")
    
    ious = []
    results = {}
    
    for date_str, gt_mask in masks:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        days = (date - first_date).days
        
        # Predicted burned at this time
        pred_burned = (pred_T <= days) & valid_mask & np.isfinite(pred_T)
        
        # Ground truth burned
        gt_burned = gt_mask & valid_mask
        
        # IoU
        intersection = (pred_burned & gt_burned).sum()
        union = (pred_burned | gt_burned).sum()
        iou = intersection / (union + 1e-6)
        
        ious.append(iou)
        results[f'iou_{date_str}'] = float(iou)
    
    results['mean_iou'] = float(np.mean(ious))
    results['std_iou'] = float(np.std(ious))
    results['n_dates'] = len(ious)
    
    return results


def compute_real_fire_metrics(pred_T: np.ndarray, true_T: np.ndarray,
                               valid_mask: np.ndarray, masks: List = None) -> Dict:
    """
    Compute metrics for real fire evaluation.
    
    Combines:
    - IoU at observation dates (primary metric)
    - Correlation on arrival times (secondary, with caveat about coarse GT)
    - Temporal ordering accuracy
    """
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    
    metrics = {
        'n_valid': int(valid.sum()),
        'n_burned': int(valid_mask.sum()),
    }
    
    if valid.sum() < 10:
        metrics['correlation'] = float('nan')
        metrics['rmse'] = float('nan')
        return metrics
    
    pred_valid = pred_T[valid]
    true_valid = true_T[valid]
    
    # Correlation
    if np.std(pred_valid) > 1e-8 and np.std(true_valid) > 1e-8:
        metrics['correlation'] = float(np.corrcoef(pred_valid, true_valid)[0, 1])
    else:
        metrics['correlation'] = float('nan')
    
    # RMSE (note: GT is coarse, so this has limited meaning)
    metrics['rmse'] = float(np.sqrt(np.mean((pred_valid - true_valid) ** 2)))
    
    # Relative RMSE
    max_time = true_valid.max()
    metrics['relative_rmse'] = metrics['rmse'] / max_time if max_time > 1e-6 else float('nan')
    
    # MAE
    metrics['mae'] = float(np.mean(np.abs(pred_valid - true_valid)))
    
    # Spearman rank correlation (temporal ordering)
    from scipy.stats import spearmanr
    try:
        spear, _ = spearmanr(pred_valid, true_valid)
        metrics['spearman'] = float(spear)
    except:
        metrics['spearman'] = float('nan')
    
    # IoU at observation dates
    if masks:
        iou_metrics = compute_iou_at_dates(pred_T, masks, valid_mask)
        metrics.update(iou_metrics)
    
    return metrics


# =============================================================================
# Real Fire Trainer
# =============================================================================

@dataclass
class RealFireConfig:
    """Configuration for real fire training."""
    # Model
    hidden_channels: int = 64
    num_cnn_layers: int = 5
    mlp_hidden: int = 128
    mlp_num_layers: int = 2
    eps: float = 0.1
    max_drift: float = 0.8
    max_eigenvalue: float = 10.0
    local_scale: float = 0.1
    
    # Training
    n_epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 20
    grad_clip: float = 1.0
    
    # Regularization
    lambda_G: float = 0.005
    lambda_B: float = 0.005
    
    # Solver
    max_iter: int = 50
    
    # Hardware
    device: str = 'cuda:0'
    
    # Output
    output_dir: str = './real_fire_results'
    seed: int = 42


class RealFireTrainer:
    """
    Trainer for real fire data using same forward/backward pipeline as simulation.
    
    Handles variable grid sizes by processing each fire at its native resolution
    with batch_size=1. The CNN is fully convolutional and works on any H×W,
    while the global MLP takes fixed-size weather features.
    """
    
    def __init__(self, config: RealFireConfig, loader: RealFireLoader):
        self.config = config
        self.loader = loader
        self.device = torch.device(config.device)
        self.adapter = RealFireCovariateAdapter()
        
        self.model = None
        self.optimizer = None
    
    def setup_model(self, n_spatial_features: int = 6):
        """
        Initialize model. 
        
        Note: No grid size needed - model is fully convolutional and handles
        any input size. Each fire is processed at its native resolution.
        """
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
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )
    
    def prepare_scenario(self, fire_id: str) -> Optional[Dict]:
        """Load and prepare a real fire scenario."""
        try:
            fire_data = self.loader.load_fire(fire_id)
            
            # Get features
            spatial_feat, global_feat = self.adapter(fire_data)
            
            # Find ignition point (centroid of first burned region)
            first_mask = fire_data['masks'][0][1]
            if first_mask.any():
                ys, xs = np.where(first_mask)
                si, sj = int(np.mean(ys)), int(np.mean(xs))
            else:
                return None
            
            H, W = fire_data['grid_size']
            
            # Create source mask (3x3 around ignition)
            source_mask = np.zeros((H, W), dtype=bool)
            source_mask[max(0, si-1):min(H, si+2), max(0, sj-1):min(W, sj+2)] = True
            
            return {
                'spatial_feat': spatial_feat,
                'global_feat': global_feat,
                'arrival_times': fire_data['arrival_times'],
                'source_mask': source_mask,
                'valid_mask': fire_data['valid_mask'],
                'masks': fire_data['masks'],
                'fire_id': fire_id,
                'grid_size': (H, W),
            }
        except Exception as e:
            print(f"Error loading {fire_id}: {e}")
            return None
    
    def train_step(self, scenario: Dict) -> Optional[Dict]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward through CNN
        spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
        global_f = scenario['global_feat'].unsqueeze(0).to(self.device)
        
        G, B, G_local, B_local = self.model(spatial, global_f)
        
        if torch.isnan(G).any() or torch.isnan(B).any():
            return None
        
        # Forward solve
        G_cpu = G.detach().cpu()
        B_cpu = B.detach().cpu()
        source = torch.from_numpy(scenario['source_mask']).unsqueeze(0)
        true_T = torch.from_numpy(scenario['arrival_times']).float()
        valid = torch.from_numpy(scenario['valid_mask'])
        
        problem = EikonalProblem(G_cpu, B_cpu, source)
        T_pred, n_iter = fast_sweeping_solve_numba(problem, max_iter=self.config.max_iter)
        
        if torch.isnan(T_pred).any():
            return None
        
        # Compute loss
        obs_mask = valid & (T_pred[0] < 1e5) & torch.isfinite(true_T)
        n_obs = obs_mask.sum().item()
        
        if n_obs < 10:
            return None
        
        residuals = T_pred[0][obs_mask] - true_T[obs_mask]
        loss_data = 0.5 * (residuals ** 2).mean()
        
        if loss_data.item() > 1e6 or torch.isnan(loss_data):
            return None
        
        # Backward pass
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
        
        # Mask gradients
        valid_tensor = torch.from_numpy(scenario['valid_mask']).float()[None, None, :, :]
        dL_dG = dL_dG * valid_tensor.expand(1, 3, -1, -1)
        dL_dB = dL_dB * valid_tensor.expand(1, 2, -1, -1)
        
        # Regularization
        if TVMetricType is not None:
            loss_reg = compute_tv_regularization(
                G, B, self.config.lambda_G, self.config.lambda_B,
                TVMetricType.FROBENIUS
            )
        else:
            loss_reg = torch.tensor(0.0)
        
        if torch.isnan(loss_reg):
            return None
        
        # Backward
        if loss_reg.requires_grad:
            loss_reg.backward(retain_graph=True)
        
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
    
    def evaluate(self, scenarios: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Evaluate model on scenarios."""
        self.model.eval()
        metrics_list = []
        
        with torch.no_grad():
            for scenario in scenarios:
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
                masks = scenario.get('masks', None)
                
                metrics = compute_real_fire_metrics(pred_T, true_T, valid_mask, masks)
                metrics['fire_id'] = scenario['fire_id']
                metrics_list.append(metrics)
        
        # Aggregate
        agg = {}
        if metrics_list:
            for key in ['correlation', 'relative_rmse', 'mae', 'mean_iou', 'spearman']:
                values = [m.get(key, float('nan')) for m in metrics_list 
                         if not np.isnan(m.get(key, float('nan')))]
                if values:
                    agg[f'{key}_mean'] = float(np.mean(values))
                    agg[f'{key}_std'] = float(np.std(values))
            agg['n_samples'] = len(metrics_list)
        
        return agg, metrics_list
    
    def train(self, train_scenarios: List[Dict], val_scenarios: List[Dict],
              verbose: bool = True) -> Dict:
        """Full training loop."""
        history = {
            'train_loss': [], 'val_correlation': [], 'val_iou': [],
            'best_epoch': 0, 'best_val_iou': 0, 'learning_rate': [],
        }
        
        best_val_iou = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            # Training
            epoch_losses = []
            np.random.shuffle(train_scenarios)
            
            for scenario in train_scenarios:
                result = self.train_step(scenario)
                if result is not None:
                    epoch_losses.append(result['loss'])
            
            if not epoch_losses:
                continue
            
            avg_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_agg, _ = self.evaluate(val_scenarios)
                val_corr = val_agg.get('correlation_mean', 0)
                val_iou = val_agg.get('mean_iou_mean', 0)
                
                history['val_correlation'].append(val_corr)
                history['val_iou'].append(val_iou)
                
                self.scheduler.step(val_iou)
                
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    history['best_epoch'] = epoch + 1
                    history['best_val_iou'] = best_val_iou
                    patience_counter = 0
                else:
                    patience_counter += 5
                
                if verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.2f}, "
                          f"val_corr={val_corr:.3f}, val_iou={val_iou:.3f}, lr={lr:.1e}")
                
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history


# =============================================================================
# Main Training Function
# =============================================================================

def run_real_fire_experiment(
    data_root: str,
    output_dir: str,
    config: RealFireConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict:
    """
    Run training experiment on real fire data.
    """
    print("=" * 70)
    print("REAL FIRE TRAINING EXPERIMENT")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading real fires from: {data_root}")
    loader = RealFireLoader(data_root)
    print(f"Found {len(loader.fires)} real fires")
    
    # Warmup numba
    if SOLVER_AVAILABLE:
        warmup_numba()
    
    # Create trainer
    trainer = RealFireTrainer(config, loader)
    
    # Get all fire IDs
    fire_ids = list(loader.fires.keys())
    np.random.seed(config.seed)
    np.random.shuffle(fire_ids)
    
    n_train = int(len(fire_ids) * train_ratio)
    n_val = int(len(fire_ids) * val_ratio)
    
    train_ids = fire_ids[:n_train]
    val_ids = fire_ids[n_train:n_train + n_val]
    test_ids = fire_ids[n_train + n_val:]
    
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Prepare scenarios
    print("\nPreparing scenarios...")
    train_scenarios = []
    for fire_id in train_ids:
        scenario = trainer.prepare_scenario(fire_id)
        if scenario is not None:
            train_scenarios.append(scenario)
    
    val_scenarios = []
    for fire_id in val_ids:
        scenario = trainer.prepare_scenario(fire_id)
        if scenario is not None:
            val_scenarios.append(scenario)
    
    test_scenarios = []
    for fire_id in test_ids:
        scenario = trainer.prepare_scenario(fire_id)
        if scenario is not None:
            test_scenarios.append(scenario)
    
    print(f"Loaded: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test")
    
    if not train_scenarios:
        print("ERROR: No training scenarios loaded!")
        return {}
    
    # Print grid size statistics
    grid_sizes = [s['grid_size'] for s in train_scenarios]
    min_h = min(h for h, w in grid_sizes)
    max_h = max(h for h, w in grid_sizes)
    min_w = min(w for h, w in grid_sizes)
    max_w = max(w for h, w in grid_sizes)
    print(f"Grid sizes range: ({min_h}-{max_h}) x ({min_w}-{max_w})")
    
    # Setup model (no grid size needed - fully convolutional)
    trainer.setup_model()
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\nTraining...")
    history = trainer.train(train_scenarios, val_scenarios, verbose=True)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_agg, test_details = trainer.evaluate(test_scenarios)
    
    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Test correlation: {test_agg.get('correlation_mean', 0):.3f} ± {test_agg.get('correlation_std', 0):.3f}")
    print(f"  Test IoU: {test_agg.get('mean_iou_mean', 0):.3f} ± {test_agg.get('mean_iou_std', 0):.3f}")
    print(f"  Test Spearman: {test_agg.get('spearman_mean', 0):.3f}")
    
    # Save results
    results = {
        'config': asdict(config),
        'n_fires': len(loader.fires),
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'history': history,
        'test_metrics': test_agg,
        'test_details': test_details,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': asdict(config),
        'grid_size_range': {
            'min': (min_h, min_w),
            'max': (max_h, max_w),
        },
    }, output_path / 'model.pt')
    
    print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train on real fire data')
    parser.add_argument('--data_root', type=str, 
                        default=None,
                        help='Path to real fire data root (required)',
                        help='Path to real_fires directory')
    parser.add_argument('--output', type=str, default='./real_fire_results',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    config = RealFireConfig(
        n_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
    )
    
    run_real_fire_experiment(args.data_root, args.output, config)


if __name__ == '__main__':
    main()
