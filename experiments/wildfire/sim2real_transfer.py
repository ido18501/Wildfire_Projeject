#!/usr/bin/env python3
"""
Evaluate Cross-Scene Synthetic Model on Real Fires (Sim2Real Transfer)
=======================================================================

Tests whether a model trained on FARSITE simulations can generalize to 
real satellite-observed wildfires.

Usage:
    python evaluate_sim2real_transfer.py \
        --model_path /path/to/results/cross_scene_results/best_model.pt \
        --real_fires_path /path/to/real_fires \
        --output ./sim2real_transfer_results

Author: Barak Gahtan
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Project imports

try:
    from eikonal.forward.pytorch import EikonalProblem
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba
    SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import forward solver: {e}")
    SOLVER_AVAILABLE = False

try:
    from eikonal.nn.encoder import FlexibleCovariateEncoder
except ImportError as e:
    print(f"Warning: fire_training.models not found: {e}")


# =============================================================================
# Real Fire Data Loading (adapted from real_fire_training.py)
# =============================================================================

class RealFireLoader:
    """Load real fire data from Sim2Real-Fire dataset."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.fires = self._discover_fires()
    
    def _discover_fires(self) -> Dict[str, Dict]:
        """Discover all real fires in the dataset."""
        fires = {}
        
        # Structure: real_fires/0050_00001/0050_00001/
        search_root = self.data_root
        
        print(f"  Searching in: {search_root}")
        
        for fire_dir in sorted(search_root.iterdir()):
            if not fire_dir.is_dir():
                continue
            
            # Skip non-fire directories
            if not fire_dir.name[0].isdigit():
                continue
            
            fire_id = fire_dir.name
            
            # Check for nested structure (0050_00001/0050_00001/)
            inner_dir = fire_dir / fire_id
            if inner_dir.exists() and inner_dir.is_dir():
                data_dir = inner_dir
            else:
                data_dir = fire_dir
            
            # Check for required subdirectories
            masks_dir = data_dir / 'Satellite_Images_Mask'
            topo_dir = data_dir / 'Topography_Map'
            fuel_dir = data_dir / 'Fuel_Map'
            weather_dir = data_dir / 'Weather_Data'
            veg_dir = data_dir / 'Vegetation_Map'
            
            if not masks_dir.exists():
                print(f"  {fire_id}: No Satellite_Images_Mask folder")
                continue
            
            # Masks might be in a subfolder named after the fire
            masks_subdir = masks_dir / fire_id
            if masks_subdir.exists():
                masks_dir = masks_subdir
            
            # Count observation dates (mask files) - could be .tif or .png
            # Format: 20180505.png (YYYYMMDD)
            mask_files = list(masks_dir.glob('*.tif')) + list(masks_dir.glob('*.png'))
            if len(mask_files) < 2:
                print(f"  {fire_id}: Only {len(mask_files)} mask files in {masks_dir}")
                continue
            
            fires[fire_id] = {
                'path': data_dir,
                'masks_dir': masks_dir,
                'topo_dir': topo_dir if topo_dir.exists() else None,
                'fuel_dir': fuel_dir if fuel_dir.exists() else None,
                'weather_dir': weather_dir if weather_dir.exists() else None,
                'veg_dir': veg_dir if veg_dir.exists() else None,
                'n_observations': len(mask_files),
            }
            print(f"  {fire_id}: Found {len(mask_files)} observations")
        
        return fires
    
    def load_fire(self, fire_id: str) -> Dict:
        """Load all data for a single fire."""
        from PIL import Image
        import re
        
        fire_info = self.fires[fire_id]
        data_dir = fire_info['path']
        
        # Load covariates
        covariates = {}
        
        # Topography
        if fire_info['topo_dir']:
            for f in fire_info['topo_dir'].glob('*.npy'):
                covariates[f.stem.lower()] = np.load(f)
            for f in fire_info['topo_dir'].glob('*.tif'):
                from PIL import Image
                covariates[f.stem.lower()] = np.array(Image.open(f))
        
        # Fuel
        if fire_info['fuel_dir']:
            for f in fire_info['fuel_dir'].glob('*.npy'):
                covariates[f.stem.lower()] = np.load(f)
            for f in fire_info['fuel_dir'].glob('*.tif'):
                covariates[f.stem.lower()] = np.array(Image.open(f))
        
        # Vegetation
        if fire_info['veg_dir']:
            for f in fire_info['veg_dir'].glob('*.npy'):
                covariates[f.stem.lower()] = np.load(f)
            for f in fire_info['veg_dir'].glob('*.tif'):
                covariates[f.stem.lower()] = np.array(Image.open(f))
        
        # Weather (may be CSV or JSON)
        if fire_info['weather_dir']:
            for f in fire_info['weather_dir'].glob('*.npy'):
                covariates[f.stem.lower()] = np.load(f)
            for f in fire_info['weather_dir'].glob('*.csv'):
                import pandas as pd
                df = pd.read_csv(f)
                for col in df.columns:
                    if col.lower() not in ['date', 'time', 'datetime']:
                        covariates[col.lower()] = df[col].values
        
        # Load masks with dates
        masks = []
        masks_dir = fire_info['masks_dir']
        
        for mask_file in sorted(masks_dir.glob('*')):
            if mask_file.suffix.lower() not in ['.png', '.tif', '.tiff']:
                continue
            
            # Extract date from filename
            # Format: 20180505.png (YYYYMMDD)
            name = mask_file.stem
            
            # Try YYYYMMDD format first
            if len(name) == 8 and name.isdigit():
                date_str = f"{name[:4]}-{name[4:6]}-{name[6:8]}"
            else:
                # Try to find date pattern in filename
                date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', name)
                if date_match:
                    date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                else:
                    date_str = name  # Use filename as-is
            
            try:
                mask = np.array(Image.open(mask_file))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # Take first channel
                mask = (mask > 0).astype(np.uint8)  # Any non-zero is burned
                masks.append((date_str, mask))
            except Exception as e:
                print(f"    Warning: could not load {mask_file}: {e}")
                continue
        
        # Sort by date
        masks.sort(key=lambda x: x[0])
        
        if not masks:
            return None
        
        return {
            'fire_id': fire_id,
            'covariates': covariates,
            'masks': masks,
            'grid_size': masks[0][1].shape if masks else (0, 0),
        }


class RealFireCovariateAdapter:
    """Adapt real fire covariates to model input format."""
    
    def __init__(self):
        # Normalization stats (from synthetic data or computed)
        self.spatial_stats = {
            'elevation': {'mean': 1500, 'std': 500},
            'slope': {'mean': 15, 'std': 10},
            'aspect_sin': {'mean': 0, 'std': 1},
            'aspect_cos': {'mean': 0, 'std': 1},
            'fuel_model': {'mean': 5, 'std': 3},
            'canopy_cover': {'mean': 50, 'std': 30},
        }
        
        self.global_stats = {
            'wind_speed': {'mean': 10, 'std': 5},
            'wind_dir_sin': {'mean': 0, 'std': 1},
            'wind_dir_cos': {'mean': 0, 'std': 1},
            'temperature': {'mean': 25, 'std': 10},
            'humidity': {'mean': 30, 'std': 15},
        }
    
    def __call__(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw covariates to normalized tensors."""
        covariates = data['covariates']
        H, W = data['grid_size']
        
        # Build spatial features
        spatial_list = []
        
        # Elevation
        if 'elevation' in covariates:
            elev = covariates['elevation'].astype(np.float32)
        elif 'dem' in covariates:
            elev = covariates['dem'].astype(np.float32)
        else:
            elev = np.zeros((H, W), dtype=np.float32)
        elev = (elev - self.spatial_stats['elevation']['mean']) / self.spatial_stats['elevation']['std']
        spatial_list.append(elev)
        
        # Slope
        if 'slope' in covariates:
            slope = covariates['slope'].astype(np.float32)
        else:
            slope = np.zeros((H, W), dtype=np.float32)
        slope = (slope - self.spatial_stats['slope']['mean']) / self.spatial_stats['slope']['std']
        spatial_list.append(slope)
        
        # Aspect (as sin/cos)
        if 'aspect' in covariates:
            aspect = covariates['aspect'].astype(np.float32)
            aspect_rad = np.deg2rad(aspect)
            spatial_list.append(np.sin(aspect_rad))
            spatial_list.append(np.cos(aspect_rad))
        else:
            spatial_list.append(np.zeros((H, W), dtype=np.float32))
            spatial_list.append(np.zeros((H, W), dtype=np.float32))
        
        # Fuel model
        if 'fuel_model' in covariates:
            fuel = covariates['fuel_model'].astype(np.float32)
        elif 'fuel' in covariates:
            fuel = covariates['fuel'].astype(np.float32)
        else:
            fuel = np.ones((H, W), dtype=np.float32) * 5
        fuel = (fuel - self.spatial_stats['fuel_model']['mean']) / self.spatial_stats['fuel_model']['std']
        spatial_list.append(fuel)
        
        # Canopy cover
        if 'canopy_cover' in covariates:
            canopy = covariates['canopy_cover'].astype(np.float32)
        elif 'canopy' in covariates:
            canopy = covariates['canopy'].astype(np.float32)
        else:
            canopy = np.ones((H, W), dtype=np.float32) * 50
        canopy = (canopy - self.spatial_stats['canopy_cover']['mean']) / self.spatial_stats['canopy_cover']['std']
        spatial_list.append(canopy)
        
        spatial_feat = torch.from_numpy(np.stack(spatial_list, axis=0))
        
        # Build global features
        global_list = []
        
        # Wind speed
        if 'wind_speed' in covariates:
            ws = float(np.nanmean(covariates['wind_speed']))
        else:
            ws = 10.0
        ws = (ws - self.global_stats['wind_speed']['mean']) / self.global_stats['wind_speed']['std']
        global_list.append(ws)
        
        # Wind direction (as sin/cos)
        if 'wind_direction' in covariates:
            wd = float(np.nanmean(covariates['wind_direction']))
            wd_rad = np.deg2rad(wd)
            global_list.append(np.sin(wd_rad))
            global_list.append(np.cos(wd_rad))
        else:
            global_list.append(0.0)
            global_list.append(1.0)
        
        # Temperature
        if 'temperature' in covariates:
            temp = float(np.nanmean(covariates['temperature']))
        else:
            temp = 25.0
        temp = (temp - self.global_stats['temperature']['mean']) / self.global_stats['temperature']['std']
        global_list.append(temp)
        
        # Humidity
        if 'humidity' in covariates:
            hum = float(np.nanmean(covariates['humidity']))
        elif 'relative_humidity' in covariates:
            hum = float(np.nanmean(covariates['relative_humidity']))
        else:
            hum = 30.0
        hum = (hum - self.global_stats['humidity']['mean']) / self.global_stats['humidity']['std']
        global_list.append(hum)
        
        global_feat = torch.tensor(global_list, dtype=torch.float32)
        
        return spatial_feat, global_feat


def compute_iou_at_dates(T_pred: np.ndarray, masks: List[Tuple[str, np.ndarray]]) -> Dict:
    """
    Compute IoU at each observation date.
    
    Args:
        T_pred: Predicted arrival times [H, W]
        masks: List of (date_str, mask_array) tuples
    
    Returns:
        Dict with per-date IoU and summary stats
    """
    if len(masks) < 2:
        return {'mean_iou': 0.0, 'n_dates': 0}
    
    # Parse first date as reference
    first_date = datetime.strptime(masks[0][0], "%Y-%m-%d")
    
    ious = {}
    
    for date_str, gt_mask in masks[1:]:  # Skip first (source)
        date = datetime.strptime(date_str, "%Y-%m-%d")
        threshold = float((date - first_date).days)
        
        if threshold <= 0:
            continue
        
        # Predicted burned region at this time
        pred_burned = (T_pred <= threshold) & np.isfinite(T_pred)
        gt_burned = gt_mask > 0
        
        # IoU
        intersection = (pred_burned & gt_burned).sum()
        union = (pred_burned | gt_burned).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        ious[f'iou_{date_str}'] = float(iou)
    
    if ious:
        iou_values = list(ious.values())
        ious['mean_iou'] = float(np.mean(iou_values))
        ious['std_iou'] = float(np.std(iou_values))
        ious['n_dates'] = len(iou_values)
    else:
        ious['mean_iou'] = 0.0
        ious['std_iou'] = 0.0
        ious['n_dates'] = 0
    
    return ious


def extract_arrival_times_from_masks(masks: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    """Extract arrival times from dated masks."""
    if not masks:
        return np.array([])
    
    H, W = masks[0][1].shape
    arrival_times = np.full((H, W), np.inf, dtype=np.float32)
    
    first_date = datetime.strptime(masks[0][0], "%Y-%m-%d")
    
    for date_str, mask in masks:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        days = float((date - first_date).days)
        
        newly_burned = (mask > 0) & np.isinf(arrival_times)
        arrival_times[newly_burned] = days
    
    return arrival_times


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_sim2real_transfer(
    model_path: str,
    real_fires_path: str,
    output_dir: str,
    device: str = 'cuda:0',
    max_grid_pixels: int = 2000000,
):
    """
    Evaluate synthetic-trained model on real fires.
    """
    print("=" * 70)
    print("SIM2REAL TRANSFER EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Real fires: {real_fires_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config matching cross-scene training
        config = {
            'hidden_channels': 64,
            'num_cnn_layers': 5,
            'mlp_hidden': 128,
            'mlp_num_layers': 2,
            'eps': 0.1,
            'max_drift': 0.8,
            'max_eigenvalue': 10.0,
            'local_scale': 0.1,
        }
    
    print(f"Config: {config}")
    
    # Create model
    model = FlexibleCovariateEncoder(
        n_spatial_features=6,
        n_global_features=5,
        hidden_channels=config.get('hidden_channels', 64),
        num_cnn_layers=config.get('num_cnn_layers', 5),
        mlp_hidden=config.get('mlp_hidden', 128),
        mlp_num_layers=config.get('mlp_num_layers', 2),
        eps=config.get('eps', 0.1),
        max_drift=config.get('max_drift', 0.8),
        max_eigenvalue=config.get('max_eigenvalue', 10.0),
        local_scale=config.get('local_scale', 0.1),
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Warmup numba
    if SOLVER_AVAILABLE:
        warmup_numba()
    
    # Load real fires
    print("\nLoading real fires...")
    loader = RealFireLoader(real_fires_path)
    adapter = RealFireCovariateAdapter()
    
    print(f"Found {len(loader.fires)} real fires")
    
    # Evaluate each fire
    results = []
    
    for fire_id in sorted(loader.fires.keys()):
        print(f"\nEvaluating {fire_id}...")
        
        try:
            data = loader.load_fire(fire_id)
            H, W = data['grid_size']
            
            # Skip very large fires
            if H * W > max_grid_pixels:
                print(f"  Skipping: grid too large ({H}x{W} = {H*W:,} pixels)")
                continue
            
            print(f"  Grid: {H}x{W}, Observations: {len(data['masks'])}")
            
            # Get covariates
            spatial_feat, global_feat = adapter(data)
            spatial_feat = spatial_feat.unsqueeze(0).to(device)
            global_feat = global_feat.unsqueeze(0).to(device)
            
            # Forward pass
            with torch.no_grad():
                G, B, _, _ = model(spatial_feat, global_feat)
            
            G_cpu = G.cpu()
            B_cpu = B.cpu()
            
            # Create source mask from first observation
            first_mask = data['masks'][0][1]
            source_mask = torch.from_numpy(first_mask > 0).unsqueeze(0)
            
            # Solve eikonal
            problem = EikonalProblem(G_cpu, B_cpu, source_mask)
            T_pred, n_iter = fast_sweeping_solve_numba(problem, max_iter=50)
            T_pred_np = T_pred[0].numpy()
            
            # Compute IoU at each observation date
            iou_metrics = compute_iou_at_dates(T_pred_np, data['masks'])
            
            # Compute correlation with GT arrival times
            gt_arrival = extract_arrival_times_from_masks(data['masks'])
            valid_mask = np.isfinite(gt_arrival) & np.isfinite(T_pred_np) & (T_pred_np < 1e5)
            n_valid = valid_mask.sum()
            
            if n_valid > 10:
                correlation = np.corrcoef(gt_arrival[valid_mask], T_pred_np[valid_mask])[0, 1]
                from scipy.stats import spearmanr
                spearman, _ = spearmanr(gt_arrival[valid_mask], T_pred_np[valid_mask])
                
                residuals = T_pred_np[valid_mask] - gt_arrival[valid_mask]
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
                
                gt_range = gt_arrival[valid_mask].max() - gt_arrival[valid_mask].min()
                rel_rmse = rmse / (gt_range + 1e-6)
            else:
                correlation = 0.0
                spearman = 0.0
                rmse = 0.0
                mae = 0.0
                rel_rmse = 0.0
            
            fire_result = {
                'fire_id': fire_id,
                'grid_size': (H, W),
                'n_pixels': H * W,
                'n_valid': int(n_valid),
                'n_observations': len(data['masks']),
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'spearman': float(spearman) if not np.isnan(spearman) else 0.0,
                'rmse': float(rmse),
                'mae': float(mae),
                'rel_rmse': float(rel_rmse),
                **iou_metrics,
            }
            
            results.append(fire_result)
            
            print(f"  Correlation: {fire_result['correlation']:.3f}, "
                  f"Mean IoU: {fire_result['mean_iou']:.3f}, "
                  f"Spearman: {fire_result['spearman']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        correlations = [r['correlation'] for r in results]
        ious = [r['mean_iou'] for r in results]
        spearmans = [r['spearman'] for r in results]
        
        summary = {
            'n_fires': len(results),
            'correlation_mean': float(np.mean(correlations)),
            'correlation_std': float(np.std(correlations)),
            'mean_iou_mean': float(np.mean(ious)),
            'mean_iou_std': float(np.std(ious)),
            'spearman_mean': float(np.mean(spearmans)),
            'spearman_std': float(np.std(spearmans)),
        }
        
        print(f"Evaluated {len(results)} fires")
        print(f"Correlation: {summary['correlation_mean']:.3f} ± {summary['correlation_std']:.3f}")
        print(f"Mean IoU: {summary['mean_iou_mean']:.3f} ± {summary['mean_iou_std']:.3f}")
        print(f"Spearman: {summary['spearman_mean']:.3f} ± {summary['spearman_std']:.3f}")
        
        # Compare to real-fire-trained model
        print("\n--- Comparison to Real-Fire-Trained Model ---")
        print("Real-trained: Corr=0.503±0.240, IoU=0.159±0.131, Spearman=0.517±0.261")
        print(f"Sim-trained:  Corr={summary['correlation_mean']:.3f}±{summary['correlation_std']:.3f}, "
              f"IoU={summary['mean_iou_mean']:.3f}±{summary['mean_iou_std']:.3f}, "
              f"Spearman={summary['spearman_mean']:.3f}±{summary['spearman_std']:.3f}")
    else:
        summary = {'n_fires': 0}
    
    # Save results
    output_data = {
        'model_path': model_path,
        'real_fires_path': real_fires_path,
        'config': config,
        'summary': summary,
        'per_fire_results': results,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path / 'sim2real_transfer_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path / 'sim2real_transfer_results.json'}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sim2Real transfer')
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='Path to trained model checkpoint (required)',
                        help='Path to trained model checkpoint')
    parser.add_argument('--real_fires_path', type=str,
                        default=None,
                        help='Path to real fire data root (required)',
                        help='Path to real fires data')
    parser.add_argument('--output', type=str, default='./sim2real_transfer_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--max_pixels', type=int, default=2000000,
                        help='Maximum grid pixels (skip larger fires)')
    
    args = parser.parse_args()
    
    evaluate_sim2real_transfer(
        args.model_path,
        args.real_fires_path,
        args.output,
        args.device,
        args.max_pixels,
    )


if __name__ == '__main__':
    main()