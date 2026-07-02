"""
Band-MSE Configuration and Metrics
====================================

Configuration dataclass and evaluation metric functions for
band-decomposed MSE training on Sim2Real-Fire data.

Author: Barak Gahtan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a band-MSE experiment."""
    experiment_name: str = ""
    experiment_type: str = "per_scene"
    scenes: List[str] = None
    test_scenes: List[str] = None

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
    patience: int = 50           # early stopping (in epoch units)
    scheduler_patience: int = 3  # ReduceLROnPlateau patience (in validation units)
    grad_clip: float = 1.0

    # Regularization
    lambda_G: float = 0.001
    lambda_B: float = 0.001

    # Solver
    max_iter: int = 50
    band_max_iter: int = 20   # band solves converge faster (large source region)
    use_sparse_backward: bool = True  # skip source/unreachable pixels in adjoint

    # Band decomposition (fractions of max_time)
    band_width: float = 0.34
    band_stride: float = 0.17

    # Subsampling (None = use all fires)
    max_train: int = None
    max_eval: int = None

    # Feature group for Sim2Real covariates
    feature_group: str = "baseline"

    # Hardware
    device: str = "cuda:0"
    n_workers: int = 8

    # I/O
    output_dir: str = "./experiments"
    seed: int = 42


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_relative_rmse(pred_T, true_T, valid_mask):
    """Relative RMSE normalized by max ground-truth time."""
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    if valid.sum() < 10:
        return float('nan')
    pred_valid, true_valid = pred_T[valid], true_T[valid]
    rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
    max_time = true_valid.max()
    return rmse / max_time if max_time > 1e-6 else float('nan')


def compute_all_metrics(pred_T, true_T, valid_mask):
    """Full metric suite: RMSE, correlation, MAE, IoU at multiple thresholds."""
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    if valid.sum() < 10:
        return {'relative_rmse': float('nan'), 'rmse': float('nan'),
                'correlation': float('nan'), 'mae': float('nan'),
                'n_valid': int(valid.sum())}

    pv, tv = pred_T[valid], true_T[valid]
    rmse = np.sqrt(np.mean((pv - tv) ** 2))
    max_time = tv.max()
    rel_rmse = rmse / max_time if max_time > 1e-6 else float('nan')
    corr = (np.corrcoef(pv, tv)[0, 1]
            if np.std(pv) > 1e-8 and np.std(tv) > 1e-8
            else float('nan'))
    mae = np.mean(np.abs(pv - tv))

    m = {'relative_rmse': float(rel_rmse), 'rmse': float(rmse),
         'correlation': float(corr), 'mae': float(mae),
         'max_time_gt': float(max_time), 'n_valid': int(valid.sum())}

    for frac in [0.25, 0.5, 0.75, 1.0]:
        t_thr = max_time * frac
        inter = ((pv <= t_thr) & (tv <= t_thr)).sum()
        union = ((pv <= t_thr) | (tv <= t_thr)).sum()
        m[f'iou_{int(frac * 100)}'] = float(inter / union) if union > 0 else 0.0

    return m


def aggregate_metrics(metrics_list):
    """Aggregate per-fire metrics into mean ± std summaries."""
    if not metrics_list:
        return {}
    vm = [m for m in metrics_list
          if isinstance(m.get('correlation'), (int, float))
          and not np.isnan(m['correlation'])]
    if not vm:
        return {'n_samples': 0}
    result = {'n_samples': len(vm)}
    for key in vm[0]:
        if key in ('n_valid', 'scene_id', 'event_id'):
            continue
        vals = [m[key] for m in vm
                if isinstance(m.get(key), (int, float))
                and not np.isnan(m[key])]
        if vals:
            result[f'{key}_mean'] = float(np.mean(vals))
            result[f'{key}_std'] = float(np.std(vals))
    return result


def get_scene_info(loader):
    """Quick summary of scene grid sizes and fire counts."""
    info = {}
    for sid, si in loader.scenes.items():
        if not si['events']:
            continue
        try:
            d = loader.load_scenario(sid, si['events'][0])
            H, W = d['topography']['elevation'].shape
            info[sid] = {'n_fires': len(si['events']),
                         'grid_size': (H, W), 'path': si['path']}
        except Exception:
            pass
    return info
