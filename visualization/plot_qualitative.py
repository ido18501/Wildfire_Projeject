#!/usr/bin/env python3
"""
Visualize Qualitative Fire Prediction Examples (Improved)
==========================================================

Creates publication-quality figures showing:
- Ground truth arrival times
- Predicted arrival times  
- Error map (prediction - GT)

Improvements:
- Shared colorbar for GT and Predicted panels
- No grey background (white/transparent for invalid regions)
- No box borders around panels
- White contours for arrival time visualization

Usage:
    python visualize_predictions_improved.py \
        --results_dir /path/to/results/results-scences/scene_0017_02731 \
        --data_root /path/to/sim2real_data \
        --output ./figures \
        --n_examples 3

Author: Barak Gahtan
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import List, Tuple, Optional

# Project imports

try:
    from experiments.wildfire.sim2real_loader import Sim2RealFireLoader, extract_arrival_times
except ImportError:
    print("Warning: sim2real_fire_loader not found")


def load_predictions(results_dir: str) -> dict:
    """Load predictions and results from a scene results directory."""
    results_dir = Path(results_dir)
    
    pred_file = results_dir / 'test_predictions.npz'
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")
    
    predictions = np.load(pred_file, allow_pickle=True)
    
    results_file = results_dir / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    test_fires_file = results_dir / 'test_fires.txt'
    if test_fires_file.exists():
        with open(test_fires_file, 'r') as f:
            test_fires = [line.strip() for line in f if line.strip()]
    else:
        test_fires = []
    
    pred_T = predictions['pred_T'] if 'pred_T' in predictions else None
    true_T = predictions['true_T'] if 'true_T' in predictions else None
    valid_mask = predictions['valid_mask'] if 'valid_mask' in predictions else None
    event_ids = predictions['event_ids'] if 'event_ids' in predictions else None
    
    return {
        'pred_T': pred_T,
        'true_T': true_T,
        'valid_mask': valid_mask,
        'event_ids': event_ids,
        'results': results,
        'test_fires': test_fires,
    }


def get_bounding_box(mask: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """Get bounding box of True values in mask with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    rmin = max(0, rmin - padding)
    rmax = min(mask.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(mask.shape[1], cmax + padding + 1)
    
    return rmin, rmax, cmin, cmax


def create_multi_fire_figure(
    fires_data: List[dict],
    output_path: str,
    case_labels: Optional[List[str]] = None,
    n_contours: int = 12,
):
    """
    Create a publication-quality multi-row figure.
    
    Improvements:
    - Shared colorbar for GT and Predicted (one colorbar per row, placed after Predicted)
    - No grey background (transparent/white for invalid regions)
    - No box borders
    - White contours on arrival time panels
    
    Args:
        fires_data: List of dicts with 'gt', 'pred', 'correlation' keys
        output_path: Where to save
        case_labels: Labels like ['Best', 'Typical', 'Challenging']
        n_contours: Number of contour lines (10-15 recommended)
    """
    n_fires = len(fires_data)
    if case_labels is None:
        case_labels = [f'Example {i+1}' for i in range(n_fires)]
    
    # Figure with custom grid: GT, Pred, shared_cbar, Error, error_cbar
    fig = plt.figure(figsize=(13, 3.8 * n_fires))
    
    # Width ratios: GT, Pred, small gap for shared cbar, Error, small gap for error cbar
    gs = GridSpec(n_fires, 5, figure=fig, 
                  width_ratios=[1, 1, 0.05, 1, 0.05],
                  wspace=0.08, hspace=0.15)
    
    # Colormaps - use white for invalid regions
    cmap_arrival = plt.cm.viridis.copy()
    cmap_arrival.set_bad(color='white', alpha=0)
    
    cmap_error = plt.cm.RdBu_r.copy()
    cmap_error.set_bad(color='white', alpha=0)
    
    for row, (data, label) in enumerate(zip(fires_data, case_labels)):
        gt = data['gt']
        pred = data['pred']
        corr = data['correlation']
        
        # Mask to GT burned region only
        valid_gt = np.isfinite(gt) & (gt < 1e5)
        
        # Crop to bounding box
        rmin, rmax, cmin, cmax = get_bounding_box(valid_gt, padding=10)
        gt_crop = gt[rmin:rmax, cmin:cmax]
        pred_crop = pred[rmin:rmax, cmin:cmax]
        valid_gt_crop = valid_gt[rmin:rmax, cmin:cmax]
        
        # Mask prediction to GT region
        pred_masked_arr = pred_crop.copy()
        pred_masked_arr[~valid_gt_crop] = np.nan
        
        valid = valid_gt_crop & np.isfinite(pred_crop) & (pred_crop < 1e5)
        
        error = np.full_like(gt_crop, np.nan)
        error[valid] = pred_masked_arr[valid] - gt_crop[valid]
        
        vmax = np.percentile(gt_crop[valid_gt_crop], 99) if valid_gt_crop.any() else 100
        error_max = np.percentile(np.abs(error[valid]), 95) if valid.any() else 10
        
        gt_masked = np.ma.masked_where(~valid_gt_crop, gt_crop)
        pred_masked = np.ma.masked_where(~valid_gt_crop, pred_masked_arr)
        error_masked = np.ma.masked_where(~valid, error)
        
        # Contour levels
        contour_levels = np.linspace(0, vmax, n_contours + 2)[1:-1]  # Skip endpoints
        
        # ============ Ground Truth (col 0) ============
        ax_gt = fig.add_subplot(gs[row, 0])
        im_gt = ax_gt.imshow(gt_masked, cmap=cmap_arrival, vmin=0, vmax=vmax,
                             aspect='equal', interpolation='nearest')
        
        # White contours
        ax_gt.contour(gt_crop, levels=contour_levels, colors='white', 
                     linewidths=0.6, alpha=0.8)
        
        if row == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax_gt.set_ylabel(f'{label}\n($\\rho$={corr:.2f})', fontsize=11, fontweight='bold')
        
        # Remove borders and ticks
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_visible(False)
        
        # ============ Predicted (col 1) ============
        ax_pred = fig.add_subplot(gs[row, 1])
        im_pred = ax_pred.imshow(pred_masked, cmap=cmap_arrival, vmin=0, vmax=vmax,
                                  aspect='equal', interpolation='nearest')
        
        # White contours
        pred_for_contour = pred_masked_arr.copy()
        pred_for_contour[~valid_gt_crop] = np.nan
        ax_pred.contour(pred_for_contour, levels=contour_levels, colors='white',
                       linewidths=0.6, alpha=0.8)
        
        if row == 0:
            ax_pred.set_title('Predicted', fontsize=12, fontweight='bold')
        
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        for spine in ax_pred.spines.values():
            spine.set_visible(False)
        
        # ============ Shared colorbar for GT & Pred (col 2) ============
        ax_cbar1 = fig.add_subplot(gs[row, 2])
        cbar1 = plt.colorbar(im_pred, cax=ax_cbar1)
        cbar1.ax.tick_params(labelsize=9)
        if row == n_fires - 1:
            cbar1.set_label('Time (hours)', fontsize=10)
        
        # ============ Error (col 3) ============
        ax_err = fig.add_subplot(gs[row, 3])
        im_err = ax_err.imshow(error_masked, cmap=cmap_error, 
                               vmin=-error_max, vmax=error_max,
                               aspect='equal', interpolation='nearest')
        
        if row == 0:
            ax_err.set_title('Error (Pred - GT)', fontsize=12, fontweight='bold')
        
        ax_err.set_xticks([])
        ax_err.set_yticks([])
        for spine in ax_err.spines.values():
            spine.set_visible(False)
        
        # ============ Error colorbar (col 4) ============
        ax_cbar2 = fig.add_subplot(gs[row, 4])
        cbar2 = plt.colorbar(im_err, cax=ax_cbar2)
        cbar2.ax.tick_params(labelsize=9)
        if row == n_fires - 1:
            cbar2.set_label('Error (hours)', fontsize=10)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved multi-fire figure: {output_path}")


def create_fire_visualization(
    gt_arrival: np.ndarray,
    pred_arrival: np.ndarray,
    fire_id: str,
    correlation: float,
    output_path: str,
    vmax: Optional[float] = None,
    n_contours: int = 12,
):
    """Create a 3-panel visualization for a single fire."""
    valid_gt = np.isfinite(gt_arrival) & (gt_arrival < 1e5)
    rmin, rmax, cmin, cmax = get_bounding_box(valid_gt, padding=15)
    
    gt_crop = gt_arrival[rmin:rmax, cmin:cmax]
    pred_crop = pred_arrival[rmin:rmax, cmin:cmax]
    valid_gt_crop = valid_gt[rmin:rmax, cmin:cmax]
    
    pred_in_gt_region = pred_crop.copy()
    pred_in_gt_region[~valid_gt_crop] = np.nan
    
    error = np.full_like(gt_crop, np.nan)
    valid = valid_gt_crop & np.isfinite(pred_crop) & (pred_crop < 1e5)
    error[valid] = pred_in_gt_region[valid] - gt_crop[valid]
    
    if vmax is None:
        vmax = np.percentile(gt_crop[valid_gt_crop], 99) if valid_gt_crop.any() else 100
    
    gt_masked = np.ma.masked_where(~valid_gt_crop, gt_crop)
    pred_masked = np.ma.masked_where(~valid_gt_crop, pred_in_gt_region)
    error_masked = np.ma.masked_where(~valid, error)
    error_max = np.percentile(np.abs(error[valid]), 95) if valid.any() else 10
    
    # Figure with custom widths
    fig = plt.figure(figsize=(13, 4))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.08)
    
    cmap_arrival = plt.cm.viridis.copy()
    cmap_arrival.set_bad(color='white', alpha=0)
    cmap_error = plt.cm.RdBu_r.copy()
    cmap_error.set_bad(color='white', alpha=0)
    
    contour_levels = np.linspace(0, vmax, n_contours + 2)[1:-1]
    
    # GT
    ax_gt = fig.add_subplot(gs[0, 0])
    im_gt = ax_gt.imshow(gt_masked, cmap=cmap_arrival, vmin=0, vmax=vmax)
    ax_gt.contour(gt_crop, levels=contour_levels, colors='white', linewidths=0.6, alpha=0.8)
    ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_visible(False)
    
    # Pred
    ax_pred = fig.add_subplot(gs[0, 1])
    im_pred = ax_pred.imshow(pred_masked, cmap=cmap_arrival, vmin=0, vmax=vmax)
    ax_pred.contour(pred_in_gt_region, levels=contour_levels, colors='white', linewidths=0.6, alpha=0.8)
    ax_pred.set_title('Predicted', fontsize=12, fontweight='bold')
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_visible(False)
    
    # Shared colorbar
    ax_cbar1 = fig.add_subplot(gs[0, 2])
    cbar1 = plt.colorbar(im_pred, cax=ax_cbar1)
    cbar1.set_label('Time (hours)', fontsize=10)
    
    # Error
    ax_err = fig.add_subplot(gs[0, 3])
    im_err = ax_err.imshow(error_masked, cmap=cmap_error, vmin=-error_max, vmax=error_max)
    ax_err.set_title('Error (Pred - GT)', fontsize=12, fontweight='bold')
    ax_err.set_xticks([]); ax_err.set_yticks([])
    for spine in ax_err.spines.values():
        spine.set_visible(False)
    
    # Error colorbar
    ax_cbar2 = fig.add_subplot(gs[0, 4])
    cbar2 = plt.colorbar(im_err, cax=ax_cbar2)
    cbar2.set_label('Error (hours)', fontsize=10)
    
    fig.suptitle(f'{fire_id} (ρ={correlation:.2f})', fontsize=13, fontweight='bold')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize fire prediction examples')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to scene results directory')
    parser.add_argument('--data_root', type=str, 
                        default=None,
                        help='Path to Sim2Real-Fire data root (required)',
                        help='Path to simulation data')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures')
    parser.add_argument('--n_examples', type=int, default=3,
                        help='Number of example fires to visualize')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='Scene ID (auto-detected from results_dir if not provided)')
    parser.add_argument('--n_contours', type=int, default=12,
                        help='Number of contour lines (10-15 recommended)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predictions from: {args.results_dir}")
    data = load_predictions(args.results_dir)
    
    pred_T = data['pred_T']
    true_T = data['true_T']
    results = data['results']
    test_fires = data['test_fires']
    
    if args.scene_id:
        scene_id = args.scene_id
    else:
        scene_id = Path(args.results_dir).name.replace('scene_', '')
    
    print(f"Scene: {scene_id}")
    print(f"Number of test fires: {len(test_fires)}")
    
    # Get per-fire metrics
    if 'test_metrics_per_fire' in results:
        per_fire_metrics = results['test_metrics_per_fire']
    else:
        per_fire_metrics = []
    
    n_fires = len(test_fires)
    
    if per_fire_metrics:
        correlations = [m.get('correlation', 0) for m in per_fire_metrics]
        sorted_indices = np.argsort(correlations)[::-1]
        
        selected_indices = []
        selected_indices.append(sorted_indices[max(0, len(sorted_indices) // 10)])
        selected_indices.append(sorted_indices[len(sorted_indices) // 2])
        selected_indices.append(sorted_indices[min(len(sorted_indices) - 1, 
                                                    len(sorted_indices) - len(sorted_indices) // 10 - 1)])
        selected_indices = selected_indices[:args.n_examples]
    else:
        correlations = []
        for i in range(min(n_fires, len(pred_T))):
            pred = pred_T[i]
            true = true_T[i]
            valid = np.isfinite(pred) & np.isfinite(true) & (pred < 1e5) & (true < 1e5)
            if valid.sum() > 10:
                corr = np.corrcoef(pred[valid], true[valid])[0, 1]
            else:
                corr = 0.0
            correlations.append(corr)
        
        sorted_indices = np.argsort(correlations)[::-1]
        selected_indices = [
            sorted_indices[len(sorted_indices) // 10],
            sorted_indices[len(sorted_indices) // 2],
            sorted_indices[-len(sorted_indices) // 10 - 1],
        ][:args.n_examples]
        
        per_fire_metrics = [{'correlation': c} for c in correlations]
    
    print(f"Selected fire indices: {selected_indices}")
    
    fires_data = []
    for idx in selected_indices:
        if idx >= len(test_fires) or idx >= len(pred_T):
            continue
        
        fire_id = test_fires[idx]
        print(f"\nProcessing fire {idx}: {fire_id}")
        
        pred_arrival = pred_T[idx]
        gt_arrival = true_T[idx]
        
        if idx < len(per_fire_metrics):
            corr = per_fire_metrics[idx].get('correlation', 0)
        else:
            valid = np.isfinite(gt_arrival) & np.isfinite(pred_arrival) & (pred_arrival < 1e5)
            if valid.sum() > 10:
                corr = np.corrcoef(gt_arrival[valid], pred_arrival[valid])[0, 1]
            else:
                corr = 0.0
        
        print(f"  Correlation: {corr:.3f}")
        
        fires_data.append({
            'gt': gt_arrival,
            'pred': pred_arrival,
            'fire_id': fire_id,
            'correlation': corr,
        })
        
        create_fire_visualization(
            gt_arrival, pred_arrival, fire_id, corr,
            output_dir / f'fire_{fire_id}.png',
            n_contours=args.n_contours
        )
    
    if len(fires_data) >= 2:
        case_labels = ['Best Case', 'Typical Case', 'Challenging Case'][:len(fires_data)]
        create_multi_fire_figure(
            fires_data,
            output_dir / f'qualitative_examples_{scene_id}.pdf',
            case_labels,
            n_contours=args.n_contours
        )
    
    print(f"\nDone! Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
