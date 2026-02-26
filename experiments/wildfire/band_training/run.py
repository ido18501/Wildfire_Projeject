#!/usr/bin/env python3
"""
Band-MSE Experiment Runner
===========================

Entry point for band-decomposed MSE wildfire training experiments.
Supports per-scene (parallel GPU) and multi-scene modes.

Usage:
    # Coarse bands (K≈5 train) on 2 scenes, 2 GPUs
    python run_band_mse.py --mode per_scene \
        --scenes 0023_00995,0024_02655 --gpus 0,2 \
        --epochs 100 --band_width 0.34 --band_stride 0.17

    # Fine bands (K≈19 train) on 2 scenes, 2 GPUs
    python run_band_mse.py --mode per_scene \
        --scenes 0023_00995,0024_02655 --gpus 1,3 \
        --epochs 100 --band_width 0.1 --band_stride 0.05

    # Subsample for quick testing
    python run_band_mse.py --mode per_scene \
        --scenes 0023_00995 --gpus 0 --epochs 30 \
        --max_train 50 --max_eval 20

    # Enable CUDA single-fire solver (auto-probed for segfault safety)
    python run_band_mse.py --mode per_scene \
        --scenes 0023_00995 --gpus 0 --epochs 100 --try-cuda

Author: Barak Gahtan
"""

import os
import sys
import json
import time
import argparse
import subprocess
import traceback
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


from eikonal.forward.numba_cpu import warmup as warmup_numba
from experiments.wildfire.sim2real_loader import Sim2RealFireLoader

from band_mse_config import ExperimentConfig
from band_mse_trainer import (
    SceneTrainer, HAVE_BATCHED_CUDA, HAVE_CUDA_SINGLE, HAVE_NUMBA,
    warmup_parallel_bands, enable_cuda_single,
    _HAVE_SPARSE_ADJOINT, warmup_sparse_adjoint)


# =============================================================================
# CUDA single-fire probe (segfault-safe via subprocess)
# =============================================================================

def _probe_cuda_single():
    """Test single-fire CUDA optimized solver in a subprocess.

    The Numba CUDA JIT compiler can segfault on certain setups during
    LLVM-to-PTX compilation.  Running the warmup in a child process
    isolates the crash — if the child segfaults, we fall back to CPU.
    """
    code = (
        "import sys; "
        "from eikonal.forward.cuda_batched import warmup_v3; "
        "warmup_v3(); "
        "print('CUDA_SINGLE_OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=120
        )
        return 'CUDA_SINGLE_OK' in result.stdout
    except Exception:
        return False


# =============================================================================
# Per-Scene Experiment
# =============================================================================

def run_single_scene_experiment(scene_id, config, data_root, output_dir,
                                scene_prefix=""):
    """Train and evaluate on a single scene."""
    print(f"\n{scene_prefix}{'=' * 60}")
    print(f"{scene_prefix}Scene: {scene_id} [Band-MSE: w={config.band_width}, "
          f"s={config.band_stride}]")
    print(f"{scene_prefix}{'=' * 60}")

    t0 = time.time()
    loader = Sim2RealFireLoader(data_root)
    warmup_numba()
    warmup_parallel_bands()
    warmup_sparse_adjoint()

    si = loader.scenes.get(scene_id)
    if si is None:
        return {'error': f'Scene {scene_id} not found'}
    events = si['events']
    print(f"{scene_prefix}Total fires: {len(events)}")

    d = loader.load_scenario(scene_id, events[0])
    H, W = d['topography']['elevation'].shape
    print(f"{scene_prefix}Grid: {H} x {W}")

    # Split into train / val / test
    np.random.seed(config.seed)
    np.random.shuffle(events)
    nt = int(len(events) * config.train_ratio)
    nv = int(len(events) * config.val_ratio)
    tr_ev = events[:nt]
    va_ev = events[nt:nt + nv]
    te_ev = events[nt + nv:]

    # Subsample if requested
    if config.max_train and len(tr_ev) > config.max_train:
        tr_ev = tr_ev[:config.max_train]
    if config.max_eval:
        if len(va_ev) > config.max_eval:
            va_ev = va_ev[:config.max_eval]
        if len(te_ev) > config.max_eval:
            te_ev = te_ev[:config.max_eval]

    print(f"{scene_prefix}Split: {len(tr_ev)}/{len(va_ev)}/{len(te_ev)}")

    trainer = SceneTrainer(config, loader)
    trainer.setup_model(H, W)

    tr = [s for s in (trainer.prepare_scenario(scene_id, e)
                      for e in tr_ev) if s]
    va = [s for s in (trainer.prepare_scenario(scene_id, e)
                      for e in va_ev) if s]
    te = [s for s in (trainer.prepare_scenario(scene_id, e)
                      for e in te_ev) if s]
    print(f"{scene_prefix}Loaded: {len(tr)}/{len(va)}/{len(te)}")

    # Show band counts at typical fire duration
    sb = trainer.generate_bands(72.0, True)
    eb = trainer.generate_bands(72.0, False)
    print(f"{scene_prefix}Bands: ~{len(sb)} train (overlapping), "
          f"~{len(eb)} eval (non-overlap)")

    fwd_label = ('CUDA single-fire' if HAVE_CUDA_SINGLE
                 else 'Numba prange' if HAVE_NUMBA
                 else 'Numba sequential')
    print(f"{scene_prefix}Forward: {fwd_label}  "
          f"(band_max_iter={config.band_max_iter})")
    print(f"{scene_prefix}Backward: "
          f"{'Numba prange' if HAVE_NUMBA else 'PyTorch (slow)'}"
          f"{' + sparse adjoint' if _HAVE_SPARSE_ADJOINT else ''}")

    # Train
    print(f"{scene_prefix}\nTraining...")
    history = trainer.train(tr, va, verbose=True, scene_prefix=scene_prefix)

    # Evaluate
    print(f"{scene_prefix}\nEvaluating...")
    tb_agg, tb_det = trainer.evaluate_bands(te)
    tf_agg, tf_det, preds = trainer.evaluate_full_fire(
        te, save_predictions=True)

    elapsed = time.time() - t0

    # Save results
    results = {
        'scene_id': scene_id,
        'grid_size': [H, W],
        'n_fires': len(events),
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'train_events': tr_ev, 'val_events': va_ev, 'test_events': te_ev,
        'history': history,
        'test_metrics_band': tb_agg,
        'test_metrics_full_fire': tf_agg,
        'test_details_band': tb_det,
        'test_details_full_fire': tf_det,
        'elapsed_seconds': elapsed,
        'config': asdict(config),
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'band_mse_batched',
            'have_batched_cuda': HAVE_BATCHED_CUDA,
            'have_cuda_single': HAVE_CUDA_SINGLE,
            'have_numba': HAVE_NUMBA,
            'device': config.device,
        }
    }

    od = output_dir / f'scene_{scene_id}'
    od.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': asdict(config),
        'grid_size': (H, W),
        'best_epoch': history.get('best_epoch', 0),
    }, od / 'model.pt')
    if preds:
        np.savez_compressed(od / 'test_predictions.npz', **preds)
    with open(od / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    for name, evts in [('train', tr_ev), ('val', va_ev), ('test', te_ev)]:
        with open(od / f'{name}_fires.txt', 'w') as f:
            f.write('\n'.join(evts))

    print(f"\n{scene_prefix}{'=' * 60}")
    print(f"{scene_prefix}Scene {scene_id} Results:")
    print(f"{scene_prefix}  [Band]      IoU: "
          f"{tb_agg.get('mean_band_iou_mean', 0):.3f} ± "
          f"{tb_agg.get('mean_band_iou_std', 0):.3f}")
    print(f"{scene_prefix}  [Full-fire] corr: "
          f"{tf_agg.get('correlation_mean', 0):.3f}, "
          f"IoU@50: {tf_agg.get('iou_50_mean', 0):.3f}, "
          f"relRMSE: {tf_agg.get('relative_rmse_mean', 0):.3f}")
    print(f"{scene_prefix}  Time: {elapsed:.1f}s  Saved: {od}")
    return results


# =============================================================================
# Parallel Execution
# =============================================================================

def _run_scene_worker(args):
    """Worker function for ProcessPoolExecutor (one scene per GPU)."""
    (sid, cd, dr, od, gpu, idx, total, tpw) = args
    os.environ.update({
        'OMP_NUM_THREADS': str(tpw),
        'MKL_NUM_THREADS': str(tpw),
        'NUMBA_NUM_THREADS': str(tpw),
        'OPENBLAS_NUM_THREADS': str(tpw),
    })
    torch.set_num_threads(tpw)
    c = ExperimentConfig(**cd)
    c.device = f'cuda:{gpu}'
    torch.cuda.set_device(gpu)
    pfx = f"[GPU{gpu}|{sid}] "
    print(f"\n{pfx}Starting scene {idx + 1}/{total}")
    try:
        r = run_single_scene_experiment(
            sid, c, dr, Path(od) / 'per_scene', scene_prefix=pfx)
        return sid, r, None
    except Exception as e:
        print(f"{pfx}ERROR: {e}")
        traceback.print_exc()
        return sid, {'error': str(e)}, traceback.format_exc()


def run_per_scene_experiments_parallel(
    data_root, output_dir, config, scenes=None,
    gpu_ids=None, total_cores=47,
):
    """Run per-scene experiments in parallel across GPUs."""
    if gpu_ids is None:
        gpu_ids = [0]
    loader = Sim2RealFireLoader(data_root)
    all_scenes = [s for s in loader.scenes
                  if scenes is None or s in scenes]
    ng = len(gpu_ids)
    tpw = total_cores // ng

    print(f"Band-MSE on {len(all_scenes)} scenes, {ng} GPUs, "
          f"band w={config.band_width} s={config.band_stride}")

    if len(all_scenes) <= 1 or ng <= 1:
        return run_per_scene_experiments_sequential(
            data_root, output_dir, config, scenes,
            gpu_ids[0] if gpu_ids else 0)

    cd = asdict(config)
    items = [(s, cd, data_root, str(output_dir),
              gpu_ids[i % ng], i, len(all_scenes), tpw)
             for i, s in enumerate(all_scenes)]

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=ng) as ex:
        futs = {ex.submit(_run_scene_worker, it): it[0] for it in items}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                sid, r, err = f.result()
                results[sid] = r
                if err is None:
                    bi = r.get('test_metrics_band', {}).get(
                        'mean_band_iou_mean', 0)
                    c = r.get('test_metrics_full_fire', {}).get(
                        'correlation_mean', 0)
                    print(f"✓ [{done}/{len(all_scenes)}] {sid}: "
                          f"biou={bi:.3f} corr={c:.3f}")
                else:
                    print(f"✗ [{done}/{len(all_scenes)}] {sid} FAILED")
            except Exception as e:
                sid = futs[f]
                results[sid] = {'error': str(e)}

    tt = time.time() - t0
    _save_summary(output_dir, results, config, tt)
    return results


def run_per_scene_experiments_sequential(
    data_root, output_dir, config, scenes=None, gpu_id=0,
):
    """Run per-scene experiments sequentially on one GPU."""
    loader = Sim2RealFireLoader(data_root)
    all_scenes = [s for s in loader.scenes
                  if scenes is None or s in scenes]
    results = {}
    for i, sid in enumerate(all_scenes):
        print(f"\n[{i + 1}/{len(all_scenes)}] {sid}")
        c = ExperimentConfig(**asdict(config))
        c.device = f'cuda:{gpu_id}'
        try:
            results[sid] = run_single_scene_experiment(
                sid, c, data_root, output_dir / 'per_scene')
        except Exception as e:
            traceback.print_exc()
            results[sid] = {'error': str(e)}
    return results


def _save_summary(output_dir, results, config, total_time):
    """Save experiment summary to JSON."""
    summary = {
        'experiment_type': 'band_mse_parallel',
        'band_width': config.band_width,
        'band_stride': config.band_stride,
        'total_time': total_time,
        'results_summary': {},
    }
    bis, cs = [], []
    for s, r in sorted(results.items()):
        if 'error' not in r:
            bi = r.get('test_metrics_band', {}).get('mean_band_iou_mean', 0)
            c = r.get('test_metrics_full_fire', {}).get('correlation_mean', 0)
            summary['results_summary'][s] = {
                'band_iou': bi, 'corr': c,
                'iou50': r.get('test_metrics_full_fire', {}).get(
                    'iou_50_mean', 0)}
            bis.append(bi)
            cs.append(c)

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}\nBAND-MSE SUMMARY\n{'=' * 70}")
    for s, r in sorted(results.items()):
        if 'error' not in r:
            bi = r.get('test_metrics_band', {}).get('mean_band_iou_mean', 0)
            c = r.get('test_metrics_full_fire', {}).get('correlation_mean', 0)
            print(f"  {s}: biou={bi:.3f} corr={c:.3f}")
        else:
            print(f"  {s}: FAILED - {r['error']}")
    if cs:
        print(f"  Avg: biou={np.mean(bis):.3f} corr={np.mean(cs):.3f} "
              f"time={total_time:.0f}s")


# =============================================================================
# Multi-Scene (cross-scene generalization)
# =============================================================================

def pad_to_size(arr, tH, tW, pad_value=0):
    """Pad 2D or 3D array to target size."""
    if arr.ndim == 2:
        p = np.full((tH, tW), pad_value, dtype=arr.dtype)
        p[:arr.shape[0], :arr.shape[1]] = arr
        return p
    elif arr.ndim == 3:
        p = np.full((arr.shape[0], tH, tW), pad_value, dtype=arr.dtype)
        p[:, :arr.shape[1], :arr.shape[2]] = arr
        return p
    raise ValueError(f"Unsupported ndim: {arr.ndim}")


def run_multi_scene_experiment(train_scenes, test_scenes, data_root,
                               output_dir, config):
    """Train on multiple scenes, evaluate on held-out scenes."""
    print(f"\nMULTI-SCENE BAND-MSE: {train_scenes} -> {test_scenes}")
    loader = Sim2RealFireLoader(data_root)
    warmup_numba()
    warmup_parallel_bands()
    warmup_sparse_adjoint()

    # Find maximum grid size for padding
    mH = mW = 0
    sg = {}
    for sid in train_scenes + test_scenes:
        if sid not in loader.scenes:
            continue
        d = loader.load_scenario(sid, loader.scenes[sid]['events'][0])
        H, W = d['topography']['elevation'].shape
        sg[sid] = (H, W)
        mH, mW = max(mH, H), max(mW, W)

    trainer = SceneTrainer(config, loader)
    trainer.setup_model(mH, mW)

    def _load(slist, lim, f0, f1):
        out = []
        for sid in slist:
            if sid not in loader.scenes:
                continue
            evts = list(loader.scenes[sid]['events'])
            np.random.seed(config.seed)
            np.random.shuffle(evts)
            n = len(evts)
            sel = evts[int(n * f0):int(n * f1)][:lim]
            for eid in sel:
                s = trainer.prepare_scenario(sid, eid)
                if s:
                    s['spatial_feat'] = torch.from_numpy(
                        pad_to_size(s['spatial_feat'].numpy(), mH, mW))
                    s['arrival_times'] = pad_to_size(
                        s['arrival_times'], mH, mW, np.inf)
                    s['source_mask'] = pad_to_size(
                        s['source_mask'].astype(np.float32),
                        mH, mW).astype(bool)
                    s['valid_mask'] = pad_to_size(
                        s['valid_mask'].astype(np.float32),
                        mH, mW).astype(bool)
                    out.append(s)
        return out

    tr = _load(train_scenes, 200, 0, 0.7)
    va = _load(train_scenes, 50, 0.7, 0.85)
    te = _load(test_scenes, 100, 0, 1)
    print(f"Loaded: {len(tr)}/{len(va)}/{len(te)}")

    history = trainer.train(tr, va, verbose=True)
    tb, _ = trainer.evaluate_bands(te)
    tf, _, preds = trainer.evaluate_full_fire(te, save_predictions=True)

    od = output_dir / 'multi_scene' / f'{len(train_scenes)}_scenes_band_mse'
    od.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': trainer.model.state_dict(),
                'config': asdict(config)}, od / 'model.pt')
    if preds:
        np.savez_compressed(od / 'test_predictions.npz', **preds)

    results = {'history': history, 'test_band': tb, 'test_full': tf}
    with open(od / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results: band_iou={tb.get('mean_band_iou_mean', 0):.3f} "
          f"corr={tf.get('correlation_mean', 0):.3f}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    p = argparse.ArgumentParser(description='Band-MSE Experiment Suite')
    p.add_argument('--mode', default='per_scene',
                   choices=['per_scene', 'multi_scene', 'all'])
    p.add_argument('--data_root',
                   default=None,
                        help='Path to Sim2Real-Fire data root (required)'
                           'simulation_data_all/data')
    p.add_argument('--output', default='./experiments_band_mse')
    p.add_argument('--scenes', default=None,
                   help='Comma-separated scene IDs (None = all)')
    p.add_argument('--n_train_scenes', type=int, default=5,
                   help='For multi_scene: number of training scenes')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=50,
                   help='Early stopping patience in epoch units (default 50)')
    p.add_argument('--scheduler_patience', type=int, default=3,
                   help='ReduceLROnPlateau patience in validation units (default 3)')
    p.add_argument('--lambda_G', type=float, default=0.001,
                   help='TV regularization weight for metric G (default 0.001)')
    p.add_argument('--lambda_B', type=float, default=0.001,
                   help='TV regularization weight for drift B (default 0.001)')
    p.add_argument('--hidden_channels', type=int, default=128,
                   help='CNN hidden channel width (default 128)')
    p.add_argument('--num_cnn_layers', type=int, default=8,
                   help='Number of CNN layers (default 8)')
    p.add_argument('--mlp_hidden', type=int, default=256,
                   help='Global MLP hidden width (default 256)')
    p.add_argument('--mlp_num_layers', type=int, default=3,
                   help='Number of global MLP layers (default 3)')
    p.add_argument('--band_width', type=float, default=0.34,
                   help='Band width as fraction of max_time')
    p.add_argument('--band_stride', type=float, default=0.17,
                   help='Band stride as fraction of max_time')
    p.add_argument('--band_max_iter', type=int, default=20,
                   help='Max sweeping iterations for band solves (default 20)')
    p.add_argument('--gpus', default='0',
                   help='Comma-separated GPU IDs')
    p.add_argument('--cores', type=int, default=47,
                   help='Total CPU cores to divide across workers')
    p.add_argument('--sequential', action='store_true',
                   help='Force sequential execution (no multiprocessing)')
    p.add_argument('--try-cuda', action='store_true',
                   help='Probe and enable CUDA single-fire solver (may '
                        'segfault on some setups; tested via subprocess)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_train', type=int, default=None,
                   help='Max training fires per scene (subsample)')
    p.add_argument('--max_eval', type=int, default=None,
                   help='Max val/test fires per scene (subsample)')
    a = p.parse_args()

    gpus = [int(g) for g in a.gpus.split(',')]
    config = ExperimentConfig(
        n_epochs=a.epochs,
        learning_rate=a.lr,
        patience=a.patience,
        scheduler_patience=a.scheduler_patience,
        lambda_G=a.lambda_G,
        lambda_B=a.lambda_B,
        hidden_channels=a.hidden_channels,
        num_cnn_layers=a.num_cnn_layers,
        mlp_hidden=a.mlp_hidden,
        mlp_num_layers=a.mlp_num_layers,
        band_width=a.band_width,
        band_stride=a.band_stride,
        band_max_iter=a.band_max_iter,
        max_train=a.max_train,
        max_eval=a.max_eval,
        seed=a.seed,
        device=f'cuda:{gpus[0]}',
    )

    od = Path(a.output)
    od.mkdir(parents=True, exist_ok=True)
    scenes = ([s.strip() for s in a.scenes.split(',')]
              if a.scenes else None)

    # Probe CUDA single-fire solver if requested
    if a.try_cuda:
        print("Probing CUDA single-fire solver (subprocess)...")
        if _probe_cuda_single():
            enable_cuda_single()
            print("  CUDA single-fire solver: ENABLED")
        else:
            print("  CUDA single-fire solver: FAILED (segfault), using CPU")

    # Re-import the flag after potential enable_cuda_single() call
    from band_mse_trainer import HAVE_CUDA_SINGLE as _hcs

    fwd_label = ('CUDA single-fire' if _hcs
                 else 'Numba prange' if HAVE_NUMBA
                 else 'Numba sequential')

    print("=" * 70)
    print("BAND-MSE EXPERIMENT SUITE")
    print(f"Model: CNN {a.num_cnn_layers}L×{a.hidden_channels}ch  "
          f"MLP {a.mlp_num_layers}L×{a.mlp_hidden}h")
    print(f"Band: w={a.band_width} s={a.band_stride}  "
          f"band_max_iter={a.band_max_iter}")
    print(f"Reg: lambda_G={a.lambda_G} lambda_B={a.lambda_B}  "
          f"LR: {a.lr} sched_pat={a.scheduler_patience} "
          f"ES_pat={a.patience}")
    print(f"GPUs: {gpus}  Forward: {fwd_label}  Numba: {HAVE_NUMBA}  "
          f"Sparse adjoint: {_HAVE_SPARSE_ADJOINT}")
    print("=" * 70)

    if a.mode in ('per_scene', 'all'):
        if len(gpus) > 1 and not a.sequential:
            run_per_scene_experiments_parallel(
                a.data_root, od, config, scenes, gpus, a.cores)
        else:
            run_per_scene_experiments_sequential(
                a.data_root, od, config, scenes, gpus[0])

    if a.mode in ('multi_scene', 'all'):
        loader = Sim2RealFireLoader(a.data_root)
        asc = sorted(loader.scenes.keys())
        nt = a.n_train_scenes
        if nt < len(asc):
            np.random.seed(a.seed)
            np.random.shuffle(asc)
            run_multi_scene_experiment(
                asc[:nt], asc[nt:], a.data_root, od, config)

    print(f"\nDone. Results: {od}")


if __name__ == '__main__':
    main()
