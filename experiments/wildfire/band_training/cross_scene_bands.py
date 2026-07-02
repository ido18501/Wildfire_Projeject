#!/usr/bin/env python3
"""
Patch-based cross-scene band training.

Comparable extension of paper-style cross-scene training:
- same train/test scenes
- same 256x256 patch extraction
- same stride/min burned filtering
- training patches are converted into teacher-forced temporal band samples
- validation/test are full native-resolution fires for primary comparable metrics
- supports arbitrary n_time_bands >= 1
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from experiments.wildfire.sim2real_loader import Sim2RealFireLoader, CovariateAdapter
from experiments.wildfire.cross_scene import (
    load_scene_patches_parallel,
    load_full_scene_scenario,
    OptimizedTrainer,
)
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.config import TVMetricType


def make_band_samples_from_patch(
    patch: Dict,
    n_time_bands: int,
    min_band_pixels: int = 10,
) -> List[Dict]:
    """Convert one full-fire patch into non-overlapping teacher-forced band samples."""
    if n_time_bands <= 0:
        raise ValueError("n_time_bands must be positive")

    arrival = patch["arrival_times"]
    finite = np.isfinite(arrival)

    if finite.sum() < min_band_pixels:
        return []

    max_time = float(np.nanmax(arrival[finite]))
    if not np.isfinite(max_time) or max_time <= 1e-6:
        return []

    samples = []
    for k in range(n_time_bands):
        t_start = (k / n_time_bands) * max_time
        t_end = ((k + 1) / n_time_bands) * max_time

        source_mask = finite & (arrival <= t_start + 1e-6)
        if k == 0:
            # Prefer the patch's ignition/component source when available.
            source_mask = np.asarray(patch["source_mask"]).astype(bool)
            if not source_mask.any():
                source_mask = finite & (arrival <= t_start + 1e-6)

        target_mask = finite & (arrival > t_start + 1e-6) & (arrival <= t_end + 1e-6)

        if source_mask.sum() == 0 or target_mask.sum() < min_band_pixels:
            continue

        band_arrival = np.full_like(arrival, np.inf, dtype=np.float32)
        band_arrival[source_mask] = 0.0
        band_arrival[target_mask] = (arrival[target_mask] - t_start).astype(np.float32)

        valid_mask = target_mask.copy()

        samples.append({
            "spatial_feat": patch["spatial_feat"],
            "global_feat": patch["global_feat"],
            "arrival_times": band_arrival,
            "source_mask": source_mask,
            "valid_mask": valid_mask,
            "scene_id": patch.get("scene_id", ""),
            "event_id": patch.get("event_id", ""),
            "patch_id": patch.get("patch_id", 0),
            "band_index": k,
            "n_time_bands": n_time_bands,
            "t_start": t_start,
            "t_end": t_end,
        })

    return samples


def to_jsonable(v):
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (int, float)):
        return v if np.isfinite(v) else None
    if isinstance(v, str) or v is None:
        return v
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [to_jsonable(x) for x in v]
    return str(v)


def run_experiment(
    data_root: str,
    output_dir: Path,
    train_scenes: List[str],
    test_scenes: List[str],
    n_time_bands: int,
    feature_group: str = "topo_weather_omitted",
    patch_size: int = 256,
    patch_stride: int = 128,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_G: float = 0.005,
    lambda_B: float = 0.005,
    num_workers: int = 8,
    n_gpus: int = 4,
    n_solver_jobs: int = 16,
    seed: int = 42,
    train_split: float = 0.70,
    val_split: float = 0.15,
    hidden_channels: int = 64,
    num_cnn_layers: int = 5,
    mlp_hidden: int = 128,
    mlp_num_layers: int = 2,
    max_fires_per_scene: Optional[int] = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATCH-BASED CROSS-SCENE BAND EXPERIMENT")
    print("=" * 70)
    print(f"Train scenes: {train_scenes}")
    print(f"Test scenes: {test_scenes}")
    print(f"Feature group: {feature_group}")
    print(f"n_time_bands: {n_time_bands}")
    print(f"Patch size: {patch_size}, stride: {patch_stride}")
    print(f"Batch size: {batch_size}")
    print(f"Model: CNN {num_cnn_layers}L x {hidden_channels}, MLP {mlp_num_layers}L x {mlp_hidden}")
    print(f"Reg: lambda_G={lambda_G}, lambda_B={lambda_B}, lr={lr}")
    if max_fires_per_scene:
        print(f"LIMITED smoke mode: max_fires_per_scene={max_fires_per_scene}")
    print("=" * 70)

    loader = Sim2RealFireLoader(data_root)
    adapter = CovariateAdapter(feature_group=feature_group)

    # Training patches -> band samples
    print("\nExtracting training patches and converting to band samples...")
    train_band_samples = []
    n_full_patches = 0

    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            print(f"  Warning: missing scene {scene_id}")
            continue

        events = loader.scenes[scene_id]["events"].copy()
        np.random.shuffle(events)
        n_train = int(len(events) * train_split)
        train_events = events[:n_train]
        if max_fires_per_scene:
            train_events = train_events[:max_fires_per_scene]

        t0 = time.perf_counter()
        patches = load_scene_patches_parallel(
            loader, adapter, scene_id, train_events,
            patch_size, patch_stride, n_jobs=max(1, num_workers)
        )
        n_full_patches += len(patches)

        scene_band_count = 0
        for p in patches:
            bands = make_band_samples_from_patch(p, n_time_bands=n_time_bands)
            train_band_samples.extend(bands)
            scene_band_count += len(bands)

        dt = time.perf_counter() - t0
        print(f"  {scene_id}: {len(patches)} patches -> {scene_band_count} band samples from {len(train_events)} fires ({dt:.1f}s)")

    print(f"\nTotal full patches: {n_full_patches}")
    print(f"Total train band samples: {len(train_band_samples)}")
    if not train_band_samples:
        raise RuntimeError("No train band samples were created.")

    # Full-scene validation scenarios from train scenes
    print("\nLoading full-scene validation scenarios...")
    val_scenarios = []
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            continue
        events = loader.scenes[scene_id]["events"].copy()
        np.random.seed(seed)
        np.random.shuffle(events)
        n_train = int(len(events) * train_split)
        n_val = int(len(events) * val_split)
        val_events = events[n_train:n_train + n_val]
        if max_fires_per_scene:
            val_events = val_events[:max_fires_per_scene]
        for event_id in val_events:
            s = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if s is not None:
                val_scenarios.append(s)
    print(f"Validation full-fire scenarios: {len(val_scenarios)}")

    # Full-scene held-out test scenarios
    print("\nLoading full-scene test scenarios...")
    test_scenarios = []
    for scene_id in test_scenes:
        if scene_id not in loader.scenes:
            print(f"  Warning: missing scene {scene_id}")
            continue
        events = loader.scenes[scene_id]["events"]
        if max_fires_per_scene:
            events = events[:max_fires_per_scene]
        for event_id in events:
            s = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if s is not None:
                test_scenarios.append(s)
        print(f"  {scene_id}: {len([s for s in test_scenarios if s['scene_id'] == scene_id])} scenarios")
    print(f"Total test full-fire scenarios: {len(test_scenarios)}")

    n_spatial = train_band_samples[0]["spatial_feat"].shape[0]
    n_global = train_band_samples[0]["global_feat"].shape[0]

    model = FlexibleCovariateEncoder(
        n_spatial_features=n_spatial,
        n_global_features=n_global,
        hidden_channels=hidden_channels,
        num_cnn_layers=num_cnn_layers,
        mlp_hidden=mlp_hidden,
        mlp_num_layers=mlp_num_layers,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = OptimizedTrainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=lr,
        lambda_G=lambda_G,
        lambda_B=lambda_B,
        batch_size=batch_size,
        num_workers=num_workers,
        n_gpus=n_gpus,
        n_solver_jobs=n_solver_jobs,
        metric_type=TVMetricType.LOG_EUCLIDEAN,
    )

    save_path = output_dir / "best_model.pt"

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    history = trainer.train(
        train_band_samples,
        val_scenarios,
        n_epochs=n_epochs,
        early_stop_patience=20,
        save_path=save_path,
    )

    print("\n" + "=" * 70)
    print("TEST EVALUATION: full native-resolution held-out fires")
    print("=" * 70)

    if save_path.exists():
        model.load_state_dict(torch.load(save_path))
        trainer.base_model = model
        if trainer.n_gpus > 1:
            trainer.model = nn.DataParallel(model, device_ids=list(range(trainer.n_gpus)))
        else:
            trainer.model = model
        trainer.model = trainer.model.to(trainer.device)

    test_agg, test_details = trainer.evaluate(test_scenarios, verbose=True, return_predictions=False)

    print("\nPer-scene test results:")
    per_scene = {}
    for scene_id in test_scenes:
        ms = [m for m in test_details if m["scene_id"] == scene_id]
        if not ms:
            continue
        corrs = [m["correlation"] for m in ms if np.isfinite(m["correlation"])]
        rmses = [m["rel_rmse"] for m in ms if np.isfinite(m["rel_rmse"])]
        ious = [m["iou_t50"] for m in ms if np.isfinite(m["iou_t50"])]
        per_scene[scene_id] = {
            "correlation": float(np.mean(corrs)) if corrs else None,
            "correlation_std": float(np.std(corrs)) if corrs else None,
            "rel_rmse": float(np.mean(rmses)) if rmses else None,
            "iou_t50": float(np.mean(ious)) if ious else None,
            "n_fires": len(ms),
        }
        print(f"  {scene_id}: Corr={per_scene[scene_id]['correlation']:.3f}, "
              f"RelRMSE={per_scene[scene_id]['rel_rmse']:.3f}, "
              f"IoU@50={per_scene[scene_id]['iou_t50']:.3f}, n={len(ms)}")

    results = {
        "config": {
            "experiment": "patch_cross_scene_bands",
            "train_scenes": train_scenes,
            "test_scenes": test_scenes,
            "feature_group": feature_group,
            "n_time_bands": n_time_bands,
            "patch_size": patch_size,
            "patch_stride": patch_stride,
            "n_full_patches": n_full_patches,
            "n_train_band_samples": len(train_band_samples),
            "n_val_scenarios": len(val_scenarios),
            "n_test_scenarios": len(test_scenarios),
            "batch_size": batch_size,
            "lr": lr,
            "lambda_G": lambda_G,
            "lambda_B": lambda_B,
            "hidden_channels": hidden_channels,
            "num_cnn_layers": num_cnn_layers,
            "mlp_hidden": mlp_hidden,
            "mlp_num_layers": mlp_num_layers,
            "seed": seed,
        },
        "history": to_jsonable(history),
        "test_metrics": to_jsonable(test_agg),
        "per_scene_test": to_jsonable(per_scene),
        "per_fire_metrics": to_jsonable([
            {k: v for k, v in m.items() if k not in ["T_pred", "T_obs", "valid_mask", "source_mask", "G", "B"]}
            for m in test_details
        ]),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_dir}")
    return results


def main():
    p = argparse.ArgumentParser(description="Patch-based cross-scene band training")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_scenes", nargs="+", required=True)
    p.add_argument("--test_scenes", nargs="+", required=True)
    p.add_argument("--n_time_bands", type=int, required=True)
    p.add_argument("--feature_group", default="topo_weather_omitted")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--patch_stride", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_G", type=float, default=0.005)
    p.add_argument("--lambda_B", type=float, default=0.005)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_gpus", type=int, default=4)
    p.add_argument("--n_solver_jobs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--num_cnn_layers", type=int, default=5)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--mlp_num_layers", type=int, default=2)
    p.add_argument("--max_fires_per_scene", type=int, default=None)
    a = p.parse_args()

    run_experiment(
        data_root=a.data_root,
        output_dir=Path(a.output_dir),
        train_scenes=a.train_scenes,
        test_scenes=a.test_scenes,
        n_time_bands=a.n_time_bands,
        feature_group=a.feature_group,
        patch_size=a.patch_size,
        patch_stride=a.patch_stride,
        n_epochs=a.n_epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        lambda_G=a.lambda_G,
        lambda_B=a.lambda_B,
        num_workers=a.num_workers,
        n_gpus=a.n_gpus,
        n_solver_jobs=a.n_solver_jobs,
        seed=a.seed,
        hidden_channels=a.hidden_channels,
        num_cnn_layers=a.num_cnn_layers,
        mlp_hidden=a.mlp_hidden,
        mlp_num_layers=a.mlp_num_layers,
        max_fires_per_scene=a.max_fires_per_scene,
    )


if __name__ == "__main__":
    main()
