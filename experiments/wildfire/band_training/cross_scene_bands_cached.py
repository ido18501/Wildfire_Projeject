#!/usr/bin/env python3
"""
Memory-safe cached patch-based cross-scene band training.

General n_time_bands >= 1.
- cache full-fire patches once
- manifest stores valid patch/band pairs
- __getitem__ creates teacher-forced band sample on demand
- validation/test are full native-resolution fires
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from experiments.wildfire.sim2real_loader import Sim2RealFireLoader, CovariateAdapter
from experiments.wildfire.cross_scene import (
    load_full_scene_scenario,
    extract_patches_grid,
    collate_patches,
    OptimizedTrainer,
)
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.config import TVMetricType


def make_band_sample_from_patch(p: Dict, band_index: int, n_time_bands: int, min_band_pixels: int = 10):
    arrival = p["arrival_times"]
    if torch.is_tensor(arrival):
        arrival = arrival.numpy()
    finite = np.isfinite(arrival)
    if finite.sum() < min_band_pixels:
        return None

    max_time = float(np.nanmax(arrival[finite]))
    if not np.isfinite(max_time) or max_time <= 1e-6:
        return None

    k = band_index
    t_start = (k / n_time_bands) * max_time
    t_end = ((k + 1) / n_time_bands) * max_time

    source_mask = finite & (arrival <= t_start + 1e-6)
    if k == 0:
        source_mask = p["source_mask"].numpy().astype(bool) if torch.is_tensor(p["source_mask"]) else p["source_mask"].astype(bool)
        if not source_mask.any():
            source_mask = finite & (arrival <= t_start + 1e-6)

    target_mask = finite & (arrival > t_start + 1e-6) & (arrival <= t_end + 1e-6)

    if source_mask.sum() == 0 or target_mask.sum() < min_band_pixels:
        return None

    band_arrival = np.full_like(arrival, np.inf, dtype=np.float32)
    band_arrival[source_mask] = 0.0
    band_arrival[target_mask] = (arrival[target_mask] - t_start).astype(np.float32)

    spatial = p["spatial_feat"].float()
    global_f = p["global_feat"].float()

    return {
        "spatial_feat": spatial,
        "global_feat": global_f,
        "arrival_times": torch.from_numpy(band_arrival).float(),
        "source_mask": torch.from_numpy(source_mask.astype(np.int8)),
        "valid_mask": torch.from_numpy(target_mask.astype(bool)),
    }


def valid_band_indices_for_patch(p: Dict, n_time_bands: int, min_band_pixels: int = 10):
    arrival = p["arrival_times"]
    if torch.is_tensor(arrival):
        arrival = arrival.numpy()
    finite = np.isfinite(arrival)
    if finite.sum() < min_band_pixels:
        return []

    max_time = float(np.nanmax(arrival[finite]))
    if not np.isfinite(max_time) or max_time <= 1e-6:
        return []

    valid = []
    for k in range(n_time_bands):
        t_start = (k / n_time_bands) * max_time
        t_end = ((k + 1) / n_time_bands) * max_time
        target_mask = finite & (arrival > t_start + 1e-6) & (arrival <= t_end + 1e-6)
        if target_mask.sum() >= min_band_pixels:
            valid.append(k)
    return valid


class CachedBandPatchDataset(Dataset):
    def __init__(self, manifest: List[Dict], n_time_bands: int):
        self.manifest = manifest
        self.n_time_bands = n_time_bands

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        p = torch.load(item["path"], map_location="cpu", weights_only=False)
        sample = make_band_sample_from_patch(p, item["band_index"], self.n_time_bands)
        if sample is None:
            raise RuntimeError(f"Invalid cached band sample at idx={idx}: {item}")
        return sample


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


def cache_band_training_manifest(
    loader,
    adapter,
    train_scenes,
    cache_dir: Path,
    patch_size: int,
    patch_stride: int,
    train_split: float,
    seed: int,
    max_fires_per_scene: Optional[int],
    n_time_bands: int,
    max_train_events_per_scene: Optional[int] = None,
    max_train_patches: Optional[int] = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / f"band_manifest_n{n_time_bands}.json"

    if manifest_path.exists():
        print(f"Using existing band manifest: {manifest_path}")
        with open(manifest_path, "r") as f:
            return json.load(f)

    # Reuse an existing full-patch cache if present.
    # This is how band n=2 and n=1 reuse the topo full-patch cache
    # created by 02_cross_old_topo.
    full_manifest_path = cache_dir / "manifest.json"
    if full_manifest_path.exists():
        print(f"Reusing existing full-patch cache: {full_manifest_path}")
        with open(full_manifest_path, "r") as f:
            full_manifest = json.load(f)

        if max_train_patches is not None:
            print(f"Filtering shared full-patch cache: {len(full_manifest)} -> {max_train_patches}")
            full_manifest = full_manifest[:max_train_patches]

        manifest = []
        for item in full_manifest:
            p = torch.load(item["path"], map_location="cpu", weights_only=False)
            band_indices = valid_band_indices_for_patch(p, n_time_bands)
            for k in band_indices:
                manifest.append({
                    "path": item["path"],
                    "scene_id": item.get("scene_id", ""),
                    "event_id": item.get("event_id", ""),
                    "patch_id": int(item.get("patch_id", 0)),
                    "band_index": int(k),
                })

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Created band manifest from shared cache: {len(manifest)} band samples")
        return manifest

    manifest = []
    patch_counter = 0
    full_patch_count = 0

    print("\nCaching full patches and band manifest...")
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            print(f"  Warning: missing train scene {scene_id}")
            continue

        events = loader.scenes[scene_id]["events"].copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(events)
        n_train = int(len(events) * train_split)
        train_events = events[:n_train]
        if max_train_events_per_scene:
            train_events = train_events[:max_train_events_per_scene]
        elif max_fires_per_scene:
            train_events = train_events[:max_fires_per_scene]

        scene_patches = 0
        scene_bands = 0
        t0 = time.perf_counter()

        for event_id in train_events:
            scenario = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if scenario is None:
                continue

            patches = extract_patches_grid(
                scenario["spatial_feat"].numpy(),
                scenario["global_feat"].numpy(),
                scenario["arrival_times"],
                patch_size=patch_size,
                stride=patch_stride,
                scene_id=scene_id,
                event_id=event_id,
            )

            for p in patches:
                band_indices = valid_band_indices_for_patch(p, n_time_bands)
                if not band_indices:
                    continue

                out = cache_dir / f"patch_{patch_counter:08d}.pt"
                torch.save(p, out)
                full_patch_count += 1
                scene_patches += 1

                for k in band_indices:
                    manifest.append({
                        "path": str(out),
                        "scene_id": scene_id,
                        "event_id": event_id,
                        "patch_id": int(p.get("patch_id", patch_counter)),
                        "band_index": int(k),
                    })
                    scene_bands += 1

                patch_counter += 1

            del scenario, patches

        dt = time.perf_counter() - t0
        print(f"  {scene_id}: cached {scene_patches} patches -> {scene_bands} band samples from {len(train_events)} fires ({dt:.1f}s)")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nTotal cached full patches: {full_patch_count}")
    print(f"Total band manifest samples: {len(manifest)}")
    return manifest


def load_validation_scenarios(loader, adapter, train_scenes, train_split, val_split, seed, max_fires_per_scene, max_val_events_per_scene=None):
    val_scenarios = []
    print("\nLoading full-scene validation scenarios...")
    for scene_id in train_scenes:
        if scene_id not in loader.scenes:
            continue
        events = loader.scenes[scene_id]["events"].copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(events)
        n_train = int(len(events) * train_split)
        n_val = int(len(events) * val_split)
        val_events = events[n_train:n_train + n_val]
        if max_val_events_per_scene:
            val_events = val_events[:max_val_events_per_scene]
        elif max_fires_per_scene:
            val_events = val_events[:max_fires_per_scene]
        for event_id in val_events:
            s = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if s is not None:
                val_scenarios.append(s)
    print(f"Validation full-fire scenarios: {len(val_scenarios)}")
    return val_scenarios


def load_test_scenarios(loader, adapter, test_scenes, max_fires_per_scene):
    test_scenarios = []
    print("\nLoading full-scene test scenarios...")
    for scene_id in test_scenes:
        if scene_id not in loader.scenes:
            print(f"  Warning: missing test scene {scene_id}")
            continue
        events = loader.scenes[scene_id]["events"]
        if max_fires_per_scene:
            events = events[:max_fires_per_scene]
        start = len(test_scenarios)
        for event_id in events:
            s = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if s is not None:
                test_scenarios.append(s)
        print(f"  {scene_id}: {len(test_scenarios) - start} scenarios")
    print(f"Total test full-fire scenarios: {len(test_scenarios)}")
    return test_scenarios


def train_cached(trainer, train_dataset, val_scenarios, n_epochs, num_workers, save_path, early_stop_patience=20):
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_patches,
        pin_memory=False,
    )

    history = {
        "train_loss": [], "train_loss_data": [], "train_loss_reg": [],
        "val_corr": [], "val_rel_rmse": [], "val_iou_t50": [], "lr": [],
    }

    best_val_corr = -float("inf")
    patience_counter = 0

    print("\n" + "=" * 70)
    print("TRAINING CACHED BAND CROSS-SCENE")
    print("=" * 70)
    print(f"  Band samples: {len(train_dataset)}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Validation full fires: {len(val_scenarios)}")
    print("=" * 70)

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        train_metrics = trainer.train_epoch(train_loader)
        total_loss = train_metrics["total_loss"]

        trainer.scheduler.step(total_loss)

        val_agg, _ = trainer.evaluate(val_scenarios)
        val_corr = val_agg["correlation"]
        val_rel_rmse = val_agg["rel_rmse"]
        val_iou = val_agg.get("iou_t50", np.nan)

        history["train_loss"].append(total_loss)
        history["train_loss_data"].append(train_metrics["data_loss"])
        history["train_loss_reg"].append(train_metrics["tv_loss"])
        history["val_corr"].append(val_corr)
        history["val_rel_rmse"].append(val_rel_rmse)
        history["val_iou_t50"].append(val_iou)
        history["lr"].append(trainer.optimizer.param_groups[0]["lr"])

        dt = time.perf_counter() - t0
        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Loss: {total_loss:.4f} "
            f"(data={train_metrics['data_loss']:.4f}, reg={train_metrics['tv_loss']:.4f}) | "
            f"Val: corr={val_corr:.3f}, relRMSE={val_rel_rmse:.3f} | "
            f"lr={trainer.optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {dt:.1f}s"
        )

        if np.isfinite(val_corr) and val_corr > best_val_corr:
            best_val_corr = val_corr
            patience_counter = 0
            torch.save(trainer.base_model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


def run_experiment(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CACHED PATCH-BASED CROSS-SCENE BAND EXPERIMENT")
    print("=" * 70)
    print(f"Train scenes: {args.train_scenes}")
    print(f"Test scenes: {args.test_scenes}")
    print(f"Feature group: {args.feature_group}")
    print(f"n_time_bands: {args.n_time_bands}")
    print(f"Patch size: {args.patch_size}, stride: {args.patch_stride}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: CNN {args.num_cnn_layers}L x {args.hidden_channels}, MLP {args.mlp_num_layers}L x {args.mlp_hidden}")
    print(f"Reg: lambda_G={args.lambda_G}, lambda_B={args.lambda_B}, lr={args.lr}")
    if args.max_fires_per_scene:
        print(f"LIMITED mode: max_fires_per_scene={args.max_fires_per_scene}")
    print("=" * 70)

    loader = Sim2RealFireLoader(args.data_root)
    adapter = CovariateAdapter(feature_group=args.feature_group)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (output_dir / "patch_cache")
    print(f"Patch cache dir: {cache_dir}")

    manifest = cache_band_training_manifest(
        loader, adapter, args.train_scenes, cache_dir,
        args.patch_size, args.patch_stride, args.train_split, args.seed,
        args.max_fires_per_scene, args.n_time_bands,
        args.max_train_events_per_scene, args.max_train_patches
    )
    if not manifest:
        raise RuntimeError("No cached band samples created.")

    train_dataset = CachedBandPatchDataset(manifest, args.n_time_bands)
    sample = train_dataset[0]
    n_spatial = sample["spatial_feat"].shape[0]
    n_global = sample["global_feat"].shape[0]

    val_scenarios = load_validation_scenarios(
        loader, adapter, args.train_scenes, args.train_split, args.val_split,
        args.seed, args.max_fires_per_scene, args.max_val_events_per_scene
    )
    test_scenarios = []
    if args.dry_run or args.n_epochs == 0:
        test_scenarios = load_test_scenarios(loader, adapter, args.test_scenes, args.max_fires_per_scene)
        print("\nDRY RUN COMPLETE")
        print(f"Band train samples used: {len(manifest)}")
        print(f"Validation full-fire scenarios: {len(val_scenarios)}")
        print(f"Test full-fire scenarios: {len(test_scenarios)}")
        return

    model = FlexibleCovariateEncoder(
        n_spatial_features=n_spatial,
        n_global_features=n_global,
        hidden_channels=args.hidden_channels,
        num_cnn_layers=args.num_cnn_layers,
        mlp_hidden=args.mlp_hidden,
        mlp_num_layers=args.mlp_num_layers,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = OptimizedTrainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=args.lr,
        lambda_G=args.lambda_G,
        lambda_B=args.lambda_B,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_gpus=args.n_gpus,
        n_solver_jobs=args.n_solver_jobs,
        metric_type=TVMetricType.LOG_EUCLIDEAN,
    )

    save_path = output_dir / "best_model.pt"
    history = train_cached(
        trainer, train_dataset, val_scenarios,
        args.n_epochs, args.num_workers, save_path, early_stop_patience=20
    )

    if args.skip_test:
        train_only = {
            "config": vars(args),
            "n_train_items": len(manifest),
            "n_val_scenarios": len(val_scenarios),
            "history": to_jsonable(history),
            "checkpoint": str(save_path),
            "note": "TRAINING COMPLETE; TEST SKIPPED. Evaluate scene-by-scene from checkpoint.",
        }
        with open(output_dir / "train_only_results.json", "w") as f:
            json.dump(train_only, f, indent=2)
        print(f"\nTRAINING COMPLETE; TEST SKIPPED. Saved: {output_dir / 'train_only_results.json'}")
        return


    if not test_scenarios:
        test_scenarios = load_test_scenarios(loader, adapter, args.test_scenes, args.max_fires_per_scene)

    if args.skip_test:
        train_only = {
            "config": vars(args),
            "n_train_items": len(manifest),
            "n_val_scenarios": len(val_scenarios),
            "history": to_jsonable(history),
            "checkpoint": str(save_path),
            "note": "Training completed; test skipped intentionally. Evaluate scene-by-scene from checkpoint.",
        }
        with open(output_dir / "train_only_results.json", "w") as f:
            json.dump(train_only, f, indent=2)
        print(f"\nSKIP TEST: saved train-only results to {output_dir / 'train_only_results.json'}")
        return

    print("\n" + "=" * 70)
    print("TEST EVALUATION: full native-resolution held-out fires")
    print("=" * 70)

    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=False))
        trainer.base_model = model
        if trainer.n_gpus > 1:
            trainer.model = nn.DataParallel(model, device_ids=list(range(trainer.n_gpus)))
        else:
            trainer.model = model
        trainer.model = trainer.model.to(trainer.device)

    test_agg, test_details = trainer.evaluate(test_scenarios, verbose=True, return_predictions=False)

    per_scene = {}
    print("\nPer-scene test results:")
    for scene_id in args.test_scenes:
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
        "config": vars(args),
        "n_band_samples": len(manifest),
        "n_val_scenarios": len(val_scenarios),
        "n_test_scenarios": len(test_scenarios),
        "history": to_jsonable(history),
        "test_metrics": to_jsonable(test_agg),
        "per_scene_test": to_jsonable(per_scene),
        "per_fire_metrics": to_jsonable(test_details),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None,
                   help="Optional existing full-patch cache dir to reuse. If omitted, uses output_dir/patch_cache.")
    p.add_argument("--train_scenes", nargs="+", required=True)
    p.add_argument("--test_scenes", nargs="+", required=True)
    p.add_argument("--n_time_bands", type=int, required=True)
    p.add_argument("--feature_group", default="topo_weather_omitted")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--patch_stride", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lambda_G", type=float, default=0.005)
    p.add_argument("--lambda_B", type=float, default=0.005)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--num_cnn_layers", type=int, default=5)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--mlp_num_layers", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_gpus", type=int, default=1)
    p.add_argument("--n_solver_jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_split", type=float, default=0.70)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--max_fires_per_scene", type=int, default=None)
    p.add_argument("--max_train_events_per_scene", type=int, default=None)
    p.add_argument("--max_train_patches", type=int, default=None)
    p.add_argument("--max_val_events_per_scene", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_test", action="store_true")
    args = p.parse_args()

    if args.n_time_bands < 1:
        raise ValueError("--n_time_bands must be >= 1")

    run_experiment(args)


if __name__ == "__main__":
    main()
