#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.wildfire.sim2real_loader import Sim2RealFireLoader, CovariateAdapter
from experiments.wildfire.cross_scene import load_full_scene_scenario, OptimizedTrainer
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.config import TVMetricType


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


def infer_feature_dims(loader, adapter, scenes):
    for scene_id in scenes:
        if scene_id not in loader.scenes:
            continue
        for event_id in loader.scenes[scene_id]["events"]:
            s = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if s is not None:
                return s["spatial_feat"].shape[0], s["global_feat"].shape[0]
    raise RuntimeError("Could not infer feature dimensions")


def mean_key(items, key):
    vals = [m.get(key) for m in items if m.get(key) is not None and np.isfinite(m.get(key))]
    return float(np.mean(vals)) if vals else None


def std_key(items, key):
    vals = [m.get(key) for m in items if m.get(key) is not None and np.isfinite(m.get(key))]
    return float(np.std(vals)) if vals else None


def aggregate_details(details):
    keys = [
        "correlation", "rel_rmse", "iou_t25", "iou_t50",
        "iou_t75", "iou_t100", "rmse", "mae",
    ]
    out = {"n_fires": len(details)}
    for k in keys:
        out[k] = mean_key(details, k)
        out[k + "_std"] = std_key(details, k)
    return out


def evaluate_scene_chunked(loader, adapter, trainer, scene_id, chunk_size, max_fires=None):
    events = loader.scenes[scene_id]["events"]
    if max_fires is not None:
        events = events[:max_fires]

    all_details = []

    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        chunk_events = events[start:end]

        print(f"  chunk {start}-{end}/{len(events)}: loading...", flush=True)
        scenarios = []
        for event_id in chunk_events:
            s = load_full_scene_scenario(loader, adapter, scene_id, event_id)
            if s is not None:
                scenarios.append(s)

        print(f"  chunk {start}-{end}: evaluating {len(scenarios)} scenarios...", flush=True)
        if scenarios:
            _, details = trainer.evaluate(scenarios, verbose=False, return_predictions=False)
            all_details.extend(details)

        del scenarios

        partial_metrics = aggregate_details(all_details)
        print(
            f"  progress {len(all_details)}/{len(events)} usable | "
            f"corr={partial_metrics.get('correlation'):.3f} "
            f"relRMSE={partial_metrics.get('rel_rmse'):.3f} "
            f"IoU50={partial_metrics.get('iou_t50'):.3f}",
            flush=True,
        )

    return aggregate_details(all_details), all_details


def aggregate_scene_files(paths):
    all_fire = []
    per_scene = {}

    for p in paths:
        data = json.load(open(p))
        per_scene[data["scene_id"]] = data["metrics"]
        all_fire.extend(data["per_fire_metrics"])

    return {
        "aggregate": aggregate_details(all_fire),
        "per_scene": per_scene,
        "per_fire_metrics": all_fire,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--test_scenes", nargs="+", required=True)
    p.add_argument("--feature_group", required=True)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--num_cnn_layers", type=int, default=5)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--mlp_num_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambda_G", type=float, default=0.01)
    p.add_argument("--lambda_B", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_gpus", type=int, default=1)
    p.add_argument("--n_solver_jobs", type=int, default=6)
    p.add_argument("--chunk_size", type=int, default=250)
    p.add_argument("--max_fires_per_scene", type=int, default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = Sim2RealFireLoader(args.data_root)
    adapter = CovariateAdapter(feature_group=args.feature_group)

    n_spatial, n_global = infer_feature_dims(loader, adapter, args.test_scenes)
    print(f"Feature dims: spatial={n_spatial}, global={n_global}")

    model = FlexibleCovariateEncoder(
        n_spatial_features=n_spatial,
        n_global_features=n_global,
        hidden_channels=args.hidden_channels,
        num_cnn_layers=args.num_cnn_layers,
        mlp_hidden=args.mlp_hidden,
        mlp_num_layers=args.mlp_num_layers,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False))

    trainer = OptimizedTrainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=args.lr,
        lambda_G=args.lambda_G,
        lambda_B=args.lambda_B,
        batch_size=args.batch_size,
        num_workers=0,
        n_gpus=args.n_gpus,
        n_solver_jobs=args.n_solver_jobs,
        metric_type=TVMetricType.LOG_EUCLIDEAN,
    )

    scene_jsons = []

    for scene_id in args.test_scenes:
        scene_out = out / f"test_results_{scene_id}.json"

        if scene_out.exists():
            print(f"Skipping existing {scene_out}")
            scene_jsons.append(scene_out)
            continue

        print("\n" + "=" * 70)
        print(f"Evaluating scene {scene_id} in chunks of {args.chunk_size}")
        print("=" * 70)

        metrics, details = evaluate_scene_chunked(
            loader, adapter, trainer, scene_id,
            chunk_size=args.chunk_size,
            max_fires=args.max_fires_per_scene,
        )

        result = {
            "scene_id": scene_id,
            "n_scenarios": len(details),
            "metrics": to_jsonable(metrics),
            "per_fire_metrics": to_jsonable(details),
        }

        with open(scene_out, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved {scene_out}")
        scene_jsons.append(scene_out)

    final = aggregate_scene_files(scene_jsons)
    final["config"] = vars(args)

    with open(out / "results.json", "w") as f:
        json.dump(to_jsonable(final), f, indent=2)

    print("\nFinal aggregate:")
    print(json.dumps(to_jsonable(final["aggregate"]), indent=2))
    print(f"Saved final results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
