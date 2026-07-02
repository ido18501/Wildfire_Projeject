#!/usr/bin/env python3
import argparse
import copy
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
BAND_DIR = REPO / "experiments" / "wildfire" / "band_training"
sys.path.insert(0, str(BAND_DIR))

from experiments.wildfire.sim2real_loader import Sim2RealFireLoader
from experiments.wildfire.band_training.trainer import SceneTrainer
from config import ExperimentConfig


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_manifest(path):
    return json.loads(Path(path).expanduser().read_text())


def maybe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def build_cross_scene_maps(manifest):
    train, val, test = {}, {}, {}
    for sid in manifest["cross_scene"]["train_scenes"]:
        train[sid] = manifest["scenes"][sid]["selected"]
    for sid in manifest["cross_scene"].get("val_scenes", []):
        val[sid] = manifest["scenes"][sid]["selected"]
    for sid in manifest["cross_scene"]["test_scenes"]:
        test[sid] = manifest["scenes"][sid]["selected"]
    return train, val, test


def build_per_scene_maps(manifest, scene_id):
    info = manifest["scenes"][scene_id]
    return (
        {scene_id: info["train"]},
        {scene_id: info["val"]},
        {scene_id: info["test"]},
    )


def collect_scenarios(trainer, scene_to_events, split_name):
    scenarios, skipped = [], []
    for sid, events in scene_to_events.items():
        for eid in events:
            try:
                s = trainer.prepare_scenario(sid, eid)
                if s is None:
                    skipped.append({"scene_id": sid, "event_id": eid, "reason": "None"})
                    continue
                scenarios.append(s)
            except Exception as e:
                skipped.append({"scene_id": sid, "event_id": eid, "reason": repr(e)})
    print(f"{split_name}: loaded {len(scenarios)} scenarios; skipped {len(skipped)}")
    return scenarios, skipped


def construct_config(args, scenes, test_scenes, exp_name):
    requested = {
        "experiment_name": exp_name,
        "experiment_type": args.experiment,
        "scenes": scenes,
        "test_scenes": test_scenes,
        "n_epochs": args.epochs,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lambda_G": args.lambda_G,
        "lambda_B": args.lambda_B,
        "device": args.device,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "band_width": 1.0 / args.n_time_bands,
        "band_stride": 1.0 / args.n_time_bands,
        "feature_group": args.feature_group,
    }

    sig = inspect.signature(ExperimentConfig)
    allowed = set(sig.parameters.keys())
    kwargs = {k: v for k, v in requested.items() if k in allowed and v is not None}
    cfg = ExperimentConfig(**kwargs)

    # Force these even if dataclass defaults differ.
    if hasattr(cfg, "band_width"):
        cfg.band_width = 1.0 / args.n_time_bands
    if hasattr(cfg, "band_stride"):
        cfg.band_stride = 1.0 / args.n_time_bands
    if hasattr(cfg, "lambda_G"):
        cfg.lambda_G = args.lambda_G
    if hasattr(cfg, "lambda_B"):
        cfg.lambda_B = args.lambda_B
    if hasattr(cfg, "feature_group"):
        cfg.feature_group = args.feature_group

    return cfg


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state):
    model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in state.items()})


def is_better(metric_name, value, best_value):
    if value is None or not np.isfinite(value):
        return False
    if best_value is None:
        return True
    if metric_name == "full_relrmse":
        return value < best_value
    return value > best_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--experiment", choices=["cross_scene", "per_scene"], default="cross_scene")
    p.add_argument("--scene_id", default=None)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_G", type=float, default=0.002)
    p.add_argument("--lambda_B", type=float, default=0.002)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_time_bands", type=int, default=2)
    p.add_argument("--feature_group", default="topo_weather_omitted")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)

    if args.experiment == "cross_scene":
        train_map, val_map, test_map = build_cross_scene_maps(manifest)
        scenes = list(train_map.keys())
        test_scenes = list(test_map.keys())
        exp_name = f"multimetric_band2_cross_lam{args.lambda_G:g}"
    else:
        if args.scene_id is None:
            raise RuntimeError("--scene_id required for per_scene")
        train_map, val_map, test_map = build_per_scene_maps(manifest, args.scene_id)
        scenes = [args.scene_id]
        test_scenes = []
        exp_name = f"multimetric_band2_{args.scene_id}_lam{args.lambda_G:g}"

    cfg = construct_config(args, scenes, test_scenes, exp_name)
    save_json(out / "run_args.json", vars(args))
    save_json(out / "config_repr.json", getattr(cfg, "__dict__", str(cfg)))
    save_json(out / "manifest_snapshot.json", manifest)

    print("=" * 90)
    print("Band n=2 multi-metric checkpoint-selection run")
    print("=" * 90)
    print("experiment:", args.experiment)
    print("scene:", args.scene_id)
    print("lambda:", args.lambda_G)
    print("epochs:", args.epochs)
    print("val_every:", args.val_every)
    print("band_width:", getattr(cfg, "band_width", None))
    print("band_stride:", getattr(cfg, "band_stride", None))
    print("=" * 90)

    loader = Sim2RealFireLoader(manifest["data_root"])
    trainer = SceneTrainer(cfg, loader)

    # Setup model.
    first_sid = next(iter(train_map.keys()))
    first_eid = train_map[first_sid][0]
    first_data = loader.load_scenario(first_sid, first_eid)
    H, W = first_data["topography"]["elevation"].shape
    trainer.setup_model(H, W)
    print(f"Initialized model for grid {H} x {W}")

    train_scenarios, train_skip = collect_scenarios(trainer, train_map, "train")
    val_scenarios, val_skip = collect_scenarios(trainer, val_map, "val")
    test_scenarios, test_skip = collect_scenarios(trainer, test_map, "test")
    save_json(out / "load_report.json", {
        "n_train": len(train_scenarios),
        "n_val": len(val_scenarios),
        "n_test": len(test_scenarios),
        "train_skip": train_skip,
        "val_skip": val_skip,
        "test_skip": test_skip,
    })

    selectors = {
        "band_iou": {"best": None, "epoch": None, "state": None},
        "full_iou50": {"best": None, "epoch": None, "state": None},
        "full_corr": {"best": None, "epoch": None, "state": None},
        "full_relrmse": {"best": None, "epoch": None, "state": None},
    }

    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        random.shuffle(train_scenarios)

        losses, data_losses, reg_losses, n_bands = [], [], [], []

        for i, sc in enumerate(train_scenarios, start=1):
            r = trainer.train_step(sc)
            if r is None:
                continue
            losses.append(r.get("loss"))
            data_losses.append(r.get("data_loss"))
            reg_losses.append(r.get("reg_loss"))
            if "n_bands" in r:
                n_bands.append(r["n_bands"])

            if i % 50 == 0:
                print(f"  epoch {epoch} fire {i}/{len(train_scenarios)}")

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_data_loss": float(np.mean(data_losses)) if data_losses else None,
            "train_reg_loss": float(np.mean(reg_losses)) if reg_losses else None,
            "mean_n_bands": float(np.mean(n_bands)) if n_bands else None,
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.time() - t0),
        }

        do_val = epoch == 1 or epoch % args.val_every == 0
        if do_val:
            print(f"\nValidation at epoch {epoch}...")

            band_agg, _ = trainer.evaluate_bands(val_scenarios)
            full_agg, _, _ = trainer.evaluate_full_fire(val_scenarios, save_predictions=False)

            band_iou = maybe_get(band_agg, "mean_band_iou_mean")
            band_mse = maybe_get(band_agg, "mean_band_mse_mean")
            full_iou50 = maybe_get(full_agg, "iou_50_mean")
            full_corr = maybe_get(full_agg, "correlation_mean")
            full_relrmse = maybe_get(full_agg, "relative_rmse_mean")

            rec.update({
                "val_band_iou": band_iou,
                "val_band_mse": band_mse,
                "val_full_iou25": maybe_get(full_agg, "iou_25_mean"),
                "val_full_iou50": full_iou50,
                "val_full_iou75": maybe_get(full_agg, "iou_75_mean"),
                "val_full_iou100": maybe_get(full_agg, "iou_100_mean"),
                "val_full_corr": full_corr,
                "val_full_relrmse": full_relrmse,
            })

            # Keep scheduler behavior aligned with band trainer: scheduler uses band IoU.
            if band_iou is not None:
                trainer.scheduler.step(band_iou)

            current = {
                "band_iou": band_iou,
                "full_iou50": full_iou50,
                "full_corr": full_corr,
                "full_relrmse": full_relrmse,
            }

            improved_any_band = False
            for name, val in current.items():
                if is_better(name, val, selectors[name]["best"]):
                    selectors[name]["best"] = float(val)
                    selectors[name]["epoch"] = epoch
                    selectors[name]["state"] = cpu_state_dict(trainer.model)
                    improved_any_band = improved_any_band or (name == "band_iou")

            if improved_any_band:
                patience_counter = 0
            else:
                patience_counter += args.val_every

            print(
                f"Epoch {epoch:3d}: "
                f"loss={rec['train_loss']:.3f} "
                f"data={rec['train_data_loss']:.3f} "
                f"reg={rec['train_reg_loss']:.4f} | "
                f"bandIoU={band_iou:.4f} "
                f"fullIoU50={full_iou50:.4f} "
                f"corr={full_corr:.4f} "
                f"relRMSE={full_relrmse:.4f} "
                f"lr={trainer.optimizer.param_groups[0]['lr']:.2e}"
            )

        else:
            print(
                f"Epoch {epoch:3d}: "
                f"loss={rec['train_loss']:.3f} "
                f"data={rec['train_data_loss']:.3f} "
                f"reg={rec['train_reg_loss']:.4f} "
                f"lr={trainer.optimizer.param_groups[0]['lr']:.2e}"
            )

        history.append(rec)
        save_json(out / "history_multimetric.json", history)

    # Save selector summary.
    selector_summary = {
        name: {"best": obj["best"], "epoch": obj["epoch"]}
        for name, obj in selectors.items()
    }
    save_json(out / "selector_summary.json", selector_summary)

    # Evaluate test under each selection criterion.
    test_results = {}
    for name, obj in selectors.items():
        if obj["state"] is None:
            continue

        print(f"\nTesting checkpoint selected by {name}: epoch {obj['epoch']}, best={obj['best']}")
        load_state(trainer.model, obj["state"])

        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "selector": name,
            "selector_best": obj["best"],
            "selector_epoch": obj["epoch"],
            "args": vars(args),
        }, out / f"model_best_by_{name}.pt")

        band_test, band_details = trainer.evaluate_bands(test_scenarios)
        full_test, full_details, preds = trainer.evaluate_full_fire(test_scenarios, save_predictions=True)

        test_results[name] = {
            "selector_best": obj["best"],
            "selector_epoch": obj["epoch"],
            "test_band": band_test,
            "test_full_fire": full_test,
        }

        save_json(out / f"test_metrics_best_by_{name}.json", test_results[name])
        save_json(out / f"test_per_fire_full_best_by_{name}.json", full_details)

        if preds is not None:
            np.savez_compressed(out / f"test_predictions_best_by_{name}.npz", **preds)

    save_json(out / "test_results_by_selector.json", test_results)

    print("\nSUMMARY")
    print("=" * 120)
    print(f"{'selector':15s} {'val_epoch':>9s} {'val_best':>10s} {'test_iou50':>10s} {'test_corr':>10s} {'test_relrmse':>12s} {'test_band_iou':>13s}")
    print("-" * 120)
    for name, r in test_results.items():
        full = r["test_full_fire"]
        band = r["test_band"]
        print(
            f"{name:15s} "
            f"{r['selector_epoch']:9d} "
            f"{r['selector_best']:10.4f} "
            f"{full.get('iou_50_mean', float('nan')):10.4f} "
            f"{full.get('correlation_mean', float('nan')):10.4f} "
            f"{full.get('relative_rmse_mean', float('nan')):12.4f} "
            f"{band.get('mean_band_iou_mean', float('nan')):13.4f}"
        )


if __name__ == "__main__":
    main()
