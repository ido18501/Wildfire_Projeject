"""
Run band training from sample-level component manifests.

Examples:
  base:
    train = base_train_samples.csv
    val/test = real-only normal_full

  robust:
    train = robust_train_samples.csv
    val/test = same real-only normal_full primary manifests

No loss/solver changes are introduced here.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ExperimentConfig
from trainer import SceneTrainer
from component_manifest import load_scenarios_from_sample_manifest
from experiments.wildfire.sim2real_loader import Sim2RealFireLoader


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=True))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return

    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_data_root(args):
    if args.data_root:
        return str(Path(args.data_root).expanduser())

    manifest_path = Path(args.manifest).expanduser()
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            for key in ["data_root", "DATA_ROOT", "root", "dataset_root"]:
                if key in m:
                    return str(Path(m[key]).expanduser())
        except Exception:
            pass

    return "/home/shirb/sim2real_fire_data/simulation_data"


def summarize_scenarios(name, scenarios):
    by_type = {}
    by_scene = {}
    for s in scenarios:
        by_type[s.get("sample_type", "")] = by_type.get(s.get("sample_type", ""), 0) + 1
        by_scene[s.get("scene_id", "")] = by_scene.get(s.get("scene_id", ""), 0) + 1

    print(f"{name}: n={len(scenarios)}")
    print(f"  by_type:  {by_type}")
    print(f"  by_scene: {by_scene}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--extended-val-manifest", default=None)
    ap.add_argument("--extended-test-manifest", default=None)

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--feature-group", default="baseline")

    ap.add_argument("--manifest", default="~/wildfire_small_subset_no014_manifest.json")
    ap.add_argument("--data-root", default=None)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=43)

    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--lambda-g", type=float, default=0.002)
    ap.add_argument("--lambda-b", type=float, default=0.002)

    ap.add_argument("--band-width", type=float, default=0.34)
    ap.add_argument("--band-stride", type=float, default=0.17)
    ap.add_argument("--band-max-iter", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=50)

    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--scheduler-patience", type=int, default=3)

    # Smoke-test caps.
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--max-test-samples", type=int, default=None)
    ap.add_argument("--max-extended-val-samples", type=int, default=None)
    ap.add_argument("--max-extended-test-samples", type=int, default=None)

    ap.add_argument("--strict-load", action="store_true")
    ap.add_argument("--save-predictions", action="store_true")

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = read_data_root(args)
    loader = Sim2RealFireLoader(data_root)

    cfg = ExperimentConfig(
        experiment_name=args.run_name,
        experiment_type="component_manifest_band",
        feature_group=args.feature_group,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_G=args.lambda_g,
        lambda_B=args.lambda_b,
        band_width=args.band_width,
        band_stride=args.band_stride,
        band_max_iter=args.band_max_iter,
        max_iter=args.max_iter,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        device=args.device,
        output_dir=str(out_dir),
        seed=args.seed,
    )

    trainer = SceneTrainer(cfg, loader)

    print("=" * 120)
    print("COMPONENT MANIFEST BAND RUN")
    print("=" * 120)
    print("run_name:", args.run_name)
    print("feature_group:", args.feature_group)
    print("data_root:", data_root)
    print("out_dir:", out_dir)
    print("train_manifest:", args.train_manifest)
    print("val_manifest:", args.val_manifest)
    print("test_manifest:", args.test_manifest)
    print()

    train_scenarios, train_skipped = load_scenarios_from_sample_manifest(
        args.train_manifest,
        trainer,
        max_samples=args.max_train_samples,
        strict=args.strict_load,
    )
    val_scenarios, val_skipped = load_scenarios_from_sample_manifest(
        args.val_manifest,
        trainer,
        max_samples=args.max_val_samples,
        strict=args.strict_load,
    )
    test_scenarios, test_skipped = load_scenarios_from_sample_manifest(
        args.test_manifest,
        trainer,
        max_samples=args.max_test_samples,
        strict=args.strict_load,
    )

    summarize_scenarios("train", train_scenarios)
    summarize_scenarios("val_primary", val_scenarios)
    summarize_scenarios("test_primary", test_scenarios)

    if not train_scenarios:
        raise RuntimeError("No train scenarios loaded")
    if not val_scenarios:
        raise RuntimeError("No val scenarios loaded")
    if not test_scenarios:
        raise RuntimeError("No test scenarios loaded")

    first = train_scenarios[0]
    H, W = first["arrival_times"].shape
    n_spatial = first["spatial_feat"].shape[0]
    n_global = first["global_feat"].shape[0]

    trainer.setup_model(H, W, n_spatial_features=n_spatial, n_global_features=n_global)

    write_json(out_dir / "run_config.json", {
        "args": vars(args),
        "data_root": data_root,
        "config": asdict(cfg),
        "n_spatial": n_spatial,
        "n_global": n_global,
        "H": H,
        "W": W,
        "load_skipped": {
            "train": train_skipped,
            "val": val_skipped,
            "test": test_skipped,
        },
    })

    t0 = time.time()
    history = trainer.train(
        train_scenarios,
        val_scenarios,
        verbose=True,
        scene_prefix=f"[{args.run_name}]",
    )
    train_seconds = time.time() - t0

    write_json(out_dir / "history.json", history)

    # Save best-loaded final model. SceneTrainer.train reloads best_state before returning.
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "config": asdict(cfg),
        "history": history,
        "args": vars(args),
    }, out_dir / "model_best_by_primary_val_band_iou.pt")

    torch.save(trainer.model.state_dict(), out_dir / "model.pt")

    print()
    print("=" * 120)
    print("PRIMARY REAL-ONLY EVALUATION")
    print("=" * 120)

    val_band_agg, val_band_rows = trainer.evaluate_bands(val_scenarios)
    test_band_agg, test_band_rows = trainer.evaluate_bands(test_scenarios)

    val_full_agg, val_full_rows, val_preds = trainer.evaluate_full_fire(
        val_scenarios,
        save_predictions=args.save_predictions,
    )
    test_full_agg, test_full_rows, test_preds = trainer.evaluate_full_fire(
        test_scenarios,
        save_predictions=args.save_predictions,
    )

    write_json(out_dir / "val_primary_band_metrics.json", val_band_agg)
    write_json(out_dir / "test_primary_band_metrics.json", test_band_agg)
    write_json(out_dir / "val_primary_full_metrics.json", val_full_agg)
    write_json(out_dir / "test_primary_full_metrics.json", test_full_agg)

    write_csv(out_dir / "val_primary_band_per_sample.csv", val_band_rows)
    write_csv(out_dir / "test_primary_band_per_sample.csv", test_band_rows)
    write_csv(out_dir / "val_primary_full_per_sample.csv", val_full_rows)
    write_csv(out_dir / "test_primary_full_per_sample.csv", test_full_rows)

    if args.save_predictions:
        if val_preds is not None:
            np.savez_compressed(out_dir / "val_primary_predictions.npz", **val_preds)
        if test_preds is not None:
            np.savez_compressed(out_dir / "test_primary_predictions.npz", **test_preds)

    ext_results = {}

    if args.extended_val_manifest:
        print()
        print("=" * 120)
        print("EXTENDED VAL DIAGNOSTIC")
        print("=" * 120)
        ext_val, ext_val_skipped = load_scenarios_from_sample_manifest(
            args.extended_val_manifest,
            trainer,
            max_samples=args.max_extended_val_samples,
            strict=args.strict_load,
        )
        summarize_scenarios("extended_val", ext_val)

        ext_val_band_agg, ext_val_band_rows = trainer.evaluate_bands(ext_val)
        ext_val_full_agg, ext_val_full_rows, _ = trainer.evaluate_full_fire(ext_val, save_predictions=False)

        write_json(out_dir / "extended_val_band_metrics.json", ext_val_band_agg)
        write_json(out_dir / "extended_val_full_metrics.json", ext_val_full_agg)
        write_csv(out_dir / "extended_val_band_per_sample.csv", ext_val_band_rows)
        write_csv(out_dir / "extended_val_full_per_sample.csv", ext_val_full_rows)

        ext_results["extended_val"] = {
            "band": ext_val_band_agg,
            "full": ext_val_full_agg,
            "skipped": ext_val_skipped,
        }

    if args.extended_test_manifest:
        print()
        print("=" * 120)
        print("EXTENDED TEST DIAGNOSTIC")
        print("=" * 120)
        ext_test, ext_test_skipped = load_scenarios_from_sample_manifest(
            args.extended_test_manifest,
            trainer,
            max_samples=args.max_extended_test_samples,
            strict=args.strict_load,
        )
        summarize_scenarios("extended_test", ext_test)

        ext_test_band_agg, ext_test_band_rows = trainer.evaluate_bands(ext_test)
        ext_test_full_agg, ext_test_full_rows, _ = trainer.evaluate_full_fire(ext_test, save_predictions=False)

        write_json(out_dir / "extended_test_band_metrics.json", ext_test_band_agg)
        write_json(out_dir / "extended_test_full_metrics.json", ext_test_full_agg)
        write_csv(out_dir / "extended_test_band_per_sample.csv", ext_test_band_rows)
        write_csv(out_dir / "extended_test_full_per_sample.csv", ext_test_full_rows)

        ext_results["extended_test"] = {
            "band": ext_test_band_agg,
            "full": ext_test_full_agg,
            "skipped": ext_test_skipped,
        }

    final_summary = {
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "train_seconds": train_seconds,
        "history": history,
        "primary": {
            "val_band": val_band_agg,
            "test_band": test_band_agg,
            "val_full": val_full_agg,
            "test_full": test_full_agg,
        },
        "extended": ext_results,
    }

    write_json(out_dir / "summary.json", final_summary)

    print()
    print("=" * 120)
    print("RUN COMPLETE")
    print("=" * 120)
    print("out_dir:", out_dir)
    print("train_seconds:", round(train_seconds, 1))
    print()
    print("Primary val band:", val_band_agg)
    print("Primary test band:", test_band_agg)
    print("Primary val full:", val_full_agg)
    print("Primary test full:", test_full_agg)


if __name__ == "__main__":
    main()
