#!/usr/bin/env python3
import argparse
import inspect
import json
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


def load_manifest(path):
    return json.loads(Path(path).expanduser().read_text())


def safe_json_dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def collect_scenarios(trainer, scene_to_events, split_name):
    scenarios = []
    skipped = []

    for scene_id, event_ids in scene_to_events.items():
        for event_id in event_ids:
            try:
                s = trainer.prepare_scenario(scene_id, event_id)
                if s is None:
                    print(f"SKIP {split_name}: {scene_id}/{event_id} -> prepare_scenario returned None")
                    skipped.append({"scene_id": scene_id, "event_id": event_id, "reason": "None"})
                    continue
                scenarios.append(s)
            except Exception as e:
                print(f"SKIP {split_name}: {scene_id}/{event_id} -> {repr(e)}")
                skipped.append({"scene_id": scene_id, "event_id": event_id, "reason": repr(e)})

    print(f"{split_name}: loaded {len(scenarios)} scenarios into RAM; skipped {len(skipped)}")
    return scenarios, skipped


def build_cross_scene_maps(manifest):
    train, val, test = {}, {}, {}

    for scene_id in manifest["cross_scene"]["train_scenes"]:
        train[scene_id] = manifest["scenes"][scene_id]["selected"]

    for scene_id in manifest["cross_scene"].get("val_scenes", []):
        val[scene_id] = manifest["scenes"][scene_id]["selected"]

    for scene_id in manifest["cross_scene"]["test_scenes"]:
        test[scene_id] = manifest["scenes"][scene_id]["selected"]

    return train, val, test


def build_per_scene_maps(manifest, scene_id):
    info = manifest["scenes"][scene_id]
    return (
        {scene_id: info["train"]},
        {scene_id: info["val"]},
        {scene_id: info["test"]},
    )


def construct_config(args, manifest, scenes, test_scenes, exp_name):
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
        "output_dir": str(args.output_dir),
        "seed": args.seed,

        "n_time_bands": args.n_time_bands,
        "num_time_bands": args.n_time_bands,
        "time_bands": args.n_time_bands,

        "feature_group": args.feature_group,
    }

    sig = inspect.signature(ExperimentConfig)
    allowed = set(sig.parameters.keys())
    kwargs = {k: v for k, v in requested.items() if k in allowed and v is not None}

    config = ExperimentConfig(**kwargs)

    # The actual band trainer uses fractional band_width/band_stride.
    # Match band_training/run.py behavior:
    #   --n_time_bands n => band_width = band_stride = 1/n
    # This creates n non-overlapping teacher-forced time bands.
    if args.n_time_bands is not None:
        bw = 1.0 / float(args.n_time_bands)

        if hasattr(config, "band_width"):
            setattr(config, "band_width", bw)
        if hasattr(config, "band_stride"):
            setattr(config, "band_stride", bw)

        # Store this too if a later config version exposes it.
        for name in ["n_time_bands", "num_time_bands", "time_bands"]:
            if hasattr(config, name):
                setattr(config, name, args.n_time_bands)

    if args.feature_group is not None and hasattr(config, "feature_group"):
        setattr(config, "feature_group", args.feature_group)

    # Required sanity checks.
    band_state = {}
    for name in ["band_width", "band_stride", "n_time_bands", "num_time_bands", "time_bands"]:
        if hasattr(config, name):
            band_state[name] = getattr(config, name)

    print("Band config fields:", band_state)

    if "band_width" not in band_state or "band_stride" not in band_state:
        print("WARNING: config has no band_width/band_stride attributes. Inspect band_training/config.py.")

    if args.feature_group is not None and not hasattr(config, "feature_group"):
        print("WARNING: config has no feature_group attribute. If the trainer hardcodes CovariateAdapter(),")
        print("then --feature_group may be ignored and we must patch trainer.py separately.")

    return config


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--experiment", choices=["cross_scene", "per_scene"], default="cross_scene")
    p.add_argument("--scene_id", default=None)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_G", type=float, default=0.005)
    p.add_argument("--lambda_B", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_time_bands", type=int, default=2)
    p.add_argument("--feature_group", default="topo_weather_omitted")

    args = p.parse_args()
    args.output_dir = str(Path(args.output_dir).expanduser())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "cross_scene":
        train_map, val_map, test_map = build_cross_scene_maps(manifest)
        exp_name = f"small_band_n{args.n_time_bands}_cross_lam{args.lambda_G:g}"
        scenes = list(train_map.keys())
        test_scenes = list(test_map.keys())
    else:
        if args.scene_id is None:
            raise RuntimeError("--scene_id is required for per_scene")
        train_map, val_map, test_map = build_per_scene_maps(manifest, args.scene_id)
        exp_name = f"small_band_n{args.n_time_bands}_per_scene_{args.scene_id}_lam{args.lambda_G:g}"
        scenes = [args.scene_id]
        test_scenes = []

    print("=" * 90)
    print("Small subset BAND manifest run")
    print("=" * 90)
    print("experiment:", args.experiment)
    print("scene_id:", args.scene_id)
    print("n_time_bands:", args.n_time_bands)
    print("lambda_G/lambda_B:", args.lambda_G, args.lambda_B)
    print("feature_group:", args.feature_group)
    print("output_dir:", output_dir)
    print("train scenes:", list(train_map.keys()))
    print("val scenes:", list(val_map.keys()))
    print("test scenes:", list(test_map.keys()))
    print("=" * 90)

    config = construct_config(args, manifest, scenes, test_scenes, exp_name)

    safe_json_dump(output_dir / "run_args.json", vars(args))
    safe_json_dump(output_dir / "manifest_snapshot.json", manifest)
    safe_json_dump(output_dir / "config_repr.json", {
        "config_type": str(type(config)),
        "config_dict": getattr(config, "__dict__", str(config)),
    })

    loader = Sim2RealFireLoader(manifest["data_root"])
    trainer = SceneTrainer(config, loader)

    # Required by the existing band trainer: __init__ leaves self.model=None.
    # The model is fully convolutional, but the original run.py still initializes it
    # explicitly before training. Use the first training scene grid/features.
    first_train_scene = next(iter(train_map.keys()))
    first_train_event = train_map[first_train_scene][0]
    first_data = loader.load_scenario(first_train_scene, first_train_event)
    elev = first_data["topography"].get("elevation")
    if elev is None:
        raise RuntimeError(f"No elevation grid found for {first_train_scene}/{first_train_event}")
    H, W = elev.shape

    # Band trainer setup_model signatures can vary across our edits.
    setup_sig = inspect.signature(trainer.setup_model)
    if "H" in setup_sig.parameters and "W" in setup_sig.parameters:
        trainer.setup_model(H, W)
    else:
        trainer.setup_model()

    print(f"Initialized band model for grid {H} x {W}")

    print("\nLoading selected train/val/test scenarios into RAM...")
    train_scenarios, train_skipped = collect_scenarios(trainer, train_map, "train")
    val_scenarios, val_skipped = collect_scenarios(trainer, val_map, "val")
    test_scenarios, test_skipped = collect_scenarios(trainer, test_map, "test")

    safe_json_dump(output_dir / "load_report.json", {
        "n_train": len(train_scenarios),
        "n_val": len(val_scenarios),
        "n_test": len(test_scenarios),
        "train_skipped": train_skipped,
        "val_skipped": val_skipped,
        "test_skipped": test_skipped,
    })

    if not train_scenarios:
        raise RuntimeError("No train scenarios loaded")
    if not val_scenarios:
        raise RuntimeError("No val scenarios loaded")
    if not test_scenarios:
        print("WARNING: no test scenarios loaded; training will still run, but test eval will be empty.")

    print("\nStarting existing band SceneTrainer.train(...)")
    t0 = time.time()
    history = trainer.train(
        train_scenarios,
        val_scenarios,
        verbose=True,
        scene_prefix=exp_name,
    )
    elapsed = time.time() - t0

    safe_json_dump(output_dir / "history.json", history)
    safe_json_dump(output_dir / "train_done.json", {
        "done": True,
        "elapsed_sec": elapsed,
    })

    print("\nTraining finished. Saving model before test evaluation...")
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config_dict": getattr(config, "__dict__", {}),
            "args": vars(args),
            "manifest": manifest,
        },
        output_dir / "final_model_before_test.pt",
    )

    print("\nEvaluating band-local test metrics...")
    band_agg, band_det = trainer.evaluate_bands(test_scenarios)
    safe_json_dump(output_dir / "test_metrics_band.json", band_agg)
    safe_json_dump(output_dir / "test_per_fire_band.json", band_det)

    print("\nEvaluating full-fire autonomous test metrics...")
    full_agg, full_det, preds = trainer.evaluate_full_fire(test_scenarios, save_predictions=True)
    safe_json_dump(output_dir / "test_metrics_full_fire.json", full_agg)
    safe_json_dump(output_dir / "test_per_fire_full_fire.json", full_det)

    if preds is not None:
        np.savez_compressed(output_dir / "test_predictions.npz", **preds)

    print("\nFINAL BAND-LOCAL TEST METRICS")
    print(json.dumps(band_agg, indent=2, default=str))

    print("\nFINAL FULL-FIRE AUTONOMOUS TEST METRICS")
    print(json.dumps(full_agg, indent=2, default=str))


if __name__ == "__main__":
    main()
