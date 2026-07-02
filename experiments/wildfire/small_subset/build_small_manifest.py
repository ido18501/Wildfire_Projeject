#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np

from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader,
    extract_arrival_times,
)


DEFAULT_SCENES = [
    "0001_02614",
    "0002_00714",
    "0018_02440",
    "0014_00426",
    "0005_00725",
]

DEFAULT_TRAIN_SCENES = [
    "0001_02614",
    "0002_00714",
    "0018_02440",
]

DEFAULT_VAL_SCENES = [
    "0014_00426",
]

DEFAULT_TEST_SCENES = [
    "0005_00725",
]


def is_valid_event(loader, scene_id, event_id, min_valid_pixels):
    try:
        data = loader.load_scenario(scene_id, event_id)
        arrival = extract_arrival_times(data["masks"], data["timestamps"])
        valid = np.isfinite(arrival)
        return int(valid.sum()) >= min_valid_pixels
    except Exception:
        return False


def split_events(events, train_ratio=0.70, val_ratio=0.15):
    n = len(events)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train = events[:n_train]
    val = events[n_train:n_train + n_val]
    test = events[n_train + n_val:]

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(Path.home() / "sim2real_fire_data" / "simulation_data"))
    parser.add_argument("--output", type=str, default=str(Path.home() / "wildfire_small_subset_manifest.json"))
    parser.add_argument("--events_per_scene", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_valid_pixels", type=int, default=500)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--train_scenes", nargs="+", default=DEFAULT_TRAIN_SCENES)
    parser.add_argument("--val_scenes", nargs="+", default=DEFAULT_VAL_SCENES)
    parser.add_argument("--test_scenes", nargs="+", default=DEFAULT_TEST_SCENES)
    parser.add_argument("--feature_group", type=str, default="topo_weather_omitted")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    loader = Sim2RealFireLoader(args.data_root)

    manifest = {
        "purpose": "controlled_small_subset_convergence_study",
        "paper_reproduction": False,
        "seed": args.seed,
        "data_root": args.data_root,
        "feature_group": args.feature_group,
        "events_per_scene_requested": args.events_per_scene,
        "min_valid_pixels": args.min_valid_pixels,
        "scenes": {},
        "cross_scene": {
            "train_scenes": args.train_scenes,
            "val_scenes": args.val_scenes,
            "test_scenes": args.test_scenes,
        },
        "per_scene": {
            "split_ratio": "70/15/15",
        },
    }

    for scene_id in args.scenes:
        if scene_id not in loader.scenes:
            raise RuntimeError(f"Scene {scene_id} not found in loader.scenes")

        all_events = list(loader.scenes[scene_id]["events"])
        rng.shuffle(all_events)

        chosen = []
        skipped = 0

        for event_id in all_events:
            if len(chosen) >= args.events_per_scene:
                break

            if args.validate:
                if not is_valid_event(loader, scene_id, event_id, args.min_valid_pixels):
                    skipped += 1
                    continue

            chosen.append(event_id)

        if len(chosen) < args.events_per_scene:
            print(
                f"WARNING: scene {scene_id} only yielded {len(chosen)} valid events "
                f"out of requested {args.events_per_scene}"
            )

        train, val, test = split_events(chosen)

        manifest["scenes"][scene_id] = {
            "selected": chosen,
            "train": train,
            "val": val,
            "test": test,
            "n_selected": len(chosen),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_available": len(all_events),
            "n_skipped_validation": skipped,
        }

        print(
            f"{scene_id}: selected={len(chosen)} "
            f"train={len(train)} val={len(val)} test={len(test)} "
            f"available={len(all_events)} skipped={skipped}"
        )

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest to: {out}")


if __name__ == "__main__":
    main()
