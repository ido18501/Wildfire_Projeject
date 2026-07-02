#!/usr/bin/env python3
"""
Verify time-band decomposition for one fire.

Checks:
  source_k = already burned by t_start
  target_k = pixels with t_start < T_gt <= t_end
  target relative times are in (0, band_width]
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from config import ExperimentConfig
from trainer import SceneTrainer
from experiments.wildfire.sim2real_loader import Sim2RealFireLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--event", default=None)
    p.add_argument("--n_time_bands", type=int, default=4)
    args = p.parse_args()

    if args.n_time_bands <= 0:
        raise ValueError("--n_time_bands must be positive")

    loader = Sim2RealFireLoader(args.data_root)
    if args.scene not in loader.scenes:
        raise ValueError(f"Scene not found: {args.scene}")

    event = args.event or loader.scenes[args.scene]["events"][0]

    cfg = ExperimentConfig(
        band_width=1.0 / args.n_time_bands,
        band_stride=1.0 / args.n_time_bands,
    )

    trainer = SceneTrainer(cfg, loader)
    scenario = trainer.prepare_scenario(args.scene, event)
    if scenario is None:
        raise RuntimeError("prepare_scenario returned None")

    true_T = torch.from_numpy(scenario["arrival_times"]).float()
    valid = torch.from_numpy(scenario["valid_mask"])
    finite_valid = valid & torch.isfinite(true_T)

    max_time = true_T[finite_valid].max().item()
    bands = trainer.generate_bands(max_time, overlapping=False)
    source_masks_K, valid_indices = trainer._build_band_sources(
        scenario, bands, true_T, valid
    )

    print("=" * 70)
    print("TIME BAND VERIFICATION")
    print("=" * 70)
    print(f"scene={args.scene}")
    print(f"event={event}")
    print(f"max_time={max_time:.6f}")
    print(f"n_time_bands={args.n_time_bands}")
    print(f"generated_bands={len(bands)}")
    print(f"valid_source_bands={len(valid_indices)}")
    print()

    total_target = 0
    for local_idx, band_idx in enumerate(valid_indices):
        t_start, t_end = bands[band_idx]
        src = source_masks_K[local_idx].bool()
        target = valid & (true_T > t_start) & (true_T <= t_end)

        if t_start < 1e-6:
            expected_src = torch.from_numpy(scenario["source_mask"]).bool()
        else:
            expected_src = (true_T <= t_start) & valid

        src_diff = (src != expected_src).sum().item()
        rel = true_T[target] - t_start

        rel_min = rel.min().item() if target.sum().item() else float("nan")
        rel_max = rel.max().item() if target.sum().item() else float("nan")

        print(
            f"band {band_idx:02d}: "
            f"[{t_start:.3f}, {t_end:.3f}] "
            f"source_pixels={src.sum().item():6d} "
            f"target_pixels={target.sum().item():6d} "
            f"rel_range=[{rel_min:.3f}, {rel_max:.3f}] "
            f"source_diff={src_diff}"
        )

        total_target += target.sum().item()

        if src_diff != 0:
            raise AssertionError(f"source mismatch in band {band_idx}")
        if target.sum().item() > 0:
            if rel_min <= 0:
                raise AssertionError(f"relative target time <= 0 in band {band_idx}")
            if rel_max > (t_end - t_start) + 1e-5:
                raise AssertionError(f"relative target time exceeds band width in band {band_idx}")

    print()
    print(f"total_target_pixels_across_bands={total_target}")
    print("VERIFY_TIME_BANDS_OK")


if __name__ == "__main__":
    main()
