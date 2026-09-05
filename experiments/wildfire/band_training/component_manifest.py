"""
Sample-manifest loader for component-augmented band training.

Purpose:
  Convert rows from final_component_training_manifests_*/.../*.csv into
  SceneTrainer-compatible scenario dicts.

Important:
  - Covariates still come from the original Sim2Real event.
  - T/source/valid are overridden from the saved NPZ sample.
  - For normal_full samples, this also uses the exact source saved in NPZ,
    not the older centroid/3x3 source fallback.
"""

import csv
from pathlib import Path

import numpy as np


def _repo_root():
    return Path(__file__).resolve().parents[3]


def read_manifest_csv(path):
    path = Path(path)
    with path.open() as f:
        return list(csv.DictReader(f))


def resolve_npz_path(npz_path, manifest_csv=None):
    p = Path(npz_path).expanduser()

    if p.is_absolute() and p.exists():
        return p

    candidates = []

    # Most of our manifests store paths relative to repo root.
    candidates.append(_repo_root() / p)

    # Also allow relative to the manifest file.
    if manifest_csv is not None:
        candidates.append(Path(manifest_csv).resolve().parent / p)

    # Also allow current working directory.
    candidates.append(Path.cwd() / p)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve npz_path={npz_path}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _copy_base_scenario(base):
    # Shallow copy is enough: we replace mutable target/source/valid arrays.
    return dict(base)


def load_scenarios_from_sample_manifest(
    manifest_csv,
    trainer,
    max_samples=None,
    strict=True,
    verbose=True,
):
    """
    Load sample-level scenarios.

    Args:
      manifest_csv:
        CSV with at least scene,event,npz_path,sample_type.
      trainer:
        SceneTrainer instance. We call trainer.prepare_scenario(scene,event)
        to load covariates/features.
      max_samples:
        Optional smoke-test cap.
      strict:
        If True, raise on first bad row. If False, skip bad rows.
    """
    rows = read_manifest_csv(manifest_csv)
    if max_samples is not None:
        rows = rows[:max_samples]

    base_cache = {}
    scenarios = []
    skipped = []

    for idx, row in enumerate(rows):
        scene = row["scene"]
        event = row["event"]
        sample_type = row.get("sample_type", "")
        sample_id = row.get("sample_id", f"row{idx:06d}")

        try:
            cache_key = (scene, event)
            if cache_key not in base_cache:
                base = trainer.prepare_scenario(scene, event)
                if base is None:
                    raise RuntimeError(f"prepare_scenario returned None for {scene}/{event}")
                base_cache[cache_key] = base

            scenario = _copy_base_scenario(base_cache[cache_key])

            npz_path = resolve_npz_path(row["npz_path"], manifest_csv=manifest_csv)

            with np.load(npz_path) as z:
                T_target = z["T_target"].astype(np.float32)
                source_mask = z["source_mask"].astype(bool)
                valid_mask = z["valid_mask"].astype(bool)

                # Optional masks are useful for diagnostics later.
                for optional_key in [
                    "final_mask",
                    "growth_mask",
                    "domain1_mask",
                    "domain2_mask",
                    "domain_union_mask",
                ]:
                    if optional_key in z:
                        scenario[optional_key] = z[optional_key].astype(bool)

            valid_mask = valid_mask & np.isfinite(T_target)

            if T_target.shape != scenario["arrival_times"].shape:
                raise ValueError(
                    f"Shape mismatch for {sample_id}: "
                    f"T_target={T_target.shape}, original={scenario['arrival_times'].shape}"
                )

            if source_mask.sum() < 1:
                raise ValueError(f"No source pixels for {sample_id}")

            if valid_mask.sum() < 10:
                raise ValueError(f"Too few valid pixels for {sample_id}")

            scenario["arrival_times"] = T_target
            scenario["source_mask"] = source_mask
            scenario["valid_mask"] = valid_mask

            # Preserve both base event identity and sample identity.
            scenario["scene_id"] = scene
            scenario["base_event_id"] = event
            scenario["event_id"] = sample_id
            scenario["sample_id"] = sample_id
            scenario["sample_type"] = sample_type
            scenario["source_mode"] = row.get("source_mode", "")
            scenario["stop_reason"] = row.get("stop_reason", "")
            scenario["split"] = row.get("split", "")
            scenario["npz_path"] = str(npz_path)

            scenarios.append(scenario)

        except Exception as e:
            msg = {
                "row_index": idx,
                "scene": scene,
                "event": event,
                "sample_id": sample_id,
                "error": repr(e),
            }
            skipped.append(msg)
            if strict:
                raise RuntimeError(f"Failed loading row {idx}: {msg}") from e

    if verbose:
        print(f"[component_manifest] {manifest_csv}")
        print(f"  rows:      {len(rows)}")
        print(f"  scenarios: {len(scenarios)}")
        print(f"  skipped:   {len(skipped)}")
        print(f"  unique base events loaded: {len(base_cache)}")

    return scenarios, skipped
