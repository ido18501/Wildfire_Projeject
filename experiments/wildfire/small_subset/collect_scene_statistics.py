#!/usr/bin/env python3
import argparse
import json
import math
import inspect
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader,
    CovariateAdapter,
    extract_arrival_times,
    find_ignition_point,
)


def safe_float(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return x
    except Exception:
        return np.nan


def arr_stats(arr, prefix):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {}

    arr = arr.astype(float, copy=False)
    finite = np.isfinite(arr)
    vals = arr[finite]

    out = {
        f"{prefix}_finite_frac": float(finite.mean()) if finite.size else np.nan,
        f"{prefix}_n_finite": int(finite.sum()),
    }

    if vals.size == 0:
        return out

    out.update({
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": float(np.std(vals)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_p05": float(np.percentile(vals, 5)),
        f"{prefix}_p25": float(np.percentile(vals, 25)),
        f"{prefix}_p50": float(np.percentile(vals, 50)),
        f"{prefix}_p75": float(np.percentile(vals, 75)),
        f"{prefix}_p95": float(np.percentile(vals, 95)),
        f"{prefix}_max": float(np.max(vals)),
        f"{prefix}_range": float(np.max(vals) - np.min(vals)),
    })

    # Useful for categorical/discrete rasters.
    if vals.size <= 5_000_000:
        rounded = vals
        if np.all(np.abs(vals - np.round(vals)) < 1e-6):
            rounded = np.round(vals).astype(int)
            uniq, counts = np.unique(rounded, return_counts=True)
            probs = counts / counts.sum()
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            out.update({
                f"{prefix}_unique": int(len(uniq)),
                f"{prefix}_entropy": entropy,
                f"{prefix}_mode": float(uniq[np.argmax(counts)]),
                f"{prefix}_mode_frac": float(counts.max() / counts.sum()),
            })

    return out


def flatten_numeric(obj, prefix="raw"):
    """
    Recursively collect numeric scalars/arrays from a loaded scenario.
    Skips masks/timestamps because those are event outputs rather than covariates.
    """
    rows = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"masks", "timestamps"}:
                continue
            rows.extend(flatten_numeric(v, f"{prefix}.{k}"))
        return rows

    if isinstance(obj, (list, tuple)):
        # Small numeric list -> array stats. Otherwise recurse.
        try:
            arr = np.asarray(obj)
            if np.issubdtype(arr.dtype, np.number):
                d = arr_stats(arr, prefix)
                rows.append({"feature": prefix, **d})
                return rows
        except Exception:
            pass

        for i, v in enumerate(obj):
            rows.extend(flatten_numeric(v, f"{prefix}.{i}"))
        return rows

    try:
        arr = np.asarray(obj)
        if np.issubdtype(arr.dtype, np.number):
            d = arr_stats(arr, prefix)
            rows.append({"feature": prefix, **d})
    except Exception:
        pass

    return rows


def make_adapter(feature_group):
    """
    Different versions of the repo used slightly different CovariateAdapter signatures.
    This tries the likely options safely.
    """
    candidates = [
        {"feature_group": feature_group},
        {"group": feature_group},
        {"features": feature_group},
        {},
    ]
    for kwargs in candidates:
        try:
            return CovariateAdapter(**kwargs)
        except TypeError:
            continue
    return CovariateAdapter()


def tensor_to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def get_feature_names(adapter, kind, n):
    candidates = [
        f"{kind}_feature_names",
        f"{kind}_features",
        f"{kind}_names",
        f"{kind}_channels",
    ]
    for name in candidates:
        if hasattr(adapter, name):
            val = getattr(adapter, name)
            if isinstance(val, (list, tuple)) and len(val) == n:
                return list(map(str, val))
    return [f"{kind}_{i}" for i in range(n)]


def adapted_feature_stats(data, feature_group):
    rows = []
    try:
        adapter = make_adapter(feature_group)
        spatial, global_f = adapter(data)

        spatial = tensor_to_numpy(spatial)
        global_f = tensor_to_numpy(global_f)

        # Spatial should be C,H,W or H,W,C. Current repo usually uses C,H,W.
        if spatial.ndim == 3:
            if spatial.shape[0] <= 64:
                C = spatial.shape[0]
                names = get_feature_names(adapter, "spatial", C)
                for c in range(C):
                    d = arr_stats(spatial[c], f"adapted.{feature_group}.spatial.{names[c]}")
                    rows.append({
                        "feature_group": feature_group,
                        "kind": "spatial",
                        "channel": names[c],
                        **d,
                    })
            else:
                C = spatial.shape[-1]
                names = get_feature_names(adapter, "spatial", C)
                for c in range(C):
                    d = arr_stats(spatial[..., c], f"adapted.{feature_group}.spatial.{names[c]}")
                    rows.append({
                        "feature_group": feature_group,
                        "kind": "spatial",
                        "channel": names[c],
                        **d,
                    })

        global_f = np.asarray(global_f).reshape(-1)
        names = get_feature_names(adapter, "global", len(global_f))
        for i, v in enumerate(global_f):
            rows.append({
                "feature_group": feature_group,
                "kind": "global",
                "channel": names[i],
                f"adapted.{feature_group}.global.{names[i]}_value": safe_float(v),
            })

    except Exception as e:
        rows.append({
            "feature_group": feature_group,
            "kind": "ERROR",
            "channel": "ERROR",
            "error": str(e),
        })

    return rows


def event_stats(loader, scene_id, event_id, terrain_valid_pixels):
    data = loader.load_scenario(scene_id, event_id)
    arrival = extract_arrival_times(data["masks"], data["timestamps"])
    ignition = find_ignition_point(data["masks"])

    H, W = arrival.shape
    burned = np.isfinite(arrival)
    n_burned = int(burned.sum())

    out = {
        "scene_id": scene_id,
        "event_id": str(event_id),
        "H": H,
        "W": W,
        "n_burned": n_burned,
        "burned_frac_grid": n_burned / float(H * W),
        "burned_frac_terrain": n_burned / float(terrain_valid_pixels) if terrain_valid_pixels > 0 else np.nan,
        "n_masks": len(data.get("masks", [])),
        "n_timestamps": len(data.get("timestamps", [])),
    }

    if ignition is not None:
        iy, ix = ignition
        out.update({
            "ignition_y": int(iy),
            "ignition_x": int(ix),
            "ignition_y_norm": iy / max(H - 1, 1),
            "ignition_x_norm": ix / max(W - 1, 1),
            "ignition_edge_dist_norm": min(iy, ix, H - 1 - iy, W - 1 - ix) / max(min(H, W) - 1, 1),
        })
    else:
        out.update({
            "ignition_y": np.nan,
            "ignition_x": np.nan,
            "ignition_y_norm": np.nan,
            "ignition_x_norm": np.nan,
            "ignition_edge_dist_norm": np.nan,
        })

    if n_burned > 0:
        vals = arrival[burned].astype(float)
        max_t = float(np.max(vals))
        mean_t = float(np.mean(vals))
        std_t = float(np.std(vals))

        out.update({
            "arrival_max": max_t,
            "arrival_mean": mean_t,
            "arrival_std": std_t,
            "arrival_cv": std_t / max_t if max_t > 1e-9 else np.nan,
            "arrival_p25": float(np.percentile(vals, 25)),
            "arrival_p50": float(np.percentile(vals, 50)),
            "arrival_p75": float(np.percentile(vals, 75)),
        })

        for frac in [0.25, 0.50, 0.75]:
            out[f"burned_by_t{int(frac*100)}_frac"] = float(np.mean(vals <= frac * max_t)) if max_t > 1e-9 else np.nan

        ys, xs = np.where(burned)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_area = bbox_h * bbox_w
        out.update({
            "burn_bbox_h": bbox_h,
            "burn_bbox_w": bbox_w,
            "burn_bbox_area": bbox_area,
            "burn_bbox_area_frac_grid": bbox_area / float(H * W),
            "burn_compactness": n_burned / float(bbox_area) if bbox_area > 0 else np.nan,
        })
    else:
        for k in [
            "arrival_max", "arrival_mean", "arrival_std", "arrival_cv",
            "arrival_p25", "arrival_p50", "arrival_p75",
            "burned_by_t25_frac", "burned_by_t50_frac", "burned_by_t75_frac",
            "burn_bbox_h", "burn_bbox_w", "burn_bbox_area",
            "burn_bbox_area_frac_grid", "burn_compactness",
        ]:
            out[k] = np.nan

    # Add global adapted features for both groups for this event.
    for fg in ["baseline", "topo_weather_omitted"]:
        try:
            adapter = make_adapter(fg)
            _, global_f = adapter(data)
            global_f = tensor_to_numpy(global_f).reshape(-1)
            names = get_feature_names(adapter, "global", len(global_f))
            for i, v in enumerate(global_f):
                out[f"{fg}_global_{names[i]}"] = safe_float(v)
        except Exception:
            pass

    return out


def aggregate_event_rows(event_df):
    metric_cols = [
        c for c in event_df.columns
        if c not in {"scene_id", "event_id"}
        and pd.api.types.is_numeric_dtype(event_df[c])
    ]

    rows = []
    for scene_id, g in event_df.groupby("scene_id"):
        row = {"scene_id": scene_id, "n_events_sampled": len(g)}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                continue
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=0))
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_p50"] = float(vals.median())
            row[f"{c}_max"] = float(vals.max())
            if abs(vals.mean()) > 1e-12:
                row[f"{c}_cv"] = float(vals.std(ddof=0) / abs(vals.mean()))
        rows.append(row)

    return pd.DataFrame(rows)


def zscore_table(scene_summary):
    numeric_cols = [
        c for c in scene_summary.columns
        if c != "scene_id" and pd.api.types.is_numeric_dtype(scene_summary[c])
    ]

    rows = []
    for c in numeric_cols:
        vals = pd.to_numeric(scene_summary[c], errors="coerce")
        mean = vals.mean()
        std = vals.std(ddof=0)
        if not np.isfinite(std) or std < 1e-12:
            continue
        for _, r in scene_summary.iterrows():
            v = r[c]
            if pd.isna(v):
                continue
            z = (v - mean) / std
            rows.append({
                "scene_id": r["scene_id"],
                "metric": c,
                "value": float(v),
                "z": float(z),
                "abs_z": float(abs(z)),
                "mean_all_scenes": float(mean),
                "std_all_scenes": float(std),
            })

    return pd.DataFrame(rows).sort_values(["scene_id", "abs_z"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "sim2real_fire_data/simulation_data"))
    ap.add_argument("--manifest", default=str(Path.home() / "wildfire_small_subset_manifest.json"))
    ap.add_argument("--max-events-per-scene", type=int, default=200,
                    help="Use 0 for all events. Default 200 for fast diagnosis.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    loader = Sim2RealFireLoader(str(data_root))

    out_dir = Path(args.out_dir) if args.out_dir else Path("small_runs") / f"scene_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    manifest = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    scene_rows = []
    event_rows = []
    raw_feature_rows = []
    adapted_rows = []

    for scene_id, info in sorted(loader.scenes.items()):
        events = list(info.get("events", []))
        n_events_total = len(events)
        if not events:
            continue

        print(f"[scene] {scene_id}: total_events={n_events_total}")

        # Load first available event to inspect static scene geometry/features.
        first = events[0]
        data0 = loader.load_scenario(scene_id, first)

        # Get geometry from elevation if possible, otherwise arrival.
        H = W = None
        terrain_valid_pixels = np.nan
        terrain_valid_frac = np.nan

        elev = None
        try:
            elev = data0.get("topography", {}).get("elevation", None)
            if elev is not None:
                elev = np.asarray(elev)
                H, W = elev.shape
                finite_elev = np.isfinite(elev)
                terrain_valid_pixels = int(finite_elev.sum())
                terrain_valid_frac = float(finite_elev.mean())
        except Exception:
            elev = None

        if H is None or W is None:
            arr0 = extract_arrival_times(data0["masks"], data0["timestamps"])
            H, W = arr0.shape
            terrain_valid_pixels = H * W
            terrain_valid_frac = 1.0

        role = "not_in_small_manifest"
        if manifest:
            # Robustly search for the scene in common manifest formats.
            text = json.dumps(manifest)
            if scene_id in text:
                role = "in_small_manifest"

        scene_row = {
            "scene_id": scene_id,
            "n_events_total": n_events_total,
            "H": H,
            "W": W,
            "grid_cells": H * W,
            "aspect_ratio_W_over_H": W / H if H else np.nan,
            "terrain_valid_pixels": terrain_valid_pixels,
            "terrain_valid_frac": terrain_valid_frac,
            "small_manifest_role": role,
        }

        # Add core elevation stats to compact scene summary.
        if elev is not None:
            e_stats = arr_stats(elev, "elevation")
            for k, v in e_stats.items():
                if k.endswith(("_mean", "_std", "_min", "_max", "_range", "_p50")):
                    scene_row[k] = v

        scene_rows.append(scene_row)

        # Long raw feature stats from first event.
        for rr in flatten_numeric(data0, prefix="raw"):
            rr["scene_id"] = scene_id
            raw_feature_rows.append(rr)

        # Long adapted spatial/global stats from first event.
        for fg in ["baseline", "topo_weather_omitted"]:
            for ar in adapted_feature_stats(data0, fg):
                ar["scene_id"] = scene_id
                adapted_rows.append(ar)

        # Sample event list.
        if args.max_events_per_scene and args.max_events_per_scene > 0 and len(events) > args.max_events_per_scene:
            idx = rng.choice(len(events), size=args.max_events_per_scene, replace=False)
            sampled_events = [events[i] for i in sorted(idx)]
        else:
            sampled_events = events

        for event_id in sampled_events:
            try:
                er = event_stats(loader, scene_id, event_id, terrain_valid_pixels)
                event_rows.append(er)
            except Exception as e:
                event_rows.append({
                    "scene_id": scene_id,
                    "event_id": str(event_id),
                    "error": str(e),
                })

    scene_df = pd.DataFrame(scene_rows)
    event_df = pd.DataFrame(event_rows)
    raw_df = pd.DataFrame(raw_feature_rows)
    adapted_df = pd.DataFrame(adapted_rows)

    event_agg = aggregate_event_rows(event_df)

    scene_summary = scene_df.merge(event_agg, on="scene_id", how="left")

    # A compact table with the most important diagnostics.
    preferred_cols = [
        "scene_id",
        "small_manifest_role",
        "n_events_total",
        "n_events_sampled",
        "H",
        "W",
        "grid_cells",
        "terrain_valid_pixels",
        "terrain_valid_frac",
        "aspect_ratio_W_over_H",
        "elevation_mean",
        "elevation_std",
        "elevation_range",
        "n_burned_mean",
        "n_burned_std",
        "burned_frac_terrain_mean",
        "burned_frac_terrain_std",
        "burned_frac_terrain_cv",
        "arrival_max_mean",
        "arrival_max_std",
        "arrival_max_cv",
        "arrival_cv_mean",
        "burned_by_t25_frac_mean",
        "burned_by_t50_frac_mean",
        "burned_by_t75_frac_mean",
        "burn_compactness_mean",
        "ignition_x_norm_mean",
        "ignition_y_norm_mean",
        "ignition_edge_dist_norm_mean",
        "n_masks_mean",
    ]
    compact_cols = [c for c in preferred_cols if c in scene_summary.columns]
    compact = scene_summary[compact_cols].copy()

    zscores = zscore_table(scene_summary)
    z0014 = zscores[zscores["scene_id"] == "0014_00426"].sort_values("abs_z", ascending=False)

    # Save files.
    scene_summary.to_csv(out_dir / "scene_summary_wide.csv", index=False)
    compact.to_csv(out_dir / "scene_summary_compact.csv", index=False)
    event_df.to_csv(out_dir / "event_stats_sampled.csv", index=False)
    raw_df.to_csv(out_dir / "raw_feature_stats_long.csv", index=False)
    adapted_df.to_csv(out_dir / "adapted_feature_stats_long.csv", index=False)
    zscores.to_csv(out_dir / "scene_metric_zscores.csv", index=False)
    z0014.to_csv(out_dir / "scene_0014_outliers.csv", index=False)

    # Markdown summaries.
    md = []
    md.append(f"# Scene statistics report\n")
    md.append(f"Data root: `{data_root}`\n")
    md.append(f"Max events per scene: `{args.max_events_per_scene}`\n")
    md.append("\n## Compact scene summary\n")
    md.append(compact.to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Top 0014 outlier metrics by absolute z-score\n")
    md.append(z0014.head(40).to_markdown(index=False, floatfmt=".4f"))
    (out_dir / "REPORT.md").write_text("\n".join(md))

    print()
    print(f"OUT_DIR={out_dir}")
    print()
    print("COMPACT SUMMARY")
    print("=" * 160)
    print(compact.to_string(index=False, max_cols=200))
    print()
    print("TOP 0014 OUTLIERS")
    print("=" * 160)
    if len(z0014):
        print(z0014.head(40).to_string(index=False))
    else:
        print("No 0014 outliers found, or scene id not present.")
    print()
    print("Files written:")
    for name in [
        "REPORT.md",
        "scene_summary_compact.csv",
        "scene_summary_wide.csv",
        "event_stats_sampled.csv",
        "raw_feature_stats_long.csv",
        "adapted_feature_stats_long.csv",
        "scene_metric_zscores.csv",
        "scene_0014_outliers.csv",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
