#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, os, re, sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

EVENT_COLS = [
    "group","run","scene_id","event_id","split","selector","selected_epoch",
    "connectivity","mask_mode",
    "gt_n_components","pred_n_components","component_count_delta",
    "gt_total_area","pred_total_area",
    "gt_seeded_area","gt_unseeded_area","gt_unseeded_area_frac",
    "gt_n_seeded_components","gt_n_unseeded_components",
    "gt_largest_component_area","gt_largest_unseeded_area","gt_largest_unseeded_area_frac",
    "gt_unseeded_components_missed","gt_unseeded_missed_area","gt_unseeded_missed_area_frac",
    "recall_seeded_area","recall_unseeded_area",
    "iou_seeded_area","iou_unseeded_area",
    "pred_area_overlapping_unseeded_gt","pred_area_overlapping_unseeded_gt_frac",
    "full_iou","full_corr","full_relrmse","full_rmse","full_mae",
    "array_file","source_mask_status",
]

GT_COLS = [
    "group","run","scene_id","event_id","split","selector","selected_epoch",
    "connectivity","mask_mode","component_id",
    "area","area_frac_of_gt",
    "overlaps_source","source_overlap_pixels",
    "earliest_true_T","latest_true_T","mean_true_T",
    "bbox_ymin","bbox_ymax","bbox_xmin","bbox_xmax",
    "centroid_y","centroid_x",
    "pred_overlap_pixels","pred_recall","is_missed_by_pred",
]

PRED_COLS = [
    "group","run","scene_id","event_id","split","selector","selected_epoch",
    "connectivity","mask_mode","component_id",
    "area","area_frac_of_pred",
    "bbox_ymin","bbox_ymax","bbox_xmin","bbox_xmax",
    "centroid_y","centroid_x",
    "overlaps_any_gt","overlaps_unseeded_gt",
    "overlap_any_gt_pixels","overlap_unseeded_gt_pixels",
]

SUM_COLS = [
    "group","run","scene_id","connectivity","mask_mode","n_events",
    "mean_gt_components","mean_pred_components",
    "median_gt_components","median_pred_components",
    "frac_events_gt_multi_component","frac_events_pred_multi_component",
    "mean_gt_unseeded_area_frac","median_gt_unseeded_area_frac",
    "frac_events_with_unseeded_gt",
    "mean_recall_seeded_area","mean_recall_unseeded_area",
    "mean_iou_seeded_area","mean_iou_unseeded_area",
    "mean_full_iou","mean_full_corr","mean_full_relrmse",
]

TRUE_KEYS = ["true_T","gt_T","T_true","T_obs","target_T","arrival_times","true_arrival_times","y_true","gt"]
PRED_KEYS = ["pred_T","T_pred","predicted_T","prediction","pred","y_pred"]
VALID_KEYS = ["valid_mask","valid","mask","burned_mask","obs_mask"]
SOURCE_KEYS = ["source_mask","ignition_mask","initial_source_mask","src_mask"]
SCENE_KEYS = ["scene_id","scene","scene_ids"]
EVENT_KEYS = ["event_id","scenario_id","fire_id","event","events"]


def xp(p):
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def first_key(npz, keys):
    for k in keys:
        if k in npz:
            return k
    lower = {k.lower(): k for k in npz.files}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def scalar_or_none(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return None


def finite_pred(a, inf_threshold):
    return np.isfinite(a) & (a < inf_threshold)


def label_components(mask, connectivity):
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    comps = []

    if connectivity == 4:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    cid = 0
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or labels[y, x] != 0:
                continue

            cid += 1
            q = deque([(y, x)])
            labels[y, x] = cid
            ys, xs = [], []

            while q:
                cy, cx = q.popleft()
                ys.append(cy)
                xs.append(cx)

                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = cid
                        q.append((ny, nx))

            ya = np.asarray(ys)
            xa = np.asarray(xs)
            comps.append({
                "component_id": cid,
                "area": int(len(ys)),
                "bbox_ymin": int(ya.min()),
                "bbox_ymax": int(ya.max()),
                "bbox_xmin": int(xa.min()),
                "bbox_xmax": int(xa.max()),
                "centroid_y": float(ya.mean()),
                "centroid_x": float(xa.mean()),
            })

    return labels, comps


def infer_source(true_T, valid):
    v = valid & np.isfinite(true_T)
    src = np.zeros_like(v, dtype=bool)
    if not v.any():
        return src, "missing_empty"

    tmin = np.nanmin(true_T[v])
    src = v & np.isclose(true_T, tmin, atol=1e-6)

    if not src.any():
        y, x = np.unravel_index(np.nanargmin(np.where(v, true_T, np.inf)), true_T.shape)
        src[max(0, y-1):y+2, max(0, x-1):x+2] = True
        src &= v

    return src, "inferred_from_min_true_T"


def mask_for_mode(true_T, pred_T, valid, mode, inf_threshold):
    gt_final = valid & np.isfinite(true_T)
    pred_final = valid & finite_pred(pred_T, inf_threshold)

    if mode == "final":
        return gt_final, pred_final

    m = re.fullmatch(r"t(\d+)", mode)
    if not m:
        raise ValueError(f"Unknown mask mode: {mode}")

    if not gt_final.any():
        return gt_final, pred_final & False

    frac = int(m.group(1)) / 100.0
    thr = float(np.nanmax(true_T[gt_final])) * frac

    return gt_final & (true_T <= thr), pred_final & (pred_T <= thr)


def safe_iou(a, b):
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else float("nan")


def safe_recall(region, pred):
    denom = int(region.sum())
    return float((region & pred).sum() / denom) if denom else float("nan")


def corr_rmse(pred_T, true_T, valid):
    v = valid & np.isfinite(true_T) & finite_pred(pred_T, 1e5)

    if int(v.sum()) < 2:
        return {
            "full_corr": float("nan"),
            "full_rmse": float("nan"),
            "full_relrmse": float("nan"),
            "full_mae": float("nan"),
        }

    p = pred_T[v].astype(float)
    t = true_T[v].astype(float)

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    denom = float(np.nanmax(t))
    rel = rmse / denom if np.isfinite(denom) and denom > 1e-9 else float("nan")
    corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-9 and np.std(t) > 1e-9 else float("nan")

    return {
        "full_corr": corr,
        "full_rmse": rmse,
        "full_relrmse": rel,
        "full_mae": mae,
    }


def analyze_one(meta, true_T, pred_T, valid, source, connectivity, mask_mode, inf_threshold):
    gt_mask, pred_mask = mask_for_mode(true_T, pred_T, valid, mask_mode, inf_threshold)

    gt_labels, gt_comps = label_components(gt_mask, connectivity)
    pred_labels, pred_comps = label_components(pred_mask, connectivity)

    total_gt = int(gt_mask.sum())
    total_pred = int(pred_mask.sum())

    seeded_mask = np.zeros_like(gt_mask, dtype=bool)
    unseeded_mask = np.zeros_like(gt_mask, dtype=bool)

    gt_rows = []
    pred_rows = []

    seeded_area = 0
    unseeded_area = 0
    largest_unseeded = 0
    n_seeded = 0
    n_unseeded = 0
    missed_unseeded_count = 0
    missed_unseeded_area = 0

    for comp in gt_comps:
        cm = gt_labels == comp["component_id"]
        overlaps_source = bool((cm & source).any())
        source_overlap_pixels = int((cm & source).sum())
        pred_overlap_pixels = int((cm & pred_mask).sum())

        if overlaps_source:
            seeded_mask |= cm
            n_seeded += 1
            seeded_area += comp["area"]
        else:
            unseeded_mask |= cm
            n_unseeded += 1
            unseeded_area += comp["area"]
            largest_unseeded = max(largest_unseeded, comp["area"])
            if pred_overlap_pixels == 0:
                missed_unseeded_count += 1
                missed_unseeded_area += comp["area"]

        vals = true_T[cm & np.isfinite(true_T)]

        gt_rows.append({
            **meta,
            **comp,
            "area_frac_of_gt": comp["area"] / total_gt if total_gt else float("nan"),
            "overlaps_source": overlaps_source,
            "source_overlap_pixels": source_overlap_pixels,
            "earliest_true_T": float(np.nanmin(vals)) if vals.size else float("nan"),
            "latest_true_T": float(np.nanmax(vals)) if vals.size else float("nan"),
            "mean_true_T": float(np.nanmean(vals)) if vals.size else float("nan"),
            "pred_overlap_pixels": pred_overlap_pixels,
            "pred_recall": pred_overlap_pixels / comp["area"] if comp["area"] else float("nan"),
            "is_missed_by_pred": pred_overlap_pixels == 0,
        })

    for comp in pred_comps:
        cm = pred_labels == comp["component_id"]
        overlap_any_gt = int((cm & gt_mask).sum())
        overlap_unseeded_gt = int((cm & unseeded_mask).sum())

        pred_rows.append({
            **meta,
            **comp,
            "area_frac_of_pred": comp["area"] / total_pred if total_pred else float("nan"),
            "overlaps_any_gt": overlap_any_gt > 0,
            "overlaps_unseeded_gt": overlap_unseeded_gt > 0,
            "overlap_any_gt_pixels": overlap_any_gt,
            "overlap_unseeded_gt_pixels": overlap_unseeded_gt,
        })

    seeded_iou = safe_iou(pred_mask & (seeded_mask | ~gt_mask), seeded_mask) if seeded_mask.any() else float("nan")
    unseeded_iou = safe_iou(pred_mask & (unseeded_mask | ~gt_mask), unseeded_mask) if unseeded_mask.any() else float("nan")

    event = {
        **meta,
        "gt_n_components": len(gt_comps),
        "pred_n_components": len(pred_comps),
        "component_count_delta": len(pred_comps) - len(gt_comps),
        "gt_total_area": total_gt,
        "pred_total_area": total_pred,
        "gt_seeded_area": seeded_area,
        "gt_unseeded_area": unseeded_area,
        "gt_unseeded_area_frac": unseeded_area / total_gt if total_gt else float("nan"),
        "gt_n_seeded_components": n_seeded,
        "gt_n_unseeded_components": n_unseeded,
        "gt_largest_component_area": max([c["area"] for c in gt_comps], default=0),
        "gt_largest_unseeded_area": largest_unseeded,
        "gt_largest_unseeded_area_frac": largest_unseeded / total_gt if total_gt else float("nan"),
        "gt_unseeded_components_missed": missed_unseeded_count,
        "gt_unseeded_missed_area": missed_unseeded_area,
        "gt_unseeded_missed_area_frac": missed_unseeded_area / unseeded_area if unseeded_area else float("nan"),
        "recall_seeded_area": safe_recall(seeded_mask, pred_mask),
        "recall_unseeded_area": safe_recall(unseeded_mask, pred_mask),
        "iou_seeded_area": seeded_iou,
        "iou_unseeded_area": unseeded_iou,
        "pred_area_overlapping_unseeded_gt": int((pred_mask & unseeded_mask).sum()),
        "pred_area_overlapping_unseeded_gt_frac": int((pred_mask & unseeded_mask).sum()) / total_pred if total_pred else float("nan"),
        "full_iou": safe_iou(gt_mask, pred_mask),
        **corr_rmse(pred_T, true_T, valid),
    }

    return event, gt_rows, pred_rows


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float) and not math.isfinite(v):
                    v = ""
                clean[c] = v
            w.writerow(clean)


def selected_epoch(run_path, selector):
    for name in ["test_results_by_selector.json", "results_by_selector.json", "summary.json", "metadata.json", "config.json"]:
        p = run_path / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        for key in [selector, "full_corr", "best_full_corr"]:
            obj = data.get(key) if isinstance(data, dict) else None
            if isinstance(obj, dict):
                for epoch_key in ["selected_epoch", "epoch", "best_epoch"]:
                    if epoch_key in obj:
                        return obj[epoch_key]

        for epoch_key in ["selected_epoch", "epoch", "best_epoch"]:
            if isinstance(data, dict) and epoch_key in data:
                return data[epoch_key]

    return ""


def group_name(run_path):
    s = str(run_path).lower()
    if "topo" in s or "band2_paperstyle_corr_20260608" in s:
        return "topo_weather_omitted"
    if "baseline" in s:
        return "baseline"
    return run_path.parent.name


def looks_prediction_file(path):
    n = path.name.lower()
    return path.suffix == ".npz" and any(t in n for t in ["pred", "test", "eval", "arrays", "inference"])


def find_npz(run_path):
    files = sorted([p for p in run_path.rglob("*.npz") if looks_prediction_file(p)])
    return files if files else sorted(run_path.rglob("*.npz"))


def split_events_from_npz(path, run_path, args):
    z = np.load(path, allow_pickle=True)

    true_key = first_key(z, TRUE_KEYS)
    pred_key = first_key(z, PRED_KEYS)

    if not true_key or not pred_key:
        return []

    true = np.asarray(z[true_key])
    pred = np.asarray(z[pred_key])

    if true.shape != pred.shape:
        return []

    batched = true.ndim == 3
    n = true.shape[0] if batched else 1

    valid_key = first_key(z, VALID_KEYS)
    source_key = first_key(z, SOURCE_KEYS)
    scene_key = first_key(z, SCENE_KEYS)
    event_key = first_key(z, EVENT_KEYS)

    events = []

    for i in range(n):
        true_T = true[i] if batched else true
        pred_T = pred[i] if batched else pred

        if true_T.ndim != 2 or pred_T.ndim != 2:
            continue

        if valid_key:
            valid_arr = np.asarray(z[valid_key])
            valid = valid_arr[i] if valid_arr.ndim == 3 else valid_arr
            valid = np.asarray(valid, dtype=bool)
        else:
            valid = np.isfinite(true_T)

        if source_key:
            source_arr = np.asarray(z[source_key])
            source = source_arr[i] if source_arr.ndim == 3 else source_arr
            source = np.asarray(source, dtype=bool)
            source_status = "loaded"
        else:
            source, source_status = infer_source(true_T, valid)
            if args.strict_source_mask:
                raise RuntimeError(f"{path} has no source_mask key. Available keys: {list(z.files)}")

        scene_id = None
        event_id = None

        if scene_key:
            a = np.asarray(z[scene_key])
            scene_id = a[i] if a.ndim > 0 and len(a) == n else scalar_or_none(a)

        if event_key:
            a = np.asarray(z[event_key])
            event_id = a[i] if a.ndim > 0 and len(a) == n else scalar_or_none(a)

        if scene_id is None:
            # Do NOT infer scene_id from the full run path, because timestamps like
            # 20260620_102205 look like scene IDs. If the npz lacks scene metadata,
            # leave it explicit instead of creating a fake scene.
            scene_id = "unknown_scene"

        if event_id is None:
            event_id = path.stem if not batched else f"{path.stem}_{i:05d}"

        events.append({
            "scene_id": str(scene_id),
            "event_id": str(event_id),
            "true_T": true_T.astype(float),
            "pred_T": pred_T.astype(float),
            "valid": valid,
            "source": source,
            "array_file": str(path.relative_to(run_path)),
            "source_mask_status": source_status,
        })

    return events


def inspect_runs(runs):
    for r in runs:
        rp = xp(r)
        print("\n" + "=" * 100)
        print(rp)

        if not rp.exists():
            print("MISSING")
            continue

        files = find_npz(rp)
        print(f"npz files found: {len(files)}")

        for p in files[:50]:
            try:
                z = np.load(p, allow_pickle=True)
                print(" ", p.relative_to(rp), "keys=", list(z.files))
            except Exception as e:
                print(" ", p.relative_to(rp), "ERROR:", e)


def finite_values(vals):
    out = []
    for v in vals:
        try:
            f = float(v)
            if math.isfinite(f):
                out.append(f)
        except Exception:
            pass
    return np.asarray(out, dtype=float)


def mean(vals):
    v = finite_values(vals)
    return float(v.mean()) if v.size else float("nan")


def median(vals):
    v = finite_values(vals)
    return float(np.median(v)) if v.size else float("nan")


def aggregate(events, by_scene=False):
    groups = defaultdict(list)

    for e in events:
        key = (
            e["group"],
            e["run"],
            e["scene_id"] if by_scene else "",
            e["connectivity"],
            e["mask_mode"],
        )
        groups[key].append(e)

    rows = []

    for (group, run, scene_id, conn, mode), xs in sorted(groups.items()):
        rows.append({
            "group": group,
            "run": run,
            "scene_id": scene_id,
            "connectivity": conn,
            "mask_mode": mode,
            "n_events": len(xs),
            "mean_gt_components": mean([x["gt_n_components"] for x in xs]),
            "mean_pred_components": mean([x["pred_n_components"] for x in xs]),
            "median_gt_components": median([x["gt_n_components"] for x in xs]),
            "median_pred_components": median([x["pred_n_components"] for x in xs]),
            "frac_events_gt_multi_component": mean([x["gt_n_components"] > 1 for x in xs]),
            "frac_events_pred_multi_component": mean([x["pred_n_components"] > 1 for x in xs]),
            "mean_gt_unseeded_area_frac": mean([x["gt_unseeded_area_frac"] for x in xs]),
            "median_gt_unseeded_area_frac": median([x["gt_unseeded_area_frac"] for x in xs]),
            "frac_events_with_unseeded_gt": mean([x["gt_n_unseeded_components"] > 0 for x in xs]),
            "mean_recall_seeded_area": mean([x["recall_seeded_area"] for x in xs]),
            "mean_recall_unseeded_area": mean([x["recall_unseeded_area"] for x in xs]),
            "mean_iou_seeded_area": mean([x["iou_seeded_area"] for x in xs]),
            "mean_iou_unseeded_area": mean([x["iou_unseeded_area"] for x in xs]),
            "mean_full_iou": mean([x["full_iou"] for x in xs]),
            "mean_full_corr": mean([x["full_corr"] for x in xs]),
            "mean_full_relrmse": mean([x["full_relrmse"] for x in xs]),
        })

    return rows


def maybe_plots(out, events):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    final = [e for e in events if e["mask_mode"] == "final"] or events

    for col in ["gt_n_components", "pred_n_components"]:
        vals = [int(e[col]) for e in final]
        plt.figure()
        plt.hist(vals, bins=range(0, max(vals + [1]) + 2))
        plt.xlabel(col)
        plt.ylabel("events")
        plt.tight_layout()
        plt.savefig(out / f"{col}_hist.png", dpi=150)
        plt.close()

    for ycol in ["full_iou", "full_corr"]:
        xs = [e["gt_unseeded_area_frac"] for e in final]
        ys = [e[ycol] for e in final]
        plt.figure()
        plt.scatter(xs, ys)
        plt.xlabel("gt_unseeded_area_frac")
        plt.ylabel(ycol)
        plt.tight_layout()
        plt.savefig(out / f"gt_unseeded_area_frac_vs_{ycol}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--data_root")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--selector", default="full_corr")
    ap.add_argument("--split", default="test")
    ap.add_argument("--connectivity", default="4", choices=["4", "8", "both"])
    ap.add_argument("--mask_modes", default="final,t25,t50,t75,t100")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--inf_threshold", type=float, default=1e5)
    ap.add_argument("--strict_source_mask", action="store_true")
    ap.add_argument("--inspect_only", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--overlay_examples", type=int, default=0)
    args = ap.parse_args()

    runs = [xp(r) for r in args.runs]

    if args.inspect_only:
        inspect_runs(runs)
        return

    out = xp(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    connectivities = [4, 8] if args.connectivity == "both" else [int(args.connectivity)]
    mask_modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]

    event_rows = []
    gt_rows = []
    pred_rows = []
    problems = []

    for run_path in runs:
        if not run_path.exists():
            problems.append(f"Missing run dir: {run_path}")
            continue

        files = find_npz(run_path)

        # Keep only the requested selector file when available.
        # Example: --selector full_corr keeps test_predictions_best_by_full_corr.npz
        selector_token = f"best_by_{args.selector}"
        selector_files = [p for p in files if selector_token in p.name]
        if selector_files:
            files = selector_files
        else:
            print(f"Warning: no files matched selector token {selector_token!r} under {run_path}; using all npz files.")

        if not files:
            problems.append(f"No .npz prediction/array files found under {run_path}")
            continue

        group = group_name(run_path)
        run = run_path.name
        epoch = selected_epoch(run_path, args.selector)
        n_events = 0

        for f in files:
            evs = split_events_from_npz(f, run_path, args)
            n_events += len(evs)

            for ev in evs:
                for conn in connectivities:
                    for mode in mask_modes:
                        scene_id = ev["scene_id"]
                        if scene_id in ("", "unknown_scene", None):
                            m_scene = re.search(r"per_scene_(\\d{4}_\\d{5})", run)
                            if m_scene:
                                scene_id = m_scene.group(1)
                            elif run.startswith("cross_"):
                                scene_id = "cross"

                        meta = {
                            "group": group,
                            "run": run,
                            "scene_id": scene_id,
                            "event_id": ev["event_id"],
                            "split": args.split,
                            "selector": args.selector,
                            "selected_epoch": epoch,
                            "connectivity": conn,
                            "mask_mode": mode,
                            "array_file": ev["array_file"],
                            "source_mask_status": ev["source_mask_status"],
                        }

                        e, gtc, prc = analyze_one(
                            meta,
                            ev["true_T"],
                            ev["pred_T"],
                            ev["valid"],
                            ev["source"],
                            conn,
                            mode,
                            args.inf_threshold,
                        )

                        event_rows.append(e)
                        gt_rows.extend(gtc)
                        pred_rows.extend(prc)

        if n_events == 0:
            problems.append(f"Found .npz files under {run_path}, but none had recognizable true/pred arrays. Use --inspect_only.")

    if not event_rows:
        print("\nERROR: no events analyzed.\n")
        for p in problems:
            print(" -", p)
        print("\nRun the same command with --inspect_only and inspect available array keys.")
        sys.exit(2)

    write_csv(out / "event_component_summary.csv", event_rows, EVENT_COLS)
    write_csv(out / "gt_component_table.csv", gt_rows, GT_COLS)
    write_csv(out / "pred_component_table.csv", pred_rows, PRED_COLS)
    write_csv(out / "run_summary.csv", aggregate(event_rows, by_scene=False), SUM_COLS)
    write_csv(out / "scene_summary.csv", aggregate(event_rows, by_scene=True), SUM_COLS)

    (out / "README_component_analysis.md").write_text(
        "# Component analysis\n\n"
        "Seeded GT components intersect source_mask. "
        "Unseeded GT components do not intersect source_mask. "
        "Treat unseeded components as diagnostic evidence only, not proof of physical later ignition.\n"
    )

    if args.plots:
        maybe_plots(out, event_rows)

    print(f"Wrote {len(event_rows)} event/mode/connectivity rows to: {out}")
    print(f"Run summary: {out / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
