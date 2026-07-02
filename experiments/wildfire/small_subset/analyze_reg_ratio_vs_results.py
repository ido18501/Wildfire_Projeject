#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def latest_root(pattern):
    roots = sorted(Path("small_runs").glob(pattern))
    return roots[-1] if roots else None


def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def get_list(h, *keys):
    # Case 1: dict of lists
    if isinstance(h, dict):
        for k in keys:
            if k in h and isinstance(h[k], list):
                out = []
                for v in h[k]:
                    fv = safe_float(v)
                    out.append(float("nan") if fv is None else fv)
                return out

    # Case 2: list of epoch dictionaries
    if isinstance(h, list):
        out = []
        for row in h:
            if not isinstance(row, dict):
                continue
            val = None
            for k in keys:
                if k in row:
                    val = row[k]
                    break
            fv = safe_float(val)
            out.append(float("nan") if fv is None else fv)
        return out

    return []


def last_valid(vals):
    clean = [v for v in vals if isinstance(v, float) and not math.isnan(v)]
    return clean[-1] if clean else None


def best_valid(vals, mode="max"):
    clean = [(i + 1, v) for i, v in enumerate(vals)
             if isinstance(v, float) and not math.isnan(v)]
    if not clean:
        return None, None
    return (max if mode == "max" else min)(clean, key=lambda x: x[1])


def parse_lambda(run_name):
    m = re.search(r"lam([0-9]+)$", run_name)
    if not m:
        return None
    s = m.group(1)
    table = {
        "002": 0.002,
        "005": 0.005,
        "01": 0.01,
        "02": 0.02,
        "05": 0.05,
        "10": 0.10,
    }
    return table.get(s)


def setup_name(run_name):
    if run_name.startswith("cross_"):
        return "cross"
    m = re.match(r"per_scene_([0-9]{4}_[0-9]{5})_", run_name)
    if m:
        return m.group(1)
    return "unknown"


def collect_run(d):
    row = {
        "root": str(d.parent),
        "run": d.name,
        "setup": setup_name(d.name),
        "lambda": parse_lambda(d.name),
    }

    status = d / "status.txt"
    if status.exists():
        row["status"] = status.read_text().strip().splitlines()[-1]
    else:
        row["status"] = "NO_STATUS"

    hp = d / "history.json"
    if hp.exists():
        try:
            h = json.loads(hp.read_text())

            losses = get_list(h,
                "losses", "loss", "train_losses", "train_loss",
                "total_losses", "total_loss"
            )
            data = get_list(h,
                "data_losses", "data_loss", "train_data_losses", "train_data_loss",
                "mse_losses", "mse_loss"
            )
            reg = get_list(h,
                "reg_losses", "reg_loss", "train_reg_losses", "train_reg_loss",
                "regularization_losses", "regularization_loss"
            )
            val_iou = get_list(h,
                "val_ious", "val_iou", "val_band_iou", "val_band_ious",
                "mean_band_iou", "mean_band_ious"
            )
            val_mse = get_list(h,
                "val_mses", "val_mse", "val_band_mse", "val_band_mses",
                "mean_band_mse", "mean_band_mses"
            )

            lens = [len(x) for x in [losses, data, reg, val_iou, val_mse] if x]
            row["epochs"] = max(lens) if lens else None

            row["last_loss"] = last_valid(losses)
            row["last_data"] = last_valid(data)
            row["last_reg"] = last_valid(reg)

            if row["last_data"] not in [None, 0] and row["last_reg"] is not None:
                row["reg_data_ratio"] = row["last_reg"] / row["last_data"]

            be, bv = best_valid(val_iou, "max")
            me, mv = best_valid(val_mse, "min")
            row["best_val_iou_epoch"] = be
            row["best_val_iou"] = bv
            row["last_val_iou"] = last_valid(val_iou)
            row["best_val_mse_epoch"] = me
            row["best_val_mse"] = mv
            row["last_val_mse"] = last_valid(val_mse)

        except Exception as e:
            row["history_error"] = repr(e)

    full = d / "test_metrics_full_fire.json"
    if full.exists():
        try:
            m = json.loads(full.read_text())
            row["full_corr"] = safe_float(m.get("correlation_mean"))
            row["full_rel_rmse"] = safe_float(m.get("relative_rmse_mean"))
            row["full_iou25"] = safe_float(m.get("iou_25_mean"))
            row["full_iou50"] = safe_float(m.get("iou_50_mean"))
            row["full_iou75"] = safe_float(m.get("iou_75_mean"))
            row["full_iou100"] = safe_float(m.get("iou_100_mean"))
        except Exception as e:
            row["full_error"] = repr(e)

    band = d / "test_metrics_band.json"
    if band.exists():
        try:
            m = json.loads(band.read_text())
            row["test_band_iou"] = safe_float(m.get("mean_band_iou_mean"))
            row["test_band_mse"] = safe_float(m.get("mean_band_mse_mean"))
        except Exception as e:
            row["band_error"] = repr(e)

    return row


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None, len(pairs)
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None, len(pairs)
    return float(np.corrcoef(x, y)[0, 1]), len(pairs)


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="*", default=None)
    p.add_argument("--include-default", action="store_true")
    p.add_argument("--include-latest-coarse", action="store_true")
    args = p.parse_args()

    roots = []

    if args.roots:
        roots.extend(Path(r).expanduser() for r in args.roots)

    if args.include_default:
        r = latest_root("band2_default_convergence_*")
        if r:
            roots.append(r)

    if args.include_latest_coarse:
        r = latest_root("band2_lambda_coarse_*")
        if r:
            roots.append(r)

    if not roots:
        # sensible default
        r1 = latest_root("band2_default_convergence_*")
        r2 = latest_root("band2_lambda_coarse_*")
        roots = [r for r in [r1, r2] if r is not None]

    rows = []
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name != "logs":
                rows.append(collect_run(d))

    rows = [r for r in rows if r.get("reg_data_ratio") is not None]

    print(f"Collected runs with reg/data ratio: {len(rows)}")
    print()

    metrics = [
        "test_band_iou",
        "full_iou50",
        "full_iou75",
        "full_corr",
        "full_rel_rmse",
        "best_val_iou",
        "last_val_iou",
        "best_val_mse",
        "last_val_mse",
    ]

    print("GLOBAL CORRELATION WITH reg/data")
    print("=" * 80)
    print(f"{'metric':20s} {'corr':>10s} {'n':>5s}")
    print("-" * 80)
    for m in metrics:
        c, n = corr([r.get("reg_data_ratio") for r in rows], [r.get(m) for r in rows])
        print(f"{m:20s} {fmt(c):>10s} {n:5d}")

    print("\nPER-SETUP CORRELATION WITH reg/data")
    print("=" * 100)
    print(f"{'setup':12s} {'metric':20s} {'corr':>10s} {'n':>5s}")
    print("-" * 100)

    setups = sorted(set(r["setup"] for r in rows))
    for setup in setups:
        sub = [r for r in rows if r["setup"] == setup]
        for m in ["test_band_iou", "full_iou50", "full_iou75", "full_corr", "best_val_iou"]:
            c, n = corr([r.get("reg_data_ratio") for r in sub], [r.get(m) for r in sub])
            if n >= 3:
                print(f"{setup:12s} {m:20s} {fmt(c):>10s} {n:5d}")

    print("\nRUN TABLE")
    print("=" * 160)
    print(
        f"{'setup':12s} {'run':48s} {'lam':>7s} {'E':>4s} "
        f"{'reg/data':>9s} {'bestvIoU':>9s} {'testBand':>9s} "
        f"{'full50':>8s} {'full75':>8s} {'corr':>8s} {'relRMSE':>8s}"
    )
    print("-" * 160)

    for r in sorted(rows, key=lambda x: (x["setup"], x.get("lambda") or 999, x["run"])):
        print(
            f"{r['setup']:12s} "
            f"{r['run'][:48]:48s} "
            f"{fmt(r.get('lambda'),3):>7s} "
            f"{fmt(r.get('epochs'),0):>4s} "
            f"{fmt(r.get('reg_data_ratio')):>9s} "
            f"{fmt(r.get('best_val_iou')):>9s} "
            f"{fmt(r.get('test_band_iou')):>9s} "
            f"{fmt(r.get('full_iou50')):>8s} "
            f"{fmt(r.get('full_iou75')):>8s} "
            f"{fmt(r.get('full_corr')):>8s} "
            f"{fmt(r.get('full_rel_rmse')):>8s}"
        )

    out = Path("small_runs") / "reg_ratio_vs_results_latest.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved raw table to {out}")


if __name__ == "__main__":
    main()
