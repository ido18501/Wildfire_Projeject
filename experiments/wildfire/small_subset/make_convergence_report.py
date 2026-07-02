#!/usr/bin/env python3
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(sorted(Path("small_runs").glob("band2_default_convergence_*"))[-1])
OUT = Path("small_runs") / f"convergence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def get_series(h, *keys):
    if isinstance(h, dict):
        for k in keys:
            if k in h and isinstance(h[k], list):
                return [safe_float(v) for v in h[k]]
    if isinstance(h, list):
        vals = []
        for row in h:
            if not isinstance(row, dict):
                continue
            val = None
            for k in keys:
                if k in row:
                    val = row[k]
                    break
            vals.append(safe_float(val))
        return vals
    return []


def clean(vals):
    return [v for v in vals if v is not None and np.isfinite(v)]


def last_valid(vals):
    c = clean(vals)
    return c[-1] if c else None


def first_valid(vals):
    c = clean(vals)
    return c[0] if c else None


def best_max(vals):
    pairs = [(i + 1, v) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
    return max(pairs, key=lambda x: x[1]) if pairs else (None, None)


def best_min(vals):
    pairs = [(i + 1, v) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
    return min(pairs, key=lambda x: x[1]) if pairs else (None, None)


def slope_last(vals, k=10):
    c = clean(vals)
    if len(c) < 3:
        return None
    y = np.array(c[-min(k, len(c)):], dtype=float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def pct_change_last(vals, k=10):
    c = clean(vals)
    if len(c) < 2:
        return None
    y = c[-min(k, len(c)):]
    if abs(y[0]) < 1e-12:
        return None
    return 100.0 * (y[-1] - y[0]) / abs(y[0])


def near_best_last(vals, maximize=True):
    last = last_valid(vals)
    if last is None:
        return None
    if maximize:
        _, best = best_max(vals)
        return last / best if best and abs(best) > 1e-12 else None
    else:
        _, best = best_min(vals)
        return last / best if best and abs(best) > 1e-12 else None


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


rows = []

for run_dir in sorted(ROOT.iterdir()):
    if not run_dir.is_dir() or run_dir.name == "logs":
        continue

    hp = run_dir / "history.json"
    if not hp.exists():
        continue

    h = json.loads(hp.read_text())

    loss = get_series(h, "losses", "loss", "train_losses", "train_loss")
    data = get_series(h, "data_losses", "data_loss", "train_data_losses", "train_data_loss")
    reg = get_series(h, "reg_losses", "reg_loss", "train_reg_losses", "train_reg_loss")
    val_iou = get_series(h, "val_ious", "val_iou", "val_band_iou", "mean_band_iou")
    val_mse = get_series(h, "val_mses", "val_mse", "val_band_mse", "mean_band_mse")
    lr = get_series(h, "learning_rates", "lr", "learning_rate")

    n_epochs = max([len(x) for x in [loss, data, reg, val_iou, val_mse, lr] if x] or [0])

    best_iou_epoch, best_iou = best_max(val_iou)
    best_mse_epoch, best_mse = best_min(val_mse)

    row = {
        "run": run_dir.name,
        "epochs": n_epochs,

        "loss_first": first_valid(loss),
        "loss_last": last_valid(loss),
        "loss_last10_pct_change": pct_change_last(loss, 10),
        "loss_slope_last10": slope_last(loss, 10),

        "data_last": last_valid(data),
        "reg_last": last_valid(reg),
        "reg_data_ratio_last": (
            last_valid(reg) / last_valid(data)
            if last_valid(reg) is not None and last_valid(data) not in [None, 0]
            else None
        ),

        "best_val_iou_epoch": best_iou_epoch,
        "best_val_iou": best_iou,
        "last_val_iou": last_valid(val_iou),
        "last_val_iou_over_best": near_best_last(val_iou, True),
        "val_iou_slope_last10": slope_last(val_iou, 10),
        "val_iou_last10_pct_change": pct_change_last(val_iou, 10),

        "best_val_mse_epoch": best_mse_epoch,
        "best_val_mse": best_mse,
        "last_val_mse": last_valid(val_mse),
        "last_val_mse_over_best": near_best_last(val_mse, False),
        "val_mse_slope_last10": slope_last(val_mse, 10),
        "val_mse_last10_pct_change": pct_change_last(val_mse, 10),

        "lr_last": last_valid(lr),
    }

    # Strict-ish convergence flags.
    # These are intentionally transparent, not magical.
    row["iou_plateau_flag"] = (
        row["val_iou_slope_last10"] is not None
        and abs(row["val_iou_slope_last10"]) < 0.0015
        and row["last_val_iou_over_best"] is not None
        and row["last_val_iou_over_best"] > 0.97
    )
    row["loss_plateau_flag"] = (
        row["loss_last10_pct_change"] is not None
        and abs(row["loss_last10_pct_change"]) < 5.0
    )
    row["mse_not_improving_flag"] = (
        row["best_val_mse_epoch"] is not None
        and row["best_val_mse_epoch"] < max(3, n_epochs - 10)
    )

    row["converged_claim_supported"] = (
        row["iou_plateau_flag"]
        and row["loss_plateau_flag"]
    )

    rows.append(row)

    # Plot.
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    x = np.arange(1, n_epochs + 1)

    def plot(ax, vals, label):
        y = np.array([np.nan if v is None else v for v in vals], dtype=float)
        if len(y) < n_epochs:
            y = np.pad(y, (0, n_epochs - len(y)), constant_values=np.nan)
        ax.plot(x, y, marker="o", markersize=2, linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    plot(axes[0], loss, "train loss")
    if data:
        y = np.array([np.nan if v is None else v for v in data], dtype=float)
        if len(y) < n_epochs:
            y = np.pad(y, (0, n_epochs - len(y)), constant_values=np.nan)
        axes[0].plot(x, y, linewidth=1, alpha=0.7, label="data")
        axes[0].legend()

    plot(axes[1], val_iou, "val band IoU")
    if best_iou_epoch is not None:
        axes[1].axvline(best_iou_epoch, linestyle="--", alpha=0.5)
        axes[1].set_title(f"best IoU epoch={best_iou_epoch}, best={best_iou:.4f}, last={last_valid(val_iou):.4f}")

    plot(axes[2], val_mse, "val band MSE")
    if best_mse_epoch is not None:
        axes[2].axvline(best_mse_epoch, linestyle="--", alpha=0.5)
        axes[2].set_title(f"best MSE epoch={best_mse_epoch}, best={best_mse:.3f}, last={last_valid(val_mse):.3f}")

    axes[2].set_xlabel("epoch")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(OUT / f"{run_dir.name}.png", dpi=160)
    plt.close(fig)


# Save JSON and CSV.
(OUT / "convergence_summary.json").write_text(json.dumps(rows, indent=2))

csv_cols = [
    "run", "epochs",
    "loss_first", "loss_last", "loss_last10_pct_change", "loss_slope_last10",
    "data_last", "reg_last", "reg_data_ratio_last",
    "best_val_iou_epoch", "best_val_iou", "last_val_iou",
    "last_val_iou_over_best", "val_iou_slope_last10", "val_iou_last10_pct_change",
    "best_val_mse_epoch", "best_val_mse", "last_val_mse",
    "last_val_mse_over_best", "val_mse_slope_last10", "val_mse_last10_pct_change",
    "lr_last",
    "iou_plateau_flag", "loss_plateau_flag", "mse_not_improving_flag",
    "converged_claim_supported",
]

with (OUT / "convergence_summary.csv").open("w") as f:
    f.write(",".join(csv_cols) + "\n")
    for r in rows:
        vals = []
        for c in csv_cols:
            v = r.get(c)
            vals.append("" if v is None else str(v))
        f.write(",".join(vals) + "\n")


print(f"ROOT: {ROOT}")
print(f"OUT:  {OUT}")
print()
print(f"{'RUN':45s} {'E':>4s} {'lossΔ10%':>9s} {'bestIoU':>8s} {'lastIoU':>8s} {'last/best':>9s} {'IoUslp10':>9s} {'bestMSEe':>8s} {'MSElast/best':>12s} {'conv?':>7s}")
print("-" * 135)

for r in rows:
    print(
        f"{r['run']:45s} "
        f"{r['epochs']:4d} "
        f"{fmt(r['loss_last10_pct_change'],2):>9s} "
        f"{fmt(r['best_val_iou']):>8s} "
        f"{fmt(r['last_val_iou']):>8s} "
        f"{fmt(r['last_val_iou_over_best']):>9s} "
        f"{fmt(r['val_iou_slope_last10']):>9s} "
        f"{fmt(r['best_val_mse_epoch'],0):>8s} "
        f"{fmt(r['last_val_mse_over_best']):>12s} "
        f"{str(r['converged_claim_supported']):>7s}"
    )

print()
print("Convergence claim rule used:")
print("  IoU plateau: |last-10 IoU slope| < 0.0015 and final IoU > 97% of best IoU")
print("  Loss plateau: |last-10 train loss percent change| < 5%")
print("  converged_claim_supported = IoU plateau AND loss plateau")
print()
print("Files written:")
print(f"  {OUT / 'convergence_summary.json'}")
print(f"  {OUT / 'convergence_summary.csv'}")
print(f"  {OUT}/*.png")
