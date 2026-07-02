#!/usr/bin/env python3
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(sorted(Path("small_runs").glob("band2_paperstyle_corr_*"))[-1])
OUT = Path("small_runs") / f"paperstyle_convergence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)


EXPECTED = ["cross_band2_lam002_fullcorr"]
for s in ["0001_02614", "0002_00714", "0018_02440", "0014_00426", "0005_00725"]:
    EXPECTED.append(f"per_scene_{s}_band2_lam002_fullcorr")


def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def get_series(history, key):
    vals = []
    for row in history:
        if isinstance(row, dict):
            vals.append(safe_float(row.get(key)))
    return vals


def valid_pairs(vals):
    return [(i + 1, v) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]


def best_max(vals):
    pairs = valid_pairs(vals)
    return max(pairs, key=lambda x: x[1]) if pairs else (None, None)


def best_min(vals):
    pairs = valid_pairs(vals)
    return min(pairs, key=lambda x: x[1]) if pairs else (None, None)


def last_valid(vals):
    pairs = valid_pairs(vals)
    return pairs[-1][1] if pairs else None


def slope_last(vals, k=5):
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    if len(clean) < 3:
        return None
    y = np.array(clean[-min(k, len(clean)):], dtype=float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def pct_change_last(vals, k=5):
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    if len(clean) < 2:
        return None
    y = clean[-min(k, len(clean)):]
    if abs(y[0]) < 1e-12:
        return None
    return 100.0 * (y[-1] - y[0]) / abs(y[0])


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


rows = []

for run in EXPECTED:
    run_dir = ROOT / run
    hp = run_dir / "history_multimetric.json"
    sp = run_dir / "selector_summary.json"
    tp = run_dir / "test_results_by_selector.json"

    if not hp.exists():
        print(f"Missing history for {run}")
        continue

    history = json.loads(hp.read_text())
    selectors = json.loads(sp.read_text()) if sp.exists() else {}
    tests = json.loads(tp.read_text()) if tp.exists() else {}

    epochs = [r.get("epoch", i + 1) for i, r in enumerate(history)]

    train_loss = get_series(history, "train_loss")
    data_loss = get_series(history, "train_data_loss")
    reg_loss = get_series(history, "train_reg_loss")
    lr = get_series(history, "lr")

    val_band_iou = get_series(history, "val_band_iou")
    val_band_mse = get_series(history, "val_band_mse")
    val_full_iou50 = get_series(history, "val_full_iou50")
    val_full_corr = get_series(history, "val_full_corr")
    val_full_relrmse = get_series(history, "val_full_relrmse")

    b_band_e, b_band = best_max(val_band_iou)
    b_iou50_e, b_iou50 = best_max(val_full_iou50)
    b_corr_e, b_corr = best_max(val_full_corr)
    b_rmse_e, b_rmse = best_min(val_full_relrmse)

    official = tests.get("full_corr", {})
    official_full = official.get("test_full_fire", {})

    row = {
        "run": run,
        "epochs": len(history),

        "best_band_iou_epoch_curve": b_band_e,
        "best_band_iou_curve": b_band,
        "best_full_iou50_epoch_curve": b_iou50_e,
        "best_full_iou50_curve": b_iou50,
        "best_full_corr_epoch_curve": b_corr_e,
        "best_full_corr_curve": b_corr,
        "best_full_relrmse_epoch_curve": b_rmse_e,
        "best_full_relrmse_curve": b_rmse,

        "selector_band_iou_epoch": selectors.get("band_iou", {}).get("epoch"),
        "selector_full_iou50_epoch": selectors.get("full_iou50", {}).get("epoch"),
        "selector_full_corr_epoch": selectors.get("full_corr", {}).get("epoch"),
        "selector_full_relrmse_epoch": selectors.get("full_relrmse", {}).get("epoch"),

        "official_selected_epoch": official.get("selector_epoch"),
        "official_val_corr": official.get("selector_best"),
        "official_test_iou50": official_full.get("iou_50_mean"),
        "official_test_corr": official_full.get("correlation_mean"),
        "official_test_relrmse": official_full.get("relative_rmse_mean"),

        "last_train_loss": last_valid(train_loss),
        "train_loss_last5_pct_change": pct_change_last(train_loss, 5),
        "val_corr_last5_slope": slope_last(val_full_corr, 5),
        "val_iou50_last5_slope": slope_last(val_full_iou50, 5),
        "val_band_iou_last5_slope": slope_last(val_band_iou, 5),
        "val_relrmse_last5_slope": slope_last(val_full_relrmse, 5),
    }
    rows.append(row)

    # Plot important curves.
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    x = np.array(epochs)

    def plot_series(ax, vals, label, marker="o"):
        y = np.array([np.nan if v is None else v for v in vals], dtype=float)
        ax.plot(x, y, marker=marker, markersize=3, linewidth=1.2, label=label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plot_series(axes[0], train_loss, "train total loss")
    plot_series(axes[0], data_loss, "train data loss")
    plot_series(axes[0], reg_loss, "train reg loss")
    axes[0].set_ylabel("loss")

    plot_series(axes[1], val_band_iou, "val band IoU")
    if b_band_e:
        axes[1].axvline(b_band_e, linestyle="--", alpha=0.6, label=f"best band IoU ep {b_band_e}")
    axes[1].set_ylabel("band IoU")

    plot_series(axes[2], val_full_iou50, "val full IoU@50")
    plot_series(axes[2], val_full_corr, "val full corr")
    if b_iou50_e:
        axes[2].axvline(b_iou50_e, linestyle="--", alpha=0.5, label=f"best IoU50 ep {b_iou50_e}")
    if b_corr_e:
        axes[2].axvline(b_corr_e, linestyle=":", alpha=0.8, label=f"best corr ep {b_corr_e}")
    axes[2].set_ylabel("full-fire val metric")
    axes[2].legend(loc="best")

    plot_series(axes[3], val_full_relrmse, "val full relRMSE")
    plot_series(axes[3], val_band_mse, "val band MSE")
    if b_rmse_e:
        axes[3].axvline(b_rmse_e, linestyle="--", alpha=0.6, label=f"best relRMSE ep {b_rmse_e}")
    axes[3].set_ylabel("error")
    axes[3].set_xlabel("epoch")
    axes[3].legend(loc="best")

    fig.suptitle(
        f"{run}\n"
        f"official full-corr selected epoch = {official.get('selector_epoch')}, "
        f"test corr={official_full.get('correlation_mean')}, "
        f"test IoU50={official_full.get('iou_50_mean')}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{run}.png", dpi=160)
    plt.close(fig)


# Save reports.
(OUT / "paperstyle_convergence_summary.json").write_text(json.dumps(rows, indent=2))

cols = [
    "run", "epochs",
    "selector_band_iou_epoch",
    "selector_full_iou50_epoch",
    "selector_full_corr_epoch",
    "selector_full_relrmse_epoch",
    "official_selected_epoch",
    "official_val_corr",
    "official_test_iou50",
    "official_test_corr",
    "official_test_relrmse",
    "best_band_iou_curve",
    "best_full_iou50_curve",
    "best_full_corr_curve",
    "best_full_relrmse_curve",
    "train_loss_last5_pct_change",
    "val_corr_last5_slope",
    "val_iou50_last5_slope",
    "val_band_iou_last5_slope",
    "val_relrmse_last5_slope",
]

with (OUT / "paperstyle_convergence_summary.csv").open("w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")


print(f"ROOT: {ROOT}")
print(f"OUT:  {OUT}")
print()
print("BEST EPOCHS / OFFICIAL PAPER-STYLE SELECTION")
print("=" * 160)
print(
    f"{'run':55s} {'E':>4s} "
    f"{'bandEp':>6s} {'iou50Ep':>7s} {'corrEp':>7s} {'rmseEp':>7s} "
    f"{'officialEp':>10s} {'valCorr':>8s} "
    f"{'testIoU50':>9s} {'testCorr':>9s} {'testRMSE':>9s}"
)
print("-" * 160)

for r in rows:
    print(
        f"{r['run'][:55]:55s} "
        f"{r['epochs']:4d} "
        f"{fmt(r['selector_band_iou_epoch'],0):>6s} "
        f"{fmt(r['selector_full_iou50_epoch'],0):>7s} "
        f"{fmt(r['selector_full_corr_epoch'],0):>7s} "
        f"{fmt(r['selector_full_relrmse_epoch'],0):>7s} "
        f"{fmt(r['official_selected_epoch'],0):>10s} "
        f"{fmt(r['official_val_corr']):>8s} "
        f"{fmt(r['official_test_iou50']):>9s} "
        f"{fmt(r['official_test_corr']):>9s} "
        f"{fmt(r['official_test_relrmse']):>9s}"
    )

print()
print("LAST-WINDOW SLOPES / STABILITY")
print("=" * 130)
print(
    f"{'run':55s} {'lossΔ5%':>9s} {'corrSlope':>10s} {'iou50Slope':>11s} "
    f"{'bandSlope':>10s} {'rmseSlope':>10s}"
)
print("-" * 130)
for r in rows:
    print(
        f"{r['run'][:55]:55s} "
        f"{fmt(r['train_loss_last5_pct_change'],2):>9s} "
        f"{fmt(r['val_corr_last5_slope']):>10s} "
        f"{fmt(r['val_iou50_last5_slope']):>11s} "
        f"{fmt(r['val_band_iou_last5_slope']):>10s} "
        f"{fmt(r['val_relrmse_last5_slope']):>10s}"
    )

print()
print("Files written:")
print(f"  {OUT / 'paperstyle_convergence_summary.json'}")
print(f"  {OUT / 'paperstyle_convergence_summary.csv'}")
print(f"  {OUT}/*.png")
