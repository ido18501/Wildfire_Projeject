#!/usr/bin/env python3
"""
Time-conditioned burn-by-time correction head.

Standalone extension:
- does NOT patch existing files
- loads an existing frozen eikonal checkpoint
- uses exact first observed burned mask as source
- trains a small residual probability head:
      logit P(true_T <= t) = (t - T_pred)/(temp * H) + residual_head(...)
- compares raw eikonal masks vs corrected masks on the original test split

Expected use:
python experiments/wildfire/extensions/time_gate_finetune.py \
  --scene 0002_00714 \
  --run_dir small_runs/.../baseline/per_scene_0002_00714 \
  --data_root /path/to/simulation_data \
  --out small_runs/time_gate_ext_test
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader,
    CovariateAdapter,
    extract_arrival_times,
)
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.forward.pytorch import EikonalProblem
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def as_float_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x)).float()


def mask_to_bool(mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == bool:
        return arr
    # We audited masks as basically binary grayscale; use conservative image threshold.
    if np.nanmax(arr) > 1.5:
        return arr > 127
    return arr > 0.5


def first_frame_exact_source(masks) -> Tuple[np.ndarray, int]:
    """
    Return exact first non-empty burned mask.
    Supports masks as arrays or tuples/lists where the mask is the last item.
    """
    for idx, item in enumerate(masks):
        m = item[-1] if isinstance(item, (tuple, list)) else item
        b = mask_to_bool(m)
        if b.sum() > 0:
            return b.astype(bool), idx
    raise ValueError("No non-empty first burned mask found")


def load_splits_or_reconstruct(run_dir: Path, loader, scene: str, seed: int):
    train = read_lines(run_dir / "train_fires.txt")
    val = read_lines(run_dir / "val_fires.txt")
    test = read_lines(run_dir / "test_fires.txt")

    if train and val and test:
        return train, val, test

    print("WARNING: split txt files not found; reconstructing 70/15/15 split from loader with seed.")
    events = list(loader.scenes[scene]["events"])
    rng = np.random.RandomState(seed)
    rng.shuffle(events)
    n_train = int(0.70 * len(events))
    n_val = int(0.15 * len(events))
    return events[:n_train], events[n_train:n_train+n_val], events[n_train+n_val:]


def get_ckpt_config(ckpt):
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if cfg is None:
        cfg = {}
    return cfg


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def build_frozen_eikonal_model(ckpt_path: Path, n_spatial: int, n_global: int, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = get_ckpt_config(ckpt)

    state = ckpt.get("model_state_dict", ckpt)
    state = strip_module_prefix(state)

    model = FlexibleCovariateEncoder(
        n_spatial_features=n_spatial,
        n_global_features=n_global,
        hidden_channels=int(cfg.get("hidden_channels", 64)),
        num_cnn_layers=int(cfg.get("num_cnn_layers", 5)),
        mlp_hidden=int(cfg.get("mlp_hidden", 128)),
        mlp_num_layers=int(cfg.get("mlp_num_layers", 2)),
        eps=float(cfg.get("eps", 0.1)),
        max_drift=float(cfg.get("max_drift", 0.8)),
        max_eigenvalue=float(cfg.get("max_eigenvalue", 10.0)),
        local_scale=float(cfg.get("local_scale", 0.1)),
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("WARNING missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("WARNING unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def prepare_scenario(loader, adapter, scene: str, event_id: str) -> Dict:
    data = loader.load_scenario(scene, event_id)

    spatial_feat, global_feat = adapter(data)
    spatial_feat = as_float_tensor(spatial_feat)
    global_feat = as_float_tensor(global_feat).view(-1)

    true_T = extract_arrival_times(data["masks"], data["timestamps"]).astype(np.float32)
    valid_mask = np.isfinite(true_T)

    source_mask, first_idx = first_frame_exact_source(data["masks"])

    if source_mask.shape != true_T.shape:
        raise ValueError(f"source mask shape {source_mask.shape} != arrival shape {true_T.shape}")

    if valid_mask.sum() < 10:
        raise ValueError("too few valid burned pixels")

    return {
        "scene_id": scene,
        "event_id": event_id,
        "spatial": spatial_feat,
        "global": global_feat,
        "true_T": true_T,
        "valid_mask": valid_mask,
        "source_mask": source_mask,
        "first_burn_frame": int(first_idx),
        "first_burn_area": int(source_mask.sum()),
    }


@torch.no_grad()
def compute_frozen_prediction(model, scenario: Dict, device: str, max_iter: int):
    spatial = scenario["spatial"].unsqueeze(0).to(device)
    global_f = scenario["global"].unsqueeze(0).to(device)

    G, B, _, _ = model(spatial, global_f)

    problem = EikonalProblem(
        G=G.detach().cpu(),
        B=B.detach().cpu(),
        source_mask=torch.from_numpy(scenario["source_mask"]).bool().unsqueeze(0),
    )
    T_pred, _ = fast_sweeping_solve_numba(problem, max_iter=max_iter)
    scenario["T_pred"] = T_pred[0].numpy().astype(np.float32)
    return scenario


def max_true_time(true_T):
    finite = np.isfinite(true_T)
    if finite.sum() == 0:
        return 0.0
    return float(np.nanmax(true_T[finite]))


# --------------------------------------------------------------------------------------
# Gate model
# --------------------------------------------------------------------------------------

class TimeGateHead(nn.Module):
    """
    Residual correction head.

    Input channels:
      spatial covariates
      T_norm
      time_frac
      delta = (t - T_pred) / H
      raw_prob = sigmoid(base_logit)

    Output:
      residual logit C(x,t)

    Final:
      logit = base_logit + C
    """
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_gate_input(scenario: Dict, t: float, temp: float, device: str):
    spatial = scenario["spatial"].float()
    C, Hh, Ww = spatial.shape

    true_T = scenario["true_T"]
    T_pred_np = scenario["T_pred"]
    horizon = max(max_true_time(true_T), 1e-6)

    T_safe = np.where(np.isfinite(T_pred_np) & (T_pred_np < 1e5), T_pred_np, horizon * 10.0)
    T_norm = np.clip(T_safe / horizon, 0.0, 10.0).astype(np.float32)
    time_frac = np.full_like(T_norm, t / horizon, dtype=np.float32)
    delta = np.clip((t - T_safe) / horizon, -10.0, 10.0).astype(np.float32)

    base_logit = np.clip((t - T_safe) / (temp * horizon), -20.0, 20.0).astype(np.float32)
    raw_prob = 1.0 / (1.0 + np.exp(-base_logit))

    extra = torch.from_numpy(np.stack([T_norm, time_frac, delta, raw_prob], axis=0)).float()
    x = torch.cat([spatial, extra], dim=0).unsqueeze(0).to(device)

    base = torch.from_numpy(base_logit).float().unsqueeze(0).unsqueeze(0).to(device)
    target = ((np.isfinite(true_T)) & (true_T <= t)).astype(np.float32)
    y = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(device)

    return x, base, y


def dice_loss_from_logits(logits, y, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    denom = p.sum() + y.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def loss_fn(logits, y):
    pos = y.sum().detach()
    total = torch.tensor(float(y.numel()), device=y.device)
    neg = total - pos
    if pos.item() < 1:
        pos_weight = torch.tensor(1.0, device=y.device)
    else:
        pos_weight = torch.clamp(neg / (pos + 1e-6), 1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, y)
    return bce + dice, bce.detach(), dice.detach()


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def bin_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = int((pred | gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    iou = tp / union if union else 0.0
    precision = tp / pred_sum if pred_sum else 0.0
    recall = tp / gt_sum if gt_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": pred_sum,
        "gt_area": gt_sum,
    }


@torch.no_grad()
def corrected_prob(head, scenario: Dict, t: float, temp: float, device: str):
    x, base, _ = make_gate_input(scenario, t, temp, device)
    residual = head(x)
    logits = base + residual
    return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


def evaluate_dataset(head, scenarios: List[Dict], fracs: List[float], alpha: float, temp: float, device: str):
    rows = []

    for sc in scenarios:
        horizon = max_true_time(sc["true_T"])
        if horizon <= 0:
            continue

        for frac in fracs:
            t = frac * horizon
            gt_t = np.isfinite(sc["true_T"]) & (sc["true_T"] <= t)

            raw_t = np.isfinite(sc["T_pred"]) & (sc["T_pred"] <= t)
            p = corrected_prob(head, sc, t, temp, device)
            corr_t = p >= alpha

            raw_m = bin_metrics(raw_t, gt_t)
            corr_m = bin_metrics(corr_t, gt_t)

            rows.append({
                "event_id": sc["event_id"],
                "frac": frac,
                "raw": raw_m,
                "corrected": corr_m,
            })

    summary = {}
    for frac in fracs:
        subset = [r for r in rows if abs(r["frac"] - frac) < 1e-9]
        for side in ["raw", "corrected"]:
            for key in ["iou", "precision", "recall", "f1", "fp", "fn", "pred_area", "gt_area"]:
                vals = [r[side][key] for r in subset]
                summary[f"{side}_{key}_{int(frac*100)}"] = float(np.mean(vals)) if vals else float("nan")

    # Overall mean across thresholds
    for side in ["raw", "corrected"]:
        vals = [r[side]["iou"] for r in rows]
        summary[f"{side}_mean_iou_all"] = float(np.mean(vals)) if vals else float("nan")

    return summary, rows


def select_threshold(head, val_scenarios, fracs, thresholds, temp, device):
    best_alpha = None
    best_iou = -1.0
    all_scores = []

    for alpha in thresholds:
        summary, _ = evaluate_dataset(head, val_scenarios, fracs, alpha, temp, device)
        score = summary["corrected_mean_iou_all"]
        all_scores.append({"alpha": alpha, "val_corrected_mean_iou_all": score})
        if score > best_iou:
            best_iou = score
            best_alpha = alpha

    return best_alpha, best_iou, all_scores


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--temp", type=float, default=0.08)
    ap.add_argument("--max_train", type=int, default=80)
    ap.add_argument("--max_val", type=int, default=25)
    ap.add_argument("--max_test", type=int, default=40)
    ap.add_argument("--fracs", type=str, default="0.25,0.50,0.75,1.00")
    ap.add_argument("--thresholds", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    args = ap.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model.pt not found in run_dir: {ckpt_path}")

    fracs = [float(x) for x in args.fracs.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print("=" * 80)
    print("Time-conditioned eikonal correction extension")
    print("=" * 80)
    print("scene:", args.scene)
    print("run_dir:", run_dir)
    print("data_root:", args.data_root)
    print("out:", out_dir)
    print("device:", args.device)
    print("fracs:", fracs)
    print("thresholds:", thresholds)

    loader = Sim2RealFireLoader(args.data_root)
    adapter = CovariateAdapter()
    warmup_numba()

    train_ids, val_ids, test_ids = load_splits_or_reconstruct(run_dir, loader, args.scene, args.seed)
    train_ids = train_ids[:args.max_train]
    val_ids = val_ids[:args.max_val]
    test_ids = test_ids[:args.max_test]

    print(f"Using split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Prepare one scenario first to infer feature sizes
    first = prepare_scenario(loader, adapter, args.scene, train_ids[0])
    n_spatial = int(first["spatial"].shape[0])
    n_global = int(first["global"].numel())
    print(f"Feature sizes: spatial={n_spatial}, global={n_global}")

    frozen_model, frozen_cfg = build_frozen_eikonal_model(ckpt_path, n_spatial, n_global, args.device)

    def load_and_predict(ids, name):
        out = []
        print(f"\nLoading + frozen predicting {name}: {len(ids)} fires")
        for i, eid in enumerate(ids, 1):
            try:
                sc = prepare_scenario(loader, adapter, args.scene, eid)
                sc = compute_frozen_prediction(frozen_model, sc, args.device, args.max_iter)
                out.append(sc)
                if i % 10 == 0 or i == len(ids):
                    print(f"  {name}: {i}/{len(ids)}")
            except Exception as e:
                print(f"  SKIP {name} {eid}: {e}")
        return out

    train_scenarios = load_and_predict(train_ids, "train")
    val_scenarios = load_and_predict(val_ids, "val")
    test_scenarios = load_and_predict(test_ids, "test")

    if not train_scenarios or not val_scenarios or not test_scenarios:
        raise RuntimeError("Need non-empty train/val/test scenarios.")

    in_channels = n_spatial + 4
    head = TimeGateHead(in_channels=in_channels, hidden=args.hidden).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_state = None
    best_alpha = 0.5
    best_val = -1.0
    history = []

    print("\nTraining correction head...")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        head.train()
        random.shuffle(train_scenarios)

        losses = []
        bces = []
        dices = []

        for sc in train_scenarios:
            horizon = max_true_time(sc["true_T"])
            if horizon <= 0:
                continue

            # Random threshold each pass; cheap data augmentation over time.
            frac = random.choice(fracs)
            t = frac * horizon

            x, base, y = make_gate_input(sc, t, args.temp, args.device)
            residual = head(x)
            logits = base + residual

            loss, bce, dice = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            bces.append(float(bce.cpu()))
            dices.append(float(dice.cpu()))

        head.eval()
        alpha, val_score, val_alpha_scores = select_threshold(
            head, val_scenarios, fracs, thresholds, args.temp, args.device
        )

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "bce": float(np.mean(bces)) if bces else float("nan"),
            "dice": float(np.mean(dices)) if dices else float("nan"),
            "best_alpha_this_epoch": alpha,
            "val_corrected_mean_iou_all": val_score,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} "
            f"loss={row['loss']:.4f} "
            f"val_corr_iou={val_score:.4f} "
            f"alpha={alpha:.2f}"
        )

        if val_score > best_val:
            best_val = val_score
            best_alpha = alpha
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    print("\nFinal evaluation...")
    val_summary, val_rows = evaluate_dataset(head, val_scenarios, fracs, best_alpha, args.temp, args.device)
    test_summary, test_rows = evaluate_dataset(head, test_scenarios, fracs, best_alpha, args.temp, args.device)

    elapsed = time.time() - start

    results = {
        "scene": args.scene,
        "run_dir": str(run_dir),
        "data_root": args.data_root,
        "out_dir": str(out_dir),
        "method": "frozen_eikonal_plus_time_conditioned_residual_gate",
        "source_policy_for_recomputed_predictions": "first_frame_exact",
        "n_spatial": n_spatial,
        "n_global": n_global,
        "fracs": fracs,
        "thresholds": thresholds,
        "best_alpha": best_alpha,
        "best_val_corrected_mean_iou_all": best_val,
        "elapsed_seconds": elapsed,
        "split_sizes": {
            "train": len(train_scenarios),
            "val": len(val_scenarios),
            "test": len(test_scenarios),
        },
        "history": history,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "test_rows": test_rows,
        "frozen_model_config": frozen_cfg,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save({
        "head_state_dict": head.state_dict(),
        "in_channels": in_channels,
        "hidden": args.hidden,
        "best_alpha": best_alpha,
        "fracs": fracs,
        "temp": args.temp,
        "source_policy": "first_frame_exact",
        "method": "base_eikonal_logit_plus_residual_head",
    }, out_dir / "time_gate_head.pt")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"best alpha selected on val: {best_alpha:.2f}")
    print(f"raw mean IoU all thresholds:       {test_summary['raw_mean_iou_all']:.4f}")
    print(f"corrected mean IoU all thresholds: {test_summary['corrected_mean_iou_all']:.4f}")
    print(f"delta:                             {test_summary['corrected_mean_iou_all'] - test_summary['raw_mean_iou_all']:+.4f}")

    for frac in fracs:
        k = int(frac * 100)
        raw = test_summary[f"raw_iou_{k}"]
        cor = test_summary[f"corrected_iou_{k}"]
        print(f"IoU@{k:03d}: raw={raw:.4f} corrected={cor:.4f} delta={cor-raw:+.4f}")

    print("\nSaved:")
    print(" ", out_dir / "results.json")
    print(" ", out_dir / "time_gate_head.pt")


if __name__ == "__main__":
    main()
