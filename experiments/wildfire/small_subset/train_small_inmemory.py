#!/usr/bin/env python3
import argparse
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader,
    CovariateAdapter,
    extract_arrival_times,
    find_ignition_point,
)

from eikonal.forward.pytorch import EikonalProblem
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.regularization import compute_tv_regularization
from eikonal.nn.config import TVMetricType

from eikonal.backward.implicit_diff import identify_active_stencils, compute_parameter_gradients

try:
    from eikonal.backward.adjoint_numba import solve_adjoint_numba
    HAVE_NUMBA_BACKWARD = True
except Exception:
    from eikonal.backward.implicit_diff import backward_pass
    HAVE_NUMBA_BACKWARD = False


def make_adapter(feature_group: str):
    sig = inspect.signature(CovariateAdapter)
    if "feature_group" in sig.parameters:
        return CovariateAdapter(feature_group=feature_group)
    if "features" in sig.parameters:
        return CovariateAdapter(features=feature_group)
    if feature_group != "baseline":
        raise RuntimeError(
            f"CovariateAdapter in this repo does not expose feature_group/features, "
            f"but requested feature_group={feature_group!r}. "
            f"Patch CovariateAdapter first or run with --feature_group baseline."
        )
    return CovariateAdapter()


def to_float_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(np.asarray(x)).float().to(device)


def compute_metrics(pred_T: np.ndarray, true_T: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    valid = (
        valid_mask
        & np.isfinite(true_T)
        & np.isfinite(pred_T)
        & (pred_T < 1e5)
    )

    if int(valid.sum()) < 10:
        return {
            "correlation": float("nan"),
            "relative_rmse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "iou_25": float("nan"),
            "iou_50": float("nan"),
            "iou_75": float("nan"),
            "iou_100": float("nan"),
            "n_valid": int(valid.sum()),
        }

    p = pred_T[valid].astype(np.float64)
    y = true_T[valid].astype(np.float64)

    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    max_time = float(np.max(y))
    rel_rmse = rmse / max_time if max_time > 1e-8 else float("nan")

    if np.std(p) > 1e-8 and np.std(y) > 1e-8:
        corr = float(np.corrcoef(p, y)[0, 1])
    else:
        corr = float("nan")

    out = {
        "correlation": corr,
        "relative_rmse": rel_rmse,
        "rmse": rmse,
        "mae": mae,
        "max_time_gt": max_time,
        "n_valid": int(valid.sum()),
    }

    for frac in [0.25, 0.50, 0.75, 1.00]:
        threshold = max_time * frac
        pred_burned = p <= threshold
        true_burned = y <= threshold
        inter = int(np.logical_and(pred_burned, true_burned).sum())
        union = int(np.logical_or(pred_burned, true_burned).sum())
        out[f"iou_{int(frac * 100)}"] = float(inter / union) if union > 0 else 0.0

    return out


def aggregate_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {"n_samples": 0}

    keys = sorted({k for m in items for k in m.keys() if isinstance(m.get(k), (int, float))})
    out = {"n_samples": len(items)}

    for key in keys:
        vals = [float(m[key]) for m in items if key in m and not math.isnan(float(m[key]))]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        else:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")

    return out


def make_source_mask(arrival: np.ndarray, masks) -> Optional[np.ndarray]:
    ignition = find_ignition_point(masks)
    if ignition is None:
        return None

    H, W = arrival.shape
    si, sj = ignition
    source = np.zeros((H, W), dtype=bool)
    source[max(0, si - 1):min(H, si + 2), max(0, sj - 1):min(W, sj + 2)] = True

    valid = np.isfinite(arrival)
    source &= valid

    if not source.any() and valid.any():
        idx = np.unravel_index(np.nanargmin(np.where(valid, arrival, np.inf)), arrival.shape)
        si, sj = idx
        source[max(0, si - 1):min(H, si + 2), max(0, sj - 1):min(W, sj + 2)] = True
        source &= valid

    return source if source.any() else None


def normalize_arrival(arrival: np.ndarray, source: np.ndarray) -> np.ndarray:
    arr = arrival.astype(np.float32).copy()
    valid = np.isfinite(arr)

    if source.any() and np.isfinite(arr[source]).any():
        t0 = float(np.nanmin(arr[source]))
    elif valid.any():
        t0 = float(np.nanmin(arr[valid]))
    else:
        t0 = 0.0

    arr[valid] -= t0
    arr[~valid] = np.inf
    return arr


def load_split_into_memory(loader, adapter, scene_to_events, preload_device: str, min_valid_pixels: int):
    scenarios = []

    for scene_id, event_ids in scene_to_events.items():
        for event_id in event_ids:
            try:
                data = loader.load_scenario(scene_id, event_id)
                spatial, global_feat = adapter(data)

                arrival = extract_arrival_times(data["masks"], data["timestamps"])
                source = make_source_mask(arrival, data["masks"])
                if source is None:
                    continue

                arrival = normalize_arrival(arrival, source)
                valid = np.isfinite(arrival)

                if int(valid.sum()) < min_valid_pixels:
                    continue

                scenario = {
                    "scene_id": scene_id,
                    "event_id": event_id,
                    "spatial": to_float_tensor(spatial, preload_device),
                    "global": to_float_tensor(global_feat, preload_device),
                    "arrival_cpu": torch.from_numpy(arrival).float(),
                    "valid_cpu": torch.from_numpy(valid),
                    "source_cpu": torch.from_numpy(source.astype(bool)),
                    "H": int(arrival.shape[0]),
                    "W": int(arrival.shape[1]),
                    "n_valid": int(valid.sum()),
                }
                scenarios.append(scenario)

            except Exception as e:
                print(f"SKIP {scene_id}/{event_id}: {e}")

    return scenarios


def build_cross_scene_splits(manifest):
    splits = {"train": {}, "val": {}, "test": {}}

    for scene_id in manifest["cross_scene"]["train_scenes"]:
        splits["train"][scene_id] = manifest["scenes"][scene_id]["selected"]

    for scene_id in manifest["cross_scene"].get("val_scenes", []):
        splits["val"][scene_id] = manifest["scenes"][scene_id]["selected"]

    for scene_id in manifest["cross_scene"]["test_scenes"]:
        splits["test"][scene_id] = manifest["scenes"][scene_id]["selected"]

    return splits


def build_per_scene_splits(manifest, scene_id):
    info = manifest["scenes"][scene_id]
    return {
        "train": {scene_id: info["train"]},
        "val": {scene_id: info["val"]},
        "test": {scene_id: info["test"]},
    }


def setup_model(first_scenario, hidden_channels, num_cnn_layers, mlp_hidden, mlp_num_layers, device):
    n_spatial = int(first_scenario["spatial"].shape[0])
    n_global = int(first_scenario["global"].numel())

    model = FlexibleCovariateEncoder(
        n_spatial_features=n_spatial,
        n_global_features=n_global,
        hidden_channels=hidden_channels,
        num_cnn_layers=num_cnn_layers,
        mlp_hidden=mlp_hidden,
        mlp_num_layers=mlp_num_layers,
        eps=0.1,
        max_drift=0.8,
        max_eigenvalue=10.0,
        local_scale=0.1,
    ).to(device)

    with torch.no_grad():
        if hasattr(model, "global_mlp"):
            model.global_mlp[-1].bias.copy_(
                torch.tensor([0.35, 0.0, 0.35, 0.0, 0.0], device=device)
            )

    return model


def forward_model(model, scenario, device):
    spatial = scenario["spatial"].unsqueeze(0).to(device)
    global_feat = scenario["global"].reshape(1, -1).to(device)
    return model(spatial, global_feat)


def solve_eikonal_cpu(G, B, source_cpu, max_iter):
    G_cpu = G.detach().cpu()
    B_cpu = B.detach().cpu()
    source = source_cpu.unsqueeze(0).cpu().bool()

    problem = EikonalProblem(G_cpu, B_cpu, source)
    T_pred, _ = fast_sweeping_solve_numba(problem, max_iter=max_iter)
    return T_pred, problem


def compute_solver_gradients(T_pred, true_T, valid_mask, source_mask, problem):
    obs_mask = (
        valid_mask.bool()
        & torch.isfinite(true_T)
        & (T_pred[0] < 1e5)
        & (~source_mask.bool())
    )

    n_obs = int(obs_mask.sum().item())
    if n_obs < 10:
        return None

    residuals = T_pred[0][obs_mask] - true_T[obs_mask]
    loss_data = 0.5 * (residuals ** 2).mean()

    dL_dT = torch.zeros_like(T_pred)
    dL_dT[0][obs_mask] = residuals / n_obs

    if HAVE_NUMBA_BACKWARD:
        stencil_info = identify_active_stencils(T_pred, problem)
        lam = solve_adjoint_numba(stencil_info, dL_dT, problem, T_pred)
        dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)
    else:
        dL_dG, dL_dB = backward_pass(T_pred, dL_dT, problem)

    valid_float = valid_mask.float()[None, None, :, :]
    dL_dG = dL_dG * valid_float.expand(1, 3, -1, -1)
    dL_dB = dL_dB * valid_float.expand(1, 2, -1, -1)

    return {
        "loss_data": loss_data,
        "dL_dG": dL_dG,
        "dL_dB": dL_dB,
        "n_obs": n_obs,
    }


def make_band_cases(scenario, n_time_bands: int):
    arrival = scenario["arrival_cpu"]
    valid = scenario["valid_cpu"].bool()
    original_source = scenario["source_cpu"].bool()

    finite_times = arrival[valid]
    if finite_times.numel() < 10:
        return []

    max_t = float(finite_times.max().item())
    if max_t <= 1e-8:
        return []

    edges = torch.linspace(0.0, max_t, n_time_bands + 1)
    cases = []

    for k in range(n_time_bands):
        start = float(edges[k].item())
        end = float(edges[k + 1].item())

        if k == 0:
            source = original_source.clone()
        else:
            source = valid & (arrival <= start)

        target = valid & (arrival > start) & (arrival <= end)

        if int(source.sum()) < 1 or int(target.sum()) < 10:
            continue

        rel_arrival = arrival.clone()
        rel_arrival[valid] = arrival[valid] - start
        rel_arrival[~valid] = float("inf")

        cases.append({
            "source_cpu": source,
            "target_cpu": target,
            "arrival_cpu": rel_arrival,
            "band_index": k,
            "band_start": start,
            "band_end": end,
        })

    return cases


def train_one_case(
    model,
    optimizer,
    scenario,
    mode,
    n_time_bands,
    lambda_G,
    lambda_B,
    max_iter,
    grad_clip,
    device,
):
    model.train()
    optimizer.zero_grad()

    G, B, G_local, B_local = forward_model(model, scenario, device)

    tv_loss = compute_tv_regularization(
        G_local,
        B_local,
        lambda_G,
        lambda_B,
        TVMetricType.LOG_EUCLIDEAN,
    )

    data_losses = []
    n_obs_total = 0

    if mode == "old":
        cases = [{
            "source_cpu": scenario["source_cpu"],
            "target_cpu": scenario["valid_cpu"] & (~scenario["source_cpu"].bool()),
            "arrival_cpu": scenario["arrival_cpu"],
            "band_index": -1,
        }]
    elif mode == "band":
        cases = make_band_cases(scenario, n_time_bands)
    else:
        raise ValueError(mode)

    if not cases:
        return None

    tv_loss.backward(retain_graph=True)

    for case in cases:
        T_pred, problem = solve_eikonal_cpu(G, B, case["source_cpu"], max_iter=max_iter)

        grad_result = compute_solver_gradients(
            T_pred=T_pred,
            true_T=case["arrival_cpu"],
            valid_mask=case["target_cpu"],
            source_mask=case["source_cpu"],
            problem=problem,
        )

        if grad_result is None:
            continue

        dL_dG = grad_result["dL_dG"].to(device)
        dL_dB = grad_result["dL_dB"].to(device)

        retain = True
        G.backward(dL_dG, retain_graph=retain)
        B.backward(dL_dB, retain_graph=retain)

        data_losses.append(float(grad_result["loss_data"].item()))
        n_obs_total += int(grad_result["n_obs"])

    if not data_losses:
        optimizer.zero_grad(set_to_none=True)
        return None

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": float(np.mean(data_losses) + tv_loss.item()),
        "data_loss": float(np.mean(data_losses)),
        "reg_loss": float(tv_loss.item()),
        "n_obs": n_obs_total,
        "n_cases": len(data_losses),
    }


@torch.no_grad()
def evaluate_full_fire(model, scenarios, max_iter, device, max_eval_fires=None):
    model.eval()
    metrics = []

    eval_scenarios = scenarios[:]
    if max_eval_fires is not None:
        eval_scenarios = eval_scenarios[:max_eval_fires]

    for sc in eval_scenarios:
        try:
            G, B, _, _ = forward_model(model, sc, device)
            T_pred, _ = solve_eikonal_cpu(G, B, sc["source_cpu"], max_iter=max_iter)

            m = compute_metrics(
                pred_T=T_pred[0].numpy(),
                true_T=sc["arrival_cpu"].numpy(),
                valid_mask=sc["valid_cpu"].numpy().astype(bool),
            )
            m["scene_id"] = sc["scene_id"]
            m["event_id"] = sc["event_id"]
            metrics.append(m)

        except Exception as e:
            print(f"EVAL SKIP {sc['scene_id']}/{sc['event_id']}: {e}")

    return aggregate_metrics(metrics), metrics


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--experiment", choices=["cross_scene", "per_scene"], default="cross_scene")
    parser.add_argument("--scene_id", type=str, default=None)

    parser.add_argument("--mode", choices=["old", "band"], required=True)
    parser.add_argument("--feature_group", type=str, default=None)
    parser.add_argument("--n_time_bands", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_G", type=float, default=0.005)
    parser.add_argument("--lambda_B", type=float, default=0.005)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_cnn_layers", type=int, default=5)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_num_layers", type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--preload_device", type=str, default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_valid_pixels", type=int, default=500)
    parser.add_argument("--max_val_fires", type=int, default=None)
    parser.add_argument("--max_test_fires", type=int, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = json.loads(Path(args.manifest).expanduser().read_text())
    feature_group = args.feature_group or manifest.get("feature_group", "topo_weather_omitted")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "run_config.json", vars(args))
    save_json(output_dir / "manifest_snapshot.json", manifest)

    if args.experiment == "cross_scene":
        split_maps = build_cross_scene_splits(manifest)
    else:
        if args.scene_id is None:
            raise RuntimeError("--scene_id is required for --experiment per_scene")
        split_maps = build_per_scene_splits(manifest, args.scene_id)

    loader = Sim2RealFireLoader(manifest["data_root"])
    adapter = make_adapter(feature_group)

    print(f"Feature group: {feature_group}")
    print(f"Mode: {args.mode}")
    print(f"Experiment: {args.experiment}")
    print(f"Preload device: {args.preload_device}")
    print(f"Backward: {'numba' if HAVE_NUMBA_BACKWARD else 'plain'}")

    print("\nLoading train split into memory...")
    train_data = load_split_into_memory(
        loader,
        adapter,
        split_maps["train"],
        args.preload_device,
        args.min_valid_pixels,
    )
    print(f"Loaded train scenarios: {len(train_data)}")

    print("\nLoading val split into memory...")
    val_data = load_split_into_memory(
        loader,
        adapter,
        split_maps["val"],
        args.preload_device,
        args.min_valid_pixels,
    )
    print(f"Loaded val scenarios: {len(val_data)}")

    print("\nLoading test split into memory...")
    test_data = load_split_into_memory(
        loader,
        adapter,
        split_maps["test"],
        args.preload_device,
        args.min_valid_pixels,
    )
    print(f"Loaded test scenarios: {len(test_data)}")

    if not train_data:
        raise RuntimeError("No training scenarios loaded")

    model = setup_model(
        train_data[0],
        hidden_channels=args.hidden_channels,
        num_cnn_layers=args.num_cnn_layers,
        mlp_hidden=args.mlp_hidden,
        mlp_num_layers=args.mlp_num_layers,
        device=args.device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    history = []
    best_val_corr = -float("inf")
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        random.shuffle(train_data)

        step_stats = []
        for i, sc in enumerate(train_data, start=1):
            result = train_one_case(
                model=model,
                optimizer=optimizer,
                scenario=sc,
                mode=args.mode,
                n_time_bands=args.n_time_bands,
                lambda_G=args.lambda_G,
                lambda_B=args.lambda_B,
                max_iter=args.max_iter,
                grad_clip=args.grad_clip,
                device=args.device,
            )
            if result is not None:
                step_stats.append(result)

            if i % 25 == 0:
                print(f"  epoch {epoch} progress {i}/{len(train_data)}")

        if step_stats:
            train_summary = {
                "loss_mean": float(np.mean([x["loss"] for x in step_stats])),
                "data_loss_mean": float(np.mean([x["data_loss"] for x in step_stats])),
                "reg_loss_mean": float(np.mean([x["reg_loss"] for x in step_stats])),
                "n_steps": len(step_stats),
                "n_obs": int(sum(x["n_obs"] for x in step_stats)),
                "n_cases": int(sum(x["n_cases"] for x in step_stats)),
            }
        else:
            train_summary = {
                "loss_mean": float("nan"),
                "data_loss_mean": float("nan"),
                "reg_loss_mean": float("nan"),
                "n_steps": 0,
                "n_obs": 0,
                "n_cases": 0,
            }

        val_summary, val_per_fire = evaluate_full_fire(
            model,
            val_data,
            max_iter=args.max_iter,
            device=args.device,
            max_eval_fires=args.max_val_fires,
        )

        val_corr = val_summary.get("correlation_mean", float("nan"))
        if not math.isnan(val_corr):
            scheduler.step(val_corr)

        epoch_record = {
            "epoch": epoch,
            "elapsed_sec": float(time.time() - t0),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_summary,
            "val_full_fire": val_summary,
        }

        history.append(epoch_record)

        save_json(output_dir / "history.json", history)
        save_json(output_dir / "latest_val_per_fire.json", val_per_fire)

        print(
            f"\nEpoch {epoch}/{args.epochs} "
            f"loss={train_summary['loss_mean']:.4f} "
            f"data={train_summary['data_loss_mean']:.4f} "
            f"reg={train_summary['reg_loss_mean']:.4f} "
            f"val_corr={val_summary.get('correlation_mean', float('nan')):.4f} "
            f"val_relRMSE={val_summary.get('relative_rmse_mean', float('nan')):.4f} "
            f"time={epoch_record['elapsed_sec']:.1f}s"
        )

        if not math.isnan(val_corr) and val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_corr": best_val_corr,
                    "feature_group": feature_group,
                    "mode": args.mode,
                    "n_time_bands": args.n_time_bands,
                    "config": vars(args),
                },
                best_path,
            )
            print(f"  saved best model: {best_path}")

    save_json(output_dir / "train_only_done.json", {"done": True, "best_val_corr": best_val_corr})

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_summary, test_per_fire = evaluate_full_fire(
        model,
        test_data,
        max_iter=args.max_iter,
        device=args.device,
        max_eval_fires=args.max_test_fires,
    )

    save_json(output_dir / "test_metrics.json", test_summary)
    save_json(output_dir / "test_per_fire.json", test_per_fire)

    print("\nFinal test metrics:")
    print(json.dumps(test_summary, indent=2))


if __name__ == "__main__":
    main()
