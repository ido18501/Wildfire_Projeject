#!/usr/bin/env python3
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter, label
except Exception:
    gaussian_filter = None
    label = None

from experiments.wildfire.sim2real_loader import Sim2RealFireLoader, CovariateAdapter
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.forward.pytorch import EikonalProblem
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba


def read_data_root():
    manifest = Path("~/wildfire_small_subset_no014_manifest.json").expanduser()
    m = json.loads(manifest.read_text())
    return m["data_root"]


def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    sd = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        sd[k] = v
    return sd


def _numeric_index(k):
    # e.g. local_cnn.4.weight -> 4
    return int(k.split(".")[1])


def infer_model_from_state_dict(sd):
    conv_keys = sorted(
        [k for k in sd if k.startswith("local_cnn.") and k.endswith(".weight")],
        key=_numeric_index,
    )
    lin_keys = sorted(
        [k for k in sd if k.startswith("global_mlp.") and k.endswith(".weight")],
        key=_numeric_index,
    )
    if not conv_keys or not lin_keys:
        raise RuntimeError("Could not infer model architecture from checkpoint keys.")

    n_spatial = int(sd[conv_keys[0]].shape[1])
    hidden_channels = int(sd[conv_keys[0]].shape[0])
    num_cnn_layers = len(conv_keys)

    n_global = int(sd[lin_keys[0]].shape[1])
    mlp_hidden = int(sd[lin_keys[0]].shape[0]) if len(lin_keys) > 1 else 64
    mlp_num_layers = len(lin_keys)

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
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, {
        "n_spatial": n_spatial,
        "n_global": n_global,
        "hidden_channels": hidden_channels,
        "num_cnn_layers": num_cnn_layers,
        "mlp_hidden": mlp_hidden,
        "mlp_num_layers": mlp_num_layers,
    }


def collect_adapter_stats(loader, scene, max_events, sample_pixels, seed, feature_group):
    rng = np.random.default_rng(seed)
    adapter = CovariateAdapter(feature_group=feature_group)

    events = list(loader.scenes[scene]["events"])
    events = events[:max_events] if max_events else events

    spatial_samples = []
    global_samples = []
    grid_shape = None
    used_events = []

    for idx, event_id in enumerate(events):
        try:
            data = loader.load_scenario(scene, event_id)
            spatial, global_f = adapter(data)

            if torch.is_tensor(spatial):
                spatial = spatial.detach().cpu().numpy()
            if torch.is_tensor(global_f):
                global_f = global_f.detach().cpu().numpy()

            spatial = np.asarray(spatial, dtype=np.float32)
            global_f = np.asarray(global_f, dtype=np.float32)

            C, H, W = spatial.shape
            if grid_shape is None:
                grid_shape = (H, W)

            flat = spatial.reshape(C, -1)
            n = flat.shape[1]
            take = min(sample_pixels, n)
            pix_idx = rng.choice(n, size=take, replace=False)
            spatial_samples.append(flat[:, pix_idx])
            global_samples.append(global_f.reshape(-1))
            used_events.append(event_id)

        except Exception as e:
            print(f"[WARN] failed loading {scene}/{event_id}: {e}")

    if not spatial_samples:
        raise RuntimeError(f"No usable events loaded for scene {scene}")

    S = np.concatenate(spatial_samples, axis=1)  # C x samples
    G = np.stack(global_samples, axis=0)         # events x F

    def stats_by_channel(A, axis):
        return {
            "mean": np.nanmean(A, axis=axis).tolist(),
            "std": np.nanstd(A, axis=axis).tolist(),
            "q01": np.nanpercentile(A, 1, axis=axis).tolist(),
            "q05": np.nanpercentile(A, 5, axis=axis).tolist(),
            "q25": np.nanpercentile(A, 25, axis=axis).tolist(),
            "q50": np.nanpercentile(A, 50, axis=axis).tolist(),
            "q75": np.nanpercentile(A, 75, axis=axis).tolist(),
            "q95": np.nanpercentile(A, 95, axis=axis).tolist(),
            "q99": np.nanpercentile(A, 99, axis=axis).tolist(),
        }

    return {
        "scene": scene,
        "feature_group": feature_group,
        "grid_shape": grid_shape,
        "n_events_used": len(used_events),
        "used_events_first20": used_events[:20],
        "spatial": stats_by_channel(S, axis=1),
        "global": stats_by_channel(G, axis=0),
        "adapter_n_spatial": int(S.shape[0]),
        "adapter_n_global": int(G.shape[1]),
    }


def smooth_noise(shape, rng, sigma):
    z = rng.normal(size=shape).astype(np.float32)
    if gaussian_filter is not None:
        z = gaussian_filter(z, sigma=sigma)
    z = z - np.nanmean(z)
    z = z / (np.nanstd(z) + 1e-6)
    return z


def make_structured_synthetic_features(stats, n_spatial, n_global, seed):
    rng = np.random.default_rng(seed)
    H, W = stats["grid_shape"]

    spatial_stats = stats["spatial"]
    global_stats = stats["global"]

    if n_spatial > stats["adapter_n_spatial"]:
        raise RuntimeError(
            f"Model expects {n_spatial} spatial channels, but adapter produced only {stats['adapter_n_spatial']}."
        )
    if n_global > stats["adapter_n_global"]:
        raise RuntimeError(
            f"Model expects {n_global} global channels, but adapter produced only {stats['adapter_n_global']}."
        )

    y = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, W, dtype=np.float32)[None, :]

    # Shared directional structures so channels are not independent salt-and-pepper noise.
    main_theta = rng.uniform(0.15 * math.pi, 0.85 * math.pi)
    coord = np.cos(main_theta) * x + np.sin(main_theta) * y
    cross = -np.sin(main_theta) * x + np.cos(main_theta) * y
    shared_gradient = coord
    shared_bands = np.sin(3.0 * math.pi * coord + rng.uniform(0, 2 * math.pi))
    shared_cross = np.cos(2.0 * math.pi * cross + rng.uniform(0, 2 * math.pi))

    # A few broad patches representing fuel/terrain heterogeneity.
    bumps = np.zeros((H, W), dtype=np.float32)
    for _ in range(5):
        cy = rng.uniform(-0.8, 0.8)
        cx = rng.uniform(-0.8, 0.8)
        sy = rng.uniform(0.15, 0.45)
        sx = rng.uniform(0.15, 0.45)
        amp = rng.normal()
        bumps += amp * np.exp(-(((y - cy) ** 2) / (2 * sy * sy) + ((x - cx) ** 2) / (2 * sx * sx)))
    bumps = (bumps - bumps.mean()) / (bumps.std() + 1e-6)

    spatial = np.zeros((n_spatial, H, W), dtype=np.float32)
    channel_summaries = []

    for c in range(n_spatial):
        q01 = float(spatial_stats["q01"][c])
        q05 = float(spatial_stats["q05"][c])
        q50 = float(spatial_stats["q50"][c])
        q95 = float(spatial_stats["q95"][c])
        q99 = float(spatial_stats["q99"][c])
        std = float(spatial_stats["std"][c])
        if not np.isfinite(std) or std < 1e-6:
            spatial[c] = q50
            continue

        sign = -1.0 if (c % 2) else 1.0
        local_theta = main_theta + rng.normal(scale=0.35)
        local_coord = np.cos(local_theta) * x + np.sin(local_theta) * y

        noise = smooth_noise((H, W), rng, sigma=max(4, min(H, W) // 35))
        pattern = (
            0.65 * sign * shared_gradient
            + 0.35 * shared_bands
            + 0.25 * shared_cross
            + 0.40 * bumps
            + 0.25 * local_coord
            + 0.15 * noise
        )
        pattern = (pattern - pattern.mean()) / (pattern.std() + 1e-6)

        # Use scene-like scale, clipped to realistic empirical percentiles.
        field = q50 + 0.85 * std * pattern
        field = np.clip(field, q01, q99)

        # Keep within the robust observed range unless q01/q99 are degenerate.
        if q99 > q01:
            field = np.clip(field, q01, q99)
        else:
            field = np.clip(field, q05, q95)

        spatial[c] = field.astype(np.float32)
        channel_summaries.append({
            "channel": c,
            "synthetic_min": float(np.nanmin(field)),
            "synthetic_mean": float(np.nanmean(field)),
            "synthetic_max": float(np.nanmax(field)),
            "scene_q05": q05,
            "scene_q50": q50,
            "scene_q95": q95,
        })

    global_f = np.zeros((n_global,), dtype=np.float32)
    for i in range(n_global):
        q05 = float(global_stats["q05"][i])
        q50 = float(global_stats["q50"][i])
        q95 = float(global_stats["q95"][i])
        std = float(global_stats["std"][i])
        val = q50 + rng.normal(scale=0.55 * max(std, 1e-6))
        global_f[i] = np.clip(val, q05, q95)

    # Common adapter convention is often temp/humidity/wind_speed/wind_sin/wind_cos.
    # If this is true, this biases the synthetic fire toward stronger directional drift.
    if n_global >= 5:
        try:
            global_f[2] = float(global_stats["q85"][2])
        except Exception:
            global_f[2] = float(global_stats["q75"][2])
        angle = rng.uniform(0, 2 * math.pi)
        global_f[3] = math.sin(angle)
        global_f[4] = math.cos(angle)

    return spatial, global_f, channel_summaries


def make_three_source_mask(H, W, radius=2):
    source = np.zeros((H, W), dtype=bool)
    centers = [
        (int(0.25 * H), int(0.25 * W)),
        (int(0.70 * H), int(0.35 * W)),
        (int(0.46 * H), int(0.76 * W)),
    ]
    yy, xx = np.ogrid[:H, :W]
    for cy, cx in centers:
        source |= ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2)
    return source, centers


def metric_diagnostics(G, B):
    g11, g12, g22 = G[0], G[1], G[2]
    trace = g11 + g22
    disc = np.sqrt((g11 - g22) ** 2 + 4 * g12 ** 2 + 1e-12)
    lam_max = (trace + disc) / 2
    lam_min = np.maximum((trace - disc) / 2, 1e-8)
    ratio = lam_max / lam_min
    drift = np.sqrt(B[0] ** 2 + B[1] ** 2)
    return ratio, drift


def spread_shape_diagnostics(T, source):
    finite = np.isfinite(T) & (T < 1e5)
    vals = T[finite & ~source]
    if vals.size < 100:
        return {"ok": False}
    horizon = float(np.nanpercentile(vals, 45))
    mask = finite & (T <= horizon)
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return {"ok": False, "horizon": horizon}
    X = np.stack([xs, ys], axis=1).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    cov = np.cov(X.T)
    evals = np.linalg.eigvalsh(cov)
    eccentricity = float(np.sqrt(max(evals) / max(min(evals), 1e-6)))
    return {
        "ok": True,
        "horizon_p45": horizon,
        "burned_pixels_at_horizon": int(mask.sum()),
        "eccentricity_sqrt_eig_ratio": eccentricity,
    }


def plot_outputs(out_png, spatial, G, B, T, source, title):
    ratio, drift = metric_diagnostics(G, B)

    finite = np.isfinite(T) & (T < 1e5)
    vals = T[finite & ~source]
    levels = np.nanpercentile(vals, [15, 30, 45, 60, 75, 90]) if vals.size else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.imshow(spatial[0])
    ax.set_title("synthetic spatial channel 0")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(spatial[min(1, spatial.shape[0]-1)])
    ax.set_title("synthetic spatial channel 1")
    ax.axis("off")

    ax = axes[0, 2]
    showT = np.where(finite, T, np.nan)
    ax.imshow(showT)
    if levels is not None:
        ax.contour(showT, levels=levels, linewidths=0.8)
    sy, sx = np.where(source)
    ax.scatter(sx, sy, s=4)
    ax.set_title("arrival time contours + 3 sources")
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(ratio)
    ax.set_title("metric anisotropy ratio")
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(drift)
    ax.set_title("drift magnitude")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(showT)
    step = max(8, min(T.shape) // 30)
    yy, xx = np.mgrid[0:T.shape[0]:step, 0:T.shape[1]:step]
    ax.quiver(xx, yy, B[0, ::step, ::step], B[1, ::step, ::step], angles="xy")
    ax.set_title("arrival time + drift quiver")
    ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="0002_00714")
    ap.add_argument("--groups", nargs="+", default=["baseline", "topo_weather_omitted"])
    ap.add_argument("--max_events", type=int, default=200)
    ap.add_argument("--sample_pixels", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_iter", type=int, default=60)
    args = ap.parse_args()

    repo = Path.cwd()
    data_root = read_data_root()
    loader = Sim2RealFireLoader(data_root)

    if args.scene not in loader.scenes:
        raise RuntimeError(f"Scene {args.scene} not found. Available: {sorted(loader.scenes)[:20]} ...")

    root_rel_path = Path("/tmp/fire_firstframe_ROOT_OUT.txt")
    if root_rel_path.exists():
        root_rel = root_rel_path.read_text().strip()
    else:
        root_rel = "small_runs/band2_firstframe_exact_all5_20260703_160412"
    root = repo / root_rel

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = repo / "analysis_outputs" / f"synthetic_3source_covstats_{args.scene}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DATA_ROOT={data_root}")
    print(f"RUN_ROOT={root}")
    print(f"OUT_DIR={out_dir}")

    summary = {
        "scene": args.scene,
        "data_root": data_root,
        "run_root": str(root),
        "out_dir": str(out_dir),
        "groups": {},
    }

    for gi, group in enumerate(args.groups):
        run_dir = root / group / f"per_scene_{args.scene}_band2_lam002_fullcorr"
        ckpt = run_dir / "model_best_by_full_corr.pt"
        if not ckpt.exists():
            print(f"[WARN] missing checkpoint for {group}: {ckpt}")
            continue

        sd = load_state_dict(ckpt)
        model, arch = infer_model_from_state_dict(sd)

        stats = collect_adapter_stats(
            loader=loader,
            scene=args.scene,
            max_events=args.max_events,
            sample_pixels=args.sample_pixels,
            seed=args.seed + 17 * gi,
            feature_group=group,
        )
        stats_path = out_dir / f"adapter_covariate_stats_{group}.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"Saved {group} adapter stats: {stats_path}")

        spatial, global_f, chan_summary = make_structured_synthetic_features(
            stats, arch["n_spatial"], arch["n_global"], seed=args.seed + 100 * gi
        )

        H, W = spatial.shape[1:]
        source, centers = make_three_source_mask(H, W, radius=2)

        with torch.no_grad():
            spatial_t = torch.from_numpy(spatial).unsqueeze(0).float()
            global_t = torch.from_numpy(global_f).unsqueeze(0).float()
            G_t, B_t, G_local, B_local = model(spatial_t, global_t)

        G_cpu = G_t.detach().cpu()
        B_cpu = B_t.detach().cpu()
        source_t = torch.from_numpy(source).unsqueeze(0).bool()

        problem = EikonalProblem(G_cpu, B_cpu, source_t)
        T, n_iter = fast_sweeping_solve_numba(problem, max_iter=args.max_iter)

        G_np = G_cpu[0].numpy()
        B_np = B_cpu[0].numpy()
        T_np = T[0].detach().cpu().numpy()

        ratio, drift = metric_diagnostics(G_np, B_np)
        spread_diag = spread_shape_diagnostics(T_np, source)

        group_npz = out_dir / f"synthetic_3source_{group}.npz"
        np.savez_compressed(
            group_npz,
            spatial_feat=spatial,
            global_feat=global_f,
            source_mask=source,
            source_centers=np.asarray(centers),
            G=G_np,
            B=B_np,
            pred_T=T_np,
            anisotropy_ratio=ratio,
            drift_magnitude=drift,
        )

        group_png = out_dir / f"synthetic_3source_{group}.png"
        plot_outputs(
            group_png,
            spatial,
            G_np,
            B_np,
            T_np,
            source,
            title=f"{args.scene} / {group} synthetic 3-source feature stress test",
        )

        group_json = out_dir / f"synthetic_3source_{group}_summary.json"
        group_summary = {
            "group": group,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            "architecture": arch,
            "source_centers": centers,
            "n_source_pixels": int(source.sum()),
            "adapter_stats": str(stats_path),
            "stats_events_used": stats["n_events_used"],
            "grid_shape": stats["grid_shape"],
            "spatial_feature_names": getattr(CovariateAdapter(feature_group=group), "spatial_feature_names", None),
            "global_feature_names": getattr(CovariateAdapter(feature_group=group), "global_feature_names", None),
            "solver_iterations": int(n_iter) if isinstance(n_iter, (int, np.integer)) else str(n_iter),
            "anisotropy_ratio_mean": float(np.nanmean(ratio)),
            "anisotropy_ratio_p95": float(np.nanpercentile(ratio, 95)),
            "drift_magnitude_mean": float(np.nanmean(drift)),
            "drift_magnitude_p95": float(np.nanpercentile(drift, 95)),
            "spread_shape": spread_diag,
            "channel_summaries": chan_summary,
            "outputs": {
                "npz": str(group_npz),
                "png": str(group_png),
                "json": str(group_json),
            },
        }
        group_json.write_text(json.dumps(group_summary, indent=2))
        summary["groups"][group] = group_summary

        print()
        print("=" * 80)
        print(group)
        print("=" * 80)
        print(f"checkpoint: {ckpt}")
        print(f"model input: spatial={arch['n_spatial']} global={arch['n_global']}")
        print(f"source centers: {centers}, source pixels={source.sum()}")
        print(f"anisotropy ratio mean/p95: {group_summary['anisotropy_ratio_mean']:.3f} / {group_summary['anisotropy_ratio_p95']:.3f}")
        print(f"drift magnitude mean/p95: {group_summary['drift_magnitude_mean']:.4f} / {group_summary['drift_magnitude_p95']:.4f}")
        print(f"spread shape: {spread_diag}")
        print(f"saved npz: {group_npz}")
        print(f"saved png: {group_png}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    tmp = Path("/tmp/synthetic_3source_covstats_OUT.txt")
    tmp.write_text(str(out_dir.relative_to(repo)))
    print()
    print(f"Saved summary: {summary_path}")
    print(f"Wrote relative output path to: {tmp}")


if __name__ == "__main__":
    main()
