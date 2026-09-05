"""Fit exact homogeneous cardiac baselines with one shared training latency.

Run: python -m experiments.cardiac.analytical_baseline
Writes only compact metrics/parameters JSON under the ignored runs_3d directory.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from eikonal.three_d.analytic import analytical_arrival
from .timeactml import load_timeactml, prepare_split, sample_training_voxels, training_latency


def fit_analytical(dataset, train_ids, model="isotropic", latency_ms=None,
                   samples_per_case=0, seed=20260905):
    """Least-squares calibration of positive log-speeds, shared over all cases.

    The canonical package protocol uses all training voxels (samples=0) with
    equal case/voxel weighting, retaining every repeated-source case. Isotropic
    least-squares slowness has a closed form; diagonal log-speeds are optimized
    with an analytic Jacobian and two deterministic initializations. Positive
    sample counts select a separately labeled subsampled experiment.
    """
    started = perf_counter()
    if model not in ("isotropic", "diagonal") or not len(train_ids):
        raise ValueError("Require nonempty training cases and isotropic or diagonal model")
    if latency_ms is None:
        latency_ms = training_latency(dataset, train_ids)
    samples = sample_training_voxels(dataset, train_ids, samples_per_case, seed)
    positions = np.indices(dataset.shape_zyx, dtype=np.float64).reshape(3, -1).T[:, ::-1]
    displacements, targets = [], []
    for case_id, indices in zip(train_ids, samples):
        source_xyz = dataset.sources_nzyx[case_id, ::-1]
        displacements.append((positions[indices] - source_xyz) * dataset.spacing_xyz)
        targets.append(dataset.activation_nzyx[case_id].ravel()[indices])
    offsets = np.concatenate(displacements)
    observed = np.concatenate(targets)
    centered = observed - latency_ms
    squares = offsets ** 2
    radius = np.sqrt(squares.sum(axis=1))
    isotropic_speed = float(np.dot(radius, radius) / np.dot(radius, centered))
    if not np.isfinite(isotropic_speed) or isotropic_speed <= 0:
        raise ValueError("No positive isotropic least-squares solution for these training data")
    normalizer = np.sqrt(len(observed))

    def residual(log_speeds):
        speeds = np.exp(log_speeds)
        return (np.sqrt(np.sum((offsets / speeds) ** 2, axis=1)) - centered) / normalizer

    def jacobian(log_speeds):
        terms = squares / np.exp(2 * log_speeds)
        times = np.sqrt(terms.sum(axis=1))
        jac = np.zeros_like(terms)
        np.divide(-terms, times[:, None], out=jac, where=times[:, None] > 0)
        return jac / normalizer

    size = 1 if model == "isotropic" else 3
    if size == 1:
        speeds = np.repeat(isotropic_speed, 3)
        mse = float(np.mean((radius / isotropic_speed - centered) ** 2))
        message, nfev, attempts = "Closed-form least-squares isotropic slowness", 0, []
    else:
        squared_slowness_guess = np.linalg.lstsq(squares, centered ** 2, rcond=None)[0]
        starts = [np.full(3, np.log(isotropic_speed)),
                  -0.5 * np.log(np.maximum(squared_slowness_guess, (1 / isotropic_speed) ** 2 * 1e-4))]
        fits = [least_squares(residual, start, jac=jacobian, xtol=1e-12, ftol=1e-12,
                              gtol=1e-10, max_nfev=500) for start in starts]
        if any(not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun).all() for fit in fits):
            raise RuntimeError("Analytical diagonal log-speed fit failed")
        fit = min(fits, key=lambda f: f.cost)
        speeds, mse = np.exp(fit.x), float(2 * fit.cost)
        message, nfev = str(fit.message), sum(f.nfev for f in fits)
        attempts = [{"speeds_xyz_mm_per_ms": np.exp(f.x).tolist(), "training_mse_ms2": float(2 * f.cost),
                     "nfev": f.nfev, "success": bool(f.success)} for f in fits]
    return {"model": model, "speeds_xyz_mm_per_ms": speeds.tolist(), "latency_ms": float(latency_ms),
            "training_sample_mse_ms2": mse, "optimizer_success": True,
            "optimizer_message": message, "nfev": nfev, "attempts": attempts,
            "speed_parameterization": "positive log-speeds; isotropic closed-form positive slowness",
            "fit_protocol": "canonical full-volume" if samples_per_case == 0 else "noncanonical subsampled training loss",
            "training_case_indices": [int(i) for i in train_ids], "samples_per_case": len(samples[0]),
            "sampling_seed": seed if samples_per_case else None,
            "sampling_axes": "all voxels in C-ravel ZYX order" if samples_per_case == 0 else "C-ravel ZYX, default_rng.choice sequentially per case",
            "elapsed_seconds": perf_counter() - started}


def evaluate_predictions(dataset, case_ids, predict_propagation, latency_ms):
    """Full-volume metrics; callback takes sourceZYX and returns propagationZYX.

    Activation prediction is the same frozen latency plus propagation for every
    source. Depth refers both to voxel Z layers and to stimulus source depth;
    distance bins use Euclidean source distance in mm, independently of speeds.
    """
    started = perf_counter()
    if not len(case_ids):
        raise ValueError("Cannot evaluate an empty partition")
    absolute_sum, squared_sum, count = 0.0, 0.0, 0
    per_case, source_depth = [], {}
    layer_abs = np.zeros(dataset.shape_zyx[0], dtype=np.float64)
    layer_sq = np.zeros_like(layer_abs)
    layer_count = np.zeros_like(layer_abs, dtype=np.int64)
    edges = np.array([0.0, 2.4, 8.0, 16.0, 32.0, np.inf])
    bin_abs, bin_sq = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    bin_count = np.zeros(len(edges) - 1, dtype=np.int64)
    for case_id in case_ids:
        source = tuple(int(s) for s in dataset.sources_nzyx[case_id])
        propagation = np.asarray(predict_propagation(source), dtype=np.float64)
        if propagation.shape != dataset.shape_zyx or not np.isfinite(propagation).all():
            raise ValueError("Prediction must be a finite ZYX volume matching the target shape")
        if abs(propagation[source]) > 1e-9:
            raise ValueError("Propagation source time must be zero")
        observed = dataset.activation_nzyx[case_id]
        error = propagation + latency_ms - observed
        absolute, squared = np.abs(error), error ** 2
        mae, rmse = float(absolute.mean()), float(np.sqrt(squared.mean()))
        norm = np.linalg.norm(observed)
        per_case.append({"case_index": int(case_id), "source_zyx": list(source), "mae_ms": mae,
                         "rmse_ms": rmse, "relative_l2": float(np.linalg.norm(error) / norm) if norm else None})
        absolute_sum += float(absolute.sum())
        squared_sum += float(squared.sum())
        count += error.size
        layer_abs += absolute.sum(axis=(1, 2))
        layer_sq += squared.sum(axis=(1, 2))
        layer_count += error.shape[1] * error.shape[2]
        group = source_depth.setdefault(source[0], [])
        group.append((mae, rmse))
        distances = analytical_arrival(dataset.shape_zyx, source, 1.0, dataset.spacing_xyz)
        for b in range(len(edges) - 1):
            mask = (distances >= edges[b]) & (distances < edges[b + 1])
            bin_abs[b] += absolute[mask].sum()
            bin_sq[b] += squared[mask].sum()
            bin_count[b] += int(mask.sum())
    case_maes = [row["mae_ms"] for row in per_case]
    return {"case_count": len(case_ids), "voxel_count": count, "mae_ms": absolute_sum / count,
            "rmse_ms": float(np.sqrt(squared_sum / count)), "latency_ms": float(latency_ms),
            "per_case_mae_distribution_ms": dict(zip(["min", "p25", "median", "p75", "p90", "max"],
                                                     np.percentile(case_maes, [0, 25, 50, 75, 90, 100]).tolist())),
            "per_case": per_case,
            "voxel_depth": [{"z_index": z, "z_mm": z * dataset.spacing_xyz[2],
                              "mae_ms": float(layer_abs[z] / layer_count[z]),
                              "rmse_ms": float(np.sqrt(layer_sq[z] / layer_count[z]))}
                             for z in range(len(layer_abs))],
            "source_depth": [{"source_z_index": int(z), "case_count": len(rows),
                               "mae_ms": float(np.mean([r[0] for r in rows])),
                               "rmse_ms": float(np.sqrt(np.mean([r[1] ** 2 for r in rows])))}
                              for z, rows in sorted(source_depth.items())],
            "source_distance": [{"min_mm_inclusive": float(edges[b]),
                                 "max_mm_exclusive": float(edges[b + 1]) if np.isfinite(edges[b + 1]) else None,
                                 "voxel_count": int(bin_count[b]),
                                 "mae_ms": float(bin_abs[b] / bin_count[b]) if bin_count[b] else None,
                                 "rmse_ms": float(np.sqrt(bin_sq[b] / bin_count[b])) if bin_count[b] else None}
                                for b in range(len(bin_count))],
            "elapsed_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path, help="Directory containing train/validation/test JSON manifests")
    parser.add_argument("--samples-per-case", type=int, default=0, help="0 (default): canonical full-volume fit; positive: noncanonical sampled fit")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, default=Path("runs_3d/analytical_baseline.json"))
    args = parser.parse_args()
    started = perf_counter()
    dataset = load_timeactml(args.data)
    split = prepare_split(dataset, args.manifest, seed=args.seed)
    latency = training_latency(dataset, split["train"])
    result = {"dataset": str(dataset.path), "shape_nzyx": list(dataset.activation_nzyx.shape),
              "spacing_xyz_mm": list(dataset.spacing_xyz), "split": split, "latency_ms": latency, "models": {}}
    print(f"Loaded shape={dataset.activation_nzyx.shape} source_groups={len(np.unique(dataset.sources_nzyx, axis=0))} "
          f"split={len(split['train'])}/{len(split['validation'])}/{len(split['test'])} latency_ms={latency:.9f}", flush=True)
    for name in ("isotropic", "diagonal"):
        fit = fit_analytical(dataset, split["train"], name, latency, args.samples_per_case, args.seed)
        speeds = fit["speeds_xyz_mm_per_ms"]
        evaluations = {part: evaluate_predictions(dataset, split[part],
                        lambda source: analytical_arrival(dataset.shape_zyx, source, speeds, dataset.spacing_xyz), latency)
                       for part in ("train", "validation", "test")}
        result["models"][name] = {"fit": fit, "evaluation": evaluations}
        print(f"ANALYTICAL {name} speeds_xyz_mm_per_ms={speeds} "
              f"heldout_MAE_ms={evaluations['test']['mae_ms']:.6f} "
              f"heldout_RMSE_ms={evaluations['test']['rmse_ms']:.6f}", flush=True)
    result["elapsed_seconds"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.output} elapsed_seconds={result['elapsed_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
