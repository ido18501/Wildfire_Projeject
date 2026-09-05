"""Small CPU calibration through the converged numerical 3D eikonal solution.

Run from repository root: python -m experiments.cardiac.numerical_training --tiny
No checkpoints or arrays are written; only a compact JSON result is saved.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from eikonal.three_d import SharedSpeedModel, analytical_arrival, solve_diagonal
from .timeactml import load_timeactml, prepare_split, sample_training_voxels, training_latency
from .analytical_baseline import fit_analytical, evaluate_predictions


def _select_groups(dataset, case_ids, limit):
    """Select a prefix of frozen source groups, retaining every original case."""
    seen = set()
    for case_id in case_ids:
        source = tuple(dataset.sources_nzyx[case_id])
        if source not in seen:
            seen.add(source)
        if len(seen) == limit:
            break
    if len(seen) < limit:
        raise ValueError(f"Requested {limit} sources, only {len(seen)} available")
    return [int(i) for i in case_ids if tuple(dataset.sources_nzyx[i]) in seen]


def fit_numerical(dataset, train_ids, latency_ms, initial_speeds, isotropic,
                  max_iter=25, samples_per_case=0, seed=20260905, backend="auto"):
    """Fit one tissue model shared by every case; gradients traverse the solver."""
    started = perf_counter()
    initial = float(initial_speeds[0]) if isotropic else initial_speeds
    model = SharedSpeedModel(initial, isotropic=isotropic)
    before = model.speeds_xyz.detach().numpy().copy()
    samples = sample_training_voxels(dataset, train_ids, samples_per_case, seed)
    target_samples = [torch.from_numpy(np.array(dataset.activation_nzyx[i].ravel()[j], copy=True))
                      for i, j in zip(train_ids, samples)]
    # L-BFGS needs few iterations for one/three shared log-speeds. Every closure
    # runs actual volumetric sweeps and their implicit backward pass per case.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                                 max_eval=max_iter * 3, tolerance_grad=1e-7,
                                 tolerance_change=1e-10, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        value = 0.0
        for case_id, indices, target in zip(train_ids, samples, target_samples):
            propagation = model(dataset.shape_zyx, tuple(dataset.sources_nzyx[case_id]),
                                dataset.spacing_xyz, backend=backend)
            if not torch.isfinite(propagation).all():
                raise FloatingPointError("Nonfinite numerical arrival")
            loss = ((propagation.flatten()[indices] + latency_ms - target)**2).mean() / len(train_ids)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite numerical training loss")
            loss.backward()
            value += float(loss.detach())
        gradient = model.log_speeds.grad.detach().numpy().copy()
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite log-speed gradient")
        history.append({"evaluation": len(history), "mse_ms2": value,
                        "speeds_xyz_mm_per_ms": model.speeds_xyz.detach().tolist(),
                        "gradient_log_speeds": gradient.tolist()})
        if len(history) == 1 or len(history) % 5 == 0:
            print(f"NUMERICAL {'isotropic' if isotropic else 'diagonal'} evaluation={len(history)} "
                  f"MSE_ms2={value:.8f} speeds={model.speeds_xyz.detach().tolist()}", flush=True)
        return torch.tensor(value, dtype=torch.float64)

    optimizer.step(closure)
    closure()  # Explicitly evaluate the final accepted parameters.
    after = model.speeds_xyz.detach().numpy().copy()
    if not np.isfinite(after).all() or (after <= 0).any():
        raise FloatingPointError("Invalid fitted propagation speeds")
    changed = not np.allclose(before, after, rtol=1e-8, atol=1e-10)
    decreased = history[-1]["mse_ms2"] < history[0]["mse_ms2"]
    if not changed or not decreased:
        raise RuntimeError("Numerical optimization did not change parameters and decrease loss")
    return {"speeds_xyz_mm_per_ms": after.tolist(), "initial_speeds_xyz_mm_per_ms": before.tolist(),
            "latency_ms": float(latency_ms), "initial_mse_ms2": history[0]["mse_ms2"],
            "final_mse_ms2": history[-1]["mse_ms2"], "parameter_changed": changed,
            "loss_decreased": decreased, "finite_gradients": True,
            "history": history, "elapsed_seconds": perf_counter() - started}


def discretization_comparison(dataset, case_ids, speeds, backend="auto", refine_factor=2):
    absolute, squared, count = 0.0, 0.0, 0
    maximum, max_residual, max_cycles, elapsed = 0.0, 0.0, 0, 0.0
    for case_id in case_ids:
        source = tuple(dataset.sources_nzyx[case_id])
        result = solve_diagonal(dataset.shape_zyx, source, speeds, dataset.spacing_xyz, backend=backend)
        exact = analytical_arrival(dataset.shape_zyx, source, speeds, dataset.spacing_xyz)
        error = result.arrival - exact
        absolute += np.abs(error).sum()
        squared += (error**2).sum()
        count += error.size
        maximum = max(maximum, float(np.abs(error).max()))
        max_residual = max(max_residual, result.max_residual)
        max_cycles = max(max_cycles, result.cycles)
        elapsed += result.elapsed_seconds
    metrics = {"mae_ms": float(absolute / count), "rmse_ms": float(np.sqrt(squared / count)),
               "max_abs_ms": maximum, "max_residual": max_residual, "max_cycles": max_cycles,
               "mean_solve_seconds": elapsed / len(case_ids)}
    if refine_factor > 1:
        case_id = case_ids[0]
        source = tuple(dataset.sources_nzyx[case_id])
        coarse = solve_diagonal(dataset.shape_zyx, source, speeds, dataset.spacing_xyz, backend=backend)
        refined = solve_diagonal(tuple((n - 1) * refine_factor + 1 for n in dataset.shape_zyx),
                                 tuple(int(s) * refine_factor for s in source), speeds,
                                 tuple(h / refine_factor for h in dataset.spacing_xyz), backend=backend)
        exact = analytical_arrival(dataset.shape_zyx, source, speeds, dataset.spacing_xyz)
        fine_at_coarse = refined.arrival[::refine_factor, ::refine_factor, ::refine_factor]
        coarse_mae = float(np.abs(coarse.arrival - exact).mean())
        refined_mae = float(np.abs(fine_at_coarse - exact).mean())
        if refined_mae >= coarse_mae:
            raise RuntimeError("Cardiac fixed-parameter refinement did not reduce analytical error")
        metrics["refinement"] = {"case_index": int(case_id), "factor": refine_factor,
                                 "coarse_mae_ms": coarse_mae, "refined_mae_ms": refined_mae,
                                 "comparison": "same physical voxels, source, and speeds",
                                 "refined_solve_seconds": refined.elapsed_seconds}
    return metrics


def run(args):
    torch.set_num_threads(1)
    started = perf_counter()
    dataset = load_timeactml(args.data)
    partition = prepare_split(dataset, manifest=args.manifest)
    counts = (2, 2, 4) if args.tiny else (args.train_cases, args.validation_cases, args.test_cases)
    ids = {name: _select_groups(dataset, partition[name], count)
           for name, count in zip(("train", "validation", "test"), counts)}
    # Even the tiny optimization uses the frozen package's 16-source latency.
    # No validation/test value contributes, and no map is individually aligned.
    latency = training_latency(dataset, partition["train"])
    max_iter = min(args.max_iter, 8) if args.tiny else args.max_iter
    print(f"DATA shape={dataset.activation_nzyx.shape} source_counts={counts} "
          f"case_counts={[len(ids[p]) for p in ('train','validation','test')]} latency_ms={latency:.9f}", flush=True)
    result = {"scope": "shared diagonal 3D eikonal, b=0; simulated cardiac first activation",
              "data_path": str(dataset.path), "shape_zyx": list(dataset.shape_zyx),
              "spacing_xyz_mm": list(dataset.spacing_xyz), "latency_ms": float(latency),
              "latency_protocol": "mean of all frozen training source activations; fixed even for tiny optimization",
              "case_indices": ids, "split_provenance": partition.get("provenance"),
              "repeat_policy": "all cases in selected frozen source groups retained; equal case/voxel weighting",
              "canonical_analytical_training_case_indices": partition["train"],
              "samples_per_training_case": args.samples_per_case, "seed": args.seed,
              "optimizer": "torch L-BFGS, strong Wolfe, shared log-speeds", "max_iter": max_iter,
              "models": {}, "discretization_caution": "A smaller cardiac error caused by numerical bias is not a model improvement."}
    analytical_fits = {}
    # Establish both exact baselines before starting numerical optimization.
    for name in ("isotropic", "diagonal"):
        analytical = fit_analytical(dataset, partition["train"], model=name, latency_ms=latency,
                                    samples_per_case=0, seed=args.seed)
        speeds = np.asarray(analytical["speeds_xyz_mm_per_ms"])
        analytical["evaluation"] = {}
        for split in ("validation", "test"):
            analytical["evaluation"][split] = evaluate_predictions(
                dataset, ids[split], lambda source: analytical_arrival(dataset.shape_zyx, source, speeds,
                                                                       dataset.spacing_xyz), latency)
        print(f"ANALYTICAL {name} speeds={speeds.tolist()}", flush=True)
        analytical_fits[name] = analytical
    for name in ("isotropic", "diagonal"):
        analytical = analytical_fits[name]
        speeds = np.asarray(analytical["speeds_xyz_mm_per_ms"])
        # Compare the numerical and exact predictions at the canonical fitted
        # parameters before optimizing any numerical model.
        fixed_parameter_test = evaluate_predictions(
            dataset, ids["test"], lambda source: solve_diagonal(dataset.shape_zyx, source, speeds,
                                                                dataset.spacing_xyz, backend=args.backend).arrival, latency)
        print(f"FIXED_PARAMETERS {name} numerical_test_MAE_ms={fixed_parameter_test['mae_ms']:.8f} "
              f"exact_test_MAE_ms={analytical['evaluation']['test']['mae_ms']:.8f}", flush=True)
        numerical = fit_numerical(dataset, ids["train"], latency, speeds * 0.85,
                                   isotropic=name == "isotropic", max_iter=max_iter,
                                   samples_per_case=args.samples_per_case, seed=args.seed, backend=args.backend)
        fitted = np.asarray(numerical["speeds_xyz_mm_per_ms"])
        numerical["evaluation"] = {}
        for split in ("validation", "test"):
            numerical["evaluation"][split] = evaluate_predictions(
                dataset, ids[split], lambda source: solve_diagonal(dataset.shape_zyx, source, fitted,
                                                                  dataset.spacing_xyz, backend=args.backend).arrival,
                latency)
        # Both cross-comparisons separate fitting effects from discretization.
        numerical["analytical_at_numerically_fitted_speeds_test"] = evaluate_predictions(
            dataset, ids["test"], lambda source: analytical_arrival(dataset.shape_zyx, source, fitted,
                                                                    dataset.spacing_xyz), latency)
        numerical["numerical_at_analytically_fitted_speeds_test"] = fixed_parameter_test
        numerical["discretization_at_numerically_fitted_speeds"] = discretization_comparison(
            dataset, ids["test"], fitted, args.backend, args.refine_factor)
        result["models"]["analytical_" + name] = analytical
        result["models"]["numerical_" + name] = numerical
        summary = lambda metrics: {key: value for key, value in metrics.items()
                                   if isinstance(value, (int, float))}
        print(f"EVALUATION {name} analytical_test={summary(analytical['evaluation']['test'])} "
              f"numerical_test={summary(numerical['evaluation']['test'])}", flush=True)
    result["elapsed_seconds"] = perf_counter() - started
    output = Path(args.output or ("runs_3d/tiny_numerical.json" if args.tiny else "runs_3d/numerical.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"PASS numerical cardiac calibration in {result['elapsed_seconds']:.3f} s; result={output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to raw dataset_acti_3d_v1.mat; auto-discovered in local_data")
    parser.add_argument("--manifest", help="Source-grouped manifest file/directory; auto-discovered if omitted")
    parser.add_argument("--tiny", action="store_true", help="2 training, 2 validation, 4 test sources; at most 8 iterations")
    parser.add_argument("--train-cases", type=int, default=16, help="Frozen training source groups; all group members retained")
    parser.add_argument("--validation-cases", type=int, default=16, help="Frozen validation source groups")
    parser.add_argument("--test-cases", type=int, default=100, help="Frozen test source groups (100 groups contain 102 cases)")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--samples-per-case", type=int, default=0, help="Numerical training loss voxels; 0 uses full volumes")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--backend", choices=("auto", "python", "numba"), default="auto")
    parser.add_argument("--refine-factor", type=int, default=2, help="Refine one held-out source at fixed physical geometry")
    parser.add_argument("--output")
    args = parser.parse_args()
    for field in ("train_cases", "validation_cases", "test_cases", "max_iter", "refine_factor"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.samples_per_case < 0:
        parser.error("--samples-per-case must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
