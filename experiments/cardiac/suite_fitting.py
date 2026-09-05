"""Isolated fitting/evaluation primitives for the unattended cardiac suite.

Every loss uses original, observed labels and equal case weight. Refinement
changes only the solver lattice: observations and source positions map exactly
to integer multiples of the refinement factor. Existing validated APIs are
intentionally left unchanged.
"""
import copy
import hashlib
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import least_squares

from eikonal.three_d import SharedSpeedModel, analytical_arrival, solve_diagonal


SOLVER_DEFAULTS = {"tol": 1e-10, "residual_tol": 1e-8, "max_cycles": 64}
DISTANCE_EDGES_MM = (0.0, 2.4, 8.0, 16.0, 32.0, np.inf)
REGION_EDGES_MM = (0.0, 2.4, 8.0, np.inf)


def refined_geometry(shape_zyx, spacing_xyz, source_zyx, factor=1):
    """Preserve physical extent and put each original source at source*factor."""
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("refine_factor must be a positive integer")
    return (tuple((int(n) - 1) * factor + 1 for n in shape_zyx),
            tuple(float(h) / factor for h in spacing_xyz),
            tuple(int(s) * factor for s in source_zyx))


def map_observation_indices(indices, shape_zyx, factor=1):
    """Map C-flat ZYX observation indices without interpolating any labels."""
    refined_shape, _, _ = refined_geometry(shape_zyx, (1., 1., 1.), (0, 0, 0), factor)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or (indices < 0).any() or (indices >= np.prod(shape_zyx)).any():
        raise ValueError("Original observation indices must be in-bounds C-flat ZYX indices")
    coordinates = np.asarray(np.unravel_index(indices, shape_zyx)) * factor
    return np.ravel_multi_index(tuple(coordinates), refined_shape).astype(np.int64)


def observation_indices(dataset, case_ids, samples_per_case=0, sampling_seed=20260905,
                        exclude_distance_mm=0.0):
    """Deterministic original-node indices, using only training source geometry.

    A single PCG64 generator draws without replacement, sequentially in the
    stored training-case order, from the C-ravel ZYX eligible nodes. Samples
    are sorted after drawing. Distances equal to the cutoff remain eligible.
    """
    if (isinstance(samples_per_case, bool) or
            not isinstance(samples_per_case, (int, np.integer)) or samples_per_case < 0):
        raise ValueError("samples_per_case must be a nonnegative integer")
    if not np.isfinite(exclude_distance_mm) or exclude_distance_mm < 0:
        raise ValueError("exclude_distance_mm must be finite and nonnegative")
    rng = np.random.default_rng(sampling_seed)
    result = []
    for case_id in case_ids:
        source = tuple(int(v) for v in dataset.sources_nzyx[case_id])
        distance = analytical_arrival(dataset.shape_zyx, source, 1., dataset.spacing_xyz).ravel()
        # One representable step avoids dropping a mathematically exact cutoff
        # node solely because its Euclidean norm rounded downward.
        eligible = np.flatnonzero(distance >= np.nextafter(float(exclude_distance_mm), -np.inf))
        if not eligible.size:
            raise ValueError("Training exclusion leaves no observations for case %s" % case_id)
        if samples_per_case and samples_per_case < eligible.size:
            eligible = np.sort(rng.choice(eligible, samples_per_case, replace=False))
        result.append(np.asarray(eligible, dtype=np.int64))
    return result


def _speeds(values, model):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Speeds must be one or three positive finite values")
    if model == "isotropic" and not np.allclose(values, values[0], rtol=1e-12, atol=0):
        raise ValueError("Isotropic model requires identical XYZ speeds")
    return values


def _observation_metadata(case_ids, indices, spec):
    return {"weighting": "mean over cases of mean squared observed-node error within each case",
            "samples_per_case_requested": int(spec.get("samples_per_case", 0)),
            "sampling_seed": spec.get("sampling_seed"),
            "sampling_rule": "PCG64 default_rng; eligible C-ravel ZYX nodes; sequential case order; choice without replacement; sorted",
            "exclude_distance_mm": float(spec.get("exclude_distance_mm", 0)),
            "refined_mapping": "original ZYX coordinate multiplied by refine_factor; target labels unchanged",
            "cases": [{"case_index": int(case_id), "observation_count": len(index),
                       "indices_sha256": hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest(),
                       **({"original_c_flat_zyx_indices": index.tolist()}
                          if spec.get("samples_per_case", 0) else {})}
                      for case_id, index in zip(case_ids, indices)]}


def _analytical_fit(dataset, case_ids, indices, model, latency, settings, checkpoint=None):
    started = perf_counter()
    positions = np.indices(dataset.shape_zyx, dtype=np.float64).reshape(3, -1).T[:, ::-1]
    offsets, centered, weights = [], [], []
    for case_id, index in zip(case_ids, indices):
        offsets.append((positions[index] - dataset.sources_nzyx[case_id, ::-1]) * dataset.spacing_xyz)
        centered.append(dataset.activation_nzyx[case_id].ravel()[index] - latency)
        weights.append(np.full(len(index), 1. / (len(case_ids) * len(index))))
    offsets, centered, weights = map(np.concatenate, (offsets, centered, weights))
    squares = offsets ** 2
    radius = np.sqrt(squares.sum(axis=1))
    denominator = np.dot(weights * radius, centered)
    speed = float(np.dot(weights, radius ** 2) / denominator) if denominator > 0 else np.nan
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("No finite positive isotropic least-squares solution")
    normalizer = np.sqrt(weights)
    history = []

    def residual(log_speeds):
        speeds = np.exp(log_speeds)
        value = (np.sqrt(np.sum((offsets / speeds) ** 2, axis=1)) - centered) * normalizer
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite analytical objective")
        row = {"evaluation": len(history) + 1, "mse_ms2": float(np.dot(value, value)),
               "speeds_xyz_mm_per_ms": speeds.tolist()}
        history.append(row)
        if checkpoint:
            checkpoint({"phase": "optimization", "fit": {"model": model,
                        "speeds_xyz_mm_per_ms": speeds.tolist(), "history": list(history),
                        "train_objective_mse_ms2": row["mse_ms2"]}})
        return value

    def jacobian(log_speeds):
        terms = squares / np.exp(2 * log_speeds)
        times = np.sqrt(terms.sum(axis=1))
        jac = np.zeros_like(terms)
        np.divide(-terms, times[:, None], out=jac, where=times[:, None] > 0)
        return jac * normalizer[:, None]

    if model == "isotropic":
        speeds = np.repeat(speed, 3)
        error = radius / speed - centered
        mse = float(np.dot(weights, error ** 2))
        gradient = float(-2. * np.dot(weights * error, radius / speed))
        attempts, nfev = [], 0
        convergence = {"converged": True, "reason": "closed_form_positive_slowness",
                       "gradient_log_speeds": [gradient], "gradient_inf_norm": abs(gradient),
                       "iteration_limit_reached": False, "evaluation_limit_reached": False}
    else:
        squared_slowness = np.linalg.lstsq(squares * normalizer[:, None],
                                          centered ** 2 * normalizer, rcond=None)[0]
        starts = [np.full(3, np.log(speed)),
                  -.5 * np.log(np.maximum(squared_slowness, (1 / speed) ** 2 * 1e-4))]
        fits = [least_squares(residual, start, jac=jacobian, xtol=1e-12, ftol=1e-12,
                              gtol=1e-10, max_nfev=int(settings.get("analytical_max_nfev", 500)))
                for start in starts]
        if any(not np.isfinite(f.x).all() or not np.isfinite(f.fun).all() for f in fits):
            raise FloatingPointError("Nonfinite analytical fitted parameters")
        best = min(fits, key=lambda f: f.cost)
        speeds, mse = np.exp(best.x), float(2 * best.cost)
        gradient = 2 * best.jac.T.dot(best.fun)
        convergence = {"converged": bool(best.success), "reason": str(best.message),
                       "scipy_status": int(best.status), "gradient_log_speeds": gradient.tolist(),
                       "gradient_inf_norm": float(np.max(np.abs(gradient))),
                       "iteration_limit_reached": False, "evaluation_limit_reached": best.status == 0}
        attempts = [{"initial_speeds_xyz_mm_per_ms": np.exp(start).tolist(),
                     "speeds_xyz_mm_per_ms": np.exp(f.x).tolist(), "mse_ms2": float(2 * f.cost),
                     "nfev": int(f.nfev), "success": bool(f.success), "message": str(f.message)}
                    for start, f in zip(starts, fits)]
        nfev = sum(f.nfev for f in fits)
    return {"model": model, "speeds_xyz_mm_per_ms": speeds.tolist(), "latency_ms": float(latency),
            "train_objective_mse_ms2": mse, "training_sample_mse_ms2": mse,
            "optimizer_success": convergence["converged"], "convergence": convergence,
            "attempts": attempts, "nfev": int(nfev), "history": history,
            "elapsed_seconds": perf_counter() - started}


def _numerical_fit(dataset, case_ids, indices, spec, config, analytical, checkpoint):
    started = perf_counter()
    settings = config.get("optimizer", {})
    model_name = spec["model"]
    initial = spec.get("initial_speeds_xyz_mm_per_ms")
    if initial is None:
        initial = np.asarray(analytical["speeds_xyz_mm_per_ms"]) * float(spec.get("initial_scale", .85))
    initial = _speeds(initial, model_name)
    device = torch.device("cuda" if config.get("backend") == "cuda" else "cpu")
    model = SharedSpeedModel(float(initial[0]) if model_name == "isotropic" else initial,
                             isotropic=model_name == "isotropic").to(device)
    max_iter, max_eval = int(settings.get("max_iter", 100)), int(settings.get("max_eval", 300))
    tolerance_grad = float(settings.get("tolerance_grad", 1e-7))
    tolerance_change = float(settings.get("tolerance_change", 1e-10))
    if max_iter < 1 or max_eval < 1 or tolerance_grad < 0 or tolerance_change < 0:
        raise ValueError("Invalid numerical optimizer settings")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=max_iter, max_eval=max_eval,
                                 tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                                 line_search_fn="strong_wolfe")
    factor = int(spec.get("refine_factor", 1))
    shape, spacing, _ = refined_geometry(dataset.shape_zyx, dataset.spacing_xyz, (0, 0, 0), factor)
    grouped = {}
    for case_id, index in zip(case_ids, indices):
        source = tuple(int(v) for v in dataset.sources_nzyx[case_id])
        target = torch.from_numpy(np.array(dataset.activation_nzyx[case_id].ravel()[index], copy=True)).to(device)
        mapped = torch.from_numpy(map_observation_indices(index, dataset.shape_zyx, factor)).to(device)
        grouped.setdefault(source, []).append((mapped, target))
    history = []

    def closure():
        optimizer.zero_grad()
        value = 0.
        for source, cases in grouped.items():
            propagation = model(shape, tuple(s * factor for s in source), spacing,
                                backend=config.get("backend", "numba"))
            if not torch.isfinite(propagation).all():
                raise FloatingPointError("Nonfinite numerical propagation")
            group_loss = sum(((propagation.flatten()[mapped] + config["latency_ms"] - target) ** 2).mean()
                             / len(case_ids) for mapped, target in cases)
            if not torch.isfinite(group_loss):
                raise FloatingPointError("Nonfinite numerical training loss")
            group_loss.backward()
            value += float(group_loss.detach())
        gradient = model.log_speeds.grad.detach().cpu().numpy().copy()
        speeds = _speeds(model.speeds_xyz.detach().cpu().numpy(), model_name)
        if not np.isfinite(gradient).all() or not np.isfinite(value):
            raise FloatingPointError("Nonfinite numerical objective or gradient")
        history.append({"evaluation": len(history) + 1, "mse_ms2": value,
                        "speeds_xyz_mm_per_ms": speeds.tolist(), "gradient_log_speeds": gradient.tolist()})
        if checkpoint:
            checkpoint({"phase": "optimization", "fit": {"model": model_name,
                        "speeds_xyz_mm_per_ms": speeds.tolist(), "history": list(history),
                        "train_objective_mse_ms2": value}})
        return torch.tensor(value, dtype=torch.float64)

    optimizer.step(closure)
    closure()  # Accepted final point; line-search trial points are not results.
    state = optimizer.state[model.log_speeds]
    iterations, evaluations = int(state.get("n_iter", 0)), int(state.get("func_evals", 0))
    gradient = np.asarray(history[-1]["gradient_log_speeds"])
    grad_norm = float(np.max(np.abs(gradient)))
    iteration_limit, evaluation_limit = iterations >= max_iter, evaluations >= max_eval
    step_norm, loss_change = None, None
    if "d" in state and "t" in state:
        step_norm = float((state["d"] * state["t"]).abs().max())
    if "prev_loss" in state:
        loss_change = abs(history[-1]["mse_ms2"] - float(state["prev_loss"]))
    if grad_norm <= tolerance_grad:
        reason, converged = "gradient_tolerance", True
    elif iteration_limit:
        reason, converged = "iteration_limit", False
    elif evaluation_limit:
        reason, converged = "evaluation_limit", False
    elif step_norm is not None and step_norm <= tolerance_change:
        reason, converged = "step_change_tolerance", True
    elif loss_change is not None and loss_change <= tolerance_change:
        reason, converged = "loss_change_tolerance", True
    else:
        reason, converged = "optimizer_stopped_without_confirmed_tolerance", False
    fitted = _speeds(model.speeds_xyz.detach().cpu().numpy(), model_name)
    return {"model": model_name, "speeds_xyz_mm_per_ms": fitted.tolist(),
            "initial_speeds_xyz_mm_per_ms": initial.tolist(), "latency_ms": float(config["latency_ms"]),
            "train_objective_mse_ms2": history[-1]["mse_ms2"], "final_mse_ms2": history[-1]["mse_ms2"],
            "initial_mse_ms2": history[0]["mse_ms2"], "history": history,
            "parameter_changed": not np.allclose(initial, fitted, rtol=1e-8, atol=1e-10),
            "loss_decreased": history[-1]["mse_ms2"] < history[0]["mse_ms2"],
            "finite_gradients": True, "optimizer_success": converged,
            "convergence": {"converged": converged, "reason": reason,
                            "gradient_log_speeds": gradient.tolist(), "gradient_inf_norm": grad_norm,
                            "iterations": iterations, "optimizer_function_evaluations": evaluations,
                            "total_closure_evaluations": len(history), "last_step_inf_norm": step_norm,
                            "last_accepted_loss_change_ms2": loss_change,
                            "iteration_limit_reached": iteration_limit,
                            "evaluation_limit_reached": evaluation_limit,
                            "note": "Step/loss tolerance denotes optimizer termination, not proof of a global optimum"},
            "training_unique_source_solves": len(history) * len(grouped),
            "elapsed_seconds": perf_counter() - started}


def _metric_rows(absolute, squared, masks, edges, names=None):
    rows = []
    for i, mask in enumerate(masks):
        count = int(mask.sum())
        row = {"min_mm_inclusive": float(edges[i]),
               "max_mm_exclusive": float(edges[i + 1]) if np.isfinite(edges[i + 1]) else None,
               "voxel_count": count, "mae_ms": float(absolute[mask].mean()) if count else None,
               "rmse_ms": float(np.sqrt(squared[mask].mean())) if count else None}
        if names:
            row["name"] = names[i]
        rows.append(row)
    return rows


def evaluate_grouped(dataset, case_ids, predict, latency_ms, exact_reference=False):
    """Full original-node errors, with repeated cases averaged within sources.

    ``exact_reference`` compares numerical propagation to exact homogeneous
    propagation supplied by the callback as (numerical, exact). Label-based
    model error and numerical error therefore never share a reference target.
    """
    started = perf_counter()
    if not len(case_ids):
        raise ValueError("Cannot evaluate an empty partition")
    groups = {}
    for case_id in case_ids:
        groups.setdefault(tuple(int(s) for s in dataset.sources_nzyx[case_id]), []).append(int(case_id))
    per_case, per_group = [], []
    pooled_abs = np.zeros(dataset.shape_zyx, dtype=np.float64)
    pooled_sq = np.zeros(dataset.shape_zyx, dtype=np.float64)
    bin_abs, bin_sq, bin_count = np.zeros(5), np.zeros(5), np.zeros(5, dtype=np.int64)
    reg_abs, reg_sq, reg_count = np.zeros(3), np.zeros(3), np.zeros(3, dtype=np.int64)
    max_abs = 0.
    for source, members in groups.items():
        output = predict(source)
        if exact_reference:
            propagation, exact = (np.asarray(v, dtype=np.float64) for v in output)
            if exact.shape != dataset.shape_zyx or not np.isfinite(exact).all():
                raise ValueError("Exact reference must be a finite original-shape volume")
        else:
            propagation = np.asarray(output, dtype=np.float64)
        if propagation.shape != dataset.shape_zyx or not np.isfinite(propagation).all():
            raise ValueError("Prediction must be a finite original-shape ZYX volume")
        if abs(propagation[source]) > 1e-9:
            raise ValueError("Propagation source time must be zero")
        absolute, squared = np.zeros_like(propagation), np.zeros_like(propagation)
        for case_id in members:
            reference = exact if exact_reference else dataset.activation_nzyx[case_id]
            error = propagation - reference if exact_reference else propagation + latency_ms - reference
            ae, se = np.abs(error), error ** 2
            absolute += ae
            squared += se
            max_abs = max(max_abs, float(ae.max()))
            norm = float(np.linalg.norm(reference))
            per_case.append({"case_index": case_id, "source_zyx": list(source),
                             "mae_ms": float(ae.mean()), "rmse_ms": float(np.sqrt(se.mean())),
                             "relative_l2": float(np.linalg.norm(error) / norm) if norm else None})
        pooled_abs += absolute
        pooled_sq += squared
        absolute /= len(members)
        squared /= len(members)
        distance = analytical_arrival(dataset.shape_zyx, source, 1., dataset.spacing_xyz)
        masks = [(distance >= lo) & (distance < hi) for lo, hi in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:])]
        regions = [(distance >= lo) & (distance < hi) for lo, hi in zip(REGION_EDGES_MM[:-1], REGION_EDGES_MM[1:])]
        distance_rows = _metric_rows(absolute, squared, masks, DISTANCE_EDGES_MM)
        region_rows = _metric_rows(absolute, squared, regions, REGION_EDGES_MM, ("near", "mid", "far"))
        for b, mask in enumerate(masks):
            bin_abs[b] += float(absolute[mask].sum()) * len(members)
            bin_sq[b] += float(squared[mask].sum()) * len(members)
            bin_count[b] += int(mask.sum()) * len(members)
        for b, mask in enumerate(regions):
            reg_abs[b] += float(absolute[mask].sum()) * len(members)
            reg_sq[b] += float(squared[mask].sum()) * len(members)
            reg_count[b] += int(mask.sum()) * len(members)
        per_group.append({"source_zyx": list(source), "source_z_index": source[0],
                          "source_z_mm": source[0] * dataset.spacing_xyz[2],
                          "case_indices": members, "case_count": len(members),
                          "mae_ms": float(absolute.mean()), "rmse_ms": float(np.sqrt(squared.mean())),
                          "voxel_depth": [{"z_index": z, "z_mm": z * dataset.spacing_xyz[2],
                                           "mae_ms": float(absolute[z].mean()),
                                           "rmse_ms": float(np.sqrt(squared[z].mean()))}
                                          for z in range(dataset.shape_zyx[0])],
                          "source_distance": distance_rows, "regions": region_rows})
    count = len(case_ids) * int(np.prod(dataset.shape_zyx))

    def pooled_bins(edges, abs_sum, sq_sum, counts, names=None):
        return [{"min_mm_inclusive": float(edges[b]),
                 "max_mm_exclusive": float(edges[b + 1]) if np.isfinite(edges[b + 1]) else None,
                 "voxel_count": int(counts[b]),
                 "mae_ms": float(abs_sum[b] / counts[b]) if counts[b] else None,
                 "rmse_ms": float(np.sqrt(sq_sum[b] / counts[b])) if counts[b] else None,
                 **({"name": names[b]} if names else {})} for b in range(len(counts))]

    return {"case_count": len(case_ids), "source_group_count": len(groups), "voxel_count": count,
            "mae_ms": float(pooled_abs.sum() / count), "rmse_ms": float(np.sqrt(pooled_sq.sum() / count)),
            "max_abs_ms": max_abs, "latency_ms": float(latency_ms),
            "reference": "exact homogeneous eikonal at the same fitted speeds" if exact_reference else "original monodomain activation labels",
            "source_group_mean_mae_ms": float(np.mean([r["mae_ms"] for r in per_group])),
            "source_group_mean_rmse_ms": float(np.mean([r["rmse_ms"] for r in per_group])),
            "group_weighting": "equal case weights within source; equal source groups for paired inference",
            "per_case": sorted(per_case, key=lambda r: list(case_ids).index(r["case_index"])),
            "per_case_mae_distribution_ms": dict(zip(("min", "p25", "median", "p75", "p90", "max"),
                                                      np.percentile([r["mae_ms"] for r in per_case], [0, 25, 50, 75, 90, 100]).tolist())),
            "per_source_group": per_group,
            "voxel_depth": [{"z_index": z, "z_mm": z * dataset.spacing_xyz[2],
                             "mae_ms": float(pooled_abs[z].mean() / len(case_ids)),
                             "rmse_ms": float(np.sqrt(pooled_sq[z].mean() / len(case_ids)))}
                            for z in range(dataset.shape_zyx[0])],
            "source_depth": [{"source_z_index": z, "source_z_mm": z * dataset.spacing_xyz[2],
                              "source_group_count": sum(r["source_z_index"] == z for r in per_group),
                              "mae_ms": float(np.mean([r["mae_ms"] for r in per_group if r["source_z_index"] == z])),
                              "rmse_ms": float(np.sqrt(np.mean([r["rmse_ms"] ** 2 for r in per_group if r["source_z_index"] == z])))}
                             for z in sorted({r["source_z_index"] for r in per_group})],
            "source_distance": pooled_bins(DISTANCE_EDGES_MM, bin_abs, bin_sq, bin_count),
            "regions": pooled_bins(REGION_EDGES_MM, reg_abs, reg_sq, reg_count, ("near", "mid", "far")),
            "elapsed_seconds": perf_counter() - started}


def fit_run(dataset, spec, config, checkpoint=None, reused_fit=None):
    """Fit one immutable run specification and evaluate full original volumes.

    ``checkpoint`` receives JSON-ready progress after objective evaluations and
    a complete accepted fit (phase ``fitted``) before held-out evaluation starts.
    Passing a complete ``reused_fit`` skips optimization, including after an
    evaluation-stage interruption. Trial-point checkpoints are not final fits.
    """
    started = perf_counter()
    method, model = spec["method"], spec["model"]
    if method not in ("analytical", "numerical") or model not in ("isotropic", "diagonal"):
        raise ValueError("Unknown fit method or model")
    latency = float(config["latency_ms"])
    if not np.isfinite(latency):
        raise ValueError("Fixed latency must be finite")
    case_ids = [int(v) for v in spec["training_case_indices"]]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Training requires nonempty unique case indices")
    if min(case_ids) < 0 or max(case_ids) >= len(dataset.sources_nzyx):
        raise ValueError("Training case index out of bounds")
    factor = spec.get("refine_factor", 1)
    shape, spacing, _ = refined_geometry(dataset.shape_zyx, dataset.spacing_xyz, (0, 0, 0), factor)
    solver = dict(SOLVER_DEFAULTS)
    solver.update({k: v for k, v in config.get("solver", {}).items() if k in SOLVER_DEFAULTS})
    if method == "numerical" and solver != SOLVER_DEFAULTS:
        raise ValueError("SharedSpeedModel uses validated fixed solver tolerances; suite solver settings must match defaults")
    indices = observation_indices(dataset, case_ids, int(spec.get("samples_per_case", 0)),
                                  spec.get("sampling_seed", 20260905), float(spec.get("exclude_distance_mm", 0)))
    observations = _observation_metadata(case_ids, indices, spec)
    settings = config.get("optimizer", {})
    analytical = None
    if reused_fit is not None:
        fit = copy.deepcopy(reused_fit)
        if "fit" in fit and "speeds_xyz_mm_per_ms" not in fit:
            fit = fit["fit"]
        _speeds(fit["speeds_xyz_mm_per_ms"], model)
        if "latency_ms" in fit and not np.isclose(fit["latency_ms"], latency, rtol=0, atol=1e-12):
            raise ValueError("Reused fit latency disagrees with the fixed suite latency")
        fit["reused"] = True
        fit.setdefault("convergence", {"converged": None, "reason": "reused_legacy_fit; no independently established convergence status"})
    elif method == "analytical":
        fit = _analytical_fit(dataset, case_ids, indices, model, latency, settings, checkpoint)
    else:
        analytical = _analytical_fit(dataset, case_ids, indices, model, latency, settings)
        fit = _numerical_fit(dataset, case_ids, indices, spec, config, analytical, checkpoint)
    speeds = _speeds(fit["speeds_xyz_mm_per_ms"], model)
    exact_cache, numerical_cache, diagnostics = {}, {}, []

    def exact(source):
        if source not in exact_cache:
            exact_cache[source] = analytical_arrival(dataset.shape_zyx, source, speeds, dataset.spacing_xyz)
        return exact_cache[source]

    def numerical(source):
        if source not in numerical_cache:
            solved = solve_diagonal(shape, tuple(s * factor for s in source), speeds, spacing,
                                    backend=config.get("backend", "numba"), **solver)
            numerical_cache[source] = solved.arrival[::factor, ::factor, ::factor].copy()
            diagnostics.append({"source_zyx": list(source), "refined_source_zyx": [s * factor for s in source],
                                "backend": solved.backend, "cycles": int(solved.cycles),
                                "max_change": solved.max_change, "max_residual": solved.max_residual,
                                "max_fixed_point_error": solved.max_fixed_point_error,
                                "elapsed_seconds": solved.elapsed_seconds})
        return numerical_cache[source]

    predict = exact if method == "analytical" else numerical
    # Recompute the precise selected-observation objective for legacy fits. A
    # completed checkpoint already has the objective at its accepted parameters.
    if reused_fit is not None and "train_objective_mse_ms2" not in fit:
        fit["train_objective_mse_ms2"] = float(np.mean([
            np.mean((predict(tuple(int(v) for v in dataset.sources_nzyx[case_id])).ravel()[index]
                     + latency - dataset.activation_nzyx[case_id].ravel()[index]) ** 2)
            for case_id, index in zip(case_ids, indices)]))
    if not np.isfinite(fit["train_objective_mse_ms2"]):
        raise FloatingPointError("Nonfinite final training objective")
    fit.update({"model": model, "method": method, "latency_ms": latency,
                "speeds_xyz_mm_per_ms": speeds.tolist(), "training_case_indices": case_ids,
                "observations": observations, "refine_factor": factor,
                "solver_shape_zyx": list(shape), "solver_spacing_xyz_mm": list(spacing)})
    if analytical is not None:
        optimum = np.asarray(analytical["speeds_xyz_mm_per_ms"])
        fit["analytical_optimum_same_training_loss"] = {"speeds_xyz_mm_per_ms": optimum.tolist(),
                                                       "train_objective_mse_ms2": analytical["train_objective_mse_ms2"],
                                                       "convergence": analytical["convergence"]}
        fit["distance_from_analytical_optimum"] = {"delta_xyz_mm_per_ms": (speeds - optimum).tolist(),
                                                   "l2_mm_per_ms": float(np.linalg.norm(speeds - optimum)),
                                                   "relative_l2": float(np.linalg.norm(speeds - optimum) / np.linalg.norm(optimum))}
    if checkpoint:
        checkpoint({"phase": "fitted", "fit": fit})
    evaluations, discretization, exact_evaluations = {}, {}, {}
    for partition in ("validation", "test"):
        heldout = [int(v) for v in spec["evaluation_case_indices"][partition]]
        evaluations[partition] = evaluate_grouped(dataset, heldout, predict, latency)
        if method == "numerical":
            exact_evaluations[partition] = evaluate_grouped(dataset, heldout, exact, latency)
            discretization[partition] = evaluate_grouped(dataset, heldout,
                                                         lambda source: (numerical(source), exact(source)),
                                                         0., exact_reference=True)
        if checkpoint:
            checkpoint({"phase": "evaluating", "completed_partition": partition, "fit": fit,
                        "evaluation": evaluations, "discretization": discretization,
                        "exact_at_fitted_speeds": exact_evaluations})
    result = {"fit": fit, "evaluation": evaluations,
              "solver_diagnostics": {"settings": solver, "scope": "final-parameter evaluation solves (includes reused-fit train objective when required)",
                                     "optimization_verification": "every differentiable forward requires convergence and causal branch verification",
                                     "unique_source_solves": len(diagnostics),
                                     "max_residual": max((r["max_residual"] for r in diagnostics), default=None),
                                     "max_fixed_point_error": max((r["max_fixed_point_error"] for r in diagnostics), default=None),
                                     "max_cycles": max((r["cycles"] for r in diagnostics), default=None),
                                     "total_solve_seconds": sum(r["elapsed_seconds"] for r in diagnostics),
                                     "per_source": diagnostics},
              "elapsed_seconds": perf_counter() - started}
    if method == "numerical":
        result.update({"discretization": discretization, "exact_at_fitted_speeds": exact_evaluations})
    return result
