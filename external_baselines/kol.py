"""Independent implementation of the timeactML centroid-IQ kernel equations.

Only training and validation arrays enter fitting. Predictions are absolute
activation times in milliseconds; no per-case target alignment is available.
The upstream audit snapshot is evidence, not an imported code dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


UPSTREAM_COMMIT = "b765384703c691e92918fe449bcf809b9e82564d"
OFFICIAL_PRESET = {"name": "official_iq4", "eta": 4.0, "nugget": 1e-7}
DEFAULT_CONFIG = {
    "eta_values": [2, 3, 4, 5, 6],
    "ridge_ratios": [0.0, 1e-8, 1e-4, 1e-2],
    "include_official": True,
    "selection_metric": "validation_full_volume_case_mean_mae_ms",
    "jitter_ratios": [0.0, 1e-12, 1e-10, 1e-8, 1e-6],
}
TARGET_SCALE = 0.002
TARGET_OFFSET = -0.1
SCHEMA_VERSION = 1


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(array):
    array = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sources(value, name, *, allow_empty=False):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (case_count, 3) in grid-index ZYX")
    if (not allow_empty and not len(value)) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite source coordinates")
    return value


def iq_kernel(sources_zyx, centers_zyx, eta=4.0):
    """Return 1/sqrt(1e-6*d + 10**(-eta)), using d, not squared d.

    Inputs are grid indices, not physical mm or unit-interval coordinates.
    A consistent XYZ/ZYX permutation leaves this Euclidean kernel unchanged.
    """
    sources = _sources(sources_zyx, "sources", allow_empty=True)
    centers = _sources(centers_zyx, "centers")
    eta = float(eta)
    if not np.isfinite(eta) or not -12 <= eta <= 12:
        raise ValueError("eta must be finite and within [-12,12]")
    distance = np.linalg.norm(sources[:, None, :] - centers[None, :, :], axis=2)
    return 1.0 / np.sqrt(1e-6 * distance + 10.0 ** (-eta))


@dataclass
class KOLPredictor:
    centers_zyx: np.ndarray
    coefficients: np.ndarray
    shape_zyx: tuple[int, int, int]
    eta: float

    def predict(self, sources_zyx):
        sources = _sources(sources_zyx, "prediction sources", allow_empty=True)
        encoded = iq_kernel(sources, self.centers_zyx, self.eta) @ self.coefficients
        result = (encoded - TARGET_OFFSET) / TARGET_SCALE
        return result.reshape((len(sources),) + self.shape_zyx)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, schema_version=np.int64(SCHEMA_VERSION),
                centers_zyx=self.centers_zyx, coefficients=self.coefficients,
                shape_zyx=np.asarray(self.shape_zyx, dtype=np.int64), eta=float(self.eta),
                target_scale=TARGET_SCALE, target_offset=TARGET_OFFSET,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def load_kol(path):
    """Restore a saved selected or candidate model without training."""
    with np.load(path, allow_pickle=False) as saved:
        if (int(saved["schema_version"]) != SCHEMA_VERSION or
                float(saved["target_scale"]) != TARGET_SCALE or
                float(saved["target_offset"]) != TARGET_OFFSET):
            raise ValueError("Unrecognized KOL model version or timing transform")
        centers = _sources(saved["centers_zyx"], "saved centers").copy()
        shape = tuple(int(n) for n in saved["shape_zyx"])
        coefficients = np.array(saved["coefficients"], dtype=np.float64)
        eta = float(saved["eta"])
    if (len(shape) != 3 or min(shape) <= 0 or
            coefficients.shape != (len(centers), int(np.prod(shape))) or
            not np.isfinite(coefficients).all() or not np.isfinite(eta)):
        raise ValueError("Invalid saved KOL model arrays")
    return KOLPredictor(centers, coefficients, shape, eta)


def _metrics(prediction, target):
    difference = (prediction - target).reshape(len(target), -1)
    if not np.isfinite(difference).all():
        raise FloatingPointError("Nonfinite predicted activation or activation error")
    norms = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    errors = np.linalg.norm(difference, axis=1)
    relative = np.divide(errors, norms, out=np.zeros_like(errors), where=norms > 0)
    relative_value = None if np.any((norms == 0) & (errors > 0)) else float(relative.mean())
    return {
        "mae_ms": float(np.mean(np.abs(difference))),
        "rmse_ms": float(np.sqrt(np.mean(difference ** 2))),
        "mse_ms2": float(np.mean(difference ** 2)),
        "relative_l2": relative_value,
        "per_case_mae_ms": np.mean(np.abs(difference), axis=1).tolist(),
        "per_case_rmse_ms": np.sqrt(np.mean(difference ** 2, axis=1)).tolist(),
        "weighting": "equal case weight, all voxels per case",
    }


def candidate_configs(config=None):
    """Build the predeclared search grid, without inspecting any labels."""
    settings = {**DEFAULT_CONFIG, **(config or {})}
    if settings["selection_metric"] != DEFAULT_CONFIG["selection_metric"]:
        raise ValueError("KOL selection metric must be fixed full-volume validation MAE")
    candidates = []
    if settings.get("include_official", True):
        candidates.append(dict(OFFICIAL_PRESET))
    if "candidates" in settings:
        proposed = settings["candidates"]
    else:
        proposed = [
            {"eta": eta, "nugget": float(ridge) * 10.0 ** (float(eta) / 2),
             "ridge_ratio": float(ridge)}
            for eta in settings["eta_values"] for ridge in settings["ridge_ratios"]
        ]
    for item in proposed:
        eta, nugget = float(item["eta"]), float(item["nugget"])
        if not np.isfinite(eta) or not -12 <= eta <= 12:
            raise ValueError("Candidate eta must be within [-12,12]")
        if not np.isfinite(nugget) or nugget < 0:
            raise ValueError("Candidate nugget must be finite and nonnegative")
        if any(eta == c["eta"] and nugget == c["nugget"] for c in candidates):
            continue
        candidates.append({"name": f"candidate_{len(candidates):02d}",
                           "eta": eta, "nugget": nugget,
                           "ridge_ratio": nugget / 10.0 ** (eta / 2)})
    if not candidates:
        raise ValueError("KOL requires at least one candidate")
    return candidates


def _factor_fit(train_sources, train_targets, candidate, jitter_ratios):
    # Collapsing duplicates with nugget/count is exactly the repeated-case
    # kernel ridge system; using the unchanged nugget would alter its weight.
    centers, inverse, counts = np.unique(train_sources, axis=0,
                                          return_inverse=True, return_counts=True)
    encoded = TARGET_SCALE * train_targets.reshape(len(train_sources), -1) + TARGET_OFFSET
    group_targets = np.zeros((len(centers), encoded.shape[1]), dtype=np.float64)
    np.add.at(group_targets, inverse, encoded)
    group_targets /= counts[:, None]
    kernel = iq_kernel(centers, centers, candidate["eta"])
    diagonal = float(kernel[0, 0])
    failures = []
    for ratio in jitter_ratios:
        jitter = float(ratio) * diagonal
        system = kernel + np.diag((candidate["nugget"] + jitter) / counts)
        try:
            factor = np.linalg.cholesky(system)
            coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, group_targets))
            if not np.isfinite(coefficients).all():
                raise FloatingPointError("Nonfinite kernel coefficients")
            residual = float(np.linalg.norm(system @ coefficients - group_targets) /
                             max(np.linalg.norm(group_targets), np.finfo(float).tiny))
            predictor = KOLPredictor(centers, coefficients, tuple(train_targets.shape[1:]),
                                     float(candidate["eta"]))
            diagnostics = {
                "status": "solved", "algorithm": "float64 Cholesky kernel ridge solve",
                "iterations": 1, "factorization_attempts": len(failures) + 1,
                "failed_factorizations": failures, "jitter": jitter,
                "effective_nugget_before_group_weighting": float(candidate["nugget"] + jitter),
                "unique_training_sources": len(centers), "group_case_counts": counts.tolist(),
                "system_condition_number": float(np.linalg.cond(system)),
                "relative_linear_system_residual": residual,
            }
            return predictor, diagnostics
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            failures.append({"jitter": jitter, "error": str(error)})
    raise np.linalg.LinAlgError(f"All predetermined factorization attempts failed: {failures}")


def fit_kol(train_sources_zyx, train_targets_zyx, val_sources_zyx, val_targets_zyx,
            *, config, seed, artifact_dir, smoke=False):
    """Fit and validation-select KOL, persisting every candidate immediately.

    Repeated sources stay grouped. ``seed`` is recorded for a common runner
    interface; KOL has no random operations. Smoke uses the same kernel grid on
    the small subset chosen by the runner, so orchestration receives real fits.
    """
    started = perf_counter()
    train_sources = _sources(train_sources_zyx, "training sources")
    val_sources = _sources(val_sources_zyx, "validation sources")
    train_targets = np.asarray(train_targets_zyx, dtype=np.float64)
    val_targets = np.asarray(val_targets_zyx, dtype=np.float64)
    if (train_targets.ndim != 4 or val_targets.ndim != 4 or
            len(train_sources) != len(train_targets) or len(val_sources) != len(val_targets) or
            train_targets.shape[1:] != val_targets.shape[1:] or min(train_targets.shape[1:]) < 1):
        raise ValueError("Training/validation targets must be NZYX on the same full-volume grid")
    if not np.isfinite(train_targets).all() or not np.isfinite(val_targets).all():
        raise ValueError("Training and validation targets must be finite")
    if {tuple(x) for x in train_sources} & {tuple(x) for x in val_sources}:
        raise ValueError("Training and validation source groups must be disjoint")
    for sources in (train_sources, val_sources):
        if np.any(sources < 0) or np.any(sources > np.asarray(train_targets.shape[1:]) - 1):
            raise ValueError("Source grid indices must be inside the target physical grid")
    settings = {**DEFAULT_CONFIG, **(config or {})}
    candidates = candidate_configs(settings)
    jitter_ratios = [float(x) for x in settings["jitter_ratios"]]
    if (not jitter_ratios or jitter_ratios[0] != 0 or
            any(not np.isfinite(x) or x < 0 for x in jitter_ratios) or
            jitter_ratios != sorted(set(jitter_ratios))):
        raise ValueError("Jitter ratios must start at zero and increase deterministically")
    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION, "implementation_sha256": _file_hash(__file__),
        "train_sources": _array_hash(train_sources), "train_targets": _array_hash(train_targets),
        "validation_sources": _array_hash(val_sources), "validation_targets": _array_hash(val_targets),
        "candidates": candidates, "jitter_ratios": jitter_ratios, "seed": int(seed),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    _write_json(artifact_dir / "kol_fit_configuration.json", {
        **fingerprint_payload, "fingerprint": fingerprint, "smoke": bool(smoke),
        "training_sources_zyx": train_sources.tolist(),
        "validation_sources_zyx": val_sources.tolist(),
        "timing_convention": "absolute activation ms; fixed encode .002*T-.1; no alignment",
    })
    candidate_results = []
    for order, candidate in enumerate(candidates):
        candidate_started = perf_counter()
        record_path = artifact_dir / "candidates" / f"{order:02d}_{candidate['name']}.json"
        model_path = record_path.with_suffix(".npz")
        record = None
        if record_path.is_file():
            try:
                saved = json.loads(record_path.read_text(encoding="utf-8"))
                if (saved.get("status") == "completed" and saved.get("fingerprint") == fingerprint and
                        saved.get("model_sha256") == _file_hash(model_path)):
                    load_kol(model_path)
                    record = saved
            except (OSError, ValueError, KeyError):
                pass
        if record is None:
            record = {"order": order, "config": candidate, "fingerprint": fingerprint}
            try:
                fit_started = perf_counter()
                predictor, diagnostics = _factor_fit(train_sources, train_targets, candidate, jitter_ratios)
                fit_seconds = perf_counter() - fit_started
                train_metrics = _metrics(predictor.predict(train_sources), train_targets)
                validation_metrics = _metrics(predictor.predict(val_sources), val_targets)
                predictor.save(model_path)
                record.update({"status": "completed", "fit_seconds": fit_seconds,
                               "elapsed_seconds": perf_counter() - candidate_started,
                               "train_metrics": train_metrics, "validation_metrics": validation_metrics,
                               "convergence": diagnostics, "model_path": str(model_path),
                               "model_sha256": _file_hash(model_path)})
            except Exception as error:
                record.update({"status": "failed", "error_type": type(error).__name__,
                               "error": str(error), "elapsed_seconds": perf_counter() - candidate_started})
                failure_dir = artifact_dir / "failures"
                failure_dir.mkdir(exist_ok=True)
                attempt = len(list(failure_dir.glob(f"candidate_{order:02d}_attempt_*.json"))) + 1
                _write_json(failure_dir / f"candidate_{order:02d}_attempt_{attempt:03d}.json", record)
            _write_json(record_path, record)
        candidate_results.append(record)
        _write_json(artifact_dir / "kol_candidates.json", candidate_results)
    complete = [x for x in candidate_results if x["status"] == "completed"]
    if not complete:
        raise RuntimeError("Every KOL candidate failed; diagnostics saved in kol_candidates.json")
    selected = min(complete, key=lambda x: (x["validation_metrics"]["mae_ms"], x["order"]))
    predictor = load_kol(selected["model_path"])
    selected_model_path = artifact_dir / "kol_model.npz"
    predictor.save(selected_model_path)
    official = next((x for x in candidate_results if x["config"]["name"] == "official_iq4"), None)
    metadata = {
        "method": "timeactml_kol", "method_label": "KOL (validation-selected centroid IQ)",
        "training_seconds": perf_counter() - started,
        "candidate_fit_seconds_total": float(sum(x.get("fit_seconds", 0) for x in complete)),
        "parameter_count": int(predictor.coefficients.size),
        "model_size_note": "dual coefficients, not a neural parameter count; includes one weight per training group and voxel",
        "model_bytes": int(predictor.coefficients.nbytes + predictor.centers_zyx.nbytes),
        "on_disk_bytes": selected_model_path.stat().st_size,
        "selected_config": selected["config"], "config": settings,
        "selection_metric": settings["selection_metric"], "selection_tie_break": "predetermined candidate order",
        "train_objective": selected["train_metrics"]["mse_ms2"],
        "train_objective_units": "ms^2 (full-volume, equal case weight)",
        "train_metrics": selected["train_metrics"], "validation_metrics": selected["validation_metrics"],
        "convergence": selected["convergence"], "candidate_results": candidate_results,
        "official_preset_metrics": official, "seed": int(seed), "stochastic": False,
        "timing_convention": "absolute activation ms; fixed .002*T-.1 transform; no target alignment",
        "upstream_commit": UPSTREAM_COMMIT, "implementation": "independent equation implementation",
        "model_path": str(selected_model_path), "model_sha256": _file_hash(selected_model_path),
        "artifact_paths": {"model": str(selected_model_path),
                           "candidate_log": str(artifact_dir / "kol_candidates.json"),
                           "configuration": str(artifact_dir / "kol_fit_configuration.json")},
        "fingerprint": fingerprint,
    }
    _write_json(artifact_dir / "kol_metadata.json", metadata)
    return predictor, metadata
