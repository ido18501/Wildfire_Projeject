"""Frozen-subset import, persistence, metrics, and integrity for external baselines."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LATENCY_MS = 1.0529606913652159
UPSTREAM_COMMIT = "b765384703c691e92918fe449bcf809b9e82564d"
MAT_SHA256 = "4c8e93028c6ccaac35912e53e67377a245493a12b88a0d4e5f7d123c0dda616d"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ensure_isolated(path, main_repository):
    path, main = Path(path).resolve(), Path(main_repository).resolve()
    try:
        path.relative_to(main)
    except ValueError:
        return path
    raise ValueError("Writes inside the running main repository are prohibited: " + str(path))


def environment():
    versions = {}
    for package in ("numpy", "scipy", "torch", "numba", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "libraries": versions}


def source_hashes():
    files = list((ROOT / "external_baselines").glob("*.py"))
    files += list((ROOT / "eikonal/three_d").glob("*.py"))
    files += [ROOT / "experiments/cardiac/timeactml.py", ROOT / "run_baselines.py"]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}


def _old_result(document, spec, semantic_manifest):
    if document.get("status") != "complete" or document.get("schema_version") != 1 or document.get("spec") != spec:
        raise ValueError("Missing, incomplete, or mismatched eikonal reference")
    if document.get("manifest_sha256") != semantic_manifest:
        raise ValueError("Foreign eikonal result manifest")
    if document.get("content_sha256") != digest({k: v for k, v in document.items() if k != "content_sha256"}):
        raise ValueError("Altered eikonal result content")
    speeds = np.asarray(document["fit"]["speeds_xyz_mm_per_ms"])
    if speeds.shape != (3,) or not np.isfinite(speeds).all() or (speeds <= 0).any():
        raise ValueError("Invalid eikonal speeds")
    for split in ("validation", "test"):
        metrics = document["evaluation"][split]
        if [row["case_index"] for row in metrics["per_case"]] != spec["evaluation_case_indices"][split]:
            raise ValueError("Eikonal held-out identities differ from the frozen specification")
        if not all(np.isfinite(metrics[k]) and metrics[k] >= 0 for k in ("mae_ms", "rmse_ms")):
            raise ValueError("Invalid eikonal metrics")


def inspect_eikonal_suite(suite):
    """Read immutable completed few-shot references; never generate subsets."""
    suite = Path(suite).resolve()
    raw = (suite / "EXPERIMENT_MANIFEST.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["config"]["smoke"]:
        raise ValueError("A smoke eikonal suite cannot define scientific comparison subsets")
    if manifest["data"]["sha256"] != MAT_SHA256:
        raise ValueError("Eikonal suite uses a different MAT")
    if manifest["shape_zyx"] != [9, 49, 49] or manifest["spacing_xyz_mm"] != [.8, .8, .8]:
        raise ValueError("Noncanonical eikonal grid")
    if abs(manifest["canonical_latency_ms"] - LATENCY_MS) > 1e-12:
        raise ValueError("Noncanonical eikonal latency")
    if any((len(manifest["split"][name]), len(manifest["source_groups"][name])) != expected
           for name, expected in {"train": (16, 16), "validation": (18, 16), "test": (102, 100)}.items()):
        raise ValueError("Noncanonical frozen partition counts")
    subsets, references = {}, []
    semantic = digest(manifest)
    for spec in manifest["runs"]:
        if spec["experiment"] != "fewshot":
            continue
        key = (spec["train_size"], spec["subset_index"])
        subset = {k: spec[k] for k in ("train_size", "subset_index", "subset_seed",
                  "training_case_indices", "training_source_groups", "evaluation_case_indices")}
        if key in subsets and subsets[key] != subset:
            raise ValueError("Models in the eikonal suite use different few-shot subsets")
        subsets[key] = subset
        if spec["method"] != "numerical":
            continue
        if spec["refine_factor"] != 1 or spec["samples_per_case"] != 0 or spec["exclude_distance_mm"] != 0:
            raise ValueError("Few-shot reference is not full-volume native-grid fitting")
        path = suite / (spec["id"] + ".json")
        content = path.read_bytes()
        document = json.loads(content)
        _old_result(document, spec, semantic)
        references.append({"method": "eikonal_" + spec["model"], **subset,
                           "source_path": str(path), "path": "references/" + spec["id"] + ".json",
                           "sha256": hashlib.sha256(content).hexdigest(), "bytes": content})
    expected = {2: 8, 4: 8, 8: 8, 16: 1}
    if {n: sum(k[0] == n for k in subsets) for n in expected} != expected or len(subsets) != 25:
        raise ValueError("Require the exact 25 predetermined eikonal subsets")
    if len(references) != 50 or len({(r["method"], r["train_size"], r["subset_index"]) for r in references}) != 50:
        raise ValueError("Require all 50 complete numerical isotropic/diagonal reference results")
    train_ids = manifest["split"]["train"]
    full_groups = manifest["source_groups"]["train"]
    for subset in subsets.values():
        ids = subset["training_case_indices"]
        if len(ids) != len(set(ids)) or not set(ids) <= set(train_ids):
            raise ValueError("Subset contains a held-out or repeated training case")
        for group in full_groups:
            overlap = set(group["case_indices"]) & set(ids)
            if overlap and overlap != set(group["case_indices"]):
                raise ValueError("Subset breaks a frozen source group")
        expected_groups = [g for g in full_groups if set(g["case_indices"]) <= set(ids)]
        if subset["training_source_groups"] != expected_groups or len(expected_groups) != subset["train_size"]:
            raise ValueError("Subset source definitions do not match frozen training groups")
        if any(subset["evaluation_case_indices"][p] != manifest["split"][p] for p in ("validation", "test")):
            raise ValueError("Subset does not use complete frozen held-out partitions")
    return manifest, raw, [subsets[k] for k in sorted(subsets)], references


def load_dataset(manifest):
    from experiments.cardiac.timeactml import load_timeactml, prepare_split, training_latency
    data = manifest["data"]
    if file_hash(data["path"]) != data["sha256"]:
        raise ValueError("MAT checksum changed")
    for part, record in manifest["frozen_manifests"].items():
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Frozen " + part + " manifest checksum changed")
    dataset = load_timeactml(data["path"])
    split = prepare_split(dataset, manifest["frozen_manifests"]["train"]["path"])
    if any(split[p] != manifest["split"][p] for p in ("train", "validation", "test")):
        raise ValueError("Frozen case memberships changed")
    if abs(training_latency(dataset, split["train"]) - LATENCY_MS) > 1e-12:
        raise ValueError("Canonical training-only latency differs")
    for part in ("train", "validation", "test"):
        for group in manifest["source_groups"][part]:
            if any(dataset.sources_nzyx[i].tolist() != group["source_zyx"] for i in group["case_indices"]):
                raise ValueError("Frozen source coordinate mismatch")
    return dataset


def common_case(dataset, test_ids):
    """Choose only by source geometry, before seeing any model prediction."""
    center = (np.asarray(dataset.shape_zyx) - 1) / 2.
    spacing = np.asarray(dataset.spacing_xyz)[::-1]
    return int(min(test_ids, key=lambda i: (float(np.linalg.norm((dataset.sources_nzyx[i] - center) * spacing)),
                                          tuple(dataset.sources_nzyx[i]), int(i))))


def evaluate_arrays(targets, predictions, sources, case_ids, inference_seconds):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 4 or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite absolute activation NZYX matching full targets")
    difference = predictions.astype(np.float64) - targets
    absolute = np.abs(difference)
    mse = (difference ** 2).reshape(len(targets), -1).mean(axis=1)
    maes = absolute.reshape(len(targets), -1).mean(axis=1)
    norms = np.linalg.norm(targets.reshape(len(targets), -1), axis=1)
    errors = np.linalg.norm(difference.reshape(len(targets), -1), axis=1)
    rows, grouped = [], {}
    for j, index in enumerate(case_ids):
        source = [int(s) for s in sources[j]]
        row = {"case_index": int(index), "source_zyx": source, "mae_ms": float(maes[j]),
               "rmse_ms": float(np.sqrt(mse[j])), "relative_l2": float(errors[j] / norms[j]) if norms[j] else None}
        rows.append(row)
        grouped.setdefault(tuple(source), []).append(j)
    groups = [{"source_zyx": list(source), "case_indices": [int(case_ids[j]) for j in members],
               "mae_ms": float(maes[members].mean()), "rmse_ms": float(np.sqrt(mse[members].mean()))}
              for source, members in sorted(grouped.items())]
    return {"case_count": len(case_ids), "source_group_count": len(groups),
            "voxel_count": int(targets.size), "mae_ms": float(maes.mean()),
            "rmse_ms": float(np.sqrt(mse.mean())), "per_case": rows, "per_source_group": groups,
            "mean_case_relative_l2": float(np.mean([r["relative_l2"] for r in rows if r["relative_l2"] is not None])),
            "inference_seconds": float(inference_seconds), "inference_seconds_per_case": float(inference_seconds / len(case_ids)),
            "timing_convention": "Absolute activation in ms; no target-dependent alignment",
            "reference": "Original monodomain activation labels"}


def valid_result(output, spec, manifest):
    output = Path(output).resolve()
    path = output / "results" / (spec["id"] + ".json")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("schema_version") != 1 or result.get("spec") != spec:
            return None
        if result.get("manifest_sha256") != digest(manifest):
            return None
        if result.get("content_sha256") != digest({k: v for k, v in result.items() if k != "content_sha256"}):
            return None
        for split in ("validation", "test"):
            metrics = result["evaluation"][split]
            if [r["case_index"] for r in metrics["per_case"]] != spec["evaluation_case_indices"][split]:
                return None
            groups = metrics["per_source_group"]
            if sorted(i for g in groups for i in g["case_indices"]) != sorted(spec["evaluation_case_indices"][split]):
                return None
            if not all(np.isfinite(metrics[k]) and metrics[k] >= 0 for k in ("mae_ms", "rmse_ms")):
                return None
        for relative, expected in result["model_artifacts"].items():
            artifact = (output / relative).resolve()
            artifact.relative_to(output)
            if not artifact.is_file() or file_hash(artifact) != expected:
                return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def output_lock(output):
    with (Path(output) / ".baseline.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This baseline output directory is already running") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
