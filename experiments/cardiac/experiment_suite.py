"""Unattended, deterministic cardiac experiments; run from the repository root.

python -m experiments.cardiac.experiment_suite --backend numba --output-root runs_3d
An interrupted fit restarts deterministically; a finished fit resumes evaluation.
Completed, validated run files are never recomputed. No wall-clock time limit.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

from .timeactml import load_timeactml, prepare_split, training_latency


SCHEMA_VERSION = 1
LATENCY_MS = 1.0529606913652159
REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("canonical", "fewshot", "refinement_refit", "near_source", "subsampling", "multistart")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_groups(dataset, case_ids):
    """Group every selected case by its exact source, preserving frozen order."""
    groups = {}
    for index in case_ids:
        source = tuple(int(x) for x in dataset.sources_nzyx[index])
        row = groups.setdefault(source, {"source_zyx": list(source), "case_indices": []})
        row["case_indices"].append(int(index))
    return list(groups.values())


def make_subsets(dataset, train_ids, counts=None, seed=20260905):
    """Predetermine unique subsets using ONLY frozen training source geometry.

    Maximize distinct z layers first, then z span. Random fixed-seed ordering
    breaks geometry ties; a least-used-group preference spreads representation.
    Subsets overlap and their across-subset spread is descriptive, not a CI.
    """
    counts = counts or {2: 8, 4: 8, 8: 8, 16: 1}
    groups = source_groups(dataset, train_ids)
    result = []
    for size, repetitions in sorted(counts.items()):
        if size > len(groups) or size < 1:
            raise ValueError("Subset size exceeds the frozen training source groups")
        candidates = list(itertools.combinations(range(len(groups)), size))
        if repetitions > len(candidates):
            raise ValueError("Requested more unique subsets than possible")
        usage = np.zeros(len(groups), dtype=int)
        selected = set()
        for repeat in range(repetitions):
            subset_seed = int(seed + size * 1000 + repeat)
            rng = np.random.default_rng(subset_seed)
            shuffled = rng.permutation(len(candidates))
            best, best_score = None, None
            for candidate_index in shuffled:
                candidate = candidates[int(candidate_index)]
                if candidate in selected:
                    continue
                depths = [groups[i]["source_zyx"][0] for i in candidate]
                score = (len(set(depths)), max(depths) - min(depths), -int(usage[list(candidate)].sum()))
                if best_score is None or score > best_score:
                    best, best_score = candidate, score
            selected.add(best)
            usage[list(best)] += 1
            chosen = {i for group in best for i in groups[group]["case_indices"]}
            ids = [int(i) for i in train_ids if i in chosen]
            result.append({"train_size": size, "subset_index": repeat, "subset_seed": subset_seed,
                           "training_case_indices": ids,
                           "training_source_groups": source_groups(dataset, ids)})
    return result


def build_plan(dataset, split, config):
    methods = ["analytical"] if config["analytical_only"] else ["analytical", "numerical"]
    train = split["train"]
    evaluation = {p: split[p] for p in ("validation", "test")}
    if config["smoke"]:
        train = [i for group in source_groups(dataset, train)[:2] for i in group["case_indices"]]
        evaluation = {p: [i for group in source_groups(dataset, ids)[:1] for i in group["case_indices"]]
                      for p, ids in evaluation.items()}
    base = {"training_case_indices": train, "training_source_groups": source_groups(dataset, train),
            "evaluation_case_indices": evaluation, "train_size": len(source_groups(dataset, train)),
            "subset_index": 0, "subset_seed": None, "refine_factor": 1,
            "exclude_distance_mm": 0.0, "samples_per_case": 128 if config["smoke"] else 0,
            "sampling_seed": config["seed"], "initial_scale": 0.85}
    plan = []

    def add(experiment, label, method, model, alias=None, **updates):
        spec = dict(base, id=f"{experiment}/{label}_{method}_{model}", experiment=experiment,
                    method=method, model=model, **updates)
        if experiment == "canonical":
            spec["id"] = f"canonical/{method}_{model}"
        if alias:
            spec["alias_of"] = alias
        plan.append(spec)

    for method in methods:
        for model in ("isotropic", "diagonal"):
            add("canonical", "", method, model)
    subsets = make_subsets(dataset, train, {2: 1} if config["smoke"] else None, config["seed"])
    for subset in subsets:
        for method in methods:
            if method == "numerical" and config["skip_numerical_fewshot"]:
                continue
            for model in ("isotropic", "diagonal"):
                alias = f"canonical/{method}_{model}" if subset["training_case_indices"] == train else None
                add("fewshot", f"n{subset['train_size']:02d}_s{subset['subset_index']:02d}",
                    method, model, alias, **subset)
    if not config["skip_refined_refit"] and not config["analytical_only"]:
        for model in ("isotropic", "diagonal"):
            add("refinement_refit", "exact", "analytical", model, f"canonical/analytical_{model}")
            add("refinement_refit", "h080", "numerical", model, f"canonical/numerical_{model}")
            add("refinement_refit", "h040", "numerical", model, refine_factor=2)
    if not config["skip_near_source"]:
        for cutoff in ([0.0, 8.0] if config["smoke"] else [0.0, 2.4, 8.0]):
            for method in methods:
                for model in ("isotropic", "diagonal"):
                    add("near_source", f"exclude{cutoff:g}mm", method, model,
                        f"canonical/{method}_{model}" if cutoff == 0 else None,
                        exclude_distance_mm=cutoff)
    if not config["skip_subsampling"]:
        for samples in (512, 2048, 0):
            seeds = [config["seed"]] if samples == 0 else config["sampling_seeds"]
            if config["smoke"]:
                seeds = seeds[:1]
                if samples == 2048:
                    continue
            for repeat, sampling_seed in enumerate(seeds):
                for method in methods:
                    for model in ("isotropic", "diagonal"):
                        add("subsampling", f"vox{samples or 'all'}_s{repeat:02d}", method, model,
                            f"canonical/{method}_{model}" if samples == 0 else None,
                            samples_per_case=base["samples_per_case"] if config["smoke"] else samples,
                            sampling_seed=sampling_seed)
                        if config["smoke"] and samples:
                            plan[-1]["samples_per_case"] = samples
    if config["multistart"] and not config["analytical_only"]:
        for number, start in enumerate(([0.25, 0.25, 0.25], [1.0, 0.8, 0.6], [0.35, 0.7, 0.9])):
            add("multistart", f"start{number}", "numerical", "diagonal",
                initial_speeds_xyz_mm_per_ms=start)
    return plan


def environment():
    libraries = {}
    for name in ("numpy", "scipy", "torch", "numba", "matplotlib"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "libraries": libraries,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def code_hashes():
    files = list((REPO / "experiments/cardiac").glob("*.py")) + list((REPO / "eikonal/three_d").glob("*.py"))
    return {str(p.relative_to(REPO)).replace("\\", "/"): file_hash(p) for p in sorted(files)}


def git_state():
    def git(*args):
        completed = subprocess.run(["git", "-c", f"safe.directory={REPO.as_posix()}", *args],
                                   cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip())
        return completed.stdout.rstrip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--porcelain=v1"),
                "working_tree_diff": git("diff", "HEAD", "--", "eikonal", "experiments/cardiac", "tests", "README.md")}
    except (OSError, RuntimeError) as error:
        return {"unavailable": str(error)}


def validate_protocol(dataset, split):
    expected = {"train": (16, 16), "validation": (16, 18), "test": (100, 102)}
    if dataset.shape_zyx != (9, 49, 49) or tuple(dataset.spacing_xyz) != (0.8, 0.8, 0.8):
        raise ValueError("Canonical data must use 9 x 49 x 49 nodes at 0.8 mm")
    for name, (groups, cases) in expected.items():
        if len(split[name]) != cases or len(source_groups(dataset, split[name])) != groups:
            raise ValueError(f"Frozen {name} must have {groups} source groups / {cases} cases")
    measured = training_latency(dataset, split["train"])
    if not np.isclose(measured, LATENCY_MS, rtol=0, atol=1e-12):
        raise ValueError("Frozen training source latency does not match the canonical protocol")


def canonical_reuse(path, dataset, split, data_sha):
    """Validate the existing handoff, snapshot its fits, never select by test score."""
    path = Path(path)
    record = {"path": str(path.resolve()), "accepted": False}
    if not path.is_file():
        return {}, dict(record, reason="Reference file absent; canonical fits will run")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        # The local data package independently binds this known canonical result
        # to its MAT checksum. No measured performance constants are embedded.
        evidence_paths = [Path(split["provenance"]["directory"]).parent / "metadata/mat_storage.json",
                          REPO / "metadata/mat_storage.json"]
        evidence = next(p for p in evidence_paths if p.is_file())
        if json.loads(evidence.read_text(encoding="utf-8"))["raw_mat_sha256"] != data_sha:
            raise ValueError("MAT differs from verified canonical package")
        if doc["shape_zyx"] != list(dataset.shape_zyx) or doc["spacing_xyz_mm"] != list(dataset.spacing_xyz):
            raise ValueError("Reference grid differs")
        if abs(doc["latency_ms"] - LATENCY_MS) > 1e-12 or doc["samples_per_training_case"] != 0:
            raise ValueError("Reference latency or observation protocol differs")
        if any(doc["case_indices"][p] != split[p] for p in ("train", "validation", "test")):
            raise ValueError("Reference frozen case lists differ")
        if doc["canonical_analytical_training_case_indices"] != split["train"]:
            raise ValueError("Reference analytical training list differs")
        fits = {}
        for name, entry in doc["models"].items():
            speeds = np.asarray(entry["speeds_xyz_mm_per_ms"], dtype=float)
            if speeds.shape != (3,) or not np.isfinite(speeds).all() or (speeds <= 0).any():
                raise ValueError("Invalid canonical speeds")
            if abs(entry["latency_ms"] - LATENCY_MS) > 1e-12:
                raise ValueError("Model latency differs")
            for part in ("validation", "test"):
                rows = entry["evaluation"][part]["per_case"]
                if [r["case_index"] for r in rows] != split[part]:
                    raise ValueError("Reference evaluation case identities differ")
                if any(r["source_zyx"] != dataset.sources_nzyx[r["case_index"]].tolist() for r in rows):
                    raise ValueError("Reference source identities differ")
                if any(not np.isfinite(r[k]) or r[k] < 0 for r in rows for k in ("mae_ms", "rmse_ms")):
                    raise ValueError("Invalid reference evaluation metrics")
            fit = {k: v for k, v in entry.items() if k not in ("evaluation",)}
            fit["reuse_provenance"] = {"path": str(path.resolve()), "sha256": file_hash(path),
                                      "policy": "Reuse fitted parameters; recompute missing source-group spatial metrics"}
            fits[name] = fit
        return fits, dict(record, accepted=True, sha256=file_hash(path),
                          data_hash_evidence={"path": str(evidence.resolve()), "sha256": file_hash(evidence)})
    except (KeyError, ValueError, TypeError, StopIteration, OSError) as error:
        return {}, dict(record, reason=str(error))


def create_manifest(args, dataset, split):
    config = {name: getattr(args, name) for name in
              ("backend", "seed", "smoke", "analytical_only", "skip_refined_refit", "skip_numerical_fewshot",
               "skip_near_source", "skip_subsampling", "multistart", "bootstrap_seed", "bootstrap_replicates")}
    config.update(latency_ms=LATENCY_MS, sampling_seeds=[args.seed + 100000 + i for i in range(args.sampling_repeats)],
                  optimizer={"max_iter": 2 if args.smoke else args.max_iter,
                             "max_eval": 6 if args.smoke else args.max_iter * 3,
                             "tolerance_grad": 1e-7, "tolerance_change": 1e-10,
                             "analytical_max_nfev": 500, "algorithm": "L-BFGS strong Wolfe, log-speeds",
                             "torch_num_threads": 1},
                  solver={"tol": 1e-10, "residual_tol": 1e-8, "max_cycles": 64,
                          "b": 0, "dtype": "float64"})
    sha = file_hash(dataset.path)
    reference_fits, reference_provenance = canonical_reuse(args.canonical_results, dataset, split, sha)
    fits, reuse = (reference_fits, reference_provenance) if not args.no_reuse and not args.smoke else (
        {}, {"accepted": False, "reason": "Disabled by --no-reuse or smoke's different training protocol"})
    qualitative_reference = (json.loads(Path(args.canonical_results).read_text(encoding="utf-8"))
                             if reference_provenance["accepted"] else None)
    root = Path(split["provenance"]["directory"])
    runs = build_plan(dataset, split, config)
    return {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(), "config": config,
            "exact_command_line": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "argv": [sys.executable, *sys.argv], "working_directory": str(Path.cwd()),
            "git": git_state(), "environment": environment(), "code_sha256": code_hashes(),
            "data": {"path": str(dataset.path), "sha256": sha, "size_bytes": dataset.path.stat().st_size},
            "frozen_manifests": {p: {"path": str((root / f"{p}.json").resolve()),
                                       "sha256": file_hash(root / f"{p}.json")} for p in ("train", "validation", "test")},
            "canonical_latency_ms": LATENCY_MS, "spacing_xyz_mm": list(dataset.spacing_xyz),
            "shape_zyx": list(dataset.shape_zyx), "split": split,
            "source_groups": {p: source_groups(dataset, split[p]) for p in ("train", "validation", "test")},
            "subset_selection_rule": make_subsets.__doc__,
            "voxel_sampling_rule": "NumPy default_rng per recorded seed, C-ravel ZYX indices; saved per-run selection hashes",
            "bootstrap_seed": config["bootstrap_seed"], "bootstrap_replicates": config["bootstrap_replicates"],
            "requested_experiments": [e for e in EXPERIMENTS if any(r["experiment"] == e for r in runs)],
            "canonical_reuse": reuse, "reusable_fits": fits, "runs": runs,
            "qualitative": {"enabled": True, "reference_provenance": reference_provenance,
                            "canonical_reference": qualitative_reference,
                            "scope": "Saved canonical fits only; held-out visualization, never refitting",
                            "fixed_activation_times_ms": [20.0, 40.0, 60.0]},
            "scientific_protocol": {"latency": "Fixed canonical training-only value in every run; no per-case alignment",
                                    "statistics": "Paired bootstrap subjects are held-out source groups, never voxels",
                                    "training_loss": "Equal case weight; mean squared error over selected original observations per case",
                                    "refinement": "Same physical domain; source indices times two, loss/evaluation only original nodes",
                                    "limits": "No CPU or wall-clock budget; optimizer iteration limits are recorded convergence safeguards"}}


def result_path(suite, spec):
    path = (Path(suite) / (spec["id"] + ".json")).resolve()
    path.relative_to(Path(suite).resolve())
    return path


def valid_result(path, spec, manifest_hash):
    """Reject truncated, tampered, foreign-protocol, incomplete or nonfinite runs."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if (value.get("status") != "complete" or value.get("schema_version") != SCHEMA_VERSION
                or value.get("manifest_sha256") != manifest_hash or value.get("spec") != spec):
            return None
        if value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"}):
            return None
        speeds = np.asarray(value["fit"]["speeds_xyz_mm_per_ms"], dtype=float)
        if speeds.shape != (3,) or not np.isfinite(speeds).all() or (speeds <= 0).any():
            return None
        if not np.isfinite(value["fit"]["train_objective_mse_ms2"]) or value["fit"]["train_objective_mse_ms2"] < 0:
            return None
        for part in ("validation", "test"):
            metrics = value["evaluation"][part]
            if not all(np.isfinite(metrics[k]) and metrics[k] >= 0 for k in ("mae_ms", "rmse_ms")):
                return None
            if sorted(r["case_index"] for r in metrics["per_case"]) != sorted(spec["evaluation_case_indices"][part]):
                return None
            groups = metrics["per_source_group"]
            if not groups or sorted(i for g in groups for i in g["case_indices"]) != sorted(spec["evaluation_case_indices"][part]):
                return None
            by_case = {r["case_index"]: r for r in metrics["per_case"]}
            if len({tuple(g["source_zyx"]) for g in groups}) != len(groups):
                return None
            for group in groups:
                if any(not np.isfinite(group[k]) or group[k] < 0 for k in ("mae_ms", "rmse_ms")):
                    return None
                if any(by_case[i]["source_zyx"] != group["source_zyx"] for i in group["case_indices"]):
                    return None
            if spec["method"] == "numerical":
                for section in ("discretization", "exact_at_fitted_speeds"):
                    if any(not np.isfinite(value[section][part][k]) or value[section][part][k] < 0 for k in ("mae_ms", "rmse_ms")):
                        return None
        return value
    except (OSError, ValueError, TypeError, KeyError):
        return None


@contextmanager
def suite_lock(suite):
    """OS advisory lock releases automatically on process death, including Windows."""
    lock_path = Path(suite) / ".suite.lock"
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This suite is already running; use a different directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute_manifest(suite, manifest, dataset, fit_executor=None, reporter=None):
    import torch
    from .suite_fitting import fit_run
    from .suite_reporting import aggregate_suite
    fit_executor = fit_executor or fit_run
    reporter = reporter or aggregate_suite
    torch.set_num_threads(manifest["config"].get("optimizer", {}).get("torch_num_threads", 1))
    suite = Path(suite).resolve()
    manifest_hash = digest(manifest)
    specs = {r["id"]: r for r in manifest["runs"]}
    counts = {"completed": 0, "skipped": 0, "failed": 0, "reporting_failed": 0, "qualitative_failed": 0}

    def event(action, **fields):
        entry = dict(time_utc=utc_now(), action=action, **fields)
        with (suite / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(action.upper(), fields.get("run_id", ""), fields.get("message", ""), flush=True)

    def report():
        try:
            output = reporter(suite)
            counts["reporting_failed"] = int(bool((output or {}).get("reporting_errors")))
            if counts["reporting_failed"]:
                event("reporting_failed", message="; ".join(output["reporting_errors"]))
        except Exception:
            counts["reporting_failed"] = 1
            failure = {"time_utc": utc_now(), "traceback": traceback.format_exc()}
            atomic_json(suite / "aggregate/REPORTING_FAILURE.json", failure)
            event("reporting_failed", message=failure["traceback"])

    with suite_lock(suite):
        event("invocation", command_line=subprocess.list2cmdline([sys.executable, *sys.argv]))
        previous_experiment = None
        try:
            for spec in manifest["runs"]:
                if previous_experiment is not None and spec["experiment"] != previous_experiment:
                    report()
                previous_experiment = spec["experiment"]
                path = result_path(suite, spec)
                existing = valid_result(path, spec, manifest_hash)
                if existing is not None:
                    counts["skipped"] += 1
                    event("skip", run_id=spec["id"])
                    continue
                progress_path = path.with_suffix(".progress.json")

                def checkpoint(progress):
                    document = dict(progress, spec=spec, manifest_sha256=manifest_hash)
                    document["content_sha256"] = digest(document)
                    atomic_json(progress_path, document)

                event("start", run_id=spec["id"])
                try:
                    if "alias_of" in spec:
                        origin_spec = specs[spec["alias_of"]]
                        origin = valid_result(result_path(suite, origin_spec), origin_spec, manifest_hash)
                        if origin is None:
                            raise RuntimeError(f"Required control is unavailable: {spec['alias_of']}")
                        result = {k: deepcopy(v) for k, v in origin.items() if k not in
                                  ("spec", "content_sha256", "completed_utc", "alias_of")}
                        result["alias_of"] = spec["alias_of"]
                        result["elapsed_seconds"] = 0.0
                    else:
                        reused_fit = manifest.get("reusable_fits", {}).get(f"{spec['method']}_{spec['model']}") if spec["experiment"] == "canonical" else None
                        if progress_path.is_file():
                            try:
                                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                                if (progress.get("phase") in ("fitted", "evaluating") and progress.get("spec") == spec
                                        and progress.get("manifest_sha256") == manifest_hash
                                        and progress.get("content_sha256") == digest({k: v for k, v in progress.items() if k != "content_sha256"})):
                                    reused_fit = progress["fit"]
                                    event("resume_evaluation", run_id=spec["id"])
                            except (OSError, ValueError, KeyError):
                                pass
                        result = fit_executor(dataset, spec, manifest["config"], checkpoint=checkpoint,
                                              reused_fit=reused_fit)
                    result.update(schema_version=SCHEMA_VERSION, status="complete", spec=spec,
                                  manifest_sha256=manifest_hash, completed_utc=utc_now())
                    result["content_sha256"] = digest(result)
                    # Preserve any failed or invalid earlier attempt, never overwrite valid results.
                    if path.is_file():
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                        archive = suite / "failures" / f"{spec['id'].replace('/', '__')}_{stamp}.json"
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        archive.write_bytes(path.read_bytes())
                    atomic_json(path, result)
                    if valid_result(path, spec, manifest_hash) is None:
                        raise ValueError("Completed result failed schema/finite/identity validation")
                    counts["completed"] += 1
                    event("complete", run_id=spec["id"])
                except Exception as error:
                    failure = {"schema_version": SCHEMA_VERSION, "status": "failed", "spec": spec,
                               "manifest_sha256": manifest_hash, "failed_utc": utc_now(),
                               "error": str(error), "traceback": traceback.format_exc()}
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                    atomic_json(suite / "failures" / f"{spec['id'].replace('/', '__')}_{stamp}.json", failure)
                    atomic_json(path, failure)
                    counts["failed"] += 1
                    event("failed", run_id=spec["id"], message=str(error))
            if manifest.get("qualitative", {}).get("enabled"):
                try:
                    from .suite_qualitative import generate_qualitative
                    reference = manifest["qualitative"]["canonical_reference"]
                    if reference is None:
                        raise ValueError("Verified saved canonical numerical reference unavailable; no retraining is permitted for figures")
                    event("qualitative_start")
                    figure_config = manifest["config"]
                    if figure_config.get("backend") == "cuda":
                        figure_config = dict(figure_config, backend="numba")
                    generate_qualitative(suite, dataset, reference, figure_config)
                    event("qualitative_complete")
                except Exception as error:
                    counts["qualitative_failed"] = 1
                    atomic_json(suite / "plots/qualitative/FAILURE.json", {"status": "failed", "error": str(error),
                                "time_utc": utc_now(), "traceback": traceback.format_exc()})
                    event("qualitative_failed", message=str(error))
            event("finished", **counts)
        except KeyboardInterrupt:
            event("interrupted", message="Saved results retained; resume this suite directory")
            raise
        finally:
            report()
            atomic_json(suite / "LAST_INVOCATION.json", dict(counts, ended_utc=utc_now()))
    return counts


def resume_inputs(suite):
    manifest = json.loads((suite / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported experiment manifest version")
    if code_hashes() != manifest["code_sha256"]:
        raise ValueError("Scientific code changed since suite creation; restore the recorded code or start a new suite")
    current = environment()
    if any(current[k] != manifest["environment"][k] for k in ("python", "libraries")):
        raise ValueError("Python/library versions changed; restore the recorded environment or start a new suite")
    data_path = manifest["data"]["path"]
    if file_hash(data_path) != manifest["data"]["sha256"]:
        raise ValueError("MAT checksum changed; resume refused")
    for part, metadata in manifest["frozen_manifests"].items():
        if file_hash(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"Frozen {part} manifest checksum changed; resume refused")
    dataset = load_timeactml(data_path)
    split = prepare_split(dataset, manifest["frozen_manifests"]["train"]["path"])
    validate_protocol(dataset, split)
    if any(split[p] != manifest["split"][p] for p in ("train", "validation", "test")):
        raise ValueError("Frozen case identities changed")
    return manifest, dataset


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--resume", type=Path, help="Resume immutable saved manifest; do not combine with experiment options")
    result.add_argument("--aggregate-only", type=Path, metavar="SUITE_DIR", help="Generate tables, plots and summaries from saved JSON only")
    result.add_argument("--output-root", type=Path, default=Path("runs_3d"))
    result.add_argument("--data", type=Path)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--canonical-results", type=Path, default=REPO / "handoff/3D_CARDIAC_RESULTS.json")
    result.add_argument("--backend", choices=("auto", "python", "numba", "cuda"), default="numba")
    for option in ("skip-refined-refit", "skip-numerical-fewshot", "skip-near-source", "skip-subsampling", "analytical-only", "smoke", "multistart", "no-reuse"):
        result.add_argument("--" + option, action="store_true")
    result.add_argument("--seed", type=int, default=20260905)
    result.add_argument("--bootstrap-seed", type=int, default=20260906)
    result.add_argument("--bootstrap-replicates", type=int, default=10000)
    result.add_argument("--sampling-repeats", type=int, default=3)
    result.add_argument("--max-iter", type=int, default=100, help="Numerical optimizer iterations per fit; no wall-clock limit")
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if any(getattr(args, field) < 1 for field in ("max_iter", "sampling_repeats", "bootstrap_replicates")):
        command.error("Iteration and repeat counts must be positive")
    if args.seed < 0 or args.bootstrap_seed < 0:
        command.error("Seeds must be nonnegative")
    if args.resume and args.aggregate_only:
        command.error("Choose --resume or --aggregate-only")
    if args.resume or args.aggregate_only:
        tokens = list(sys.argv[1:] if argv is None else argv)
        allowed = "--resume" if args.resume else "--aggregate-only"
        if any(t.startswith("--") and t.split("=")[0] != allowed for t in tokens):
            command.error(f"{allowed} uses the saved configuration; omit all other options")
    if args.aggregate_only:
        from .suite_reporting import aggregate_suite
        output = aggregate_suite(args.aggregate_only.resolve())
        return 1 if output.get("reporting_errors") else 0
    if args.resume:
        suite = args.resume.resolve()
        manifest, dataset = resume_inputs(suite)
    else:
        data_path = args.data
        if data_path is None and not list(Path("local_data").rglob("dataset_acti_3d_v1.mat")):
            data_path = REPO / "raw/dataset_3D_structured/dataset_acti_3d_v1.mat"
        dataset = load_timeactml(data_path)
        split = prepare_split(dataset, args.manifest)
        validate_protocol(dataset, split)
        manifest = create_manifest(args, dataset, split)
        suite = args.output_root.resolve() / ("local_experiment_suite_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        suite.mkdir(parents=True, exist_ok=False)
        for directory in (*EXPERIMENTS, "aggregate", "plots", "failures"):
            (suite / directory).mkdir()
        atomic_json(suite / "EXPERIMENT_MANIFEST.json", manifest)
        for relative, expected in manifest["code_sha256"].items():
            destination = suite / "provenance/code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / relative).read_bytes())
            if file_hash(destination) != expected:
                raise RuntimeError("Scientific code changed while snapshotting the suite")
        for part, metadata in manifest["frozen_manifests"].items():
            destination = suite / "provenance/manifests" / (part + ".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(metadata["path"]).read_bytes())
        if manifest["qualitative"]["canonical_reference"] is not None:
            atomic_json(suite / "plots/qualitative/CANONICAL_REFERENCE.json", manifest["qualitative"]["canonical_reference"])
    print(f"SUITE_DIR={suite}", flush=True)
    print(f"Planned results={len(manifest['runs'])}; smoke={manifest['config']['smoke']}; no wall-clock limit", flush=True)
    try:
        counts = execute_manifest(suite, manifest, dataset)
    except KeyboardInterrupt:
        return 130
    print(f"Saved {suite / 'SUMMARY.md'}; {counts}", flush=True)
    return 1 if counts["failed"] or counts["reporting_failed"] or counts["qualitative_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
