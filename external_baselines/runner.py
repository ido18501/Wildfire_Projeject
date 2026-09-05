"""Unattended external baselines on the exact saved eikonal few-shot protocol."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np

from .protocol import (ROOT, LATENCY_MS, UPSTREAM_COMMIT, atomic_json, atomic_npz,
                       common_case, digest, ensure_isolated, environment, evaluate_arrays,
                       file_hash, inspect_eikonal_suite, load_dataset, now, output_lock,
                       source_hashes, valid_result)


def make_plan(subsets, seeds, methods, smoke=False):
    selected = [s for s in subsets if s["train_size"] == 2][:1] if smoke else subsets
    runs = []
    for method in methods:
        for subset in selected:
            for seed in ([0] if method == "kol" else seeds[:1] if smoke else seeds):
                runs.append(dict(subset, method=method, seed=int(seed),
                                 id=f"{method}/n{subset['train_size']:02d}_s{subset['subset_index']:02d}_seed{seed}"))
    return runs


def snapshot_suite(args):
    from .fno import DEFAULTS as FNO_DEFAULTS
    from .kol import DEFAULT_CONFIG as KOL_DEFAULTS, candidate_configs
    links = json.loads((ROOT / "PROJECT_LINKS.json").read_text(encoding="utf-8"))
    source_suite = args.eikonal_suite or Path(links["eikonal_suite"])
    source_manifest, raw, subsets, references = inspect_eikonal_suite(source_suite)
    if args.eikonal_suite:
        # Server copies retain the exact frozen files/hashes, not Windows paths.
        # Existing suite layout is <project>/runs_3d/<suite>; no split is rebuilt.
        server_project = Path(source_suite).resolve().parents[1]
        links["main_repository"] = str(server_project)
        links["data"] = str(server_project / "local_data/timeactml/raw/dataset_3D_structured/dataset_acti_3d_v1.mat")
        for part, record in source_manifest["frozen_manifests"].items():
            record["path"] = str(server_project / "local_data/timeactml/poc" / (part + ".json"))
    ensure_isolated(ROOT, links["main_repository"])
    output_root = ensure_isolated(args.output_root, links["main_repository"])
    # Validate the MAT and coordinate/split semantics before creating a runnable manifest.
    source_manifest["data"]["path"] = str(Path(args.data or links["data"]).resolve())
    dataset = load_dataset(source_manifest)
    upstream = (ROOT / "external_sources/timeactML").resolve()
    completed = subprocess.run(["git", "-c", f"safe.directory={upstream.as_posix()}",
                                "-C", str(upstream), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True)
    if completed.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError("External timeactML checkout is not at the audited pinned commit")
    seed_values = [args.seed + i for i in range(args.repeats)]
    config = {"smoke": args.smoke, "methods": args.methods, "seeds": seed_values,
              "device": args.device, "torch_num_threads": args.threads, "latency_ms": LATENCY_MS,
              "fno": {**FNO_DEFAULTS, "device": args.device, "epochs": args.epochs, "patience": args.patience,
                      "min_epochs": min(200, args.epochs), "validation_every": 10,
                      "batch_size": args.batch_size, "lr": 1e-3, "weight_decay": 1e-4,
                      "inference_batch_size": 1},
              "kol": dict(KOL_DEFAULTS), "save_predictions": True,
              "selection": "Hyperparameters/checkpoints use validation only; test is evaluated after fit is fixed"}
    if args.smoke:
        config["fno"].update(epochs=2, min_epochs=0, validation_every=1, checkpoint_every=1)
    manifest = {"schema_version": 1, "created_utc": now(), "config": config,
                "source_project": links["main_repository"], "data": source_manifest["data"],
                "source_eikonal_suite": str(Path(source_suite).resolve()),
                "source_eikonal_manifest_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                "source_eikonal_manifest_semantic_sha256": digest(json.loads(raw)),
                "source_git": source_manifest.get("git"),
                "frozen_manifests": source_manifest["frozen_manifests"],
                "split": source_manifest["split"], "source_groups": source_manifest["source_groups"],
                "shape_zyx": source_manifest["shape_zyx"], "spacing_xyz_mm": source_manifest["spacing_xyz_mm"],
                "canonical_latency_ms": LATENCY_MS, "subsets": subsets,
                "runs": make_plan(subsets, seed_values, args.methods, args.smoke),
                "environment": environment(), "code_sha256": source_hashes(),
                "command_line": subprocess.list2cmdline(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
                "argv": getattr(sys, "orig_argv", [sys.executable, *sys.argv]), "working_directory": str(Path.cwd()),
                "research_sha256": file_hash(ROOT / "BASELINE_RESEARCH.md"),
                "upstream": json.loads((ROOT / "external_sources/timeactML_source/SOURCE_PROVENANCE.json").read_text(encoding="utf-8")),
                "kol_candidates": candidate_configs(config["kol"]),
                "references": [{k: v for k, v in row.items() if k != "bytes"} for row in references],
                "qualitative_case_index": common_case(dataset, source_manifest["split"]["test"]),
                "qualitative_selection_rule": "Source nearest physical domain center; tie source ZYX then lowest case index. No labels or performance used.",
                "winner_selection_rule": "Lowest mean N=16 validation MAE across every planned initialization seed; no winner with incomplete coverage",
                "fidelity": {"kol": "Exact pinned kernel/timing preset included in predetermined validation search; no upstream KOL source is imported",
                             "fno": "Original pinned architecture imported from separate unlicensed checkout; training uses our frozen cases and validation scheduler/checkpoint repair"},
                "interpretation": "Comparison measures data efficiency on this extreme few-shot frozen dataset, not general method quality."}
    output = output_root / ("external_baselines_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("results", "models", "predictions", "references", "logs", "failures", "aggregate", "plots", "qualitative", "provenance"):
        (output / folder).mkdir()
    atomic_json(output / "BASELINE_MANIFEST.json", manifest)
    shutil.copyfile(ROOT / "BASELINE_RESEARCH.md", output / "BASELINE_RESEARCH.md")
    (output / "references/EXPERIMENT_MANIFEST.json").write_bytes(raw)
    for reference in references:
        destination = output / reference["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(reference["bytes"])
    for part, record in manifest["frozen_manifests"].items():
        shutil.copyfile(record["path"], output / "references" / (part + ".json"))
    for relative in manifest["code_sha256"]:
        destination = output / "provenance/code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    atomic_json(output / "provenance/UPSTREAM_SOURCE.json", manifest["upstream"])
    return output, manifest, dataset


def resume_suite(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    ensure_isolated(output, manifest["source_project"])
    ensure_isolated(ROOT, manifest["source_project"])
    if manifest["schema_version"] != 1 or source_hashes() != manifest["code_sha256"]:
        raise ValueError("Baseline code differs from the frozen manifest; restore the source snapshot or create a new suite")
    current = environment()
    if any(current[k] != manifest["environment"][k] for k in ("python", "libraries")):
        raise ValueError("Python/library versions differ from the frozen baseline manifest")
    for row in manifest["references"]:
        if file_hash(output / row["path"]) != row["sha256"]:
            raise ValueError("Saved eikonal comparison snapshot changed")
    dataset = load_dataset(manifest)
    return manifest, dataset


def fit_one(output, manifest, dataset, spec):
    import torch
    trainer_config = dict(manifest["config"][spec["method"]])
    if spec["method"] == "fno":
        from .fno import fit_fno as train
        trainer_config["upstream_dir"] = str(ROOT / "external_sources/timeactML")
    else:
        from .kol import fit_kol as train
    train_ids = spec["training_case_indices"]
    validation_ids = spec["evaluation_case_indices"]["validation"]
    artifact_dir = output / "models" / spec["id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    # Trainers receive only the selected training and frozen validation arrays.
    # No test array, dataset object, or test score is available in either API.
    predictor, metadata = train(dataset.sources_nzyx[train_ids], dataset.activation_nzyx[train_ids],
                                dataset.sources_nzyx[validation_ids], dataset.activation_nzyx[validation_ids],
                                config=trainer_config, seed=spec["seed"], artifact_dir=artifact_dir,
                                smoke=manifest["config"]["smoke"])
    training_wall_seconds = perf_counter() - started
    evaluation = {}
    prediction_paths = {}
    for part in ("validation", "test"):
        case_ids = spec["evaluation_case_indices"][part]
        started = perf_counter()
        prediction = np.asarray(predictor.predict(dataset.sources_nzyx[case_ids]), dtype=np.float64)
        elapsed = perf_counter() - started
        evaluation[part] = evaluate_arrays(dataset.activation_nzyx[case_ids], prediction,
                                          dataset.sources_nzyx[case_ids], case_ids, elapsed)
        relative = "predictions/" + spec["id"] + "_" + part + ".npz"
        atomic_npz(output / relative, predictions_nzyx=prediction, case_indices=np.asarray(case_ids, dtype=np.int64))
        prediction_paths[part] = relative
    metadata["adapter_training_wall_seconds"] = training_wall_seconds
    artifact_paths = sorted(p for p in artifact_dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    hashes = {p.relative_to(output).as_posix(): file_hash(p) for p in artifact_paths}
    hashes.update({relative: file_hash(output / relative) for relative in prediction_paths.values()})
    return {"fit": metadata, "evaluation": evaluation, "model_artifacts": hashes,
            "prediction_paths": prediction_paths,
            "timing_caveat": "External inference excludes metric calculation and disk writes; saved eikonal elapsed times include diagnostics and are not directly comparable."}


def generate_common_qualitative(output, manifest, dataset):
    from .reporting import read_results, select_strongest
    results = read_results(output, manifest)
    # read_results is documented to return an ID mapping; tolerate a tuple carrying statuses.
    if isinstance(results, tuple):
        results = results[0]
    winner = select_strongest(manifest, results)
    smoke = manifest["config"]["smoke"]
    if smoke:
        candidates = [s for s in manifest["runs"] if s["method"] == "kol"]
        selected = next((s for s in candidates if valid_result(output, s, manifest)), None)
        criterion = "SMOKE ONLY: predeclared KOL example, not a selected winner"
    elif winner:
        candidates = [s for s in manifest["runs"] if s["method"] == winner and s["train_size"] == 16]
        selected = min(candidates, key=lambda s: s["seed"]) if candidates else None
        criterion = manifest["winner_selection_rule"] + "; first predetermined seed displayed"
    else:
        selected, criterion = None, manifest["winner_selection_rule"]
    if selected is None:
        atomic_json(output / "qualitative/COMPARISON.json", {"status": "pending",
                    "reason": "Complete N=16 validation coverage required before selecting the strongest external method"})
        return
    result = valid_result(output, selected, manifest)
    if result is None:
        return
    case_id = manifest["qualitative_case_index"]
    reference = next(r for r in manifest["references"] if r["method"] == "eikonal_diagonal"
                     and r["train_size"] == selected["train_size"] and r["subset_index"] == selected["subset_index"])
    input_fingerprint = digest({"reference": reference["sha256"], "result": result["content_sha256"],
                                "case_index": case_id, "criterion": criterion})
    metadata_path, array_path = output / "qualitative/COMPARISON.json", output / "qualitative/COMPARISON.npz"
    if metadata_path.is_file() and array_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("input_fingerprint") == input_fingerprint and previous.get("array_sha256") == file_hash(array_path):
            return
    with np.load(output / result["prediction_paths"]["test"], allow_pickle=False) as arrays:
        positions = np.flatnonzero(arrays["case_indices"] == case_id)
        if len(positions) != 1:
            raise ValueError("Common held-out case absent from saved external predictions")
        prediction = arrays["predictions_nzyx"][positions[0]].copy()
    reference_result = json.loads((output / reference["path"]).read_text(encoding="utf-8"))
    from eikonal.three_d.solver import solve_diagonal
    source = tuple(int(v) for v in dataset.sources_nzyx[case_id])
    # Regenerate one eikonal volume from already fitted saved parameters; never fit.
    numerical = solve_diagonal(dataset.shape_zyx, source, reference_result["fit"]["speeds_xyz_mm_per_ms"],
                               dataset.spacing_xyz, backend="numba")
    eikonal = numerical.arrival + LATENCY_MS
    atomic_npz(array_path, ground_truth_zyx=dataset.activation_nzyx[case_id], eikonal_zyx=eikonal,
               external_zyx=prediction, source_zyx=np.asarray(source), case_index=np.asarray(case_id),
               spacing_xyz_mm=np.asarray(dataset.spacing_xyz))
    atomic_json(metadata_path, {"status": "complete", "input_fingerprint": input_fingerprint,
                "array_sha256": file_hash(array_path), "method": selected["method"], "seed": selected["seed"],
                "train_size": selected["train_size"], "subset_index": selected["subset_index"],
                "case_index": case_id, "selection_rule": criterion, "smoke": smoke})


def execute(output, manifest, dataset, trainer=None, reporter=None):
    import torch
    from .reporting import aggregate
    trainer = trainer or fit_one
    reporter = reporter or aggregate
    torch.set_num_threads(manifest["config"].get("torch_num_threads", 1))
    counts = {"completed": 0, "skipped": 0, "failed": 0, "reporting_failed": 0, "qualitative_failed": 0}

    def log(action, **fields):
        row = dict(time_utc=now(), action=action, **fields)
        with (output / "logs/events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(action.upper(), fields.get("run_id", ""), fields.get("error", ""), flush=True)

    def report():
        try:
            summary = reporter(output)
            counts["reporting_failed"] = int(bool((summary or {}).get("reporting_errors")))
        except Exception as error:
            counts["reporting_failed"] = 1
            atomic_json(output / "failures/reporting.json", {"error": str(error), "traceback": traceback.format_exc()})
            log("reporting_failed", error=str(error))

    with output_lock(output):
        log("invocation", command_line=subprocess.list2cmdline(getattr(sys, "orig_argv", [sys.executable, *sys.argv])))
        try:
            for spec in manifest["runs"]:
                path = output / "results" / (spec["id"] + ".json")
                if valid_result(output, spec, manifest):
                    counts["skipped"] += 1
                    log("skip", run_id=spec["id"])
                    continue
                log("start", run_id=spec["id"])
                if path.is_file():
                    backup = output / "failures" / (spec["id"].replace("/", "__") + "_" +
                             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
                    shutil.copyfile(path, backup)
                try:
                    result = trainer(output, manifest, dataset, spec)
                    result.update(schema_version=1, status="complete", spec=spec,
                                  manifest_sha256=digest(manifest), completed_utc=now())
                    result["content_sha256"] = digest(result)
                    atomic_json(path, result)
                    if valid_result(output, spec, manifest) is None:
                        raise ValueError("Completed baseline result failed integrity/identity checks")
                    counts["completed"] += 1
                    log("complete", run_id=spec["id"])
                except Exception as error:
                    failure = {"schema_version": 1, "status": "failed", "spec": spec,
                               "manifest_sha256": digest(manifest), "time_utc": now(),
                               "error": str(error), "traceback": traceback.format_exc()}
                    atomic_json(path, failure)
                    atomic_json(output / "failures" / (spec["id"].replace("/", "__") + "_" +
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"), failure)
                    counts["failed"] += 1
                    log("failed", run_id=spec["id"], error=str(error))
                report()
            try:
                generate_common_qualitative(output, manifest, dataset)
            except Exception as error:
                counts["qualitative_failed"] = 1
                atomic_json(output / "failures/qualitative.json", {"error": str(error), "traceback": traceback.format_exc()})
                log("qualitative_failed", error=str(error))
        except KeyboardInterrupt:
            log("interrupted", error="Progress saved. Resume this baseline directory.")
            raise
        finally:
            report()
            atomic_json(output / "LAST_INVOCATION.json", dict(counts, ended_utc=now()))
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--aggregate-only", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs_3d")
    parser.add_argument("--eikonal-suite", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--methods", nargs="+", choices=("kol", "fno"), default=["kol", "fno"])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods):
        parser.error("Each method may be requested only once")
    if any(getattr(args, p) < 1 for p in ("epochs", "patience", "batch_size", "repeats", "threads")) or args.seed < 0:
        parser.error("Positive epoch/repeat/thread counts and nonnegative seed required")
    if args.resume and args.aggregate_only:
        parser.error("Choose --resume or --aggregate-only")
    if args.resume or args.aggregate_only:
        option = "--resume" if args.resume else "--aggregate-only"
        tokens = sys.argv[1:] if argv is None else argv
        if any(t.startswith("--") and t.split("=")[0] != option for t in tokens):
            parser.error(option + " uses the saved configuration; omit other options")
    if args.aggregate_only:
        from .reporting import aggregate
        document = json.loads((args.aggregate_only / "BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
        ensure_isolated(args.aggregate_only, document["source_project"])
        result = aggregate(args.aggregate_only)
        return int(bool(result.get("reporting_errors")))
    if args.resume:
        output = args.resume.resolve()
        manifest, dataset = resume_suite(output)
    else:
        output, manifest, dataset = snapshot_suite(args)
    print("BASELINE_DIR=" + str(output), flush=True)
    print(f"Planned baseline runs={len(manifest['runs'])}; smoke={manifest['config']['smoke']}", flush=True)
    try:
        counts = execute(output, manifest, dataset)
    except KeyboardInterrupt:
        return 130
    print("FINISHED " + json.dumps(counts), flush=True)
    print("SUMMARY=" + str(output / "SUMMARY.md"), flush=True)
    return 1 if counts["failed"] or counts["reporting_failed"] or counts["qualitative_failed"] else 0
