"""Saved-results-only summaries and figures for the isolated baseline suite.

Subsets are the replication unit of the few-shot curve. Initialization seeds
are averaged *within* a subset; they never increase its statistical weight.
No reporting function trains, selects a test-optimal seed, or changes a split.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


LABELS = {
    "eikonal_diagonal": "Anisotropic eikonal (ours)",
    "eikonal_isotropic": "Isotropic eikonal ablation",
    "kol": "Kernel operator learning",
    "fno": "Fourier neural operator",
}
COLORS = {"eikonal_diagonal": "#124e78", "eikonal_isotropic": "#86a6b9",
          "kol": "#c86129", "fno": "#42744e"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_child(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_evaluation(record, spec):
    for part in ("validation", "test"):
        metrics = record["evaluation"][part]
        if any(not _finite(metrics[k]) or metrics[k] < 0 for k in ("mae_ms", "rmse_ms")):
            return False
        expected = spec["evaluation_case_indices"][part]
        cases = metrics["per_case"]
        groups = metrics["per_source_group"]
        if (sorted(row["case_index"] for row in cases) != sorted(expected) or
                sorted(i for group in groups for i in group["case_indices"]) != sorted(expected)):
            return False
        if len({tuple(g["source_zyx"]) for g in groups}) != len(groups):
            return False
        by_case = {row["case_index"]: row for row in cases}
        for row in [*cases, *groups]:
            if any(not _finite(row[k]) or row[k] < 0 for k in ("mae_ms", "rmse_ms")):
                return False
        for group in groups:
            if any(by_case[i]["source_zyx"] != group["source_zyx"] for i in group["case_indices"]):
                return False
    return True


def read_results(output_dir, manifest=None):
    """Read only complete, integrity-checked, correctly paired baseline JSONs.

    Returns (records_by_id, states_by_id). Invalid and failed files are never
    used to make figures, including partially saved evaluation records.
    """
    root = Path(output_dir).resolve()
    manifest = manifest or json.loads((root / "BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_sha = digest(manifest)
    records, states = {}, {}
    for spec in manifest["runs"]:
        run_id = spec["id"]
        try:
            path = _safe_child(root, "results/" + run_id + ".json")
            if not path.exists():
                states[run_id] = "pending"
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "failed":
                states[run_id] = "failed"
                continue
            valid = (record.get("status") == "complete" and record.get("schema_version") == 1 and
                     record.get("manifest_sha256") == manifest_sha and record.get("spec") == spec and
                     record.get("content_sha256") == digest({k: v for k, v in record.items() if k != "content_sha256"}) and
                     _valid_evaluation(record, spec))
            if valid:
                for relative, expected_hash in record.get("model_artifacts", {}).items():
                    artifact = _safe_child(root, relative)
                    if not artifact.is_file() or file_hash(artifact) != expected_hash:
                        valid = False
                        break
            states[run_id] = "complete" if valid else "invalid"
            if valid:
                records[run_id] = record
        except (OSError, ValueError, KeyError, TypeError):
            states[run_id] = "invalid"
    return records, states


def _records_map(results):
    if isinstance(results, dict):
        return results
    return {row["spec"]["id"]: row for row in results}


def select_strongest(manifest, results):
    """Choose by N=16 validation MAE after ALL planned methods/seeds complete.

    Method name breaks exact ties. No test field is read. Averaging seeds within
    subset is retained for compatibility if a future manifest has >1 N=16 set.
    """
    records = _records_map(results)
    expected = [s for s in manifest["runs"] if s["train_size"] == 16]
    if not expected or any(s["id"] not in records for s in expected):
        return None
    grouped = defaultdict(lambda: defaultdict(list))
    for spec in expected:
        record = records[spec["id"]]
        if record.get("status") != "complete":
            return None
        value = record["evaluation"]["validation"]["mae_ms"]
        if not _finite(value):
            return None
        grouped[spec["method"]][spec["subset_index"]].append(value)
    scores = {method: float(np.mean([np.mean(values) for values in subsets.values()]))
              for method, subsets in grouped.items()}
    return min(scores, key=lambda method: (scores[method], method))


def _runtime(record, part=None):
    if part:
        data = record["evaluation"][part]
        return data.get("inference_seconds", data.get("elapsed_seconds"))
    fit = record.get("fit", {})
    return next((fit[key] for key in ("training_seconds", "train_seconds", "elapsed_seconds") if key in fit), None)


def _flatten(record, method=None, source="external"):
    spec, fit = record["spec"], record.get("fit", {})
    method = method or spec["method"]
    return {"method": method, "train_size": spec["train_size"], "subset_index": spec["subset_index"],
            "subset_seed": spec.get("subset_seed"), "initialization_seed": spec.get("seed"),
            "run_id": spec["id"], "source": source,
            "validation_mae_ms": record["evaluation"]["validation"]["mae_ms"],
            "validation_rmse_ms": record["evaluation"]["validation"]["rmse_ms"],
            "test_mae_ms": record["evaluation"]["test"]["mae_ms"],
            "test_rmse_ms": record["evaluation"]["test"]["rmse_ms"],
            "training_seconds": _runtime(record),
            "validation_inference_seconds": _runtime(record, "validation"),
            "test_inference_seconds": _runtime(record, "test"),
            "runtime_caveat": "saved eikonal evaluation includes diagnostic calculations" if source == "eikonal_snapshot" else "",
            "parameter_count": fit.get("parameter_count", fit.get("trainable_parameters")),
            "real_scalar_parameter_count": fit.get("real_scalar_parameter_count", fit.get("parameter_count")),
            "model_size_bytes": fit.get("model_size_bytes", fit.get("serialized_model_bytes", fit.get("model_bytes", fit.get("size_bytes")))),
            "training_case_indices": json.dumps(spec["training_case_indices"], separators=(",", ":"))}


def _reference_records(root, manifest):
    result, errors = [], []
    for ref in manifest.get("references", []):
        try:
            path = _safe_child(root, ref["path"])
            if file_hash(path) != ref["sha256"]:
                raise ValueError("reference file checksum mismatch")
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value.get("status") != "complete" or
                    value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"}) or
                    not _valid_evaluation(value, value["spec"])):
                raise ValueError("reference result integrity or metrics invalid")
            if (value["spec"]["train_size"] != ref["train_size"] or
                    value["spec"]["subset_index"] != ref["subset_index"]):
                raise ValueError("reference subset identity mismatch")
            expected = next((s for s in manifest["runs"] if s["train_size"] == ref["train_size"] and
                             s["subset_index"] == ref["subset_index"]), None)
            if expected is not None and (value["spec"]["training_case_indices"] != expected["training_case_indices"] or
                                         value["spec"]["evaluation_case_indices"] != expected["evaluation_case_indices"]):
                # Smoke runs may deliberately evaluate just one complete source
                # group; do not compare those measurements with full references.
                if manifest.get("config", {}).get("smoke"):
                    continue
                raise ValueError("reference and baseline training/evaluation cases differ")
            result.append((ref, value))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append("%s: %s" % (ref.get("path", "unknown reference"), error))
    return result, errors


def _mean(values):
    values = [x for x in values if _finite(x)]
    return float(np.mean(values)) if values else None


def summarize_rows(rows, manifest):
    """Equal-subset means, with initialization and between-subset spread separate."""
    grouped = defaultdict(lambda: defaultdict(list))
    expected = defaultdict(lambda: defaultdict(set))
    for spec in manifest["runs"]:
        expected[(spec["method"], spec["train_size"])][spec["subset_index"]].add(spec["id"])
    for row in rows:
        grouped[(row["method"], row["train_size"])][row["subset_index"]].append(row)
    summaries = []
    for (method, size), subsets in sorted(grouped.items()):
        expected_subsets = expected.get((method, size), {})
        # Exclude incomplete stochastic subsets from the curve. A one-seed
        # partial result otherwise changes a subset's weight mid-experiment.
        complete = {index: values for index, values in subsets.items()
                    if not expected_subsets or {r["run_id"] for r in values} == expected_subsets[index]}
        row = {"method": method, "train_size": size, "completed_subsets": len(complete),
               "available_runs": sum(map(len, subsets.values())),
               "expected_subsets": len(expected_subsets) if expected_subsets else len(subsets),
               "expected_runs": sum(map(len, expected_subsets.values())) if expected_subsets else sum(map(len, subsets.values()))}
        metrics = ("validation_mae_ms", "validation_rmse_ms", "test_mae_ms", "test_rmse_ms",
                   "training_seconds", "validation_inference_seconds", "test_inference_seconds", "parameter_count", "model_size_bytes")
        for metric in metrics:
            means = [_mean([r.get(metric) for r in values]) for values in complete.values()]
            means = [x for x in means if x is not None]
            init_sd = [float(np.std([r[metric] for r in values], ddof=1)) for values in complete.values()
                       if len(values) > 1 and all(_finite(r.get(metric)) for r in values)]
            row.update({metric + "_mean": _mean(means),
                        metric + "_subset_std": float(np.std(means, ddof=1)) if len(means) > 1 else (0. if means else None),
                        metric + "_median": float(np.median(means)) if means else None,
                        metric + "_min": min(means) if means else None,
                        metric + "_max": max(means) if means else None,
                        metric + "_initialization_std_mean": _mean(init_sd)})
        summaries.append(row)
    return summaries


def _atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows, empty_fields=("method", "train_size")):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _pyplot(root):
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    return plt


def _save(plt, fig, root, name):
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / (name + ".png"), bbox_inches="tight")
    fig.savefig(root / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _method_order(methods):
    return sorted(methods, key=lambda m: (list(LABELS).index(m) if m in LABELS else 99, m))


def _plot_curves(plt, root, summaries, smoke):
    available = [r for r in summaries if r["test_mae_ms_mean"] is not None]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8), layout="constrained")
    for method in _method_order({r["method"] for r in available}):
        rows = sorted((r for r in available if r["method"] == method), key=lambda r: r["train_size"])
        ax.errorbar([r["train_size"] for r in rows], [r["test_mae_ms_mean"] for r in rows],
                    yerr=[r["test_mae_ms_subset_std"] for r in rows], marker="o", capsize=4,
                    linewidth=2, color=COLORS.get(method), label=LABELS.get(method, method))
    ax.set(xlabel="Training source groups", ylabel="Held-out full-volume MAE (ms)",
           xticks=[2, 4, 8, 16], title="Smoke verification only" if smoke else "Few-shot cardiac activation prediction")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    fig.supxlabel("Seeds averaged within each subset; bars = SD across complete subset means (descriptive).", fontsize=8)
    _save(plt, fig, root, "test_mae_vs_training_sources")
    n16 = [r for r in available if r["train_size"] == 16]
    if not n16:
        return
    order = _method_order({x["method"] for x in n16})
    n16.sort(key=lambda r: order.index(r["method"]))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    positions = np.arange(len(n16))
    ax.bar(positions, [r["test_mae_ms_mean"] for r in n16],
           yerr=[r["test_mae_ms_initialization_std_mean"] or 0. for r in n16],
           capsize=5, color=[COLORS.get(r["method"], "#555555") for r in n16], width=.67)
    ax.set(xticks=positions, xticklabels=[LABELS.get(r["method"], r["method"]).replace(" (", "\n(") for r in n16],
           ylabel="Held-out full-volume MAE (ms)", title="Canonical training set: 16 source groups")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.supxlabel("Bars = mean over predetermined initialization seeds; error bars = initialization SD.", fontsize=8)
    _save(plt, fig, root, "n16_method_comparison")


def _source_comparison(root, selected, records, references):
    external = [r for r in records.values() if r["spec"]["method"] == selected and r["spec"]["train_size"] == 16]
    reference = next((r for ref, r in references if ref["method"] == "eikonal_diagonal" and ref["train_size"] == 16), None)
    if not external or reference is None:
        return []
    by_source = [{tuple(g["source_zyx"]): g for g in r["evaluation"]["test"]["per_source_group"]} for r in external]
    rows = []
    for group in reference["evaluation"]["test"]["per_source_group"]:
        key = tuple(group["source_zyx"])
        if any(key not in groups or sorted(groups[key]["case_indices"]) != sorted(group["case_indices"]) for groups in by_source):
            raise ValueError("Per-source comparison requires exactly matching held-out groups")
        values = [groups[key]["mae_ms"] for groups in by_source]
        avg = float(np.mean(values))
        rows.append({"source_z": key[0], "source_y": key[1], "source_x": key[2],
                     "case_indices": json.dumps(group["case_indices"]), "external_method": selected,
                     "eikonal_mae_ms": group["mae_ms"], "external_seed_mean_mae_ms": avg,
                     "external_initialization_sd_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                     "external_minus_eikonal_mae_ms": avg - group["mae_ms"]})
    return rows


def _plot_sources(plt, root, rows, selected):
    if not rows:
        return
    x, y = [r["eikonal_mae_ms"] for r in rows], [r["external_seed_mean_mae_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.9, 5.5), layout="constrained")
    scatter = ax.scatter(x, y, c=[r["source_z"] * .8 for r in rows], cmap="viridis", s=30, alpha=.85)
    lo, hi = min(0, min(x), min(y)), max(max(x), max(y)) * 1.04
    ax.plot([lo, hi], [lo, hi], color="#666666", linestyle="--", linewidth=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Anisotropic eikonal source-group MAE (ms)",
           ylabel=LABELS.get(selected, selected) + " source-group MAE (ms)",
           title="Held-out source groups at N=16")
    fig.colorbar(scatter, ax=ax, label="Source depth (mm)", shrink=.8)
    fig.supxlabel("External method selected by validation only; displayed error averaged over fixed seeds.", fontsize=8)
    _save(plt, fig, root, "per_source_comparison")


def _plot_qualitative(plt, root, manifest=None):
    # Generated figures must not outlive the evidence that authorizes them.
    # They are rebuilt below only after the current metadata/arrays validate.
    for suffix in ("png", "pdf"):
        _safe_child(root, "common_heldout_case." + suffix).unlink(missing_ok=True)
    path = root.parent / "qualitative/COMPARISON.npz"
    if not path.exists():
        return False
    info_path = path.with_suffix(".json")
    if not info_path.exists():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("status") != "complete":
        # A stale NPZ can survive a later pending/failed selection. Its presence
        # alone must never turn that selection into a publishable figure.
        return False
    if "array_sha256" in info and info["array_sha256"] != file_hash(path):
        raise ValueError("Qualitative array checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        truth, ours, external = [np.array(arrays[k], dtype=float) for k in ("ground_truth_zyx", "eikonal_zyx", "external_zyx")]
        source = np.asarray(arrays["source_zyx"], dtype=int)
        spacing = np.asarray(arrays["spacing_xyz_mm"], dtype=float)
        case = int(np.asarray(arrays["case_index"]).item())
    if truth.ndim != 3 or truth.shape != ours.shape or truth.shape != external.shape or not all(np.isfinite(a).all() for a in (truth, ours, external)):
        raise ValueError("Qualitative arrays must be matching finite ZYX volumes")
    if source.shape != (3,) or (source < 0).any() or (source >= np.asarray(truth.shape)).any():
        raise ValueError("Qualitative source must be an in-bounds ZYX coordinate")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Qualitative spacing must contain three positive finite XYZ values")
    if "case_index" in info and info["case_index"] != case:
        raise ValueError("Qualitative metadata and array case identities differ")
    selected = info.get("method", info.get("external_method", "external baseline"))
    z_slices = sorted({0, truth.shape[0] // 2, truth.shape[0] - 1, int(source[0])})
    volumes = [truth, ours, external, np.abs(ours - truth), np.abs(external - truth)]
    labels = ["Monodomain target", "Anisotropic eikonal", LABELS.get(selected, selected), "Eikonal absolute error", "External absolute error"]
    time_min = min(a.min() for a in volumes[:3])
    time_max = max(a.max() for a in volumes[:3])
    err_max = max(a.max() for a in volumes[3:]) or 1.
    extent = [0., (truth.shape[2] - 1) * spacing[0], 0., (truth.shape[1] - 1) * spacing[1]]
    fig, axes = plt.subplots(len(z_slices), 5, figsize=(15.5, 3.0 * len(z_slices)), squeeze=False, layout="constrained")
    for row, z in enumerate(z_slices):
        for col, volume in enumerate(volumes):
            ax = axes[row, col]
            im = ax.imshow(volume[z], origin="lower", extent=extent, cmap="viridis" if col < 3 else "magma",
                           vmin=time_min if col < 3 else 0., vmax=time_max if col < 3 else err_max,
                           interpolation="nearest", aspect="equal")
            if z == source[0]:
                ax.plot(source[2] * spacing[0], source[1] * spacing[1], marker="+", color="white", markersize=8, markeredgewidth=1.2)
            if row == 0:
                ax.set_title(labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel("z = %.1f mm\ny (mm)" % (z * spacing[2]))
            if row == len(z_slices) - 1:
                ax.set_xlabel("x (mm)")
            if col == 2 and row == 0:
                time_image = im
            if col == 4 and row == 0:
                error_image = im
    fig.colorbar(time_image, ax=axes[:, :3].ravel().tolist(), location="bottom", label="Absolute activation time (ms)", shrink=.65, pad=.03)
    fig.colorbar(error_image, ax=axes[:, 3:].ravel().tolist(), location="bottom", label="Absolute error (ms)", shrink=.7, pad=.03)
    seed = info.get("seed", info.get("initialization_seed", "fixed"))
    train_size = info.get("train_size", "unrecorded")
    smoke = bool(info.get("smoke", False) or (manifest or {}).get("config", {}).get("smoke", False))
    status = "SMOKE VERIFICATION ONLY — no scientific conclusions" if smoke else "Scientific run — descriptive held-out visualization"
    fig.suptitle("%s\nFrozen test case %d · source ZYX %s · %s · N=%s training source groups · seed %s\nIdentical physical slices and global scales; no target-based alignment" %
                 (status, case, source.tolist(), LABELS.get(selected, selected), train_size, seed), fontsize=12)
    _save(plt, fig, root, "common_heldout_case")
    return True


def _summary(root, manifest, states, summaries, selected, rows, errors):
    smoke = manifest.get("config", {}).get("smoke", False)
    complete = sum(v == "complete" for v in states.values())
    lines = ["# External cardiac activation baselines", "",
             "**Smoke verification only: these limited runs are not scientific performance results.**" if smoke else
             "All measurements below come from saved runs using the frozen source-group protocol.", "",
             "%d/%d baseline runs complete; %d pending; %d failed; %d invalid." %
             (complete, len(states), sum(v == "pending" for v in states.values()), sum(v == "failed" for v in states.values()), sum(v == "invalid" for v in states.values())), "",
             "## Available measured results", "",
             "Seeds are averaged within each training subset, then subsets receive equal weight. Between-subset SD is descriptive; overlapping subsets are not independent replicates. Initialization SD is saved separately in `aggregate/comparison_summary.csv`.", "",
             "| Method | N | Complete subsets | Test MAE, mean ± subset SD (ms) | Test RMSE (ms) | Validation MAE (ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["test_mae_ms_mean"] is None:
            continue
        lines.append("| %s | %d | %d/%d | %.6f ± %.6f | %.6f | %.6f |" %
                     (LABELS.get(row["method"], row["method"]), row["train_size"], row["completed_subsets"], row["expected_subsets"],
                      row["test_mae_ms_mean"], row["test_mae_ms_subset_std"], row["test_rmse_ms_mean"], row["validation_mae_ms_mean"]))
    lines += ["", "## Interpretation and limitations", ""]
    if selected:
        lines.append("The selected external method is **%s**, chosen by the mean N=16 validation MAE over every predetermined initialization seed. Test error did not choose the method or the displayed seed." % LABELS.get(selected, selected))
        external = next((r for r in summaries if r["method"] == selected and r["train_size"] == 16), None)
        ours = next((r for r in summaries if r["method"] == "eikonal_diagonal" and r["train_size"] == 16), None)
        if external and ours and not smoke:
            delta = external["test_mae_ms_mean"] - ours["test_mae_ms_mean"]
            lines.append("At N=16, the external minus anisotropic-eikonal test MAE is **%+.6f ms** (%s has the lower observed MAE). This is an observed difference, not a significance claim." %
                         (delta, "eikonal" if delta > 0 else LABELS.get(selected, selected) if delta < 0 else "neither method"))
    else:
        lines.append("External-method selection is pending until every planned N=16 method and initialization has a valid result. No winner is inferred from incomplete coverage.")
    lines += ["", "These experiments address data efficiency with 2–16 source groups. Poor performance here would not establish that a baseline is generally poor. Learned effective propagation speeds are not physiological conductivities. Any eikonal advantage concerns this frozen low-data task; physical structure, parameter interpretation and numerical error remain separate questions.", "",
              "Activation errors compare predictions with original monodomain labels in milliseconds. They are not numerical error against an exact homogeneous eikonal solution. No test volume is aligned using its own target. The eikonal reference uses its saved fixed training-only latency; baseline absolute-time conventions are documented in `BASELINE_RESEARCH.md` and fitted-run metadata.", "",
              "Training and inference timings, model sizes, and all seed/subset measurements are in `aggregate/comparison.csv`. Eikonal saved evaluation timings include additional diagnostics and are not directly equivalent to baseline-only forward timings. Validation is used for reasonable hyperparameter/early-stopping selection, so validation errors are selection statistics, not held-out generalization estimates.", "",
              "The common qualitative case is fixed using source geometry, with a predetermined seed realization. Images are descriptive; global activation/error scales and identical z slices are used.", "",
              "## Pending or failed work", ""]
    incomplete = [(key, value) for key, value in states.items() if value != "complete"]
    lines.extend(["- `%s`: %s" % pair for pair in incomplete] or ["All planned baseline runs have valid saved results."])
    if errors:
        lines += ["", "## Reporting issues", ""] + ["- " + e for e in errors]
    _atomic_text(root / "SUMMARY.md", "\n".join(lines) + "\n")
    slides = ["# Presentation results", ""]
    if smoke or not selected:
        slides += ["Scientific comparison is pending; smoke and incomplete results should not be presented as evidence.", ""]
    else:
        slides += ["- The comparison uses the same predetermined N=2, 4, 8 and 16 source-group subsets, with full held-out activation volumes.",
                   "- The external method used for detailed comparison is %s, selected using complete N=16 validation results only." % LABELS.get(selected, selected)]
        if external and ours:
            slides.append("- At N=16, anisotropic eikonal test MAE is %.6f ms and %s is %.6f ms (mean over predetermined initialization seeds)." %
                          (ours["test_mae_ms_mean"], LABELS.get(selected, selected), external["test_mae_ms_mean"]))
        slides += ["- Conclusions concern this extreme few-shot regime; error bars do not imply statistical significance.", ""]
    slides += ["Recommended plots: `plots/test_mae_vs_training_sources.png`, `plots/n16_method_comparison.png`, `plots/per_source_comparison.png`, and `plots/common_heldout_case.png` when available."]
    _atomic_text(root / "PRESENTATION_RESULTS.md", "\n".join(slides) + "\n")


def aggregate(output_dir):
    """Regenerate machine-readable summaries and publication PNG/PDF figures."""
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    records, states = read_results(root, manifest)
    references, errors = _reference_records(root, manifest)
    rows = [_flatten(r) for r in records.values()]
    rows += [_flatten(r, ref["method"], "eikonal_snapshot") for ref, r in references]
    rows.sort(key=lambda r: (r["method"], r["train_size"], r["subset_index"], str(r["initialization_seed"])))
    summaries = summarize_rows(rows, manifest)
    selected = select_strongest(manifest, records)
    _csv(root / "aggregate/comparison.csv", rows)
    _csv(root / "aggregate/comparison_summary.csv", summaries)
    _json(root / "aggregate/comparison.json", {"runs": rows, "summary": summaries,
          "selection_policy": "smallest N=16 mean validation MAE across all planned seeds; all methods must complete; method-name ties",
          "selected_external": selected, "states": states})
    try:
        source_rows = _source_comparison(root, selected, records, references) if selected else []
        _csv(root / "aggregate/per_source_comparison.csv", source_rows,
             ("source_z", "source_y", "source_x", "eikonal_mae_ms", "external_seed_mean_mae_ms"))
    except Exception as error:
        source_rows = []
        errors.append("per-source aggregation: " + str(error))
    try:
        plt = _pyplot(root)
        plots = root / "plots"
        _plot_curves(plt, plots, summaries, manifest.get("config", {}).get("smoke", False))
        _plot_sources(plt, plots, source_rows, selected)
        _plot_qualitative(plt, plots, manifest)
    except Exception as error:
        errors.append("plotting: " + str(error))
    _summary(root, manifest, states, summaries, selected, rows, errors)
    outcome = {"completed": sum(v == "complete" for v in states.values()),
               "pending": sum(v == "pending" for v in states.values()),
               "failed": sum(v == "failed" for v in states.values()),
               "invalid": sum(v == "invalid" for v in states.values()),
               "selected_external": selected, "reporting_errors": errors}
    _json(root / "aggregate/STATUS.json", outcome)
    return outcome
