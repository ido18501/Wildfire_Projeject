"""Read-only timeactML loading and source-grouped experimental partitions.

Raw MAT axes are case, X, Y, Z; all exposed volumes are case, Z, Y, X.
Physical spacing is XYZ in mm and activation is in ms. No map is shifted by
its minimum: the sole shared latency is determined from training sources.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class TimeActML:
    activation_nzyx: np.ndarray
    sources_nzyx: np.ndarray
    path: Path
    spacing_xyz: tuple = (0.8, 0.8, 0.8)

    @property
    def shape_zyx(self):
        return self.activation_nzyx.shape[1:]


def load_timeactml(path=None, expected_shape=(1000, 49, 49, 9)):
    """Load and validate the structured-v1 MAT without modifying downloaded data.

    ``expected_shape`` overrides the exact raw shape only to permit tiny test
    fixtures; normal experiments assert the verified 1000x49x49x9 dataset.
    """
    if path is None:
        matches = sorted(Path("local_data").rglob("dataset_acti_3d_v1.mat"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one dataset_acti_3d_v1.mat under local_data, found {len(matches)}")
        path = matches[0]
    path = Path(path).resolve()
    content = loadmat(path, variable_names=["actis", "iapps"])
    if "actis" not in content or "iapps" not in content:
        raise ValueError("MAT must contain actis and iapps")
    activation, stimulus = content["actis"], content["iapps"]
    if activation.shape != tuple(expected_shape) or stimulus.shape != tuple(expected_shape):
        raise ValueError(f"Expected actis/iapps raw NXYZ shape {expected_shape}; got {activation.shape}/{stimulus.shape}")
    if not np.issubdtype(activation.dtype, np.number) or np.iscomplexobj(activation):
        raise ValueError("actis must contain real numeric activation times")
    if not np.issubdtype(stimulus.dtype, np.number) or np.iscomplexobj(stimulus):
        raise ValueError("iapps must contain real numeric stimulus values")
    if not np.isfinite(activation).all() or not np.isfinite(stimulus).all():
        raise ValueError("All activation and stimulus values must be finite")
    nonzero = stimulus != 0
    counts = nonzero.reshape(len(stimulus), -1).sum(axis=1)
    if not np.all(counts == 1):
        bad = np.flatnonzero(counts != 1)
        raise ValueError(f"Each case must have exactly one nonzero source voxel; invalid cases {bad[:10].tolist()}")
    flat_sources = nonzero.reshape(len(stimulus), -1).argmax(axis=1)
    sources_xyz = np.column_stack(np.unravel_index(flat_sources, stimulus.shape[1:]))
    sources = np.ascontiguousarray(sources_xyz[:, ::-1], dtype=np.int64)
    volumes = np.ascontiguousarray(activation.transpose(0, 3, 2, 1), dtype=np.float64)
    volumes.setflags(write=False)
    sources.setflags(write=False)
    return TimeActML(volumes, sources, path)


def _read_manifest_file(path, dataset, split):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("case_index_base") != 0 or document.get("source_index_base") != 0:
        raise ValueError("Manifests must explicitly declare zero-based case and source indices")
    if document.get("split") != split:
        raise ValueError(f"Manifest split mismatch for {path}")
    if "spacing_mm" in document and not np.allclose(document["spacing_mm"], dataset.spacing_xyz):
        raise ValueError("Manifest physical spacing disagrees with dataset")
    if "raw_array_axes" in document and document["raw_array_axes"] != ["case", "x", "y", "z"]:
        raise ValueError("Manifest must describe raw case,X,Y,Z axes")
    ids = []
    for row in document["cases"]:
        index = row["case_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(dataset.sources_nzyx):
            raise ValueError("Manifest case index must be an in-bounds integer")
        declared = (row["source_z"], row["source_y"], row["source_x"])
        if tuple(dataset.sources_nzyx[index]) != declared:
            raise ValueError(f"Manifest source disagrees with iapps for case {index}")
        if row.get("split", split) != split:
            raise ValueError("Manifest row split mismatch")
        ids.append(index)
    if "case_count" in document and document["case_count"] != len(ids):
        raise ValueError("Manifest case_count disagrees with its case rows")
    distinct_count = len({tuple(dataset.sources_nzyx[i]) for i in ids})
    if "unique_source_count" in document and document["unique_source_count"] != distinct_count:
        raise ValueError("Manifest unique_source_count disagrees with its source rows")
    return ids


def _validate_partitions(dataset, partitions):
    seen_sources, seen_ids, result, counts = {}, set(), {}, {}
    for name in ("train", "validation", "test"):
        ids = partitions[name]
        if not ids:
            raise ValueError(f"Empty {name} partition")
        local_sources = set()
        for index in ids:
            if index in seen_ids:
                raise ValueError(f"Repeated case index {index} in manifests")
            if not 0 <= index < len(dataset.sources_nzyx):
                raise ValueError("Out-of-bounds split index")
            seen_ids.add(index)
            source = tuple(int(s) for s in dataset.sources_nzyx[index])
            if source in seen_sources and seen_sources[source] != name:
                raise ValueError(f"Source-location leakage between {seen_sources[source]} and {name}: {source}")
            seen_sources[source] = name
            local_sources.add(source)
        result[name] = [int(index) for index in ids]
        counts[name] = {"manifest_case_count": len(ids), "distinct_source_count": len(local_sources)}
    result["group_counts"] = counts
    result["case_policy"] = "All frozen manifest cases in original order, including repeated sources; equal case/voxel weight"
    return result


def prepare_split(dataset, manifest=None, seed=20260905):
    """Require and consume the frozen source-grouped package manifests.

    Manifest is a directory holding train.json, validation.json and test.json.
    Downloaded partitions are never changed or regenerated. All original cases
    are retained, including repeated sources: canonical counts are 16/18/102
    cases for 16/16/100 distinct source groups. ``seed`` is retained for API
    compatibility and never changes or replaces a frozen partition.
    """
    if manifest is None:
        candidates = [parent / "poc" for parent in dataset.path.parents]
        manifest = next((p for p in candidates if (p / "train.json").is_file()), None)
    if manifest is not None:
        root = Path(manifest)
        if root.is_file():
            root = root.parent
        if not (root / "train.json").is_file() and (root / "poc" / "train.json").is_file():
            root = root / "poc"
        partitions = {name: _read_manifest_file(root / f"{name}.json", dataset, name)
                      for name in ("train", "validation", "test")}
        result = _validate_partitions(dataset, partitions)
        result["provenance"] = {"kind": "data_agent_source_grouped_manifests", "directory": str(root.resolve())}
        return result
    raise FileNotFoundError("Frozen source-grouped train/validation/test JSON manifests are required; pass --manifest or use local_data/timeactml/poc")


def training_latency(dataset, train_ids):
    """One shared latency in ms, using only selected training source activations."""
    if not len(train_ids):
        raise ValueError("Cannot determine latency without training cases")
    values = [dataset.activation_nzyx[int(i)][tuple(dataset.sources_nzyx[int(i)])] for i in train_ids]
    return float(np.mean(values))


def sample_training_voxels(dataset, train_ids, samples_per_case=0, seed=20260905):
    """C-ravel ZYX indices: zero means all voxels, otherwise frozen samples."""
    if samples_per_case < 0:
        raise ValueError("samples_per_case must be nonnegative; zero uses all voxels")
    rng = np.random.default_rng(seed)
    size = int(np.prod(dataset.shape_zyx))
    if samples_per_case == 0:
        return [np.arange(size, dtype=np.int64) for _ in train_ids]
    return [rng.choice(size, min(samples_per_case, size), replace=False) for _ in train_ids]
