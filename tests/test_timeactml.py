"""Small generated fixtures test data semantics without requiring a download."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from eikonal.three_d.analytic import analytical_arrival
from experiments.cardiac.timeactml import (
    TimeActML, load_timeactml, prepare_split, sample_training_voxels, training_latency,
)
from experiments.cardiac.analytical_baseline import evaluate_predictions, fit_analytical


class TimeActMLTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def _mat(self, source_xyz=((1, 2, 0), (3, 0, 1))):
        shape = (len(source_xyz), 4, 3, 2)
        activation = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 0.25
        stimuli = np.zeros(shape, dtype=np.uint8)
        for i, source in enumerate(source_xyz):
            stimuli[(i,) + tuple(source)] = 1
        path = self.root / "fixture.mat"
        savemat(path, {"actis": activation, "iapps": stimuli})
        return path, activation, stimuli

    def _dataset(self):
        sources = np.array([(0, 0, 0), (0, 0, 0), (1, 1, 1), (0, 2, 3), (1, 0, 3)], dtype=np.int64)
        volumes = np.stack([analytical_arrival((2, 3, 4), s, (0.6, 0.4, 0.3)) + 1.2 for s in sources])
        return TimeActML(volumes, sources, self.root / "raw.mat")

    def _manifests(self, dataset, partitions):
        root = self.root / "poc"
        root.mkdir(exist_ok=True)
        for split, ids in partitions.items():
            rows = []
            for index in ids:
                z, y, x = dataset.sources_nzyx[index]
                rows.append({"case_index": index, "source_x": int(x), "source_y": int(y), "source_z": int(z)})
            document = {"split": split, "case_index_base": 0, "source_index_base": 0,
                        "spacing_mm": [0.8, 0.8, 0.8], "cases": rows}
            (root / f"{split}.json").write_text(json.dumps(document), encoding="utf-8")
        return root

    def test_axis_mapping_and_immutable_targets(self):
        path, raw, _ = self._mat()
        dataset = load_timeactml(path, expected_shape=raw.shape)
        self.assertEqual(dataset.shape_zyx, (2, 3, 4))
        np.testing.assert_array_equal(dataset.sources_nzyx, [(0, 2, 1), (1, 0, 3)])
        self.assertEqual(dataset.activation_nzyx[0, 1, 2, 3], raw[0, 3, 2, 1])
        self.assertFalse(dataset.activation_nzyx.flags.writeable)
        self.assertEqual(dataset.spacing_xyz, (0.8, 0.8, 0.8))

    def test_shape_and_nonfinite_rejected(self):
        path, raw, stimuli = self._mat()
        with self.assertRaisesRegex(ValueError, "raw NXYZ shape"):
            load_timeactml(path)
        raw[0, 0, 0, 0] = np.nan
        savemat(path, {"actis": raw, "iapps": stimuli})
        with self.assertRaisesRegex(ValueError, "finite"):
            load_timeactml(path, expected_shape=raw.shape)

    def test_missing_or_multiple_source_rejected(self):
        path, raw, stimuli = self._mat()
        for count in (0, 2):
            stimuli[0] = 0
            stimuli[0, 0, 0, :count] = 1
            savemat(path, {"actis": raw, "iapps": stimuli})
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_timeactml(path, expected_shape=raw.shape)

    def test_manifest_source_group_leakage_rejected(self):
        dataset = self._dataset()
        root = self._manifests(dataset, {"train": [0], "validation": [1], "test": [2]})
        with self.assertRaisesRegex(ValueError, "leakage"):
            prepare_split(dataset, root)

    def test_manifest_retains_all_group_members_and_order(self):
        dataset = self._dataset()
        root = self._manifests(dataset, {"train": [1, 0], "validation": [2], "test": [4, 3]})
        split = prepare_split(dataset, root)
        self.assertEqual(split["train"], [1, 0])
        self.assertEqual(split["test"], [4, 3])
        self.assertEqual(split["group_counts"]["train"], {"manifest_case_count": 2, "distinct_source_count": 1})
        document = json.loads((root / "test.json").read_text())
        document["cases"][0]["source_x"] = 0
        (root / "test.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "disagrees with iapps"):
            prepare_split(dataset, root)

    def test_missing_frozen_manifests_rejected(self):
        dataset = self._dataset()
        with self.assertRaisesRegex(FileNotFoundError, "Frozen source-grouped"):
            prepare_split(dataset)

    def test_latency_training_only_and_evaluation_no_test_alignment(self):
        dataset = self._dataset()
        latency = training_latency(dataset, [0, 2])
        self.assertAlmostEqual(latency, 1.2)
        changed = dataset.activation_nzyx.copy()
        changed[4] += 100.0
        shifted = TimeActML(changed, dataset.sources_nzyx, dataset.path)
        self.assertEqual(training_latency(shifted, [0, 2]), latency)
        metrics = evaluate_predictions(shifted, [4],
                    lambda s: analytical_arrival(dataset.shape_zyx, s, (0.6, 0.4, 0.3)), latency)
        self.assertAlmostEqual(metrics["mae_ms"], 100.0)
        self.assertAlmostEqual(metrics["rmse_ms"], 100.0)

    def test_analytical_fit_recovers_shared_speeds(self):
        dataset = self._dataset()
        result = fit_analytical(dataset, [0, 2, 3], model="diagonal", samples_per_case=24)
        np.testing.assert_allclose(result["speeds_xyz_mm_per_ms"], (0.6, 0.4, 0.3), atol=1e-8)
        self.assertLess(result["training_sample_mse_ms2"], 1e-16)
        indices = sample_training_voxels(dataset, [0, 2], 5)
        rng = np.random.default_rng(20260905)
        for actual in indices:
            np.testing.assert_array_equal(actual, rng.choice(24, 5, replace=False))
        full_indices = sample_training_voxels(dataset, [0, 2], 0)
        for actual in full_indices:
            np.testing.assert_array_equal(actual, np.arange(24))

    def test_full_volume_isotropic_fit_recovers_shared_speed(self):
        dataset = self._dataset()
        volumes = np.stack([analytical_arrival(dataset.shape_zyx, s, 0.51) + 1.2 for s in dataset.sources_nzyx])
        isotropic = TimeActML(volumes, dataset.sources_nzyx, dataset.path)
        result = fit_analytical(isotropic, [0, 2, 3], model="isotropic")
        np.testing.assert_allclose(result["speeds_xyz_mm_per_ms"], [0.51] * 3, atol=1e-12)
        self.assertEqual(result["samples_per_case"], 24)
        self.assertEqual(result["fit_protocol"], "canonical full-volume")
        self.assertLess(result["training_sample_mse_ms2"], 1e-24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
