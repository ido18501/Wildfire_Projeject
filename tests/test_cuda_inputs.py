"""Server input-path override preserves the exact frozen experiment identity."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from external_baselines import runner
from external_baselines.protocol import ROOT, inspect_eikonal_suite


class ServerInputTests(unittest.TestCase):
    def test_server_paths_preserve_hashes_and_subsets_without_fitting(self):
        links = json.loads((ROOT / "PROJECT_LINKS.json").read_text())
        original, raw, subsets, references = inspect_eikonal_suite(links["eikonal_suite"])
        with tempfile.TemporaryDirectory() as temporary:
            server = Path(temporary) / "project"
            frozen = server / "local_data/timeactml/poc"
            frozen.mkdir(parents=True)
            for part, record in original["frozen_manifests"].items():
                shutil.copyfile(record["path"], frozen / (part + ".json"))
            args = argparse.Namespace(eikonal_suite=server / "runs_3d/frozen_suite", data=None,
                                      output_root=Path(temporary) / "baselines", seed=20260907,
                                      repeats=3, methods=["kol", "fno"], smoke=True, device="cuda",
                                      threads=1, epochs=4000, patience=250, batch_size=20)
            count = 1 + max(i for group in original["split"].values() if isinstance(group, list) for i in group)
            dummy = SimpleNamespace(shape_zyx=(9, 49, 49), spacing_xyz=(.8, .8, .8),
                                    sources_nzyx=np.zeros((count, 3), dtype=np.int64))

            def check_paths(manifest):
                for part, record in manifest["frozen_manifests"].items():
                    self.assertEqual(Path(record["path"]), frozen / (part + ".json"))
                    self.assertEqual(record["sha256"], original["frozen_manifests"][part]["sha256"])
                self.assertTrue(Path(manifest["data"]["path"]).is_relative_to(server))
                self.assertEqual(manifest["split"], original["split"])
                return dummy

            with patch.object(runner, "inspect_eikonal_suite", return_value=(deepcopy(original), raw, subsets, references)), \
                 patch.object(runner, "load_dataset", side_effect=check_paths):
                _, manifest, _ = runner.snapshot_suite(args)
            self.assertEqual(manifest["subsets"], subsets)
            self.assertEqual(manifest["source_eikonal_manifest_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest["source_project"], str(server))


if __name__ == "__main__":
    unittest.main()
