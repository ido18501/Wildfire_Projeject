"""Tiny FNO device/checkpoint smoke; no MAT, KOL, or scientific suite run."""
import argparse
import json
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from . import fno


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: FNO GPU smoke was NOT run")
    torch.set_num_threads(1)
    coordinates = np.moveaxis(np.indices((3, 5, 7)), 0, -1)
    sources = np.array([[0, 0, 0], [2, 4, 6], [1, 2, 3]])
    targets = np.stack([np.linalg.norm(coordinates - source, axis=-1) + 1. for source in sources])
    config = dict(device=args.device, epochs=2, min_epochs=0, validation_every=1,
                  checkpoint_every=1, width=2, layers=2, modes_xyz=[2, 2, 2])
    checked = []
    original_loss = fno._relative_l2

    def checked_loss(prediction, target):
        assert prediction.device.type == target.device.type == args.device
        checked.append(target.device.type)
        return original_loss(prediction, target)

    with tempfile.TemporaryDirectory() as directory, patch.object(fno, "_relative_l2", checked_loss):
        predictor, info = fno.fit_fno(sources[:2], targets[:2], sources[2:], targets[2:],
                                       config=config, seed=20260907, artifact_dir=directory)
        assert next(predictor.model.parameters()).device.type == args.device
        inputs_seen = []
        hook = predictor.model.register_forward_pre_hook(lambda model, inputs: inputs_seen.append(inputs[0].device.type))
        prediction = predictor.predict(sources[2:])
        hook.remove()
        assert inputs_seen and all(d == args.device for d in inputs_seen)
        assert checked and info["device"] == args.device and np.isfinite(prediction).all()
        portable = fno.load_predictor(directory, device="cpu")
        assert next(portable.model.parameters()).device.type == "cpu"
        np.testing.assert_allclose(portable.predict(sources[2:]), prediction, rtol=1e-4, atol=1e-5)
    print(json.dumps({"passed": True, "device": args.device, "cuda_used": args.device == "cuda",
                      "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                      "training_and_inference_device_checked": True, "checkpoint_reload_on_cpu": True}))


if __name__ == "__main__":
    main()
