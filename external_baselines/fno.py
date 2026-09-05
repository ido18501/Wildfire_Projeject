"""Frozen-split adapter for the isolated, pinned timeactML FNO implementation.

This file is original adapter code. The upstream architecture is read from an
isolated audit snapshot whose license is unresolved; it is not redistributed as
part of this package. No pretrained upstream weights or upstream split loader
are used. Inputs and outputs at this boundary use the project's Z,Y,X order.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


UPSTREAM_COMMIT = "b765384703c691e92918fe449bcf809b9e82564d"
ARCHITECTURE_SHA256 = "d165cd1f66aff944327e78ee23e486df514450fe5d0317a633ba986f4dd820ba"
DEFAULTS = {
    "device": "cpu", "epochs": 4000, "patience": 250, "min_epochs": 200,
    "validation_every": 10, "checkpoint_every": 50, "batch_size": 20,
    "inference_batch_size": 1, "lr": 0.001, "weight_decay": 0.0001,
    "scheduler_patience_checks": 25, "scheduler_factor": 0.95,
    "scheduler_min_lr": 1e-6, "width": 32, "modes_xyz": [16, 8, 4],
    "layers": 4, "deterministic": True,
}


def _upstream_path(config):
    default = Path(__file__).resolve().parents[1] / "external_sources" / "timeactML"
    upstream = Path(config.get("upstream_dir", default)).resolve()
    return upstream / "FNO" / "FNO_3D" / "fno.py"


def _architecture(config):
    source = _upstream_path(config)
    upstream_root = source.parents[2]
    if (upstream_root / ".git").exists():
        commit = subprocess.check_output(["git", "-c", f"safe.directory={upstream_root.as_posix()}",
                                          "-C", str(upstream_root), "rev-parse", "HEAD"], text=True).strip()
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"Upstream FNO checkout commit differs from pinned {UPSTREAM_COMMIT}: {commit}")
    # Git's Windows checkout may convert LF to CRLF; audit canonical LF bytes.
    digest = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != ARCHITECTURE_SHA256:
        raise ValueError(f"Upstream FNO source hash mismatch: {source}")
    spec = importlib.util.spec_from_file_location("isolated_timeactml_fno", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FNO3d(*config["modes_xyz"], config["width"], config["layers"])


def _device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch installation has no available CUDA device")
    return torch.device(name)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_inputs(sources_zyx, shape_zyx):
    """The official four channels: binary pulse and per-axis unit-cube XYZ."""
    shape = tuple(int(x) for x in shape_zyx)
    sources = np.asarray(sources_zyx)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("Source array must be N×3 in Z,Y,X order")
    if not np.all(np.isfinite(sources)) or not np.all(sources == np.round(sources)):
        raise ValueError("Sources must be finite integer grid indices")
    sources = sources.astype(np.int64)
    if np.any(sources < 0) or np.any(sources >= np.asarray(shape)):
        raise ValueError("Source index is outside the observation grid")
    z, y, x = shape
    grid = np.stack(np.meshgrid(np.linspace(0, 1, x, dtype=np.float32),
                               np.linspace(0, 1, y, dtype=np.float32),
                               np.linspace(0, 1, z, dtype=np.float32), indexing="ij"), axis=-1)
    inputs = np.zeros((len(sources), x, y, z, 4), dtype=np.float32)
    inputs[..., 1:] = grid
    for row, (sz, sy, sx) in enumerate(sources):
        inputs[row, sx, sy, sz, 0] = 1.0
    return torch.from_numpy(inputs)


def _targets(targets):
    result = np.asarray(targets, dtype=np.float32)
    if result.ndim != 4 or not np.all(np.isfinite(result)):
        raise ValueError("Activation targets must be finite N×Z×Y×X arrays in milliseconds")
    return torch.from_numpy(np.ascontiguousarray(result.transpose(0, 3, 2, 1)[..., None]))


def _relative_l2(prediction, target):
    denominator = target.flatten(1).norm(dim=1)
    if torch.any(denominator <= 0):
        raise ValueError("Relative L2 needs nonzero activation target norm")
    return (prediction - target).flatten(1).norm(dim=1) / denominator


def _atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load(path):
    # Only checkpoints generated locally by this adapter are accepted here.
    return torch.load(path, map_location="cpu", weights_only=True)


class FNOPredictor:
    def __init__(self, model, config, shape_zyx, device):
        self.model = model.to(device).eval()
        self.config = config
        self.shape_zyx = tuple(shape_zyx)
        self.device = device

    def predict(self, sources_zyx, batch_size=None):
        inputs = source_inputs(sources_zyx, self.shape_zyx)
        batch_size = int(batch_size or self.config["inference_batch_size"])
        if batch_size < 1:
            raise ValueError("Inference batch size must be positive")
        outputs = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                output = self.model(inputs[start:start + batch_size].to(self.device))
                outputs.append(output[..., 0].permute(0, 3, 2, 1).cpu().numpy())
        if not outputs:
            return np.empty((0, *self.shape_zyx), dtype=np.float32)
        result = np.concatenate(outputs)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("FNO produced nonfinite activation predictions")
        return result


def load_predictor(artifact_dir, *, device="cpu", upstream_dir=None):
    """Reload our fitted best checkpoint, never upstream pretrained weights."""
    saved = _load(Path(artifact_dir) / "model.pt")
    config = saved["config"].copy()
    if upstream_dir is not None:
        config["upstream_dir"] = str(upstream_dir)
    model = _architecture(config)
    model.load_state_dict(saved["model"])
    return FNOPredictor(model, config, saved["shape_zyx"], _device(device))


def fit_fno(train_sources_zyx, train_targets_zyx, val_sources_zyx, val_targets_zyx,
            *, config, seed, artifact_dir, smoke=False):
    """Fit only supplied training cases; select checkpoints only on validation.

    Returns ``(FNOPredictor, metadata)``. A matching ``last_checkpoint.pt``
    resumes optimizer, scheduler, deterministic shuffle state and elapsed time.
    The caller retains responsibility for the frozen source-group protocol.
    """
    options = dict(DEFAULTS)
    options.update(config or {})
    options["modes_xyz"] = list(options["modes_xyz"])
    options["upstream_dir"] = str(_upstream_path(options).parents[2])
    if smoke:
        options.update(epochs=2, min_epochs=0, validation_every=1, checkpoint_every=1)
    integer_keys = ("epochs", "patience", "min_epochs", "validation_every", "checkpoint_every",
                    "batch_size", "inference_batch_size", "width", "layers")
    for key in integer_keys:
        options[key] = int(options[key])
    if any(options[key] < 1 for key in integer_keys if key != "min_epochs") or options["min_epochs"] < 0:
        raise ValueError("Invalid nonpositive FNO configuration")
    train_y, val_y = _targets(train_targets_zyx), _targets(val_targets_zyx)
    shape = tuple(np.asarray(train_targets_zyx).shape[1:])
    if len(train_y) < 1 or len(val_y) < 1 or tuple(np.asarray(val_targets_zyx).shape[1:]) != shape:
        raise ValueError("Training and validation require nonempty matching observation grids")
    train_x, val_x = source_inputs(train_sources_zyx, shape), source_inputs(val_sources_zyx, shape)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("Source and target case counts differ")
    signature = hashlib.sha256(json.dumps({"config": options, "seed": int(seed), "smoke": bool(smoke)}, sort_keys=True).encode())
    for array in (train_x, train_y, val_x, val_y):
        signature.update(array.numpy().tobytes())
    signature = signature.hexdigest()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = artifact_dir / "last_checkpoint.pt", artifact_dir / "model.pt"
    log_path = artifact_dir / "training_log.jsonl"
    device = _device(options["device"])
    if options["deterministic"]:
        # Needed before creating CUDA matrices for deterministic CUBLAS kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(bool(options["deterministic"]))
    try:
        model = _architecture(options).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["lr"]), weight_decay=float(options["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(options["scheduler_factor"]),
            patience=int(options["scheduler_patience_checks"]), threshold=0.001,
            threshold_mode="rel", min_lr=float(options["scheduler_min_lr"]))
        shuffle = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        start_epoch, best_epoch, best_val, previous_seconds = 0, 0, math.inf, 0.0
        final_train = math.inf
        resumed = False
        if last_path.exists():
            saved = _load(last_path)
            if saved["signature"] != signature:
                raise ValueError("FNO checkpoint configuration/data identity differs; use a new run directory")
            if not best_path.exists():
                raise ValueError("Last checkpoint exists but best checkpoint is missing")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            shuffle.set_state(saved["shuffle_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            start_epoch, best_epoch, best_val = saved["epoch"], saved["best_epoch"], saved["best_val"]
            previous_seconds, final_train = saved["training_seconds"], saved["train_relative_l2"]
            resumed = True
        _synchronize(device)
        began = time.perf_counter()
        epoch, reason, last_val = start_epoch, "maximum_epochs", best_val
        epoch_seconds = []
        effective_batch = min(options["batch_size"], len(train_x))

        def elapsed():
            _synchronize(device)
            return previous_seconds + time.perf_counter() - began

        def checkpoint(stop_reason=None):
            _atomic_save({"signature": signature, "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                          "best_val": best_val, "training_seconds": elapsed(), "train_relative_l2": final_train,
                          "shuffle_rng": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                          "stop_reason": stop_reason}, last_path)

        if start_epoch >= options["min_epochs"] and start_epoch - best_epoch >= options["patience"]:
            reason = "validation_early_stopping"
        else:
            for epoch in range(start_epoch + 1, options["epochs"] + 1):
                _synchronize(device)
                epoch_began = time.perf_counter()
                model.train()
                permutation = torch.randperm(len(train_x), generator=shuffle)
                total_loss = 0.0
                for start in range(0, len(permutation), effective_batch):
                    indices = permutation[start:start + effective_batch]
                    inputs, targets = train_x[indices].to(device), train_y[indices].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _relative_l2(model(inputs), targets).sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite FNO training loss at epoch {epoch}")
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                final_train = total_loss / len(train_y)
                validation_due = epoch == 1 or epoch % options["validation_every"] == 0 or epoch == options["epochs"]
                if validation_due:
                    model.eval()
                    total = 0.0
                    with torch.no_grad():
                        for start in range(0, len(val_x), options["inference_batch_size"]):
                            inputs = val_x[start:start + options["inference_batch_size"]].to(device)
                            targets = val_y[start:start + options["inference_batch_size"]].to(device)
                            total += float(_relative_l2(model(inputs), targets).sum().cpu())
                    last_val = total / len(val_y)
                    if not math.isfinite(last_val):
                        raise FloatingPointError(f"Nonfinite FNO validation loss at epoch {epoch}")
                    scheduler.step(last_val)
                    if last_val < best_val:
                        best_val, best_epoch = last_val, epoch
                        _atomic_save({"signature": signature, "model": _cpu_state(model), "config": options,
                                      "shape_zyx": list(shape), "seed": int(seed), "epoch": epoch,
                                      "validation_relative_l2": best_val, "upstream_commit": UPSTREAM_COMMIT,
                                      "architecture_sha256": ARCHITECTURE_SHA256}, best_path)
                _synchronize(device)
                epoch_seconds.append(time.perf_counter() - epoch_began)
                entry = {"epoch": epoch, "epoch_seconds": epoch_seconds[-1], "train_relative_l2": final_train,
                         "validation_relative_l2": last_val if validation_due else None,
                         "best_epoch": best_epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                         "training_seconds": elapsed()}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
                    handle.flush()
                if epoch == 1 or epoch % options["checkpoint_every"] == 0:
                    checkpoint()
                if validation_due and epoch >= options["min_epochs"] and epoch - best_epoch >= options["patience"]:
                    reason = "validation_early_stopping"
                    break
        checkpoint(reason)
        training_seconds = elapsed()
        best = _load(best_path)
        if best["signature"] != signature:
            raise ValueError("Best checkpoint identity does not match current frozen training data")
        model.load_state_dict(best["model"])
        model.eval()
        selected_train_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(train_x), options["inference_batch_size"]):
                targets = train_y[start:start + options["inference_batch_size"]].to(device)
                selected_train_loss += float(_relative_l2(model(train_x[start:start + options["inference_batch_size"]].to(device)), targets).sum().cpu())
        parameter_count = sum(p.numel() for p in model.parameters())
        scalar_count = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        metadata = {"training_seconds": training_seconds, "parameter_count": parameter_count,
                    "real_scalar_parameter_count": scalar_count, "size_bytes": size_bytes,
                    "epochs_completed": epoch, "best_epoch": best_epoch, "best_validation_relative_l2": best_val,
                    "train_objective": selected_train_loss / len(train_y), "objective_name": "mean_case_relative_L2",
                    "stop_reason": reason, "resumed_checkpoint": resumed, "seed": int(seed), "config": options,
                    "effective_batch_size": effective_batch, "device": str(device), "signature": signature,
                    "mean_epoch_seconds_this_invocation": float(np.mean(epoch_seconds)) if epoch_seconds else None,
                    "epoch_seconds_this_invocation": epoch_seconds,
                    "checkpoint": str(best_path.resolve()), "training_log": str(log_path.resolve()),
                    "upstream_commit": UPSTREAM_COMMIT, "upstream_architecture_sha256": ARCHITECTURE_SHA256,
                    "timing_convention": "Direct absolute activation milliseconds; no latency shift or target alignment",
                    "fidelity": "Exact pinned upstream FNO3d architecture/input; frozen grouped split replaces upstream80/20; validation-only scheduling/early-stop repairs inactive upstream plateau scheduler; no pretrained weights"}
        return FNOPredictor(model, options, shape, device), metadata
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
