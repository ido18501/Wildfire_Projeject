import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ExperimentConfig
from trainer import SceneTrainer
from component_manifest import load_scenarios_from_sample_manifest
from experiments.wildfire.sim2real_loader import Sim2RealFireLoader


def write_json(path, obj):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, tuple):
            return [conv(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), indent=2, allow_nan=True))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return

    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def summarize(name, scenarios):
    by_type = {}
    by_scene = {}
    for s in scenarios:
        by_type[s.get("sample_type", "")] = by_type.get(s.get("sample_type", ""), 0) + 1
        by_scene[s.get("scene_id", "")] = by_scene.get(s.get("scene_id", ""), 0) + 1

    print(f"{name}: n={len(scenarios)}")
    print(f"  by_type:  {by_type}")
    print(f"  by_scene: {by_scene}")


def read_data_root(args):
    if args.data_root:
        return str(Path(args.data_root).expanduser())

    manifest_path = Path(args.manifest).expanduser()
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            for k in ["data_root", "DATA_ROOT", "root", "dataset_root"]:
                if k in m:
                    return str(Path(m[k]).expanduser())
        except Exception:
            pass

    return "/home/shirb/sim2real_fire_data/simulation_data"


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state):
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def evaluate_all(trainer, scenarios, prefix, out_dir):
    band_agg, band_rows = trainer.evaluate_bands(scenarios)
    full_agg, full_rows, _ = trainer.evaluate_full_fire(scenarios, save_predictions=False)

    write_json(out_dir / f"{prefix}_band_metrics.json", band_agg)
    write_json(out_dir / f"{prefix}_full_metrics.json", full_agg)
    write_csv(out_dir / f"{prefix}_band_per_sample.csv", band_rows)
    write_csv(out_dir / f"{prefix}_full_per_sample.csv", full_rows)

    return {"band": band_agg, "full": full_agg}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--extended-val-manifest", required=True)
    ap.add_argument("--extended-test-manifest", required=True)

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--feature-group", default="baseline")

    ap.add_argument("--manifest", default="~/wildfire_small_subset_no014_manifest.json")
    ap.add_argument("--data-root", default=None)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--lambda-g", type=float, default=0.002)
    ap.add_argument("--lambda-b", type=float, default=0.002)

    ap.add_argument("--band-width", type=float, default=0.5)
    ap.add_argument("--band-stride", type=float, default=0.5)
    ap.add_argument("--band-max-iter", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=50)

    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--scheduler-patience", type=int, default=3)

    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--max-test-samples", type=int, default=None)
    ap.add_argument("--max-extended-val-samples", type=int, default=None)
    ap.add_argument("--max-extended-test-samples", type=int, default=None)

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = read_data_root(args)
    loader = Sim2RealFireLoader(data_root)

    cfg = ExperimentConfig(
        experiment_name=args.run_name,
        experiment_type="component_manifest_band_dual_fullcorr",
        feature_group=args.feature_group,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_G=args.lambda_g,
        lambda_B=args.lambda_b,
        band_width=args.band_width,
        band_stride=args.band_stride,
        band_max_iter=args.band_max_iter,
        max_iter=args.max_iter,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        device=args.device,
        output_dir=str(out_dir),
        seed=args.seed,
    )

    trainer = SceneTrainer(cfg, loader)

    print("=" * 120)
    print("COMPONENT MANIFEST BAND RUN: DUAL FULL-CORR VALIDATION")
    print("=" * 120)
    print("run_name:", args.run_name)
    print("feature_group:", args.feature_group)
    print("out_dir:", out_dir)
    print("data_root:", data_root)
    print()

    train_scenarios, _ = load_scenarios_from_sample_manifest(
        args.train_manifest, trainer, max_samples=args.max_train_samples, strict=True
    )
    val_scenarios, _ = load_scenarios_from_sample_manifest(
        args.val_manifest, trainer, max_samples=args.max_val_samples, strict=True
    )
    test_scenarios, _ = load_scenarios_from_sample_manifest(
        args.test_manifest, trainer, max_samples=args.max_test_samples, strict=True
    )
    ext_val_scenarios, _ = load_scenarios_from_sample_manifest(
        args.extended_val_manifest, trainer, max_samples=args.max_extended_val_samples, strict=True
    )
    ext_test_scenarios, _ = load_scenarios_from_sample_manifest(
        args.extended_test_manifest, trainer, max_samples=args.max_extended_test_samples, strict=True
    )

    summarize("train", train_scenarios)
    summarize("primary_val", val_scenarios)
    summarize("primary_test", test_scenarios)
    summarize("extended_val", ext_val_scenarios)
    summarize("extended_test", ext_test_scenarios)

    if not train_scenarios or not val_scenarios or not test_scenarios:
        raise RuntimeError("Missing train/val/test scenarios")

    first = train_scenarios[0]
    H, W = first["arrival_times"].shape
    n_spatial = first["spatial_feat"].shape[0]
    n_global = first["global_feat"].shape[0]

    trainer.setup_model(H, W, n_spatial_features=n_spatial, n_global_features=n_global)

    write_json(out_dir / "run_config.json", {
        "args": vars(args),
        "config": asdict(cfg),
        "H": H,
        "W": W,
        "n_spatial": n_spatial,
        "n_global": n_global,
    })

    history = {
        "train_loss": [],
        "train_data_loss": [],
        "train_reg_loss": [],
        "learning_rate": [],
        "primary_val_full_corr": [],
        "primary_val_full_rel_rmse": [],
        "extended_val_full_corr": [],
        "extended_val_full_rel_rmse": [],
        "best_primary_epoch": 0,
        "best_primary_val_full_corr": -1.0,
        "best_extended_epoch": 0,
        "best_extended_val_full_corr": -1.0,
    }

    best_primary_state = None
    best_extended_state = None
    best_primary_corr = -1.0
    best_extended_corr = -1.0
    patience_counter = 0

    t0 = time.time()

    for epoch in range(args.epochs):
        t_epoch = time.time()
        losses, dloss, rloss, nbands = [], [], [], []
        n_skip = 0

        np.random.shuffle(train_scenarios)

        for fi, scenario in enumerate(train_scenarios):
            r = trainer.train_step(scenario)

            if r is None:
                n_skip += 1
                continue

            losses.append(r["loss"])
            dloss.append(r["data_loss"])
            rloss.append(r["reg_loss"])
            nbands.append(r["n_bands"])

            if (fi + 1) % 50 == 0:
                print(
                    f"[{args.run_name}] ep{epoch+1} fire {fi+1}/{len(train_scenarios)} "
                    f"loss={r['loss']:.3f} bands={r['n_bands']}"
                )

        if not losses:
            print(f"[{args.run_name}] Epoch {epoch+1}: all fires skipped")
            continue

        al = float(np.mean(losses))
        ad = float(np.mean(dloss))
        ar = float(np.mean(rloss))
        ab = float(np.mean(nbands))
        lr = trainer.optimizer.param_groups[0]["lr"]

        history["train_loss"].append(al)
        history["train_data_loss"].append(ad)
        history["train_reg_loss"].append(ar)
        history["learning_rate"].append(lr)

        do_eval = (epoch == 0) or ((epoch + 1) % args.eval_every == 0)

        if do_eval:
            primary_full_agg, _, _ = trainer.evaluate_full_fire(val_scenarios, save_predictions=False)
            extended_full_agg, _, _ = trainer.evaluate_full_fire(ext_val_scenarios, save_predictions=False)

            primary_corr = float(primary_full_agg.get("correlation_mean", -1.0))
            primary_rel_rmse = float(primary_full_agg.get("relative_rmse_mean", float("inf")))
            extended_corr = float(extended_full_agg.get("correlation_mean", -1.0))
            extended_rel_rmse = float(extended_full_agg.get("relative_rmse_mean", float("inf")))

            history["primary_val_full_corr"].append(primary_corr)
            history["primary_val_full_rel_rmse"].append(primary_rel_rmse)
            history["extended_val_full_corr"].append(extended_corr)
            history["extended_val_full_rel_rmse"].append(extended_rel_rmse)

            trainer.scheduler.step(primary_corr)

            if primary_corr > best_primary_corr:
                best_primary_corr = primary_corr
                best_primary_state = clone_state(trainer.model)
                history["best_primary_epoch"] = epoch + 1
                history["best_primary_val_full_corr"] = primary_corr
                patience_counter = 0
                torch.save({
                    "model_state_dict": best_primary_state,
                    "epoch": epoch + 1,
                    "selection_metric": "primary_val_full_corr",
                    "selection_value": primary_corr,
                    "config": asdict(cfg),
                    "args": vars(args),
                }, out_dir / "model_best_by_primary_val_full_corr.pt")
            else:
                patience_counter += args.eval_every

            if extended_corr > best_extended_corr:
                best_extended_corr = extended_corr
                best_extended_state = clone_state(trainer.model)
                history["best_extended_epoch"] = epoch + 1
                history["best_extended_val_full_corr"] = extended_corr
                torch.save({
                    "model_state_dict": best_extended_state,
                    "epoch": epoch + 1,
                    "selection_metric": "extended_val_full_corr",
                    "selection_value": extended_corr,
                    "config": asdict(cfg),
                    "args": vars(args),
                }, out_dir / "model_best_by_extended_val_full_corr.pt")

            print(
                f"[{args.run_name}] Epoch {epoch+1:3d}: "
                f"loss={al:.3f} data={ad:.3f} reg={ar:.4f} bands={ab:.1f} "
                f"primary_full_corr={primary_corr:.3f} extended_full_corr={extended_corr:.3f} "
                f"lr={lr:.1e} skip={n_skip} {time.time()-t_epoch:.0f}s"
            )

            if patience_counter >= args.patience:
                print(f"[{args.run_name}] Early stopping at epoch {epoch+1}")
                break
        else:
            print(
                f"[{args.run_name}] Epoch {epoch+1:3d}: "
                f"loss={al:.3f} data={ad:.3f} reg={ar:.4f} bands={ab:.1f} "
                f"skip={n_skip} {time.time()-t_epoch:.0f}s"
            )

        write_json(out_dir / "history_live.json", history)

    train_seconds = time.time() - t0

    torch.save({
        "model_state_dict": clone_state(trainer.model),
        "config": asdict(cfg),
        "args": vars(args),
        "history": history,
    }, out_dir / "model_final.pt")

    results = {}

    for ckpt_name, state in [
        ("primary_selected", best_primary_state),
        ("extended_selected", best_extended_state),
    ]:
        if state is None:
            continue

        load_state(trainer.model, state)

        results[ckpt_name] = {
            "primary_val": evaluate_all(trainer, val_scenarios, f"{ckpt_name}_primary_val", out_dir),
            "primary_test": evaluate_all(trainer, test_scenarios, f"{ckpt_name}_primary_test", out_dir),
            "extended_val": evaluate_all(trainer, ext_val_scenarios, f"{ckpt_name}_extended_val", out_dir),
            "extended_test": evaluate_all(trainer, ext_test_scenarios, f"{ckpt_name}_extended_test", out_dir),
        }

    summary = {
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "train_seconds": train_seconds,
        "history": history,
        "results": results,
    }

    write_json(out_dir / "history.json", history)
    write_json(out_dir / "summary.json", summary)

    print()
    print("=" * 120)
    print("RUN COMPLETE: DUAL FULL-CORR VALIDATION")
    print("=" * 120)
    print("out_dir:", out_dir)
    print("train_seconds:", round(train_seconds, 1))
    print("best_primary_epoch:", history["best_primary_epoch"])
    print("best_primary_val_full_corr:", history["best_primary_val_full_corr"])
    print("best_extended_epoch:", history["best_extended_epoch"])
    print("best_extended_val_full_corr:", history["best_extended_val_full_corr"])


if __name__ == "__main__":
    main()
