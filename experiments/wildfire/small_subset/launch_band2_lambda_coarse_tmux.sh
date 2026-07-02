#!/usr/bin/env bash
set -Eeo pipefail

cd ~/differentiable-eikonal-wildfire-merged-test
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wildfire
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-$HOME/wildfire_small_subset_manifest.json}"
ROOT_OUT="${ROOT_OUT:-$PWD/small_runs/band2_lambda_coarse_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-60}"
DEVICE="${DEVICE:-cuda:0}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_TIME_BANDS="${N_TIME_BANDS:-2}"
FEATURE_GROUP="${FEATURE_GROUP:-topo_weather_omitted}"

LAMBDAS=("0.002" "0.01" "0.02" "0.05" "0.10")
SCENES=("0001_02614" "0002_00714" "0018_02440" "0014_00426" "0005_00725")

mkdir -p "$ROOT_OUT/logs"

safe_lam () {
  echo "$1" | sed 's/0\.//; s/\.//g'
}

run_job () {
  local name="$1"
  shift

  local job_dir="$ROOT_OUT/$name"
  local log_file="$ROOT_OUT/logs/${name}.log"
  local cmd_file="$job_dir/command.sh"
  local status_file="$job_dir/status.txt"

  mkdir -p "$job_dir"

  {
    echo "#!/usr/bin/env bash"
    echo "set -Eeo pipefail"
    echo "cd ~/differentiable-eikonal-wildfire-merged-test"
    echo "source ~/miniconda3/etc/profile.d/conda.sh"
    echo "conda activate wildfire"
    echo 'export PYTHONPATH="$PWD:${PYTHONPATH:-}"'
    printf "%q " "$@"
    echo
  } > "$cmd_file"
  chmod +x "$cmd_file"

  echo
  echo "================================================================================"
  echo "START JOB: $name"
  echo "LOG: $log_file"
  echo "================================================================================"

  echo "START $(date)" > "$status_file"

  set +e
  stdbuf -oL -eL bash "$cmd_file" > "$log_file" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "DONE $(date)" >> "$status_file"
    echo "SUCCESS: $name"
  else
    echo "FAILED $(date) rc=$rc" >> "$status_file"
    echo "FAILED: $name rc=$rc"
  fi
}

echo "Root output: $ROOT_OUT"
echo "Epoch cap: $EPOCHS"
echo "Device: $DEVICE"
echo "Lambdas: ${LAMBDAS[*]}"

# ---------------------------------------------------------------------
# Cross-scene coarse lambda sweep
# ---------------------------------------------------------------------
for LAM in "${LAMBDAS[@]}"; do
  SAFE=$(safe_lam "$LAM")

  run_job "cross_band2_lam${SAFE}" \
    python experiments/wildfire/small_subset/run_manifest_band.py \
      --manifest "$MANIFEST" \
      --output_dir "$ROOT_OUT/cross_band2_lam${SAFE}" \
      --experiment cross_scene \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --lr "$LR" \
      --lambda_G "$LAM" \
      --lambda_B "$LAM" \
      --batch_size "$BATCH_SIZE" \
      --n_time_bands "$N_TIME_BANDS" \
      --feature_group "$FEATURE_GROUP"
done

# ---------------------------------------------------------------------
# Per-scene coarse lambda sweep
# ---------------------------------------------------------------------
for SCENE in "${SCENES[@]}"; do
  for LAM in "${LAMBDAS[@]}"; do
    SAFE=$(safe_lam "$LAM")

    run_job "per_scene_${SCENE}_band2_lam${SAFE}" \
      python experiments/wildfire/small_subset/run_manifest_band.py \
        --manifest "$MANIFEST" \
        --output_dir "$ROOT_OUT/per_scene_${SCENE}_band2_lam${SAFE}" \
        --experiment per_scene \
        --scene_id "$SCENE" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --lr "$LR" \
        --lambda_G "$LAM" \
        --lambda_B "$LAM" \
        --batch_size "$BATCH_SIZE" \
        --n_time_bands "$N_TIME_BANDS" \
        --feature_group "$FEATURE_GROUP"
  done
done

echo
echo "================================================================================"
echo "COARSE LAMBDA SWEEP FINISHED"
echo "Root output: $ROOT_OUT"
echo "================================================================================"

python - <<PY
from pathlib import Path
import json
import re
import math

root = Path("$ROOT_OUT")
rows = []

def max_len_dict_lists(h):
    if not isinstance(h, dict):
        return None
    lens = [len(v) for v in h.values() if isinstance(v, list)]
    return max(lens) if lens else None

def get_list(h, *keys):
    if not isinstance(h, dict):
        return []
    for k in keys:
        if k in h and isinstance(h[k], list):
            return h[k]
    return []

def best(vals, mode="max"):
    clean = []
    for i, v in enumerate(vals, start=1):
        try:
            v = float(v)
            if not math.isnan(v):
                clean.append((i, v))
        except Exception:
            pass
    if not clean:
        return None, None
    return (max if mode == "max" else min)(clean, key=lambda x: x[1])

def last(vals):
    clean = []
    for v in vals:
        try:
            v = float(v)
            if not math.isnan(v):
                clean.append(v)
        except Exception:
            pass
    return clean[-1] if clean else None

def parse_lambda(name):
    # lam002 -> 0.002, lam01 -> 0.01, lam10 -> 0.10
    m = re.search(r"lam([0-9]+)$", name)
    if not m:
        return None
    s = m.group(1)
    if s == "002":
        return 0.002
    if s == "005":
        return 0.005
    if s == "01":
        return 0.01
    if s == "02":
        return 0.02
    if s == "05":
        return 0.05
    if s == "10":
        return 0.10
    return s

for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name == "logs":
        continue

    row = {"run": d.name, "lambda": parse_lambda(d.name)}

    status = d / "status.txt"
    if status.exists():
        row["status"] = status.read_text().strip().splitlines()[-1]

    hp = d / "history.json"
    if hp.exists():
        try:
            h = json.loads(hp.read_text())
            row["epochs"] = max_len_dict_lists(h)

            losses = get_list(h, "losses", "loss", "train_losses")
            data = get_list(h, "data_losses", "data_loss")
            reg = get_list(h, "reg_losses", "reg_loss")
            ious = get_list(h, "val_ious", "val_iou", "val_band_iou")
            mses = get_list(h, "val_mses", "val_mse", "val_band_mse")
            lrs = get_list(h, "learning_rates", "lr")

            be, bv = best(ious, "max")
            me, mv = best(mses, "min")

            row["best_val_iou_epoch"] = be
            row["best_val_iou"] = bv
            row["last_val_iou"] = last(ious)
            row["best_val_mse_epoch"] = me
            row["best_val_mse"] = mv
            row["last_val_mse"] = last(mses)
            row["last_loss"] = last(losses)
            row["last_data"] = last(data)
            row["last_reg"] = last(reg)
            if row["last_data"] not in [None, 0] and row["last_reg"] is not None:
                row["reg_data_ratio"] = row["last_reg"] / row["last_data"]
            row["last_lr"] = last(lrs)
        except Exception as e:
            row["history_error"] = repr(e)

    full = d / "test_metrics_full_fire.json"
    if full.exists():
        try:
            m = json.loads(full.read_text())
            row["full_corr"] = m.get("correlation_mean")
            row["full_rel_rmse"] = m.get("relative_rmse_mean")
            row["full_iou50"] = m.get("iou_50_mean")
            row["full_iou75"] = m.get("iou_75_mean")
        except Exception as e:
            row["full_error"] = repr(e)

    band = d / "test_metrics_band.json"
    if band.exists():
        try:
            m = json.loads(band.read_text())
            row["test_band_iou"] = m.get("mean_band_iou_mean")
            row["test_band_mse"] = m.get("mean_band_mse_mean")
        except Exception as e:
            row["band_error"] = repr(e)

    rows.append(row)

(root / "summary.json").write_text(json.dumps(rows, indent=2))

print(f"{'RUN':48s} {'LAM':>6s} {'E':>4s} {'BESTvIOU':>9s} {'LASTvIOU':>9s} {'FULL50':>8s} {'FULL75':>8s} {'CORR':>8s} {'REG/DATA':>9s}")
print("-" * 120)
for r in rows:
    def f(x, nd=4):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)
    print(
        f"{r.get('run','')[:48]:48s} "
        f"{f(r.get('lambda'),3):>6s} "
        f"{f(r.get('epochs'),0):>4s} "
        f"{f(r.get('best_val_iou')):>9s} "
        f"{f(r.get('last_val_iou')):>9s} "
        f"{f(r.get('full_iou50')):>8s} "
        f"{f(r.get('full_iou75')):>8s} "
        f"{f(r.get('full_corr')):>8s} "
        f"{f(r.get('reg_data_ratio')):>9s}"
    )

print(f"Saved summary to {root / 'summary.json'}")
PY
