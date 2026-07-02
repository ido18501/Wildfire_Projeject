#!/usr/bin/env bash
set -Eeo pipefail

cd ~/differentiable-eikonal-wildfire-merged-test
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wildfire
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-$HOME/wildfire_small_subset_manifest.json}"
ROOT_OUT="${ROOT_OUT:-$PWD/small_runs/band2_paperstyle_corr_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-200}"
VAL_EVERY="${VAL_EVERY:-5}"
DEVICE="${DEVICE:-cuda:0}"
LR="${LR:-1e-3}"
LAMBDA="${LAMBDA:-0.002}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_TIME_BANDS="${N_TIME_BANDS:-2}"
FEATURE_GROUP="${FEATURE_GROUP:-topo_weather_omitted}"

mkdir -p "$ROOT_OUT/logs"

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
echo "Official selection criterion: best validation full-fire correlation"
echo "Lambda: $LAMBDA"
echo "Epochs: $EPOCHS"
echo "Validation every: $VAL_EVERY epoch(s)"

run_job "cross_band2_lam002_fullcorr" \
  python experiments/wildfire/small_subset/run_manifest_band_multimetric.py \
    --manifest "$MANIFEST" \
    --output_dir "$ROOT_OUT/cross_band2_lam002_fullcorr" \
    --experiment cross_scene \
    --epochs "$EPOCHS" \
    --val_every "$VAL_EVERY" \
    --device "$DEVICE" \
    --lr "$LR" \
    --lambda_G "$LAMBDA" \
    --lambda_B "$LAMBDA" \
    --batch_size "$BATCH_SIZE" \
    --n_time_bands "$N_TIME_BANDS" \
    --feature_group "$FEATURE_GROUP"

for SCENE in 0001_02614 0002_00714 0018_02440 0014_00426 0005_00725; do
  run_job "per_scene_${SCENE}_band2_lam002_fullcorr" \
    python experiments/wildfire/small_subset/run_manifest_band_multimetric.py \
      --manifest "$MANIFEST" \
      --output_dir "$ROOT_OUT/per_scene_${SCENE}_band2_lam002_fullcorr" \
      --experiment per_scene \
      --scene_id "$SCENE" \
      --epochs "$EPOCHS" \
      --val_every "$VAL_EVERY" \
      --device "$DEVICE" \
      --lr "$LR" \
      --lambda_G "$LAMBDA" \
      --lambda_B "$LAMBDA" \
      --batch_size "$BATCH_SIZE" \
      --n_time_bands "$N_TIME_BANDS" \
      --feature_group "$FEATURE_GROUP"
done

echo
echo "================================================================================"
echo "PAPER-STYLE FULL-CORR BAND2 RUNS FINISHED"
echo "Root output: $ROOT_OUT"
echo "================================================================================"

python - <<PY
from pathlib import Path
import json

root = Path("$ROOT_OUT")
rows = []

for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name == "logs":
        continue

    p = d / "test_results_by_selector.json"
    row = {"run": d.name}

    status = d / "status.txt"
    if status.exists():
        row["status"] = status.read_text().strip().splitlines()[-1]

    if p.exists():
        r = json.loads(p.read_text())
        official = r.get("full_corr")
        if official:
            full = official["test_full_fire"]
            band = official["test_band"]
            row.update({
                "selected_epoch": official["selector_epoch"],
                "val_full_corr": official["selector_best"],
                "test_iou25": full.get("iou_25_mean"),
                "test_iou50": full.get("iou_50_mean"),
                "test_iou75": full.get("iou_75_mean"),
                "test_iou100": full.get("iou_100_mean"),
                "test_corr": full.get("correlation_mean"),
                "test_relrmse": full.get("relative_rmse_mean"),
                "test_band_iou": band.get("mean_band_iou_mean"),
            })

    rows.append(row)

(root / "official_fullcorr_summary.json").write_text(json.dumps(rows, indent=2))

print(f"{'run':55s} {'ep':>4s} {'valCorr':>8s} {'iou50':>8s} {'iou75':>8s} {'corr':>8s} {'relRMSE':>8s} {'bandIoU':>8s}")
print("-" * 120)
for r in rows:
    def f(x, nd=4):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    print(
        f"{r.get('run','')[:55]:55s} "
        f"{f(r.get('selected_epoch'),0):>4s} "
        f"{f(r.get('val_full_corr')):>8s} "
        f"{f(r.get('test_iou50')):>8s} "
        f"{f(r.get('test_iou75')):>8s} "
        f"{f(r.get('test_corr')):>8s} "
        f"{f(r.get('test_relrmse')):>8s} "
        f"{f(r.get('test_band_iou')):>8s}"
    )

print(f"Saved official summary to {root / 'official_fullcorr_summary.json'}")
PY
