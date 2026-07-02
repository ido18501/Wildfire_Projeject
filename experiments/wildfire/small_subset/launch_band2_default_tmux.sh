#!/usr/bin/env bash
set -Eeo pipefail

cd ~/differentiable-eikonal-wildfire-merged-test
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wildfire
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-$HOME/wildfire_small_subset_manifest.json}"
ROOT_OUT="${ROOT_OUT:-$PWD/small_runs/band2_default_convergence_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda:0}"
LR="${LR:-1e-3}"
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

run_job "cross_band2_lam0005" \
  python experiments/wildfire/small_subset/run_manifest_band.py \
    --manifest "$MANIFEST" \
    --output_dir "$ROOT_OUT/cross_band2_lam0005" \
    --experiment cross_scene \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --lr "$LR" \
    --lambda_G 0.005 \
    --lambda_B 0.005 \
    --batch_size "$BATCH_SIZE" \
    --n_time_bands "$N_TIME_BANDS" \
    --feature_group "$FEATURE_GROUP"

for SCENE in 0001_02614 0002_00714 0018_02440 0014_00426 0005_00725; do
  run_job "per_scene_${SCENE}_band2_lam0005" \
    python experiments/wildfire/small_subset/run_manifest_band.py \
      --manifest "$MANIFEST" \
      --output_dir "$ROOT_OUT/per_scene_${SCENE}_band2_lam0005" \
      --experiment per_scene \
      --scene_id "$SCENE" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --lr "$LR" \
      --lambda_G 0.005 \
      --lambda_B 0.005 \
      --batch_size "$BATCH_SIZE" \
      --n_time_bands "$N_TIME_BANDS" \
      --feature_group "$FEATURE_GROUP"
done

echo
echo "ALL DEFAULT-REG BAND2 JOBS FINISHED"
echo "Root output: $ROOT_OUT"
