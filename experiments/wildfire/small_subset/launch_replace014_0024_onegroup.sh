#!/usr/bin/env bash
set -euo pipefail

FEATURE_GROUP="$1"
ROOT_OUT="$2"

cd ~/differentiable-eikonal-wildfire-merged-test
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wildfire
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$ROOT_OUT/logs"

echo "FEATURE_GROUP=$FEATURE_GROUP"
echo "ROOT_OUT=$ROOT_OUT"

python experiments/wildfire/small_subset/run_manifest_band_multimetric.py \
  --manifest ~/wildfire_small_subset_no014_manifest.json \
  --output_dir "$ROOT_OUT/cross_band2_lam002_fullcorr" \
  --experiment cross_scene \
  --feature_group "$FEATURE_GROUP" \
  --n_time_bands 2 \
  --lambda_G 0.002 \
  --lambda_B 0.002 \
  --epochs 200 \
  --val_every 5 \
  --device cuda:0 \
  --lr 0.001 \
  --batch_size 32 \
  --seed 42 \
  > "$ROOT_OUT/logs/cross.log" 2>&1

python experiments/wildfire/small_subset/run_manifest_band_multimetric.py \
  --manifest ~/wildfire_small_subset_no014_manifest.json \
  --output_dir "$ROOT_OUT/per_scene_0024_02655_band2_lam002_fullcorr" \
  --experiment per_scene \
  --scene_id 0024_02655 \
  --feature_group "$FEATURE_GROUP" \
  --n_time_bands 2 \
  --lambda_G 0.002 \
  --lambda_B 0.002 \
  --epochs 200 \
  --val_every 5 \
  --device cuda:0 \
  --lr 0.001 \
  --batch_size 32 \
  --seed 42 \
  > "$ROOT_OUT/logs/per_scene_0024.log" 2>&1
