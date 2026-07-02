# Time-band / Band-MSE Training

This folder supports teacher-forced time-band training.

For each fire with GT arrival times `T_gt`, max duration `T`, and `n_time_bands = n`,
the fire is split into non-overlapping bands:

    [kT/n, (k+1)T/n], for k = 0, ..., n-1

For band `k`:

    source_k = pixels already burned by t_start
    target_k = pixels with t_start < T_gt <= t_end
    target_time_k = T_gt - t_start

So each band trains the model to predict short-horizon spread from the known perimeter at the band start.

## Smoke test

    python experiments/wildfire/band_training/run.py \
      --mode per_scene \
      --data_root ~/sim2real_fire_data/simulation_data \
      --scenes 0014_00426 \
      --gpus 0 \
      --epochs 1 \
      --max_train 2 \
      --max_eval 1 \
      --n_time_bands 4 \
      --output ./timeband_smoke_clean

## Verify band construction

    python experiments/wildfire/band_training/verify_time_bands.py \
      --data_root ~/sim2real_fire_data/simulation_data \
      --scene 0014_00426 \
      --n_time_bands 4

Expected final line:

    VERIFY_TIME_BANDS_OK

## Notes

`--n_time_bands n` is a convenience flag. It sets:

    band_width = 1/n
    band_stride = 1/n

For overlapping bands, use `--band_width` and `--band_stride` directly.
