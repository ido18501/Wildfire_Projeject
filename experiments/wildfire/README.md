# Wildfire Experiments

Experiments on the [Sim2Real-Fire](https://github.com/TJU-IDVLab/Sim2Real-Fire) dataset, corresponding to Section 6 and Appendices G–H of the paper.

## Dataset

Download the Sim2Real-Fire dataset:

```bash
python download_data.py --output_dir /path/to/sim2real_fire_data/simulation_data
```

This requires `gdown` (`pip install gdown`). The full dataset includes 19 geographic scenes with over 32,000 simulated fires and 50 real wildfire observations.

## Experiments

**Per-scene evaluation** (Section 6.1, Table 3, Figure 6):
```bash
python per_scene.py --mode per_scene --gpus 0,1,2,3 --epochs 100
```

**Cross-scene transfer** (Section 6.2, Table 4):
```bash
python cross_scene.py \
    --data_root /path/to/simulation_data \
    --train_scenes 0001_02614 0003_01715 ... \
    --test_scenes 0005_00725 0014_00426 0023_00995 0024_02655 \
    --patch_size 256 --stride 128 --epochs 30
```

**Model capacity analysis** (Section 6.3, Table 5):
```bash
python capacity_analysis.py --mode nn --scenes all --fires_per_scene 3
python capacity_analysis.py --mode pixel --scenes all --fires_per_scene 3
```

**Band-decomposed training** (Appendix G):
```bash
cd band_training
python run.py --mode per_scene \
    --scenes 0024_02655 --gpus 0 \
    --epochs 100 --band_width 0.5 --band_stride 0.25
```

**Real fire evaluation** (Appendix H):
```bash
# Train on real fires
python real_fire_eval.py --data_root /path/to/real_fires --epochs 100

# Evaluate sim-to-real transfer (zero-shot)
python sim2real_transfer.py \
    --model_path /path/to/cross_scene_model.pt \
    --real_fires_path /path/to/real_fires
```
