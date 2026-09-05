1. Use an activated **Conda** environment with Python 3.11, NumPy, SciPy, Numba, Matplotlib and **PyTorch 2.6.0 cu124** ([official build](https://docs.pytorch.org/get-started/previous-versions/#v260)), on the existing CUDA-12.4-compatible driver. No new `.venv`, system CUDA or driver changes. Commands below assume adjacent `Wildfire_Projeject` and `Cardiac_External_Baselines` copies with the existing data and frozen results retained.

2. CUDA availability:
   ```sh
   python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
   ```

3. Tiny GPU checks: from `Wildfire_Projeject`, then from `Cardiac_External_Baselines`, respectively. They fail if CUDA is unavailable.
   ```sh
   python -m tests.test_eikonal3d_cuda --require-cuda
   python -m external_baselines.cuda_smoke --device cuda
   ```

4. Normal 3D cardiac calibration, from `Wildfire_Projeject`:
   ```sh
   python -m experiments.cardiac.numerical_training --backend cuda
   ```

5. Full external baselines, from `Cardiac_External_Baselines` (FNO CUDA; KOL CPU; existing suite option remaps frozen input paths without changing hashes/cases):
   ```sh
   python run_baselines.py --device cuda --eikonal-suite ../Wildfire_Projeject/runs_3d/local_experiment_suite_20260905T152712_824014Z
   ```

6. CPU fallbacks, from the respective directories:
   ```sh
   python -m experiments.cardiac.numerical_training --backend numba
   python run_baselines.py --device cpu --eikonal-suite ../Wildfire_Projeject/runs_3d/local_experiment_suite_20260905T152712_824014Z
   ```
