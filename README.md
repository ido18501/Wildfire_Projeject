# Differentiable Eikonal Models for Wildfire Simulation

Research code for modeling wildfire propagation with differentiable Randers-Finsler Eikonal solvers.

Based on the framework by **Barak Gahtan, Jacob Shpund, and Alex M. Bronstein**.

## Overview

The framework solves and differentiates through the Randers-Finsler eikonal equation

$$(\nabla T - \mathbf{b})^\top \mathbf{G}^{-1} (\nabla T - \mathbf{b}) = 1,$$

where $\mathbf{G}(\mathbf{x})$ is a spatially-varying metric tensor encoding anisotropic propagation and $\mathbf{b}(\mathbf{x})$ is a drift vector capturing directional bias. The forward solver uses fast sweeping with eight triangular stencils, converging in 2–3 iterations independent of grid size. The backward pass applies implicit differentiation to the converged solution, computing parameter gradients by solving a sparse lower-triangular adjoint system in a single reverse-time pass.

A neural network encoder maps environmental covariates (terrain, fuel, weather) to Randers metric parameters, trained end-to-end through the solver. On the Sim2Real-Fire dataset, the framework achieves mean test correlation of 0.824 for within-scene generalization and 0.766 for cross-scene transfer to unseen geographic regions.

## Installation

```bash
git clone https://github.com/ido18501/Wildfire_Projeject.git
cd Wildfire_Projeject
pip install -e .
```

For Numba-accelerated solvers (recommended):
```bash
pip install -e ".[numba]"
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.12, NumPy, SciPy, Matplotlib. Optional: Numba ≥ 0.56 (for JIT-compiled solvers), CUDA toolkit (for GPU batched solver), gdown (for dataset download).

## Quick Start

```python
import torch
from eikonal import EikonalProblem, fast_sweeping_solve, backward_pass

# Define a Randers-Finsler problem on an 80x80 grid
M, N = 80, 80
G = torch.stack([
    torch.ones(M, N),   # g11
    torch.zeros(M, N),  # g12
    torch.ones(M, N),   # g22
], dim=0).unsqueeze(0)   # (1, 3, M, N) — isotropic identity metric
B = torch.zeros(1, 2, M, N)  # no drift
source_mask = torch.zeros(1, M, N, dtype=torch.bool)
source_mask[0, M//2, N//2] = True

problem = EikonalProblem(G=G, B=B, source_mask=source_mask)

# Solve forward
T, n_iter = fast_sweeping_solve(problem, max_iter=30, tol=1e-6)

# Compute gradients via implicit differentiation
dL_dT = 2 * (T - T_observed)  # example MSE gradient
dL_dG, dL_dB = backward_pass(T, dL_dT, problem)
```

## Repository Structure

```
Wildfire_Projeject/
│
├── eikonal/                               Core library
│   ├── forward/
│   │   ├── pytorch.py                     Reference PyTorch solver
│   │   ├── numba_cpu.py                   Numba JIT solver (83× speedup)
│   │   └── cuda_batched.py               Batched CUDA solver (2778× speedup)
│   ├── backward/
│   │   ├── implicit_diff.py              Implicit differentiation backward pass
│   │   ├── hybrid.py                      Stencil-boundary hybrid gradients
│   │   ├── adjoint_numba.py              Numba-optimized adjoint solve (3700× speedup)
│   │   ├── adjoint_sparse.py            Sparse adjoint for band training
│   │   └── adjoint_batched.py           Batched backward for multi-fire training
│   └── nn/
│       ├── encoder.py                     FlexibleCovariateEncoder (CNN + MLP)
│       ├── regularization.py             TV regularization for metric fields
│       └── config.py                      Device and type configuration
│
├── experiments/
│   ├── synthetic/                         Sections 5, Appendices A–E (37 experiments)
│   │   ├── run_experiments.py            Main runner
│   │   ├── A_forward_solver/             Forward solver validation
│   │   ├── B_gradient_verification/      Gradient correctness
│   │   ├── C_inverse_problem/            Metric recovery
│   │   ├── D_synthetic_wildfire/         Synthetic wildfire scenarios
│   │   └── E_comparisons/               Method comparisons
│   ├── wildfire/                          Section 6, Appendices G–H
│   │   ├── per_scene.py                  Within-scene generalization (Sec 6.1)
│   │   ├── cross_scene.py               Cross-scene transfer (Sec 6.2)
│   │   ├── capacity_analysis.py         Model capacity diagnostic (Sec 6.3)
│   │   ├── band_training/               Temporal band decomposition (App G)
│   │   ├── real_fire_eval.py            Real wildfire evaluation (App H)
│   │   ├── sim2real_transfer.py         Sim-to-real transfer (App H)
│   │   ├── sim2real_loader.py           Sim2Real-Fire data loader
│   │   └── download_data.py            Dataset download script
│   └── benchmarks/                        Section 5.3 timing figures
│       ├── benchmark_forward.py          Forward solver benchmarks (Fig 4)
│       ├── benchmark_backward.py         Backward pass benchmarks (Fig 5)
│       ├── plot_forward_benchmark.py     Generate forward timing plot
│       └── plot_backward_benchmark.py    Generate backward timing plot
│
├── visualization/                         Figure generation
│   ├── plot_per_scene_results.py         Per-scene summary (Fig 6)
│   ├── plot_qualitative.py              Qualitative examples (Fig 7)
│   ├── plot_stencil_boundaries.py       Stencil structure (Fig 3)
│   ├── plot_covariates_vs_metric.py     Covariates-to-metric visualization
│   ├── plot_multisource_recovery.py     Multi-source recovery (Figs C10–C11)
│   ├── plot_metric_components.py        Metric field visualization
│   └── generate_teaser.py              Teaser figure (Fig 1)
│
├── tests/                                 Correctness verification
│   ├── verify_solver_correctness.py     All solver variants produce identical results
│   ├── verify_gradients.py              FD vs implicit differentiation (Table 2)
│   └── verify_batched_pipeline.py       Batched matches sequential
│
├── pyproject.toml
├── requirements.txt
├── LICENSE                                MIT
└── README.md
```

## Experiments and result scripts

The table below maps the reference experiments to the scripts used to run them. All commands assume the repository root as working directory and that the `eikonal` package is installed (`pip install -e .`).

| Experiment | Result | Command |
|---|---|---|
| **Section 5.1** – Forward Solver | Table 1, Figs A1–A9 | `cd experiments/synthetic && python run_experiments.py --phase A` |
| **Section 5.2** – Gradient Verification | Table 2, Figs B1–B9 | `cd experiments/synthetic && python run_experiments.py --phase B` |
| **Section 5.3** – Efficiency | Figs 4–5 | `cd experiments/benchmarks && python benchmark_forward.py && python benchmark_backward.py` |
| **Section 5.4** – Inverse Problem | Table 3, Figs C1–C11 | `cd experiments/synthetic && python run_experiments.py --phase C` |
| **Section 6.1** – Per-Scene | Fig 6, Table F.1 | `cd experiments/wildfire && python per_scene.py --mode per_scene --gpus 0,1,2,3` |
| **Section 6.2** – Cross-Scene | Table 4 | `cd experiments/wildfire && python cross_scene.py --data_root /path/to/data` |
| **Section 6.3** – Capacity | Table 5 | `cd experiments/wildfire && python capacity_analysis.py` |
| **Appendix D** – Synthetic Wildfire | Figs D1–D6 | `cd experiments/synthetic && python run_experiments.py --phase D` |
| **Appendix E** – Comparisons | Figs E1–E5 | `cd experiments/synthetic && python run_experiments.py --phase E` |
| **Appendix G** – Band Training | App G results | `cd experiments/wildfire/band_training && python run.py` |
| **Appendix H** – Real Fire | Table 6 | `cd experiments/wildfire && python real_fire_eval.py && python sim2real_transfer.py` |

## Datasets

**Sim2Real-Fire** ([Li et al., NeurIPS 2024](https://github.com/TJU-IDVLab/Sim2Real-Fire)): Over one million FARSITE-simulated wildfire scenarios across 19 geographic scenes with aligned topography, vegetation, fuel, and weather covariates at 30m spatial and hourly temporal resolution. Includes 50 real wildfire observations from satellite imagery.

```bash
cd experiments/wildfire
python download_data.py --output_dir /path/to/sim2real_fire_data/simulation_data
```

## Key Implementation Details

**Forward solver.** The fast sweeping method processes the grid in four alternating sweep directions (left-right, top-bottom, right-left, bottom-top), with entire columns or rows updated simultaneously for GPU parallelism. At each node, eight triangular stencils are evaluated; the minimum valid candidate becomes the new arrival time. Convergence occurs in 2–3 iterations independent of grid size, yielding O(n) total work. Three implementations are provided: a PyTorch reference, a Numba JIT-compiled version achieving 83× speedup, and a CUDA batched version solving multiple fires in parallel.

**Backward pass.** Implicit differentiation avoids unrolling the iterative solver. The Jacobian ∂R/∂T is sparse and lower-triangular when nodes are ordered by increasing arrival time, enabling the adjoint system to be solved by a single reverse-time back-substitution. The Numba-compiled adjoint solve achieves 3700× speedup over naive Python. For band-decomposed training, a sparse variant processes only active pixels (those with 0 < T < ∞), providing additional speedup when large portions of the grid are teacher-forced sources.

**Neural encoder.** The `FlexibleCovariateEncoder` combines a global MLP (for domain-wide weather features) with a fully-convolutional network (for spatial terrain/fuel covariates). Outputs are projected to satisfy feasibility constraints: G is made positive-definite via eigenvalue clamping, and the drift norm is rescaled to satisfy the Randers condition ‖b‖_{G⁻¹} < 1.

```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
