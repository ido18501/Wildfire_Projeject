#!/usr/bin/env python3
"""
Repository Verification Script
===============================

Verifies that every source file in the differentiable-eikonal-wildfire
repository has correct imports and that the core library functions work.

Usage:
    cd differentiable-eikonal-wildfire
    pip install -e .
    python verify_repo.py

The script performs four levels of checks:
  1. STRUCTURE  - All expected files exist
  2. IMPORTS    - Every .py file can be imported or parsed without error
  3. HYGIENE    - No hardcoded paths or stale sys.path.insert remain
  4. FUNCTIONAL - Core eikonal library solves a small problem end-to-end

Exit code 0 means everything passed.
"""

import os
import sys
import importlib
import traceback
import ast
from pathlib import Path

# =========================================================================
# Configuration
# =========================================================================

REPO_ROOT = Path(__file__).parent.resolve()
os.chdir(REPO_ROOT)

# Ensure the repo root is on sys.path (simulates pip install -e .)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

n_pass = 0
n_fail = 0
n_warn = 0


def report(status, msg):
    global n_pass, n_fail, n_warn
    if status == "pass":
        n_pass += 1
        print(f"  {PASS} {msg}")
    elif status == "fail":
        n_fail += 1
        print(f"  {FAIL} {msg}")
    elif status == "warn":
        n_warn += 1
        print(f"  {WARN} {msg}")


# =========================================================================
# Phase 1: STRUCTURE — Check all expected files exist
# =========================================================================

EXPECTED_FILES = [
    # Top-level
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",

    # eikonal core library
    "eikonal/__init__.py",
    "eikonal/forward/__init__.py",
    "eikonal/forward/pytorch.py",
    "eikonal/forward/numba_cpu.py",
    "eikonal/forward/cuda_batched.py",
    "eikonal/backward/__init__.py",
    "eikonal/backward/implicit_diff.py",
    "eikonal/backward/hybrid.py",
    "eikonal/backward/adjoint_numba.py",
    "eikonal/backward/adjoint_sparse.py",
    "eikonal/backward/adjoint_batched.py",
    "eikonal/nn/__init__.py",
    "eikonal/nn/config.py",
    "eikonal/nn/encoder.py",
    "eikonal/nn/regularization.py",

    # Synthetic experiments
    "experiments/synthetic/__init__.py",
    "experiments/synthetic/experiment_base.py",
    "experiments/synthetic/run_experiments.py",
    "experiments/synthetic/A_forward_solver/__init__.py",
    "experiments/synthetic/A_forward_solver/experiments.py",
    "experiments/synthetic/B_gradient_verification/__init__.py",
    "experiments/synthetic/B_gradient_verification/experiments.py",
    "experiments/synthetic/C_inverse_problem/__init__.py",
    "experiments/synthetic/C_inverse_problem/experiments.py",
    "experiments/synthetic/D_synthetic_wildfire/__init__.py",
    "experiments/synthetic/D_synthetic_wildfire/experiments.py",
    "experiments/synthetic/E_comparisons/__init__.py",
    "experiments/synthetic/E_comparisons/experiments.py",

    # Wildfire experiments
    "experiments/wildfire/__init__.py",
    "experiments/wildfire/per_scene.py",
    "experiments/wildfire/cross_scene.py",
    "experiments/wildfire/capacity_analysis.py",
    "experiments/wildfire/real_fire_eval.py",
    "experiments/wildfire/sim2real_transfer.py",
    "experiments/wildfire/sim2real_loader.py",
    "experiments/wildfire/download_data.py",
    "experiments/wildfire/README.md",
    "experiments/wildfire/band_training/__init__.py",
    "experiments/wildfire/band_training/config.py",
    "experiments/wildfire/band_training/trainer.py",
    "experiments/wildfire/band_training/run.py",

    # Benchmarks
    "experiments/benchmarks/__init__.py",
    "experiments/benchmarks/benchmark_forward.py",
    "experiments/benchmarks/benchmark_backward.py",
    "experiments/benchmarks/plot_forward_benchmark.py",
    "experiments/benchmarks/plot_backward_benchmark.py",

    # Visualization
    "visualization/__init__.py",
    "visualization/plot_per_scene_results.py",
    "visualization/plot_qualitative.py",
    "visualization/plot_covariates_vs_metric.py",
    "visualization/plot_stencil_boundaries.py",
    "visualization/plot_multisource_recovery.py",
    "visualization/plot_metric_components.py",
    "visualization/generate_teaser.py",

    # Tests
    "tests/__init__.py",
    "tests/verify_solver_correctness.py",
    "tests/verify_gradients.py",
    "tests/verify_batched_pipeline.py",
]


def check_structure():
    print(f"\n{BOLD}Phase 1: STRUCTURE — All expected files exist{RESET}")
    for fpath in EXPECTED_FILES:
        full = REPO_ROOT / fpath
        if full.exists():
            report("pass", fpath)
        else:
            report("fail", f"{fpath}  — FILE MISSING")

    # Soft check for .gitignore (dotfiles may not survive all unzip tools)
    gitignore = REPO_ROOT / ".gitignore"
    if gitignore.exists():
        report("pass", ".gitignore (optional)")
    else:
        report("warn", ".gitignore missing — create manually or re-extract with: unzip -o archive.zip")


# =========================================================================
# Phase 2: IMPORTS — Module imports and syntax checks
# =========================================================================

IMPORTABLE_MODULES = [
    ("eikonal",                          "Top-level package"),
    ("eikonal.forward",                  "Forward solver package"),
    ("eikonal.forward.pytorch",          "PyTorch reference solver"),
    ("eikonal.backward",                 "Backward pass package"),
    ("eikonal.backward.implicit_diff",   "Implicit differentiation"),
    ("eikonal.nn",                       "Neural network package"),
    ("eikonal.nn.config",                "Config (DEVICE, TVMetricType)"),
    ("eikonal.nn.encoder",               "FlexibleCovariateEncoder"),
    ("eikonal.nn.regularization",        "TV regularization"),
]

# Specific symbols that should be accessible after import
EXPECTED_SYMBOLS = {
    "eikonal": [
        "EikonalProblem", "fast_sweeping_solve", "create_test_problem",
        "backward_pass", "identify_active_stencils", "UpdateType", "StencilInfo",
        "FlexibleCovariateEncoder", "compute_tv_regularization",
        "DEVICE", "TVMetricType",
    ],
    "eikonal.forward.pytorch": [
        "EikonalProblem", "fast_sweeping_solve", "create_test_problem",
        "compute_two_point_update", "stencil_update",
    ],
    "eikonal.backward.implicit_diff": [
        "backward_pass", "identify_active_stencils", "solve_adjoint_direct",
        "compute_parameter_gradients", "verify_gradients_finite_difference",
        "UpdateType", "StencilInfo", "NEIGHBOR_OFFSETS", "get_stencil_pair",
    ],
    "eikonal.nn.encoder": [
        "FlexibleCovariateEncoder", "project_G_eigenvalue", "project_B_norm",
    ],
    "eikonal.nn.regularization": [
        "compute_tv_regularization", "tv_spd_frobenius",
        "tv_spd_log_euclidean", "tv_vector_field",
    ],
    "eikonal.nn.config": [
        "DEVICE", "TVMetricType",
    ],
}


def check_imports():
    print(f"\n{BOLD}Phase 2a: IMPORTS — Core library modules{RESET}")

    for modname, desc in IMPORTABLE_MODULES:
        try:
            mod = importlib.import_module(modname)
            report("pass", f"import {modname}  ({desc})")
        except Exception as e:
            report("fail", f"import {modname}  — {type(e).__name__}: {e}")

    print(f"\n{BOLD}Phase 2b: SYMBOLS — Expected names accessible{RESET}")

    for modname, symbols in EXPECTED_SYMBOLS.items():
        try:
            mod = importlib.import_module(modname)
        except Exception:
            report("fail", f"{modname} could not be imported")
            continue
        missing = [s for s in symbols if not hasattr(mod, s)]
        if not missing:
            report("pass", f"{modname}: all {len(symbols)} symbols present")
        else:
            report("fail", f"{modname}: missing {missing}")

    print(f"\n{BOLD}Phase 2c: SYNTAX — All .py files parse correctly{RESET}")

    all_py = sorted(REPO_ROOT.rglob("*.py"))
    for pyfile in all_py:
        if "__pycache__" in str(pyfile):
            continue
        rel = pyfile.relative_to(REPO_ROOT)
        try:
            with open(pyfile, "r") as f:
                ast.parse(f.read(), filename=str(rel))
            report("pass", str(rel))
        except SyntaxError as e:
            report("fail", f"{rel}  — SyntaxError: {e}")

    print(f"\n{BOLD}Phase 2d: IMPORT REFS — eikonal.* imports resolve{RESET}")

    for pyfile in all_py:
        if "__pycache__" in str(pyfile):
            continue
        rel = pyfile.relative_to(REPO_ROOT)
        try:
            with open(pyfile, "r") as f:
                tree = ast.parse(f.read(), filename=str(rel))
        except SyntaxError:
            continue

        eikonal_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("eikonal"):
                    names = [alias.name for alias in node.names]
                    eikonal_imports.append((node.module, names, node.lineno))

        if not eikonal_imports:
            continue

        all_ok = True
        issues = []
        for mod, names, lineno in eikonal_imports:
            try:
                m = importlib.import_module(mod)
                for name in names:
                    if not hasattr(m, name):
                        all_ok = False
                        issues.append(f"L{lineno}: {mod}.{name} not found")
            except Exception as e:
                err = str(e).lower()
                if "numba" in err or "cuda" in err:
                    continue  # optional dependency
                all_ok = False
                issues.append(f"L{lineno}: {mod} — {type(e).__name__}")

        n_syms = sum(len(names) for _, names, _ in eikonal_imports)
        if all_ok:
            report("pass", f"{rel}  ({n_syms} symbols from eikonal.*)")
        else:
            report("fail", f"{rel}  — {'; '.join(issues)}")


# =========================================================================
# Phase 3: HYGIENE — No hardcoded paths or stale sys.path
# =========================================================================

def check_hygiene():
    print(f"\n{BOLD}Phase 3: HYGIENE — No hardcoded paths or stale sys.path{RESET}")

    all_py = sorted(REPO_ROOT.rglob("*.py"))
    for pyfile in all_py:
        if "__pycache__" in str(pyfile) or pyfile.name == "verify_repo.py":
            continue
        rel = pyfile.relative_to(REPO_ROOT)
        issues = []

        with open(pyfile, "r") as f:
            lines = f.readlines()

        in_docstring = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Rough docstring tracking
            triple_count = line.count('"""') + line.count("'''")
            if triple_count == 1:
                in_docstring = not in_docstring
            elif triple_count >= 2:
                pass  # single-line docstring, stays same state

            if in_docstring or stripped.startswith("#"):
                continue

            # Check for hardcoded absolute paths
            if "/home/barak" in line:
                issues.append(f"L{i}: contains /home/barak")
            if "/home/claude" in line:
                issues.append(f"L{i}: contains hardcoded path")
            if "/mnt/user-data" in line:
                issues.append(f"L{i}: contains /mnt/user-data")

            # Check for sys.path.insert (outside comments/docstrings)
            if "sys.path.insert" in stripped and not stripped.startswith("#"):
                issues.append(f"L{i}: sys.path.insert found")

        if not issues:
            report("pass", str(rel))
        else:
            for issue in issues:
                report("fail", f"{rel}:{issue}")


# =========================================================================
# Phase 4: FUNCTIONAL — Core library end-to-end test
# =========================================================================

def check_functional():
    print(f"\n{BOLD}Phase 4: FUNCTIONAL — End-to-end solver test{RESET}")

    # 4a. EikonalProblem creation using the built-in factory
    try:
        import torch
        import numpy as np
        from eikonal.forward.pytorch import (
            EikonalProblem, fast_sweeping_solve, create_test_problem
        )

        M, N = 40, 40
        problem = create_test_problem(M=M, N=N, case="isotropic")
        assert problem.G.shape == (1, 3, M, N)
        assert problem.B.shape == (1, 2, M, N)
        assert problem.source_mask.shape == (1, M, N)
        assert problem.M == M and problem.N == N
        report("pass", "EikonalProblem construction via create_test_problem")
    except Exception as e:
        report("fail", f"EikonalProblem construction — {e}")
        return

    # 4b. Forward solve
    try:
        T, n_iter = fast_sweeping_solve(problem, max_iter=30, tol=1e-6)
        assert T.shape == (1, M, N), f"Bad shape: {T.shape}"
        assert T[0, M // 2, N // 2] == 0.0, "Source should be 0"
        T_interior = T[0, 1:-1, 1:-1]
        assert T_interior.max() < 1e5, f"Solve did not converge (max={T_interior.max():.1f})"
        assert T.min() >= 0.0, "No negative arrival times"
        report("pass", f"Forward solve (n_iter={n_iter}, T interior max={T_interior.max():.2f})")
    except Exception as e:
        report("fail", f"Forward solve — {e}")
        return

    # 4c. Analytical comparison (isotropic G=I, B=0 → Euclidean pixel distance)
    try:
        cx, cy = M // 2, N // 2
        y_coords, x_coords = np.mgrid[0:M, 0:N]
        # Solver uses pixel-unit coordinates (no h factor)
        T_exact = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(np.float32)
        T_np = T[0].numpy()
        # Compare only interior (boundary stays at inf) and away from source
        valid = (T_exact > 1.0) & (T_np < 1e5)
        if valid.sum() > 0:
            rel_err = np.abs(T_np[valid] - T_exact[valid]) / T_exact[valid]
            mean_rel = rel_err.mean()
            max_rel = rel_err.max()
            ok = mean_rel < 0.05
            report("pass" if ok else "fail",
                   f"Isotropic accuracy: mean rel err = {mean_rel:.4f}, max = {max_rel:.4f}")
        else:
            report("fail", "No valid interior points for comparison")
    except Exception as e:
        report("fail", f"Analytical comparison — {e}")

    # 4d. Backward pass (implicit differentiation)
    #     Signature: backward_pass(T, dL_dT, problem)
    try:
        from eikonal.backward.implicit_diff import backward_pass
        dL_dT = 2.0 * T
        dL_dG, dL_dB = backward_pass(T, dL_dT, problem)
        assert dL_dG.shape == problem.G.shape, f"dL_dG shape mismatch: {dL_dG.shape}"
        assert dL_dB.shape == problem.B.shape, f"dL_dB shape mismatch: {dL_dB.shape}"
        assert torch.isfinite(dL_dG).all(), "dL_dG has non-finite values"
        assert torch.isfinite(dL_dB).all(), "dL_dB has non-finite values"
        nz_G = (dL_dG != 0).sum().item()
        nz_B = (dL_dB != 0).sum().item()
        report("pass", f"Backward pass (dL_dG nonzero: {nz_G}, dL_dB nonzero: {nz_B})")
    except Exception as e:
        report("fail", f"Backward pass — {e}")
        traceback.print_exc()

    # 4e. Neural network encoder
    #     Parameter is num_cnn_layers (not n_conv_layers)
    encoder = None
    n_spatial, n_global = 5, 3
    try:
        from eikonal.nn.encoder import FlexibleCovariateEncoder
        encoder = FlexibleCovariateEncoder(
            n_spatial_features=n_spatial,
            n_global_features=n_global,
            hidden_channels=16,
            num_cnn_layers=3,
        )
        spatial = torch.randn(1, n_spatial, M, N)
        global_feat = torch.randn(1, n_global)
        G_pred, B_pred, G_local, B_local = encoder(spatial, global_feat)
        assert G_pred.shape == (1, 3, M, N), f"G shape: {G_pred.shape}"
        assert B_pred.shape == (1, 2, M, N), f"B shape: {B_pred.shape}"
        report("pass", f"FlexibleCovariateEncoder forward pass")
    except Exception as e:
        report("fail", f"FlexibleCovariateEncoder — {e}")

    # 4f. TV Regularization
    #     Signature: compute_tv_regularization(G, B, lambda_G, lambda_B, metric_type) -> Tensor
    try:
        from eikonal.nn.regularization import compute_tv_regularization
        from eikonal.nn.config import TVMetricType
        assert encoder is not None, "Skipped (encoder failed)"
        loss_reg = compute_tv_regularization(
            G_pred, B_pred, 1.0, 1.0, TVMetricType.FROBENIUS
        )
        assert loss_reg.ndim == 0, f"TV should be scalar, got shape {loss_reg.shape}"
        assert loss_reg >= 0, "TV should be non-negative"
        report("pass", f"TV regularization (loss_reg={loss_reg.item():.4f})")
    except Exception as e:
        report("fail", f"TV regularization — {e}")

    # 4g. Numba forward solver (optional)
    try:
        from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup
        warmup()
        T_numba, _ = fast_sweeping_solve_numba(problem, max_iter=30, tol=1e-6)
        valid_mask = (T[0].numpy() < 1e5) & (T_numba[0].numpy() < 1e5)
        max_diff = abs(T[0].numpy() - T_numba[0].numpy())[valid_mask].max()
        # Tolerance accounts for float32 vs float64 differences
        ok = max_diff < 1e-3
        report("pass" if ok else "fail",
               f"Numba solver matches PyTorch (max diff = {max_diff:.2e})")
    except ImportError:
        report("warn", "Numba solver skipped (numba not installed)")
    except Exception as e:
        report("fail", f"Numba solver — {e}")

    # 4h. Full pipeline: encoder → solver → implicit diff → encoder.grad
    if encoder is not None:
        try:
            encoder.zero_grad()
            spatial_in = torch.randn(1, n_spatial, M, N)
            global_in = torch.randn(1, n_global)

            G_out, B_out, _, _ = encoder(spatial_in, global_in)

            prob = EikonalProblem(
                G=G_out.detach(), B=B_out.detach(),
                source_mask=problem.source_mask,
            )
            T_pred, _ = fast_sweeping_solve(prob, max_iter=30, tol=1e-6)

            dL_dT = 2.0 * T_pred
            dL_dG, dL_dB = backward_pass(T_pred, dL_dT, prob)

            # Backprop through encoder
            G_out.backward(dL_dG)
            has_grads = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in encoder.parameters()
            )
            report("pass" if has_grads else "fail",
                   "Full pipeline: encoder → solver → backward → encoder.grad")
        except Exception as e:
            report("fail", f"Full pipeline — {e}")
            traceback.print_exc()
    else:
        report("fail", "Full pipeline skipped (encoder construction failed)")


# =========================================================================
# Main
# =========================================================================

def main():
    print(f"{BOLD}{'=' * 65}")
    print("  Repository Verification: differentiable-eikonal-wildfire")
    print(f"  Root: {REPO_ROOT}")
    print(f"{'=' * 65}{RESET}")

    check_structure()
    check_imports()
    check_hygiene()
    check_functional()

    total = n_pass + n_fail + n_warn
    print(f"\n{BOLD}{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}{RESET}")
    print(f"  {PASS} Passed:   {n_pass}")
    print(f"  {FAIL} Failed:   {n_fail}")
    print(f"  {WARN} Warnings: {n_warn}")
    print(f"  Total:    {total}")

    if n_fail == 0:
        print(f"\n  {BOLD}{PASS} ALL CHECKS PASSED{RESET}\n")
    else:
        print(f"\n  {BOLD}{FAIL} {n_fail} CHECK(S) FAILED — see above{RESET}\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
