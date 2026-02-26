#!/usr/bin/env python3
"""
Quick Smoke Test — Reproduce Selected Paper Figures
=====================================================

Runs a small subset of synthetic experiments to verify the full pipeline
works end-to-end and produces figures matching the paper.

Generates:
  - A1: Forward solver convergence (Table 1, Fig A1)
  - B1: Gradient verification — isotropic (Fig B1)
  - C1: Inverse problem — isotropic metric recovery (Fig C1)

Usage:
    cd differentiable-eikonal-wildfire
    pip install -e .
    python smoke_test.py

Figures are saved to experiments/synthetic/figures/
"""

import os
import sys
import time

# Run from the synthetic experiments directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTH_DIR = os.path.join(SCRIPT_DIR, "experiments", "synthetic")
os.chdir(SYNTH_DIR)
sys.path.insert(0, SYNTH_DIR)

from experiment_base import DEVICE, FIGURES_DIR

BOLD = "\033[1m"
RESET = "\033[0m"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def run_experiment(exp_class, label, **kwargs):
    """Run a single experiment and report results."""
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}{RESET}")
    t0 = time.time()
    try:
        exp = exp_class(**kwargs)
        result = exp.execute(save=True, visualize=True, verbose=True)
        elapsed = time.time() - t0
        print(f"\n  {PASS} {label} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  {FAIL} {label} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"{BOLD}{'=' * 60}")
    print("  Smoke Test: Reproduce Selected Paper Figures")
    print(f"  Device: {DEVICE}")
    print(f"  Figures directory: {FIGURES_DIR}")
    print(f"{'=' * 60}{RESET}")

    results = []

    # A1: Forward solver convergence (Table 1, Figure A1)
    from A_forward_solver.experiments import A1_IsotropicConvergence
    results.append(run_experiment(
        A1_IsotropicConvergence,
        "A1: Isotropic Convergence (Table 1, Fig A1)",
        grid_sizes=[25, 50, 100, 200],  # skip 400 for speed
    ))

    # B1: Gradient verification — isotropic (Figure B1)
    from B_gradient_verification.experiments import B1_FD_Isotropic
    results.append(run_experiment(
        B1_FD_Isotropic,
        "B1: Gradient Verification — Isotropic (Fig B1)",
    ))

    # C1: Inverse problem — isotropic recovery (Figure C1)
    from C_inverse_problem.experiments import C1_IsotropicFull
    results.append(run_experiment(
        C1_IsotropicFull,
        "C1: Isotropic Metric Recovery (Fig C1)",
    ))

    # Summary
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  SMOKE TEST SUMMARY: {n_pass}/{n_total} passed")
    print(f"{'=' * 60}{RESET}")

    if n_pass == n_total:
        print(f"\n  {PASS} All experiments passed. Check figures in:")
        print(f"      {FIGURES_DIR}/")
        # List generated figures
        if os.path.exists(FIGURES_DIR):
            figs = sorted(f for f in os.listdir(FIGURES_DIR) if f.endswith('.pdf') or f.endswith('.png'))
            for fig in figs:
                print(f"      - {fig}")
    else:
        print(f"\n  {FAIL} Some experiments failed — see output above")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
