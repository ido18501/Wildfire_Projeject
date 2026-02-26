#!/usr/bin/env python3
"""
SGP Paper Experiments - Main Runner
===================================

Run experiments for the Differentiable Randers-Finsler Eikonal Solver paper.

Usage:
    python run_experiments.py                    # Run ALL experiments
    python run_experiments.py --phase A          # Run only Category A
    python run_experiments.py --phase A B        # Run Categories A and B
    python run_experiments.py --phase A --quick  # Quick mode (smaller grids)
    python run_experiments.py --list             # List all experiments
    python run_experiments.py --exp A1 B3 C7     # Run specific experiments

Categories:
    A: Forward Solver Validation (9 experiments)
    B: Gradient Verification (9 experiments)
    C: Inverse Problem / Metric Recovery (8 experiments)
    D: Synthetic Wildfire Scenarios (6 experiments)
    E: Comparisons (5 experiments)
    
Total: 37 experiments
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for project path
# Project imports handled via installed eikonal package

from experiment_base import DEVICE, RESULTS_DIR, FIGURES_DIR


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

def get_all_experiments() -> Dict[str, List]:
    """Get all experiment classes organized by category."""
    
    experiments = {}
    
    # Category A: Forward Solver
    from A_forward_solver.experiments import (
        A1_IsotropicConvergence, A2_UniformAnisotropic, A3_RotatedAnisotropic,
        A4_ConstantDrift, A5_CombinedAnisotropicDrift, A6_PiecewiseMetric,
        A7_SpatiallyVaryingMetric, A8_MultiSource, A9_IterationComplexity
    )
    experiments['A'] = [
        ('A1', A1_IsotropicConvergence, {}, {'grid_sizes': [25, 50, 100]}),
        ('A2', A2_UniformAnisotropic, {}, {'N': 100}),
        ('A3', A3_RotatedAnisotropic, {}, {'N': 100}),
        ('A4', A4_ConstantDrift, {}, {'N': 100}),
        ('A5', A5_CombinedAnisotropicDrift, {}, {'N': 50, 'N_fine': 200}),
        ('A6', A6_PiecewiseMetric, {}, {'N': 100}),
        ('A7', A7_SpatiallyVaryingMetric, {}, {'N': 100}),
        ('A8', A8_MultiSource, {}, {'N': 100}),
        ('A9', A9_IterationComplexity, {}, {'grid_sizes': [50, 100, 200]}),
    ]
    
    # Category B: Gradient Verification
    from B_gradient_verification.experiments import (
        B1_FD_Isotropic, B2_FD_Anisotropic, B3_FD_Drift,
        B4_RingLoss, B5_GradientDecay, B6_StencilMap,
        B7_StencilBoundaries, B8_RandomDirections, B9_GradientStability
    )
    experiments['B'] = [
        ('B1', B1_FD_Isotropic, {}, {'N': 30, 'n_test_points': 10}),
        ('B2', B2_FD_Anisotropic, {}, {'N': 30, 'n_test_points': 10}),
        ('B3', B3_FD_Drift, {}, {'N': 30, 'n_test_points': 10}),
        ('B4', B4_RingLoss, {}, {'N': 60}),
        ('B5', B5_GradientDecay, {}, {'N': 60}),
        ('B6', B6_StencilMap, {}, {'N': 60}),
        ('B7', B7_StencilBoundaries, {}, {'N': 60}),
        ('B8', B8_RandomDirections, {}, {'N': 30, 'n_directions': 50}),
        ('B9', B9_GradientStability, {}, {'N': 30, 'n_perturbations': 10}),
    ]
    
    # Category C: Inverse Problem
    from C_inverse_problem.experiments import (
        C1_IsotropicFull, C2_IsotropicSparse, C3_DiagonalAnisotropic,
        C5_DriftRecovery, C7_RegularizationAblation, C8_ObservationDensity,
        C9_NoiseRobustness, C10_MultipleSources
    )
    experiments['C'] = [
        ('C1', C1_IsotropicFull, {}, {'N': 50, 'n_iter': 150}),
        ('C2', C2_IsotropicSparse, {}, {'N': 50, 'n_iter': 200}),
        ('C3', C3_DiagonalAnisotropic, {}, {'N': 50, 'n_iter': 200}),
        ('C5', C5_DriftRecovery, {}, {'N': 40, 'n_iter': 150}),
        ('C7', C7_RegularizationAblation, {}, {'N': 40, 'n_iter': 100}),
        ('C8', C8_ObservationDensity, {}, {'N': 40, 'n_iter': 100}),
        ('C9', C9_NoiseRobustness, {}, {'N': 40, 'n_iter': 100}),
        ('C10', C10_MultipleSources, {}, {'N': 40, 'n_iter': 100}),
    ]
    
    # Category D: Synthetic Wildfire
    from D_synthetic_wildfire.experiments import (
        D1_TerrainDriven, D2_WindDriven, D3_FuelHeterogeneity,
        D4_CombinedScenario, D5_FireLineReconstruction, D6_ParameterSensitivity
    )
    experiments['D'] = [
        ('D1', D1_TerrainDriven, {}, {'N': 80}),
        ('D2', D2_WindDriven, {}, {'N': 80}),
        ('D3', D3_FuelHeterogeneity, {}, {'N': 80}),
        ('D4', D4_CombinedScenario, {}, {'N': 100}),
        ('D5', D5_FireLineReconstruction, {}, {'N': 60}),
        ('D6', D6_ParameterSensitivity, {}, {'N': 60}),
    ]
    
    # Category E: Comparisons
    from E_comparisons.experiments import (
        E1_SweepingVsMarching, E2_ImplicitVsFD, E3_RegularizationStrategies,
        E4_OptimizationMethods, E5_Scalability
    )
    experiments['E'] = [
        ('E1', E1_SweepingVsMarching, {}, {'grid_sizes': [30, 50, 100]}),
        ('E2', E2_ImplicitVsFD, {}, {'grid_sizes': [30, 50]}),
        ('E3', E3_RegularizationStrategies, {}, {'N': 40, 'n_iter': 100}),
        ('E4', E4_OptimizationMethods, {}, {'N': 40, 'n_iter': 100}),
        ('E5', E5_Scalability, {}, {'grid_sizes': [50, 100, 200]}),
    ]
    
    return experiments


def list_experiments():
    """Print list of all experiments."""
    experiments = get_all_experiments()
    
    print("\n" + "="*70)
    print("SGP PAPER EXPERIMENTS")
    print("="*70)
    
    total = 0
    for cat, exp_list in experiments.items():
        cat_names = {
            'A': 'Forward Solver Validation',
            'B': 'Gradient Verification',
            'C': 'Inverse Problem / Metric Recovery',
            'D': 'Synthetic Wildfire Scenarios',
            'E': 'Comparisons'
        }
        print(f"\nCategory {cat}: {cat_names.get(cat, '')}")
        print("-"*50)
        for exp_id, exp_class, _, _ in exp_list:
            print(f"  {exp_id}: {exp_class.description}")
            total += 1
    
    print(f"\n{'='*70}")
    print(f"Total: {total} experiments")
    print("="*70)


def run_experiments(phases: List[str] = None, 
                    specific_exps: List[str] = None,
                    quick: bool = False,
                    save: bool = True,
                    visualize: bool = True,
                    verbose: bool = True) -> Dict:
    """
    Run experiments.
    
    Args:
        phases: List of categories to run (e.g., ['A', 'B'])
        specific_exps: List of specific experiment IDs (e.g., ['A1', 'B3'])
        quick: Use smaller grids for faster runs
        save: Save results to disk
        visualize: Generate figures
        verbose: Print progress
    
    Returns:
        Dictionary of experiment results
    """
    all_experiments = get_all_experiments()
    
    # Determine which experiments to run
    to_run = []
    
    if specific_exps:
        # Run specific experiments
        for exp_id in specific_exps:
            cat = exp_id[0].upper()
            if cat in all_experiments:
                for eid, cls, kwargs, quick_kwargs in all_experiments[cat]:
                    if eid.upper() == exp_id.upper():
                        to_run.append((eid, cls, quick_kwargs if quick else kwargs))
                        break
    else:
        # Run by phase/category
        if phases is None:
            phases = list(all_experiments.keys())
        
        for phase in phases:
            phase = phase.upper()
            if phase in all_experiments:
                for exp_id, cls, kwargs, quick_kwargs in all_experiments[phase]:
                    to_run.append((exp_id, cls, quick_kwargs if quick else kwargs))
    
    if not to_run:
        print("No experiments to run!")
        return {}
    
    # Run experiments
    if verbose:
        print("\n" + "="*70)
        print("SGP PAPER EXPERIMENTS")
        print("="*70)
        print(f"Device: {DEVICE}")
        print(f"Results: {RESULTS_DIR}")
        print(f"Figures: {FIGURES_DIR}")
        print(f"Quick mode: {quick}")
        print(f"Experiments to run: {len(to_run)}")
        print("="*70)
    
    results = {}
    start_time = datetime.now()
    
    for exp_id, exp_class, kwargs in to_run:
        try:
            exp = exp_class(**kwargs)
            result = exp.execute(save=save, visualize=visualize, verbose=verbose)
            results[exp_id] = result
        except Exception as e:
            print(f"\nERROR in {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = None
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Experiment':<40} {'Status':<10} {'Time':<10}")
        print("-"*70)
        
        n_pass, n_fail, n_error = 0, 0, 0
        for exp_id, result in results.items():
            if result is None:
                status = "ERROR"
                n_error += 1
                time_str = "N/A"
            elif result.success:
                status = "✓ PASS"
                n_pass += 1
                time_str = f"{result.runtime_seconds:.1f}s"
            else:
                status = "✗ FAIL"
                n_fail += 1
                time_str = f"{result.runtime_seconds:.1f}s"
            
            print(f"{exp_id:<40} {status:<10} {time_str:<10}")
        
        print("-"*70)
        print(f"Total: {len(results)} | Pass: {n_pass} | Fail: {n_fail} | Error: {n_error}")
        print(f"Total time: {elapsed:.1f}s")
        print("="*70)
    
    # Save summary
    if save:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': DEVICE,
            'quick_mode': quick,
            'total_time': elapsed,
            'results': {
                exp_id: {
                    'success': r.success if r else False,
                    'runtime': r.runtime_seconds if r else 0,
                    'metrics': r.metrics if r else {},
                } for exp_id, r in results.items()
            }
        }
        
        summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.json')
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if verbose:
            print(f"\nSummary saved to: {summary_path}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run SGP paper experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--phase', '-p', nargs='+', default=None,
                        help='Categories to run: A, B, C, D, E (default: all)')
    parser.add_argument('--exp', '-e', nargs='+', default=None,
                        help='Specific experiments: A1, B3, C7, etc.')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode with smaller grids')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--no-viz', action='store_true',
                        help='Do not generate figures')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all experiments')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    run_experiments(
        phases=args.phase,
        specific_exps=args.exp,
        quick=args.quick,
        save=not args.no_save,
        visualize=not args.no_viz,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
