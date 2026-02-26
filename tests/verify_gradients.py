#!/usr/bin/env python
"""
Gradient Verification Statistics for Paper Table
=================================================

This script computes the exact mean and max relative errors for gradient
verification, comparing implicit differentiation against finite differences.

Run from the directory containing forward_solver.py and backward_pass.py:
    python gradient_verification_stats.py

Output: LaTeX table rows with actual error statistics.

Author: Barak Gahtan
Date: January 2025
"""

import torch
import numpy as np
import sys
import os

# Import from your codebase
from eikonal.forward.pytorch import EikonalProblem, create_test_problem, fast_sweeping_solve
from eikonal.backward.implicit_diff import backward_pass

def verify_gradients_for_config(problem, n_points=20, eps=1e-5, seed=42):
    """
    Verify gradients for a given problem configuration.
    
    Returns dict with mean and max relative errors for G and B components.
    """
    np.random.seed(seed)
    
    # Solve forward
    T, _ = fast_sweeping_solve(problem, verbose=False)
    
    M, N = problem.M, problem.N
    source_i, source_j = M // 2, N // 2
    
    # Pick a target point for the loss (away from source)
    target_i, target_j = M // 2 + M // 4, N // 2 + N // 4
    
    # Loss gradient: dL/dT = 1 at target, 0 elsewhere
    dL_dT = torch.zeros_like(T)
    dL_dT[0, target_i, target_j] = 1.0
    
    # Compute gradients via implicit differentiation
    dL_dG, dL_dB = backward_pass(T, dL_dT, problem, verbose=False)
    
    # Generate random interior test points (avoid source, target, boundaries)
    interior_margin = 3
    test_points = []
    attempts = 0
    while len(test_points) < n_points and attempts < 1000:
        i = np.random.randint(interior_margin, M - interior_margin)
        j = np.random.randint(interior_margin, N - interior_margin)
        # Avoid source and target
        if (i, j) != (source_i, source_j) and (i, j) != (target_i, target_j):
            # Also avoid immediate neighbors of source
            if abs(i - source_i) > 1 or abs(j - source_j) > 1:
                test_points.append((i, j))
        attempts += 1
    
    def compute_loss(T_val):
        return (T_val * dL_dT).sum().item()
    
    results = {'G': [], 'B': []}
    
    # Test dL/dB
    for i, j in test_points:
        for c in [0, 1]:  # b1 and b2
            # Perturb B[0, c, i, j] by +eps
            B_plus = problem.B.clone()
            B_plus[0, c, i, j] += eps
            problem_plus = EikonalProblem(problem.G.clone(), B_plus, problem.source_mask.clone())
            T_plus, _ = fast_sweeping_solve(problem_plus, verbose=False)
            L_plus = compute_loss(T_plus)
            
            # Perturb by -eps
            B_minus = problem.B.clone()
            B_minus[0, c, i, j] -= eps
            problem_minus = EikonalProblem(problem.G.clone(), B_minus, problem.source_mask.clone())
            T_minus, _ = fast_sweeping_solve(problem_minus, verbose=False)
            L_minus = compute_loss(T_minus)
            
            # Finite difference gradient
            fd_grad = (L_plus - L_minus) / (2 * eps)
            computed_grad = dL_dB[0, c, i, j].item()
            
            # Compute relative error
            if abs(fd_grad) > 1e-10:
                rel_err = abs(fd_grad - computed_grad) / abs(fd_grad)
            else:
                rel_err = abs(computed_grad) if abs(computed_grad) > 1e-10 else 0.0
            
            results['B'].append(rel_err)
    
    # Test dL/dG
    for i, j in test_points:
        for c in [0, 1, 2]:  # g11, g12, g22
            G_plus = problem.G.clone()
            G_plus[0, c, i, j] += eps
            problem_plus = EikonalProblem(G_plus, problem.B.clone(), problem.source_mask.clone())
            T_plus, _ = fast_sweeping_solve(problem_plus, verbose=False)
            L_plus = compute_loss(T_plus)
            
            G_minus = problem.G.clone()
            G_minus[0, c, i, j] -= eps
            problem_minus = EikonalProblem(G_minus, problem.B.clone(), problem.source_mask.clone())
            T_minus, _ = fast_sweeping_solve(problem_minus, verbose=False)
            L_minus = compute_loss(T_minus)
            
            fd_grad = (L_plus - L_minus) / (2 * eps)
            computed_grad = dL_dG[0, c, i, j].item()
            
            if abs(fd_grad) > 1e-10:
                rel_err = abs(fd_grad - computed_grad) / abs(fd_grad)
            else:
                rel_err = abs(computed_grad) if abs(computed_grad) > 1e-10 else 0.0
            
            results['G'].append(rel_err)
    
    return results


def main():
    print("=" * 70)
    print("GRADIENT VERIFICATION STATISTICS FOR PAPER")
    print("=" * 70)
    print()
    print("Settings:")
    print("  - Finite difference step size: eps = 1e-5")
    print("  - Points tested per configuration: 20")
    print("  - Grid size: 50 x 50")
    print()
    
    # Test configurations
    configs = [
        ("Isotropic", "isotropic"),
        ("Anisotropic", "anisotropic_uniform"),
        ("Drift only", "drift_piecewise"),
    ]
    
    all_results = []
    
    for name, case in configs:
        print(f"Testing {name}...", end=" ", flush=True)
        
        # Create problem
        problem = create_test_problem(M=50, N=50, case=case)
        
        # For "Drift only", we want isotropic metric with drift
        # The drift_piecewise case already has this
        
        # Run verification
        results = verify_gradients_for_config(problem, n_points=20, eps=1e-5)
        
        # Compute statistics
        all_errors = results['G'] + results['B']
        mean_err = np.mean(all_errors) * 100  # as percentage
        max_err = np.max(all_errors) * 100
        
        all_results.append({
            'name': name,
            'case': case,
            'mean': mean_err,
            'max': max_err,
            'G_mean': np.mean(results['G']) * 100,
            'G_max': np.max(results['G']) * 100,
            'B_mean': np.mean(results['B']) * 100,
            'B_max': np.max(results['B']) * 100,
            'n_G': len(results['G']),
            'n_B': len(results['B']),
        })
        
        print(f"done. Mean: {mean_err:.4f}%, Max: {max_err:.4f}%")
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    for r in all_results:
        print(f"{r['name']}:")
        print(f"  G components ({r['n_G']} tests): Mean = {r['G_mean']:.4f}%, Max = {r['G_max']:.4f}%")
        print(f"  B components ({r['n_B']} tests): Mean = {r['B_mean']:.4f}%, Max = {r['B_max']:.4f}%")
        print(f"  Combined: Mean = {r['mean']:.4f}%, Max = {r['max']:.4f}%")
        print()
    
    print("=" * 70)
    print("LATEX TABLE (copy this to your paper)")
    print("=" * 70)
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Pointwise gradient verification against central finite differences ($\epsilon = 10^{-5}$). Mean and maximum relative errors computed over 20 randomly sampled interior points per configuration.}")
    print(r"\label{tab:gradient_accuracy}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Configuration & Components & Points Tested & Mean Rel.\ Error & Max Rel.\ Error \\")
    print(r"\midrule")
    
    for r in all_results:
        if r['name'] == 'Isotropic':
            components = r"$\mathbf{G}$, $\mathbf{b}$"
        elif r['name'] == 'Anisotropic':
            components = r"$g_{11}$, $g_{12}$, $g_{22}$"
        else:
            components = r"$b_1$, $b_2$"
        
        # Format mean error
        if r['mean'] < 0.01:
            mean_str = r"$< 0.01\%$"
        else:
            mean_str = f"${r['mean']:.2f}\\%$"
        
        # Format max error
        max_str = f"${r['max']:.2f}\\%$"
        
        print(f"{r['name']} & {components} & 20 & {mean_str} & {max_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
