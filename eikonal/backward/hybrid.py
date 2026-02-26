"""
Hybrid Backward Pass: Stencil-Aware Gradient Computation
=========================================================

This module extends the backward pass to handle spatially varying metrics
by detecting nodes near stencil boundaries and using finite differences there.

The key insight: implicit differentiation assumes a fixed stencil structure.
When perturbing parameters causes stencils to switch, the implicit gradients
are incorrect. We detect these "unstable" nodes and use FD instead.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import numpy as np

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.backward.implicit_diff import (
    identify_active_stencils, solve_adjoint_direct, compute_parameter_gradients,
    StencilInfo, UpdateType, NEIGHBOR_OFFSETS, get_stencil_pair
)


def compute_stencil_stability(T: Tensor, problem: EikonalProblem,
                              stencil_info: StencilInfo) -> Tensor:
    """
    Compute stability margin at each node.

    Stability margin = (second_best_T - best_T) / best_T

    Small margin means the node is near a stencil boundary and sensitive
    to parameter perturbations.

    Returns:
        stability: (K, M-2, N-2) tensor of stability margins (0 = unstable, large = stable)
    """
    K, M, N = T.shape
    H, W = M - 2, N - 2
    device = T.device

    # Get metric and drift at interior nodes
    G = problem.G[:, :, 1:-1, 1:-1]  # (K, 3, H, W)
    B = problem.B[:, :, 1:-1, 1:-1]  # (K, 2, H, W)
    T_interior = T[:, 1:-1, 1:-1]  # (K, H, W)

    g11, g12, g22 = G[:, 0], G[:, 1], G[:, 2]
    b1, b2 = B[:, 0], B[:, 1]

    # Compute arrival time candidates from all 8 stencils
    all_candidates = []

    for stencil_idx in range(8):
        m1, m2 = get_stencil_pair(stencil_idx)
        m1x, m1y = float(m1[0]), float(m1[1])
        m2x, m2y = float(m2[0]), float(m2[1])

        # Get neighbor arrival times
        # m1 = (dx, dy) means neighbor at (row + dy, col + dx)
        T1 = T[:, 1 + m1[1]:M - 1 + m1[1], 1 + m1[0]:N - 1 + m1[0]]
        T2 = T[:, 1 + m2[1]:M - 1 + m2[1], 1 + m2[0]:N - 1 + m2[0]]

        # Pad if necessary (boundary handling)
        if T1.shape != T_interior.shape:
            T1 = torch.full_like(T_interior, float('inf'))
        if T2.shape != T_interior.shape:
            T2 = torch.full_like(T_interior, float('inf'))

        # Compute 2-point update
        e11 = g11 * (m1x ** 2) + 2 * g12 * (m1x * m1y) + g22 * (m1y ** 2)
        e22 = g11 * (m2x ** 2) + 2 * g12 * (m2x * m2y) + g22 * (m2y ** 2)
        e12 = g11 * (m1x * m2x) + g12 * (m1x * m2y + m1y * m2x) + g22 * (m1y * m2y)

        detE = e11 * e22 - e12 ** 2
        eps = 1e-12
        det_safe = torch.clamp(detE, min=eps)

        q11 = e22 / det_safe
        q12 = -e12 / det_safe
        q22 = e11 / det_safe

        m1b = m1x * b1 + m1y * b2
        m2b = m2x * b1 + m2y * b2
        S1 = T1 + m1b
        S2 = T2 + m2b

        alpha = q11 + 2 * q12 + q22
        beta = (q11 + q12) * S1 + (q12 + q22) * S2
        gamma = q11 * S1 ** 2 + 2 * q12 * S1 * S2 + q22 * S2 ** 2 - 1.0

        disc = beta ** 2 - alpha * gamma
        disc_safe = torch.clamp(disc, min=0.0)
        alpha_safe = torch.clamp(alpha, min=eps)

        t0_2pt = (beta + torch.sqrt(disc_safe)) / alpha_safe

        # Causality check for 2-point
        lb = torch.maximum(T1, T2)
        t0_2pt = torch.maximum(t0_2pt, lb)

        # Check if inside triangle (simplified)
        u1 = S1 - t0_2pt
        u2 = S2 - t0_2pt
        Qu1 = q11 * u1 + q12 * u2
        Qu2 = q12 * u1 + q22 * u2
        valid_2pt = (Qu1 >= -eps) & (Qu2 >= -eps) & (detE > eps) & (T1 < T_interior) & (T2 < T_interior)

        INF = torch.tensor(1e10, device=device)
        t0_2pt = torch.where(valid_2pt, t0_2pt, INF)

        # Compute 1-point updates
        t0_1pt_m1 = T1 + m1b + torch.sqrt(torch.clamp(e11, min=0.0))
        t0_1pt_m2 = T2 + m2b + torch.sqrt(torch.clamp(e22, min=0.0))

        # Apply causality
        t0_1pt_m1 = torch.where(T1 < T_interior, t0_1pt_m1, INF)
        t0_1pt_m2 = torch.where(T2 < T_interior, t0_1pt_m2, INF)

        # Best for this stencil
        t0_stencil = torch.minimum(t0_2pt, torch.minimum(t0_1pt_m1, t0_1pt_m2))
        all_candidates.append(t0_stencil)

    # Stack all candidates: (8, K, H, W)
    candidates = torch.stack(all_candidates, dim=0)

    # Find best and second-best
    sorted_candidates, _ = torch.sort(candidates, dim=0)
    best = sorted_candidates[0]  # (K, H, W)
    second_best = sorted_candidates[1]  # (K, H, W)

    # Stability margin
    # Avoid division by zero
    best_safe = torch.clamp(torch.abs(best), min=1e-10)
    stability = (second_best - best) / best_safe

    # Clamp to reasonable range
    stability = torch.clamp(stability, min=0.0, max=10.0)

    return stability


def identify_critical_unstable_nodes(
        stability: Tensor,
        lam: Tensor,
        stability_threshold: float = 0.01,
        lambda_threshold: float = 0.01,
) -> Tensor:
    """
    Identify nodes that are both unstable AND have significant adjoint value.

    These are the "critical" nodes where we need FD instead of implicit diff.

    Args:
        stability: (K, M-2, N-2) stability margins
        lam: (K, M, N) adjoint values (with boundary padding)
        stability_threshold: nodes with margin below this are unstable
        lambda_threshold: relative threshold for significant λ

    Returns:
        critical_mask: (K, M-2, N-2) boolean mask of critical nodes
    """
    K = stability.shape[0]

    # Get interior λ
    lam_interior = lam[:, 1:-1, 1:-1]  # (K, M-2, N-2)

    # Normalize λ by max
    lam_max = torch.abs(lam_interior).max()
    if lam_max < 1e-10:
        # No significant adjoint, nothing is critical
        return torch.zeros_like(stability, dtype=torch.bool)

    lam_normalized = torch.abs(lam_interior) / lam_max

    # Critical = unstable AND significant λ
    unstable = stability < stability_threshold
    significant = lam_normalized > lambda_threshold
    critical = unstable & significant

    return critical


def compute_fd_gradient_at_node(
        problem: EikonalProblem,
        T: Tensor,
        dL_dT: Tensor,
        i: int, j: int, k: int = 0,
        eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    """
    Compute gradient dL/dG and dL/dB at a single node using finite differences.

    Returns:
        dL_dG: (3,) gradient w.r.t. [g11, g12, g22] at (k, i, j)
        dL_dB: (2,) gradient w.r.t. [b1, b2] at (k, i, j)
    """
    device = T.device

    # Compute base loss
    # Loss = sum of dL_dT * T at all nodes
    # But since we're perturbing one node, we just need T at target points
    target_mask = dL_dT != 0

    def compute_loss(prob):
        T_new, _ = fast_sweeping_solve(prob, verbose=False)
        return (dL_dT * T_new).sum().item()

    dL_dG = torch.zeros(3, device=device)
    dL_dB = torch.zeros(2, device=device)

    # dL/dG
    for c in range(3):
        G_plus = problem.G.clone()
        G_plus[k, c, i, j] += eps
        G_minus = problem.G.clone()
        G_minus[k, c, i, j] -= eps

        L_plus = compute_loss(EikonalProblem(G_plus, problem.B, problem.source_mask))
        L_minus = compute_loss(EikonalProblem(G_minus, problem.B, problem.source_mask))

        dL_dG[c] = (L_plus - L_minus) / (2 * eps)

    # dL/dB
    for c in range(2):
        B_plus = problem.B.clone()
        B_plus[k, c, i, j] += eps
        B_minus = problem.B.clone()
        B_minus[k, c, i, j] -= eps

        L_plus = compute_loss(EikonalProblem(problem.G, B_plus, problem.source_mask))
        L_minus = compute_loss(EikonalProblem(problem.G, B_minus, problem.source_mask))

        dL_dB[c] = (L_plus - L_minus) / (2 * eps)

    return dL_dG, dL_dB


def backward_pass_hybrid(
        T: Tensor,
        dL_dT: Tensor,
        problem: EikonalProblem,
        stability_threshold: float = 0.01,
        lambda_threshold: float = 0.01,
        max_fd_nodes: int = 100,
        verbose: bool = False,
) -> Tuple[Tensor, Tensor, dict]:
    """
    Hybrid backward pass: implicit differentiation + FD for unstable nodes.

    Args:
        T: (K, M, N) arrival time field from forward solve
        dL_dT: (K, M, N) gradient of loss w.r.t. T
        problem: EikonalProblem with G, B, source_mask
        stability_threshold: margin below which node is considered unstable
        lambda_threshold: relative λ below which node is not critical
        max_fd_nodes: maximum number of nodes to use FD for (computational limit)
        verbose: print diagnostic information

    Returns:
        dL_dG: (K, 3, M, N) gradient w.r.t. metric G
        dL_dB: (K, 2, M, N) gradient w.r.t. drift B
        info: dict with diagnostic information
    """
    K, M, N = T.shape
    device = T.device

    # Step 1: Identify active stencils
    if verbose:
        print("Step 1: Identifying active stencils...")
    stencil_info = identify_active_stencils(T, problem)

    # Step 2: Compute stability margins
    if verbose:
        print("Step 2: Computing stability margins...")
    stability = compute_stencil_stability(T, problem, stencil_info)

    # Step 3: Solve adjoint system
    if verbose:
        print("Step 3: Solving adjoint system...")
    lam = solve_adjoint_direct(stencil_info, dL_dT, problem, T, verbose=False)

    # Step 4: Identify critical unstable nodes
    if verbose:
        print("Step 4: Identifying critical unstable nodes...")
    critical_mask = identify_critical_unstable_nodes(
        stability, lam, stability_threshold, lambda_threshold
    )

    n_critical = critical_mask.sum().item()
    n_unstable = (stability < stability_threshold).sum().item()

    if verbose:
        print(f"   Total interior nodes: {(M - 2) * (N - 2)}")
        print(f"   Unstable nodes (margin < {stability_threshold}): {n_unstable}")
        print(f"   Critical nodes (unstable + significant λ): {n_critical}")

    # Step 5: Compute gradients using implicit differentiation
    if verbose:
        print("Step 5: Computing gradients via implicit differentiation...")
    dL_dG, dL_dB = compute_parameter_gradients(stencil_info, lam, problem)

    # Step 6: Override with FD for critical nodes
    if n_critical > 0:
        if verbose:
            print(f"Step 6: Computing FD gradients for {min(n_critical, max_fd_nodes)} critical nodes...")

        # Get indices of critical nodes
        critical_indices = torch.nonzero(critical_mask, as_tuple=False)

        # Limit to max_fd_nodes
        if len(critical_indices) > max_fd_nodes:
            # Sort by lambda magnitude and take top ones
            lam_interior = lam[:, 1:-1, 1:-1]
            lam_at_critical = torch.abs(lam_interior[critical_mask])
            _, top_idx = torch.topk(lam_at_critical, max_fd_nodes)
            critical_indices = critical_indices[top_idx]

            if verbose:
                print(f"   Limited to top {max_fd_nodes} by λ magnitude")

        # Compute FD for each critical node
        for idx in critical_indices:
            k_idx = idx[0].item()
            i_int = idx[1].item()  # interior index
            j_int = idx[2].item()

            i = i_int + 1  # full grid index
            j = j_int + 1

            fd_dG, fd_dB = compute_fd_gradient_at_node(
                problem, T, dL_dT, i, j, k_idx
            )

            # Override implicit diff values with FD values
            dL_dG[k_idx, :, i, j] = fd_dG
            dL_dB[k_idx, :, i, j] = fd_dB
    else:
        if verbose:
            print("Step 6: No critical nodes, skipping FD.")

    # Collect diagnostic info
    info = {
        'n_unstable': n_unstable,
        'n_critical': n_critical,
        'stability': stability,
        'critical_mask': critical_mask,
    }

    if verbose:
        print("Done.")

    return dL_dG, dL_dB, info


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from eikonal.forward.pytorch import create_test_problem
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("TESTING HYBRID BACKWARD PASS")
    print("=" * 70)

    # Test on varying metric case
    case = "anisotropic_varying"
    target = (70, 70)
    test_points = [(55, 55), (60, 60), (65, 65)]

    print(f"\nTest case: {case}")
    print(f"Target: {target}")

    problem = create_test_problem(M=100, N=100, case=case)
    T, _ = fast_sweeping_solve(problem, verbose=False)

    ti, tj = target
    dL_dT = torch.zeros_like(T)
    dL_dT[0, ti, tj] = 1.0

    # Run hybrid backward pass
    print("\n--- HYBRID BACKWARD PASS ---")
    dL_dG_hybrid, dL_dB_hybrid, info = backward_pass_hybrid(
        T, dL_dT, problem,
        stability_threshold=0.05,
        lambda_threshold=0.01,
        max_fd_nodes=200,
        verbose=True
    )

    # Run original backward pass for comparison
    print("\n--- ORIGINAL BACKWARD PASS ---")
    from eikonal.backward.implicit_diff import backward_pass

    dL_dG_orig, dL_dB_orig = backward_pass(T, dL_dT, problem, verbose=False)

    # Verify against FD
    print("\n--- FINITE DIFFERENCE VERIFICATION ---")
    eps = 1e-5


    def compute_loss(prob):
        T_new, _ = fast_sweeping_solve(prob, verbose=False)
        return T_new[0, ti, tj].item()


    print(f"\n{'Point':<15} {'FD':<12} {'Original':<12} {'Hybrid':<12} {'Orig Err':<12} {'Hybrid Err':<12}")
    print("-" * 75)

    for (pi, pj) in test_points:
        # FD for g11
        G_plus = problem.G.clone()
        G_plus[0, 0, pi, pj] += eps
        G_minus = problem.G.clone()
        G_minus[0, 0, pi, pj] -= eps

        fd = (compute_loss(EikonalProblem(G_plus, problem.B, problem.source_mask)) -
              compute_loss(EikonalProblem(G_minus, problem.B, problem.source_mask))) / (2 * eps)

        orig = dL_dG_orig[0, 0, pi, pj].item()
        hybrid = dL_dG_hybrid[0, 0, pi, pj].item()

        orig_err = abs(orig - fd) / abs(fd) if abs(fd) > 1e-10 else abs(orig)
        hybrid_err = abs(hybrid - fd) / abs(fd) if abs(fd) > 1e-10 else abs(hybrid)

        print(f"g11[{pi},{pj}]    {fd:<12.4f} {orig:<12.4f} {hybrid:<12.4f} {orig_err:<12.2%} {hybrid_err:<12.2%}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Stability and critical nodes
    ax = axes[0, 0]
    im = ax.imshow(info['stability'][0].numpy(), cmap='viridis')
    ax.set_title('Stability Margin\n(low = unstable)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(info['critical_mask'][0].numpy().astype(float), cmap='Reds')
    ax.set_title(f'Critical Nodes (n={info["n_critical"]})\n(unstable + significant λ)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    lam = solve_adjoint_direct(identify_active_stencils(T, problem), dL_dT, problem, T)
    im = ax.imshow(lam[0, 1:-1, 1:-1].numpy())
    ax.set_title('Adjoint λ')
    plt.colorbar(im, ax=ax)

    # Row 2: Gradient comparison
    ax = axes[1, 0]
    im = ax.imshow(dL_dG_orig[0, 0, 1:-1, 1:-1].numpy(), cmap='RdBu_r')
    ax.set_title('Original dL/dg11')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(dL_dG_hybrid[0, 0, 1:-1, 1:-1].numpy(), cmap='RdBu_r')
    ax.set_title('Hybrid dL/dg11')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    diff = dL_dG_hybrid[0, 0, 1:-1, 1:-1] - dL_dG_orig[0, 0, 1:-1, 1:-1]
    im = ax.imshow(diff.numpy(), cmap='RdBu_r')
    ax.set_title('Difference (Hybrid - Original)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('hybrid_backward_pass_test.png', dpi=150)
    print("\nSaved 'hybrid_backward_pass_test.png'")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)