import numpy as np
import torch
from torch import Tensor
from typing import Tuple
from dataclasses import dataclass


@dataclass
class EikonalProblem:
    """
    Defines a Randers-Finsler eikonal problem on a 2D Cartesian grid.

    The PDE being solved is:
        (∇T - b)ᵀ A (∇T - b) = 1,  where A = G⁻¹

    This describes wavefront propagation where:
    - G(x) is a Riemannian metric (symmetric positive-definite 2×2 matrix)
    - b(x) is a drift vector that biases the propagation direction
    - T(x) is the arrival time (what we solve for)

    Attributes:
        G: Metric tensor field, shape (K, 3, M, N)
           Stores [g11, g12, g22] for symmetric matrix [[g11, g12], [g12, g22]]
        B: Drift vector field, shape (K, 2, M, N)
           Stores [b1, b2]
        source_mask: Boolean mask, shape (K, M, N)
           True at source points where T = 0
        M, N: Grid dimensions (rows, columns)
        K: Batch dimension (number of independent problems)
    """
    G: Tensor  # (K, 3, M, N) metric
    B: Tensor  # (K, 2, M, N) drift
    source_mask: Tensor  # (K, M, N) bool

    @property
    def K(self) -> int:
        return self.G.shape[0]

    @property
    def M(self) -> int:
        return self.G.shape[2]

    @property
    def N(self) -> int:
        return self.G.shape[3]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.K, self.M, self.N)

    def initialize_arrival_times(self, inf_value: float = 1e6) -> Tensor:
        """
        Create initial arrival time field.
        T = 0 at source points, T = inf elsewhere.
        """
        T = torch.full(self.shape, inf_value, dtype=self.G.dtype, device=self.G.device)
        T[self.source_mask] = 0.0
        return T


# =============================================================================
# TEST PROBLEM FACTORY
# =============================================================================

def create_test_problem(
        M: int = 100,
        N: int = 100,
        case: str = "isotropic",
        device: str = "cpu",
) -> EikonalProblem:
    """
    Create test problems for the eikonal solver.

    Cases:
        "isotropic": Identity metric, no drift (standard Eikonal)
        "drift_piecewise": Identity metric with piecewise constant drift
        "drift_gaussian": Identity metric with Gaussian-shaped drift
        "drift_circular": Identity metric with circular drift pattern
        "anisotropic_uniform": Uniform anisotropic metric (rotated ellipse)
        "anisotropic_piecewise": Piecewise anisotropic metric
        "anisotropic_varying": Smoothly varying anisotropic metric
    """
    # Source at center
    source_mask = torch.zeros(1, M, N, dtype=torch.bool, device=device)
    source_mask[0, M // 2, N // 2] = True

    # Coordinate grids for defining spatially-varying fields
    X = torch.linspace(0, N - 1, N, device=device)[None, None, :]  # (1, 1, N)
    Y = torch.linspace(0, M - 1, M, device=device)[None, :, None]  # (1, M, 1)

    # Default: identity metric (isotropic)
    G = torch.stack([
        torch.ones(M, N, device=device),  # g11
        torch.zeros(M, N, device=device),  # g12
        torch.ones(M, N, device=device),  # g22
    ], dim=0).unsqueeze(0)  # (1, 3, M, N)

    # Default: zero drift
    B = torch.zeros(1, 2, M, N, device=device)

    if case == "isotropic":
        pass  # Use defaults

    elif case == "drift_piecewise":
        B[0, 0, :M // 3, :] = -0.1
        B[0, 1, :M // 3, :] = 0.1
        B[0, 0, M // 3:2 * M // 3, :] = -0.1 * np.sqrt(2)
        B[0, 1, M // 3:2 * M // 3, :] = 0.0
        B[0, 0, 2 * M // 3:, :] = -0.05
        B[0, 1, 2 * M // 3:, :] = -0.2

    elif case == "drift_gaussian":
        y_normalized = (Y - 0.5 * M) / M * 0.5
        B[0, 0, :, :] = (torch.exp(-10 * y_normalized ** 2) + y_normalized - 0.5).squeeze(0)

    elif case == "drift_circular":
        y_normalized = (Y - 0.5 * M) / M * 0.5
        x_normalized = (X - 0.5 * N) / N * 0.5 * np.pi
        magnitude = 0.5 * torch.exp(-10 * y_normalized ** 2)
        B[0, 0, :, :] = (magnitude * torch.cos(x_normalized)).squeeze(0)
        B[0, 1, :, :] = (magnitude * torch.sin(x_normalized)).squeeze(0)

    elif case == "anisotropic_uniform":
        a, b = 2.0, 0.5
        theta = np.deg2rad(-45)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        S = np.diag([a, b])
        G_matrix = R.T @ S @ R

        G = torch.stack([
            torch.full((M, N), G_matrix[0, 0], device=device),
            torch.full((M, N), G_matrix[0, 1], device=device),
            torch.full((M, N), G_matrix[1, 1], device=device),
        ], dim=0).unsqueeze(0)

    elif case == "anisotropic_piecewise":
        a, b = 2.0, 0.5
        theta = np.deg2rad(-45)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        S = np.diag([a, b])
        G_aniso = R.T @ S @ R

        G = torch.stack([
            torch.ones(M, N, device=device),
            torch.zeros(M, N, device=device),
            torch.ones(M, N, device=device),
        ], dim=0).unsqueeze(0)

        G[0, 0, :M // 2, :N // 2] = 5 * G_aniso[0, 0]
        G[0, 1, :M // 2, :N // 2] = 5 * G_aniso[0, 1]
        G[0, 2, :M // 2, :N // 2] = 5 * G_aniso[1, 1]

        G[0, 0, M // 2:, :N // 2] = 5.0
        G[0, 2, M // 2:, :N // 2] = 5.0

        G[0, 0, M // 2:, N // 2:] = G_aniso[0, 0]
        G[0, 1, M // 2:, N // 2:] = G_aniso[0, 1]
        G[0, 2, M // 2:, N // 2:] = G_aniso[1, 1]

    elif case == "anisotropic_varying":
        y_scale = ((Y - 0.5 * M) / M * 0.5 + 0.5) + 0.1

        G = torch.zeros(1, 3, M, N, device=device)
        for i in range(M):
            theta = ((i - 0.5 * N) / N * 0.75 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, s], [-s, c]])
            S = np.diag([2.0, 0.5])
            G_matrix = R.T @ S @ R
            G[0, 0, i, :] = G_matrix[0, 0]
            G[0, 1, i, :] = G_matrix[0, 1]
            G[0, 2, i, :] = G_matrix[1, 1]

        G = G * y_scale.unsqueeze(1)

    else:
        raise ValueError(f"Unknown case: {case}")

    return EikonalProblem(G=G, B=B, source_mask=source_mask)


# =============================================================================
# LOCAL UPDATE FORMULAS
# =============================================================================

def _sign(x: int) -> int:
    """Return -1 if x < 0, else +1."""
    return -1 if x < 0 else +1


@torch.no_grad()
def compute_two_point_update(
        T1: Tensor,
        T2: Tensor,
        g11: Tensor,
        g12: Tensor,
        g22: Tensor,
        b1: Tensor,
        b2: Tensor,
        m1: Tuple[int, int],
        m2: Tuple[int, int],
        eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the 2-point (planar wavefront) update.

    Solves: (S - t0·1)ᵀ Q (S - t0·1) = 1
    where E = Mᵀ G M, Q = E⁻¹, S = [T1 + m1·b, T2 + m2·b]ᵀ

    Returns:
        t0_2pt: Computed arrival time
        valid: Boolean mask where update is valid (causal + inside triangle)
    """
    m1x, m1y = float(m1[0]), float(m1[1])
    m2x, m2y = float(m2[0]), float(m2[1])

    # E = Mᵀ G M
    e11 = g11 * (m1x ** 2) + 2 * g12 * (m1x * m1y) + g22 * (m1y ** 2)
    e22 = g11 * (m2x ** 2) + 2 * g12 * (m2x * m2y) + g22 * (m2y ** 2)
    e12 = g11 * (m1x * m2x) + g12 * (m1x * m2y + m1y * m2x) + g22 * (m1y * m2y)

    # Q = E⁻¹
    detE = e11 * e22 - e12 ** 2
    det_safe = torch.clamp(detE, min=eps)
    q11 = e22 / det_safe
    q12 = -e12 / det_safe
    q22 = e11 / det_safe

    # Drift-adjusted arrival times
    m1_dot_b = m1x * b1 + m1y * b2
    m2_dot_b = m2x * b1 + m2y * b2
    S1 = T1 + m1_dot_b
    S2 = T2 + m2_dot_b

    # Solve quadratic
    alpha = q11 + 2 * q12 + q22
    beta = (q11 + q12) * S1 + (q12 + q22) * S2
    gamma = q11 * S1 ** 2 + 2 * q12 * S1 * S2 + q22 * S2 ** 2 - 1.0

    discriminant = beta ** 2 - alpha * gamma
    disc_safe = torch.clamp(discriminant, min=0.0)
    alpha_safe = torch.clamp(alpha, min=eps)

    t0_2pt = (beta + torch.sqrt(disc_safe)) / alpha_safe

    # Causality
    causality_bound = torch.maximum(T1, T2)
    t0_2pt = torch.maximum(t0_2pt, causality_bound)

    # Barycentric test
    lam1 = q11 * (t0_2pt - S1) + q12 * (t0_2pt - S2)
    lam2 = q12 * (t0_2pt - S1) + q22 * (t0_2pt - S2)
    inside_triangle = (lam1 >= 0) & (lam2 >= 0)
    valid = (detE > eps) & (alpha > eps) & torch.isfinite(t0_2pt) & inside_triangle

    return t0_2pt, valid


@torch.no_grad()
def compute_one_point_update(
        T_nbr: Tensor,
        g11: Tensor,
        g12: Tensor,
        g22: Tensor,
        b1: Tensor,
        b2: Tensor,
        m: Tuple[int, int],
) -> Tensor:
    """
    Compute the 1-point (edge) update: t0 = T_nbr + m·b + √(mᵀ G m)
    """
    mx, my = float(m[0]), float(m[1])
    m_G_m = g11 * (mx ** 2) + 2 * g12 * (mx * my) + g22 * (my ** 2)
    distance = torch.sqrt(torch.clamp(m_G_m, min=0.0))
    m_dot_b = mx * b1 + my * b2
    return T_nbr + distance + m_dot_b


@torch.no_grad()
def stencil_update(
        T1: Tensor,
        T2: Tensor,
        T0: Tensor,
        g11: Tensor,
        g12: Tensor,
        g22: Tensor,
        b1: Tensor,
        b2: Tensor,
        m1: Tuple[int, int],
        m2: Tuple[int, int],
        eps: float = 1e-12,
) -> Tensor:
    """
    Compute update for one triangular stencil.
    Returns: T0_new = min{T0, t0^(2pt), t0^(1pt_m1), t0^(1pt_m2)}
    """
    t0_2pt, valid_2pt = compute_two_point_update(
        T1, T2, g11, g12, g22, b1, b2, m1, m2, eps
    )
    t0_1pt_m1 = compute_one_point_update(T1, g11, g12, g22, b1, b2, m1)
    t0_1pt_m2 = compute_one_point_update(T2, g11, g12, g22, b1, b2, m2)

    INF = torch.tensor(float('inf'), device=T0.device)
    t0_2pt_effective = torch.where(valid_2pt, t0_2pt, INF)
    t0_candidate = torch.minimum(t0_2pt_effective, torch.minimum(t0_1pt_m1, t0_1pt_m2))

    return torch.minimum(T0, t0_candidate)


# =============================================================================
# SWEEP FUNCTIONS
# =============================================================================

@torch.no_grad()
def sweep_column(
        T: Tensor,
        G: Tensor,
        B: Tensor,
        j: int,
        direction: int,
):
    """Update column j using information from adjacent column."""
    p = direction
    j_nbr = j + _sign(p)

    g11 = G[:, 0, 1:-1, j]
    g12 = G[:, 1, 1:-1, j]
    g22 = G[:, 2, 1:-1, j]
    b1 = B[:, 0, 1:-1, j]
    b2 = B[:, 1, 1:-1, j]
    T0 = T[:, 1:-1, j]

    # Upper neighbor stencil
    T1_up = T[:, 0:-2, j_nbr]
    T2_up = T[:, 1:-1, j_nbr]
    T_updated = stencil_update(T1_up, T2_up, T0, g11, g12, g22, b1, b2, (p, -1), (p, 0))

    # Lower neighbor stencil
    T1_down = T[:, 2:, j_nbr]
    T2_down = T[:, 1:-1, j_nbr]
    T_updated = stencil_update(T1_down, T2_down, T_updated, g11, g12, g22, b1, b2, (p, +1), (p, 0))

    T[:, 1:-1, j] = T_updated


@torch.no_grad()
def sweep_row(
        T: Tensor,
        G: Tensor,
        B: Tensor,
        i: int,
        direction: int,
):
    """Update row i using information from adjacent row."""
    p = direction
    i_nbr = i + _sign(p)

    g11 = G[:, 0, i, 1:-1]
    g12 = G[:, 1, i, 1:-1]
    g22 = G[:, 2, i, 1:-1]
    b1 = B[:, 0, i, 1:-1]
    b2 = B[:, 1, i, 1:-1]
    T0 = T[:, i, 1:-1]

    # Left neighbor stencil
    T1_left = T[:, i_nbr, 0:-2]
    T2_left = T[:, i_nbr, 1:-1]
    T_updated = stencil_update(T1_left, T2_left, T0, g11, g12, g22, b1, b2, (-1, p), (0, p))

    # Right neighbor stencil
    T1_right = T[:, i_nbr, 2:]
    T2_right = T[:, i_nbr, 1:-1]
    T_updated = stencil_update(T1_right, T2_right, T_updated, g11, g12, g22, b1, b2, (+1, p), (0, p))

    T[:, i, 1:-1] = T_updated


# =============================================================================
# MAIN SOLVER
# =============================================================================

@torch.no_grad()
def fast_sweeping_solve(
        problem: EikonalProblem,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
) -> Tuple[Tensor, int]:
    """
    Solve the Randers-Finsler eikonal equation using fast sweeping.

    Returns:
        T: Converged arrival time field, shape (K, M, N)
        n_iter: Number of iterations performed
    """
    G, B = problem.G, problem.B
    K, M, N = problem.shape

    T = problem.initialize_arrival_times()

    for iteration in range(max_iter):
        T_prev = T.clone()

        # Four sweeps covering all quadrant directions
        for j in range(1, N - 1):
            sweep_column(T, G, B, j, direction=-1)
        for i in range(1, M - 1):
            sweep_row(T, G, B, i, direction=-1)
        for j in range(N - 2, 0, -1):
            sweep_column(T, G, B, j, direction=+1)
        for i in range(M - 2, 0, -1):
            sweep_row(T, G, B, i, direction=+1)

        max_change = torch.abs(T - T_prev).max().item()

        if verbose:
            print(f"Iteration {iteration + 1}: max change = {max_change:.2e}")

        if max_change <= tol:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            return T, iteration + 1

    if verbose:
        print(f"Warning: did not converge after {max_iter} iterations")

    return T, max_iter


# =============================================================================
# VISUALIZATION
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_solution(
        T: Tensor,
        G: Tensor,
        B: Tensor,
        k: int = 0,
        contour_step: int = 10,
        glyph_step: int = 5,
        figsize: Tuple[int, int] = (10, 8),
):
    """Visualize the eikonal solution with metric ellipses and drift vectors."""
    M, N = T.shape[1], T.shape[2]

    fig, ax = plt.subplots(figsize=figsize)

    T_plot = T[k].cpu().numpy()
    T_plot = np.where(np.isinf(T_plot), np.nan, T_plot)
    im = ax.imshow(T_plot, origin='upper')
    plt.colorbar(im, ax=ax, label='Arrival time T(x)')

    max_T = np.nanmax(T_plot)
    if max_T > 0:
        levels = np.arange(0, max_T, contour_step)
        ax.contour(T_plot, levels=levels, colors='black', linewidths=0.5)

    X, Y = np.arange(N), np.arange(M)
    B_np = B[k].cpu().numpy()
    ax.quiver(X[::glyph_step], Y[::glyph_step],
              B_np[0, ::glyph_step, ::glyph_step],
              B_np[1, ::glyph_step, ::glyph_step],
              color='red', alpha=0.7, scale=5)

    G_np = G[k].cpu().numpy()
    for i in range(0, M, glyph_step):
        for j in range(0, N, glyph_step):
            g = G_np[:, i, j]
            G_mat = np.array([[g[0], g[1]], [g[1], g[2]]])
            try:
                vals, vecs = np.linalg.eigh(np.linalg.inv(G_mat))
                r = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
                scale = 0.8
                rx, ry = scale * r[1], scale * r[0]
                major_vec = vecs[:, 1] if r[1] >= r[0] else vecs[:, 0]
                angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
                ellipse = Ellipse((j, i), width=2 * rx, height=2 * ry, angle=angle,
                                  fill=False, edgecolor='blue', linewidth=0.5, alpha=0.6)
                ax.add_patch(ellipse)
            except:
                pass

    ax.set_xlabel('x (column)')
    ax.set_ylabel('y (row)')
    ax.set_title('Eikonal Solution')

    return fig, ax


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    problem = create_test_problem(M=100, N=100, case="isotropic")
    T, n_iter = fast_sweeping_solve(problem, max_iter=20, tol=1e-6, verbose=True)

    print(f"\nSolution statistics:")
    T_interior = T[0, 1:-1, 1:-1]
    print(f"  Interior range: [{T_interior.min().item():.2f}, {T_interior.max().item():.2f}]")
    print(f"  T at source: {T[0, 50, 50].item():.2f}")
    print(f"  T at (50,60): {T[0, 50, 60].item():.2f} (expected: 10.00)")
    print(f"  T at (60,60): {T[0, 60, 60].item():.2f} (expected: {np.sqrt(200):.2f})")