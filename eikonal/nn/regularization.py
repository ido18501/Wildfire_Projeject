"""
Total Variation Regularization for Randers-Finsler Metric Fields
================================================================

Implements both Frobenius and Log-Euclidean TV regularization for:
- SPD tensor fields G (metric tensor)
- Vector fields B (drift)

The Log-Euclidean metric respects the Riemannian structure of the SPD manifold
and provides rotation-invariant regularization.

For 2×2 SPD matrix G with eigendecomposition G = U D U^T:
    log(G) = U log(D) U^T

Since G is projected to have eigenvalues >= eps, log(D) is always well-defined.

Author: Barak Gahtan
Date: January 2025
"""

import torch
from torch import Tensor
from .config import TVMetricType


# =============================================================================
# FROBENIUS NORM TV (NOT rotation invariant)
# =============================================================================

def tv_spd_frobenius(G: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Total Variation for SPD tensor field using Frobenius norm.

    G: (B, 3, M, N) representing [g11, g12, g22]

    ||dG||²_F = dg11² + 2·dg12² + dg22²

    Note: NOT rotation invariant. Use Log-Euclidean for rotation invariance.
    """
    dG_dx = G[:, :, :, 1:] - G[:, :, :, :-1]
    dG_dy = G[:, :, 1:, :] - G[:, :, :-1, :]

    norm_sq_dx = dG_dx[:, 0]**2 + 2*dG_dx[:, 1]**2 + dG_dx[:, 2]**2
    norm_sq_dy = dG_dy[:, 0]**2 + 2*dG_dy[:, 1]**2 + dG_dy[:, 2]**2

    tv_dx = torch.sqrt(norm_sq_dx + eps)
    tv_dy = torch.sqrt(norm_sq_dy + eps)

    return tv_dx.sum() + tv_dy.sum()


# =============================================================================
# LOG-EUCLIDEAN TV (rotation invariant)
# =============================================================================

def matrix_log_2x2(G: Tensor) -> Tensor:
    """
    Closed-form matrix logarithm for 2×2 SPD matrices.

    For G = U D U^T with D = diag(λ1, λ2):
        log(G) = U log(D) U^T

    The eigenvector matrix U is a rotation by angle θ where:
        θ = 0.5 * atan2(2*g12, g11 - g22)

    Reconstruction:
        log(G)_11 = log(λ1)*cos²θ + log(λ2)*sin²θ
        log(G)_12 = (log(λ1) - log(λ2))*cosθ*sinθ
        log(G)_22 = log(λ1)*sin²θ + log(λ2)*cos²θ

    Args:
        G: (B, 3, M, N) representing [g11, g12, g22]
           Must have positive eigenvalues (guaranteed by projection)

    Returns:
        (B, 3, M, N) representing [log(G)_11, log(G)_12, log(G)_22]
    """
    g11 = G[:, 0:1]
    g12 = G[:, 1:2]
    g22 = G[:, 2:3]

    # Eigenvalues: λ = (tr ± √((g11-g22)² + 4g12²)) / 2
    trace = g11 + g22
    diff = g11 - g22
    disc = torch.sqrt(diff**2 + 4*g12**2 + 1e-12)

    lambda1 = (trace + disc) / 2
    lambda2 = (trace - disc) / 2

    # Safety clamp - projection should guarantee λ >= eps, but defend against edge cases
    lambda1 = torch.clamp(lambda1, min=1e-6)
    lambda2 = torch.clamp(lambda2, min=1e-6)

    log_lambda1 = torch.log(lambda1)
    log_lambda2 = torch.log(lambda2)

    # Principal angle of eigenvector
    theta = 0.5 * torch.atan2(2*g12, diff + 1e-12)
    c = torch.cos(theta)
    s = torch.sin(theta)

    # Reconstruct log(G) = U log(D) U^T
    log_g11 = log_lambda1 * c**2 + log_lambda2 * s**2
    log_g12 = (log_lambda1 - log_lambda2) * c * s
    log_g22 = log_lambda1 * s**2 + log_lambda2 * c**2

    return torch.cat([log_g11, log_g12, log_g22], dim=1)


def tv_spd_log_euclidean(G: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Total Variation for SPD tensor field using Log-Euclidean metric.

    d(G1, G2) = ||log(G1) - log(G2)||_F

    This is rotation invariant and respects the SPD manifold geometry.

    Args:
        G: (B, 3, M, N) representing [g11, g12, g22]
        eps: Small constant for numerical stability in TV norm
    """
    log_G = matrix_log_2x2(G)

    dlogG_dx = log_G[:, :, :, 1:] - log_G[:, :, :, :-1]
    dlogG_dy = log_G[:, :, 1:, :] - log_G[:, :, :-1, :]

    norm_sq_dx = dlogG_dx[:, 0]**2 + 2*dlogG_dx[:, 1]**2 + dlogG_dx[:, 2]**2
    norm_sq_dy = dlogG_dy[:, 0]**2 + 2*dlogG_dy[:, 1]**2 + dlogG_dy[:, 2]**2

    tv_dx = torch.sqrt(norm_sq_dx + eps)
    tv_dy = torch.sqrt(norm_sq_dy + eps)

    return tv_dx.sum() + tv_dy.sum()


# =============================================================================
# VECTOR FIELD TV
# =============================================================================

def tv_vector_field(B: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Total Variation for vector field using Euclidean norm.

    B: (B, 2, M, N)
    """
    dB_dx = B[:, :, :, 1:] - B[:, :, :, :-1]
    dB_dy = B[:, :, 1:, :] - B[:, :, :-1, :]

    norm_sq_dx = dB_dx[:, 0]**2 + dB_dx[:, 1]**2
    norm_sq_dy = dB_dy[:, 0]**2 + dB_dy[:, 1]**2

    tv_dx = torch.sqrt(norm_sq_dx + eps)
    tv_dy = torch.sqrt(norm_sq_dy + eps)

    return tv_dx.sum() + tv_dy.sum()


# =============================================================================
# COMBINED REGULARIZATION
# =============================================================================

def compute_tv_regularization(
    G: Tensor,
    B: Tensor,
    lambda_G: float,
    lambda_B: float,
    metric_type: TVMetricType = TVMetricType.LOG_EUCLIDEAN
) -> Tensor:
    """
    Compute total TV regularization for G and B fields.

    Args:
        G: Metric tensor field (B, 3, M, N)
        B: Drift vector field (B, 2, M, N)
        lambda_G: Weight for G regularization
        lambda_B: Weight for B regularization
        metric_type: FROBENIUS or LOG_EUCLIDEAN

    Returns:
        Scalar regularization loss (differentiable)
    """
    if lambda_G == 0 and lambda_B == 0:
        return torch.tensor(0.0, device=G.device)

    if metric_type == TVMetricType.LOG_EUCLIDEAN:
        tv_G = tv_spd_log_euclidean(G)
    else:
        tv_G = tv_spd_frobenius(G)

    tv_B = tv_vector_field(B)

    return lambda_G * tv_G + lambda_B * tv_B


def compute_tv_components(
    G: Tensor,
    B: Tensor,
    metric_type: TVMetricType = TVMetricType.LOG_EUCLIDEAN
) -> dict:
    """
    Compute individual TV components for logging.

    Returns:
        Dictionary with 'tv_G' and 'tv_B' values
    """
    if metric_type == TVMetricType.LOG_EUCLIDEAN:
        tv_G = tv_spd_log_euclidean(G).item()
    else:
        tv_G = tv_spd_frobenius(G).item()

    tv_B = tv_vector_field(B).item()

    return {'tv_G': tv_G, 'tv_B': tv_B}


# =============================================================================
# DIRICHLET ENERGY (L2 smoothness)
# =============================================================================

def dirichlet_energy_G(G: Tensor, eps: float = 1e-8) -> Tensor:
    """Dirichlet energy E(G) = Σ ||∇G||²_F (promotes smooth fields)."""
    weight = torch.tensor([1., 2., 1.], device=G.device).view(1, 3, 1, 1)
    dx = G[:, :, :, 1:] - G[:, :, :, :-1]
    dy = G[:, :, 1:, :] - G[:, :, :-1, :]
    return (weight * dx**2).sum() + (weight * dy**2).sum()


def dirichlet_energy_B(B: Tensor, eps: float = 1e-8) -> Tensor:
    """Dirichlet energy E(B) = Σ ||∇B||²_F."""
    dx = B[:, :, :, 1:] - B[:, :, :, :-1]
    dy = B[:, :, 1:, :] - B[:, :, :-1, :]
    return (dx**2).sum() + (dy**2).sum()


def compute_dirichlet_regularization(
    G: Tensor,
    B: Tensor,
    lambda_G: float,
    lambda_B: float,
) -> Tensor:
    """Dirichlet regularization (smooth fields, not piecewise-constant)."""
    if lambda_G == 0 and lambda_B == 0:
        return torch.tensor(0.0, device=G.device)
    return lambda_G * dirichlet_energy_G(G) + lambda_B * dirichlet_energy_B(B)