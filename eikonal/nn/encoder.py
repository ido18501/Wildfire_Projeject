"""
Neural Network Models for Metric Field Prediction
==================================================

Contains the FlexibleCovariateEncoder: a CNN that maps spatial and global
covariates to Randers-Finsler metric parameters (G, B).

Architecture:
- Global pathway: MLP(weather features) → uniform baseline (G₀, B₀)
- Local pathway: CNN(spatial covariates) → spatially-varying residual

Output: G = G₀ + G_local, B = B₀ + B_local

Author: Barak Gahtan
Date: January 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


# =============================================================================
# CONSTRAINT PROJECTIONS
# =============================================================================

def project_G_eigenvalue(G: torch.Tensor, eps: float = 0.1,
                         relative: bool = False, max_eigenvalue: float = 10.0) -> torch.Tensor:
    """
    Project G to positive definite via eigenvalue thresholding.

    Uses stable atan2-based eigenvector computation (same as matrix_log_2x2).

    Args:
        G: Metric tensor field (B, 3, M, N) as [g11, g12, g22]
        eps: Minimum eigenvalue (ABSOLUTE - relative mode is unstable)
        relative: Ignored for stability (always uses absolute eps)
        max_eigenvalue: Maximum allowed eigenvalue (prevents extreme slowdowns)

    Returns:
        Projected G with eps <= eigenvalues <= max_eigenvalue
    """
    g11 = G[:, 0:1, :, :]
    g12 = G[:, 1:2, :, :]
    g22 = G[:, 2:3, :, :]

    # Eigenvalues via stable formula
    trace = g11 + g22
    diff = g11 - g22
    disc = torch.sqrt(diff**2 + 4 * g12**2 + 1e-12)

    lambda_max = (trace + disc) / 2
    lambda_min = (trace - disc) / 2

    # Clamp eigenvalues to valid range
    lambda_max_new = torch.clamp(lambda_max, min=eps, max=max_eigenvalue)
    lambda_min_new = torch.clamp(lambda_min, min=eps, max=max_eigenvalue)

    # Principal angle via atan2 (stable, no division)
    theta = 0.5 * torch.atan2(2 * g12, diff + 1e-12)
    c = torch.cos(theta)
    s = torch.sin(theta)

    # Reconstruct G = U D U^T with clamped eigenvalues
    g11_proj = lambda_max_new * c**2 + lambda_min_new * s**2
    g12_proj = (lambda_max_new - lambda_min_new) * c * s
    g22_proj = lambda_max_new * s**2 + lambda_min_new * c**2

    return torch.cat([g11_proj, g12_proj, g22_proj], dim=1)


def project_B_norm(B: torch.Tensor, G: torch.Tensor, max_norm: float = 0.9) -> torch.Tensor:
    """
    Project B so that ||B||_G < max_norm (Randers feasibility constraint).

    Also clips B in Euclidean norm as a safety measure.

    Args:
        B: Drift vector field (B, 2, M, N)
        G: Metric tensor field (B, 3, M, N)
        max_norm: Maximum allowed G-norm of B

    Returns:
        Projected B with ||B||_G < max_norm everywhere
    """
    b1, b2 = B[:, 0:1, :, :], B[:, 1:2, :, :]
    g11, g12, g22 = G[:, 0:1, :, :], G[:, 1:2, :, :], G[:, 2:3, :, :]

    # First: clip Euclidean norm as safety (prevents extreme B even if G is weird)
    euclidean_norm = torch.sqrt(b1**2 + b2**2 + 1e-12)
    euclidean_max = max_norm * 2  # Allow some headroom for G-norm scaling
    euclidean_scale = torch.where(euclidean_norm > euclidean_max,
                                   euclidean_max / euclidean_norm,
                                   torch.ones_like(euclidean_norm))
    b1 = b1 * euclidean_scale
    b2 = b2 * euclidean_scale

    # Then: project in G-norm
    det_G = torch.clamp(g11 * g22 - g12**2, min=1e-6)  # Increased from 1e-8
    norm_B_sq = (b1**2 * g22 - 2*b1*b2*g12 + b2**2 * g11) / det_G
    norm_B = torch.sqrt(torch.clamp(norm_B_sq, min=0))

    scale = torch.where(norm_B > max_norm, max_norm / (norm_B + 1e-8), torch.ones_like(norm_B))
    return torch.cat([b1 * scale, b2 * scale], dim=1)


# =============================================================================
# FLEXIBLE COVARIATE ENCODER
# =============================================================================

class FlexibleCovariateEncoder(nn.Module):
    """
    CNN encoder that handles variable input sizes.

    Architecture:
        Global pathway: MLP(global_features) → (g11₀, g12₀, g22₀, b1₀, b2₀)
        Local pathway: CNN(spatial_features) → local residuals

    The global pathway captures fire-wide effects (wind, weather).
    The local pathway captures terrain and fuel variations.

    Only the local components are regularized, allowing global parameters
    to capture fire-wide drift without penalty.
    """

    def __init__(
        self,
        n_spatial_features: int,
        n_global_features: int,
        hidden_channels: int = 128,
        num_cnn_layers: int = 6,
        mlp_hidden: int = 64,
        mlp_num_layers: int = 3,
        eps: float = 0.1,
        max_drift: float = 0.9,
        local_scale: float = 0.1,
        max_eigenvalue: float = 10.0,
    ):
        """
        Args:
            n_spatial_features: Number of spatial covariate channels
            n_global_features: Number of global (per-fire) features
            hidden_channels: Width of CNN hidden layers
            num_cnn_layers: Depth of CNN
            mlp_hidden: Width of global MLP hidden layers
            mlp_num_layers: Depth of global MLP
            eps: Minimum eigenvalue for G projection
            max_drift: Maximum ||B||_G for Randers feasibility
            local_scale: Scaling factor for local residuals
            max_eigenvalue: Maximum eigenvalue for G projection
        """
        super().__init__()

        self.n_spatial = n_spatial_features
        self.n_global = n_global_features
        self.eps = eps
        self.max_drift = max_drift
        self.local_scale = local_scale
        self.max_eigenvalue = max_eigenvalue

        # Global MLP: global features → (g11, g12, g22, b1, b2)
        mlp_layers = []
        in_dim = n_global_features
        for i in range(mlp_num_layers - 1):
            mlp_layers.extend([
                nn.Linear(in_dim, mlp_hidden),
                nn.ReLU(),
            ])
            in_dim = mlp_hidden
        mlp_layers.append(nn.Linear(in_dim, 5))
        self.global_mlp = nn.Sequential(*mlp_layers)

        # Local CNN: spatial features → local residuals
        cnn_layers = []
        in_ch = n_spatial_features
        for i in range(num_cnn_layers - 1):
            cnn_layers.extend([
                nn.Conv2d(in_ch, hidden_channels, 3, padding=1),
                nn.ReLU(),
            ])
            in_ch = hidden_channels
        cnn_layers.append(nn.Conv2d(hidden_channels, 5, 1))
        self.local_cnn = nn.Sequential(*cnn_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize to identity metric, zero drift."""
        nn.init.zeros_(self.global_mlp[-1].weight)
        with torch.no_grad():
            self.global_mlp[-1].bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]))

        nn.init.normal_(self.local_cnn[-1].weight, std=0.01)
        nn.init.zeros_(self.local_cnn[-1].bias)

    def forward(self, spatial_feat: torch.Tensor, global_feat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_feat: (B, C_spatial, M, N) - any M, N
            global_feat: (B, C_global)

        Returns:
            G: (B, 3, M, N) - projected metric tensor field
            B: (B, 2, M, N) - projected drift vector field
            G_local: (B, 3, M, N) - local component (for regularization)
            B_local: (B, 2, M, N) - local component (for regularization)
        """
        batch, _, M, N = spatial_feat.shape

        # Global baseline
        g_out = self.global_mlp(global_feat)  # (B, 5)
        G_global = g_out[:, :3, None, None].expand(-1, -1, M, N)
        B_global = g_out[:, 3:, None, None].expand(-1, -1, M, N)

        # Local residual from CNN
        local = self.local_scale * self.local_cnn(spatial_feat)
        G_local = local[:, :3, :, :]
        B_local = local[:, 3:, :, :]

        # Combine
        G_raw = G_global + G_local
        B_raw = B_global + B_local

        # Project to valid metric/drift
        G = project_G_eigenvalue(G_raw, self.eps, relative=False, max_eigenvalue=self.max_eigenvalue)
        B = project_B_norm(B_raw, G, self.max_drift)

        return G, B, G_local, B_local

    def get_global_params(self, global_feat: torch.Tensor) -> Dict:
        """Extract global baseline values for logging."""
        out = self.global_mlp(global_feat)
        return {
            'g11': out[0, 0].item(),
            'g12': out[0, 1].item(),
            'g22': out[0, 2].item(),
            'b1': out[0, 3].item(),
            'b2': out[0, 4].item(),
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)