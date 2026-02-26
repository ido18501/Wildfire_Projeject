"""
Neural network components for learning covariate-to-metric mappings.

Components:
    encoder         - FlexibleCovariateEncoder (CNN + MLP architecture)
    regularization  - Total variation regularization for metric fields
    config          - Device configuration and metric type enum
"""

from eikonal.nn.encoder import FlexibleCovariateEncoder, project_G_eigenvalue, project_B_norm
from eikonal.nn.regularization import compute_tv_regularization
from eikonal.nn.config import DEVICE, TVMetricType
