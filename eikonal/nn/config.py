"""
Configuration for eikonal neural network components.
"""

import torch
from enum import Enum

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TVMetricType(Enum):
    """TV regularization metric types for SPD tensor fields."""
    FROBENIUS = "frobenius"
    LOG_EUCLIDEAN = "log_euclidean"
