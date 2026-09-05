"""Exact homogeneous, zero-drift travel times on a Cartesian 3D grid.

Volumes and integer source coordinates use ZYX order. Physical spacing and
positive propagation speeds use XYZ order, in mm and mm/ms respectively.
"""

import numpy as np


def analytical_arrival(shape_zyx, source_zyx, speeds_xyz, spacing_xyz=(0.8, 0.8, 0.8)):
    """Return an exact float64 arrival volume in ms with the source fixed at 0.

    ``speeds_xyz`` may be one scalar isotropic speed or three axis-aligned
    speeds. The metric is G = diag(1/c_x**2, 1/c_y**2, 1/c_z**2), so
    T(x) = sqrt(sum_a ((x_a - source_a) / c_a)**2).
    """
    shape = np.asarray(shape_zyx)
    source = np.asarray(source_zyx)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    speeds = np.asarray(speeds_xyz, dtype=np.float64)
    if shape.shape != (3,) or not np.all(np.isfinite(shape)):
        raise ValueError("shape_zyx must contain three positive integers")
    if not np.all(shape == np.floor(shape)) or np.any(shape < 1):
        raise ValueError("shape_zyx must contain three positive integers")
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("source_zyx must contain three in-bounds integers")
    if not np.all(source == np.floor(source)) or np.any(source < 0) or np.any(source >= shape):
        raise ValueError("source_zyx must contain three in-bounds integers")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three finite positive spacings")
    if speeds.ndim == 0 or speeds.shape == (1,):
        speeds = np.repeat(speeds.reshape(1), 3)
    if speeds.shape != (3,) or not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError("speeds_xyz must be a finite positive scalar or XYZ triple")
    axes = [(np.arange(int(n), dtype=np.float64) - s) * h / c
            for n, s, h, c in zip(shape, source, spacing[::-1], speeds[::-1])]
    return np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2
                   + axes[2][None, None, :] ** 2)
