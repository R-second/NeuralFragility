"""NeuralFragility package entrypoint.

This package provides tools to compute neural fragility based on
the `sreedhar` algorithm implemented by the user in the workspace.
"""

from .sreedhar_alg import *
from .fragility_from_eeg import calculate_ols_estimates, compute_A_from_time_series, compute_fragility_heatmap
from .openneuro_utils import (
    compute_fragility_from_matrices,
    create_sliding_windows,
    estimate_linear_models,
    estimate_noise_covariances,
    grid_search_fragility,
    is_stable,
    load_fragility_npz,
    model_fitting_errors,
    normalize_fragility,
    save_fragility_npz,
    spectral_radii,
)

__all__ = [
    "calculate_ols_estimates",
    "compute_fragility_from_matrices",
    "compute_A_from_time_series",
    "compute_fragility_heatmap",
    "create_sliding_windows",
    "estimate_linear_models",
    "estimate_noise_covariances",
    "grid_search_fragility",
    "is_stable",
    "load_fragility_npz",
    "model_fitting_errors",
    "neural_fragility_inf",
    "normalize_fragility",
    "save_fragility_npz",
    "spectral_radii",
]
