"""NeuralFragility package entrypoint.

This package provides tools to compute neural fragility based on
the `sreedhar` algorithm implemented by the user in the workspace.
"""

from .sreedhar_alg import *
from .fragility_from_eeg import calculate_ols_estimates, compute_A_from_time_series, compute_fragility_heatmap

__all__ = [
    "calculate_ols_estimates",
    "compute_A_from_time_series",
    "compute_fragility_heatmap",
    "neural_fragility_inf",
]
