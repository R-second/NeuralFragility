"""NeuralFragility package entrypoint."""

from .eeg_fragility import (
    compute_stability_radius_from_matrices,
    compute_stability_radius_heatmap,
    create_sliding_windows,
    estimate_linear_dynamics_ols,
    estimate_transition_matrix,
    estimate_transition_matrices,
    load_fragility_npz,
    calculate_neural_fragility,
    save_fragility_npz,
)
from .fragility_algorithm import (
    FragilityMethod,
    compute_level_value,
    compute_stability_radius,
    compute_stability_radius_grid_search,
    compute_regularized_level_value,
    extract_unit_circle_angles,
    filter_level_crossings,
    find_best_interval_midpoint,
    maximize_level_value,
    solve_level_set_eigenproblem,
)

__all__ = [
    "compute_stability_radius_from_matrices",
    "compute_stability_radius_grid_search",
    "compute_stability_radius_heatmap",
    "FragilityMethod",
    "compute_level_value",
    "compute_stability_radius",
    "compute_regularized_level_value",
    "create_sliding_windows",
    "estimate_linear_dynamics_ols",
    "estimate_transition_matrices",
    "estimate_transition_matrix",
    "extract_unit_circle_angles",
    "filter_level_crossings",
    "find_best_interval_midpoint",
    "load_fragility_npz",
    "maximize_level_value",
    "calculate_neural_fragility",
    "save_fragility_npz",
    "solve_level_set_eigenproblem",
]
