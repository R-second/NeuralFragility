"""Compute Neural Fragility heatmaps from EEG-like time series.

Input EEG arrays are expected to have shape `(n_channels, n_times)`.
"""

import numpy as np
from numpy.linalg import pinv

try:
    from .fragility_algorithm import compute_neural_fragility
except ImportError:
    from fragility_algorithm import compute_neural_fragility


def estimate_linear_dynamics_ols(samples_by_time, only_transition_matrix=False):
    """Estimate `x(t+1) = A x(t)` by ordinary least squares.

    Parameters
    ----------
    samples_by_time:
        Array with shape `(n_times, n_channels)`.
    only_transition_matrix:
        If true, return only the transition matrix. Otherwise, also return the
        residual covariance matrix.
    """
    n_timepoints, _ = samples_by_time.shape
    n_transitions = n_timepoints - 1
    previous_samples = samples_by_time[:-1, :]
    next_samples = samples_by_time[1:, :]

    covariance = previous_samples.T @ previous_samples
    transition_matrix = (
        np.linalg.inv(covariance) @ previous_samples.T @ next_samples
    ).T
    residuals = next_samples - previous_samples @ transition_matrix.T
    noise_covariance = residuals.T @ residuals / n_transitions

    if only_transition_matrix:
        return transition_matrix
    return transition_matrix, noise_covariance


def compute_fragility_heatmap(
    eeg,
    fs,
    window_sec=0.2,
    step_sec=0.1,
    gamma=0.01,
    apply_svd=False,
    svd_log_threshold=-10,
    verbose=False,
):
    """Compute a neural fragility heatmap from EEG samples.

    Returns
    -------
    heatmap:
        Array with shape `(n_channels, n_windows)`.
    times:
        Center time of each window in seconds.
    """
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_times)")

    n_channels, n_timepoints = eeg.shape
    window_length = int(round(window_sec * fs))
    step_length = int(round(step_sec * fs))
    if window_length < 2:
        raise ValueError("window_sec too small for sampling frequency")

    window_starts = list(range(0, n_timepoints - window_length + 1, step_length))
    n_windows = len(window_starts)
    heatmap = np.zeros((n_channels, n_windows))
    times = []

    for window_index, start_index in enumerate(window_starts):
        window_data = eeg[:, start_index : start_index + window_length]
        times.append((start_index + window_length / 2) / fs)

        if apply_svd:
            centered_window = window_data - window_data.mean(axis=1, keepdims=True)
            left_vectors, singular_values, right_vectors = np.linalg.svd(
                centered_window,
                full_matrices=False,
            )
            kept_components = np.where(
                np.log10(singular_values + 1e-16) > svd_log_threshold
            )[0]

            if kept_components.size == 0:
                projection_matrix = (
                    left_vectors @ np.diag(singular_values) @ right_vectors
                )
                projected_samples = right_vectors.T[:, :1]
            else:
                n_components = len(kept_components)
                projection_matrix = left_vectors[:, :n_components] @ np.diag(
                    singular_values[:n_components]
                )
                projected_samples = right_vectors[:n_components, :].T

            if projected_samples.shape[0] < 2:
                transition_matrix = np.eye(n_channels)
            else:
                reduced_transition = estimate_transition_matrix(projected_samples)
                try:
                    transition_matrix = (
                        projection_matrix @ reduced_transition @ pinv(projection_matrix)
                    )
                except Exception:
                    transition_matrix = np.eye(n_channels)
        else:
            samples_by_time = window_data.T
            if samples_by_time.shape[0] < 2:
                transition_matrix = np.eye(n_channels)
            else:
                transition_matrix = estimate_transition_matrix(samples_by_time)

        for channel_index in range(n_channels):
            try:
                fragility, _, _ = compute_neural_fragility(
                    transition_matrix,
                    channel_index,
                    gamma=gamma,
                    print_progress=False,
                )
            except Exception:
                fragility = np.nan
            heatmap[channel_index, window_index] = fragility

    return heatmap, np.array(times)


def estimate_transition_matrix(samples_by_time):
    """Estimate a transition matrix from `(n_times, n_channels)` samples."""
    previous_samples = samples_by_time[:-1, :]
    next_samples = samples_by_time[1:, :]
    return (np.linalg.pinv(previous_samples) @ next_samples).T
