"""EEG and time-series utilities for Neural Fragility analyses.

Most functions accept data with shape `(n_channels, n_times)`.  Functions that
estimate one transition matrix from one window use `(n_times, n_channels)`.
"""

import numpy as np
from numpy.linalg import pinv

try:
    from .fragility_algorithm import compute_level_value, compute_neural_fragility
except ImportError:
    from fragility_algorithm import compute_level_value, compute_neural_fragility


def _get_data_and_times(raw_or_data, fs=None):
    if hasattr(raw_or_data, "get_data") and hasattr(raw_or_data, "info"):
        data = raw_or_data.get_data()
        sampling_frequency = float(raw_or_data.info["sfreq"])
        times = np.asarray(raw_or_data.times)
        return data, sampling_frequency, times

    if fs is None:
        raise ValueError("fs is required when raw_or_data is a NumPy array.")

    data = np.asarray(raw_or_data)
    if data.ndim != 2:
        raise ValueError("raw_or_data must have shape (n_channels, n_times).")

    sampling_frequency = float(fs)
    times = np.arange(data.shape[1]) / sampling_frequency
    return data, sampling_frequency, times


def create_sliding_windows(raw_or_data, fs=None, window_size_ms=250, step_size_ms=125):
    """Create sliding windows with shape `(n_windows, n_channels, n_samples)`."""
    data, sampling_frequency, times = _get_data_and_times(raw_or_data, fs=fs)
    n_channels, n_times = data.shape

    n_samples_window = int(round(window_size_ms * sampling_frequency / 1000))
    n_samples_step = int(round(step_size_ms * sampling_frequency / 1000))

    if n_samples_window < 2:
        raise ValueError("window_size_ms is too small for the sampling frequency.")
    if n_samples_step < 1:
        raise ValueError("step_size_ms is too small for the sampling frequency.")
    if n_times < n_samples_window:
        raise ValueError("Data length is shorter than the requested window size.")

    starts = np.arange(0, n_times - n_samples_window + 1, n_samples_step, dtype=int)
    windows = np.empty((len(starts), n_channels, n_samples_window), dtype=data.dtype)

    for window_index, start_index in enumerate(starts):
        windows[window_index] = data[:, start_index : start_index + n_samples_window]

    window_times = times[starts] + (n_samples_window / sampling_frequency / 2)
    return windows, window_times


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


def estimate_transition_matrix(samples_by_time):
    """Estimate a transition matrix from `(n_times, n_channels)` samples."""
    previous_samples = samples_by_time[:-1, :]
    next_samples = samples_by_time[1:, :]
    return (np.linalg.pinv(previous_samples) @ next_samples).T


def estimate_transition_matrices(windows, l2_lambda=1e-4):
    """Estimate `x(t+1) = A x(t)` for each window using ridge regression."""
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples).")

    n_windows, n_channels, n_samples = windows.shape
    if n_samples < 2:
        raise ValueError("Each window must contain at least two samples.")

    transition_matrices = np.zeros((n_windows, n_channels, n_channels), dtype=float)
    regularization = l2_lambda * np.eye(n_channels)

    for window_index, window in enumerate(windows):
        current_samples = window[:, :-1]
        next_samples = window[:, 1:]

        lhs = current_samples @ current_samples.T + regularization
        rhs = next_samples @ current_samples.T
        transition_matrices[window_index] = rhs @ np.linalg.pinv(lhs)

    return transition_matrices


def estimate_transition_matrices_from_windows(windows, apply_svd=False, svd_log_threshold=-10):
    """Estimate one transition matrix per window, with optional SVD projection."""
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples).")

    if not apply_svd:
        return np.array([estimate_transition_matrix(window.T) for window in windows])

    n_windows, n_channels, _ = windows.shape
    transition_matrices = np.zeros((n_windows, n_channels, n_channels), dtype=float)

    for window_index, window in enumerate(windows):
        transition_matrices[window_index] = _estimate_svd_transition_matrix(
            window,
            svd_log_threshold=svd_log_threshold,
        )

    return transition_matrices


def _estimate_svd_transition_matrix(window, svd_log_threshold=-10):
    n_channels = window.shape[0]
    centered_window = window - window.mean(axis=1, keepdims=True)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        centered_window,
        full_matrices=False,
    )
    kept_components = np.where(np.log10(singular_values + 1e-16) > svd_log_threshold)[0]

    if kept_components.size == 0:
        projection_matrix = left_vectors @ np.diag(singular_values) @ right_vectors
        projected_samples = right_vectors.T[:, :1]
    else:
        n_components = len(kept_components)
        projection_matrix = left_vectors[:, :n_components] @ np.diag(
            singular_values[:n_components]
        )
        projected_samples = right_vectors[:n_components, :].T

    if projected_samples.shape[0] < 2:
        return np.eye(n_channels)

    reduced_transition = estimate_transition_matrix(projected_samples)
    try:
        return projection_matrix @ reduced_transition @ pinv(projection_matrix)
    except Exception:
        return np.eye(n_channels)


def estimate_noise_covariances(windows, transition_matrices):
    """Estimate residual covariance matrices for fitted linear models."""
    windows = np.asarray(windows)
    transition_matrices = np.asarray(transition_matrices)

    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples).")
    if (
        transition_matrices.shape[:2] != windows.shape[:2]
        or transition_matrices.shape[2] != windows.shape[1]
    ):
        raise ValueError(
            "transition_matrices must have shape (n_windows, n_channels, n_channels)."
        )

    n_windows, n_channels, n_samples = windows.shape
    noise_covariances = np.zeros((n_windows, n_channels, n_channels), dtype=float)

    for window_index, window in enumerate(windows):
        current_samples = window[:, :-1]
        next_samples = window[:, 1:]
        residuals = next_samples - transition_matrices[window_index] @ current_samples
        noise_covariances[window_index] = (
            residuals @ residuals.T / max(n_samples - 1, 1)
        )

    return noise_covariances


def model_fitting_errors(windows, transition_matrices):
    """Return relative one-step prediction errors for each window."""
    windows = np.asarray(windows)
    transition_matrices = np.asarray(transition_matrices)
    errors = np.zeros(windows.shape[0], dtype=float)

    for window_index, window in enumerate(windows):
        current_samples = window[:, :-1]
        next_samples = window[:, 1:]
        predicted = transition_matrices[window_index] @ current_samples
        errors[window_index] = np.linalg.norm(next_samples - predicted, "fro") / (
            np.linalg.norm(next_samples, "fro") + 1e-12
        )

    return errors


def spectral_radii(transition_matrices):
    """Return the maximum absolute eigenvalue for each matrix."""
    transition_matrices = np.asarray(transition_matrices)
    return np.array(
        [
            np.max(np.abs(np.linalg.eigvals(transition_matrix)))
            for transition_matrix in transition_matrices
        ]
    )


def is_stable_transition_matrix(transition_matrix, tol=0.0):
    """Return whether all eigenvalues lie inside the unit circle."""
    return np.max(np.abs(np.linalg.eigvals(transition_matrix))) <= 1.0 + tol


def compute_fragility_grid_search(transition_matrix, channel_index, num_points=1000):
    """Approximate neural fragility by grid-searching `max_theta inf sigma2`.

    `compute_neural_fragility` returns `1 / max_theta inf sigma2`, so this helper
    returns the same quantity rather than the raw level-set value.
    """
    thetas = np.linspace(0, np.pi, num_points)
    peak_level = max(
        compute_level_value(transition_matrix, channel_index, theta) for theta in thetas
    )
    return 1.0 / peak_level if peak_level != 0 else np.inf


def compute_fragility_from_matrices(
    transition_matrices,
    gamma=0.01,
    method="proposed",
    grid_points=1000,
    max_iter=20,
    epsilon=1e-6,
    progress=False,
):
    """Compute raw neural fragility with shape `(n_channels, n_windows)`."""
    transition_matrices = np.asarray(transition_matrices)
    if (
        transition_matrices.ndim != 3
        or transition_matrices.shape[1] != transition_matrices.shape[2]
    ):
        raise ValueError(
            "transition_matrices must have shape (n_windows, n_channels, n_channels)."
        )

    n_windows, n_channels, _ = transition_matrices.shape
    raw_fragility = np.zeros((n_channels, n_windows), dtype=float)

    iterator = range(n_windows)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        except ImportError:
            pass

    for window_index in iterator:
        transition_matrix = transition_matrices[window_index]
        for channel_index in range(n_channels):
            if method == "proposed":
                value, _, _ = compute_neural_fragility(
                    transition_matrix,
                    channel_index,
                    gamma=gamma,
                    max_iter=max_iter,
                    print_progress=False,
                    epsilon=epsilon,
                )
            elif method == "grid":
                value = compute_fragility_grid_search(
                    transition_matrix,
                    channel_index,
                    num_points=grid_points,
                )
            else:
                raise ValueError("method must be 'proposed' or 'grid'.")
            raw_fragility[channel_index, window_index] = value

    return raw_fragility


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
    if np.asarray(eeg).ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_times)")

    windows, times = create_sliding_windows(
        eeg,
        fs=fs,
        window_size_ms=window_sec * 1000,
        step_size_ms=step_sec * 1000,
    )
    transition_matrices = estimate_transition_matrices_from_windows(
        windows,
        apply_svd=apply_svd,
        svd_log_threshold=svd_log_threshold,
    )
    heatmap = compute_fragility_from_matrices(
        transition_matrices,
        gamma=gamma,
        progress=verbose,
    )
    return heatmap, times


def normalize_fragility(raw_fragility, eps=1e-14):
    """Normalize fragility across channels at each time window.

    This matches the notebook convention: `(max - raw) / max`.
    """
    raw_fragility = np.asarray(raw_fragility)
    max_vals = np.max(raw_fragility, axis=0)
    return (max_vals - raw_fragility) / (max_vals + eps)


def save_fragility_npz(
    path,
    raw_fragility,
    normalized_fragility,
    times=None,
    channel_names=None,
):
    """Save fragility arrays without pickle."""
    payload = {
        "raw_fragility": np.asarray(raw_fragility),
        "normalized_fragility": np.asarray(normalized_fragility),
    }
    if times is not None:
        payload["times"] = np.asarray(times)
    if channel_names is not None:
        payload["channel_names"] = np.asarray(channel_names, dtype=str)
    np.savez_compressed(path, **payload)


def load_fragility_npz(path):
    """Load fragility arrays saved by `save_fragility_npz`."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
