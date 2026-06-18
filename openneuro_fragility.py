"""Utilities for OpenNeuro-style EEG neural fragility analyses.

The functions in this module are intentionally independent of MNE.  Pass an
MNE Raw object to `create_sliding_windows` when MNE is installed, or pass a
plain NumPy array with shape `(n_channels, n_times)` and an explicit sampling
frequency.
"""

import numpy as np

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

    for i, start_idx in enumerate(starts):
        windows[i] = data[:, start_idx : start_idx + n_samples_window]

    window_times = times[starts] + (n_samples_window / sampling_frequency / 2)
    return windows, window_times


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

    for i, window in enumerate(windows):
        x_t = window[:, :-1]
        x_t1 = window[:, 1:]

        lhs = x_t @ x_t.T + regularization
        rhs = x_t1 @ x_t.T
        transition_matrices[i] = rhs @ np.linalg.pinv(lhs)

    return transition_matrices


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
    sigma_matrices = np.zeros((n_windows, n_channels, n_channels), dtype=float)

    for i, window in enumerate(windows):
        x_t = window[:, :-1]
        x_t1 = window[:, 1:]
        residuals = x_t1 - transition_matrices[i] @ x_t
        sigma_matrices[i] = residuals @ residuals.T / max(n_samples - 1, 1)

    return sigma_matrices


def model_fitting_errors(windows, transition_matrices):
    """Return relative one-step prediction errors for each window."""
    windows = np.asarray(windows)
    transition_matrices = np.asarray(transition_matrices)
    errors = np.zeros(windows.shape[0], dtype=float)

    for i, window in enumerate(windows):
        x_t = window[:, :-1]
        x_t1 = window[:, 1:]
        predicted = transition_matrices[i] @ x_t
        errors[i] = np.linalg.norm(x_t1 - predicted, "fro") / (
            np.linalg.norm(x_t1, "fro") + 1e-12
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

    for window_idx in iterator:
        transition_matrix = transition_matrices[window_idx]
        for channel_index in range(n_channels):
            if method == "proposed":
                val, _, _ = compute_neural_fragility(
                    transition_matrix,
                    channel_index,
                    gamma=gamma,
                    max_iter=max_iter,
                    print_progress=False,
                    epsilon=epsilon,
                )
            elif method == "grid":
                val = compute_fragility_grid_search(
                    transition_matrix,
                    channel_index,
                    num_points=grid_points,
                )
            else:
                raise ValueError("method must be 'proposed' or 'grid'.")
            raw_fragility[channel_index, window_idx] = val

    return raw_fragility


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
