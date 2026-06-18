"""EEG and time-series utilities for Neural Fragility analyses.

Most functions accept data with shape `(n_channels, n_times)`.  Functions that
estimate one transition matrix from one window use `(n_times, n_channels)`.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from .fragility_algorithm import compute_level_value, compute_stability_radius
except ImportError:
    from fragility_algorithm import compute_level_value, compute_stability_radius

FloatArray: TypeAlias = NDArray[np.floating]
StringArray: TypeAlias = NDArray[np.str_]
PathLikeStr: TypeAlias = str | PathLike[str]
FragilityMethod: TypeAlias = Literal["proposed", "grid"]


def _get_data_and_times(
    raw_or_data: Any,
    fs: float | None = None,
) -> tuple[FloatArray, float, FloatArray]:
    """Extract data array, sampling frequency, and time points from an MNE Raw-like object or a raw data array.

    Args:
        raw_or_data: MNE Raw-like object or a raw data array of shape `(n_channels, n_times)`.
        fs: Sampling frequency required when `raw_or_data` is a NumPy array.

    Returns:
        Data array, sampling frequency, and time points.
    """
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


def create_sliding_windows(
    raw_or_data: Any,
    fs: float | None = None,
    window_size_ms: float = 250,
    step_size_ms: float = 125,
) -> tuple[FloatArray, FloatArray]:
    """Create sliding windows from an MNE Raw-like object or a raw data array.

    Args:
        raw_or_data: MNE Raw-like object or a raw data array of shape `(n_channels, n_times)`.
        fs: Sampling frequency required when `raw_or_data` is a NumPy array.
        window_size_ms: Window length. Unit is milliseconds.
        step_size_ms: Window step size. Unit is milliseconds.

    Returns:
        Array of windows with shape `(n_windows, n_channels, n_samples)` and center times.
    """
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


def estimate_linear_dynamics_ols(
    samples_by_time: FloatArray,
    only_transition_matrix: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Estimate linear dynamics using ordinary least squares regression.

    Args:
        samples_by_time: Array of shape `(n_times, n_channels)` representing the time series.
        only_transition_matrix: If `True`, only the transition matrix is returned.

    Returns:
        The transition matrix. If `only_transition_matrix=False`, also returns the residual covariance matrix.
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


def estimate_transition_matrix(
    samples_by_time: FloatArray,
    l2_lambda: float = 0.0,
) -> FloatArray:
    """Estimate the transition matrix from a single time series window.

    Args:
        samples_by_time: Array of shape `(n_times, n_channels)` representing the time series.
        l2_lambda: Ridge regularization coefficient. Use `0.0` for ordinary least squares.

    Returns:
        Transition matrix of shape `(n_channels, n_channels)`.
    """
    samples_by_time = np.asarray(samples_by_time)
    if samples_by_time.ndim != 2:
        raise ValueError("samples_by_time must have shape (n_times, n_channels).")

    n_times, n_channels = samples_by_time.shape
    if n_times < 2:
        raise ValueError("samples_by_time must contain at least two time points.")

    current_samples = samples_by_time[:-1, :].T
    next_samples = samples_by_time[1:, :].T
    regularization = l2_lambda * np.eye(n_channels)

    lhs = current_samples @ current_samples.T + regularization
    rhs = next_samples @ current_samples.T
    return rhs @ np.linalg.pinv(lhs)


def estimate_transition_matrices(
    windows: FloatArray,
    l2_lambda: float = 1e-4,
) -> FloatArray:
    """Estimate transition matrices for multiple windows.

    Args:
        windows: Array of shape `(n_windows, n_channels, n_samples)` representing the windows.
        l2_lambda: Ridge regularization coefficient passed to `estimate_transition_matrix`.

    Returns:
        Array of transition matrices with shape `(n_windows, n_channels, n_channels)`.
    """
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples).")

    n_windows, n_channels, n_samples = windows.shape
    if n_samples < 2:
        raise ValueError("Each window must contain at least two samples.")

    transition_matrices = np.zeros((n_windows, n_channels, n_channels), dtype=float)

    for window_index, window in enumerate(windows):
        transition_matrices[window_index] = estimate_transition_matrix(
            window.T,
            l2_lambda=l2_lambda,
        )

    return transition_matrices


def model_fitting_errors(
    windows: FloatArray,
    transition_matrices: FloatArray,
) -> FloatArray:
    """Compute the relative prediction errors for each window given the transition matrices.

    Args:
        windows: Array of shape `(n_windows, n_channels, n_samples)` representing the windows.
        transition_matrices: Array of transition matrices with shape `(n_windows, n_channels, n_channels)`.

    Returns:
        Array of relative prediction errors for each window.
    """
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

def compute_stability_radius_grid_search(
    transition_matrix: FloatArray,
    channel_index: int,
    num_points: int = 1000,
) -> float:
    """Approximate Stability Radius using grid search.

    Args:
        transition_matrix: Transition matrix of shape `(n_channels, n_channels)`.
        channel_index: Channel index for which to compute stability radius.
        num_points: Number of grid points to evaluate on `[0, pi]`.

    Returns:
        Approximated stability radius value as the reciprocal of the maximum level value.
    """
    thetas = np.linspace(0, np.pi, num_points)
    peak_level = max(
        compute_level_value(transition_matrix, channel_index, theta) for theta in thetas
    )
    return 1.0 / peak_level if peak_level != 0 else np.inf


def compute_stability_radius_from_matrices(
    transition_matrices: FloatArray,
    gamma: float = 0.01,
    method: FragilityMethod = "proposed",
    grid_points: int = 1000,
    max_iter: int = 20,
    epsilon: float = 1e-6,
    progress: bool = False,
) -> FloatArray:
    """Compute Stability Radius heatmap from transition matrices.

    Args:
        transition_matrices: Array of shape `(n_windows, n_channels, n_channels)` representing the transition matrices.
        gamma: Regularization parameter for the level set method.
        method: Stability radius computation method, either `"proposed"` or `"grid"`.
        grid_points: Number of grid points to use when `method="grid"`.
        max_iter: Maximum number of iterations when `method="proposed"`.
        epsilon: Convergence threshold when `method="proposed"`.
        progress: Whether to display a progress bar if possible.

    Returns:
        Array of shape `(n_channels, n_windows)` representing the raw stability radius heatmap.
    """
    transition_matrices = np.asarray(transition_matrices)
    if (
        transition_matrices.ndim != 3
        or transition_matrices.shape[1] != transition_matrices.shape[2]
    ):
        raise ValueError(
            "transition_matrices must have shape (n_windows, n_channels, n_channels)."
        )

    n_windows, n_channels, _ = transition_matrices.shape
    raw_stability_radius = np.zeros((n_channels, n_windows), dtype=float)

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
                value, _, _ = compute_stability_radius(
                    transition_matrix,
                    channel_index,
                    gamma=gamma,
                    max_iter=max_iter,
                    print_progress=False,
                    epsilon=epsilon,
                )
            elif method == "grid":
                value = compute_stability_radius_grid_search(
                    transition_matrix,
                    channel_index,
                    num_points=grid_points,
                )
            else:
                raise ValueError("method must be 'proposed' or 'grid'.")
            raw_stability_radius[channel_index, window_index] = value

    return raw_stability_radius


def compute_stability_radius_heatmap(
    eeg: FloatArray,
    fs: float,
    window_sec: float = 0.2,
    step_sec: float = 0.1,
    gamma: float = 0.01,
    verbose: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Compute Stability Radius heatmap from EEG array.

    Args:
        eeg: EEG array of shape `(n_channels, n_times)`.
        fs: Sampling frequency.
        window_sec: Window length in seconds.
        step_sec: Window step size in seconds.
        gamma: Regularization parameter for the level set method.
        verbose: Whether to display progress if possible.

    Returns:
        Stability Radius heatmap and the center times of each window.
    """
    if np.asarray(eeg).ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_times)")

    windows, times = create_sliding_windows(
        eeg,
        fs=fs,
        window_size_ms=window_sec * 1000,
        step_size_ms=step_sec * 1000,
    )
    transition_matrices = estimate_transition_matrices(windows, l2_lambda=0.0)
    heatmap = compute_stability_radius_from_matrices(
        transition_matrices,
        gamma=gamma,
        progress=verbose,
    )
    return heatmap, times


def calculate_neural_fragility(raw_stability_radius: FloatArray, eps: float = 1e-14) -> FloatArray:
    """Calculate neural fragility from raw stability radius values along the channel dimension for each time window.

    Args:
        raw_stability_radius: Raw stability radius array of shape `(n_channels, n_windows)`.
        eps: Small value to avoid division by zero.

    Returns:
        Neural fragility array following the convention `(max - raw) / max`.
    """
    raw_stability_radius = np.asarray(raw_stability_radius)
    max_vals = np.max(raw_stability_radius, axis=0)
    return (max_vals - raw_stability_radius) / (max_vals + eps)


def save_fragility_npz(
    path: PathLikeStr,
    raw_stability_radius: FloatArray,
    neural_fragility: FloatArray,
    times: FloatArray | None = None,
    channel_names: StringArray | list[str] | None = None,
) -> None:
    """Save fragility-related arrays in a compressed NumPy format without pickle.

    Args:
        path: Path to the target `.npz` file.
        raw_stability_radius: Raw stability radius array.
        neural_fragility: Neural fragility array.
        times: Array of center times for each window.
        channel_names: List or array of channel names.

    Returns:
        None.
    """
    payload: dict[str, Any] = {
        "raw_stability_radius": np.asarray(raw_stability_radius),
        "neural_fragility": np.asarray(neural_fragility),
    }
    if times is not None:
        payload["times"] = np.asarray(times)
    if channel_names is not None:
        payload["channel_names"] = np.asarray(channel_names, dtype=str)
    np.savez_compressed(path, allow_pickle=False, **payload)


def load_fragility_npz(path: PathLikeStr) -> dict[str, NDArray[Any]]:
    """Load fragility arrays saved with `save_fragility_npz`.

    Args:
        path: Path to the `.npz` file to load.

    Returns:
        Dictionary containing the loaded arrays indexed by their names.
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
