"""Utilities to compute Neural Fragility heatmaps from EEG-like time series.

Functions:
 - calculate_ols_estimates(X, only_A=False)
 - compute_fragility_heatmap(eeg, fs, window_sec, step_sec, gamma=0.01, apply_svd=False)

Expect `eeg` shape: (n_channels, n_times).
"""
import numpy as np
from numpy.linalg import pinv

try:
    from .sreedhar_alg import neural_fragility_inf
except ImportError:
    from sreedhar_alg import neural_fragility_inf

def calculate_ols_estimates(X, only_A=False):
    # X: (T, N) time x channels as in notebooks; here we accept (T, N)
    T_full, N = X.shape
    T = T_full - 1
    X_t_minus_1 = X[:-1, :]
    X_t = X[1:, :]
    S_XX = X_t_minus_1.T @ X_t_minus_1
    A_ols = (np.linalg.inv(S_XX) @ X_t_minus_1.T @ X_t).T
    Sigma_ols = ((X_t - X_t_minus_1 @ A_ols.T).T @ (X_t - X_t_minus_1 @ A_ols.T)) / T
    if only_A:
        return A_ols
    return A_ols, Sigma_ols

def compute_fragility_heatmap(eeg, fs, window_sec=0.2, step_sec=0.1, gamma=0.01, apply_svd=False, svd_log_threshold=-10, verbose=False):
    """Compute fragility heatmap.

    Parameters:
    - eeg: ndarray, shape (n_channels, n_times)
    - fs: sampling frequency (Hz)
    - window_sec: window length in seconds
    - step_sec: step size in seconds
    - gamma: fragility algorithm parameter
    - apply_svd: if True, perform SVD-based dimensionality reduction per window (like original notebook)

    Returns:
    - heatmap: ndarray (n_channels, n_windows)
    - times: center times for each window (seconds)
    """
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D (n_channels, n_times)")
    n_ch, n_t = eeg.shape
    win_len = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win_len < 2:
        raise ValueError("window_sec too small for sampling frequency")
    starts = list(range(0, n_t - win_len + 1, step))
    n_win = len(starts)
    heatmap = np.zeros((n_ch, n_win))
    times = []
    for i, s in enumerate(starts):
        e = eeg[:, s:s+win_len]
        times.append((s + win_len/2) / fs)
        # convert to shape (T, N) as used by calculate_ols_estimates
        Xorig = e
        # optionally apply SVD PCA similar to notebooks
        if apply_svd:
            # center in time dimension
            Xc = Xorig - Xorig.mean(axis=1, keepdims=True)
            U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
            # choose components with log singular value > threshold
            keep = np.where(np.log10(S + 1e-16) > svd_log_threshold)[0]
            if keep.size == 0:
                P = U @ np.diag(S) @ Vh
                Xpca = (Vh.T[:, :1])  # fallback
            else:
                d = len(keep)
                P = U[:, :d] @ np.diag(S[:d])
                Xpca = Vh[:d, :].T
            # Xpca has shape (T, d) but we want (T, N_pca)
            # For OLS we need time x channels, so use Xpca (time x components)
            X_for_ols = Xpca
            # compute A00 and map back: A0 = P * A00 * pinv(P)
            if X_for_ols.shape[0] < 2:
                A = np.eye(n_ch)
            else:
                A00 = compute_A_from_time_series(X_for_ols)
                try:
                    A = P @ A00 @ pinv(P)
                except Exception:
                    A = np.eye(n_ch)
        else:
            # use raw channels; construct X with shape (T, N)
            X_ts = Xorig.T  # (T, N)
            if X_ts.shape[0] < 2:
                A = np.eye(n_ch)
            else:
                A = compute_A_from_time_series(X_ts)

        # compute fragility per channel k
        for k in range(n_ch):
            try:
                f, _, _ = neural_fragility_inf(A, k, gamma=gamma, print_progress=False)
            except Exception:
                f = np.nan
            heatmap[k, i] = f
    return heatmap, np.array(times)

def compute_A_from_time_series(X_ts):
    # X_ts: (T, N) as in notebooks (time x channels)
    # returns A of shape (N, N)
    T_full, N = X_ts.shape
    X_t_minus_1 = X_ts[:-1, :]
    X_t = X_ts[1:, :]
    # use pseudo-inverse for stability
    A_ols = (np.linalg.pinv(X_t_minus_1) @ X_t).T
    return A_ols
