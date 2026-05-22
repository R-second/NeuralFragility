"""Small NumPy-only example for the OpenNeuro-style analysis pipeline."""
try:
    from ._bootstrap import output_path
except ImportError:
    from _bootstrap import output_path

import argparse
from pathlib import Path

import numpy as np

from openneuro_utils import (
    compute_fragility_from_matrices,
    create_sliding_windows,
    estimate_linear_models,
    normalize_fragility,
    save_fragility_npz,
)


def generate_synthetic_eeg(n_channels, n_times, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_channels, n_channels))
    A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))

    eeg = np.zeros((n_channels, n_times))
    eeg[:, 0] = rng.standard_normal(n_channels)
    for t in range(1, n_times):
        eeg[:, t] = A @ eeg[:, t - 1] + 0.05 * rng.standard_normal(n_channels)
    return eeg


def plot_heatmap(normalized_fragility, times, output_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(normalized_fragility, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Neural Fragility (normalized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")

    n_ticks = min(8, len(times))
    tick_indices = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
    plt.xticks(tick_indices, [f"{times[i]:.2f}" for i in tick_indices], rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Saved heatmap to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run a NumPy-only OpenNeuro-style fragility pipeline.")
    parser.add_argument("--channels", type=int, default=6, help="Synthetic channel count.")
    parser.add_argument("--samples", type=int, default=500, help="Synthetic sample count.")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling frequency.")
    parser.add_argument("--window-ms", type=float, default=250.0, help="Window size in milliseconds.")
    parser.add_argument("--step-ms", type=float, default=125.0, help="Step size in milliseconds.")
    parser.add_argument("--method", choices=["proposed", "grid"], default="proposed", help="Fragility solver.")
    parser.add_argument("--output", default=str(output_path("openneuro_numpy_heatmap.png")), help="Output heatmap path.")
    parser.add_argument("--data-output", default=str(output_path("openneuro_numpy_fragility.npz")), help="Output data path.")
    args = parser.parse_args()

    eeg = generate_synthetic_eeg(args.channels, args.samples)
    windows, times = create_sliding_windows(eeg, fs=args.fs, window_size_ms=args.window_ms, step_size_ms=args.step_ms)
    A_matrices = estimate_linear_models(windows, l2_lambda=1e-4)
    raw_fragility = compute_fragility_from_matrices(A_matrices, method=args.method, gamma=0.01)
    normalized_fragility = normalize_fragility(raw_fragility)

    save_fragility_npz(args.data_output, raw_fragility, normalized_fragility, times=times)
    print(f"Saved fragility arrays to {args.data_output}")
    plot_heatmap(normalized_fragility, times, args.output)


if __name__ == "__main__":
    main()
