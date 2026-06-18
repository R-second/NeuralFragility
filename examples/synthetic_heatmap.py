"""Generate synthetic EEG-like data and save a fragility heatmap."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

try:
    from ._bootstrap import output_path
except ImportError:
    from _bootstrap import output_path

import argparse
import numpy as np
from numpy.typing import NDArray

from eeg_fragility import compute_neural_fragility_heatmap

FloatArray: TypeAlias = NDArray[np.floating]
PathLikeStr: TypeAlias = str | PathLike[str]


def generate_synthetic_data(
    transition_matrix: FloatArray,
    n_times: int = 500,
    noise_scale: float = 0.1,
) -> FloatArray:
    """Generate synthetic EEG-like data using a linear dynamical system.

    Args:
        transition_matrix: Transition matrix of the linear dynamical system (shape: `(n_channels, n_channels)`).
        n_times: Number of samples to generate.
        noise_scale: Standard deviation of the Gaussian noise added at each time step.

    Returns:
        Synthetic data with shape `(n_channels, n_times)`.
    """
    n_channels = transition_matrix.shape[0]
    samples = np.zeros((n_times, n_channels))
    samples[0, :] = np.random.normal(0, 1, n_channels)
    for t in range(1, n_times):
        samples[t, :] = transition_matrix @ samples[t - 1, :] + np.random.normal(
            0,
            noise_scale,
            n_channels,
        )
    return samples.T


def save_heatmap(heatmap: FloatArray, output_file: PathLikeStr) -> None:
    """Generate a heatmap image of the neural fragility for synthetic data.

    Args:
        heatmap: Neural fragility heatmap with shape `(n_channels, n_windows)`.
        output_file: Path to the output image file.

    Returns:
        None.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Neural Fragility")
    plt.xlabel("Window index")
    plt.ylabel("Channel")
    plt.title("Synthetic Neural Fragility Heatmap")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def main() -> None:
    """Parse CLI arguments and run the synthetic data heatmap example."""
    parser = argparse.ArgumentParser(
        description="Run a self-contained NeuralFragility example."
    )
    parser.add_argument(
        "--output",
        default=str(output_path("synthetic_fragility_heatmap.png")),
        help="Path for the generated heatmap image.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Compute the heatmap without writing an image.",
    )
    args = parser.parse_args()

    np.random.seed(0)
    N = 8
    A_true = np.random.randn(N, N) * 0.2
    A_true = 0.9 * A_true / np.max(np.abs(np.linalg.eigvals(A_true)))
    eeg = generate_synthetic_data(A_true, n_times=1000, noise_scale=0.05)
    fs = 200.0
    heatmap, times = compute_neural_fragility_heatmap(
        eeg, fs, window_sec=0.2, step_sec=0.1, gamma=0.01
    )

    if not args.no_plot:
        output_file = Path(args.output)
        save_heatmap(heatmap, output_file)
        print(f"Saved heatmap to {output_file}")

    print(f"Computed heatmap with shape {heatmap.shape} across {len(times)} windows.")


if __name__ == "__main__":
    main()
