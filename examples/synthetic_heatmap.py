"""Generate synthetic EEG-like data and save a fragility heatmap."""
from pathlib import Path

try:
    from ._bootstrap import output_path
except ImportError:
    from _bootstrap import output_path

import argparse
import numpy as np

from fragility_from_eeg import compute_fragility_heatmap


def generate_synthetic_data(A, T=500, noise_scale=0.1):
    N = A.shape[0]
    X = np.zeros((T, N))
    X[0, :] = np.random.normal(0, 1, N)
    for t in range(1, T):
        X[t, :] = A @ X[t - 1, :] + np.random.normal(0, noise_scale, N)
    return X.T


def save_heatmap(heatmap, output_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def main():
    parser = argparse.ArgumentParser(description="Run a self-contained NeuralFragility example.")
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
    eeg = generate_synthetic_data(A_true, T=1000, noise_scale=0.05)
    fs = 200.0
    heatmap, times = compute_fragility_heatmap(eeg, fs, window_sec=0.2, step_sec=0.1, gamma=0.01)

    if not args.no_plot:
        output_file = Path(args.output)
        save_heatmap(heatmap, output_file)
        print(f"Saved heatmap to {output_file}")

    print(f"Computed heatmap with shape {heatmap.shape} across {len(times)} windows.")


if __name__ == "__main__":
    main()
