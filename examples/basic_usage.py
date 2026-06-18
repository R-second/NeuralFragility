"""Run the README basic usage examples as one executable script."""

from __future__ import annotations

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

import numpy as np

from eeg_fragility import compute_neural_fragility_heatmap
from eeg_fragility import compute_stability_radius_from_matrices
from fragility_algorithm import compute_stability_radius


def main() -> None:
    """Run the basic NeuralFragility usage examples.

    Args:
        なし。

    Returns:
        なし。
    """
    rng = np.random.default_rng(0)

    print("1. Compute stability radius for one matrix and one channel")
    transition_matrix = np.array(
        [
            [0.8, 0.1, 0.0],
            [0.0, 0.7, 0.2],
            [0.1, 0.0, 0.6],
        ]
    )
    value, theta, _ = compute_stability_radius(
        transition_matrix=transition_matrix,
        channel_index=0,
        gamma=0.01,
        print_progress=False,
    )
    print(f"channel 0 stability radius: {value:.6f}")
    print(f"best theta: {theta:.6f}")

    print("\n2. Compute a neural fragility heatmap from EEG-like data")
    eeg = rng.normal(size=(4, 2000))
    sampling_frequency = 200.0
    heatmap, times = compute_neural_fragility_heatmap(
        eeg=eeg,
        fs=sampling_frequency,
        window_sec=0.25,
        step_sec=0.125,
    )
    print(f"heatmap shape: {heatmap.shape}")
    print(f"times shape: {times.shape}")

    print("\n3. Compute stability radius values from pre-estimated transition matrices")
    transition_matrices = rng.normal(size=(10, 4, 4)) * 0.1
    stability_radius = compute_stability_radius_from_matrices(
        transition_matrices=transition_matrices,
        gamma=0.01,
    )
    print(f"stability radius shape: {stability_radius.shape}")


if __name__ == "__main__":
    main()
