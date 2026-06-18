"""Convergence analysis for random stable matrices."""

from __future__ import annotations

try:
    from ._bootstrap import output_path
    from .plotting import configure_matplotlib
except ImportError:
    from _bootstrap import output_path
    from plotting import configure_matplotlib

import argparse
from os import PathLike
from pathlib import Path
from typing import TypeAlias, Any

import numpy as np
from numpy.typing import NDArray

from fragility_algorithm import maximize_level_value

FloatArray: TypeAlias = NDArray[np.floating]
IterationLog: TypeAlias = list[dict[str, Any]]
PathLikeStr: TypeAlias = str | PathLike[str]


def generate_stable_matrix(
    n: int,
    spectral_radius: float = 0.95,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate a random stable matrix with a specified spectral radius.

    Args:
        n: Matrix size.
        spectral_radius: Spectral radius to match after generation.
        rng: Random number generator. A new one will be created if `None`.

    Returns:
        A random stable matrix with shape `(n, n)`.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = rng.standard_normal((n, n))
    eigenvalues = np.linalg.eigvals(A)
    max_abs_eig = np.max(np.abs(eigenvalues))

    if max_abs_eig == 0:
        return A
    return A * (spectral_radius / max_abs_eig)


def extract_level_history(log: IterationLog) -> FloatArray:
    """Extract the history of level values from the iteration log.

    Args:
        log: log returned by `maximize_level_value`, containing entries for each iteration.

    Returns:
        Array of level values for each iteration.
    """
    history = [float(log[0]["level"])]
    for entry in log[1:]:
        next_level = entry.get("next_level")
        if next_level is not None:
            history.append(float(next_level))
    return np.array(history)


def run_trials(
    num_trials: int,
    matrix_size: int,
    spectral_radius: float,
    k_idx: int,
    gamma_algo: float,
    max_iter: int,
    seed: int,
) -> list[FloatArray]:
    """Run multiple random trials for convergence analysis.

    Args:
        num_trials: Number of trials.
        matrix_size: Matrix size.
        spectral_radius: Spectral radius of the generated matrix.
        k_idx: Index of the channel to evaluate.
        gamma_algo: Regularization parameter for the level-set method.
        max_iter: Maximum number of iterations.
        seed: Random seed.

    Returns:
        List of level value histories for each trial.
    """
    rng = np.random.default_rng(seed)
    results = []

    print(f"Running {num_trials} trials with N={matrix_size}...")
    for i in range(num_trials):
        A = generate_stable_matrix(
            matrix_size, spectral_radius=spectral_radius, rng=rng
        )
        final_level, final_theta, log = maximize_level_value(
            A,
            k_idx,
            gamma_algo,
            max_iter=max_iter,
            print_progress=False,
            epsilon=1e-15,
        )
        results.append(extract_level_history(log))
        print(
            f"Trial {i + 1:>3}/{num_trials}: level={final_level:.6f}, theta={final_theta:.4f}, steps={len(log) - 1}"
        )

    return results


def plot_convergence(
    results: list[FloatArray],
    output_file: PathLikeStr,
    show: bool,
) -> None:
    """Plot the convergence error curve and save it.

    Args:
        results: List of level value histories for each trial.
        output_file: Path to the output image file.
        show: Whether to display the figure interactively after saving.

    Returns:
        None.
    """
    plt, line_alpha = configure_matplotlib(output_file, show)
    plt.figure(figsize=(10, 6))

    max_error_index = 0
    for history in results:
        if len(history) < 2:
            continue

        final_val = history[-1]
        errors = np.abs(history[:-1] - final_val)
        mask = errors > 1e-16

        if np.any(mask):
            x_values = np.arange(len(errors))[mask]
            max_error_index = max(max_error_index, int(np.max(x_values)))
            plt.semilogy(x_values, errors[mask], "k-", alpha=line_alpha)

    plt.xlabel("Iteration")
    plt.ylabel(r"Error $|\Xi_{final} - \Xi_{k}|$ (log scale)")
    plt.xticks(np.arange(0, max_error_index + 1, step=1))
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=350)
    print(f"Saved figure to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    """Parse CLI arguments and run convergence analysis.
    """
    parser = argparse.ArgumentParser(
        description="Run convergence analysis on random stable matrices."
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of random trials."
    )
    parser.add_argument("--size", type=int, default=100, help="Matrix size.")
    parser.add_argument(
        "--spectral-radius", type=float, default=0.95, help="Target spectral radius."
    )
    parser.add_argument("--k", type=int, default=0, help="Perturbed node index.")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Algorithm gamma.")
    parser.add_argument(
        "--max-iter", type=int, default=20, help="Maximum level-set iterations."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        default=str(output_path("convergence_analysis.png")),
        help="Output figure path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    args = parser.parse_args()

    results = run_trials(
        num_trials=args.trials,
        matrix_size=args.size,
        spectral_radius=args.spectral_radius,
        k_idx=args.k,
        gamma_algo=args.gamma,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    plot_convergence(results, args.output, args.show)
    print("Visualization complete.")


if __name__ == "__main__":
    main()
