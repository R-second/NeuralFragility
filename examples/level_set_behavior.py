"""Visualize the level-set iterations used in the paper's numerical tests."""

from __future__ import annotations

try:
    from ._bootstrap import output_path
    from .matrix_examples import get_matrix
    from .plotting import configure_matplotlib
except ImportError:
    from _bootstrap import output_path
    from matrix_examples import get_matrix
    from plotting import configure_matplotlib

import argparse
from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np

from fragility_algorithm import compute_level_value, maximize_level_value

PathLikeStr: TypeAlias = str | PathLike[str]


def run_experiment(
    matrix_name: str,
    k: int,
    gamma_algo: float,
    grid_size: int,
    max_iter: int,
    output_file: PathLikeStr,
    show: bool,
) -> None:
    """Run a single experiment to visualize the level-set method behavior and save the diagnostic plot.

    Args:
        matrix_name: Name of the matrix registered in `matrix_examples.py`.
        k: Index of the channel to evaluate.
        gamma_algo: Regularization parameter for the level-set method.
        grid_size: Number of points to use in the reference grid search.
        max_iter: Maximum number of iterations for the level-set method.
        output_file: Path to the output image file.
        show: Whether to display the figure interactively after saving.

    Returns:
        None.
    """
    A = get_matrix(matrix_name)

    theta_plot = np.linspace(0, np.pi, grid_size)
    inf_vals = np.array([compute_level_value(A, k, t) for t in theta_plot])

    true_max_idx = np.argmax(inf_vals)
    true_max_val = inf_vals[true_max_idx]
    true_max_theta = theta_plot[true_max_idx]

    print(f"True Max (Grid Search): {true_max_val:.6f} at theta ~ {true_max_theta:.4f}")

    final_level, final_theta, log = maximize_level_value(
        A,
        k,
        gamma_algo,
        max_iter=max_iter,
        print_progress=True,
    )

    print(
        f"Algo Max (Level Set):   {final_level:.6f} at theta = {final_theta:.4f} (inv: {1 / final_level:.4f})"
    )

    plt, level_alpha = configure_matplotlib(output_file, show)
    plt.figure(figsize=(12, 7))
    plt.plot(theta_plot, inf_vals, "k-", linewidth=2, label=r"True $\inf \sigma_2$")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(log)))
    for idx, entry in enumerate(log):
        iter_num = entry["iter"]
        level_value = entry["level"]
        crossings = entry["crossings"]
        next_t = entry.get("next_theta")

        plt.axhline(
            y=level_value,
            color=colors[idx],
            linestyle="--",
            alpha=level_alpha,
            label=rf"Iter {iter_num} level",
        )

        if len(crossings) > 0:
            plt.plot(
                crossings,
                [level_value] * len(crossings),
                "o",
                color=colors[idx],
                markersize=6,
            )

        if next_t is not None and iter_num > 0:
            plt.plot(
                next_t,
                entry["next_level"],
                "*",
                color=colors[idx],
                markersize=12,
                markeredgecolor="k",
            )

    plt.plot(
        final_theta, final_level, "rD", markersize=10, label="Final Result", zorder=10
    )
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\inf \sigma_2$")
    plt.legend(loc="upper right", ncol=1)
    plt.grid(True)
    plt.xlim(0, np.pi)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, format=output_file.suffix.lstrip(".") or None, dpi=350)
    print(f"Saved figure to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()

    error = abs(final_level - true_max_val)
    print(f"Error: {error:.2e}")
    if error < 1e-4:
        print("Test Passed: Algorithm successfully found the peak.")
    else:
        print("Test Warning: Discrepancy found (check gamma or grid resolution).")


def main() -> None:
    """Parse CLI arguments and run level-set behavior visualization."""
    parser = argparse.ArgumentParser(
        description="Run the paper level-set behavior check."
    )
    parser.add_argument(
        "--matrix",
        default="A_hard",
        help="Matrix name defined in examples/matrix_examples.py.",
    )
    parser.add_argument("--k", type=int, default=0, help="Channel index.")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Algorithm gamma.")
    parser.add_argument(
        "--grid-size", type=int, default=1000, help="Grid size for reference search."
    )
    parser.add_argument(
        "--max-iter", type=int, default=10, help="Maximum level-set iterations."
    )
    parser.add_argument(
        "--output",
        default=str(output_path("behaviour_check.png")),
        help="Output figure path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    args = parser.parse_args()

    run_experiment(
        matrix_name=args.matrix,
        k=args.k,
        gamma_algo=args.gamma,
        grid_size=args.grid_size,
        max_iter=args.max_iter,
        output_file=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
