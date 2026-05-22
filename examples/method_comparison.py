"""Compare grid search and the proposed level-set method."""
try:
    from ._bootstrap import output_path
    from .convergence_analysis import generate_stable_matrix
    from .plotting import configure_paper_matplotlib
except ImportError:
    from _bootstrap import output_path
    from convergence_analysis import generate_stable_matrix
    from plotting import configure_paper_matplotlib

import argparse
from pathlib import Path
import time

import numpy as np

from sreedhar_alg import calculate_inf_sigma2_single, optimize_sigma2_inf_main


DEFAULT_GRID_RESOLUTIONS = [100, 300, 1000, 3000, 6000]
DEFAULT_EPSILON_VALUES = [1e-1, 1e-3, 1e-6]


def run_grid_search(A, k, num_points):
    thetas = np.linspace(0, np.pi, num_points)
    max_val = -1.0

    start_time = time.perf_counter()
    for theta in thetas:
        val = calculate_inf_sigma2_single(A, k, theta)
        if val > max_val:
            max_val = val
    elapsed = time.perf_counter() - start_time

    return max_val, elapsed


def run_proposed_method(A, k, epsilon, gamma=0.01, max_iter=20):
    start_time = time.perf_counter()
    val, _, _ = optimize_sigma2_inf_main(
        A,
        k,
        gamma,
        max_iter=max_iter,
        print_progress=False,
        epsilon=epsilon,
    )
    elapsed = time.perf_counter() - start_time
    return val, elapsed


def run_comparison_experiment(
    matrix_size,
    num_trials,
    grid_resolutions,
    epsilon_values,
    k_idx,
    gamma_algo,
    max_iter,
    spectral_radius,
    seed,
):
    rng = np.random.default_rng(seed)

    avg_grid_times = np.zeros(len(grid_resolutions))
    avg_grid_errors = np.zeros(len(grid_resolutions))
    avg_prop_times = np.zeros(len(epsilon_values))
    avg_prop_errors = np.zeros(len(epsilon_values))

    print(f"Comparing methods for N={matrix_size} with {num_trials} trials...")
    for trial_idx in range(num_trials):
        A = generate_stable_matrix(matrix_size, spectral_radius=spectral_radius, rng=rng)
        true_val, _ = run_proposed_method(A, k_idx, epsilon=1e-12, gamma=gamma_algo, max_iter=max_iter)

        for idx, resolution in enumerate(grid_resolutions):
            val, elapsed = run_grid_search(A, k_idx, resolution)
            avg_grid_times[idx] += elapsed
            avg_grid_errors[idx] += abs(val - true_val)

        for idx, epsilon in enumerate(epsilon_values):
            val, elapsed = run_proposed_method(A, k_idx, epsilon=epsilon, gamma=gamma_algo, max_iter=max_iter)
            avg_prop_times[idx] += elapsed
            avg_prop_errors[idx] += abs(val - true_val)

        print(f"Trial {trial_idx + 1:>3}/{num_trials}: reference xi={true_val:.6f}")

    avg_grid_times /= num_trials
    avg_grid_errors /= num_trials
    avg_prop_times /= num_trials
    avg_prop_errors /= num_trials

    return {
        "matrix_size": matrix_size,
        "num_trials": num_trials,
        "grid_resolutions": np.array(grid_resolutions, dtype=int),
        "epsilon_values": np.array(epsilon_values, dtype=float),
        "avg_grid_times": avg_grid_times,
        "avg_grid_errors": avg_grid_errors,
        "avg_prop_times": avg_prop_times,
        "avg_prop_errors": avg_prop_errors,
    }


def save_results(results, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **results)
    print(f"Saved results to {output_file}")


def load_results(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def plot_single_comparison(results, output_file, show):
    plt, _ = configure_paper_matplotlib(output_file, show)
    plt.figure(figsize=(10, 7))

    grid_times = np.maximum(results["avg_grid_times"], 1e-16)
    grid_errors = np.maximum(results["avg_grid_errors"], 1e-16)
    prop_times = np.maximum(results["avg_prop_times"], 1e-16)
    prop_errors = np.maximum(results["avg_prop_errors"], 1e-16)

    plt.loglog(grid_times, grid_errors, "o-", label="Ad-hoc (Grid Search)", color="blue", markersize=6)
    for i, resolution in enumerate(results["grid_resolutions"]):
        plt.annotate(
            f"M={int(resolution)}",
            (grid_times[i], grid_errors[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="blue",
        )

    plt.loglog(prop_times, prop_errors, "*-", label="Proposed Method", color="red", markersize=10)
    for i, epsilon in enumerate(results["epsilon_values"]):
        plt.annotate(
            f"eps={epsilon:.0e}",
            (prop_times[i], prop_errors[i]),
            textcoords="offset points",
            xytext=(30, 0),
            ha="center",
            fontsize=8,
            color="red",
        )

    matrix_size = int(results["matrix_size"])
    plt.xlabel("Computation Time [s]")
    plt.ylabel("Error")
    plt.title(f"Performance Comparison (N={matrix_size})")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(1e-14, 1)
    plt.legend()
    save_figure(plt, output_file, show)


def plot_combined_comparison(results_by_size, output_file, show):
    plt, _ = configure_paper_matplotlib(output_file, show)
    plt.rcParams["font.size"] = 15
    plt.figure(figsize=(10, 7))

    line_styles = ["-", "--", ":", "-."]
    for i, results in enumerate(results_by_size):
        matrix_size = int(results["matrix_size"])
        line_style = line_styles[i % len(line_styles)]

        grid_times = np.maximum(results["avg_grid_times"], 1e-16)
        grid_errors = np.maximum(results["avg_grid_errors"], 1e-16)
        prop_times = np.maximum(results["avg_prop_times"], 1e-16)
        prop_errors = np.maximum(results["avg_prop_errors"], 1e-16)

        plt.loglog(
            grid_times,
            grid_errors,
            "o" + line_style,
            label=f"Grid Search N={matrix_size}",
            markersize=6,
            color="blue",
            alpha=0.7,
        )
        for j, resolution in enumerate(results["grid_resolutions"]):
            if j % 2 == 0:
                plt.annotate(
                    f"M={int(resolution)}",
                    (grid_times[j], grid_errors[j]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=12,
                    color="black",
                )

        plt.loglog(
            prop_times,
            prop_errors,
            "*" + line_style,
            label=f"Proposed Method N={matrix_size}",
            markersize=12,
            color="red",
            alpha=0.7,
        )
        for j, epsilon in enumerate(results["epsilon_values"]):
            if j % 2 == 0:
                plt.annotate(
                    f"eps={epsilon:.0e}",
                    (prop_times[j], prop_errors[j]),
                    textcoords="offset points",
                    xytext=(10, -8),
                    ha="left",
                    fontsize=12,
                    color="black",
                )

    plt.xlabel("Computation Time [s]")
    plt.ylabel("Error")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(1e-10, 1)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, borderaxespad=0.0)
    plt.subplots_adjust(bottom=0.25)
    save_figure(plt, output_file, show, tight_layout=False)


def save_figure(plt, output_file, show, tight_layout=True):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(output_file, dpi=350)
    print(f"Saved figure to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare grid search and proposed method runtimes/errors.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100], help="Matrix sizes to run.")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per matrix size.")
    parser.add_argument(
        "--grid-resolutions",
        nargs="+",
        default=DEFAULT_GRID_RESOLUTIONS,
        type=int,
        help="Grid search resolutions.",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        default=DEFAULT_EPSILON_VALUES,
        type=float,
        help="Convergence tolerances for the proposed method.",
    )
    parser.add_argument("--k", type=int, default=0, help="Perturbed node index.")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Algorithm gamma.")
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum level-set iterations.")
    parser.add_argument("--spectral-radius", type=float, default=0.95, help="Target spectral radius.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--results-dir", default=str(output_path("comparison_results")), help="Directory for .npz results.")
    parser.add_argument("--plot-dir", default=str(output_path("")), help="Directory for plot images.")
    parser.add_argument("--skip-run", action="store_true", help="Load existing .npz results and only plot.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plot_dir = Path(args.plot_dir)
    all_results = []

    for offset, matrix_size in enumerate(args.sizes):
        result_file = results_dir / f"comparison_results_N{matrix_size}.npz"

        if args.skip_run:
            results = load_results(result_file)
        else:
            results = run_comparison_experiment(
                matrix_size=matrix_size,
                num_trials=args.trials,
                grid_resolutions=args.grid_resolutions,
                epsilon_values=args.epsilons,
                k_idx=args.k,
                gamma_algo=args.gamma,
                max_iter=args.max_iter,
                spectral_radius=args.spectral_radius,
                seed=args.seed + offset,
            )
            save_results(results, result_file)

        all_results.append(results)
        single_plot = plot_dir / f"comparison_N{matrix_size}.png"
        plot_single_comparison(results, single_plot, args.show)

    if len(all_results) > 1:
        plot_combined_comparison(all_results, plot_dir / "comparison_all_N.png", args.show)

    print("Comparison complete.")


if __name__ == "__main__":
    main()
