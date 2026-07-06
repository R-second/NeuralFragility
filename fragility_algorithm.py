"""Core Neural Fragility (or Stability Radius) level-set algorithm."""

from __future__ import annotations

from typing import Literal, TypeAlias, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.linalg import eig

FloatArray: TypeAlias = NDArray[np.floating]
ComplexArray: TypeAlias = NDArray[np.complexfloating]
IterationLog: TypeAlias = list[dict[str, Any]]
FragilityMethod: TypeAlias = Literal["branch filtering method", "grid search"]


def compute_level_value(
    transition_matrix: FloatArray,
    channel_index: int,
    theta: float,
) -> float:
    """Calculate d(G_R(s)^T, G_I(s)^TR) where s = exp(i*theta).

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to evaluate the perturbation response.
        theta: angle on the unit circle in radians.

    Returns:
        The value of d(G_R(s)^T, G_I(s)^TR). Returns `0.0` if the linear system is singular.
    """
    n_channels = transition_matrix.shape[0]
    point_on_unit_circle = np.exp(1j * theta)
    identity = np.eye(n_channels)

    try:
        selector = np.zeros(n_channels)
        selector[channel_index] = 1.0
        transfer_vector = linalg.solve(
            (point_on_unit_circle * identity - transition_matrix).T,
            selector,
        )
    except np.linalg.LinAlgError:
        return 0.0

    real_part = np.real(transfer_vector)
    imag_part = np.imag(transfer_vector)
    real_norm_sq = np.sum(real_part**2)
    imag_norm_sq = np.sum(imag_part**2)

    if imag_norm_sq < 1e-12:
        return float(np.sqrt(real_norm_sq))

    real_imag_dot = np.dot(real_part, imag_part)
    value_sq = real_norm_sq - (real_imag_dot**2 / imag_norm_sq)
    return float(np.sqrt(np.maximum(value_sq, 0.0)))


def compute_regularized_level_value(
    transition_matrix: FloatArray,
    channel_index: int,
    theta: float,
    gamma: float,
) -> float:
    """Calculate the second singular value of M(gamma, G(s)) where s = exp(i*theta).

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to evaluate the perturbation response.
        theta: angle on the unit circle in radians.
        gamma: parameter

    Returns:
        The second singular value of M(gamma, G(s)) where s = exp(i*theta). Returns `0.0` if the linear system is singular.
    """
    n_channels = transition_matrix.shape[0]
    point_on_unit_circle = np.exp(1j * theta)
    identity = np.eye(n_channels)

    try:
        selector = np.zeros(n_channels)
        selector[channel_index] = 1.0
        transfer_vector = linalg.solve(
            (point_on_unit_circle * identity - transition_matrix).T,
            selector,
        )
    except np.linalg.LinAlgError:
        return 0.0

    real_part = np.real(transfer_vector)
    imag_part = np.imag(transfer_vector)
    real_norm_sq = np.sum(real_part**2)
    imag_norm_sq = np.sum(imag_part**2)
    real_imag_dot_sq = np.dot(real_part, imag_part) ** 2

    base_term = real_norm_sq
    regularized_imag_term = 0.5 * (gamma**2 + 1 / gamma**2) * imag_norm_sq
    sqrt_inner = (gamma + 1 / gamma) ** 2 * (imag_norm_sq**2) + 4 * real_imag_dot_sq
    correction_term = 0.5 * np.abs(gamma - 1 / gamma) * np.sqrt(sqrt_inner)
    value_sq = base_term + regularized_imag_term - correction_term
    return float(np.sqrt(np.maximum(value_sq, 0.0)))


def solve_level_set_eigenproblem(
    transition_matrix: FloatArray,
    channel_index: int,
    level_value: float,
    gamma: float,
) -> ComplexArray:
    """Solve the generalized eigenvalue problem for the level-set method.

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to evaluate the perturbation response.
        level_value: current level-set value.
        gamma: regularization parameter.

    Returns:
        Array of generalized eigenvalues. Returns an empty array if the problem cannot be solved.
    """
    n_channels = transition_matrix.shape[0]
    alpha = (1 + gamma**2) / (2 * gamma)
    beta = (1 - gamma**2) / (2 * gamma)
    inverse_level = 1.0 / level_value

    identity = np.eye(n_channels)
    zeros = np.zeros((n_channels, n_channels))
    phi = inverse_level * identity
    psi = np.zeros((n_channels, n_channels))
    psi[channel_index, channel_index] = inverse_level

    left_matrix = np.block(
        [
            [-transition_matrix, zeros, zeros, beta * phi],
            [alpha * psi, identity, zeros, zeros],
            [zeros, zeros, identity, alpha * phi],
            [-beta * psi, zeros, zeros, -transition_matrix.T],
        ]
    )
    right_matrix = np.block(
        [
            [-identity, -alpha * phi, zeros, zeros],
            [zeros, transition_matrix.T, beta * psi, zeros],
            [zeros, -beta * phi, transition_matrix, zeros],
            [zeros, zeros, -alpha * psi, -identity],
        ]
    )
    return np.array(linalg.eig(left_matrix, right_matrix, right=False))


def extract_unit_circle_angles(
    eigenvalues: ComplexArray,
    tolerance: float = 1e-4,
) -> FloatArray:
    """Extract angles of eigenvalues that are close to the unit circle.

    Args:
        eigenvalues: Array of generalized eigenvalues.
        tolerance: Tolerance for considering an eigenvalue to be on the unit circle.

    Returns:
        Ascending array of candidate angles normalized to `[0, pi]`.
    """
    angles: list[float] = []
    for eigenvalue in eigenvalues:
        if np.abs(np.abs(eigenvalue) - 1.0) < tolerance:
            angles.append(np.abs(np.angle(eigenvalue)))

    if not angles:
        return np.array([])

    rounded_angles = np.round(np.array(angles), 5)
    return np.sort(np.unique(rounded_angles))


def filter_level_crossings(
    transition_matrix: FloatArray,
    channel_index: int,
    gamma: float,
    level_value: float,
    candidate_angles: FloatArray,
    tolerance: float = 1e-4,
) -> FloatArray:
    """Verify candidate angles from the generalized eigenvalue problem against the regularized level-set objective function.

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to evaluate the perturbation response.
        gamma: regularization parameter.
        level_value: level-set value to verify against.
        candidate_angles: candidate angles obtained from the generalized eigenvalue problem.
        tolerance: tolerance for the consistency check with the regularized objective function.

    Returns:
        Array of verified angles.
    """
    verified_angles: list[float] = []
    for theta in candidate_angles:
        candidate_value = compute_regularized_level_value(
            transition_matrix,
            channel_index,
            theta,
            gamma,
        )
        if np.abs(candidate_value - level_value) < tolerance:
            verified_angles.append(theta)
    return np.array(verified_angles)


def find_best_interval_midpoint(
    transition_matrix: FloatArray,
    channel_index: int,
    crossing_angles: FloatArray,
    current_level: float,
) -> tuple[float | None, float]:
    """Find the best midpoint angle between level-set crossings that maximizes the level-set objective function.

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to evaluate the perturbation response.
        crossing_angles: angles at which the current level-set crosses.
        current_level: current best level value.

    Returns:
        Best midpoint angle and its level value. Returns `None` for the angle if no improvement is found.
    """
    boundaries = np.concatenate(([0.0], crossing_angles, [np.pi]))
    boundaries = np.unique(boundaries)
    boundaries.sort()

    best_theta = None
    best_level = current_level
    for start_theta, end_theta in zip(boundaries[:-1], boundaries[1:]):
        if (end_theta - start_theta) < 1e-6:
            continue

        midpoint_theta = (start_theta + end_theta) / 2.0
        candidate_level = compute_level_value(
            transition_matrix,
            channel_index,
            midpoint_theta,
        )
        if candidate_level > best_level:
            best_level = candidate_level
            best_theta = midpoint_theta

    return best_theta, best_level


def maximize_level_value(
    transition_matrix: FloatArray,
    channel_index: int,
    gamma: float,
    max_iter: int = 20,
    print_progress: bool = True,
    epsilon: float = 1e-6,
) -> tuple[float, float, IterationLog]:
    """Maximize the level-set objective function using the level-set method.

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to evaluate the perturbation response.
        gamma: regularization parameter.
        max_iter: maximum number of level-set iterations.
        print_progress: whether to print iteration logs to standard output.
        epsilon: minimum improvement width for convergence criterion.

    Returns:
        Best level value, corresponding angle, and diagnostic log for each iteration.
    """
    value_at_zero = compute_level_value(transition_matrix, channel_index, 0.0)
    value_at_pi = compute_level_value(transition_matrix, channel_index, np.pi)

    if value_at_zero >= value_at_pi:
        current_level = value_at_zero
        best_theta = 0.0
    else:
        current_level = value_at_pi
        best_theta = np.pi

    iteration_log = [
        {"iter": 0, "level": current_level, "crossings": [], "next_theta": best_theta}
    ]
    if print_progress:
        print(
            f"Iter 0: Initial level = {current_level:.6f} at theta = {best_theta:.4f}"
        )

    for iteration in range(1, max_iter + 1):
        eigenvalues = solve_level_set_eigenproblem(
            transition_matrix,
            channel_index,
            current_level,
            gamma,
        )
        if print_progress:
            print(
                f"Iter {iteration}: Solved GEP for level = {current_level:.6f}, "
                f"found {len(eigenvalues)} eigenvalues."
            )

        candidate_angles = extract_unit_circle_angles(eigenvalues)
        if print_progress:
            print(
                f"Iter {iteration}: Extracted {len(candidate_angles)} candidate "
                f"angles on unit circle. Candidates: {candidate_angles}"
            )

        crossing_angles = filter_level_crossings(
            transition_matrix,
            channel_index,
            gamma,
            current_level,
            candidate_angles,
            tolerance=1e-2,
        )
        if print_progress:
            print(
                f"Iter {iteration}: Verified {len(crossing_angles)} angles close "
                f"to current level. Candidates: {crossing_angles}"
            )

        next_theta, next_level = find_best_interval_midpoint(
            transition_matrix,
            channel_index,
            crossing_angles,
            current_level,
        )
        iteration_log.append(
            {
                "iter": iteration,
                "level": current_level,
                "crossings": crossing_angles,
                "next_theta": next_theta if next_theta is not None else best_theta,
                "next_level": next_level,
            }
        )

        if next_theta is None or (next_level - current_level) < epsilon:
            if print_progress:
                print(f"Converged at Iter {iteration}")
                if next_theta is None:
                    print("No new theta found that exceeds current level.")
                else:
                    improvement = next_level - current_level
                    print(
                        f"Improvement {improvement:.6f} is less than epsilon {epsilon:.6f}."
                    )
            break

        if print_progress:
            print(
                f"Iter {iteration}: level updated {current_level:.6f} -> "
                f"{next_level:.6f} at theta = {next_theta:.4f}"
            )
        current_level = next_level
        best_theta = next_theta

    return current_level, best_theta, iteration_log


def compute_stability_radius_grid_search(
    transition_matrix: FloatArray,
    channel_index: int,
    num_points: int = 1000,
) -> float:
    """Approximate Stability Radius using grid search.

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to compute stability radius for.
        num_points: number of grid points to evaluate on `[0, pi]`.

    Returns:
        Approximated stability radius value as the reciprocal of the maximum level value.
    """
    thetas = np.linspace(0, np.pi, num_points)
    peak_level = max(
        compute_level_value(transition_matrix, channel_index, theta) for theta in thetas
    )
    return 1.0 / peak_level if peak_level != 0 else np.inf


def compute_stability_radius(
    transition_matrix: FloatArray,
    channel_index: int,
    gamma: float = 0.01,
    max_iter: int = 20,
    print_progress: bool = True,
    epsilon: float = 1e-6,
    method: FragilityMethod = "branch filtering method",
    grid_points: int = 1000,
) -> tuple[float, float, IterationLog]:
    """Compute the stability radius of a given channel.

    Args:
        transition_matrix: transition matrix of shape `(n_channels, n_channels)`.
        channel_index: index of the channel or node to compute stability radius for.
        gamma: regularization parameter when `method="branch filtering method"`.
        max_iter: maximum number of level-set iterations when `method="branch filtering method"`.
        print_progress: whether to print iteration logs to standard output when `method="branch filtering method"`.
        epsilon: minimum improvement width for convergence criterion when `method="branch filtering method"`.
        method: stability radius computation method, either `"branch filtering method"` or `"grid search"`.
        grid_points: number of grid points to use when `method="grid search"`.

    Returns:
        Best stability radius value, corresponding angle, and diagnostic log for each iteration.
    """
    if method == "grid search":
        thetas = np.linspace(0, np.pi, grid_points)
        levels = np.array(
            [
                compute_level_value(transition_matrix, channel_index, theta)
                for theta in thetas
            ]
        )
        peak_index = int(np.argmax(levels))
        peak_level = float(levels[peak_index])
        peak_theta = float(thetas[peak_index])
        stability_radius = 1.0 / peak_level if peak_level != 0 else np.inf
        iteration_log = [
            {
                "method": method,
                "grid_points": grid_points,
                "level": peak_level,
                "theta": peak_theta,
            }
        ]
        return stability_radius, peak_theta, iteration_log

    if method != "branch filtering method":
        raise ValueError("method must be 'branch filtering method' or 'grid search'.")

    peak_level, peak_theta, iteration_log = maximize_level_value(
        transition_matrix,
        channel_index,
        gamma,
        max_iter,
        print_progress=print_progress,
        epsilon=epsilon,
    )
    stability_radius = 1.0 / peak_level if peak_level != 0 else np.inf
    return stability_radius, peak_theta, iteration_log
