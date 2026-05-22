"""Copy of research_matsuda/sreedhar_alg.py adapted for package use."""
import numpy as np
from scipy import linalg
from scipy.linalg import eig

def calculate_inf_sigma2_single(A, k, theta):
    n = A.shape[0]
    s = np.exp(1j * theta)
    I = np.eye(n)
    try:
        rhs = np.zeros(n)
        rhs[k] = 1.0
        x = linalg.solve((s * I - A).T, rhs)
        G_s = x
    except np.linalg.LinAlgError:
        return 0.0

    G_R = np.real(G_s)
    G_I = np.imag(G_s)
    n_gr2 = np.sum(G_R**2)
    n_gi2 = np.sum(G_I**2)
    if n_gi2 < 1e-12:
        return np.sqrt(n_gr2)
    dot_prod = np.dot(G_R, G_I)
    val_sq = n_gr2 - (dot_prod**2 / n_gi2)
    return np.sqrt(np.maximum(val_sq, 0.0))

def calculate_sigma2_with_gamma(A, k, theta, gamma):
    n = A.shape[0]
    s = np.exp(1j * theta)
    I = np.eye(n)
    try:
        rhs = np.zeros(n)
        rhs[k] = 1.0
        x = linalg.solve((s * I - A).T, rhs)
        G_s = x
    except np.linalg.LinAlgError:
        return 0.0

    G_R = np.real(G_s)
    G_I = np.imag(G_s)
    n_gr2 = np.sum(G_R**2)
    n_gi2 = np.sum(G_I**2)
    dot_prod = np.dot(G_R, G_I)
    dot_prod2 = dot_prod**2
    term1 = n_gr2
    term2 = 0.5 * (gamma**2 + 1/gamma**2) * n_gi2
    sqrt_inner = (gamma + 1/gamma)**2 * (n_gi2**2) + 4 * dot_prod2
    term3 = 0.5 * np.abs(gamma - 1/gamma) * np.sqrt(sqrt_inner)
    val_sq = term1 + term2 - term3
    return np.sqrt(np.maximum(val_sq, 0.0))


def solve_generalized_eigenvalue_problem(A, k, xi, gamma):
    n = A.shape[0]
    alpha = (1 + gamma**2) / (2 * gamma)
    beta  = (1 - gamma**2) / (2 * gamma)
    inv_xi = 1.0 / xi
    Phi = inv_xi * np.eye(n)
    Psi = np.zeros((n, n))
    Psi[k, k] = inv_xi
    Z = np.zeros((n, n))
    I = np.eye(n)
    F = np.block([
        [-A,             Z, Z, beta * Phi],
        [alpha * Psi,    I, Z,          Z],
        [Z,              Z, I, alpha * Phi],
        [-beta * Psi,    Z, Z,       -A.T]
    ])
    G = np.block([
        [-I, -alpha * Phi,           Z,           Z],
        [ Z,          A.T,  beta * Psi,           Z],
        [ Z,  -beta * Phi,           A,           Z],
        [ Z,            Z, -alpha * Psi,          -I]
    ])
    eigenvalues = eig(F, G, right=False)
    return eigenvalues

def extract_thetas_on_unit_circle(eigenvalues, tolerance=1e-4):
    thetas = []
    for lam in eigenvalues:
        if np.abs(np.abs(lam) - 1.0) < tolerance:
            angle = np.angle(lam)
            theta_norm = np.abs(angle)
            thetas.append(theta_norm)
    if not thetas:
        return np.array([])
    thetas = np.array(thetas)
    thetas_rounded = np.round(thetas, 5)
    unique_thetas = np.unique(thetas_rounded)
    return np.sort(unique_thetas)

def verify_crossings(A, k, gamma, xi, candidate_thetas, tol=1e-4):
    verified_list = []
    for theta in candidate_thetas:
        sigma2_val = calculate_sigma2_with_gamma(A, k, theta, gamma)
        if np.abs(sigma2_val - xi) < tol:
            verified_list.append(theta)
    return np.array(verified_list)

def find_next_peak_in_intervals(A, k, sorted_thetas, current_xi):
    boundaries = np.concatenate(([0.0], sorted_thetas, [np.pi]))
    boundaries = np.unique(boundaries)
    boundaries.sort()
    best_theta = None
    best_val = current_xi
    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i+1]
        if (t_end - t_start) < 1e-6:
            continue
        t_mid = (t_start + t_end) / 2.0
        val = calculate_inf_sigma2_single(A, k, t_mid)
        if val > best_val:
            best_val = val
            best_theta = t_mid
    return best_theta, best_val

def optimize_sigma2_inf_main(A, k, gamma, max_iter=20, print_progress=True, epsilon=1e-6):
    val_0 = calculate_inf_sigma2_single(A, k, 0.0)
    val_pi = calculate_inf_sigma2_single(A, k, np.pi)
    if val_0 >= val_pi:
        current_xi = val_0
        best_theta = 0.0
    else:
        current_xi = val_pi
        best_theta = np.pi
    log = []
    log.append({'iter': 0,'xi': current_xi,'crossings': [],'next_theta': best_theta})
    if print_progress:
        print(f"Iter 0: Initial xi = {current_xi:.6f} at theta = {best_theta:.4f}")
    for i in range(1, max_iter + 1):
        eigenvalues = solve_generalized_eigenvalue_problem(A, k, current_xi, gamma)
        if print_progress:
            print(f"Iter {i}: Solved GEP for xi = {current_xi:.6f}, found {len(eigenvalues)} eigenvalues.")
        raw_candidates = extract_thetas_on_unit_circle(eigenvalues)
        if print_progress:
            print(f"Iter {i}: Extracted {len(raw_candidates)} candidate thetas on unit circle. Candidates: {raw_candidates}")
        verified_thetas = verify_crossings(A, k, gamma, current_xi, raw_candidates, tol=1e-2)
        if print_progress:
            print(f"Iter {i}: Verified {len(verified_thetas)} thetas that are close to current xi. Candidates: {verified_thetas}")
        next_theta, next_val = find_next_peak_in_intervals(A, k, verified_thetas, current_xi)
        log.append({'iter': i,'xi': current_xi,'crossings': verified_thetas,'next_theta': next_theta if next_theta is not None else best_theta,'next_val': next_val})
        if next_theta is None or (next_val - current_xi) < epsilon:
            if print_progress:
                print(f"Converged at Iter {i}")
                if next_theta is None:
                    print("No new theta found that exceeds current xi.")
                else:
                    print(f"Improvement {next_val - current_xi:.6f} is less than epsilon {epsilon:.6f}.")
            break
        if print_progress:
            print(f"Iter {i}: xi updated {current_xi:.6f} -> {next_val:.6f} at theta = {next_theta:.4f}")
        current_xi = next_val
        best_theta = next_theta
    return current_xi, best_theta, log

def neural_fragility_inf(A, k, gamma=0.01, max_iter=20, print_progress=True, epsilon=1e-6):
    final_xi, final_theta, log = optimize_sigma2_inf_main(
        A,
        k,
        gamma,
        max_iter,
        print_progress=print_progress,
        epsilon=epsilon,
    )
    fragility = 1.0 / final_xi if final_xi != 0 else np.inf
    return fragility, final_theta, log
