"""Frank-Wolfe minimax solver for anisotropic alpha-complex filtration values.

Mathematical background
-----------------------
Given a simplex sigma = {0, ..., k-1} with points x_i in R^d and SPD matrices A_i,
the filtration value is defined (LeanEllAlpha T1-T3) as:

    alpha(sigma) = min_x  max_{i in sigma} f_i(x),   f_i(x) = (x - x_i)^T A_i (x - x_i)

By strong duality (LeanEllAlpha T5, concavity of g on Delta), this equals:

    alpha(sigma) = max_{mu in Delta^{k-1}} g(mu)

where (LeanEllAlpha T4):

    A(mu) = sum_i mu_i A_i
    b(mu) = sum_i mu_i A_i x_i
    g(mu) = sum_i mu_i x_i^T A_i x_i - b(mu)^T A(mu)^{-1} b(mu)

The circumcenter (optimal point) is x*(mu) = A(mu)^{-1} b(mu)  (T4).
The gradient is dg/dmu_i = f_i(x*(mu))  (used by Frank-Wolfe direction finding).

Algorithm: Pairwise Frank-Wolfe with exact 1D line search
---------------------------------------------------------
Each iteration:
  s = argmax_i f_i(x*)         (most violated, FW vertex)
  v = argmin_{i: mu_i > 0} f_i  (least violated, away vertex)
  Pairwise step: mu <- mu + gamma*(e_s - e_v), gamma in [0, mu_v]
  Exact gamma: bisect on dg/dgamma = f_s(x*(gamma)) - f_v(x*(gamma)) = 0

For the pairwise case |sigma| = 2, this finds the optimum in one step.
For |sigma| >= 3, linear convergence is achieved (superior to O(1/T) fixed-step FW).

Correspondence with ellphi
--------------------------
For the pairwise case |sigma| = 2:

- ellphi solves: F(mu) = f_i(x*(mu)) - f_j(x*(mu)) = 0  (1D root-finding)
- This solver maximises: g over Delta^1 via pairwise FW  (equivalent)
- Both find the same x*; dg/dmu = -F(mu).

Sign convention note:
  ellphi packed coef stores b_ellphi = -A x_bar  (negative linear term)
  so the center is recovered as  x_bar = -A^{-1} b_ellphi
  Use `solve_minimax_from_coefs` to handle this automatically.

Value correspondence:
  alpha(sigma) here  =  t^2  where t = ellphi.tangency(...).t
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import linalg

__all__ = [
    "MinimaxResult",
    "solve_minimax",
    "solve_minimax_from_coefs",
]

_DEFAULT_TOL = 1e-9
_DEFAULT_MAX_ITER = 2000
_DEFAULT_WEIGHT_TOL = 1e-10
_N_BISECT = 52  # ~machine precision for double


class MinimaxResult(NamedTuple):
    """Result of the pairwise Frank-Wolfe minimax solver.

    Attributes:
        alpha: Filtration value alpha(sigma) = max_i f_i(x*).
               Equals t^2 where t is ellphi's tangency distance (pairwise case).
        circumcenter: Optimal point x* = A(mu*)^{-1} b(mu*).
        weights: Optimal weights mu* in Delta^{k-1}, shape (k,).
        active_set: Indices i where mu*_i > weight_tol  (KKT active set, T6-T7).
        converged: Whether the Frank-Wolfe gap fell below tol.
        n_iter: Number of Frank-Wolfe iterations performed.
    """

    alpha: float
    circumcenter: np.ndarray
    weights: np.ndarray
    active_set: list[int]
    converged: bool
    n_iter: int


def _cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b via Cholesky; fall back to least-squares if singular."""
    try:
        chol = linalg.cho_factor(A, check_finite=False)
        return linalg.cho_solve(chol, b, check_finite=False)
    except linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _eval_f(mu: np.ndarray, matrices: np.ndarray, Ax: np.ndarray, centers: np.ndarray):
    """Compute circumcenter x*(mu) and f_i(x*(mu)) for all i."""
    A_mu = np.einsum("k,kij->ij", mu, matrices)
    b_mu = np.einsum("k,ki->i", mu, Ax)
    xstar = _cholesky_solve(A_mu, b_mu)
    diff = xstar[np.newaxis, :] - centers
    f = np.einsum("ki,kij,kj->k", diff, matrices, diff)
    return xstar, f


def _exact_line_search(
    s: int,
    v: int,
    mu: np.ndarray,
    matrices: np.ndarray,
    Ax: np.ndarray,
    centers: np.ndarray,
    gamma_max: float,
) -> float:
    """Find gamma* in [0, gamma_max] that maximises g along the pairwise direction.

    Bisects on h(gamma) = f_s(x*(mu + gamma*(e_s - e_v))) - f_v(...) = 0.
    dg/dgamma = f_s(x*(gamma)) - f_v(x*(gamma)) (positive at 0, negative at gamma_max).
    """
    if gamma_max <= 0.0:
        return 0.0

    def h(gamma: float) -> float:
        mu_g = mu.copy()
        mu_g[s] += gamma
        mu_g[v] -= gamma
        _, f_g = _eval_f(mu_g, matrices, Ax, centers)
        return float(f_g[s] - f_g[v])

    h0 = h(0.0)
    if h0 <= 0.0:
        return 0.0  # already at or past optimum in this direction

    h_max = h(gamma_max)
    if h_max >= 0.0:
        return gamma_max  # optimum is at the boundary

    lo, hi = 0.0, gamma_max
    for _ in range(_N_BISECT):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_minimax(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
    weight_tol: float = _DEFAULT_WEIGHT_TOL,
) -> MinimaxResult:
    """Compute filtration value alpha(sigma) via pairwise Frank-Wolfe on dual g(mu).

    Uses exact 1D line search (bisection) at each step, giving linear convergence.
    For the pairwise case |sigma| = 2, the optimum is found in a single step.

    Args:
        matrices: SPD matrices A_i, shape (k, d, d).
        centers: Points x_bar_i, shape (k, d).
        tol: Convergence tolerance on the Frank-Wolfe duality gap
             gap = f_s(x*) - sum_i mu_i f_i(x*)  >= 0.
        max_iter: Maximum number of pairwise Frank-Wolfe iterations.
        weight_tol: Threshold below which mu_i is treated as zero.

    Returns:
        MinimaxResult.
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)

    # Accept single-simplex input (2D matrix, 1D center)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis]
        centers = centers[np.newaxis]

    k, d = centers.shape

    # --- Trivial case: single point ---
    if k == 1:
        return MinimaxResult(
            alpha=0.0,
            circumcenter=centers[0].copy(),
            weights=np.array([1.0]),
            active_set=[0],
            converged=True,
            n_iter=0,
        )

    # Precompute A_i x_bar_i  (shape: k x d)
    Ax = np.einsum("kij,kj->ki", matrices, centers)

    # Uniform initialisation
    mu = np.full(k, 1.0 / k)

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        xstar, f = _eval_f(mu, matrices, Ax, centers)

        # FW vertex (most violated) and away vertex (least violated in active set)
        s = int(np.argmax(f))
        active_mask = mu > weight_tol
        f_active = np.where(active_mask, f, np.inf)
        v = int(np.argmin(f_active))

        # FW duality gap: >= 0; zero iff mu = mu*
        fw_gap = float(f[s] - np.dot(mu, f))

        if fw_gap < tol:
            converged = True
            break

        if s == v:
            # Degenerate: all active vertices have the same f value
            converged = True
            break

        # Pairwise step: d = e_s - e_v, gamma in [0, mu_v]
        gamma_max = float(mu[v])
        gamma = _exact_line_search(s, v, mu, matrices, Ax, centers, gamma_max)

        mu = mu.copy()
        mu[s] += gamma
        mu[v] -= gamma
        # Numerical safety
        mu = np.clip(mu, 0.0, None)
        mu /= mu.sum()

    # Final evaluation
    xstar, f = _eval_f(mu, matrices, Ax, centers)
    alpha = float(np.max(f))
    active_set = [i for i in range(k) if mu[i] > weight_tol]

    return MinimaxResult(
        alpha=alpha,
        circumcenter=xstar,
        weights=mu,
        active_set=active_set,
        converged=converged,
        n_iter=n_iter,
    )


def solve_minimax_from_coefs(
    coefs: np.ndarray,
    **kwargs,
) -> MinimaxResult:
    """Convenience wrapper: accept ellphi packed conic coefficient vectors.

    Converts ellphi's packed representation (A, b_ellphi, c) to (A_i, x_bar_i)
    and calls solve_minimax.

    Sign convention:
        ellphi stores  b_ellphi = -A x_bar  (negative linear term in x^T A x + 2b^T x + c)
        so  x_bar = -A^{-1} b_ellphi

    Value correspondence:
        result.alpha  ==  ellphi.tangency(coefs[0], coefs[1]).t ** 2  (pairwise case)

    Args:
        coefs: Packed conic coefficient array, shape (k, m) or (m,) for k=1.
        **kwargs: Forwarded to solve_minimax (tol, max_iter, weight_tol).

    Returns:
        MinimaxResult.
    """
    from ellphi.geometry import unpack_conic

    coefs = np.asarray(coefs, dtype=float)
    if coefs.ndim == 1:
        coefs = coefs[np.newaxis]

    A_arr, b_arr, _ = unpack_conic(coefs)  # (k,d,d), (k,d), (k,)

    # x_bar_i = -A_i^{-1} b_i  (ellphi sign convention: b = -A x_bar)
    k = A_arr.shape[0]
    centers = np.empty_like(b_arr)
    for i in range(k):
        centers[i] = _cholesky_solve(A_arr[i], -b_arr[i])

    return solve_minimax(A_arr, centers, **kwargs)
