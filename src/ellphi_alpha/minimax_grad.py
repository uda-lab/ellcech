"""Gradient computation for the anisotropic minimax filtration value.

Mathematical background
-----------------------
Given a solved MinimaxResult with optimal weights mu* and circumcenter x*,
the filtration value alpha(sigma; theta) is differentiable for alpha > 0
exactly when the dual optimal set is a singleton (Theorem 4.1,
minimax_solver_analysis.tex §4).  A unique optimizer may still have zero
weights; those indices simply have zero gradient blocks.  The alpha = 0 case
is exceptional.

By the envelope theorem applied to the dual:

    d alpha / d xbar_k = 2 mu_k* A_k (xbar_k - x*)      (Eq. grad_center)
    d alpha / d A_k     = mu_k* outer(xbar_k - x*, xbar_k - x*)  (Eq. grad_matrix)

For k not in the active set (mu_k* = 0), both gradients are zero.

These formulas are the k-simplex generalization of ellphi's pairwise gradients:
    For |sigma|=2: d alpha/d xbar_1 = 2 mu_1* A_1 (xbar_1 - x*), etc.

Usage
-----
    result = solve_minimax(matrices, centers, method="fw+brentq+newton")
    grad = compute_gradient(result, centers, matrices)
    # grad.d_xbar[k]: shape (d,) — gradient w.r.t. k-th center
    # grad.d_A[k]:    shape (d,d) — gradient w.r.t. k-th metric matrix

Reference
---------
minimax_solver_analysis.tex §4 (Differentiability), Eq. (grad_center), (grad_matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .minimax import MinimaxResult

__all__ = [
    "GradientResult",
    "compute_gradient",
]


@dataclass
class GradientResult:
    """Gradient of alpha(sigma) with respect to input parameters.

    Attributes:
        d_xbar: Gradients w.r.t. centers xbar_k, shape (d,) each.
                Zero for k not in the active set.
        d_A:    Gradients w.r.t. metric matrices A_k, shape (d,d) each.
                Symmetric outer product; zero for k not in the active set.
        alpha:  The filtration value (from the forward pass).
        circumcenter: The optimal x* (from the forward pass).
        weights: The optimal mu* (from the forward pass).
    """

    d_xbar: list[np.ndarray]
    d_A: list[np.ndarray]
    alpha: float
    circumcenter: np.ndarray
    weights: np.ndarray


def compute_gradient(
    result: MinimaxResult,
    centers: np.ndarray | Sequence[np.ndarray],
    matrices: np.ndarray | Sequence[np.ndarray],
) -> GradientResult:
    """Compute gradients of alpha(sigma) w.r.t. input centers and metric matrices.

    Uses the envelope theorem (Eq. grad_center and grad_matrix from
    minimax_solver_analysis.tex §4):

        d alpha / d xbar_k = 2 mu_k* A_k (xbar_k - x*)
        d alpha / d A_k    = mu_k* outer(xbar_k - x*, xbar_k - x*)

    Args:
        result:   A MinimaxResult from solve_minimax (any method).
        centers:  Array of centers xbar_i, shape (k, d).
        matrices: Array of SPD matrices A_i, shape (k, d, d).

    Returns:
        GradientResult with d_xbar and d_A lists (length k each).

    Notes:
        - For alpha > 0, the gradient exists exactly when the dual optimal
          set is a singleton; this does not require every mu_k* to be
          strictly positive.
        - When the unique optimizer has mu_k* = 0, the corresponding center
          and matrix gradient blocks are zero, not evidence of
          non-differentiability.
        - The alpha = 0 case is exceptional: the value is differentiable with
          zero gradient even when the dual optimizer is not unique.
        - The formula is exact (no finite differences), O(k d^2) cost.
    """
    centers = np.asarray(centers, dtype=float)
    matrices = np.asarray(matrices, dtype=float)

    if centers.ndim == 1:
        centers = centers[np.newaxis]
        matrices = matrices[np.newaxis]

    k, d = centers.shape
    if matrices.shape != (k, d, d):
        raise ValueError(
            f"matrices shape {matrices.shape} inconsistent with centers shape {centers.shape}"
        )
    if len(result.weights) != k:
        raise ValueError(
            f"result.weights length {len(result.weights)} does not match k={k}"
        )

    xstar = result.circumcenter  # shape (d,)
    mu = result.weights          # shape (k,)

    d_xbar = []
    d_A = []

    for i in range(k):
        diff = centers[i] - xstar           # xbar_i - x*, shape (d,)
        mu_i = float(mu[i])

        # d alpha / d xbar_i = 2 mu_i A_i (xbar_i - x*)
        grad_xbar_i = 2.0 * mu_i * (matrices[i] @ diff)   # shape (d,)

        # d alpha / d A_i = mu_i outer(xbar_i - x*, xbar_i - x*)
        grad_A_i = mu_i * np.outer(diff, diff)             # shape (d, d)

        d_xbar.append(grad_xbar_i)
        d_A.append(grad_A_i)

    return GradientResult(
        d_xbar=d_xbar,
        d_A=d_A,
        alpha=result.alpha,
        circumcenter=xstar.copy(),
        weights=mu.copy(),
    )
