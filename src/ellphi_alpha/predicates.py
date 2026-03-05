"""Predicate checks for candidate anisotropic alpha-complex simplices."""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np

from .minimax import MinimaxResult, solve_minimax

__all__ = [
    "PredicateResult",
    "evaluate_predicates",
]

_DEFAULT_BOUNDARY_TOL = 1e-7
_DEFAULT_EMPTY_TOL = 1e-7


class PredicateResult(NamedTuple):
    """Result of predicate checks for a single simplex."""

    simplex: tuple[int, ...]
    minimax: MinimaxResult
    p1_converged: bool
    p2_boundary: bool
    p3_empty: bool
    accepted: bool


def _canonical_simplex(simplex: Sequence[int], n_vertices: int) -> tuple[int, ...]:
    simplex_t = tuple(int(i) for i in simplex)
    if not simplex_t:
        raise ValueError("simplex must contain at least one vertex")
    if len(set(simplex_t)) != len(simplex_t):
        raise ValueError("simplex cannot contain duplicate vertices")
    if min(simplex_t) < 0 or max(simplex_t) >= n_vertices:
        raise ValueError("simplex indices are out of range")
    return tuple(sorted(simplex_t))


def evaluate_predicates(
    simplex: Sequence[int],
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    minimax_result: MinimaxResult | None = None,
    minimax_kwargs: Mapping[str, object] | None = None,
    boundary_tol: float = _DEFAULT_BOUNDARY_TOL,
    empty_tol: float = _DEFAULT_EMPTY_TOL,
) -> PredicateResult:
    """Evaluate predicates P1-P3 for a simplex.

    P1 (duality/optimization): minimax solver converged to a finite result.
    P2 (boundary contact): all simplex vertices are approximately on the
    minimax boundary level.
    P3 (emptiness): no point outside the simplex lies strictly inside the same
    anisotropic alpha-ball.
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n_vertices = centers.shape[0]
    simplex_t = _canonical_simplex(simplex, n_vertices)
    simplex_idx = np.asarray(simplex_t, dtype=int)

    if minimax_result is None:
        kwargs = dict(minimax_kwargs or {})
        minimax_result = solve_minimax(
            matrices[simplex_idx],
            centers[simplex_idx],
            **kwargs,
        )
    else:
        if len(minimax_result.weights) != len(simplex_t):
            raise ValueError(
                "minimax_result is incompatible with simplex: "
                f"weights length {len(minimax_result.weights)} != {len(simplex_t)}"
            )
        if len(minimax_result.circumcenter) != centers.shape[1]:
            raise ValueError(
                "minimax_result is incompatible with centers: "
                f"circumcenter dimension {len(minimax_result.circumcenter)} "
                f"!= {centers.shape[1]}"
            )

    alpha = float(minimax_result.alpha)
    xstar = minimax_result.circumcenter

    simplex_diff = xstar[np.newaxis, :] - centers[simplex_idx]
    simplex_f = np.einsum("ki,kij,kj->k", simplex_diff, matrices[simplex_idx], simplex_diff)

    p1 = bool(
        minimax_result.converged
        and np.isfinite(alpha)
        and np.all(np.isfinite(minimax_result.weights))
        and np.isclose(np.sum(minimax_result.weights), 1.0, atol=1e-8)
        and np.min(minimax_result.weights) >= -1e-10
    )

    boundary_scale = max(1.0, abs(alpha))
    p2 = bool(np.max(simplex_f) - np.min(simplex_f) <= boundary_tol * boundary_scale)

    outside_mask = np.ones(n_vertices, dtype=bool)
    outside_mask[simplex_idx] = False
    if np.any(outside_mask):
        diff_all = xstar[np.newaxis, :] - centers
        f_all = np.einsum("ki,kij,kj->k", diff_all, matrices, diff_all)
        p3 = bool(np.min(f_all[outside_mask]) >= alpha - empty_tol * boundary_scale)
    else:
        p3 = True

    accepted = p1 and p2 and p3
    return PredicateResult(
        simplex=simplex_t,
        minimax=minimax_result,
        p1_converged=p1,
        p2_boundary=p2,
        p3_empty=p3,
        accepted=accepted,
    )
