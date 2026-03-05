"""Incremental anisotropic alpha-filtration construction."""

from __future__ import annotations

from collections.abc import Hashable, MutableMapping
from typing import Mapping, NamedTuple, TypeAlias

import numpy as np

from .enumeration import iter_candidate_simplices
from .predicates import PredicateResult, evaluate_predicates

__all__ = [
    "FiltrationEntry",
    "PredicateCacheKey",
    "build_incremental_filtration",
]

PredicateCacheKey: TypeAlias = tuple[
    tuple[int, ...],  # simplex
    tuple[tuple[str, Hashable], ...],  # normalized minimax kwargs
    float,  # boundary_tol
    float,  # empty_tol
]


class FiltrationEntry(NamedTuple):
    """Single simplex entry in an alpha filtration."""

    simplex: tuple[int, ...]
    alpha: float
    predicates: PredicateResult | None


def _codim_one_faces(simplex: tuple[int, ...]) -> list[tuple[int, ...]]:
    if len(simplex) <= 1:
        return []
    return [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))]


def _freeze_cache_value(value: object) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_cache_value(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_cache_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_cache_value(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(value.shape), str(value.dtype), value.tobytes())
    return repr(value)


def _predicate_cache_key(
    simplex: tuple[int, ...],
    *,
    minimax_kwargs: Mapping[str, object] | None,
    boundary_tol: float,
    empty_tol: float,
) -> PredicateCacheKey:
    frozen_kwargs = tuple(
        sorted(
            (str(k), _freeze_cache_value(v))
            for k, v in (minimax_kwargs or {}).items()
        )
    )
    return (simplex, frozen_kwargs, float(boundary_tol), float(empty_tol))


def build_incremental_filtration(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    max_dim: int,
    minimax_kwargs: Mapping[str, object] | None = None,
    boundary_tol: float = 1e-7,
    empty_tol: float = 1e-7,
    predicate_cache: MutableMapping[PredicateCacheKey, PredicateResult] | None = None,
) -> list[FiltrationEntry]:
    """Build filtration entries using candidate enumeration + P1-P3 checks.

    The builder enforces:
    1. Downward closure: each inserted simplex has all codimension-1 faces present.
    2. Monotone alpha values along face inclusions.
    3. Output sorted by (alpha, simplex dimension, lexicographic simplex).

    Optional caching:
    - Provide ``predicate_cache`` (typically a dict) to reuse predicate/minimax
      results across repeated builds on the same dataset and settings.
    - Cache keys include simplex, ``minimax_kwargs``, ``boundary_tol``, and
      ``empty_tol``.
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)

    if matrices.ndim != 3:
        raise ValueError("matrices must have shape (n, d, d)")
    if centers.ndim != 2:
        raise ValueError("centers must have shape (n, d)")
    if matrices.shape[0] != centers.shape[0]:
        raise ValueError("matrices and centers must have the same number of vertices")
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError("each matrix must be square")
    if matrices.shape[1] != centers.shape[1]:
        raise ValueError("matrix and center dimensions must agree")
    if max_dim < 0:
        return []

    n_vertices = centers.shape[0]
    alpha_by_simplex: dict[tuple[int, ...], float] = {}
    entries: list[FiltrationEntry] = []

    for simplex in iter_candidate_simplices(n_vertices, max_dim=max_dim):
        if len(simplex) == 1:
            alpha_by_simplex[simplex] = 0.0
            entries.append(FiltrationEntry(simplex=simplex, alpha=0.0, predicates=None))
            continue

        faces = _codim_one_faces(simplex)
        if any(face not in alpha_by_simplex for face in faces):
            # Faces-first prune: if downward closure cannot be satisfied, skip
            # expensive predicate/minimax evaluation for this simplex.
            continue

        cache_key = None
        pred = None
        if predicate_cache is not None:
            cache_key = _predicate_cache_key(
                simplex,
                minimax_kwargs=minimax_kwargs,
                boundary_tol=boundary_tol,
                empty_tol=empty_tol,
            )
            pred = predicate_cache.get(cache_key)

        if pred is None:
            pred = evaluate_predicates(
                simplex,
                matrices,
                centers,
                minimax_kwargs=minimax_kwargs,
                boundary_tol=boundary_tol,
                empty_tol=empty_tol,
            )
            if predicate_cache is not None and cache_key is not None:
                predicate_cache[cache_key] = pred

        if not pred.accepted:
            continue

        # Numerical monotonicity: enforce alpha(face) <= alpha(simplex).
        alpha = max(float(pred.minimax.alpha), *(alpha_by_simplex[face] for face in faces))
        alpha_by_simplex[simplex] = alpha
        entries.append(FiltrationEntry(simplex=simplex, alpha=alpha, predicates=pred))

    entries.sort(key=lambda e: (e.alpha, len(e.simplex), e.simplex))
    return entries
