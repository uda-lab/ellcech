"""Simplex enumeration utilities for anisotropic alpha-complex candidates."""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

__all__ = [
    "iter_candidate_simplices",
    "enumerate_candidate_simplices",
]


def iter_candidate_simplices(
    n_vertices: int,
    *,
    max_dim: int,
    min_dim: int = 0,
) -> Iterator[tuple[int, ...]]:
    """Yield candidate simplices in dimension-then-lexicographic order.

    Args:
        n_vertices: Number of vertices in the dataset (indices are ``0..n-1``).
        max_dim: Maximum simplex dimension to include.
        min_dim: Minimum simplex dimension to include.

    Yields:
        Tuples of vertex indices representing simplices.
    """
    if n_vertices < 0:
        raise ValueError("n_vertices must be non-negative")
    if min_dim < 0:
        raise ValueError("min_dim must be non-negative")
    if max_dim < min_dim:
        return

    max_dim_eff = min(max_dim, n_vertices - 1)
    for dim in range(min_dim, max_dim_eff + 1):
        size = dim + 1
        for simplex in combinations(range(n_vertices), size):
            yield simplex


def enumerate_candidate_simplices(
    n_vertices: int,
    *,
    max_dim: int,
    min_dim: int = 0,
) -> list[tuple[int, ...]]:
    """Return candidate simplices up to ``max_dim`` as a list."""
    return list(
        iter_candidate_simplices(
            n_vertices=n_vertices,
            max_dim=max_dim,
            min_dim=min_dim,
        )
    )
