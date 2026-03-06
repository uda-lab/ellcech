"""Utilities for backend-neutral filtration entry normalization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from ..filtration import FiltrationEntry

__all__ = [
    "FiltrationLikeEntry",
    "NormalizedFiltrationEntry",
    "iter_normalized_filtration",
    "normalize_filtration",
    "normalize_filtration_entry",
    "normalize_simplex",
]

@dataclass(frozen=True)
class NormalizedFiltrationEntry:
    """Canonical filtration entry representation for backend adapters."""

    simplex: tuple[int, ...]
    alpha: float


# NormalizedFiltrationEntry is included so that the output of normalize_filtration()
# can be fed back into a second normalization call without type errors.
FiltrationLikeEntry: TypeAlias = FiltrationEntry | NormalizedFiltrationEntry | tuple[Sequence[int], float]


def normalize_simplex(simplex: Sequence[int]) -> tuple[int, ...]:
    """Normalize simplex vertices to a sorted tuple of unique integers.

    Raises ValueError for non-integer vertex labels to prevent silent
    truncation (e.g. float 3.7 → int 3) from corrupting simplex identities.
    """
    try:
        import numpy as np
        _int_types = (int, np.integer)
    except ImportError:
        _int_types = (int,)
    normalized = []
    for v in simplex:
        if not isinstance(v, _int_types):
            raise ValueError(
                f"simplex vertex {v!r} has type {type(v).__name__!r}; "
                "expected an integer type"
            )
        normalized.append(int(v))
    normalized_t = tuple(normalized)
    if not normalized_t:
        raise ValueError("simplex must be non-empty")
    if len(set(normalized_t)) != len(normalized_t):
        raise ValueError(f"simplex has duplicate vertices: {normalized_t}")
    if normalized_t != tuple(sorted(normalized_t)):
        normalized_t = tuple(sorted(normalized_t))
    return normalized_t


def normalize_filtration_entry(entry: FiltrationLikeEntry) -> NormalizedFiltrationEntry:
    """Normalize one filtration entry from public input shapes."""
    if isinstance(entry, (FiltrationEntry, NormalizedFiltrationEntry)):
        simplex = normalize_simplex(entry.simplex)
        alpha = float(entry.alpha)
    else:
        simplex, alpha = entry
        simplex = normalize_simplex(simplex)
        alpha = float(alpha)
    return NormalizedFiltrationEntry(simplex=simplex, alpha=alpha)


def iter_normalized_filtration(
    filtration: Iterable[FiltrationLikeEntry],
) -> Iterator[NormalizedFiltrationEntry]:
    """Yield normalized filtration entries."""
    for entry in filtration:
        yield normalize_filtration_entry(entry)


def normalize_filtration(
    filtration: Iterable[FiltrationLikeEntry],
    *,
    sort_entries: bool = False,
) -> list[NormalizedFiltrationEntry]:
    """Return a normalized filtration entry list."""
    normalized = list(iter_normalized_filtration(filtration))
    if sort_entries:
        normalized.sort(key=lambda e: (e.alpha, len(e.simplex), e.simplex))
    return normalized
