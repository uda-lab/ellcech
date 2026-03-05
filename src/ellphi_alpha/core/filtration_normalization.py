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

FiltrationLikeEntry: TypeAlias = FiltrationEntry | tuple[Sequence[int], float]


@dataclass(frozen=True)
class NormalizedFiltrationEntry:
    """Canonical filtration entry representation for backend adapters."""

    simplex: tuple[int, ...]
    alpha: float


def normalize_simplex(simplex: Sequence[int]) -> tuple[int, ...]:
    """Normalize simplex vertices to a sorted tuple of unique integers."""
    normalized = tuple(int(v) for v in simplex)
    if not normalized:
        raise ValueError("simplex must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"simplex has duplicate vertices: {normalized}")
    if normalized != tuple(sorted(normalized)):
        normalized = tuple(sorted(normalized))
    return normalized


def normalize_filtration_entry(entry: FiltrationLikeEntry) -> NormalizedFiltrationEntry:
    """Normalize one filtration entry from public input shapes."""
    if isinstance(entry, FiltrationEntry):
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
