"""Core backend protocol contracts used by adapter implementations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .filtration_normalization import FiltrationLikeEntry

__all__ = ["PersistenceBackend"]


@runtime_checkable
class PersistenceBackend(Protocol):
    """Backend contract for persistence-based acceptance checks."""

    name: str

    def is_available(self) -> bool:
        """Return True when backend dependencies are importable."""

    def simplex_tree_from_filtration(
        self,
        filtration: Iterable[FiltrationLikeEntry],
    ) -> Any:
        """Build backend simplex-tree object from filtration entries."""

    def alpha_complex_simplex_tree(self, points: np.ndarray) -> Any:
        """Build backend alpha-complex simplex tree from points."""

    def persistence_intervals(
        self,
        simplex_tree: Any,
        *,
        dimension: int,
        homology_coeff_field: int = 2,
    ) -> np.ndarray:
        """Compute persistence intervals in a fixed dimension."""

    def edge_map(self, simplex_tree: Any) -> dict[tuple[int, int], float]:
        """Extract edge filtration values from a backend simplex tree."""

    def bottleneck_distance(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """Compute bottleneck distance between two persistence diagrams."""
