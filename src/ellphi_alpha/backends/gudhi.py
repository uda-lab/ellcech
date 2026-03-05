"""GUDHI-backed persistence adapter."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.backend_contracts import PersistenceBackend
from ..core.filtration_normalization import FiltrationLikeEntry, iter_normalized_filtration

__all__ = ["GudhiBackend"]

_MISSING_GUDHI_MESSAGE = (
    "gudhi is not installed. Install it via `pip install gudhi` "
    "or `poetry install --with demo`."
)


def _import_gudhi():
    try:
        return importlib.import_module("gudhi")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_MISSING_GUDHI_MESSAGE) from exc


class GudhiBackend(PersistenceBackend):
    """Adapter that exposes persistence operations via GUDHI."""

    name = "gudhi"

    def is_available(self) -> bool:
        return importlib.util.find_spec("gudhi") is not None

    def simplex_tree_from_filtration(
        self,
        filtration: Iterable[FiltrationLikeEntry],
    ) -> Any:
        gudhi = _import_gudhi()
        simplex_tree = gudhi.SimplexTree()

        for entry in iter_normalized_filtration(filtration):
            simplex_tree.insert(simplex=list(entry.simplex), filtration=entry.alpha)

        if hasattr(simplex_tree, "make_filtration_non_decreasing"):
            simplex_tree.make_filtration_non_decreasing()
        return simplex_tree

    def alpha_complex_simplex_tree(self, points: np.ndarray) -> Any:
        gudhi = _import_gudhi()
        points = np.asarray(points, dtype=float)
        return gudhi.AlphaComplex(points=points).create_simplex_tree()

    def persistence_intervals(
        self,
        simplex_tree: Any,
        *,
        dimension: int,
        homology_coeff_field: int = 2,
    ) -> np.ndarray:
        simplex_tree.persistence(homology_coeff_field=homology_coeff_field)
        intervals = simplex_tree.persistence_intervals_in_dimension(int(dimension))
        return np.asarray(intervals, dtype=float)

    def edge_map(self, simplex_tree: Any) -> dict[tuple[int, int], float]:
        edge_map: dict[tuple[int, int], float] = {}
        for simplex, alpha in simplex_tree.get_filtration():
            if len(simplex) != 2:
                continue
            i, j = sorted(int(v) for v in simplex)
            edge_map[(i, j)] = float(alpha)
        return edge_map

    def bottleneck_distance(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        gudhi = _import_gudhi()
        return float(gudhi.bottleneck_distance(left.tolist(), right.tolist()))
