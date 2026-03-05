"""Bridge helpers for exporting filtrations to GUDHI."""

from __future__ import annotations

import importlib
from typing import Iterable, Sequence

from .filtration import FiltrationEntry

__all__ = ["to_gudhi_simplex_tree"]


def to_gudhi_simplex_tree(
    filtration: Iterable[FiltrationEntry | tuple[Sequence[int], float]],
):
    """Convert filtration entries into a ``gudhi.SimplexTree``.

    Raises:
        RuntimeError: If ``gudhi`` is not installed.
    """
    try:
        gudhi = importlib.import_module("gudhi")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gudhi is not installed. Install it via `pip install gudhi` "
            "or `poetry install --with demo`."
        ) from exc

    simplex_tree = gudhi.SimplexTree()
    for entry in filtration:
        if isinstance(entry, FiltrationEntry):
            simplex = entry.simplex
            alpha = entry.alpha
        else:
            simplex, alpha = entry
        simplex_tree.insert(simplex=list(simplex), filtration=float(alpha))

    if hasattr(simplex_tree, "make_filtration_non_decreasing"):
        simplex_tree.make_filtration_non_decreasing()
    return simplex_tree
