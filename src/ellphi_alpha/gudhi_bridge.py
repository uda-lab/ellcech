"""Backward-compatible bridge helpers for exporting filtrations to GUDHI."""

from __future__ import annotations

from collections.abc import Iterable

from .backends.gudhi import GudhiBackend
from .core.filtration_normalization import FiltrationLikeEntry

__all__ = ["to_gudhi_simplex_tree"]


def to_gudhi_simplex_tree(
    filtration: Iterable[FiltrationLikeEntry],
):
    """Convert filtration entries into a ``gudhi.SimplexTree``.

    This function remains public for compatibility and delegates to
    :class:`ellphi_alpha.backends.gudhi.GudhiBackend`.
    """
    return GudhiBackend().simplex_tree_from_filtration(filtration)
