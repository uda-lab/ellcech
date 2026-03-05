"""Backend-neutral core contracts and utilities."""

from .backend_contracts import PersistenceBackend
from .filtration_normalization import (
    FiltrationLikeEntry,
    NormalizedFiltrationEntry,
    iter_normalized_filtration,
    normalize_filtration,
    normalize_filtration_entry,
    normalize_simplex,
)

__all__ = [
    "PersistenceBackend",
    "FiltrationLikeEntry",
    "NormalizedFiltrationEntry",
    "iter_normalized_filtration",
    "normalize_filtration",
    "normalize_filtration_entry",
    "normalize_simplex",
]
