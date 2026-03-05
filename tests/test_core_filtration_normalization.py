import pytest

from ellphi_alpha.core.filtration_normalization import (
    normalize_filtration,
    normalize_simplex,
)
from ellphi_alpha.filtration import FiltrationEntry


def test_normalize_simplex_sorts_vertices():
    assert normalize_simplex((3, 1, 2)) == (1, 2, 3)


def test_normalize_simplex_rejects_duplicate_vertices():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_simplex((1, 1, 2))


def test_normalize_filtration_handles_mixed_entry_types():
    entries = normalize_filtration(
        [
            FiltrationEntry(simplex=(2,), alpha=0.0, predicates=None),
            ((2, 1), 3.5),
        ]
    )
    assert [(entry.simplex, entry.alpha) for entry in entries] == [
        ((2,), 0.0),
        ((1, 2), 3.5),
    ]
