import pytest

from ellphi_alpha.core.filtration_normalization import (
    NormalizedFiltrationEntry,
    normalize_filtration,
    normalize_simplex,
)
from ellphi_alpha.filtration import FiltrationEntry


def test_normalize_simplex_sorts_vertices():
    assert normalize_simplex((3, 1, 2)) == (1, 2, 3)


def test_normalize_simplex_rejects_duplicate_vertices():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_simplex((1, 1, 2))


def test_normalize_simplex_rejects_float_vertices():
    """Float vertices must raise rather than silently truncating (e.g. 3.7 → 3)."""
    with pytest.raises(ValueError, match="integer type"):
        normalize_simplex((1, 2.0, 3))


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


def test_normalize_filtration_accepts_normalized_entries_as_input():
    """Output of normalize_filtration() must be accepted as input again."""
    source = [
        FiltrationEntry(simplex=(0,), alpha=0.0, predicates=None),
        FiltrationEntry(simplex=(1,), alpha=0.0, predicates=None),
        ((1, 0), 1.5),
    ]
    first_pass = normalize_filtration(source)
    # Must not raise; NormalizedFiltrationEntry must be in FiltrationLikeEntry union.
    second_pass = normalize_filtration(first_pass)
    assert [(e.simplex, e.alpha) for e in second_pass] == [
        (e.simplex, e.alpha) for e in first_pass
    ]
    assert all(isinstance(e, NormalizedFiltrationEntry) for e in second_pass)
