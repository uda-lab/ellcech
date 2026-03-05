import pytest
import numpy as np

import ellphi_alpha.phase4_acceptance as acceptance
from ellphi_alpha.core.filtration_normalization import normalize_filtration


class _UnavailableBackend:
    name = "fake"

    def is_available(self):
        return False


class _FakeBackend:
    name = "fake"

    def __init__(self, *, h0_distance=0.0, h1_intervals=None):
        self.h0_distance = float(h0_distance)
        if h1_intervals is None:
            h1_intervals = np.array([[0.0, 0.5], [0.1, 0.45]], dtype=float)
        self.h1_intervals = np.asarray(h1_intervals, dtype=float)
        self._edge_alpha = 0.0

    def is_available(self):
        return True

    def simplex_tree_from_filtration(self, filtration):
        normalized = normalize_filtration(filtration)
        edge_entries = [e for e in normalized if len(e.simplex) == 2]
        if edge_entries:
            self._edge_alpha = float(edge_entries[0].alpha)
        return {"kind": "ours", "entries": normalized}

    def alpha_complex_simplex_tree(self, points):
        return {"kind": "reference", "points": np.asarray(points, dtype=float)}

    def persistence_intervals(self, _simplex_tree, *, dimension, homology_coeff_field=2):
        assert homology_coeff_field == 2
        if dimension == 0:
            return np.array([[0.0, 0.6], [0.0, np.inf]], dtype=float)
        if dimension == 1:
            return self.h1_intervals
        raise ValueError(f"unsupported test dimension: {dimension}")

    def edge_map(self, _simplex_tree):
        return {(0, 1): self._edge_alpha}

    def bottleneck_distance(self, _left, _right):
        return self.h0_distance


def test_baseline_barcode_agreement_skips_without_backend():
    result = acceptance.run_baseline_barcode_agreement(
        n_points=24,
        dimension=2,
        backend=_UnavailableBackend(),
    )
    assert result.status == "skipped"
    assert result.passed is None
    assert "fake" in result.message
    assert result.reference_backend == "fake"


def test_baseline_barcode_agreement_with_fake_backend():
    result = acceptance.run_baseline_barcode_agreement(
        n_points=2,
        dimension=2,
        random_seed=20260306,
        backend=_FakeBackend(h0_distance=0.0),
    )
    assert result.status == "ok"
    assert result.passed is True
    assert result.reference_backend == "fake"
    assert result.h0_distance is not None
    assert result.h0_distance <= 1e-6
    # Backward-compatible alias fields stay populated.
    assert result.h0_bottleneck == result.h0_distance
    assert result.h0_count_gudhi == result.h0_count_reference


def test_baseline_barcode_agreement_against_gudhi():
    if not acceptance.has_gudhi():
        pytest.skip("gudhi is not installed")

    result = acceptance.run_baseline_barcode_agreement(
        n_points=100,
        dimension=2,
        random_seed=20260306,
    )
    assert result.status == "ok"
    assert result.passed is True
    assert result.h0_bottleneck is not None
    assert result.h0_bottleneck <= 1e-6


def test_six_rings_h1_check_skips_without_backend():
    result = acceptance.run_six_rings_h1_check(
        points_per_ring=10,
        backend=_UnavailableBackend(),
    )
    assert result.status == "skipped"
    assert result.passed is None
    assert "fake" in result.message
    assert result.reference_backend == "fake"


def test_six_rings_h1_check_with_fake_backend():
    fake_backend = _FakeBackend(
        h1_intervals=np.array(
            [
                [0.0, 0.8],
                [0.0, 0.7],
                [0.1, 0.4],
                [0.2, np.inf],
            ],
            dtype=float,
        )
    )
    result = acceptance.run_six_rings_h1_check(
        points_per_ring=10,
        lifetime_threshold=0.25,
        min_long_lived_h1=3,
        backend=fake_backend,
    )
    assert result.status == "ok"
    assert result.passed is True
    assert result.reference_backend == "fake"
    assert result.h1_count_reference == result.h1_count
    assert result.long_lived_h1_count_reference == result.long_lived_h1_count
    assert result.top_h1_lifetimes_reference == result.top_h1_lifetimes


def test_six_rings_h1_check_against_threshold():
    if not acceptance.has_gudhi():
        pytest.skip("gudhi is not installed")

    result = acceptance.run_six_rings_h1_check(
        points_per_ring=20,
        random_seed=20260307,
    )
    assert result.status == "ok"
    assert result.passed is True
    assert result.long_lived_h1_count is not None
    assert result.long_lived_h1_count >= 6


def test_conditioning_stress_check_tracks_alpha_error():
    result = acceptance.run_conditioning_stress_check(
        n_cases=8,
        dimension=2,
        min_condition_number=1e6,
        max_condition_number=1e7,
        random_seed=20260308,
        rel_error_threshold=1e-4,
        abs_error_threshold=1e-3,
    )
    assert result.status == "ok"
    assert result.passed is True
    assert result.failed_cases == 0
    assert result.over_threshold_cases == 0
    assert result.cond_min_observed > 1e6
