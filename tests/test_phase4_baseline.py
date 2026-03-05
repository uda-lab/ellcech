import pytest

import ellphi_alpha.phase4_acceptance as acceptance


def test_baseline_barcode_agreement_skips_without_gudhi(monkeypatch):
    monkeypatch.setattr(acceptance, "has_gudhi", lambda: False)
    result = acceptance.run_baseline_barcode_agreement(n_points=24, dimension=2)
    assert result.status == "skipped"
    assert result.passed is None
    assert "gudhi" in result.message


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


def test_six_rings_h1_check_skips_without_gudhi(monkeypatch):
    monkeypatch.setattr(acceptance, "has_gudhi", lambda: False)
    result = acceptance.run_six_rings_h1_check(points_per_ring=10)
    assert result.status == "skipped"
    assert result.passed is None
    assert "gudhi" in result.message


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
