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
