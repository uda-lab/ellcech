import numpy as np
import pytest

from ellphi_alpha import solve_minimax
from ellphi_alpha.act_map import (
    SimplexInfo,
    act_support,
    booleanity_report,
    certify_support,
    coface_set,
    critical_simplices,
    critical_value_check,
    fiber_partition,
)


def _make_spd(rng, d, cond_max=10.0):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = rng.uniform(1.0, cond_max, size=d)
    return (q * eigs) @ q.T


def _random_instance(n, d, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.0, 1.0, size=(n, d))
    matrices = np.stack([_make_spd(rng, d) for _ in range(n)])
    return matrices, centers


# The d=2, n=4 degenerate counterexample of
# ../paper-ellalpha/docs/early_gate_counterexample_note.md: two minimal
# positive KKT supports {0,1} and {2,3} at the same center (origin) and
# level (alpha = 1).
T1_CENTERS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
T1_MATRICES = np.stack(
    [
        np.diag([1.0, 2.0]),
        np.diag([1.0, 3.0]),
        np.diag([4.0, 1.0]),
        np.diag([5.0, 1.0]),
    ]
)


def test_certify_support_generic_edge():
    matrices, centers = _random_instance(2, 2, seed=11)
    result = solve_minimax(matrices, centers)
    cert = certify_support(matrices, centers, result)
    assert cert.status == "certified"
    assert 1 <= len(cert.support) <= 3
    assert cert.spread <= 1e-6
    assert cert.stationarity <= 1e-6


def test_act_support_maps_local_to_global():
    matrices, centers = _random_instance(5, 2, seed=12)
    simplex = (0, 2, 4)
    idx = np.asarray(simplex)
    result = solve_minimax(matrices[idx], centers[idx])
    cert = certify_support(matrices[idx], centers[idx], result)
    support = act_support(simplex, cert)
    assert set(support) <= set(simplex)
    assert support == tuple(sorted(support))


def test_act_idempotency_on_random_simplices():
    # Rigidity (act_map_proofs.md, Lemma 2.3): the act support of the act
    # support is itself.
    matrices, centers = _random_instance(7, 2, seed=13)
    for simplex in [(0, 1, 2), (1, 3, 5), (2, 4, 6), (0, 3, 4, 6)]:
        idx = np.asarray(simplex)
        result = solve_minimax(matrices[idx], centers[idx])
        cert = certify_support(matrices[idx], centers[idx], result)
        if cert.status not in ("certified", "escalated"):
            continue
        tau = act_support(simplex, cert)
        tau_idx = np.asarray(tau)
        result_tau = solve_minimax(matrices[tau_idx], centers[tau_idx])
        cert_tau = certify_support(matrices[tau_idx], centers[tau_idx], result_tau)
        assert cert_tau.status in ("certified", "escalated")
        assert act_support(tau, cert_tau) == tau
        assert result_tau.alpha == pytest.approx(result.alpha, rel=1e-6)


def test_t1_pair_values():
    for pair in [(0, 1), (2, 3)]:
        idx = np.asarray(pair)
        result = solve_minimax(T1_MATRICES[idx], T1_CENTERS[idx])
        assert result.alpha == pytest.approx(1.0, rel=1e-8)
        np.testing.assert_allclose(result.circumcenter, [0.0, 0.0], atol=1e-8)


def test_t1_full_simplex_tie_detected():
    result = solve_minimax(T1_MATRICES, T1_CENTERS)
    cert = certify_support(T1_MATRICES, T1_CENTERS, result)
    assert result.alpha == pytest.approx(1.0, rel=1e-8)
    # Two minimal supports at the same level: the certificate must flag the
    # degeneracy instead of returning a spuriously unique support.
    assert cert.status == "degenerate"
    assert len(cert.support) > 3


def test_seed_20260708_certificate_escalation():
    # Regression: near the degenerate configuration, fw+bisect returns a
    # bogus 4-element support with an equal-value spread at the 4th decimal;
    # the certificate must reject it and escalate to scipy-slsqp.
    rng = np.random.default_rng(20260708)
    centers = T1_CENTERS + 1e-3 * rng.standard_normal(T1_CENTERS.shape)
    raw = solve_minimax(T1_MATRICES, centers)

    unescalated = certify_support(T1_MATRICES, centers, raw, escalate=False)
    assert unescalated.status == "uncertain"

    cert = certify_support(T1_MATRICES, centers, raw)
    assert cert.status == "escalated"
    assert cert.support == (0, 1, 2)
    reference = solve_minimax(T1_MATRICES, centers, method="scipy-slsqp")
    assert cert.result.alpha == pytest.approx(reference.alpha, rel=1e-9)


def test_coface_contains_simplex():
    matrices, centers = _random_instance(8, 2, seed=14)
    simplex = (1, 4, 6)
    idx = np.asarray(simplex)
    result = solve_minimax(matrices[idx], centers[idx])
    coface = coface_set(matrices, centers, result.circumcenter, result.alpha)
    assert set(simplex) <= coface


def _info(simplex, alpha, support, coface, status="certified", source="solved"):
    return SimplexInfo(
        simplex=simplex,
        alpha=alpha,
        support=support,
        coface=frozenset(coface),
        status=status,
        source=source,
        circumcenter=None,
    )


def test_fiber_partition_groups_by_support():
    infos = {
        (0,): _info((0,), 0.0, (0,), {0}),
        (1,): _info((1,), 0.0, (1,), {1}),
        (0, 1): _info((0, 1), 1.0, (0, 1), {0, 1, 2}),
        (0, 1, 2): _info((0, 1, 2), 1.0, (0, 1), {0, 1, 2}, source="reused"),
    }
    fibers = fiber_partition(infos)
    assert fibers[(0, 1)] == [(0, 1), (0, 1, 2)]
    assert fibers[(0,)] == [(0,)]


def test_booleanity_report_flags_missing_member():
    infos = {
        (0,): _info((0,), 0.0, (0,), {0}),
        (1,): _info((1,), 0.0, (1,), {1}),
        (2,): _info((2,), 0.0, (2,), {2}),
        # Fiber [ (0,1), {0,1,2} ] should contain (0,1) and (0,1,2); the
        # triangle is absent, so the report must flag it as missing.
        (0, 1): _info((0, 1), 1.0, (0, 1), {0, 1, 2}),
    }
    report = booleanity_report(infos, max_dim=2)
    assert not report.is_boolean
    assert any(v.support == (0, 1) and (0, 1, 2) in v.missing for v in report.violations)


def test_booleanity_report_accepts_full_interval():
    infos = {
        (0,): _info((0,), 0.0, (0,), {0}),
        (1,): _info((1,), 0.0, (1,), {1}),
        (2,): _info((2,), 0.0, (2,), {2}),
        (0, 1): _info((0, 1), 1.0, (0, 1), {0, 1, 2}),
        (0, 1, 2): _info((0, 1, 2), 1.0, (0, 1), {0, 1, 2}, source="reused"),
    }
    report = booleanity_report(infos, max_dim=2)
    assert report.is_boolean
    assert report.fiber_sizes == {1: 3, 2: 1}


def test_critical_simplices_selects_fixed_points():
    infos = {
        (0,): _info((0,), 0.0, (0,), {0}),
        (0, 1): _info((0, 1), 1.0, (0, 1), {0, 1}),
        (0, 2): _info((0, 2), 1.5, (0, 2), {0, 1, 2}),
    }
    assert critical_simplices(infos) == [(0,), (0, 1)]


def test_critical_value_check():
    intervals = {0: np.array([[0.0, 1.0], [0.0, np.inf]]), 1: np.array([[1.0, 2.0]])}
    ok = critical_value_check(intervals, [0.0, 1.0, 2.0], tol=1e-9)
    assert ok.passed
    assert ok.checked_endpoints == 5
    bad = critical_value_check(intervals, [0.0, 1.0], tol=1e-9)
    assert not bad.passed
    assert bad.violations == ((1, 2.0),)
