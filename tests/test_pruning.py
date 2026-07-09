from itertools import combinations

import numpy as np
import pytest

from ellphi_alpha import solve_minimax
from ellphi_alpha.act_map import booleanity_report
from ellphi_alpha.pruning import (
    brute_force_filtration,
    certified_filtration,
    clique_candidates,
    neighbour_graph,
    rayleigh_lower_bound,
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


def _pairwise_alpha_percentile(matrices, centers, q):
    n = len(centers)
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(solve_minimax(matrices[[i, j]], centers[[i, j]]).alpha)
    return float(np.percentile(values, q))


def test_rayleigh_lower_bound():
    matrices = np.stack([np.diag([2.0, 5.0]), np.diag([3.0, 0.5])])
    assert rayleigh_lower_bound(matrices) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        rayleigh_lower_bound(np.stack([np.diag([1.0, -1.0])]))


def test_neighbour_graph():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    adj = neighbour_graph(centers, 1.5)
    assert adj == [{1}, {0}, set()]
    adj = neighbour_graph(centers, 2.5)
    assert adj == [{1}, {0, 2}, {1}]


def test_clique_candidates_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 8
    adjacency = [set() for _ in range(n)]
    for i, j in combinations(range(n), 2):
        if rng.uniform() < 0.5:
            adjacency[i].add(j)
            adjacency[j].add(i)

    def is_clique(simplex):
        return all(j in adjacency[i] for i, j in combinations(simplex, 2))

    for max_dim in (1, 2, 3):
        expected = [
            simplex
            for size in range(1, max_dim + 2)
            for simplex in combinations(range(n), size)
            if is_clique(simplex)
        ]
        got = list(clique_candidates(adjacency, max_dim))
        assert sorted(got, key=lambda s: (len(s), s)) == sorted(
            expected, key=lambda s: (len(s), s)
        )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_certified_filtration_matches_brute_force(seed):
    # Theorem I soundness semantics: the certified pipeline must emit exactly
    # the simplices with alpha <= r_max, with matching values.
    matrices, centers = _random_instance(10, 2, seed)
    r_max = _pairwise_alpha_percentile(matrices, centers, 40.0)
    cert = certified_filtration(matrices, centers, r_max=r_max, max_dim=3)
    brute = brute_force_filtration(matrices, centers, r_max=r_max, max_dim=3)

    cert_map = {e.simplex: e.alpha for e in cert.entries}
    brute_map = {e.simplex: e.alpha for e in brute}
    assert set(cert_map) == set(brute_map)
    for simplex, alpha in brute_map.items():
        assert cert_map[simplex] == pytest.approx(alpha, abs=1e-8)


def test_certified_filtration_prunes_and_reuses():
    matrices, centers = _random_instance(10, 2, seed=1)
    r_max = _pairwise_alpha_percentile(matrices, centers, 40.0)
    result = certified_filtration(matrices, centers, r_max=r_max, max_dim=3)
    stats = result.stats
    assert stats.emitted == len(result.entries)
    assert stats.solves < stats.brute_force_candidates
    assert stats.reuses > 0
    assert stats.uncertain == 0
    assert stats.gl_clique_candidates <= stats.brute_force_candidates
    assert stats.after_pairwise_candidates <= stats.gl_clique_candidates


def test_reused_alpha_matches_direct_solve():
    matrices, centers = _random_instance(10, 2, seed=2)
    r_max = _pairwise_alpha_percentile(matrices, centers, 40.0)
    result = certified_filtration(matrices, centers, r_max=r_max, max_dim=3)
    reused = [s for s, i in result.info.items() if i.source == "reused"]
    assert reused
    for simplex in reused:
        idx = np.asarray(simplex)
        direct = solve_minimax(matrices[idx], centers[idx])
        assert result.info[simplex].alpha == pytest.approx(direct.alpha, abs=1e-7)


def test_output_is_downward_closed_and_monotone():
    matrices, centers = _random_instance(10, 2, seed=3)
    r_max = _pairwise_alpha_percentile(matrices, centers, 40.0)
    result = certified_filtration(matrices, centers, r_max=r_max, max_dim=3)
    alpha_map = {e.simplex: e.alpha for e in result.entries}
    for simplex, alpha in alpha_map.items():
        for k in range(len(simplex)):
            face = simplex[:k] + simplex[k + 1 :]
            if not face:
                continue
            assert face in alpha_map
            assert alpha_map[face] <= alpha + 1e-12


def test_booleanity_holds_on_generic_data():
    matrices, centers = _random_instance(10, 2, seed=1)
    r_max = _pairwise_alpha_percentile(matrices, centers, 40.0)
    result = certified_filtration(matrices, centers, r_max=r_max, max_dim=3)
    assert all(i.status == "certified" for i in result.info.values())
    report = booleanity_report(result.info, max_dim=3)
    assert report.is_boolean
    assert not report.skipped_fibers


def test_r_max_zero_keeps_only_vertices():
    matrices, centers = _random_instance(5, 2, seed=4)
    result = certified_filtration(matrices, centers, r_max=0.0, max_dim=2)
    assert [e.simplex for e in result.entries] == [(v,) for v in range(5)]
