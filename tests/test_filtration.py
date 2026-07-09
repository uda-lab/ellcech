import numpy as np

import ellphi_alpha.filtration as filtration_mod
from ellphi_alpha.filtration import build_incremental_filtration
from ellphi_alpha.minimax import MinimaxResult
from ellphi_alpha.predicates import PredicateResult


def _as_set(entries):
    return {entry.simplex for entry in entries}


def _fake_predicate(simplex, *, accepted: bool, alpha: float, dim: int) -> PredicateResult:
    k = len(simplex)
    minimax = MinimaxResult(
        alpha=float(alpha),
        circumcenter=np.zeros(dim),
        weights=np.full(k, 1.0 / k),
        active_set=list(range(k)),
        converged=True,
        n_iter=1,
    )
    return PredicateResult(
        simplex=tuple(simplex),
        minimax=minimax,
        p1_converged=True,
        p2_boundary=True,
        p3_empty=True,
        accepted=accepted,
    )


def test_build_incremental_filtration_triangle_downward_closed():
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)

    filt = build_incremental_filtration(matrices, pts, max_dim=2)
    simplices = _as_set(filt)
    assert simplices == {
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    }

    alpha_map = {entry.simplex: entry.alpha for entry in filt}
    alphas = [entry.alpha for entry in filt]
    assert alphas == sorted(alphas)
    for simplex, alpha in alpha_map.items():
        if len(simplex) == 1:
            continue
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1 :]
            assert face in alpha_map
            assert alpha_map[face] <= alpha + 1e-12


def test_build_incremental_filtration_critical_only_rejects_triangle():
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
            [1.0, np.sqrt(3.0) / 3.0],  # inside circum-ball of (0,1,2)
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 4, axis=0)

    filt = build_incremental_filtration(matrices, pts, max_dim=2, mode="critical-only")
    simplices = _as_set(filt)
    assert (0, 1, 2) not in simplices


def test_build_incremental_filtration_cech_mode_keeps_non_empty_simplices():
    # Cech semantics (default): P3 emptiness does not gate membership, so the
    # triangle whose circum-ball contains point 3 still enters with its alpha.
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
            [1.0, np.sqrt(3.0) / 3.0],  # inside circum-ball of (0,1,2)
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 4, axis=0)

    filt = build_incremental_filtration(matrices, pts, max_dim=2)
    alpha_map = {entry.simplex: entry.alpha for entry in filt}
    assert (0, 1, 2) in alpha_map
    # Equilateral side 2: circumradius^2 = 4/3.
    assert abs(alpha_map[(0, 1, 2)] - 4.0 / 3.0) < 1e-8


def test_build_incremental_filtration_max_dim_one():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((6, 3))
    matrices = np.repeat(np.eye(3)[np.newaxis], 6, axis=0)
    filt = build_incremental_filtration(matrices, pts, max_dim=1)
    assert all(len(entry.simplex) <= 2 for entry in filt)


def test_build_incremental_filtration_faces_first_prune_skips_eval(monkeypatch):
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 4, axis=0)
    calls: list[tuple[int, ...]] = []

    def fake_evaluate_predicates(simplex, *_args, **_kwargs):
        simplex_t = tuple(simplex)
        calls.append(simplex_t)
        if simplex_t == (0, 1):
            return _fake_predicate(simplex_t, accepted=False, alpha=2.0, dim=2)
        return _fake_predicate(simplex_t, accepted=True, alpha=float(len(simplex_t)), dim=2)

    monkeypatch.setattr(filtration_mod, "evaluate_predicates", fake_evaluate_predicates)

    filt = filtration_mod.build_incremental_filtration(
        matrices, pts, max_dim=2, mode="critical-only"
    )
    simplices = _as_set(filt)

    # (0,1) is rejected, so triangles containing this edge cannot be inserted
    # and should be pruned before predicate evaluation.
    assert (0, 1, 2) not in calls
    assert (0, 1, 3) not in calls
    assert (0, 2, 3) in calls
    assert (1, 2, 3) in calls

    assert (0, 1) not in simplices


def test_build_incremental_filtration_predicate_cache_reuses_results(monkeypatch):
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)
    calls: list[tuple[int, ...]] = []

    def fake_evaluate_predicates(simplex, *_args, **_kwargs):
        simplex_t = tuple(simplex)
        calls.append(simplex_t)
        return _fake_predicate(simplex_t, accepted=True, alpha=float(len(simplex_t)), dim=2)

    monkeypatch.setattr(filtration_mod, "evaluate_predicates", fake_evaluate_predicates)

    cache = {}
    first = filtration_mod.build_incremental_filtration(
        matrices,
        pts,
        max_dim=2,
        predicate_cache=cache,
    )
    n_calls_first = len(calls)
    assert n_calls_first == 4  # 3 edges + 1 triangle
    assert len(cache) == 4

    second = filtration_mod.build_incremental_filtration(
        matrices,
        pts,
        max_dim=2,
        predicate_cache=cache,
    )
    n_calls_second = len(calls) - n_calls_first
    assert n_calls_second == 0
    assert [entry.simplex for entry in second] == [entry.simplex for entry in first]
    assert [entry.alpha for entry in second] == [entry.alpha for entry in first]


def test_build_incremental_filtration_predicate_cache_invalidated_on_dataset_change(monkeypatch):
    """Cache must not reuse results when the dataset (matrices/centers) changes."""
    pts_a = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])
    pts_b = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, np.sqrt(3.0)]])  # different
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)
    calls: list[tuple] = []

    def fake_evaluate_predicates(simplex, *_args, **_kwargs):
        simplex_t = tuple(simplex)
        calls.append(simplex_t)
        return _fake_predicate(simplex_t, accepted=True, alpha=float(len(simplex_t)), dim=2)

    monkeypatch.setattr(filtration_mod, "evaluate_predicates", fake_evaluate_predicates)

    cache = {}
    filtration_mod.build_incremental_filtration(matrices, pts_a, max_dim=2, predicate_cache=cache)
    n_after_first = len(calls)

    # Different dataset: cache must miss for all simplices.
    filtration_mod.build_incremental_filtration(matrices, pts_b, max_dim=2, predicate_cache=cache)
    n_after_second = len(calls)
    assert n_after_second - n_after_first == n_after_first  # same number of fresh evals


def test_build_incremental_filtration_predicate_cache_respects_settings(monkeypatch):
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)
    calls: list[tuple[int, ...]] = []

    def fake_evaluate_predicates(simplex, *_args, **_kwargs):
        simplex_t = tuple(simplex)
        calls.append(simplex_t)
        return _fake_predicate(simplex_t, accepted=True, alpha=float(len(simplex_t)), dim=2)

    monkeypatch.setattr(filtration_mod, "evaluate_predicates", fake_evaluate_predicates)

    cache = {}
    filtration_mod.build_incremental_filtration(
        matrices,
        pts,
        max_dim=2,
        boundary_tol=1e-7,
        predicate_cache=cache,
    )
    n_calls_after_first = len(calls)
    filtration_mod.build_incremental_filtration(
        matrices,
        pts,
        max_dim=2,
        boundary_tol=1e-6,  # different setting: should miss cache
        predicate_cache=cache,
    )
    n_calls_after_second = len(calls)
    assert n_calls_after_second - n_calls_after_first == 4
