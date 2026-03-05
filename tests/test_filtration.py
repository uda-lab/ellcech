import numpy as np

from ellphi_alpha.filtration import build_incremental_filtration


def _as_set(entries):
    return {entry.simplex for entry in entries}


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


def test_build_incremental_filtration_predicate_rejects_triangle():
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
    simplices = _as_set(filt)
    assert (0, 1, 2) not in simplices


def test_build_incremental_filtration_max_dim_one():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((6, 3))
    matrices = np.repeat(np.eye(3)[np.newaxis], 6, axis=0)
    filt = build_incremental_filtration(matrices, pts, max_dim=1)
    assert all(len(entry.simplex) <= 2 for entry in filt)
