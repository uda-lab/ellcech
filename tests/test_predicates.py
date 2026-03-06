import numpy as np
import pytest

from ellphi_alpha.minimax import solve_minimax
from ellphi_alpha.predicates import evaluate_predicates


def _make_spd(rng, d):
    L = rng.standard_normal((d, d))
    return L @ L.T + np.eye(d) * 0.5


def test_predicates_accept_equilateral_triangle():
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)

    pred = evaluate_predicates((0, 1, 2), matrices, pts)
    assert pred.p1_converged
    assert pred.p2_boundary
    assert pred.p3_empty
    assert pred.accepted


def test_predicates_reject_when_outside_point_is_inside_ball():
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
            [1.0, np.sqrt(3.0) / 3.0],  # circumcenter of first 3 points
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 4, axis=0)

    pred = evaluate_predicates((0, 1, 2), matrices, pts)
    assert pred.p1_converged
    assert pred.p2_boundary
    assert not pred.p3_empty
    assert not pred.accepted


def test_predicates_reject_non_converged_minimax():
    rng = np.random.default_rng(123)
    pts = rng.standard_normal((5, 4))
    matrices = np.stack([_make_spd(rng, 4) for _ in range(5)])

    pred = evaluate_predicates(
        (0, 1, 2, 3, 4),
        matrices,
        pts,
        minimax_kwargs={"max_iter": 1, "tol": 1e-14},
    )
    assert not pred.p1_converged
    assert not pred.accepted


def test_predicates_raise_for_incompatible_supplied_minimax_result():
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 3, axis=0)

    # Minimax computed for edge (2 weights), but supplied for triangle (3 vertices).
    edge_result = solve_minimax(matrices[:2], pts[:2])
    with pytest.raises(ValueError, match="incompatible with simplex"):
        evaluate_predicates((0, 1, 2), matrices, pts, minimax_result=edge_result)


def test_predicates_raise_for_wrong_simplex_matching_shape():
    """A result from a *different* simplex of the same shape must be rejected.

    Shape-only checks (matching weight count and circumcenter dimension) would
    accept this, but the geometry consistency check must detect the mismatch.
    """
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],  # far away — a result for (0,1) is wrong for (2,3)
        ]
    )
    matrices = np.repeat(np.eye(2)[np.newaxis], 4, axis=0)

    # Result computed for edge (0, 1): circumcenter=[0.5,0], alpha=0.25
    result_for_01 = solve_minimax(matrices[:2], pts[:2])

    # Supplying it for edge (2, 3) has matching shape but wrong geometry.
    with pytest.raises(ValueError, match="inconsistent with the supplied simplex"):
        evaluate_predicates((2, 3), matrices, pts, minimax_result=result_for_01)
