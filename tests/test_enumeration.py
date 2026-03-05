import pytest

from ellphi_alpha.enumeration import (
    enumerate_candidate_simplices,
    iter_candidate_simplices,
)


def test_enumerate_candidate_simplices_count_and_order():
    simplices = enumerate_candidate_simplices(4, max_dim=2)
    assert len(simplices) == 14  # C(4,1)+C(4,2)+C(4,3)
    assert simplices[:4] == [(0,), (1,), (2,), (3,)]
    assert simplices[4:10] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert simplices[10:] == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def test_enumerate_candidate_simplices_clamps_max_dim():
    simplices = enumerate_candidate_simplices(3, max_dim=10)
    assert simplices[-1] == (0, 1, 2)
    assert len(simplices) == 7


def test_iter_candidate_simplices_min_dim():
    simplices = list(iter_candidate_simplices(5, min_dim=2, max_dim=2))
    assert simplices[0] == (0, 1, 2)
    assert simplices[-1] == (2, 3, 4)
    assert len(simplices) == 10


def test_enumeration_argument_validation():
    with pytest.raises(ValueError):
        list(iter_candidate_simplices(-1, max_dim=2))
    with pytest.raises(ValueError):
        list(iter_candidate_simplices(4, min_dim=-1, max_dim=2))
