"""Tests for the Frank-Wolfe minimax solver.

Key validation:
  For |σ| = 2, alpha(σ) must equal ellphi.tangency(p, q).t ** 2.
  This cross-validates the Frank-Wolfe solver against ellphi's 1D root-finding,
  which is the established reference implementation for the pairwise case.
"""

import numpy as np
import pytest

import ellphi
from ellphi_alpha.minimax import solve_minimax, solve_minimax_from_coefs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spd(rng, d):
    """Return a random d×d SPD matrix."""
    L = rng.standard_normal((d, d))
    return L @ L.T + np.eye(d) * 0.5


def _pairwise_alpha_ellphi(pcoef, qcoef):
    """Ground truth: α({i,j}) = t² from ellphi."""
    return ellphi.tangency(pcoef, qcoef).t ** 2


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------

class TestTrivialCases:
    def test_single_point_alpha_zero(self):
        A = np.eye(2)
        x = np.array([1.0, 2.0])
        res = solve_minimax(A[np.newaxis], x[np.newaxis])
        assert res.alpha == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(res.circumcenter, x)
        assert res.converged
        assert res.n_iter == 0

    def test_single_point_3d(self):
        rng = np.random.default_rng(0)
        A = _make_spd(rng, 3)
        x = rng.standard_normal(3)
        res = solve_minimax(A[np.newaxis], x[np.newaxis])
        assert res.alpha == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Isotropic (A_i = I) special cases
# ---------------------------------------------------------------------------

class TestIsotropic:
    def test_two_points_midpoint(self):
        """With A_i = I, x* is the midpoint of two equidistant points."""
        x0 = np.array([-1.0, 0.0])
        x1 = np.array([+1.0, 0.0])
        A = np.eye(2)[np.newaxis].repeat(2, axis=0)
        centers = np.stack([x0, x1])
        res = solve_minimax(A, centers)
        assert res.alpha == pytest.approx(1.0, rel=1e-6)
        np.testing.assert_allclose(res.circumcenter, [0.0, 0.0], atol=1e-6)
        assert res.converged

    def test_two_points_alpha_is_half_sq_distance(self):
        """For A=I, α({i,j}) = ||x̄_i - x̄_j||² / 4 (midpoint equidistant)."""
        rng = np.random.default_rng(42)
        x0, x1 = rng.standard_normal((2, 3))
        A = np.eye(3)[np.newaxis].repeat(2, axis=0)
        centers = np.stack([x0, x1])
        res = solve_minimax(A, centers)
        expected = np.sum((x0 - x1) ** 2) / 4.0
        assert res.alpha == pytest.approx(expected, rel=1e-6)

    def test_three_points_circumcenter_equidistant(self):
        """With A_i = I, x* is the circumcenter (equidistant from all points)."""
        # Equilateral triangle
        c = 2.0 * np.pi / 3.0
        x0 = np.array([1.0, 0.0])
        x1 = np.array([np.cos(c), np.sin(c)])
        x2 = np.array([np.cos(2 * c), np.sin(2 * c)])
        A = np.eye(2)[np.newaxis].repeat(3, axis=0)
        centers = np.stack([x0, x1, x2])
        res = solve_minimax(A, centers, tol=1e-10)
        # All f_i should be equal
        diff = res.circumcenter[np.newaxis] - centers
        f = np.einsum("ki,kij,kj->k", diff, A, diff)
        assert np.max(f) - np.min(f) == pytest.approx(0.0, abs=1e-6)
        assert res.converged


# ---------------------------------------------------------------------------
# Cross-validation with ellphi (pairwise case |σ| = 2)
# ---------------------------------------------------------------------------

class TestCrossValidationEllphi:
    """For |σ| = 2, alpha must match ellphi.tangency(p, q).t**2."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_pairwise_random_2d(self, seed):
        rng = np.random.default_rng(seed)
        x0, x1 = rng.standard_normal((2, 2))
        cov0 = _make_spd(rng, 2)
        cov1 = _make_spd(rng, 2)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        res = solve_minimax_from_coefs(coefs, tol=1e-10)
        expected = _pairwise_alpha_ellphi(pcoef, qcoef)

        assert res.alpha == pytest.approx(expected, rel=1e-5), (
            f"seed={seed}: alpha={res.alpha:.8f}, expected={expected:.8f}"
        )
        assert res.converged, f"seed={seed}: did not converge in {res.n_iter} iters"

    @pytest.mark.parametrize("seed", [0, 1, 5])
    def test_pairwise_random_3d(self, seed):
        rng = np.random.default_rng(seed + 100)
        x0, x1 = rng.standard_normal((2, 3))
        cov0 = _make_spd(rng, 3)
        cov1 = _make_spd(rng, 3)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        res = solve_minimax_from_coefs(coefs, tol=1e-10)
        expected = _pairwise_alpha_ellphi(pcoef, qcoef)

        assert res.alpha == pytest.approx(expected, rel=1e-5), (
            f"seed={seed}: alpha={res.alpha:.8f}, expected={expected:.8f}"
        )

    def test_pairwise_identical_ellipses(self):
        """Two identical ellipses: α = 0, x* = center."""
        x = np.array([1.0, 2.0])
        cov = np.diag([0.5, 2.0])
        coef = ellphi.coef_from_cov(x, cov)[0]
        coefs = np.stack([coef, coef])
        res = solve_minimax_from_coefs(coefs, tol=1e-10)
        assert res.alpha == pytest.approx(0.0, abs=1e-7)
        np.testing.assert_allclose(res.circumcenter, x, atol=1e-6)

    def test_pairwise_tangent_point_location(self):
        """The circumcenter x* should satisfy f_i(x*) = f_j(x*) = alpha."""
        rng = np.random.default_rng(99)
        x0, x1 = rng.standard_normal((2, 2))
        cov0 = _make_spd(rng, 2)
        cov1 = _make_spd(rng, 2)
        pcoef = ellphi.coef_from_cov(x0, cov0)[0]
        qcoef = ellphi.coef_from_cov(x1, cov1)[0]
        coefs = np.stack([pcoef, qcoef])

        from ellphi.geometry import unpack_conic
        A_arr, b_arr, _ = unpack_conic(coefs)
        centers = np.stack([-np.linalg.solve(A_arr[i], b_arr[i]) for i in range(2)])

        res = solve_minimax(A_arr, centers, tol=1e-12)
        xstar = res.circumcenter
        diff = xstar[np.newaxis] - centers
        f = np.einsum("ki,kij,kj->k", diff, A_arr, diff)

        assert f[0] == pytest.approx(f[1], rel=1e-5)
        assert f[0] == pytest.approx(res.alpha, rel=1e-5)


# ---------------------------------------------------------------------------
# Higher-dimensional simplex (|σ| ≥ 3)
# ---------------------------------------------------------------------------

class TestHigherOrder:
    def test_triangle_active_set(self):
        """For a centered equilateral triangle, all 3 weights should be active."""
        c = 2.0 * np.pi / 3.0
        pts = np.array([
            [1.0, 0.0],
            [np.cos(c), np.sin(c)],
            [np.cos(2 * c), np.sin(2 * c)],
        ])
        A = np.eye(2)[np.newaxis].repeat(3, axis=0)
        res = solve_minimax(A, pts, tol=1e-10)
        assert len(res.active_set) == 3
        assert res.converged

    def test_degenerate_triangle_reduces_to_edge(self):
        """If one point is much closer, its weight should be near zero (inactive)."""
        # x2 is very close to x* of {x0, x1}, so active set should be {0, 1}
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1e-6]])
        A = np.eye(2)[np.newaxis].repeat(3, axis=0)
        res = solve_minimax(A, pts, tol=1e-8)
        # alpha should be close to alpha({0, 1}) = 1.0
        assert res.alpha == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("k,d", [(3, 2), (4, 3), (5, 4)])
    def test_random_simplex_convergence(self, k, d):
        """Frank-Wolfe should converge for random k-simplex in R^d."""
        rng = np.random.default_rng(k * 100 + d)
        centers = rng.standard_normal((k, d))
        matrices = np.stack([_make_spd(rng, d) for _ in range(k)])
        res = solve_minimax(matrices, centers, tol=1e-8, max_iter=5000)
        assert res.converged, f"k={k}, d={d}: did not converge in {res.n_iter} iters"
        assert res.alpha >= 0.0
        assert len(res.active_set) >= 1
