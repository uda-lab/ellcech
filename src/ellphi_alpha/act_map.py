"""Act-map structure for the anisotropic alpha filtration.

Implements the numerical counterpart of the act-map program described in
``docs/certified_pruning_sync.md`` (paper-side sources:
``../paper-ellalpha/alpha_like_reduction_research_plan.md`` and
``../paper-ellalpha/notes/act_map_proofs.md``):

- KKT support certificates: the raw ``MinimaxResult.active_set`` is unreliable
  near degenerate configurations, so a support is only trusted after checking
  the equal-value spread, the stationarity residual, and affine independence
  of the support gradients, with escalation to ``scipy-slsqp`` on failure.
- The coface set C(sigma) = {j : f_j(c_sigma) <= alpha(sigma)} (theorem E),
  exposed instead of being discarded inside predicate P3.
- Fiber partition, Booleanity report, and critical simplices (the Del^aniso
  candidates a(sigma) = sigma, C(sigma) = sigma).
- ``critical_value_check``: every finite persistence endpoint of the full
  filtration must be a critical value — the observable Morse-theoretic
  consequence that avoids implementing the Morse boundary operator.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from .minimax import MinimaxResult, solve_minimax

__all__ = [
    "SupportCertificate",
    "SimplexInfo",
    "certify_support",
    "act_support",
    "coface_set",
    "fiber_partition",
    "BooleanityViolation",
    "BooleanityReport",
    "booleanity_report",
    "critical_simplices",
    "CriticalValueCheck",
    "critical_value_check",
]

_DEFAULT_SUPPORT_TOL = 1e-6
_DEFAULT_CERT_TOL = 1e-6
_DEFAULT_COFACE_TOL = 1e-7

#: Certificate statuses, from most to least trusted:
#: - "certified":  KKT conditions verified on the reported support, gradients
#:   affinely independent (the support is the unique minimal one locally).
#: - "escalated":  same, but only after re-solving with scipy-slsqp.
#: - "degenerate": KKT conditions verified but the support gradients are
#:   affinely dependent — a tie between minimal supports is suspected
#:   (nondegeneracy (N) fails or nearly fails at this simplex).
#: - "uncertain":  no verified certificate even after escalation.
CERT_STATUSES = ("certified", "escalated", "degenerate", "uncertain")


class SupportCertificate(NamedTuple):
    """Verified (or failed) KKT support certificate for one minimax solve."""

    support: tuple[int, ...]  # local indices into the simplex
    status: str  # one of CERT_STATUSES
    spread: float  # max_i |f_i(c) - alpha| / max(1, |alpha|) over the support
    stationarity: float  # ||sum_i lambda_i grad f_i(c)|| (relative)
    result: MinimaxResult  # result the certificate refers to (may be re-solved)


class SimplexInfo(NamedTuple):
    """Act-map data attached to one filtration simplex."""

    simplex: tuple[int, ...]
    alpha: float
    support: tuple[int, ...]  # act support a(sigma), global vertex ids
    coface: frozenset[int]  # C at the simplex's center and level
    status: str  # certificate status of the underlying solve
    source: str  # "vertex" | "solved" | "reused"
    circumcenter: np.ndarray | None


def _simplex_values_and_gradients(
    matrices: np.ndarray, centers: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    diff = x[np.newaxis, :] - centers
    values = np.einsum("ki,kij,kj->k", diff, matrices, diff)
    grads = 2.0 * np.einsum("kij,kj->ki", matrices, diff)
    return values, grads


def _affinely_independent(grads: np.ndarray) -> bool:
    """Whether the rows of ``grads`` are affinely independent."""
    k = grads.shape[0]
    if k <= 1:
        return True
    rel = grads[1:] - grads[0]
    scale = max(1.0, float(np.max(np.abs(grads))))
    s = np.linalg.svd(rel, compute_uv=False)
    rank = int(np.sum(s > 1e-8 * scale))
    return rank == k - 1


def _check_certificate(
    matrices: np.ndarray,
    centers: np.ndarray,
    result: MinimaxResult,
    *,
    support_tol: float,
    cert_tol: float,
) -> SupportCertificate:
    weights = np.asarray(result.weights, dtype=float)
    support = tuple(int(i) for i in np.nonzero(weights > support_tol)[0])
    alpha = float(result.alpha)
    d = centers.shape[1]

    if not support or not result.converged or not np.isfinite(alpha):
        return SupportCertificate(support, "uncertain", np.inf, np.inf, result)

    x = np.asarray(result.circumcenter, dtype=float)
    values, grads = _simplex_values_and_gradients(matrices, centers, x)
    sub = np.asarray(support, dtype=int)

    scale = max(1.0, abs(alpha))
    spread = float(np.max(np.abs(values[sub] - alpha))) / scale
    # alpha must actually be attained: no simplex value may exceed it.
    overshoot = float(np.max(values) - alpha) / scale

    lam = weights[sub]
    lam = lam / float(np.sum(lam))
    grad_scale = max(1.0, float(np.max(np.linalg.norm(grads[sub], axis=1))))
    stationarity = float(np.linalg.norm(lam @ grads[sub])) / grad_scale

    kkt_ok = (
        spread <= cert_tol
        and overshoot <= cert_tol
        and stationarity <= cert_tol
        and len(support) <= d + 1
    )
    if not kkt_ok:
        # A support larger than d+1 with an otherwise valid certificate is a
        # degeneracy symptom, not a numerical failure.
        if (
            spread <= cert_tol
            and overshoot <= cert_tol
            and stationarity <= cert_tol
            and len(support) > d + 1
        ):
            return SupportCertificate(support, "degenerate", spread, stationarity, result)
        return SupportCertificate(support, "uncertain", spread, stationarity, result)

    if not _affinely_independent(grads[sub]):
        return SupportCertificate(support, "degenerate", spread, stationarity, result)
    return SupportCertificate(support, "certified", spread, stationarity, result)


def certify_support(
    matrices: np.ndarray,
    centers: np.ndarray,
    result: MinimaxResult,
    *,
    support_tol: float = _DEFAULT_SUPPORT_TOL,
    cert_tol: float = _DEFAULT_CERT_TOL,
    escalate: bool = True,
    minimax_kwargs: Mapping[str, object] | None = None,
) -> SupportCertificate:
    """Verify the KKT support of a minimax solve, escalating if needed.

    ``matrices`` and ``centers`` are the simplex-local arrays that were passed
    to :func:`ellphi_alpha.solve_minimax` (shapes ``(k, d, d)`` and ``(k, d)``).

    The raw ``active_set`` (all weights above ``weight_tol``) is *not* trusted:
    weights are re-thresholded at ``support_tol`` and the certificate checks

    1. equal-value spread: ``f_i(c) == alpha`` on the support,
    2. attainment: ``max_i f_i(c) <= alpha`` over the whole simplex,
    3. stationarity: ``sum_i lambda_i grad f_i(c) == 0``,
    4. minimality proxy: support gradients affinely independent
       (hence ``|support| <= d+1``).

    On failure the instance is re-solved with ``scipy-slsqp`` and re-checked;
    a certificate that only passes after escalation gets status "escalated".
    A certificate passing 1–3 but failing 4 gets status "degenerate" (a tie
    between minimal supports is suspected).  Anything else is "uncertain" and
    must never be used to prune or reuse values.
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)
    cert = _check_certificate(
        matrices, centers, result, support_tol=support_tol, cert_tol=cert_tol
    )
    if cert.status != "uncertain" or not escalate:
        return cert
    if result.method == "scipy-slsqp":
        return cert

    kwargs = dict(minimax_kwargs or {})
    kwargs["method"] = "scipy-slsqp"
    retried = solve_minimax(matrices, centers, **kwargs)
    cert2 = _check_certificate(
        matrices, centers, retried, support_tol=support_tol, cert_tol=cert_tol
    )
    if cert2.status == "certified":
        return cert2._replace(status="escalated")
    if cert2.status == "degenerate":
        return cert2
    # Keep whichever attempt has the smaller residual evidence.
    return cert2 if cert2.spread + cert2.stationarity <= cert.spread + cert.stationarity else cert


def act_support(simplex: Sequence[int], certificate: SupportCertificate) -> tuple[int, ...]:
    """Map a certificate's local support indices to sorted global vertex ids."""
    simplex_t = tuple(int(i) for i in simplex)
    return tuple(sorted(simplex_t[i] for i in certificate.support))


def coface_set(
    matrices: np.ndarray,
    centers: np.ndarray,
    circumcenter: np.ndarray,
    alpha: float,
    *,
    tol: float = _DEFAULT_COFACE_TOL,
) -> frozenset[int]:
    """C(sigma) = {j : f_j(c_sigma) <= alpha(sigma)} over *all* vertices.

    This is the interval upper end of theorem E: every simplex between the
    act support and C carries the same filtration value.  Exposes the scan
    that predicate P3 performs internally and then discards.
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)
    x = np.asarray(circumcenter, dtype=float)
    diff = x[np.newaxis, :] - centers
    f_all = np.einsum("ki,kij,kj->k", diff, matrices, diff)
    scale = max(1.0, abs(float(alpha)))
    return frozenset(int(j) for j in np.nonzero(f_all <= alpha + tol * scale)[0])


def fiber_partition(
    infos: Mapping[tuple[int, ...], SimplexInfo],
) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """Group simplices by their act support a(sigma)."""
    fibers: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for simplex, info in infos.items():
        fibers.setdefault(info.support, []).append(simplex)
    for members in fibers.values():
        members.sort(key=lambda s: (len(s), s))
    return fibers


class BooleanityViolation(NamedTuple):
    """One act fiber that is not the Boolean interval [tau, C(tau)]."""

    support: tuple[int, ...]
    coface: tuple[int, ...]
    missing: tuple[tuple[int, ...], ...]  # in [tau, C(tau)] but not in the fiber
    extra: tuple[tuple[int, ...], ...]  # in the fiber but not in [tau, C(tau)]


class BooleanityReport(NamedTuple):
    n_simplices: int
    n_fibers: int
    fiber_sizes: dict[int, int]  # size -> count of fibers of that size
    violations: tuple[BooleanityViolation, ...]
    skipped_fibers: tuple[tuple[int, ...], ...]  # coface too large to expand

    @property
    def is_boolean(self) -> bool:
        return not self.violations


def booleanity_report(
    infos: Mapping[tuple[int, ...], SimplexInfo],
    *,
    max_dim: int,
    max_interval_bits: int = 20,
) -> BooleanityReport:
    """Check that each act fiber equals [tau, C(tau)] cut at ``max_dim``.

    Under nondegeneracy (N) this is the Booleanity theorem (act fibers are
    Boolean intervals); violations are expected exactly on degenerate inputs.
    Fibers whose interval would need more than ``2**max_interval_bits``
    subsets are reported as skipped rather than expanded.
    """
    fibers = fiber_partition(infos)
    violations: list[BooleanityViolation] = []
    skipped: list[tuple[int, ...]] = []
    sizes: dict[int, int] = {}

    for support, members in fibers.items():
        sizes[len(members)] = sizes.get(len(members), 0) + 1
        # The interval is defined by the coface set recorded at the fiber
        # bottom (the support itself, when present as a simplex).
        base = infos.get(support)
        if base is None:
            base = infos[members[0]]
        coface = base.coface
        free = sorted(coface - set(support))
        if len(free) > max_interval_bits:
            skipped.append(support)
            continue
        expected: set[tuple[int, ...]] = set()
        for size in range(0, len(free) + 1):
            if len(support) + size > max_dim + 1:
                break
            for extra_vertices in combinations(free, size):
                expected.add(tuple(sorted(set(support) | set(extra_vertices))))
        actual = set(members)
        if expected != actual:
            violations.append(
                BooleanityViolation(
                    support=support,
                    coface=tuple(sorted(coface)),
                    missing=tuple(sorted(expected - actual)),
                    extra=tuple(sorted(actual - expected)),
                )
            )

    return BooleanityReport(
        n_simplices=len(infos),
        n_fibers=len(fibers),
        fiber_sizes=dict(sorted(sizes.items())),
        violations=tuple(violations),
        skipped_fibers=tuple(skipped),
    )


def critical_simplices(
    infos: Mapping[tuple[int, ...], SimplexInfo],
) -> list[tuple[int, ...]]:
    """Del^aniso candidates: a(sigma) = sigma and C(sigma) = sigma."""
    out = [
        simplex
        for simplex, info in infos.items()
        if info.support == simplex and info.coface == frozenset(simplex)
    ]
    out.sort(key=lambda s: (len(s), s))
    return out


class CriticalValueCheck(NamedTuple):
    checked_endpoints: int
    violations: tuple[tuple[int, float], ...]  # (homology dim, endpoint value)

    @property
    def passed(self) -> bool:
        return not self.violations


def critical_value_check(
    intervals_by_dim: Mapping[int, np.ndarray],
    critical_alphas: Iterable[float],
    *,
    tol: float = 1e-9,
) -> CriticalValueCheck:
    """Check every finite persistence endpoint is a critical alpha value.

    ``intervals_by_dim`` maps homology dimension to an ``(k, 2)`` array of
    (birth, death) pairs (death may be ``inf``).  This is the observable
    consequence of the Morse chain model: all births and deaths happen at
    critical values.
    """
    crit = np.asarray(sorted(set(float(a) for a in critical_alphas)), dtype=float)
    violations: list[tuple[int, float]] = []
    checked = 0
    for dim, intervals in intervals_by_dim.items():
        arr = np.asarray(intervals, dtype=float).reshape(-1, 2)
        for endpoint in arr.ravel():
            if not np.isfinite(endpoint):
                continue
            checked += 1
            scale = max(1.0, abs(endpoint))
            if crit.size == 0 or np.min(np.abs(crit - endpoint)) > tol * scale:
                violations.append((int(dim), float(endpoint)))
    return CriticalValueCheck(checked_endpoints=checked, violations=tuple(violations))
