"""Certified pruning-based filtration construction (theorems B, C, D, E, G).

Implements the certified candidate-generation pipeline of
``docs/certified_pruning_sync.md`` §3 WP1, backed by the Lean-verified
theorems of the paper repo (``anisotropic_cech_pruning_research_plan.md``):

- theorem B (`distance_pruning_soundness`): every simplex with
  ``alpha <= r_max`` is a clique of the neighbour graph G_L with
  ``L = 2 sqrt(r_max / m)``, ``m`` the Rayleigh lower bound.
- theorem C (`pairwise_minimax_pruning`): an edge with ``alpha > r_max``
  kills every coface.
- theorem D (`face_monotonicity_pruning`): a face with ``alpha > r_max``
  kills every coface.
- theorem E (`active_support_interval_value`): for a certified act support
  ``tau`` with coface set ``C(tau)``, every ``tau ⊆ sigma ⊆ C(tau)`` has
  ``alpha(sigma) = alpha(tau)`` — value reuse without a solve.
- theorem G (`card_lowSkeletonCliqueFinset_le`): clique-candidate counts are
  bounded by ``|V| * sum_t Delta^t``.

Tolerance contract (mirrors the paper's oracle semantics): acceptance uses
the primal upper bound ``max_i f_i(c)`` (valid at *any* candidate center, so
an accepted simplex is certified even if the solver did not converge);
rejection uses the weak-duality lower bound ``g(mu)`` at the returned weights
(valid for any feasible ``mu``).  Candidates caught between the two bounds
are retained and flagged "uncertain", never silently pruned.
"""

from __future__ import annotations

import math
import time
from typing import Iterator, Mapping, NamedTuple, Sequence

import numpy as np

from .act_map import SimplexInfo, act_support, certify_support, coface_set
from .filtration import FiltrationEntry
from .minimax import solve_minimax

__all__ = [
    "PruningStats",
    "CertifiedFiltration",
    "rayleigh_lower_bound",
    "neighbour_graph",
    "clique_candidates",
    "certified_filtration",
    "brute_force_filtration",
]

_DEFAULT_VALUE_TOL = 1e-9
_DEFAULT_COFACE_TOL = 1e-7

# Certificate statuses trusted enough to register interval value reuse.
_REUSABLE_STATUSES = ("certified", "escalated")


class PruningStats(NamedTuple):
    """Candidate counts per pruning stage plus solver effort."""

    n_vertices: int
    dimension: int
    r_max: float
    max_dim: int
    rayleigh_m: float
    radius_L: float
    brute_force_candidates: int  # all simplices with dim <= max_dim
    gl_clique_candidates: int  # cliques of G_L (theorem B stage)
    after_pairwise_candidates: int  # cliques of the retained-edge graph (theorem C)
    evaluated_candidates: int  # survived the face check (theorem D)
    solves: int
    reuses: int  # theorem E value assignments
    rejected: int  # certified alpha > r_max
    uncertain: int  # retained without a certified value
    emitted: int
    wall_time_s: float


class CertifiedFiltration(NamedTuple):
    entries: list[FiltrationEntry]
    info: dict[tuple[int, ...], SimplexInfo]
    stats: PruningStats


def rayleigh_lower_bound(matrices: np.ndarray) -> float:
    """Uniform Rayleigh lower bound m: min over sites of lambda_min(A_i)."""
    matrices = np.asarray(matrices, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrices)
    m = float(np.min(eigenvalues))
    if m <= 0.0:
        raise ValueError("all site matrices must be positive definite")
    return m


def neighbour_graph(centers: np.ndarray, radius: float) -> list[set[int]]:
    """Adjacency sets of G_L: i ~ j iff ||x_i - x_j|| <= radius (i != j)."""
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    within = dist2 <= radius * radius
    np.fill_diagonal(within, False)
    return [set(int(j) for j in np.nonzero(within[i])[0]) for i in range(n)]


def clique_candidates(
    adjacency: Sequence[set[int]],
    max_dim: int,
    *,
    min_size: int = 1,
) -> Iterator[tuple[int, ...]]:
    """Yield cliques of sizes ``min_size .. max_dim+1`` in dimension-lex order.

    Enumeration extends each clique by higher-indexed common neighbours, so
    the work matches the theorem G bounded-degree accounting.
    """
    n = len(adjacency)
    level: list[tuple[int, ...]] = [(v,) for v in range(n)]
    size = 1
    while level and size <= max_dim + 1:
        if size >= min_size:
            yield from level
        if size == max_dim + 1:
            break
        next_level: list[tuple[int, ...]] = []
        for clique in level:
            common = set(adjacency[clique[-1]])
            for v in clique[:-1]:
                common &= adjacency[v]
            for w in sorted(common):
                if w > clique[-1]:
                    next_level.append(clique + (w,))
        level = next_level
        size += 1


def _codim_one_faces(simplex: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))]


def _primal_upper_bound(
    matrices: np.ndarray, centers: np.ndarray, x: np.ndarray
) -> float:
    diff = x[np.newaxis, :] - centers
    return float(np.max(np.einsum("ki,kij,kj->k", diff, matrices, diff)))


def _dual_lower_bound(
    matrices: np.ndarray, centers: np.ndarray, weights: np.ndarray
) -> float:
    """Weak-duality lower bound g(mu) at any feasible mu (clipped/renormalized)."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    w = w / total
    a_mu = np.einsum("k,kij->ij", w, matrices)
    b_mu = np.einsum("k,kij,kj->i", w, matrices, centers)
    c_mu = float(np.einsum("k,ki,kij,kj->", w, centers, matrices, centers))
    try:
        x_mu = np.linalg.solve(a_mu, b_mu)
    except np.linalg.LinAlgError:
        return 0.0
    return max(0.0, c_mu - float(b_mu @ x_mu))


def certified_filtration(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    r_max: float,
    max_dim: int,
    value_tol: float = _DEFAULT_VALUE_TOL,
    coface_tol: float = _DEFAULT_COFACE_TOL,
    minimax_kwargs: Mapping[str, object] | None = None,
) -> CertifiedFiltration:
    """Build the anisotropic Cech filtration up to ``r_max`` with pruning.

    Emits exactly the simplices with ``alpha(sigma) <= r_max`` (theorem A
    semantics — predicates P2/P3 do *not* gate membership) for
    ``dim <= max_dim``, with their filtration values, using theorems B/C/D
    to prune candidates and theorem E to reuse values across act intervals.
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if matrices.ndim != 3 or centers.ndim != 2:
        raise ValueError("matrices must be (n, d, d) and centers (n, d)")
    if matrices.shape[0] != centers.shape[0]:
        raise ValueError("matrices and centers must have the same number of vertices")
    if matrices.shape[1] != matrices.shape[2] or matrices.shape[1] != centers.shape[1]:
        raise ValueError("matrix and center dimensions must agree")
    if r_max < 0.0:
        raise ValueError("r_max must be non-negative")

    t0 = time.perf_counter()
    n, d = centers.shape
    kwargs = dict(minimax_kwargs or {})

    m = rayleigh_lower_bound(matrices)
    # value_tol under the root keeps boundary simplices (alpha == r_max) in
    # G_L despite sqrt/square rounding.
    radius = 2.0 * math.sqrt((r_max + value_tol) / m)
    adjacency = neighbour_graph(centers, radius)

    brute_force = sum(math.comb(n, k) for k in range(1, max_dim + 2))
    gl_cliques = sum(1 for _ in clique_candidates(adjacency, max_dim))

    entries: list[FiltrationEntry] = []
    info: dict[tuple[int, ...], SimplexInfo] = {}
    alpha_by_simplex: dict[tuple[int, ...], float] = {}
    solves = reuses = rejected = uncertain = 0

    accept_threshold = r_max + value_tol

    def solve_and_certify(simplex: tuple[int, ...]) -> tuple[float, float, object]:
        """Return (upper, lower, certificate) for one candidate."""
        idx = np.asarray(simplex, dtype=int)
        result = solve_minimax(matrices[idx], centers[idx], **kwargs)
        cert = certify_support(
            matrices[idx], centers[idx], result, minimax_kwargs=kwargs
        )
        result = cert.result  # possibly the escalated re-solve
        upper = _primal_upper_bound(
            matrices[idx], centers[idx], np.asarray(result.circumcenter, dtype=float)
        )
        lower = _dual_lower_bound(matrices[idx], centers[idx], result.weights)
        return upper, lower, cert

    def register(
        simplex: tuple[int, ...], alpha: float, cert, source: str
    ) -> None:
        result = cert.result
        entries.append(FiltrationEntry(simplex=simplex, alpha=alpha, predicates=None))
        alpha_by_simplex[simplex] = alpha
        info[simplex] = SimplexInfo(
            simplex=simplex,
            alpha=alpha,
            support=act_support(simplex, cert),
            coface=coface_set(
                matrices,
                centers,
                np.asarray(result.circumcenter, dtype=float),
                alpha,
                tol=coface_tol,
            ),
            status=cert.status,
            source=source,
            circumcenter=np.asarray(result.circumcenter, dtype=float),
        )

    # Dimension 0: every vertex enters at alpha = 0.
    for v in range(n):
        simplex = (v,)
        entries.append(FiltrationEntry(simplex=simplex, alpha=0.0, predicates=None))
        alpha_by_simplex[simplex] = 0.0
        info[simplex] = SimplexInfo(
            simplex=simplex,
            alpha=0.0,
            support=simplex,
            coface=coface_set(matrices, centers, centers[v], 0.0, tol=coface_tol),
            status="certified",
            source="vertex",
            circumcenter=centers[v].copy(),
        )

    # Dimension 1: solve G_L edges; removed edges realize theorem C.
    retained_adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in sorted(adjacency[i]):
            if j <= i:
                continue
            simplex = (i, j)
            upper, lower, cert = solve_and_certify(simplex)
            solves += 1
            if upper <= accept_threshold:
                retained_adjacency[i].add(j)
                retained_adjacency[j].add(i)
                register(simplex, max(upper, 0.0), cert, "solved")
            elif lower > accept_threshold:
                rejected += 1
            else:
                # Ambiguous: retain (superset semantics), flag uncertain.
                uncertain += 1
                retained_adjacency[i].add(j)
                retained_adjacency[j].add(i)
                register(simplex, max(upper, 0.0), cert._replace(status="uncertain"), "solved")

    after_pairwise = n + sum(len(a) for a in retained_adjacency) // 2
    evaluated = n + solves

    # Dimensions >= 2: cliques of the retained-edge graph, face-checked
    # (theorem D), value-reused when inside a certified act interval
    # (theorem E), solved otherwise.
    for simplex in clique_candidates(retained_adjacency, max_dim, min_size=3):
        after_pairwise += 1
        faces = _codim_one_faces(simplex)
        if any(face not in alpha_by_simplex for face in faces):
            continue  # a certified-rejected face kills this candidate
        evaluated += 1

        reused = False
        simplex_set = set(simplex)
        for face in faces:
            face_info = info[face]
            if face_info.status not in _REUSABLE_STATUSES:
                continue
            if simplex_set <= face_info.coface:
                alpha = max(
                    face_info.alpha, *(alpha_by_simplex[f] for f in faces)
                )
                entries.append(
                    FiltrationEntry(simplex=simplex, alpha=alpha, predicates=None)
                )
                alpha_by_simplex[simplex] = alpha
                info[simplex] = face_info._replace(
                    simplex=simplex, alpha=alpha, source="reused"
                )
                reuses += 1
                reused = True
                break
        if reused:
            continue

        upper, lower, cert = solve_and_certify(simplex)
        solves += 1
        if upper <= accept_threshold:
            alpha = max(upper, *(alpha_by_simplex[f] for f in faces))
            register(simplex, alpha, cert, "solved")
        elif lower > accept_threshold:
            rejected += 1
        else:
            uncertain += 1
            alpha = max(upper, *(alpha_by_simplex[f] for f in faces))
            register(simplex, alpha, cert._replace(status="uncertain"), "solved")

    entries.sort(key=lambda e: (e.alpha, len(e.simplex), e.simplex))
    stats = PruningStats(
        n_vertices=n,
        dimension=d,
        r_max=float(r_max),
        max_dim=max_dim,
        rayleigh_m=m,
        radius_L=radius,
        brute_force_candidates=brute_force,
        gl_clique_candidates=gl_cliques,
        after_pairwise_candidates=after_pairwise,
        evaluated_candidates=evaluated,
        solves=solves,
        reuses=reuses,
        rejected=rejected,
        uncertain=uncertain,
        emitted=len(entries),
        wall_time_s=time.perf_counter() - t0,
    )
    return CertifiedFiltration(entries=entries, info=info, stats=stats)


def brute_force_filtration(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    r_max: float,
    max_dim: int,
    value_tol: float = _DEFAULT_VALUE_TOL,
    minimax_kwargs: Mapping[str, object] | None = None,
) -> list[FiltrationEntry]:
    """Reference pipeline: solve every candidate simplex, no pruning.

    Uses the same acceptance rule (primal upper bound ``<= r_max + tol``) as
    :func:`certified_filtration`, so the two outputs are directly comparable
    (theorem I soundness semantics).
    """
    matrices = np.asarray(matrices, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    kwargs = dict(minimax_kwargs or {})

    entries: list[FiltrationEntry] = []
    alpha_by_simplex: dict[tuple[int, ...], float] = {}
    accept_threshold = r_max + value_tol

    from .enumeration import iter_candidate_simplices

    for simplex in iter_candidate_simplices(n, max_dim=max_dim):
        if len(simplex) == 1:
            alpha_by_simplex[simplex] = 0.0
            entries.append(FiltrationEntry(simplex=simplex, alpha=0.0, predicates=None))
            continue
        faces = _codim_one_faces(simplex)
        if any(face not in alpha_by_simplex for face in faces):
            continue
        idx = np.asarray(simplex, dtype=int)
        result = solve_minimax(matrices[idx], centers[idx], **kwargs)
        upper = _primal_upper_bound(
            matrices[idx], centers[idx], np.asarray(result.circumcenter, dtype=float)
        )
        if upper > accept_threshold:
            continue
        alpha = max(upper, *(alpha_by_simplex[face] for face in faces))
        alpha_by_simplex[simplex] = alpha
        entries.append(FiltrationEntry(simplex=simplex, alpha=alpha, predicates=None))

    entries.sort(key=lambda e: (e.alpha, len(e.simplex), e.simplex))
    return entries
