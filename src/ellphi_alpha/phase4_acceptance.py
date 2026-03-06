"""Phase-4 acceptance measurement helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends.gudhi import GudhiBackend
from .core.backend_contracts import PersistenceBackend
from .filtration import FiltrationEntry, build_incremental_filtration
from .minimax import solve_minimax_from_coefs

DEFAULT_BASELINE_SEED = 20260306
DEFAULT_SIX_RINGS_SEED = 20260307
DEFAULT_CONDITIONING_SEED = 20260308


@dataclass(frozen=True)
class BaselineBarcodeAgreement:
    """Result bundle for the Phase-4 baseline barcode agreement check."""

    status: str
    message: str
    passed: bool | None
    n_points: int
    dimension: int
    random_seed: int
    simplex_count: int
    edge_count: int
    reference_backend: str | None
    h0_count_ours: int | None
    h0_count_reference: int | None
    h0_distance: float | None
    h0_distance_metric: str | None
    h0_count_gudhi: int | None
    h0_bottleneck: float | None
    max_abs_edge_alpha_diff: float | None
    missing_edge_count: int | None
    extra_edge_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SixRingsH1Check:
    """Result bundle for the 6-rings long-lived H1 check."""

    status: str
    message: str
    passed: bool | None
    rings: int
    points_per_ring: int
    n_points: int
    random_seed: int
    reference_backend: str | None
    h1_count_reference: int | None
    long_lived_h1_count_reference: int | None
    top_h1_lifetimes_reference: list[float] | None
    h1_count: int | None
    long_lived_h1_count: int | None
    min_long_lived_h1: int
    lifetime_threshold: float
    top_h1_lifetimes: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditioningStressCheck:
    """Result bundle for the high-conditioning alpha error benchmark."""

    status: str
    message: str
    passed: bool
    n_cases: int
    evaluated_cases: int
    failed_cases: int
    dimension: int
    random_seed: int
    min_condition_number: float
    max_condition_number: float
    cond_min_observed: float
    cond_max_observed: float
    rel_error_threshold: float
    abs_error_threshold: float
    max_rel_error: float
    mean_rel_error: float
    max_abs_error: float
    mean_abs_error: float
    over_threshold_cases: int
    worst_cases: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_gudhi() -> bool:
    return GudhiBackend().is_available()


def _extract_edge_map_from_entries(
    filtration: list[FiltrationEntry],
) -> dict[tuple[int, int], float]:
    edge_map: dict[tuple[int, int], float] = {}
    for entry in filtration:
        if len(entry.simplex) != 2:
            continue
        i, j = entry.simplex
        edge_map[(i, j)] = float(entry.alpha)
    return edge_map


def _make_six_rings_points(
    *,
    points_per_ring: int,
    ring_radius: float,
    spacing: float,
    noise: float,
    random_seed: int,
) -> np.ndarray:
    if points_per_ring < 8:
        raise ValueError("points_per_ring must be >= 8")
    if ring_radius <= 0:
        raise ValueError("ring_radius must be positive")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(random_seed)
    centers = np.array(
        [
            [-spacing, -spacing / 2.0],
            [0.0, -spacing / 2.0],
            [spacing, -spacing / 2.0],
            [-spacing, spacing / 2.0],
            [0.0, spacing / 2.0],
            [spacing, spacing / 2.0],
        ],
        dtype=float,
    )

    points: list[np.ndarray] = []
    base_angles = np.linspace(0.0, 2.0 * np.pi, points_per_ring, endpoint=False)
    for c in centers:
        angle_offset = rng.uniform(0.0, 2.0 * np.pi)
        angles = base_angles + angle_offset
        ring = np.column_stack([np.cos(angles), np.sin(angles)]) * ring_radius
        if noise > 0:
            ring = ring + rng.normal(scale=noise, size=ring.shape)
        points.append(ring + c)
    return np.vstack(points)


def _random_spd_with_target_condition(
    rng: np.random.Generator,
    *,
    dimension: int,
    target_condition: float,
) -> np.ndarray:
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    if target_condition <= 1.0:
        raise ValueError("target_condition must be > 1")

    q, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    eigenvalues = np.geomspace(1.0, target_condition, num=dimension)
    matrix = q @ np.diag(eigenvalues) @ q.T
    return 0.5 * (matrix + matrix.T)


def run_baseline_barcode_agreement(
    *,
    n_points: int = 100,
    dimension: int = 2,
    random_seed: int = DEFAULT_BASELINE_SEED,
    edge_alpha_tol: float = 1e-6,
    bottleneck_tol: float = 1e-6,
    backend: PersistenceBackend | None = None,
) -> BaselineBarcodeAgreement:
    """Check H0 barcode agreement against a reference alpha-complex backend."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    backend = GudhiBackend() if backend is None else backend

    rng = np.random.default_rng(random_seed)
    centers = rng.standard_normal((n_points, dimension))
    matrices = np.repeat(np.eye(dimension, dtype=float)[np.newaxis, :, :], n_points, axis=0)
    filtration = build_incremental_filtration(
        matrices,
        centers,
        max_dim=1,
        minimax_kwargs={"tol": 1e-10, "max_iter": 2000},
    )
    our_edges = _extract_edge_map_from_entries(filtration)

    if not backend.is_available():
        backend_name = getattr(backend, "name", None)
        return BaselineBarcodeAgreement(
            status="skipped",
            message=f"{backend_name} is not installed; baseline comparison skipped.",
            passed=None,
            n_points=n_points,
            dimension=dimension,
            random_seed=random_seed,
            simplex_count=len(filtration),
            edge_count=len(our_edges),
            reference_backend=backend_name,
            h0_count_ours=None,
            h0_count_reference=None,
            h0_distance=None,
            h0_distance_metric="bottleneck",
            h0_count_gudhi=None,
            h0_bottleneck=None,
            max_abs_edge_alpha_diff=None,
            missing_edge_count=None,
            extra_edge_count=None,
        )

    ours_tree = backend.simplex_tree_from_filtration(filtration)
    reference_tree = backend.alpha_complex_simplex_tree(centers)

    ours_h0 = backend.persistence_intervals(ours_tree, dimension=0, homology_coeff_field=2)
    reference_h0 = backend.persistence_intervals(
        reference_tree,
        dimension=0,
        homology_coeff_field=2,
    )
    h0_distance = backend.bottleneck_distance(ours_h0, reference_h0)

    reference_edges = backend.edge_map(reference_tree)
    our_edge_set = set(our_edges)
    ref_edge_set = set(reference_edges)
    common_edges = our_edge_set & ref_edge_set
    missing_edge_count = int(len(ref_edge_set - our_edge_set))
    extra_edge_count = int(len(our_edge_set - ref_edge_set))
    if common_edges:
        max_abs_edge_alpha_diff: float | None = max(
            abs(our_edges[e] - reference_edges[e]) for e in common_edges
        )
    else:
        # No common edges: cannot compute alpha difference; report as None
        # (float("inf") is not strict JSON and is misleading when edge sets are disjoint).
        max_abs_edge_alpha_diff = None

    edge_alpha_ok = max_abs_edge_alpha_diff is None or max_abs_edge_alpha_diff <= edge_alpha_tol
    passed = (
        len(ours_h0) == len(reference_h0)
        and h0_distance <= bottleneck_tol
        and edge_alpha_ok
        and missing_edge_count == 0
        and extra_edge_count == 0
    )
    message = (
        "H0 barcode agreement passed."
        if passed
        else "H0 barcode agreement failed threshold checks."
    )

    return BaselineBarcodeAgreement(
        status="ok" if passed else "failed",
        message=message,
        passed=passed,
        n_points=n_points,
        dimension=dimension,
        random_seed=random_seed,
        simplex_count=len(filtration),
        edge_count=len(our_edges),
        reference_backend=backend.name,
        h0_count_ours=int(len(ours_h0)),
        h0_count_reference=int(len(reference_h0)),
        h0_distance=h0_distance,
        h0_distance_metric="bottleneck",
        h0_count_gudhi=int(len(reference_h0)),
        h0_bottleneck=h0_distance,
        max_abs_edge_alpha_diff=float(max_abs_edge_alpha_diff) if max_abs_edge_alpha_diff is not None else None,
        missing_edge_count=missing_edge_count,
        extra_edge_count=extra_edge_count,
    )


def run_six_rings_h1_check(
    *,
    points_per_ring: int = 24,
    ring_radius: float = 1.0,
    spacing: float = 6.0,
    noise: float = 0.03,
    random_seed: int = DEFAULT_SIX_RINGS_SEED,
    lifetime_threshold: float = 0.25,
    min_long_lived_h1: int = 6,
    backend: PersistenceBackend | None = None,
) -> SixRingsH1Check:
    """Check that a 6-rings dataset exposes at least 6 long-lived H1 classes."""
    backend = GudhiBackend() if backend is None else backend

    points = _make_six_rings_points(
        points_per_ring=points_per_ring,
        ring_radius=ring_radius,
        spacing=spacing,
        noise=noise,
        random_seed=random_seed,
    )

    if not backend.is_available():
        backend_name = getattr(backend, "name", None)
        return SixRingsH1Check(
            status="skipped",
            message=f"{backend_name} is not installed; 6-rings H1 check skipped.",
            passed=None,
            rings=6,
            points_per_ring=points_per_ring,
            n_points=int(points.shape[0]),
            random_seed=random_seed,
            reference_backend=backend_name,
            h1_count_reference=None,
            long_lived_h1_count_reference=None,
            top_h1_lifetimes_reference=None,
            h1_count=None,
            long_lived_h1_count=None,
            min_long_lived_h1=min_long_lived_h1,
            lifetime_threshold=lifetime_threshold,
            top_h1_lifetimes=None,
        )

    simplex_tree = backend.alpha_complex_simplex_tree(points)
    h1 = backend.persistence_intervals(simplex_tree, dimension=1, homology_coeff_field=2)
    if h1.size == 0:
        lifetimes = np.array([], dtype=float)
    else:
        finite_mask = np.isfinite(h1[:, 1])
        finite_h1 = h1[finite_mask]
        lifetimes = finite_h1[:, 1] - finite_h1[:, 0]

    sorted_lifetimes = np.sort(lifetimes)[::-1]
    long_lived_count = int(np.sum(sorted_lifetimes >= lifetime_threshold))
    passed = long_lived_count >= min_long_lived_h1
    message = (
        "6-rings H1 long-lived check passed."
        if passed
        else "6-rings H1 long-lived check failed threshold."
    )

    return SixRingsH1Check(
        status="ok" if passed else "failed",
        message=message,
        passed=passed,
        rings=6,
        points_per_ring=points_per_ring,
        n_points=int(points.shape[0]),
        random_seed=random_seed,
        reference_backend=backend.name,
        h1_count_reference=int(len(h1)),
        long_lived_h1_count_reference=long_lived_count,
        top_h1_lifetimes_reference=[float(v) for v in sorted_lifetimes[:10]],
        h1_count=int(len(h1)),
        long_lived_h1_count=long_lived_count,
        min_long_lived_h1=min_long_lived_h1,
        lifetime_threshold=lifetime_threshold,
        top_h1_lifetimes=[float(v) for v in sorted_lifetimes[:10]],
    )


def run_conditioning_stress_check(
    *,
    n_cases: int = 32,
    dimension: int = 2,
    min_condition_number: float = 1e6,
    max_condition_number: float = 1e7,
    random_seed: int = DEFAULT_CONDITIONING_SEED,
    rel_error_threshold: float = 1e-4,
    abs_error_threshold: float = 1e-3,
) -> ConditioningStressCheck:
    """Benchmark alpha error on high-condition-number pairwise problems."""
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    if min_condition_number <= 1.0:
        raise ValueError("min_condition_number must be > 1")
    if max_condition_number < min_condition_number:
        raise ValueError("max_condition_number must be >= min_condition_number")

    import ellphi

    rng = np.random.default_rng(random_seed)
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    observed_conds: list[float] = []
    over_threshold_cases = 0
    failed_cases = 0
    worst_cases: list[dict[str, float]] = []

    for idx in range(n_cases):
        target_cond_0 = float(
            10 ** rng.uniform(np.log10(min_condition_number), np.log10(max_condition_number))
        )
        target_cond_1 = float(
            10 ** rng.uniform(np.log10(min_condition_number), np.log10(max_condition_number))
        )
        A0 = _random_spd_with_target_condition(
            rng,
            dimension=dimension,
            target_condition=target_cond_0,
        )
        A1 = _random_spd_with_target_condition(
            rng,
            dimension=dimension,
            target_condition=target_cond_1,
        )
        # Use min so that BOTH matrices in the pair must exceed the threshold.
        # Using max would allow one under-conditioned matrix to slip through.
        observed_cond = float(min(np.linalg.cond(A0), np.linalg.cond(A1)))
        observed_conds.append(observed_cond)

        centers = rng.standard_normal((2, dimension)) * 2.0
        cov0 = np.linalg.inv(A0)
        cov1 = np.linalg.inv(A1)

        try:
            pcoef = ellphi.coef_from_cov(centers[0], cov0)[0]
            qcoef = ellphi.coef_from_cov(centers[1], cov1)[0]
            expected_alpha = float(ellphi.tangency(pcoef, qcoef).t ** 2)
            # No regularization: A(mu) is a convex combination of SPD matrices,
            # hence always SPD.  cond(A(mu)) is bounded by the input condition
            # numbers, which double precision handles without geometric distortion.
            measured_alpha = float(
                solve_minimax_from_coefs(
                    np.stack([pcoef, qcoef]),
                    tol=1e-10,
                    max_iter=4000,
                ).alpha
            )
        except Exception:
            failed_cases += 1
            continue

        abs_err = abs(measured_alpha - expected_alpha)
        rel_err = abs_err / max(abs(expected_alpha), 1e-12)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

        if rel_err > rel_error_threshold or abs_err > abs_error_threshold:
            over_threshold_cases += 1

        worst_cases.append(
            {
                "case": float(idx),
                "condition_number": observed_cond,
                "expected_alpha": expected_alpha,
                "measured_alpha": measured_alpha,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )

    if rel_errors:
        max_rel_error = float(np.max(rel_errors))
        mean_rel_error = float(np.mean(rel_errors))
        max_abs_error = float(np.max(abs_errors))
        mean_abs_error = float(np.mean(abs_errors))
    else:
        max_rel_error = float("inf")
        mean_rel_error = float("inf")
        max_abs_error = float("inf")
        mean_abs_error = float("inf")

    if observed_conds:
        cond_min_observed = float(np.min(observed_conds))
        cond_max_observed = float(np.max(observed_conds))
    else:
        cond_min_observed = 0.0
        cond_max_observed = 0.0

    # Keep only the most error-prone samples for concise reporting.
    worst_cases.sort(key=lambda row: row["rel_error"], reverse=True)
    top_worst_cases = worst_cases[:5]

    evaluated_cases = len(rel_errors)
    passed = (
        evaluated_cases > 0
        and failed_cases == 0
        and cond_min_observed > min_condition_number
        and over_threshold_cases == 0
    )
    message = (
        "Conditioning stress benchmark passed."
        if passed
        else "Conditioning stress benchmark failed threshold checks."
    )

    return ConditioningStressCheck(
        status="ok" if passed else "failed",
        message=message,
        passed=passed,
        n_cases=n_cases,
        evaluated_cases=evaluated_cases,
        failed_cases=failed_cases,
        dimension=dimension,
        random_seed=random_seed,
        min_condition_number=min_condition_number,
        max_condition_number=max_condition_number,
        cond_min_observed=cond_min_observed,
        cond_max_observed=cond_max_observed,
        rel_error_threshold=rel_error_threshold,
        abs_error_threshold=abs_error_threshold,
        max_rel_error=max_rel_error,
        mean_rel_error=mean_rel_error,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        over_threshold_cases=over_threshold_cases,
        worst_cases=top_worst_cases,
    )


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path
