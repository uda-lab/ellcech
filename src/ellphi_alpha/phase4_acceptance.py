"""Phase-4 acceptance measurement helpers."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .filtration import FiltrationEntry, build_incremental_filtration
from .gudhi_bridge import to_gudhi_simplex_tree

DEFAULT_BASELINE_SEED = 20260306
DEFAULT_SIX_RINGS_SEED = 20260307


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
    h0_count_ours: int | None
    h0_count_gudhi: int | None
    h0_bottleneck: float | None
    max_abs_edge_alpha_diff: float | None

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
    h1_count: int | None
    long_lived_h1_count: int | None
    min_long_lived_h1: int
    lifetime_threshold: float
    top_h1_lifetimes: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_gudhi() -> bool:
    return importlib.util.find_spec("gudhi") is not None


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


def _extract_edge_map_from_simplex_tree(simplex_tree) -> dict[tuple[int, int], float]:
    edge_map: dict[tuple[int, int], float] = {}
    for simplex, alpha in simplex_tree.get_filtration():
        if len(simplex) != 2:
            continue
        i, j = sorted(int(v) for v in simplex)
        edge_map[(i, j)] = float(alpha)
    return edge_map


def _h0_intervals(simplex_tree) -> np.ndarray:
    simplex_tree.persistence(homology_coeff_field=2)
    intervals = simplex_tree.persistence_intervals_in_dimension(0)
    return np.asarray(intervals, dtype=float)


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


def run_baseline_barcode_agreement(
    *,
    n_points: int = 100,
    dimension: int = 2,
    random_seed: int = DEFAULT_BASELINE_SEED,
    edge_alpha_tol: float = 1e-6,
    bottleneck_tol: float = 1e-6,
) -> BaselineBarcodeAgreement:
    """Check H0 barcode agreement against GUDHI alpha complex on isotropic data."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

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

    if not has_gudhi():
        return BaselineBarcodeAgreement(
            status="skipped",
            message="gudhi is not installed; baseline comparison skipped.",
            passed=None,
            n_points=n_points,
            dimension=dimension,
            random_seed=random_seed,
            simplex_count=len(filtration),
            edge_count=len(our_edges),
            h0_count_ours=None,
            h0_count_gudhi=None,
            h0_bottleneck=None,
            max_abs_edge_alpha_diff=None,
        )

    import gudhi

    ours_tree = to_gudhi_simplex_tree(filtration)
    gudhi_tree = gudhi.AlphaComplex(points=centers).create_simplex_tree()

    ours_h0 = _h0_intervals(ours_tree)
    gudhi_h0 = _h0_intervals(gudhi_tree)
    h0_bottleneck = float(gudhi.bottleneck_distance(ours_h0.tolist(), gudhi_h0.tolist()))

    gudhi_edges = _extract_edge_map_from_simplex_tree(gudhi_tree)
    common_edges = set(our_edges).intersection(gudhi_edges)
    if common_edges:
        max_abs_edge_alpha_diff = max(
            abs(our_edges[e] - gudhi_edges[e]) for e in common_edges
        )
    else:
        max_abs_edge_alpha_diff = float("inf")

    passed = (
        len(ours_h0) == len(gudhi_h0)
        and h0_bottleneck <= bottleneck_tol
        and max_abs_edge_alpha_diff <= edge_alpha_tol
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
        h0_count_ours=int(len(ours_h0)),
        h0_count_gudhi=int(len(gudhi_h0)),
        h0_bottleneck=h0_bottleneck,
        max_abs_edge_alpha_diff=float(max_abs_edge_alpha_diff),
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
) -> SixRingsH1Check:
    """Check that a 6-rings dataset exposes at least 6 long-lived H1 classes."""
    points = _make_six_rings_points(
        points_per_ring=points_per_ring,
        ring_radius=ring_radius,
        spacing=spacing,
        noise=noise,
        random_seed=random_seed,
    )

    if not has_gudhi():
        return SixRingsH1Check(
            status="skipped",
            message="gudhi is not installed; 6-rings H1 check skipped.",
            passed=None,
            rings=6,
            points_per_ring=points_per_ring,
            n_points=int(points.shape[0]),
            random_seed=random_seed,
            h1_count=None,
            long_lived_h1_count=None,
            min_long_lived_h1=min_long_lived_h1,
            lifetime_threshold=lifetime_threshold,
            top_h1_lifetimes=None,
        )

    import gudhi

    simplex_tree = gudhi.AlphaComplex(points=points).create_simplex_tree()
    simplex_tree.persistence(homology_coeff_field=2)
    h1 = np.asarray(simplex_tree.persistence_intervals_in_dimension(1), dtype=float)
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
        h1_count=int(len(h1)),
        long_lived_h1_count=long_lived_count,
        min_long_lived_h1=min_long_lived_h1,
        lifetime_threshold=lifetime_threshold,
        top_h1_lifetimes=[float(v) for v in sorted_lifetimes[:10]],
    )


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path
