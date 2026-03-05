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


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path
