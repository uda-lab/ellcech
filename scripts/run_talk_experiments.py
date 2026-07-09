#!/usr/bin/env python
"""Talk experiments T1-T4 (docs/certified_pruning_sync.md, WP4).

Produces JSON raw reports, CSV summaries, and PDF+PNG figures under
``artifacts/talk/`` for the 2026-07 study-group talk on computing the
anisotropic Cech filtration:

- T1: the d=2, n=4 degenerate counterexample — exact tie detection, and
  restoration of unique supports / Booleanity under perturbation.
- T2: pruning efficacy tables (brute force vs G_L cliques vs pairwise vs
  face pruning; solves vs reuses; wall time).
- T3: Booleanity and Del^aniso statistics on small random instances, with
  the critical-value persistence check.
- T4: two-ring anisotropic toy figure (ellipse field, retained complex,
  persistence diagram with critical values marked).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ellphi_alpha import (
    act_support,
    booleanity_report,
    brute_force_filtration,
    certified_filtration,
    certify_support,
    critical_simplices,
    critical_value_check,
    solve_minimax,
)
from ellphi_alpha.backends.gudhi import GudhiBackend
from ellphi_alpha.phase4_acceptance import write_json_report

# The degenerate counterexample of
# ../paper-ellalpha/docs/early_gate_counterexample_note.md.
T1_CENTERS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
T1_MATRICES = np.stack(
    [
        np.diag([1.0, 2.0]),
        np.diag([1.0, 3.0]),
        np.diag([4.0, 1.0]),
        np.diag([5.0, 1.0]),
    ]
)
T1_PERTURB_SEED = 20260708

T2_BASE_SEED = 20260700
T2_QUANTILE = 20.0

T3_SEEDS = (101, 102, 103)
T3_QUANTILE = 50.0

T4_SEED = 20260713


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_fig(fig, outdir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _random_spd(rng: np.random.Generator, d: int, cond_max: float = 10.0) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = rng.uniform(1.0, cond_max, size=d)
    return (q * eigs) @ q.T


def _pairwise_alpha_percentile(matrices, centers, q: float) -> float:
    n = len(centers)
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(solve_minimax(matrices[[i, j]], centers[[i, j]]).alpha)
    return float(np.percentile(values, q))


def _ellipse_points(matrix, center, level: float, n: int = 200) -> np.ndarray:
    """Boundary of {x : (x-c)^T A (x-c) = level}."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    axes = eigenvectors @ np.diag(np.sqrt(level / eigenvalues))
    return center + circle @ axes.T


def _status_counts(info) -> dict[str, int]:
    return dict(Counter(i.status for i in info.values()))


def _persistence_by_dim(entries, max_hom_dim: int) -> dict[int, np.ndarray]:
    backend = GudhiBackend()
    tree = backend.simplex_tree_from_filtration(entries)
    return {
        dim: backend.persistence_intervals(tree, dimension=dim, homology_coeff_field=2)
        for dim in range(max_hom_dim + 1)
    }


# ---------------------------------------------------------------------------
# T1 — degenerate counterexample
# ---------------------------------------------------------------------------


def run_t1(outdir: Path) -> None:
    print("[talk-t1] degenerate counterexample")
    report: dict = {"centers": T1_CENTERS, "matrices": T1_MATRICES}
    rows = []
    for simplex in [(0, 1), (2, 3), (1, 2, 3), (0, 1, 2, 3)]:
        idx = np.asarray(simplex)
        result = solve_minimax(T1_MATRICES[idx], T1_CENTERS[idx])
        cert = certify_support(T1_MATRICES[idx], T1_CENTERS[idx], result)
        rows.append(
            {
                "simplex": simplex,
                "alpha": float(result.alpha),
                "circumcenter": [round(float(v), 12) for v in result.circumcenter],
                "raw_active_set": list(result.active_set),
                "certified_support": act_support(simplex, cert),
                "status": cert.status,
            }
        )
    report["exact"] = rows

    # Unperturbed sweep: ties must be detected, Booleanity must fail.
    exact_cf = certified_filtration(T1_MATRICES, T1_CENTERS, r_max=1.5, max_dim=3)
    exact_rep = booleanity_report(exact_cf.info, max_dim=3)
    report["exact_sweep"] = {
        "statuses": _status_counts(exact_cf.info),
        "booleanity_violations": len(exact_rep.violations),
        "violation_supports": [v.support for v in exact_rep.violations],
    }

    # Perturbed sweeps: generic structure must be restored.
    for delta in (1e-2, 1e-3):
        rng = np.random.default_rng(T1_PERTURB_SEED)
        perturbed = T1_CENTERS + delta * rng.standard_normal(T1_CENTERS.shape)
        cf = certified_filtration(T1_MATRICES, perturbed, r_max=1.5, max_dim=3)
        rep = booleanity_report(cf.info, max_dim=3)
        full = cf.info.get((0, 1, 2, 3))
        report[f"perturbed_delta_{delta:g}"] = {
            "statuses": _status_counts(cf.info),
            "booleanity_violations": len(rep.violations),
            "full_simplex_support": full.support if full is not None else None,
            "full_simplex_alpha": full.alpha if full is not None else None,
            "n_critical": len(critical_simplices(cf.info)),
            "n_simplices": len(cf.info),
        }

    write_json_report(outdir / "t1_report.json", _jsonable(report))
    _write_csv(
        outdir / "t1_summary.csv",
        ["simplex", "alpha", "raw_active_set", "certified_support", "status"],
        [
            {
                "simplex": " ".join(map(str, r["simplex"])),
                "alpha": f"{r['alpha']:.9f}",
                "raw_active_set": " ".join(map(str, r["raw_active_set"])),
                "certified_support": " ".join(map(str, r["certified_support"])),
                "status": r["status"],
            }
            for r in rows
        ],
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i in range(4):
        pts = _ellipse_points(T1_MATRICES[i], T1_CENTERS[i], 1.0)
        ax.plot(pts[:, 0], pts[:, 1], color=colors[i], label=f"site {i}")
        ax.plot(*T1_CENTERS[i], marker="x", color=colors[i])
    ax.plot(0.0, 0.0, marker="o", color="black", markersize=6, zorder=5)
    ax.annotate(r"$f_i = 1$ for all $i$", (0.0, 0.0), textcoords="offset points",
                xytext=(8, 8), fontsize=9)
    ax.set_aspect("equal")
    ax.set_title(r"Degenerate configuration: all $f_i = 1$ at the origin")
    ax.legend(fontsize=8, loc="upper right")
    _save_fig(fig, outdir, "t1_ellipses")
    print(f"[talk-t1] exact statuses={report['exact_sweep']['statuses']} "
          f"violations={report['exact_sweep']['booleanity_violations']}")


# ---------------------------------------------------------------------------
# T2 — pruning efficacy
# ---------------------------------------------------------------------------


def run_t2(outdir: Path) -> None:
    print("[talk-t2] pruning efficacy")
    rows = []
    for n in (20, 50, 100):
        max_dims = (2, 3) if n == 20 else (2,)
        rng = np.random.default_rng(T2_BASE_SEED + n)
        centers = rng.uniform(-1.0, 1.0, size=(n, 2))
        matrices = np.stack([_random_spd(rng, 2) for _ in range(n)])
        r_max = _pairwise_alpha_percentile(matrices, centers, T2_QUANTILE)
        for max_dim in max_dims:
            cf = certified_filtration(matrices, centers, r_max=r_max, max_dim=max_dim)
            stats = cf.stats
            row = {
                "n": n,
                "max_dim": max_dim,
                "r_max": f"{r_max:.6f}",
                "brute_force": stats.brute_force_candidates,
                "gl_cliques": stats.gl_clique_candidates,
                "after_pairwise": stats.after_pairwise_candidates,
                "after_face": stats.evaluated_candidates,
                "emitted": stats.emitted,
                "solves": stats.solves,
                "reuses": stats.reuses,
                "rejected": stats.rejected,
                "uncertain": stats.uncertain,
                "wall_time_s": f"{stats.wall_time_s:.3f}",
                "brute_wall_time_s": "",
            }
            if n == 20:
                t0 = time.perf_counter()
                brute = brute_force_filtration(
                    matrices, centers, r_max=r_max, max_dim=max_dim
                )
                brute_time = time.perf_counter() - t0
                row["brute_wall_time_s"] = f"{brute_time:.3f}"
                cert_set = {e.simplex for e in cf.entries}
                brute_set = {e.simplex for e in brute}
                assert cert_set == brute_set, "soundness violated in T2"
            rows.append(row)
            print(f"[talk-t2] n={n} max_dim={max_dim} "
                  f"brute={row['brute_force']} gl={row['gl_cliques']} "
                  f"pairwise={row['after_pairwise']} face={row['after_face']} "
                  f"solves={row['solves']} reuses={row['reuses']} "
                  f"time={row['wall_time_s']}s")

    fields = list(rows[0].keys())
    _write_csv(outdir / "t2_pruning_efficacy.csv", fields, rows)
    write_json_report(outdir / "t2_report.json", _jsonable({"rows": rows}))


# ---------------------------------------------------------------------------
# T3 — Booleanity + Del^aniso statistics
# ---------------------------------------------------------------------------


def run_t3(outdir: Path) -> None:
    print("[talk-t3] Booleanity and Del^aniso statistics")
    rows = []
    raw: dict = {}
    for n in (10, 15):
        for seed in T3_SEEDS:
            rng = np.random.default_rng(seed)
            centers = rng.uniform(-1.0, 1.0, size=(n, 2))
            matrices = np.stack([_random_spd(rng, 2) for _ in range(n)])
            r_max = _pairwise_alpha_percentile(matrices, centers, T3_QUANTILE)
            cf = certified_filtration(matrices, centers, r_max=r_max, max_dim=3)
            rep = booleanity_report(cf.info, max_dim=3)
            crit = critical_simplices(cf.info)
            critical_alphas = {cf.info[s].alpha for s in crit}
            intervals = _persistence_by_dim(cf.entries, max_hom_dim=2)
            cv = critical_value_check(intervals, critical_alphas, tol=1e-6)
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "r_max": f"{r_max:.6f}",
                    "n_simplices": len(cf.info),
                    "n_fibers": rep.n_fibers,
                    "booleanity_violations": len(rep.violations),
                    "n_critical": len(crit),
                    "checked_endpoints": cv.checked_endpoints,
                    "critical_value_violations": len(cv.violations),
                    "statuses": " ".join(
                        f"{k}:{v}" for k, v in sorted(_status_counts(cf.info).items())
                    ),
                }
            )
            raw[f"n{n}_seed{seed}"] = {
                "fiber_sizes": rep.fiber_sizes,
                "critical": crit,
                "critical_value_violations": cv.violations,
            }
            print(f"[talk-t3] n={n} seed={seed} simplices={len(cf.info)} "
                  f"fibers={rep.n_fibers} violations={len(rep.violations)} "
                  f"critical={len(crit)} cv_violations={len(cv.violations)}")

    fields = list(rows[0].keys())
    _write_csv(outdir / "t3_booleanity_summary.csv", fields, rows)
    write_json_report(outdir / "t3_report.json", _jsonable(raw))

    # Fiber-size histogram over all runs.
    sizes = Counter()
    for entry in raw.values():
        for size, count in entry["fiber_sizes"].items():
            sizes[int(size)] += count
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(list(sizes.keys()), list(sizes.values()), color="tab:blue")
    ax.set_xlabel("act fiber size (Boolean interval cardinality)")
    ax.set_ylabel("count")
    ax.set_title("Fiber sizes across T3 runs")
    _save_fig(fig, outdir, "t3_fiber_sizes")


# ---------------------------------------------------------------------------
# T4 — two-ring anisotropic toy
# ---------------------------------------------------------------------------


def _two_rings(n_per_ring: int, rng: np.random.Generator):
    """Two rings with tangentially aligned anisotropy."""
    ring_centers = np.array([[-1.5, 0.0], [1.5, 0.0]])
    centers = []
    matrices = []
    for rc in ring_centers:
        angles = np.linspace(0.0, 2.0 * np.pi, n_per_ring, endpoint=False)
        angles = angles + rng.uniform(0.0, 2.0 * np.pi)
        for theta in angles:
            radius = 1.0 + rng.normal(scale=0.02)
            point = rc + radius * np.array([np.cos(theta), np.sin(theta)])
            tangent = np.array([-np.sin(theta), np.cos(theta)])
            normal = np.array([np.cos(theta), np.sin(theta)])
            matrix = 1.0 * np.outer(tangent, tangent) + 4.0 * np.outer(normal, normal)
            centers.append(point)
            matrices.append(matrix)
    return np.stack(matrices), np.asarray(centers)


def run_t4(outdir: Path) -> None:
    print("[talk-t4] two-ring anisotropic toy")
    rng = np.random.default_rng(T4_SEED)
    matrices, centers = _two_rings(20, rng)
    r_max = _pairwise_alpha_percentile(matrices, centers, 15.0)
    cf = certified_filtration(matrices, centers, r_max=r_max, max_dim=2)
    rep = booleanity_report(cf.info, max_dim=2)
    crit = critical_simplices(cf.info)
    critical_alphas = sorted({cf.info[s].alpha for s in crit})
    intervals = _persistence_by_dim(cf.entries, max_hom_dim=1)
    cv = critical_value_check(intervals, critical_alphas, tol=1e-6)

    h1 = intervals[1]
    finite_h1 = h1[np.isfinite(h1[:, 1])] if h1.size else h1
    essential_h1 = h1[~np.isfinite(h1[:, 1])] if h1.size else h1
    lifetimes = (finite_h1[:, 1] - finite_h1[:, 0]) if finite_h1.size else np.array([])
    top = np.sort(lifetimes)[::-1][:6]
    report = {
        "n_points": len(centers),
        "r_max": r_max,
        "stats": cf.stats._asdict(),
        "booleanity_violations": len(rep.violations),
        "statuses": _status_counts(cf.info),
        "n_critical": len(crit),
        "critical_value_check": {
            "checked": cv.checked_endpoints,
            "violations": cv.violations,
        },
        "h1_count": int(len(h1)),
        # The two ring classes are expected to be essential at r_max: born
        # when each ring closes, not yet filled by triangles.
        "essential_h1_count": int(len(essential_h1)),
        "essential_h1_births": essential_h1[:, 0] if essential_h1.size else [],
        "top_finite_h1_lifetimes": top,
    }
    write_json_report(outdir / "t4_report.json", _jsonable(report))

    # Complex figure: ellipse field + retained complex.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for simplex, info in cf.info.items():
        if len(simplex) == 3:
            tri = centers[list(simplex)]
            ax.fill(tri[:, 0], tri[:, 1], color="tab:blue", alpha=0.15, lw=0)
    for simplex in cf.info:
        if len(simplex) == 2:
            seg = centers[list(simplex)]
            ax.plot(seg[:, 0], seg[:, 1], color="tab:blue", lw=0.8, alpha=0.7)
    for i in range(len(centers)):
        pts = _ellipse_points(matrices[i], centers[i], r_max)
        ax.plot(pts[:, 0], pts[:, 1], color="tab:gray", lw=0.5, alpha=0.5)
    ax.plot(centers[:, 0], centers[:, 1], "k.", markersize=4)
    ax.set_aspect("equal")
    ax.set_title(
        rf"Two anisotropic rings: retained complex at $r_{{\max}}={r_max:.3f}$"
        f" ({cf.stats.emitted} simplices, {cf.stats.solves} solves, "
        f"{cf.stats.reuses} reuses)"
    )
    _save_fig(fig, outdir, "t4_two_rings")

    # Persistence diagram with critical values marked.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    finite_max = r_max * 1.05
    for dim, color, marker in ((0, "tab:orange", "o"), (1, "tab:blue", "s")):
        arr = intervals[dim]
        if not arr.size:
            continue
        deaths = np.where(np.isfinite(arr[:, 1]), arr[:, 1], finite_max)
        ax.scatter(arr[:, 0], deaths, s=18, c=color, marker=marker,
                   label=f"$H_{dim}$", alpha=0.8, zorder=3)
    ax.plot([0, finite_max], [0, finite_max], "k-", lw=0.8)
    ax.axhline(finite_max, color="gray", lw=0.6, ls="--")
    for value in critical_alphas:
        ax.plot([value], [0], marker="|", color="tab:red", markersize=8, zorder=4)
    ax.plot([], [], marker="|", color="tab:red", ls="none", label="critical values")
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("Persistence of the certified filtration")
    ax.legend(fontsize=8, loc="lower right")
    _save_fig(fig, outdir, "t4_persistence")
    print(f"[talk-t4] r_max={r_max:.4f} simplices={cf.stats.emitted} "
          f"violations={len(rep.violations)} critical={len(crit)} "
          f"cv_violations={len(cv.violations)} "
          f"essential_h1={len(essential_h1)} top_finite_h1={[round(v,3) for v in top]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "talk",
    )
    parser.add_argument(
        "--only",
        choices=["t1", "t2", "t3", "t4"],
        nargs="*",
        default=None,
        help="Run only the listed experiments.",
    )
    args = parser.parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only) if args.only else {"t1", "t2", "t3", "t4"}
    t0 = time.perf_counter()
    if "t1" in selected:
        run_t1(outdir)
    if "t2" in selected:
        run_t2(outdir)
    if "t3" in selected:
        run_t3(outdir)
    if "t4" in selected:
        run_t4(outdir)
    print(f"[talk] done in {time.perf_counter() - t0:.1f}s -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
