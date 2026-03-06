"""Benchmarking utilities for minimax solver methods.

Usage example::

    from ellphi_alpha.benchmarks import benchmark_solve_minimax, make_random_simplex

    rng = np.random.default_rng(42)
    matrices, centers = make_random_simplex(k=5, d=3, rng=rng)
    results = benchmark_solve_minimax(matrices, centers, n_repeat=20)

    for r in results:
        print(r)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from ellphi_alpha.minimax import MethodName, MinimaxResult, _VALID_METHODS, solve_minimax

__all__ = [
    "BenchmarkResult",
    "ExperimentConfig",
    "ExperimentResult",
    "benchmark_solve_minimax",
    "bootstrap_ci",
    "make_degenerate_simplex",
    "make_ill_conditioned_simplex",
    "make_random_simplex",
    "run_experiment",
    "summarize_experiment",
    "wilson_interval",
]


class BenchmarkResult(NamedTuple):
    """Single-method benchmark summary.

    Attributes:
        method: Solver method name.
        alpha: Computed filtration value.
        converged: Whether the solver reported convergence.
        n_iter: Number of iterations (FW steps + Newton steps for fw+newton).
        time_mean_s: Mean wall-clock time in seconds over n_repeat runs.
        time_std_s: Std of wall-clock time.
        rel_error: |alpha - alpha_ref| / max(|alpha_ref|, 1e-15).
                   NaN if this is the reference method or alpha_ref is not set.
    """

    method: str
    alpha: float
    converged: bool
    n_iter: int
    time_mean_s: float
    time_std_s: float
    rel_error: float


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------


def make_random_simplex(
    k: int,
    d: int,
    *,
    rng: np.random.Generator | None = None,
    cond_bound: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random simplex (SPD matrices + centers) for benchmarking.

    Args:
        k: Number of vertices.
        d: Ambient dimension.
        rng: Random number generator.  Defaults to ``np.random.default_rng(0)``.
        cond_bound: Each A_i is constructed so cond(A_i) <= cond_bound.

    Returns:
        (matrices, centers): shapes (k, d, d) and (k, d).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    centers = rng.standard_normal((k, d))
    matrices = np.empty((k, d, d))
    for i in range(k):
        U = np.linalg.qr(rng.standard_normal((d, d)))[0]
        eigvals = rng.uniform(1.0, cond_bound, size=d)
        matrices[i] = U @ np.diag(eigvals) @ U.T

    return matrices, centers


def make_ill_conditioned_simplex(
    k: int,
    d: int,
    *,
    rng: np.random.Generator | None = None,
    cond_target: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simplex with matrices having condition number ~cond_target.

    Args:
        k: Number of vertices.
        d: Ambient dimension.
        rng: Random number generator.
        cond_target: Target condition number for each A_i.

    Returns:
        (matrices, centers): shapes (k, d, d) and (k, d).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    centers = rng.standard_normal((k, d))
    matrices = np.empty((k, d, d))
    for i in range(k):
        U = np.linalg.qr(rng.standard_normal((d, d)))[0]
        # Spread eigenvalues logarithmically from 1 to cond_target
        eigvals = np.logspace(0, np.log10(cond_target), d)
        # Shuffle to avoid systematic ordering
        rng.shuffle(eigvals)
        matrices[i] = U @ np.diag(eigvals) @ U.T

    return matrices, centers


def make_degenerate_simplex(
    k: int,
    d: int,
    *,
    rng: np.random.Generator | None = None,
    n_far: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simplex with n_far distant vertices (near-degenerate active set).

    The first (k - n_far) vertices are clustered near the origin;
    the last n_far vertices are pushed far away so their weights tend to zero.

    Args:
        k: Number of vertices (must be > n_far).
        d: Ambient dimension.
        rng: Random number generator.
        n_far: Number of far-away vertices.

    Returns:
        (matrices, centers): shapes (k, d, d) and (k, d).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_far >= k:
        raise ValueError(f"n_far={n_far} must be < k={k}")

    centers = np.empty((k, d))
    # Near vertices: clustered around origin
    centers[: k - n_far] = rng.standard_normal((k - n_far, d)) * 0.5
    # Far vertices: offset by 10-20 units
    centers[k - n_far :] = rng.standard_normal((n_far, d)) * 0.5 + rng.uniform(
        10.0, 20.0, size=(n_far, d)
    )

    matrices = np.empty((k, d, d))
    for i in range(k):
        U = np.linalg.qr(rng.standard_normal((d, d)))[0]
        eigvals = rng.uniform(1.0, 5.0, size=d)
        matrices[i] = U @ np.diag(eigvals) @ U.T

    return matrices, centers


# ---------------------------------------------------------------------------
# Experiment data classes
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Result from a single solver run on a single instance."""

    experiment_id: str
    instance_id: int
    k: int
    d: int
    seed: int
    method: str
    alpha: float
    alpha_ref: float
    rel_error: float
    converged: bool
    n_iter: int
    time_seconds: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "instance_id": self.instance_id,
            "k": self.k,
            "d": self.d,
            "seed": self.seed,
            "method": self.method,
            "alpha": self.alpha,
            "alpha_ref": self.alpha_ref,
            "rel_error": self.rel_error,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "time_seconds": self.time_seconds,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class ExperimentConfig:
    """Configuration for a batch experiment."""

    experiment_id: str
    k_values: list[int]
    d_values: list[int]
    methods: list[str]
    n_instances: int
    n_timing_repeats: int = 1
    generator: str = "random"
    generator_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    solver_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the median.

    Returns:
        (median, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    boot_medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_medians[i] = float(np.median(sample))

    alpha = (1.0 - ci) / 2.0
    return (
        float(np.median(data)),
        float(np.percentile(boot_medians, 100 * alpha)),
        float(np.percentile(boot_medians, 100 * (1.0 - alpha))),
    )


def wilson_interval(
    n_success: int, n_total: int, *, z: float = 1.96
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns:
        (p_hat, ci_low, ci_high)
    """
    if n_total == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = n_success / n_total
    denom = 1.0 + z**2 / n_total
    center = (p + z**2 / (2.0 * n_total)) / denom
    half_width = z * np.sqrt(p * (1.0 - p) / n_total + z**2 / (4.0 * n_total**2)) / denom
    return (p, max(0.0, center - half_width), min(1.0, center + half_width))


def summarize_experiment(
    results: list[ExperimentResult],
) -> dict[str, dict[str, Any]]:
    """Aggregate experiment results by (method, k, d) factor levels.

    Returns:
        Dict keyed by ``"method:k:d"`` with summary statistics.
    """
    from collections import defaultdict

    groups: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        key = f"{r.method}:k={r.k}:d={r.d}"
        groups[key].append(r)

    summary = {}
    for key, group in sorted(groups.items()):
        errors = np.array([r.rel_error for r in group if np.isfinite(r.rel_error)])
        times = np.array([r.time_seconds for r in group])
        iters = np.array([r.n_iter for r in group])
        n_converged = sum(1 for r in group if r.converged)
        n_total = len(group)

        summary[key] = {
            "n": n_total,
            "converge_rate": n_converged / n_total if n_total > 0 else 0.0,
            "rel_error_median": float(np.median(errors)) if len(errors) > 0 else float("nan"),
            "rel_error_max": float(np.max(errors)) if len(errors) > 0 else float("nan"),
            "time_median": float(np.median(times)),
            "time_mean": float(np.mean(times)),
            "iter_median": float(np.median(iters)),
        }

    return summary


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

_GENERATORS = {
    "random": make_random_simplex,
    "ill_conditioned": make_ill_conditioned_simplex,
    "degenerate": make_degenerate_simplex,
}


def _compute_reference_alpha(
    matrices: np.ndarray,
    centers: np.ndarray,
) -> float:
    """Compute reference alpha as max of tight fw+bisect and scipy-slsqp."""
    a1 = solve_minimax(
        matrices, centers, method="fw+bisect", tol=1e-14, max_iter=10000,
    ).alpha
    a2 = solve_minimax(
        matrices, centers, method="scipy-slsqp",
    ).alpha
    return max(a1, a2)


def run_experiment(config: ExperimentConfig) -> list[ExperimentResult]:
    """Run a full experiment according to *config*.

    Returns:
        List of ExperimentResult for every (instance, method) combination.
    """
    gen_func = _GENERATORS.get(config.generator)
    if gen_func is None:
        raise ValueError(
            f"Unknown generator {config.generator!r}. "
            f"Valid: {list(_GENERATORS.keys())}"
        )

    results: list[ExperimentResult] = []
    rng = np.random.default_rng(config.seed)
    instance_id = 0

    for k in config.k_values:
        for d in config.d_values:
            for inst in range(config.n_instances):
                inst_seed = int(rng.integers(0, 2**31))
                inst_rng = np.random.default_rng(inst_seed)

                gen_kwargs = dict(config.generator_kwargs)
                # Handle generators with different kwarg names
                if config.generator == "ill_conditioned":
                    gen_kwargs.setdefault("cond_target", gen_kwargs.pop("cond_bound", 10.0))
                matrices, centers = gen_func(k, d, rng=inst_rng, **gen_kwargs)

                alpha_ref = _compute_reference_alpha(matrices, centers)

                for method in config.methods:
                    # Timing: best of n_timing_repeats
                    best_time = float("inf")
                    last_res: MinimaxResult | None = None
                    for _ in range(config.n_timing_repeats):
                        t0 = time.perf_counter()
                        last_res = solve_minimax(
                            matrices, centers, method=method,
                            **config.solver_kwargs,
                        )
                        elapsed = time.perf_counter() - t0
                        best_time = min(best_time, elapsed)

                    assert last_res is not None
                    rel_err = (
                        abs(last_res.alpha - alpha_ref) / max(abs(alpha_ref), 1e-15)
                    )

                    results.append(ExperimentResult(
                        experiment_id=config.experiment_id,
                        instance_id=instance_id,
                        k=k,
                        d=d,
                        seed=inst_seed,
                        method=method,
                        alpha=last_res.alpha,
                        alpha_ref=alpha_ref,
                        rel_error=rel_err,
                        converged=last_res.converged,
                        n_iter=last_res.n_iter,
                        time_seconds=best_time,
                        metadata=last_res.metadata,
                    ))

                instance_id += 1

    return results


# ---------------------------------------------------------------------------
# Legacy benchmark API
# ---------------------------------------------------------------------------


def benchmark_solve_minimax(
    matrices: np.ndarray,
    centers: np.ndarray,
    *,
    methods: list[str] | None = None,
    n_repeat: int = 10,
    reference_method: str = "fw+bisect",
    solver_kwargs: dict | None = None,
) -> list[BenchmarkResult]:
    """Benchmark all (or selected) minimax solver methods on a single simplex.

    For each method the solver is run ``n_repeat`` times and the mean/std of
    wall-clock time is reported.  Relative error is computed against the
    ``reference_method`` result.

    Args:
        matrices: SPD matrices A_i, shape (k, d, d).
        centers: Points x_bar_i, shape (k, d).
        methods: List of method names to benchmark.  Defaults to all valid methods.
        n_repeat: Number of timing repetitions per method.
        reference_method: Method used as the ground-truth alpha for rel_error.
        solver_kwargs: Extra kwargs forwarded to ``solve_minimax`` for every method
            (e.g. ``{"tol": 1e-12}``).

    Returns:
        List of BenchmarkResult, one per method.
    """
    if methods is None:
        methods = list(_VALID_METHODS)
    if solver_kwargs is None:
        solver_kwargs = {}

    # First compute reference alpha
    ref_result: MinimaxResult = solve_minimax(
        matrices, centers, method=reference_method, **solver_kwargs
    )
    alpha_ref = ref_result.alpha

    results: list[BenchmarkResult] = []

    for method in methods:
        times: list[float] = []
        last: MinimaxResult | None = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            last = solve_minimax(matrices, centers, method=method, **solver_kwargs)
            times.append(time.perf_counter() - t0)

        assert last is not None
        alpha_m = last.alpha
        rel_error = (
            float("nan")
            if method == reference_method
            else abs(alpha_m - alpha_ref) / max(abs(alpha_ref), 1e-15)
        )

        results.append(
            BenchmarkResult(
                method=method,
                alpha=alpha_m,
                converged=last.converged,
                n_iter=last.n_iter,
                time_mean_s=float(np.mean(times)),
                time_std_s=float(np.std(times)),
                rel_error=rel_error,
            )
        )

    return results
