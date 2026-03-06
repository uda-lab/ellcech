"""CLI benchmark for minimax solver methods.

Usage
-----
# Benchmark all methods on k=5 vertices in d=3 dimensions, 30 repetitions:
    python scripts/benchmark_minimax.py --k 5 --d 3 --repeat 30

# Vary k from 2 to 8, fixed d=4:
    python scripts/benchmark_minimax.py --k-range 2 8 --d 4

# Only specific methods:
    python scripts/benchmark_minimax.py --methods fw+bisect fw+newton --k 4 --d 3

# Set random seed for reproducibility:
    python scripts/benchmark_minimax.py --seed 42 --k 6 --d 4 --repeat 20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ellphi_alpha.benchmarks import BenchmarkResult, benchmark_solve_minimax, make_random_simplex
from ellphi_alpha.minimax import _VALID_METHODS


def _fmt_time(t: float) -> str:
    if t < 1e-6:
        return f"{t * 1e9:7.1f} ns"
    if t < 1e-3:
        return f"{t * 1e6:7.1f} us"
    if t < 1.0:
        return f"{t * 1e3:7.1f} ms"
    return f"{t:7.3f} s "


def _print_results(results: list[BenchmarkResult], k: int, d: int) -> None:
    header = f"k={k}, d={d}"
    print(f"\n{'='*62}")
    print(f"  {header}")
    print(f"{'='*62}")
    print(f"  {'method':<16} {'alpha':>14} {'iters':>6}  {'time (mean)':>11}  {'± std':>10}  {'rel_err':>12}  cvg")
    print(f"  {'-'*16} {'-'*14} {'-'*6}  {'-'*11}  {'-'*10}  {'-'*12}  ---")
    for r in results:
        rel_err = f"{r.rel_error:.3e}" if not (r.rel_error != r.rel_error) else "  (ref)     "
        print(
            f"  {r.method:<16} {r.alpha:>14.8g} {r.n_iter:>6}  "
            f"{_fmt_time(r.time_mean_s)}  ±{_fmt_time(r.time_std_s)}  "
            f"{rel_err:>12}  {'Y' if r.converged else 'N'}"
        )


def _run_single(
    k: int,
    d: int,
    methods: list[str],
    repeat: int,
    seed: int,
    cond_bound: float,
) -> None:
    rng = np.random.default_rng(seed)
    matrices, centers = make_random_simplex(k, d, rng=rng, cond_bound=cond_bound)
    results = benchmark_solve_minimax(
        matrices,
        centers,
        methods=methods,
        n_repeat=repeat,
    )
    _print_results(results, k, d)


def _run_k_range(
    k_min: int,
    k_max: int,
    d: int,
    methods: list[str],
    repeat: int,
    seed: int,
    cond_bound: float,
) -> None:
    rng = np.random.default_rng(seed)
    for k in range(k_min, k_max + 1):
        matrices, centers = make_random_simplex(k, d, rng=rng, cond_bound=cond_bound)
        results = benchmark_solve_minimax(
            matrices,
            centers,
            methods=methods,
            n_repeat=repeat,
        )
        _print_results(results, k, d)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ellphi-alpha minimax solver methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Simplex geometry
    parser.add_argument("--k", type=int, default=None,
                        help="Number of simplex vertices (single run).")
    parser.add_argument("--k-range", type=int, nargs=2, metavar=("K_MIN", "K_MAX"),
                        default=None,
                        help="Range of k values to sweep (inclusive).")
    parser.add_argument("--d", type=int, default=3,
                        help="Ambient dimension (default: 3).")
    parser.add_argument("--cond-bound", type=float, default=10.0,
                        help="Max condition number for generated matrices (default: 10).")

    # Solver options
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=list(_VALID_METHODS) + ["all"],
                        help="Methods to benchmark (default: all).")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Number of timing repetitions per method (default: 20).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")

    args = parser.parse_args(argv)

    methods = args.methods
    if methods is None or methods == ["all"]:
        methods = list(_VALID_METHODS)

    if args.k_range is not None:
        k_min, k_max = args.k_range
        _run_k_range(k_min, k_max, args.d, methods, args.repeat, args.seed, args.cond_bound)
    elif args.k is not None:
        _run_single(args.k, args.d, methods, args.repeat, args.seed, args.cond_bound)
    else:
        # Default: sweep k=2..6, d=3
        print("No --k or --k-range specified. Running default sweep k=2..6, d=3.")
        _run_k_range(2, 6, args.d, methods, args.repeat, args.seed, args.cond_bound)


if __name__ == "__main__":
    main()
