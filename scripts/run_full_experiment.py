"""Run the 5 minimax solver numerical experiments.

Usage
-----
    # Run all experiments:
    python scripts/run_full_experiment.py --experiments 1 2 3 4 5 --seed 42

    # Run only experiment 1 with specific methods:
    python scripts/run_full_experiment.py --experiments 1 --methods fw+bisect fw+brentq+newton

    # Run experiment 2 (ill-conditioning):
    python scripts/run_full_experiment.py --experiments 2 --seed 42

Experiments
-----------
1. Basic speed/precision: k in {2..10}, d in {2,3,5,10}
2. Ill-conditioning:      k in {3,4,5}, d=3, cond in {2..10000}
3. Near-degenerate:        k in {4,6,8}, n_far in {1,2,3}
4. FW->Newton handoff:    k=5, d=3, fw_tol sweep
5. Brentq vs bisect:      k in {3..8}, d=3, line search eval count
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from ellphi_alpha.benchmarks import (
    ExperimentConfig,
    ExperimentResult,
    bootstrap_ci,
    run_experiment,
    summarize_experiment,
    wilson_interval,
)
from ellphi_alpha.minimax import _VALID_METHODS, solve_minimax
from ellphi_alpha.benchmarks import make_random_simplex

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "experiments",
)

ALL_METHODS = list(_VALID_METHODS)


def _ensure_artifacts_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _save_results(exp_id: str, results: list[ExperimentResult]):
    _ensure_artifacts_dir()
    raw_path = os.path.join(ARTIFACTS_DIR, f"{exp_id}_raw.json")
    data = [r.to_dict() for r in results]
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved raw results to {raw_path}")

    # Summary CSV
    summary = summarize_experiment(results)
    csv_path = os.path.join(ARTIFACTS_DIR, f"{exp_id}_summary.csv")
    with open(csv_path, "w") as f:
        f.write("key,n,converge_rate,rel_error_median,rel_error_max,time_median,time_mean,iter_median\n")
        for key, s in summary.items():
            f.write(
                f"{key},{s['n']},{s['converge_rate']:.4f},"
                f"{s['rel_error_median']:.2e},{s['rel_error_max']:.2e},"
                f"{s['time_median']:.6f},{s['time_mean']:.6f},{s['iter_median']:.0f}\n"
            )
    print(f"  Saved summary to {csv_path}")


def _print_summary(results: list[ExperimentResult]):
    summary = summarize_experiment(results)
    print(f"\n  {'key':<45} {'n':>4} {'cvg%':>6} {'err_med':>10} {'err_max':>10} {'t_med':>10} {'iters':>6}")
    print(f"  {'-'*45} {'-'*4} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for key, s in summary.items():
        err_med = f"{s['rel_error_median']:.2e}" if np.isfinite(s["rel_error_median"]) else "N/A"
        err_max = f"{s['rel_error_max']:.2e}" if np.isfinite(s["rel_error_max"]) else "N/A"
        print(
            f"  {key:<45} {s['n']:>4} {s['converge_rate']*100:>5.1f}% "
            f"{err_med:>10} {err_max:>10} "
            f"{s['time_median']:>10.6f} {s['iter_median']:>6.0f}"
        )


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------


def run_exp1(methods: list[str], seed: int):
    """Exp 1: Basic speed/precision across k and d."""
    print("\n" + "=" * 70)
    print("  Experiment 1: Basic speed/precision")
    print("=" * 70)

    config = ExperimentConfig(
        experiment_id="exp1",
        k_values=[2, 3, 4, 5, 6, 8, 10],
        d_values=[2, 3, 5, 10],
        methods=methods,
        n_instances=50,
        n_timing_repeats=3,
        generator="random",
        generator_kwargs={"cond_bound": 10.0},
        seed=seed,
    )
    t0 = time.perf_counter()
    results = run_experiment(config)
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s ({len(results)} rows)")
    _print_summary(results)
    _save_results("exp1", results)
    return results


def run_exp2(methods: list[str], seed: int):
    """Exp 2: Ill-conditioning study."""
    print("\n" + "=" * 70)
    print("  Experiment 2: Ill-conditioning")
    print("=" * 70)

    all_results: list[ExperimentResult] = []
    for cond_target in [2, 10, 100, 1000, 10000]:
        config = ExperimentConfig(
            experiment_id="exp2",
            k_values=[3, 4, 5],
            d_values=[3],
            methods=methods,
            n_instances=200,
            n_timing_repeats=1,
            generator="ill_conditioned",
            generator_kwargs={"cond_target": cond_target},
            seed=seed + cond_target,
        )
        results = run_experiment(config)
        # Tag with cond_target
        for r in results:
            if r.metadata is None:
                r.metadata = {}
            r.metadata["cond_target"] = cond_target
        all_results.extend(results)
        n_fail = sum(1 for r in results if not r.converged)
        fail_rate, ci_lo, ci_hi = wilson_interval(n_fail, len(results))
        print(f"  cond={cond_target:>6}: {len(results)} runs, "
              f"fail_rate={fail_rate:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    _print_summary(all_results)
    _save_results("exp2", all_results)
    return all_results


def run_exp3(methods: list[str], seed: int):
    """Exp 3: Near-degenerate (far vertices)."""
    print("\n" + "=" * 70)
    print("  Experiment 3: Near-degenerate simplices")
    print("=" * 70)

    all_results: list[ExperimentResult] = []
    for n_far in [1, 2, 3]:
        for k in [4, 6, 8]:
            config = ExperimentConfig(
                experiment_id="exp3",
                k_values=[k],
                d_values=[3],
                methods=methods,
                n_instances=200,
                n_timing_repeats=1,
                generator="degenerate",
                generator_kwargs={"n_far": n_far},
                seed=seed + n_far * 100 + k,
            )
            results = run_experiment(config)
            for r in results:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata["n_far"] = n_far
            all_results.extend(results)

    _print_summary(all_results)
    _save_results("exp3", all_results)
    return all_results


def run_exp4(methods: list[str], seed: int):
    """Exp 4: FW→Newton handoff tolerance sweep."""
    print("\n" + "=" * 70)
    print("  Experiment 4: FW→Newton handoff (fw_tol sweep)")
    print("=" * 70)

    # Only methods with Newton polishing make sense
    newton_methods = [m for m in methods if "newton" in m]
    if not newton_methods:
        print("  Skipped: no Newton methods selected.")
        return []

    fw_tols = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-12]
    all_results: list[ExperimentResult] = []
    rng = np.random.default_rng(seed)

    for fw_tol in fw_tols:
        config = ExperimentConfig(
            experiment_id="exp4",
            k_values=[5],
            d_values=[3],
            methods=newton_methods,
            n_instances=100,
            n_timing_repeats=3,
            generator="random",
            generator_kwargs={"cond_bound": 10.0},
            seed=seed + int(-np.log10(fw_tol)),
            solver_kwargs={"tol": fw_tol},
        )
        results = run_experiment(config)
        for r in results:
            if r.metadata is None:
                r.metadata = {}
            r.metadata["fw_tol"] = fw_tol
        all_results.extend(results)

        # Quick summary per tol
        errors = np.array([r.rel_error for r in results if np.isfinite(r.rel_error)])
        iters = np.array([r.n_iter for r in results])
        med_err = f"{np.median(errors):.2e}" if len(errors) > 0 else "N/A"
        print(f"  fw_tol={fw_tol:.0e}: median_err={med_err}, median_iter={np.median(iters):.0f}")

    _save_results("exp4", all_results)
    return all_results


def run_exp5(methods: list[str], seed: int):
    """Exp 5: Brentq vs bisect line search comparison."""
    print("\n" + "=" * 70)
    print("  Experiment 5: Brentq vs bisect line search")
    print("=" * 70)

    # Focus on the FW-only methods
    ls_methods = [m for m in methods if m in ("fw+bisect", "fw+brentq")]
    if len(ls_methods) < 2:
        # Include both if not already
        ls_methods = ["fw+bisect", "fw+brentq"]

    config = ExperimentConfig(
        experiment_id="exp5",
        k_values=[3, 4, 5, 6, 7, 8],
        d_values=[3],
        methods=ls_methods,
        n_instances=200,
        n_timing_repeats=3,
        generator="random",
        generator_kwargs={"cond_bound": 10.0},
        seed=seed,
    )
    t0 = time.perf_counter()
    results = run_experiment(config)
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s ({len(results)} rows)")
    _print_summary(results)
    _save_results("exp5", results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: run_exp1,
    2: run_exp2,
    3: run_exp3,
    4: run_exp4,
    5: run_exp5,
}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run minimax solver numerical experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Which experiments to run (default: all).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Methods to test (default: all). Choices: {ALL_METHODS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    args = parser.parse_args(argv)

    methods = args.methods or ALL_METHODS
    for m in methods:
        if m not in _VALID_METHODS:
            parser.error(f"Unknown method: {m!r}")

    print(f"Methods: {methods}")
    print(f"Seed: {args.seed}")

    for exp_num in args.experiments:
        EXPERIMENTS[exp_num](methods, args.seed)

    print("\nAll done.")


if __name__ == "__main__":
    main()
