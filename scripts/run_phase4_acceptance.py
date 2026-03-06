#!/usr/bin/env python
"""Run Phase-4 acceptance measurements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ellphi_alpha.phase4_acceptance import (
    run_conditioning_stress_check,
    run_baseline_barcode_agreement,
    run_six_rings_h1_check,
    write_json_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase4_acceptance"),
        help="Directory where JSON reports are written.",
    )
    parser.add_argument(
        "--baseline-seed",
        type=int,
        default=None,
        help="Random seed for baseline agreement check (default: 20260306).",
    )
    parser.add_argument(
        "--rings-seed",
        type=int,
        default=None,
        help="Random seed for 6-rings H1 check (default: 20260307).",
    )
    parser.add_argument(
        "--conditioning-seed",
        type=int,
        default=None,
        help="Random seed for high-conditioning stress check (default: 20260308).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "[Deprecated] Set the same random seed for all three checks. "
            "Use --baseline-seed, --rings-seed, --conditioning-seed instead. "
            "Specific seed flags take precedence over --seed when both are given."
        ),
    )
    return parser.parse_args()


_DEFAULT_BASELINE_SEED = 20260306
_DEFAULT_RINGS_SEED = 20260307
_DEFAULT_CONDITIONING_SEED = 20260308


def main() -> int:
    args = parse_args()

    # Resolve seeds: specific flag > deprecated --seed > hard-coded default.
    baseline_seed = args.baseline_seed if args.baseline_seed is not None else (
        args.seed if args.seed is not None else _DEFAULT_BASELINE_SEED
    )
    rings_seed = args.rings_seed if args.rings_seed is not None else (
        args.seed if args.seed is not None else _DEFAULT_RINGS_SEED
    )
    conditioning_seed = args.conditioning_seed if args.conditioning_seed is not None else (
        args.seed if args.seed is not None else _DEFAULT_CONDITIONING_SEED
    )

    baseline = run_baseline_barcode_agreement(
        n_points=100,
        dimension=2,
        random_seed=baseline_seed,
    )
    baseline_path = write_json_report(
        args.output_dir / "baseline_barcode_agreement.json",
        baseline.to_dict(),
    )

    print(
        "[phase4-baseline] "
        f"status={baseline.status} passed={baseline.passed} "
        f"h0_bottleneck={baseline.h0_bottleneck}"
    )
    print(f"[phase4-baseline] report={baseline_path}")

    six_rings = run_six_rings_h1_check(
        points_per_ring=24,
        random_seed=rings_seed,
    )
    six_rings_path = write_json_report(
        args.output_dir / "six_rings_h1_check.json",
        six_rings.to_dict(),
    )
    print(
        "[phase4-6rings] "
        f"status={six_rings.status} passed={six_rings.passed} "
        f"long_lived_h1={six_rings.long_lived_h1_count}"
    )
    print(f"[phase4-6rings] report={six_rings_path}")

    conditioning = run_conditioning_stress_check(
        n_cases=32,
        dimension=2,
        min_condition_number=1e6,
        max_condition_number=1e7,
        random_seed=conditioning_seed,
        rel_error_threshold=1e-4,
        abs_error_threshold=1e-3,
    )
    conditioning_path = write_json_report(
        args.output_dir / "conditioning_stress_check.json",
        conditioning.to_dict(),
    )
    print(
        "[phase4-conditioning] "
        f"status={conditioning.status} passed={conditioning.passed} "
        f"max_rel_error={conditioning.max_rel_error:.3e} "
        f"cond_range=[{conditioning.cond_min_observed:.3e}, "
        f"{conditioning.cond_max_observed:.3e}]"
    )
    print(f"[phase4-conditioning] report={conditioning_path}")

    summary_path = write_json_report(
        args.output_dir / "phase4_acceptance_summary.json",
        {
            "baseline": baseline.to_dict(),
            "six_rings": six_rings.to_dict(),
            "conditioning": conditioning.to_dict(),
        },
    )
    print(f"[phase4-summary] report={summary_path}")

    if (
        baseline.status == "failed"
        or six_rings.status == "failed"
        or conditioning.status == "failed"
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
