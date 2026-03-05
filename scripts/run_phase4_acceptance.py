#!/usr/bin/env python
"""Run Phase-4 acceptance measurements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ellphi_alpha.phase4_acceptance import (
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
        default=20260306,
        help="Random seed for baseline agreement check.",
    )
    parser.add_argument(
        "--rings-seed",
        type=int,
        default=20260307,
        help="Random seed for 6-rings H1 check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline = run_baseline_barcode_agreement(
        n_points=100,
        dimension=2,
        random_seed=args.baseline_seed,
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
        random_seed=args.rings_seed,
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

    if baseline.status == "failed" or six_rings.status == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
