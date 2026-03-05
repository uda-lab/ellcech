#!/usr/bin/env python
"""Run Phase-4 acceptance measurements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ellphi_alpha.phase4_acceptance import (
    run_baseline_barcode_agreement,
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
    parser.add_argument("--seed", type=int, default=20260306, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline = run_baseline_barcode_agreement(
        n_points=100,
        dimension=2,
        random_seed=args.seed,
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

    if baseline.status == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
