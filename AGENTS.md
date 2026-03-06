# AGENTS.md

## Scope
- Applies only to `ellphi-alpha/`.

## Core Policy (Backend-Neutral)
- Keep core logic backend-neutral.
- Do not import backend-specific libraries in core modules.
- Put adapters under `src/ellphi_alpha/backends/`.

## Canonical Notebook
- Canonical notebook: `notebooks/practical_demo.ipynb`.
- When updated, execute in place (`--execute --inplace`) before handoff.

## Commits
- Use small, logical commits.

## Required Validation Before Handoff
- `poetry run pytest`
- `poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/practical_demo.ipynb`
- `poetry run python scripts/run_phase4_acceptance.py --output-dir artifacts/phase4_acceptance`

## Acceptance Checks
- A/B checks (`baseline_barcode_agreement`, `six_rings_h1_check`) may be skipped when `gudhi` is unavailable.
- `conditioning_stress_check` must pass.
