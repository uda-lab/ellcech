# ellphi-alpha

**Experimental** implementation of the anisotropic alpha complex for ellipsoid-based TDA.

> Status: Pre-alpha. API and behaviour may change without notice.

## What this is

This package computes filtration values for the *anisotropic alpha complex*:

    α(σ) = min_x  max_{i ∈ σ} f_i(x),   f_i(x) = (x − x̄_i)ᵀ Aᵢ (x − x̄_i)

The mathematical foundations are formalised in Lean 4 in the companion repo
`paper-ellalpha/` (theorems T1–T9 in `AnisotropicKKT.lean` / `Duality.lean`).

## Relation to ellphi

[ellphi](../ellphi/) computes pairwise tangency distances and feeds them into
Vietoris–Rips filtrations. `ellphi-alpha` computes alpha-complex filtration
values for simplices of any dimension, giving a sparser (Delaunay-like) complex.

**Pairwise case correspondence:**

    ellphi_alpha.solve_minimax_from_coefs([p, q]).alpha  ==  ellphi.tangency(p, q).t ** 2

For |σ| ≥ 3 there is no direct ellphi equivalent — this is the core challenge
that this package addresses.

## Installation (development)

```bash
cd ellphi-alpha
poetry install
```

To also install notebook/demo dependencies:

```bash
poetry install --with demo
```

## Quick start

```python
import numpy as np
import ellphi
import ellphi_alpha as ea

# Two ellipses (from ellphi coef vectors)
p = ellphi.coef_from_cov([0.0, 0.0], [[0.4, 0.0], [0.0, 0.2]])[0]
q = ellphi.coef_from_cov([1.0, 0.5], [[0.3, 0.1], [0.1, 0.5]])[0]

# Pairwise filtration value
res = ea.solve_minimax_from_coefs(np.stack([p, q]))
print(f"α({{i,j}}) = {res.alpha:.6f}")
print(f"ellphi t  = {ellphi.tangency(p, q).t:.6f}  (should equal sqrt of above)")
print(f"circumcenter = {res.circumcenter}")

# Direct interface (matrices + centers)
A0 = np.linalg.inv([[0.4, 0.0], [0.0, 0.2]])
A1 = np.linalg.inv([[0.3, 0.1], [0.1, 0.5]])
res3 = ea.solve_minimax(
    np.stack([A0, A1, np.eye(2)]),
    np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]]),
)
print(f"α(triangle) = {res3.alpha:.6f},  active set = {res3.active_set}")
```

## Running tests

```bash
poetry run pytest
```

## Backend architecture

- Core solver, predicates, and filtration construction are backend-agnostic.
- Optional persistence adapters live under `ellphi_alpha.backends/`.
- `ellphi_alpha.backends.GudhiBackend` provides the GUDHI integration path.
- `to_gudhi_simplex_tree(...)` is kept as a public compatibility wrapper over
  the GUDHI adapter.

## Filtration Cache (Optional)

For repeated builds on the same dataset/settings, you can reuse predicate/minimax
results with a plain dictionary cache:

```python
cache = {}
filt1 = ea.build_incremental_filtration(A, centers, max_dim=2, predicate_cache=cache)
filt2 = ea.build_incremental_filtration(A, centers, max_dim=2, predicate_cache=cache)
```

Cache keys include simplex, `minimax_kwargs`, `boundary_tol`, and `empty_tol`.

## Phase-4 Acceptance Measurements

```bash
poetry run python scripts/run_phase4_acceptance.py --output-dir artifacts/phase4_acceptance
```

The runner writes:

- `baseline_barcode_agreement.json` (d=2, n=100 check vs GUDHI baseline)
- `six_rings_h1_check.json` (6-rings long-lived H1 check)
- `conditioning_stress_check.json` (cond > 1e6 alpha-error tracking)
- `phase4_acceptance_summary.json` (combined report)

If `gudhi` is unavailable, the first two checks are reported as `skipped` while
the conditioning benchmark still runs.

Both `run_baseline_barcode_agreement(...)` and `run_six_rings_h1_check(...)`
accept an optional `backend=` argument; by default they use `GudhiBackend` to
preserve existing behavior.

## Mathematical references

- `paper-ellalpha/LeanEllAlpha/` — Lean 4 formalisations (T1–T9)
- `paper-ellalpha/next_steps_plan.md` — implementation roadmap (Gap A–F)
