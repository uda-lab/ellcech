# ellphi-alpha

**Experimental** implementation of the anisotropic alpha complex for ellipsoid-based TDA.

> Status: Pre-alpha. API and behaviour may change without notice.

## What this is

This package computes filtration values for the *anisotropic alpha complex*:

    α(σ) = min_x  max_{i ∈ σ} f_i(x),   f_i(x) = (x − x̄_i)ᵀ Aᵢ (x − x̄_i)

The mathematical foundations are formalised in Lean 4 in the companion repo
`paper-ellalpha/`: solver-level facts as theorems T1–T9 in
`AnisotropicKKT.lean` / `Duality.lean`, and the certified pruning stack as
letter-named theorems A–G, I in `CechPruning.lean` (see the naming map in
[docs/certified_pruning_sync.md](docs/certified_pruning_sync.md) §1).

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

## Solver methods

Seven solver methods are available via `method=`:

| Method | Description |
|---|---|
| `fw+bisect` (default) | Pairwise FW with 52-step bisection line search |
| `fw+brentq` | Pairwise FW with adaptive brentq line search (~5x faster) |
| `fw+bisect+newton` | FW(bisect) + Newton polishing on active face |
| `fw+brentq+newton` | FW(brentq) + Armijo-damped Newton (recommended) |
| `fw+bisect+damped-newton` | FW(bisect) + Armijo-damped Newton |
| `scipy-slsqp` | Direct SLSQP solve (reference solver) |
| `newton-cold` | Newton from uniform start (stability baseline) |

```python
res = ea.solve_minimax(matrices, centers, method="fw+brentq+newton")
print(res.metadata)  # {'fw_iters': 12, 'newton_iters': 3, 'hessian_cond': 42.1, ...}
```

## Running tests

```bash
poetry run pytest
```

## Notebook demo

Use `notebooks/practical_demo.ipynb` as the single canonical notebook for users.
It combines quick numeric checks and visual walkthroughs in one place.

Execute it in-place to embed fresh outputs:

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/practical_demo.ipynb
```

Open [notebooks/practical_demo.ipynb](notebooks/practical_demo.ipynb) to inspect:
- pairwise `alpha` vs `ellphi.tangency(...).t**2`
- `|sigma| >= 3` minimax and active set
- filtration graph snapshot up to an alpha threshold
- optional GUDHI persistence summary (with H1 lifetime plot when available)

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

Membership semantics: the default `mode="cech"` admits every P1-trusted
candidate with its alpha value (theorem A semantics; persistence-correct).
`mode="critical-only"` restores the historical P1–P3 gating, which keeps only
critical-simplex candidates and is *not* persistence-correct in general.

## Certified pruning and the act map

`certified_filtration` builds the filtration up to a value bound `r_max`
without brute-force enumeration, using the Lean-verified pruning theorems
(B: neighbour graph, C/D: pairwise and face pruning, E: act-interval value
reuse) and KKT support certificates:

```python
from ellphi_alpha import certified_filtration, booleanity_report, critical_simplices

result = certified_filtration(matrices, centers, r_max=0.5, max_dim=2)
print(result.stats)             # candidate counts per pruning stage, solves, reuses
report = booleanity_report(result.info, max_dim=2)   # act fibers vs Boolean intervals
crit = critical_simplices(result.info)               # Del^aniso candidates
```

Acceptance uses the primal upper bound `max_i f_i(c)`; certified rejection
uses the weak-duality lower bound; ambiguous candidates are retained and
flagged `uncertain`.  Support certificates re-threshold the solver's active
set and verify equal-value spread, stationarity, and affine independence,
escalating to `scipy-slsqp` near degenerate configurations.  See
[docs/certified_pruning_sync.md](docs/certified_pruning_sync.md) for the
full specification and the paper-repo naming map.

Talk experiments (degenerate counterexample, pruning efficacy, Booleanity /
Del^aniso statistics, two-ring toy):

```bash
poetry run python scripts/run_talk_experiments.py   # -> artifacts/talk/
```

## Phase-4 Acceptance Measurements

```bash
poetry run python scripts/run_phase4_acceptance.py --output-dir artifacts/phase4_acceptance
```

The runner writes:

- `baseline_barcode_agreement.json` (d=2, n=100: H0+H1 bottleneck agreement
  of the certified filtration vs the GUDHI alpha complex, truncated at r_max)
- `six_rings_h1_check.json` (6-rings long-lived H1 check)
- `conditioning_stress_check.json` (cond > 1e6 alpha-error tracking)
- `phase4_acceptance_summary.json` (combined report)

If `gudhi` is unavailable, the first two checks are reported as `skipped` while
the conditioning benchmark still runs.

Both `run_baseline_barcode_agreement(...)` and `run_six_rings_h1_check(...)`
accept an optional `backend=` argument; by default they use `GudhiBackend` to
preserve existing behavior.

## Benchmarking

```bash
# Quick benchmark (k=2..6, all methods):
poetry run python scripts/benchmark_minimax.py

# Full numerical experiments (5 experiments):
poetry run python scripts/run_full_experiment.py --experiments 1 2 3 4 5 --seed 42

# Specific experiment with specific methods:
poetry run python scripts/run_full_experiment.py --experiments 5 --methods fw+bisect fw+brentq
```

Results are saved to `artifacts/experiments/`.

## Mathematical references

- [docs/certified_pruning_sync.md](docs/certified_pruning_sync.md) — sync
  with the paper repo: theorem naming map (T1–T9 vs A–G, I), drift findings,
  and the implementation specification for pruning/act-map/talk experiments
- `paper-ellalpha/LeanEllAlpha/` — Lean 4 formalisations (solver-level
  T1–T9; certified pruning A–E, I, G)
- `paper-ellalpha/anisotropic_cech_pruning_research_plan.md` — certified
  pruning program
- `paper-ellalpha/alpha_like_reduction_research_plan.md` — act map /
  Del^aniso reduction program
- `paper-ellalpha/next_steps_plan.md` — implementation roadmap (Gap A–F)
