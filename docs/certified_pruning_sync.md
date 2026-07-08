# Sync with paper-ellalpha: certified pruning and the act-map program (2026-07-08)

This document synchronizes this implementation repo with the current state of
the companion paper repo `../paper-ellalpha` and specifies the implementation
work needed for (a) certified pruning-based efficiency and (b) toy-example
numerics for the 2026-07 study-group talk.

Paper-repo sources of truth:

- `../paper-ellalpha/anisotropic_cech_pruning_research_plan.md` — certified pruning program (theorems A–G, I)
- `../paper-ellalpha/alpha_like_reduction_research_plan.md` — reduction program (act map, nondegeneracy, Del^aniso)
- `../paper-ellalpha/notes/alpha_like_reduction_status.ja.md` — Japanese status note (talk-oriented)
- `../paper-ellalpha/docs/early_gate_counterexample_note.md` — the d=2, n=4 degenerate counterexample
- `../paper-ellalpha/LeanEllAlpha/FormalizationRoadmap.md` — Lean status (A–E, I, G proven; F deferred)

## 1. Theorem naming map

This repo's README and docstrings cite "T1–T9" (the older numbering for the
minimax/KKT solver-level facts in `AnisotropicKKT.lean` / `Duality.lean`).
Those results still exist and back the solver. The paper repo has since added
a second, Lean-verified stack with letter names:

| Paper name | Statement (informal) | Lean name |
|---|---|---|
| A | σ ∈ Čech_r ⟺ α(σ) ≤ r | `cech_birth_value_equiv` |
| B | m·I ⪯ A_i, α(σ) ≤ r_max ⟹ σ is a clique of G_L, L = 2√(r_max/m) | `distance_pruning_soundness` |
| C | α({i,j}) > r_max kills every coface | `pairwise_minimax_pruning` |
| D | α(τ) > r_max, τ ⊆ σ ⟹ α(σ) > r_max | `face_monotonicity_pruning` |
| E | τ ⊆ σ ⊆ C(τ) ⟹ α(σ) = α(τ), where C(τ) = {j : f_j(c_τ) ≤ r_τ} | `active_support_interval_value` |
| I | certified candidate set retains every true simplex up to r_max | `integrated_candidate_superset_correctness` |
| G | max degree Δ ⟹ ≤ \|V\|·Σ_{t≤k} Δ^t clique candidates | `card_lowSkeletonCliqueFinset_le` |
| F | d-skeleton determines PH in degrees < d | deferred (citation-level) |

Reduction program (paper-side, not yet formalized): the act map
a(σ) = canonical (lex-smallest) minimal positive KKT support of σ; under
nondegeneracy (N) (unique minimal positive KKT support per simplex), the
fibers a⁻¹(τ) = [τ, C(τ)] are Boolean intervals, α is a generalized discrete
Morse function, and the filtration collapses persistence-preservingly onto

    Del^aniso = { σ : a(σ) = σ and C(σ) = σ }.

## 2. Drift findings (survey of this repo, 2026-07-08)

1. **No pruning is implemented.** `enumeration.py` generates all
   C(n, k+1) subsets; there is no r_max, no Rayleigh bound m, no neighbour
   graph G_L, no pairwise or face-value pruning (theorems B, C, D, G unused).
2. **P3 is used as a filtration-membership condition.** `filtration.py`
   accepts a simplex only if predicates P1–P3 pass. P2 (boundary contact with
   positive weights) and P3 (emptiness) are, in the current theory, the
   *critical-simplex* conditions defining Del^aniso — not membership
   conditions for the Čech filtration. Consequence, confirmed numerically:
   the phase-4 baseline check reports `missing_edge_count: 98` against the
   GUDHI alpha complex on n=100 isotropic data (non-Gabriel Delaunay edges are
   deleted instead of receiving inherited values). H0 agrees by luck
   (MST ⊆ Gabriel graph); H1 correctness is not guaranteed by the current
   pipeline. GUDHI resolves the same issue by value inheritance; the certified
   counterpart here is theorem E interval value reuse.
3. **Theorem E is absent.** The `alpha = max(own, faces)` clamp at
   `filtration.py:182` enforces monotonicity numerically but never reuses
   values across [τ, C(τ)], and every candidate is solved from scratch.
4. **Raw ingredients for the act map already exist.** `MinimaxResult.active_set`
   is the positive-weight KKT support; P3 internally computes f_j(c_σ) for all
   outside j (an almost-C(τ)) and discards it.

## 3. Implementation work packages

Conventions for all WPs: pure NumPy/SciPy + existing `solve_minimax`;
tolerance-aware comparisons everywhere (accept if α ≤ r_max + tol; a certified
*rejection* requires α > r_max + tol; ambiguous cases are retained, never
silently pruned — mirror the paper's oracle contract). Follow AGENTS.md
(backend-neutral core; GUDHI only via the backend).

### WP1 — `pruning.py`: certified candidate generation (theorems B, C, D, G)

- `rayleigh_lower_bound(matrices) -> m`: min over sites of λ_min(A_i).
- `neighbour_graph(centers, L)`: adjacency lists for ‖x̄_i − x̄_j‖ ≤ L with
  L = 2√(r_max/m).
- `clique_candidates(adj, max_dim)`: k-clique enumeration from sorted
  adjacency lists (per-vertex combinations of higher-indexed neighbours with
  clique check) — complexity matches theorem G accounting.
- `certified_filtration(coefs, r_max, max_dim, tol)`: dimension-ordered sweep.
  Edges of G_L are solved first; edges with α > r_max + tol are removed and
  higher candidates must be cliques of retained edges (theorem C). A candidate
  whose recorded facet has α > r_max + tol is skipped (theorem D). Everything
  else is solved or value-reused (WP2) and emitted with its α.
- Return a `PruningStats` record: brute-force count, G_L clique count, counts
  after pairwise and face pruning, number of minimax solves, number of reuse
  assignments, wall time.
- Soundness test (theorem I semantics): on random instances small enough for
  brute force (n ≤ 12, d = 2, max_dim = 3), the certified pipeline must emit
  exactly the simplices with α ≤ r_max, with α values matching the brute-force
  pipeline within tolerance.

### WP2 — `act_map.py`: act map, interval reuse, Del^aniso

- `act_support(sigma, result) -> tau`: map `active_set` indices back to vertex
  ids, sorted (canonical representative).
- `coface_set(tau, result, coefs, tol) -> C(tau)`: all j with
  f_j(c_τ) ≤ r_τ + tol (reuse/expose the P3 internals instead of discarding).
- Interval value reuse inside the WP1 sweep: after solving τ, register
  (τ, C(τ), r_τ); a later candidate σ with τ ⊆ σ ⊆ C(τ) gets α(σ) = r_τ
  without a solve (theorem E). Count reuses.
- `fiber_partition(filtration_entries)`: group simplices by a(σ);
  `booleanity_report(...)`: for each fiber, check it equals
  [τ, C(τ)] ∩ {dim ≤ max_dim}; report violations (expected: none on generic
  data; expected: present on the degenerate counterexample).
- `critical_simplices(...)`: σ with a(σ) = σ and C(σ) = σ (within the
  dimension bound) — the Del^aniso candidates.
- `critical_value_check(...)`: every birth/death endpoint of the GUDHI
  persistence of the full certified filtration lies (within tol) in the set
  {α(σ) : σ critical}. This is the observable Morse-theoretic consequence
  that avoids implementing the Morse boundary.

### WP3 — Čech-semantics fix for the baseline acceptance

Add a filtration mode in which P1 gates numerical trust but P2/P3 do NOT gate
membership: every certified candidate with α ≤ r_max enters with its α
(theorem A semantics). Reframe `run_baseline_barcode_agreement` to compare
H0 *and* H1 bottleneck distances against the GUDHI alpha complex on isotropic
data (persistence-level agreement is the correct acceptance; edge-set equality
against Delaunay is not implied by the theory and should be dropped or
demoted to an informational count). The current default behaviour can remain
available as an explicit `mode="critical-only"` for comparison, clearly
documented as NOT persistence-correct in general.

### WP4 — `scripts/run_talk_experiments.py` + figures (artifacts/talk/)

- **T1 (degenerate counterexample).** The d=2, n=4 configuration:
  x̄ = (−1,0), (1,0), (0,−1), (0,1); A = diag(1,2), diag(1,3), diag(4,1),
  diag(5,1). Verify α = 1 with center 0 for {1,2}, {3,4}, and {1,2,3,4}
  (0-indexed accordingly); show the solver's active_set on the full simplex
  and on {2,3,4}; then perturb centers by δ ∈ {1e-2, 1e-3} (fixed seed) and
  show unique supports and a Boolean fiber report with no violations.
  Output: a small table + one figure with the four ellipses through the
  origin at level r = 1.
- **T2 (pruning efficacy).** Random anisotropic clouds, d = 2,
  n ∈ {20, 50, 100}, A_i random SPD with condition number in [1, 10], fixed
  seeds, max_dim = 2 (optionally 3 for n = 20), r_max = a fixed percentile of
  pairwise α. Table: brute-force candidate count vs G_L cliques vs after
  pairwise vs after face pruning; solves performed; reuse count; wall time
  (brute-force pipeline timed only for n = 20).
- **T3 (Booleanity + Del^aniso on toys).** n ∈ {10, 15}, d = 2, several
  seeds, full sweep to max_dim = 3: Booleanity report (violations expected 0),
  fiber-size histogram, |{α ≤ r_max}| vs |critical|, critical_value_check
  against GUDHI persistence of the full filtration.
- **T4 (talk figure).** A two-ring anisotropic toy (rings with tangentially
  aligned anisotropy, n ≈ 40): ellipse field + G_L graph + retained complex
  at r_max; persistence diagram of the certified filtration with critical
  values marked. Save all figures as PDF+PNG under `artifacts/talk/`.
- Every experiment: fixed seeds, JSON raw output + CSV summary next to the
  figures (mirroring the existing artifacts/experiments layout).

### WP5 — docs and tests

- README: replace the T1–T9 citation with the naming map of §1 (keep T-names
  only where they refer to solver-level facts); link this document.
- DEVELOPMENT.md: append a dated session entry.
- Tests: pruning soundness vs brute force (WP1), reuse-equals-solve within
  tol (WP2), act-map idempotency on random simplices, tie detection on the
  T1 configuration, baseline acceptance green under WP3.
- All existing tests must stay green (`poetry run pytest`); the notebook
  stays untouched in this round.

## 4. Non-goals in this round

- No Morse boundary / collapsed persistence computation (critical_value_check
  is the stand-in).
- No arboricity-sensitive enumeration; no d = 3 point clouds.
- No changes to the solver's numerical methods.
- No symbolic perturbation; degenerate inputs are only *detected* (tie
  reporting), not resolved.
