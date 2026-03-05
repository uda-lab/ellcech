# Development Log

## Session 2026-03-05 — Initial Setup

### What was built

**Package scaffold** (`pyproject.toml`, `src/ellphi_alpha/`, `tests/`):
- Poetry project, `ellphi>=0.1` from PyPI, `requires-python = ">=3.10"`
- Matches ellphi's `src/` layout for smooth eventual integration

**Gap A: minimax solver** (`src/ellphi_alpha/minimax.py`):
- Computes `alpha(sigma) = min_x max_i f_i(x)` via the Lagrange dual `max_{mu in Delta} g(mu)`
- Algorithm: **Pairwise Frank-Wolfe with exact 1D line search (bisection)**
  - Each step chooses `s = argmax f_i` (FW vertex) and `v = argmin f_i` (away vertex)
  - Bisects on `dg/dgamma = f_s(x*(gamma)) - f_v(x*(gamma)) = 0` over `[0, mu_v]`
  - Pairwise case (k=2): converges in **1 step** (equivalent to ellphi's 1D root-finding)
  - General case (k>=3): **linear convergence** (vs O(1/T) for naive fixed-step FW)
- Public API: `solve_minimax(matrices, centers)` and `solve_minimax_from_coefs(coefs)`

**Test suite** (`tests/test_minimax.py`, 20 tests, all passing):
- Trivial cases (k=1, alpha=0)
- Isotropic (A=I): midpoint, half-squared-distance, circumcenter of triangle
- Cross-validation vs ellphi for pairwise case (k=2, 2D and 3D, rel tol 1e-5)
- Higher-order simplices (k=3,4,5): convergence and active set

### Mathematical correspondence with ellphi

| | ellphi | ellphi-alpha |
|---|---|---|
| Pairwise problem | `F(mu)=0`, 1D root-finding (Brentq+Newton) | `max g(mu)`, pairwise FW + bisection |
| Solve time (k=2) | ~microseconds (native Newton) | ~microseconds (1 bisection pass) |
| k>=3 | not applicable | linear convergence |
| Return value | `t = sqrt(alpha)` | `alpha = t^2` |
| `b` sign | `b = -A x_bar` | center input as `x_bar` directly |

### Key sign convention

ellphi packed coef: `Q(x) = x^T A x + 2 b^T x + c` with `b = -A x_bar`, `c = x_bar^T A x_bar`
→ recover center: `x_bar = -A^{-1} b`  (handled in `solve_minimax_from_coefs`)

ellalpha Lean definition: `f_i(x) = (x - x_bar_i)^T A_i (x - x_bar_i)`
→ inputs to `solve_minimax`: `(A_i, x_bar_i)` directly

### Why exact line search matters

The naive Frank-Wolfe with step size `gamma = 2/(t+2)` has O(1/T) convergence.
For the pairwise case, achieving `rel=1e-5` accuracy would require ~200,000 iterations.
With bisection-based exact line search, the pairwise case converges in **1 iteration**
because `dg/dgamma = 0` at the root is exactly the condition `f_i = f_j`,
which is also the optimality condition for `alpha({i,j})`.

### Pending work (from next_steps_plan.md)

| Gap | Status | Next action |
|-----|--------|-------------|
| Gap A (minimax solver) | **Done** (pairwise FW + bisection) | Consider C++ port when stable |
| Gap B (simplex enumeration) | Not started | Phase 4.1: enumerate all (d+1)-simplices for n<=50 |
| Gap C (predicates P1-P3) | Not started | Implement after Gap B |
| Gap D (incremental construction) | Not started | Depends on B+C |
| Gap E (GUDHI interface) | Not started | Depends on D |
| Gap F (numerical stability) | Partial | active set shrinking via weight_tol in solver |

### Environment

- Python 3.12.4
- ellphi 0.1.1.post1 (PyPI)
- numpy 2.4.2, scipy 1.17.1
- All 20 tests pass in 0.59s
