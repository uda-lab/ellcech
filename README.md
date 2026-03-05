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

## Mathematical references

- `paper-ellalpha/LeanEllAlpha/` — Lean 4 formalisations (T1–T9)
- `paper-ellalpha/next_steps_plan.md` — implementation roadmap (Gap A–F)
