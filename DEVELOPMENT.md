# Development Log

## Session 2026-07-10 — Certified Pruning / Act Map / Talk Experiments (WP1–WP5)

paper repo との同期文書 `docs/certified_pruning_sync.md` §3 の作業パッケージを実装した．

### What was built

- **`pruning.py`（WP1）**: 定理 B（近傍グラフ G_L, L = 2√((r_max+tol)/m)）・
  C/D（対毎・面 pruning）・E（act 区間の値再利用）・G（clique 列挙）による
  `certified_filtration`．受理は primal 上界 max_i f_i(c)，certified な棄却は
  弱双対下界 g(μ)，中間は保持して `uncertain` フラグ（oracle 契約）．
  `PruningStats` が段階ごとの候補数・solve 数・再利用数を記録．
  brute force 対照 `brute_force_filtration` と健全性テスト（定理 I 意味論）付き．
- **`act_map.py`（WP2）**: KKT support 証明書 `certify_support`
  （support_tol=1e-6 で再閾値処理 → 等値スプレッド・停留残差・勾配のアフィン
  独立性を検証，不合格時は scipy-slsqp へエスカレーション）．
  `coface_set`（P3 内部計算の公開），`fiber_partition` / `booleanity_report` /
  `critical_simplices`（Del^aniso 候補）/ `critical_value_check`．
  回帰テスト: seed 20260708 の摂動反例で fw+bisect の偽 4 要素 support を
  証明書が棄却し，エスカレーションで support {0,1,2} に回復することを固定．
- **膜性の修正（WP3）**: `build_incremental_filtration` に `mode` を追加．
  既定 `"cech"` は P1 のみで膜性判定（定理 A 意味論・persistence 正）．
  旧挙動は `"critical-only"`（一般には persistence 非保存）として残置．
  phase4 baseline を H0+H1 bottleneck（対 GUDHI alpha complex，r_max で切断）
  に改修：n=100 で H0 1.9e-16，H1 9.8e-11．Delaunay との辺集合一致は
  情報表示に降格（健全な不変量は片側不等式 Čech 値 ≤ alpha 複体値）．
- **講演実験（WP4）**: `scripts/run_talk_experiments.py` → `artifacts/talk/`．
  T1: 退化反例の tie 検出（全単体 support 証明書 `degenerate`，Booleanity
  違反 1）と摂動 δ∈{1e-2,1e-3} での回復（違反 0，support {0,1,2}）．
  T2: pruning 効率（n=100, max_dim=2: brute 166,750 候補 → solve 4,572 +
  再利用 2,762，33 s）．T3: 6 例全てで Booleanity 違反 0・臨界値チェック
  違反 0，臨界単体は保持複体の 15–40%．T4: 二重リング（essential H1 = 2）．

### Notes

- 全テスト green（`uv run pytest`）．
- 既定 `mode="cech"` により，notebook の filtration セルの出力は再実行時に
  単体数が増える（P2/P3 で落ちていた単体が値付きで入る）．notebook 本文は
  本ラウンドでは未編集．

## Session 2026-03-06 — Minimax Solver Numerical Experiment (7 Methods)

### What changed

ellphi の root-finding では Brent 法 + Newton の「brentq+newton」が速度・精度・安定性の
最適バランスとして確立された．ellphi-alpha の minimax solver でも同様の体系的実験を行い，
最良の方法を選定するため，3 法→ **7 法**に拡張した．

### 新メソッド（`minimax.py`）

| method | 説明 | 新規/既存 |
|---|---|---|
| `fw+bisect` | 52 回固定 bisection line search (デフォルト) | 既存 |
| `fw+brentq` | 適応的 brentq line search (通常 8–15 回) | **新規** |
| `fw+bisect+newton` | FW(bisect) → Newton polishing | 既存 `fw+newton` のリネーム |
| `fw+brentq+newton` | FW(brentq) → Armijo damped Newton | **新規（本命候補）** |
| `fw+bisect+damped-newton` | FW(bisect) → Armijo damped Newton | **新規** |
| `scipy-slsqp` | scipy SLSQP 直接解法 | 既存 |
| `newton-cold` | 一様初期値からの Newton (安定性下界) | **新規** |

`"fw+newton"` は `"fw+bisect+newton"` への deprecated エイリアスとして保持．

### 新インフラ

- **`_ensure_finite()`**: NaN/Inf 検出ガード（ellphi パターン）
- **`_brentq_line_search()`**: scipy.optimize.brentq による適応的 line search，失敗時は bisection fallback
- **`_damped_newton_polish()`**: Armijo backtracking + ヘシアン条件数チェック（>1e12 で正則化）
- **`MinimaxResult.metadata`**: `fw_iters`, `newton_iters`, `hessian_cond`, `n_fevals`, `line_search_evals`

### 実験インフラ（`benchmarks.py`）

- **`make_ill_conditioned_simplex()`**: 対数スケール固有値で cond_target を達成
- **`make_degenerate_simplex()`**: n_far 個の遠方頂点で近退化 active set を生成
- **`ExperimentResult` / `ExperimentConfig`**: 実験データクラス
- **`bootstrap_ci()`, `wilson_interval()`, `summarize_experiment()`**: 統計ヘルパー
- **`run_experiment()`**: 因子格子の自動実行 + 参照値計算

### 実験スクリプト（`scripts/run_full_experiment.py`）

5 つの実験を CLI で実行可能：
1. 基本速度・精度 (k×d 格子)
2. 高条件数 (cond ∈ {2..10000})
3. 近退化 (n_far ∈ {1,2,3})
4. FW→Newton handoff (fw_tol sweep)
5. brentq vs bisect

出力: `artifacts/experiments/exp{N}_raw.json`, `exp{N}_summary.csv`

### Exp 5 結果（brentq vs bisect, 200 instances/cell, seed=42）

| k | fw+bisect t_med | fw+brentq t_med | speedup |
|---|---|---|---|
| 3 | 3.6 ms | 1.0 ms | **3.5x** |
| 4 | 15.0 ms | 3.2 ms | **4.7x** |
| 5 | 20.4 ms | 3.9 ms | **5.2x** |
| 6 | 22.3 ms | 4.3 ms | **5.2x** |
| 7 | 21.6 ms | 4.3 ms | **5.0x** |
| 8 | 23.6 ms | 4.7 ms | **5.0x** |

同一精度（rel_error median ~1e-9）で **brentq は bisect の約 5 倍高速**．
brentq の適応的終了（~8–15 回の関数評価）vs bisect の固定 52 回による差．

### テスト

- 78 tests in `test_minimax_methods.py` (全 pass)
- 全テストスイート: 133 passed, 2 skipped (GUDHI)

## Session 2026-03-06 — Backend-Decoupling Refactor

### What changed

- Added backend-neutral core contracts/utilities in `src/ellphi_alpha/core/`:
  - `backend_contracts.py` (`PersistenceBackend` protocol)
  - `filtration_normalization.py` (canonical filtration entry normalization)
- Added optional GUDHI adapter at `src/ellphi_alpha/backends/gudhi.py`.
- Kept public `to_gudhi_simplex_tree(...)` in `gudhi_bridge.py` as a
  compatibility wrapper over the adapter.
- Refactored Phase-4 acceptance checks to accept `backend=` injection while
  defaulting to `GudhiBackend` for behavior compatibility.
- Extended result schemas with neutral `reference_*` fields while preserving
  existing GUDHI-named serialized fields.

## Session 2026-03-06 — Phase-4 Acceptance Measurements

### What was built

- Added `src/ellphi_alpha/phase4_acceptance.py` to house reproducible
  measurement routines and JSON report writing.
- Added `scripts/run_phase4_acceptance.py` to run three checks and emit:
  - `baseline_barcode_agreement.json`
  - `six_rings_h1_check.json`
  - `conditioning_stress_check.json`
  - `phase4_acceptance_summary.json`
- Added/extended pytest coverage in `tests/test_phase4_baseline.py`:
  - baseline barcode agreement skip + GUDHI-backed test
  - 6-rings H1 skip + GUDHI-backed test
  - conditioning stress benchmark test (`cond > 1e6`, alpha error thresholds)

### Measurement definitions

- **A) Baseline barcode agreement**:
  isotropic `d=2, n=100` point cloud, compare H0 barcode against
  `gudhi.AlphaComplex` on the same points.
- **B) 6-rings long-lived H1**:
  deterministic 6-ring synthetic dataset; require at least six H1 intervals
  above a persistence threshold.
- **C) High-conditioning stress benchmark**:
  random pairwise anisotropic instances with
  `cond(A_i)` sampled in `[1e6, 1e7]`, reference alpha from
  `ellphi.tangency(...).t**2`, tracked by max/mean abs+rel error thresholds.

### Notes

- If `gudhi` is not installed, A/B are marked `skipped` (not failed) and the
  conditioning benchmark still runs.

### Measured outcomes (with GUDHI available)

Run date: 2026-03-06 (`scripts/run_phase4_acceptance.py`).

- **A) Baseline `d=2, n=100` barcode agreement vs GUDHI**:
  `status=ok`, `passed=True`, `h0_count_ours=100`, `h0_count_gudhi=100`,
  `h0_bottleneck=1.942890293094024e-16`,
  `max_abs_edge_alpha_diff=1.942890293094024e-16`.
- **B) 6-rings long-lived H1 check**:
  `status=ok`, `passed=True`, `h1_count=14`, `long_lived_h1_count=8`
  (threshold: `>= 6` at lifetime `>= 0.25`).

## Session 2026-03-06 — Gaps B-F

### What was built

- **Gap B (simplex enumeration)**: added `enumeration.py` with
  `iter_candidate_simplices` and `enumerate_candidate_simplices` for
  candidate simplices up to `max_dim`.
- **Gap C (predicates P1-P3)**: added `predicates.py` with
  `evaluate_predicates` and `PredicateResult`:
  - P1: minimax convergence/feasibility check
  - P2: boundary-contact geometric check on simplex vertices
  - P3: empty-ball geometric check against outside vertices
- **Gap D (incremental filtration)**: added `filtration.py` with
  `build_incremental_filtration` and `FiltrationEntry`; uses B+C, enforces
  downward closure and alpha monotonicity, and outputs alpha-sorted entries.
- **Gap E (GUDHI bridge)**: added `gudhi_bridge.py` with
  `to_gudhi_simplex_tree`; raises a clear runtime error when `gudhi` is absent.
- **Gap F (numerical stability controls)**: extended `solve_minimax` and
  `solve_minimax_from_coefs` with optional `regularization`,
  `condition_number_limit`, and `max_conditioning_steps`; defaults preserve
  prior behavior.
- Updated package exports in `__init__.py`.
- Added tests for all gaps and public API exports.

## Session 2026-03-05 — Initial Setup

### What was built

**Package scaffold** (`pyproject.toml`, `src/ellphi_alpha/`, `tests/`):
- uv project, `ellphi>=0.1` from PyPI, `requires-python = ">=3.10"`
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
| Gap B (simplex enumeration) | **Done** | Add candidate-pruning heuristics if needed |
| Gap C (predicates P1-P3) | **Done** | Refine tolerance defaults empirically |
| Gap D (incremental construction) | **Done** | Add larger-scale benchmarks |
| Gap E (GUDHI interface) | **Done** | Add end-to-end PH examples in notebooks |
| Gap F (numerical stability) | **Done** | Tune conditioning thresholds for production datasets |

### Environment

- Python 3.12.4
- ellphi 0.1.1.post1 (PyPI)
- numpy 2.4.2, scipy 1.17.1
- All 20 tests pass in 0.59s
