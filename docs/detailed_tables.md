# Detailed Evaluation Tables

Detailed numerical tables supporting [`README.md`](../README.md) across `Qwen/Qwen3-4B` (`n = 278` scored Ulto85 neurons, `424` active poles) and `Qwen/Qwen3-1.7B` (`n = 50` neurons, `100` poles). Offline outputs in `results/` come directly from the analysis scripts (`results/summary_tables.md`, `results/qwen4b/summary_4b.md`, `results/qwen4b/surrogate_summary_4b.md`, `results/qwen17b_causal/summary_17b_causal.md`, and `results/qwen4b/effect_weighted.json`).

---

## 1. Cross-Model Per-Band Detection vs. Exact `float32` KL Share & EWL

### 1.1 `Qwen/Qwen3-4B` (`n = 278` Scored Neurons, Exact `float32` Band KL, `zero_kl = 0.0`, Additivity Median `1.00001`)

| Subset | n | Band | Det Acc (Detected Only) [95% Neuron CI] | Miss | Exact `fp32` KL Share [95% Neuron CI] | EWL (All Bands) [95% CI] | EWL (`>= p75` Renorm) | `p99.9+` Lift / `top100` Lift |
|---|---:|---|---|---:|---:|---:|---:|---:|
| All KL neurons | 278 | `p99.9+` | 0.839 [0.820, 0.857] | 0 | 0.293 [0.262, 0.324] | **0.345 [0.309, 0.382]** (eq `0.336`) | **0.421 [0.384, 0.458]** | 0.679 [0.643, 0.713] / 0.794 |
| All KL neurons | 278 | `p97–99.9` (energy-merged) | 0.664 [0.643, 0.686] (eq `0.649`) | 1 | 0.334 [0.312, 0.356] | — | — | — |
| All KL neurons | 278 | `p75–97` | 0.549 [0.528, 0.570] | 5 | 0.192 [0.175, 0.210] | — | — | — |
| All KL neurons | 278 | `< p75` (`p25–75` tested) | 0.499 [0.478, 0.519] | 15 | 0.182 [0.158, 0.206] | — | — | — |
| Bipolar (2-sided) | 146 | `p99.9+ / p97–99.9 / p75–97 / <p75` | `0.807 / 0.639 / 0.528 / 0.517` | 0 | `0.316 / 0.408 / 0.249 / 0.027` | **0.359 [0.314, 0.407]** | **0.364 [0.320, 0.410]** | 0.614 / 0.742 |
| Unipolar (1-sided) | 132 | `p99.9+ / p97–99.9 / p75–97 / <p75` | `0.875 / 0.691 / 0.572 / 0.477` | 0/1/5/15 | `0.267 / 0.251 / 0.128 / 0.353` | **0.329 [0.274, 0.385]** | **0.485 [0.426, 0.541]** | 0.751 / 0.851 |

- **Ratio, merge sensitivity & spread (`Qwen3-4B`, `n = 278`):** `Qwen3-4B` EWL drops roughly in half relative to `p99.9+` lift (`0.679 -> 0.345`, ratio `0.51`). Merge sensitivity across `p97–99.9` weights spans `0.327–0.345` (token-width `0.9/2.0`: `0.327`, equal `50/50`: `0.336`, corpus L2 energy: `0.345`), and per-neuron EWL across `278` neurons has median `0.348` (`0.35`, IQR `[0.104, 0.574]`) with `15.1%` (`42/278`) negative.
- **Note on `< p75` unipolar KL (`0.353`) vs. bipolar KL (`0.027`):** On unipolar (`1`-sided) neurons (`n = 132`), the excluded opposite-sign tail carries `mean_opposite_sign_eval_energy_share = 0.3505` inside `< p75` (contributing to `KLShare(< p75) = 0.353`, vs. `0.027` and `0.000` opposite-sign energy on `146` bipolar neurons). Detection on `< p75` tests `p25–75`; `< p25` was not tested.
- **Note on `n = 60` pilot subset vs. full `n = 278`:** The stratified `n = 60` KL subset over-represented rare passing causal neurons (`5/60 = 8.3%` with `sim_cal_rec >= 0.50` vs. `11/278 = 4.0%`; `n = 60` `EWL = 0.371 [0.295, 0.446]`, `p99.9+` lift = `0.753`).
- **`EvalEnergy` vs. `KLShare` sanity check (`4B`, `n = 278`, `1,112` pairs):** Pooled Spearman $\rho = 0.861$ (neuron-clustered 95% CI `[0.830, 0.887]`, pair-level CI `[0.838, 0.882]`; Pearson $r = 0.827$, clustered `[0.788, 0.860]`, pair `[0.798, 0.854]`); 1,000-permutation neuron-shuffled null $\rho = 0.109$ `[q025=0.045, q975=0.172]`, empirical $p = 0.001$; within-band $\rho$: `p99.9+ = 0.790 [0.730, 0.839]`, `p97_99.9 = 0.805 [0.743, 0.858]`, `p75_97 = 0.867 [0.820, 0.902]`, `lt_p75 = 0.885 [0.847, 0.916]` (`CorpusEnergy` pooled $\rho = 0.776$, clustered `[0.738, 0.812]`).

### 1.2 `Qwen/Qwen3-1.7B` (`n = 50` All / `n = 40` Held-Out, Exact `float32` Band KL)

| Cohort | n | Band | `own` Acc [95% CI] | `no_label` Acc | Exact `fp32` KL Share [95% CI] | EWL (`own`, Equal Merge) [95% CI] | `own - no_label` EWL (2× Rescaled / Raw) | `p99.9+` Lift |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `all_50` | 50 | `p99.9+ / p97–99.9 / p75–97 / <p75` | `0.782 / 0.653 / 0.550 / N/A` | `0.532 / 0.457 / 0.468 / N/A` | `0.410 / 0.431 / 0.148 / 0.011` | **0.397 [0.330, 0.470]** (`en 0.398 [0.325, 0.474]`, `>=p75 0.402 [0.332, 0.470]`) | **+0.410 [+0.290, +0.539] / +0.205 [+0.139, +0.270]** | 0.563 [0.483, 0.640] |
| `heldout_40` | 40 | `p99.9+ / p97–99.9 / p75–97 / <p75` | `0.777 / 0.623 / 0.547 / N/A` | `0.535 / 0.458 / 0.481 / N/A` | `0.422 / 0.431 / 0.137 / 0.010` | **0.369 [0.297, 0.439]** (`en 0.369 [0.298, 0.442]`, `>=p75 0.373 [0.301, 0.445]`) | **+0.372 [+0.240, +0.504] / +0.186 [+0.121, +0.250]** | 0.554 [0.467, 0.646] |

- **`EvalEnergy` vs. `KLShare` sanity check (`1.7B`, `n = 50`, `200` pairs):** Pooled Spearman $\rho = 0.907$ (neuron-clustered 95% CI `[0.855, 0.942]`, pair-level CI `[0.863, 0.936]`; Pearson $r = 0.855$); 1,000-permutation neuron-shuffled null $\rho = 0.632$ `[q025=0.561, q975=0.706]`, empirical $p = 0.001$; within-band $\rho$: `p99.9+ = 0.782 [0.586, 0.895]`, `p97_99.9 = 0.636 [0.385, 0.806]`, `p75_97 = 0.873 [0.727, 0.952]`, `lt_p75 = 0.581 [0.333, 0.777]`.

---

## 2. `Qwen/Qwen3-4B` (`Ulto85/neurons-kinda-clean`, MIT) Detection Sweeps & Means (`n = 278` Neurons, `424` Poles)

### 2.1 Per-Pole and Per-Neuron Means & Slot-A Preference (`gpt-6.1-terra@none` unless noted)

| Sweep / Condition | Band | Calls | Scored / Unp / Tie | Per-Pole Acc [95% Neuron CI] | Per-Neuron Acc | `P(pred = A)` | `P(pos = A)` | Spend USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `p999_own_terra` | `p99.9+` | 8,480 | `8480 / 0 / 0` | 0.828 [0.810, 0.846] | 0.839 | 0.454 | 0.500 | $4.5317 |
| `p999_own_luna` | `p99.9+` | 8,480 | `8480 / 0 / 0` | 0.789 [0.773, 0.806] | 0.801 | 0.511 | 0.500 | $0.4532 |
| `p999_wrong_terra` | `p99.9+` | 2,544 | `2544 / 0 / 0` | 0.509 [0.488, 0.534] | — | 0.407 | 0.500 | $1.3578 |
| `p999_no_label_terra` | `p99.9+` | 2,544 | `2544 / 0 / 0` | 0.579 [0.554, 0.606] | — | 0.415 | 0.500 | $0.9050 |
| `p97_99_own_terra` | `p97–99` | 4,190 | `4190 / 0 / 0` | 0.608 [0.588, 0.629] | 0.613 | 0.411 | 0.500 | $2.2391 |
| `p999_full_expl_terra` (`n=50`) | `p99.9+` | 1,480 | `1480 / 0 / 0` | 0.857 [0.819, 0.891] | — | 0.477 | 0.500 | $1.4550 |
| `top100_own_terra` (post hoc) | `top100` | 8,480 | `8480 / 0 / 0` | 0.888 [0.873, 0.901] | 0.897 | 0.474 | 0.500 | $4.5314 |

- **Additional multi-band sweeps (`terra@none`, neuron / pole mean):** `p99_99.9 = 0.686 / 0.677` (`2,538` calls, `423` poles / `277` neurons), `p75_97 = 0.549 / 0.542` (`2,496` calls, `416` / `273`), `p25_75 = 0.499 / 0.504` (`2,316` calls, `386` / `263`). (`luna@none` neuron means: `p99_99.9 = 0.660`, `p97_99 = 0.597`, `p75_97 = 0.523`, `p25_75 = 0.531`.)
- **Top-100 lexical overlap & post-hoc status:** `top100` was added post hoc after observing `295/424` (`69.6%`) active poles had `p99.9 < 0.20 * max_act_dir`. Across `8,480` `top100` pairs, `30.6%` (`2,591/8,480`) match an exact case-insensitive word (`len >= 2`) in the pole description, and `37.9%` (`3,210/8,480`) match case-insensitive substring containment; both counts were computed from the `top100` snippet file (`snippets_4b_top100.jsonl`), which is not shipped.

### 2.2 Base Rates vs. Conditional Pass Rates (`n = 278` Scored Ulto85 Neurons)

- **Overall base rates (`n = 278`):** `267/278` (`96.0%`) have `sim_cal_rec < 0.50`; `251/278` (`90.3%`) have `sim_cal_rec < 0.25`; `155/278` (`55.8%`) have `sim_cal_rec <= 0.0`. Only `11/278` (`4.0%`) reach `sim_cal_rec >= 0.50`, `27/278` (`9.7%`) reach `>= 0.25`, and `64/278` (`23.0%`) reach `sim_cal_rec_active >= 0.50`.
- **Failing `top100 >= 0.80` (`n = 43/278`):** `0/43` (`0.0%`) reach `sim_cal_rec >= 0.25`; `1/43` (`2.3%`) reaches `sim_cal_rec_active >= 0.50`.
- **Passing `top100 >= 0.80` (`n = 235/278`):** lifts `sim_cal_rec_active >= 0.50` from `1/43` to `63/235` (`26.8%`), while `224/235` (`95.3%`) stay at `sim_cal_rec < 0.50`, `208/235` (`88.5%`) below `0.25`, `124/235` (`52.8%`) at `<= 0`, and only `11/235` (`4.7%`) reach `sim_cal_rec >= 0.50`.
- **Rank correlation & spread:** `top100` vs `sim_cal_rec` Spearman $\rho = 0.142$ `[0.025, 0.264]` (`sim_cal_rec_active`: $\rho = 0.485$ `[0.386, 0.570]`, `strong20_rec`: $\rho = 0.320$). Across the `86` neurons with `top100 == 1.00`, `sim_cal_rec` spans `[-7.110, +0.909]` (`median = 0.020`).
- **Provenance exclusions:** `278` of `280` neurons in `e3_neurons.json` are scored in `e3_scores.json` (`(12, 1597)` and `(24, 289)` unscored); `132` weak/negligible poles were excluded (`424` active poles kept). Ulto85 explanations and causal scores were written by Claude models (Opus explainer, Sonnet/Opus simulator).

---

## 3. `Qwen/Qwen3-1.7B` Held-Out Detection Audit (`heldout_80`, `gpt-6.1-terra@none`, `dead_zone`)

| Condition / Rule | `p75–90` [95% CI] | `p90–97` [95% CI] | `p97–99.9` [95% CI] | `p99.9+` [95% CI] | `scored / unp / tie` |
|---|---:|---:|---:|---:|---:|
| `own` (`gpt-4.1` label) | 0.548 [0.502, 0.590] | 0.546 [0.487, 0.600] | 0.623 [0.571, 0.673] | 0.777 [0.733, 0.825] | `1920 / 0 / 0` |
| `token_only` (top-3 train tokens) | 0.548 [0.504, 0.592] | 0.523 [0.481, 0.569] | 0.583 [0.537, 0.629] | 0.746 [0.700, 0.790] | `1920 / 0 / 0` |
| `no_label` (`"(none provided)"`) | 0.471 [0.427, 0.510] | 0.492 [0.435, 0.552] | 0.458 [0.392, 0.521] | 0.535 [0.465, 0.606] | `1920 / 0 / 0` |
| `wrong` (cross-layer cosine `<= 0.65`) | 0.515 [0.469, 0.558] | 0.487 [0.438, 0.540] | 0.450 [0.410, 0.487] | 0.469 [0.408, 0.527] | `1920 / 0 / 0` |

- **Sub-band & distractor arms (`heldout_80`, `own`, `terra@none`):**
  - `p97–99` (`n = 344` pairs) = `0.594` vs. `p99–99.9` (`n = 136` pairs) = `0.679`.
  - `examples_as_label` (20 training firings instead of label text): `p97–99.9 = 0.648`, `p99.9+ = 0.879` (`+0.102 [+0.058, +0.150]` over `own`).
  - `goodfire_lt_p90` at `p99.9+`: `0.796` (vs `0.777` `dead_zone`).
  - `nearest_wgate_tier0` at `p99.9+` (`n = 32` surviving pairs): `0.706` vs matched `dead_zone` `0.719` (`-0.013`).
  - Weak-band gate split (`own`): `p90–97` `gate_open = 0.561` vs `gate_closed = 0.532` (`+0.029`); paired `own - token_only` at `p99.9+` = `+0.031 [-0.015, +0.079]`.

---

## 4. Goodfire Table 4 Reference Comparison

| Benchmark | Model & Routing | Explainer / Judge (Batching) | Unit (n) | `p75–97` | `p97–99` | `p99–99.9` | `p99.9+` | `top100` / `top1000` |
|---|---|---|---|---:|---:|---:|---:|---:|
| Goodfire Table 4 | `Qwen3.6-35B-A3B` (MoE, routed) | Goodfire modes / `Opus 5.5` (10/call) | Poles (496) | — | 0.906 | 0.932 | `0.952 (p99.9–99.99) / 0.972 (p99.99+)` | 0.970 (`top1000`) |
| `Qwen3-4B` (per-neuron) | `Qwen/Qwen3-4B` (dense) | Ulto85 Claude / `terra@none` (1/call) | Neurons (278) | 0.549 | 0.613 (`n=275`) | 0.686 (`n=277`) | 0.839 | 0.897 (`top100`) |
| `Qwen3-4B` (per-pole) | `Qwen/Qwen3-4B` (dense) | Ulto85 Claude / `terra@none` (1/call) | Poles (424) | 0.542 | 0.608 (`n=419`) | 0.677 (`n=423`) | 0.828 | 0.888 (`top100`) |
| `Qwen3-1.7B` (`heldout_80`, per-neuron) | `Qwen/Qwen3-1.7B` (dense) | `gpt-4.1` / `terra@none` (1/call) | Neurons (40) | 0.547 (`eq 90-97/75-90`) | 0.594 | 0.659 | 0.777 | — |
| `Qwen3-1.7B` (`heldout_80`, per-pole) | `Qwen/Qwen3-1.7B` (dense) | `gpt-4.1` / `terra@none` (1/call) | Poles (80; 73 at `p99–99.9`) | 0.547 (`eq 90-97/75-90`) | 0.594 | 0.679 | 0.777 | — |
| `Qwen3-1.7B` (`all_100`, per-neuron / pole) | `Qwen/Qwen3-1.7B` (dense) | `gpt-4.1` / `terra@none` (1/call) | Neurons (50) / Poles (100) | 0.550 (`eq 90-97/75-90`) | 0.618 / 0.618 | 0.698 / 0.722 | 0.782 / 0.782 | — |

*Caveat: Goodfire evaluates routed MoE activations (`a * w_router`) using Goodfire Neuron Scientist mode descriptions, `Claude Opus 5.5` in batches of 10, and random-text distractors; our evaluations use dense SwiGLU activations, Ulto85 Claude (`4B`) or `gpt-4.1` (`1.7B`) labels, `gpt-6.1-terra@none` at 1 pair per call, and `dead_zone` distractors.*
