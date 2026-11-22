# Exploratory and Post Hoc Appendix

> **Status: Exploratory and Post Hoc.** Every analysis in this document was designed after inspecting initial detection and Ulto85 causal outcomes (`sim_cal_rec`, `strong20_rec`) or produced inconclusive results across models. None of these measures represent pre-specified hypotheses or causal verification tools.

---

## 1. Activation-Energy Share & `top100 >= 0.80` Diagnostic Buckets (`Qwen3-4B`)

While `top100` detection (`>= 0.80` on `235/278` neurons) does not predict `sim_cal_rec` (`ρ = 0.142 [0.025, 0.264]`), zero-label centered activation variance share (`EnergyShare(B) = sum_{t in B} (a_t - mean_a)^2 / sum_t (a_t - mean_a)^2` over `1,536,000` FineWeb tokens) tracks oracle strong-firing recovery `strong20_rec` (`ρ = 0.708 [0.635, 0.770]`, `r = 0.691`, `R^2 = 0.477`; using `EnergyShare(>=50% max)` vs `strong50_rec`: `ρ = 0.639 [0.554, 0.713]`). Notice that `EnergyShare(|a| >= 20% max)` reads zero labels and already yields `AUROC(sim_cal_rec >= 0.25) = 0.921` (`ρ = 0.348 [0.234, 0.464]`, `AUROC >= 0.50 = 0.968`, `P@20 = 0.700`, `P@50 = 0.440`), reflecting the fact that neurons whose activation energy is concentrated in strong firings (`>= 20% max`) are structurally necessary (`strong20_rec` ceiling) for non-trivial full-distribution simulation recovery.

Post-hoc mutually exclusive failure buckets among the `235` neurons passing `top100 >= 0.80` (`n = 278` total):

| Bucket | n | Share of 235 | Med `E(p99.9+)` | Med `E(<p97)` | Med `E(>=20%)` | Med `strong20` | Med `E(gate_closed)` | Med `sim_cal_rec` | Example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pass_causal` (`sim >= 0.50`) | 11 | 4.7% | 0.664 | 0.107 | 0.650 | 0.759 | 0.131 | 0.666 | `info` (`k=26`, `0.859`) |
| `overbroad_fp_damage` (`sim < 0`, `active >= 0.50`) | 23 | 9.8% | 0.430 | 0.305 | 0.274 | 0.342 | 0.301 | -0.151 | `PCR` (`k=23`, `-0.570`) |
| `weak_mass_dominated` (`strong20 < 0.25`) | 84 | 35.7% | 0.162 | 0.471 | 0.136 | 0.114 | 0.582 | -0.021 | `insurance` (`k=236`, `-0.008`) |
| `active_sim_failure` (`strong20 >= 0.25`, `active < 0.50`) | 94 | 40.0% | 0.300 | 0.279 | 0.272 | 0.433 | 0.252 | 0.003 | — |
| `other_partial` (`0 <= sim < 0.50`, `strong20 >= 0.25`, `active >= 0.50`) | 23 | 9.8% | 0.507 | 0.261 | 0.381 | 0.460 | 0.227 | 0.245 | — |

---

## 2. Energy-Weighted Detection (`WDet`) & Ridge Surrogate Benchmarks (`Qwen3-4B`)

We tested post hoc whether weighting band-wise 2AFC detection accuracies by `EnergyShare` (`WDet`, zero free parameters) or fitting nested layer-stratified 5-fold OOF ridge / rank-ridge models could predict `sim_cal_rec` better than `EnergyShare` alone (`results/qwen4b/surrogate_metrics_4b.json`, API spend `$4.5396` multiband + `$2.0644` YES/NO probes = `$6.6039`):

- **`WDet` adds zero rank information above zero-label `EnergyShare` on `sim_cal_rec`:**
  - `s2_wdet_clamped_terra` (`ρ = 0.194`) has **partial Spearman correlation `-0.060 [-0.169, +0.060]` (`p = 0.316`)** controlling for `energy_ge_20pct_max` (`Δρ vs E(>=20%) = -0.153 [-0.244, -0.058]`).
  - `s2_wdet_clamped_top100_terra` (`ρ = 0.244`) has **partial Spearman correlation `-0.009 [-0.126, +0.109]` (`p = 0.879`)**.
  - `s2_wdet_clamped_luna` (`ρ = 0.068`, partial `-0.190`).
  - Reason: below `p97`, 2AFC detection approaches chance (`p75_97 = 0.549`, `p25_75 = 0.499` on `terra`), so multiplying near-chance lifts by band energy dilutes raw `EnergyShare`.
- **60-neuron KL-weighted `WDet` check (`results/qwen4b/band_kl_4b.json`):** `WDet_energy (4-band)` correlated with `WDet_KL (norm)` (`ρ = 0.776 [0.626, 0.880]`), and replacing corpus energy weights with 8,192-token KL weights did not improve `sim_cal_rec` prediction (`wdet_kl_norm_terra` `ρ = 0.194 [-0.068, 0.417]` vs `b1_energy_ge_20pct_max` `ρ = 0.501` on the same 60 neurons).
- **Out-of-fold 5-fold CV ridge models:** `s4_cv_ridge_full_terra` (`OOF ρ = 0.311`, partial `+0.010`, `p = 0.872`) and `s4_cv_rank_ridge_full_terra` (`0.297`) sit below raw zero-label `EnergyShare(>=20% max)` (`0.348`). `s4_cv_rank_ridge_full_luna` reaches `0.359 [0.242, 0.461]` (`Δρ vs E(>=20%) = +0.012 [-0.037, +0.064]`, CI spans zero) only by placing a *negative* standardized weight (`-0.073`) on `WDet_luna` (acting as a spikiness contrast against `energy_p99.9+`, not positive label-lift weight).

---

## 3. Single-Snippet Ordinary-Text YES/NO Precision Probe (`FPR_ord` / `1 - P(YES | ord)`)

We queried `gpt-6.1-terra@none` (`10` ordinary `< p90` + `2` top-100 firing probes/pole, `5,088` calls) and `gpt-6-luna@none` (`20` ordinary + `4` top-100/pole, `10,176` calls) on single marked snippets asking `YES` or `NO` whether the pole activates on `<<token>>`.

### Critical Logprob Imputation Caveat & Failure on Overbroad Damage
- **Probability-floor imputation:** Because ordinary random tokens rarely look like trigger tokens and `top_logprobs=5` was used, **`96.37%` (`4,086 / 4,240`) of `terra@none` ordinary-probe calls and `89.22%` (`7,566 / 8,480`) of `luna@none` ordinary-probe calls did not have `"YES"` inside the top-5 logprobs** and required imputing `logprob_YES` from the residual probability-mass floor (`log(max(1 - sum_top5 p, eps))`). Subsequent `P(YES | ordinary)` and `yes_margin` estimates are therefore sensitive to top-5 floor truncation rather than resolved `YES` logit differences.
- **Complete failure to catch negative recovery / overbroad labels:** Overall mean `FPR_ord` was `0.032` (`terra`) / `0.028` (`luna`). On named neuron `PCR` (`sim_cal_rec = -0.570`), `FPR_ord` was `0.000` (`0/10` on `terra`). Across the `overbroad_fp_damage` bucket (`n = 23`), mean `FPR_ord` (`0.009` `terra`, `0.005` `luna`) was lower than on `sim_cal_rec >= 0` (`0.026` / `0.027`), giving `ρ(FPR_ord, sim_cal_rec) = -0.030 [-0.129, +0.062]` and `AUROC(overbroad_fp) = 0.428` (`terra`) / `0.354` (`luna`). Uniformly sampled ordinary text misses sub-1% rare triggers and near-miss calibration errors.
- **Exploratory correlation on 4B `sim_cal_rec_active`:** Controlling for `E(>=20%)`, `1 - mean P(YES | ord)` (`terra@none`) correlated with active-token causal recovery `sim_cal_rec_active` on 4B (`ρ = 0.596 [0.516, 0.665]`, partial `+0.557`; `luna@none` `yes_margin_gap`: `ρ = 0.525 [0.422, 0.615]`, partial `+0.468`). Because this was post hoc, relied largely on floor-imputed ordinary `logprob_YES` values, and was inconclusive on `Qwen3-1.7B`, it should be treated strictly as an exploratory observation rather than a validated predictor.

---

## 4. Inconclusive `Qwen3-1.7B` Causal Simulation & `4B -> 1.7B` Transfer Attempt (`n = 50`)

We attempted to test cross-model transfer on all `50` `Qwen3-1.7B` neurons (`3,200` sparse JSON window simulations on `gpt-6.1-terra@none` over `64` windows/neuron + `3,600` YES/NO probes, `$3.6386` API spend on disk / `$3.6437` incl. `$0.0051` discarded 5-call pre-optimization test; exact `float32` causal patching under `allow_tf32=False`, zero-ablation KL `0.00e+00`):

- **Why `1.7B` causal simulation was inconclusive:** Using single-pass `gpt-4.1` labels (mean `FPR_ord = 0.141` vs `0.032` on 4B) and a lightweight `gpt-6.1-terra@none` sparse-JSON simulator, simulated activations achieved only **`median corr = 0.079`**, oracle `strong20_rec >= 0.25` held in **`34 / 50`** neurons (`28 / 50` on the 8k slice), while **calibrated simulator recovery reached `sim_cal_rec >= 0.25` in `0 / 50` neurons** (`2-fold CV` median `-0.029`) and `sim_cal_rec_active >= 0.50` in **`0 / 50` neurons** (median `+0.034`, `max = 0.392`).
- **Measured `1.7B` surrogate & transfer numbers (severe floor compression):** Under this simulator/label floor failure, `1 - P(YES | ord)` (`terra`) scored `ρ = +0.013 [-0.274, +0.300]` (`partial -0.006`) on `sim_cal_rec_active`, `yes_margin_gap` (`luna`) scored `+0.121 [-0.181, +0.390]`, zero-label `EnergyShare(>=20%)` flipped sign on `sim_cal_rec` (`ρ = -0.190 [-0.440, +0.089]`, `energy_ge_50pct_max = -0.460`, because misspecified labels incur larger negative KL penalties on spiky high-energy neurons when no neuron succeeds), and all `4B -> 1.7B` zero-shot ridge/rank-ridge models failed (`sim_cal_rec` `ρ in [-0.164, -0.038]`, `sim_cal_rec_active` `ρ in [+0.003, +0.168]`).
- **What did hold structurally (`fp32` energy & KL second-order sanity checks, zero simulator dependence):**
  - On `Qwen/Qwen3-1.7B` (`n = 50`, `200` pairs): pooled `EvalEnergy` vs exact `fp32` `KLShare` $\rho = 0.907$ (neuron-clustered 95% CI `[0.855, 0.942]`, pair-level 95% CI `[0.863, 0.936]`, within-band $\rho = 0.782 / 0.636 / 0.873 / 0.581$, 1,000-perm neuron-shuffled null `[0.561, 0.706]`); `59.0%` of exact `fp32` `KLShare` fell below `p99.9+` (`p99.9+ / p97_99.9 / p75_97 / lt_p75` = `0.410 / 0.431 / 0.148 / 0.011`, total `< p99.9+` = `0.590`).
  - On `Qwen/Qwen3-4B` (`n = 278`, `1,112` pairs): pooled `EvalEnergy` vs exact `fp32` `KLShare` $\rho = 0.861$ (neuron-clustered 95% CI `[0.830, 0.887]`, pair-level 95% CI `[0.838, 0.882]`, within-band $\rho = 0.790 / 0.805 / 0.867 / 0.885$, 1,000-perm neuron-shuffled null `[0.045, 0.172]`); `70.7%` of exact `fp32` `KLShare` fell below `p99.9+`.
  - Zero-label `EnergyShare(>=20%)` tracked oracle `strong20_rec` on `1.7B` (`ρ = 0.576 [0.279, 0.775]` on 64 windows, `0.823 [0.696, 0.906]` on the 8k slice).
