# Methodology & Experimental Protocol

## 1. Primary References & Scope

This repository audits two-snippet neuron-description detection against downstream activation-band causal impact in dense SwiGLU transformer MLPs (`Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`).

- **Primary detection task & SwiGLU quadrant baseline:** Nick Cammarata, [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (essay), and Curt Tigges, Matthew Kowal, Thomas McGrath, Owen Lewis, & Nick Cammarata (Goodfire, Nov 2026), [*Neurons as an Interpretable Basis*](https://www.goodfire.com/research/neurons-as-an-interpretable-basis) (paper).
- **Dense causal simulator & Qwen3-4B neuron scores:** Aayan Rizvi, [`Ulto85/neurons-kinda-clean`](https://github.com/Ulto85/neurons-kinda-clean) (commit `85c71b7c2876a38f2a17ee2c6bf3da1ec16f09c3`, Oct 2026, MIT license; verbatim license at [`data/qwen4b/ulto85/LICENSE`](../data/qwen4b/ulto85/LICENSE)).
- **Prior weak-band commentary & ablation results:** Leo Gao, [*Neural networks are not already clean*](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean); Stepan Shabalin & Mateusz Piotrowski, *Comparing Sparsified MLPs and Transcoders: Many Transcoder Features Are Neurons in Disguise* (`arXiv:2610.24521`, 2026).

### Important Protocol Differences vs. Goodfire Table 4

Our detection measurements should not be read as a direct benchmark reproduction of Goodfire's numbers because three experimental parameters differ:
1. **Model & routing:** Goodfire evaluates routed MoE neurons in `Qwen3.6-35B-A3B` (`8/256` active experts, router-weighted activations $a \cdot w_{\text{router}}$); we evaluate dense MLPs in `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` ($w_{\text{router}} \equiv 1$).
2. **Neuron labels & simulator labels:** Goodfire uses its own mode descriptions; our `Qwen3-1.7B` audit uses single-pass `gpt-4.1` pole descriptions generated on held-aside training documents, while our `Qwen3-4B` audit uses Ulto85 descriptions written by Claude Opus (`explainer`) and scored by Claude Sonnet/Opus (`simulator`).
3. **Judge model & batching:** Goodfire uses `Opus 5.5` evaluating batches of 10 snippet pairs per call. We evaluate strictly **1 snippet pair per API call** using `gpt-6.1-terra@none` (primary) and `gpt-6-luna@none` (secondary) at `temperature=0.0`, `reasoning_effort="none"`, `top_logprobs=5`.

---

## 2. SwiGLU Quadrant Definitions & Signed Percentile Bands

For a bias-free SwiGLU MLP neuron $n$ with RMSNorm input $x$:
$$g = w_{\text{gate}, n}^\top x, \qquad u = w_{\text{up}, n}^\top x, \qquad a = \text{SiLU}(g) \cdot u, \qquad y = a \, w_{\text{down}}[:, n]$$

Each token activation falls into one of four SwiGLU quadrants:
- **Q1** ($g > 0, u > 0$, gate open, $a > 0$, positive pole)
- **Q2** ($g > 0, u < 0$, gate open, $a < 0$, negative pole)
- **Q3** ($g < 0, u < 0$, gate closed / double-negative, $a > 0$, positive pole)
- **Q4** ($g < 0, u > 0$, gate closed, $a < 0$, negative pole)

Following Goodfire's mirrored-tail convention (`p97/3`, `p99/1`), percentiles are computed over each neuron's **signed** non-BOS corpus activation distribution $a_{n,t}$, so directional activation $d_{n,s,t} = s \cdot a_{n,t}$ ($s \in \{+1, -1\}$) places top positive firings above $P_{99.9}(a_n)$ and top negative firings below $P_{0.1}(a_n)$.

For downstream causal KL attribution (`KLShare`) and Effect-Weighted Detection Lift (`EWL`), non-BOS tokens are partitioned into four mutually exclusive bands covering 100% of activations:
- `p99.9+`: top $0.1\%$ directional/magnitude band ($|d| \ge P_{99.9}$)
- `p97_99.9`: $[P_{97}, P_{99.9})$
- `p75_97`: $[P_{75}, P_{97})$
- `lt_p75`: $|a| < P_{75}$ (plus any sub-threshold dead-zone tail)

---

## 3. Effect-Weighted Detection Lift (EWL)

Binary 2-alternative forced-choice (2AFC) detection has chance accuracy $0.50$. For neuron $i$ with per-band detection accuracy $\text{acc}_i(b) \in [0, 1]$ and exact `float32` mean-ablation KL share $w_i(b) = \text{KLShare}_i(b)$ (non-negative and normalized across bands $B = \{\text{p99.9+}, \text{p97\_99.9}, \text{p75\_97}, \text{lt\_p75}\}$):

$$\text{EWL}_i = \sum_{b \in B} w_i(b) \cdot 2 \cdot \bigl(\text{acc}_i(b) - 0.5\bigr) \in [-1, +1]$$

- $\text{EWL}_i = 0$ corresponds to chance detection weighted by downstream KL impact; $\text{EWL}_i = +1$ corresponds to $100\%$ accuracy on every band carrying non-zero KL share.
- Missing neuron×band cells (`0` valid clean pairs in a very sparse band) contribute $0$ lift (equivalent to $\text{acc} = 0.5$).
- **Subgroup & merge sensitivity variants (`src/qwen4b_eval/effect_weighted.py`):**
  - Full-distribution EWL (`all 4 bands`) and renormalized $\ge p_{75}$ EWL (restricting weights to $\{\text{p99.9+}, \text{p97\_99.9}, \text{p75\_97}\}$, renormalized to sum to $1$).
  - On `Qwen3-4B`, `p97_99.9` combines our `p99_99.9` and `p97_99` detection cells weighted by each neuron's corpus $L^2$ activation energy from `energy_l2.bands_5` (with equal `50/50` and token-width `0.9/2.0` sensitivity checks also computed). One-sided (unipolar) vs. two-sided (bipolar) neurons are reported separately as well as pooled.
  - On `Qwen3-1.7B`, we also report **no-label-adjusted EWL**, substituting $2 \cdot (\text{acc}_{\text{own}}(b) - \text{acc}_{\text{no\_label}}(b))$ for chance-subtracted lift, and **unweighted chance-subtracted lift** $\frac{1}{4}\sum_b 2(\text{acc}_i(b) - 0.5)$ for comparison.

---

## 4. Two-Snippet Detection Window & Context Controls

### Window Format & Structural Invariants
Every evaluation snippet is a `69`-token interior chunk window constructed as:
$$\text{prefix (60 tokens)} + \text{``<<''} + \text{marked target token (1 token)} + \text{``>>''} + \text{suffix (8 tokens)}$$
Each token is decoded separately with `clean_up_tokenization_spaces=False`. Document chunks require `60 <= target_idx <= chunk_len - 9` (preventing left/right truncation and excluding token position `0` / `<bos>`). In every paired comparison:
- `100%` of firing and distractor snippets have exact length `69` tokens and exact target offset `60`.
- Firing and distractor snippets are sampled from disjoint documents (`pos.doc_id != dist.doc_id`).
- Every `(pole_id, band, distractor_type)` cell has an exact balanced `k/k` Slot-A / Slot-B assignment (`pos_slot`).
- **Context cleanliness & `< p90` distractor strength-overlap limitation:** For weak-band firing snippets (`p75_90`, `p90_97`) on `Qwen3-1.7B` and for all distractors (`dead_zone`, `goodfire_lt_p90`, `nearest_wgate_tier0`), every non-target token in the 69-token window satisfies $s \cdot a_{n,j} < P_{90}(s \cdot a_n)$ (`95.3%`–`97.0%` of raw `p75_90`/`p90_97` candidates violate this before filtering). Conversely, because distractor windows only have to stay below $P_{90}$, on `p25_75` and part of `p75_97` positives the distractor window can contain an unmarked token stronger than the marked positive target token (`dist.window_max_act_dir > pos.act_dir`); near-chance accuracy in those lower bands is partly by design. Measured on `Qwen3-1.7B` (`dead_zone`, `data/snippets.jsonl`), this occurs in **`473/600` (`78.8%`)** pairs overall in `p75_90` (**`378/480` = `78.8%`** on `heldout_80`), compared with **`0/600` (`0.0%`)** in `p90_97`, `p97_99.9`, and `p99.9+`. The `Qwen3-4B` snippet JSONL files are not shipped in this export.

### Distractor & Label Arms (`Qwen3-1.7B`, `n = 50` Neurons / `100` Poles)
- **Corpus & split:** `2,000,073` tokens from `HuggingFaceFW/fineweb-edu` (`sample-10BT`, ODC-By), split strictly by `doc_id % 2` into `train_label` (`1,007,383` tokens) and `eval` (`992,690` tokens), plus 16-token suffix deduplication against training examples.
- **Cohorts:** `40` neurons (`80` poles, `heldout_80`) held out until all evaluation criteria were locked; `10` neurons (`20` poles, `pilot_20`) used for prompt/pipeline checks.
- **Distractor arms:** `dead_zone` (`|a_target| < P25(|a|)` and whole window `< p90`), `goodfire_lt_p90` (`p99.9+` comparison allowing `|a_target| >= P25(|a|)` while whole window `< p90`), and `nearest_wgate_tier0` (strict same-layer `w_gate`-cosine neighbor $m \ne n$ firing at `|a_{m,t}| >= P99.9` with gate open $g_{m,t} > 0$, while target $n$ sits in its dead zone with whole window `< p90`).
- **Label conditions (unified prompt template, no `"strongly"`):** `own` (`gpt-4.1` description), `wrong` (cross-layer donor, boilerplate-stripped `text-embedding-3-small` cosine `<= 0.65`), `no_label` (`"(none provided)"`), `token_only` (`Tokens such as: "<t1>", "<t2>", "<t3>"` from `train_label` `p99.9+`).
- **Unparseables & logprob ties:** Excluded from accuracy numerator and denominator rather than scored as `0` or `0.5` (`0` excluded across all calls).

### Multi-Band Detection & Causal Comparison (`Qwen3-4B` on Ulto85)
- **Neuron & pole scope:** Ulto85's `e3_neurons.json` lists `280` neurons (`560` poles); `278` are scored in `e3_scores.json` (`(12, 1597)` and `(24, 289)` unscored), and `132` weak/negligible poles are inactive/excluded, leaving **`424` active poles** (`132` unipolar neurons + `146` bipolar neurons = `278` scored neurons).
- **Disjoint evaluation text:** `1,536,000` tokens from `HuggingFaceFW/fineweb` (`sample/10BT/000_00000.parquet`, ODC-By), reading rows `60,000..64,789` (strictly disjoint from Ulto85's rows `0..59,999`) for activation extraction and detection snippets (`8,480` `p99.9+` pairs, `8,480` `top100` pairs added post hoc after `295/424` active poles had `p99.9 < 0.20 * max_act`, `4,190` `p97_99` pairs, `2,538` `p99_99.9` pairs, `2,496` `p75_97` pairs, `2,316` `p25_75` pairs).
- **Cross-layer `wrong` and `no_label` baselines on `4B`:** Scored on `2,544` `p99.9+` pairs (`6` pairs/pole, `wrong` donor cosine `<= 0.65`, `0` violations).

---

## 5. Exact `float32` Band-KL Attribution & Statistical Checks

Mean-ablation KL divergence on `Qwen/Qwen3-4B` (`n = 60` pilot and `n = 278` full corpus) and `Qwen/Qwen3-1.7B` (`n = 50`) uses exact IEEE-754 `float32` (`allow_tf32=False`) on `8,192` disjoint tokens (FineWeb rows `65,000+` for `4B`; odd-doc `fineweb-edu` tokens for `1.7B`), after catching a `~4.5e-4` mantissa floor in `bf16` that masked `~1e-6–2e-5` single-neuron KLs. Every `fp32` run verifies `zero_ablation_sanity_kl == 0.00e+00`. Normalized band shares sum to $1.0$ (`median additivity ratio sum_B KL(B) / KL(full) ~ 1.000`).

### Statistical Inference & Required Disclosures
- **Unit of bootstrap (`neuron` vs. `pair`):** All neuron-level means, Spearman $\rho$, Pearson $r$, EWL, and `within_band` KL correlations use **neuron-clustered bootstrap CIs** (`B = 1,000` or `2,000`, `seed = 42`), drawing whole neurons with replacement so paired poles/bands stay clustered. When reporting pooled `4 * N` `(neuron, band)` correlation for the `EnergyShare` vs. `KLShare` sanity check, **both** genuine neuron-clustered bootstrap CIs (`neuron_cluster_all_4_bands`) and naive pair-level CIs (`flattened_neuron_band_pair`) are reported alongside a **1,000-permutation permuted-neuron null** (shuffling neuron rows of `KLShare` while preserving per-band means and marginal distributions).
- **Base rates:** All pass-rate conditioning claims report full base rates across `n = 278` alongside conditional subsets (`267/278` have `sim_cal_rec < 0.50`, `251/278 < 0.25`; `0/43` failing `top100 >= 0.80` reach `sim_cal_rec >= 0.25`).
- **Position bias & lexical overlap:** Both judge order preference (`P(pred = A)`) and `top100` target-token lexical overlap (`30.6%` exact word / `37.9%` case-insensitive substring overlap against Ulto85 labels, computed from the `top100` snippet file `snippets_4b_top100.jsonl`, which is not shipped) are reported directly in [detailed tables](detailed_tables.md) and [`README.md`](../README.md).
