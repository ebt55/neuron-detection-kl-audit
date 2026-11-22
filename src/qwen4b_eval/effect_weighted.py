"""Effect-weighted detection lift, cross-model band comparison, and statistical disclosures.

Computes CPU-only, offline analyses directly from committed JSON/JSONL files:
  - Per-band detection accuracy vs. exact IEEE-754 float32 mean-ablation KL share
    on Qwen/Qwen3-4B (Ulto85/neurons-kinda-clean, MIT) and Qwen/Qwen3-1.7B.
  - Effect-weighted detection lift (EWL) with neuron-level bootstrap 95% CIs,
    two-sided (bipolar) subgroup splits, >= p75 renormalised sensitivity, and
    lift over the no-label floor on Qwen3-1.7B.
  - Side-by-side comparison against Goodfire Table 4 random-text accuracy.
  - Statistical disclosures: within-band and pooled EnergyShare vs. KLShare
    correlations with both genuine neuron-clustered and pair-level bootstrap CIs
    plus a 1,000-permutation permuted-neuron null; base rates; YES/NO logprob
    imputation rates; slot-B preference; and top-100 lexical overlap.
  - Publication figure: figures/main_detection_vs_kl_by_band.png.
  - Manual reading sheet: audit/reading_sheet_12_neurons.md (when snippet JSONLs
    are present on disk).

Zero API calls, zero GPU usage, and zero mutation of pre-existing result files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from qwen4b_eval.analyze_4b import ULTO85_CREDIT_FOOTER

FOUR_BANDS: tuple[str, ...] = ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
GE_P75_BANDS: tuple[str, ...] = ("p99.9+", "p97_99.9", "p75_97")

DEFAULT_KL_278_PATH = Path("results/qwen4b/band_kl_4b_all278.json")
DEFAULT_KL_60_PATH = Path("results/qwen4b/band_kl_4b.json")
DEFAULT_GAP_4B_PATH = Path("results/qwen4b/gap_decomposition_4b.json")
DEFAULT_ACTIVE_POLES_4B_PATH = Path("data/qwen4b/active_poles.json")
DEFAULT_ULTO_NEURONS_PATH = Path("data/qwen4b/ulto85/e3_neurons.json")
DEFAULT_ULTO_SCORES_PATH = Path("data/qwen4b/ulto85/e3_scores.json")
DEFAULT_JUDGEMENTS_4B_PATH = Path("results/qwen4b/judgements_4b.jsonl")
DEFAULT_JUDGEMENTS_4B_TOP100_PATH = Path("results/qwen4b/judgements_4b_top100.jsonl")
DEFAULT_JUDGEMENTS_4B_MULTIBAND_PATH = Path("results/qwen4b/judgements_4b_multiband.jsonl")
DEFAULT_JUDGEMENTS_4B_FPR_PATH = Path("results/qwen4b/judgements_4b_fpr.jsonl")
DEFAULT_SNIPPETS_4B_PATH = Path("data/qwen4b/snippets_4b.jsonl")
DEFAULT_SNIPPETS_4B_TOP100_PATH = Path("data/qwen4b/snippets_4b_top100.jsonl")

DEFAULT_EK_17B_PATH = Path("results/qwen17b_causal/energy_and_kl_17b.json")
DEFAULT_CAUSAL_17B_PATH = Path("results/qwen17b_causal/causal_scores_17b.json")
DEFAULT_JUDGEMENTS_17B_PATH = Path("results/judgements.jsonl")
DEFAULT_SUBBAND_17B_PATH = Path("results/p97_p99_subband_split.json")
DEFAULT_POLES_17B_PATH = Path("data/poles.json")
DEFAULT_SNIPPETS_17B_PATH = Path("data/snippets.jsonl")

DEFAULT_OUTPUT_JSON_PATH = Path("results/qwen4b/effect_weighted.json")
DEFAULT_FIGURE_PATH = Path("figures/main_detection_vs_kl_by_band.png")
DEFAULT_READING_SHEET_PATH = Path("audit/reading_sheet_12_neurons.md")

# Fallback lexical overlap summary recorded when optional snippet JSONLs are omitted.
PRECOMPUTED_TOP100_LEXICAL_OVERLAP: dict[str, Any] = {
    "source": "precomputed_fallback_when_snippets_omitted",
    "n_pairs": 8480,
    "exact_word_ci_len_ge_2_count": 2591,
    "exact_word_ci_len_ge_2_rate": 2591 / 8480,
    "substring_ci_len_ge_2_count": 3210,
    "substring_ci_len_ge_2_rate": 3210 / 8480,
}

# Goodfire Table 4 reference accuracies (random-text distractors).
GOODFIRE_TABLE4_REFERENCE: dict[str, Any] = {
    "source": "Goodfire (2026-11-10), Neurons as an Interpretable Basis, Table 4 (random-text distractors)",
    "url": "https://www.goodfire.com/research/neurons-as-an-interpretable-basis",
    "model": "Qwen3.6-35B-A3B (MoE)",
    "labeler": "Neuron Scientist agent",
    "judge": "Claude Opus 5.5",
    "n_poles": 496,
    "bands": {
        "p97_99": 0.906,
        "p99_99.9": 0.932,
        "p99.9_99.99": 0.952,
        "p99.99_plus": 0.972,
        "top_1000_or_top100": 0.970,
        "p75_97": None,
        "p25_75": None,
    },
}


# ---------------------------------------------------------------------------
# Statistical helpers (warning-free under -W error::RuntimeWarning)
# ---------------------------------------------------------------------------


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Spearman rank correlation safely, returning (nan, nan) on constant inputs."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 3 or np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return float("nan"), float("nan")
    res = stats.spearmanr(xv, yv)
    return float(res.statistic), float(res.pvalue)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation safely, returning (nan, nan) on constant inputs."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 3 or np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return float("nan"), float("nan")
    r, p = stats.pearsonr(xv, yv)
    return float(r), float(p)


def _quantile_ci(samples: np.ndarray) -> tuple[float | None, float | None]:
    """Return [2.5%, 97.5%] percentile interval over finite samples."""
    fin = samples[np.isfinite(samples)]
    if fin.size == 0:
        return None, None
    lo, hi = np.percentile(fin, [2.5, 97.5])
    return float(lo), float(hi)


def neuron_bootstrap_mean(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute mean, median, std, and neuron-level bootstrap 95% CI."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_unit": "neuron",
            "n_bootstrap": int(n_bootstrap),
        }
    mean_val = float(np.mean(arr))
    med_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n == 1:
        ci_lo = ci_hi = mean_val
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(int(n_bootstrap), n))
        boot_means = arr[idx].mean(axis=1)
        ci_lo, ci_hi = _quantile_ci(boot_means)
    return {
        "n": n,
        "mean": mean_val,
        "median": med_val,
        "std": std_val,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "bootstrap_unit": "neuron",
        "n_bootstrap": int(n_bootstrap),
    }


def neuron_bootstrap_corr(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Spearman rho and Pearson r with neuron-level bootstrap 95% CIs."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xv = xa[mask]
    yv = ya[mask]
    n = int(xv.size)
    rho, rho_p = _safe_spearman(xv, yv)
    r, r_p = _safe_pearson(xv, yv)

    rng = np.random.default_rng(seed)
    rhos = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    rs = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    if n >= 3:
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            xb = xv[idx]
            yb = yv[idx]
            if np.all(xb == xb[0]) or np.all(yb == yb[0]):
                continue
            rhos[b] = float(stats.spearmanr(xb, yb).statistic)
            rs[b] = float(stats.pearsonr(xb, yb)[0])

    rho_lo, rho_hi = _quantile_ci(rhos)
    r_lo, r_hi = _quantile_ci(rs)
    return {
        "n": n,
        "spearman_rho": None if math.isnan(rho) else rho,
        "spearman_p": None if math.isnan(rho_p) else rho_p,
        "spearman_ci_low": rho_lo,
        "spearman_ci_high": rho_hi,
        "pearson_r": None if math.isnan(r) else r,
        "pearson_p": None if math.isnan(r_p) else r_p,
        "pearson_ci_low": r_lo,
        "pearson_ci_high": r_hi,
        "bootstrap_unit": "neuron",
        "n_bootstrap": int(n_bootstrap),
    }


def _fast_pearson_batch(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r across 2D arrays of shape (B, M)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = np.sum(Xc * Yc, axis=1)
    den = np.sqrt(np.sum(Xc * Xc, axis=1) * np.sum(Yc * Yc, axis=1))
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    valid = den > 0.0
    out[valid] = num[valid] / den[valid]
    return out


def _fast_spearman_row(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman rank correlation on 1D finite arrays with tie handling."""
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    den = float(np.sqrt(np.dot(rxc, rxc) * np.dot(ryc, ryc)))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(rxc, ryc) / den)


def pooled_energy_kl_analysis(
    E_mat: np.ndarray,
    K_mat: np.ndarray,
    band_names: tuple[str, ...] = FOUR_BANDS,
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Second-order EnergyShare vs. KLShare sanity check with clustered & pair CIs + permuted-neuron null.

    Parameters
    ----------
    E_mat, K_mat : ndarray of shape (N_neurons, N_bands)
        Per-neuron band energy shares and band KL shares.
    """
    E = np.asarray(E_mat, dtype=np.float64)
    K = np.asarray(K_mat, dtype=np.float64)
    if E.shape != K.shape or E.ndim != 2:
        raise ValueError(f"Expected matching 2D matrices, got {E.shape} and {K.shape}")
    n_neurons, n_bands = E.shape
    if n_bands != len(band_names):
        raise ValueError(f"band_names length {len(band_names)} != {n_bands}")

    # 1. Within-band correlations (strictly neuron-level)
    within_band: dict[str, Any] = {}
    for bi, bname in enumerate(band_names):
        within_band[bname] = neuron_bootstrap_corr(
            E[:, bi], K[:, bi], n_bootstrap=n_bootstrap, seed=seed + 100 + bi
        )

    # 2. Pooled point estimates across N * 4 (neuron, band) pairs
    e_flat = E.ravel()
    k_flat = K.ravel()
    obs_rho, obs_rho_p = _safe_spearman(e_flat, k_flat)
    obs_r, obs_r_p = _safe_pearson(e_flat, k_flat)

    # 3. Genuine neuron-clustered bootstrap CI (resample N whole neurons with replacement)
    rng_cl = np.random.default_rng(seed + 1000)
    cl_rhos = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    cl_rs = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    if n_neurons >= 3:
        for b in range(int(n_bootstrap)):
            idx = rng_cl.integers(0, n_neurons, size=n_neurons)
            eb = E[idx].ravel()
            kb = K[idx].ravel()
            cl_rhos[b] = _fast_spearman_row(eb, kb)
            r_val, _ = _safe_pearson(eb, kb)
            cl_rs[b] = r_val
    cl_rho_lo, cl_rho_hi = _quantile_ci(cl_rhos)
    cl_r_lo, cl_r_hi = _quantile_ci(cl_rs)

    # 4. Pair-level bootstrap CI (resample 4*N flattened pairs independently)
    rng_pr = np.random.default_rng(seed + 2000)
    n_pairs = int(e_flat.size)
    pr_rhos = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    pr_rs = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    if n_pairs >= 3:
        for b in range(int(n_bootstrap)):
            idx = rng_pr.integers(0, n_pairs, size=n_pairs)
            eb = e_flat[idx]
            kb = k_flat[idx]
            pr_rhos[b] = _fast_spearman_row(eb, kb)
            r_val, _ = _safe_pearson(eb, kb)
            pr_rs[b] = r_val
    pr_rho_lo, pr_rho_hi = _quantile_ci(pr_rhos)
    pr_r_lo, pr_r_hi = _quantile_ci(pr_rs)

    # 5. Permuted-neuron null (1,000 permutations): permute neuron rows of K, preserving band means
    rng_null = np.random.default_rng(seed + 3000)
    null_rhos = np.full(int(n_permutations), np.nan, dtype=np.float64)
    null_rs = np.full(int(n_permutations), np.nan, dtype=np.float64)
    if n_neurons >= 3:
        for p_idx in range(int(n_permutations)):
            perm = rng_null.permutation(n_neurons)
            kp = K[perm].ravel()
            null_rhos[p_idx] = _fast_spearman_row(e_flat, kp)
            r_val, _ = _safe_pearson(e_flat, kp)
            null_rs[p_idx] = r_val
    null_rho_fin = null_rhos[np.isfinite(null_rhos)]
    null_r_fin = null_rs[np.isfinite(null_rs)]
    null_rho_lo, null_rho_hi = _quantile_ci(null_rhos)
    null_r_lo, null_r_hi = _quantile_ci(null_rs)

    p_emp_rho = (
        float((1 + np.sum(null_rho_fin >= obs_rho)) / (null_rho_fin.size + 1))
        if (null_rho_fin.size > 0 and not math.isnan(obs_rho))
        else None
    )
    p_emp_r = (
        float((1 + np.sum(null_r_fin >= obs_r)) / (null_r_fin.size + 1))
        if (null_r_fin.size > 0 and not math.isnan(obs_r))
        else None
    )

    return {
        "n_neurons": int(n_neurons),
        "n_bands": int(n_bands),
        "n_pairs": int(n_pairs),
        "within_band": within_band,
        "pooled": {
            "spearman_rho": None if math.isnan(obs_rho) else obs_rho,
            "spearman_p_naive_pair": None if math.isnan(obs_rho_p) else obs_rho_p,
            "pearson_r": None if math.isnan(obs_r) else obs_r,
            "pearson_p_naive_pair": None if math.isnan(obs_r_p) else obs_r_p,
            "neuron_clustered_bootstrap_ci": {
                "bootstrap_unit": "neuron_cluster_all_4_bands",
                "n_bootstrap": int(n_bootstrap),
                "spearman_ci_low": cl_rho_lo,
                "spearman_ci_high": cl_rho_hi,
                "pearson_ci_low": cl_r_lo,
                "pearson_ci_high": cl_r_hi,
            },
            "pair_level_bootstrap_ci": {
                "bootstrap_unit": "flattened_neuron_band_pair",
                "n_bootstrap": int(n_bootstrap),
                "spearman_ci_low": pr_rho_lo,
                "spearman_ci_high": pr_rho_hi,
                "pearson_ci_low": pr_r_lo,
                "pearson_ci_high": pr_r_hi,
            },
            "permuted_neuron_null": {
                "description": (
                    "Randomly permute neuron rows of KLShare matrix (preserving exact per-band "
                    "marginal distributions and band means) and recompute pooled correlation."
                ),
                "n_permutations": int(n_permutations),
                "seed": int(seed + 3000),
                "spearman_mean": float(np.mean(null_rho_fin)) if null_rho_fin.size else None,
                "spearman_std": float(np.std(null_rho_fin, ddof=1)) if null_rho_fin.size > 1 else None,
                "spearman_q025": null_rho_lo,
                "spearman_q975": null_rho_hi,
                "spearman_p_empirical_ge_obs": p_emp_rho,
                "pearson_mean": float(np.mean(null_r_fin)) if null_r_fin.size else None,
                "pearson_std": float(np.std(null_r_fin, ddof=1)) if null_r_fin.size > 1 else None,
                "pearson_q025": null_r_lo,
                "pearson_q975": null_r_hi,
                "pearson_p_empirical_ge_obs": p_emp_r,
            },
        },
    }


def compute_neuron_ewl(
    kl_shares: dict[str, float],
    band_accs: dict[str, float | None],
    bands: tuple[str, ...] = FOUR_BANDS,
    renormalize: bool = False,
) -> float:
    """Compute single-neuron Effect-Weighted Lift: sum_b w(b) * 2 * (acc(b) - 0.5).

    Missing band accuracies (None or NaN) contribute 0 lift (equivalent to acc = 0.5).
    When renormalize=True, weights are divided by sum_{b in bands} kl_shares[b].
    """
    raw_weights = [float(kl_shares.get(b, 0.0)) for b in bands]
    denom = sum(raw_weights)
    if renormalize:
        if denom <= 0.0:
            return 0.0
        weights = [w / denom for w in raw_weights]
    else:
        weights = raw_weights
    total = 0.0
    for b, w in zip(bands, weights):
        acc = band_accs.get(b)
        if acc is None or not math.isfinite(float(acc)):
            lift = 0.0
        else:
            lift = 2.0 * (float(acc) - 0.5)
        total += w * lift
    return float(total)


def compute_neuron_ewl_over_floor(
    kl_shares: dict[str, float],
    own_accs: dict[str, float | None],
    floor_accs: dict[str, float | None],
    bands: tuple[str, ...] = GE_P75_BANDS,
    renormalize: bool = False,
    scale: float = 1.0,
) -> float:
    """Compute scale * sum_b w(b) * (own(b) - floor(b)) over specified bands."""
    raw_weights = [float(kl_shares.get(b, 0.0)) for b in bands]
    denom = sum(raw_weights)
    if renormalize:
        if denom <= 0.0:
            return 0.0
        weights = [w / denom for w in raw_weights]
    else:
        weights = raw_weights
    total = 0.0
    for b, w in zip(bands, weights):
        own_v = own_accs.get(b)
        fl_v = floor_accs.get(b)
        if own_v is None or fl_v is None:
            diff = 0.0
        else:
            diff = float(own_v) - float(fl_v)
        total += w * diff
    return float(scale * total)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_qwen4b_detection_tables(
    active_poles_path: Path,
    judgements_4b_path: Path,
    judgements_top100_path: Path,
    judgements_multiband_path: Path,
    judge_model: str = "gpt-6.1-terra@none",
) -> dict[str, Any]:
    """Load Qwen3-4B scored detection judgements aggregated at both pole and neuron levels."""
    ap_doc = _load_json(active_poles_path)
    poles = ap_doc["poles"]
    neuron_to_poles: dict[int, list[str]] = {}
    pole_to_k: dict[str, int] = {}
    pole_info: dict[str, dict[str, Any]] = {}
    for p in poles:
        k = int(p["k"])
        pid = str(p["pole_id"])
        neuron_to_poles.setdefault(k, []).append(pid)
        pole_to_k[pid] = k
        pole_info[pid] = p

    # Collect (condition, band, pole_id) -> list of binary correct values, plus slot stats
    pole_band_scores: dict[tuple[str, str], dict[str, list[int]]] = {}
    slot_a_counts: dict[tuple[str, str, str], list[int]] = {}  # (judge, band, cond) -> [a_count, total]

    for jpath in (judgements_4b_path, judgements_top100_path, judgements_multiband_path):
        if not jpath.is_file():
            continue
        for row in _iter_jsonl(jpath):
            if row.get("outcome") != "scored":
                continue
            jm = str(row["judge_model"])
            band = str(row["band"])
            cond = str(row["condition"])
            slot_key = (jm, band, cond)
            slot_a_counts.setdefault(slot_key, [0, 0])
            slot_a_counts[slot_key][1] += 1
            if row.get("predicted_slot") == "A":
                slot_a_counts[slot_key][0] += 1

            if jm != judge_model:
                continue
            pid = str(row["pole_id"])
            corr = 1 if bool(row["correct"]) else 0
            pole_band_scores.setdefault((cond, band), {}).setdefault(pid, []).append(corr)

    raw_bands = ("top100", "p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75")
    pole_means: dict[str, dict[str, float]] = {}
    neuron_means: dict[str, dict[int, float | None]] = {}
    for b in raw_bands:
        p_map = pole_band_scores.get(("own", b), {})
        pole_means[b] = {pid: float(np.mean(vals)) for pid, vals in p_map.items() if vals}
        n_map: dict[int, float | None] = {}
        for k, pids in neuron_to_poles.items():
            vals = [pole_means[b][pid] for pid in pids if pid in pole_means[b]]
            n_map[k] = float(np.mean(vals)) if vals else None
        neuron_means[b] = n_map

    # Also store p99.9_plus wrong / no_label pole & neuron means for completeness
    for floor_cond in ("no_label", "wrong"):
        p_map = pole_band_scores.get((floor_cond, "p99.9_plus"), {})
        pole_means[f"p99.9_plus_{floor_cond}"] = {
            pid: float(np.mean(vals)) for pid, vals in p_map.items() if vals
        }
        neuron_means[f"p99.9_plus_{floor_cond}"] = {
            k: (
                float(np.mean([pole_means[f"p99.9_plus_{floor_cond}"][pid] for pid in pids if pid in pole_means[f"p99.9_plus_{floor_cond}"]]))
                if any(pid in pole_means[f"p99.9_plus_{floor_cond}"] for pid in pids)
                else None
            )
            for k, pids in neuron_to_poles.items()
        }

    return {
        "neuron_to_poles": neuron_to_poles,
        "pole_to_k": pole_to_k,
        "pole_info": pole_info,
        "pole_means": pole_means,
        "neuron_means": neuron_means,
        "slot_a_counts": {
            f"{jm}__{band}__{cond}": {
                "pred_a_count": a_cnt,
                "total": tot,
                "p_pred_a": float(a_cnt / tot) if tot else None,
            }
            for (jm, band, cond), (a_cnt, tot) in sorted(slot_a_counts.items())
        },
    }


def merge_4b_p97_99_9_accuracy(
    acc_p99_999: float | None,
    acc_p97_99: float | None,
    w_p99_999: float,
    w_p97_99: float,
    mode: str = "energy",
) -> float | None:
    """Merge p99_99.9 and p97_99 detection accuracies into p97_99.9.

    mode == "energy": corpus L2-energy-weighted merge using bands_5 weights.
    mode == "equal": 50/50 unweighted merge.
    If only one sub-band has scored pairs, uses that sub-band cleanly.
    """
    has_hi = acc_p99_999 is not None and math.isfinite(acc_p99_999)
    has_lo = acc_p97_99 is not None and math.isfinite(acc_p97_99)
    if not has_hi and not has_lo:
        return None
    if has_hi and not has_lo:
        return float(acc_p99_999)
    if has_lo and not has_hi:
        return float(acc_p97_99)
    assert acc_p99_999 is not None and acc_p97_99 is not None
    if mode == "equal":
        return float(0.5 * acc_p99_999 + 0.5 * acc_p97_99)
    denom = float(w_p99_999 + w_p97_99)
    if denom <= 0.0:
        return float(0.5 * acc_p99_999 + 0.5 * acc_p97_99)
    return float((w_p99_999 * acc_p99_999 + w_p97_99 * acc_p97_99) / denom)


def _compute_4b_subset_summary(
    ks: list[int],
    kl_per_neuron: dict[str, dict[str, Any]],
    gap_per_neuron: dict[str, dict[str, Any]],
    det_tables: dict[str, Any],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute band accuracies, KL shares, EWL variants, and missing counts for a 4B neuron subset."""
    n_neurons = len(ks)
    neuron_means = det_tables["neuron_means"]

    per_band_det_only: dict[str, list[float]] = {b: [] for b in FOUR_BANDS}
    per_band_imputed: dict[str, list[float]] = {b: [] for b in FOUR_BANDS}
    per_band_equal_p97_det_only: list[float] = []
    per_band_kl: dict[str, list[float]] = {b: [] for b in FOUR_BANDS}
    missing_counts: dict[str, int] = {b: 0 for b in FOUR_BANDS}
    missing_subbands = {"p99_99.9": 0, "p97_99": 0}

    ewl_full_energy: list[float] = []
    ewl_full_equal: list[float] = []
    ewl_full_token_width: list[float] = []
    ewl_ge75_energy: list[float] = []
    ewl_ge75_equal: list[float] = []
    ewl_top_contrib: list[float] = []
    ewl_below_top_contrib: list[float] = []
    top_p999_lift: list[float] = []
    top100_lift: list[float] = []
    top100_acc_vals: list[float] = []
    p99_999_weights_share: list[float] = []
    opp_sign_eval_shares: list[float] = []
    corp_dead_or_opp_shares: list[float] = []
    corp_p25_75_shares: list[float] = []

    for k in ks:
        ks_str = str(k)
        kl_rec = kl_per_neuron[ks_str]
        gap_rec = gap_per_neuron[ks_str]
        kl_sh = kl_rec["kl_share_4band_norm"]

        b5 = gap_rec["energy_l2"]["bands_5"]
        w_hi = float(b5["p99_99.9"])
        w_lo = float(b5["p97_99"])
        if w_hi + w_lo > 0.0:
            p99_999_weights_share.append(w_hi / (w_hi + w_lo))

        acc_p999 = neuron_means["p99.9_plus"].get(k)
        acc_p99_999 = neuron_means["p99_99.9"].get(k)
        acc_p97_99 = neuron_means["p97_99"].get(k)
        acc_p75_97 = neuron_means["p75_97"].get(k)
        acc_p25_75 = neuron_means["p25_75"].get(k)
        acc_top100 = neuron_means["top100"].get(k)

        if acc_p99_999 is None:
            missing_subbands["p99_99.9"] += 1
        if acc_p97_99 is None:
            missing_subbands["p97_99"] += 1

        acc_p97_999_energy = merge_4b_p97_99_9_accuracy(
            acc_p99_999, acc_p97_99, w_hi, w_lo, mode="energy"
        )
        acc_p97_999_equal = merge_4b_p97_99_9_accuracy(
            acc_p99_999, acc_p97_99, w_hi, w_lo, mode="equal"
        )
        acc_p97_999_token_width = merge_4b_p97_99_9_accuracy(
            acc_p99_999, acc_p97_99, 0.9, 2.0, mode="energy"
        )

        accs_energy = {
            "p99.9+": acc_p999,
            "p97_99.9": acc_p97_999_energy,
            "p75_97": acc_p75_97,
            "lt_p75": acc_p25_75,
        }
        accs_equal = {
            "p99.9+": acc_p999,
            "p97_99.9": acc_p97_999_equal,
            "p75_97": acc_p75_97,
            "lt_p75": acc_p25_75,
        }
        accs_token_width = {
            "p99.9+": acc_p999,
            "p97_99.9": acc_p97_999_token_width,
            "p75_97": acc_p75_97,
            "lt_p75": acc_p25_75,
        }

        if acc_p97_999_equal is not None:
            per_band_equal_p97_det_only.append(acc_p97_999_equal)

        for b in FOUR_BANDS:
            per_band_kl[b].append(float(kl_sh[b]))
            val = accs_energy[b]
            if val is None or not math.isfinite(val):
                missing_counts[b] += 1
                per_band_imputed[b].append(0.5)
            else:
                per_band_det_only[b].append(float(val))
                per_band_imputed[b].append(float(val))

        ewl_e = compute_neuron_ewl(kl_sh, accs_energy, FOUR_BANDS, renormalize=False)
        ewl_q = compute_neuron_ewl(kl_sh, accs_equal, FOUR_BANDS, renormalize=False)
        ewl_tw = compute_neuron_ewl(kl_sh, accs_token_width, FOUR_BANDS, renormalize=False)
        ewl_ge_e = compute_neuron_ewl(kl_sh, accs_energy, GE_P75_BANDS, renormalize=True)
        ewl_ge_q = compute_neuron_ewl(kl_sh, accs_equal, GE_P75_BANDS, renormalize=True)

        top_term = float(kl_sh["p99.9+"]) * (
            2.0 * (float(acc_p999) - 0.5) if acc_p999 is not None else 0.0
        )
        ewl_full_energy.append(ewl_e)
        ewl_full_equal.append(ewl_q)
        ewl_full_token_width.append(ewl_tw)
        ewl_ge75_energy.append(ewl_ge_e)
        ewl_ge75_equal.append(ewl_ge_q)
        ewl_top_contrib.append(top_term)
        ewl_below_top_contrib.append(ewl_e - top_term)

        top_p999_lift.append(2.0 * (float(acc_p999) - 0.5) if acc_p999 is not None else 0.0)
        if acc_top100 is not None:
            top100_acc_vals.append(float(acc_top100))
            top100_lift.append(2.0 * (float(acc_top100) - 0.5))

        if kl_rec.get("opposite_sign_eval_energy_share") is not None:
            opp_sign_eval_shares.append(float(kl_rec["opposite_sign_eval_energy_share"]))
        bw = gap_rec["energy_l2"].get("bands_with_p25_75", {})
        if "dead_or_opp" in bw:
            corp_dead_or_opp_shares.append(float(bw["dead_or_opp"]))
        if "p25_75" in bw:
            corp_p25_75_shares.append(float(bw["p25_75"]))

    bands_summary: dict[str, Any] = {}
    for bi, b in enumerate(FOUR_BANDS):
        bands_summary[b] = {
            "kl_share": neuron_bootstrap_mean(per_band_kl[b], n_bootstrap=n_bootstrap, seed=seed + 10 + bi),
            "det_acc_detected_only": neuron_bootstrap_mean(
                per_band_det_only[b], n_bootstrap=n_bootstrap, seed=seed + 20 + bi
            ),
            "det_acc_imputed_0_5": neuron_bootstrap_mean(
                per_band_imputed[b], n_bootstrap=n_bootstrap, seed=seed + 30 + bi
            ),
            "n_missing_neurons": int(missing_counts[b]),
        }
    bands_summary["p97_99.9"]["det_acc_equal_weight_detected_only"] = neuron_bootstrap_mean(
        per_band_equal_p97_det_only, n_bootstrap=n_bootstrap, seed=seed + 41
    )
    bands_summary["p97_99.9"]["mean_energy_weight_on_p99_99.9"] = (
        float(np.mean(p99_999_weights_share)) if p99_999_weights_share else None
    )

    ewl_full_stat = neuron_bootstrap_mean(ewl_full_energy, n_bootstrap=n_bootstrap, seed=seed + 50)
    top_lift_stat = neuron_bootstrap_mean(top_p999_lift, n_bootstrap=n_bootstrap, seed=seed + 55)
    top100_lift_stat = neuron_bootstrap_mean(top100_lift, n_bootstrap=n_bootstrap, seed=seed + 56)

    ratio_vs_p999 = (
        float(ewl_full_stat["mean"] / top_lift_stat["mean"])
        if (ewl_full_stat["mean"] is not None and top_lift_stat["mean"] and top_lift_stat["mean"] != 0.0)
        else None
    )
    ratio_vs_top100 = (
        float(ewl_full_stat["mean"] / top100_lift_stat["mean"])
        if (ewl_full_stat["mean"] is not None and top100_lift_stat["mean"] and top100_lift_stat["mean"] != 0.0)
        else None
    )

    arr_ewl = np.asarray(ewl_full_energy, dtype=np.float64)
    if arr_ewl.size > 0:
        ewl_q25, ewl_med, ewl_q75 = (float(x) for x in np.quantile(arr_ewl, [0.25, 0.50, 0.75]))
        ewl_n_neg = int(np.sum(arr_ewl < 0.0))
        ewl_neg_share = float(ewl_n_neg / arr_ewl.size)
    else:
        ewl_q25 = ewl_med = ewl_q75 = float("nan")
        ewl_n_neg = 0
        ewl_neg_share = float("nan")

    return {
        "n_neurons": n_neurons,
        "bands": bands_summary,
        "top100_acc": neuron_bootstrap_mean(top100_acc_vals, n_bootstrap=n_bootstrap, seed=seed + 49),
        "missing_subbands": missing_subbands,
        "ewl_full_energy_merge": ewl_full_stat,
        "ewl_full_equal_merge": neuron_bootstrap_mean(ewl_full_equal, n_bootstrap=n_bootstrap, seed=seed + 51),
        "ewl_full_token_width_merge": neuron_bootstrap_mean(
            ewl_full_token_width, n_bootstrap=n_bootstrap, seed=seed + 58
        ),
        "ewl_ge_p75_renorm_energy_merge": neuron_bootstrap_mean(
            ewl_ge75_energy, n_bootstrap=n_bootstrap, seed=seed + 52
        ),
        "ewl_ge_p75_renorm_equal_merge": neuron_bootstrap_mean(
            ewl_ge75_equal, n_bootstrap=n_bootstrap, seed=seed + 53
        ),
        "ewl_top_band_contribution": neuron_bootstrap_mean(ewl_top_contrib, n_bootstrap=n_bootstrap, seed=seed + 54),
        "ewl_below_top_contribution": neuron_bootstrap_mean(
            ewl_below_top_contrib, n_bootstrap=n_bootstrap, seed=seed + 57
        ),
        "ewl_distribution": {
            "n_neurons": int(arr_ewl.size),
            "median": ewl_med,
            "q25": ewl_q25,
            "q75": ewl_q75,
            "iqr": [ewl_q25, ewl_q75],
            "n_negative": ewl_n_neg,
            "negative_share": ewl_neg_share,
        },
        "top_p999_lift": top_lift_stat,
        "top100_lift": top100_lift_stat,
        "ewl_ratio_to_p999_lift": ratio_vs_p999,
        "ewl_ratio_to_top100_lift": ratio_vs_top100,
        "lt_p75_diagnostics": {
            "opposite_sign_eval_energy_share": (
                neuron_bootstrap_mean(opp_sign_eval_shares, n_bootstrap=n_bootstrap, seed=seed + 60)
                if opp_sign_eval_shares
                else None
            ),
            "corpus_dead_or_opp_energy_share": neuron_bootstrap_mean(
                corp_dead_or_opp_shares, n_bootstrap=n_bootstrap, seed=seed + 61
            ),
            "corpus_p25_75_energy_share": neuron_bootstrap_mean(
                corp_p25_75_shares, n_bootstrap=n_bootstrap, seed=seed + 62
            ),
        },
    }


def analyze_qwen4b(
    kl_path: Path,
    kl_n60_path: Path | None,
    gap_path: Path,
    active_poles_path: Path,
    det_tables: dict[str, Any],
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full Qwen3-4B per-band KL vs. detection and EWL analysis."""
    kl_doc = _load_json(kl_path)
    gap_doc = _load_json(gap_path)
    kl_per_neuron = kl_doc["per_neuron"]
    gap_per_neuron = gap_doc["per_neuron"]

    ks_all = sorted(int(k) for k in kl_per_neuron.keys())
    ks_bipolar = [
        k for k in ks_all
        if bool(kl_per_neuron[str(k)].get("is_bipolar", gap_per_neuron[str(k)]["is_bipolar"]))
    ]
    ks_unipolar = [k for k in ks_all if k not in set(ks_bipolar)]

    all_summary = _compute_4b_subset_summary(
        ks_all, kl_per_neuron, gap_per_neuron, det_tables, n_bootstrap=n_bootstrap, seed=seed
    )
    bipolar_summary = _compute_4b_subset_summary(
        ks_bipolar, kl_per_neuron, gap_per_neuron, det_tables, n_bootstrap=n_bootstrap, seed=seed + 200
    )
    unipolar_summary = _compute_4b_subset_summary(
        ks_unipolar, kl_per_neuron, gap_per_neuron, det_tables, n_bootstrap=n_bootstrap, seed=seed + 400
    )

    # All-278 detection missing cell counts (independent of KL subset size)
    all278_ks = sorted(int(k) for k in gap_per_neuron.keys())
    missing_all278: dict[str, int] = {}
    for raw_b in ("top100", "p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75"):
        missing_all278[raw_b] = sum(
            1 for k in all278_ks if det_tables["neuron_means"][raw_b].get(k) is None
        )

    # Second-order sanity check: EvalEnergy & CorpEnergy vs KLShare
    E_eval = np.array(
        [[float(kl_per_neuron[str(k)]["eval_energy_4band"][b]) for b in FOUR_BANDS] for k in ks_all],
        dtype=np.float64,
    )
    E_corp = np.array(
        [[float(kl_per_neuron[str(k)]["corpus_energy_4band"][b]) for b in FOUR_BANDS] for k in ks_all],
        dtype=np.float64,
    )
    K_mat = np.array(
        [[float(kl_per_neuron[str(k)]["kl_share_4band_norm"][b]) for b in FOUR_BANDS] for k in ks_all],
        dtype=np.float64,
    )

    eval_vs_kl = pooled_energy_kl_analysis(
        E_eval, K_mat, n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed + 500
    )
    corp_vs_kl = pooled_energy_kl_analysis(
        E_corp, K_mat, n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed + 600
    )
    ge20_eval_vs_kl = neuron_bootstrap_corr(
        [float(kl_per_neuron[str(k)]["eval_energy_ge_20pct_max"]) for k in ks_all],
        [float(kl_per_neuron[str(k)]["kl_share_ge_20pct_max_norm"]) for k in ks_all],
        n_bootstrap=n_bootstrap,
        seed=seed + 700,
    )
    ge20_corp_vs_kl = neuron_bootstrap_corr(
        [float(kl_per_neuron[str(k)]["corpus_energy_ge_20pct_max"]) for k in ks_all],
        [float(kl_per_neuron[str(k)]["kl_share_ge_20pct_max_norm"]) for k in ks_all],
        n_bootstrap=n_bootstrap,
        seed=seed + 701,
    )
    gc_eval_vs_kl = neuron_bootstrap_corr(
        [float(kl_per_neuron[str(k)]["eval_energy_gate_closed"]) for k in ks_all],
        [float(kl_per_neuron[str(k)]["kl_share_gate_closed_norm"]) for k in ks_all],
        n_bootstrap=n_bootstrap,
        seed=seed + 702,
    )
    gc_corp_vs_kl = neuron_bootstrap_corr(
        [float(kl_per_neuron[str(k)]["corpus_energy_gate_closed"]) for k in ks_all],
        [float(kl_per_neuron[str(k)]["kl_share_gate_closed_norm"]) for k in ks_all],
        n_bootstrap=n_bootstrap,
        seed=seed + 703,
    )

    # n=60 comparison block (especially informative when kl_path is the full 278-neuron file)
    n60_comparison: dict[str, Any] | None = None
    if kl_n60_path is not None and kl_n60_path.is_file():
        n60_doc = _load_json(kl_n60_path)
        ks_60 = sorted(int(k) for k in n60_doc["per_neuron"].keys())
        n60_pass = sum(
            1 for k in ks_60 if float(gap_per_neuron[str(k)]["sim_cal_rec"]) >= 0.50
        )
        n278_pass = sum(
            1 for k in all278_ks if float(gap_per_neuron[str(k)]["sim_cal_rec"]) >= 0.50
        )
        n60_sub = (
            all_summary
            if len(ks_all) == len(ks_60) and ks_all == ks_60
            else _compute_4b_subset_summary(
                ks_60, n60_doc["per_neuron"], gap_per_neuron, det_tables,
                n_bootstrap=n_bootstrap, seed=seed + 800,
            )
        )
        n60_comparison = {
            "n_neurons": len(ks_60),
            "sim_ge_0_50_in_n60": n60_pass,
            "sim_ge_0_50_in_n60_rate": float(n60_pass / len(ks_60)),
            "sim_ge_0_50_in_all278": n278_pass,
            "sim_ge_0_50_in_all278_rate": float(n278_pass / len(all278_ks)),
            "over_representation_note": (
                f"Stratified n=60 subset contains {n60_pass}/60 ({100.0 * n60_pass / len(ks_60):.1f}%) "
                f"neurons with sim_cal_rec >= 0.50 vs {n278_pass}/278 ({100.0 * n278_pass / len(all278_ks):.1f}%) "
                "across all 278 neurons."
            ),
            "ewl_full_energy_merge": n60_sub["ewl_full_energy_merge"],
            "ewl_ge_p75_renorm_energy_merge": n60_sub["ewl_ge_p75_renorm_energy_merge"],
            "top_p999_lift": n60_sub["top_p999_lift"],
            "top100_lift": n60_sub["top100_lift"],
        }

    return {
        "kl_source_file": kl_path.as_posix(),
        "n_kl_neurons": len(ks_all),
        "n_bipolar_in_kl": len(ks_bipolar),
        "n_unipolar_in_kl": len(ks_unipolar),
        "zero_ablation_sanity_kl": float(kl_doc.get("zero_ablation_sanity_kl", 0.0)),
        "kl_4band_additivity_ratio_median": float(kl_doc.get("kl_4band_additivity_ratio_median", float("nan"))),
        "band_mapping": {
            "p99.9+": "p99.9_plus detection accuracy (20 pairs/pole, gpt-6.1-terra@none, own label)",
            "p97_99.9": (
                "Corpus L2-energy-weighted merge of p99_99.9 and p97_99 using energy_l2.bands_5 "
                "(equal-weight 50/50 sensitivity also reported)"
            ),
            "p75_97": "p75_97 detection accuracy (6 pairs/pole, gpt-6.1-terra@none, own label)",
            "lt_p75": (
                "p25_75 detection accuracy (6 pairs/pole); < p25 is untested, and on 1-sided "
                "(unipolar) neurons the < p75 KL band includes the excluded opposite-sign activation side"
            ),
        },
        "missing_raw_band_cells_all278": missing_all278,
        "all_kl_neurons": all_summary,
        "bipolar_only": bipolar_summary,
        "unipolar_only": unipolar_summary,
        "n60_subset_comparison": n60_comparison,
        "energy_vs_kl_sanity_check": {
            "eval_energy_vs_kl_share": eval_vs_kl,
            "corpus_energy_vs_kl_share": corp_vs_kl,
            "ge_20pct_max_eval_vs_kl": ge20_eval_vs_kl,
            "ge_20pct_max_corpus_vs_kl": ge20_corp_vs_kl,
            "gate_closed_eval_vs_kl": gc_eval_vs_kl,
            "gate_closed_corpus_vs_kl": gc_corp_vs_kl,
        },
    }


# ---------------------------------------------------------------------------
# Qwen3-1.7B analysis (A.3 + Item B sanity check)
# ---------------------------------------------------------------------------


def _compute_17b_subset_summary(
    neuron_ids: list[str],
    ek_per_neuron: dict[str, dict[str, Any]],
    own_by_nb: dict[tuple[str, str], float],
    nolabel_by_nb: dict[tuple[str, str], float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Qwen3-1.7B band accuracies, KL shares, EWL, and lift over no_label floor."""
    per_band_kl: dict[str, list[float]] = {b: [] for b in FOUR_BANDS}
    per_band_own_eq: dict[str, list[float]] = {b: [] for b in GE_P75_BANDS}
    per_band_own_en: dict[str, list[float]] = {b: [] for b in GE_P75_BANDS}
    per_band_nl_eq: dict[str, list[float]] = {b: [] for b in GE_P75_BANDS}
    per_band_nl_en: dict[str, list[float]] = {b: [] for b in GE_P75_BANDS}

    ewl_full_eq: list[float] = []
    ewl_full_en: list[float] = []
    ewl_ge75_eq: list[float] = []
    ewl_ge75_en: list[float] = []
    top_lift_chance: list[float] = []

    raw_over_nl_full_eq: list[float] = []
    scaled_over_nl_full_eq: list[float] = []
    raw_over_nl_ge75_eq: list[float] = []
    scaled_over_nl_ge75_eq: list[float] = []
    raw_over_nl_full_en: list[float] = []
    scaled_over_nl_full_en: list[float] = []
    top_raw_over_nl: list[float] = []
    top_scaled_over_nl: list[float] = []

    for nid in neuron_ids:
        rec = ek_per_neuron[nid]
        kl_sh = rec["kl_share_4band_norm"]
        c5 = rec["corpus_energy_5band"]
        w90 = float(c5["p90_97"])
        w75 = float(c5["p75_90"])
        w_tot = w90 + w75

        own_999 = own_by_nb[(nid, "p99.9_plus")]
        own_97 = own_by_nb[(nid, "p97_99.9")]
        own_90 = own_by_nb[(nid, "p90_97")]
        own_75 = own_by_nb[(nid, "p75_90")]

        nl_999 = nolabel_by_nb[(nid, "p99.9_plus")]
        nl_97 = nolabel_by_nb[(nid, "p97_99.9")]
        nl_90 = nolabel_by_nb[(nid, "p90_97")]
        nl_75 = nolabel_by_nb[(nid, "p75_90")]

        own_75_97_eq = 0.5 * (own_90 + own_75)
        nl_75_97_eq = 0.5 * (nl_90 + nl_75)
        if w_tot > 0.0:
            own_75_97_en = (w90 * own_90 + w75 * own_75) / w_tot
            nl_75_97_en = (w90 * nl_90 + w75 * nl_75) / w_tot
        else:
            own_75_97_en = own_75_97_eq
            nl_75_97_en = nl_75_97_eq

        own_eq_map: dict[str, float | None] = {
            "p99.9+": own_999,
            "p97_99.9": own_97,
            "p75_97": own_75_97_eq,
            "lt_p75": None,
        }
        own_en_map: dict[str, float | None] = {
            "p99.9+": own_999,
            "p97_99.9": own_97,
            "p75_97": own_75_97_en,
            "lt_p75": None,
        }
        nl_eq_map: dict[str, float | None] = {
            "p99.9+": nl_999,
            "p97_99.9": nl_97,
            "p75_97": nl_75_97_eq,
            "lt_p75": None,
        }
        nl_en_map: dict[str, float | None] = {
            "p99.9+": nl_999,
            "p97_99.9": nl_97,
            "p75_97": nl_75_97_en,
            "lt_p75": None,
        }

        for b in FOUR_BANDS:
            per_band_kl[b].append(float(kl_sh[b]))
        for b in GE_P75_BANDS:
            per_band_own_eq[b].append(float(own_eq_map[b]))  # type: ignore[arg-type]
            per_band_own_en[b].append(float(own_en_map[b]))  # type: ignore[arg-type]
            per_band_nl_eq[b].append(float(nl_eq_map[b]))  # type: ignore[arg-type]
            per_band_nl_en[b].append(float(nl_en_map[b]))  # type: ignore[arg-type]

        ewl_full_eq.append(compute_neuron_ewl(kl_sh, own_eq_map, FOUR_BANDS, renormalize=False))
        ewl_full_en.append(compute_neuron_ewl(kl_sh, own_en_map, FOUR_BANDS, renormalize=False))
        ewl_ge75_eq.append(compute_neuron_ewl(kl_sh, own_eq_map, GE_P75_BANDS, renormalize=True))
        ewl_ge75_en.append(compute_neuron_ewl(kl_sh, own_en_map, GE_P75_BANDS, renormalize=True))
        top_lift_chance.append(2.0 * (own_999 - 0.5))

        raw_over_nl_full_eq.append(
            compute_neuron_ewl_over_floor(kl_sh, own_eq_map, nl_eq_map, GE_P75_BANDS, renormalize=False, scale=1.0)
        )
        scaled_over_nl_full_eq.append(
            compute_neuron_ewl_over_floor(kl_sh, own_eq_map, nl_eq_map, GE_P75_BANDS, renormalize=False, scale=2.0)
        )
        raw_over_nl_ge75_eq.append(
            compute_neuron_ewl_over_floor(kl_sh, own_eq_map, nl_eq_map, GE_P75_BANDS, renormalize=True, scale=1.0)
        )
        scaled_over_nl_ge75_eq.append(
            compute_neuron_ewl_over_floor(kl_sh, own_eq_map, nl_eq_map, GE_P75_BANDS, renormalize=True, scale=2.0)
        )
        raw_over_nl_full_en.append(
            compute_neuron_ewl_over_floor(kl_sh, own_en_map, nl_en_map, GE_P75_BANDS, renormalize=False, scale=1.0)
        )
        scaled_over_nl_full_en.append(
            compute_neuron_ewl_over_floor(kl_sh, own_en_map, nl_en_map, GE_P75_BANDS, renormalize=False, scale=2.0)
        )
        top_raw_over_nl.append(own_999 - nl_999)
        top_scaled_over_nl.append(2.0 * (own_999 - nl_999))

    bands_summary: dict[str, Any] = {}
    for bi, b in enumerate(FOUR_BANDS):
        entry: dict[str, Any] = {
            "kl_share": neuron_bootstrap_mean(per_band_kl[b], n_bootstrap=n_bootstrap, seed=seed + 10 + bi)
        }
        if b in GE_P75_BANDS:
            entry["own_acc_equal_merge"] = neuron_bootstrap_mean(
                per_band_own_eq[b], n_bootstrap=n_bootstrap, seed=seed + 20 + bi
            )
            entry["no_label_acc_equal_merge"] = neuron_bootstrap_mean(
                per_band_nl_eq[b], n_bootstrap=n_bootstrap, seed=seed + 30 + bi
            )
            if b == "p75_97":
                entry["own_acc_energy_merge"] = neuron_bootstrap_mean(
                    per_band_own_en[b], n_bootstrap=n_bootstrap, seed=seed + 40
                )
                entry["no_label_acc_energy_merge"] = neuron_bootstrap_mean(
                    per_band_nl_en[b], n_bootstrap=n_bootstrap, seed=seed + 41
                )
        else:
            entry["own_acc_equal_merge"] = None
            entry["no_label_acc_equal_merge"] = None
            entry["status"] = "untested"
        bands_summary[b] = entry

    ewl17_stat = neuron_bootstrap_mean(ewl_full_eq, n_bootstrap=n_bootstrap, seed=seed + 50)
    top17_stat = neuron_bootstrap_mean(top_lift_chance, n_bootstrap=n_bootstrap, seed=seed + 54)
    ratio17 = (
        float(ewl17_stat["mean"] / top17_stat["mean"])
        if (ewl17_stat["mean"] is not None and top17_stat["mean"] and top17_stat["mean"] != 0.0)
        else None
    )

    return {
        "n_neurons": len(neuron_ids),
        "bands": bands_summary,
        "ewl_full_equal_merge": ewl17_stat,
        "ewl_full_energy_merge": neuron_bootstrap_mean(ewl_full_en, n_bootstrap=n_bootstrap, seed=seed + 51),
        "ewl_ge_p75_renorm_equal_merge": neuron_bootstrap_mean(ewl_ge75_eq, n_bootstrap=n_bootstrap, seed=seed + 52),
        "ewl_ge_p75_renorm_energy_merge": neuron_bootstrap_mean(ewl_ge75_en, n_bootstrap=n_bootstrap, seed=seed + 53),
        "top_p999_lift": top17_stat,
        "ewl_ratio_to_p999_lift": ratio17,
        "lift_over_no_label_floor": {
            "ewl_raw_full_equal_merge": neuron_bootstrap_mean(
                raw_over_nl_full_eq, n_bootstrap=n_bootstrap, seed=seed + 60
            ),
            "ewl_scaled_2x_full_equal_merge": neuron_bootstrap_mean(
                scaled_over_nl_full_eq, n_bootstrap=n_bootstrap, seed=seed + 61
            ),
            "ewl_raw_ge_p75_renorm_equal_merge": neuron_bootstrap_mean(
                raw_over_nl_ge75_eq, n_bootstrap=n_bootstrap, seed=seed + 62
            ),
            "ewl_scaled_2x_ge_p75_renorm_equal_merge": neuron_bootstrap_mean(
                scaled_over_nl_ge75_eq, n_bootstrap=n_bootstrap, seed=seed + 63
            ),
            "ewl_raw_full_energy_merge": neuron_bootstrap_mean(
                raw_over_nl_full_en, n_bootstrap=n_bootstrap, seed=seed + 64
            ),
            "ewl_scaled_2x_full_energy_merge": neuron_bootstrap_mean(
                scaled_over_nl_full_en, n_bootstrap=n_bootstrap, seed=seed + 65
            ),
            "top_p999_raw_own_minus_no_label": neuron_bootstrap_mean(
                top_raw_over_nl, n_bootstrap=n_bootstrap, seed=seed + 66
            ),
            "top_p999_scaled_2x_own_minus_no_label": neuron_bootstrap_mean(
                top_scaled_over_nl, n_bootstrap=n_bootstrap, seed=seed + 67
            ),
        },
    }


def analyze_qwen17b(
    ek_17b_path: Path,
    causal_17b_path: Path,
    judgements_17b_path: Path,
    poles_17b_path: Path,
    judge_model: str = "gpt-6.1-terra@none",
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Qwen3-1.7B per-band KL vs. detection, EWL, no-label lift, and sanity checks."""
    ek_doc = _load_json(ek_17b_path)
    causal_doc = _load_json(causal_17b_path)
    poles_doc = _load_json(poles_17b_path)

    pole_to_nid: dict[str, str] = {}
    for p in poles_doc["poles"]:
        pid = str(p["pole_id"])
        nid = pid.rsplit("_", 1)[0]
        pole_to_nid[pid] = nid

    raw_scores: dict[tuple[str, str, str], list[int]] = {}
    for row in _iter_jsonl(judgements_17b_path):
        if row.get("outcome") != "scored":
            continue
        if str(row.get("judge_model")) != judge_model:
            continue
        if str(row.get("distractor_type")) != "dead_zone":
            continue
        if str(row.get("order")) != "fwd":
            continue
        cond = str(row.get("condition"))
        if cond not in ("own", "no_label"):
            continue
        band = str(row.get("band"))
        pid = str(row["pole_id"])
        nid = pole_to_nid.get(pid, pid.rsplit("_", 1)[0])
        raw_scores.setdefault((cond, nid, band), []).append(1 if bool(row["correct"]) else 0)

    own_by_nb = {
        (nid, b): float(np.mean(vals))
        for (cond, nid, b), vals in raw_scores.items()
        if cond == "own" and vals
    }
    nolabel_by_nb = {
        (nid, b): float(np.mean(vals))
        for (cond, nid, b), vals in raw_scores.items()
        if cond == "no_label" and vals
    }

    ek_per_neuron = ek_doc["per_neuron"]
    all_nids = sorted(ek_per_neuron.keys())
    heldout_nids = [nid for nid in all_nids if not bool(ek_per_neuron[nid].get("in_pilot", False))]

    all_50 = _compute_17b_subset_summary(
        all_nids, ek_per_neuron, own_by_nb, nolabel_by_nb, n_bootstrap=n_bootstrap, seed=seed
    )
    heldout_40 = _compute_17b_subset_summary(
        heldout_nids, ek_per_neuron, own_by_nb, nolabel_by_nb, n_bootstrap=n_bootstrap, seed=seed + 200
    )

    # EnergyShare vs KLShare second-order sanity check on Qwen3-1.7B (n = 50)
    E_eval = np.array(
        [[float(ek_per_neuron[nid]["eval_energy_4band"][b]) for b in FOUR_BANDS] for nid in all_nids],
        dtype=np.float64,
    )
    E_corp = np.array(
        [[float(ek_per_neuron[nid]["corpus_energy_4band"][b]) for b in FOUR_BANDS] for nid in all_nids],
        dtype=np.float64,
    )
    K_mat = np.array(
        [[float(ek_per_neuron[nid]["kl_share_4band_norm"][b]) for b in FOUR_BANDS] for nid in all_nids],
        dtype=np.float64,
    )
    eval_vs_kl = pooled_energy_kl_analysis(
        E_eval, K_mat, n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed + 500
    )
    corp_vs_kl = pooled_energy_kl_analysis(
        E_corp, K_mat, n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed + 600
    )
    gc_eval_vs_kl = neuron_bootstrap_corr(
        [float(ek_per_neuron[nid]["eval_energy_gate_closed"]) for nid in all_nids],
        [float(ek_per_neuron[nid]["kl_share_gate_closed_norm"]) for nid in all_nids],
        n_bootstrap=n_bootstrap,
        seed=seed + 700,
    )
    gc_corp_vs_kl = neuron_bootstrap_corr(
        [float(ek_per_neuron[nid]["corpus_energy_gate_closed"]) for nid in all_nids],
        [float(ek_per_neuron[nid]["kl_share_gate_closed_norm"]) for nid in all_nids],
        n_bootstrap=n_bootstrap,
        seed=seed + 701,
    )

    # Inconclusive causal simulation disclosure
    c_per = causal_doc["per_neuron"]
    s20_eval64 = [float(c_per[nid]["strong20_rec_eval64"]) for nid in all_nids]
    s20_8k = [float(ek_per_neuron[nid]["strong20_rec"]) for nid in all_nids]
    sim_cv = [float(c_per[nid]["sim_cal_rec"]) for nid in all_nids]
    sim_act = [float(c_per[nid]["sim_cal_rec_active"]) for nid in all_nids]
    corrs = [float(c_per[nid]["corr"]) for nid in all_nids]

    inconclusive_disclosure = {
        "status": "inconclusive_simulator_floor_compression",
        "n_neurons": len(all_nids),
        "median_simulator_corr": float(np.median(corrs)),
        "strong20_rec_eval64_ge_0_25_count": int(sum(x >= 0.25 for x in s20_eval64)),
        "strong20_rec_8k_ge_0_25_count": int(sum(x >= 0.25 for x in s20_8k)),
        "sim_cal_rec_ge_0_25_count": int(sum(x >= 0.25 for x in sim_cv)),
        "sim_cal_rec_active_ge_0_50_count": int(sum(x >= 0.50 for x in sim_act)),
        "median_sim_cal_rec": float(np.median(sim_cv)),
        "max_sim_cal_rec": float(np.max(sim_cv)),
        "median_sim_cal_rec_active": float(np.median(sim_act)),
        "max_sim_cal_rec_active": float(np.max(sim_act)),
        "median_strong20_rec_eval64": float(np.median(s20_eval64)),
        "median_strong20_rec_8k": float(np.median(s20_8k)),
    }

    return {
        "n_neurons": len(all_nids),
        "n_heldout_neurons": len(heldout_nids),
        "zero_ablation_sanity_kl": float(ek_doc.get("zero_ablation_sanity_kl", 0.0)),
        "band_mapping": {
            "p99.9+": "p99.9_plus detection accuracy (dead_zone, own & no_label, gpt-6.1-terra@none, fwd)",
            "p97_99.9": "p97_99.9 detection accuracy (dead_zone, own & no_label)",
            "p75_97": (
                "Equal-weight (50/50) merge of p90_97 and p75_90 detection accuracy "
                "(corpus L2-energy-weighted merge also reported)"
            ),
            "lt_p75": "Untested in detection (1.1% mean KL share; lift set to 0 in full EWL)",
        },
        "all_50": all_50,
        "heldout_40": heldout_40,
        "energy_vs_kl_sanity_check": {
            "eval_energy_vs_kl_share": eval_vs_kl,
            "corpus_energy_vs_kl_share": corp_vs_kl,
            "gate_closed_eval_vs_kl": gc_eval_vs_kl,
            "gate_closed_corpus_vs_kl": gc_corp_vs_kl,
        },
        "inconclusive_causal_disclosure": inconclusive_disclosure,
    }


# ---------------------------------------------------------------------------
# Goodfire Table 4 comparison (A.4) & statistical disclosures (Item B)
# ---------------------------------------------------------------------------


def build_goodfire_table4_comparison(
    det_tables_4b: dict[str, Any],
    subband_17b_path: Path,
    res_17b: dict[str, Any],
    judgements_17b_path: Path,
    poles_17b_path: Path,
    snippets_17b_path: Path = DEFAULT_SNIPPETS_17B_PATH,
    judge_model: str = "gpt-6.1-terra@none",
) -> dict[str, Any]:
    """Assemble side-by-side comparison across Goodfire Table 4, Qwen3-4B, and Qwen3-1.7B."""
    # Qwen3-4B pole means and neuron means
    pm4 = det_tables_4b["pole_means"]
    nm4 = det_tables_4b["neuron_means"]

    def _mean_dict_vals(d: dict[Any, float | None]) -> tuple[float | None, int]:
        vals = [float(v) for v in d.values() if v is not None and math.isfinite(float(v))]
        return (float(np.mean(vals)) if vals else None), len(vals)

    q4_pole: dict[str, Any] = {}
    q4_neuron: dict[str, Any] = {}
    for b in ("p97_99", "p99_99.9", "p99.9_plus", "top100", "p75_97", "p25_75"):
        m_p, n_p = _mean_dict_vals(pm4[b])
        m_n, n_n = _mean_dict_vals(nm4[b])
        q4_pole[b] = {"mean": m_p, "n_poles": n_p}
        q4_neuron[b] = {"mean": m_n, "n_neurons": n_n}

    # Qwen3-1.7B heldout_80 (40 neurons / 80 poles) + all_100 (50 neurons / 100 poles)
    sub_doc = _load_json(subband_17b_path)
    cohorts = sub_doc.get("cohorts", sub_doc)
    h80_sub = cohorts["heldout_80"][judge_model]
    all_sub = cohorts["all_100"][judge_model]

    # Exact p75-90, p90-97, and optional sub-band per-neuron means
    poles_doc = _load_json(poles_17b_path)
    pole_to_nid = {
        str(p["pole_id"]): str(p.get("neuron_id") or str(p["pole_id"]).rsplit("_", 1)[0])
        for p in poles_doc["poles"]
    }
    pilot_pids = {str(p["pole_id"]) for p in poles_doc["poles"] if bool(p.get("in_pilot", False))}
    scores_17b_h80: dict[str, dict[str, list[int]]] = {}
    scores_17b_all: dict[str, dict[str, list[int]]] = {}
    scored_rows_h80: list[dict[str, Any]] = []
    scored_rows_all: list[dict[str, Any]] = []
    for row in _iter_jsonl(judgements_17b_path):
        if (
            row.get("outcome") != "scored"
            or str(row.get("judge_model")) != judge_model
            or str(row.get("distractor_type")) != "dead_zone"
            or str(row.get("order")) != "fwd"
            or str(row.get("condition")) != "own"
        ):
            continue
        b = str(row["band"])
        pid = str(row["pole_id"])
        c = 1 if bool(row["correct"]) else 0
        scores_17b_all.setdefault(b, {}).setdefault(pid, []).append(c)
        scored_rows_all.append(row)
        if pid not in pilot_pids:
            scores_17b_h80.setdefault(b, {}).setdefault(pid, []).append(c)
            scored_rows_h80.append(row)

    def _band_pole_mean(bucket: dict[str, dict[str, list[int]]], band: str) -> float | None:
        pmap = bucket.get(band, {})
        if not pmap:
            return None
        return float(np.mean([float(np.mean(v)) for v in pmap.values() if v]))

    def _band_neuron_mean(bucket: dict[str, dict[str, list[int]]], band: str) -> float | None:
        pmap = bucket.get(band, {})
        if not pmap:
            return None
        nmap: dict[str, list[float]] = {}
        for pid, vals in pmap.items():
            if vals:
                nmap.setdefault(pole_to_nid[pid], []).append(float(np.mean(vals)))
        return float(np.mean([float(np.mean(vs)) for vs in nmap.values()])) if nmap else None

    sub_p97_neuron_h80 = None
    sub_p99_neuron_h80 = None
    sub_p97_neuron_all = None
    sub_p99_neuron_all = None
    p99_thrs = {k: float(v["p99"]) for k, v in sub_doc.get("pole_thresholds", {}).items()}
    if snippets_17b_path.is_file() and p99_thrs:
        act_by_pair = {
            str(s["pair_id"]): float(s["pos"]["act_dir"])
            for s in _iter_jsonl(snippets_17b_path)
            if str(s.get("band")) == "p97_99.9" and str(s.get("distractor_type")) == "dead_zone"
        }

        def _sub_neuron_mean(rows: list[dict[str, Any]], ge_p99: bool) -> float | None:
            nmap: dict[str, dict[str, list[int]]] = {}
            for r in rows:
                if str(r["band"]) != "p97_99.9":
                    continue
                pair_id = str(r["pair_id"])
                pid = str(r["pole_id"])
                if pair_id not in act_by_pair or pid not in p99_thrs:
                    continue
                is_hi = act_by_pair[pair_id] >= p99_thrs[pid]
                if is_hi != ge_p99:
                    continue
                nid = pole_to_nid[pid]
                nmap.setdefault(nid, {}).setdefault(pid, []).append(1 if bool(r["correct"]) else 0)
            if not nmap:
                return None
            return float(np.mean([float(np.mean([float(np.mean(vs)) for vs in pd.values()])) for pd in nmap.values()]))

        sub_p97_neuron_h80 = _sub_neuron_mean(scored_rows_h80, ge_p99=False)
        sub_p99_neuron_h80 = _sub_neuron_mean(scored_rows_h80, ge_p99=True)
        sub_p97_neuron_all = _sub_neuron_mean(scored_rows_all, ge_p99=False)
        sub_p99_neuron_all = _sub_neuron_mean(scored_rows_all, ge_p99=True)

    return {
        "mandatory_caveat": (
            "Goodfire Table 4 uses a different model (Qwen3.6-35B-A3B MoE), labeler "
            "(Neuron Scientist agent), judge (Claude Opus 5.5), and random-text distractors; "
            "Qwen3-4B uses Ulto85 Claude Opus explanations and Qwen3-1.7B uses single-pass "
            "gpt-4.1 labels, both judged by gpt-6.1-terra@none against dead-zone distractors."
        ),
        "goodfire_table4": GOODFIRE_TABLE4_REFERENCE,
        "qwen3_4b_pole_means": {
            "unit": "active_pole",
            "judge": judge_model,
            "distractor": "dead_zone",
            "p97_99": q4_pole["p97_99"],
            "p99_99.9": q4_pole["p99_99.9"],
            "p99.9_plus_combined": q4_pole["p99.9_plus"],
            "top100": q4_pole["top100"],
            "p75_97": q4_pole["p75_97"],
            "p25_75": q4_pole["p25_75"],
        },
        "qwen3_4b_neuron_means": {
            "unit": "neuron_mean_over_active_poles",
            "judge": judge_model,
            "distractor": "dead_zone",
            "p97_99": q4_neuron["p97_99"],
            "p99_99.9": q4_neuron["p99_99.9"],
            "p99.9_plus_combined": q4_neuron["p99.9_plus"],
            "top100": q4_neuron["top100"],
            "p75_97": q4_neuron["p75_97"],
            "p25_75": q4_neuron["p25_75"],
        },
        "qwen3_17b_heldout_80": {
            "unit": "pole_and_neuron_reported",
            "n_poles": 80,
            "n_neurons": 40,
            "judge": judge_model,
            "distractor": "dead_zone",
            "p97_99": float(h80_sub["p97_99"]["conditions"]["own"]["mean"]),
            "p99_99.9": float(h80_sub["p99_99.9"]["conditions"]["own"]["mean"]),
            "p97_99_per_neuron": sub_p97_neuron_h80,
            "p99_99.9_per_neuron": sub_p99_neuron_h80,
            "p97_99.9_combined": _band_pole_mean(scores_17b_h80, "p97_99.9"),
            "p99.9_plus_combined": _band_pole_mean(scores_17b_h80, "p99.9_plus"),
            "p99.9_plus_per_neuron": _band_neuron_mean(scores_17b_h80, "p99.9_plus"),
            "p90_97": _band_pole_mean(scores_17b_h80, "p90_97"),
            "p75_90": _band_pole_mean(scores_17b_h80, "p75_90"),
            "p75_97_equal_merge": res_17b["heldout_40"]["bands"]["p75_97"]["own_acc_equal_merge"]["mean"],
            "p25_75": None,
        },
        "qwen3_17b_all_100": {
            "unit": "pole_and_neuron_reported",
            "n_poles": 100,
            "n_neurons": 50,
            "judge": judge_model,
            "distractor": "dead_zone",
            "p97_99": float(all_sub["p97_99"]["conditions"]["own"]["mean"]),
            "p99_99.9": float(all_sub["p99_99.9"]["conditions"]["own"]["mean"]),
            "p97_99_per_neuron": sub_p97_neuron_all,
            "p99_99.9_per_neuron": sub_p99_neuron_all,
            "p97_99.9_combined": _band_pole_mean(scores_17b_all, "p97_99.9"),
            "p99.9_plus_combined": _band_pole_mean(scores_17b_all, "p99.9_plus"),
            "p99.9_plus_per_neuron": _band_neuron_mean(scores_17b_all, "p99.9_plus"),
            "p90_97": _band_pole_mean(scores_17b_all, "p90_97"),
            "p75_90": _band_pole_mean(scores_17b_all, "p75_90"),
            "p75_97_equal_merge": res_17b["all_50"]["bands"]["p75_97"]["own_acc_equal_merge"]["mean"],
            "p25_75": None,
        },
    }


def build_statistical_disclosures(
    det_tables_4b: dict[str, Any],
    gap_path: Path,
    ulto_neurons_path: Path,
    ulto_scores_path: Path,
    fpr_4b_path: Path,
    snippets_top100_path: Path,
    res_17b: dict[str, Any],
    existing_output_path: Path | None = None,
) -> dict[str, Any]:
    """Compute exact base rates, saturation range, imputation rates, slot bias, and lexical overlap."""
    gap_doc = _load_json(gap_path)
    per_n = gap_doc["per_neuron"]
    e3_neurons = _load_json(ulto_neurons_path)
    e3_scores = _load_json(ulto_scores_path)["neurons"]
    scored_set = {(int(x["layer"]), int(x["neuron"])) for x in e3_scores}
    unscored_neurons = [
        [int(pair[0]), int(pair[1])]
        for pair in e3_neurons
        if (int(pair[0]), int(pair[1])) not in scored_set
    ]

    n_total = len(per_n)
    n_active_poles = len(det_tables_4b["pole_info"])
    n_excluded_poles = 2 * n_total - n_active_poles

    # Per-neuron arrays on all 278 Qwen3-4B neurons
    top100_map = det_tables_4b["neuron_means"]["top100"]
    rows_278 = []
    for ks, rec in sorted(per_n.items(), key=lambda kv: int(kv[0])):
        k = int(ks)
        rows_278.append(
            {
                "k": k,
                "top100": float(top100_map[k]),
                "sim_cal_rec": float(rec["sim_cal_rec"]),
                "sim_cal_rec_active": float(rec["sim_cal_rec_active"]),
                "strong20_rec": float(rec["strong20_rec"]),
            }
        )

    pass_80 = [r for r in rows_278 if r["top100"] >= 0.80]
    fail_80 = [r for r in rows_278 if r["top100"] < 0.80]
    perfect_100 = [r for r in rows_278 if abs(r["top100"] - 1.0) < 1e-9]

    rho_top100_sim, _ = _safe_spearman(
        np.array([r["top100"] for r in rows_278]),
        np.array([r["sim_cal_rec"] for r in rows_278]),
    )
    rho_top100_act, _ = _safe_spearman(
        np.array([r["top100"] for r in rows_278]),
        np.array([r["sim_cal_rec_active"] for r in rows_278]),
    )

    base_rates = {
        "n_ulto85_catalog_neurons": len(e3_neurons),
        "n_scored_neurons": n_total,
        "unscored_ulto85_neurons": unscored_neurons,
        "n_active_poles": n_active_poles,
        "n_excluded_poles": n_excluded_poles,
        "sim_lt_0_50_all": sum(1 for r in rows_278 if r["sim_cal_rec"] < 0.50),
        "sim_lt_0_25_all": sum(1 for r in rows_278 if r["sim_cal_rec"] < 0.25),
        "sim_le_0_00_all": sum(1 for r in rows_278 if r["sim_cal_rec"] <= 0.0),
        "sim_ge_0_25_all": sum(1 for r in rows_278 if r["sim_cal_rec"] >= 0.25),
        "sim_ge_0_50_all": sum(1 for r in rows_278 if r["sim_cal_rec"] >= 0.50),
        "active_ge_0_50_all": sum(1 for r in rows_278 if r["sim_cal_rec_active"] >= 0.50),
        "top100_ge_0_80_pass_count": len(pass_80),
        "top100_lt_0_80_fail_count": len(fail_80),
        "pass_sim_ge_0_25": sum(1 for r in pass_80 if r["sim_cal_rec"] >= 0.25),
        "pass_sim_ge_0_50": sum(1 for r in pass_80 if r["sim_cal_rec"] >= 0.50),
        "pass_sim_lt_0_50": sum(1 for r in pass_80 if r["sim_cal_rec"] < 0.50),
        "pass_active_ge_0_50": sum(1 for r in pass_80 if r["sim_cal_rec_active"] >= 0.50),
        "fail_sim_ge_0_25": sum(1 for r in fail_80 if r["sim_cal_rec"] >= 0.25),
        "fail_sim_ge_0_50": sum(1 for r in fail_80 if r["sim_cal_rec"] >= 0.50),
        "fail_active_ge_0_50": sum(1 for r in fail_80 if r["sim_cal_rec_active"] >= 0.50),
        "top100_eq_1_00_count": len(perfect_100),
        "top100_eq_1_00_sim_min": float(min(r["sim_cal_rec"] for r in perfect_100)),
        "top100_eq_1_00_sim_max": float(max(r["sim_cal_rec"] for r in perfect_100)),
        "top100_eq_1_00_sim_median": float(np.median([r["sim_cal_rec"] for r in perfect_100])),
        "spearman_top100_vs_sim_cal_rec": rho_top100_sim,
        "spearman_top100_vs_sim_cal_rec_active": rho_top100_act,
    }

    # YES/NO logprob imputation audit on Qwen3-4B
    fpr_imp: dict[str, dict[str, Any]] = {}
    if fpr_4b_path.is_file():
        acc_imp: dict[tuple[str, str], list[int]] = {}
        for row in _iter_jsonl(fpr_4b_path):
            key = (str(row["judge_model"]), str(row["probe_type"]))
            acc_imp.setdefault(key, [0, 0])
            acc_imp[key][1] += 1
            if bool(row.get("logprob_imputed", False)):
                acc_imp[key][0] += 1
        for (jm, ptype), (imp_cnt, tot) in sorted(acc_imp.items()):
            fpr_imp[f"{jm}__{ptype}"] = {
                "imputed_count": imp_cnt,
                "total": tot,
                "imputed_rate": float(imp_cnt / tot) if tot else None,
            }

    # Lexical token-in-label overlap on top100 (live when snippet JSONL is present; fallback otherwise)
    if snippets_top100_path.is_file():
        pole_text = {pid: str(info["pole_text"]) for pid, info in det_tables_4b["pole_info"].items()}
        n_pairs = 0
        exact_w_ci = 0
        sub_ci = 0
        for row in _iter_jsonl(snippets_top100_path):
            n_pairs += 1
            tok = str(row["pos"]["target_token_str"]).strip()
            lbl = pole_text[str(row["pole_id"])]
            if len(tok) >= 2:
                if re.search(rf"\b{re.escape(tok)}\b", lbl, flags=re.I):
                    exact_w_ci += 1
                if tok.lower() in lbl.lower():
                    sub_ci += 1
        lexical_overlap = {
            "source": "computed_live_from_snippets_4b_top100",
            "n_pairs": n_pairs,
            "exact_word_ci_len_ge_2_count": exact_w_ci,
            "exact_word_ci_len_ge_2_rate": float(exact_w_ci / n_pairs) if n_pairs else None,
            "substring_ci_len_ge_2_count": sub_ci,
            "substring_ci_len_ge_2_rate": float(sub_ci / n_pairs) if n_pairs else None,
        }
    elif existing_output_path is not None and existing_output_path.is_file():
        prev = _load_json(existing_output_path)
        lexical_overlap = prev.get("statistical_disclosures", {}).get(
            "top100_token_in_label_lexical_overlap", PRECOMPUTED_TOP100_LEXICAL_OVERLAP
        )
    else:
        lexical_overlap = dict(PRECOMPUTED_TOP100_LEXICAL_OVERLAP)

    return {
        "base_rates_and_detection_vs_causal_4b": base_rates,
        "qwen17b_inconclusive_causal_disclosure": res_17b["inconclusive_causal_disclosure"],
        "yesno_logprob_imputation_4b": fpr_imp,
        "slot_b_preference_p_pred_a_4b": det_tables_4b["slot_a_counts"],
        "top100_token_in_label_lexical_overlap": lexical_overlap,
        "post_hoc_top100_disclosure": (
            "The top100 sweep on Qwen3-4B was added post hoc after observing that 295/424 active "
            "poles (69.6%) had p99.9 < 20% of max activation."
        ),
    }


# ---------------------------------------------------------------------------
# Publication figure: figures/main_detection_vs_kl_by_band.png
# ---------------------------------------------------------------------------


def plot_main_detection_vs_kl_by_band(
    res_4b: dict[str, Any],
    res_17b: dict[str, Any],
    output_path: Path,
) -> None:
    """Render clean 2-panel grouped bar chart of detection accuracy vs. exact fp32 KL share."""
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8), dpi=250, sharey=True)
    fig.subplots_adjust(left=0.058, right=0.985, top=0.815, bottom=0.195, wspace=0.075)

    x_labels = ["p99.9+", "p97–99.9", "p75–97", "< p75"]
    xs = np.array([0.0, 1.0, 2.0, 3.22], dtype=np.float64)
    sep_x = 2.61
    bar_w = 0.34

    color_det = "#2563EB"
    color_kl = "#D97706"
    color_det_lt75 = "#93C5FD"
    color_kl_lt75 = "#FCD34D"
    color_nl = "#374151"

    # ---- Panel A: Qwen3-4B ----
    ax0 = axes[0]
    n4 = res_4b["n_kl_neurons"]
    all4 = res_4b["all_kl_neurons"]
    bi4 = res_4b["bipolar_only"]
    uni4 = res_4b["unipolar_only"]

    det4_m = [all4["bands"][b]["det_acc_detected_only"]["mean"] for b in FOUR_BANDS]
    det4_lo = [det4_m[i] - all4["bands"][b]["det_acc_detected_only"]["ci_low"] for i, b in enumerate(FOUR_BANDS)]
    det4_hi = [all4["bands"][b]["det_acc_detected_only"]["ci_high"] - det4_m[i] for i, b in enumerate(FOUR_BANDS)]

    kl4_m = [all4["bands"][b]["kl_share"]["mean"] for b in FOUR_BANDS]
    kl4_lo = [kl4_m[i] - all4["bands"][b]["kl_share"]["ci_low"] for i, b in enumerate(FOUR_BANDS)]
    kl4_hi = [all4["bands"][b]["kl_share"]["ci_high"] - kl4_m[i] for i, b in enumerate(FOUR_BANDS)]

    # Grey shaded < p75 region
    ax0.axvspan(sep_x, 3.85, color="#F3F4F6", alpha=0.85, zorder=0)
    ax0.axvline(sep_x, color="#6B7280", linestyle="--", linewidth=1.25, zorder=1)
    h_chance = ax0.axhline(0.50, color="#6B7280", linestyle=":", linewidth=1.3, zorder=1)

    # First 3 active-tail bands
    b_det4 = ax0.bar(
        xs[:3] - bar_w / 2,
        det4_m[:3],
        width=bar_w,
        color=color_det,
        edgecolor="#1E3A8A",
        linewidth=0.7,
        yerr=[det4_lo[:3], det4_hi[:3]],
        capsize=3.5,
        error_kw={"ecolor": "#1E3A8A", "elinewidth": 1.1},
        label="Detection accuracy (own label, 95% neuron CI)",
        zorder=3,
    )
    b_kl4 = ax0.bar(
        xs[:3] + bar_w / 2,
        kl4_m[:3],
        width=bar_w,
        color=color_kl,
        edgecolor="#92400E",
        linewidth=0.7,
        yerr=[kl4_lo[:3], kl4_hi[:3]],
        capsize=3.5,
        error_kw={"ecolor": "#92400E", "elinewidth": 1.1},
        label="Exact fp32 KL share (95% neuron CI)",
        zorder=3,
    )
    # Hatched < p75 bars on 4B
    ax0.bar(
        [xs[3] - bar_w / 2],
        [det4_m[3]],
        width=bar_w,
        color=color_det_lt75,
        edgecolor="#1E3A8A",
        linewidth=0.8,
        hatch="///",
        yerr=[[det4_lo[3]], [det4_hi[3]]],
        capsize=3.5,
        error_kw={"ecolor": "#1E3A8A", "elinewidth": 1.1},
        zorder=3,
    )
    ax0.bar(
        [xs[3] + bar_w / 2],
        [kl4_m[3]],
        width=bar_w,
        color=color_kl_lt75,
        edgecolor="#92400E",
        linewidth=0.8,
        hatch="///",
        yerr=[[kl4_lo[3]], [kl4_hi[3]]],
        capsize=3.5,
        error_kw={"ecolor": "#92400E", "elinewidth": 1.1},
        zorder=3,
    )

    for i in range(4):
        ax0.text(
            xs[i] - bar_w / 2,
            det4_m[i] + det4_hi[i] + 0.018,
            f"{det4_m[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#1E3A8A",
        )
        ax0.text(
            xs[i] + bar_w / 2,
            kl4_m[i] + kl4_hi[i] + 0.018,
            f"{kl4_m[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#92400E",
        )

    # Callout for < p75 on 4B
    uni_lt75 = uni4["bands"]["lt_p75"]["kl_share"]["mean"]
    bi_lt75 = bi4["bands"]["lt_p75"]["kl_share"]["mean"]
    ax0.text(
        3.23,
        0.74,
        f"p25–75 tested;\n< p25 untested\nKL mostly opp-sign\nof 1-sided neurons\n(1-sided: {100*uni_lt75:.1f}%\n2-sided: {100*bi_lt75:.1f}%)",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#9CA3AF", "alpha": 0.94},
        zorder=5,
    )

    # Compact EWL vs top-band lift box in upper-center clear zone
    ewl4 = all4["ewl_full_energy_merge"]["mean"]
    ewl4_ge75 = all4["ewl_ge_p75_renorm_energy_merge"]["mean"]
    ewl4_bi = bi4["ewl_full_energy_merge"]["mean"]
    top4_lift = all4["top_p999_lift"]["mean"]
    top100_lift = all4["top100_lift"]["mean"]
    ax0.text(
        1.52,
        0.965,
        (
            f"Top lift: p99.9+ = {top4_lift:.3f} | top100 = {top100_lift:.3f}\n"
            f"Effect-weighted lift (EWL): all = {ewl4:.3f}\n"
            f"  2-sided = {ewl4_bi:.3f} | >=p75 renorm = {ewl4_ge75:.3f}"
        ),
        ha="center",
        va="top",
        fontsize=7.9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EFF6FF", "edgecolor": "#93C5FD", "alpha": 0.95},
        zorder=5,
    )

    ax0.set_title(
        f"A. Qwen3-4B (n = {n4} fp32 KL neurons; Ulto85 labels, gpt-6.1-terra)",
        fontsize=10.6,
        fontweight="bold",
        pad=8,
    )
    ax0.set_xticks(xs)
    ax0.set_xticklabels(x_labels, fontsize=9.6, fontweight="bold")
    ax0.set_xlim(-0.55, 3.85)
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("Accuracy or Normalized KL Share", fontsize=10.2, fontweight="bold")
    ax0.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)

    # ---- Panel B: Qwen3-1.7B ----
    ax1 = axes[1]
    all17 = res_17b["all_50"]
    det17_m = [all17["bands"][b]["own_acc_equal_merge"]["mean"] for b in GE_P75_BANDS]
    det17_lo = [det17_m[i] - all17["bands"][b]["own_acc_equal_merge"]["ci_low"] for i, b in enumerate(GE_P75_BANDS)]
    det17_hi = [all17["bands"][b]["own_acc_equal_merge"]["ci_high"] - det17_m[i] for i, b in enumerate(GE_P75_BANDS)]

    nl17_m = [all17["bands"][b]["no_label_acc_equal_merge"]["mean"] for b in GE_P75_BANDS]
    nl17_lo = [nl17_m[i] - all17["bands"][b]["no_label_acc_equal_merge"]["ci_low"] for i, b in enumerate(GE_P75_BANDS)]
    nl17_hi = [all17["bands"][b]["no_label_acc_equal_merge"]["ci_high"] - nl17_m[i] for i, b in enumerate(GE_P75_BANDS)]

    kl17_m = [all17["bands"][b]["kl_share"]["mean"] for b in FOUR_BANDS]
    kl17_lo = [kl17_m[i] - all17["bands"][b]["kl_share"]["ci_low"] for i, b in enumerate(FOUR_BANDS)]
    kl17_hi = [all17["bands"][b]["kl_share"]["ci_high"] - kl17_m[i] for i, b in enumerate(FOUR_BANDS)]

    ax1.axvspan(sep_x, 3.85, color="#F3F4F6", alpha=0.85, zorder=0)
    ax1.axvline(sep_x, color="#6B7280", linestyle="--", linewidth=1.25, zorder=1)
    ax1.axhline(0.50, color="#9CA3AF", linestyle=":", linewidth=1.2, zorder=1)

    ax1.bar(
        xs[:3] - bar_w / 2,
        det17_m,
        width=bar_w,
        color=color_det,
        edgecolor="#1E3A8A",
        linewidth=0.7,
        yerr=[det17_lo, det17_hi],
        capsize=3.5,
        error_kw={"ecolor": "#1E3A8A", "elinewidth": 1.1},
        zorder=3,
    )
    ax1.bar(
        xs[:3] + bar_w / 2,
        kl17_m[:3],
        width=bar_w,
        color=color_kl,
        edgecolor="#92400E",
        linewidth=0.7,
        yerr=[kl17_lo[:3], kl17_hi[:3]],
        capsize=3.5,
        error_kw={"ecolor": "#92400E", "elinewidth": 1.1},
        zorder=3,
    )
    # Hatched < p75 KL bar on 1.7B (detection untested)
    ax1.bar(
        [xs[3] + bar_w / 2],
        [kl17_m[3]],
        width=bar_w,
        color=color_kl_lt75,
        edgecolor="#92400E",
        linewidth=0.8,
        hatch="///",
        yerr=[[kl17_lo[3]], [kl17_hi[3]]],
        capsize=3.5,
        error_kw={"ecolor": "#92400E", "elinewidth": 1.1},
        zorder=3,
    )

    # No-label floor diamond markers on 1.7B
    h_nl = ax1.errorbar(
        xs[:3] - bar_w / 2,
        nl17_m,
        yerr=[nl17_lo, nl17_hi],
        fmt="D",
        color=color_nl,
        markersize=5.6,
        markerfacecolor="#FFFFFF",
        markeredgewidth=1.6,
        elinewidth=1.0,
        capsize=2.5,
        label="No-label floor accuracy (1.7B, 95% neuron CI)",
        zorder=5,
    )

    for i in range(3):
        ax1.text(
            xs[i] - bar_w / 2,
            det17_m[i] + det17_hi[i] + 0.018,
            f"{det17_m[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#1E3A8A",
        )
        ax1.text(
            xs[i] + bar_w / 2,
            kl17_m[i] + kl17_hi[i] + 0.018,
            f"{kl17_m[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#92400E",
        )
    ax1.text(
        xs[3] + bar_w / 2,
        kl17_m[3] + kl17_hi[3] + 0.018,
        f"{kl17_m[3]:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color="#92400E",
    )

    ax1.text(
        3.23,
        0.58,
        f"Untested\nin detection\n(KL share:\n{100*kl17_m[3]:.1f}%,\nboth poles\nincluded)",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#9CA3AF", "alpha": 0.94},
        zorder=5,
    )

    ewl17 = all17["ewl_full_equal_merge"]["mean"]
    ewl17_ge75 = all17["ewl_ge_p75_renorm_equal_merge"]["mean"]
    top17_lift = all17["top_p999_lift"]["mean"]
    nl_raw = all17["lift_over_no_label_floor"]["ewl_raw_full_equal_merge"]["mean"]
    nl_top = all17["lift_over_no_label_floor"]["top_p999_raw_own_minus_no_label"]["mean"]

    ho17 = res_17b["heldout_40"]
    ho_ewl = ho17["ewl_full_equal_merge"]["mean"]
    ho_ge75 = ho17["ewl_ge_p75_renorm_equal_merge"]["mean"]
    ho_top = ho17["top_p999_lift"]["mean"]
    ho_nl_raw = ho17["lift_over_no_label_floor"]["ewl_raw_full_equal_merge"]["mean"]

    ax1.text(
        1.78,
        0.995,
        (
            f"all_50 (incl. 10 pilot): top = {top17_lift:.3f} | EWL = {ewl17:.3f} | >=p75 = {ewl17_ge75:.3f}\n"
            f"  own−no_label: top = {nl_top:+.3f} | raw = {nl_raw:+.3f} | 2x = {2*nl_raw:+.3f}\n"
            f"heldout_40: top = {ho_top:.3f} | EWL = {ho_ewl:.3f} | >=p75 = {ho_ge75:.3f}\n"
            f"  own−no_label: raw = {ho_nl_raw:+.3f} | 2x = {2*ho_nl_raw:+.3f}"
        ),
        ha="center",
        va="top",
        fontsize=7.3,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#EFF6FF", "edgecolor": "#93C5FD", "alpha": 0.95},
        zorder=5,
    )

    ax1.set_title(
        "B. Qwen3-1.7B (n = 50: 40 heldout + 10 pilot; gpt-4.1, terra)",
        fontsize=10.4,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(x_labels, fontsize=9.6, fontweight="bold")
    ax1.set_xlim(-0.55, 3.85)
    ax1.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)

    # Shared title, subtitle, legend, and credit footer
    fig.suptitle(
        "Two-Snippet Detection Accuracy Falls Where Most Downstream Ablation KL Lives",
        fontsize=13.2,
        fontweight="bold",
        y=0.968,
    )
    fig.text(
        0.5,
        0.903,
        (
            "Band mapping — Qwen3-4B: p97–99.9 merges p99–99.9 and p97–99 by corpus L2 energy; "
            "Qwen3-1.7B: p75–97 merges p90–97 and p75–90 (50/50; energy-weighted sensitivity matches within 0.002). "
            "Error bars: 95% neuron bootstrap CIs (B = 2,000)."
        ),
        ha="center",
        va="center",
        fontsize=8.6,
        color="#374151",
    )

    handles = [b_det4, b_kl4, h_nl, h_chance]
    labels = [
        "Detection accuracy (own label, 95% neuron CI)",
        "Exact fp32 mean-ablation KL share (95% neuron CI)",
        "No-label floor accuracy (Qwen3-1.7B, 95% neuron CI)",
        "Chance (0.50)",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D1D5DB",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.015,
        f"{ULTO85_CREDIT_FOOTER} | Qwen3-1.7B & Qwen3-4B KL ablations computed in exact IEEE-754 float32 (zero-ablation sanity KL = 0.0).",
        ha="center",
        va="center",
        fontsize=8.1,
        color="#4B5563",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manual 12-neuron reading sheet (audit/reading_sheet_12_neurons.md)
# ---------------------------------------------------------------------------


def select_12_reading_sheet_neurons(
    gap_per_neuron: dict[str, dict[str, Any]],
    det_tables_4b: dict[str, Any],
) -> list[tuple[str, int]]:
    """Deterministically select 12 Qwen3-4B neurons (4 high-sim, 4 neg-sim, 4 failing top100)."""
    top100_map = det_tables_4b["neuron_means"]["top100"]
    rows = []
    for ks, rec in sorted(gap_per_neuron.items(), key=lambda kv: int(kv[0])):
        k = int(ks)
        rows.append(
            {
                "k": k,
                "layer": int(rec["layer"]),
                "top100": float(top100_map[k]),
                "sim": float(rec["sim_cal_rec"]),
                "active": float(rec["sim_cal_rec_active"]),
            }
        )

    selected: list[tuple[str, int]] = []

    # Group 1: 4 neurons with top100 == 1.00 and high sim_cal_rec (must include info k=26)
    g1_pool = [r for r in rows if abs(r["top100"] - 1.0) < 1e-9]
    g1_ks = [26]
    g1_layers = {next(r["layer"] for r in rows if r["k"] == 26)}
    for r in sorted(g1_pool, key=lambda x: (-x["sim"], x["k"])):
        if len(g1_ks) >= 4:
            break
        if r["k"] in g1_ks or r["layer"] in g1_layers:
            continue
        g1_ks.append(r["k"])
        g1_layers.add(r["layer"])
    for k in g1_ks:
        selected.append(("high_causal_top100_1_00", k))

    # Group 2: 4 neurons with top100 == 1.00 and negative sim_cal_rec (must include PCR k=23 + extreme min + varied layers)
    g2_pool = [r for r in rows if abs(r["top100"] - 1.0) < 1e-9 and r["sim"] < 0.0]
    g2_ks = [23]
    min_neg = min(g2_pool, key=lambda x: (x["sim"], x["k"]))
    if min_neg["k"] not in g2_ks:
        g2_ks.append(min_neg["k"])
    g2_layers = {next(r["layer"] for r in rows if r["k"] == kk) for kk in g2_ks}
    for r in sorted(g2_pool, key=lambda x: (x["sim"], x["k"])):
        if len(g2_ks) >= 4:
            break
        if r["k"] in g2_ks or r["layer"] in g2_layers:
            continue
        g2_ks.append(r["k"])
        g2_layers.add(r["layer"])
    for k in g2_ks:
        selected.append(("negative_causal_top100_1_00", k))

    # Group 3: 4 neurons failing detection (top100 < 0.80), spread across layers & causal outcomes
    g3_pool = [r for r in rows if r["top100"] < 0.80]
    g3_ks: list[int] = []
    g3_layers: set[int] = set()

    # 3a: lowest top100
    r_lowest_det = min(g3_pool, key=lambda x: (x["top100"], x["k"]))
    g3_ks.append(r_lowest_det["k"])
    g3_layers.add(r_lowest_det["layer"])
    # 3b: highest active recovery among failing neurons (distinct layer)
    r_best_act = max(
        [r for r in g3_pool if r["layer"] not in g3_layers],
        key=lambda x: (x["active"], -x["k"]),
    )
    g3_ks.append(r_best_act["k"])
    g3_layers.add(r_best_act["layer"])
    # 3c: lowest sim_cal_rec among deep layers >= 24 (distinct layer)
    deep_pool = [r for r in g3_pool if r["layer"] >= 24 and r["layer"] not in g3_layers and r["k"] not in g3_ks]
    if not deep_pool:
        deep_pool = [r for r in g3_pool if r["k"] not in g3_ks]
    r_deep_neg = min(deep_pool, key=lambda x: (x["sim"], x["k"]))
    g3_ks.append(r_deep_neg["k"])
    g3_layers.add(r_deep_neg["layer"])
    # 3d: middle-layer (12 <= layer <= 20) near-zero failing neuron (distinct layer)
    mid_pool = [r for r in g3_pool if 12 <= r["layer"] <= 20 and r["layer"] not in g3_layers and r["k"] not in g3_ks]
    if not mid_pool:
        mid_pool = [r for r in g3_pool if r["k"] not in g3_ks]
    r_mid = min(mid_pool, key=lambda x: (x["top100"], x["k"]))
    g3_ks.append(r_mid["k"])
    g3_layers.add(r_mid["layer"])

    for k in g3_ks:
        selected.append(("failing_detection_top100_lt_0_80", k))

    return selected


def _sample_five_pairs_for_neuron(
    pairs_by_pole: dict[str, list[dict[str, Any]]],
    ordered_poles: list[str],
    n_total: int = 5,
) -> list[dict[str, Any]]:
    """Deterministically sample up to n_total positive pairs spread across a neuron's active poles."""
    available_poles = [pid for pid in ordered_poles if pairs_by_pole.get(pid)]
    if not available_poles:
        return []
    if len(available_poles) == 1:
        pool = sorted(pairs_by_pole[available_poles[0]], key=lambda r: int(r["pair_idx"]))
        if len(pool) <= n_total:
            return pool
        idxs = [int(round(i * (len(pool) - 1) / (n_total - 1))) for i in range(n_total)]
        return [pool[i] for i in idxs]
    # Bipolar: 3 from first pole + 2 from second pole
    quotas = [3, n_total - 3]
    out: list[dict[str, Any]] = []
    for pid, q in zip(available_poles[:2], quotas):
        pool = sorted(pairs_by_pole[pid], key=lambda r: int(r["pair_idx"]))
        if len(pool) <= q:
            out.extend(pool)
        else:
            idxs = [int(round(i * (len(pool) - 1) / max(1, q - 1))) for i in range(q)]
            out.extend(pool[i] for i in idxs)
    return out[:n_total]


def generate_reading_sheet_12_neurons(
    kl_path: Path,
    gap_path: Path,
    det_tables_4b: dict[str, Any],
    snippets_top100_path: Path,
    snippets_4b_path: Path,
    judgements_top100_path: Path,
    judgements_4b_path: Path,
    output_path: Path,
    judge_model: str = "gpt-6.1-terra@none",
) -> dict[str, Any]:
    """Generate audit/reading_sheet_12_neurons.md with verbatim labels, marked snippets, and verdicts."""
    kl_doc = _load_json(kl_path)
    gap_doc = _load_json(gap_path)
    kl_per_neuron = kl_doc["per_neuron"]
    gap_per_neuron = gap_doc["per_neuron"]

    selected = select_12_reading_sheet_neurons(gap_per_neuron, det_tables_4b)
    selected_ks = {k for _, k in selected}

    # Index judgements for selected neurons (gpt-6.1-terra@none, own)
    j_top100: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(judgements_top100_path):
        if (
            int(row["k"]) in selected_ks
            and str(row["judge_model"]) == judge_model
            and str(row["condition"]) == "own"
            and str(row["band"]) == "top100"
        ):
            j_top100[str(row["pair_id"])] = row

    j_p97_99: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(judgements_4b_path):
        if (
            int(row["k"]) in selected_ks
            and str(row["judge_model"]) == judge_model
            and str(row["condition"]) == "own"
            and str(row["band"]) == "p97_99"
        ):
            j_p97_99[str(row["pair_id"])] = row

    # Index snippets for selected neurons
    snips_top100: dict[int, dict[str, list[dict[str, Any]]]] = {k: {} for k in selected_ks}
    for row in _iter_jsonl(snippets_top100_path):
        k = int(row["k"])
        if k in selected_ks and str(row["band"]) == "top100":
            snips_top100[k].setdefault(str(row["pole_id"]), []).append(row)

    snips_p97_99: dict[int, dict[str, list[dict[str, Any]]]] = {k: {} for k in selected_ks}
    for row in _iter_jsonl(snippets_4b_path):
        k = int(row["k"])
        if k in selected_ks and str(row["band"]) == "p97_99":
            snips_p97_99[k].setdefault(str(row["pole_id"]), []).append(row)

    group_titles = {
        "high_causal_top100_1_00": "Group A: Perfect Top-100 Detection (`1.000`) & High Causal Recovery",
        "negative_causal_top100_1_00": "Group B: Perfect Top-100 Detection (`1.000`) & Negative Causal Recovery",
        "failing_detection_top100_lt_0_80": "Group C: Failing Top-100 Detection (`< 0.800`)",
    }

    lines: list[str] = [
        "# Manual Reading Sheet: 12 Selected Qwen3-4B Neurons",
        "",
        "Twelve deterministically selected `Qwen/Qwen3-4B` neurons from `Ulto85/neurons-kinda-clean` (MIT): "
        "4 with `top100 == 1.000` and high `sim_cal_rec` (including `info`, `k=26`), "
        "4 with `top100 == 1.000` and negative `sim_cal_rec` (including `PCR`, `k=23`, and the minimum `-7.110`), "
        "and 4 failing top-100 detection (`top100 < 0.800`) spread across layers and causal outcomes. "
        "Snippets come from `HuggingFaceFW/fineweb` (`sample/10BT/000_00000.parquet`, ODC-By). "
        f"KL shares are loaded from `{kl_path.as_posix()}` (`n = {len(kl_per_neuron)}` neurons).",
        "",
        "## Overview Table",
        "",
        "| # | Group | `k` | Layer / Neuron | Polarity | `top100` | `p99.9+` | `p97–99` | `sim_cal_rec` | `active` | `strong20` | KL `p99.9+` / `p97–99.9` / `p75–97` / `<p75` |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    nm = det_tables_4b["neuron_means"]
    pole_info = det_tables_4b["pole_info"]

    for idx, (grp, k) in enumerate(selected, start=1):
        ks = str(k)
        g = gap_per_neuron[ks]
        pol = "bipolar" if g["is_bipolar"] else "unipolar"
        t100 = nm["top100"].get(k)
        p999 = nm["p99.9_plus"].get(k)
        p97 = nm["p97_99"].get(k)
        if ks in kl_per_neuron:
            ksh = kl_per_neuron[ks]["kl_share_4band_norm"]
            kl_str = (
                f"{ksh['p99.9+']:.3f} / {ksh['p97_99.9']:.3f} / "
                f"{ksh['p75_97']:.3f} / {ksh['lt_p75']:.3f}"
            )
        else:
            kl_str = "pending 278-neuron KL"
        short_grp = {"high_causal_top100_1_00": "A (high)", "negative_causal_top100_1_00": "B (neg)", "failing_detection_top100_lt_0_80": "C (fail)"}[grp]
        name_tag = f" ({g['named_neuron']})" if g.get("named_neuron") else ""
        p97_s = f"{p97:.3f}" if p97 is not None else "n/a"
        lines.append(
            f"| {idx} | {short_grp} | `{k}`{name_tag} | `L{g['layer']:02d} N{g['neuron']}` | {pol} "
            f"| {t100:.3f} | {p999:.3f} | {p97_s} | {g['sim_cal_rec']:+.3f} | {g['sim_cal_rec_active']:+.3f} "
            f"| {g['strong20_rec']:+.3f} | {kl_str} |"
        )

    current_group = None
    for idx, (grp, k) in enumerate(selected, start=1):
        if grp != current_group:
            current_group = grp
            lines.extend(["", f"## {group_titles[grp]}"])

        ks = str(k)
        g = gap_per_neuron[ks]
        pol = "bipolar" if g["is_bipolar"] else "unipolar"
        name_tag = f" — `{g['named_neuron']}`" if g.get("named_neuron") else ""
        t100 = nm["top100"].get(k)
        p999 = nm["p99.9_plus"].get(k)
        p97 = nm["p97_99"].get(k)
        p97_s = f"{p97:.3f}" if p97 is not None else "n/a"
        max_abs = float(g["max_abs_act"])

        lines.extend(
            [
                "",
                f"### {idx}. Neuron `k={k}` (`L{g['layer']:02d}_N{g['neuron']}`{name_tag})",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Polarity / Active Poles | `{pol}` ({', '.join(f'`{p}`' for p in g['active_pole_ids'])}) |",
                f"| Detection (`gpt-6.1-terra@none`) | `top100 = {t100:.3f}`, `p99.9+ = {p999:.3f}`, `p97–99 = {p97_s}` |",
                f"| Ulto85 Causal Scores | `sim_cal_rec = {g['sim_cal_rec']:+.3f}`, `sim_cal_rec_active = {g['sim_cal_rec_active']:+.3f}`, `strong20_rec = {g['strong20_rec']:+.3f}` |",
            ]
        )
        if ks in kl_per_neuron:
            ksh = kl_per_neuron[ks]["kl_share_4band_norm"]
            lines.append(
                f"| 4-Band Exact `fp32` KL Share | `p99.9+ = {ksh['p99.9+']:.3f}`, `p97–99.9 = {ksh['p97_99.9']:.3f}`, "
                f"`p75–97 = {ksh['p75_97']:.3f}`, `< p75 = {ksh['lt_p75']:.3f}` |"
            )
        else:
            lines.append("| 4-Band Exact `fp32` KL Share | Not in `n=60` subset (populated on full 278-neuron KL run) |")

        lines.extend(["", "**Verbatim Ulto85 Pole Label(s):**"])
        ordered_poles = list(g["active_pole_ids"])
        # Put dominant pole first
        dom = g.get("dominant_pole_id")
        if dom in ordered_poles:
            ordered_poles = [dom] + [p for p in ordered_poles if p != dom]
        for pid in ordered_poles:
            p_text = str(pole_info[pid]["pole_text"]).replace("\n", " ").strip()
            lines.append(f"- `{pid}`: {p_text}")

        for sec_label, snip_dict, j_dict in (
            ("Top-100 Positive Snippets (5 sampled)", snips_top100[k], j_top100),
            ("`p97–99` Positive Snippets (5 sampled)", snips_p97_99[k], j_p97_99),
        ):
            lines.extend(["", f"**{sec_label}:**"])
            sampled = _sample_five_pairs_for_neuron(snip_dict, ordered_poles, n_total=5)
            if not sampled:
                lines.append("- *(No scored positive pairs available in this band.)*")
                continue
            for si, pair in enumerate(sampled, start=1):
                pid = str(pair["pole_id"])
                pair_id = str(pair["pair_id"])
                pos = pair["pos"]
                act_dir = float(pos["act_dir"])
                pct_max = (100.0 * act_dir / max_abs) if max_abs > 0.0 else 0.0
                marked = str(pos["marked_text"]).replace("\n", " ⏎ ").replace("`", "'")
                jrow = j_dict.get(pair_id)
                if jrow is not None:
                    verdict = "CORRECT" if bool(jrow["correct"]) else "WRONG"
                    pred_slot = str(jrow["predicted_slot"])
                    pos_slot = str(jrow["pos_slot"])
                    v_str = f"`{verdict}` (pred `{pred_slot}`, pos `{pos_slot}`)"
                else:
                    v_str = "`UNMATCHED`"
                lines.append(
                    f"{si}. `{pid}` (`pair_idx={pair['pair_idx']}`) | `act_dir={act_dir:.3f}` "
                    f"(`{pct_max:.1f}%` of max) | judge: {v_str}"
                )
                lines.append(f"   - `{marked}`")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": output_path.as_posix(), "selected": [{"group": g, "k": k} for g, k in selected]}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def run_effect_weighted(
    kl_path: Path | None = None,
    kl_n60_path: Path | None = DEFAULT_KL_60_PATH,
    gap_path: Path = DEFAULT_GAP_4B_PATH,
    active_poles_path: Path = DEFAULT_ACTIVE_POLES_4B_PATH,
    ulto_neurons_path: Path = DEFAULT_ULTO_NEURONS_PATH,
    ulto_scores_path: Path = DEFAULT_ULTO_SCORES_PATH,
    judgements_4b_path: Path = DEFAULT_JUDGEMENTS_4B_PATH,
    judgements_top100_path: Path = DEFAULT_JUDGEMENTS_4B_TOP100_PATH,
    judgements_multiband_path: Path = DEFAULT_JUDGEMENTS_4B_MULTIBAND_PATH,
    judgements_fpr_path: Path = DEFAULT_JUDGEMENTS_4B_FPR_PATH,
    snippets_4b_path: Path = DEFAULT_SNIPPETS_4B_PATH,
    snippets_top100_path: Path = DEFAULT_SNIPPETS_4B_TOP100_PATH,
    ek_17b_path: Path = DEFAULT_EK_17B_PATH,
    causal_17b_path: Path = DEFAULT_CAUSAL_17B_PATH,
    judgements_17b_path: Path = DEFAULT_JUDGEMENTS_17B_PATH,
    subband_17b_path: Path = DEFAULT_SUBBAND_17B_PATH,
    poles_17b_path: Path = DEFAULT_POLES_17B_PATH,
    snippets_17b_path: Path = DEFAULT_SNIPPETS_17B_PATH,
    output_json_path: Path = DEFAULT_OUTPUT_JSON_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    reading_sheet_path: Path | None = DEFAULT_READING_SHEET_PATH,
    skip_reading_sheet: bool = False,
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all effect-weighted lift analyses, figure rendering, and optional reading sheet."""
    if kl_path is None:
        kl_path = DEFAULT_KL_278_PATH if DEFAULT_KL_278_PATH.is_file() else DEFAULT_KL_60_PATH

    det_tables_4b = load_qwen4b_detection_tables(
        active_poles_path=active_poles_path,
        judgements_4b_path=judgements_4b_path,
        judgements_top100_path=judgements_top100_path,
        judgements_multiband_path=judgements_multiband_path,
    )

    res_4b = analyze_qwen4b(
        kl_path=kl_path,
        kl_n60_path=kl_n60_path,
        gap_path=gap_path,
        active_poles_path=active_poles_path,
        det_tables=det_tables_4b,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
    )

    res_17b = analyze_qwen17b(
        ek_17b_path=ek_17b_path,
        causal_17b_path=causal_17b_path,
        judgements_17b_path=judgements_17b_path,
        poles_17b_path=poles_17b_path,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
    )

    gf_table4 = build_goodfire_table4_comparison(
        det_tables_4b=det_tables_4b,
        subband_17b_path=subband_17b_path,
        res_17b=res_17b,
        judgements_17b_path=judgements_17b_path,
        poles_17b_path=poles_17b_path,
        snippets_17b_path=snippets_17b_path,
    )

    disclosures = build_statistical_disclosures(
        det_tables_4b=det_tables_4b,
        gap_path=gap_path,
        ulto_neurons_path=ulto_neurons_path,
        ulto_scores_path=ulto_scores_path,
        fpr_4b_path=judgements_fpr_path,
        snippets_top100_path=snippets_top100_path,
        res_17b=res_17b,
        existing_output_path=output_json_path if output_json_path.is_file() else None,
    )

    plot_main_detection_vs_kl_by_band(res_4b=res_4b, res_17b=res_17b, output_path=figure_path)

    reading_sheet_meta: dict[str, Any] | None = None
    if (
        not skip_reading_sheet
        and reading_sheet_path is not None
        and snippets_top100_path.is_file()
        and snippets_4b_path.is_file()
    ):
        reading_sheet_meta = generate_reading_sheet_12_neurons(
            kl_path=kl_path,
            gap_path=gap_path,
            det_tables_4b=det_tables_4b,
            snippets_top100_path=snippets_top100_path,
            snippets_4b_path=snippets_4b_path,
            judgements_top100_path=judgements_top100_path,
            judgements_4b_path=judgements_4b_path,
            output_path=reading_sheet_path,
        )

    payload: dict[str, Any] = {
        "schema_version": "effect_weighted_1.0",
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "qwen3_4b": res_4b,
        "qwen3_17b": res_17b,
        "goodfire_table4_comparison": gf_table4,
        "statistical_disclosures": disclosures,
        "figure_path": figure_path.as_posix(),
        "reading_sheet": reading_sheet_meta,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute effect-weighted detection lift, KL band comparison, and statistical disclosures."
    )
    p.add_argument("--kl-json", type=Path, default=None, help="Path to 4B fp32 KL JSON (auto-detects all278 vs n60).")
    p.add_argument("--kl-n60-json", type=Path, default=DEFAULT_KL_60_PATH)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    p.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    p.add_argument("--reading-sheet", type=Path, default=DEFAULT_READING_SHEET_PATH)
    p.add_argument("--skip-reading-sheet", action="store_true", help="Do not regenerate the 12-neuron reading sheet.")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_effect_weighted(
        kl_path=args.kl_json,
        kl_n60_path=args.kl_n60_json,
        output_json_path=args.output_json,
        figure_path=args.figure,
        reading_sheet_path=args.reading_sheet,
        skip_reading_sheet=bool(args.skip_reading_sheet),
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
    )
    q4 = payload["qwen3_4b"]
    q17 = payload["qwen3_17b"]
    print(
        f"effect_weighted complete (4B n={q4['n_kl_neurons']}, 1.7B n={q17['n_neurons']}): "
        f"4B EWL={q4['all_kl_neurons']['ewl_full_energy_merge']['mean']:.3f} "
        f"(2-sided={q4['bipolar_only']['ewl_full_energy_merge']['mean']:.3f}, "
        f">=p75={q4['all_kl_neurons']['ewl_ge_p75_renorm_energy_merge']['mean']:.3f}, "
        f"p99.9+ lift={q4['all_kl_neurons']['top_p999_lift']['mean']:.3f}); "
        f"1.7B EWL={q17['all_50']['ewl_full_equal_merge']['mean']:.3f} "
        f"(p99.9+ lift={q17['all_50']['top_p999_lift']['mean']:.3f}, "
        f"raw over no_label={q17['all_50']['lift_over_no_label_floor']['ewl_raw_full_equal_merge']['mean']:+.3f}) "
        f"-> wrote {args.output_json} & {args.figure}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
