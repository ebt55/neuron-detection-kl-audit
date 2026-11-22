"""Activation-energy gap decomposition and multi-band + FPR snippet construction for Qwen3-4B.

Strictly CPU-only, strictly isolated from Qwen3-1.7B files and existing Qwen3-4B files
(`snippets_4b.jsonl`, `snippets_4b_top100.jsonl`, `judgements_4b.jsonl` are read-only):

Outputs:
1. `results/qwen4b/gap_decomposition_4b.json`:
   - Exact centered L2 variance shares (and L1 sensitivity shares) across:
     - 4-band (`p99.9+`, `p97_99.9`, `p75_97`, `lt_p75`) summing strictly to 1.000
     - 5-band (`p99.9+`, `p99_99.9`, `p97_99`, `p75_97`, `lt_p75`) summing strictly to 1.000
     - 6-band (`top100_thresh`, `rest_of_p99.9+`, `p99_99.9`, `p97_99`, `p75_97`, `bulk_lt_p75`)
     - 6-band-with-p25 (`p99.9+`, `p99_99.9`, `p97_99`, `p75_97`, `p25_75`, `dead_or_opp`)
     - Amplitude shares (`energy_ge_50pct_max`, `energy_ge_20pct_max`, `energy_lt_20pct_max`)
     - SwiGLU gate shares (`energy_gate_open` [Q1+Q2] vs `energy_gate_closed` [Q3+Q4])
   - Correlations (Spearman rho, Pearson r, R^2, neuron-bootstrap 95% CIs) vs Ulto85
     `strong20_rec`, `strong50_rec`, `sim_cal_rec`, `sim_cal_rec_active`.
   - Mutually exclusive diagnostic failure-bucket partition of the 235 neurons passing
     `top100 >= 0.80` (and separately 204 passing `p99.9+ >= 0.80`, and all 278 neurons).
2. `data/qwen4b/snippets_4b_multiband.jsonl`:
   - Exact symmetric 69-token pairs (`60 + <<target>> + 8`, 0 `<bos>`, 0 same-doc leaks,
     strict Tier-0 `dead_zone` distractors with `|a| < p25_abs` & whole-window `< p90`,
     exact 3/3 A/B balance per pole x band) for 3 new bands:
     - `p99_99.9` (`p99 <= d_pos < p99.9`, 6 pairs/pole)
     - `p75_97` (`p75 <= d_pos < p97`, 6 pairs/pole, clean-window-first with tier logging)
     - `p25_75` (`p25_abs <= d_pos < p75`, 6 pairs/pole, clean-window-first with tier logging)
3. `data/qwen4b/snippets_4b_fpr.jsonl`:
   - Ordinary-text single-snippet YES/NO precision probes for all 424 active poles:
     - `tier="core"` (`5,088` probes for `terra@none` & `luna@none`): 10 random ordinary
       non-activating interior snippets (`|a| < 0.20 * max_act_dir` & `d < p90`, distinct docs)
       + 2 top-100 firing snippets per pole.
     - `tier="luna_extra"` (`5,088` extra probes so `luna@none` gets 20 ordinary + 4 top-100
       = `10,176` probes total).
4. `results/qwen4b/structural_checks_4b_multiband.json`:
   - Full hard-invariant audit covering both `snippets_4b_multiband.jsonl` and `snippets_4b_fpr.jsonl`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from qwen4b_eval import (
    FINEWEB_MIN_ROW_IDX,
    MODEL_ID_4B,
    SEED_4B,
    ensure_4b_dirs,
)
from qwen4b_eval.snippets_4b import _stable_rng_4b, balanced_even_slots
from w_audit.config import DEDUP_SUFFIX_LEN, QUADRANTS, WindowSpec
from w_audit.extract import QUAD_CODE_TO_NAME
from w_audit.snippets import (
    _sample_distinct_doc_indices,
    _select_strict_tier0_pairs,
    _suffix_key,
    compute_other_and_window_max_1d,
    render_marked_window,
)

MULTIBAND_NAMES: tuple[str, ...] = ("p99_99.9", "p75_97", "p25_75")
MULTIBAND_PAIRS_PER_POLE: int = 6
FPR_ORD_CORE_PER_POLE: int = 10
FPR_ORD_EXTRA_PER_POLE: int = 10
FPR_TOP_CORE_PER_POLE: int = 2
FPR_TOP_EXTRA_PER_POLE: int = 2

FAILURE_BUCKETS: tuple[str, ...] = (
    "pass_causal",
    "overbroad_fp_damage",
    "weak_mass_dominated",
    "active_sim_failure",
    "other_partial",
)


# ---------------------------------------------------------------------------
# Bootstrap Correlation & Bucket Helpers
# ---------------------------------------------------------------------------


def bootstrap_corr_stats(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 1000,
    seed: int = SEED_4B,
) -> dict[str, Any]:
    """Compute Spearman rho, Pearson r, R^2, and 95% bootstrap CIs over the input sequence."""
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax, ay = ax[mask], ay[mask]
    n = int(len(ax))
    if n < 3 or float(np.ptp(ax)) <= 1e-12 or float(np.ptp(ay)) <= 1e-12:
        return {
            "n": n,
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "spearman_ci_low": float("nan"),
            "spearman_ci_high": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "pearson_ci_low": float("nan"),
            "pearson_ci_high": float("nan"),
            "r_squared": float("nan"),
        }

    rho, rho_p = spearmanr(ax, ay)
    r, r_p = pearsonr(ax, ay)
    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(n_resamples, n))
    boot_rho: list[float] = []
    boot_r: list[float] = []
    for idx in idx_mat:
        bx, by = ax[idx], ay[idx]
        if float(np.ptp(bx)) <= 1e-12 or float(np.ptp(by)) <= 1e-12:
            continue
        brho, _ = spearmanr(bx, by)
        br, _ = pearsonr(bx, by)
        if math.isfinite(brho):
            boot_rho.append(float(brho))
        if math.isfinite(br):
            boot_r.append(float(br))

    return {
        "n": n,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "spearman_ci_low": float(np.percentile(boot_rho, 2.5)) if boot_rho else float("nan"),
        "spearman_ci_high": float(np.percentile(boot_rho, 97.5)) if boot_rho else float("nan"),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "pearson_ci_low": float(np.percentile(boot_r, 2.5)) if boot_r else float("nan"),
        "pearson_ci_high": float(np.percentile(boot_r, 97.5)) if boot_r else float("nan"),
        "r_squared": float(r * r),
    }


def assign_failure_bucket(
    sim_cal_rec: float,
    sim_cal_rec_active: float,
    strong20_rec: float,
) -> str:
    """Mutually exclusive diagnostic bucket for explaining causal recovery failure.

    Precedence order (guaranteeing PCR lands in `overbroad_fp_damage` and all 5 buckets
    form a strict 100%-exhaustive partition):
      1. `pass_causal`: `sim_cal_rec >= 0.50`
      2. `overbroad_fp_damage`: `sim_cal_rec < 0.0` AND `sim_cal_rec_active >= 0.50`
      3. `weak_mass_dominated`: `strong20_rec < 0.25` (remaining `sim_cal_rec < 0.50`)
      4. `active_sim_failure`: `strong20_rec >= 0.25` AND `sim_cal_rec_active < 0.50`
      5. `other_partial`: remaining (`sim_cal_rec in [0, 0.50)`, `strong20_rec >= 0.25`,
         `sim_cal_rec_active >= 0.50`)
    """
    if sim_cal_rec >= 0.50:
        return "pass_causal"
    if sim_cal_rec < 0.0 and sim_cal_rec_active >= 0.50:
        return "overbroad_fp_damage"
    if strong20_rec < 0.25:
        return "weak_mass_dominated"
    if sim_cal_rec_active < 0.50:
        return "active_sim_failure"
    return "other_partial"


# ---------------------------------------------------------------------------
# Exact Energy Partition Helpers (L2 variance share + L1 sensitivity share)
# ---------------------------------------------------------------------------


def _assign_pole_band_codes(
    d_nb: np.ndarray,
    abs_nb: np.ndarray,
    p25_abs: float,
    p75: float,
    p97: float,
    p99: float,
    p99_9: float,
    top100_thresh: float,
) -> np.ndarray:
    """Assign each token an exclusive fine band code 0..6:
      0: top100_thresh   (d >= max(top100_thresh, p99.9))
      1: rest_of_p99.9+  (p99.9 <= d < max(top100_thresh, p99.9))
      2: p99_99.9        (p99 <= d < p99.9)
      3: p97_99          (p97 <= d < p99)
      4: p75_97          (p75 <= d < p97)
      5: p25_75          (d < p75 & d > 0 & |a| >= p25_abs)
      6: dead_or_opp     (everything else in d < p75)
    """
    top_eff = max(float(top100_thresh), float(p99_9))
    codes = np.full(d_nb.shape, 6, dtype=np.uint8)
    codes[(d_nb > 0.0) & (abs_nb >= p25_abs) & (d_nb < p75)] = 5
    codes[(d_nb >= p75) & (d_nb < p97)] = 4
    codes[(d_nb >= p97) & (d_nb < p99)] = 3
    codes[(d_nb >= p99) & (d_nb < p99_9)] = 2
    codes[(d_nb >= p99_9) & (d_nb < top_eff)] = 1
    codes[d_nb >= top_eff] = 0
    return codes


def _codes_to_band_shares(codes: np.ndarray, weights: np.ndarray, total_w: float) -> dict[str, dict[str, float]]:
    """Aggregate fine band codes (0..6) into exact 4-band, 5-band, 6-band, and with-p25 partitions."""
    denom = total_w if total_w > 0.0 else 1.0
    sums = [float(np.sum(weights[codes == c])) / denom for c in range(7)]
    c0, c1, c2, c3, c4, c5, c6 = sums
    p999_plus = c0 + c1
    p97_999 = c2 + c3
    bulk_lt_p75 = c5 + c6
    return {
        "bands_4": {
            "p99.9+": p999_plus,
            "p97_99.9": p97_999,
            "p75_97": c4,
            "lt_p75": bulk_lt_p75,
        },
        "bands_5": {
            "p99.9+": p999_plus,
            "p99_99.9": c2,
            "p97_99": c3,
            "p75_97": c4,
            "lt_p75": bulk_lt_p75,
        },
        "bands_6": {
            "top100_thresh": c0,
            "rest_of_p99.9+": c1,
            "p99_99.9": c2,
            "p97_99": c3,
            "p75_97": c4,
            "bulk_lt_p75": bulk_lt_p75,
        },
        "bands_with_p25_75": {
            "p99.9+": p999_plus,
            "p99_99.9": c2,
            "p97_99": c3,
            "p75_97": c4,
            "p25_75": c5,
            "dead_or_opp": c6,
        },
    }


def compute_gap_decomposition_4b(
    base_dir: Path | None = None,
    seed: int = SEED_4B,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute activation-energy shares, correlations, and failure-bucket decomposition."""
    t0 = time.time()
    dirs = ensure_4b_dirs(base_dir)
    cache_dir = dirs["cache"]
    data_dir = dirs["data"]
    results_dir = dirs["results"]

    catalog = json.loads((data_dir / "active_poles.json").read_text(encoding="utf-8"))
    cache_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    ulto_scores_raw = json.loads((dirs["ulto85"] / "e3_scores.json").read_text(encoding="utf-8"))
    ulto_by_k = {int(r["k"]): r for r in ulto_scores_raw["neurons"]}

    metrics_path = results_dir / "metrics_4b.json"
    metrics_per_neuron: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics_per_neuron = json.loads(metrics_path.read_text(encoding="utf-8")).get("per_neuron", {})

    sparsity_path = results_dir / "pole_sparsity_diagnostics.json"
    top100_thresh_by_pole: dict[str, float] = {}
    if sparsity_path.is_file():
        sp_doc = json.loads(sparsity_path.read_text(encoding="utf-8"))
        for r in sp_doc.get("poles", []):
            top100_thresh_by_pole[str(r["pole_id"])] = float(r["top100_pool"]["top100_min_act_dir"])

    acts = np.load(cache_dir / "acts.npy", mmap_mode="r")
    quads = np.load(cache_dir / "quads.npy", mmap_mode="r")
    token_ids = np.load(cache_dir / "token_ids.npy", mmap_mode="r")
    pos_in_chunk = np.load(cache_dir / "pos_in_chunk.npy", mmap_mode="r")

    col_to_neuron = [tuple(x) for x in cache_meta["col_to_neuron"]]
    neuron_to_col = {(int(L), int(n)): col for col, (L, n) in enumerate(col_to_neuron)}
    special_ids = set(int(x) for x in cache_meta.get("special_token_ids", []))

    non_bos = pos_in_chunk > 0
    if special_ids:
        non_bos = non_bos & (~np.isin(token_ids, list(special_ids)))
    n_non_bos = int(np.sum(non_bos))

    poles_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in catalog["poles"]:
        poles_by_k[int(p["k"])].append(p)

    per_pole_out: dict[str, dict[str, Any]] = {}
    per_neuron_out: dict[str, dict[str, Any]] = {}

    for n_rec in sorted(catalog["neurons"], key=lambda r: int(r["k"])):
        k = int(n_rec["k"])
        L = int(n_rec["layer"])
        n_idx = int(n_rec["neuron"])
        col = neuron_to_col[(L, n_idx)]
        a_nb = acts[non_bos, col].astype(np.float64)
        q_nb = quads[non_bos, col]
        abs_nb = np.abs(a_nb)

        mean_a = float(np.mean(a_nb))
        dev = a_nb - mean_a
        l2 = dev * dev
        l1 = np.abs(dev)
        tot_l2 = float(np.sum(l2))
        tot_l1 = float(np.sum(l1))
        denom_l2 = tot_l2 if tot_l2 > 0.0 else 1.0
        denom_l1 = tot_l1 if tot_l1 > 0.0 else 1.0

        m_abs = float(np.max(abs_nb))
        ulto_rec = ulto_by_k.get(k, {})
        ulto_M = float(ulto_rec.get("M", m_abs))

        # Amplitude shares (matching Ulto85 strong20 / strong50 directly)
        mask_50 = abs_nb >= (0.50 * m_abs)
        mask_20 = abs_nb >= (0.20 * m_abs)
        mask_20_ulto = abs_nb >= (0.20 * ulto_M)
        energy_ge_50 = float(np.sum(l2[mask_50]) / denom_l2)
        energy_ge_20 = float(np.sum(l2[mask_20]) / denom_l2)
        energy_lt_20 = float(1.0 - energy_ge_20)
        energy_ge_20_ulto = float(np.sum(l2[mask_20_ulto]) / denom_l2)
        l1_ge_50 = float(np.sum(l1[mask_50]) / denom_l1)
        l1_ge_20 = float(np.sum(l1[mask_20]) / denom_l1)
        l1_lt_20 = float(1.0 - l1_ge_20)

        # SwiGLU gate energy shares (Q1+Q2 open vs Q3+Q4 closed)
        gate_open = q_nb < 2
        energy_gate_open = float(np.sum(l2[gate_open]) / denom_l2)
        energy_gate_closed = float(np.sum(l2[~gate_open]) / denom_l2)
        l1_gate_open = float(np.sum(l1[gate_open]) / denom_l1)
        l1_gate_closed = float(np.sum(l1[~gate_open]) / denom_l1)
        energy_by_quad = {
            QUAD_CODE_TO_NAME[c]: float(np.sum(l2[q_nb == c]) / denom_l2) for c in range(4)
        }

        # Per-pole band partitions & thresholds
        p25_abs = float(n_rec.get("p25_abs", np.quantile(abs_nb, 0.25)))
        pole_codes_by_sign: dict[int, np.ndarray] = {}
        pole_list = poles_by_k[k]

        for p in pole_list:
            pid = str(p["pole_id"])
            sign = int(p["sign"])
            d_nb = float(sign) * a_nb
            edges = p["band_edges_dir"]
            p75 = float(np.quantile(d_nb, 0.75))
            p90 = float(edges["p90"])
            p97 = float(edges["p97"])
            p99 = float(edges["p99"])
            p99_9 = float(edges["p99.9"])
            top100_th = float(top100_thresh_by_pole.get(pid, p99_9))

            p_codes = _assign_pole_band_codes(
                d_nb=d_nb,
                abs_nb=abs_nb,
                p25_abs=p25_abs,
                p75=p75,
                p97=p97,
                p99=p99,
                p99_9=p99_9,
                top100_thresh=top100_th,
            )
            pole_codes_by_sign[sign] = p_codes
            l2_parts = _codes_to_band_shares(p_codes, l2, tot_l2)
            l1_parts = _codes_to_band_shares(p_codes, l1, tot_l1)

            max_d = float(np.max(d_nb))
            d_mask_20 = d_nb >= (0.20 * max_d) if max_d > 0.0 else np.zeros_like(d_nb, dtype=bool)
            d_mask_50 = d_nb >= (0.50 * max_d) if max_d > 0.0 else np.zeros_like(d_nb, dtype=bool)
            per_pole_out[pid] = {
                "pole_id": pid,
                "k": k,
                "layer": L,
                "neuron": n_idx,
                "sign": sign,
                "is_bipolar": bool(p["is_bipolar"]),
                "is_dominant_pole": bool(p.get("is_dominant_pole", False)),
                "p25_abs": p25_abs,
                "p75": p75,
                "p90": p90,
                "p97": p97,
                "p99": p99,
                "p99.9": p99_9,
                "top100_thresh": top100_th,
                "max_act_dir": max_d,
                "energy_l2": l2_parts,
                "energy_l1": l1_parts,
                "energy_dir_ge_20pct_max": float(np.sum(l2[d_mask_20]) / denom_l2),
                "energy_dir_ge_50pct_max": float(np.sum(l2[d_mask_50]) / denom_l2),
                "energy_same_sign_total": float(np.sum(l2[d_nb > 0.0]) / denom_l2),
            }

        # Per-neuron 100%-exhaustive partition (unipolar: active pole; bipolar: sign(a_t) pole)
        if len(pole_codes_by_sign) == 1:
            neuron_codes = next(iter(pole_codes_by_sign.values()))
        else:
            neuron_codes = np.where(a_nb >= 0.0, pole_codes_by_sign[1], pole_codes_by_sign[-1])

        n_l2_parts = _codes_to_band_shares(neuron_codes, l2, tot_l2)
        n_l1_parts = _codes_to_band_shares(neuron_codes, l1, tot_l1)

        # Strict sanity assertion: 4-band and 6-band sum to 1.000000
        s4 = sum(n_l2_parts["bands_4"].values())
        s6 = sum(n_l2_parts["bands_6"].values())
        if abs(s4 - 1.0) > 1e-9 or abs(s6 - 1.0) > 1e-9:
            raise AssertionError(f"Neuron k={k} energy shares do not sum to 1: s4={s4}, s6={s6}")

        m_row = metrics_per_neuron.get(str(k), {})
        sim_cal_rec = float(ulto_rec.get("sim_cal_rec", m_row.get("sim_cal_rec", float("nan"))))
        sim_cal_rec_active = float(
            ulto_rec.get("sim_cal_rec_active", m_row.get("sim_cal_rec_active", float("nan")))
        )
        strong20_rec = float(ulto_rec.get("strong20_rec", m_row.get("strong20_rec", float("nan"))))
        strong50_rec = float(ulto_rec.get("strong50_rec", float("nan")))
        det_top100 = m_row.get("det_top100_terra_mean")
        det_p999 = m_row.get("det_p999_terra_mean")
        bucket = assign_failure_bucket(sim_cal_rec, sim_cal_rec_active, strong20_rec)

        per_neuron_out[str(k)] = {
            "k": k,
            "neuron_id": str(n_rec["neuron_id"]),
            "layer": L,
            "neuron": n_idx,
            "is_bipolar": bool(n_rec["is_bipolar"]),
            "n_active_poles": int(n_rec["n_active_poles"]),
            "active_pole_ids": list(n_rec["active_pole_ids"]),
            "dominant_pole_id": n_rec.get("dominant_pole_id"),
            "named_neuron": n_rec.get("named_neuron"),
            "label": str(n_rec["label"]),
            "mean_act": mean_a,
            "var_act": float(tot_l2 / max(n_non_bos, 1)),
            "max_abs_act": m_abs,
            "ulto85_M": ulto_M,
            "energy_l2": n_l2_parts,
            "energy_l1": n_l1_parts,
            "energy_p99.9+": float(n_l2_parts["bands_4"]["p99.9+"]),
            "energy_p97+": float(n_l2_parts["bands_4"]["p99.9+"] + n_l2_parts["bands_4"]["p97_99.9"]),
            "energy_lt_p97": float(n_l2_parts["bands_4"]["p75_97"] + n_l2_parts["bands_4"]["lt_p75"]),
            "energy_ge_50pct_max": energy_ge_50,
            "energy_ge_20pct_max": energy_ge_20,
            "energy_lt_20pct_max": energy_lt_20,
            "energy_ge_20pct_ulto_M": energy_ge_20_ulto,
            "l1_ge_50pct_max": l1_ge_50,
            "l1_ge_20pct_max": l1_ge_20,
            "l1_lt_20pct_max": l1_lt_20,
            "energy_gate_open": energy_gate_open,
            "energy_gate_closed": energy_gate_closed,
            "l1_gate_open": l1_gate_open,
            "l1_gate_closed": l1_gate_closed,
            "energy_by_quadrant": energy_by_quad,
            "sim_cal_rec": sim_cal_rec,
            "sim_cal_rec_active": sim_cal_rec_active,
            "strong20_rec": strong20_rec,
            "strong50_rec": strong50_rec,
            "corr": ulto_rec.get("corr"),
            "mono": ulto_rec.get("mono"),
            "det_top100_terra_mean": det_top100,
            "det_p999_terra_mean": det_p999,
            "failure_bucket": bucket,
        }

    # ---- Compute key correlations across all 278 neurons ----
    all_rows = [per_neuron_out[str(k)] for k in sorted(int(x) for x in per_neuron_out)]
    correlations = {
        "energy_ge_20pct_max_vs_strong20_rec": bootstrap_corr_stats(
            [r["energy_ge_20pct_max"] for r in all_rows],
            [r["strong20_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed,
        ),
        "l1_ge_20pct_max_vs_strong20_rec": bootstrap_corr_stats(
            [r["l1_ge_20pct_max"] for r in all_rows],
            [r["strong20_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 1,
        ),
        "energy_ge_20pct_ulto_M_vs_strong20_rec": bootstrap_corr_stats(
            [r["energy_ge_20pct_ulto_M"] for r in all_rows],
            [r["strong20_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 2,
        ),
        "energy_ge_50pct_max_vs_strong50_rec": bootstrap_corr_stats(
            [r["energy_ge_50pct_max"] for r in all_rows],
            [r["strong50_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 3,
        ),
        "energy_ge_20pct_max_vs_sim_cal_rec": bootstrap_corr_stats(
            [r["energy_ge_20pct_max"] for r in all_rows],
            [r["sim_cal_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 4,
        ),
        "energy_p99_9_plus_vs_sim_cal_rec": bootstrap_corr_stats(
            [r["energy_p99.9+"] for r in all_rows],
            [r["sim_cal_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 5,
        ),
        "energy_p97_plus_vs_sim_cal_rec": bootstrap_corr_stats(
            [r["energy_p97+"] for r in all_rows],
            [r["sim_cal_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 6,
        ),
        "energy_gate_open_vs_sim_cal_rec": bootstrap_corr_stats(
            [r["energy_gate_open"] for r in all_rows],
            [r["sim_cal_rec"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 7,
        ),
        "energy_ge_20pct_max_vs_sim_cal_rec_active": bootstrap_corr_stats(
            [r["energy_ge_20pct_max"] for r in all_rows],
            [r["sim_cal_rec_active"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 8,
        ),
        "energy_p99_9_plus_vs_sim_cal_rec_active": bootstrap_corr_stats(
            [r["energy_p99.9+"] for r in all_rows],
            [r["sim_cal_rec_active"] for r in all_rows],
            n_resamples=n_bootstrap,
            seed=seed + 9,
        ),
    }

    # ---- Summarize failure-bucket partitions for top100>=0.80, p99.9+>=0.80, and all 278 ----
    def _summarize_bucket_subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        n_sub = len(rows)
        buckets_summary: dict[str, Any] = {}
        overlap_overbroad_and_weak = sum(
            1
            for r in rows
            if r["sim_cal_rec"] < 0.0
            and r["sim_cal_rec_active"] >= 0.50
            and r["strong20_rec"] < 0.25
        )
        for bname in FAILURE_BUCKETS:
            brows = [r for r in rows if r["failure_bucket"] == bname]
            cnt = len(brows)
            layer_counts = Counter(int(r["layer"]) for r in brows)
            if cnt > 0:
                med_e999 = float(np.median([r["energy_p99.9+"] for r in brows]))
                med_elt97 = float(np.median([r["energy_lt_p97"] for r in brows]))
                med_e20 = float(np.median([r["energy_ge_20pct_max"] for r in brows]))
                med_s20 = float(np.median([r["strong20_rec"] for r in brows]))
                med_gc = float(np.median([r["energy_gate_closed"] for r in brows]))
                med_sim = float(np.median([r["sim_cal_rec"] for r in brows]))
                med_act = float(np.median([r["sim_cal_rec_active"] for r in brows]))
                mean_bands_4 = {
                    bk: float(np.mean([r["energy_l2"]["bands_4"][bk] for r in brows]))
                    for bk in ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
                }
            else:
                med_e999 = med_elt97 = med_e20 = med_s20 = med_gc = med_sim = med_act = None
                mean_bands_4 = {bk: None for bk in ("p99.9+", "p97_99.9", "p75_97", "lt_p75")}
            buckets_summary[bname] = {
                "count": cnt,
                "share": cnt / max(n_sub, 1),
                "by_layer": {str(L): int(layer_counts[L]) for L in sorted(layer_counts)},
                "median_energy_p99_9_plus": med_e999,
                "median_energy_lt_p97": med_elt97,
                "median_energy_ge_20pct_max": med_e20,
                "median_strong20_rec": med_s20,
                "median_energy_gate_closed": med_gc,
                "median_sim_cal_rec": med_sim,
                "median_sim_cal_rec_active": med_act,
                "mean_bands_4_energy": mean_bands_4,
                "named_neurons": [r["named_neuron"] for r in brows if r.get("named_neuron")],
            }
        return {
            "n_total": n_sub,
            "causal_fail_counts": {
                "sim_cal_rec_lt_0_50": sum(1 for r in rows if r["sim_cal_rec"] < 0.50),
                "sim_cal_rec_lt_0_25": sum(1 for r in rows if r["sim_cal_rec"] < 0.25),
                "sim_cal_rec_le_0_00": sum(1 for r in rows if r["sim_cal_rec"] <= 0.0),
            },
            "overlap_overbroad_fp_and_strong20_lt_0_25": overlap_overbroad_and_weak,
            "buckets": buckets_summary,
        }

    rows_top100_ge_80 = [
        r for r in all_rows if r["det_top100_terra_mean"] is not None and float(r["det_top100_terra_mean"]) >= 0.80
    ]
    rows_p999_ge_80 = [
        r for r in all_rows if r["det_p999_terra_mean"] is not None and float(r["det_p999_terra_mean"]) >= 0.80
    ]

    named_summary = {}
    for r in all_rows:
        if r.get("named_neuron"):
            named_summary[str(r["named_neuron"])] = {
                "k": r["k"],
                "neuron_id": r["neuron_id"],
                "failure_bucket": r["failure_bucket"],
                "det_top100_terra_mean": r["det_top100_terra_mean"],
                "det_p999_terra_mean": r["det_p999_terra_mean"],
                "sim_cal_rec": r["sim_cal_rec"],
                "sim_cal_rec_active": r["sim_cal_rec_active"],
                "strong20_rec": r["strong20_rec"],
                "strong50_rec": r["strong50_rec"],
                "energy_ge_20pct_max": r["energy_ge_20pct_max"],
                "energy_ge_50pct_max": r["energy_ge_50pct_max"],
                "energy_p99.9+": r["energy_p99.9+"],
                "energy_lt_p97": r["energy_lt_p97"],
                "energy_gate_closed": r["energy_gate_closed"],
                "bands_4": r["energy_l2"]["bands_4"],
            }

    doc = {
        "model_id": catalog.get("model_id", MODEL_ID_4B),
        "n_neurons": len(all_rows),
        "n_active_poles": len(per_pole_out),
        "n_non_bos_tokens": n_non_bos,
        "bucket_precedence": list(FAILURE_BUCKETS),
        "correlations": correlations,
        "named_neurons": named_summary,
        "failure_buckets_top100_ge_0_80": _summarize_bucket_subset(rows_top100_ge_80),
        "failure_buckets_p99_9_plus_ge_0_80": _summarize_bucket_subset(rows_p999_ge_80),
        "failure_buckets_all_278": _summarize_bucket_subset(all_rows),
        "per_neuron": per_neuron_out,
        "per_pole": per_pole_out,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = results_dir / "gap_decomposition_4b.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[gap_decomposition_4b] COMPLETE in {doc['elapsed_sec']:.1f}s | "
        f"energy_ge_20 vs strong20_rec: rho={correlations['energy_ge_20pct_max_vs_strong20_rec']['spearman_rho']:.3f}, "
        f"r={correlations['energy_ge_20pct_max_vs_strong20_rec']['pearson_r']:.3f} | "
        f"top100>=0.80 n={len(rows_top100_ge_80)}"
    )
    return doc


# ---------------------------------------------------------------------------
# Multi-Band Snippets + Ordinary-Text Precision (FPR) Probes
# ---------------------------------------------------------------------------


def _sample_tiered_band_positives(
    base_mask: np.ndarray,
    other_max_dir: np.ndarray,
    p90: float,
    p97: float,
    doc_ids: np.ndarray,
    chunk_ids: np.ndarray,
    pos_in_chunk: np.ndarray,
    token_ids: np.ndarray,
    n_want: int,
    rng: np.random.Generator,
    min_spacing: int,
    excluded_docs: set[int],
    forbidden_suffixes: set[tuple[int, ...]],
    dedup_len: int,
    use_clean_tiers: bool,
) -> tuple[list[int], list[int]]:
    """Sample up to `n_want` positives preferring clean non-target windows first.

    Returns `(picked_flat_indices, picked_tiers)` where tier is:
      - 0: clean `other_max_dir < p90` (or base band for `p99_99.9` where `use_clean_tiers=False`)
      - 1: fallback `other_max_dir < p97`
      - 2: unfiltered fallback
    """
    if not use_clean_tiers:
        cands = np.flatnonzero(base_mask)
        if cands.size == 0:
            return [], []
        picked = _sample_distinct_doc_indices(
            candidate_indices=cands,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            n_want=n_want,
            rng=rng,
            min_spacing=min_spacing,
            excluded_docs=excluded_docs,
            token_ids=token_ids,
            forbidden_suffixes=forbidden_suffixes,
            dedup_within_sample=True,
            suffix_len=dedup_len,
            allow_same_doc_fallback=False,
        )
        if len(picked) < n_want:
            used_d = {int(doc_ids[i]) for i in picked}
            used_s = forbidden_suffixes | {_suffix_key(token_ids, i, dedup_len) for i in picked}
            extra = _sample_distinct_doc_indices(
                candidate_indices=cands,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=n_want - len(picked),
                rng=rng,
                min_spacing=min_spacing,
                excluded_docs=used_d,
                token_ids=token_ids,
                forbidden_suffixes=used_s,
                dedup_within_sample=True,
                suffix_len=dedup_len,
                allow_same_doc_fallback=True,
            )
            picked.extend(extra)
        return picked[:n_want], [0] * min(len(picked), n_want)

    tier_masks = [
        (0, base_mask & (other_max_dir < p90)),
        (1, base_mask & (other_max_dir < p97)),
        (2, base_mask),
    ]
    picked: list[int] = []
    tiers: list[int] = []
    used_docs = set(excluded_docs)
    used_sufs = set(forbidden_suffixes)

    for tier_id, tmask in tier_masks:
        if len(picked) >= n_want:
            break
        cands = np.flatnonzero(tmask)
        if cands.size == 0:
            continue
        take = _sample_distinct_doc_indices(
            candidate_indices=cands,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            n_want=n_want - len(picked),
            rng=rng,
            min_spacing=min_spacing,
            excluded_docs=used_docs,
            token_ids=token_ids,
            forbidden_suffixes=used_sufs,
            dedup_within_sample=True,
            suffix_len=dedup_len,
            allow_same_doc_fallback=False,
        )
        for idx in take:
            picked.append(idx)
            tiers.append(tier_id)
            used_docs.add(int(doc_ids[idx]))
            used_sufs.add(_suffix_key(token_ids, idx, dedup_len))

    # Last-resort pass allowing docs used in other bands (still strictly distinct within this band)
    if len(picked) < n_want:
        in_band_docs = {int(doc_ids[i]) for i in picked}
        for tier_id, tmask in tier_masks:
            if len(picked) >= n_want:
                break
            cands = np.flatnonzero(tmask)
            if cands.size == 0:
                continue
            take = _sample_distinct_doc_indices(
                candidate_indices=cands,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=n_want - len(picked),
                rng=rng,
                min_spacing=min_spacing,
                excluded_docs=in_band_docs,
                token_ids=token_ids,
                forbidden_suffixes=used_sufs,
                dedup_within_sample=True,
                suffix_len=dedup_len,
                allow_same_doc_fallback=False,
            )
            for idx in take:
                picked.append(idx)
                tiers.append(tier_id)
                in_band_docs.add(int(doc_ids[idx]))
                used_sufs.add(_suffix_key(token_ids, idx, dedup_len))

    # Keep strictly even count for exact k/k A/B balance
    even_len = 2 * (len(picked) // 2)
    return picked[:even_len], tiers[:even_len]


def build_multiband_and_fpr_snippets_4b(
    tokenizer: Any,
    base_dir: Path | None = None,
    seed: int = SEED_4B,
    pairs_per_band: int = MULTIBAND_PAIRS_PER_POLE,
    fpr_ord_core: int = FPR_ORD_CORE_PER_POLE,
    fpr_ord_extra: int = FPR_ORD_EXTRA_PER_POLE,
    fpr_top_core: int = FPR_TOP_CORE_PER_POLE,
    fpr_top_extra: int = FPR_TOP_EXTRA_PER_POLE,
    min_allowed_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Build `snippets_4b_multiband.jsonl` and `snippets_4b_fpr.jsonl` + full invariant checks."""
    t0 = time.time()
    dirs = ensure_4b_dirs(base_dir)
    cache_dir = dirs["cache"]
    data_dir = dirs["data"]
    results_dir = dirs["results"]

    catalog = json.loads((data_dir / "active_poles.json").read_text(encoding="utf-8"))
    cache_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))

    if window_spec is not None:
        wspec = window_spec
    else:
        wcfg = catalog.get("window", cache_meta.get("window", {}))
        wspec = WindowSpec(
            before=int(wcfg.get("before", 60)),
            after=int(wcfg.get("after", 8)),
            marker_open=str(wcfg.get("marker_open", "<<")),
            marker_close=str(wcfg.get("marker_close", ">>")),
        )

    token_ids = np.load(cache_dir / "token_ids.npy", mmap_mode="r")
    doc_ids = np.load(cache_dir / "doc_ids.npy", mmap_mode="r")
    chunk_ids = np.load(cache_dir / "chunk_ids.npy", mmap_mode="r")
    pos_in_chunk = np.load(cache_dir / "pos_in_chunk.npy", mmap_mode="r")
    chunk_lengths = np.load(cache_dir / "chunk_lengths.npy", mmap_mode="r")
    acts = np.load(cache_dir / "acts.npy", mmap_mode="r")
    quads = np.load(cache_dir / "quads.npy", mmap_mode="r")
    token_counts = np.load(cache_dir / "token_counts.npy")

    col_to_neuron = [tuple(x) for x in cache_meta["col_to_neuron"]]
    neuron_to_col = {(int(L), int(n)): col for col, (L, n) in enumerate(col_to_neuron)}
    special_ids = set(int(x) for x in cache_meta.get("special_token_ids", []))

    chunk_len_at_pos = chunk_lengths[chunk_ids]
    non_bos = pos_in_chunk > 0
    if special_ids:
        non_bos = non_bos & (~np.isin(token_ids, list(special_ids)))
    interior_mask = (
        non_bos
        & (pos_in_chunk >= wspec.before)
        & (pos_in_chunk <= (chunk_len_at_pos - wspec.after - 1))
    )
    dedup_len = min(DEDUP_SUFFIX_LEN, wspec.before + 1)

    # Load top-100 positive snippets (read-only!) so FPR positive probes come straight from top100
    top100_pos_by_pole: dict[str, list[dict[str, Any]]] = defaultdict(list)
    top100_path = data_dir / "snippets_4b_top100.jsonl"
    if top100_path.is_file():
        with top100_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                top100_pos_by_pole[str(row["pole_id"])].append(row["pos"])

    def _build_snippet_obj(
        flat_idx: int,
        act_val: float,
        act_dir: float,
        quad_code: int,
        window_max_act_dir: float,
        other_max_act_dir: float,
        p90_dir: float,
        p97_dir: float,
        include_suffix_hash: bool = False,
    ) -> dict[str, Any]:
        marked_text, target_tok_str, extra_markers = render_marked_window(
            token_ids, flat_idx, tokenizer, wspec
        )
        tid = int(token_ids[flat_idx])
        obj: dict[str, Any] = {
            "doc_id": int(doc_ids[flat_idx]),
            "chunk_id": int(chunk_ids[flat_idx]),
            "pos_in_chunk": int(pos_in_chunk[flat_idx]),
            "window_tokens": int(wspec.total),
            "target_offset": int(wspec.target_offset),
            "target_token_id": tid,
            "target_token_str": target_tok_str,
            "target_corpus_count": int(token_counts[tid]),
            "act_val": float(act_val),
            "act_dir": float(act_dir),
            "gate_open": bool(quad_code < 2),
            "quadrant": QUAD_CODE_TO_NAME[int(quad_code)],
            "window_max_act_dir": float(window_max_act_dir),
            "other_max_act_dir": float(other_max_act_dir),
            "window_has_ge_p97": bool(window_max_act_dir >= p97_dir),
            "other_has_ge_p90": bool(other_max_act_dir >= p90_dir),
            "extra_marker_count": int(extra_markers),
            "marked_text": marked_text,
        }
        if include_suffix_hash:
            sk = _suffix_key(token_ids, flat_idx, dedup_len)
            obj["suffix_16_hash"] = hashlib.sha256(
                ",".join(str(x) for x in sk).encode("ascii")
            ).hexdigest()[:16]
        return obj

    multiband_rows: list[dict[str, Any]] = []
    fpr_rows: list[dict[str, Any]] = []
    omitted_pole_bands: list[dict[str, Any]] = []
    window_tier_counts: dict[str, Counter[int]] = {b: Counter() for b in MULTIBAND_NAMES}

    n_top_want = fpr_top_core + fpr_top_extra
    n_ord_want = fpr_ord_core + fpr_ord_extra

    for pole in catalog["poles"]:
        pole_id = str(pole["pole_id"])
        neuron_id = str(pole["neuron_id"])
        k = int(pole["k"])
        layer = int(pole["layer"])
        neuron = int(pole["neuron"])
        sign = int(pole["sign"])
        in_full_expl_50 = bool(pole.get("in_full_expl_50", False))

        p25_abs = float(pole["p25_abs"])
        edges = pole["band_edges_dir"]
        p90 = float(edges["p90"])
        p97 = float(edges["p97"])
        p99 = float(edges["p99"])
        p99_9 = float(edges["p99.9"])

        col = neuron_to_col[(layer, neuron)]
        a_n = acts[:, col].astype(np.float32)
        q_n = quads[:, col]
        d_n = float(sign) * a_n
        abs_n = np.abs(a_n)
        d_nb = d_n[non_bos]
        p75 = float(np.quantile(d_nb, 0.75))
        max_act_dir = float(np.max(d_nb))

        other_max_dir, win_max_dir = compute_other_and_window_max_1d(d_n, wspec)
        pos_guard = interior_mask & (d_n > 0.0) & (abs_n >= p25_abs)
        dz_tier0_mask = interior_mask & (abs_n < p25_abs) & (win_max_dir < p90)

        # ---- Part 1: Three new bands (p99_99.9, p75_97, p25_75) ----
        band_specs = [
            ("p99_99.9", pos_guard & (d_n >= p99) & (d_n < p99_9), False),
            ("p75_97", pos_guard & (d_n >= p75) & (d_n < p97), True),
            ("p25_75", interior_mask & (d_n >= p25_abs) & (d_n < p75), True),
        ]
        used_pos_docs_pole: set[int] = set()
        used_pos_suffixes_pole: set[tuple[int, ...]] = set()
        dz_used_docs_pole: set[int] = set()

        for bname, bmask, use_clean in band_specs:
            rng_pos = _stable_rng_4b(seed, pole_id, "multiband_pos", bname)
            picked, tiers = _sample_tiered_band_positives(
                base_mask=bmask,
                other_max_dir=other_max_dir,
                p90=p90,
                p97=p97,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                token_ids=token_ids,
                n_want=pairs_per_band,
                rng=rng_pos,
                min_spacing=wspec.total,
                excluded_docs=used_pos_docs_pole,
                forbidden_suffixes=used_pos_suffixes_pole,
                dedup_len=dedup_len,
                use_clean_tiers=use_clean,
            )
            if len(picked) < 2:
                omitted_pole_bands.append(
                    {
                        "pole_id": pole_id,
                        "band": bname,
                        "eligible_tokens": int(np.sum(bmask)),
                        "p25_abs": p25_abs,
                        "p75": p75,
                        "p97": p97,
                        "p99": p99,
                        "p99.9": p99_9,
                    }
                )
                continue

            # Shuffle positive order deterministically (keeping paired tier aligned)
            rng_shuf = _stable_rng_4b(seed, pole_id, "multiband_shuf", bname)
            perm = rng_shuf.permutation(len(picked)).tolist()
            picked = [picked[i] for i in perm]
            tiers = [tiers[i] for i in perm]

            for idx in picked:
                used_pos_docs_pole.add(int(doc_ids[idx]))
                used_pos_suffixes_pole.add(_suffix_key(token_ids, idx, dedup_len))

            rng_dz = _stable_rng_4b(seed, pole_id, "multiband_dz", bname)
            paired = _select_strict_tier0_pairs(
                tier0_mask=dz_tier0_mask,
                pos_indices=picked,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                rng=rng_dz,
                used_docs_across_bands=dz_used_docs_pole,
                min_spacing=wspec.total,
                allow_same_doc_fallback=False,
            )
            if len(paired) < 2:
                omitted_pole_bands.append(
                    {
                        "pole_id": pole_id,
                        "band": bname,
                        "eligible_tokens": int(np.sum(bmask)),
                        "reason": "insufficient_dead_zone_pairs",
                    }
                )
                continue

            slots = balanced_even_slots(len(paired), seed=seed, pole_id=pole_id, band=bname)
            for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired, slots)):
                p_idx = picked[pos_k]
                p_tier = tiers[pos_k]
                window_tier_counts[bname][p_tier] += 1
                multiband_rows.append(
                    {
                        "pair_id": f"{pole_id}__{bname}__dead_zone__{pair_idx}",
                        "pole_id": pole_id,
                        "neuron_id": neuron_id,
                        "k": k,
                        "layer": layer,
                        "neuron": neuron,
                        "sign": sign,
                        "band": bname,
                        "distractor_type": "dead_zone",
                        "distractor_tier": 0,
                        "pos_window_tier": p_tier,
                        "pair_idx": pair_idx,
                        "pos_slot": slot,
                        "in_full_expl_50": in_full_expl_50,
                        "pos": _build_snippet_obj(
                            flat_idx=p_idx,
                            act_val=float(a_n[p_idx]),
                            act_dir=float(d_n[p_idx]),
                            quad_code=int(q_n[p_idx]),
                            window_max_act_dir=float(win_max_dir[p_idx]),
                            other_max_act_dir=float(other_max_dir[p_idx]),
                            p90_dir=p90,
                            p97_dir=p97,
                            include_suffix_hash=True,
                        ),
                        "dist": _build_snippet_obj(
                            flat_idx=d_idx,
                            act_val=float(a_n[d_idx]),
                            act_dir=float(d_n[d_idx]),
                            quad_code=int(q_n[d_idx]),
                            window_max_act_dir=float(win_max_dir[d_idx]),
                            other_max_act_dir=float(other_max_dir[d_idx]),
                            p90_dir=p90,
                            p97_dir=p97,
                            include_suffix_hash=False,
                        ),
                    }
                )

        # ---- Part 2: Ordinary-Text YES/NO Precision Probes (snippets_4b_fpr.jsonl) ----
        # 2a. Top-100 positive probes (2 core + 2 luna_extra = 4 distinct docs)
        top_pos_objs = top100_pos_by_pole.get(pole_id, [])[:n_top_want]
        if len(top_pos_objs) < n_top_want:
            # Fallback for synthetic tests where snippets_4b_top100.jsonl may not be pre-built
            cands_top = np.flatnonzero(pos_guard & (d_n >= p99_9))
            rng_ft = _stable_rng_4b(seed, pole_id, "fpr_top_fallback")
            fb_idxs = _sample_distinct_doc_indices(
                candidate_indices=cands_top,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=n_top_want,
                rng=rng_ft,
                min_spacing=wspec.total,
                token_ids=token_ids,
                dedup_within_sample=True,
                suffix_len=dedup_len,
            )
            top_pos_objs = [
                _build_snippet_obj(
                    flat_idx=idx,
                    act_val=float(a_n[idx]),
                    act_dir=float(d_n[idx]),
                    quad_code=int(q_n[idx]),
                    window_max_act_dir=float(win_max_dir[idx]),
                    other_max_act_dir=float(other_max_dir[idx]),
                    p90_dir=p90,
                    p97_dir=p97,
                )
                for idx in fb_idxs
            ]
        if len(top_pos_objs) < n_top_want:
            raise RuntimeError(f"Pole {pole_id}: only found {len(top_pos_objs)}/{n_top_want} top100 FPR probes")

        top_probe_docs = {int(o["doc_id"]) for o in top_pos_objs}
        for i_top, o in enumerate(top_pos_objs):
            tier_tag = "core" if i_top < fpr_top_core else "luna_extra"
            fpr_rows.append(
                {
                    "probe_id": f"{pole_id}__top100_firing__{i_top}",
                    "pole_id": pole_id,
                    "neuron_id": neuron_id,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "probe_type": "top100_firing",
                    "ground_truth": "YES",
                    "tier": tier_tag,
                    "probe_idx": i_top,
                    "doc_id": int(o["doc_id"]),
                    "chunk_id": int(o["chunk_id"]),
                    "pos_in_chunk": int(o["pos_in_chunk"]),
                    "window_tokens": int(o["window_tokens"]),
                    "target_offset": int(o["target_offset"]),
                    "target_token_id": int(o["target_token_id"]),
                    "target_token_str": str(o["target_token_str"]),
                    "act_val": float(o["act_val"]),
                    "act_dir": float(o["act_dir"]),
                    "frac_of_max": float(o["act_dir"]) / max_act_dir if max_act_dir > 0.0 else 0.0,
                    "quadrant": str(o["quadrant"]),
                    "marked_text": str(o["marked_text"]),
                }
            )

        # 2b. Random ordinary-text non-activating probes (10 core + 10 luna_extra = 20 distinct docs)
        # Strict criteria: |a_target| < 0.20 * max_act_dir AND d_target < p90
        ord_mask = interior_mask & (abs_n < (0.20 * max_act_dir)) & (d_n < p90)
        ord_cands = np.flatnonzero(ord_mask)
        rng_ord = _stable_rng_4b(seed, pole_id, "fpr_ordinary_nonfiring")
        # Fast uniform corpus-position sampling across distinct docs
        ord_picked: list[int] = []
        ord_used_docs = set(top_probe_docs)
        if ord_cands.size > 0:
            probes = rng_ord.integers(0, ord_cands.size, size=min(4096, max(256, n_ord_want * 64)))
            for p_i in probes.tolist():
                cand = int(ord_cands[p_i])
                did = int(doc_ids[cand])
                if did not in ord_used_docs:
                    ord_used_docs.add(did)
                    ord_picked.append(cand)
                    if len(ord_picked) >= n_ord_want:
                        break
        if len(ord_picked) < n_ord_want:
            extra_ord = _sample_distinct_doc_indices(
                candidate_indices=ord_cands,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=n_ord_want - len(ord_picked),
                rng=rng_ord,
                min_spacing=wspec.total,
                excluded_docs=ord_used_docs,
                allow_same_doc_fallback=False,
            )
            ord_picked.extend(extra_ord)
        if len(ord_picked) < n_ord_want:
            raise RuntimeError(
                f"Pole {pole_id}: only found {len(ord_picked)}/{n_ord_want} ordinary non-firing FPR probes "
                f"(ord_cands={ord_cands.size})"
            )

        for i_ord, f_idx in enumerate(ord_picked):
            tier_tag = "core" if i_ord < fpr_ord_core else "luna_extra"
            marked_text, target_tok_str, _ = render_marked_window(
                token_ids, f_idx, tokenizer, wspec
            )
            tid = int(token_ids[f_idx])
            q_code = int(q_n[f_idx])
            fpr_rows.append(
                {
                    "probe_id": f"{pole_id}__ordinary_nonfiring__{i_ord}",
                    "pole_id": pole_id,
                    "neuron_id": neuron_id,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "probe_type": "ordinary_nonfiring",
                    "ground_truth": "NO",
                    "tier": tier_tag,
                    "probe_idx": i_ord,
                    "doc_id": int(doc_ids[f_idx]),
                    "chunk_id": int(chunk_ids[f_idx]),
                    "pos_in_chunk": int(pos_in_chunk[f_idx]),
                    "window_tokens": int(wspec.total),
                    "target_offset": int(wspec.target_offset),
                    "target_token_id": tid,
                    "target_token_str": target_tok_str,
                    "act_val": float(a_n[f_idx]),
                    "act_dir": float(d_n[f_idx]),
                    "frac_of_max": float(d_n[f_idx]) / max_act_dir if max_act_dir > 0.0 else 0.0,
                    "quadrant": QUAD_CODE_TO_NAME[q_code],
                    "marked_text": marked_text,
                }
            )

    # ---- Verify all hard invariants across BOTH files ----
    poles_by_id = {str(p["pole_id"]): p for p in catalog["poles"]}
    mb_bos = mb_same_doc = mb_overlap = mb_dz_p25 = mb_dz_p90 = mb_win_err = 0
    mb_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in multiband_rows:
        pid = str(row["pole_id"])
        pole = poles_by_id[pid]
        p25 = float(pole["p25_abs"])
        p90 = float(pole["band_edges_dir"]["p90"])
        pos, dist = row["pos"], row["dist"]
        mb_by_cell[(pid, str(row["band"]))].append(row)
        if int(pos["window_tokens"]) != wspec.total or int(dist["window_tokens"]) != wspec.total:
            mb_win_err += 1
        if int(pos["target_offset"]) != wspec.target_offset or int(dist["target_offset"]) != wspec.target_offset:
            mb_win_err += 1
        if int(pos["pos_in_chunk"]) <= 0 or int(dist["pos_in_chunk"]) <= 0:
            mb_bos += 1
        if int(pos["doc_id"]) == int(dist["doc_id"]):
            mb_same_doc += 1
        if min_allowed_doc_id is not None and (
            int(pos["doc_id"]) < min_allowed_doc_id or int(dist["doc_id"]) < min_allowed_doc_id
        ):
            mb_overlap += 1
        if abs(float(dist["act_val"])) >= p25:
            mb_dz_p25 += 1
        if float(dist["window_max_act_dir"]) >= p90:
            mb_dz_p90 += 1

    mb_imbalanced = 0
    mb_by_band_summary: dict[str, Any] = {}
    for bname in MULTIBAND_NAMES:
        b_cells = [rows for (pid, b), rows in mb_by_cell.items() if b == bname]
        b_pairs = sum(len(r) for r in b_cells)
        full_6 = sum(1 for r in b_cells if len(r) == pairs_per_band)
        for r in b_cells:
            if sum(1 for x in r if x["pos_slot"] == "A") * 2 != len(r):
                mb_imbalanced += 1
        mb_by_band_summary[bname] = {
            "pairs": b_pairs,
            "surviving_poles": len(b_cells),
            "full_6_poles": full_6,
            "omitted_poles": len(poles_by_id) - len(b_cells),
            "pos_window_tier_counts": {str(k): int(v) for k, v in sorted(window_tier_counts[bname].items())},
        }

    fpr_bos = fpr_overlap = fpr_win_err = fpr_dup_doc = 0
    fpr_docs_by_pole: dict[str, list[int]] = defaultdict(list)
    fpr_tier_counts = Counter()
    for row in fpr_rows:
        fpr_tier_counts[f"{row['tier']}:{row['probe_type']}"] += 1
        fpr_docs_by_pole[str(row["pole_id"])].append(int(row["doc_id"]))
        if int(row["window_tokens"]) != wspec.total or int(row["target_offset"]) != wspec.target_offset:
            fpr_win_err += 1
        if int(row["pos_in_chunk"]) <= 0:
            fpr_bos += 1
        if min_allowed_doc_id is not None and int(row["doc_id"]) < min_allowed_doc_id:
            fpr_overlap += 1
    for pid, dlist in fpr_docs_by_pole.items():
        fpr_dup_doc += len(dlist) - len(set(dlist))

    all_passed = bool(
        len(multiband_rows) > 0
        and mb_bos == 0
        and mb_same_doc == 0
        and mb_overlap == 0
        and mb_dz_p25 == 0
        and mb_dz_p90 == 0
        and mb_win_err == 0
        and mb_imbalanced == 0
        and len(fpr_rows) == len(poles_by_id) * (n_ord_want + n_top_want)
        and fpr_bos == 0
        and fpr_overlap == 0
        and fpr_win_err == 0
        and fpr_dup_doc == 0
    )

    checks = {
        "all_hard_invariants_passed": all_passed,
        "multiband": {
            "total_pairs": len(multiband_rows),
            "by_band": mb_by_band_summary,
            "bos_pos0_leak_count": mb_bos,
            "same_doc_leak_count": mb_same_doc,
            "ulto85_doc_overlap_count": mb_overlap,
            "dist_abs_ge_p25_count": mb_dz_p25,
            "dist_window_ge_p90_count": mb_dz_p90,
            "window_or_offset_errors": mb_win_err,
            "imbalanced_cells": mb_imbalanced,
            "omitted_pole_bands_count": len(omitted_pole_bands),
            "omitted_pole_bands": omitted_pole_bands,
        },
        "fpr": {
            "total_probes": len(fpr_rows),
            "core_probes": sum(1 for r in fpr_rows if r["tier"] == "core"),
            "luna_extra_probes": sum(1 for r in fpr_rows if r["tier"] == "luna_extra"),
            "counts_by_tier_type": dict(sorted(fpr_tier_counts.items())),
            "bos_pos0_leak_count": fpr_bos,
            "ulto85_doc_overlap_count": fpr_overlap,
            "window_or_offset_errors": fpr_win_err,
            "within_pole_dup_doc_count": fpr_dup_doc,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    mb_path = data_dir / "snippets_4b_multiband.jsonl"
    fpr_path = data_dir / "snippets_4b_fpr.jsonl"
    checks_path = results_dir / "structural_checks_4b_multiband.json"

    with mb_path.open("w", encoding="utf-8") as f:
        for row in multiband_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with fpr_path.open("w", encoding="utf-8") as f:
        for row in fpr_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    if not all_passed:
        raise AssertionError(f"Multiband/FPR invariant check FAILED: {json.dumps(checks, indent=2)}")

    print(
        f"[multiband_and_fpr] COMPLETE in {checks['elapsed_sec']:.1f}s | "
        f"multiband={len(multiband_rows):,} pairs ({ {b: mb_by_band_summary[b]['pairs'] for b in MULTIBAND_NAMES} }) | "
        f"fpr={len(fpr_rows):,} probes (core={checks['fpr']['core_probes']:,}, luna_extra={checks['fpr']['luna_extra_probes']:,}) | "
        f"all_hard_invariants_passed={all_passed}"
    )
    return checks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run activation-energy gap decomposition and multiband+FPR snippet construction."
    )
    parser.add_argument("--model-id", default=MODEL_ID_4B)
    parser.add_argument("--seed", type=int, default=SEED_4B)
    parser.add_argument("--skip-gap", action="store_true")
    parser.add_argument("--skip-snippets", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_gap:
        compute_gap_decomposition_4b(seed=args.seed)
    if not args.skip_snippets:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        build_multiband_and_fpr_snippets_4b(tokenizer=tokenizer, seed=args.seed)


if __name__ == "__main__":
    main()
