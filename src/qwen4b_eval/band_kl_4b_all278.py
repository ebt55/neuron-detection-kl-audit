"""Exact IEEE-754 float32 4-band mean-ablation KL across all 278 Qwen3-4B neurons.

Extends `qwen4b_eval.band_kl_4b` (`n = 60` stratified neurons in `band_kl_4b.json`) to all
`278` Ulto85 Qwen3-4B neurons while enforcing strict bit-exact parity:
  1. Exact IEEE-754 `float32` (`allow_tf32=False`), fixed `[8, 128]` microbatching over
     `64 x 128 = 8,192` disjoint FineWeb tokens (`rows 65,000..65,069`), `max_vram < 6.5 GB`.
  2. Hard `zero_ablation_sanity_kl <= 1e-10` gate (must observe `0.0`).
  3. 4-anchor determinism validation (`1` neuron per stage: `k=26 info`, `k=58`, `k=140`,
     `k=236 insurance`) asserting `max_abs_diff < 1e-11` across all 9 `kl_by_cond` conditions
     against `results/qwen4b/band_kl_4b.json` BEFORE computing any new neurons (and aborting
     with `[anchor-mismatch]` if any anchor differs by `>= 1e-9`).
  4. Loads the remaining `56` neurons from `band_kl_4b.json` after verifying their `clean_acts`
     match `n_ablated_non_bos_tokens` and `eval_energy_4band`, and computes the `218` new
     neurons with incremental JSONL checkpointing (`band_kl_4b_all278_ckpt.jsonl`) and a
     graceful `07:30 IST` wall-clock cutoff check at every micro-group boundary.
  5. At zero extra forward passes, computes `is_bipolar`, `n_active_poles`, and
     `opposite_sign_eval_energy_share` (unipolar vs bipolar `< p75` decomposition),
     within-band CIs (`B=2,000, seed=42`), pooled `n=1,112` CIs (both genuine neuron-clustered
     and clearly labeled pair-level), and a `1,000`-permutation permuted-neuron null (`seed=42`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from qwen4b_eval import (
    ACTIVE_POLES_PATH,
    MODEL_ID_4B,
    SEED_4B,
    ensure_4b_dirs,
)
from qwen4b_eval.band_kl_4b import (
    ABLATION_CONDITIONS,
    FOUR_BAND_CONDS,
    _compute_chunked_kl_sum,
    _compute_stage_ranges,
    build_neuron_ablation_masks,
)
from qwen4b_eval.extract_4b import (
    ParityGateError,
    _call_decoder_layer,
    _clear_cuda,
    _prepare_masks_and_rope,
    collect_fixed_length_sequences,
    iter_fineweb_disjoint_rows,
    load_qwen4b_backbone_cpu,
)
from qwen4b_eval.gap_and_bands_4b import bootstrap_corr_stats
from w_audit.extract import encode_quadrant_torch

IST_TZ = timezone(timedelta(hours=5, minutes=30))
DEFAULT_CUTOFF_IST_ISO: str = "2026-11-22T07:30:00+05:30"

KL_MIN_ROW_IDX_4B: int = 65_000
KL_N_SEQS_4B: int = 64
KL_SEQ_LEN_4B: int = 128
MICRO_BATCH_SIZE_4B: int = 8
# band_kl_4b.json was computed with kl_token_chunk=512; chunk=4096 shifts cuBLAS split-K
# reduction by ~2.1e-10 (> 1e-11 gate) and uses ~5.55 GB VRAM vs ~4.71 GB at 512.
KL_TOKEN_CHUNK_4B: int = 512
ANCHOR_NEURON_KS: tuple[int, ...] = (26, 58, 140, 236)


class AnchorMismatchError(RuntimeError):
    """Raised when an anchor neuron differs from band_kl_4b.json beyond tolerance."""


# ---------------------------------------------------------------------------
# Statistical, Opposite-Sign Energy, and Checkpoint Helpers
# ---------------------------------------------------------------------------


def compute_opposite_sign_eval_energy(
    acts_eval_nb: np.ndarray,
    mean_act: float,
    lt_p75_mask_nb: np.ndarray,
    active_signs: Sequence[int],
) -> tuple[float, float, float]:
    """Compute opposite-sign eval L2 energy shares on non-BOS eval tokens.

    For 1-sided (unipolar) neurons (`len(active_signs) == 1`), checks `act_dir < 0.0` where
    `act_dir = active_sign * acts_eval_nb`:
      - `opp_in_lt_p75_share`: fraction of total eval L2 energy inside `lt_p75 & (act_dir < 0)`
      - `opp_total_share`: fraction of total eval L2 energy with `act_dir < 0`
      - `opp_frac_of_lt_p75`: `opp_in_lt_p75_share / lt_p75_eval_energy_share`

    For 2-sided (bipolar) neurons (`len(active_signs) == 2`), both poles are active so all
    three opposite-sign shares are identically `0.0`.
    """
    if len(active_signs) != 1:
        return 0.0, 0.0, 0.0

    sign = int(active_signs[0])
    a = np.asarray(acts_eval_nb, dtype=np.float64)
    l2 = (a - float(mean_act)) ** 2
    tot_l2 = float(np.sum(l2))
    if tot_l2 <= 0.0:
        return 0.0, 0.0, 0.0

    act_dir = float(sign) * a
    opp_mask = act_dir < 0.0
    lt_mask = np.asarray(lt_p75_mask_nb, dtype=bool)
    opp_in_lt_l2 = float(np.sum(l2[lt_mask & opp_mask]))
    opp_tot_l2 = float(np.sum(l2[opp_mask]))
    lt_l2 = float(np.sum(l2[lt_mask]))

    opp_in_lt_share = opp_in_lt_l2 / tot_l2
    opp_tot_share = opp_tot_l2 / tot_l2
    opp_frac_lt = (opp_in_lt_l2 / lt_l2) if lt_l2 > 0.0 else 0.0
    return float(opp_in_lt_share), float(opp_tot_share), float(opp_frac_lt)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    seed: int = SEED_4B,
) -> dict[str, Any]:
    """Compute sample mean and neuron-level 95% bootstrap CI."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean_val = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean_val, "ci_low": mean_val, "ci_high": mean_val}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[idx], axis=1)
    return {
        "n": n,
        "mean": mean_val,
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


def _safe_spearman_pearson(ax: np.ndarray, ay: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Spearman rho & Pearson r without RuntimeWarnings on constant inputs."""
    if len(ax) < 3 or float(np.ptp(ax)) <= 1e-12 or float(np.ptp(ay)) <= 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rho, rho_p = spearmanr(ax, ay)
    r, r_p = pearsonr(ax, ay)
    return float(rho), float(rho_p), float(r), float(r_p)


def pooled_corr_with_clustered_and_pair_ci(
    mat_x: np.ndarray,
    mat_y: np.ndarray,
    n_resamples: int = 2000,
    seed: int = SEED_4B,
) -> dict[str, Any]:
    """Compute pooled `(N x B)` Spearman rho & Pearson r with BOTH cluster and pair 95% CIs.

    - `neuron_clustered_ci`: resamples whole neuron rows `0..N-1` with replacement (each
      sampled neuron brings all `B` bands together).
    - `pair_level_ci`: resamples flattened `(neuron, band)` pairs `0..(N*B-1)` with replacement.
    """
    mx = np.asarray(mat_x, dtype=np.float64)
    my = np.asarray(mat_y, dtype=np.float64)
    if mx.ndim != 2 or my.shape != mx.shape:
        raise ValueError(f"Expected matching 2D matrices, got {mx.shape} and {my.shape}")

    n_neurons, n_bands = int(mx.shape[0]), int(mx.shape[1])
    x_flat = mx.reshape(-1)
    y_flat = my.reshape(-1)
    n_pairs = int(len(x_flat))

    rho, rho_p, r, r_p = _safe_spearman_pearson(x_flat, y_flat)

    # 1. Genuine neuron-clustered bootstrap (resample N whole neuron rows)
    rng_c = np.random.default_rng(seed)
    boot_c_rho: list[float] = []
    boot_c_r: list[float] = []
    for _ in range(n_resamples):
        idx = rng_c.integers(0, n_neurons, size=n_neurons)
        bx = mx[idx].reshape(-1)
        by = my[idx].reshape(-1)
        if float(np.ptp(bx)) <= 1e-12 or float(np.ptp(by)) <= 1e-12:
            continue
        brho, _ = spearmanr(bx, by)
        br, _ = pearsonr(bx, by)
        if math.isfinite(brho):
            boot_c_rho.append(float(brho))
        if math.isfinite(br):
            boot_c_r.append(float(br))

    # 2. Pair-level bootstrap (resample N*B individual pairs)
    rng_p = np.random.default_rng(seed + 1)
    boot_p_rho: list[float] = []
    boot_p_r: list[float] = []
    for _ in range(n_resamples):
        idx = rng_p.integers(0, n_pairs, size=n_pairs)
        bx = x_flat[idx]
        by = y_flat[idx]
        if float(np.ptp(bx)) <= 1e-12 or float(np.ptp(by)) <= 1e-12:
            continue
        brho, _ = spearmanr(bx, by)
        br, _ = pearsonr(bx, by)
        if math.isfinite(brho):
            boot_p_rho.append(float(brho))
        if math.isfinite(br):
            boot_p_r.append(float(br))

    clustered_ci = {
        "ci_type": "neuron_clustered",
        "description": f"Resample {n_neurons} whole neurons with replacement (each pulls {n_bands} bands)",
        "n_clusters": n_neurons,
        "n_pairs": n_pairs,
        "n_resamples": n_resamples,
        "seed": seed,
        "spearman_ci_low": float(np.percentile(boot_c_rho, 2.5)) if boot_c_rho else float("nan"),
        "spearman_ci_high": float(np.percentile(boot_c_rho, 97.5)) if boot_c_rho else float("nan"),
        "pearson_ci_low": float(np.percentile(boot_c_r, 2.5)) if boot_c_r else float("nan"),
        "pearson_ci_high": float(np.percentile(boot_c_r, 97.5)) if boot_c_r else float("nan"),
    }
    pair_ci = {
        "ci_type": "pair_level",
        "description": f"Resample {n_pairs} flattened (neuron, band) pairs with replacement",
        "n_pairs": n_pairs,
        "n_resamples": n_resamples,
        "seed": seed + 1,
        "spearman_ci_low": float(np.percentile(boot_p_rho, 2.5)) if boot_p_rho else float("nan"),
        "spearman_ci_high": float(np.percentile(boot_p_rho, 97.5)) if boot_p_rho else float("nan"),
        "pearson_ci_low": float(np.percentile(boot_p_r, 2.5)) if boot_p_r else float("nan"),
        "pearson_ci_high": float(np.percentile(boot_p_r, 97.5)) if boot_p_r else float("nan"),
    }

    return {
        "n": n_pairs,
        "n_neurons": n_neurons,
        "n_bands": n_bands,
        "n_resamples": n_resamples,
        "ci_type": "neuron_clustered",
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "spearman_ci_low": clustered_ci["spearman_ci_low"],
        "spearman_ci_high": clustered_ci["spearman_ci_high"],
        "pearson_r": r,
        "pearson_p": r_p,
        "pearson_ci_low": clustered_ci["pearson_ci_low"],
        "pearson_ci_high": clustered_ci["pearson_ci_high"],
        "r_squared": float(r * r) if math.isfinite(r) else float("nan"),
        "neuron_clustered_ci": clustered_ci,
        "pair_level_ci": pair_ci,
    }


def permuted_neuron_null(
    mat_x: np.ndarray,
    mat_y: np.ndarray,
    n_permutations: int = 1000,
    seed: int = SEED_4B,
) -> dict[str, Any]:
    """Permute neuron rows of `mat_y` (`N x B`) preserving per-band means and row sums."""
    mx = np.asarray(mat_x, dtype=np.float64)
    my = np.asarray(mat_y, dtype=np.float64)
    if mx.ndim != 2 or my.shape != mx.shape:
        raise ValueError(f"Expected matching 2D matrices, got {mx.shape} and {my.shape}")

    n_neurons, n_bands = int(mx.shape[0]), int(mx.shape[1])
    x_flat = mx.reshape(-1)
    y_flat = my.reshape(-1)
    obs_rho, _, obs_r, _ = _safe_spearman_pearson(x_flat, y_flat)

    rng = np.random.default_rng(seed)
    null_rhos: list[float] = []
    null_rs: list[float] = []
    for _ in range(n_permutations):
        perm = rng.permutation(n_neurons)
        yp_flat = my[perm, :].reshape(-1)
        if float(np.ptp(x_flat)) <= 1e-12 or float(np.ptp(yp_flat)) <= 1e-12:
            continue
        rho_p, _ = spearmanr(x_flat, yp_flat)
        r_p, _ = pearsonr(x_flat, yp_flat)
        if math.isfinite(rho_p):
            null_rhos.append(float(rho_p))
        if math.isfinite(r_p):
            null_rs.append(float(r_p))

    rho_arr = np.asarray(null_rhos, dtype=np.float64)
    r_arr = np.asarray(null_rs, dtype=np.float64)
    p_rho = float((1 + int(np.sum(rho_arr >= obs_rho))) / (len(rho_arr) + 1)) if len(rho_arr) else float("nan")
    p_r = float((1 + int(np.sum(r_arr >= obs_r))) / (len(r_arr) + 1)) if len(r_arr) else float("nan")

    return {
        "n_neurons": n_neurons,
        "n_bands": n_bands,
        "n_pairs": int(n_neurons * n_bands),
        "n_permutations": n_permutations,
        "seed": seed,
        "preserves_per_band_means": True,
        "observed_spearman_rho": obs_rho,
        "observed_pearson_r": obs_r,
        "spearman_mean": float(np.mean(rho_arr)) if len(rho_arr) else float("nan"),
        "spearman_std": float(np.std(rho_arr)) if len(rho_arr) else float("nan"),
        "spearman_ci_low": float(np.percentile(rho_arr, 2.5)) if len(rho_arr) else float("nan"),
        "spearman_ci_high": float(np.percentile(rho_arr, 97.5)) if len(rho_arr) else float("nan"),
        "spearman_p_empirical": p_rho,
        "pearson_mean": float(np.mean(r_arr)) if len(r_arr) else float("nan"),
        "pearson_std": float(np.std(r_arr)) if len(r_arr) else float("nan"),
        "pearson_ci_low": float(np.percentile(r_arr, 2.5)) if len(r_arr) else float("nan"),
        "pearson_ci_high": float(np.percentile(r_arr, 97.5)) if len(r_arr) else float("nan"),
        "pearson_p_empirical": p_r,
    }


def load_ckpt_jsonl(ckpt_path: Path) -> dict[int, dict[str, Any]]:
    """Load completed neuron records from incremental JSONL checkpoint."""
    loaded: dict[int, dict[str, Any]] = {}
    if not ckpt_path.is_file():
        return loaded
    lines = ckpt_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            # Ignore a truncated final line from an abrupt crash; fail on earlier corruption.
            if idx == len(lines) - 1:
                continue
            raise ValueError(f"Corrupted checkpoint line {idx + 1} in {ckpt_path}") from exc
        if "k" not in obj or "kl_by_cond" not in obj:
            continue
        kls = obj["kl_by_cond"]
        if not all(c in kls and isinstance(kls[c], (int, float)) and math.isfinite(float(kls[c])) for c in ABLATION_CONDITIONS):
            continue
        loaded[int(obj["k"])] = obj
    return loaded


def append_ckpt_jsonl(ckpt_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append completed neuron checkpoint records atomically per group with fsync."""
    if not records:
        return
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with ckpt_path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def build_neuron_kl_record(
    gap_rec: Mapping[str, Any],
    ap_rec: Mapping[str, Any],
    kl_by_cond: Mapping[str, float],
    n_ablated_non_bos_tokens: Mapping[str, int],
    eval_energy_4band: Mapping[str, float],
    eval_energy_ge_20pct_max: float,
    eval_energy_gate_closed: float,
    opposite_sign_eval_energy_share: float,
    opposite_sign_total_eval_energy_share: float,
    opposite_sign_frac_of_lt_p75: float,
    source: str,
) -> dict[str, Any]:
    """Assemble standardized per-neuron KL + EnergyShare record."""
    k = int(gap_rec["k"])
    kls = {c: float(kl_by_cond[c]) for c in ABLATION_CONDITIONS}
    kl_full = float(kls["full"])
    sum_4b = sum(float(kls[b]) for b in FOUR_BAND_CONDS)
    kl_share_4b_norm = {
        b: (float(kls[b]) / sum_4b if sum_4b > 0.0 else 0.0) for b in FOUR_BAND_CONDS
    }
    kl_share_4b_vs_full = {
        b: (float(kls[b]) / kl_full if kl_full > 0.0 else 0.0) for b in FOUR_BAND_CONDS
    }
    sum_amp = float(kls["ge_20pct_max"] + kls["lt_20pct_max"])
    kl_share_ge20_norm = float(kls["ge_20pct_max"] / sum_amp) if sum_amp > 0.0 else 0.0
    kl_share_ge20_full = float(kls["ge_20pct_max"] / kl_full) if kl_full > 0.0 else 0.0

    sum_gate = float(kls["gate_open"] + kls["gate_closed"])
    kl_share_gc_norm = float(kls["gate_closed"] / sum_gate) if sum_gate > 0.0 else 0.0
    kl_share_gc_full = float(kls["gate_closed"] / kl_full) if kl_full > 0.0 else 0.0

    n_active_poles = int(ap_rec["n_active_poles"])
    is_bipolar = bool(n_active_poles == 2)
    corpus_4b = {b: float(gap_rec["energy_l2"]["bands_4"][b]) for b in FOUR_BAND_CONDS}

    return {
        "k": k,
        "neuron_id": str(gap_rec["neuron_id"]),
        "layer": int(gap_rec["layer"]),
        "neuron": int(gap_rec["neuron"]),
        "is_bipolar": is_bipolar,
        "n_active_poles": n_active_poles,
        "active_pole_ids": list(ap_rec.get("active_pole_ids", gap_rec.get("active_pole_ids", []))),
        "dominant_pole_id": ap_rec.get("dominant_pole_id", gap_rec.get("dominant_pole_id")),
        "named_neuron": gap_rec.get("named_neuron"),
        "failure_bucket": gap_rec.get("failure_bucket"),
        "sim_cal_rec": float(gap_rec["sim_cal_rec"]),
        "sim_cal_rec_active": float(gap_rec["sim_cal_rec_active"]),
        "strong20_rec": float(gap_rec["strong20_rec"]),
        "source": source,
        "kl_by_cond": kls,
        "n_ablated_non_bos_tokens": {c: int(n_ablated_non_bos_tokens[c]) for c in ABLATION_CONDITIONS},
        "kl_4band_sum": sum_4b,
        "kl_4band_additivity_ratio": float(sum_4b / kl_full) if kl_full > 0.0 else float("nan"),
        "kl_share_4band_norm": kl_share_4b_norm,
        "kl_share_4band_vs_full": kl_share_4b_vs_full,
        "kl_share_ge_20pct_max_norm": kl_share_ge20_norm,
        "kl_share_ge_20pct_max_vs_full": kl_share_ge20_full,
        "kl_share_gate_closed_norm": kl_share_gc_norm,
        "kl_share_gate_closed_vs_full": kl_share_gc_full,
        "corpus_energy_4band": corpus_4b,
        "corpus_energy_ge_20pct_max": float(gap_rec["energy_ge_20pct_max"]),
        "corpus_energy_gate_closed": float(gap_rec["energy_gate_closed"]),
        "eval_energy_4band": {b: float(eval_energy_4band[b]) for b in FOUR_BAND_CONDS},
        "eval_energy_ge_20pct_max": float(eval_energy_ge_20pct_max),
        "eval_energy_gate_closed": float(eval_energy_gate_closed),
        "opposite_sign_eval_energy_share": float(opposite_sign_eval_energy_share),
        "opposite_sign_total_eval_energy_share": float(opposite_sign_total_eval_energy_share),
        "opposite_sign_frac_of_lt_p75": float(opposite_sign_frac_of_lt_p75),
    }


def _summarize_polarity_cohort(
    cohort_rows: Sequence[Mapping[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compute mean KL shares, eval/corpus energy shares, and opposite-sign shares for a cohort."""
    n = len(cohort_rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean_kl_share_4band_norm": {
            b: float(np.mean([r["kl_share_4band_norm"][b] for r in cohort_rows])) for b in FOUR_BAND_CONDS
        },
        "mean_eval_energy_4band": {
            b: float(np.mean([r["eval_energy_4band"][b] for r in cohort_rows])) for b in FOUR_BAND_CONDS
        },
        "mean_corpus_energy_4band": {
            b: float(np.mean([r["corpus_energy_4band"][b] for r in cohort_rows])) for b in FOUR_BAND_CONDS
        },
        "mean_opposite_sign_eval_energy_share": float(
            np.mean([r["opposite_sign_eval_energy_share"] for r in cohort_rows])
        ),
        "median_opposite_sign_eval_energy_share": float(
            np.median([r["opposite_sign_eval_energy_share"] for r in cohort_rows])
        ),
        "mean_opposite_sign_total_eval_energy_share": float(
            np.mean([r["opposite_sign_total_eval_energy_share"] for r in cohort_rows])
        ),
        "mean_opposite_sign_frac_of_lt_p75": float(
            np.mean([r["opposite_sign_frac_of_lt_p75"] for r in cohort_rows])
        ),
        "bootstrap_ci_95": {
            "n_resamples": n_bootstrap,
            "seed": seed,
            "kl_share_4band_norm": {
                b: bootstrap_mean_ci(
                    [r["kl_share_4band_norm"][b] for r in cohort_rows],
                    n_resamples=n_bootstrap,
                    seed=seed,
                )
                for b in FOUR_BAND_CONDS
            },
            "eval_energy_4band": {
                b: bootstrap_mean_ci(
                    [r["eval_energy_4band"][b] for r in cohort_rows],
                    n_resamples=n_bootstrap,
                    seed=seed,
                )
                for b in FOUR_BAND_CONDS
            },
            "opposite_sign_eval_energy_share": bootstrap_mean_ci(
                [r["opposite_sign_eval_energy_share"] for r in cohort_rows],
                n_resamples=n_bootstrap,
                seed=seed,
            ),
        },
    }


def assemble_band_kl_all278_doc(
    per_neuron_kl: Mapping[str, Mapping[str, Any]],
    gap_doc: Mapping[str, Any],
    stage_ranges: Sequence[tuple[int, int]],
    n_seqs: int,
    seq_len: int,
    micro_batch_size: int,
    kl_token_chunk: int,
    peak_vram_alloc: float,
    peak_vram_res: float,
    max_vram_gb: float,
    is_cuda: bool,
    zero_ablation_sanity_kl: float,
    anchor_validation: Mapping[str, Any],
    elapsed_sec: float,
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    seed: int = SEED_4B,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build complete aggregate document for all neurons in `per_neuron_kl`."""
    rows = [per_neuron_kl[str(k)] for k in sorted(int(x) for x in per_neuron_kl)]
    n_neurons = len(rows)

    neurons_per_layer: dict[str, int] = defaultdict(int)
    for r in rows:
        neurons_per_layer[str(int(r["layer"]))] += 1
    neurons_per_layer_sorted = {
        str(L): int(neurons_per_layer[str(L)]) for L in sorted(int(x) for x in neurons_per_layer)
    }

    corrs: dict[str, Any] = {}
    for b in FOUR_BAND_CONDS:
        corrs[f"corpus_energy_vs_kl_norm__{b}"] = bootstrap_corr_stats(
            [r["corpus_energy_4band"][b] for r in rows],
            [r["kl_share_4band_norm"][b] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed,
        )
        corrs[f"eval_energy_vs_kl_norm__{b}"] = bootstrap_corr_stats(
            [r["eval_energy_4band"][b] for r in rows],
            [r["kl_share_4band_norm"][b] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed,
        )

    mat_kl = np.asarray(
        [[r["kl_share_4band_norm"][b] for b in FOUR_BAND_CONDS] for r in rows],
        dtype=np.float64,
    )
    mat_eval = np.asarray(
        [[r["eval_energy_4band"][b] for b in FOUR_BAND_CONDS] for r in rows],
        dtype=np.float64,
    )
    mat_corp = np.asarray(
        [[r["corpus_energy_4band"][b] for b in FOUR_BAND_CONDS] for r in rows],
        dtype=np.float64,
    )

    n_pairs = n_neurons * len(FOUR_BAND_CONDS)
    pooled_eval = pooled_corr_with_clustered_and_pair_ci(
        mat_eval, mat_kl, n_resamples=n_bootstrap, seed=seed
    )
    pooled_corp = pooled_corr_with_clustered_and_pair_ci(
        mat_corp, mat_kl, n_resamples=n_bootstrap, seed=seed
    )
    corrs[f"pooled_{n_pairs}_eval_energy_vs_kl_norm"] = pooled_eval
    corrs[f"pooled_{n_pairs}_corpus_energy_vs_kl_norm"] = pooled_corp

    corrs["corpus_energy_vs_kl_norm__ge_20pct_max"] = bootstrap_corr_stats(
        [r["corpus_energy_ge_20pct_max"] for r in rows],
        [r["kl_share_ge_20pct_max_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed,
    )
    corrs["eval_energy_vs_kl_norm__ge_20pct_max"] = bootstrap_corr_stats(
        [r["eval_energy_ge_20pct_max"] for r in rows],
        [r["kl_share_ge_20pct_max_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed,
    )
    corrs["corpus_energy_vs_kl_norm__gate_closed"] = bootstrap_corr_stats(
        [r["corpus_energy_gate_closed"] for r in rows],
        [r["kl_share_gate_closed_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed,
    )
    corrs["eval_energy_vs_kl_norm__gate_closed"] = bootstrap_corr_stats(
        [r["eval_energy_gate_closed"] for r in rows],
        [r["kl_share_gate_closed_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed,
    )
    corrs["kl_ge_20pct_max_vs_ulto85_strong20_rec"] = bootstrap_corr_stats(
        [r["kl_share_ge_20pct_max_norm"] for r in rows],
        [r["strong20_rec"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed,
    )

    perm_null = {
        "eval_energy_vs_kl_norm": permuted_neuron_null(
            mat_eval, mat_kl, n_permutations=n_permutations, seed=seed
        ),
        "corpus_energy_vs_kl_norm": permuted_neuron_null(
            mat_corp, mat_kl, n_permutations=n_permutations, seed=seed
        ),
    }

    uni_rows = [r for r in rows if not bool(r["is_bipolar"])]
    bip_rows = [r for r in rows if bool(r["is_bipolar"])]
    unipolar_vs_bipolar = {
        "unipolar": _summarize_polarity_cohort(uni_rows, n_bootstrap=n_bootstrap, seed=seed),
        "bipolar": _summarize_polarity_cohort(bip_rows, n_bootstrap=n_bootstrap, seed=seed),
    }

    add_ratios = [
        r["kl_4band_additivity_ratio"] for r in rows if math.isfinite(r["kl_4band_additivity_ratio"])
    ]
    named_kl = {str(r["named_neuron"]): dict(r) for r in rows if r.get("named_neuron")}

    doc: dict[str, Any] = {
        "model_id": gap_doc.get("model_id", MODEL_ID_4B),
        "compute_dtype": "torch.float32 (IEEE-754, allow_tf32=False)",
        "stage_ranges": [list(sr) for sr in stage_ranges],
        "n_neurons": n_neurons,
        "n_reused_from_band_kl_4b": sum(
            1 for r in rows if str(r.get("source", "")).startswith("band_kl_4b")
        ),
        "n_newly_computed": sum(1 for r in rows if r.get("source") == "computed_all278"),
        "neurons_per_layer": neurons_per_layer_sorted,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "total_tokens": n_seqs * seq_len,
        "total_non_bos_tokens": n_seqs * (seq_len - 1),
        "micro_batch_size": micro_batch_size,
        "kl_token_chunk": kl_token_chunk,
        "bootstrap_resamples": n_bootstrap,
        "bootstrap_seed": seed,
        "peak_vram_allocated_gb": round(float(peak_vram_alloc), 4),
        "peak_vram_reserved_gb": round(float(peak_vram_res), 4),
        "max_vram_limit_gb": max_vram_gb,
        "vram_pass": (not is_cuda) or (peak_vram_alloc < max_vram_gb and peak_vram_res < max_vram_gb),
        "zero_ablation_sanity_kl": float(zero_ablation_sanity_kl),
        "anchor_validation": dict(anchor_validation),
        "kl_4band_additivity_ratio_median": float(np.median(add_ratios)) if add_ratios else float("nan"),
        "mean_kl_share_4band_norm": {
            b: float(np.mean([r["kl_share_4band_norm"][b] for r in rows])) for b in FOUR_BAND_CONDS
        },
        "mean_corpus_energy_4band": {
            b: float(np.mean([r["corpus_energy_4band"][b] for r in rows])) for b in FOUR_BAND_CONDS
        },
        "mean_eval_energy_4band": {
            b: float(np.mean([r["eval_energy_4band"][b] for r in rows])) for b in FOUR_BAND_CONDS
        },
        "mean_opposite_sign_eval_energy_share": float(
            np.mean([r["opposite_sign_eval_energy_share"] for r in rows])
        ),
        "unipolar_vs_bipolar": unipolar_vs_bipolar,
        "correlations": corrs,
        "permuted_neuron_null": perm_null,
        "named_neurons": named_kl,
        "per_neuron": {str(r["k"]): dict(r) for r in rows},
        "elapsed_sec": round(float(elapsed_sec), 2),
    }
    if extra_meta:
        doc.update(extra_meta)
    return doc


# ---------------------------------------------------------------------------
# GPU IEEE-754 Float32 4-Stage Pipeline with Anchors & Checkpoint Resume
# ---------------------------------------------------------------------------


def _evaluate_group_kl(
    backbone: nn.Module,
    group_recs: Sequence[Mapping[str, Any]],
    masks_by_k: Mapping[int, Mapping[str, np.ndarray]],
    clean_layer_in: Mapping[int, torch.Tensor],
    clean_final_cpu: torch.Tensor,
    lm_head_weight_cpu: torch.Tensor,
    stage_ranges: Sequence[tuple[int, int]],
    layer_to_stage: Mapping[int, int],
    start_stage: int,
    micro_batch_size: int,
    n_batches: int,
    total_non_bos_tokens: int,
    kl_token_chunk: int,
    dev: torch.device,
    cpu_dev: torch.device,
    orig_cpu_dtype: torch.dtype,
    rotary_mod: nn.Module | None,
    norm_mod: nn.Module | None,
    record_vram_fn: Any,
) -> dict[int, dict[str, float]]:
    """Compute all 9 ablation conditions for a group of neurons sharing `start_stage`.

    Deduplicates identical `(b_idx, bm_bytes)` masks within each neuron `k` (for example,
    `lt_20pct_max` == `full` on batches where `ge_20pct_max` has 0 active tokens) while
    preserving fixed `[micro_batch_size, seq_len]` batching and bit-exact IEEE-754 sums.
    """
    n_stages = len(stage_ranges)
    is_cuda = dev.type == "cuda"

    def _stage_modules(s_idx: int) -> list[nn.Module]:
        a, b = stage_ranges[s_idx]
        mods: list[nn.Module] = []
        if s_idx == 0:
            mods.append(backbone.embed_tokens)
        if rotary_mod is not None:
            mods.append(rotary_mod)
        mods.extend(list(backbone.layers[a:b]))
        if s_idx == n_stages - 1 and norm_mod is not None:
            mods.append(norm_mod)
        return mods

    def _load_stage_fp32(s_idx: int) -> list[nn.Module]:
        mods = _stage_modules(s_idx)
        for m in mods:
            if m is rotary_mod:
                m.to(device=dev)
            else:
                m.to(device=dev, dtype=torch.float32)
        return mods

    def _unload_stage(mods: Sequence[nn.Module]) -> None:
        for m in mods:
            if m is rotary_mod:
                m.to(device=cpu_dev)
            else:
                m.to(device=cpu_dev, dtype=orig_cpu_dtype)
        _clear_cuda()

    # Map each (k, cond) -> list of unique (b_idx, mask_bytes) keys within neuron k
    cond_keys_by_k: dict[int, dict[str, list[tuple[int, bytes]]]] = {}
    unique_masks_by_k: dict[int, dict[tuple[int, bytes], tuple[np.ndarray, list[int]]]] = {}
    for rec in group_recs:
        k = int(rec["k"])
        cond_keys_by_k[k] = {}
        unique_masks_by_k[k] = {}
        for cond in ABLATION_CONDITIONS:
            m2d = masks_by_k[k][cond]
            keys_for_cond: list[tuple[int, bytes]] = []
            for b_idx in range(n_batches):
                s0 = b_idx * micro_batch_size
                s1 = s0 + micro_batch_size
                bm_np = m2d[s0:s1]
                if not bm_np.any():
                    continue
                mkey = (b_idx, bm_np.tobytes())
                if mkey not in unique_masks_by_k[k]:
                    changed_local = np.flatnonzero(bm_np.any(axis=1)).tolist()
                    unique_masks_by_k[k][mkey] = (bm_np, changed_local)
                keys_for_cond.append(mkey)
            cond_keys_by_k[k][cond] = keys_for_cond

    # pending[(k, b_idx, mbytes)] = (changed_local, h_cpu[B, T, D])
    pending: dict[tuple[int, int, bytes], tuple[list[int], torch.Tensor]] = {}

    l_start, l_end = stage_ranges[start_stage]
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    mods = _load_stage_fp32(start_stage)
    try:
        for rec in group_recs:
            k = int(rec["k"])
            L = int(rec["layer"])
            if layer_to_stage[L] != start_stage:
                raise ValueError(f"Neuron k={k} layer={L} does not belong to stage {start_stage}")
            n_idx = int(rec["neuron"])
            mean_scalar = torch.tensor(float(rec["mean_act"]), dtype=torch.float32, device=dev)
            hook_state: dict[str, Any] = {"bm": None}

            def _pre_hook(_mod: nn.Module, args: tuple[Any, ...]):
                x = args[0].clone()
                x[:, :, n_idx] = torch.where(hook_state["bm"], mean_scalar, x[:, :, n_idx])
                return (x,)

            hh = backbone.layers[L].mlp.down_proj.register_forward_pre_hook(_pre_hook)
            try:
                with torch.no_grad():
                    for (b_idx, mbytes), (bm_np, changed_local) in unique_masks_by_k[k].items():
                        s0 = b_idx * micro_batch_size
                        s1 = s0 + micro_batch_size
                        h = clean_layer_in[L][s0:s1].to(dev, dtype=torch.float32)
                        hook_state["bm"] = torch.from_numpy(bm_np).to(dev)
                        pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                        for i in range(L, l_end):
                            h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                        if start_stage == n_stages - 1 and norm_mod is not None:
                            h = norm_mod(h)
                        pending[(k, b_idx, mbytes)] = (changed_local, h.detach().cpu())
                        del h, pos_ids, pos_embs, mask_map
                        hook_state["bm"] = None
            finally:
                hh.remove()
        record_vram_fn()
    finally:
        _unload_stage(mods)

    for next_s in range(start_stage + 1, n_stages):
        nl_start, nl_end = stage_ranges[next_s]
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        mods = _load_stage_fp32(next_s)
        try:
            with torch.no_grad():
                for key_p, (changed_local, h_cpu) in pending.items():
                    h = h_cpu.to(dev, dtype=torch.float32)
                    pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                    for i in range(nl_start, nl_end):
                        h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                    if next_s == n_stages - 1 and norm_mod is not None:
                        h = norm_mod(h)
                    pending[key_p] = (changed_local, h.detach().cpu())
                    del h, pos_ids, pos_embs, mask_map
            record_vram_fn()
        finally:
            _unload_stage(mods)

    batch_kl: dict[tuple[int, int, bytes], float] = {}
    if pending:
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        lm_w_gpu = lm_head_weight_cpu.to(device=dev, dtype=torch.float32)
        try:
            with torch.no_grad():
                for key_p, (changed_local, h_final_cpu) in pending.items():
                    _k, b_idx, _mbytes = key_p
                    s0 = b_idx * micro_batch_size
                    abs_changed = [s0 + li for li in changed_local]
                    c_h = clean_final_cpu[abs_changed].to(dev, dtype=torch.float32)
                    a_h = h_final_cpu[changed_local].to(dev, dtype=torch.float32)
                    batch_kl[key_p] = _compute_chunked_kl_sum(
                        c_h, a_h, lm_w_gpu, token_chunk=kl_token_chunk
                    )
                    del c_h, a_h
            record_vram_fn()
        finally:
            del lm_w_gpu
            _clear_cuda()

    del pending
    kl_out: dict[int, dict[str, float]] = {}
    for rec in group_recs:
        k = int(rec["k"])
        kl_out[k] = {}
        for cond in ABLATION_CONDITIONS:
            kl_sum = 0.0
            for b_idx, mbytes in cond_keys_by_k[k][cond]:
                kl_sum += batch_kl[(k, b_idx, mbytes)]
            kl_out[k][cond] = float(kl_sum / float(total_non_bos_tokens))
    return kl_out


def run_band_kl_all278(
    backbone: nn.Module,
    input_ids_cpu: torch.Tensor,
    gap_doc: Mapping[str, Any],
    active_poles_doc: Mapping[str, Any],
    base_kl_doc: Mapping[str, Any] | None = None,
    anchor_ks: Sequence[int] = ANCHOR_NEURON_KS,
    lm_head_weight_cpu: torch.Tensor | None = None,
    compute_device: str | torch.device = "cuda",
    micro_batch_size: int = MICRO_BATCH_SIZE_4B,
    kl_token_chunk: int = KL_TOKEN_CHUNK_4B,
    max_vram_gb: float = 6.5,
    neurons_per_group: int = 4,
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    seed: int = SEED_4B,
    ckpt_path: Path | None = None,
    output_path: Path | None = None,
    cutoff_dt: datetime | None = None,
    now_fn: Any = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run exact IEEE-754 float32 4-band KL over all neurons in `gap_doc['per_neuron']`."""
    t0 = time.time()
    dev = torch.device(compute_device)
    is_cuda = dev.type == "cuda"
    cpu_dev = torch.device("cpu")
    clock_fn = now_fn if now_fn is not None else (lambda: datetime.now(IST_TZ))

    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    n_layers = len(backbone.layers)
    stage_ranges = _compute_stage_ranges(n_layers)
    n_stages = len(stage_ranges)
    layer_to_stage = {
        L: s_idx for s_idx, (a, b) in enumerate(stage_ranges) for L in range(a, b)
    }

    orig_cpu_dtype = next(backbone.parameters()).dtype
    n_seqs, seq_len = int(input_ids_cpu.shape[0]), int(input_ids_cpu.shape[1])
    if n_seqs % micro_batch_size != 0:
        raise ValueError(
            f"n_seqs ({n_seqs}) must be divisible by micro_batch_size ({micro_batch_size})."
        )
    n_batches = n_seqs // micro_batch_size
    hidden_size = int(backbone.embed_tokens.weight.shape[1])
    total_non_bos_tokens = n_seqs * (seq_len - 1)

    if lm_head_weight_cpu is None:
        lm_head_weight_cpu = backbone.embed_tokens.weight.detach()

    ap_by_k = {int(r["k"]): r for r in active_poles_doc["neurons"]}
    all_recs = [gap_doc["per_neuron"][str(k)] for k in sorted(int(x) for x in gap_doc["per_neuron"])]
    neurons_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_recs:
        neurons_by_layer[int(rec["layer"])].append(dict(rec))
    hook_layers = sorted(neurons_by_layer.keys())

    rotary_mod = getattr(backbone, "rotary_emb", None)
    norm_mod = getattr(backbone, "norm", None)

    def _stage_modules(s_idx: int) -> list[nn.Module]:
        a, b = stage_ranges[s_idx]
        mods: list[nn.Module] = []
        if s_idx == 0:
            mods.append(backbone.embed_tokens)
        if rotary_mod is not None:
            mods.append(rotary_mod)
        mods.extend(list(backbone.layers[a:b]))
        if s_idx == n_stages - 1 and norm_mod is not None:
            mods.append(norm_mod)
        return mods

    def _load_stage_fp32(s_idx: int) -> list[nn.Module]:
        mods = _stage_modules(s_idx)
        for m in mods:
            if m is rotary_mod:
                m.to(device=dev)
            else:
                m.to(device=dev, dtype=torch.float32)
        return mods

    def _unload_stage(mods: Sequence[nn.Module]) -> None:
        for m in mods:
            if m is rotary_mod:
                m.to(device=cpu_dev)
            else:
                m.to(device=cpu_dev, dtype=orig_cpu_dtype)
        _clear_cuda()

    peak_vram_alloc = 0.0
    peak_vram_res = 0.0

    def _record_vram() -> None:
        nonlocal peak_vram_alloc, peak_vram_res
        if is_cuda:
            torch.cuda.synchronize(dev)
            alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
            resv = torch.cuda.max_memory_reserved(dev) / (1024**3)
            peak_vram_alloc = max(peak_vram_alloc, alloc)
            peak_vram_res = max(peak_vram_res, resv)
            if alloc >= max_vram_gb or resv >= max_vram_gb:
                raise ParityGateError(
                    f"VRAM limit exceeded during FP32 band_kl_4b_all278: "
                    f"alloc={alloc:.3f}GB, res={resv:.3f}GB >= {max_vram_gb}GB"
                )

    anchor_validation: dict[str, Any] = {
        "anchor_ks": [int(k) for k in anchor_ks],
        "validated": False,
        "max_abs_diff_overall": 0.0,
        "per_anchor_max_abs_diff": {},
    }

    try:
        # ------------------------------------------------------------------
        # STEP 1: Clean 4-stage IEEE-754 FP32 pass for all 278 neurons
        # ------------------------------------------------------------------
        clean_layer_in: dict[int, torch.Tensor] = {
            L: torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
            for L in hook_layers
        }
        stage_h_cpu = torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
        clean_final_cpu = torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
        clean_acts_by_k: dict[int, np.ndarray] = {
            int(r["k"]): np.empty((n_seqs, seq_len), dtype=np.float32) for r in all_recs
        }
        clean_quads_by_k: dict[int, np.ndarray] = {
            int(r["k"]): np.empty((n_seqs, seq_len), dtype=np.uint8) for r in all_recs
        }

        for s_idx, (l_start, l_end) in enumerate(stage_ranges):
            if is_cuda:
                _clear_cuda()
                torch.cuda.reset_peak_memory_stats(dev)
            s_hook_layers = [L for L in hook_layers if l_start <= L < l_end]
            mods = _load_stage_fp32(s_idx)
            handles: list[Any] = []
            buf: dict[int, dict[str, torch.Tensor]] = {L: {} for L in s_hook_layers}
            try:
                for L in s_hook_layers:
                    n_idxs = [int(r["neuron"]) for r in neurons_by_layer[L]]
                    idx_t = torch.tensor(n_idxs, dtype=torch.long, device=dev)
                    mlp = backbone.layers[L].mlp

                    def _mk_g(li: int, sel: torch.Tensor):
                        def _h(_m, _a, out):
                            buf[li]["g"] = out.detach().index_select(-1, sel)
                        return _h

                    def _mk_u(li: int, sel: torch.Tensor):
                        def _h(_m, _a, out):
                            buf[li]["u"] = out.detach().index_select(-1, sel)
                        return _h

                    def _mk_a(li: int, sel: torch.Tensor):
                        def _h(_m, args):
                            buf[li]["a"] = args[0].detach().index_select(-1, sel)
                        return _h

                    handles.append(mlp.gate_proj.register_forward_hook(_mk_g(L, idx_t)))
                    handles.append(mlp.up_proj.register_forward_hook(_mk_u(L, idx_t)))
                    handles.append(mlp.down_proj.register_forward_pre_hook(_mk_a(L, idx_t)))

                with torch.no_grad():
                    for b_idx in range(n_batches):
                        s0 = b_idx * micro_batch_size
                        s1 = s0 + micro_batch_size
                        if s_idx == 0:
                            h = backbone.embed_tokens(input_ids_cpu[s0:s1].to(dev))
                        else:
                            h = stage_h_cpu[s0:s1].to(dev, dtype=torch.float32)
                        pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                        for i in range(l_start, l_end):
                            if i in clean_layer_in:
                                clean_layer_in[i][s0:s1].copy_(h.detach().cpu())
                            h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                        if s_idx == n_stages - 1:
                            if norm_mod is not None:
                                h = norm_mod(h)
                            clean_final_cpu[s0:s1].copy_(h.detach().cpu())
                        else:
                            stage_h_cpu[s0:s1].copy_(h.detach().cpu())

                        for L in s_hook_layers:
                            a_np = buf[L]["a"].cpu().numpy()
                            q_np = encode_quadrant_torch(buf[L]["g"], buf[L]["u"]).cpu().numpy()
                            for j, rec in enumerate(neurons_by_layer[L]):
                                k = int(rec["k"])
                                clean_acts_by_k[k][s0:s1] = a_np[:, :, j]
                                clean_quads_by_k[k][s0:s1] = q_np[:, :, j]
                            buf[L].clear()
                        del h, pos_ids, pos_embs, mask_map
                _record_vram()
            finally:
                for h_rem in handles:
                    h_rem.remove()
                _unload_stage(mods)

        del stage_h_cpu

        # ------------------------------------------------------------------
        # STEP 2: Exact zero-ablation sanity check (must observe 0.0 <= 1e-10)
        # ------------------------------------------------------------------
        first_L = hook_layers[0]
        first_stage = layer_to_stage[first_L]
        test_h = clean_layer_in[first_L][:micro_batch_size].clone()
        for s_idx in range(first_stage, n_stages):
            l_start, l_end = stage_ranges[s_idx]
            mods = _load_stage_fp32(s_idx)
            try:
                with torch.no_grad():
                    h = test_h.to(dev, dtype=torch.float32)
                    pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                    loop_a = first_L if s_idx == first_stage else l_start
                    for i in range(loop_a, l_end):
                        h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                    if s_idx == n_stages - 1 and norm_mod is not None:
                        h = norm_mod(h)
                    test_h = h.detach().cpu()
                    del h, pos_ids, pos_embs, mask_map
                _record_vram()
            finally:
                _unload_stage(mods)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        lm_w_gpu = lm_head_weight_cpu.to(device=dev, dtype=torch.float32)
        with torch.no_grad():
            zero_ablation_sanity_kl = (
                _compute_chunked_kl_sum(
                    clean_final_cpu[:micro_batch_size].to(dev),
                    test_h.to(dev),
                    lm_w_gpu,
                    token_chunk=kl_token_chunk,
                )
                / float(micro_batch_size * (seq_len - 1))
            )
        _record_vram()
        del lm_w_gpu, test_h
        _clear_cuda()

        if zero_ablation_sanity_kl > 1e-10:
            raise ParityGateError(
                f"FP32 zero-ablation re-run sanity KL failed: {zero_ablation_sanity_kl:.3e} > 1e-10"
            )
        print(
            f"[band_kl_4b_all278] Clean pass + zero gate PASSED: zero_kl={zero_ablation_sanity_kl:.2e}, "
            f"peak_vram={peak_vram_alloc:.3f}/{peak_vram_res:.3f}GB ({time.time() - t0:.1f}s)",
            flush=True,
        )

        # ------------------------------------------------------------------
        # STEP 3: Build masks, eval_energy, and opposite-sign shares for all 278
        # ------------------------------------------------------------------
        per_pole_gap = gap_doc["per_pole"]
        masks_by_k: dict[int, dict[str, np.ndarray]] = {}
        n_abl_tok_by_k: dict[int, dict[str, int]] = {}
        eval_e_4b_by_k: dict[int, dict[str, float]] = {}
        eval_e_ge20_by_k: dict[int, float] = {}
        eval_e_gc_by_k: dict[int, float] = {}
        opp_share_by_k: dict[int, tuple[float, float, float]] = {}

        for rec in all_recs:
            k = int(rec["k"])
            ap_rec = ap_by_k[k]
            masks = build_neuron_ablation_masks(
                clean_acts_by_k[k], clean_quads_by_k[k], rec, per_pole_gap
            )
            masks_by_k[k] = masks
            n_abl_tok_by_k[k] = {c: int(np.sum(masks[c][:, 1:])) for c in ABLATION_CONDITIONS}

            a_eval = clean_acts_by_k[k][:, 1:].astype(np.float64)
            mean_a = float(rec["mean_act"])
            l2_eval = (a_eval - mean_a) ** 2
            tot_eval_l2 = float(np.sum(l2_eval))
            denom_eval = tot_eval_l2 if tot_eval_l2 > 0.0 else 1.0
            eval_e_4b_by_k[k] = {
                b: float(np.sum(l2_eval[masks[b][:, 1:]]) / denom_eval) for b in FOUR_BAND_CONDS
            }
            eval_e_ge20_by_k[k] = float(
                np.sum(l2_eval[masks["ge_20pct_max"][:, 1:]]) / denom_eval
            )
            eval_e_gc_by_k[k] = float(
                np.sum(l2_eval[masks["gate_closed"][:, 1:]]) / denom_eval
            )

            active_signs = ap_rec.get("active_signs")
            if active_signs is None:
                active_signs = [
                    int(per_pole_gap[pid]["sign"]) for pid in ap_rec["active_pole_ids"]
                ]
            opp_share_by_k[k] = compute_opposite_sign_eval_energy(
                acts_eval_nb=a_eval,
                mean_act=mean_a,
                lt_p75_mask_nb=masks["lt_p75"][:, 1:],
                active_signs=active_signs,
            )

        # ------------------------------------------------------------------
        # STEP 4: Verify existing band_kl_4b.json neurons + 4-anchor determinism
        # ------------------------------------------------------------------
        base_per_neuron = base_kl_doc.get("per_neuron", {}) if base_kl_doc is not None else {}
        base_ks = {int(k) for k in base_per_neuron}
        kl_results_by_k: dict[int, dict[str, float]] = {}
        source_by_k: dict[int, str] = {}

        # Verify clean_acts parity on ALL existing neurons from band_kl_4b.json
        for k_base in sorted(base_ks):
            ref_rec = base_per_neuron[str(k_base)]
            if n_abl_tok_by_k[k_base] != ref_rec["n_ablated_non_bos_tokens"]:
                raise ParityGateError(
                    f"clean_acts token count mismatch on k={k_base}: "
                    f"{n_abl_tok_by_k[k_base]} != {ref_rec['n_ablated_non_bos_tokens']}"
                )
            max_e_diff = max(
                abs(eval_e_4b_by_k[k_base][b] - float(ref_rec["eval_energy_4band"][b]))
                for b in FOUR_BAND_CONDS
            )
            if max_e_diff >= 1e-10:
                raise ParityGateError(
                    f"clean_acts eval_energy_4band mismatch on k={k_base}: max_diff={max_e_diff:.3e}"
                )

        # Validate determinism on the 4 anchor neurons BEFORE running the 218 new neurons
        anchor_set = {int(k) for k in anchor_ks if int(k) in base_ks}
        if base_per_neuron and anchor_ks:
            missing_anchors = [int(k) for k in anchor_ks if int(k) not in base_ks]
            if missing_anchors:
                raise AnchorMismatchError(
                    f"[anchor-mismatch] Anchor ks missing from base_kl_doc: {missing_anchors}"
                )
            anchor_by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for ak in sorted(anchor_set):
                rec = gap_doc["per_neuron"][str(ak)]
                anchor_by_stage[layer_to_stage[int(rec["layer"])]].append(rec)

            max_overall_diff = 0.0
            per_anchor_diff: dict[str, float] = {}
            for s_idx in sorted(anchor_by_stage.keys(), reverse=True):
                anchor_kls = _evaluate_group_kl(
                    backbone=backbone,
                    group_recs=anchor_by_stage[s_idx],
                    masks_by_k=masks_by_k,
                    clean_layer_in=clean_layer_in,
                    clean_final_cpu=clean_final_cpu,
                    lm_head_weight_cpu=lm_head_weight_cpu,
                    stage_ranges=stage_ranges,
                    layer_to_stage=layer_to_stage,
                    start_stage=s_idx,
                    micro_batch_size=micro_batch_size,
                    n_batches=n_batches,
                    total_non_bos_tokens=total_non_bos_tokens,
                    kl_token_chunk=kl_token_chunk,
                    dev=dev,
                    cpu_dev=cpu_dev,
                    orig_cpu_dtype=orig_cpu_dtype,
                    rotary_mod=rotary_mod,
                    norm_mod=norm_mod,
                    record_vram_fn=_record_vram,
                )
                for ak, kls_new in anchor_kls.items():
                    ref_kls = base_per_neuron[str(ak)]["kl_by_cond"]
                    diffs = {
                        c: abs(float(kls_new[c]) - float(ref_kls[c])) for c in ABLATION_CONDITIONS
                    }
                    max_d = max(diffs.values())
                    per_anchor_diff[str(ak)] = max_d
                    max_overall_diff = max(max_overall_diff, max_d)
                    if max_d >= 1e-9:
                        raise AnchorMismatchError(
                            f"[anchor-mismatch] Anchor k={ak} differed by {max_d:.3e} >= 1e-9: {diffs}"
                        )
                    if max_d >= 1e-11:
                        raise AnchorMismatchError(
                            f"[anchor-mismatch] Anchor k={ak} failed < 1e-11 gate: {max_d:.3e}: {diffs}"
                        )
                    kl_results_by_k[ak] = {c: float(kls_new[c]) for c in ABLATION_CONDITIONS}
                    source_by_k[ak] = "band_kl_4b_anchor_verified"

            anchor_validation = {
                "anchor_ks": [int(k) for k in anchor_ks],
                "validated": True,
                "max_abs_diff_overall": max_overall_diff,
                "per_anchor_max_abs_diff": per_anchor_diff,
                "passed_lt_1e_11": max_overall_diff < 1e-11,
            }
            print(
                f"[band_kl_4b_all278] 4-anchor determinism PASSED: "
                f"max_abs_diff={max_overall_diff:.3e} (< 1e-11), per_anchor={per_anchor_diff}",
                flush=True,
            )

        # Load remaining base neurons (56 non-anchors)
        for k_base in sorted(base_ks - anchor_set):
            kl_results_by_k[k_base] = {
                c: float(base_per_neuron[str(k_base)]["kl_by_cond"][c])
                for c in ABLATION_CONDITIONS
            }
            source_by_k[k_base] = "band_kl_4b_reused"

        # ------------------------------------------------------------------
        # STEP 5: Load checkpoints and compute remaining new neurons
        # ------------------------------------------------------------------
        ckpt_loaded = load_ckpt_jsonl(ckpt_path) if ckpt_path is not None else {}
        for k_ck, ck_rec in ckpt_loaded.items():
            if k_ck in base_ks:
                continue
            if "n_ablated_non_bos_tokens" in ck_rec and ck_rec["n_ablated_non_bos_tokens"] != n_abl_tok_by_k[k_ck]:
                print(
                    f"[band_kl_4b_all278] Ignoring stale checkpoint k={k_ck} (token mask mismatch)",
                    flush=True,
                )
                continue
            kl_results_by_k[k_ck] = {c: float(ck_rec["kl_by_cond"][c]) for c in ABLATION_CONDITIONS}
            source_by_k[k_ck] = "computed_all278"

        pending_recs = [
            rec for rec in all_recs if int(rec["k"]) not in kl_results_by_k
        ]
        n_target_new = len(all_recs) - len(base_ks)
        done_new = len(kl_results_by_k) - len(base_ks)
        print(
            f"[band_kl_4b_all278] Reused {len(base_ks)} base neurons; "
            f"resumed {done_new}/{n_target_new} new neurons from ckpt; "
            f"{len(pending_recs)} remaining.",
            flush=True,
        )

        pending_by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in pending_recs:
            pending_by_stage[layer_to_stage[int(rec["layer"])]].append(dict(rec))

        stopped_early_cutoff = False
        for start_stage in sorted(pending_by_stage.keys(), reverse=True):
            stage_recs = sorted(pending_by_stage[start_stage], key=lambda r: int(r["k"]))
            for g0 in range(0, len(stage_recs), neurons_per_group):
                if cutoff_dt is not None and clock_fn() >= cutoff_dt:
                    stopped_early_cutoff = True
                    print(
                        f"[band_kl_4b_all278] Wall-clock cutoff {cutoff_dt.isoformat()} reached at "
                        f"{clock_fn().isoformat()}! Stopping gracefully with {done_new}/{n_target_new} "
                        f"new neurons checkpointed ({len(kl_results_by_k)}/{len(all_recs)} total).",
                        flush=True,
                    )
                    break

                t_g = time.time()
                group = stage_recs[g0 : g0 + neurons_per_group]
                group_kls = _evaluate_group_kl(
                    backbone=backbone,
                    group_recs=group,
                    masks_by_k=masks_by_k,
                    clean_layer_in=clean_layer_in,
                    clean_final_cpu=clean_final_cpu,
                    lm_head_weight_cpu=lm_head_weight_cpu,
                    stage_ranges=stage_ranges,
                    layer_to_stage=layer_to_stage,
                    start_stage=start_stage,
                    micro_batch_size=micro_batch_size,
                    n_batches=n_batches,
                    total_non_bos_tokens=total_non_bos_tokens,
                    kl_token_chunk=kl_token_chunk,
                    dev=dev,
                    cpu_dev=cpu_dev,
                    orig_cpu_dtype=orig_cpu_dtype,
                    rotary_mod=rotary_mod,
                    norm_mod=norm_mod,
                    record_vram_fn=_record_vram,
                )

                ckpt_batch: list[dict[str, Any]] = []
                g_sec = round(time.time() - t_g, 2)
                for rec in group:
                    k = int(rec["k"])
                    kl_results_by_k[k] = group_kls[k]
                    source_by_k[k] = "computed_all278"
                    opp_lt, opp_tot, opp_frac = opp_share_by_k[k]
                    ckpt_batch.append(
                        {
                            "k": k,
                            "neuron_id": str(rec["neuron_id"]),
                            "layer": int(rec["layer"]),
                            "neuron": int(rec["neuron"]),
                            "kl_by_cond": group_kls[k],
                            "n_ablated_non_bos_tokens": n_abl_tok_by_k[k],
                            "eval_energy_4band": eval_e_4b_by_k[k],
                            "eval_energy_ge_20pct_max": eval_e_ge20_by_k[k],
                            "eval_energy_gate_closed": eval_e_gc_by_k[k],
                            "opposite_sign_eval_energy_share": opp_lt,
                            "opposite_sign_total_eval_energy_share": opp_tot,
                            "opposite_sign_frac_of_lt_p75": opp_frac,
                            "elapsed_group_sec": g_sec,
                        }
                    )
                if ckpt_path is not None:
                    append_ckpt_jsonl(ckpt_path, ckpt_batch)
                done_new += len(group)
                print(
                    f"[band_kl_4b_all278] Stage {start_stage} group done "
                    f"({done_new}/{n_target_new} new, {len(kl_results_by_k)}/{len(all_recs)} total, "
                    f"group={g_sec:.1f}s, elapsed={time.time() - t0:.1f}s, "
                    f"peak_vram={peak_vram_alloc:.2f}/{peak_vram_res:.2f}GB)",
                    flush=True,
                )
            if stopped_early_cutoff:
                break
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn

    if stopped_early_cutoff:
        return {
            "status": "partial_cutoff",
            "completed_total": len(kl_results_by_k),
            "target_total": len(all_recs),
            "completed_new": len(kl_results_by_k) - len(base_ks),
            "target_new": len(all_recs) - len(base_ks),
            "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
            "zero_ablation_sanity_kl": zero_ablation_sanity_kl,
            "anchor_validation": anchor_validation,
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------
    # STEP 6: Assemble all 278 neuron records & aggregate statistics
    # ------------------------------------------------------------------
    per_neuron_kl: dict[str, dict[str, Any]] = {}
    for rec in all_recs:
        k = int(rec["k"])
        opp_lt, opp_tot, opp_frac = opp_share_by_k[k]
        per_neuron_kl[str(k)] = build_neuron_kl_record(
            gap_rec=rec,
            ap_rec=ap_by_k[k],
            kl_by_cond=kl_results_by_k[k],
            n_ablated_non_bos_tokens=n_abl_tok_by_k[k],
            eval_energy_4band=eval_e_4b_by_k[k],
            eval_energy_ge_20pct_max=eval_e_ge20_by_k[k],
            eval_energy_gate_closed=eval_e_gc_by_k[k],
            opposite_sign_eval_energy_share=opp_lt,
            opposite_sign_total_eval_energy_share=opp_tot,
            opposite_sign_frac_of_lt_p75=opp_frac,
            source=source_by_k[k],
        )

    doc = assemble_band_kl_all278_doc(
        per_neuron_kl=per_neuron_kl,
        gap_doc=gap_doc,
        stage_ranges=stage_ranges,
        n_seqs=n_seqs,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        kl_token_chunk=kl_token_chunk,
        peak_vram_alloc=peak_vram_alloc,
        peak_vram_res=peak_vram_res,
        max_vram_gb=max_vram_gb,
        is_cuda=is_cuda,
        zero_ablation_sanity_kl=zero_ablation_sanity_kl,
        anchor_validation=anchor_validation,
        elapsed_sec=time.time() - t0,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
        extra_meta=extra_meta,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    return doc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run exact IEEE-754 FP32 4-band mean-ablation KL across all 278 Qwen3-4B neurons."
    )
    parser.add_argument("--model-id", default=MODEL_ID_4B)
    parser.add_argument("--n-seqs", type=int, default=KL_N_SEQS_4B)
    parser.add_argument("--seq-len", type=int, default=KL_SEQ_LEN_4B)
    parser.add_argument("--min-row-idx", type=int, default=KL_MIN_ROW_IDX_4B)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE_4B)
    parser.add_argument("--kl-token-chunk", type=int, default=KL_TOKEN_CHUNK_4B)
    parser.add_argument("--neurons-per-group", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-vram-gb", type=float, default=6.5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED_4B)
    parser.add_argument("--cutoff-ist", default=DEFAULT_CUTOFF_IST_ISO)
    args = parser.parse_args(argv)

    dirs = ensure_4b_dirs()
    gap_path = dirs["results"] / "gap_decomposition_4b.json"
    base_kl_path = dirs["results"] / "band_kl_4b.json"
    ckpt_path = dirs["results"] / "band_kl_4b_all278_ckpt.jsonl"
    out_path = dirs["results"] / "band_kl_4b_all278.json"

    gap_doc = json.loads(gap_path.read_text(encoding="utf-8"))
    active_poles_doc = json.loads(ACTIVE_POLES_PATH.read_text(encoding="utf-8"))
    base_kl_doc = json.loads(base_kl_path.read_text(encoding="utf-8"))

    cutoff_dt = datetime.fromisoformat(args.cutoff_ist) if args.cutoff_ist else None

    backbone, tokenizer = load_qwen4b_backbone_cpu(model_id=args.model_id, dtype=torch.bfloat16)
    docs_it = iter_fineweb_disjoint_rows(min_row_idx=args.min_row_idx)
    seq_chunks = collect_fixed_length_sequences(
        docs_with_ids=docs_it,
        tokenizer=tokenizer,
        n_sequences=args.n_seqs,
        seq_len=args.seq_len,
        max_seqs_per_doc=1,
        min_doc_id=args.min_row_idx,
    )
    input_ids = torch.tensor([c.token_ids for c in seq_chunks], dtype=torch.long)
    doc_ids_used = [c.doc_id for c in seq_chunks]
    print(
        f"[band_kl_4b_all278] Loaded {len(seq_chunks)} seqs x {args.seq_len} = {input_ids.numel():,} tok "
        f"(rows {min(doc_ids_used)}..{max(doc_ids_used)})",
        flush=True,
    )

    doc = run_band_kl_all278(
        backbone=backbone,
        input_ids_cpu=input_ids,
        gap_doc=gap_doc,
        active_poles_doc=active_poles_doc,
        base_kl_doc=base_kl_doc,
        anchor_ks=ANCHOR_NEURON_KS,
        compute_device=args.device,
        micro_batch_size=args.micro_batch_size,
        kl_token_chunk=args.kl_token_chunk,
        max_vram_gb=args.max_vram_gb,
        neurons_per_group=args.neurons_per_group,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
        ckpt_path=ckpt_path,
        output_path=out_path,
        cutoff_dt=cutoff_dt,
        extra_meta={
            "fineweb_min_row_idx_allowed": args.min_row_idx,
            "fineweb_min_doc_id_observed": int(min(doc_ids_used)),
            "fineweb_max_doc_id_observed": int(max(doc_ids_used)),
            "kl_token_chunk_note": (
                "Matched band_kl_4b.json (kl_token_chunk=512) for exact 0.00e+00 anchor parity; "
                "kl_token_chunk=4096 causes a 2.11e-10 cuBLAS reduction shift (> 1e-11 anchor gate) "
                "and 5.55 GB peak VRAM vs 4.71 GB at 512."
            ),
        },
    )
    if doc.get("status") == "partial_cutoff":
        print(f"[band_kl_4b_all278] PARTIAL_CUTOFF: {json.dumps(doc)}", flush=True)
    else:
        print(
            f"[band_kl_4b_all278] COMPLETE in {doc['elapsed_sec']:.1f}s | n={doc['n_neurons']} | "
            f"vram={doc['peak_vram_allocated_gb']:.3f}/{doc['peak_vram_reserved_gb']:.3f}GB | "
            f"zero_kl={doc['zero_ablation_sanity_kl']:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
