"""Cross-model causal comparison & structural band-KL evaluation (Qwen3-1.7B vs Qwen3-4B).

Strictly isolated under `results/qwen17b_causal/`:
- Reuses frozen `results/judgements.jsonl` (`dead_zone`, `own`, `fwd`) — ZERO new detection calls
- Reads 1.7B `fp32` energy/KL (`results/qwen17b_causal/energy_and_kl_17b.json`),
  causal substitution scores (`results/qwen17b_causal/causal_scores_17b.json`),
  simulation calls (`results/qwen17b_causal/sim_calls_17b.jsonl`),
  YES/NO precision calls (`results/qwen17b_causal/fpr_17b.jsonl`),
  and Qwen3-4B benchmarks (`results/qwen4b/surrogate_metrics_4b.json`, `results/qwen4b/band_kl_4b.json`)
- Writes strictly to:
  - `results/qwen17b_causal/metrics_17b_causal.json`
  - `results/qwen17b_causal/summary_17b_causal.md`
  - `figures/cross_model_17b_vs_4b.png`

All summaries and figures are strictly factual (zero interpretive prose).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from qwen17b_causal.simulate_17b import (
    DEFAULT_FPR_17B_PATH,
    DEFAULT_POLES_17B_PATH,
    DEFAULT_SIM_CALLS_17B_PATH,
    assert_17b_causal_output_path,
    load_poles_17b,
)
from qwen4b_eval.analyze_4b import (
    PRIMARY_JUDGE_4B,
    SECONDARY_JUDGE_4B,
    _dedup_judgements_4b,
    _fmt_ci,
    _fmt_num,
    _safe_spearman,
    bootstrap_corr_ci,
)
from qwen4b_eval.effect_weighted import pooled_energy_kl_analysis
from qwen4b_eval.surrogate_4b import (
    DEFAULT_BAND_KL_PATH,
    DEFAULT_SURROGATE_METRICS_PATH,
    RIDGE_ALPHA_GRID,
    _dedup_fpr_rows,
    _fit_ridge_closed_form,
    _train_only_ecdf_rank_transform,
    bootstrap_paired_spearman_diff,
    compute_wdet_score,
    exact_auroc,
    expected_precision_at_k,
    make_stratified_folds,
    partial_spearman_corr,
)
from w_audit.analyze import DEFAULT_BOOTSTRAP_B, classify_row_outcome
from w_audit.config import SEED
from w_audit.labeler import load_jsonl

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGEMENTS_17B_PATH: Path = Path("results/judgements.jsonl")
DEFAULT_ENERGY_KL_17B_PATH: Path = Path("results/qwen17b_causal/energy_and_kl_17b.json")
DEFAULT_CAUSAL_SCORES_17B_PATH: Path = Path("results/qwen17b_causal/causal_scores_17b.json")

DEFAULT_METRICS_17B_CAUSAL_PATH: Path = Path("results/qwen17b_causal/metrics_17b_causal.json")
DEFAULT_SUMMARY_17B_CAUSAL_PATH: Path = Path("results/qwen17b_causal/summary_17b_causal.md")
DEFAULT_FIGURE_CROSS_MODEL_PATH: Path = Path("figures/cross_model_17b_vs_4b.png")

CROSS_MODEL_CREDIT_FOOTER: str = (
    "Qwen3-4B labels & causal recovery scores: Ulto85/neurons-kinda-clean (MIT, Qwen3-4B); "
    "Qwen3-1.7B: frozen gpt-4.1 labels & fp32 causal simulation"
)

BANDS_4_KEYS: tuple[str, ...] = ("p99.9+", "p97_99.9", "p75_97", "lt_p75")


# ---------------------------------------------------------------------------
# Flexible Schema Extractors for Step 3B / 3C JSONs
# ---------------------------------------------------------------------------


def _normalize_per_neuron_map(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index per-neuron records cleanly by canonical neuron_id ('L04_N2467')."""
    raw = doc.get("per_neuron", doc.get("neurons", {}))
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if not isinstance(v, Mapping):
                continue
            nid = str(v.get("neuron_id") or k)
            if not nid.startswith("L") and "layer" in v and "neuron" in v:
                nid = f"L{int(v['layer']):02d}_N{int(v['neuron']):04d}"
            out[nid] = dict(v)
    elif isinstance(raw, list):
        for v in raw:
            if not isinstance(v, Mapping):
                continue
            nid = str(v.get("neuron_id") or f"L{int(v['layer']):02d}_N{int(v['neuron']):04d}")
            out[nid] = dict(v)
    return out


def _extract_4band_dict(entry: Mapping[str, Any], candidate_paths: Sequence[tuple[str, ...]]) -> dict[str, float] | None:
    """Walk candidate nested paths to extract a 4-band share dict."""
    for path in candidate_paths:
        cur: Any = entry
        ok = True
        for seg in path:
            if isinstance(cur, Mapping) and seg in cur:
                cur = cur[seg]
            else:
                ok = False
                break
        if ok and isinstance(cur, Mapping):
            # Support minor alias differences like p99.9_plus vs p99.9+
            alias_map = {
                "p99.9+": ("p99.9+", "p99.9_plus", "p99_9_plus"),
                "p97_99.9": ("p97_99.9", "p97_99_9"),
                "p75_97": ("p75_97",),
                "lt_p75": ("lt_p75", "bulk_lt_p75"),
            }
            parsed: dict[str, float] = {}
            found_all = True
            for bk, aliases in alias_map.items():
                val = None
                for a in aliases:
                    if a in cur and cur[a] is not None:
                        val = float(cur[a])
                        break
                if val is None:
                    found_all = False
                    break
                parsed[bk] = val
            if found_all:
                return parsed
    return None


def _get_float_alias(entry: Mapping[str, Any], keys: Sequence[str], default: float = float("nan")) -> float:
    for k in keys:
        if k in entry and entry[k] is not None:
            v = float(entry[k])
            if math.isfinite(v):
                return v
    return default


# ---------------------------------------------------------------------------
# Aggregation: Frozen 1.7B Detection + YES/NO FPR
# ---------------------------------------------------------------------------


def aggregate_17b_detection_dead_zone(
    judgements_path: Path,
    poles_by_nid: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[tuple[str, str, str], float], dict[str, dict[str, Any]]]:
    """Load frozen Qwen3-1.7B own/fwd/dead_zone detection accuracy per (neuron_id, band, judge)."""
    rows = _dedup_judgements_4b(load_jsonl(judgements_path))
    pole_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    audit: dict[str, dict[str, Any]] = {}

    valid_judges = {PRIMARY_JUDGE_4B, SECONDARY_JUDGE_4B}
    for row in rows:
        if str(row.get("condition")) != "own":
            continue
        if str(row.get("order", "fwd")) != "fwd":
            continue
        if str(row.get("distractor_type")) != "dead_zone":
            continue
        jm = str(row.get("judge_model"))
        if jm not in valid_judges:
            continue
        band = str(row["band"])
        pid = str(row["pole_id"])
        outcome = classify_row_outcome(row)
        akey = f"{jm}:{band}"
        if akey not in audit:
            audit[akey] = {"judge_model": jm, "band": band, "n_total": 0, "n_scored": 0, "n_unp": 0, "n_tie": 0}
        audit[akey]["n_total"] += 1
        if outcome == "scored":
            audit[akey]["n_scored"] += 1
            pole_scores[(pid, band, jm)].append(float(row["correct"]))
        elif outcome == "logprob_tie":
            audit[akey]["n_tie"] += 1
        else:
            audit[akey]["n_unp"] += 1

    pole_acc = {k: float(np.mean(v)) for k, v in pole_scores.items() if v}

    neuron_acc: dict[tuple[str, str, str], float] = {}
    bands_present = ("p99.9_plus", "p97_99.9", "p90_97", "p75_90")
    for nid, plist in poles_by_nid.items():
        pids = [str(p["pole_id"]) for p in plist]
        for jm in valid_judges:
            for band in bands_present:
                vals = [pole_acc[(pid, band, jm)] for pid in pids if (pid, band, jm) in pole_acc]
                if vals:
                    neuron_acc[(nid, band, jm)] = float(np.mean(vals))
            # Composite p75_97 matching 4-band energy/KL partition
            v90 = neuron_acc.get((nid, "p90_97", jm))
            v75 = neuron_acc.get((nid, "p75_90", jm))
            if v90 is not None and v75 is not None:
                neuron_acc[(nid, "p75_97", jm)] = 0.5 * (v90 + v75)
            elif v90 is not None:
                neuron_acc[(nid, "p75_97", jm)] = v90
            elif v75 is not None:
                neuron_acc[(nid, "p75_97", jm)] = v75

    for akey, a in audit.items():
        jm, band = a["judge_model"], a["band"]
        nvals = [neuron_acc[(nid, band, jm)] for nid in poles_by_nid if (nid, band, jm) in neuron_acc]
        a["n_neurons"] = len(nvals)
        a["mean_neuron_acc"] = float(np.mean(nvals)) if nvals else float("nan")

    return neuron_acc, audit


def aggregate_17b_fpr(
    fpr_rows: Sequence[Mapping[str, Any]],
    poles_by_nid: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, Any]]]:
    """Aggregate 1.7B YES/NO precision metrics to neuron level per judge."""
    pole_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    audit: dict[str, dict[str, Any]] = {}

    for row in fpr_rows:
        jm = str(row["judge_model"])
        ptype = str(row["probe_type"])
        pid = str(row["pole_id"])
        outcome = str(row.get("outcome", "unparseable"))
        akey = f"{jm}:{ptype}"
        if akey not in audit:
            audit[akey] = {
                "judge_model": jm,
                "probe_type": ptype,
                "n_total": 0,
                "n_scored": 0,
                "n_unparseable": 0,
                "n_logprob_tie": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }
        a = audit[akey]
        a["n_total"] += 1
        a["reasoning_tokens"] += int(row.get("reasoning_tokens", 0) or 0)
        a["cost_usd"] += float(row.get("cost_usd", 0.0) or 0.0)
        if outcome == "scored":
            a["n_scored"] += 1
            pole_buckets[(pid, jm, ptype)].append(dict(row))
        elif outcome == "logprob_tie":
            a["n_logprob_tie"] += 1
        else:
            a["n_unparseable"] += 1

    for a in audit.values():
        a["cost_usd"] = round(a["cost_usd"], 6)

    def _mean_f(sub: Sequence[Mapping[str, Any]], fld: str) -> float:
        vals = [float(r[fld]) for r in sub if r.get(fld) is not None and math.isfinite(float(r[fld]))]
        return float(np.mean(vals)) if vals else float("nan")

    judges = sorted({str(r["judge_model"]) for r in fpr_rows})
    pole_m: dict[tuple[str, str], dict[str, float]] = {}
    for plist in poles_by_nid.values():
        for p in plist:
            pid = str(p["pole_id"])
            for jm in judges:
                ord_r = pole_buckets.get((pid, jm, "ordinary_nonfiring"), [])
                top_r = pole_buckets.get((pid, jm, "top100_firing"), [])
                if not ord_r and not top_r:
                    continue
                fpr_ord = _mean_f(ord_r, "predicted_yes")
                tpr_top = _mean_f(top_r, "predicted_yes")
                py_ord = _mean_f(ord_r, "p_yes")
                py_top = _mean_f(top_r, "p_yes")
                ym_ord = _mean_f(ord_r, "yes_margin")
                ym_top = _mean_f(top_r, "yes_margin")
                pole_m[(pid, jm)] = {
                    "fpr_ord": fpr_ord,
                    "tpr_top": tpr_top,
                    "one_minus_fpr_ord": 1.0 - fpr_ord if math.isfinite(fpr_ord) else float("nan"),
                    "tpr_minus_fpr": tpr_top - fpr_ord if math.isfinite(tpr_top) and math.isfinite(fpr_ord) else float("nan"),
                    "balanced_acc_yesno": 0.5 * (tpr_top + 1.0 - fpr_ord) if math.isfinite(tpr_top) and math.isfinite(fpr_ord) else float("nan"),
                    "p_yes_ord": py_ord,
                    "one_minus_p_yes_ord": 1.0 - py_ord if math.isfinite(py_ord) else float("nan"),
                    "yes_margin_ord": ym_ord,
                    "yes_margin_top": ym_top,
                    "yes_margin_gap": ym_top - ym_ord if math.isfinite(ym_top) and math.isfinite(ym_ord) else float("nan"),
                }

    fields = (
        "fpr_ord",
        "tpr_top",
        "one_minus_fpr_ord",
        "tpr_minus_fpr",
        "balanced_acc_yesno",
        "p_yes_ord",
        "one_minus_p_yes_ord",
        "yes_margin_ord",
        "yes_margin_top",
        "yes_margin_gap",
    )
    neuron_m: dict[tuple[str, str], dict[str, float]] = {}
    for nid, plist in poles_by_nid.items():
        pids = [str(p["pole_id"]) for p in plist]
        for jm in judges:
            pms = [pole_m[(pid, jm)] for pid in pids if (pid, jm) in pole_m]
            if not pms:
                continue
            neuron_m[(nid, jm)] = {
                fld: (
                    float(np.mean([pm[fld] for pm in pms if math.isfinite(pm[fld])]))
                    if any(math.isfinite(pm[fld]) for pm in pms)
                    else float("nan")
                )
                for fld in fields
            }

    return neuron_m, audit


# ---------------------------------------------------------------------------
# Zero-Shot Out-of-Model Transfer: Fit Strictly on Qwen3-4B -> Predict Qwen3-1.7B
# ---------------------------------------------------------------------------


def fit_4b_transfer_and_predict_17b(
    X_train_4b: np.ndarray,
    y_train_4b: np.ndarray,
    layers_4b: Sequence[int],
    X_eval_17b: np.ndarray,
    rank_mode: bool = False,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit ridge or rank-ridge STRICTLY on Qwen3-4B (inner 5-fold CV for alpha) and predict Qwen3-1.7B zero-shot."""
    X_tr = np.asarray(X_train_4b, dtype=np.float64).copy()
    y_tr = np.asarray(y_train_4b, dtype=np.float64).copy()
    X_ev = np.asarray(X_eval_17b, dtype=np.float64).copy()
    p = X_tr.shape[1]

    # Impute any non-finite entries strictly from 4B medians
    for j in range(p):
        fin_tr = np.isfinite(X_tr[:, j])
        med_j = float(np.median(X_tr[fin_tr, j])) if np.any(fin_tr) else 0.0
        X_tr[~fin_tr, j] = med_j
        fin_ev = np.isfinite(X_ev[:, j])
        X_ev[~fin_ev, j] = med_j

    folds_4b = make_stratified_folds(layers_4b, n_folds=5, seed=seed)
    best_alpha = float(alpha_grid[0])
    best_score = -float("inf") if rank_mode else float("inf")

    for alpha in alpha_grid:
        oof = np.zeros(len(y_tr), dtype=np.float64)
        for f in range(5):
            va = folds_4b == f
            tr = ~va
            if rank_mode:
                Xi_tr, Xi_va = _train_only_ecdf_rank_transform(X_tr[tr], X_tr[va])
                yi_tr = rankdata(y_tr[tr], method="average") / float(np.sum(tr) + 1)
            else:
                Xi_tr, Xi_va = X_tr[tr], X_tr[va]
                yi_tr = y_tr[tr]
            mu = np.mean(Xi_tr, axis=0)
            sd = np.std(Xi_tr, axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            y_mu = float(np.mean(yi_tr))
            w = _fit_ridge_closed_form((Xi_tr - mu) / sd, yi_tr - y_mu, alpha)
            oof[va] = y_mu + ((Xi_va - mu) / sd) @ w

        if rank_mode:
            rho, _ = _safe_spearman(oof, y_tr)
            sc = rho if math.isfinite(rho) else -1.0
            if sc > best_score:
                best_score = sc
                best_alpha = float(alpha)
        else:
            mse = float(np.mean((oof - y_tr) ** 2))
            if mse < best_score:
                best_score = mse
                best_alpha = float(alpha)

    # Fit strictly on all 278 Qwen3-4B neurons and transform 1.7B using 4B-only stats
    if rank_mode:
        X_tr_f, X_ev_f = _train_only_ecdf_rank_transform(X_tr, X_ev)
        y_tr_f = rankdata(y_tr, method="average") / float(len(y_tr) + 1)
    else:
        X_tr_f, X_ev_f = X_tr, X_ev
        y_tr_f = y_tr

    mu_f = np.mean(X_tr_f, axis=0)
    sd_f = np.std(X_tr_f, axis=0)
    sd_f = np.where(sd_f < 1e-12, 1.0, sd_f)
    y_mu_f = float(np.mean(y_tr_f))
    w_f = _fit_ridge_closed_form((X_tr_f - mu_f) / sd_f, y_tr_f - y_mu_f, best_alpha)
    preds_17b = y_mu_f + ((X_ev_f - mu_f) / sd_f) @ w_f

    return {
        "best_alpha_4b_cv": best_alpha,
        "std_weights_4b": [round(float(v), 6) for v in w_f],
        "preds_17b": preds_17b,
    }


def _eval_17b_candidate(
    name: str,
    scores: Sequence[float],
    y_sim: Sequence[float],
    y_sim_insample: Sequence[float],
    y_active: Sequence[float],
    z_e20: Sequence[float],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluate one surrogate vector on Qwen3-1.7B across all targets, partials, and AUROCs."""
    sc = np.asarray(scores, dtype=np.float64)
    ys = np.asarray(y_sim, dtype=np.float64)
    yi = np.asarray(y_sim_insample, dtype=np.float64)
    ya = np.asarray(y_active, dtype=np.float64)
    ze = np.asarray(z_e20, dtype=np.float64)

    return {
        "name": name,
        "sim_cal_rec": bootstrap_corr_ci(sc, ys, n_resamples=n_bootstrap, seed=seed),
        "sim_cal_rec_insample": bootstrap_corr_ci(sc, yi, n_resamples=n_bootstrap, seed=seed + 5),
        "sim_cal_rec_active": bootstrap_corr_ci(sc, ya, n_resamples=n_bootstrap, seed=seed + 11),
        "partial_above_e20_sim_cal_rec": partial_spearman_corr(sc, ys, ze, n_resamples=n_bootstrap, seed=seed + 17),
        "partial_above_e20_sim_cal_rec_insample": partial_spearman_corr(sc, yi, ze, n_resamples=n_bootstrap, seed=seed + 23),
        "partial_above_e20_sim_cal_rec_active": partial_spearman_corr(sc, ya, ze, n_resamples=n_bootstrap, seed=seed + 29),
        "delta_rho_vs_e20_sim_cal_rec": bootstrap_paired_spearman_diff(sc, ze, ys, n_resamples=n_bootstrap, seed=seed + 37),
        "classification_sim_cal_rec": {
            "pass_gt_0_00": {
                "n_pos": int(np.sum(ys > 0.0)),
                "auroc": exact_auroc(sc, ys > 0.0),
                "precision_at_10": expected_precision_at_k(sc, ys > 0.0, 10),
                "precision_at_20": expected_precision_at_k(sc, ys > 0.0, 20),
            },
            "pass_ge_0_25": {
                "n_pos": int(np.sum(ys >= 0.25)),
                "auroc": exact_auroc(sc, ys >= 0.25),
                "precision_at_10": expected_precision_at_k(sc, ys >= 0.25, 10),
                "precision_at_20": expected_precision_at_k(sc, ys >= 0.25, 20),
            },
            "pass_ge_0_50": {
                "n_pos": int(np.sum(ys >= 0.50)),
                "auroc": exact_auroc(sc, ys >= 0.50),
                "precision_at_10": expected_precision_at_k(sc, ys >= 0.50, 10),
                "precision_at_20": expected_precision_at_k(sc, ys >= 0.50, 20),
            },
            "fail_le_0_00": {
                "n_pos": int(np.sum(ys <= 0.0)),
                "auroc": exact_auroc(-sc, ys <= 0.0),
                "precision_at_10": expected_precision_at_k(-sc, ys <= 0.0, 10),
                "precision_at_20": expected_precision_at_k(-sc, ys <= 0.0, 20),
            },
        },
    }


# ---------------------------------------------------------------------------
# Main Cross-Model Analysis Pipeline
# ---------------------------------------------------------------------------


def analyze_17b_causal(
    poles_path: Path = DEFAULT_POLES_17B_PATH,
    judgements_17b_path: Path = DEFAULT_JUDGEMENTS_17B_PATH,
    energy_kl_17b_path: Path = DEFAULT_ENERGY_KL_17B_PATH,
    causal_scores_17b_path: Path = DEFAULT_CAUSAL_SCORES_17B_PATH,
    sim_calls_17b_path: Path = DEFAULT_SIM_CALLS_17B_PATH,
    fpr_17b_path: Path = DEFAULT_FPR_17B_PATH,
    surrogate_4b_metrics_path: Path = DEFAULT_SURROGATE_METRICS_PATH,
    band_kl_4b_path: Path = DEFAULT_BAND_KL_PATH,
    output_metrics_path: Path = DEFAULT_METRICS_17B_CAUSAL_PATH,
    output_summary_path: Path = DEFAULT_SUMMARY_17B_CAUSAL_PATH,
    figure_path: Path = DEFAULT_FIGURE_CROSS_MODEL_PATH,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute 1.7B causal evaluation & cross-model comparison vs Qwen3-4B."""
    for p in (output_metrics_path, output_summary_path, figure_path):
        assert_17b_causal_output_path(p)

    poles = load_poles_17b(poles_path)
    poles_by_nid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in poles:
        nid = str(p.get("neuron_id") or f"L{int(p['layer']):02d}_N{int(p['neuron']):04d}")
        poles_by_nid[nid].append(p)
    nids_sorted = sorted(poles_by_nid.keys())

    # 1. Load frozen 1.7B 4-band dead_zone detection
    det_17b, det_audit = aggregate_17b_detection_dead_zone(judgements_17b_path, poles_by_nid)

    # 2. Load 1.7B FPR & simulation call spend/parse audits
    fpr_rows = _dedup_fpr_rows(load_jsonl(fpr_17b_path)) if fpr_17b_path.is_file() else []
    fpr_17b, fpr_audit = aggregate_17b_fpr(fpr_rows, poles_by_nid)

    sim_rows_raw = load_jsonl(sim_calls_17b_path) if sim_calls_17b_path.is_file() else []
    sim_dedup = {(str(r["window_id"]), str(r["judge_model"])): r for r in sim_rows_raw}
    sim_rows = list(sim_dedup.values())
    sim_spend_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in sim_rows)
    fpr_spend_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in fpr_rows)
    sim_parse_counts = Counter(str(r.get("parse_status", "unknown")) for r in sim_rows)
    sim_reas_toks = sum(int(r.get("reasoning_tokens", 0) or 0) for r in sim_rows)
    fpr_reas_toks = sum(int(r.get("reasoning_tokens", 0) or 0) for r in fpr_rows)

    # 3. Load Step 3B energy+KL & Step 3C causal scores
    ekl_doc = json.loads(energy_kl_17b_path.read_text(encoding="utf-8"))
    cs_doc = json.loads(causal_scores_17b_path.read_text(encoding="utf-8"))
    ekl_by_nid = _normalize_per_neuron_map(ekl_doc)
    cs_by_nid = _normalize_per_neuron_map(cs_doc)

    # Build unified per-neuron table for all 50 Qwen3-1.7B neurons
    per_neuron: dict[str, dict[str, Any]] = {}
    b4_map = (
        ("p99.9+", "p99.9_plus"),
        ("p97_99.9", "p97_99.9"),
        ("p75_97", "p75_97"),
        ("lt_p75", "lt_p75"),
    )

    pooled_eval_e: list[float] = []
    pooled_corp_e: list[float] = []
    pooled_kl: list[float] = []
    band_e_eval: dict[str, list[float]] = {b: [] for b in BANDS_4_KEYS}
    band_e_corp: dict[str, list[float]] = {b: [] for b in BANDS_4_KEYS}
    band_kl_list: dict[str, list[float]] = {b: [] for b in BANDS_4_KEYS}
    ek_rows_eval: list[list[float]] = []
    ek_rows_kl: list[list[float]] = []

    for nid in nids_sorted:
        e_rec = ekl_by_nid[nid]
        c_rec = cs_by_nid[nid]
        layer = int(poles_by_nid[nid][0]["layer"])
        neuron = int(poles_by_nid[nid][0]["neuron"])

        corp_4b = _extract_4band_dict(
            e_rec,
            (
                ("corpus_energy_4band",),
                ("energy_l2", "bands_4"),
                ("corpus_energy_l2", "bands_4"),
                ("energy_4band",),
                ("bands_4",),
            ),
        )
        eval_4b = _extract_4band_dict(
            e_rec,
            (
                ("eval_energy_4band",),
                ("eval_energy_l2", "bands_4"),
            ),
        )
        kl_4b = _extract_4band_dict(
            e_rec,
            (
                ("kl_share_4band_norm",),
                ("kl_shares_4band",),
                ("kl_share_4band",),
            ),
        )
        if corp_4b is None:
            raise KeyError(f"Missing corpus 4-band energy shares for {nid} in {energy_kl_17b_path}")
        if kl_4b is not None:
            if eval_4b is not None:
                ek_rows_eval.append([float(eval_4b[bk]) for bk in BANDS_4_KEYS])
                ek_rows_kl.append([float(kl_4b[bk]) for bk in BANDS_4_KEYS])
            for bk in BANDS_4_KEYS:
                pooled_corp_e.append(float(corp_4b[bk]))
                pooled_kl.append(float(kl_4b[bk]))
                band_e_corp[bk].append(float(corp_4b[bk]))
                band_kl_list[bk].append(float(kl_4b[bk]))
                if eval_4b is not None:
                    pooled_eval_e.append(float(eval_4b[bk]))
                    band_e_eval[bk].append(float(eval_4b[bk]))

        e20 = _get_float_alias(e_rec, ("corpus_energy_ge_20pct_max", "energy_ge_20pct_max"))
        e50 = _get_float_alias(e_rec, ("corpus_energy_ge_50pct_max", "energy_ge_50pct_max"))
        e20_eval = _get_float_alias(e_rec, ("eval_energy_ge_20pct_max",))
        e_gc = _get_float_alias(e_rec, ("corpus_energy_gate_closed", "energy_gate_closed"))
        e_gc_eval = _get_float_alias(e_rec, ("eval_energy_gate_closed",))
        kl_gc = _get_float_alias(e_rec, ("kl_share_gate_closed_norm", "kl_share_gate_closed_vs_full", "kl_share_gate_closed"))

        strong20_ord8k = _get_float_alias(e_rec, ("strong20_rec",))
        strong50_ord8k = _get_float_alias(e_rec, ("strong50_rec",))
        strong20 = _get_float_alias(c_rec, ("strong20_rec_eval64",), strong20_ord8k)
        strong50 = _get_float_alias(c_rec, ("strong50_rec_eval64",), strong50_ord8k)
        sim_cal = _get_float_alias(c_rec, ("sim_cal_rec",))
        sim_cal_cv = _get_float_alias(c_rec, ("sim_cal_rec_cv", "sim_cal_rec_2fold_cv"), sim_cal)
        sim_cal_insample = _get_float_alias(c_rec, ("sim_cal_rec_insample",), sim_cal)
        sim_act = _get_float_alias(c_rec, ("sim_cal_rec_active",))
        sim_act_insample = _get_float_alias(c_rec, ("sim_cal_rec_active_insample",), sim_act)
        corr_v = _get_float_alias(c_rec, ("corr", "pearson_corr"))
        sim_raw = _get_float_alias(c_rec, ("sim_raw_rec",))

        accs_t = {
            "p99.9_plus": det_17b.get((nid, "p99.9_plus", PRIMARY_JUDGE_4B)),
            "p97_99.9": det_17b.get((nid, "p97_99.9", PRIMARY_JUDGE_4B)),
            "p75_97": det_17b.get((nid, "p75_97", PRIMARY_JUDGE_4B)),
            "p90_97": det_17b.get((nid, "p90_97", PRIMARY_JUDGE_4B)),
            "p75_90": det_17b.get((nid, "p75_90", PRIMARY_JUDGE_4B)),
            "lt_p75": None,
        }
        accs_l = {
            "p99.9_plus": det_17b.get((nid, "p99.9_plus", SECONDARY_JUDGE_4B)),
            "p97_99.9": det_17b.get((nid, "p97_99.9", SECONDARY_JUDGE_4B)),
            "p75_97": det_17b.get((nid, "p75_97", SECONDARY_JUDGE_4B)),
            "p90_97": det_17b.get((nid, "p90_97", SECONDARY_JUDGE_4B)),
            "p75_90": det_17b.get((nid, "p75_90", SECONDARY_JUDGE_4B)),
            "lt_p75": None,
        }

        wdet_t = compute_wdet_score(corp_4b, accs_t, b4_map, mode="clamped")
        wdet_t_signed = compute_wdet_score(corp_4b, accs_t, b4_map, mode="signed")
        wdet_l = compute_wdet_score(corp_4b, accs_l, b4_map, mode="clamped")
        corp_5b = e_rec.get("corpus_energy_5band")
        b5_map_17b = (
            ("p99.9_plus", "p99.9_plus"),
            ("p97_99.9", "p97_99.9"),
            ("p90_97", "p90_97"),
            ("p75_90", "p75_90"),
            ("lt_p75", "lt_p75"),
        )
        wdet_5b_t = (
            compute_wdet_score(corp_5b, accs_t, b5_map_17b, mode="clamped")
            if isinstance(corp_5b, Mapping)
            else wdet_t
        )
        wdet_5b_l = (
            compute_wdet_score(corp_5b, accs_l, b5_map_17b, mode="clamped")
            if isinstance(corp_5b, Mapping)
            else wdet_l
        )
        wdet_kl_t = compute_wdet_score(kl_4b, accs_t, b4_map, mode="clamped") if kl_4b is not None else float("nan")
        wdet_kl_l = compute_wdet_score(kl_4b, accs_l, b4_map, mode="clamped") if kl_4b is not None else float("nan")

        unw_t = float(np.mean([accs_t[b] for b in ("p99.9_plus", "p97_99.9", "p90_97", "p75_90") if accs_t[b] is not None]))
        unw_l = float(np.mean([accs_l[b] for b in ("p99.9_plus", "p97_99.9", "p90_97", "p75_90") if accs_l[b] is not None]))

        ft = fpr_17b.get((nid, PRIMARY_JUDGE_4B), {})
        fl = fpr_17b.get((nid, SECONDARY_JUDGE_4B), {})
        fpr_t = float(ft.get("fpr_ord", float("nan")))
        tpr_t = float(ft.get("tpr_top", float("nan")))
        fpr_l = float(fl.get("fpr_ord", float("nan")))
        tpr_l = float(fl.get("tpr_top", float("nan")))

        per_neuron[nid] = {
            "neuron_id": nid,
            "layer": layer,
            "neuron": neuron,
            "sim_cal_rec": sim_cal,
            "sim_cal_rec_cv": sim_cal_cv,
            "sim_cal_rec_insample": sim_cal_insample,
            "sim_cal_rec_active": sim_act,
            "sim_cal_rec_active_insample": sim_act_insample,
            "strong20_rec": strong20,
            "strong50_rec": strong50,
            "strong20_rec_ord8k": strong20_ord8k,
            "strong50_rec_ord8k": strong50_ord8k,
            "corr": corr_v,
            "sim_raw_rec": sim_raw,
            # B0
            "b0_p999_terra": float(accs_t["p99.9_plus"]) if accs_t["p99.9_plus"] is not None else float("nan"),
            "b0_p999_luna": float(accs_l["p99.9_plus"]) if accs_l["p99.9_plus"] is not None else float("nan"),
            # B1
            "b1_energy_ge_20pct_max": e20,
            "b1_energy_ge_50pct_max": e50,
            "b1_energy_p999": float(corp_4b["p99.9+"]),
            "b1_energy_p97_plus": float(corp_4b["p99.9+"] + corp_4b["p97_99.9"]),
            "b1_one_minus_gate_closed_energy": 1.0 - e_gc if math.isfinite(e_gc) else float("nan"),
            "eval_energy_ge_20pct_max": e20_eval,
            "eval_energy_gate_closed": e_gc_eval,
            "kl_gate_closed": kl_gc,
            # S1
            "s1_unweighted_4band_terra": unw_t,
            "s1_unweighted_4band_luna": unw_l,
            # S2
            "s2_wdet_clamped_terra": wdet_t,
            "s2_wdet_5band_clamped_terra": wdet_5b_t,
            "s2_wdet_signed_terra": wdet_t_signed,
            "s2_wdet_clamped_luna": wdet_l,
            "s2_wdet_5band_clamped_luna": wdet_5b_l,
            "s2_wdet_kl_terra": wdet_kl_t,
            "s2_wdet_kl_luna": wdet_kl_l,
            # S3
            "fpr_ord_terra": fpr_t,
            "tpr_top_terra": tpr_t,
            "s3_one_minus_fpr_ord_terra": float(ft.get("one_minus_fpr_ord", float("nan"))),
            "s3_tpr_minus_fpr_terra": float(ft.get("tpr_minus_fpr", float("nan"))),
            "s3_balanced_acc_yesno_terra": float(ft.get("balanced_acc_yesno", float("nan"))),
            "s3_pyes_margin_gap_terra": float(ft.get("yes_margin_gap", float("nan"))),
            "s3_one_minus_pyes_ord_terra": float(ft.get("one_minus_p_yes_ord", float("nan"))),
            "fpr_ord_luna": fpr_l,
            "tpr_top_luna": tpr_l,
            "s3_one_minus_fpr_ord_luna": float(fl.get("one_minus_fpr_ord", float("nan"))),
            "s3_tpr_minus_fpr_luna": float(fl.get("tpr_minus_fpr", float("nan"))),
            "s3_balanced_acc_yesno_luna": float(fl.get("balanced_acc_yesno", float("nan"))),
            "s3_pyes_margin_gap_luna": float(fl.get("yes_margin_gap", float("nan"))),
            "s3_one_minus_pyes_ord_luna": float(fl.get("one_minus_p_yes_ord", float("nan"))),
            # S4 zero-param
            "s4_wdet_minus_fpr_terra": wdet_t - fpr_t if math.isfinite(fpr_t) else float("nan"),
            "s4_wdet_times_1m_fpr_terra": wdet_t * (1.0 - fpr_t) if math.isfinite(fpr_t) else float("nan"),
            "s4_wdet_minus_fpr_luna": wdet_l - fpr_l if math.isfinite(fpr_l) else float("nan"),
            "s4_wdet_times_1m_fpr_luna": wdet_l * (1.0 - fpr_l) if math.isfinite(fpr_l) else float("nan"),
            # Raw band details
            "corpus_energy_4band": corp_4b,
            "eval_energy_4band": eval_4b,
            "kl_share_4band_norm": kl_4b,
            "band_accs_terra": accs_t,
            "band_accs_luna": accs_l,
        }

    rows_ord = [per_neuron[nid] for nid in nids_sorted]
    y_sim = np.asarray([r["sim_cal_rec"] for r in rows_ord], dtype=np.float64)
    y_sim_cv = np.asarray([r["sim_cal_rec_cv"] for r in rows_ord], dtype=np.float64)
    y_sim_insample = np.asarray([r["sim_cal_rec_insample"] for r in rows_ord], dtype=np.float64)
    y_act = np.asarray([r["sim_cal_rec_active"] for r in rows_ord], dtype=np.float64)
    y_s20 = np.asarray([r["strong20_rec"] for r in rows_ord], dtype=np.float64)
    y_s50 = np.asarray([r["strong50_rec"] for r in rows_ord], dtype=np.float64)
    z_e20 = np.asarray([r["b1_energy_ge_20pct_max"] for r in rows_ord], dtype=np.float64)

    # 4. Structural EnergyShare vs strong20/50 & EvalEnergy vs KLShare on 1.7B
    pooled_eval_ci = (
        {
            **bootstrap_corr_ci(pooled_eval_e, pooled_kl, n_resamples=n_bootstrap, seed=seed + 3),
            "bootstrap_unit": "flattened_neuron_band_pair",
        }
        if len(pooled_eval_e) == len(pooled_kl) and len(pooled_kl) >= 3
        else None
    )
    pooled_corp_ci = (
        {
            **bootstrap_corr_ci(pooled_corp_e, pooled_kl, n_resamples=n_bootstrap, seed=seed + 4),
            "bootstrap_unit": "flattened_neuron_band_pair",
        }
        if len(pooled_kl) >= 3
        else None
    )
    eval_ek_analysis = (
        pooled_energy_kl_analysis(
            np.asarray(ek_rows_eval, dtype=np.float64),
            np.asarray(ek_rows_kl, dtype=np.float64),
            band_names=BANDS_4_KEYS,
            n_bootstrap=n_bootstrap,
            n_permutations=1000,
            seed=seed,
        )
        if len(ek_rows_eval) >= 3
        else None
    )

    structural_17b = {
        "energy_ge_20pct_max_vs_strong20_rec": bootstrap_corr_ci(z_e20, y_s20, n_resamples=n_bootstrap, seed=seed + 1),
        "energy_ge_50pct_max_vs_strong50_rec": bootstrap_corr_ci(
            [r["b1_energy_ge_50pct_max"] for r in rows_ord], y_s50, n_resamples=n_bootstrap, seed=seed + 2
        ),
        "eval_energy_ge_20pct_vs_strong20_ord8k": bootstrap_corr_ci(
            [r["eval_energy_ge_20pct_max"] for r in rows_ord],
            [r["strong20_rec_ord8k"] for r in rows_ord],
            n_resamples=n_bootstrap,
            seed=seed + 5,
        ),
        "pooled_200_eval_energy_vs_kl_share": pooled_eval_ci,
        "pooled_200_corpus_energy_vs_kl_share": pooled_corp_ci,
        "pooled_200_eval_energy_vs_kl_share_detailed": eval_ek_analysis,
        "within_band_eval_energy_vs_kl_share": eval_ek_analysis["within_band"] if eval_ek_analysis else None,
        "permuted_neuron_null_eval_vs_kl": eval_ek_analysis["pooled"]["permuted_neuron_null"] if eval_ek_analysis else None,
        "eval_energy_gate_closed_vs_kl_gate_closed": bootstrap_corr_ci(
            [r["eval_energy_gate_closed"] for r in rows_ord],
            [r["kl_gate_closed"] for r in rows_ord],
            n_resamples=n_bootstrap,
            seed=seed + 6,
        ),
        "mean_kl_share_4band": {
            bk: float(np.mean(band_kl_list[bk])) if band_kl_list[bk] else float("nan")
            for bk in BANDS_4_KEYS
        },
        "mean_corpus_energy_4band": {
            bk: float(np.mean(band_e_corp[bk])) if band_e_corp[bk] else float("nan")
            for bk in BANDS_4_KEYS
        },
        "source_meta": {
            k: ekl_doc.get(k)
            for k in (
                "n_neurons",
                "total_tokens",
                "total_non_bos_tokens",
                "peak_vram_allocated_gb",
                "peak_vram_reserved_gb",
                "zero_ablation_sanity_kl",
                "kl_4band_additivity_ratio_median",
            )
            if k in ekl_doc
        },
    }

    # 5. Pass-rate failure table at p99.9+ >= 0.75 and >= 0.80
    pass_rate_audit: dict[str, Any] = {}
    det_p999_t = np.asarray([r["b0_p999_terra"] for r in rows_ord], dtype=np.float64)
    for thr_lbl, thr in (("p99.9+ >= 0.75", 0.75), ("p99.9+ >= 0.80", 0.80), ("all_50", -1.0)):
        m = np.ones(len(rows_ord), dtype=bool) if thr < 0 else (det_p999_t >= thr)
        n_m = int(np.sum(m))
        pass_rate_audit[thr_lbl] = {
            "n_passing_det": n_m,
            "share_of_50": n_m / max(len(rows_ord), 1),
            "sim_cal_rec_ge_0_50": int(np.sum(m & (y_sim >= 0.50))),
            "sim_cal_rec_ge_0_25": int(np.sum(m & (y_sim >= 0.25))),
            "sim_cal_rec_gt_0_00": int(np.sum(m & (y_sim > 0.0))),
            "sim_cal_rec_le_0_00": int(np.sum(m & (y_sim <= 0.0))),
            "sim_cal_rec_insample_ge_0_25": int(np.sum(m & (y_sim_insample >= 0.25))),
            "sim_cal_rec_cv_ge_0_50": int(np.sum(m & (y_sim_cv >= 0.50))),
            "sim_cal_rec_cv_ge_0_25": int(np.sum(m & (y_sim_cv >= 0.25))),
            "sim_cal_rec_active_ge_0_50": int(np.sum(m & (y_act >= 0.50))),
            "strong20_rec_ge_0_50": int(np.sum(m & (y_s20 >= 0.50))),
        }

    # 6. Zero-shot Out-of-Model Transfer from Qwen3-4B (n=278) -> Qwen3-1.7B (n=50)
    m4b_doc: dict[str, Any] = (
        json.loads(surrogate_4b_metrics_path.read_text(encoding="utf-8"))
        if surrogate_4b_metrics_path.is_file()
        else {}
    )
    transfer_results: dict[str, Any] = {}
    if "per_neuron" in m4b_doc:
        pn4 = m4b_doc["per_neuron"]
        ks4 = sorted(int(k) for k in pn4.keys())
        rows4 = [pn4[str(k)] for k in ks4]
        lays4 = [int(r["layer"]) for r in rows4]
        y4_sim = np.asarray([float(r["sim_cal_rec"]) for r in rows4], dtype=np.float64)
        y4_act = np.asarray([float(r["sim_cal_rec_active"]) for r in rows4], dtype=np.float64)

        transfer_specs = [
            (
                "transfer_4b_to_17b_ridge_sim_terra",
                ["s2_wdet_clamped_terra", "fpr_ord_terra", "tpr_top_terra", "b1_energy_ge_20pct_max"],
                y4_sim,
                False,
            ),
            (
                "transfer_4b_to_17b_rank_ridge_sim_terra",
                ["s2_wdet_clamped_terra", "fpr_ord_terra", "tpr_top_terra", "b1_energy_ge_20pct_max"],
                y4_sim,
                True,
            ),
            (
                "transfer_4b_to_17b_rank_ridge_sim_luna",
                ["s2_wdet_clamped_luna", "fpr_ord_luna", "tpr_top_luna", "b1_energy_ge_20pct_max"],
                y4_sim,
                True,
            ),
            (
                "transfer_4b_to_17b_ridge_active_terra",
                ["s3_one_minus_pyes_ord_terra", "s3_pyes_margin_gap_terra", "b0_p999_terra", "b1_energy_ge_20pct_max"],
                y4_act,
                False,
            ),
            (
                "transfer_4b_to_17b_rank_ridge_active_terra",
                ["s3_one_minus_pyes_ord_terra", "s3_pyes_margin_gap_terra", "b0_p999_terra", "b1_energy_ge_20pct_max"],
                y4_act,
                True,
            ),
            (
                "transfer_4b_to_17b_rank_ridge_active_luna",
                ["s3_one_minus_pyes_ord_luna", "s3_pyes_margin_gap_luna", "b0_p999_luna", "b1_energy_ge_20pct_max"],
                y4_act,
                True,
            ),
        ]
        for tname, feats, y4_tgt, rmode in transfer_specs:
            X4 = np.column_stack([np.asarray([float(r[f]) for r in rows4], dtype=np.float64) for f in feats])
            X17 = np.column_stack([np.asarray([float(r[f]) for r in rows_ord], dtype=np.float64) for f in feats])
            tr_out = fit_4b_transfer_and_predict_17b(
                X_train_4b=X4,
                y_train_4b=y4_tgt,
                layers_4b=lays4,
                X_eval_17b=X17,
                rank_mode=rmode,
                seed=seed,
            )
            preds = tr_out["preds_17b"]
            for i, r in enumerate(rows_ord):
                r[tname] = float(preds[i])
            ev = _eval_17b_candidate(
                name=tname,
                scores=preds,
                y_sim=y_sim,
                y_sim_insample=y_sim_insample,
                y_active=y_act,
                z_e20=z_e20,
                n_bootstrap=n_bootstrap,
                seed=seed + 500 + len(transfer_results) * 19,
            )
            ev["transfer_fit_4b"] = {
                "features": feats,
                "rank_mode": rmode,
                "best_alpha_4b_cv": tr_out["best_alpha_4b_cv"],
                "std_weights_4b": dict(zip(feats, tr_out["std_weights_4b"])),
            }
            transfer_results[tname] = ev

    # 7. Benchmark ALL candidates on Qwen3-1.7B (n = 50)
    cand_keys = (
        "b0_p999_terra",
        "b0_p999_luna",
        "b1_energy_ge_20pct_max",
        "b1_energy_ge_50pct_max",
        "b1_energy_p999",
        "b1_energy_p97_plus",
        "b1_one_minus_gate_closed_energy",
        "s1_unweighted_4band_terra",
        "s1_unweighted_4band_luna",
        "s2_wdet_clamped_terra",
        "s2_wdet_5band_clamped_terra",
        "s2_wdet_signed_terra",
        "s2_wdet_clamped_luna",
        "s2_wdet_5band_clamped_luna",
        "s2_wdet_kl_terra",
        "s2_wdet_kl_luna",
        "s3_one_minus_fpr_ord_terra",
        "s3_tpr_minus_fpr_terra",
        "s3_balanced_acc_yesno_terra",
        "s3_pyes_margin_gap_terra",
        "s3_one_minus_pyes_ord_terra",
        "s3_one_minus_fpr_ord_luna",
        "s3_tpr_minus_fpr_luna",
        "s3_balanced_acc_yesno_luna",
        "s3_pyes_margin_gap_luna",
        "s3_one_minus_pyes_ord_luna",
        "s4_wdet_minus_fpr_terra",
        "s4_wdet_times_1m_fpr_terra",
        "s4_wdet_minus_fpr_luna",
        "s4_wdet_times_1m_fpr_luna",
    )
    benchmarks_17b: dict[str, dict[str, Any]] = {}
    for idx_c, ck in enumerate(cand_keys):
        benchmarks_17b[ck] = _eval_17b_candidate(
            name=ck,
            scores=[r[ck] for r in rows_ord],
            y_sim=y_sim,
            y_sim_insample=y_sim_insample,
            y_active=y_act,
            z_e20=z_e20,
            n_bootstrap=n_bootstrap,
            seed=seed + 100 * (idx_c + 1),
        )
    benchmarks_17b.update(transfer_results)

    # 8. Build concise cross-model side-by-side comparison table (4B n=278 vs 1.7B n=50)
    b4_benches = m4b_doc.get("benchmarks", {})
    b4_gap = m4b_doc.get("gap_decomposition_summary", {}).get("correlations", {})
    kl4_doc = json.loads(band_kl_4b_path.read_text(encoding="utf-8")) if band_kl_4b_path.is_file() else {}

    cross_model_table: list[dict[str, Any]] = [
        {
            "metric": "EnergyShare(>=20% max) vs strong20_rec",
            "qwen4b": b4_gap.get("energy_ge_20pct_max_vs_strong20_rec"),
            "qwen17b": structural_17b["energy_ge_20pct_max_vs_strong20_rec"],
        },
        {
            "metric": "EvalEnergy vs true fp32 KLShare (4-band pooled)",
            "qwen4b": kl4_doc.get("correlations", {}).get("pooled_240_eval_energy_vs_kl_norm"),
            "qwen17b": structural_17b["pooled_200_eval_energy_vs_kl_share"],
        },
        {
            "metric": "p99.9+ Det (terra) vs sim_cal_rec",
            "qwen4b": b4_benches.get("b0_p999_terra", {}).get("sim_cal_rec"),
            "qwen17b": benchmarks_17b["b0_p999_terra"]["sim_cal_rec"],
        },
        {
            "metric": "p99.9+ Det (terra) vs sim_cal_rec_active",
            "qwen4b": b4_benches.get("b0_p999_terra", {}).get("sim_cal_rec_active"),
            "qwen17b": benchmarks_17b["b0_p999_terra"]["sim_cal_rec_active"],
        },
        {
            "metric": "Zero-label EnergyShare(>=20%) vs sim_cal_rec",
            "qwen4b": b4_benches.get("b1_energy_ge_20pct_max", {}).get("sim_cal_rec"),
            "qwen17b": benchmarks_17b["b1_energy_ge_20pct_max"]["sim_cal_rec"],
        },
        {
            "metric": "WDet clamped (terra) vs sim_cal_rec",
            "qwen4b": b4_benches.get("s2_wdet_clamped_terra", {}).get("sim_cal_rec"),
            "qwen17b": benchmarks_17b["s2_wdet_clamped_terra"]["sim_cal_rec"],
        },
        {
            "metric": "WDet clamped (terra) partial | E(>=20%) on sim_cal_rec",
            "qwen4b": b4_benches.get("s2_wdet_clamped_terra", {}).get("partial_above_energy_ge_20pct_sim_cal_rec"),
            "qwen17b": benchmarks_17b["s2_wdet_clamped_terra"]["partial_above_e20_sim_cal_rec"],
        },
        {
            "metric": "1 - P(YES|ord) (terra) vs sim_cal_rec_active",
            "qwen4b": b4_benches.get("s3_one_minus_pyes_ord_terra", {}).get("sim_cal_rec_active"),
            "qwen17b": benchmarks_17b["s3_one_minus_pyes_ord_terra"]["sim_cal_rec_active"],
        },
        {
            "metric": "1 - P(YES|ord) (terra) partial | E(>=20%) on active",
            "qwen4b": b4_benches.get("s3_one_minus_pyes_ord_terra", {}).get("partial_above_energy_ge_20pct_sim_cal_rec_active"),
            "qwen17b": benchmarks_17b["s3_one_minus_pyes_ord_terra"]["partial_above_e20_sim_cal_rec_active"],
        },
        {
            "metric": "yes_margin_gap (luna) vs sim_cal_rec_active",
            "qwen4b": b4_benches.get("s3_pyes_margin_gap_luna", {}).get("sim_cal_rec_active"),
            "qwen17b": benchmarks_17b["s3_pyes_margin_gap_luna"]["sim_cal_rec_active"],
        },
        {
            "metric": "yes_margin_gap (luna) partial | E(>=20%) on active",
            "qwen4b": b4_benches.get("s3_pyes_margin_gap_luna", {}).get("partial_above_energy_ge_20pct_sim_cal_rec_active"),
            "qwen17b": benchmarks_17b["s3_pyes_margin_gap_luna"]["partial_above_e20_sim_cal_rec_active"],
        },
    ]

    metrics_doc: dict[str, Any] = {
        "model_id": "Qwen/Qwen3-1.7B",
        "n_neurons": len(rows_ord),
        "n_poles": len(poles),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "causal_17b_spend": {
            "sim_calls": len(sim_rows),
            "sim_spend_usd": round(sim_spend_usd, 6),
            "sim_parse_status": dict(sim_parse_counts),
            "sim_reasoning_tokens": sim_reas_toks,
            "fpr_calls": len(fpr_rows),
            "fpr_spend_usd": round(fpr_spend_usd, 6),
            "fpr_reasoning_tokens": fpr_reas_toks,
            "combined_spend_usd": round(sim_spend_usd + fpr_spend_usd, 6),
            "budget_cap_usd": 4.50,
            "within_budget": bool((sim_spend_usd + fpr_spend_usd) <= 4.50),
        },
        "causal_distribution_17b": {
            "median_sim_cal_rec": float(np.nanmedian(y_sim)),
            "mean_sim_cal_rec": float(np.nanmean(y_sim)),
            "median_sim_cal_rec_cv": float(np.nanmedian(y_sim_cv)),
            "median_sim_cal_rec_insample": float(np.nanmedian(y_sim_insample)),
            "median_sim_cal_rec_active": float(np.nanmedian(y_act)),
            "mean_sim_cal_rec_active": float(np.nanmean(y_act)),
            "median_strong20_rec": float(np.nanmedian(y_s20)),
            "median_energy_ge_20pct_max": float(np.nanmedian(z_e20)),
        },
        "detection_audit_17b": det_audit,
        "fpr_audit_17b": fpr_audit,
        "structural_checks_17b": structural_17b,
        "pass_rate_audit_17b": pass_rate_audit,
        "benchmarks_17b": benchmarks_17b,
        "cross_model_comparison_4b_vs_17b": cross_model_table,
        "per_neuron": per_neuron,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = render_17b_causal_summary_md(metrics_doc)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_md, encoding="utf-8")

    plot_cross_model_17b_vs_4b_figure(
        metrics_17b=metrics_doc,
        metrics_4b=m4b_doc,
        output_path=figure_path,
    )

    return metrics_doc


# ---------------------------------------------------------------------------
# Strictly Factual Summary Markdown
# ---------------------------------------------------------------------------


def _fmt_stat_entry(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "—"
    if "partial_spearman_rho" in entry:
        return _fmt_ci(entry.get("partial_spearman_rho"), entry.get("ci_low"), entry.get("ci_high"), signed=True)
    return _fmt_ci(entry.get("spearman_rho"), entry.get("spearman_ci_low"), entry.get("spearman_ci_high"))


def render_17b_causal_summary_md(metrics_doc: Mapping[str, Any]) -> str:
    """Render strictly factual cross-model comparison markdown (zero hype)."""
    lines: list[str] = []
    sp = metrics_doc["causal_17b_spend"]
    dist = metrics_doc["causal_distribution_17b"]
    g1 = metrics_doc["structural_checks_17b"]
    b17 = metrics_doc["benchmarks_17b"]

    lines.append("# Cross-Model Causal Comparison (`Qwen3-1.7B` `n = 50` vs `Qwen3-4B` `n = 278`)")
    lines.append("")
    lines.append(
        f"- 1.7B causal & precision API spend: **`${sp['combined_spend_usd']:.4f}`** "
        f"(simulations: `${sp['sim_spend_usd']:.4f}` across `{sp['sim_calls']:,}` calls, "
        f"parse status `{sp['sim_parse_status']}`, `{sp['sim_reasoning_tokens']}` reas tok; "
        f"FPR YES/NO: `${sp['fpr_spend_usd']:.4f}` across `{sp['fpr_calls']:,}` calls, "
        f"`{sp['fpr_reasoning_tokens']}` reas tok; cap `$4.50`)."
    )
    lines.append(
        f"- `Qwen3-1.7B` causal medians (`n = {metrics_doc['n_neurons']}`): "
        f"`sim_cal_rec = {dist['median_sim_cal_rec']:.3f}` (`2-fold CV = {dist['median_sim_cal_rec_cv']:.3f}`), "
        f"`sim_cal_rec_active = {dist['median_sim_cal_rec_active']:.3f}`, "
        f"`strong20_rec = {dist['median_strong20_rec']:.3f}`, "
        f"`EnergyShare(>=20%) = {dist['median_energy_ge_20pct_max']:.3f}`."
    )
    lines.append("")

    # Section 1: Cross-model side-by-side headline table
    lines.append("## 1. Cross-Model Side-by-Side Comparison (`Qwen3-4B n=278` vs `Qwen3-1.7B n=50`)")
    lines.append("")
    lines.append("| Metric | `Qwen3-4B` (`n = 278`, Ulto85 MIT) ρ [95% CI] | `Qwen3-1.7B` (`n = 50`) ρ [95% CI] |")
    lines.append("|---|---:|---:|")
    for row in metrics_doc["cross_model_comparison_4b_vs_17b"]:
        lines.append(f"| {row['metric']} | {_fmt_stat_entry(row.get('qwen4b'))} | {_fmt_stat_entry(row.get('qwen17b'))} |")
    lines.append("")

    # Section 2: Pass-rate audit on 1.7B
    lines.append("## 2. `Qwen3-1.7B` Top-Band Detection Pass Rates vs Causal Recovery (`n = 50`)")
    lines.append("")
    lines.append("| Subset | n | `sim >= 0.50` | `sim >= 0.25` | `sim > 0` | `sim <= 0` | `sim_cv >= 0.50` | `sim_cv >= 0.25` | `active >= 0.50` | `strong20 >= 0.50` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k_sub, pr in metrics_doc["pass_rate_audit_17b"].items():
        lines.append(
            f"| `{k_sub}` | {pr['n_passing_det']} | {pr['sim_cal_rec_ge_0_50']} | {pr['sim_cal_rec_ge_0_25']} | "
            f"{pr['sim_cal_rec_gt_0_00']} | {pr['sim_cal_rec_le_0_00']} | {pr['sim_cal_rec_cv_ge_0_50']} | "
            f"{pr['sim_cal_rec_cv_ge_0_25']} | {pr['sim_cal_rec_active_ge_0_50']} | {pr['strong20_rec_ge_0_50']} |"
        )
    lines.append("")

    # Section 3: Exact fp32 band KL validation on 1.7B
    lines.append("## 3. `Qwen3-1.7B` Exact `fp32` Band KL & Energy Validation (`n = 50`)")
    lines.append("")
    mk = g1["mean_kl_share_4band"]
    me = g1["mean_corpus_energy_4band"]
    lines.append(
        f"- Mean 4-band true `KLShare` (`p99.9+ / p97_99.9 / p75_97 / lt_p75`): "
        f"`{_fmt_num(mk['p99.9+'])} / {_fmt_num(mk['p97_99.9'])} / {_fmt_num(mk['p75_97'])} / {_fmt_num(mk['lt_p75'])}`."
    )
    lines.append(
        f"- Mean 4-band corpus `EnergyShare`: "
        f"`{_fmt_num(me['p99.9+'])} / {_fmt_num(me['p97_99.9'])} / {_fmt_num(me['p75_97'])} / {_fmt_num(me['lt_p75'])}`."
    )
    lines.append(
        f"- `EvalEnergy(>=20%)` vs `strong20_rec` (8k slice): {_fmt_stat_entry(g1.get('eval_energy_ge_20pct_vs_strong20_ord8k'))}; "
        f"`EvalEnergy(gate_closed)` vs `KLShare(gate_closed)`: {_fmt_stat_entry(g1.get('eval_energy_gate_closed_vs_kl_gate_closed'))}."
    )
    det_ek = g1.get("pooled_200_eval_energy_vs_kl_share_detailed")
    if isinstance(det_ek, Mapping) and "pooled" in det_ek and "within_band" in det_ek:
        pl = det_ek["pooled"]
        wb = det_ek["within_band"]
        cl_ci = pl["neuron_clustered_bootstrap_ci"]
        pr_ci = pl["pair_level_bootstrap_ci"]
        pnull = pl["permuted_neuron_null"]
        wb_txt = " / ".join(f"{bk}: {_fmt_num(wb[bk].get('spearman_rho'))}" for bk in BANDS_4_KEYS if bk in wb)
        lines.append(
            f"- Pooled `EvalEnergy` vs `KLShare` (second-order sanity check, {det_ek.get('n_pairs', pl.get('n_pairs', 0))} neuron-band pairs): "
            f"`rho = {_fmt_num(pl.get('spearman_rho'))}` "
            f"(neuron-clustered 95% CI `[{_fmt_num(cl_ci.get('spearman_ci_low'))}, {_fmt_num(cl_ci.get('spearman_ci_high'))}]`; "
            f"pair-level 95% CI `[{_fmt_num(pr_ci.get('spearman_ci_low'))}, {_fmt_num(pr_ci.get('spearman_ci_high'))}]`; "
            f"{pnull['n_permutations']}-perm neuron-shuffled null `[q025={_fmt_num(pnull.get('spearman_q025'))}, q975={_fmt_num(pnull.get('spearman_q975'))}]`, empirical p={_fmt_num(pnull.get('spearman_p_empirical_ge_obs'), 4)})."
        )
        lines.append(f"- Within-band `EvalEnergy` vs `KLShare` (`rho`): {wb_txt}.")
    lines.append("")

    # Section 4: Full Surrogate & Zero-Shot 4B->1.7B Transfer Table
    lines.append("## 4. Full `Qwen3-1.7B` Surrogate & Zero-Shot `4B -> 1.7B` Transfer Benchmark (`n = 50`)")
    lines.append("")
    lines.append("| Candidate | `sim_cal_rec` (CV) ρ [95% CI] | `sim_insamp` ρ | Partial ρ \\| `E(>=20%)` (`sim`) | `active` ρ [95% CI] | Partial ρ \\| `E(>=20%)` (`active`) | AUC `>0` | P@10 `>0` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ck, ev in b17.items():
        cs = ev["sim_cal_rec"]
        ci = ev["sim_cal_rec_insample"]
        ps = ev["partial_above_e20_sim_cal_rec"]
        ca = ev["sim_cal_rec_active"]
        pa = ev["partial_above_e20_sim_cal_rec_active"]
        clf = ev["classification_sim_cal_rec"]
        lines.append(
            f"| `{ck}` | {_fmt_stat_entry(cs)} | {_fmt_num(ci.get('spearman_rho'))} | "
            f"{_fmt_stat_entry(ps)} | {_fmt_stat_entry(ca)} | {_fmt_stat_entry(pa)} | "
            f"{_fmt_num(clf['pass_gt_0_00']['auroc'])} | "
            f"{_fmt_num(clf['pass_gt_0_00']['precision_at_10'])} |"
        )
    lines.append("")
    lines.append(f"*{CROSS_MODEL_CREDIT_FOOTER}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clean 250-DPI Cross-Model Figure (Zero Interpretive Prose)
# ---------------------------------------------------------------------------


def plot_cross_model_17b_vs_4b_figure(
    metrics_17b: Mapping[str, Any],
    metrics_4b: Mapping[str, Any],
    output_path: Path = DEFAULT_FIGURE_CROSS_MODEL_PATH,
) -> None:
    """Plot 2-panel cross-model comparison (Panel A: 4B vs 1.7B bars; Panel B: 1.7B det vs causal scatter)."""
    assert_17b_causal_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(14.6, 6.4), dpi=250)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.05], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    b4 = metrics_4b.get("benchmarks", {})
    b17 = metrics_17b["benchmarks_17b"]

    # ---- Panel A: Grouped Spearman rho [95% CI] bars (Qwen3-4B n=278 vs Qwen3-1.7B n=50) ----
    rows_spec: list[tuple[str, str, str]] = [
        ("sim_cal_rec: p99.9+ Det (terra)", "b0_p999_terra", "sim_cal_rec"),
        ("sim_cal_rec: Energy >=20% (0-label)", "b1_energy_ge_20pct_max", "sim_cal_rec"),
        ("sim_cal_rec: WDet clamped (terra)", "s2_wdet_clamped_terra", "sim_cal_rec"),
        ("sim_cal_rec: 1 - P(YES|ord) (terra)", "s3_one_minus_pyes_ord_terra", "sim_cal_rec"),
        ("active: p99.9+ Det (terra)", "b0_p999_terra", "sim_cal_rec_active"),
        ("active: Energy >=20% (0-label)", "b1_energy_ge_20pct_max", "sim_cal_rec_active"),
        ("active: WDet clamped (terra)", "s2_wdet_clamped_terra", "sim_cal_rec_active"),
        ("active: 1 - P(YES|ord) (terra)", "s3_one_minus_pyes_ord_terra", "sim_cal_rec_active"),
    ]

    def _get_rho_ci(bench_dict: Mapping[str, Any], key: str, tgt: str) -> tuple[float, float, float]:
        sub = bench_dict.get(key, {}).get(tgt, {})
        rho = float(sub.get("spearman_rho", float("nan")))
        lo = float(sub.get("spearman_ci_low", float("nan")))
        hi = float(sub.get("spearman_ci_high", float("nan")))
        err_lo = max(0.0, rho - lo) if math.isfinite(rho) and math.isfinite(lo) else 0.0
        err_hi = max(0.0, hi - rho) if math.isfinite(rho) and math.isfinite(hi) else 0.0
        return rho, err_lo, err_hi

    n_rows = len(rows_spec)
    # Add visual gap between sim_cal_rec block (top 4) and sim_cal_rec_active block (bottom 4)
    y_centers = np.array([8.5, 7.5, 6.5, 5.5, 3.8, 2.8, 1.8, 0.8], dtype=np.float64)
    bh = 0.34

    vals_4b = [_get_rho_ci(b4, k, t) for _, k, t in rows_spec]
    vals_17b = [_get_rho_ci(b17, k, t) for _, k, t in rows_spec]

    r4 = [v[0] for v in vals_4b]
    lo4 = [v[1] for v in vals_4b]
    hi4 = [v[2] for v in vals_4b]
    r17 = [v[0] for v in vals_17b]
    lo17 = [v[1] for v in vals_17b]
    hi17 = [v[2] for v in vals_17b]

    ax_a.barh(
        y_centers + bh * 0.52,
        [x if math.isfinite(x) else 0.0 for x in r4],
        height=bh,
        color="#2a6f97",
        edgecolor="white",
        linewidth=0.6,
        label="Qwen3-4B (n = 278, Ulto85 MIT)",
        zorder=3,
    )
    ax_a.errorbar(
        [x if math.isfinite(x) else 0.0 for x in r4],
        y_centers + bh * 0.52,
        xerr=[lo4, hi4],
        fmt="none",
        ecolor="#1a1a1a",
        elinewidth=1.0,
        capsize=2.2,
        zorder=4,
    )
    ax_a.barh(
        y_centers - bh * 0.52,
        [x if math.isfinite(x) else 0.0 for x in r17],
        height=bh,
        color="#e76f51",
        edgecolor="white",
        linewidth=0.6,
        label=f"Qwen3-1.7B (n = {metrics_17b['n_neurons']})",
        zorder=3,
    )
    ax_a.errorbar(
        [x if math.isfinite(x) else 0.0 for x in r17],
        y_centers - bh * 0.52,
        xerr=[lo17, hi17],
        fmt="none",
        ecolor="#1a1a1a",
        elinewidth=1.0,
        capsize=2.2,
        zorder=4,
    )

    ax_a.axvline(0.0, color="#222222", linewidth=0.8, zorder=2)
    ax_a.axhline(4.65, color="#888888", linestyle="--", linewidth=0.85, alpha=0.75, zorder=2)

    all_hi = [r + h for r, h in zip(r4 + r17, hi4 + hi17) if math.isfinite(r)]
    all_lo = [r - l for r, l in zip(r4 + r17, lo4 + lo17) if math.isfinite(r)]
    x_lo = min(-0.22, (min(all_lo) - 0.08) if all_lo else -0.22)
    x_hi = max(0.88, (max(all_hi) + 0.18) if all_hi else 0.88)
    ax_a.set_xlim(x_lo, x_hi)

    for yc, rv4, hv4, rv17, hv17 in zip(y_centers, r4, hi4, r17, hi17):
        if math.isfinite(rv4):
            ax_a.text(
                max(rv4 + hv4, 0.0) + 0.015,
                yc + bh * 0.52,
                f"{rv4:+.2f}",
                va="center",
                ha="left",
                fontsize=7.4,
                color="#1b4965",
            )
        if math.isfinite(rv17):
            ax_a.text(
                max(rv17 + hv17, 0.0) + 0.015,
                yc - bh * 0.52,
                f"{rv17:+.2f}",
                va="center",
                ha="left",
                fontsize=7.4,
                color="#9d3b22",
            )

    ax_a.set_yticks(y_centers)
    ax_a.set_yticklabels([lbl for lbl, _, _ in rows_spec], fontsize=8.2)
    ax_a.set_ylim(0.15, 9.25)
    ax_a.set_xlabel("Spearman ρ (95% neuron bootstrap CI)")
    ax_a.set_title("A. Cross-model rank correlation: Qwen3-4B vs. Qwen3-1.7B", fontsize=9.8, fontweight="bold", pad=8)
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.14),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    ax_a.grid(axis="x", linestyle=":", alpha=0.45)
    ax_a.set_axisbelow(True)

    # ---- Panel B: Qwen3-1.7B (n=50) scatter of p99.9+ detection vs sim_cal_rec & sim_cal_rec_active ----
    per_n = metrics_17b["per_neuron"]
    nids = sorted(per_n.keys())
    x_det = np.asarray([float(per_n[nid]["b0_p999_terra"]) for nid in nids], dtype=np.float64)
    y_sim = np.asarray([float(per_n[nid]["sim_cal_rec"]) for nid in nids], dtype=np.float64)
    y_act = np.asarray([float(per_n[nid]["sim_cal_rec_active"]) for nid in nids], dtype=np.float64)

    y_lo_clip = -0.80
    y_hi_clip = 1.08

    # Shade p99.9+ >= 0.80 pass region
    ax_b.axvspan(0.80, 1.04, color="#f1f5f9", alpha=0.8, zorder=0)
    ax_b.axvline(0.80, color="#64748b", linestyle="--", linewidth=0.9, zorder=1)
    ax_b.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.55, zorder=1)
    ax_b.axhline(0.50, color="#666666", linestyle=":", linewidth=0.9, alpha=0.65, zorder=1)

    sim_clip_lo = y_sim < y_lo_clip
    act_clip_lo = y_act < y_lo_clip

    ax_b.scatter(
        x_det[~sim_clip_lo],
        np.clip(y_sim[~sim_clip_lo], y_lo_clip, y_hi_clip),
        c="#1d3557",
        marker="o",
        s=36,
        alpha=0.84,
        edgecolors="white",
        linewidths=0.5,
        label="sim_cal_rec (full distribution)",
        zorder=4,
    )
    if np.any(sim_clip_lo):
        ax_b.scatter(
            x_det[sim_clip_lo],
            np.full(int(np.sum(sim_clip_lo)), y_lo_clip + 0.025),
            c="#1d3557",
            marker="v",
            s=40,
            alpha=0.92,
            edgecolors="#111111",
            linewidths=0.6,
            zorder=5,
        )

    ax_b.scatter(
        x_det[~act_clip_lo],
        np.clip(y_act[~act_clip_lo], y_lo_clip, y_hi_clip),
        c="#e76f51",
        marker="^",
        s=34,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.45,
        label="sim_cal_rec_active (active windows)",
        zorder=3,
    )
    if np.any(act_clip_lo):
        ax_b.scatter(
            x_det[act_clip_lo],
            np.full(int(np.sum(act_clip_lo)), y_lo_clip + 0.055),
            c="#e76f51",
            marker="v",
            s=38,
            alpha=0.9,
            edgecolors="#111111",
            linewidths=0.6,
            zorder=5,
        )

    c_sim = b17["b0_p999_terra"]["sim_cal_rec"]
    c_act = b17["b0_p999_terra"]["sim_cal_rec_active"]
    pr80 = metrics_17b["pass_rate_audit_17b"]["p99.9+ >= 0.80"]
    n_clip = int(np.sum(sim_clip_lo | act_clip_lo))
    box_txt = (
        f"p99.9+ vs sim_cal_rec: ρ = {_fmt_num(c_sim['spearman_rho'])}\n"
        f"p99.9+ vs active: ρ = {_fmt_num(c_act['spearman_rho'])}\n"
        f"At p99.9+ >= 0.80 (n={pr80['n_passing_det']}/50):\n"
        f"  sim >= 0.50: {pr80['sim_cal_rec_ge_0_50']}/{pr80['n_passing_det']} | "
        f"sim <= 0: {pr80['sim_cal_rec_le_0_00']}/{pr80['n_passing_det']}"
    )
    if n_clip > 0:
        box_txt += f"\nClipped at y={y_lo_clip:.2f}: {n_clip} pts"
    ax_b.text(
        0.03,
        0.97,
        box_txt,
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95),
        zorder=6,
    )

    x_min = min(0.35, float(np.nanmin(x_det)) - 0.04) if len(x_det) else 0.35
    ax_b.set_xlim(x_min, 1.04)
    ax_b.set_ylim(y_lo_clip, y_hi_clip)
    ax_b.set_xlabel("Qwen3-1.7B p99.9+ own-label detection (gpt-6.1-terra@none)")
    ax_b.set_ylabel("Exact fp32 causal recovery (KL damage avoided)")
    ax_b.set_title(
        f"B. Qwen3-1.7B (n = {metrics_17b['n_neurons']}): p99.9+ detection vs. causal recovery",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax_b.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=7.9)
    ax_b.grid(True, linestyle=":", alpha=0.45)
    ax_b.set_axisbelow(True)

    fig.text(
        0.5,
        0.012,
        CROSS_MODEL_CREDIT_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#444444",
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.89, bottom=0.20)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-1.7B causal evaluation & cross-model comparison vs Qwen3-4B."
    )
    parser.add_argument("--poles", type=Path, default=DEFAULT_POLES_17B_PATH)
    parser.add_argument("--judgements-17b", type=Path, default=DEFAULT_JUDGEMENTS_17B_PATH)
    parser.add_argument("--energy-kl-17b", type=Path, default=DEFAULT_ENERGY_KL_17B_PATH)
    parser.add_argument("--causal-scores-17b", type=Path, default=DEFAULT_CAUSAL_SCORES_17B_PATH)
    parser.add_argument("--sim-calls-17b", type=Path, default=DEFAULT_SIM_CALLS_17B_PATH)
    parser.add_argument("--fpr-17b", type=Path, default=DEFAULT_FPR_17B_PATH)
    parser.add_argument("--surrogate-4b-metrics", type=Path, default=DEFAULT_SURROGATE_METRICS_PATH)
    parser.add_argument("--band-kl-4b", type=Path, default=DEFAULT_BAND_KL_PATH)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_METRICS_17B_CAUSAL_PATH)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_17B_CAUSAL_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_CROSS_MODEL_PATH)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    doc = analyze_17b_causal(
        poles_path=args.poles,
        judgements_17b_path=args.judgements_17b,
        energy_kl_17b_path=args.energy_kl_17b,
        causal_scores_17b_path=args.causal_scores_17b,
        sim_calls_17b_path=args.sim_calls_17b,
        fpr_17b_path=args.fpr_17b,
        surrogate_4b_metrics_path=args.surrogate_4b_metrics,
        band_kl_4b_path=args.band_kl_4b,
        output_metrics_path=args.output_metrics,
        output_summary_path=args.output_summary,
        figure_path=args.figure,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "n_neurons": doc["n_neurons"],
                "causal_17b_spend": doc["causal_17b_spend"],
                "causal_distribution_17b": doc["causal_distribution_17b"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
