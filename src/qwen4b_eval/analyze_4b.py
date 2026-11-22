"""Statistical audit and figures for exploratory Qwen3-4B (Ulto85) evaluation.

Strictly isolated from Qwen3-1.7B Part 1 outputs:
- Reads inputs from `data/qwen4b/` and `results/qwen4b/`
- Writes strictly to:
  - `results/qwen4b/metrics_4b.json`
  - `results/qwen4b/summary_4b.md`
  - `figures/qwen4b_detection_vs_causal.png`
  - `figures/qwen4b_top100_two_panel.png`
  - `figures/qwen4b_detection_vs_strong20.png`
  - `figures/qwen4b_main_plain.png`

All tables, labels, and figure titles are strictly factual (zero interpretive prose).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import pearsonr, spearmanr

from qwen4b_eval.judge_4b import (
    DEFAULT_ACTIVE_POLES_PATH,
    DEFAULT_JUDGEMENTS_4B_PATH,
    DEFAULT_JUDGEMENTS_4B_TOP100_PATH,
    DEFAULT_SNIPPETS_4B_PATH,
    DEFAULT_SNIPPETS_4B_TOP100_PATH,
    DEFAULT_WRONG_DONORS_4B_PATH,
    load_active_poles,
    resolve_pole_sign,
)
from w_audit.analyze import (
    DEFAULT_BOOTSTRAP_B,
    classify_row_outcome,
    cohens_kappa,
    neuron_clustered_bootstrap_accuracy,
    neuron_clustered_bootstrap_paired_diff,
)
from w_audit.config import SEED
from w_audit.labeler import load_jsonl

DEFAULT_SCORES_PATH: Path = Path("data/qwen4b/ulto85/e3_scores.json")
DEFAULT_OPUS_SCORES_PATH: Path = Path("data/qwen4b/ulto85/e3_scores_opus.json")
DEFAULT_SPARSITY_DIAG_PATH: Path = Path("results/qwen4b/pole_sparsity_diagnostics.json")
DEFAULT_METRICS_4B_PATH: Path = Path("results/qwen4b/metrics_4b.json")
DEFAULT_SUMMARY_4B_PATH: Path = Path("results/qwen4b/summary_4b.md")
DEFAULT_FIGURE_4B_PATH: Path = Path("figures/qwen4b_detection_vs_causal.png")
DEFAULT_FIGURE_4B_TOP100_TWO_PANEL_PATH: Path = Path("figures/qwen4b_top100_two_panel.png")
DEFAULT_FIGURE_4B_STRONG20_PATH: Path = Path("figures/qwen4b_detection_vs_strong20.png")
DEFAULT_FIGURE_4B_MAIN_PLAIN_PATH: Path = Path("figures/qwen4b_main_plain.png")

ULTO85_CREDIT_FOOTER: str = (
    "Neuron labels & causal recovery scores: Ulto85/neurons-kinda-clean (MIT, Qwen3-4B)"
)

PRIMARY_JUDGE_4B: str = "gpt-6.1-terra@none"
SECONDARY_JUDGE_4B: str = "gpt-6-luna@none"

NAMED_NEURONS: tuple[tuple[str, int, int, int], ...] = (
    ("info", 26, 0, 9209),
    ("PCR", 23, 0, 8146),
    ("insurance", 236, 32, 4362),
)

CAUSAL_METRICS: tuple[str, ...] = (
    "sim_cal_rec",
    "sim_cal_rec_active",
    "strong20_rec",
    "corr",
    "mono",
)

DET_THRESHOLDS: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90)
CAUSAL_PASS_SPECS: tuple[tuple[str, str, float, bool], ...] = (
    ("sim_cal_rec > 0", "sim_cal_rec", 0.0, True),
    ("sim_cal_rec >= 0.25", "sim_cal_rec", 0.25, False),
    ("sim_cal_rec >= 0.50", "sim_cal_rec", 0.50, False),
    ("sim_cal_rec_active >= 0.50", "sim_cal_rec_active", 0.50, False),
    ("strong20_rec >= 0.25", "strong20_rec", 0.25, False),
    ("strong20_rec >= 0.50", "strong20_rec", 0.50, False),
)


def _fmt_num(val: float | None, digits: int = 3) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:.{digits}f}"


def _fmt_signed(val: float | None, digits: int = 3) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:+.{digits}f}"


def _fmt_ci(val: float | None, lo: float | None, hi: float | None, signed: bool = False) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    f = _fmt_signed if signed else _fmt_num
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return f(val)
    return f"{f(val)} [{f(lo)}, {f(hi)}]"


def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax, ay = ax[mask], ay[mask]
    if len(ax) < 3 or float(np.ptp(ax)) <= 1e-12 or float(np.ptp(ay)) <= 1e-12:
        return float("nan"), float("nan")
    r, p = pearsonr(ax, ay)
    return float(r), float(p)


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax, ay = ax[mask], ay[mask]
    if len(ax) < 3 or float(np.ptp(ax)) <= 1e-12 or float(np.ptp(ay)) <= 1e-12:
        return float("nan"), float("nan")
    rho, p = spearmanr(ax, ay)
    return float(rho), float(p)


def spearman_brown(r: float) -> float:
    """Spearman-Brown prophecy formula 2r / (1 + r) for split-half reliability."""
    if not math.isfinite(r) or (1.0 + r) <= 1e-12:
        return float("nan")
    return float((2.0 * r) / (1.0 + r))


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute Spearman rho & Pearson r with 95% bootstrap CIs over the passed rows."""
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax, ay = ax[mask], ay[mask]
    n = int(len(ax))
    rho, rho_p = _safe_spearman(ax, ay)
    r, r_p = _safe_pearson(ax, ay)
    if n < 3 or not math.isfinite(r):
        return {
            "n": n,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "spearman_ci_low": float("nan"),
            "spearman_ci_high": float("nan"),
            "pearson_r": r,
            "pearson_p": r_p,
            "pearson_ci_low": float("nan"),
            "pearson_ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(n_resamples, n))
    boot_r: list[float] = []
    boot_rho: list[float] = []
    for idx in idx_mat:
        bx = ax[idx]
        by = ay[idx]
        if float(np.ptp(bx)) <= 1e-12 or float(np.ptp(by)) <= 1e-12:
            continue
        br, _ = pearsonr(bx, by)
        brho, _ = spearmanr(bx, by)
        if math.isfinite(br):
            boot_r.append(float(br))
        if math.isfinite(brho):
            boot_rho.append(float(brho))

    r_lo, r_hi = (
        (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5)))
        if boot_r
        else (float("nan"), float("nan"))
    )
    rho_lo, rho_hi = (
        (float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5)))
        if boot_rho
        else (float("nan"), float("nan"))
    )
    return {
        "n": n,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "spearman_ci_low": rho_lo,
        "spearman_ci_high": rho_hi,
        "pearson_r": r,
        "pearson_p": r_p,
        "pearson_ci_low": r_lo,
        "pearson_ci_high": r_hi,
    }


def _pole_p999_abs(pole: Mapping[str, Any]) -> float:
    """Extract |p99.9| magnitude from active_poles.json across schema variants."""
    for container_key in ("band_edges_dir", "thresholds", "quantiles"):
        sub = pole.get(container_key)
        if isinstance(sub, Mapping):
            for k in ("p99.9", "p99_9", "q99.9", "q999"):
                if k in sub and sub[k] is not None:
                    return abs(float(sub[k]))
    for k in ("p99_9_abs", "p99.9_abs", "p99_9", "p99.9"):
        if k in pole and pole[k] is not None:
            return abs(float(pole[k]))
    return 0.0


def _resolve_dominant_pole_id(poles_for_neuron: Sequence[Mapping[str, Any]]) -> str:
    """Return dominant pole_id for a neuron (explicit flag or argmax |p99.9|)."""
    if len(poles_for_neuron) == 1:
        return str(poles_for_neuron[0]["pole_id"])
    flagged = [
        p
        for p in poles_for_neuron
        if bool(p.get("is_dominant", p.get("is_dominant_pole", False)))
    ]
    if len(flagged) == 1:
        return str(flagged[0]["pole_id"])
    ordered = sorted(
        poles_for_neuron,
        key=lambda p: (_pole_p999_abs(p), resolve_pole_sign(p)),
        reverse=True,
    )
    return str(ordered[0]["pole_id"])


def _dedup_judgements_4b(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate judgement rows by canonical (pair_id, condition, judge_model, order)."""
    dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["pair_id"]),
            str(row["condition"]),
            str(row["judge_model"]),
            str(row.get("order") or "fwd"),
        )
        dedup[key] = dict(row)
    return list(dedup.values())


def _extract_neuron_score_map(
    scores_raw: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Load Ulto85 e3_scores*.json into {k: neuron_score_dict}."""
    items = scores_raw["neurons"] if isinstance(scores_raw, Mapping) else scores_raw
    return {int(r["k"]): dict(r) for r in items}


def _build_neuron_pole_index(
    active_poles: Sequence[Mapping[str, Any]],
    scores_by_k: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    """Map pole_id -> pole, k -> list of active poles, and k -> dominant pole_id."""
    ln_to_k = {(int(r["layer"]), int(r["neuron"])): int(k) for k, r in scores_by_k.items()}
    poles_by_id: dict[str, dict[str, Any]] = {}
    poles_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for p in active_poles:
        rec = dict(p)
        pid = str(rec["pole_id"])
        if "k" in rec and rec["k"] is not None:
            k = int(rec["k"])
        else:
            k = ln_to_k[(int(rec["layer"]), int(rec["neuron"]))]
            rec["k"] = k
        rec["sign"] = resolve_pole_sign(rec)
        poles_by_id[pid] = rec
        poles_by_k[k].append(rec)

    dominant_by_k: dict[int, str] = {}
    for k, plist in poles_by_k.items():
        plist.sort(key=lambda r: (resolve_pole_sign(r) * -1, str(r["pole_id"])))
        dominant_by_k[k] = _resolve_dominant_pole_id(plist)

    return poles_by_id, dict(poles_by_k), dominant_by_k


def _summarize_one_sweep(
    rows: Sequence[Mapping[str, Any]],
    judge_model: str,
    band: str,
    condition: str,
    poles_by_id: Mapping[str, Mapping[str, Any]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    acc = neuron_clustered_bootstrap_accuracy(
        rows, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
    )
    scored_rows = [r for r in rows if classify_row_outcome(r) == "scored"]
    n_calls = len(rows)
    p_pred_a = (
        float(np.mean([1.0 if r.get("predicted_slot") == "A" else 0.0 for r in scored_rows]))
        if scored_rows
        else float("nan")
    )
    p_pos_a = (
        float(np.mean([1.0 if r.get("pos_slot") == "A" else 0.0 for r in scored_rows]))
        if scored_rows
        else float("nan")
    )
    in_tok = sum(int(r.get("prompt_tokens", 0) or 0) for r in rows)
    out_tok = sum(int(r.get("completion_tokens", 0) or 0) for r in rows)
    reas_tok = sum(int(r.get("reasoning_tokens", 0) or 0) for r in rows)
    vis_tok = sum(
        int(r.get("visible_output_tokens", r.get("completion_tokens", 0)) or 0) for r in rows
    )
    usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in rows)
    return {
        "judge_model": judge_model,
        "band": band,
        "condition": condition,
        "n_calls": n_calls,
        "acc": acc,
        "p_pred_A": p_pred_a,
        "p_pos_A": p_pos_a,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "reasoning_tokens": reas_tok,
        "visible_output_tokens": vis_tok,
        "mean_prompt_tokens": (in_tok / n_calls) if n_calls else 0.0,
        "mean_visible_output_tokens": (vis_tok / n_calls) if n_calls else 0.0,
        "mean_reasoning_tokens": (reas_tok / n_calls) if n_calls else 0.0,
        "total_cost_usd": round(usd, 6),
    }


def compute_sweep_summaries(
    j_rows: Sequence[Mapping[str, Any]],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    n_resamples: int,
    seed: int,
    top100_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute per-sweep accuracy, neuron-clustered 95% CIs, position bias, and token/USD spend."""
    sweep_specs = (
        ("p999_own_terra", PRIMARY_JUDGE_4B, "p99.9_plus", "own"),
        ("p999_own_luna", SECONDARY_JUDGE_4B, "p99.9_plus", "own"),
        ("p999_wrong_terra", PRIMARY_JUDGE_4B, "p99.9_plus", "wrong"),
        ("p999_no_label_terra", PRIMARY_JUDGE_4B, "p99.9_plus", "no_label"),
        ("p97_99_own_terra", PRIMARY_JUDGE_4B, "p97_99", "own"),
        ("p999_full_expl_terra", PRIMARY_JUDGE_4B, "p99.9_plus", "full_explanation"),
    )
    out: dict[str, Any] = {}
    core_calls = core_scored = core_unp = core_tie = 0
    core_usd = 0.0

    for name, jm, band, cond in sweep_specs:
        rows = [
            r
            for r in j_rows
            if str(r["judge_model"]) == jm
            and str(r["band"]) == band
            and str(r["condition"]) == cond
        ]
        sw = _summarize_one_sweep(rows, jm, band, cond, poles_by_id, n_resamples, seed)
        out[name] = sw
        core_calls += sw["n_calls"]
        core_scored += int(sw["acc"]["n_scored"])
        core_unp += int(sw["acc"]["n_unparseable"])
        core_tie += int(sw["acc"]["n_logprob_tie"])
        core_usd += float(sw["total_cost_usd"])

    grand_calls = core_calls
    grand_scored = core_scored
    grand_unp = core_unp
    grand_tie = core_tie
    grand_usd = core_usd

    top100_summary: dict[str, Any] | None = None
    if top100_rows:
        t_rows = [
            r
            for r in top100_rows
            if str(r["judge_model"]) == PRIMARY_JUDGE_4B and str(r["condition"]) == "own"
        ]
        top100_summary = _summarize_one_sweep(
            t_rows, PRIMARY_JUDGE_4B, "top100", "own", poles_by_id, n_resamples, seed
        )
        out["top100_own_terra"] = top100_summary
        grand_calls += top100_summary["n_calls"]
        grand_scored += int(top100_summary["acc"]["n_scored"])
        grand_unp += int(top100_summary["acc"]["n_unparseable"])
        grand_tie += int(top100_summary["acc"]["n_logprob_tie"])
        grand_usd += float(top100_summary["total_cost_usd"])

    return {
        "sweeps": out,
        "core_5_sweeps_total": {
            "n_calls": core_calls,
            "n_scored": core_scored,
            "n_unparseable": core_unp,
            "n_logprob_tie": core_tie,
            "total_cost_usd": round(core_usd, 6),
        },
        "top100_sweep_total": (
            {
                "n_calls": top100_summary["n_calls"],
                "n_scored": int(top100_summary["acc"]["n_scored"]),
                "n_unparseable": int(top100_summary["acc"]["n_unparseable"]),
                "n_logprob_tie": int(top100_summary["acc"]["n_logprob_tie"]),
                "total_cost_usd": round(float(top100_summary["total_cost_usd"]), 6),
            }
            if top100_summary
            else None
        ),
        "grand_total": {
            "n_calls": grand_calls,
            "n_scored": grand_scored,
            "n_unparseable": grand_unp,
            "n_logprob_tie": grand_tie,
            "total_cost_usd": round(grand_usd, 6),
        },
    }


def compute_split_half_reliability(
    j_rows: Sequence[Mapping[str, Any]],
    snippets_by_pair: Mapping[str, Mapping[str, Any]],
    poles_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
    judge_model: str,
    band: str = "p99.9_plus",
    n_random_splits: int = 100,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute deterministic and 100-random 5/5-balanced split-half reliability (Spearman-Brown)."""
    by_pole: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for r in j_rows:
        if (
            str(r["judge_model"]) == judge_model
            and str(r["band"]) == band
            and str(r["condition"]) == "own"
            and classify_row_outcome(r) == "scored"
        ):
            pid = str(r["pole_id"])
            pair_id = str(r["pair_id"])
            snip = snippets_by_pair.get(pair_id, {})
            pidx = int(r.get("pair_idx", snip.get("pair_idx", -1)))
            slot = str(r.get("pos_slot", snip.get("pos_slot", "A")))
            by_pole[pid].append((pidx, slot, float(r["correct"])))

    for pid in by_pole:
        by_pole[pid].sort(key=lambda x: x[0])

    valid_ks = sorted(
        k
        for k, plist in poles_by_k.items()
        if all(len(by_pole.get(str(p["pole_id"]), [])) >= 2 for p in plist)
    )
    if len(valid_ks) < 3:
        return {"judge_model": judge_model, "band": band, "n_neurons": len(valid_ks)}

    det_h1: list[float] = []
    det_h2: list[float] = []
    pole_det_h1: list[float] = []
    pole_det_h2: list[float] = []

    for k in valid_ks:
        nh1_vals: list[float] = []
        nh2_vals: list[float] = []
        for p in poles_by_k[k]:
            items = by_pole[str(p["pole_id"])]
            mid = len(items) // 2
            m1 = float(np.mean([x[2] for x in items[:mid]]))
            m2 = float(np.mean([x[2] for x in items[mid:]]))
            nh1_vals.append(m1)
            nh2_vals.append(m2)
            pole_det_h1.append(m1)
            pole_det_h2.append(m2)
        det_h1.append(float(np.mean(nh1_vals)))
        det_h2.append(float(np.mean(nh2_vals)))

    det_r, det_r_p = _safe_pearson(det_h1, det_h2)
    det_rho, det_rho_p = _safe_spearman(det_h1, det_h2)
    pole_det_r, _ = _safe_pearson(pole_det_h1, pole_det_h2)
    pole_det_rho, _ = _safe_spearman(pole_det_h1, pole_det_h2)

    rng = np.random.default_rng(seed)
    pole_ab_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k in valid_ks:
        for p in poles_by_k[k]:
            pid = str(p["pole_id"])
            items = by_pole[pid]
            a_vals = np.asarray([x[2] for x in items if x[1] == "A"], dtype=np.float64)
            b_vals = np.asarray([x[2] for x in items if x[1] == "B"], dtype=np.float64)
            pole_ab_arrays[pid] = (a_vals, b_vals)

    rand_r_list: list[float] = []
    rand_sb_r_list: list[float] = []
    rand_rho_list: list[float] = []
    rand_sb_rho_list: list[float] = []

    for _ in range(n_random_splits):
        h1_vec: list[float] = []
        h2_vec: list[float] = []
        for k in valid_ks:
            nh1_vals = []
            nh2_vals = []
            for p in poles_by_k[k]:
                pid = str(p["pole_id"])
                a_arr, b_arr = pole_ab_arrays[pid]
                if len(a_arr) >= 2 and len(b_arr) >= 2:
                    pa = rng.permutation(len(a_arr))
                    pb = rng.permutation(len(b_arr))
                    ha = len(a_arr) // 2
                    hb = len(b_arr) // 2
                    m1 = float(np.mean(np.concatenate([a_arr[pa[:ha]], b_arr[pb[:hb]]])))
                    m2 = float(np.mean(np.concatenate([a_arr[pa[ha:]], b_arr[pb[hb:]]])))
                else:
                    all_arr = np.concatenate([a_arr, b_arr])
                    perm = rng.permutation(len(all_arr))
                    mid = len(all_arr) // 2
                    m1 = float(np.mean(all_arr[perm[:mid]]))
                    m2 = float(np.mean(all_arr[perm[mid:]]))
                nh1_vals.append(m1)
                nh2_vals.append(m2)
            h1_vec.append(float(np.mean(nh1_vals)))
            h2_vec.append(float(np.mean(nh2_vals)))

        rr, _ = _safe_pearson(h1_vec, h2_vec)
        rrho, _ = _safe_spearman(h1_vec, h2_vec)
        if math.isfinite(rr):
            rand_r_list.append(rr)
            rand_sb_r_list.append(spearman_brown(rr))
        if math.isfinite(rrho):
            rand_rho_list.append(rrho)
            rand_sb_rho_list.append(spearman_brown(rrho))

    def _summarize(vals: Sequence[float]) -> dict[str, float]:
        if not vals:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "p2_5": float("nan"),
                "p97_5": float("nan"),
            }
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p2_5": float(np.percentile(arr, 2.5)),
            "p97_5": float(np.percentile(arr, 97.5)),
        }

    return {
        "judge_model": judge_model,
        "band": band,
        "n_neurons": len(valid_ks),
        "n_poles": len(pole_det_h1),
        "n_random_splits": n_random_splits,
        "deterministic_halves": {
            "pearson_r": det_r,
            "pearson_p": det_r_p,
            "spearman_brown_r": spearman_brown(det_r),
            "spearman_rho": det_rho,
            "spearman_p": det_rho_p,
            "spearman_brown_rho": spearman_brown(det_rho),
            "per_pole_pearson_r": pole_det_r,
            "per_pole_spearman_brown_r": spearman_brown(pole_det_r),
            "per_pole_spearman_rho": pole_det_rho,
            "per_pole_spearman_brown_rho": spearman_brown(pole_det_rho),
        },
        "random_balanced_splits": {
            "pearson_r": _summarize(rand_r_list),
            "spearman_brown_r": _summarize(rand_sb_r_list),
            "spearman_rho": _summarize(rand_rho_list),
            "spearman_brown_rho": _summarize(rand_sb_rho_list),
        },
    }


def compute_cross_judge_agreement_4b(
    j_rows: Sequence[Mapping[str, Any]],
    neuron_table: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute pair-by-pair and per-neuron agreement between terra@none and luna@none on p99.9+ own."""
    terra_map: dict[str, Mapping[str, Any]] = {}
    luna_map: dict[str, Mapping[str, Any]] = {}
    for r in j_rows:
        if str(r["band"]) != "p99.9_plus" or str(r["condition"]) != "own":
            continue
        if classify_row_outcome(r) != "scored":
            continue
        if str(r["judge_model"]) == PRIMARY_JUDGE_4B:
            terra_map[str(r["pair_id"])] = r
        elif str(r["judge_model"]) == SECONDARY_JUDGE_4B:
            luna_map[str(r["pair_id"])] = r

    shared = sorted(set(terra_map) & set(luna_map))
    if not shared:
        return {"n_overlap_pairs": 0}

    y1 = [str(terra_map[pid]["predicted_slot"]) for pid in shared]
    y2 = [str(luna_map[pid]["predicted_slot"]) for pid in shared]
    exact_agree = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(y1, y2)]))
    kappa = cohens_kappa(y1, y2)

    xs: list[float] = []
    ys: list[float] = []
    for k in sorted(neuron_table.keys()):
        t_val = neuron_table[k].get("det_p999_terra_mean")
        l_val = neuron_table[k].get("det_p999_luna_mean")
        if t_val is not None and l_val is not None and math.isfinite(t_val) and math.isfinite(l_val):
            xs.append(float(t_val))
            ys.append(float(l_val))
    nr, nr_p = _safe_pearson(xs, ys)
    nrho, nrho_p = _safe_spearman(xs, ys)
    mad = float(np.mean(np.abs(np.asarray(xs) - np.asarray(ys)))) if xs else float("nan")

    return {
        "n_overlap_pairs": len(shared),
        "n_overlap_neurons": len(xs),
        "pair_exact_agreement": exact_agree,
        "pair_cohens_kappa": kappa,
        "neuron_pearson_r": nr,
        "neuron_pearson_p": nr_p,
        "neuron_spearman_rho": nrho,
        "neuron_spearman_p": nrho_p,
        "neuron_mean_abs_diff": mad,
    }


def compute_causal_opus_reliability(
    scores_by_k: Mapping[int, Mapping[str, Any]],
    opus_by_k: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare Sonnet (e3_scores.json) vs. Opus (e3_scores_opus.json) on overlapping neurons."""
    shared_ks = sorted(set(scores_by_k.keys()) & set(opus_by_k.keys()))
    if not shared_ks:
        return {"n_overlap": 0, "metrics": {}}

    metric_thresholds: dict[str, list[tuple[str, float, bool]]] = {
        "sim_cal_rec": [("> 0", 0.0, True), (">= 0.25", 0.25, False), (">= 0.50", 0.50, False)],
        "sim_cal_rec_active": [(">= 0.50", 0.50, False)],
        "strong20_rec": [(">= 0.50", 0.50, False)],
        "corr": [(">= 0.50", 0.50, False)],
    }

    metrics_out: dict[str, Any] = {}
    for mname, thrs in metric_thresholds.items():
        s_vals = np.asarray([float(scores_by_k[k][mname]) for k in shared_ks], dtype=np.float64)
        o_vals = np.asarray([float(opus_by_k[k][mname]) for k in shared_ks], dtype=np.float64)
        rho, rho_p = _safe_spearman(s_vals, o_vals)
        r, r_p = _safe_pearson(s_vals, o_vals)
        diffs = np.abs(s_vals - o_vals)
        thr_agreements: dict[str, Any] = {}
        for tlabel, tval, strict_gt in thrs:
            s_pass = (s_vals > tval) if strict_gt else (s_vals >= tval)
            o_pass = (o_vals > tval) if strict_gt else (o_vals >= tval)
            agree = float(np.mean(s_pass == o_pass))
            kap = cohens_kappa(
                ["A" if b else "B" for b in s_pass], ["A" if b else "B" for b in o_pass]
            )
            thr_agreements[tlabel] = {
                "sonnet_pass_count": int(np.sum(s_pass)),
                "opus_pass_count": int(np.sum(o_pass)),
                "both_pass_count": int(np.sum(s_pass & o_pass)),
                "exact_agreement": agree,
                "cohens_kappa": kap,
            }
        metrics_out[mname] = {
            "n_overlap": len(shared_ks),
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "pearson_r": r,
            "pearson_p": r_p,
            "median_abs_diff": float(np.median(diffs)),
            "mean_abs_diff": float(np.mean(diffs)),
            "sonnet_mean": float(np.mean(s_vals)),
            "opus_mean": float(np.mean(o_vals)),
            "threshold_agreement": thr_agreements,
        }

    return {
        "n_overlap": len(shared_ks),
        "overlap_ks": shared_ks,
        "metrics": metrics_out,
    }


def build_per_neuron_table(
    scores_by_k: Mapping[int, Mapping[str, Any]],
    poles_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
    dominant_by_k: Mapping[int, str],
    j_rows: Sequence[Mapping[str, Any]],
    top100_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Aggregate per-pole detection scores to per-neuron primary (mean active) and sensitivity (dominant)."""
    acc_by_key: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in j_rows:
        if classify_row_outcome(r) != "scored":
            continue
        acc_by_key[
            (str(r["pole_id"]), str(r["judge_model"]), str(r["band"]), str(r["condition"]))
        ].append(float(r["correct"]))
    for r in top100_rows or []:
        if classify_row_outcome(r) != "scored":
            continue
        acc_by_key[
            (str(r["pole_id"]), str(r["judge_model"]), "top100", str(r["condition"]))
        ].append(float(r["correct"]))

    def _pole_mean(pid: str, jm: str, band: str, cond: str) -> float | None:
        vals = acc_by_key.get((pid, jm, band, cond))
        return float(np.mean(vals)) if vals else None

    def _agg(
        plist: Sequence[Mapping[str, Any]], dom_pid: str, jm: str, band: str, cond: str
    ) -> tuple[float | None, float | None, dict[str, float]]:
        per_pole: dict[str, float] = {}
        for p in plist:
            pid = str(p["pole_id"])
            v = _pole_mean(pid, jm, band, cond)
            if v is not None:
                per_pole[pid] = v
        mean_val = float(np.mean(list(per_pole.values()))) if per_pole else None
        dom_val = per_pole.get(dom_pid)
        return mean_val, dom_val, per_pole

    table: dict[int, dict[str, Any]] = {}
    for k in sorted(scores_by_k.keys()):
        sc = scores_by_k[k]
        plist = poles_by_k.get(k, [])
        dom_pid = dominant_by_k.get(k, str(plist[0]["pole_id"]) if plist else "")
        in_full_50 = any(bool(p.get("in_full_expl_50", False)) for p in plist)

        t_m, t_d, t_pp = _agg(plist, dom_pid, PRIMARY_JUDGE_4B, "p99.9_plus", "own")
        t100_m, t100_d, t100_pp = _agg(plist, dom_pid, PRIMARY_JUDGE_4B, "top100", "own")
        l_m, l_d, l_pp = _agg(plist, dom_pid, SECONDARY_JUDGE_4B, "p99.9_plus", "own")
        w_m, w_d, _ = _agg(plist, dom_pid, PRIMARY_JUDGE_4B, "p99.9_plus", "wrong")
        nl_m, nl_d, _ = _agg(plist, dom_pid, PRIMARY_JUDGE_4B, "p99.9_plus", "no_label")
        p97_m, p97_d, _ = _agg(plist, dom_pid, PRIMARY_JUDGE_4B, "p97_99", "own")
        fe_m, fe_d, _ = _agg(plist, dom_pid, PRIMARY_JUDGE_4B, "p99.9_plus", "full_explanation")

        table[k] = {
            "k": k,
            "layer": int(sc["layer"]),
            "neuron": int(sc["neuron"]),
            "label": str(sc.get("label", "")),
            "n_active_poles": len(plist),
            "polarity": "bipolar" if len(plist) >= 2 else "unipolar",
            "active_pole_ids": [str(p["pole_id"]) for p in plist],
            "dominant_pole_id": dom_pid,
            "in_full_expl_50": in_full_50,
            "sim_cal_rec": float(sc["sim_cal_rec"]),
            "sim_cal_rec_active": float(sc["sim_cal_rec_active"]),
            "strong20_rec": float(sc["strong20_rec"]),
            "corr": float(sc["corr"]),
            "mono": float(sc["mono"]),
            "det_p999_terra_mean": t_m,
            "det_p999_terra_dom": t_d,
            "det_p999_terra_per_pole": t_pp,
            "det_top100_terra_mean": t100_m,
            "det_top100_terra_dom": t100_d,
            "det_top100_terra_per_pole": t100_pp,
            "det_p999_luna_mean": l_m,
            "det_p999_luna_dom": l_d,
            "det_p999_luna_per_pole": l_pp,
            "det_p999_wrong_mean": w_m,
            "det_p999_wrong_dom": w_d,
            "det_p999_no_label_mean": nl_m,
            "det_p999_no_label_dom": nl_d,
            "det_p97_99_terra_mean": p97_m,
            "det_p97_99_terra_dom": p97_d,
            "det_p999_full_expl_mean": fe_m,
            "det_p999_full_expl_dom": fe_d,
        }

    return table


def _eval_pass_matrix_for_field(
    neuron_table: Mapping[int, Mapping[str, Any]],
    det_field: str,
    nl_field: str,
    global_no_label_floor: float | None,
) -> list[dict[str, Any]]:
    """Compute 2x2 pass counts across fixed AND above-no_label detection criteria vs. causal targets."""
    ordered_ks = sorted(neuron_table.keys())
    valid = [
        neuron_table[k]
        for k in ordered_ks
        if neuron_table[k].get(det_field) is not None
        and math.isfinite(float(neuron_table[k][det_field]))
    ]
    n_tot = len(valid)
    if n_tot == 0:
        return []

    det_criteria: list[tuple[str, float | None, Any]] = []
    if global_no_label_floor is not None and math.isfinite(global_no_label_floor):
        gnl = float(global_no_label_floor)
        det_criteria.append(
            (f"> global no_label ({gnl:.3f})", gnl, lambda rec, g=gnl: float(rec[det_field]) > g)
        )
    has_per_neuron_nl = all(
        rec.get(nl_field) is not None and math.isfinite(float(rec[nl_field])) for rec in valid
    )
    if has_per_neuron_nl:
        det_criteria.append(
            ("> per-neuron no_label", None, lambda rec: float(rec[det_field]) > float(rec[nl_field]) + 1e-12)
        )
        det_criteria.append(
            (
                "own - no_label >= 0.20",
                0.20,
                lambda rec: (float(rec[det_field]) - float(rec[nl_field])) >= 0.20 - 1e-12,
            )
        )
    for d_thr in DET_THRESHOLDS:
        det_criteria.append(
            (f">= {d_thr:.2f}", d_thr, lambda rec, t=d_thr: float(rec[det_field]) >= t - 1e-12)
        )

    rows_out: list[dict[str, Any]] = []
    for c_label, c_field, c_thr, strict_gt in CAUSAL_PASS_SPECS:
        for d_label, d_thr_num, d_fn in det_criteria:
            both_pass = det_only = causal_only = both_fail = 0
            for rec in valid:
                d_pass = bool(d_fn(rec))
                c_val = float(rec[c_field])
                c_pass = (c_val > c_thr) if strict_gt else (c_val >= c_thr - 1e-12)
                if d_pass and c_pass:
                    both_pass += 1
                elif d_pass and not c_pass:
                    det_only += 1
                elif (not d_pass) and c_pass:
                    causal_only += 1
                else:
                    both_fail += 1
            n_det_pass = both_pass + det_only
            n_causal_pass = both_pass + causal_only
            rows_out.append(
                {
                    "det_criterion": d_label,
                    "det_threshold": d_thr_num,
                    "causal_criterion": c_label,
                    "causal_field": c_field,
                    "n_total": n_tot,
                    "n_det_pass": n_det_pass,
                    "det_pass_rate": (n_det_pass / n_tot) if n_tot else float("nan"),
                    "n_causal_pass": n_causal_pass,
                    "causal_pass_rate": (n_causal_pass / n_tot) if n_tot else float("nan"),
                    "both_pass": both_pass,
                    "det_only": det_only,
                    "causal_only": causal_only,
                    "both_fail": both_fail,
                    "causal_fail_given_det_pass": (
                        (det_only / n_det_pass) if n_det_pass > 0 else float("nan")
                    ),
                    "causal_pass_given_det_pass": (
                        (both_pass / n_det_pass) if n_det_pass > 0 else float("nan")
                    ),
                }
            )
    return rows_out


def compute_pass_rates_and_correlations(
    neuron_table: Mapping[int, Mapping[str, Any]],
    global_no_label_floor: float | None,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute 2x2 pass-rate quadrants and bootstrap correlations against Ulto85 causal scores."""
    ordered_ks = sorted(neuron_table.keys())

    pass_matrices: dict[str, list[dict[str, Any]]] = {
        "mean_active_poles": _eval_pass_matrix_for_field(
            neuron_table, "det_p999_terra_mean", "det_p999_no_label_mean", global_no_label_floor
        ),
        "dominant_pole": _eval_pass_matrix_for_field(
            neuron_table, "det_p999_terra_dom", "det_p999_no_label_dom", global_no_label_floor
        ),
    }
    has_top100 = any(
        neuron_table[k].get("det_top100_terra_mean") is not None for k in ordered_ks
    )
    if has_top100:
        pass_matrices["top100_mean_active_poles"] = _eval_pass_matrix_for_field(
            neuron_table, "det_top100_terra_mean", "det_p999_no_label_mean", global_no_label_floor
        )
        pass_matrices["top100_dominant_pole"] = _eval_pass_matrix_for_field(
            neuron_table, "det_top100_terra_dom", "det_p999_no_label_dom", global_no_label_floor
        )

    det_predictors: list[tuple[str, str]] = [
        ("p999_terra_mean", "det_p999_terra_mean"),
        ("p999_terra_dom", "det_p999_terra_dom"),
        ("p999_luna_mean", "det_p999_luna_mean"),
        ("p97_99_terra_mean", "det_p97_99_terra_mean"),
    ]
    if has_top100:
        det_predictors.extend(
            [
                ("top100_terra_mean", "det_top100_terra_mean"),
                ("top100_terra_dom", "det_top100_terra_dom"),
            ]
        )

    correlations: dict[str, dict[str, Any]] = {}
    for pname, pfield in det_predictors:
        correlations[pname] = {}
        for target in CAUSAL_METRICS:
            xs: list[float] = []
            ys: list[float] = []
            for k in ordered_ks:
                v = neuron_table[k].get(pfield)
                t = neuron_table[k].get(target)
                if v is not None and t is not None and math.isfinite(float(v)) and math.isfinite(float(t)):
                    xs.append(float(v))
                    ys.append(float(t))
            correlations[pname][target] = bootstrap_corr_ci(
                xs, ys, n_resamples=n_resamples, seed=seed
            )

    return {
        "pass_matrices": pass_matrices,
        "correlations": correlations,
    }


def compute_full_explanation_comparison(
    j_rows: Sequence[Mapping[str, Any]],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    poles_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute exact paired full_explanation - pole_only on the 50-neuron stratified arm."""
    fe_rows = [
        r
        for r in j_rows
        if str(r["judge_model"]) == PRIMARY_JUDGE_4B
        and str(r["band"]) == "p99.9_plus"
        and str(r["condition"]) == "full_explanation"
    ]
    if not fe_rows:
        return {"n_pairs": 0}

    fe_pole_ids = {str(r["pole_id"]) for r in fe_rows}
    own_rows = [
        r
        for r in j_rows
        if str(r["judge_model"]) == PRIMARY_JUDGE_4B
        and str(r["band"]) == "p99.9_plus"
        and str(r["condition"]) == "own"
        and str(r["pole_id"]) in fe_pole_ids
    ]

    fe_by_pair = {
        str(r["pair_id"]): r for r in fe_rows if classify_row_outcome(r) == "scored"
    }
    own_by_pair = {
        str(r["pair_id"]): r for r in own_rows if classify_row_outcome(r) == "scored"
    }
    shared_pairs = sorted(set(fe_by_pair) & set(own_by_pair))

    def _subset_stats(allowed_pids: set[str]) -> dict[str, Any]:
        sub_fe = [r for r in fe_rows if str(r["pole_id"]) in allowed_pids]
        sub_own = [r for r in own_rows if str(r["pole_id"]) in allowed_pids]
        per_pole_diff: dict[str, list[float]] = defaultdict(list)
        for pair_id in shared_pairs:
            pid = str(fe_by_pair[pair_id]["pole_id"])
            if pid in allowed_pids:
                per_pole_diff[pid].append(
                    float(fe_by_pair[pair_id]["correct"]) - float(own_by_pair[pair_id]["correct"])
                )
        return {
            "full_explanation": neuron_clustered_bootstrap_accuracy(
                sub_fe, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
            ),
            "pole_only": neuron_clustered_bootstrap_accuracy(
                sub_own, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
            ),
            "paired_full_minus_pole": neuron_clustered_bootstrap_paired_diff(
                per_pole_diff, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
            ),
        }

    polarity_by_pid: dict[str, str] = {}
    for k, plist in poles_by_k.items():
        pol = "bipolar" if len(plist) >= 2 else "unipolar"
        for p in plist:
            polarity_by_pid[str(p["pole_id"])] = pol

    unipolar_pids = {pid for pid in fe_pole_ids if polarity_by_pid.get(pid) == "unipolar"}
    bipolar_pids = {pid for pid in fe_pole_ids if polarity_by_pid.get(pid) == "bipolar"}
    layers = sorted({int(poles_by_id[pid]["layer"]) for pid in fe_pole_ids if pid in poles_by_id})

    by_layer: dict[int, Any] = {}
    for ly in layers:
        ly_pids = {
            pid for pid in fe_pole_ids if pid in poles_by_id and int(poles_by_id[pid]["layer"]) == ly
        }
        by_layer[ly] = _subset_stats(ly_pids)

    return {
        "n_pairs": len(shared_pairs),
        "overall": _subset_stats(fe_pole_ids),
        "unipolar": _subset_stats(unipolar_pids),
        "bipolar": _subset_stats(bipolar_pids),
        "by_layer": by_layer,
    }


def enrich_sparsity_diagnostics(
    sparsity_diag: Mapping[str, Any] | None,
    neuron_table: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Attach p99.9+ and top100 per-pole accuracies to sparse poles (<20% max) and named diagnostics."""
    if not sparsity_diag:
        return None
    pole_acc_p999: dict[str, float] = {}
    pole_acc_top100: dict[str, float] = {}
    for rec in neuron_table.values():
        for pid, val in (rec.get("det_p999_terra_per_pole") or {}).items():
            pole_acc_p999[str(pid)] = float(val)
        for pid, val in (rec.get("det_top100_terra_per_pole") or {}).items():
            pole_acc_top100[str(pid)] = float(val)

    sparse_rows: list[dict[str, Any]] = []
    for row in sparsity_diag.get("sparse_poles_lt_20pct_max", []):
        pid = str(row["pole_id"])
        rec = dict(row)
        rec["det_p999_terra"] = pole_acc_p999.get(pid)
        rec["det_top100_terra"] = pole_acc_top100.get(pid)
        sparse_rows.append(rec)

    named_diag: dict[str, list[dict[str, Any]]] = {}
    for nm, rows in (sparsity_diag.get("named_neurons_diagnostic") or {}).items():
        enriched: list[dict[str, Any]] = []
        for row in rows:
            pid = str(row["pole_id"])
            rec = dict(row)
            rec["det_p999_terra"] = pole_acc_p999.get(pid)
            rec["det_top100_terra"] = pole_acc_top100.get(pid)
            enriched.append(rec)
        named_diag[str(nm)] = enriched

    return {
        "n_active_poles": int(sparsity_diag.get("n_active_poles", 0)),
        "n_poles_lt_20pct_max": int(sparsity_diag.get("n_poles_lt_20pct_max", len(sparse_rows))),
        "frac_poles_lt_20pct_max": float(sparsity_diag.get("frac_poles_lt_20pct_max", 0.0)),
        "n_neurons_any_pole_lt_20pct_max": int(
            sparsity_diag.get("n_neurons_any_pole_lt_20pct_max", 0)
        ),
        "p99_9_frac_of_max_stats": dict(sparsity_diag.get("p99_9_frac_of_max_stats", {})),
        "named_neurons_diagnostic": named_diag,
        "sparse_poles_lt_20pct_max": sparse_rows,
    }


def extract_named_neurons_table(
    neuron_table: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract exact readout rows for info (L0 N9209), PCR (L0 N8146), and insurance (L32 N4362)."""
    rows: list[dict[str, Any]] = []
    for short_name, k, expected_layer, expected_neuron in NAMED_NEURONS:
        rec = neuron_table.get(k)
        if rec is None:
            for cand in neuron_table.values():
                if int(cand["layer"]) == expected_layer and int(cand["neuron"]) == expected_neuron:
                    rec = cand
                    break
        if rec is None:
            continue
        row = dict(rec)
        row["short_name"] = short_name
        rows.append(row)
    return rows


def compute_all_4b_metrics(
    active_poles: Sequence[Mapping[str, Any]],
    scores_raw: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    opus_scores_raw: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    snippets: Sequence[Mapping[str, Any]],
    judgements: Sequence[Mapping[str, Any]],
    wrong_donors_meta: Mapping[str, Any] | None = None,
    top100_snippets: Sequence[Mapping[str, Any]] | None = None,
    top100_judgements: Sequence[Mapping[str, Any]] | None = None,
    sparsity_diag: Mapping[str, Any] | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    n_random_splits: int = 100,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute all structured metrics for the exploratory Qwen3-4B (Ulto85) audit."""
    scores_by_k = _extract_neuron_score_map(scores_raw)
    opus_by_k = _extract_neuron_score_map(opus_scores_raw) if opus_scores_raw is not None else {}
    poles_by_id, poles_by_k, dominant_by_k = _build_neuron_pole_index(active_poles, scores_by_k)
    snippets_by_pair = {str(s["pair_id"]): s for s in snippets}
    top100_snips_by_pair = {str(s["pair_id"]): s for s in (top100_snippets or [])}
    j_rows = _dedup_judgements_4b(judgements)
    t100_rows = _dedup_judgements_4b(top100_judgements or [])

    sweep_audit = compute_sweep_summaries(
        j_rows=j_rows,
        poles_by_id=poles_by_id,
        n_resamples=n_resamples,
        seed=seed,
        top100_rows=t100_rows if t100_rows else None,
    )
    global_nl_mean = (
        sweep_audit["sweeps"].get("p999_no_label_terra", {}).get("acc", {}).get("mean")
    )
    neuron_table = build_per_neuron_table(
        scores_by_k=scores_by_k,
        poles_by_k=poles_by_k,
        dominant_by_k=dominant_by_k,
        j_rows=j_rows,
        top100_rows=t100_rows if t100_rows else None,
    )
    split_half_terra = compute_split_half_reliability(
        j_rows=j_rows,
        snippets_by_pair=snippets_by_pair,
        poles_by_k=poles_by_k,
        judge_model=PRIMARY_JUDGE_4B,
        band="p99.9_plus",
        n_random_splits=n_random_splits,
        seed=seed,
    )
    split_half_luna = compute_split_half_reliability(
        j_rows=j_rows,
        snippets_by_pair=snippets_by_pair,
        poles_by_k=poles_by_k,
        judge_model=SECONDARY_JUDGE_4B,
        band="p99.9_plus",
        n_random_splits=n_random_splits,
        seed=seed,
    )
    split_half_map: dict[str, Any] = {
        PRIMARY_JUDGE_4B: split_half_terra,
        SECONDARY_JUDGE_4B: split_half_luna,
    }
    if t100_rows:
        split_half_map["top100_terra"] = compute_split_half_reliability(
            j_rows=t100_rows,
            snippets_by_pair=top100_snips_by_pair,
            poles_by_k=poles_by_k,
            judge_model=PRIMARY_JUDGE_4B,
            band=str(t100_rows[0].get("band", "top100")),
            n_random_splits=n_random_splits,
            seed=seed,
        )

    cross_judge = compute_cross_judge_agreement_4b(j_rows=j_rows, neuron_table=neuron_table)
    causal_opus = compute_causal_opus_reliability(scores_by_k=scores_by_k, opus_by_k=opus_by_k)
    pass_and_corr = compute_pass_rates_and_correlations(
        neuron_table=neuron_table,
        global_no_label_floor=float(global_nl_mean) if global_nl_mean is not None else None,
        n_resamples=n_resamples,
        seed=seed,
    )
    full_expl = compute_full_explanation_comparison(
        j_rows=j_rows,
        poles_by_id=poles_by_id,
        poles_by_k=poles_by_k,
        n_resamples=n_resamples,
        seed=seed,
    )
    named_neurons = extract_named_neurons_table(neuron_table)
    sparsity_enriched = enrich_sparsity_diagnostics(sparsity_diag, neuron_table)

    n_unipolar = sum(1 for plist in poles_by_k.values() if len(plist) == 1)
    n_bipolar = sum(1 for plist in poles_by_k.values() if len(plist) >= 2)

    return {
        "n_neurons": len(scores_by_k),
        "n_active_poles": len(poles_by_id),
        "n_unipolar_neurons": n_unipolar,
        "n_bipolar_neurons": n_bipolar,
        "n_snippets": len(snippets) if snippets else len({str(r["pair_id"]) for r in j_rows}),
        "n_top100_snippets": (
            len(top100_snippets)
            if top100_snippets
            else len({str(r["pair_id"]) for r in t100_rows})
        ),
        "n_judgements": len(j_rows),
        "n_top100_judgements": len(t100_rows),
        "global_no_label_floor": float(global_nl_mean) if global_nl_mean is not None else None,
        "wrong_donors_summary": {
            "embed_model": wrong_donors_meta.get("embed_model"),
            "max_embed_cosine_cutoff": wrong_donors_meta.get("max_embed_cosine_cutoff"),
            "same_layer_violations": wrong_donors_meta.get("same_layer_violations"),
            "threshold_relaxed_count": wrong_donors_meta.get("threshold_relaxed_count"),
            "cosine_stats": wrong_donors_meta.get("cosine_stats"),
            "embed_cost_usd": wrong_donors_meta.get("embed_cost_usd"),
        }
        if wrong_donors_meta
        else None,
        "sweep_audit": sweep_audit,
        "split_half_reliability": split_half_map,
        "cross_judge_agreement": cross_judge,
        "causal_opus_reliability": causal_opus,
        "pass_matrices": pass_and_corr["pass_matrices"],
        "correlations": pass_and_corr["correlations"],
        "full_explanation_50": full_expl,
        "named_neurons": named_neurons,
        "sparsity_analysis": sparsity_enriched,
        "per_neuron": {str(k): v for k, v in neuron_table.items()},
    }


def _annotate_named_neurons(
    ax: Any,
    named_lookup: Mapping[int, Mapping[str, Any]],
    det_field: str,
    y_field: str,
    edge_lo: float,
    edge_hi: float,
    offsets_for_panel: Mapping[str, tuple[int, int]],
    include_layer_neuron: bool = True,
) -> None:
    """Highlight and annotate info, PCR, and insurance cleanly without collisions."""
    for short_name, k, ly, nn in NAMED_NEURONS:
        nrec = named_lookup.get(k)
        if not nrec or nrec.get(det_field) is None:
            continue
        nx = float(nrec[det_field])
        ny_raw = float(nrec[y_field])
        ny = max(edge_lo, min(edge_hi, ny_raw))
        ax.scatter(
            [nx],
            [ny],
            s=95,
            facecolors="none",
            edgecolors="#D81B60",
            linewidths=1.8,
            zorder=7,
        )
        dx, dy = offsets_for_panel.get(short_name, (-75, 16))
        if nx < 0.28 and dx < 0:
            dx = abs(dx)
        if ny > 0.90 and dy > 0:
            dy = -abs(dy)
        ann_label = f"{short_name} (L{ly} N{nn})" if include_layer_neuron else short_name
        ax.annotate(
            ann_label,
            xy=(nx, ny),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.4,
            fontweight="bold",
            color="#880E4F",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FFFFFF",
                "edgecolor": "#D81B60",
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": "#D81B60",
                "lw": 1.15,
                "shrinkA": 0,
                "shrinkB": 4,
            },
            zorder=8,
        )


def plot_detection_vs_causal(
    metrics: Mapping[str, Any],
    output_path: Path = DEFAULT_FIGURE_4B_PATH,
    y_clip: tuple[float, float] = (-0.85, 1.05),
    det_field: str = "det_p999_terra_mean",
    corr_key: str = "p999_terra_mean",
    band_label: str = "p99.9+",
) -> Path:
    """Plot 2-panel scatter: detection accuracy vs sim_cal_rec and sim_cal_rec_active."""
    per_neuron = metrics.get("per_neuron", {})
    rows = [
        rec
        for rec in per_neuron.values()
        if rec.get(det_field) is not None and math.isfinite(float(rec[det_field]))
    ]

    xs = np.asarray([float(r[det_field]) for r in rows], dtype=np.float64)
    layers = np.asarray([int(r["layer"]) for r in rows], dtype=np.float64)
    y_left_raw = np.asarray([float(r["sim_cal_rec"]) for r in rows], dtype=np.float64)
    y_right_raw = np.asarray([float(r["sim_cal_rec_active"]) for r in rows], dtype=np.float64)

    sweeps = metrics.get("sweep_audit", {}).get("sweeps", {})
    wrong_mean = sweeps.get("p999_wrong_terra", {}).get("acc", {}).get("mean")
    nl_mean = sweeps.get("p999_no_label_terra", {}).get("acc", {}).get("mean")

    y_min, y_max = y_clip
    edge_lo = y_min + 0.025
    edge_hi = y_max - 0.025

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13.6, 6.0), dpi=250, sharex=True, sharey=True
    )
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=35.0)

    named_lookup = {int(rec["k"]): rec for rec in metrics.get("named_neurons", [])}
    corr_map = metrics.get("correlations", {}).get(corr_key, {})

    panels = (
        (
            ax_l,
            y_left_raw,
            "Ordinary Text Causal Recovery (sim_cal_rec)",
            "sim_cal_rec",
        ),
        (
            ax_r,
            y_right_raw,
            "Active-Token Causal Recovery (sim_cal_rec_active)",
            "sim_cal_rec_active",
        ),
    )

    # Panel-tailored offsets preventing overlap on both p99.9+ and top100
    is_top100 = det_field == "det_top100_terra_mean"
    offsets_map = {
        0: {
            "info": (-115, -18) if is_top100 else (-68, -26),
            "PCR": (-112, 20),
            "insurance": (-108, -26) if is_top100 else (-82, -26),
        },
        1: {
            "info": (-120, -14) if is_top100 else (-66, -24),
            "PCR": (-115, -36),
            "insurance": (-108, -26) if is_top100 else (-82, -26),
        },
    }

    scatter_mappable = None
    for panel_idx, (ax, y_raw, ptitle, ckey) in enumerate(panels):
        lo_mask = y_raw < y_min
        hi_mask = y_raw > y_max
        in_mask = (~lo_mask) & (~hi_mask)

        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.80, zorder=1)
        ax.axhline(0.50, color="#2E7D32", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
        if wrong_mean is not None and math.isfinite(float(wrong_mean)):
            ax.axvline(
                float(wrong_mean),
                color="#B83333",
                linestyle=":",
                linewidth=1.15,
                alpha=0.80,
                zorder=1,
                label=f"wrong floor ({float(wrong_mean):.3f})" if panel_idx == 0 else None,
            )
        if nl_mean is not None and math.isfinite(float(nl_mean)):
            ax.axvline(
                float(nl_mean),
                color="#C87D1A",
                linestyle="-.",
                linewidth=1.15,
                alpha=0.80,
                zorder=1,
                label=f"no_label floor ({float(nl_mean):.3f})" if panel_idx == 0 else None,
            )

        sc = ax.scatter(
            xs[in_mask],
            y_raw[in_mask],
            c=layers[in_mask],
            cmap=cmap,
            norm=norm,
            s=30,
            alpha=0.82,
            edgecolors="#FFFFFF",
            linewidths=0.35,
            zorder=3,
        )
        scatter_mappable = sc

        n_lo = int(np.sum(lo_mask))
        n_hi = int(np.sum(hi_mask))
        if n_lo > 0:
            ax.scatter(
                xs[lo_mask],
                np.full(n_lo, edge_lo),
                c=layers[lo_mask],
                cmap=cmap,
                norm=norm,
                marker="v",
                s=44,
                alpha=0.95,
                edgecolors="#111111",
                linewidths=0.75,
                zorder=4,
            )
        if n_hi > 0:
            ax.scatter(
                xs[hi_mask],
                np.full(n_hi, edge_hi),
                c=layers[hi_mask],
                cmap=cmap,
                norm=norm,
                marker="^",
                s=44,
                alpha=0.95,
                edgecolors="#111111",
                linewidths=0.75,
                zorder=4,
            )

        c_stats = corr_map.get(ckey, {})
        rho_val = c_stats.get("spearman_rho", float("nan"))
        r_val = c_stats.get("pearson_r", float("nan"))
        stat_txt = f"n={len(rows)} | ρ={_fmt_num(rho_val)} | r={_fmt_num(r_val)}"
        ax.text(
            0.02,
            0.97,
            stat_txt,
            transform=ax.transAxes,
            fontsize=8.6,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#FFFFFF",
                "edgecolor": "#CCCCCC",
                "alpha": 0.90,
            },
            zorder=6,
        )

        clip_notes: list[str] = []
        if n_lo > 0:
            clip_notes.append(f"▼ {n_lo} outlier{'s' if n_lo != 1 else ''} < {y_min:.2f} pinned at edge")
        if n_hi > 0:
            clip_notes.append(f"▲ {n_hi} outlier{'s' if n_hi != 1 else ''} > {y_max:.2f} pinned at edge")
        if clip_notes:
            ax.text(
                0.02,
                0.02,
                "; ".join(clip_notes),
                transform=ax.transAxes,
                fontsize=8.0,
                va="bottom",
                ha="left",
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "#FFFDF7",
                    "edgecolor": "#CCCCCC",
                    "alpha": 0.92,
                },
                zorder=6,
            )

        _annotate_named_neurons(
            ax=ax,
            named_lookup=named_lookup,
            det_field=det_field,
            y_field=ckey,
            edge_lo=edge_lo,
            edge_hi=edge_hi,
            offsets_for_panel=offsets_map[panel_idx],
            include_layer_neuron=True,
        )

        ax.set_title(ptitle, fontsize=10.6, fontweight="bold", pad=8)
        ax.set_xlabel(
            f"Per-Neuron Detection Accuracy ({band_label}, own, gpt-6.1-terra@none)",
            fontsize=9.6,
            fontweight="bold",
        )
        if panel_idx == 0:
            ax.set_ylabel("Ulto85 Causal Substitution Recovery", fontsize=10.2, fontweight="bold")
        ax.set_xlim(-0.02, 1.03)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, linestyle=":", color="#D5D5D5", alpha=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_l.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.085),
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fontsize=7.9,
    )

    fig.suptitle(
        f"Qwen3-4B (Ulto85, n={len(rows)} neurons / {metrics.get('n_active_poles', 0)} active poles, {band_label}): "
        f"Two-Snippet Detection Accuracy vs. Causal Recovery",
        fontsize=11.4,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.50,
        0.014,
        ULTO85_CREDIT_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.905, top=0.865, bottom=0.145, wspace=0.08)
    if scatter_mappable is not None:
        cax = fig.add_axes([0.92, 0.155, 0.016, 0.70])
        cbar = fig.colorbar(scatter_mappable, cax=cax)
        cbar.set_label("MLP Layer (L0–L35)", fontsize=9.5, fontweight="bold")
        cbar.set_ticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_detection_vs_strong20(
    metrics: Mapping[str, Any],
    output_path: Path = DEFAULT_FIGURE_4B_STRONG20_PATH,
    y_clip: tuple[float, float] = (-0.45, 1.05),
) -> Path:
    """Plot single-panel top100 detection accuracy vs. strong20_rec (250 DPI)."""
    per_neuron = metrics.get("per_neuron", {})
    det_field = "det_top100_terra_mean"
    corr_key = "top100_terra_mean"
    rows = [
        rec
        for rec in per_neuron.values()
        if rec.get(det_field) is not None and math.isfinite(float(rec[det_field]))
    ]
    if not rows:
        det_field = "det_p999_terra_mean"
        corr_key = "p999_terra_mean"
        rows = [
            rec
            for rec in per_neuron.values()
            if rec.get(det_field) is not None and math.isfinite(float(rec[det_field]))
        ]

    xs = np.asarray([float(r[det_field]) for r in rows], dtype=np.float64)
    layers = np.asarray([int(r["layer"]) for r in rows], dtype=np.float64)
    y_raw = np.asarray([float(r["strong20_rec"]) for r in rows], dtype=np.float64)

    y_min, y_max = y_clip
    edge_lo = y_min + 0.02
    edge_hi = y_max - 0.02
    lo_mask = y_raw < y_min
    hi_mask = y_raw > y_max
    in_mask = (~lo_mask) & (~hi_mask)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=250)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=35.0)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.80, zorder=1)
    ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1.0, alpha=0.75, zorder=1)
    ax.axhline(0.50, color="#2E7D32", linestyle="--", linewidth=1.05, alpha=0.80, zorder=1)

    sc = ax.scatter(
        xs[in_mask],
        y_raw[in_mask],
        c=layers[in_mask],
        cmap=cmap,
        norm=norm,
        s=34,
        alpha=0.84,
        edgecolors="#FFFFFF",
        linewidths=0.35,
        zorder=3,
    )
    n_lo = int(np.sum(lo_mask))
    n_hi = int(np.sum(hi_mask))
    if n_lo > 0:
        ax.scatter(
            xs[lo_mask],
            np.full(n_lo, edge_lo),
            c=layers[lo_mask],
            cmap=cmap,
            norm=norm,
            marker="v",
            s=48,
            alpha=0.95,
            edgecolors="#111111",
            linewidths=0.75,
            zorder=4,
        )
    if n_hi > 0:
        ax.scatter(
            xs[hi_mask],
            np.full(n_hi, edge_hi),
            c=layers[hi_mask],
            cmap=cmap,
            norm=norm,
            marker="^",
            s=48,
            alpha=0.95,
            edgecolors="#111111",
            linewidths=0.75,
            zorder=4,
        )

    c_stats = metrics.get("correlations", {}).get(corr_key, {}).get("strong20_rec", {})
    stat_txt = (
        f"n={len(rows)} | ρ={_fmt_num(c_stats.get('spearman_rho'))} | "
        f"r={_fmt_num(c_stats.get('pearson_r'))}"
    )
    ax.text(
        0.02,
        0.97,
        stat_txt,
        transform=ax.transAxes,
        fontsize=8.8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFFFF", "edgecolor": "#CCCCCC", "alpha": 0.90},
        zorder=6,
    )
    clip_notes: list[str] = []
    if n_lo > 0:
        clip_notes.append(f"▼ {n_lo} outlier{'s' if n_lo != 1 else ''} < {y_min:.2f} pinned at edge")
    if n_hi > 0:
        clip_notes.append(f"▲ {n_hi} outlier{'s' if n_hi != 1 else ''} > {y_max:.2f} pinned at edge")
    if clip_notes:
        ax.text(
            0.02,
            0.02,
            "; ".join(clip_notes),
            transform=ax.transAxes,
            fontsize=8.2,
            va="bottom",
            ha="left",
            color="#333333",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFDF7", "edgecolor": "#CCCCCC", "alpha": 0.92},
            zorder=6,
        )

    named_lookup = {int(rec["k"]): rec for rec in metrics.get("named_neurons", [])}
    _annotate_named_neurons(
        ax=ax,
        named_lookup=named_lookup,
        det_field=det_field,
        y_field="strong20_rec",
        edge_lo=edge_lo,
        edge_hi=edge_hi,
        offsets_for_panel={
            "info": (-118, -16),
            "PCR": (-115, -32),
            "insurance": (-110, 26),
        },
        include_layer_neuron=True,
    )

    band_lbl = "top100" if det_field == "det_top100_terra_mean" else "p99.9+"
    ax.set_title(
        f"Qwen3-4B (n={len(rows)} neurons, {band_lbl}): Detection Accuracy vs. Strong-20% Causal Recovery",
        fontsize=10.8,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(
        f"Per-Neuron Detection Accuracy ({band_lbl}, own, gpt-6.1-terra@none)",
        fontsize=9.8,
        fontweight="bold",
    )
    ax.set_ylabel("Strong-20% Causal Recovery (strong20_rec)", fontsize=10.0, fontweight="bold")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, linestyle=":", color="#D5D5D5", alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.47,
        0.014,
        ULTO85_CREDIT_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.095, right=0.87, top=0.90, bottom=0.145)
    cax = fig.add_axes([0.89, 0.155, 0.020, 0.73])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("MLP Layer (L0–L35)", fontsize=9.5, fontweight="bold")
    cbar.set_ticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_main_plain_4b(
    metrics: Mapping[str, Any],
    output_path: Path = DEFAULT_FIGURE_4B_MAIN_PLAIN_PATH,
    y_clip: tuple[float, float] = (-0.85, 1.19),
) -> Path:
    """Plot plain single-panel main figure (top100 vs ordinary-text sim_cal_rec, NO stats box)."""
    per_neuron = metrics.get("per_neuron", {})
    det_field = "det_top100_terra_mean"
    rows = [
        rec
        for rec in per_neuron.values()
        if rec.get(det_field) is not None and math.isfinite(float(rec[det_field]))
    ]
    if not rows:
        det_field = "det_p999_terra_mean"
        rows = [
            rec
            for rec in per_neuron.values()
            if rec.get(det_field) is not None and math.isfinite(float(rec[det_field]))
        ]

    xs = np.asarray([float(r[det_field]) for r in rows], dtype=np.float64)
    layers = np.asarray([int(r["layer"]) for r in rows], dtype=np.float64)
    y_raw = np.asarray([float(r["sim_cal_rec"]) for r in rows], dtype=np.float64)

    # Exact factual counts in the shaded x >= 0.80 region
    det80_mask = xs >= 0.80 - 1e-12
    n_det80 = int(np.sum(det80_mask))
    n_ge50 = int(np.sum(det80_mask & (y_raw >= 0.50 - 1e-12)))
    n_gt0 = int(np.sum(det80_mask & (y_raw > 0.0)))
    n_le0 = int(np.sum(det80_mask & (y_raw <= 0.0)))

    y_min, y_max = y_clip
    edge_lo = y_min + 0.025
    edge_hi = y_max - 0.025
    lo_mask = y_raw < y_min
    hi_mask = y_raw > 1.05
    in_mask = (~lo_mask) & (~hi_mask)

    fig, ax = plt.subplots(figsize=(10.4, 6.2), dpi=250)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=35.0)

    # Lightly shaded detection >= 0.80 region
    ax.axvspan(0.80, 1.035, color="#E7EFF9", alpha=0.65, zorder=0)
    ax.axvline(0.80, color="#6B8AB8", linestyle="--", linewidth=1.1, alpha=0.85, zorder=1)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.80, zorder=1)
    ax.axhline(0.50, color="#2E7D32", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)

    sc = ax.scatter(
        xs[in_mask],
        y_raw[in_mask],
        c=layers[in_mask],
        cmap=cmap,
        norm=norm,
        s=36,
        alpha=0.85,
        edgecolors="#FFFFFF",
        linewidths=0.40,
        zorder=3,
    )

    n_lo = int(np.sum(lo_mask))
    n_hi = int(np.sum(hi_mask))
    if n_lo > 0:
        ax.scatter(
            xs[lo_mask],
            np.full(n_lo, edge_lo),
            c=layers[lo_mask],
            cmap=cmap,
            norm=norm,
            marker="v",
            s=50,
            alpha=0.95,
            edgecolors="#111111",
            linewidths=0.75,
            zorder=4,
        )
    if n_hi > 0:
        ax.scatter(
            xs[hi_mask],
            np.full(n_hi, edge_hi),
            c=layers[hi_mask],
            cmap=cmap,
            norm=norm,
            marker="^",
            s=50,
            alpha=0.95,
            edgecolors="#111111",
            linewidths=0.75,
            zorder=4,
        )

    # Exact factual counts cleanly centered INSIDE the shaded detection >= 0.80 region (above all dots)
    shaded_box_txt = (
        f"Detection >= 0.80\n"
        f"{n_det80} / {len(rows)} neurons\n"
        f">= 50% rec.: {n_ge50}\n"
        f"> 0% rec.: {n_gt0}\n"
        f"<= 0% rec.: {n_le0}"
    )
    ax.text(
        0.917,
        1.165,
        shaded_box_txt,
        fontsize=8.0,
        va="top",
        ha="center",
        color="#1B2A41",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "#FFFFFF",
            "edgecolor": "#6B8AB8",
            "linewidth": 1.0,
            "alpha": 0.96,
        },
        zorder=9,
    )

    # Clipped outlier edge note (strictly factual, no stats box)
    clip_notes: list[str] = []
    if n_lo > 0:
        clip_notes.append(f"▼ {n_lo} outlier{'s' if n_lo != 1 else ''} < {int(round(y_min * 100))}% pinned at edge")
    if n_hi > 0:
        clip_notes.append(f"▲ {n_hi} outlier{'s' if n_hi != 1 else ''} > 105% pinned at edge")
    if clip_notes:
        ax.text(
            0.02,
            0.02,
            "; ".join(clip_notes),
            transform=ax.transAxes,
            fontsize=8.2,
            va="bottom",
            ha="left",
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FFFDF7",
                "edgecolor": "#CCCCCC",
                "alpha": 0.92,
            },
            zorder=6,
        )

    # Three named neurons marked cleanly (zero overlap with shaded counts box!)
    named_lookup = {int(rec["k"]): rec for rec in metrics.get("named_neurons", [])}
    _annotate_named_neurons(
        ax=ax,
        named_lookup=named_lookup,
        det_field=det_field,
        y_field="sim_cal_rec",
        edge_lo=edge_lo,
        edge_hi=edge_hi,
        offsets_for_panel={
            "info": (-155, -18),
            "PCR": (-135, 22),
            "insurance": (-130, -28),
        },
        include_layer_neuron=True,
    )

    ax.set_xlabel("Detection test accuracy on top activations", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(
        "Share of the neuron's effect the label recovers on ordinary text",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.035)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([-0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.00])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle=":", color="#D5D5D5", alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.47,
        0.016,
        ULTO85_CREDIT_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.87, top=0.96, bottom=0.145)
    cax = fig.add_axes([0.89, 0.155, 0.019, 0.79])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("MLP Layer (L0–L35)", fontsize=9.5, fontweight="bold")
    cbar.set_ticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def render_summary_4b_md(metrics: Mapping[str, Any]) -> str:
    """Render strictly factual Markdown summary tables for the Qwen3-4B (Ulto85) audit."""
    lines: list[str] = []
    lines.append("# Exploratory Audit: Two-Snippet Detection vs. Ulto85 Causal Scores on Qwen3-4B")
    lines.append("")
    lines.append(
        f"- **Scope:** `{metrics['n_neurons']}` scored neurons from `Ulto85/neurons-kinda-clean` (MIT) "
        f"(`{metrics['n_unipolar_neurons']}` unipolar + `{metrics['n_bipolar_neurons']}` bipolar = "
        f"`{metrics['n_active_poles']}` active poles), `{metrics['n_snippets']:,}` core pairs + "
        f"`{metrics.get('n_top100_snippets', 0):,}` `top100` pairs, `{metrics['sweep_audit']['grand_total']['n_calls']:,}` judge rows."
    )
    lines.append(
        f"- **Primary judge:** `{PRIMARY_JUDGE_4B}` | **Secondary judge:** `{SECONDARY_JUDGE_4B}` | "
        f"**Figures:** `figures/qwen4b_main_plain.png`, `figures/qwen4b_top100_two_panel.png`, "
        f"`figures/qwen4b_detection_vs_causal.png`, `figures/qwen4b_detection_vs_strong20.png`"
    )
    wd = metrics.get("wrong_donors_summary")
    if wd:
        cs = wd.get("cosine_stats") or {}
        lines.append(
            f"- **Cross-layer `wrong` donors (`{wd.get('embed_model')}`):** cutoff `<= {wd.get('max_embed_cosine_cutoff')}`, "
            f"same-layer violations = `{wd.get('same_layer_violations')}`, relaxed = `{wd.get('threshold_relaxed_count')}`, "
            f"cosine `min={_fmt_num(cs.get('min'))}`, `med={_fmt_num(cs.get('median'))}`, "
            f"`p90={_fmt_num(cs.get('p90'))}`, `max={_fmt_num(cs.get('max'))}`, "
            f"embed cost = `${float(wd.get('embed_cost_usd') or 0.0):.6f}`."
        )
    lines.append("")

    # 1. Sweep overview, position bias, and USD spend
    lines.append("## 1. Sweep Summary, Position Bias, and API Spend")
    lines.append(
        "| Sweep | Judge Model | Band | Condition | Calls | `scored/unp/tie` | Mean Pole Acc [95% Neuron CI] | "
        "`P(pred=A)` | `P(pos=A)` | Mean In Tok | Mean Vis Out | Mean Reas Tok | Total USD |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    sw_map = metrics["sweep_audit"]["sweeps"]
    for name, sw in sw_map.items():
        a = sw["acc"]
        lines.append(
            f"| `{name}` | `{sw['judge_model']}` | `{sw['band']}` | `{sw['condition']}` | "
            f"{sw['n_calls']:,} | `{a['n_scored']}/{a['n_unparseable']}/{a['n_logprob_tie']}` | "
            f"{_fmt_ci(a['mean'], a['ci_low'], a['ci_high'])} | "
            f"{_fmt_num(sw['p_pred_A'])} | {_fmt_num(sw['p_pos_A'])} | "
            f"{sw['mean_prompt_tokens']:.1f} | {sw['mean_visible_output_tokens']:.1f} | "
            f"{sw['mean_reasoning_tokens']:.1f} | ${sw['total_cost_usd']:.4f} |"
        )
    gt = metrics["sweep_audit"]["grand_total"]
    lines.append(
        f"| **Grand Total** | — | — | — | **{gt['n_calls']:,}** | "
        f"**`{gt['n_scored']}/{gt['n_unparseable']}/{gt['n_logprob_tie']}`** | — | — | — | — | — | — | "
        f"**${gt['total_cost_usd']:.4f}** |"
    )
    lines.append("")

    # 2. Measurement reliability: split-half detection (p99.9+ & top100) + Sonnet vs Opus causal
    lines.append("## 2. Measurement Reliability (Detection Split-Half & Ulto85 Causal Sonnet vs. Opus)")
    lines.append("### 2.1 Detection Split-Half Reliability (`own`, `20` pairs/pole, `5/5` A/B per half)")
    lines.append(
        "| Sweep / Judge | Band | `n_neurons / n_poles` | Det. Halves `r` (`SB r`) | Det. Halves `ρ` (`SB ρ`) | "
        "100 Random `5/5` `SB r` Mean ± SD [p2.5, p97.5] | 100 Random `5/5` `SB ρ` Mean ± SD [p2.5, p97.5] |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for sh_key, sh_label in (
        (PRIMARY_JUDGE_4B, PRIMARY_JUDGE_4B),
        ("top100_terra", f"{PRIMARY_JUDGE_4B} (`top100`)"),
        (SECONDARY_JUDGE_4B, SECONDARY_JUDGE_4B),
    ):
        sh = metrics["split_half_reliability"].get(sh_key, {})
        det = sh.get("deterministic_halves", {})
        rnd = sh.get("random_balanced_splits", {})
        if not det:
            continue
        sbr = rnd["spearman_brown_r"]
        sbrho = rnd["spearman_brown_rho"]
        lines.append(
            f"| {sh_label} | `{sh.get('band', '')}` | `{sh['n_neurons']} / {sh['n_poles']}` | "
            f"{_fmt_num(det['pearson_r'])} ({_fmt_num(det['spearman_brown_r'])}) | "
            f"{_fmt_num(det['spearman_rho'])} ({_fmt_num(det['spearman_brown_rho'])}) | "
            f"{_fmt_num(sbr['mean'])} ± {_fmt_num(sbr['std'])} [{_fmt_num(sbr['p2_5'])}, {_fmt_num(sbr['p97_5'])}] | "
            f"{_fmt_num(sbrho['mean'])} ± {_fmt_num(sbrho['std'])} [{_fmt_num(sbrho['p2_5'])}, {_fmt_num(sbrho['p97_5'])}] |"
        )
    cj = metrics.get("cross_judge_agreement", {})
    if cj.get("n_overlap_pairs", 0) > 0:
        lines.append(
            f"- **Cross-judge agreement (`{PRIMARY_JUDGE_4B}` vs. `{SECONDARY_JUDGE_4B}`, `p99.9_plus` `own`):** "
            f"`n_pairs={cj['n_overlap_pairs']:,}`, pair exact agree = `{_fmt_num(cj['pair_exact_agreement'])}`, "
            f"Cohen's κ = `{_fmt_num(cj['pair_cohens_kappa'])}`, per-neuron `r = {_fmt_num(cj['neuron_pearson_r'])}`, "
            f"Spearman `ρ = {_fmt_num(cj['neuron_spearman_rho'])}`, mean `|Δ| = {_fmt_num(cj['neuron_mean_abs_diff'])}`."
        )
    lines.append("")

    lines.append("### 2.2 Causal Score Reliability: Sonnet (`e3_scores.json`) vs. Opus (`e3_scores_opus.json`)")
    co = metrics.get("causal_opus_reliability", {})
    if co.get("n_overlap", 0) > 0:
        lines.append(
            "| Metric | `n_overlap` | Spearman `ρ` (`p`) | Pearson `r` (`p`) | Median `|Δ|` | Mean `|Δ|` | Threshold Pass Agreement (`agree`, `κ`) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for mname, ms in co["metrics"].items():
            tparts = []
            for tl, td in ms["threshold_agreement"].items():
                tparts.append(
                    f"`{tl}`: {td['exact_agreement']:.3f} (κ={_fmt_num(td['cohens_kappa'])}, "
                    f"S={td['sonnet_pass_count']}, O={td['opus_pass_count']}, both={td['both_pass_count']})"
                )
            lines.append(
                f"| `{mname}` | {ms['n_overlap']} | "
                f"{_fmt_num(ms['spearman_rho'])} ({ms['spearman_p']:.2e}) | "
                f"{_fmt_num(ms['pearson_r'])} ({ms['pearson_p']:.2e}) | "
                f"{_fmt_num(ms['median_abs_diff'])} | {_fmt_num(ms['mean_abs_diff'])} | "
                f"{'; '.join(tparts)} |"
            )
    lines.append("")

    # 3. Side-by-side p99.9+ vs top100 correlations
    lines.append("## 3. Per-Neuron Correlations: Detection Accuracy (`p99.9+`, `top100`, `p97–99`) vs. Ulto85 Metrics")
    lines.append(
        "| Detection Predictor | Ulto85 Target | `n` | Spearman `ρ` [95% CI] | Spearman `p` | Pearson `r` [95% CI] | Pearson `p` |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for pname, tmap in metrics["correlations"].items():
        for target in CAUSAL_METRICS:
            c = tmap[target]
            sp_p = f"{c['spearman_p']:.2e}" if math.isfinite(c["spearman_p"]) else "—"
            pr_p = f"{c['pearson_p']:.2e}" if math.isfinite(c["pearson_p"]) else "—"
            lines.append(
                f"| `{pname}` | `{target}` | {c['n']} | "
                f"{_fmt_ci(c['spearman_rho'], c['spearman_ci_low'], c['spearman_ci_high'])} | {sp_p} | "
                f"{_fmt_ci(c['pearson_r'], c['pearson_ci_low'], c['pearson_ci_high'])} | {pr_p} |"
            )
    lines.append("")

    # 4. Pass-rate tables & 2x2 quadrants (above no_label floor + fixed thresholds, p99.9+ & top100)
    lines.append("## 4. Detection vs. Causal Pass Rates & 2×2 Quadrant Counts (`gpt-6.1-terra@none`, `n=278`)")
    matrix_sections = (
        ("mean_active_poles", "### 4.1 `p99.9+` — Primary Aggregation: Mean Over Active Poles"),
        ("top100_mean_active_poles", "### 4.2 `top100` — Primary Aggregation: Mean Over Active Poles"),
        ("dominant_pole", "### 4.3 `p99.9+` — Sensitivity Aggregation: Dominant Pole by `|p99.9|`"),
        ("top100_dominant_pole", "### 4.4 `top100` — Sensitivity Aggregation: Dominant Pole by `|p99.9|`"),
    )
    for agg_name, heading in matrix_sections:
        rows_mat = metrics["pass_matrices"].get(agg_name)
        if not rows_mat:
            continue
        lines.append(heading)
        lines.append(
            "| Causal Criterion | Detection Criterion | `Det Pass` (%) | `Causal Pass` (%) | `Both Pass` | `Det Only` | `Causal Only` | `Both Fail` | `P(Causal Fail | Det Pass)` |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows_mat:
            lines.append(
                f"| `{r['causal_criterion']}` | `{r['det_criterion']}` | "
                f"{r['n_det_pass']} ({100.0 * r['det_pass_rate']:.1f}%) | "
                f"{r['n_causal_pass']} ({100.0 * r['causal_pass_rate']:.1f}%) | "
                f"{r['both_pass']} | {r['det_only']} | {r['causal_only']} | {r['both_fail']} | "
                f"{_fmt_num(r['causal_fail_given_det_pass'])} |"
            )
        lines.append("")

    # 5. 50-neuron full_explanation vs pole_only
    lines.append("## 5. Stratified 50-Neuron `full_explanation` vs. `pole_only` (`p99.9_plus`, `gpt-6.1-terra@none`)")
    fe = metrics.get("full_explanation_50", {})
    if fe.get("n_pairs", 0) > 0:
        lines.append(
            "| Slice | `n_neurons / n_poles / n_pairs` | `full_explanation` [95% CI] | `pole_only` (`own`) [95% CI] | Paired `full - pole_only` [95% CI] |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for sname in ("overall", "unipolar", "bipolar"):
            st = fe[sname]
            f_a = st["full_explanation"]
            o_a = st["pole_only"]
            d_a = st["paired_full_minus_pole"]
            lines.append(
                f"| `{sname}` | `{d_a['n_neurons']} / {d_a['n_poles']} / {d_a['n_pairs']}` | "
                f"{_fmt_ci(f_a['mean'], f_a['ci_low'], f_a['ci_high'])} | "
                f"{_fmt_ci(o_a['mean'], o_a['ci_low'], o_a['ci_high'])} | "
                f"{_fmt_ci(d_a['mean'], d_a['ci_low'], d_a['ci_high'], signed=True)} |"
            )
        for ly in sorted(fe.get("by_layer", {}).keys()):
            st = fe["by_layer"][ly]
            f_a = st["full_explanation"]
            o_a = st["pole_only"]
            d_a = st["paired_full_minus_pole"]
            lines.append(
                f"| `L{ly}` | `{d_a['n_neurons']} / {d_a['n_poles']} / {d_a['n_pairs']}` | "
                f"{_fmt_ci(f_a['mean'], f_a['ci_low'], f_a['ci_high'])} | "
                f"{_fmt_ci(o_a['mean'], o_a['ci_low'], o_a['ci_high'])} | "
                f"{_fmt_ci(d_a['mean'], d_a['ci_low'], d_a['ci_high'], signed=True)} |"
            )
        lines.append("")

    # 6. Named neurons table (side-by-side p99.9+ and top100)
    lines.append("## 6. Named Neurons Readout (`info`, `PCR`, `insurance`)")
    lines.append(
        "| Name | `k` | Layer/Neuron | Polarity | `p99.9+` `terra` (`dom`) | `top100` `terra` (`dom`) | `p99.9+` `luna` | "
        "`wrong` | `no_label` | `full_expl` | `p97–99` `terra` | `sim_cal_rec` | `sim_cal_rec_active` | `strong20_rec` | `corr` | `mono` |"
    )
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nr in metrics.get("named_neurons", []):
        pp_note = ""
        if nr["n_active_poles"] >= 2 and nr.get("det_p999_terra_per_pole"):
            bits = ", ".join(
                f"{'pos' if pid.endswith('_pos') else 'neg'}={val:.3f}"
                for pid, val in sorted(nr["det_p999_terra_per_pole"].items())
            )
            pp_note = f" [{bits}]"
        t100_note = ""
        if nr["n_active_poles"] >= 2 and nr.get("det_top100_terra_per_pole"):
            bits = ", ".join(
                f"{'pos' if pid.endswith('_pos') else 'neg'}={val:.3f}"
                for pid, val in sorted(nr["det_top100_terra_per_pole"].items())
            )
            t100_note = f" [{bits}]"
        lines.append(
            f"| `{nr['short_name']}` | {nr['k']} | `L{nr['layer']} N{nr['neuron']}` | `{nr['polarity']}` | "
            f"{_fmt_num(nr['det_p999_terra_mean'])} ({_fmt_num(nr['det_p999_terra_dom'])}){pp_note} | "
            f"{_fmt_num(nr.get('det_top100_terra_mean'))} ({_fmt_num(nr.get('det_top100_terra_dom'))}){t100_note} | "
            f"{_fmt_num(nr['det_p999_luna_mean'])} | {_fmt_num(nr['det_p999_wrong_mean'])} | "
            f"{_fmt_num(nr['det_p999_no_label_mean'])} | {_fmt_num(nr['det_p999_full_expl_mean'])} | "
            f"{_fmt_num(nr['det_p97_99_terra_mean'])} | {_fmt_num(nr['sim_cal_rec'])} | "
            f"{_fmt_num(nr['sim_cal_rec_active'])} | {_fmt_num(nr['strong20_rec'])} | "
            f"{_fmt_num(nr['corr'])} | {_fmt_num(nr['mono'], digits=0)} |"
        )
    lines.append("")

    # 7. Item (1) Sparsity & info Diagnostic Table
    sp = metrics.get("sparsity_analysis")
    if sp:
        lines.append("## 7. Pole Sparsity Diagnostic (`p99.9_thresh < 0.20 * max_act_dir`)")
        fstats = sp.get("p99_9_frac_of_max_stats") or {}
        lines.append(
            f"- **Counts:** `{sp['n_poles_lt_20pct_max']} / {sp['n_active_poles']}` active poles "
            f"(`{100.0 * sp['frac_poles_lt_20pct_max']:.1f}%`, `{sp['n_neurons_any_pole_lt_20pct_max']}` neurons) "
            f"have `p99.9_thresh < 0.20 * max_act_dir`. "
            f"Corpus `p99.9 / max` distribution: `min={_fmt_num(fstats.get('min'))}`, "
            f"`p10={_fmt_num(fstats.get('p10'))}`, `p25={_fmt_num(fstats.get('p25'))}`, "
            f"`med={_fmt_num(fstats.get('median'))}`, `p75={_fmt_num(fstats.get('p75'))}`, "
            f"`max={_fmt_num(fstats.get('max'))}`."
        )
        nd = sp.get("named_neurons_diagnostic") or {}
        if nd:
            lines.append("")
            lines.append("### 7.1 Named Pole Sampled Positive Activation Check (`p99.9+` vs. `top100`)")
            lines.append(
                "| Name | Pole | `p99.9_thresh` | `max_act_dir` | `p99.9 / max` | `n_ge_20pct_max` | "
                "`p99.9+` sampled `>=20% max` | `top100` sampled `>=20% max` | `p99.9+` acc | `top100` acc |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for nm in ("info", "PCR", "insurance"):
                for row in nd.get(nm, []):
                    p999_s = row.get("p99_9_plus_sampled") or {}
                    t100_s = row.get("top100_sampled") or {}
                    p999_n = p999_s.get("n_pairs", p999_s.get("n_sampled", 0))
                    t100_n = t100_s.get("n_pairs", t100_s.get("n_sampled", 0))
                    lines.append(
                        f"| `{nm}` | `{row['pole_id']}` | {_fmt_num(row['p99.9_thresh'], 4)} | "
                        f"{_fmt_num(row['max_act_dir'], 4)} | {_fmt_num(row['p99.9_frac_of_max'], 4)} | "
                        f"{int(row['n_tokens_ge_20pct_max']):,} | "
                        f"`{p999_s.get('n_ge_20pct_max', 0)}/{p999_n}` | "
                        f"`{t100_s.get('n_ge_20pct_max', 0)}/{t100_n}` | "
                        f"{_fmt_num(row.get('det_p999_terra'))} | {_fmt_num(row.get('det_top100_terra'))} |"
                    )
        lines.append("")
        lines.append(
            f"### 7.2 All Poles with `p99.9_thresh < 0.20 * max_act_dir` (`n={len(sp['sparse_poles_lt_20pct_max'])}`)"
        )
        lines.append(
            "| Pole ID | `k` | Layer | `p99.9_thresh` | `max_act_dir` | `p99.9 / max` | "
            "`frac_pos` | `frac_ge_20pct_max` | `n_ge_20pct_max` | `p99.9+` acc | `top100` acc | `sim_cal_rec` |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sp["sparse_poles_lt_20pct_max"]:
            lines.append(
                f"| `{row['pole_id']}` | {int(row['k'])} | L{int(row['layer'])} | "
                f"{_fmt_num(row['p99.9_thresh'], 4)} | {_fmt_num(row['max_act_dir'], 4)} | "
                f"{_fmt_num(row['p99.9_frac_of_max'], 4)} | {_fmt_num(row['frac_pos'], 4)} | "
                f"{float(row['frac_ge_20pct_max']):.6f} | {int(row['n_tokens_ge_20pct_max']):,} | "
                f"{_fmt_num(row.get('det_p999_terra'))} | {_fmt_num(row.get('det_top100_terra'))} | "
                f"{_fmt_num(row.get('sim_cal_rec'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_analysis_4b(
    active_poles_path: Path = DEFAULT_ACTIVE_POLES_PATH,
    scores_path: Path = DEFAULT_SCORES_PATH,
    opus_scores_path: Path = DEFAULT_OPUS_SCORES_PATH,
    snippets_path: Path = DEFAULT_SNIPPETS_4B_PATH,
    judgements_path: Path = DEFAULT_JUDGEMENTS_4B_PATH,
    wrong_donors_path: Path = DEFAULT_WRONG_DONORS_4B_PATH,
    metrics_path: Path = DEFAULT_METRICS_4B_PATH,
    summary_path: Path = DEFAULT_SUMMARY_4B_PATH,
    figure_path: Path = DEFAULT_FIGURE_4B_PATH,
    top100_snippets_path: Path | None = None,
    top100_judgements_path: Path | None = None,
    sparsity_diag_path: Path | None = None,
    top100_two_panel_figure_path: Path | None = None,
    strong20_figure_path: Path | None = None,
    main_plain_figure_path: Path | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    n_random_splits: int = 100,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run full Qwen3-4B statistical audit and write metrics JSON, Markdown tables, and figures."""
    active_poles = load_active_poles(active_poles_path)
    scores_raw = json.loads(scores_path.read_text(encoding="utf-8"))
    opus_raw = (
        json.loads(opus_scores_path.read_text(encoding="utf-8"))
        if opus_scores_path.is_file()
        else None
    )
    snippets = (
        load_jsonl(snippets_path)
        if (snippets_path is not None and snippets_path.is_file())
        else []
    )
    judgements = load_jsonl(judgements_path)
    wrong_meta = (
        json.loads(wrong_donors_path.read_text(encoding="utf-8"))
        if wrong_donors_path.is_file()
        else None
    )
    t100_snips = (
        load_jsonl(top100_snippets_path)
        if (top100_snippets_path is not None and top100_snippets_path.is_file())
        else None
    )
    t100_judg = (
        load_jsonl(top100_judgements_path)
        if (top100_judgements_path is not None and top100_judgements_path.is_file())
        else None
    )
    sp_diag = (
        json.loads(sparsity_diag_path.read_text(encoding="utf-8"))
        if (sparsity_diag_path is not None and sparsity_diag_path.is_file())
        else None
    )

    metrics = compute_all_4b_metrics(
        active_poles=active_poles,
        scores_raw=scores_raw,
        opus_scores_raw=opus_raw,
        snippets=snippets,
        judgements=judgements,
        wrong_donors_meta=wrong_meta,
        top100_snippets=t100_snips,
        top100_judgements=t100_judg,
        sparsity_diag=sp_diag,
        n_resamples=n_resamples,
        n_random_splits=n_random_splits,
        seed=seed,
    )

    # 1. Existing p99.9+ two-panel figure (with Ulto85 credit footer)
    plot_detection_vs_causal(
        metrics,
        output_path=figure_path,
        det_field="det_p999_terra_mean",
        corr_key="p999_terra_mean",
        band_label="p99.9+",
    )

    # 2. New top100 figures whenever top100 judgements are present (or paths explicitly passed)
    has_top100 = bool(t100_judg)
    if has_top100 or top100_two_panel_figure_path is not None:
        t100_fig = top100_two_panel_figure_path or (figure_path.parent / "qwen4b_top100_two_panel.png")
        plot_detection_vs_causal(
            metrics,
            output_path=t100_fig,
            det_field="det_top100_terra_mean" if has_top100 else "det_p999_terra_mean",
            corr_key="top100_terra_mean" if has_top100 else "p999_terra_mean",
            band_label="top100" if has_top100 else "p99.9+",
        )
    if has_top100 or strong20_figure_path is not None:
        s20_fig = strong20_figure_path or (figure_path.parent / "qwen4b_detection_vs_strong20.png")
        plot_detection_vs_strong20(metrics, output_path=s20_fig)
    if has_top100 or main_plain_figure_path is not None:
        plain_fig = main_plain_figure_path or (figure_path.parent / "qwen4b_main_plain.png")
        plot_main_plain_4b(metrics, output_path=plain_fig)

    summary_md = render_summary_4b_md(metrics)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Qwen3-4B (Ulto85) detection vs. causal scores and render figures."
    )
    parser.add_argument("--active-poles", type=Path, default=DEFAULT_ACTIVE_POLES_PATH)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument("--opus-scores", type=Path, default=DEFAULT_OPUS_SCORES_PATH)
    parser.add_argument("--snippets", type=Path, default=DEFAULT_SNIPPETS_4B_PATH)
    parser.add_argument("--judgements", type=Path, default=DEFAULT_JUDGEMENTS_4B_PATH)
    parser.add_argument("--wrong-donors", type=Path, default=DEFAULT_WRONG_DONORS_4B_PATH)
    parser.add_argument("--top100-snippets", type=Path, default=DEFAULT_SNIPPETS_4B_TOP100_PATH)
    parser.add_argument("--top100-judgements", type=Path, default=DEFAULT_JUDGEMENTS_4B_TOP100_PATH)
    parser.add_argument("--sparsity-diag", type=Path, default=DEFAULT_SPARSITY_DIAG_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_4B_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_4B_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_4B_PATH)
    parser.add_argument(
        "--top100-two-panel-figure", type=Path, default=DEFAULT_FIGURE_4B_TOP100_TWO_PANEL_PATH
    )
    parser.add_argument(
        "--strong20-figure", type=Path, default=DEFAULT_FIGURE_4B_STRONG20_PATH
    )
    parser.add_argument(
        "--main-plain-figure", type=Path, default=DEFAULT_FIGURE_4B_MAIN_PLAIN_PATH
    )
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--random-splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    metrics = run_analysis_4b(
        active_poles_path=args.active_poles,
        scores_path=args.scores,
        opus_scores_path=args.opus_scores,
        snippets_path=args.snippets,
        judgements_path=args.judgements,
        wrong_donors_path=args.wrong_donors,
        metrics_path=args.metrics,
        summary_path=args.summary,
        figure_path=args.figure,
        top100_snippets_path=args.top100_snippets,
        top100_judgements_path=args.top100_judgements,
        sparsity_diag_path=args.sparsity_diag,
        top100_two_panel_figure_path=args.top100_two_panel_figure,
        strong20_figure_path=args.strong20_figure,
        main_plain_figure_path=args.main_plain_figure,
        n_resamples=args.bootstrap,
        n_random_splits=args.random_splits,
        seed=args.seed,
    )
    gt = metrics["sweep_audit"]["grand_total"]
    print(
        f"4B analysis complete: {metrics['n_neurons']} neurons / {metrics['n_active_poles']} active poles, "
        f"{gt['n_calls']:,} calls (${gt['total_cost_usd']:.4f}); "
        f"wrote {args.summary}, {args.metrics}, and figures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
