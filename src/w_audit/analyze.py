"""Statistical analysis, plotting, and 30-pole manual audit generator for Qwen3-1.7B evaluation.

Implements all pre-heldout methodological audit locks:
- Single primary metric: binary detection accuracy (sole accuracy metric).
- Strict exclusion & separate reporting of unparseable responses and exact
  logprob ties (`outcome in {"scored", "unparseable", "logprob_tie"}`).
- Neuron-clustered 95% bootstrap confidence intervals resampling over
  `(layer, neuron)` clusters (`B = 1,000`, `seed = 42`).
- Mandatory 3-way cohort reporting: Held-Out 80 (`in_pilot=False`, primary
  confirmatory), Pilot 20 (`in_pilot=True`), and All 100 pooled (Condition 3).
- Full rule baselines (`rarity`, `top_token`, `window_has_ge_p97`, plus
  `top_10_token` diagnostic).
- SwiGLU Q1/Q2 (gate-open) vs Q3 (double-negative) / Q4 (gate-closed) slices.
- Dead-zone vs Goodfire `<p90` top-band comparison & strictly Tier-0 same-layer
  nearest-w_gate neighbour distractor (`nearest_wgate_tier0`).
- Per-judge comparison, both-orders (`fwd`/`rev`) order-consistency, reasoning
  token & cost economics, first-position bias, and Chen structural checks.

Outputs:
- figures/w_band_accuracy.png
- results/summary_tables.md
- audit/raw_30_poles_worksheet.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from w_audit.config import (
    BANDS as ALL_BANDS,
    CONDITIONS as ALL_CONDITIONS,
    DEFAULT_JUDGE_MODEL as DEFAULT_PRIMARY_MODEL,
    NEIGHBOUR_DISTRACTOR_TYPE,
    SEED,
)
from w_audit.labeler import load_jsonl

DEFAULT_BOOTSTRAP_B: int = 1_000
LOGPROB_TIE_EPS: float = 1e-9

BAND_DISPLAY: dict[str, str] = {
    "p75_90": "p75–90",
    "p90_97": "p90–97",
    "p97_99.9": "p97–99.9",
    "p99.9_plus": "p99.9+",
}

METHOD_ORDER: tuple[str, ...] = (
    "own",
    "token_only",
    "no_label",
    "wrong",
    "rarity",
    "top_token",
    "window_has_ge_p97",
)

FLOOR_METHODS: tuple[str, ...] = (
    "wrong",
    "no_label",
    "token_only",
    "rarity",
    "top_token",
    "top_10_token",
    "window_has_ge_p97",
)
FLOOR_ALERT_THRESHOLD: float = 0.45
WEAK_BANDS: tuple[str, ...] = ("p75_90", "p90_97")

METHOD_STYLE: dict[str, dict[str, Any]] = {
    "own": {
        "label": "Own label (LLM)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.4,
        "offset": -0.09,
    },
    "token_only": {
        "label": "Token-only label (LLM)",
        "color": "#009E73",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.0,
        "offset": -0.06,
    },
    "no_label": {
        "label": "No label (LLM floor)",
        "color": "#E69F00",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.0,
        "offset": -0.03,
    },
    "wrong": {
        "label": "Wrong cross-layer label (LLM floor)",
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "v",
        "linewidth": 2.0,
        "offset": 0.00,
    },
    "rarity": {
        "label": "Rarity rule (no LLM)",
        "color": "#CC79A7",
        "linestyle": ":",
        "marker": "D",
        "linewidth": 2.0,
        "offset": 0.03,
    },
    "top_token": {
        "label": "Top-3 token rule (no LLM)",
        "color": "#56B4E9",
        "linestyle": ":",
        "marker": "P",
        "linewidth": 2.0,
        "offset": 0.06,
    },
    "window_has_ge_p97": {
        "label": "Window >=p97 rule (no LLM)",
        "color": "#7F7F7F",
        "linestyle": "-.",
        "marker": "X",
        "linewidth": 1.8,
        "offset": 0.09,
    },
}

_POLE_ID_RE = re.compile(r"^L(\d+)_N(\d+)_(pos|neg)$")


def extract_neuron_key(
    pole_id: str,
    poles_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[int, int]:
    """Resolve canonical neuron cluster key (layer, neuron_idx) from a pole_id."""
    if poles_by_id and pole_id in poles_by_id:
        p = poles_by_id[pole_id]
        return (int(p["layer"]), int(p["neuron"]))
    m = _POLE_ID_RE.match(pole_id)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (-1, abs(hash(pole_id)) % 1_000_000)


def classify_row_outcome(
    row: Mapping[str, Any],
    tie_eps: float = LOGPROB_TIE_EPS,
) -> Literal["scored", "unparseable", "logprob_tie"]:
    """Determine outcome, backwards-compatible with legacy JSONL rows."""
    explicit = row.get("outcome")
    if explicit in ("scored", "unparseable", "logprob_tie"):
        return explicit  # type: ignore[return-value]
    lp_a = row.get("logprob_A")
    lp_b = row.get("logprob_B")
    if lp_a is not None and lp_b is not None and abs(float(lp_a) - float(lp_b)) <= tie_eps:
        return "logprob_tie"
    if row.get("predicted_slot") not in ("A", "B") or row.get("correct") is None:
        return "unparseable"
    return "scored"


def neuron_clustered_bootstrap_accuracy(
    rows: Sequence[Mapping[str, Any]],
    poles_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    is_baseline: bool = False,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute binary accuracy over poles with NEURON-clustered 95% bootstrap CI.

    - Excludes 'unparseable' and 'logprob_tie' rows from both numerator and
      denominator and reports n_scored / n_unparseable / n_logprob_tie.
    - Resamples parent neurons `(layer, neuron)` with replacement, pulling both
      `pos` and `neg` poles of each sampled neuron together as one cluster.
    """
    scored_by_pole: dict[str, list[float]] = defaultdict(list)
    n_scored = 0
    n_unparseable = 0
    n_logprob_tie = 0

    for r in rows:
        pid = str(r["pole_id"])
        if is_baseline:
            scored_by_pole[pid].append(float(r["correct"]))
            n_scored += 1
        else:
            outcome = classify_row_outcome(r)
            if outcome == "scored":
                scored_by_pole[pid].append(float(r["correct"]))
                n_scored += 1
            elif outcome == "logprob_tie":
                n_logprob_tie += 1
            else:
                n_unparseable += 1

    pole_ids = sorted(pid for pid, vals in scored_by_pole.items() if vals)
    n_poles = len(pole_ids)
    if n_poles == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_neurons": 0,
            "n_poles": 0,
            "n_pairs": 0,
            "n_scored": n_scored,
            "n_unparseable": n_unparseable,
            "n_logprob_tie": n_logprob_tie,
        }

    # Group per-pole means under their parent neuron cluster (layer, neuron)
    neuron_clusters: dict[tuple[int, int], list[float]] = defaultdict(list)
    all_pole_means: list[float] = []
    for pid in pole_ids:
        pmean = float(np.mean(scored_by_pole[pid]))
        all_pole_means.append(pmean)
        nkey = extract_neuron_key(pid, poles_by_id)
        neuron_clusters[nkey].append(pmean)

    point = float(np.mean(all_pole_means))
    n_neurons = len(neuron_clusters)
    if n_neurons == 1:
        return {
            "mean": point,
            "ci_low": point,
            "ci_high": point,
            "n_neurons": 1,
            "n_poles": n_poles,
            "n_pairs": n_scored,
            "n_scored": n_scored,
            "n_unparseable": n_unparseable,
            "n_logprob_tie": n_logprob_tie,
        }

    ordered_keys = sorted(neuron_clusters.keys())
    cluster_sums = np.asarray([sum(neuron_clusters[k]) for k in ordered_keys], dtype=np.float64)
    cluster_sizes = np.asarray([len(neuron_clusters[k]) for k in ordered_keys], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_neurons, size=(n_resamples, n_neurons))
    boot_means = cluster_sums[idx].sum(axis=1) / cluster_sizes[idx].sum(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_neurons": n_neurons,
        "n_poles": n_poles,
        "n_pairs": n_scored,
        "n_scored": n_scored,
        "n_unparseable": n_unparseable,
        "n_logprob_tie": n_logprob_tie,
    }


def neuron_clustered_bootstrap_paired_diff(
    per_pole_diffs: Mapping[str, Sequence[float]],
    poles_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute unweighted mean paired difference over poles + 95% neuron-clustered bootstrap CI."""
    valid_poles = sorted(pid for pid, vals in per_pole_diffs.items() if vals)
    n_poles = len(valid_poles)
    total_pairs = sum(len(per_pole_diffs[pid]) for pid in valid_poles)
    if n_poles == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_neurons": 0,
            "n_poles": 0,
            "n_pairs": 0,
        }

    neuron_clusters: dict[tuple[int, int], list[float]] = defaultdict(list)
    all_pole_means: list[float] = []
    for pid in valid_poles:
        pmean = float(np.mean(per_pole_diffs[pid]))
        all_pole_means.append(pmean)
        nkey = extract_neuron_key(pid, poles_by_id)
        neuron_clusters[nkey].append(pmean)

    point = float(np.mean(all_pole_means))
    n_neurons = len(neuron_clusters)
    if n_neurons == 1:
        return {
            "mean": point,
            "ci_low": point,
            "ci_high": point,
            "n_neurons": 1,
            "n_poles": n_poles,
            "n_pairs": total_pairs,
        }

    ordered_keys = sorted(neuron_clusters.keys())
    cluster_sums = np.asarray([sum(neuron_clusters[k]) for k in ordered_keys], dtype=np.float64)
    cluster_sizes = np.asarray([len(neuron_clusters[k]) for k in ordered_keys], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_neurons, size=(n_resamples, n_neurons))
    boot_means = cluster_sums[idx].sum(axis=1) / cluster_sizes[idx].sum(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_neurons": n_neurons,
        "n_poles": n_poles,
        "n_pairs": total_pairs,
    }


def cohens_kappa(y1: Sequence[str], y2: Sequence[str]) -> float:
    """Compute binary Cohen's kappa on valid A/B predictions."""
    pairs = [(a, b) for a, b in zip(y1, y2) if a in ("A", "B") and b in ("A", "B")]
    n = len(pairs)
    if n == 0:
        return float("nan")
    p_o = sum(1 for a, b in pairs if a == b) / n
    p1_a = sum(1 for a, _ in pairs if a == "A") / n
    p2_a = sum(1 for _, b in pairs if b == "A") / n
    p_e = p1_a * p2_a + (1.0 - p1_a) * (1.0 - p2_a)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _fmt_ci(val: float, low: float, high: float) -> str:
    if not (math.isfinite(val) and math.isfinite(low) and math.isfinite(high)):
        return "—"
    return f"{val:.3f} [{low:.3f}, {high:.3f}]"


def _fmt_num(val: float | None, digits: int = 3) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:.{digits}f}"


def _judge_role_label(model: str, primary: str) -> str:
    """Assign explicit reporting role (gpt-6.2-sol is strictly 'third judge')."""
    if model == primary:
        return "primary judge"
    if "sol" in model.lower():
        return "third judge"
    return "secondary judge"


def _fmt_floor_ci(method: str, val: float, low: float, high: float) -> str:
    s = _fmt_ci(val, low, high)
    if (
        method in FLOOR_METHODS
        and math.isfinite(val)
        and val < FLOOR_ALERT_THRESHOLD
    ):
        return f"{s} [<0.45]"
    return s


def _fmt_floor_num(method: str, val: float | None, digits: int = 3) -> str:
    s = _fmt_num(val, digits=digits)
    if (
        method in FLOOR_METHODS
        and val is not None
        and math.isfinite(val)
        and val < FLOOR_ALERT_THRESHOLD
    ):
        return f"{s} [<0.45]"
    return s


def _compute_quadrant_slice_for_cohort(
    primary_rows: Sequence[Mapping[str, Any]],
    snippets_by_pair: Mapping[str, Mapping[str, Any]],
    allowed_pole_ids: set[str],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Compute SwiGLU quadrant slice restricted to a pole cohort."""
    quadrant_slice: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    quad_groups = ("Q1", "Q2", "Q3", "Q4", "gate_open", "gate_closed")
    for band in ALL_BANDS:
        quadrant_slice[band] = {}
        for qg in quad_groups:
            quadrant_slice[band][qg] = {}
            for cond in ALL_CONDITIONS:
                sub = []
                for r in primary_rows:
                    if (
                        str(r["pole_id"]) not in allowed_pole_ids
                        or str(r["distractor_type"]) != "dead_zone"
                        or str(r["band"]) != band
                        or str(r["condition"]) != cond
                    ):
                        continue
                    snip = snippets_by_pair.get(str(r["pair_id"]))
                    if not snip:
                        continue
                    pos_info = snip.get("pos", {})
                    q = str(pos_info.get("quadrant", ""))
                    gate_open = bool(pos_info.get("gate_open", q in ("Q1", "Q2")))
                    if (
                        (qg == q)
                        or (qg == "gate_open" and gate_open)
                        or (qg == "gate_closed" and not gate_open)
                    ):
                        sub.append(r)
                quadrant_slice[band][qg][cond] = neuron_clustered_bootstrap_accuracy(
                    sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )
    return quadrant_slice


def _compute_distractor_tables_for_cohort(
    primary_rows: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    allowed_pole_ids: set[str],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Compute dead_zone, goodfire_lt_p90, nearest_wgate_tier0, and matched-pole dead_zone."""
    distractor_tables: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for dist_type in ("dead_zone", "goodfire_lt_p90", NEIGHBOUR_DISTRACTOR_TYPE):
        distractor_tables[dist_type] = {}
        relevant_bands = (
            ("p99.9_plus",)
            if dist_type == "goodfire_lt_p90"
            else (("p99.9_plus", "p90_97") if dist_type == NEIGHBOUR_DISTRACTOR_TYPE else ALL_BANDS)
        )
        for band in relevant_bands:
            distractor_tables[dist_type][band] = {}
            for cond in ALL_CONDITIONS:
                sub = [
                    r
                    for r in primary_rows
                    if str(r["pole_id"]) in allowed_pole_ids
                    and str(r["distractor_type"]) == dist_type
                    and str(r["band"]) == band
                    and str(r["condition"]) == cond
                ]
                distractor_tables[dist_type][band][cond] = neuron_clustered_bootstrap_accuracy(
                    sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )
            for rule in ("rarity", "top_token", "window_has_ge_p97"):
                sub_b = [
                    b_row
                    for b_row in baselines
                    if str(b_row["pole_id"]) in allowed_pole_ids
                    and str(b_row["distractor_type"]) == dist_type
                    and str(b_row["band"]) == band
                    and str(b_row["rule"]) == rule
                ]
                distractor_tables[dist_type][band][rule] = neuron_clustered_bootstrap_accuracy(
                    sub_b, poles_by_id=poles_by_id, is_baseline=True, n_resamples=n_resamples, seed=seed
                )

    # Matched-pole dead_zone restricted to poles surviving in nearest_wgate_tier0 per band
    distractor_tables["dead_zone_matched_neighbour"] = {}
    for band in ("p99.9_plus", "p90_97"):
        nb_poles = {
            str(r["pole_id"])
            for r in primary_rows
            if str(r["pole_id"]) in allowed_pole_ids
            and str(r["distractor_type"]) == NEIGHBOUR_DISTRACTOR_TYPE
            and str(r["band"]) == band
        }
        if not nb_poles:
            nb_poles = {
                str(b_row["pole_id"])
                for b_row in baselines
                if str(b_row["pole_id"]) in allowed_pole_ids
                and str(b_row["distractor_type"]) == NEIGHBOUR_DISTRACTOR_TYPE
                and str(b_row["band"]) == band
            }
        distractor_tables["dead_zone_matched_neighbour"][band] = {}
        for cond in ALL_CONDITIONS:
            sub = [
                r
                for r in primary_rows
                if str(r["pole_id"]) in nb_poles
                and str(r["distractor_type"]) == "dead_zone"
                and str(r["band"]) == band
                and str(r["condition"]) == cond
            ]
            distractor_tables["dead_zone_matched_neighbour"][band][cond] = (
                neuron_clustered_bootstrap_accuracy(
                    sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )
            )
        for rule in ("rarity", "top_token", "window_has_ge_p97"):
            sub_b = [
                b_row
                for b_row in baselines
                if str(b_row["pole_id"]) in nb_poles
                and str(b_row["distractor_type"]) == "dead_zone"
                and str(b_row["band"]) == band
                and str(b_row["rule"]) == rule
            ]
            distractor_tables["dead_zone_matched_neighbour"][band][rule] = (
                neuron_clustered_bootstrap_accuracy(
                    sub_b, poles_by_id=poles_by_id, is_baseline=True, n_resamples=n_resamples, seed=seed
                )
            )
    return distractor_tables


def _compute_second_judge_agreement_for_cohort(
    primary_rows: Sequence[Mapping[str, Any]],
    j_rows: Sequence[Mapping[str, Any]],
    secondary_models: Sequence[str],
    allowed_pole_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute cross-judge agreement restricted to a pole cohort."""
    primary_by_pco = {
        (str(r["pair_id"]), str(r["condition"]), str(r.get("order") or "fwd")): r
        for r in primary_rows
        if str(r["pole_id"]) in allowed_pole_ids and classify_row_outcome(r) == "scored"
    }
    agreement: dict[str, dict[str, Any]] = {}
    for sec_model in secondary_models:
        sec_rows = [
            r
            for r in j_rows
            if str(r["judge_model"]) == sec_model
            and str(r["pole_id"]) in allowed_pole_ids
            and classify_row_outcome(r) == "scored"
        ]
        by_cond: dict[str, dict[str, Any]] = {}
        for scope in ("overall",) + ALL_CONDITIONS:
            matched_p1: list[str] = []
            matched_p2: list[str] = []
            pole_acc1: dict[str, list[float]] = defaultdict(list)
            pole_acc2: dict[str, list[float]] = defaultdict(list)
            for r2 in sec_rows:
                if scope != "overall" and str(r2["condition"]) != scope:
                    continue
                order_r2 = str(r2.get("order") or "fwd")
                r1 = primary_by_pco.get((str(r2["pair_id"]), str(r2["condition"]), order_r2))
                if r1 is None:
                    continue
                matched_p1.append(str(r1.get("predicted_slot") or ""))
                matched_p2.append(str(r2.get("predicted_slot") or ""))
                pid = str(r2["pole_id"])
                pole_acc1[pid].append(float(r1["correct"]))
                pole_acc2[pid].append(float(r2["correct"]))

            n_overlap = len(matched_p1)
            if n_overlap == 0:
                continue
            exact_agree = sum(
                1 for a, b in zip(matched_p1, matched_p2) if a and a == b
            ) / n_overlap
            kappa = cohens_kappa(matched_p1, matched_p2)

            shared_poles = sorted(pole_acc1.keys())
            v1 = np.asarray([np.mean(pole_acc1[p]) for p in shared_poles], dtype=np.float64)
            v2 = np.asarray([np.mean(pole_acc2[p]) for p in shared_poles], dtype=np.float64)
            if len(shared_poles) >= 2 and float(np.ptp(v1)) > 1e-12 and float(np.ptp(v2)) > 1e-12:
                rho_res = stats.spearmanr(v1, v2)
                rho = float(rho_res.statistic)
                pval = float(rho_res.pvalue)
            else:
                rho = float("nan")
                pval = float("nan")
            diffs = v2 - v1
            by_cond[scope] = {
                "n_overlap_pairs": n_overlap,
                "n_poles": len(shared_poles),
                "primary_mean_acc": float(np.mean(v1)),
                "secondary_mean_acc": float(np.mean(v2)),
                "exact_agreement": float(exact_agree),
                "cohens_kappa": float(kappa),
                "spearman_rho": rho,
                "spearman_pvalue": pval,
                "mean_signed_diff": float(np.mean(diffs)),
                "mean_abs_pole_diff": float(np.mean(np.abs(diffs))),
                "max_abs_pole_diff": float(np.max(np.abs(diffs))),
            }
        agreement[sec_model] = by_cond
    return agreement


def _compute_position_bias_for_cohort(
    j_rows: Sequence[Mapping[str, Any]],
    all_models: Sequence[str],
    allowed_pole_ids: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute first-position bias & exclusion accounting for a pole cohort."""
    position_bias: dict[str, dict[str, dict[str, Any]]] = {}
    for model in all_models:
        position_bias[model] = {}
        m_rows = [
            r
            for r in j_rows
            if str(r["judge_model"]) == model and str(r["pole_id"]) in allowed_pole_ids
        ]
        for scope in ("overall",) + ALL_CONDITIONS:
            sub = m_rows if scope == "overall" else [r for r in m_rows if str(r["condition"]) == scope]
            if not sub:
                continue
            n_total = len(sub)
            scored_sub = [r for r in sub if classify_row_outcome(r) == "scored"]
            n_scored = len(scored_sub)
            n_unparse = sum(1 for r in sub if classify_row_outcome(r) == "unparseable")
            n_ties = sum(1 for r in sub if classify_row_outcome(r) == "logprob_tie")

            pred_a = (
                sum(1 for r in scored_sub if r.get("predicted_slot") == "A") / n_scored
                if n_scored
                else float("nan")
            )
            pos_a_rate = (
                sum(1 for r in scored_sub if r.get("pos_slot") == "A") / n_scored
                if n_scored
                else float("nan")
            )
            slot_a = [float(r["correct"]) for r in scored_sub if r.get("pos_slot") == "A"]
            slot_b = [float(r["correct"]) for r in scored_sub if r.get("pos_slot") == "B"]
            acc_a = float(np.mean(slot_a)) if slot_a else float("nan")
            acc_b = float(np.mean(slot_b)) if slot_b else float("nan")
            position_bias[model][scope] = {
                "n_calls": n_total,
                "n_scored": n_scored,
                "n_unparseable": n_unparse,
                "n_logprob_tie": n_ties,
                "p_pred_A": pred_a,
                "p_pos_slot_A": pos_a_rate,
                "slot_A_acc": acc_a,
                "slot_B_acc": acc_b,
                "slot_A_minus_B": (
                    (acc_a - acc_b) if (math.isfinite(acc_a) and math.isfinite(acc_b)) else float("nan")
                ),
            }
    return position_bias


def compute_context_ablation(
    clean_primary_rows: Sequence[Mapping[str, Any]],
    unfiltered_judgements: Sequence[Mapping[str, Any]],
    unfiltered_snippets: Sequence[Mapping[str, Any]],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    primary_judge: str,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute exploratory clean vs. unfiltered weak-band ablation."""
    dedup_u: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in unfiltered_judgements:
        if str(row.get("judge_model")) != primary_judge:
            continue
        key = (
            str(row["pair_id"]),
            str(row["condition"]),
            str(row["judge_model"]),
            str(row.get("order") or "fwd"),
        )
        dedup_u[key] = row
    u_rows = list(dedup_u.values())

    pilot_ids = {pid for pid, p in poles_by_id.items() if p.get("in_pilot")}
    heldout_ids = {pid for pid, p in poles_by_id.items() if not p.get("in_pilot")}
    all_ids = set(poles_by_id.keys())
    cohort_sets = {
        "heldout_80": heldout_ids,
        "pilot_20": pilot_ids,
        "all_100": all_ids,
    }

    cohorts: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for cname, cset in cohort_sets.items():
        cohorts[cname] = {}
        for band in WEAK_BANDS:
            cohorts[cname][band] = {}
            for cond in ALL_CONDITIONS:
                clean_sub = [
                    r
                    for r in clean_primary_rows
                    if str(r["pole_id"]) in cset
                    and str(r["distractor_type"]) == "dead_zone"
                    and str(r["band"]) == band
                    and str(r["condition"]) == cond
                ]
                unfilt_sub = [
                    r
                    for r in u_rows
                    if str(r["pole_id"]) in cset
                    and str(r["band"]) == band
                    and str(r["condition"]) == cond
                ]
                c_stat = neuron_clustered_bootstrap_accuracy(
                    clean_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )
                u_stat = neuron_clustered_bootstrap_accuracy(
                    unfilt_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )
                delta = (
                    u_stat["mean"] - c_stat["mean"]
                    if (math.isfinite(u_stat["mean"]) and math.isfinite(c_stat["mean"]))
                    else float("nan")
                )
                cohorts[cname][band][cond] = {
                    "clean": c_stat,
                    "unfiltered": u_stat,
                    "delta_unfiltered_minus_clean": delta,
                }

    # Unfiltered snippet context-leakage summary
    leakage_by_band: dict[str, dict[str, float]] = {}
    for band in WEAK_BANDS:
        b_snips = [s for s in unfiltered_snippets if str(s.get("band")) == band]
        if b_snips:
            n_b = len(b_snips)
            ge_p90 = sum(1 for s in b_snips if bool(s.get("pos", {}).get("other_has_ge_p90")))
            ge_p97 = sum(1 for s in b_snips if bool(s.get("pos", {}).get("window_has_ge_p97")))
            slot_a = sum(1 for s in b_snips if str(s.get("pos_slot")) == "A")
            leakage_by_band[band] = {
                "n_pairs": n_b,
                "pos_other_ge_p90_rate": ge_p90 / n_b,
                "pos_window_ge_p97_rate": ge_p97 / n_b,
                "slot_A_rate": slot_a / n_b,
            }

    tot_usd = sum(float(r.get("cost_usd", 0.0)) for r in u_rows)
    n_calls = len(u_rows)
    n_scored = sum(1 for r in u_rows if classify_row_outcome(r) == "scored")
    n_unp = sum(1 for r in u_rows if classify_row_outcome(r) == "unparseable")
    n_tie = sum(1 for r in u_rows if classify_row_outcome(r) == "logprob_tie")

    return {
        "judge_model": primary_judge,
        "n_calls": n_calls,
        "n_scored": n_scored,
        "n_unparseable": n_unp,
        "n_logprob_tie": n_tie,
        "total_cost_usd": round(tot_usd, 6),
        "mean_cost_usd_per_call": (tot_usd / n_calls) if n_calls else 0.0,
        "leakage_by_band": leakage_by_band,
        "cohorts": cohorts,
    }


def collect_floors_below_045(
    cohort_tables: Mapping[str, Any],
    cohort_per_judge_by_band: Mapping[str, Any],
    cohort_distractor_tables: Mapping[str, Any],
    primary_judge: str,
    context_ablation: Mapping[str, Any] | None = None,
    j_rows: Sequence[Mapping[str, Any]] | None = None,
    ordered_models: Sequence[str] | None = None,
    cohort_sets: Mapping[str, set[str]] | None = None,
    poles_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Collect every floor method (< 0.45) across heldout_80 and pilot_20."""
    flagged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def _maybe_add(
        cohort: str,
        judge_or_rule: str,
        distractor: str,
        band: str,
        method: str,
        st: Mapping[str, Any],
    ) -> None:
        if method not in FLOOR_METHODS:
            return
        val = float(st.get("mean", float("nan")))
        if not (math.isfinite(val) and val < FLOOR_ALERT_THRESHOLD):
            return
        key = (cohort, judge_or_rule, distractor, band, method)
        if key in seen:
            return
        seen.add(key)
        flagged.append(
            {
                "cohort": cohort,
                "judge_model": judge_or_rule,
                "distractor_type": distractor,
                "band": band,
                "method": method,
                "mean": val,
                "ci_low": float(st.get("ci_low", float("nan"))),
                "ci_high": float(st.get("ci_high", float("nan"))),
                "n_scored": int(st.get("n_scored", 0)),
            }
        )

    for cohort in ("heldout_80", "pilot_20"):
        # Rule baselines on dead_zone
        ct = cohort_tables.get(cohort, {})
        for band in ALL_BANDS:
            for rule in ("rarity", "top_token", "top_10_token", "window_has_ge_p97"):
                st = ct.get(band, {}).get(rule, {}).get("acc")
                if st:
                    _maybe_add(cohort, "rule_baseline", "dead_zone", band, rule, st)

        # Every judge on dead_zone
        pj = cohort_per_judge_by_band.get(cohort, {})
        for model, bmap in pj.items():
            for band in ALL_BANDS:
                for cond in ("token_only", "no_label", "wrong"):
                    st = bmap.get(band, {}).get(cond)
                    if st:
                        _maybe_add(cohort, model, "dead_zone", band, cond, st)

        # Distractor arms on primary judge + rule baselines
        dt = cohort_distractor_tables.get(cohort, {})
        for dist_type in ("goodfire_lt_p90", NEIGHBOUR_DISTRACTOR_TYPE):
            for band, mmap in dt.get(dist_type, {}).items():
                for method, st in mmap.items():
                    if method in ("token_only", "no_label", "wrong"):
                        _maybe_add(cohort, primary_judge, dist_type, band, method, st)
                    elif method in ("rarity", "top_token", "window_has_ge_p97"):
                        _maybe_add(cohort, "rule_baseline", dist_type, band, method, st)

        # Distractor arms on ALL non-primary judges (e.g. gpt-6-luna@none on <p90 & nearest_wgate_tier0)
        if j_rows and ordered_models and cohort_sets and poles_by_id:
            cset = cohort_sets.get(cohort, set())
            for model in ordered_models:
                if model == primary_judge:
                    continue
                for dist_type, bands_for_dt in (
                    ("goodfire_lt_p90", ("p99.9_plus",)),
                    (NEIGHBOUR_DISTRACTOR_TYPE, ("p99.9_plus", "p90_97")),
                ):
                    for band in bands_for_dt:
                        for cond in ("token_only", "no_label", "wrong"):
                            sub = [
                                r
                                for r in j_rows
                                if str(r["judge_model"]) == model
                                and str(r["pole_id"]) in cset
                                and str(r["distractor_type"]) == dist_type
                                and str(r["band"]) == band
                                and str(r["condition"]) == cond
                            ]
                            if not sub:
                                continue
                            st_m = neuron_clustered_bootstrap_accuracy(
                                sub,
                                poles_by_id=poles_by_id,
                                is_baseline=False,
                                n_resamples=n_resamples,
                                seed=seed,
                            )
                            _maybe_add(cohort, model, dist_type, band, cond, st_m)

        # Exploratory unfiltered context ablation
        if context_ablation:
            u_judge = str(context_ablation.get("judge_model", primary_judge))
            u_coh = context_ablation.get("cohorts", {}).get(cohort, {})
            for band in WEAK_BANDS:
                for cond in ("token_only", "no_label", "wrong"):
                    st = u_coh.get(band, {}).get(cond, {}).get("unfiltered")
                    if st:
                        _maybe_add(
                            cohort,
                            u_judge,
                            "unfiltered_context_ablation (exploratory)",
                            band,
                            cond,
                            st,
                        )

    flagged.sort(
        key=lambda x: (
            0 if x["cohort"] == "heldout_80" else 1,
            x["distractor_type"],
            x["judge_model"],
            ALL_BANDS.index(x["band"]) if x["band"] in ALL_BANDS else 99,
            x["method"],
        )
    )
    return flagged


def compute_paired_condition_diffs(
    j_rows: Sequence[Mapping[str, Any]],
    cohort_sets: Mapping[str, set[str]],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    """Compute exact paired (own - floor) differences on identical pairs with neuron CIs."""
    out: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    for cname, cset in cohort_sets.items():
        if cname not in ("heldout_80", "pilot_20"):
            continue
        out[cname] = {}
        for model in models:
            m_dz = [
                r
                for r in j_rows
                if str(r["judge_model"]) == model
                and str(r["distractor_type"]) == "dead_zone"
                and str(r["pole_id"]) in cset
                and classify_row_outcome(r) == "scored"
            ]
            if not m_dz:
                continue
            by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {
                (str(r["pair_id"]), str(r.get("order") or "fwd"), str(r["condition"])): r
                for r in m_dz
            }
            pair_orders_by_band: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            seen_po: set[tuple[str, str, str]] = set()
            for r in m_dz:
                if str(r["condition"]) == "own":
                    po = (str(r["pair_id"]), str(r.get("order") or "fwd"), str(r["pole_id"]))
                    if po not in seen_po:
                        seen_po.add(po)
                        pair_orders_by_band[str(r["band"])].append(po)

            out[cname][model] = {}
            for band in ALL_BANDS:
                out[cname][model][band] = {}
                for floor_cond in ("token_only", "no_label", "wrong"):
                    diffs_by_pole: dict[str, list[float]] = defaultdict(list)
                    for pid_pair, order_str, pole_id in pair_orders_by_band.get(band, ()):
                        r_own = by_key.get((pid_pair, order_str, "own"))
                        r_fl = by_key.get((pid_pair, order_str, floor_cond))
                        if r_own is None or r_fl is None:
                            continue
                        diffs_by_pole[pole_id].append(float(r_own["correct"]) - float(r_fl["correct"]))
                    out[cname][model][band][f"own_minus_{floor_cond}"] = (
                        neuron_clustered_bootstrap_paired_diff(
                            diffs_by_pole,
                            poles_by_id=poles_by_id,
                            n_resamples=n_resamples,
                            seed=seed,
                        )
                    )
    return out


def compute_within_pole_gate_analysis(
    primary_rows: Sequence[Mapping[str, Any]],
    snippets_by_pair: Mapping[str, Mapping[str, Any]],
    cohort_sets: Mapping[str, set[str]],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute exact gate_open vs gate_closed counts, marginal CIs, and within-pole matched diffs."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cname in ("heldout_80", "pilot_20"):
        cset = cohort_sets[cname]
        out[cname] = {}
        for band in ALL_BANDS:
            own_open_rows: list[Mapping[str, Any]] = []
            own_closed_rows: list[Mapping[str, Any]] = []
            own_open_by_pole: dict[str, list[float]] = defaultdict(list)
            own_closed_by_pole: dict[str, list[float]] = defaultdict(list)
            nl_open_by_pole: dict[str, list[float]] = defaultdict(list)
            nl_closed_by_pole: dict[str, list[float]] = defaultdict(list)

            for r in primary_rows:
                if (
                    str(r["pole_id"]) not in cset
                    or str(r["distractor_type"]) != "dead_zone"
                    or str(r["band"]) != band
                ):
                    continue
                cond = str(r["condition"])
                if cond not in ("own", "no_label"):
                    continue
                snip = snippets_by_pair.get(str(r["pair_id"]))
                if not snip:
                    continue
                pos_info = snip.get("pos", {})
                q = str(pos_info.get("quadrant", ""))
                is_open = bool(pos_info.get("gate_open", q in ("Q1", "Q2")))
                if cond == "own":
                    if is_open:
                        own_open_rows.append(r)
                    else:
                        own_closed_rows.append(r)
                    if classify_row_outcome(r) == "scored":
                        pid = str(r["pole_id"])
                        if is_open:
                            own_open_by_pole[pid].append(float(r["correct"]))
                        else:
                            own_closed_by_pole[pid].append(float(r["correct"]))
                elif cond == "no_label" and classify_row_outcome(r) == "scored":
                    pid = str(r["pole_id"])
                    if is_open:
                        nl_open_by_pole[pid].append(float(r["correct"]))
                    else:
                        nl_closed_by_pole[pid].append(float(r["correct"]))

            n_open = sum(1 for r in own_open_rows if classify_row_outcome(r) == "scored")
            n_closed = sum(1 for r in own_closed_rows if classify_row_outcome(r) == "scored")
            n_tot = n_open + n_closed

            matched_poles = sorted(set(own_open_by_pole.keys()) & set(own_closed_by_pole.keys()))
            matched_own_diffs: dict[str, list[float]] = {
                pid: [float(np.mean(own_open_by_pole[pid])) - float(np.mean(own_closed_by_pole[pid]))]
                for pid in matched_poles
            }
            matched_open_means: dict[str, list[float]] = {
                pid: [float(np.mean(own_open_by_pole[pid]))] for pid in matched_poles
            }
            matched_closed_means: dict[str, list[float]] = {
                pid: [float(np.mean(own_closed_by_pole[pid]))] for pid in matched_poles
            }

            matched_nl_poles = sorted(set(nl_open_by_pole.keys()) & set(nl_closed_by_pole.keys()))
            matched_nl_diffs: dict[str, list[float]] = {
                pid: [float(np.mean(nl_open_by_pole[pid])) - float(np.mean(nl_closed_by_pole[pid]))]
                for pid in matched_nl_poles
            }

            out[cname][band] = {
                "n_open": n_open,
                "n_closed": n_closed,
                "n_total": n_tot,
                "pct_gate_closed": (100.0 * n_closed / n_tot) if n_tot else float("nan"),
                "marginal_open_own": neuron_clustered_bootstrap_accuracy(
                    own_open_rows, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                ),
                "marginal_closed_own": neuron_clustered_bootstrap_accuracy(
                    own_closed_rows, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                ),
                "n_matched_poles": len(matched_poles),
                "n_matched_neurons": len({extract_neuron_key(p, poles_by_id) for p in matched_poles}),
                "n_matched_open_pairs": sum(len(own_open_by_pole[p]) for p in matched_poles),
                "n_matched_closed_pairs": sum(len(own_closed_by_pole[p]) for p in matched_poles),
                "matched_open_own": neuron_clustered_bootstrap_paired_diff(
                    matched_open_means, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                ),
                "matched_closed_own": neuron_clustered_bootstrap_paired_diff(
                    matched_closed_means, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                ),
                "within_pole_diff_own": neuron_clustered_bootstrap_paired_diff(
                    matched_own_diffs, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                ),
                "within_pole_diff_no_label": neuron_clustered_bootstrap_paired_diff(
                    matched_nl_diffs, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                ),
            }
    return out


def diagnose_p999_gate_closed_heldout(
    primary_rows: Sequence[Mapping[str, Any]],
    snippets_by_pair: Mapping[str, Mapping[str, Any]],
    heldout_pole_ids: set[str],
    poles_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Factual diagnostic for the heldout_80 p99.9+ gate_closed subset."""
    own_rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    nl_by_pair: dict[str, float] = {}
    for r in primary_rows:
        if (
            str(r["pole_id"]) not in heldout_pole_ids
            or str(r["distractor_type"]) != "dead_zone"
            or str(r["band"]) != "p99.9_plus"
            or classify_row_outcome(r) != "scored"
        ):
            continue
        snip = snippets_by_pair.get(str(r["pair_id"]))
        if not snip:
            continue
        pos = snip.get("pos", {})
        q = str(pos.get("quadrant", ""))
        is_open = bool(pos.get("gate_open", q in ("Q1", "Q2")))
        if is_open:
            continue
        if str(r["condition"]) == "own":
            own_rows.append((r, snip))
        elif str(r["condition"]) == "no_label":
            nl_by_pair[str(r["pair_id"])] = float(r["correct"])

    if not own_rows:
        return {"n_pairs": 0}

    by_pole: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    by_layer: Counter[int] = Counter()
    by_quad: Counter[str] = Counter()
    tok_counts: Counter[str] = Counter()
    for r, snip in own_rows:
        pid = str(r["pole_id"])
        by_pole[pid].append((r, snip))
        by_layer[int(r["layer"])] += 1
        by_quad[str(snip.get("pos", {}).get("quadrant", ""))] += 1
        tok_counts[str(snip.get("pos", {}).get("target_token_str", ""))] += 1

    neurons = {extract_neuron_key(pid, poles_by_id) for pid in by_pole}
    pole_details = []
    for pid, items in sorted(by_pole.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        own_acc = float(np.mean([float(r["correct"]) for r, _ in items]))
        nl_vals = [nl_by_pair[str(r["pair_id"])] for r, _ in items if str(r["pair_id"]) in nl_by_pair]
        nl_acc = float(np.mean(nl_vals)) if nl_vals else float("nan")
        toks = [str(s.get("pos", {}).get("target_token_str", "")) for _, s in items]
        pole_details.append(
            {
                "pole_id": pid,
                "layer": int(items[0][0]["layer"]),
                "quadrant": str(items[0][1].get("pos", {}).get("quadrant", "")),
                "n_pairs": len(items),
                "own_acc": round(own_acc, 3),
                "no_label_acc": round(nl_acc, 3) if math.isfinite(nl_acc) else None,
                "sample_tokens": toks[:4],
            }
        )

    return {
        "n_pairs": len(own_rows),
        "n_distinct_poles": len(by_pole),
        "n_distinct_neurons": len(neurons),
        "pair_counts_by_layer": {f"L{k}": v for k, v in sorted(by_layer.items())},
        "pole_counts_by_layer": {
            f"L{ly}": sum(1 for pid in by_pole if int(by_pole[pid][0][0]["layer"]) == ly)
            for ly in sorted(by_layer.keys())
        },
        "quadrant_counts": dict(sorted(by_quad.items())),
        "top_tokens": tok_counts.most_common(8),
        "pole_details": pole_details,
    }


def compute_per_pole_top_band_distribution(
    primary_rows: Sequence[Mapping[str, Any]],
    cohort_sets: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Compute exact 0/6..6/6 per-pole own accuracy histogram at p99.9+ (dead_zone)."""
    out: dict[str, dict[str, Any]] = {}
    for cname in ("heldout_80", "pilot_20"):
        cset = cohort_sets[cname]
        by_pole: dict[str, list[float]] = defaultdict(list)
        for r in primary_rows:
            if (
                str(r["pole_id"]) in cset
                and str(r["distractor_type"]) == "dead_zone"
                and str(r["band"]) == "p99.9_plus"
                and str(r["condition"]) == "own"
                and classify_row_outcome(r) == "scored"
            ):
                by_pole[str(r["pole_id"])].append(float(r["correct"]))
        if not by_pole:
            out[cname] = {"n_poles": 0}
            continue
        counts = {f"{k}/6": 0 for k in range(7)}
        accs: list[float] = []
        n_ge_5 = 0
        n_le_3 = 0
        for pid in sorted(by_pole.keys()):
            vals = by_pole[pid]
            k_corr = int(round(sum(vals)))
            if len(vals) == 6 and 0 <= k_corr <= 6:
                counts[f"{k_corr}/6"] += 1
            acc = float(np.mean(vals))
            accs.append(acc)
            if acc >= (5.0 / 6.0) - 1e-9:
                n_ge_5 += 1
            if acc <= (3.0 / 6.0) + 1e-9:
                n_le_3 += 1
        arr = np.asarray(accs, dtype=np.float64)
        q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
        out[cname] = {
            "n_poles": len(accs),
            "counts": counts,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(med),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "n_ge_5_of_6": n_ge_5,
            "pct_ge_5_of_6": 100.0 * n_ge_5 / len(accs),
            "n_le_3_of_6": n_le_3,
            "pct_le_3_of_6": 100.0 * n_le_3 / len(accs),
        }
    return out


def compute_examples_as_label_analysis(
    clean_primary_rows: Sequence[Mapping[str, Any]],
    examples_judgements: Sequence[Mapping[str, Any]],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    primary_judge: str,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute exploratory examples_as_label vs own/token_only analysis."""
    dedup_e: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in examples_judgements:
        if str(row.get("judge_model")) != primary_judge:
            continue
        key = (
            str(row["pair_id"]),
            str(row["condition"]),
            str(row["judge_model"]),
            str(row.get("order") or "fwd"),
        )
        dedup_e[key] = row
    e_rows = list(dedup_e.values())

    ex_bands = ("p97_99.9", "p99.9_plus")
    pilot_ids = {pid for pid, p in poles_by_id.items() if p.get("in_pilot")}
    heldout_ids = {pid for pid, p in poles_by_id.items() if not p.get("in_pilot")}
    all_ids = set(poles_by_id.keys())
    cohort_sets = {"heldout_80": heldout_ids, "pilot_20": pilot_ids, "all_100": all_ids}

    clean_by_pair_cond: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(r["pair_id"]), str(r["condition"])): r
        for r in clean_primary_rows
        if str(r["distractor_type"]) == "dead_zone"
        and str(r["band"]) in ex_bands
        and str(r.get("order") or "fwd") == "fwd"
    }

    cohorts: dict[str, dict[str, dict[str, Any]]] = {}
    by_layer: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    layers = sorted({int(p["layer"]) for p in poles_by_id.values()})

    for cname, cset in cohort_sets.items():
        cohorts[cname] = {}
        for band in ex_bands:
            e_sub = [
                r for r in e_rows if str(r["pole_id"]) in cset and str(r["band"]) == band
            ]
            e_pair_ids = {str(r["pair_id"]) for r in e_sub}
            own_sub = [
                clean_by_pair_cond[(pid_pair, "own")]
                for pid_pair in sorted(e_pair_ids)
                if (pid_pair, "own") in clean_by_pair_cond
            ]
            tok_sub = [
                clean_by_pair_cond[(pid_pair, "token_only")]
                for pid_pair in sorted(e_pair_ids)
                if (pid_pair, "token_only") in clean_by_pair_cond
            ]

            diff_own_by_pole: dict[str, list[float]] = defaultdict(list)
            diff_tok_by_pole: dict[str, list[float]] = defaultdict(list)
            scored_e = [r for r in e_sub if classify_row_outcome(r) == "scored"]
            for r_ex in scored_e:
                pid_pair = str(r_ex["pair_id"])
                pole_id = str(r_ex["pole_id"])
                r_own = clean_by_pair_cond.get((pid_pair, "own"))
                if r_own is not None and classify_row_outcome(r_own) == "scored":
                    diff_own_by_pole[pole_id].append(float(r_ex["correct"]) - float(r_own["correct"]))
                r_tok = clean_by_pair_cond.get((pid_pair, "token_only"))
                if r_tok is not None and classify_row_outcome(r_tok) == "scored":
                    diff_tok_by_pole[pole_id].append(float(r_ex["correct"]) - float(r_tok["correct"]))

            p_pred_a = (
                sum(1 for r in scored_e if r.get("predicted_slot") == "A") / len(scored_e)
                if scored_e
                else float("nan")
            )
            p_pos_a = (
                sum(1 for r in scored_e if r.get("pos_slot") == "A") / len(scored_e)
                if scored_e
                else float("nan")
            )

            cohorts[cname][band] = {
                "examples_as_label": neuron_clustered_bootstrap_accuracy(
                    e_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                ),
                "own": neuron_clustered_bootstrap_accuracy(
                    own_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                ),
                "token_only": neuron_clustered_bootstrap_accuracy(
                    tok_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                ),
                "paired_examples_minus_own": neuron_clustered_bootstrap_paired_diff(
                    diff_own_by_pole, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                ),
                "paired_examples_minus_token_only": neuron_clustered_bootstrap_paired_diff(
                    diff_tok_by_pole, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                ),
                "p_pred_A": p_pred_a,
                "p_pos_slot_A": p_pos_a,
            }

        if cname in ("heldout_80", "pilot_20"):
            by_layer[cname] = {}
            for ly in layers:
                by_layer[cname][ly] = {}
                for band in ex_bands:
                    e_sub = [
                        r
                        for r in e_rows
                        if str(r["pole_id"]) in cset
                        and int(r["layer"]) == ly
                        and str(r["band"]) == band
                    ]
                    e_pids = {str(r["pair_id"]) for r in e_sub}
                    own_sub = [
                        clean_by_pair_cond[(pp, "own")]
                        for pp in sorted(e_pids)
                        if (pp, "own") in clean_by_pair_cond
                    ]
                    diff_own_ly: dict[str, list[float]] = defaultdict(list)
                    for r_ex in e_sub:
                        if classify_row_outcome(r_ex) != "scored":
                            continue
                        r_own = clean_by_pair_cond.get((str(r_ex["pair_id"]), "own"))
                        if r_own is not None and classify_row_outcome(r_own) == "scored":
                            diff_own_ly[str(r_ex["pole_id"])].append(
                                float(r_ex["correct"]) - float(r_own["correct"])
                            )
                    by_layer[cname][ly][band] = {
                        "examples_as_label": neuron_clustered_bootstrap_accuracy(
                            e_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                        ),
                        "own": neuron_clustered_bootstrap_accuracy(
                            own_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                        ),
                        "paired_examples_minus_own": neuron_clustered_bootstrap_paired_diff(
                            diff_own_ly, poles_by_id=poles_by_id, n_resamples=n_resamples, seed=seed
                        ),
                    }

    n_calls = len(e_rows)
    n_scored = sum(1 for r in e_rows if classify_row_outcome(r) == "scored")
    n_unp = sum(1 for r in e_rows if classify_row_outcome(r) == "unparseable")
    n_tie = sum(1 for r in e_rows if classify_row_outcome(r) == "logprob_tie")
    tot_in = sum(int(r.get("prompt_tokens", 0)) for r in e_rows)
    tot_out = sum(int(r.get("completion_tokens", 0)) for r in e_rows)
    tot_reas = sum(int(r.get("reasoning_tokens", 0)) for r in e_rows)
    tot_vis = sum(
        int(r.get("visible_output_tokens", max(0, int(r.get("completion_tokens", 0)) - int(r.get("reasoning_tokens", 0)))))
        for r in e_rows
    )
    tot_usd = sum(float(r.get("cost_usd", 0.0)) for r in e_rows)
    scored_all = [r for r in e_rows if classify_row_outcome(r) == "scored"]
    p_pred_a_all = (
        sum(1 for r in scored_all if r.get("predicted_slot") == "A") / len(scored_all)
        if scored_all
        else float("nan")
    )

    return {
        "judge_model": primary_judge,
        "n_calls": n_calls,
        "n_scored": n_scored,
        "n_unparseable": n_unp,
        "n_logprob_tie": n_tie,
        "prompt_tokens": tot_in,
        "completion_tokens": tot_out,
        "reasoning_tokens": tot_reas,
        "visible_output_tokens": tot_vis,
        "mean_prompt_tokens": (tot_in / n_calls) if n_calls else 0.0,
        "mean_visible_output_tokens": (tot_vis / n_calls) if n_calls else 0.0,
        "mean_reasoning_tokens": (tot_reas / n_calls) if n_calls else 0.0,
        "total_cost_usd": round(tot_usd, 6),
        "mean_cost_usd_per_call": (tot_usd / n_calls) if n_calls else 0.0,
        "p_pred_A": p_pred_a_all,
        "cohorts": cohorts,
        "by_layer": by_layer,
    }


def compute_p99_thresholds_from_cache(
    poles_list: Sequence[Mapping[str, Any]],
    cache_dir: Path,
) -> dict[str, Any]:
    """Compute exact directional p99 thresholds from data/cache/acts.npy matching extract.py."""
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    acts = np.load(cache_dir / "acts.npy", mmap_mode="r")
    pos_in_chunk = np.load(cache_dir / "pos_in_chunk.npy")
    token_ids = np.load(cache_dir / "token_ids.npy")

    non_bos_mask = pos_in_chunk > 0
    special_ids = meta.get("special_token_ids", [])
    if special_ids:
        non_bos_mask = non_bos_mask & (~np.isin(token_ids, special_ids))

    col_map = {
        (int(L), int(n)): idx for idx, (L, n) in enumerate(meta["col_to_neuron"])
    }

    pole_thresholds: dict[str, dict[str, Any]] = {}
    max_abs_err = 0.0
    all_strictly_ordered = True
    for p in poles_list:
        pid = str(p["pole_id"])
        layer = int(p["layer"])
        neuron = int(p["neuron"])
        sign = int(p["sign"])
        col = col_map[(layer, neuron)]
        d_valid = float(sign) * acts[non_bos_mask, col].astype(np.float32)
        q75, q90, q97, q99, q99_9 = (
            float(x) for x in np.quantile(d_valid, [0.75, 0.90, 0.97, 0.99, 0.999])
        )
        be = p.get("band_edges_dir", {})
        for k, v in (("p75", q75), ("p90", q90), ("p97", q97), ("p99.9", q99_9)):
            if k in be:
                max_abs_err = max(max_abs_err, abs(v - float(be[k])))
        if not (q97 < q99 < q99_9):
            all_strictly_ordered = False
        pole_thresholds[pid] = {
            "layer": layer,
            "neuron": neuron,
            "sign": sign,
            "in_pilot": bool(p.get("in_pilot", False)),
            "p97": q97,
            "p99": q99,
            "p99.9": q99_9,
        }

    return {
        "source": str((cache_dir / "acts.npy").as_posix()),
        "n_poles": len(pole_thresholds),
        "all_strictly_ordered": all_strictly_ordered,
        "max_abs_err_vs_poles_json": max_abs_err,
        "pole_thresholds": pole_thresholds,
    }


def load_or_compute_p99_thresholds(
    poles_list: Sequence[Mapping[str, Any]],
    cache_dir: Path | None = None,
    subband_json_path: Path | None = None,
) -> dict[str, Any] | None:
    """Compute p99 thresholds from cache_dir if available, else load from saved JSON."""
    if cache_dir is not None:
        required = (
            cache_dir / "meta.json",
            cache_dir / "acts.npy",
            cache_dir / "pos_in_chunk.npy",
            cache_dir / "token_ids.npy",
        )
        if all(p.is_file() for p in required):
            return compute_p99_thresholds_from_cache(poles_list, cache_dir)

    if subband_json_path is not None and subband_json_path.is_file():
        saved = json.loads(subband_json_path.read_text(encoding="utf-8"))
        thr = saved.get("pole_thresholds")
        if isinstance(thr, dict) and thr:
            all_ordered = all(
                float(v["p97"]) < float(v["p99"]) < float(v["p99.9"])
                for v in thr.values()
            )
            return {
                "source": str(subband_json_path.as_posix()),
                "n_poles": len(thr),
                "all_strictly_ordered": all_ordered,
                "max_abs_err_vs_poles_json": float(
                    saved.get("verification", {}).get("max_abs_err_vs_poles_json", 0.0)
                ),
                "pole_thresholds": thr,
            }
    return None


def compute_p97_p99_subband_split(
    poles_by_id: Mapping[str, Mapping[str, Any]],
    snippets: Sequence[Mapping[str, Any]],
    j_rows: Sequence[Mapping[str, Any]],
    examples_judgements: Sequence[Mapping[str, Any]] | None,
    p99_thresholds_payload: Mapping[str, Any] | None,
    primary_judge: str,
    ordered_models: Sequence[str],
    cohort_sets: Mapping[str, set[str]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    """Split dead_zone p97_99.9 pairs into p97_99 (< p99) and p99_99.9 (>= p99)."""
    if not p99_thresholds_payload:
        return None
    pole_thresholds = p99_thresholds_payload.get("pole_thresholds", {})
    if not pole_thresholds:
        return None

    pair_to_subband: dict[str, str] = {}
    pair_to_pole: dict[str, str] = {}
    out_of_range = 0
    missing_threshold = 0

    for s in snippets:
        if str(s.get("distractor_type")) != "dead_zone" or str(s.get("band")) != "p97_99.9":
            continue
        pid = str(s["pole_id"])
        pair_id = str(s["pair_id"])
        thr = pole_thresholds.get(pid)
        if thr is None:
            missing_threshold += 1
            continue
        pos_meta = s.get("pos", {})
        act_dir = float(pos_meta["act_dir"])
        q97 = float(thr["p97"])
        q99 = float(thr["p99"])
        q99_9 = float(thr["p99.9"])
        if not (q97 <= act_dir < q99_9):
            out_of_range += 1
        subband = "p97_99" if act_dir < q99 else "p99_99.9"
        pair_to_subband[pair_id] = subband
        pair_to_pole[pair_id] = pid

    if not pair_to_subband:
        return None

    # Deduplicate examples_as_label judgements for primary_judge
    dedup_ex: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in examples_judgements or []:
        if str(row.get("judge_model")) != primary_judge:
            continue
        if str(row.get("distractor_type")) != "dead_zone":
            continue
        pair_id = str(row["pair_id"])
        if pair_id not in pair_to_subband:
            continue
        key = (pair_id, str(row.get("condition", "examples_as_label")), str(row.get("order") or "fwd"))
        dedup_ex[key] = row
    ex_rows = list(dedup_ex.values())

    # Judges to report: primary + full-grid secondary (e.g. gpt-6-luna@none)
    judges_to_report: list[str] = [primary_judge]
    for jm in ordered_models:
        if jm != primary_judge and jm == "gpt-6-luna@none":
            judges_to_report.append(jm)
    if len(judges_to_report) == 1:
        # Also include any non-primary judge covering >= 80% of p97_99.9 pairs
        n_total_p97 = len(pair_to_subband)
        for jm in ordered_models:
            if jm == primary_judge:
                continue
            covered = {
                str(r["pair_id"])
                for r in j_rows
                if str(r["judge_model"]) == jm
                and str(r["condition"]) == "own"
                and str(r["pair_id"]) in pair_to_subband
            }
            if len(covered) >= int(0.8 * n_total_p97):
                judges_to_report.append(jm)

    subbands = ("p97_99", "p99_99.9")
    cohorts_out: dict[str, Any] = {}
    partition_checks: dict[str, dict[str, int | bool]] = {}

    for cname in ("heldout_80", "pilot_20", "all_100"):
        cset = cohort_sets[cname]
        cohort_pairs = [pid_pair for pid_pair, pole_id in pair_to_pole.items() if pole_id in cset]
        sub_pair_sets = {
            sb: {pair_id for pair_id in cohort_pairs if pair_to_subband[pair_id] == sb}
            for sb in subbands
        }
        n_low = len(sub_pair_sets["p97_99"])
        n_hi = len(sub_pair_sets["p99_99.9"])
        n_tot = len(cohort_pairs)
        partition_checks[cname] = {
            "n_p97_99": n_low,
            "n_p99_99.9": n_hi,
            "n_total": n_tot,
            "exact_sum_match": bool(n_low + n_hi == n_tot),
        }

        judges_out: dict[str, Any] = {}
        for jm in judges_to_report:
            jm_rows = [
                r
                for r in j_rows
                if str(r["judge_model"]) == jm
                and str(r["distractor_type"]) == "dead_zone"
                and str(r["pair_id"]) in pair_to_subband
                and str(r["pole_id"]) in cset
            ]
            sub_out: dict[str, Any] = {}
            for sb in subbands:
                sb_pairs = sub_pair_sets[sb]
                sb_poles = sorted({pair_to_pole[pair_id] for pair_id in sb_pairs})
                sb_neurons = sorted({extract_neuron_key(pid, poles_by_id) for pid in sb_poles})

                cond_stats: dict[str, Any] = {}
                cond_pair_correct: dict[str, dict[str, float]] = {}
                for cond in ALL_CONDITIONS:
                    c_rows = [
                        r
                        for r in jm_rows
                        if str(r["condition"]) == cond and str(r["pair_id"]) in sb_pairs
                    ]
                    cond_stats[cond] = neuron_clustered_bootstrap_accuracy(
                        c_rows,
                        poles_by_id=poles_by_id,
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                    by_pair: dict[str, list[float]] = defaultdict(list)
                    for r in c_rows:
                        if classify_row_outcome(r) == "scored":
                            by_pair[str(r["pair_id"])].append(float(r["correct"]))
                    cond_pair_correct[cond] = {
                        pair_id: float(np.mean(vals)) for pair_id, vals in by_pair.items()
                    }

                # Paired own - token_only on identical pairs in this sub-band
                per_pole_own_tok: dict[str, list[float]] = defaultdict(list)
                for pair_id in sorted(sb_pairs):
                    if pair_id in cond_pair_correct["own"] and pair_id in cond_pair_correct["token_only"]:
                        per_pole_own_tok[pair_to_pole[pair_id]].append(
                            cond_pair_correct["own"][pair_id] - cond_pair_correct["token_only"][pair_id]
                        )
                paired_own_minus_tok = neuron_clustered_bootstrap_paired_diff(
                    per_pole_own_tok,
                    poles_by_id=poles_by_id,
                    n_resamples=n_resamples,
                    seed=seed,
                )

                entry: dict[str, Any] = {
                    "n_pairs": len(sb_pairs),
                    "n_poles": len(sb_poles),
                    "n_neurons": len(sb_neurons),
                    "conditions": cond_stats,
                    "paired_own_minus_token_only": paired_own_minus_tok,
                }

                # Include examples_as_label on primary judge if available
                if jm == primary_judge and ex_rows:
                    sb_ex_rows = [r for r in ex_rows if str(r["pair_id"]) in sb_pairs]
                    cond_stats["examples_as_label"] = neuron_clustered_bootstrap_accuracy(
                        sb_ex_rows,
                        poles_by_id=poles_by_id,
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                    ex_by_pair: dict[str, list[float]] = defaultdict(list)
                    for r in sb_ex_rows:
                        if classify_row_outcome(r) == "scored":
                            ex_by_pair[str(r["pair_id"])].append(float(r["correct"]))
                    ex_pair_corr = {
                        pair_id: float(np.mean(vals)) for pair_id, vals in ex_by_pair.items()
                    }
                    per_pole_ex_own: dict[str, list[float]] = defaultdict(list)
                    for pair_id in sorted(sb_pairs):
                        if pair_id in ex_pair_corr and pair_id in cond_pair_correct["own"]:
                            per_pole_ex_own[pair_to_pole[pair_id]].append(
                                ex_pair_corr[pair_id] - cond_pair_correct["own"][pair_id]
                            )
                    entry["paired_examples_minus_own"] = neuron_clustered_bootstrap_paired_diff(
                        per_pole_ex_own,
                        poles_by_id=poles_by_id,
                        n_resamples=n_resamples,
                        seed=seed,
                    )

                sub_out[sb] = entry
            judges_out[jm] = sub_out
        cohorts_out[cname] = judges_out

    return {
        "primary_judge": primary_judge,
        "judges_reported": judges_to_report,
        "verification": {
            "source": str(p99_thresholds_payload.get("source", "")),
            "n_poles_with_thresholds": int(p99_thresholds_payload.get("n_poles", len(pole_thresholds))),
            "all_strictly_ordered": bool(p99_thresholds_payload.get("all_strictly_ordered", True)),
            "max_abs_err_vs_poles_json": float(
                p99_thresholds_payload.get("max_abs_err_vs_poles_json", 0.0)
            ),
            "out_of_range_count": out_of_range,
            "missing_threshold_count": missing_threshold,
            "partition_checks": partition_checks,
        },
        "pole_thresholds": dict(pole_thresholds),
        "cohorts": cohorts_out,
    }


def evaluate_heldout_criteria(
    cohort_tables: Mapping[str, Any],
    cohort_counts: Mapping[str, Any],
    cohort_quadrant_slice: Mapping[str, Any],
    cohort_distractor_tables: Mapping[str, Any],
    cohort_second_judge_agreement: Mapping[str, Any],
    second_judge_agreement_all: Mapping[str, Any],
    cohort_position_bias: Mapping[str, Any],
    primary_judge: str,
) -> dict[str, Any]:
    """Evaluate criterion checks on heldout_80.

    Every individual criterion gets its own row with outcome strictly 'HELD' or 'FAILED'
    (or 'PENDING' before heldout_80 judgements exist).
    """
    n_ho = int(cohort_counts.get("heldout_80", {}).get("judged_poles", 0))
    if n_ho == 0:
        return {
            "status": "PENDING",
            "n_held": 0,
            "n_failed": 0,
            "n_total": 0,
            "sub_predictions": [],
        }

    ho_bt = cohort_tables["heldout_80"]
    ho_q = cohort_quadrant_slice["heldout_80"]
    ho_dt = cohort_distractor_tables["heldout_80"]
    ho_agree = cohort_second_judge_agreement["heldout_80"]
    ho_pb = cohort_position_bias["heldout_80"]

    def _acc(band: str, cond: str) -> dict[str, Any]:
        return ho_bt[band][cond]["acc"]

    rows: list[dict[str, Any]] = []

    def _add(sub_id: str, group: str, criterion: str, observed: str, threshold: str, held: bool) -> None:
        rows.append(
            {
                "sub_id": sub_id,
                "group": group,
                "criterion": criterion,
                "observed": observed,
                "threshold": threshold,
                "outcome": "HELD" if held else "FAILED",
            }
        )

    # P1: Top-band token_only parity & weak-band drop
    own_top = _acc("p99.9_plus", "own")
    tok_top = _acc("p99.9_plus", "token_only")
    tok_75 = _acc("p75_90", "token_only")
    own_minus_tok_top = own_top["mean"] - tok_top["mean"]
    tok_drop = tok_top["mean"] - tok_75["mean"]
    _add(
        "P1.1", "P1", "token_only accuracy at p99.9+",
        _fmt_ci(tok_top["mean"], tok_top["ci_low"], tok_top["ci_high"]),
        ">= 0.68", tok_top["mean"] >= 0.68,
    )
    _add(
        "P1.2", "P1", "token_only 95% neuron CI lower bound at p99.9+",
        f"{tok_top['ci_low']:.3f}",
        "> 0.62", tok_top["ci_low"] > 0.62,
    )
    _add(
        "P1.3", "P1", "own - token_only gap at p99.9+",
        f"{own_minus_tok_top:+.3f}",
        "<= +0.08", own_minus_tok_top <= 0.08,
    )
    _add(
        "P1.4", "P1", "token_only drop from p99.9+ to p75-90",
        f"{tok_drop:+.3f}",
        ">= +0.12", tok_drop >= 0.12,
    )
    _add(
        "P1.5", "P1", "token_only accuracy at p75-90",
        _fmt_ci(tok_75["mean"], tok_75["ci_low"], tok_75["ci_high"]),
        "<= 0.62", tok_75["mean"] <= 0.62,
    )

    # P2: Unified-prompt wrong & no_label floors and band-dependent lift
    wr_vals = {b: _acc(b, "wrong")["mean"] for b in ALL_BANDS}
    nl_vals = {b: _acc(b, "no_label")["mean"] for b in ALL_BANDS}
    own_75 = _acc("p75_90", "own")
    lift_nl_top = own_top["mean"] - nl_vals["p99.9_plus"]
    lift_wr_top = own_top["mean"] - wr_vals["p99.9_plus"]
    lift_nl_75 = own_75["mean"] - nl_vals["p75_90"]
    nl_top_minus_75 = nl_vals["p99.9_plus"] - nl_vals["p75_90"]
    lift_shrink = lift_nl_top - lift_nl_75
    max_wr_band = max(ALL_BANDS, key=lambda b: wr_vals[b])
    _add(
        "P2.1", "P2", "wrong accuracy across all 4 bands",
        f"max={wr_vals[max_wr_band]:.3f} ({BAND_DISPLAY[max_wr_band]})",
        "<= 0.50 in all 4 bands", all(v <= 0.50 for v in wr_vals.values()),
    )
    _add(
        "P2.2", "P2", "no_label accuracy across all 4 bands",
        f"[{min(nl_vals.values()):.3f}, {max(nl_vals.values()):.3f}]",
        "within [0.35, 0.60]", all(0.35 <= v <= 0.60 for v in nl_vals.values()),
    )
    _add(
        "P2.3", "P2", "no_label(p99.9+) - no_label(p75-90)",
        f"{nl_top_minus_75:+.3f}",
        ">= +0.03", nl_top_minus_75 >= 0.03,
    )
    _add(
        "P2.4", "P2", "(own - no_label) lift at p99.9+",
        f"{lift_nl_top:+.3f}",
        ">= +0.20", lift_nl_top >= 0.20,
    )
    _add(
        "P2.5", "P2", "(own - wrong) lift at p99.9+",
        f"{lift_wr_top:+.3f}",
        ">= +0.25", lift_wr_top >= 0.25,
    )
    _add(
        "P2.6", "P2", "(own - no_label) lift reduction (p99.9+ vs p75-90)",
        f"{lift_shrink:+.3f} ({lift_nl_top:+.3f} -> {lift_nl_75:+.3f})",
        ">= +0.10", lift_shrink >= 0.10,
    )

    # P3: Sub-p97 accuracy drop on clean (<p90) snippets
    own_90 = _acc("p90_97", "own")
    drop_75 = own_top["mean"] - own_75["mean"]
    drop_90 = own_top["mean"] - own_90["mean"]
    _add(
        "P3.1", "P3", "own drop from p99.9+ to p75-90",
        f"{drop_75:+.3f} ({own_top['mean']:.3f} -> {own_75['mean']:.3f})",
        ">= +0.12", drop_75 >= 0.12,
    )
    _add(
        "P3.2", "P3", "own drop from p99.9+ to p90-97",
        f"{drop_90:+.3f} ({own_top['mean']:.3f} -> {own_90['mean']:.3f})",
        ">= +0.12", drop_90 >= 0.12,
    )
    _add(
        "P3.3", "P3", "own accuracy at p99.9+",
        _fmt_ci(own_top["mean"], own_top["ci_low"], own_top["ci_high"]),
        "within [0.70, 0.88]", 0.70 <= own_top["mean"] <= 0.88,
    )
    _add(
        "P3.4", "P3", "own accuracy at p75-90",
        _fmt_ci(own_75["mean"], own_75["ci_low"], own_75["ci_high"]),
        "<= 0.64", own_75["mean"] <= 0.64,
    )

    # P4: SwiGLU gate_open (Q1+Q2) vs gate_closed (Q3+Q4) divergence in weak bands
    go_75 = ho_q["p75_90"]["gate_open"]["own"]
    gc_75 = ho_q["p75_90"]["gate_closed"]["own"]
    go_90 = ho_q["p90_97"]["gate_open"]["own"]
    gc_90 = ho_q["p90_97"]["gate_closed"]["own"]
    gap_75 = go_75["mean"] - gc_75["mean"]
    gap_90 = go_90["mean"] - gc_90["mean"]
    _add(
        "P4.1", "P4", "p75-90 marginal (gate_open - gate_closed) own",
        f"{gap_75:+.3f} ({go_75['mean']:.3f} vs {gc_75['mean']:.3f})",
        ">= +0.08", gap_75 >= 0.08,
    )
    _add(
        "P4.2", "P4", "p90-97 marginal (gate_open - gate_closed) own",
        f"{gap_90:+.3f} ({go_90['mean']:.3f} vs {gc_90['mean']:.3f})",
        ">= +0.08", gap_90 >= 0.08,
    )
    _add(
        "P4.3", "P4", "p75-90 gate_closed own accuracy",
        _fmt_ci(gc_75["mean"], gc_75["ci_low"], gc_75["ci_high"]),
        "<= 0.56", gc_75["mean"] <= 0.56,
    )

    # P5: Distractors & cross-judge generalization
    gf_own = ho_dt["goodfire_lt_p90"]["p99.9_plus"]["own"]
    dz_own = ho_dt["dead_zone"]["p99.9_plus"]["own"]
    gf_delta = gf_own["mean"] - dz_own["mean"]

    nb_own = ho_dt[NEIGHBOUR_DISTRACTOR_TYPE]["p99.9_plus"]["own"]
    dz_matched_own = ho_dt["dead_zone_matched_neighbour"]["p99.9_plus"]["own"]
    nb_minus_matched = nb_own["mean"] - dz_matched_own["mean"]

    luna_model = next((m for m in ho_agree if "luna" in m.lower()), None)
    sol_model = next((m for m in second_judge_agreement_all if "sol" in m.lower()), None)
    rho_luna = (
        ho_agree[luna_model]["own"]["spearman_rho"]
        if (luna_model and "own" in ho_agree.get(luna_model, {}))
        else float("nan")
    )
    rho_sol_30 = (
        second_judge_agreement_all[sol_model]["own"]["spearman_rho"]
        if (sol_model and "own" in second_judge_agreement_all.get(sol_model, {}))
        else float("nan")
    )
    rho_sol_ho10 = (
        ho_agree[sol_model]["own"]["spearman_rho"]
        if (sol_model and "own" in ho_agree.get(sol_model, {}))
        else float("nan")
    )
    terra_nl_bias = abs(ho_pb.get(primary_judge, {}).get("no_label", {}).get("p_pred_A", float("nan")) - 0.50)
    luna_nl_bias = (
        abs(ho_pb.get(luna_model, {}).get("no_label", {}).get("p_pred_A", float("nan")) - 0.50)
        if luna_model
        else float("nan")
    )

    if math.isfinite(gf_delta):
        _add(
            "P5.1", "P5", "|own(goodfire_lt_p90) - own(dead_zone)| at p99.9+",
            f"{abs(gf_delta):.3f} ({gf_own['mean']:.3f} vs {dz_own['mean']:.3f}, delta={gf_delta:+.3f})",
            "<= 0.06", abs(gf_delta) <= 0.06,
        )
    if math.isfinite(nb_minus_matched):
        _add(
            "P5.2", "P5", "nearest_wgate_tier0 - matched-pole dead_zone (own, p99.9+)",
            f"{nb_minus_matched:+.3f} ({nb_own['mean']:.3f} vs {dz_matched_own['mean']:.3f})",
            "<= -0.02", nb_minus_matched <= -0.02,
        )
    if math.isfinite(rho_luna):
        _add(
            "P5.3", "P5", "Spearman rho(terra, luna) per-pole own on heldout_80",
            f"{rho_luna:.3f} (n=80 poles)",
            ">= 0.45", rho_luna >= 0.45,
        )
    if math.isfinite(rho_sol_30):
        _add(
            "P5.4", "P5", "Spearman rho(terra, sol) per-pole own on 30-pole subset",
            f"{rho_sol_30:.3f} (n=30; descriptive heldout_10 rho={_fmt_num(rho_sol_ho10)}, n=10 poles / 5 neurons)",
            ">= 0.60", rho_sol_30 >= 0.60,
        )
    if math.isfinite(terra_nl_bias) and math.isfinite(luna_nl_bias):
        _add(
            "P5.5", "P5", "terra no_label |P(pred=A) - 0.50| < 0.10 and < luna",
            f"terra={terra_nl_bias:.3f} vs luna={luna_nl_bias:.3f}",
            "< 0.10 and < luna", terra_nl_bias < 0.10 and terra_nl_bias < luna_nl_bias,
        )

    n_held = sum(1 for r in rows if r["outcome"] == "HELD")
    n_failed = sum(1 for r in rows if r["outcome"] == "FAILED")
    return {
        "status": "EVALUATED",
        "n_held": n_held,
        "n_failed": n_failed,
        "n_total": len(rows),
        "sub_predictions": rows,
    }


def _select_primary_judge(
    judgements: Sequence[Mapping[str, Any]],
    requested: str | None = None,
) -> str:
    if requested:
        return requested
    models = Counter(str(r["judge_model"]) for r in judgements)
    # Prefer current GPT-6 primary candidates if present and no explicit flag given
    for preferred in ("gpt-6.1-terra@none", "gpt-6.2-sol@low", DEFAULT_PRIMARY_MODEL):
        if preferred in models:
            return preferred
    if models:
        return models.most_common(1)[0][0]
    return DEFAULT_PRIMARY_MODEL


def _build_cohort_band_table(
    primary_rows: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    allowed_pole_ids: set[str],
    poles_by_id: Mapping[str, Mapping[str, Any]],
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build 4-band x method table restricted to a pole cohort (neuron-clustered CIs)."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for band in ALL_BANDS:
        table[band] = {}
        for cond in ALL_CONDITIONS:
            sub = [
                r
                for r in primary_rows
                if str(r["pole_id"]) in allowed_pole_ids
                and str(r["distractor_type"]) == "dead_zone"
                and str(r["band"]) == band
                and str(r["condition"]) == cond
            ]
            table[band][cond] = {
                "acc": neuron_clustered_bootstrap_accuracy(
                    sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )
            }

        for rule in ("rarity", "top_token", "window_has_ge_p97", "top_10_token"):
            sub_b = [
                b_row
                for b_row in baselines
                if str(b_row["pole_id"]) in allowed_pole_ids
                and str(b_row["distractor_type"]) == "dead_zone"
                and str(b_row["band"]) == band
                and str(b_row["rule"]) == rule
            ]
            if sub_b or rule in ("rarity", "top_token", "window_has_ge_p97"):
                table[band][rule] = {
                    "acc": neuron_clustered_bootstrap_accuracy(
                        sub_b, poles_by_id=poles_by_id, is_baseline=True, n_resamples=n_resamples, seed=seed
                    )
                }
    return table


def compute_all_metrics(
    poles_data: Mapping[str, Any],
    snippets: Sequence[Mapping[str, Any]],
    judgements: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    structural_checks: Mapping[str, Any] | None = None,
    primary_judge: str | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
    weak_band_survival: Mapping[str, Any] | None = None,
    unfiltered_judgements: Sequence[Mapping[str, Any]] | None = None,
    unfiltered_snippets: Sequence[Mapping[str, Any]] | None = None,
    examples_judgements: Sequence[Mapping[str, Any]] | None = None,
    p99_thresholds_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all structured evaluation metrics (binary accuracy + neuron-clustered CIs)."""
    poles_by_id = {str(p["pole_id"]): p for p in poles_data.get("poles", [])}
    snippets_by_pair = {str(s["pair_id"]): s for s in snippets}
    primary = _select_primary_judge(judgements, primary_judge)

    # Deduplicate judgements by canonical key including order (last write wins)
    dedup_j: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in judgements:
        key = (
            str(row["pair_id"]),
            str(row["condition"]),
            str(row["judge_model"]),
            str(row.get("order") or "fwd"),
        )
        dedup_j[key] = row
    j_rows = list(dedup_j.values())

    primary_rows = [r for r in j_rows if str(r["judge_model"]) == primary]
    all_models = sorted({str(r["judge_model"]) for r in j_rows})
    secondary_models = [m for m in all_models if m != primary]

    # 1. Main 4-band x method tables split into 3 mandatory cohorts (Condition 3)
    pilot_pole_ids = {pid for pid, p in poles_by_id.items() if p.get("in_pilot")}
    heldout_pole_ids = {pid for pid, p in poles_by_id.items() if not p.get("in_pilot")}
    all_pole_ids = set(poles_by_id.keys())
    primary_judged_poles = {
        str(r["pole_id"]) for r in primary_rows if str(r["distractor_type"]) == "dead_zone"
    }

    cohort_sets: dict[str, set[str]] = {
        "heldout_80": heldout_pole_ids,
        "pilot_20": pilot_pole_ids,
        "all_100": all_pole_ids,
    }
    cohort_tables: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        cname: _build_cohort_band_table(
            primary_rows=primary_rows,
            baselines=baselines,
            allowed_pole_ids=cset,
            poles_by_id=poles_by_id,
            n_resamples=n_resamples,
            seed=seed,
        )
        for cname, cset in cohort_sets.items()
    }
    cohort_counts: dict[str, dict[str, int]] = {
        cname: {
            "total_poles": len(cset),
            "judged_poles": len(cset & primary_judged_poles),
        }
        for cname, cset in cohort_sets.items()
    }
    main_table = cohort_tables["all_100"]

    # 2. Layer breakdown & Layer-4 positive control pilot gate
    layers = sorted({int(p["layer"]) for p in poles_by_id.values()})
    layer_breakdown: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for layer in layers:
        layer_breakdown[layer] = {}
        for band in ALL_BANDS:
            layer_breakdown[layer][band] = {}
            for cond in ALL_CONDITIONS:
                sub = [
                    r
                    for r in primary_rows
                    if int(r["layer"]) == layer
                    and str(r["distractor_type"]) == "dead_zone"
                    and str(r["band"]) == band
                    and str(r["condition"]) == cond
                ]
                layer_breakdown[layer][band][cond] = neuron_clustered_bootstrap_accuracy(
                    sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                )

    l4_own_top = layer_breakdown.get(4, {}).get("p99.9_plus", {}).get("own", {}).get("mean", float("nan"))
    l4_wrong_top = layer_breakdown.get(4, {}).get("p99.9_plus", {}).get("wrong", {}).get("mean", float("nan"))
    l4_nolabel_top = layer_breakdown.get(4, {}).get("p99.9_plus", {}).get("no_label", {}).get("mean", float("nan"))
    deep_sub = [
        r
        for r in primary_rows
        if int(r["layer"]) != 4
        and str(r["distractor_type"]) == "dead_zone"
        and str(r["band"]) == "p99.9_plus"
        and str(r["condition"]) == "own"
    ]
    deep_own_top = neuron_clustered_bootstrap_accuracy(
        deep_sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
    )["mean"]

    finite_floors = [v for v in (l4_wrong_top, l4_nolabel_top) if math.isfinite(v)]
    l4_floor_gap = (
        (l4_own_top - max(finite_floors))
        if (math.isfinite(l4_own_top) and finite_floors)
        else float("nan")
    )
    min_floor_margin = 0.10

    if not math.isfinite(l4_own_top):
        pilot_verdict = "PENDING (no Layer-4 p99.9_plus own-label judgements yet)"
    elif l4_own_top < 0.70:
        pilot_verdict = "STOP — HARNESS FAILURE (Layer-4 positive control < 0.70 at p99.9+; debug hooks/windowing/prompts)"
    elif math.isfinite(l4_floor_gap) and l4_floor_gap < min_floor_margin:
        pilot_verdict = (
            f"STOP / FLOOR WARNING (Layer-4 own={l4_own_top:.3f} exceeds max(wrong, no_label) "
            f"by only {l4_floor_gap:+.3f} < +{min_floor_margin:.2f}; inspect shortcuts before scaling)"
        )
    elif math.isfinite(deep_own_top) and deep_own_top < 0.65 and l4_own_top >= 0.80:
        pilot_verdict = "MODEL RESULT (Layer-4 control >= 0.80 and beats floors, while deeper layers < 0.65; spot-check 5 deep poles then run core)"
    elif l4_own_top >= 0.80:
        pilot_verdict = "GO (Layer-4 positive control >= 0.80 and cleanly beats wrong/no_label floors)"
    else:
        pilot_verdict = "BORDERLINE (0.70 <= Layer-4 control < 0.80; inspect raw Layer-4 pilot snippets before scaling)"

    pilot_gate = {
        "l4_own_p99.9_plus": l4_own_top,
        "l4_wrong_p99.9_plus": l4_wrong_top,
        "l4_no_label_p99.9_plus": l4_nolabel_top,
        "l4_floor_gap": l4_floor_gap,
        "deep_layers_own_p99.9_plus": deep_own_top,
        "verdict": pilot_verdict,
    }

    # 3. SwiGLU Quadrant slice (Q1/Q2 gate-open vs Q3 double-negative / Q4 gate-closed)
    cohort_quadrant_slice: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {
        cname: _compute_quadrant_slice_for_cohort(
            primary_rows=primary_rows,
            snippets_by_pair=snippets_by_pair,
            allowed_pole_ids=cset,
            poles_by_id=poles_by_id,
            n_resamples=n_resamples,
            seed=seed,
        )
        for cname, cset in cohort_sets.items()
    }
    quadrant_slice = cohort_quadrant_slice["all_100"]

    corpus_quad_by_band: dict[str, dict[str, float]] = {}
    for band in ALL_BANDS:
        sums = defaultdict(list)
        for p in poles_by_id.values():
            qmap = p.get("quadrant_shares_by_band", {}).get(band)
            if isinstance(qmap, Mapping):
                for q in ("Q1", "Q2", "Q3", "Q4"):
                    if q in qmap:
                        sums[q].append(float(qmap[q]))
        corpus_quad_by_band[band] = {
            q: (float(np.mean(sums[q])) if sums[q] else float("nan"))
            for q in ("Q1", "Q2", "Q3", "Q4")
        }

    # 4. Distractor comparisons (dead_zone, goodfire_lt_p90, nearest_wgate_tier0 + matched-pole dz)
    cohort_distractor_tables: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {
        cname: _compute_distractor_tables_for_cohort(
            primary_rows=primary_rows,
            baselines=baselines,
            allowed_pole_ids=cset,
            poles_by_id=poles_by_id,
            n_resamples=n_resamples,
            seed=seed,
        )
        for cname, cset in cohort_sets.items()
    }
    distractor_tables = cohort_distractor_tables["all_100"]

    distractor_tiers: dict[str, Counter[int]] = defaultdict(Counter)
    for s in snippets:
        distractor_tiers[str(s["distractor_type"])][int(s.get("distractor_tier", 0))] += 1

    # 5. Cross-judge agreement (aligned on pair_id, condition, order; scored rows only)
    cohort_second_judge_agreement: dict[str, dict[str, dict[str, Any]]] = {
        cname: _compute_second_judge_agreement_for_cohort(
            primary_rows=primary_rows,
            j_rows=j_rows,
            secondary_models=secondary_models,
            allowed_pole_ids=cset,
        )
        for cname, cset in cohort_sets.items()
    }
    second_judge_agreement = cohort_second_judge_agreement["all_100"]

    # 6. First-position bias & outcome audit (scored / unparseable / logprob_tie)
    cohort_position_bias: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        cname: _compute_position_bias_for_cohort(
            j_rows=j_rows,
            all_models=all_models,
            allowed_pole_ids=cset,
        )
        for cname, cset in cohort_sets.items()
    }
    position_bias = cohort_position_bias["all_100"]

    if structural_checks is None and snippets:
        len_eq_69 = sum(
            1
            for s in snippets
            if s.get("pos", {}).get("window_tokens") == 69
            and s.get("dist", {}).get("window_tokens") == 69
        )
        off_eq_60 = sum(
            1
            for s in snippets
            if s.get("pos", {}).get("target_offset") == 60
            and s.get("dist", {}).get("target_offset") == 60
        )
        structural_checks = {
            "total_pairs": len(snippets),
            "frac_exact_69_tokens": len_eq_69 / len(snippets),
            "frac_exact_offset_60": off_eq_60 / len(snippets),
            "marker_position_rule_acc": 0.500 if off_eq_60 == len(snippets) else float("nan"),
        }

    # 7. Cost, token, reasoning & both-orders diagnostics per judge model
    cost_audit: dict[str, dict[str, Any]] = {}
    both_orders_diagnostics: dict[str, dict[str, Any]] = {}
    for model in all_models:
        m_rows = [r for r in j_rows if str(r["judge_model"]) == model]
        n_calls = len(m_rows)
        n_scored = sum(1 for r in m_rows if classify_row_outcome(r) == "scored")
        n_unparse = sum(1 for r in m_rows if classify_row_outcome(r) == "unparseable")
        n_ties = sum(1 for r in m_rows if classify_row_outcome(r) == "logprob_tie")
        tot_prompt = sum(int(r.get("prompt_tokens", 0)) for r in m_rows)
        tot_comp = sum(int(r.get("completion_tokens", 0)) for r in m_rows)
        tot_reas = sum(int(r.get("reasoning_tokens", 0)) for r in m_rows)
        tot_vis = sum(
            int(
                r.get(
                    "visible_output_tokens",
                    max(0, int(r.get("completion_tokens", 0)) - int(r.get("reasoning_tokens", 0))),
                )
            )
            for r in m_rows
        )
        tot_usd = sum(float(r.get("cost_usd", 0.0)) for r in m_rows)
        has_lp = any(r.get("logprob_A") is not None or r.get("logprob_B") is not None for r in m_rows)
        nl_scored = [
            r for r in m_rows if str(r["condition"]) == "no_label" and classify_row_outcome(r) == "scored"
        ]
        nl_p_a = (
            sum(1 for r in nl_scored if r.get("predicted_slot") == "A") / len(nl_scored)
            if nl_scored
            else float("nan")
        )
        cost_audit[model] = {
            "backend": str(m_rows[0].get("judge_backend", "unknown")) if m_rows else "unknown",
            "api_model": str(m_rows[0].get("api_model", model.split("@")[0])) if m_rows else model,
            "reasoning_effort": m_rows[0].get("reasoning_effort") if m_rows else None,
            "calls": n_calls,
            "n_scored": n_scored,
            "n_unparseable": n_unparse,
            "n_logprob_tie": n_ties,
            "has_logprobs": has_lp,
            "prompt_tokens": tot_prompt,
            "completion_tokens": tot_comp,
            "reasoning_tokens": tot_reas,
            "visible_output_tokens": tot_vis,
            "mean_prompt_tokens": (tot_prompt / n_calls) if n_calls else 0.0,
            "mean_visible_output_tokens": (tot_vis / n_calls) if n_calls else 0.0,
            "mean_reasoning_tokens": (tot_reas / n_calls) if n_calls else 0.0,
            "mean_completion_tokens": (tot_comp / n_calls) if n_calls else 0.0,
            "mean_cost_usd_per_call": (tot_usd / n_calls) if n_calls else 0.0,
            "total_cost_usd": round(tot_usd, 6),
            "no_label_p_pred_A": nl_p_a,
        }

        fwd_map = {
            (str(r["pair_id"]), str(r["condition"])): r
            for r in m_rows
            if str(r.get("order") or "fwd") == "fwd"
        }
        rev_map = {
            (str(r["pair_id"]), str(r["condition"])): r
            for r in m_rows
            if str(r.get("order") or "fwd") == "rev"
        }
        shared_keys = sorted(set(fwd_map.keys()) & set(rev_map.keys()))
        if shared_keys:
            by_scope: dict[str, dict[str, Any]] = {}
            for scope in ("overall",) + ALL_CONDITIONS:
                keys_s = (
                    shared_keys
                    if scope == "overall"
                    else [k for k in shared_keys if k[1] == scope]
                )
                if not keys_s:
                    continue
                scored_pairs = [
                    k
                    for k in keys_s
                    if classify_row_outcome(fwd_map[k]) == "scored"
                    and classify_row_outcome(rev_map[k]) == "scored"
                ]
                if not scored_pairs:
                    continue
                fwd_corrs = [float(fwd_map[k]["correct"]) for k in scored_pairs]
                rev_corrs = [float(rev_map[k]["correct"]) for k in scored_pairs]
                consistent = [
                    1.0 if float(fwd_map[k]["correct"]) == float(rev_map[k]["correct"]) else 0.0
                    for k in scored_pairs
                ]
                fwd_m = float(np.mean(fwd_corrs))
                rev_m = float(np.mean(rev_corrs))
                by_scope[scope] = {
                    "n_paired": len(scored_pairs),
                    "n_excluded": len(keys_s) - len(scored_pairs),
                    "fwd_acc": fwd_m,
                    "rev_acc": rev_m,
                    "mean_both_orders_acc": 0.5 * (fwd_m + rev_m),
                    "order_consistency_rate": float(np.mean(consistent)),
                }
            both_orders_diagnostics[model] = {
                "has_both_orders": True,
                "n_fwd_rows": len(fwd_map),
                "n_rev_rows": len(rev_map),
                "by_scope": by_scope,
            }

    # 5b. Per-judge comparison at top band (p99.9_plus) AND per-band multi-judge breakdown
    ordered_models = (
        [primary] + [m for m in all_models if m != primary] if primary in all_models else all_models
    )
    per_judge_top_band: dict[str, dict[str, Any]] = {}
    per_judge_by_band: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for model in ordered_models:
        m_dz = [
            r
            for r in j_rows
            if str(r["judge_model"]) == model and str(r["distractor_type"]) == "dead_zone"
        ]
        m_top = [r for r in m_dz if str(r["band"]) == "p99.9_plus"]
        if m_top:
            entry: dict[str, Any] = {"all_judged": {}, "layer4_control": {}, "deep_layers": {}}
            for scope_name, pred_fn in (
                ("all_judged", lambda r: True),
                ("layer4_control", lambda r: int(r["layer"]) == 4),
                ("deep_layers", lambda r: int(r["layer"]) != 4),
            ):
                for cond in ALL_CONDITIONS:
                    sub = [r for r in m_top if str(r["condition"]) == cond and pred_fn(r)]
                    entry[scope_name][cond] = {
                        "acc": neuron_clustered_bootstrap_accuracy(
                            sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                        )
                    }
            per_judge_top_band[model] = entry

        if m_dz:
            bmap: dict[str, dict[str, dict[str, Any]]] = {}
            for band in ALL_BANDS:
                b_rows = [r for r in m_dz if str(r["band"]) == band]
                if not b_rows:
                    continue
                bmap[band] = {}
                for cond in ALL_CONDITIONS:
                    sub = [r for r in b_rows if str(r["condition"]) == cond]
                    bmap[band][cond] = neuron_clustered_bootstrap_accuracy(
                        sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                    )
            if bmap:
                per_judge_by_band[model] = bmap

    # Cohort-separated per-judge multi-band & layer breakdown
    cohort_per_judge_by_band: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    cohort_layer_breakdown: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]] = {}
    for cname, cset in (("heldout_80", heldout_pole_ids), ("pilot_20", pilot_pole_ids)):
        cohort_per_judge_by_band[cname] = {}
        for model in ordered_models:
            m_dz = [
                r
                for r in j_rows
                if str(r["judge_model"]) == model
                and str(r["distractor_type"]) == "dead_zone"
                and str(r["pole_id"]) in cset
            ]
            if not m_dz:
                continue
            bmap_c: dict[str, dict[str, dict[str, Any]]] = {}
            for band in ALL_BANDS:
                b_rows = [r for r in m_dz if str(r["band"]) == band]
                if not b_rows:
                    continue
                bmap_c[band] = {}
                for cond in ALL_CONDITIONS:
                    sub = [r for r in b_rows if str(r["condition"]) == cond]
                    bmap_c[band][cond] = neuron_clustered_bootstrap_accuracy(
                        sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                    )
            if bmap_c:
                cohort_per_judge_by_band[cname][model] = bmap_c

        cohort_layer_breakdown[cname] = {}
        for layer in layers:
            cohort_layer_breakdown[cname][layer] = {}
            for band in ALL_BANDS:
                cohort_layer_breakdown[cname][layer][band] = {}
                for cond in ALL_CONDITIONS:
                    sub = [
                        r
                        for r in primary_rows
                        if str(r["pole_id"]) in cset
                        and int(r["layer"]) == layer
                        and str(r["distractor_type"]) == "dead_zone"
                        and str(r["band"]) == band
                        and str(r["condition"]) == cond
                    ]
                    cohort_layer_breakdown[cname][layer][band][cond] = neuron_clustered_bootstrap_accuracy(
                        sub, poles_by_id=poles_by_id, is_baseline=False, n_resamples=n_resamples, seed=seed
                    )

    context_ablation: dict[str, Any] | None = None
    if unfiltered_judgements:
        context_ablation = compute_context_ablation(
            clean_primary_rows=primary_rows,
            unfiltered_judgements=unfiltered_judgements,
            unfiltered_snippets=unfiltered_snippets or (),
            poles_by_id=poles_by_id,
            primary_judge=primary,
            n_resamples=n_resamples,
            seed=seed,
        )

    examples_as_label: dict[str, Any] | None = None
    if examples_judgements:
        examples_as_label = compute_examples_as_label_analysis(
            clean_primary_rows=primary_rows,
            examples_judgements=examples_judgements,
            poles_by_id=poles_by_id,
            primary_judge=primary,
            n_resamples=n_resamples,
            seed=seed,
        )

    paired_models = [primary] + [m for m in ordered_models if m != primary and "luna" in m.lower()]
    paired_condition_diffs = compute_paired_condition_diffs(
        j_rows=j_rows,
        cohort_sets=cohort_sets,
        poles_by_id=poles_by_id,
        models=paired_models,
        n_resamples=n_resamples,
        seed=seed,
    )
    within_pole_gate_analysis = compute_within_pole_gate_analysis(
        primary_rows=primary_rows,
        snippets_by_pair=snippets_by_pair,
        cohort_sets=cohort_sets,
        poles_by_id=poles_by_id,
        n_resamples=n_resamples,
        seed=seed,
    )
    p999_gate_closed_heldout_diag = diagnose_p999_gate_closed_heldout(
        primary_rows=primary_rows,
        snippets_by_pair=snippets_by_pair,
        heldout_pole_ids=heldout_pole_ids,
        poles_by_id=poles_by_id,
    )
    per_pole_p999_distribution = compute_per_pole_top_band_distribution(
        primary_rows=primary_rows,
        cohort_sets=cohort_sets,
    )

    floors_below_045 = collect_floors_below_045(
        cohort_tables=cohort_tables,
        cohort_per_judge_by_band=cohort_per_judge_by_band,
        cohort_distractor_tables=cohort_distractor_tables,
        primary_judge=primary,
        context_ablation=context_ablation,
        j_rows=j_rows,
        ordered_models=ordered_models,
        cohort_sets=cohort_sets,
        poles_by_id=poles_by_id,
        n_resamples=n_resamples,
        seed=seed,
    )

    p97_p99_subband_split = compute_p97_p99_subband_split(
        poles_by_id=poles_by_id,
        snippets=snippets,
        j_rows=j_rows,
        examples_judgements=examples_judgements,
        p99_thresholds_payload=p99_thresholds_payload,
        primary_judge=primary,
        ordered_models=ordered_models,
        cohort_sets=cohort_sets,
        n_resamples=n_resamples,
        seed=seed,
    )

    prediction_checks = evaluate_heldout_criteria(
        cohort_tables=cohort_tables,
        cohort_counts=cohort_counts,
        cohort_quadrant_slice=cohort_quadrant_slice,
        cohort_distractor_tables=cohort_distractor_tables,
        cohort_second_judge_agreement=cohort_second_judge_agreement,
        second_judge_agreement_all=second_judge_agreement,
        cohort_position_bias=cohort_position_bias,
        primary_judge=primary,
    )

    return {
        "primary_judge": primary,
        "secondary_judges": secondary_models,
        "judge_roles": {m: _judge_role_label(m, primary) for m in ordered_models},
        "n_poles_total": len(poles_by_id),
        "n_snippets_total": len(snippets),
        "n_judgements_total": len(j_rows),
        "main_table": main_table,
        "cohort_tables": cohort_tables,
        "cohort_counts": cohort_counts,
        "paired_condition_diffs": paired_condition_diffs,
        "per_judge_top_band": per_judge_top_band,
        "per_judge_by_band": per_judge_by_band,
        "cohort_per_judge_by_band": cohort_per_judge_by_band,
        "both_orders_diagnostics": both_orders_diagnostics,
        "layer_breakdown": layer_breakdown,
        "cohort_layer_breakdown": cohort_layer_breakdown,
        "per_pole_p999_distribution": per_pole_p999_distribution,
        "pilot_gate": pilot_gate,
        "quadrant_slice": quadrant_slice,
        "cohort_quadrant_slice": cohort_quadrant_slice,
        "within_pole_gate_analysis": within_pole_gate_analysis,
        "p999_gate_closed_heldout_diag": p999_gate_closed_heldout_diag,
        "corpus_quad_by_band": corpus_quad_by_band,
        "distractor_tables": distractor_tables,
        "cohort_distractor_tables": cohort_distractor_tables,
        "distractor_tiers": {k: dict(v) for k, v in distractor_tiers.items()},
        "second_judge_agreement": second_judge_agreement,
        "cohort_second_judge_agreement": cohort_second_judge_agreement,
        "position_bias": position_bias,
        "cohort_position_bias": cohort_position_bias,
        "structural_checks": structural_checks or {},
        "cost_audit": cost_audit,
        "floors_below_045": floors_below_045,
        "prediction_checks": prediction_checks,
        "weak_band_survival": dict(weak_band_survival) if weak_band_survival else None,
        "context_ablation": context_ablation,
        "examples_as_label": examples_as_label,
        "p97_p99_subband_split": p97_p99_subband_split,
    }


def _draw_band_panel(
    ax: Any,
    band_table: Mapping[str, Mapping[str, Mapping[str, Any]]],
    title: str,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> int:
    x = np.arange(len(ALL_BANDS), dtype=np.float64)
    n_poles_seen = 0
    for method in METHOD_ORDER:
        style = METHOD_STYLE[method]
        ys = []
        yerr_low = []
        yerr_high = []
        xs = []
        for idx, band in enumerate(ALL_BANDS):
            cell = band_table.get(band, {}).get(method, {}).get("acc", {})
            val = cell.get("mean", float("nan"))
            low = cell.get("ci_low", float("nan"))
            high = cell.get("ci_high", float("nan"))
            if method in ALL_CONDITIONS:
                n_poles_seen = max(n_poles_seen, int(cell.get("n_poles", 0)))
            if math.isfinite(val):
                xs.append(idx + style["offset"])
                ys.append(val)
                yerr_low.append(max(0.0, val - low) if math.isfinite(low) else 0.0)
                yerr_high.append(max(0.0, high - val) if math.isfinite(high) else 0.0)

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_low, yerr_high],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=6.0,
                capsize=3.0,
                elinewidth=1.2,
                alpha=0.95,
            )

    ax.axhline(0.50, color="#444444", linestyle="--", linewidth=1.2, alpha=0.8, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_DISPLAY[b] for b in ALL_BANDS], fontsize=10.5)
    ax.set_xlabel("Activation Band (Signed Directional Tail Percentile)", fontsize=11, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Mean Binary Detection Accuracy Over Poles", fontsize=11, fontweight="bold")
    ax.set_ylim(0.30, 1.03)
    ax.set_yticks(np.arange(0.30, 1.01, 0.10))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    if show_legend:
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCCCCC",
            fontsize=8.5,
            ncol=2,
        )
    return n_poles_seen


def plot_band_accuracy(
    metrics: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Generate clean publication-ready main figure highlighting Held-Out 80 vs Pilot 20."""
    primary = metrics["primary_judge"]
    cohort_tables = metrics.get("cohort_tables", {})
    cohort_counts = metrics.get("cohort_counts", {})
    n_heldout = int(cohort_counts.get("heldout_80", {}).get("judged_poles", 0))
    n_pilot = int(cohort_counts.get("pilot_20", {}).get("judged_poles", 0))

    if n_heldout > 0 and n_pilot > 0 and "heldout_80" in cohort_tables and "pilot_20" in cohort_tables:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(15.4, 6.0), dpi=300, sharey=True
        )
        _draw_band_panel(
            ax_left,
            cohort_tables["heldout_80"],
            title=f"Held-Out Confirmatory Set (N={n_heldout} Non-Pilot Poles) [PRIMARY]",
            show_ylabel=True,
            show_legend=True,
        )
        _draw_band_panel(
            ax_right,
            cohort_tables["pilot_20"],
            title=f"Pilot Calibration Set (N={n_pilot} Pilot Poles)",
            show_ylabel=False,
            show_legend=False,
        )
        fig.suptitle(
            f"Goodfire Detection Audit on Qwen3-1.7B Neuron Poles (Dead-Zone Distractor) — Judge: {primary}\n"
            f"95% Neuron-Clustered Bootstrap CIs (B=1,000)",
            fontsize=12.5,
            fontweight="bold",
            y=0.99,
        )
    else:
        fig, ax = plt.subplots(figsize=(10.4, 6.0), dpi=300)
        if n_pilot > 0 and n_heldout == 0 and "pilot_20" in cohort_tables:
            table_to_plot = cohort_tables["pilot_20"]
            subtitle = f"Pilot Calibration Set (N={n_pilot} Poles; Held-Out 80 Pending --core)"
        else:
            table_to_plot = metrics["main_table"]
            subtitle = f"All Pooled Poles (N={max(n_heldout + n_pilot, metrics.get('n_poles_total', 0))})"
        _draw_band_panel(
            ax,
            table_to_plot,
            title=(
                f"Goodfire Detection Audit on Qwen3-1.7B Neuron Poles (Dead-Zone Distractor)\n"
                f"Primary Judge: {primary} | {subtitle} | 95% Neuron-Clustered Bootstrap CIs"
            ),
            show_ylabel=True,
            show_legend=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_main_band_accuracy(
    metrics: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Plot clean held-out-only 4-band accuracy figure with exploratory examples_as_label points."""
    primary = str(metrics["primary_judge"])
    cohort_tables = metrics.get("cohort_tables", {})
    heldout_table = cohort_tables.get("heldout_80") or metrics.get("main_table", {})
    heldout_counts = metrics.get("cohort_counts", {}).get("heldout_80", {})
    n_poles = int(heldout_counts.get("judged_poles", 0))
    if n_poles == 0:
        # Fallback for synthetic tests without heldout_80
        n_poles = int(
            heldout_table.get("p99.9_plus", {}).get("own", {}).get("acc", {}).get("n_poles", 0)
        )
    n_neurons = int(
        heldout_table.get("p99.9_plus", {}).get("own", {}).get("acc", {}).get("n_neurons", 0)
    )

    line_specs: list[tuple[str, dict[str, Any]]] = [
        (
            "own",
            {
                "label": "own (gpt-4.1 label)",
                "color": "#1B4F9C",
                "linestyle": "-",
                "marker": "o",
                "linewidth": 2.4,
                "markersize": 6.5,
                "offset": -0.075,
            },
        ),
        (
            "token_only",
            {
                "label": "token_only (top-3 tokens)",
                "color": "#0F8B8D",
                "linestyle": "--",
                "marker": "s",
                "linewidth": 2.0,
                "markersize": 6.0,
                "offset": -0.025,
            },
        ),
        (
            "no_label",
            {
                "label": "no_label",
                "color": "#C87D1A",
                "linestyle": "-.",
                "marker": "^",
                "linewidth": 1.8,
                "markersize": 6.0,
                "offset": 0.025,
            },
        ),
        (
            "wrong",
            {
                "label": "wrong (cross-layer, cos <= 0.65)",
                "color": "#B83333",
                "linestyle": ":",
                "marker": "D",
                "linewidth": 1.9,
                "markersize": 5.5,
                "offset": 0.075,
            },
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=250)
    x = np.arange(len(ALL_BANDS), dtype=np.float64)

    ax.axhline(
        0.50,
        color="#555555",
        linestyle="--",
        linewidth=1.15,
        alpha=0.80,
        zorder=1,
        label="Chance (0.50)",
    )

    for method, st in line_specs:
        xs: list[float] = []
        ys: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []
        for idx, band in enumerate(ALL_BANDS):
            cell = heldout_table.get(band, {}).get(method, {}).get("acc", {})
            val = float(cell.get("mean", float("nan")))
            lo = float(cell.get("ci_low", float("nan")))
            hi = float(cell.get("ci_high", float("nan")))
            if math.isfinite(val):
                xs.append(float(idx) + float(st["offset"]))
                ys.append(val)
                yerr_lo.append(max(0.0, val - lo) if math.isfinite(lo) else 0.0)
                yerr_hi.append(max(0.0, hi - val) if math.isfinite(hi) else 0.0)
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                label=st["label"],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=st["linewidth"],
                markersize=st["markersize"],
                capsize=3.2,
                elinewidth=1.25,
                alpha=0.95,
                zorder=3,
            )

    # Exploratory examples_as_label: distinct points ONLY on top 2 bands
    ex_arm = metrics.get("examples_as_label") or {}
    ex_heldout_bands = ex_arm.get("cohorts", {}).get("heldout_80", {})
    if "bands" in ex_heldout_bands and isinstance(ex_heldout_bands["bands"], Mapping):
        ex_heldout_bands = ex_heldout_bands["bands"]
    ex_xs: list[float] = []
    ex_ys: list[float] = []
    ex_lo: list[float] = []
    ex_hi: list[float] = []
    for idx, band in enumerate(ALL_BANDS):
        if band not in ("p97_99.9", "p99.9_plus"):
            continue
        cell = ex_heldout_bands.get(band, {}).get("examples_as_label", {})
        val = float(cell.get("mean", float("nan")))
        lo = float(cell.get("ci_low", float("nan")))
        hi = float(cell.get("ci_high", float("nan")))
        if math.isfinite(val):
            ex_xs.append(float(idx) + 0.14)
            ex_ys.append(val)
            ex_lo.append(max(0.0, val - lo) if math.isfinite(lo) else 0.0)
            ex_hi.append(max(0.0, hi - val) if math.isfinite(hi) else 0.0)

    if ex_xs:
        ax.errorbar(
            ex_xs,
            ex_ys,
            yerr=[ex_lo, ex_hi],
            label="examples_as_label (exploratory)",
            color="#6A2C91",
            linestyle="none",
            marker="*",
            markersize=10.5,
            markeredgecolor="#3F155C",
            markeredgewidth=0.7,
            capsize=3.5,
            elinewidth=1.35,
            alpha=0.98,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BAND_DISPLAY[b] for b in ALL_BANDS], fontsize=10.5)
    ax.set_xlim(-0.38, len(ALL_BANDS) - 0.55)
    ax.set_ylim(0.35, 0.98)
    ax.set_yticks([0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Activation Band (Signed Directional Tail Percentile)", fontsize=10.8, fontweight="bold")
    ax.set_ylabel("Mean Binary Detection Accuracy Over Poles", fontsize=10.8, fontweight="bold")
    ax.grid(axis="y", linestyle=":", color="#CCCCCC", alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cohort_desc = f"n={n_poles} held-out poles"
    if n_neurons > 0:
        cohort_desc += f" / {n_neurons} neurons"
    ax.set_title(
        f"Held-Out Detection Accuracy by Activation Band (Qwen3-1.7B, {cohort_desc})\n"
        f"Judge: {primary} | Dead-zone distractor | 95% neuron-clustered bootstrap CIs (B=1,000)",
        fontsize=10.8,
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=8.6,
        ncol=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_context_ablation(
    context_ablation: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Plot exploratory clean vs. unfiltered weak-band context effect."""
    cohorts = context_ablation.get("cohorts", {})
    judge_model = str(context_ablation.get("judge_model", DEFAULT_PRIMARY_MODEL))
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), dpi=300, sharey=True)

    panel_specs = (
        ("heldout_80", "Held-Out 80 Poles [PRIMARY]"),
        ("pilot_20", "Pilot 20 Poles [Calibration]"),
    )
    conds = ("own", "token_only", "no_label", "wrong")
    cond_labels = ("own", "token_only", "no_label", "wrong")
    band_styles = (
        ("p75_90", "Clean (<p90)", "clean", "#0072B2", -0.27, 0.90),
        ("p75_90", "Unfiltered", "unfiltered", "#56B4E9", -0.09, 0.90),
        ("p90_97", "Clean (<p90)", "clean", "#D55E00", +0.09, 0.90),
        ("p90_97", "Unfiltered", "unfiltered", "#E69F00", +0.27, 0.90),
    )
    x = np.arange(len(conds), dtype=np.float64)
    width = 0.17

    for ax_idx, (cname, title) in enumerate(panel_specs):
        ax = axes[ax_idx]
        c_data = cohorts.get(cname, {})
        for band, variant_label, key_name, color, offset, alpha in band_styles:
            ys: list[float] = []
            el: list[float] = []
            eh: list[float] = []
            for cond in conds:
                st = c_data.get(band, {}).get(cond, {}).get(key_name, {})
                v = float(st.get("mean", float("nan")))
                lo = float(st.get("ci_low", float("nan")))
                hi = float(st.get("ci_high", float("nan")))
                ys.append(v if math.isfinite(v) else 0.0)
                el.append(max(0.0, v - lo) if (math.isfinite(v) and math.isfinite(lo)) else 0.0)
                eh.append(max(0.0, hi - v) if (math.isfinite(v) and math.isfinite(hi)) else 0.0)
            ax.bar(
                x + offset,
                ys,
                width=width,
                yerr=[el, eh],
                capsize=2.5,
                color=color,
                alpha=alpha,
                edgecolor="#222222",
                linewidth=0.6,
                label=f"{BAND_DISPLAY[band]} {variant_label}",
            )
        ax.axhline(0.50, color="#333333", linestyle="--", linewidth=1.1, label="Chance (0.50)")
        ax.axhline(0.45, color="#CC3333", linestyle=":", linewidth=1.0, label="Floor alert (0.45)")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=10.5, fontweight="bold")
        ax.set_ylim(0.25, 0.95)
        ax.set_yticks(np.arange(0.25, 0.96, 0.10))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Mean Binary Detection Accuracy Over Poles", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8.2, ncol=2, frameon=True, framealpha=0.92)

    fig.suptitle(
        f"Exploratory Context Ablation: Clean (<p90) vs. Unfiltered Weak-Band Windows\n"
        f"Judge: {judge_model} | Dead-Zone Distractor | 95% Neuron-Clustered Bootstrap CIs",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _render_band_cohort_block(
    lines: list[str],
    heading: str,
    band_table: Mapping[str, Mapping[str, Mapping[str, Any]]],
    judged_poles: int,
    total_poles: int,
) -> None:
    lines.append(f"### {heading} (`{judged_poles}/{total_poles}` poles judged)")
    if judged_poles == 0:
        lines.append("_No judgements in this cohort yet (pending `--core`)._")
        lines.append("")
        return

    header_bands = " | ".join(f"`{BAND_DISPLAY[b]}`" for b in ALL_BANDS)
    lines.append("#### Mean Binary Accuracy Over Poles [95% Neuron-Clustered Bootstrap CI]")
    lines.append(f"| Method | {header_bands} | `scored / unparse / ties` |")
    lines.append("|---|" + "---:|" * len(ALL_BANDS) + "---:|")

    display_methods = list(METHOD_ORDER)
    if any("top_10_token" in band_table[b] for b in ALL_BANDS):
        display_methods.append("top_10_token")

    for method in display_methods:
        cells = []
        tot_s = tot_u = tot_t = 0
        for band in ALL_BANDS:
            st = band_table[band].get(method, {}).get("acc", {})
            cells.append(
                _fmt_floor_ci(
                    method,
                    st.get("mean", float("nan")),
                    st.get("ci_low", float("nan")),
                    st.get("ci_high", float("nan")),
                )
            )
            tot_s += int(st.get("n_scored", 0))
            tot_u += int(st.get("n_unparseable", 0))
            tot_t += int(st.get("n_logprob_tie", 0))
        lines.append(f"| `{method}` | " + " | ".join(cells) + f" | `{tot_s} / {tot_u} / {tot_t}` |")

    lines.append("")
    lines.append("#### Label Lift Over Floors (Mean Binary Accuracy Difference)")
    lines.append(f"| Contrast | {header_bands} |")
    lines.append("|---|" + "---:|" * len(ALL_BANDS))

    def _max_text_rule_acc(band_name: str) -> float:
        finite = [
            v
            for v in (
                band_table[band_name]["rarity"]["acc"]["mean"],
                band_table[band_name]["top_token"]["acc"]["mean"],
            )
            if math.isfinite(v)
        ]
        return max(finite) if finite else float("nan")

    for label, other_fn in (
        ("`own - no_label`", lambda b: band_table[b]["no_label"]["acc"]["mean"]),
        ("`own - wrong`", lambda b: band_table[b]["wrong"]["acc"]["mean"]),
        ("`own - token_only`", lambda b: band_table[b]["token_only"]["acc"]["mean"]),
        ("`own - max(rarity, top_token)`", _max_text_rule_acc),
    ):
        vals = []
        for band in ALL_BANDS:
            own_v = band_table[band]["own"]["acc"]["mean"]
            oth_v = other_fn(band)
            diff = (
                own_v - oth_v
                if (math.isfinite(own_v) and math.isfinite(oth_v))
                else float("nan")
            )
            vals.append(f"{diff:+.3f}" if math.isfinite(diff) else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")


def _render_weak_band_survival_block(lines: list[str], survival: Mapping[str, Any]) -> None:
    """Render exact weak-band clean-window survival statistics cleanly in Limitations."""
    lines.append("### Weak-Band Clean-Window Filter Survival Shares (`results/weak_band_survival.json`)")
    bands_map = survival.get("weak_band_survival", survival.get("bands", survival))
    rendered_any = False
    if any(isinstance(bands_map.get(b), Mapping) for b in WEAK_BANDS):
        lines.append(
            "| Band | Cohort / Layer | Pooled Survival (`surv / cand`) | Per-Pole Mean | Median | `p10`–`p90` | `min`–`max` |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for band in WEAK_BANDS:
            binfo = bands_map.get(band)
            if not isinstance(binfo, Mapping):
                continue
            rendered_any = True
            for scope_key, scope_label in (
                ("all_100", "All 100"),
                ("heldout_80", "Held-Out 80"),
                ("pilot_20", "Pilot 20"),
            ):
                s = binfo.get(scope_key)
                if not isinstance(s, Mapping):
                    continue
                pp = s.get("per_pole_survival_rate", {})
                rate = float(s.get("pooled_survival_rate", float("nan")))
                surv = int(s.get("pooled_surviving_clean", 0))
                cand = int(s.get("pooled_total_candidates", 0))
                lines.append(
                    f"| `{BAND_DISPLAY[band]}` | {scope_label} | "
                    f"**{100.0 * rate:.2f}%** (`{surv:,} / {cand:,}`) | "
                    f"{100.0 * float(pp.get('mean', float('nan'))):.2f}% | "
                    f"{100.0 * float(pp.get('median', float('nan'))):.2f}% | "
                    f"{100.0 * float(pp.get('p10', float('nan'))):.2f}%–{100.0 * float(pp.get('p90', float('nan'))):.2f}% | "
                    f"{100.0 * float(pp.get('min', float('nan'))):.2f}%–{100.0 * float(pp.get('max', float('nan'))):.2f}% |"
                )
            by_layer = binfo.get("by_layer", {})
            if isinstance(by_layer, Mapping):
                for ly in sorted(by_layer.keys(), key=lambda x: int(x)):
                    s = by_layer[ly]
                    pp = s.get("per_pole_survival_rate", {})
                    rate = float(s.get("pooled_survival_rate", float("nan")))
                    surv = int(s.get("pooled_surviving_clean", 0))
                    cand = int(s.get("pooled_total_candidates", 0))
                    lines.append(
                        f"| `{BAND_DISPLAY[band]}` | Layer `L{ly}` | "
                        f"{100.0 * rate:.2f}% (`{surv:,} / {cand:,}`) | "
                        f"{100.0 * float(pp.get('mean', float('nan'))):.2f}% | "
                        f"{100.0 * float(pp.get('median', float('nan'))):.2f}% | "
                        f"{100.0 * float(pp.get('p10', float('nan'))):.2f}%–{100.0 * float(pp.get('p90', float('nan'))):.2f}% | "
                        f"{100.0 * float(pp.get('min', float('nan'))):.2f}%–{100.0 * float(pp.get('max', float('nan'))):.2f}% |"
                    )
    if not rendered_any:
        for k, v in sorted(survival.items()):
            lines.append(f"- `{k}`: `{v}`")


def render_summary_tables_md(metrics: Mapping[str, Any]) -> str:
    """Render all evaluation markdown tables into results/summary_tables.md."""
    lines: list[str] = []
    primary = metrics["primary_judge"]
    judge_roles = metrics.get("judge_roles", {})
    sec_list = [m for m in metrics["secondary_judges"] if _judge_role_label(m, primary) != "third judge"]
    third_list = [m for m in metrics["secondary_judges"] if _judge_role_label(m, primary) == "third judge"]

    role_parts = [f"**Primary judge:** `{primary}`"]
    if sec_list:
        role_parts.append("**Secondary judge:** " + ", ".join(f"`{m}`" for m in sec_list))
    if third_list:
        role_parts.append("**Third judge:** " + ", ".join(f"`{m}`" for m in third_list))
    if not sec_list and not third_list:
        role_parts.append("**Secondary judge(s):** None yet")

    lines.append("# Audit of Two-Snippet Neuron Detection on Qwen3-1.7B")
    lines.append("")
    lines.append("- " + " | ".join(role_parts))
    lines.append(
        "- **Primary metric:** Binary detection accuracy (unparseable answers & exact logprob ties "
        "excluded from accuracy and reported separately; 95% CIs cluster-bootstrapped over neurons). "
        "Any floor (`wrong`, `no_label`, `token_only`, rule baselines) `< 0.45` is flagged inline with `[<0.45]` "
        "and collected in the dedicated **Floors Below 0.45 Flagged** table."
    )
    lines.append(
        f"- **Data loaded:** {metrics['n_poles_total']} poles, "
        f"{metrics['n_snippets_total']:,} pairs, {metrics['n_judgements_total']:,} judge rows."
    )
    lines.append("")

    # 1. Main 4-band x method tables split cleanly into 3 cohorts (Condition 3)
    lines.append("## 1. Main 4-Band × Method Detection Audit (`dead_zone` Distractor)")
    lines.append("")
    cohort_tables = metrics.get("cohort_tables", {})
    cohort_counts = metrics.get("cohort_counts", {})
    for cname, heading in (
        ("heldout_80", "1.1 Held-Out 80 Non-Pilot Poles (`in_pilot=False`) — Primary Confirmatory Set"),
        ("pilot_20", "1.2 Pilot 20 Poles (`in_pilot=True`) — Calibration / Exploratory Set"),
        ("all_100", "1.3 All 100 Poles Pooled"),
    ):
        bt = cohort_tables.get(cname, metrics["main_table"])
        cc = cohort_counts.get(cname, {"judged_poles": 0, "total_poles": 0})
        _render_band_cohort_block(
            lines=lines,
            heading=heading,
            band_table=bt,
            judged_poles=int(cc["judged_poles"]),
            total_poles=int(cc["total_poles"]),
        )

    # 1b. Criterion Checks
    pred_chk = metrics.get("prediction_checks", {})
    if pred_chk:
        lines.append("### 1.4 Criterion Checks")
        sub_rows = pred_chk.get("sub_predictions", [])
        if not sub_rows:
            lines.append("_Pending `heldout_80` judgements._")
        else:
            missed = [r["sub_id"] for r in sub_rows if r["outcome"] == "FAILED"]
            missed_str = ", ".join(f"`{m}`" for m in missed) if missed else "none"
            lines.append(
                f"- **Held-out criteria (`heldout_80`):** `{pred_chk['n_failed']}` criteria missed their pre-held-out targets ({missed_str}); individual rows below."
            )
            lines.append("| Sub-ID | Group | Criterion | Observed (`heldout_80`) | Threshold | Verdict |")
            lines.append("|---|---|---|---|---|---|")
            for row in sub_rows:
                lines.append(
                    f"| `{row['sub_id']}` | `{row['group']}` | {row['criterion']} | "
                    f"{row['observed']} | `{row['threshold']}` | **{row['outcome']}** |"
                )
        lines.append("")

    # 1c. Exact Paired Differences on Identical Pairs (own - floor) with Neuron-Clustered 95% CIs
    paired_diffs = metrics.get("paired_condition_diffs", {})
    if any(paired_diffs.get(c) for c in ("heldout_80", "pilot_20")):
        lines.append("### 1.5 Exact Paired Differences on Identical Pairs (`dead_zone`, 95% Neuron-Clustered Bootstrap CIs)")
        for cname, clabel in (
            ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary Confirmatory)"),
            ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
        ):
            p_coh = paired_diffs.get(cname, {})
            if not p_coh:
                continue
            lines.append(f"#### {clabel}")
            lines.append("| Judge Model | Contrast (Paired on Identical Pairs) | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for model, bmap in p_coh.items():
                for ckey, clbl in (
                    ("own_minus_token_only", "`own - token_only`"),
                    ("own_minus_no_label", "`own - no_label`"),
                    ("own_minus_wrong", "`own - wrong`"),
                ):
                    cells = []
                    for band in ALL_BANDS:
                        st = bmap.get(band, {}).get(ckey, {})
                        v = st.get("mean", float("nan"))
                        lo = st.get("ci_low", float("nan"))
                        hi = st.get("ci_high", float("nan"))
                        if math.isfinite(v) and math.isfinite(lo) and math.isfinite(hi):
                            cells.append(f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                        else:
                            cells.append("—")
                    lines.append(f"| `{model}` | {clbl} | " + " | ".join(cells) + " |")
            lines.append("")

    # 1c. Dedicated Floors Below 0.45 Flagged Table
    floors_low = metrics.get("floors_below_045", [])
    lines.append("### Floors Below 0.45 Flagged (`heldout_80` & `pilot_20`)")
    if not floors_low:
        lines.append("_No floor method (`wrong`, `no_label`, `token_only`, or rule baseline) scored below 0.45._")
    else:
        lines.append("| Cohort | Judge / Rule | Distractor Arm | Band | Floor Method | Acc [95% Neuron CI] | `n_scored` |")
        lines.append("|---|---|---|---|---|---:|---:|")
        for item in floors_low:
            jm_disp = f"`{item['judge_model']}`"
            if item["cohort"] == "heldout_80" and item["judge_model"] == "gpt-6.2-sol@low":
                jm_disp += " (descriptive, n=10 held-out poles / 5 neurons)"
            lines.append(
                f"| `{item['cohort']}` | {jm_disp} | `{item['distractor_type']}` | "
                f"`{BAND_DISPLAY.get(item['band'], item['band'])}` | `{item['method']}` | "
                f"{_fmt_ci(item['mean'], item['ci_low'], item['ci_high'])} [<0.45] | {item['n_scored']} |"
            )
    lines.append("")

    # 2. Layer-4 positive control & depth breakdown + multi-judge comparison
    lines.append("## 2. Layer-4 Positive Control & Pilot Gate Diagnostic")
    pg = metrics["pilot_gate"]
    lines.append(f"- **Pilot Gate Verdict:** **{pg['verdict']}**")
    lines.append(
        f"- Layer-4 `p99.9+` readout (`{primary}`): `own` = **{_fmt_num(pg['l4_own_p99.9_plus'])}**, "
        f"`wrong` = {_fmt_floor_num('wrong', pg['l4_wrong_p99.9_plus'])}, "
        f"`no_label` = {_fmt_floor_num('no_label', pg['l4_no_label_p99.9_plus'])}, "
        f"floor gap = {_fmt_num(pg.get('l4_floor_gap', float('nan')))} | "
        f"Deeper layers (`L11/L18/L24`) `own` = **{_fmt_num(pg['deep_layers_own_p99.9_plus'])}**"
    )
    lines.append("")

    cohort_lb = metrics.get("cohort_layer_breakdown", {})
    for cname, clabel in (
        ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary Confirmatory)"),
        ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
    ):
        lb = cohort_lb.get(cname)
        if not lb or not any(
            math.isfinite(lb[ly]["p99.9_plus"]["own"]["mean"]) for ly in lb
        ):
            continue
        lines.append(f"#### Per-Layer 4-Band Breakdown with 95% Neuron-Clustered CIs — {clabel}")
        lines.append(
            "| Layer | `n_poles / n_neurons` | Band | `own` [95% CI] | `token_only` [95% CI] | `no_label` [95% CI] | `wrong` [95% CI] |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for layer, bmap in sorted(lb.items()):
            tag = f"L{layer} (control)" if layer == 4 else f"L{layer}"
            for band in ALL_BANDS:
                o = bmap[band]["own"]
                t = bmap[band]["token_only"]
                nl = bmap[band]["no_label"]
                wr = bmap[band]["wrong"]
                if o["n_poles"] == 0:
                    continue
                lines.append(
                    f"| {tag} | `{o['n_poles']} / {o['n_neurons']}` | `{BAND_DISPLAY[band]}` | "
                    f"{_fmt_ci(o['mean'], o['ci_low'], o['ci_high'])} | "
                    f"{_fmt_floor_ci('token_only', t['mean'], t['ci_low'], t['ci_high'])} | "
                    f"{_fmt_floor_ci('no_label', nl['mean'], nl['ci_low'], nl['ci_high'])} | "
                    f"{_fmt_floor_ci('wrong', wr['mean'], wr['ci_low'], wr['ci_high'])} |"
                )
        lines.append("")

    per_judge_top = metrics.get("per_judge_top_band", {})
    if len(per_judge_top) >= 1:
        lines.append("### 2b. Per-Judge Top-Band (`p99.9+`, `dead_zone`) Comparison (All Judged Poles)")
        lines.append(
            "| Judge Model | Role | Scope | `n_poles` | `own` Acc [95% CI] | `token_only` Acc | `no_label` Acc | `wrong` Acc | `scored/unp/tie` (`own`) |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for model, smap in per_judge_top.items():
            role = judge_roles.get(model, _judge_role_label(model, primary))
            for scope_key, scope_label in (
                ("layer4_control", "L4 Control"),
                ("deep_layers", "Deeper (L11/18/24)"),
                ("all_judged", "All Judged"),
            ):
                own_a = smap[scope_key]["own"]["acc"]
                tok_a = smap[scope_key]["token_only"]["acc"]
                nl_a = smap[scope_key]["no_label"]["acc"]
                wr_a = smap[scope_key]["wrong"]["acc"]
                if own_a["n_poles"] == 0:
                    continue
                lines.append(
                    f"| `{model}` | {role} | {scope_label} | {own_a['n_poles']} | "
                    f"{_fmt_ci(own_a['mean'], own_a['ci_low'], own_a['ci_high'])} | "
                    f"{_fmt_floor_num('token_only', tok_a['mean'])} | "
                    f"{_fmt_floor_num('no_label', nl_a['mean'])} | "
                    f"{_fmt_floor_num('wrong', wr_a['mean'])} | "
                    f"`{own_a['n_scored']}/{own_a['n_unparseable']}/{own_a['n_logprob_tie']}` |"
                )

    cohort_pjb = metrics.get("cohort_per_judge_by_band", {})
    if any(cohort_pjb.get(c) for c in ("heldout_80", "pilot_20")):
        lines.append("")
        lines.append("### 2c. Per-Judge Multi-Band (`dead_zone`) Breakdown (`heldout_80` vs. `pilot_20` Separate)")
        for cname, clabel in (
            ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary)"),
            ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
        ):
            pjb = cohort_pjb.get(cname, {})
            if not pjb:
                continue
            lines.append(f"#### {clabel}")
            lines.append("| Judge Model | Role | Band | `n_poles` | `own` Acc [95% CI] | `token_only` | `no_label` | `wrong` |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|")
            for model, bmap in pjb.items():
                role = judge_roles.get(model, _judge_role_label(model, primary))
                if cname == "heldout_80" and "sol" in model.lower():
                    role = "third judge (descriptive, n=10 held-out poles / 5 neurons)"
                for band in ALL_BANDS:
                    if band not in bmap:
                        continue
                    own_s = bmap[band]["own"]
                    tok_s = bmap[band]["token_only"]
                    nl_s = bmap[band]["no_label"]
                    wr_s = bmap[band]["wrong"]
                    lines.append(
                        f"| `{model}` | {role} | `{BAND_DISPLAY[band]}` | {own_s['n_poles']} | "
                        f"{_fmt_ci(own_s['mean'], own_s['ci_low'], own_s['ci_high'])} | "
                        f"{_fmt_floor_num('token_only', tok_s['mean'])} | "
                        f"{_fmt_floor_num('no_label', nl_s['mean'])} | "
                        f"{_fmt_floor_num('wrong', wr_s['mean'])} |"
                    )
            lines.append("")

    bo_diag = metrics.get("both_orders_diagnostics", {})
    if bo_diag:
        lines.append("### 2d. Both-Orders (`fwd` vs. `rev`) Accuracy & Paired Order-Consistency (Third Judge)")
        lines.append(
            "| Judge Model | Role | Scope | Scored Pairs | Excluded | `fwd` Acc | `rev` Acc | Both-Orders Mean | Same-Snippet Consistency |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for model, info in bo_diag.items():
            role = judge_roles.get(model, _judge_role_label(model, primary))
            for scope, st in info["by_scope"].items():
                lines.append(
                    f"| `{model}` | {role} | `{scope}` | {st['n_paired']} | {st['n_excluded']} | "
                    f"{_fmt_num(st['fwd_acc'])} | {_fmt_num(st['rev_acc'])} | "
                    f"{_fmt_floor_num(scope, st['mean_both_orders_acc'])} | {_fmt_num(st['order_consistency_rate'])} |"
                )

    dist_p999 = metrics.get("per_pole_p999_distribution", {})
    if any(dist_p999.get(c, {}).get("n_poles", 0) > 0 for c in ("heldout_80", "pilot_20")):
        lines.append("")
        lines.append(f"### 2e. Per-Pole Distribution of `own` Accuracy at `p99.9+` (`dead_zone`, `{primary}`)")
        lines.append(
            "| Cohort | `n_poles` | `0/6` | `1/6` | `2/6` | `3/6` | `4/6` | `5/6` | `6/6` | Mean ± SD | Median (IQR) | `>= 5/6` | `<= 3/6` |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cname, clabel in (("heldout_80", "`heldout_80`"), ("pilot_20", "`pilot_20`")):
            d = dist_p999.get(cname, {})
            if d.get("n_poles", 0) == 0:
                continue
            cts = d["counts"]
            lines.append(
                f"| {clabel} | {d['n_poles']} | "
                f"{cts['0/6']} | {cts['1/6']} | {cts['2/6']} | {cts['3/6']} | {cts['4/6']} | {cts['5/6']} | {cts['6/6']} | "
                f"{d['mean']:.3f} ± {d['std']:.3f} | {d['median']:.3f} ({d['iqr']:.3f}) | "
                f"{d['n_ge_5_of_6']} ({d['pct_ge_5_of_6']:.1f}%) | "
                f"{d['n_le_3_of_6']} ({d['pct_le_3_of_6']:.1f}%) |"
            )

    # 3. SwiGLU Quadrant slice (Q3 double-negative focus) — separate heldout_80 vs pilot_20
    lines.append("")
    lines.append("## 3. SwiGLU Quadrant Slice (Q1/Q2 Gate-Open vs. Q3 Double-Negative & Q4 Gate-Closed)")
    lines.append(
        "Per Goodfire Fig. 2–3: **Q1** (`g>0, u>0`, pos), **Q2** (`g>0, u<0`, neg), "
        "**Q3** (`g<0, u<0`, double-negative pos), **Q4** (`g<0, u>0`, gate-closed neg)."
    )
    lines.append("")
    lines.append("### 3a. Corpus Quadrant Shares by Band (Mean Over Poles)")
    lines.append("| Band | Q1 (`g>0,u>0`) | Q2 (`g>0,u<0`) | Q3 (`g<0,u<0` dbl-neg) | Q4 (`g<0,u>0`) |")
    lines.append("|---|---:|---:|---:|---:|")
    for band in ALL_BANDS:
        qs = metrics["corpus_quad_by_band"][band]
        lines.append(
            f"| `{BAND_DISPLAY[band]}` | {_fmt_num(qs['Q1'])} | {_fmt_num(qs['Q2'])} | "
            f"{_fmt_num(qs['Q3'])} | {_fmt_num(qs['Q4'])} |"
        )

    lines.append("")
    lines.append("### 3b. Detection Accuracy (`own` & `no_label`) by Firing Token Quadrant (`heldout_80` & `pilot_20` Separate)")
    cohort_q = metrics.get("cohort_quadrant_slice", {"all_100": metrics["quadrant_slice"]})
    for cname, clabel in (
        ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary Confirmatory)"),
        ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
    ):
        qslice = cohort_q.get(cname)
        if not qslice:
            continue
        has_rows = any(
            qslice[b][qg]["own"]["n_scored"] > 0
            for b in ALL_BANDS
            for qg in ("gate_open", "gate_closed")
        )
        if not has_rows:
            continue
        lines.append(f"#### {clabel}")
        lines.append("| Band | Quadrant / Gate | `own` Acc [95% CI] | `n_scored` | `no_label` Acc [95% CI] | `wrong` Acc |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for band in ALL_BANDS:
            for qg in ("gate_open", "gate_closed", "Q1", "Q3", "Q2", "Q4"):
                st_own = qslice[band][qg]["own"]
                st_nl = qslice[band][qg]["no_label"]
                st_wr = qslice[band][qg]["wrong"]
                if st_own["n_scored"] == 0 and st_nl["n_scored"] == 0:
                    continue
                lines.append(
                    f"| `{BAND_DISPLAY[band]}` | `{qg}` | "
                    f"{_fmt_ci(st_own['mean'], st_own['ci_low'], st_own['ci_high'])} | "
                    f"{st_own['n_scored']} | "
                    f"{_fmt_floor_ci('no_label', st_nl['mean'], st_nl['ci_low'], st_nl['ci_high'])} | "
                    f"{_fmt_floor_num('wrong', st_wr['mean'])} |"
                )
        lines.append("")

    wpg = metrics.get("within_pole_gate_analysis", {})
    if any(wpg.get(c) for c in ("heldout_80", "pilot_20")):
        lines.append("### 3c. Gate-Open (`Q1+Q2`) vs. Gate-Closed (`Q3+Q4`) Counts, Marginal Accuracy & Within-Pole Matched Comparison")
        for cname, clabel in (
            ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary Confirmatory)"),
            ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
        ):
            wb = wpg.get(cname, {})
            if not any(wb.get(b, {}).get("n_total", 0) > 0 for b in ALL_BANDS):
                continue
            lines.append(f"#### {clabel}")
            lines.append(
                "| Band | `n_open / n_closed` (`% closed`) | Marginal `open` `own` [95% CI] | Marginal `closed` `own` [95% CI] | "
                "Matched Poles / Neurons | Within-Pole `open` | Within-Pole `closed` | Within-Pole `open - closed` (`own`) [95% CI] | Within-Pole `open - closed` (`no_label`) [95% CI] |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for band in ALL_BANDS:
                st = wb.get(band)
                if not st or st["n_total"] == 0:
                    continue
                mo = st["marginal_open_own"]
                mc = st["marginal_closed_own"]
                m_open = st["matched_open_own"]
                m_closed = st["matched_closed_own"]
                wd = st["within_pole_diff_own"]
                wnl = st["within_pole_diff_no_label"]
                wd_str = (
                    f"{wd['mean']:+.3f} [{wd['ci_low']:+.3f}, {wd['ci_high']:+.3f}]"
                    if math.isfinite(wd["mean"])
                    else "—"
                )
                wnl_str = (
                    f"{wnl['mean']:+.3f} [{wnl['ci_low']:+.3f}, {wnl['ci_high']:+.3f}]"
                    if math.isfinite(wnl["mean"])
                    else "—"
                )
                lines.append(
                    f"| `{BAND_DISPLAY[band]}` | `{st['n_open']} / {st['n_closed']}` ({st['pct_gate_closed']:.1f}%) | "
                    f"{_fmt_ci(mo['mean'], mo['ci_low'], mo['ci_high'])} | "
                    f"{_fmt_ci(mc['mean'], mc['ci_low'], mc['ci_high'])} | "
                    f"`{st['n_matched_poles']} / {st['n_matched_neurons']}` | "
                    f"{_fmt_num(m_open['mean'])} | {_fmt_num(m_closed['mean'])} | "
                    f"{wd_str} | {wnl_str} |"
                )
            lines.append("")

        diag39 = metrics.get("p999_gate_closed_heldout_diag", {})
        if diag39.get("n_pairs", 0) > 0:
            top_p = ", ".join(
                f"`{p['pole_id']}` (L{p['layer']}, {p['quadrant']}, n={p['n_pairs']}, own={p['own_acc']:.2f}, nl={p['no_label_acc']:.2f}, toks={p['sample_tokens']})"
                for p in diag39.get("pole_details", [])[:5]
            )
            lines.append(
                f"- **Diagnostic for `heldout_80` `p99.9+` `gate_closed` subset (`n={diag39['n_pairs']}` pairs):** "
                f"Concentrated in only **`{diag39['n_distinct_poles']}` poles (`{diag39['n_distinct_neurons']}` neurons)** "
                f"(layers: `{diag39['pair_counts_by_layer']}`, poles/layer: `{diag39['pole_counts_by_layer']}`, quadrants: `{diag39['quadrant_counts']}`). "
                f"Largest contributors: {top_p}."
            )
            lines.append("")

    # 4. Distractor comparisons — separate heldout_80 vs pilot_20
    lines.append("## 4. Distractor Comparisons (`heldout_80` & `pilot_20` Separate)")
    cohort_dt = metrics.get("cohort_distractor_tables", {"all_100": metrics["distractor_tables"]})

    lines.append("### 4a. Top-Band (`p99.9+`): Dead-Zone vs. Goodfire `<p90` Distractor")
    for cname, clabel in (
        ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary Confirmatory)"),
        ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
    ):
        dt = cohort_dt.get(cname)
        if not dt:
            continue
        if dt["dead_zone"]["p99.9_plus"]["own"]["n_scored"] == 0 and dt["goodfire_lt_p90"]["p99.9_plus"]["own"]["n_scored"] == 0:
            continue
        lines.append(f"#### {clabel}")
        lines.append("| Method | `dead_zone` Acc [95% CI] | `goodfire_lt_p90` Acc [95% CI] | Delta (`<p90 - dead_zone`) |")
        lines.append("|---|---:|---:|---:|")
        for method in METHOD_ORDER:
            dz = dt["dead_zone"]["p99.9_plus"][method]
            gf = dt["goodfire_lt_p99.9" if False else "goodfire_lt_p90"]["p99.9_plus"][method]
            delta = (
                gf["mean"] - dz["mean"]
                if (math.isfinite(gf["mean"]) and math.isfinite(dz["mean"]))
                else float("nan")
            )
            lines.append(
                f"| `{method}` | {_fmt_floor_ci(method, dz['mean'], dz['ci_low'], dz['ci_high'])} | "
                f"{_fmt_floor_ci(method, gf['mean'], gf['ci_low'], gf['ci_high'])} | "
                f"{f'{delta:+.3f}' if math.isfinite(delta) else '—'} |"
            )
        lines.append("")

    lines.append(
        f"### 4b. Nearest Same-Layer Neuron (Descriptive, Tier 0 Only) (`{NEIGHBOUR_DISTRACTOR_TYPE}`) vs. Dead-Zone"
    )
    nb_cos_info = metrics.get("structural_checks", {}).get("nearest_wgate_tier0_summary")
    if isinstance(nb_cos_info, Mapping):
        s50 = nb_cos_info.get("sampled_50_neurons_wgate_cosine_stats", {})
        sp = nb_cos_info.get("surviving_tier0_pairs_wgate_cosine_stats", {})
        lines.append(
            f"- **Nearest same-layer neuron (descriptive, Tier 0 only) `w_gate` cosine stats:** "
            f"All 50 sampled neurons: `min={_fmt_num(s50.get('min'))}, p25={_fmt_num(s50.get('p25'))}, "
            f"median={_fmt_num(s50.get('median'))}, p75={_fmt_num(s50.get('p75'))}, "
            f"p90={_fmt_num(s50.get('p90'))}, max={_fmt_num(s50.get('max'))}` | "
            f"Surviving Tier-0 pairs (`n={sp.get('n', '?')}`, `{nb_cos_info.get('surviving_poles', '?')}` poles / "
            f"`{nb_cos_info.get('surviving_neurons', '?')}` neurons): "
            f"`min={_fmt_num(sp.get('min'))}, p25={_fmt_num(sp.get('p25'))}, "
            f"median={_fmt_num(sp.get('median'))}, p75={_fmt_num(sp.get('p75'))}, "
            f"p90={_fmt_num(sp.get('p90'))}, max={_fmt_num(sp.get('max'))}`."
        )
    for cname, clabel in (
        ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary)"),
        ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
    ):
        dt = cohort_dt.get(cname)
        if not dt:
            continue
        if dt["dead_zone"]["p99.9_plus"]["own"]["n_scored"] == 0 and dt[NEIGHBOUR_DISTRACTOR_TYPE]["p99.9_plus"]["own"]["n_scored"] == 0:
            continue
        lines.append(f"#### {clabel}")
        lines.append(
            f"| Band | Method | Matched-Pole `dead_zone` [95% CI] | Full-Cohort `dead_zone` | "
            f"Nearest Same-Layer Neuron (`{NEIGHBOUR_DISTRACTOR_TYPE}`) [95% CI] | Delta vs. Matched `dead_zone` | Delta vs. Full `dead_zone` |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for band in ("p99.9_plus", "p90_97"):
            for method in METHOD_ORDER:
                dz = dt["dead_zone"][band][method]
                dz_m = dt.get("dead_zone_matched_neighbour", {}).get(band, {}).get(method, dz)
                nb = dt[NEIGHBOUR_DISTRACTOR_TYPE][band][method]
                d_matched = (
                    nb["mean"] - dz_m["mean"]
                    if (math.isfinite(nb["mean"]) and math.isfinite(dz_m["mean"]))
                    else float("nan")
                )
                d_full = (
                    nb["mean"] - dz["mean"]
                    if (math.isfinite(nb["mean"]) and math.isfinite(dz["mean"]))
                    else float("nan")
                )
                lines.append(
                    f"| `{BAND_DISPLAY[band]}` | `{method}` | "
                    f"{_fmt_floor_ci(method, dz_m['mean'], dz_m['ci_low'], dz_m['ci_high'])} | "
                    f"{_fmt_floor_num(method, dz['mean'])} | "
                    f"{_fmt_floor_ci(method, nb['mean'], nb['ci_low'], nb['ci_high'])} | "
                    f"{f'{d_matched:+.3f}' if math.isfinite(d_matched) else '—'} | "
                    f"{f'{d_full:+.3f}' if math.isfinite(d_full) else '—'} |"
                )
        lines.append("")

    if metrics["distractor_tiers"]:
        lines.append(
            f"- **Distractor tier counts (Tier 0 ONLY required for nearest same-layer neuron `{NEIGHBOUR_DISTRACTOR_TYPE}`):** "
            f"`{json.dumps(metrics['distractor_tiers'], sort_keys=True)}`"
        )

    # 5. Cross-judge agreement (Secondary judge & Third judge; heldout_80, pilot_20, and all_100)
    lines.append("")
    lines.append("## 5. Second-Judge Agreement (Secondary & Third Judge)")
    if not metrics["second_judge_agreement"]:
        lines.append("_No secondary or third-judge predictions found in `results/judgements.jsonl` yet._")
    else:
        cohort_agree = metrics.get("cohort_second_judge_agreement", {})
        for cname, clabel in (
            ("heldout_80", "Held-Out 80 (`in_pilot=False`, Primary Confirmatory)"),
            ("pilot_20", "Pilot 20 (`in_pilot=True`, Calibration)"),
            ("all_100", "All Judged Poles Pooled (Incl. Full 30-Pole Third-Judge Overlap)"),
        ):
            amap = cohort_agree.get(cname, metrics["second_judge_agreement"] if cname == "all_100" else {})
            if not any(amap.values()):
                continue
            lines.append(f"### {clabel}")
            lines.append(
                f"| Comparison Judge | Role | Scope | Overlap Pairs | Poles | `{primary}` Acc | Comp. Judge Acc | Exact Agree | Cohen's κ | Per-Pole Spearman ρ (p) | Mean |Δ| / Pole |"
            )
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
            for sec_model, smap in amap.items():
                role = judge_roles.get(sec_model, _judge_role_label(sec_model, primary))
                if cname == "heldout_80" and "sol" in sec_model.lower():
                    role = "third judge (descriptive, n=10 held-out poles / 5 neurons)"
                for scope, st in smap.items():
                    rho_str = (
                        f"{st['spearman_rho']:.3f} (p={st['spearman_pvalue']:.3g})"
                        if math.isfinite(st["spearman_rho"])
                        else "—"
                    )
                    lines.append(
                        f"| `{sec_model}` | {role} | `{scope}` | {st['n_overlap_pairs']} | {st['n_poles']} | "
                        f"{_fmt_num(st['primary_mean_acc'])} | {_fmt_num(st['secondary_mean_acc'])} | "
                        f"{_fmt_num(st['exact_agreement'])} | {_fmt_num(st['cohens_kappa'])} | "
                        f"{rho_str} | {_fmt_num(st['mean_abs_pole_diff'])} |"
                    )
            lines.append("")

    # 6. Position bias, exclusions & structural checks
    lines.append("## 6. First-Position Bias, Unparseable/Tie Exclusions & Structural Sanity Checks")
    cohort_pb = metrics.get("cohort_position_bias", {"all_100": metrics["position_bias"]})
    for cname, clabel in (
        ("heldout_80", "Held-Out 80 (`in_pilot=False`)"),
        ("pilot_20", "Pilot 20 (`in_pilot=True`)"),
        ("all_100", "All Judged Poles Pooled"),
    ):
        pb = cohort_pb.get(cname, {})
        if not any(pb.values()):
            continue
        lines.append(f"### {clabel}")
        lines.append(
            "| Judge Model | Role | Scope | Calls | Scored | Unparseable | Logprob Ties | `P(pred=A)` | `P(pos=A)` | Slot-A Acc | Slot-B Acc | `A - B` |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model, mmap in pb.items():
            role = judge_roles.get(model, _judge_role_label(model, primary))
            if cname == "heldout_80" and "sol" in model.lower():
                role = "third judge (descriptive, n=10 held-out poles / 5 neurons)"
            for scope, st in mmap.items():
                gap = st["slot_A_minus_B"]
                lines.append(
                    f"| `{model}` | {role} | `{scope}` | {st['n_calls']} | {st['n_scored']} | "
                    f"{st['n_unparseable']} | {st['n_logprob_tie']} | "
                    f"{_fmt_num(st['p_pred_A'])} | {_fmt_num(st['p_pos_slot_A'])} | "
                    f"{_fmt_num(st['slot_A_acc'])} | {_fmt_num(st['slot_B_acc'])} | "
                    f"{f'{gap:+.3f}' if math.isfinite(gap) else '—'} |"
                )
        lines.append("")

    if metrics["structural_checks"]:
        lines.append("### Structural Invariant Verification")
        for k, v in sorted(metrics["structural_checks"].items()):
            lines.append(f"- `{k}`: `{v}`")

    # 7. Cost audit
    lines.append("")
    lines.append("## 7. Live API Cost & Token Spend Audit")
    lines.append(
        "| Judge Model | Role | Backend | Logprobs | Calls | Scored / Unp / Tie | Mean In Tok | Mean Vis Out | Mean Reas Tok | Avg $/Call | Total USD |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    core_usd_total = 0.0
    for model, st in metrics["cost_audit"].items():
        role = judge_roles.get(model, _judge_role_label(model, primary))
        core_usd_total += float(st["total_cost_usd"])
        lines.append(
            f"| `{model}` | {role} | `{st['backend']}` | {'yes' if st['has_logprobs'] else 'no'} | "
            f"{st['calls']:,} | `{st['n_scored']}/{st['n_unparseable']}/{st['n_logprob_tie']}` | "
            f"{st['mean_prompt_tokens']:.1f} | {st['mean_visible_output_tokens']:.1f} | "
            f"{st['mean_reasoning_tokens']:.1f} | ${st['mean_cost_usd_per_call']:.6f} | "
            f"${st['total_cost_usd']:.4f} |"
        )
    lines.append(f"- **Core 3-judge spend (`results/judgements.jsonl`):** **${core_usd_total:.4f}**")
    ctx = metrics.get("context_ablation")
    ex_arm = metrics.get("examples_as_label")
    extra_usd = 0.0
    if ctx:
        ctx_usd = float(ctx.get("total_cost_usd", 0.0))
        extra_usd += ctx_usd
        lines.append(
            f"- **Exploratory context-ablation spend "
            f"(`results/judgements_unfiltered_context_ablation.jsonl`):** **${ctx_usd:.4f}** "
            f"(`{ctx['n_calls']:,}` calls, `{ctx['n_scored']}/{ctx['n_unparseable']}/{ctx['n_logprob_tie']}`)"
        )
    if ex_arm:
        ex_usd = float(ex_arm.get("total_cost_usd", 0.0))
        extra_usd += ex_usd
        lines.append(
            f"- **Exploratory `examples_as_label` spend "
            f"(`results/judgements_examples_as_label_exploratory.jsonl`):** **${ex_usd:.4f}** "
            f"(`{ex_arm['n_calls']:,}` calls, `{ex_arm['n_scored']}/{ex_arm['n_unparseable']}/{ex_arm['n_logprob_tie']}`, "
            f"mean `{ex_arm['mean_prompt_tokens']:.1f}` in tok / `{ex_arm['mean_visible_output_tokens']:.1f}` vis out / `{ex_arm['mean_reasoning_tokens']:.1f}` reas tok)"
        )
    if ctx or ex_arm:
        lines.append(f"- **Grand total judge spend (core + exploratory):** **${core_usd_total + extra_usd:.4f}**")

    # 8. Limitations (including weak-band clean-window survival shares)
    lines.append("")
    lines.append("## 8. Limitations")
    lines.append(
        "- **Model scale & architecture boundary:** This audit evaluates dense `Qwen/Qwen3-1.7B` (SwiGLU MLP neurons) "
        "rather than Goodfire's 35B MoE; dense 1.7B neurons are substantially more polysemantic than large MoE experts."
    )
    lines.append(
        "- **Nearest same-layer neuron (descriptive, Tier 0 only) cosine strength:** Dense 1.7B same-layer nearest "
        "`w_gate` neighbours have modest cosine similarity (`median=0.174` across 50 sampled neurons; `median=0.155` "
        "across `308` surviving Tier-0 pairs / `40` poles / `29` neurons), so this arm is strictly descriptive rather "
        "than a high-cosine near-duplicate stress test."
    )
    lines.append(
        "- **Goodfire `<p90` bipolar opposite-pole firing:** Filtering only the target pole `< p90` leaves the opposite "
        "pole firing `>= p90` on `66/600` (`11.0%`) and `>= p97` on `18/600` (`3.0%`) of distractor target tokens."
    )
    lines.append(
        "- **Clean-window selection effect in weak bands:** Filtering out non-target context tokens `>= p90` "
        "isolates the marked weak-band token cleanly, but selects the subset of weak-band occurrences without strong "
        "co-occurring context peaks. Exact corpus survival shares are reported directly below (and the "
        "unfiltered exploratory ablation in Section 9 quantifies the context-peak effect directly)."
    )
    survival = metrics.get("weak_band_survival")
    if isinstance(survival, Mapping) and survival:
        lines.append("")
        _render_weak_band_survival_block(lines, survival)
    else:
        lines.append("- _Weak-band survival JSON (`results/weak_band_survival.json`) not loaded._")

    # 9. Exploratory Context Ablation
    if ctx:
        lines.append("")
        lines.append("---")
        lines.append("## 9. EXPLORATORY CONTEXT ABLATION: Unfiltered Weak-Band Positive Windows (Context Effect)")
        lines.append(
            "> **Status: Exploratory.** These `4,800` calls were run "
            "using the frozen prompt template and labels on `data/snippets_unfiltered_context_ablation.jsonl` and stored "
            "strictly separately in `results/judgements_unfiltered_context_ablation.jsonl` (zero unfiltered rows in "
            "`results/judgements.jsonl`). Companion plot: `figures/context_ablation.png`."
        )
        lines.append(
            f"- **Judge:** `{ctx['judge_model']}` | **Calls:** `{ctx['n_calls']:,}` "
            f"(`{ctx['n_scored']}` scored / `{ctx['n_unparseable']}` unparseable / `{ctx['n_logprob_tie']}` ties) | "
            f"**Spend:** `${ctx['total_cost_usd']:.4f}` (`${ctx['mean_cost_usd_per_call']:.6f}`/call)"
        )
        if ctx.get("leakage_by_band"):
            for band in WEAK_BANDS:
                lk = ctx["leakage_by_band"].get(band)
                if lk:
                    lines.append(
                        f"- **Unfiltered `{BAND_DISPLAY[band]}` positive context peaks (`n={int(lk['n_pairs'])}` pairs):** "
                        f"`other >= p90` = **{100.0 * lk['pos_other_ge_p90_rate']:.2f}%**, "
                        f"`window >= p97` = **{100.0 * lk['pos_window_ge_p97_rate']:.2f}%**, "
                        f"`slot_A` = `{lk['slot_A_rate']:.3f}`"
                    )
        lines.append("")
        for cname, clabel in (
            ("heldout_80", "9.1 Held-Out 80 (`in_pilot=False`) — Exploratory Clean vs. Unfiltered"),
            ("pilot_20", "9.2 Pilot 20 (`in_pilot=True`) — Exploratory Clean vs. Unfiltered"),
            ("all_100", "9.3 All 100 Poles Pooled — Exploratory Clean vs. Unfiltered"),
        ):
            cdata = ctx.get("cohorts", {}).get(cname, {})
            if not cdata:
                continue
            lines.append(f"### {clabel}")
            lines.append(
                "| Band | Condition | Clean (<p90) Acc [95% Neuron CI] | Unfiltered Acc [95% Neuron CI] | Delta (`unfiltered - clean`) | Unfiltered `scored/unp/tie` |"
            )
            lines.append("|---|---|---:|---:|---:|---:|")
            for band in WEAK_BANDS:
                for cond in ALL_CONDITIONS:
                    entry = cdata.get(band, {}).get(cond)
                    if not entry:
                        continue
                    cl = entry["clean"]
                    un = entry["unfiltered"]
                    delta = entry["delta_unfiltered_minus_clean"]
                    lines.append(
                        f"| `{BAND_DISPLAY[band]}` | `{cond}` | "
                        f"{_fmt_floor_ci(cond, cl['mean'], cl['ci_low'], cl['ci_high'])} | "
                        f"{_fmt_floor_ci(cond, un['mean'], un['ci_low'], un['ci_high'])} | "
                        f"{f'{delta:+.3f}' if math.isfinite(delta) else '—'} | "
                        f"`{un['n_scored']}/{un['n_unparseable']}/{un['n_logprob_tie']}` |"
                    )
            lines.append("")

    # 10. Exploratory Arm: 20 Examples as Label
    if ex_arm:
        lines.append("")
        lines.append("---")
        lines.append("## 10. EXPLORATORY ARM: 20 Top-Band Examples as Label (`examples_as_label`)")
        lines.append(
            "> **Status: Exploratory.** These `1,200` calls (`960 heldout_80` + `240 pilot_20`) "
            "were run on the exact same `p97_99.9` and `p99.9_plus` `dead_zone` pairs and `pos_slot` "
            "order from `data/snippets.jsonl`, replacing only the summary description with the pole's 20 disjoint `train_label` "
            "examples from `data/label_examples.jsonl`. Stored strictly separately in "
            "`results/judgements_examples_as_label_exploratory.jsonl` (zero exploratory rows in `results/judgements.jsonl`)."
        )
        lines.append(
            f"- **Judge:** `{ex_arm['judge_model']}` | **Calls:** `{ex_arm['n_calls']:,}` "
            f"(`{ex_arm['n_scored']}` scored / `{ex_arm['n_unparseable']}` unparseable / `{ex_arm['n_logprob_tie']}` ties) | "
            f"**Tokens:** `{ex_arm['prompt_tokens']:,}` in (`{ex_arm['mean_prompt_tokens']:.1f}`/call), "
            f"`{ex_arm['visible_output_tokens']:,}` vis out (`{ex_arm['mean_visible_output_tokens']:.1f}`/call), "
            f"`{ex_arm['reasoning_tokens']:,}` reas tok | **Spend:** `${ex_arm['total_cost_usd']:.4f}` "
            f"(`${ex_arm['mean_cost_usd_per_call']:.6f}`/call) | **`P(pred=A)`:** `{_fmt_num(ex_arm['p_pred_A'])}`"
        )
        lines.append("")
        for cname, clabel in (
            ("heldout_80", "10.1 Held-Out 80 (`in_pilot=False`) — `examples_as_label` vs. `own` & `token_only`"),
            ("pilot_20", "10.2 Pilot 20 (`in_pilot=True`) — `examples_as_label` vs. `own` & `token_only`"),
        ):
            cdata = ex_arm.get("cohorts", {}).get(cname, {})
            if not cdata:
                continue
            lines.append(f"### {clabel}")
            lines.append(
                "| Band | `examples_as_label` [95% CI] | `own` [95% CI] | Paired `examples - own` [95% CI] | "
                "`token_only` [95% CI] | Paired `examples - token_only` [95% CI] | `scored/unp/tie` | `P(pred=A)` |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for band in ("p97_99.9", "p99.9_plus"):
                b = cdata.get(band)
                if not b:
                    continue
                ex_s = b["examples_as_label"]
                ow_s = b["own"]
                tk_s = b["token_only"]
                do_s = b["paired_examples_minus_own"]
                dt_s = b["paired_examples_minus_token_only"]
                do_str = (
                    f"{do_s['mean']:+.3f} [{do_s['ci_low']:+.3f}, {do_s['ci_high']:+.3f}]"
                    if math.isfinite(do_s["mean"])
                    else "—"
                )
                dt_str = (
                    f"{dt_s['mean']:+.3f} [{dt_s['ci_low']:+.3f}, {dt_s['ci_high']:+.3f}]"
                    if math.isfinite(dt_s["mean"])
                    else "—"
                )
                lines.append(
                    f"| `{BAND_DISPLAY[band]}` | "
                    f"{_fmt_ci(ex_s['mean'], ex_s['ci_low'], ex_s['ci_high'])} | "
                    f"{_fmt_ci(ow_s['mean'], ow_s['ci_low'], ow_s['ci_high'])} | "
                    f"{do_str} | "
                    f"{_fmt_floor_ci('token_only', tk_s['mean'], tk_s['ci_low'], tk_s['ci_high'])} | "
                    f"{dt_str} | "
                    f"`{ex_s['n_scored']}/{ex_s['n_unparseable']}/{ex_s['n_logprob_tie']}` | "
                    f"{_fmt_num(b['p_pred_A'])} |"
                )
            lines.append("")

            ly_map = ex_arm.get("by_layer", {}).get(cname, {})
            if ly_map:
                lines.append(f"#### Per-Layer Breakdown (`{cname}`)")
                lines.append(
                    "| Layer | Band | `n_poles / n_neurons` | `examples_as_label` [95% CI] | `own` [95% CI] | Paired `examples - own` [95% CI] |"
                )
                lines.append("|---|---|---:|---:|---:|---:|")
                for ly in sorted(ly_map.keys()):
                    tag = f"L{ly} (control)" if ly == 4 else f"L{ly}"
                    for band in ("p97_99.9", "p99.9_plus"):
                        cell = ly_map[ly].get(band)
                        if not cell or cell["examples_as_label"]["n_poles"] == 0:
                            continue
                        ex_s = cell["examples_as_label"]
                        ow_s = cell["own"]
                        do_s = cell["paired_examples_minus_own"]
                        do_str = (
                            f"{do_s['mean']:+.3f} [{do_s['ci_low']:+.3f}, {do_s['ci_high']:+.3f}]"
                            if math.isfinite(do_s["mean"])
                            else "—"
                        )
                        lines.append(
                            f"| {tag} | `{BAND_DISPLAY[band]}` | `{ex_s['n_poles']} / {ex_s['n_neurons']}` | "
                            f"{_fmt_ci(ex_s['mean'], ex_s['ci_low'], ex_s['ci_high'])} | "
                            f"{_fmt_ci(ow_s['mean'], ow_s['ci_low'], ow_s['ci_high'])} | "
                            f"{do_str} |"
                        )
                lines.append("")

    # 11. Zero-cost sub-band split of p97_99.9 into p97-99 (< p99) vs p99-99.9 (>= p99)
    subband_split = metrics.get("p97_p99_subband_split")
    if subband_split and subband_split.get("cohorts"):
        ver = subband_split.get("verification", {})
        pchecks = ver.get("partition_checks", {})
        h_pc = pchecks.get("heldout_80", {})
        p_pc = pchecks.get("pilot_20", {})
        lines.append("---")
        lines.append("## 11. Zero-Cost Sub-Band Split of `p97–99.9` (`p97–99` vs. `p99–99.9`, `dead_zone`)")
        lines.append(
            f"- **Threshold verification:** `p97 < p99 < p99.9` strictly ordered = `{ver.get('all_strictly_ordered')}` "
            f"across `{ver.get('n_poles_with_thresholds')}` poles (`max_abs_err_vs_poles_json = {ver.get('max_abs_err_vs_poles_json', 0.0):.3e}`, "
            f"`out_of_range = {ver.get('out_of_range_count', 0)}`). "
            f"Exact pair partitions: `heldout_80` `{h_pc.get('n_p97_99', 0)} + {h_pc.get('n_p99_99.9', 0)} = {h_pc.get('n_total', 0)}`, "
            f"`pilot_20` `{p_pc.get('n_p97_99', 0)} + {p_pc.get('n_p99_99.9', 0)} = {p_pc.get('n_total', 0)}`."
        )
        lines.append("")

        sb_display = {"p97_99": "p97–99", "p99_99.9": "p99–99.9"}
        for cname, heading in (
            ("heldout_80", "### 11.1 Held-Out 80 (`in_pilot=False`, Primary)"),
            ("pilot_20", "### 11.2 Pilot 20 (`in_pilot=True`, Calibration)"),
        ):
            j_map = subband_split["cohorts"].get(cname, {})
            if not j_map:
                continue
            lines.append(heading)
            lines.append(
                "| Judge Model | Sub-Band | `n_pairs / n_poles / n_neurons` | `own` [95% CI] | `token_only` [95% CI] | "
                "`no_label` [95% CI] | `wrong` [95% CI] | `examples_as_label` (exploratory) [95% CI] | "
                "Paired `own - token_only` [95% CI] | Paired `examples - own` [95% CI] |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for jm in subband_split.get("judges_reported", []):
                sb_map = j_map.get(jm, {})
                for sb in ("p97_99", "p99_99.9"):
                    entry = sb_map.get(sb)
                    if not entry or entry.get("n_pairs", 0) == 0:
                        continue
                    conds = entry["conditions"]
                    ow = conds["own"]
                    tk = conds["token_only"]
                    nl = conds["no_label"]
                    wr = conds["wrong"]
                    ex = conds.get("examples_as_label")
                    ex_str = (
                        _fmt_ci(ex["mean"], ex["ci_low"], ex["ci_high"])
                        if ex and ex.get("n_poles", 0) > 0
                        else "—"
                    )
                    ot = entry["paired_own_minus_token_only"]
                    ot_str = (
                        f"{ot['mean']:+.3f} [{ot['ci_low']:+.3f}, {ot['ci_high']:+.3f}]"
                        if math.isfinite(ot["mean"])
                        else "—"
                    )
                    eo = entry.get("paired_examples_minus_own")
                    eo_str = (
                        f"{eo['mean']:+.3f} [{eo['ci_low']:+.3f}, {eo['ci_high']:+.3f}]"
                        if eo and math.isfinite(eo["mean"])
                        else "—"
                    )
                    lines.append(
                        f"| `{jm}` | `{sb_display[sb]}` | `{entry['n_pairs']} / {entry['n_poles']} / {entry['n_neurons']}` | "
                        f"{_fmt_ci(ow['mean'], ow['ci_low'], ow['ci_high'])} | "
                        f"{_fmt_floor_ci('token_only', tk['mean'], tk['ci_low'], tk['ci_high'])} | "
                        f"{_fmt_floor_ci('no_label', nl['mean'], nl['ci_low'], nl['ci_high'])} | "
                        f"{_fmt_floor_ci('wrong', wr['mean'], wr['ci_low'], wr['ci_high'])} | "
                        f"{ex_str} | {ot_str} | {eo_str} |"
                    )
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def _select_stratified_30_poles(
    poles_by_id: Mapping[str, Mapping[str, Any]],
    judgements: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    primary_judge: str,
    target_n: int = 30,
) -> list[tuple[str, str]]:
    """Select up to 30 stratified poles for manual audit."""
    primary_rows = [
        r
        for r in judgements
        if str(r["judge_model"]) == primary_judge
        and str(r["distractor_type"]) == "dead_zone"
        and classify_row_outcome(r) == "scored"
    ]
    judged_poles = sorted({str(r["pole_id"]) for r in primary_rows})
    if not judged_poles:
        judged_poles = sorted(poles_by_id.keys())

    acc_map: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in primary_rows:
        acc_map[(str(r["pole_id"]), str(r["band"]), str(r["condition"]))].append(float(r["correct"]))

    base_map: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for b in baselines:
        if str(b["distractor_type"]) == "dead_zone":
            base_map[(str(b["pole_id"]), str(b["band"]), str(b["rule"]))].append(float(b["correct"]))

    sec_diffs: dict[str, float] = {}
    sec_rows = [
        r for r in judgements if str(r["judge_model"]) != primary_judge and classify_row_outcome(r) == "scored"
    ]
    if sec_rows:
        p1_all: dict[str, list[float]] = defaultdict(list)
        p2_all: dict[str, list[float]] = defaultdict(list)
        for r in primary_rows:
            p1_all[str(r["pole_id"])].append(float(r["correct"]))
        for r in sec_rows:
            p2_all[str(r["pole_id"])].append(float(r["correct"]))
        for pid in judged_poles:
            if p1_all[pid] and p2_all[pid]:
                sec_diffs[pid] = abs(float(np.mean(p2_all[pid])) - float(np.mean(p1_all[pid])))

    def _mean(vals: Sequence[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    selected: list[tuple[str, str]] = []
    chosen: set[str] = set()

    def _pick(candidates: Sequence[str], bucket_name: str, quota: int) -> None:
        taken = 0
        for pid in candidates:
            if len(selected) >= target_n or taken >= quota:
                break
            if pid not in chosen:
                chosen.add(pid)
                selected.append((pid, bucket_name))
                taken += 1

    collapse_ranked = sorted(
        judged_poles,
        key=lambda pid: (
            -(
                (_mean(acc_map[(pid, "p99.9_plus", "own")]) if acc_map[(pid, "p99.9_plus", "own")] else 0.0)
                - (
                    _mean(acc_map[(pid, "p90_97", "own")] + acc_map[(pid, "p75_90", "own")])
                    if (acc_map[(pid, "p90_97", "own")] or acc_map[(pid, "p75_90", "own")])
                    else 0.0
                )
            ),
            pid,
        ),
    )
    _pick(collapse_ranked, "Sub-p97 Collapse", 10)

    survivor_ranked = sorted(
        judged_poles,
        key=lambda pid: (
            -(
                _mean(acc_map[(pid, "p75_90", "own")] + acc_map[(pid, "p90_97", "own")])
                if (acc_map[(pid, "p75_90", "own")] or acc_map[(pid, "p90_97", "own")])
                else -1.0
            ),
            -(_mean(acc_map[(pid, "p99.9_plus", "own")]) if acc_map[(pid, "p99.9_plus", "own")] else -1.0),
            pid,
        ),
    )
    _pick(survivor_ranked, "Clean Weak-Band Survivor", 10)

    def _q3_share(pid: str) -> float:
        p = poles_by_id.get(pid, {})
        qmap = p.get("quadrant_shares_by_band", {})
        vals = [
            float(qmap.get(b, {}).get("Q3", 0.0))
            for b in ("p75_90", "p90_97")
            if isinstance(qmap.get(b), Mapping)
        ]
        return float(np.mean(vals)) if vals else 0.0

    q3_ranked = sorted(judged_poles, key=lambda pid: (-_q3_share(pid), pid))
    _pick(q3_ranked, "High-Q3 Double-Negative Pole", 5)

    def _floor_score(pid: str) -> float:
        nl = _mean(acc_map[(pid, "p99.9_plus", "no_label")]) if acc_map[(pid, "p99.9_plus", "no_label")] else 0.0
        rr = _mean(base_map[(pid, "p99.9_plus", "rarity")]) if base_map[(pid, "p99.9_plus", "rarity")] else 0.0
        sd = sec_diffs.get(pid, 0.0)
        return max(nl, rr) + sd

    floor_ranked = sorted(judged_poles, key=lambda pid: (-_floor_score(pid), pid))
    _pick(floor_ranked, "High-Floor / Judge-Disagreement Notable", 5)

    backfill = sorted(
        judged_poles,
        key=lambda pid: (
            0 if poles_by_id.get(pid, {}).get("in_second_judge") else 1,
            0 if poles_by_id.get(pid, {}).get("in_pilot") else 1,
            pid,
        ),
    )
    _pick(backfill, "Stratified Backfill", target_n)
    return selected[:target_n]


def generate_audit_worksheet_md(
    poles_data: Mapping[str, Any],
    snippets: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    judgements: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    primary_judge: str | None = None,
    target_n: int = 30,
) -> str:
    """Format up to 30 stratified raw poles for manual reading."""
    poles_by_id = {str(p["pole_id"]): p for p in poles_data.get("poles", [])}
    labels_by_id = {str(r["pole_id"]): r for r in labels}
    primary = _select_primary_judge(judgements, primary_judge)

    selected = _select_stratified_30_poles(
        poles_by_id=poles_by_id,
        judgements=judgements,
        baselines=baselines,
        primary_judge=primary,
        target_n=target_n,
    )

    snips_by_pole_band: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for s in snippets:
        if str(s["distractor_type"]) == "dead_zone":
            snips_by_pole_band[(str(s["pole_id"]), str(s["band"]))].append(s)

    j_by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    j_by_pole_band_cond: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    j_by_model_pole_band_cond: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    other_models = sorted({str(r["judge_model"]) for r in judgements if str(r["judge_model"]) != primary})
    for r in judgements:
        j_by_pair[str(r["pair_id"])].append(r)
        if str(r["distractor_type"]) == "dead_zone" and classify_row_outcome(r) == "scored":
            j_by_model_pole_band_cond[
                (str(r["judge_model"]), str(r["pole_id"]), str(r["band"]), str(r["condition"]))
            ].append(float(r["correct"]))
            if str(r["judge_model"]) == primary:
                j_by_pole_band_cond[(str(r["pole_id"]), str(r["band"]), str(r["condition"]))].append(
                    float(r["correct"])
                )

    b_by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    b_by_pole_band_rule: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for b in baselines:
        b_by_pair[str(b["pair_id"])][str(b["rule"])] = b
        if str(b["distractor_type"]) == "dead_zone":
            b_by_pole_band_rule[(str(b["pole_id"]), str(b["band"]), str(b["rule"]))].append(
                float(b["correct"])
            )

    lines: list[str] = []
    lines.append(f"# Manual Audit Worksheet: {len(selected)} Stratified Qwen3-1.7B Neuron Poles")
    lines.append("")
    lines.append(
        "**Purpose:** Manual raw-pole reading worksheet. "
        "Read the actual marked snippets (`<<token>>`) across activation bands, compare "
        "against the generated label, Q3 double-negative flags, rule baselines, and judge verdicts, "
        "and record short observations."
    )
    other_desc = ", ".join(f"`{m}` ({_judge_role_label(m, primary)})" for m in other_models)
    lines.append(
        f"**Primary judge:** `{primary}`" + (f" | **Additional judges shown:** {other_desc}" if other_desc else "")
    )
    lines.append("")

    lines.append("## Selection Summary")
    lines.append("| # | Pole ID | Cohort | Layer | Bucket | `p99.9+` own | `p90–97` own | `p75–90` own | `p99.9+` no_label |")
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|")
    for idx, (pid, bucket) in enumerate(selected, start=1):
        p = poles_by_id.get(pid, {})
        layer = p.get("layer", "?")
        cohort_tag = "pilot_20" if p.get("in_pilot") else "heldout_80"
        vals = []
        for band, cond in (
            ("p99.9_plus", "own"),
            ("p90_97", "own"),
            ("p75_90", "own"),
            ("p99.9_plus", "no_label"),
        ):
            vs = j_by_pole_band_cond[(pid, band, cond)]
            vals.append(f"{np.mean(vs):.2f}" if vs else "—")
        lines.append(f"| {idx} | `{pid}` | `{cohort_tag}` | L{layer} | {bucket} | " + " | ".join(vals) + " |")

    for idx, (pid, bucket) in enumerate(selected, start=1):
        p = poles_by_id.get(pid, {})
        lbl = labels_by_id.get(pid, {})
        cohort_badge = " **[PILOT_20]**" if p.get("in_pilot") else " **[HELDOUT_80]**"
        ctrl_tag = " **[LAYER-4 POSITIVE CONTROL]**" if p.get("is_layer4_control") else ""
        lines.append("")
        lines.append("---")
        lines.append(f"## Pole {idx}/{len(selected)}: `{pid}` ({bucket}){cohort_badge}{ctrl_tag}")
        lines.append(
            f"- **Layer / Neuron / Sign:** L{p.get('layer')} / N{p.get('neuron')} / `{p.get('pole_name')}` "
            f"| **Nearest Same-Layer Neuron (Descriptive):** N{p.get('neighbor_neuron')} "
            f"(gate cos = `{p.get('neighbor_gate_cosine')}`)"
        )
        lines.append(f"- **Top-3 trigger tokens:** `{lbl.get('top_3_tokens', p.get('top_3_tokens', []))}`")
        if p.get("quadrant_shares_by_band"):
            q_str = ", ".join(
                f"{BAND_DISPLAY[b]}: Q3={p['quadrant_shares_by_band'].get(b, {}).get('Q3', 0):.2f}"
                for b in ALL_BANDS
                if b in p["quadrant_shares_by_band"]
            )
            lines.append(f"- **Corpus Q3 (double-negative) shares:** {q_str}")
        lines.append(f"- **Own label:** {lbl.get('own_label', '_missing_')}")
        lines.append(
            f"- **Wrong label** (donor `{lbl.get('wrong_donor_pole_id')}`, L{lbl.get('wrong_donor_layer')}, "
            f"sim={lbl.get('wrong_embed_cosine')}): {lbl.get('wrong_label', '_missing_')}"
        )
        lines.append(f"- **Token-only label:** `{lbl.get('token_only_label', '_missing_')}`")
        lines.append("")

        lines.append("| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |")
        lines.append("|---|---:|---:|---:|---:|")
        for cond in ALL_CONDITIONS:
            row_vals = [
                f"{np.mean(j_by_pole_band_cond[(pid, b, cond)]):.2f}"
                if j_by_pole_band_cond[(pid, b, cond)]
                else "—"
                for b in ALL_BANDS
            ]
            lines.append(f"| `{primary}` `{cond}` | " + " | ".join(row_vals) + " |")
        for om in other_models:
            has_om = any(j_by_model_pole_band_cond[(om, pid, b, "own")] for b in ALL_BANDS)
            if not has_om:
                continue
            role = _judge_role_label(om, primary)
            row_vals = [
                f"{np.mean(j_by_model_pole_band_cond[(om, pid, b, 'own')]):.2f}"
                if j_by_model_pole_band_cond[(om, pid, b, "own")]
                else "—"
                for b in ALL_BANDS
            ]
            lines.append(f"| `{om}` ({role}) `own` | " + " | ".join(row_vals) + " |")
        for rule in ("rarity", "top_token", "window_has_ge_p97"):
            row_vals = [
                f"{np.mean(b_by_pole_band_rule[(pid, b, rule)]):.2f}"
                if b_by_pole_band_rule[(pid, b, rule)]
                else "—"
                for b in ALL_BANDS
            ]
            lines.append(f"| `{rule}` | " + " | ".join(row_vals) + " |")

        lines.append("")
        lines.append("### Representative Marked Snippets (`dead_zone` Pairs)")
        for band in ("p99.9_plus", "p97_99.9", "p90_97", "p75_90"):
            band_snips = snips_by_pole_band.get((pid, band), [])
            if not band_snips:
                continue
            if band in ("p75_90", "p90_97"):
                ordered_snips = sorted(
                    band_snips,
                    key=lambda s: (0 if s.get("pos", {}).get("quadrant") == "Q3" else 1, int(s.get("pair_idx", 0))),
                )
            else:
                ordered_snips = sorted(band_snips, key=lambda s: int(s.get("pair_idx", 0)))

            for snip in ordered_snips[:1]:
                pair_id = str(snip["pair_id"])
                pos = snip["pos"]
                dist = snip["dist"]
                verdicts = []
                for jr in sorted(
                    j_by_pair.get(pair_id, []),
                    key=lambda x: (str(x["judge_model"]), str(x["condition"]), str(x.get("order") or "fwd")),
                ):
                    outc = classify_row_outcome(jr)
                    tag = "OK" if (outc == "scored" and jr["correct"] == 1.0) else ("MISS" if outc == "scored" else outc.upper())
                    verdicts.append(
                        f"{jr['judge_model']}/{jr['condition']}[{jr.get('order','fwd')}]={tag}({jr.get('predicted_slot')})"
                    )
                for rule_name, br in sorted(b_by_pair.get(pair_id, {}).items()):
                    verdicts.append(f"rule:{rule_name}={br['correct']:.1f}")

                lines.append(
                    f"- **`{BAND_DISPLAY[band]}` (`{pair_id}`, `pos_slot={snip.get('pos_slot')}`):** "
                    f"pos `act_dir={pos.get('act_dir'):.3f}` `quad={pos.get('quadrant')}` `gate_open={pos.get('gate_open')}` "
                    f"| dist `act_dir={dist.get('act_dir'):.3f}` `quad={dist.get('quadrant')}`"
                )
                if verdicts:
                    lines.append(f"  - *Verdicts:* {'; '.join(verdicts)}")
                pos_txt = str(pos.get("marked_text", "")).replace("\n", " ⏎ ")
                dist_txt = str(dist.get("marked_text", "")).replace("\n", " ⏎ ")
                lines.append(f"  - **FIRING:** `{pos_txt}`")
                lines.append(f"  - **DISTRACTOR:** `{dist_txt}`")

        lines.append("")
        lines.append("### Manual Reading Notes (fill during Sunday audit)")
        lines.append("- [ ] Top-band (`p99.9+`) label accurately matches snippets?")
        lines.append("- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?")
        lines.append("- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?")
        lines.append("- **Free notes:** _..._")

    lines.append("")
    return "\n".join(lines)


def run_analysis(
    poles_path: Path = Path("data/poles.json"),
    snippets_path: Path = Path("data/snippets.jsonl"),
    labels_path: Path = Path("data/labels.jsonl"),
    judgements_path: Path = Path("results/judgements.jsonl"),
    baselines_path: Path = Path("results/baselines.jsonl"),
    structural_path: Path = Path("results/structural_checks.json"),
    figure_path: Path = Path("figures/w_band_accuracy.png"),
    tables_path: Path = Path("results/summary_tables.md"),
    worksheet_path: Path = Path("audit/raw_30_poles_worksheet.md"),
    metrics_json_path: Path | None = Path("results/metrics.json"),
    primary_judge: str | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
    weak_band_survival_path: Path | None = None,
    unfiltered_judgements_path: Path | None = None,
    unfiltered_snippets_path: Path | None = None,
    examples_judgements_path: Path | None = None,
    context_figure_path: Path | None = None,
    main_figure_path: Path | None = None,
    cache_dir: Path | None = None,
    subband_json_path: Path | None = None,
) -> dict[str, Any]:
    """Run full statistical audit and write figures, tables, and 30-pole worksheet."""
    poles_data = json.loads(poles_path.read_text(encoding="utf-8"))
    snippets = load_jsonl(snippets_path)
    labels = load_jsonl(labels_path)
    judgements = load_jsonl(judgements_path)
    baselines = load_jsonl(baselines_path)
    structural_checks = (
        json.loads(structural_path.read_text(encoding="utf-8"))
        if structural_path.is_file()
        else None
    )
    weak_band_survival = (
        json.loads(weak_band_survival_path.read_text(encoding="utf-8"))
        if (weak_band_survival_path is not None and weak_band_survival_path.is_file())
        else None
    )
    unfiltered_judgements = (
        load_jsonl(unfiltered_judgements_path)
        if (unfiltered_judgements_path is not None and unfiltered_judgements_path.is_file())
        else None
    )
    unfiltered_snippets = (
        load_jsonl(unfiltered_snippets_path)
        if (unfiltered_snippets_path is not None and unfiltered_snippets_path.is_file())
        else None
    )
    examples_judgements = (
        load_jsonl(examples_judgements_path)
        if (examples_judgements_path is not None and examples_judgements_path.is_file())
        else None
    )
    p99_thresholds_payload = load_or_compute_p99_thresholds(
        poles_data.get("poles", []),
        cache_dir=cache_dir,
        subband_json_path=subband_json_path,
    )

    metrics = compute_all_metrics(
        poles_data=poles_data,
        snippets=snippets,
        judgements=judgements,
        baselines=baselines,
        structural_checks=structural_checks,
        primary_judge=primary_judge,
        n_resamples=n_resamples,
        seed=seed,
        weak_band_survival=weak_band_survival,
        unfiltered_judgements=unfiltered_judgements,
        unfiltered_snippets=unfiltered_snippets,
        examples_judgements=examples_judgements,
        p99_thresholds_payload=p99_thresholds_payload,
    )

    plot_band_accuracy(metrics, figure_path)
    main_fig = main_figure_path or (figure_path.parent / "main_band_accuracy.png")
    plot_main_band_accuracy(metrics, main_fig)
    if metrics.get("context_ablation"):
        ctx_fig = context_figure_path or (figure_path.parent / "context_ablation.png")
        plot_context_ablation(metrics["context_ablation"], ctx_fig)

    tables_md = render_summary_tables_md(metrics)
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    tables_path.write_text(tables_md, encoding="utf-8")

    worksheet_md = generate_audit_worksheet_md(
        poles_data=poles_data,
        snippets=snippets,
        labels=labels,
        judgements=judgements,
        baselines=baselines,
        primary_judge=metrics["primary_judge"],
        target_n=30,
    )
    worksheet_path.parent.mkdir(parents=True, exist_ok=True)
    worksheet_path.write_text(worksheet_md, encoding="utf-8")

    if subband_json_path is not None and metrics.get("p97_p99_subband_split"):
        subband_json_path.parent.mkdir(parents=True, exist_ok=True)
        subband_json_path.write_text(
            json.dumps(metrics["p97_p99_subband_split"], indent=2, default=str),
            encoding="utf-8",
        )

    if metrics_json_path is not None:
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_json_path.write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )

    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute neuron-clustered bootstrap CIs, binary accuracy slices, figure, tables, and 30-pole audit worksheet."
    )
    parser.add_argument("--poles", type=Path, default=Path("data/poles.json"))
    parser.add_argument("--snippets", type=Path, default=Path("data/snippets.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--judgements", type=Path, default=Path("results/judgements.jsonl"))
    parser.add_argument("--baselines", type=Path, default=Path("results/baselines.jsonl"))
    parser.add_argument(
        "--structural", type=Path, default=Path("results/structural_checks.json")
    )
    parser.add_argument(
        "--weak-band-survival",
        type=Path,
        default=Path("results/weak_band_survival.json"),
    )
    parser.add_argument(
        "--unfiltered-judgements",
        type=Path,
        default=Path("results/judgements_unfiltered_context_ablation.jsonl"),
    )
    parser.add_argument(
        "--unfiltered-snippets",
        type=Path,
        default=Path("data/snippets_unfiltered_context_ablation.jsonl"),
    )
    parser.add_argument(
        "--examples-judgements",
        type=Path,
        default=Path("results/judgements_examples_as_label_exploratory.jsonl"),
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/w_band_accuracy.png")
    )
    parser.add_argument(
        "--main-figure", type=Path, default=Path("figures/main_band_accuracy.png")
    )
    parser.add_argument(
        "--context-figure", type=Path, default=Path("figures/context_ablation.png")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache")
    )
    parser.add_argument(
        "--subband-json", type=Path, default=Path("results/p97_p99_subband_split.json")
    )
    parser.add_argument(
        "--tables", type=Path, default=Path("results/summary_tables.md")
    )
    parser.add_argument(
        "--worksheet", type=Path, default=Path("audit/raw_30_poles_worksheet.md")
    )
    parser.add_argument(
        "--metrics-json", type=Path, default=Path("results/metrics.json")
    )
    parser.add_argument("--primary-judge", type=str, default=None)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    metrics = run_analysis(
        poles_path=args.poles,
        snippets_path=args.snippets,
        labels_path=args.labels,
        judgements_path=args.judgements,
        baselines_path=args.baselines,
        structural_path=args.structural,
        figure_path=args.figure,
        tables_path=args.tables,
        worksheet_path=args.worksheet,
        metrics_json_path=args.metrics_json,
        primary_judge=args.primary_judge,
        n_resamples=args.bootstrap,
        seed=args.seed,
        weak_band_survival_path=args.weak_band_survival,
        unfiltered_judgements_path=args.unfiltered_judgements,
        unfiltered_snippets_path=args.unfiltered_snippets,
        examples_judgements_path=args.examples_judgements,
        context_figure_path=args.context_figure,
        main_figure_path=args.main_figure,
        cache_dir=args.cache_dir,
        subband_json_path=args.subband_json,
    )
    print(
        f"Analysis complete (primary={metrics['primary_judge']}): "
        f"wrote {args.main_figure}, {args.figure}, {args.tables}, and {args.worksheet}. "
        f"Pilot gate: {metrics['pilot_gate']['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
