"""Explanation-free no-LLM rule baselines and structural sanity checks.

Checks window, balance, and Tier-0 distractor invariants:
1. Corpus Rarity Rule (`rarity`): picks snippet whose marked target token has lower corpus
   frequency across the ~2M-token stream: s(tok) = -log2(count(tok) / N).
2. Top-Token Rule (`top_token`, plus `top_10_token` diagnostic): picks snippet whose marked
   target token belongs to the pole's train_label top-3 (or top-10) trigger token list.
3. Window Peak Rule (`window_has_ge_p97`): picks snippet whose 69-token window
   contains any token with directional activation >= P_97(s * a) for that pole; tie-breaks
   at 0.500 and explicitly records per-cell tie rates.
4. Structural Cue Sanity Checks (`results/structural_checks.json`):
   - 100% exact 69-token windows, 100% target_offset=60, 0% truncation, 0% <bos>/pos-0 leaks,
     0% same-doc leaks, exact 0.500 marker-position & token-length rules.
   - Weak-band guard: verifies other_token_ge_p90_rate == 0.0 on ALL surviving `p75_90`/`p90_97`
     positives and ALL `dead_zone`/`goodfire_lt_p90`/`nearest_wgate_tier0` distractors,
     plus exact surviving pair/pole/neuron counts per band and distractor type.
   - Slot balance: verifies exact 50/50 (k/k) Slot A/B balance in 100% of surviving cells.
   - Same-layer neighbour arm: verifies strictly Tier-0 ONLY `nearest_wgate_tier0` (0 fallback tiers) and
     logs exact `w_gate` nearest-neighbour cosine quantiles across all 50 sampled neurons
     AND across surviving Tier-0 pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from w_audit.config import (
    PAIRS_PER_CELL,
    TARGET_OFFSET,
    WINDOW_TOTAL,
    ensure_dirs,
)

WEAK_BANDS: tuple[str, ...] = ("p75_90", "p90_97")
NEIGHBOUR_DISTRACTOR_TYPE: str = "nearest_wgate_tier0"


def _quantile_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Compute exact min/p25/median/p75/p90/max summary for a sequence of floats."""
    if not values:
        return {
            "n": 0,
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def score_rarity_pair(
    pos_count: int,
    dist_count: int,
    total_tokens: int,
) -> tuple[float, bool, float]:
    """Score one pair under Corpus Rarity Rule: s(tok) = -log2(count(tok) / N)."""
    n = max(int(total_tokens), 1)
    c_pos = max(int(pos_count), 1)
    c_dist = max(int(dist_count), 1)
    s_pos = -math.log2(c_pos / n)
    s_dist = -math.log2(c_dist / n)
    score_diff = s_pos - s_dist
    is_tie = c_pos == c_dist
    if is_tie or abs(score_diff) == 0.0:
        return 0.0, True, 0.5
    return float(score_diff), False, (1.0 if score_diff > 0.0 else 0.0)


def score_top_token_pair(
    pos_token_id: int,
    dist_token_id: int,
    top_token_ids: Sequence[int] | set[int],
) -> tuple[float, bool, float]:
    """Score one pair under Top-K Trigger Token Rule."""
    top_set = top_token_ids if isinstance(top_token_ids, set) else set(top_token_ids)
    h_pos = 1.0 if int(pos_token_id) in top_set else 0.0
    h_dist = 1.0 if int(dist_token_id) in top_set else 0.0
    score_diff = h_pos - h_dist
    is_tie = h_pos == h_dist
    if is_tie:
        return 0.0, True, 0.5
    return float(score_diff), False, (1.0 if score_diff > 0.0 else 0.0)


def score_window_ge_p97_pair(
    pos_window_has_ge_p97: bool,
    dist_window_has_ge_p97: bool,
) -> tuple[float, bool, float]:
    """Score one pair under Window Peak Rule (`window_has_ge_p97`).

    Picks the snippet whose 69-token window contains any token with directional
    activation >= P_97(s * a) for that pole; tie -> 0.5 (is_tie=True).
    """
    h_pos = 1.0 if bool(pos_window_has_ge_p97) else 0.0
    h_dist = 1.0 if bool(dist_window_has_ge_p97) else 0.0
    score_diff = h_pos - h_dist
    is_tie = h_pos == h_dist
    if is_tie:
        return 0.0, True, 0.5
    return float(score_diff), False, (1.0 if score_diff > 0.0 else 0.0)


def run_structural_checks(
    pairs: Sequence[dict[str, Any]],
    poles_doc: dict[str, Any] | None = None,
    expected_window_tokens: int = WINDOW_TOTAL,
    expected_target_offset: int = TARGET_OFFSET,
    target_pairs_per_cell: int = PAIRS_PER_CELL,
) -> dict[str, Any]:
    """Verify all structural invariants across all snippet pairs."""
    n = len(pairs)
    if n == 0:
        raise ValueError("Cannot run structural checks on empty snippet pair list.")

    pole_by_id: dict[str, dict[str, Any]] = {}
    if poles_doc is not None:
        pole_by_id = {str(p["pole_id"]): p for p in poles_doc.get("poles", [])}

    exact_win_count = 0
    exact_offset_count = 0
    trunc_pos_count = 0
    trunc_dist_count = 0
    bos_leak_count = 0
    same_doc_leak_count = 0
    marker_pos_scores: list[float] = []
    token_len_scores: list[float] = []
    char_len_scores: list[float] = []
    char_len_by_cell: dict[str, list[float]] = defaultdict(list)

    pos_extra_total = 0
    dist_extra_total = 0
    pos_extra_nonzero = 0
    dist_extra_nonzero = 0
    extra_marker_scores: list[float] = []

    slot_a_count = 0
    slot_b_count = 0

    # Weak-band guard: Weak-band positive other-token < p90 & distractor whole-window < p90 checks
    weak_pos_total = 0
    weak_pos_other_ge_p90_count = 0
    dist_window_ge_p90_count = 0

    # Slot balance: Exact k/k A/B cell balance tracking
    cell_slot_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"A": 0, "B": 0}
    )

    # Same-layer neighbour: Tier-0 neighbour verification & cosine distribution
    nb_non_tier0_count = 0
    nb_pair_cosines: list[float] = []
    nb_neuron_cosines: dict[tuple[int, int], float] = {}
    nb_pairs_by_band: dict[str, int] = defaultdict(int)
    nb_poles_by_band: dict[str, set[str]] = defaultdict(set)
    nb_all_poles: set[str] = set()
    nb_all_neurons: set[tuple[int, int]] = set()

    # Exact surviving pair/pole counts per (distractor_type, band)
    pairs_per_pole_cell: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    neurons_per_cell: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)

    for row in pairs:
        pole_id = str(row["pole_id"])
        band = str(row["band"])
        dist_type = str(row["distractor_type"])
        pos = row["pos"]
        dist = row["dist"]

        p_win = int(pos["window_tokens"])
        d_win = int(dist["window_tokens"])
        p_off = int(pos["target_offset"])
        d_off = int(dist["target_offset"])

        if p_win == expected_window_tokens and d_win == expected_window_tokens:
            exact_win_count += 1
        if p_win != expected_window_tokens:
            trunc_pos_count += 1
        if d_win != expected_window_tokens:
            trunc_dist_count += 1
        if p_off == expected_target_offset and d_off == expected_target_offset:
            exact_offset_count += 1
        if int(pos["pos_in_chunk"]) < expected_target_offset or int(dist["pos_in_chunk"]) < expected_target_offset:
            bos_leak_count += 1
        if int(pos["doc_id"]) == int(dist["doc_id"]):
            same_doc_leak_count += 1

        marker_pos_scores.append(1.0 if p_off > d_off else (0.0 if p_off < d_off else 0.5))
        token_len_scores.append(1.0 if p_win > d_win else (0.0 if p_win < d_win else 0.5))

        p_chars = len(pos["marked_text"])
        d_chars = len(dist["marked_text"])
        c_score = 1.0 if p_chars > d_chars else (0.0 if p_chars < d_chars else 0.5)
        char_len_scores.append(c_score)
        cell_key = f"{dist_type}:{band}"
        char_len_by_cell[cell_key].append(c_score)

        p_ex = int(pos.get("extra_marker_count", 0))
        d_ex = int(dist.get("extra_marker_count", 0))
        pos_extra_total += p_ex
        dist_extra_total += d_ex
        if p_ex > 0:
            pos_extra_nonzero += 1
        if d_ex > 0:
            dist_extra_nonzero += 1
        extra_marker_scores.append(1.0 if p_ex > d_ex else (0.0 if p_ex < d_ex else 0.5))

        slot = str(row["pos_slot"])
        if slot == "A":
            slot_a_count += 1
            cell_slot_counts[(pole_id, band, dist_type)]["A"] += 1
        elif slot == "B":
            slot_b_count += 1
            cell_slot_counts[(pole_id, band, dist_type)]["B"] += 1

        # Weak-band guard checks using stored flags (and raw window_max_act_dir vs p90 if poles_doc provided)
        pole_info = pole_by_id.get(pole_id)
        p90_edge = float(pole_info["band_edges_dir"]["p90"]) if pole_info is not None else None

        if band in WEAK_BANDS:
            weak_pos_total += 1
            if "other_has_ge_p90" in pos:
                is_leak = bool(pos["other_has_ge_p90"])
            elif p90_edge is not None and pos.get("other_max_act_dir") is not None:
                is_leak = float(pos["other_max_act_dir"]) >= p90_edge
            else:
                is_leak = False
            if is_leak:
                weak_pos_other_ge_p90_count += 1

        if p90_edge is not None and dist.get("window_max_act_dir") is not None:
            if float(dist["window_max_act_dir"]) >= p90_edge:
                dist_window_ge_p90_count += 1
        elif dist.get("other_has_ge_p90", False):
            dist_window_ge_p90_count += 1

        # Survival tracking
        pairs_per_pole_cell[(dist_type, band)][pole_id] += 1
        if "layer" in row and "neuron" in row:
            nkey = (int(row["layer"]), int(row["neuron"]))
            neurons_per_cell[(dist_type, band)].add(nkey)
        elif pole_info is not None:
            nkey = (int(pole_info["layer"]), int(pole_info["neuron"]))
            neurons_per_cell[(dist_type, band)].add(nkey)
        else:
            nkey = (-1, -1)

        # Tier-0 neighbour tracking
        if dist_type == NEIGHBOUR_DISTRACTOR_TYPE:
            if int(row.get("distractor_tier", 0)) != 0:
                nb_non_tier0_count += 1
            nb_pairs_by_band[band] += 1
            nb_poles_by_band[band].add(pole_id)
            nb_all_poles.add(pole_id)
            if nkey != (-1, -1):
                nb_all_neurons.add(nkey)
            nb_cos = dist.get("neighbor_gate_cosine")
            if nb_cos is None and pole_info is not None:
                nb_cos = pole_info.get("neighbor_gate_cosine")
            if nb_cos is not None:
                nb_pair_cosines.append(float(nb_cos))
                if nkey != (-1, -1):
                    nb_neuron_cosines[nkey] = float(nb_cos)

    # Slot balance: Verify exact k/k A/B balance in every surviving cell
    total_cells = len(cell_slot_counts)
    imbalanced_cells = [
        {"pole_id": k[0], "band": k[1], "distractor_type": k[2], **v}
        for k, v in cell_slot_counts.items()
        if v["A"] != v["B"] or (v["A"] + v["B"]) == 0
    ]
    exact_balance_rate = (total_cells - len(imbalanced_cells)) / max(total_cells, 1)

    # Build exact surviving pair/pole/neuron report
    n_total_poles = len(pole_by_id) if pole_by_id else len({str(r["pole_id"]) for r in pairs})
    surviving_counts: dict[str, dict[str, Any]] = {}
    for (dist_type, band) in sorted(pairs_per_pole_cell.keys()):
        pmap = pairs_per_pole_cell[(dist_type, band)]
        counts_list = list(pmap.values())
        surv_poles = len(counts_list)
        surviving_counts[f"{dist_type}:{band}"] = {
            "surviving_pairs": int(sum(counts_list)),
            "target_max_pairs": int(n_total_poles * target_pairs_per_cell),
            "surviving_poles": int(surv_poles),
            "total_poles": int(n_total_poles),
            "surviving_neurons": int(len(neurons_per_cell[(dist_type, band)])),
            "poles_with_full_6": int(sum(c == target_pairs_per_cell for c in counts_list)),
            "poles_with_4": int(sum(c == 4 for c in counts_list)),
            "poles_with_2": int(sum(c == 2 for c in counts_list)),
            "poles_with_0": int(max(n_total_poles - surv_poles, 0)),
        }

    # Same-layer neighbour: Sampled 50 neurons' w_gate cosine distribution + surviving Tier-0 distribution
    sampled_neuron_cosines: dict[tuple[int, int], float] = {}
    if poles_doc is not None:
        for p in poles_doc.get("poles", []):
            nk = (int(p["layer"]), int(p["neuron"]))
            if nk not in sampled_neuron_cosines and "neighbor_gate_cosine" in p:
                sampled_neuron_cosines[nk] = float(p["neighbor_gate_cosine"])

    weak_pos_other_ge_p90_rate = (
        weak_pos_other_ge_p90_count / weak_pos_total if weak_pos_total > 0 else 0.0
    )
    dist_window_ge_p90_rate = dist_window_ge_p90_count / n

    hard_passed = (
        exact_win_count == n
        and exact_offset_count == n
        and trunc_pos_count == 0
        and trunc_dist_count == 0
        and bos_leak_count == 0
        and same_doc_leak_count == 0
        and weak_pos_other_ge_p90_count == 0
        and dist_window_ge_p90_count == 0
        and len(imbalanced_cells) == 0
        and nb_non_tier0_count == 0
    )

    return {
        "total_pairs": n,
        "expected_window_tokens": expected_window_tokens,
        "expected_target_offset": expected_target_offset,
        "window_length_exact_rate": exact_win_count / n,
        "target_offset_exact_rate": exact_offset_count / n,
        "truncation_rate_pos": trunc_pos_count / n,
        "truncation_rate_dist": trunc_dist_count / n,
        "bos_pos0_leak_count": bos_leak_count,
        "same_doc_leak_count": same_doc_leak_count,
        "weak_pos_total": weak_pos_total,
        "weak_pos_other_token_ge_p90_count": weak_pos_other_ge_p90_count,
        "other_token_ge_p90_rate": weak_pos_other_ge_p90_rate,
        "dist_window_ge_p90_count": dist_window_ge_p90_count,
        "dist_window_ge_p90_rate": dist_window_ge_p90_rate,
        "total_cells": total_cells,
        "imbalanced_cells_count": len(imbalanced_cells),
        "exact_50_50_cell_balance_rate": exact_balance_rate,
        "nearest_wgate_tier0_non_tier0_count": nb_non_tier0_count,
        "marker_position_rule_accuracy": sum(marker_pos_scores) / n,
        "token_length_rule_accuracy": sum(token_len_scores) / n,
        "char_length_rule_accuracy": sum(char_len_scores) / n,
        "char_length_rule_by_cell": {
            k: sum(v) / len(v) for k, v in sorted(char_len_by_cell.items())
        },
        "extra_marker_balance": {
            "pos_mean_extra_markers": pos_extra_total / n,
            "dist_mean_extra_markers": dist_extra_total / n,
            "pos_nonzero_rate": pos_extra_nonzero / n,
            "dist_nonzero_rate": dist_extra_nonzero / n,
            "extra_marker_rule_accuracy": sum(extra_marker_scores) / n,
        },
        "pos_slot_balance": {
            "slot_A_count": slot_a_count,
            "slot_B_count": slot_b_count,
            "slot_A_rate": slot_a_count / n,
        },
        "surviving_pair_counts": surviving_counts,
        "nearest_wgate_tier0_summary": {
            "surviving_pairs_total": int(sum(nb_pairs_by_band.values())),
            "surviving_pairs_by_band": dict(sorted(nb_pairs_by_band.items())),
            "surviving_poles": int(len(nb_all_poles)),
            "surviving_poles_by_band": {k: len(v) for k, v in sorted(nb_poles_by_band.items())},
            "surviving_neurons": int(len(nb_all_neurons)),
            "sampled_50_neurons_wgate_cosine_stats": _quantile_summary(
                list(sampled_neuron_cosines.values())
            ),
            "surviving_tier0_pairs_wgate_cosine_stats": _quantile_summary(nb_pair_cosines),
            "surviving_tier0_neurons_wgate_cosine_stats": _quantile_summary(
                list(nb_neuron_cosines.values())
            ),
        },
        "all_hard_invariants_passed": hard_passed,
    }


def run_baselines(
    base_dir: Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Compute rarity, top_token, window_has_ge_p97, and top_10_token rules + structural checks."""
    dirs = ensure_dirs(base_dir)
    poles_path = dirs["data"] / "poles.json"
    snippets_path = dirs["data"] / "snippets.jsonl"
    baselines_path = dirs["results"] / "baselines.jsonl"
    structural_path = dirs["results"] / "structural_checks.json"

    poles_doc = json.loads(poles_path.read_text(encoding="utf-8"))
    total_tokens = int(poles_doc["total_tokens"])
    wcfg = poles_doc.get("window", {})
    expected_win = int(wcfg.get("total", WINDOW_TOTAL))
    expected_off = int(wcfg.get("before", TARGET_OFFSET))

    top3_by_pole: dict[str, set[int]] = {}
    top10_by_pole: dict[str, set[int]] = {}
    p97_by_pole: dict[str, float] = {}
    for pole in poles_doc["poles"]:
        pid = str(pole["pole_id"])
        top3_by_pole[pid] = {int(x) for x in pole.get("top_3_token_ids", [])}
        top10_by_pole[pid] = {int(x) for x in pole.get("top_10_token_ids", [])}
        p97_by_pole[pid] = float(pole["band_edges_dir"]["p97"])

    pairs: list[dict[str, Any]] = []
    with snippets_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    # 1. Structural Cue Sanity Checks (length, offset, context, and cell balance)
    structural = run_structural_checks(
        pairs=pairs,
        poles_doc=poles_doc,
        expected_window_tokens=expected_win,
        expected_target_offset=expected_off,
    )
    if strict and not structural["all_hard_invariants_passed"]:
        raise AssertionError(
            f"Hard structural invariant violated: {json.dumps(structural, indent=2)}"
        )

    # 2. Evaluate rarity, top_token, window_has_ge_p97, and top_10_token rules across all pairs
    out_rows: list[dict[str, Any]] = []
    acc_by_cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tie_by_cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in pairs:
        pair_id = str(row["pair_id"])
        pole_id = str(row["pole_id"])
        band = str(row["band"])
        dist_type = str(row["distractor_type"])
        pos = row["pos"]
        dist = row["dist"]
        p97_edge = p97_by_pole[pole_id]

        # Rule 1: Corpus Rarity
        r_diff, r_tie, r_corr = score_rarity_pair(
            pos_count=int(pos["target_corpus_count"]),
            dist_count=int(dist["target_corpus_count"]),
            total_tokens=total_tokens,
        )
        # Rule 2: Top-3 Trigger Token
        t3_diff, t3_tie, t3_corr = score_top_token_pair(
            pos_token_id=int(pos["target_token_id"]),
            dist_token_id=int(dist["target_token_id"]),
            top_token_ids=top3_by_pole[pole_id],
        )
        # Rule 3: Window contains >= p97 directional activation token
        pos_has_p97 = (
            bool(pos["window_has_ge_p97"])
            if "window_has_ge_p97" in pos
            else (float(pos["window_max_act_dir"]) >= p97_edge)
        )
        dist_has_p97 = (
            bool(dist["window_has_ge_p97"])
            if "window_has_ge_p97" in dist
            else (float(dist["window_max_act_dir"]) >= p97_edge)
        )
        w97_diff, w97_tie, w97_corr = score_window_ge_p97_pair(pos_has_p97, dist_has_p97)

        # Diagnostic Rule 4: Top-10 Trigger Token
        t10_diff, t10_tie, t10_corr = score_top_token_pair(
            pos_token_id=int(pos["target_token_id"]),
            dist_token_id=int(dist["target_token_id"]),
            top_token_ids=top10_by_pole[pole_id],
        )

        for rule_name, s_diff, is_tie, corr in (
            ("rarity", r_diff, r_tie, r_corr),
            ("top_token", t3_diff, t3_tie, t3_corr),
            ("window_has_ge_p97", w97_diff, w97_tie, w97_corr),
            ("top_10_token", t10_diff, t10_tie, t10_corr),
        ):
            out_rows.append(
                {
                    "pair_id": pair_id,
                    "pole_id": pole_id,
                    "band": band,
                    "distractor_type": dist_type,
                    "rule": rule_name,
                    "score_diff": float(s_diff),
                    "is_tie": bool(is_tie),
                    "correct": float(corr),
                }
            )
            ck = f"{dist_type}:{band}"
            acc_by_cell[ck][rule_name].append(float(corr))
            tie_by_cell[ck][rule_name].append(1.0 if is_tie else 0.0)

    with baselines_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rule_summary: dict[str, dict[str, float]] = {}
    rule_tie_summary: dict[str, dict[str, float]] = {}
    for cell_key in sorted(acc_by_cell.keys()):
        rule_summary[cell_key] = {
            rname: sum(vals) / len(vals) for rname, vals in sorted(acc_by_cell[cell_key].items())
        }
        rule_tie_summary[cell_key] = {
            rname: sum(vals) / len(vals) for rname, vals in sorted(tie_by_cell[cell_key].items())
        }
    structural["rule_baseline_summary"] = rule_summary
    structural["rule_tie_rate_summary"] = rule_tie_summary
    structural_path.write_text(json.dumps(structural, indent=2), encoding="utf-8")

    print(f"[baselines] Wrote {len(out_rows):,} rows -> {baselines_path}")
    print(
        f"[baselines] Hard invariants passed: {structural['all_hard_invariants_passed']} | "
        f"exact_50_50_cell_balance={structural['exact_50_50_cell_balance_rate']:.3f} | "
        f"weak_other_ge_p90={structural['other_token_ge_p90_rate']:.4f} | "
        f"dist_ge_p90={structural['dist_window_ge_p90_rate']:.4f} | "
        f"slot_A_rate={structural['pos_slot_balance']['slot_A_rate']:.3f}"
    )
    for cell_key, rmap in rule_summary.items():
        tmap = rule_tie_summary[cell_key]
        print(
            f"  {cell_key:34s} | rarity={rmap['rarity']:.3f} | "
            f"top3={rmap['top_token']:.3f} | "
            f"win>=p97={rmap['window_has_ge_p97']:.3f} (tie={tmap['window_has_ge_p97']:.2f})"
        )

    return structural


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute explanation-free rule baselines and structural sanity checks."
    )
    parser.add_argument("--no-strict", action="store_true", help="Do not raise on invariant mismatch.")
    args = parser.parse_args(argv)
    run_baselines(strict=not args.no_strict)


if __name__ == "__main__":
    main()
