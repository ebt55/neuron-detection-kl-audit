"""CPU diagnostics & exploratory unfiltered weak-band ablation.

STRICTLY READ-ONLY on all frozen files (`data/poles.json`,
`data/label_examples.jsonl`, `data/snippets.jsonl`, `data/labels.jsonl`,
`results/baselines.jsonl`). Produces TWO standalone output artifacts from `data/cache/`:

1. `results/weak_band_survival.json` (Limitations — Condition 2):
   Exact share of candidate weak-band positive eval tokens (`doc_id % 2 == 1`,
   `60 <= pos_in_chunk <= chunk_len - 9`, `d_pos in band`, `|a_pos| >= p25_abs`)
   that survive the `< p90` clean-window filter (`max_{j != pos} d_j < p90`),
   reported pooled AND per-pole (`mean`, `median`, `p10`, `p25`, `p75`, `p90`,
   `min`, `max`) across `all_100`, `heldout_80`, `pilot_20`, and per layer
   (`L4`, `L11`, `L18`, `L24`).

2. `data/snippets_unfiltered_context_ablation.jsonl` (Exploratory — Condition 5):
   Strictly separate 1,200-pair weak-band (`p75_90`, `p90_97`) evaluation set
   (`240 pilot_20` + `960 heldout_80`) under `dead_zone` distractors where
   positives are sampled WITHOUT the `other < p90` window filter (retaining
   interior 69-token windowing, odd eval docs, `s*a > 0`, `|a| >= p25_abs`,
   16-token suffix dedup vs. `data/label_examples.jsonl`, `<<token>>` marking at
   offset 60, Tier-0 `dead_zone` distractors on distinct odd docs, and exact 3/3
   A/B cell balance across 100% of 200 cells).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from w_audit.baselines import (
    score_rarity_pair,
    score_top_token_pair,
    score_window_ge_p97_pair,
)
from w_audit.config import (
    DEDUP_SUFFIX_LEN,
    MODEL_ID,
    PAIRS_PER_CELL,
    SEED,
    WindowSpec,
    ensure_dirs,
)
from w_audit.extract import QUAD_CODE_TO_NAME
from w_audit.snippets import (
    WEAK_BANDS,
    _sample_distinct_doc_indices,
    _select_strict_tier0_pairs,
    _stable_rng,
    _suffix_key,
    balanced_cell_pos_slots,
    compute_other_and_window_max_1d,
    render_marked_window,
)

UNFILTERED_SNIPPETS_FILENAME: str = "snippets_unfiltered_context_ablation.jsonl"
SURVIVAL_REPORT_FILENAME: str = "weak_band_survival.json"


def _dist_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Compute exact distribution summary (n, mean, median, p10, p25, p75, p90, min, max)."""
    if not values:
        return {
            "n_poles": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_poles": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.quantile(arr, 0.50)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _aggregate_survival_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pooled and per-pole survival stats over a cohort/layer of pole records."""
    tot_cand = int(sum(int(r["total_candidates"]) for r in records))
    tot_surv = int(sum(int(r["surviving_clean"]) for r in records))
    tot_ge90 = int(sum(int(r["other_ge_p90_count"]) for r in records))
    tot_ge97 = int(sum(int(r["other_ge_p97_count"]) for r in records))

    pooled_surv_rate = (tot_surv / tot_cand) if tot_cand > 0 else float("nan")
    pooled_ge90_rate = (tot_ge90 / tot_cand) if tot_cand > 0 else float("nan")
    pooled_ge97_rate = (tot_ge97 / tot_cand) if tot_cand > 0 else float("nan")

    surv_rates = [float(r["survival_rate"]) for r in records]
    ge90_rates = [float(r["other_ge_p90_rate"]) for r in records]
    ge97_rates = [float(r["other_ge_p97_rate"]) for r in records]

    return {
        "n_poles": len(records),
        "pooled_total_candidates": tot_cand,
        "pooled_surviving_clean": tot_surv,
        "pooled_survival_rate": pooled_surv_rate,
        "pooled_other_ge_p90_rate": pooled_ge90_rate,
        "pooled_other_ge_p97_rate": pooled_ge97_rate,
        "per_pole_survival_rate": _dist_summary(surv_rates),
        "per_pole_other_ge_p90_rate": _dist_summary(ge90_rates),
        "per_pole_other_ge_p97_rate": _dist_summary(ge97_rates),
    }


def _load_train_suffixes_from_label_examples(
    label_examples_path: Path,
    token_ids: np.ndarray,
    chunk_starts: np.ndarray,
    dedup_len: int,
) -> dict[str, set[tuple[int, ...]]]:
    """Reconstruct exact 16-token train_label suffixes per pole from data/label_examples.jsonl."""
    suffixes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    with label_examples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row["pole_id"])
            flat_idx = int(chunk_starts[int(row["chunk_id"])]) + int(row["pos_in_chunk"])
            suffixes[pid].add(_suffix_key(token_ids, flat_idx, dedup_len))
    return suffixes


def build_weak_band_survival_and_unfiltered_ablation(
    tokenizer: Any,
    base_dir: Path | None = None,
    seed: int = SEED,
    pairs_per_cell: int = PAIRS_PER_CELL,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Compute results/weak_band_survival.json and data/snippets_unfiltered_context_ablation.jsonl.

    Does NOT write or modify any frozen file (`poles.json`,
    `label_examples.jsonl`, `snippets.jsonl`, `labels.jsonl`, `baselines.jsonl`).
    """
    if pairs_per_cell % 2 != 0:
        raise ValueError(f"pairs_per_cell must be even for exact k/k A/B balance, got {pairs_per_cell}")

    t0 = time.time()
    dirs = ensure_dirs(base_dir)
    cache_dir = dirs["cache"]
    poles_path = dirs["data"] / "poles.json"
    label_examples_path = dirs["data"] / "label_examples.jsonl"
    unfiltered_path = dirs["data"] / UNFILTERED_SNIPPETS_FILENAME
    survival_path = dirs["results"] / SURVIVAL_REPORT_FILENAME

    poles_doc = json.loads(poles_path.read_text(encoding="utf-8"))
    cache_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    total_tokens = int(poles_doc["total_tokens"])

    if window_spec is not None:
        wspec = window_spec
    else:
        wcfg = poles_doc.get("window", {})
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
    chunk_starts = np.load(cache_dir / "chunk_starts.npy", mmap_mode="r")
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
    eval_interior = interior_mask & ((doc_ids % 2) == 1)

    dedup_len = min(DEDUP_SUFFIX_LEN, wspec.before + 1)
    train_suffixes_by_pole = _load_train_suffixes_from_label_examples(
        label_examples_path=label_examples_path,
        token_ids=token_ids,
        chunk_starts=chunk_starts,
        dedup_len=dedup_len,
    )

    def _build_snippet_obj(
        flat_idx: int,
        act_val: float,
        act_dir: float,
        quad_code: int,
        window_max_act_dir: float,
        other_max_act_dir: float,
        p90_dir: float,
        p97_dir: float,
        is_dist: bool = False,
    ) -> dict[str, Any]:
        marked_text, target_tok_str, extra_markers = render_marked_window(
            token_ids, flat_idx, tokenizer, wspec
        )
        tid = int(token_ids[flat_idx])
        obj: dict[str, Any] = {
            "doc_id": int(doc_ids[flat_idx]),
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
            "other_has_ge_p97": bool(other_max_act_dir >= p97_dir),
        }
        if is_dist:
            obj["neighbor_act_val"] = None
            obj["neighbor_gate_cosine"] = None
        obj["extra_marker_count"] = int(extra_markers)
        obj["marked_text"] = marked_text
        return obj

    survival_records_by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in WEAK_BANDS}
    unfiltered_pairs: list[dict[str, Any]] = []
    pilot_pole_ids: set[str] = set()
    top3_by_pole: dict[str, set[int]] = {}

    for pole in poles_doc["poles"]:
        pole_id = str(pole["pole_id"])
        layer = int(pole["layer"])
        neuron = int(pole["neuron"])
        sign = int(pole["sign"])
        in_pilot = bool(pole["in_pilot"])
        if in_pilot:
            pilot_pole_ids.add(pole_id)
        top3_by_pole[pole_id] = {int(x) for x in pole.get("top_3_token_ids", [])}

        p25_abs = float(pole["p25_abs"])
        edges = pole["band_edges_dir"]
        p75 = float(edges["p75"])
        p90 = float(edges["p90"])
        p97 = float(edges["p97"])

        col = neuron_to_col[(layer, neuron)]
        a_n = acts[:, col].astype(np.float32)
        q_n = quads[:, col]
        d_n = float(sign) * a_n
        abs_n = np.abs(a_n)

        other_max_dir, win_max_dir = compute_other_and_window_max_1d(d_n, wspec)

        eval_guard = eval_interior & (d_n > 0.0) & (abs_n >= p25_abs)
        in_dead_zone = eval_interior & (abs_n < p25_abs)
        dz_tier0_mask = in_dead_zone & (win_max_dir < p90)
        dz_used_docs: set[int] = set()

        band_bounds = {
            "p75_90": (p75, p90),
            "p90_97": (p90, p97),
        }
        for band in WEAK_BANDS:
            lo, hi = band_bounds[band]
            cand_mask = eval_guard & (d_n >= lo) & (d_n < hi)
            surv_mask = cand_mask & (other_max_dir < p90)
            ge90_mask = cand_mask & (other_max_dir >= p90)
            ge97_mask = cand_mask & (other_max_dir >= p97)

            n_cand = int(np.sum(cand_mask))
            n_surv = int(np.sum(surv_mask))
            n_ge90 = int(np.sum(ge90_mask))
            n_ge97 = int(np.sum(ge97_mask))
            surv_rate = (n_surv / n_cand) if n_cand > 0 else float("nan")
            ge90_rate = (n_ge90 / n_cand) if n_cand > 0 else float("nan")
            ge97_rate = (n_ge97 / n_cand) if n_cand > 0 else float("nan")

            survival_records_by_band[band].append(
                {
                    "pole_id": pole_id,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "in_pilot": in_pilot,
                    "total_candidates": n_cand,
                    "surviving_clean": n_surv,
                    "other_ge_p90_count": n_ge90,
                    "other_ge_p97_count": n_ge97,
                    "survival_rate": surv_rate,
                    "other_ge_p90_rate": ge90_rate,
                    "other_ge_p97_rate": ge97_rate,
                }
            )

            # Sample 6 UNFILTERED positives (NO `other_max_dir < p90` filter!)
            cands = np.flatnonzero(cand_mask)
            rng_pos = _stable_rng(seed, pole_id, "eval_pos_unfiltered_ablation", band)
            picked_pos = _sample_distinct_doc_indices(
                candidate_indices=cands,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=pairs_per_cell,
                rng=rng_pos,
                min_spacing=wspec.total,
                token_ids=token_ids,
                forbidden_suffixes=train_suffixes_by_pole.get(pole_id, set()),
                suffix_len=dedup_len,
            )
            even_pos = picked_pos[: 2 * (len(picked_pos) // 2)]

            rng_dz = _stable_rng(seed, pole_id, "dist_dead_zone_unfiltered_ablation", band)
            paired = _select_strict_tier0_pairs(
                tier0_mask=dz_tier0_mask,
                pos_indices=even_pos,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                rng=rng_dz,
                used_docs_across_bands=dz_used_docs,
                min_spacing=wspec.total,
            )
            slots = balanced_cell_pos_slots(
                len(paired), seed, pole_id, band, "dead_zone_unfiltered_ablation"
            )

            for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired, slots)):
                p_idx = even_pos[pos_k]
                pair_id = f"{pole_id}__{band}__dead_zone_unfiltered__{pair_idx}"
                unfiltered_pairs.append(
                    {
                        "pair_id": pair_id,
                        "pole_id": pole_id,
                        "layer": layer,
                        "neuron": neuron,
                        "sign": sign,
                        "band": band,
                        "distractor_type": "dead_zone",
                        "distractor_tier": 0,
                        "ablation_arm": "unfiltered_weak_context",
                        "pair_idx": pair_idx,
                        "pos_slot": slot,
                        "pos": _build_snippet_obj(
                            flat_idx=p_idx,
                            act_val=float(a_n[p_idx]),
                            act_dir=float(d_n[p_idx]),
                            quad_code=int(q_n[p_idx]),
                            window_max_act_dir=float(win_max_dir[p_idx]),
                            other_max_act_dir=float(other_max_dir[p_idx]),
                            p90_dir=p90,
                            p97_dir=p97,
                            is_dist=False,
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
                            is_dist=True,
                        ),
                    }
                )

    # Write strictly separate unfiltered snippet file
    with unfiltered_path.open("w", encoding="utf-8") as f:
        for row in unfiltered_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Audit the sampled unfiltered pairs (balance, exact windows, leakage rates, descriptive rules)
    cell_slots: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    same_doc_leaks = 0
    bos_leaks = 0
    exact_win = 0
    dist_ge90_leaks = 0
    for r in unfiltered_pairs:
        cell_slots[(r["pole_id"], r["band"])][r["pos_slot"]] += 1
        pos, dist = r["pos"], r["dist"]
        if pos["doc_id"] == dist["doc_id"] or (pos["doc_id"] % 2 != 1) or (dist["doc_id"] % 2 != 1):
            same_doc_leaks += 1
        if pos["pos_in_chunk"] < wspec.before or dist["pos_in_chunk"] < wspec.before:
            bos_leaks += 1
        if (
            pos["window_tokens"] == wspec.total
            and dist["window_tokens"] == wspec.total
            and pos["target_offset"] == wspec.target_offset
            and dist["target_offset"] == wspec.target_offset
        ):
            exact_win += 1
        if dist["window_max_act_dir"] >= float(
            next(p for p in poles_doc["poles"] if p["pole_id"] == r["pole_id"])["band_edges_dir"]["p90"]
        ):
            dist_ge90_leaks += 1

    imbalanced_cells = [
        k for k, cnt in cell_slots.items() if cnt["A"] != cnt["B"] or (cnt["A"] + cnt["B"]) == 0
    ]

    def _subset_pair_metrics(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        m = len(subset)
        if m == 0:
            return {"n_pairs": 0}
        ge90 = sum(1 for r in subset if r["pos"]["other_has_ge_p90"])
        ge97 = sum(1 for r in subset if r["pos"]["other_has_ge_p97"])
        w97_scores = []
        rar_scores = []
        top3_scores = []
        for r in subset:
            _, _, w_acc = score_window_ge_p97_pair(
                r["pos"]["window_has_ge_p97"], r["dist"]["window_has_ge_p97"]
            )
            _, _, r_acc = score_rarity_pair(
                r["pos"]["target_corpus_count"], r["dist"]["target_corpus_count"], total_tokens
            )
            _, _, t_acc = score_top_token_pair(
                r["pos"]["target_token_id"],
                r["dist"]["target_token_id"],
                top3_by_pole[r["pole_id"]],
            )
            w97_scores.append(w_acc)
            rar_scores.append(r_acc)
            top3_scores.append(t_acc)
        return {
            "n_pairs": m,
            "pos_other_ge_p90_count": ge90,
            "pos_other_ge_p90_rate": ge90 / m,
            "pos_other_ge_p97_count": ge97,
            "pos_other_ge_p97_rate": ge97 / m,
            "descriptive_rules": {
                "window_has_ge_p97": sum(w97_scores) / m,
                "rarity": sum(rar_scores) / m,
                "top_token": sum(top3_scores) / m,
            },
        }

    # Build structured weak_band_survival.json report
    survival_by_band_report: dict[str, Any] = {}
    layers_sorted = sorted({int(p["layer"]) for p in poles_doc["poles"]})
    for band in WEAK_BANDS:
        recs = survival_records_by_band[band]
        recs_heldout = [r for r in recs if not r["in_pilot"]]
        recs_pilot = [r for r in recs if r["in_pilot"]]
        by_layer = {
            str(L): _aggregate_survival_records([r for r in recs if r["layer"] == L])
            for L in layers_sorted
        }
        survival_by_band_report[band] = {
            "all_100": _aggregate_survival_records(recs),
            "heldout_80": _aggregate_survival_records(recs_heldout),
            "pilot_20": _aggregate_survival_records(recs_pilot),
            "by_layer": by_layer,
            "per_pole": recs,
        }

    n_pairs = len(unfiltered_pairs)
    pilot_pairs = [r for r in unfiltered_pairs if r["pole_id"] in pilot_pole_ids]
    heldout_pairs = [r for r in unfiltered_pairs if r["pole_id"] not in pilot_pole_ids]
    slot_a = sum(1 for r in unfiltered_pairs if r["pos_slot"] == "A")

    ablation_summary = {
        "status": "EXPLORATORY",
        "snippets_path": str(unfiltered_path.relative_to(dirs["root"])).replace("\\", "/"),
        "total_pairs": n_pairs,
        "pilot_20_pairs": len(pilot_pairs),
        "heldout_80_pairs": len(heldout_pairs),
        "total_cells": len(cell_slots),
        "imbalanced_cells_count": len(imbalanced_cells),
        "exact_50_50_cell_balance_rate": (
            (len(cell_slots) - len(imbalanced_cells)) / max(len(cell_slots), 1)
        ),
        "slot_A_count": slot_a,
        "slot_B_count": n_pairs - slot_a,
        "slot_A_rate": slot_a / max(n_pairs, 1),
        "window_length_exact_rate": exact_win / max(n_pairs, 1),
        "bos_pos0_leak_count": bos_leaks,
        "same_doc_or_split_leak_count": same_doc_leaks,
        "dist_window_ge_p90_count": dist_ge90_leaks,
        "dist_window_ge_p90_rate": dist_ge90_leaks / max(n_pairs, 1),
        "all_invariants_passed": (
            len(imbalanced_cells) == 0
            and exact_win == n_pairs
            and bos_leaks == 0
            and same_doc_leaks == 0
            and dist_ge90_leaks == 0
        ),
        "overall": _subset_pair_metrics(unfiltered_pairs),
        "by_band": {
            b: _subset_pair_metrics([r for r in unfiltered_pairs if r["band"] == b])
            for b in WEAK_BANDS
        },
        "by_cohort_band": {
            f"{cname}:{b}": _subset_pair_metrics(
                [r for r in c_pairs if r["band"] == b]
            )
            for cname, c_pairs in (("heldout_80", heldout_pairs), ("pilot_20", pilot_pairs))
            for b in WEAK_BANDS
        },
    }

    report = {
        "schema_version": "1.0",
        "model_id": poles_doc.get("model_id", MODEL_ID),
        "seed": seed,
        "window": wspec.to_dict(),
        "weak_band_survival": survival_by_band_report,
        "unfiltered_ablation_summary": ablation_summary,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    survival_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not ablation_summary["all_invariants_passed"]:
        raise AssertionError(f"Unfiltered ablation invariants failed: {ablation_summary}")

    print(f"[context_ablation] Wrote {survival_path} and {unfiltered_path} ({n_pairs} pairs) in {report['elapsed_sec']}s")
    for b in WEAK_BANDS:
        s_all = survival_by_band_report[b]["all_100"]
        s_ho = survival_by_band_report[b]["heldout_80"]
        s_pi = survival_by_band_report[b]["pilot_20"]
        print(
            f"  survival {b:6s} | pooled={s_all['pooled_survival_rate']:.4%} "
            f"(mean={s_all['per_pole_survival_rate']['mean']:.4%}, "
            f"med={s_all['per_pole_survival_rate']['median']:.4%}, "
            f"p10={s_all['per_pole_survival_rate']['p10']:.4%}, "
            f"p90={s_all['per_pole_survival_rate']['p90']:.4%}) | "
            f"heldout_80={s_ho['pooled_survival_rate']:.4%} | pilot_20={s_pi['pooled_survival_rate']:.4%}"
        )
    for b in WEAK_BANDS:
        m = ablation_summary["by_band"][b]
        print(
            f"  unfiltered {b:6s} (n={m['n_pairs']}): "
            f"pos_other>=p90={m['pos_other_ge_p90_rate']:.3%}, "
            f"pos_other>=p97={m['pos_other_ge_p97_rate']:.3%}, "
            f"win>=p97_rule={m['descriptive_rules']['window_has_ge_p97']:.3f}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute weak-band clean-window survival report and unfiltered context ablation snippets."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pairs-per-cell", type=int, default=PAIRS_PER_CELL)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    build_weak_band_survival_and_unfiltered_ablation(
        tokenizer=tokenizer,
        seed=args.seed,
        pairs_per_cell=args.pairs_per_cell,
    )


if __name__ == "__main__":
    main()
