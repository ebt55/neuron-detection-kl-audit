"""Pole sparsity diagnostics (Item 1) & percentile-free top100 band builder (Item 2).

Strictly CPU-only, reading from `data/qwen4b/cache/` and `data/qwen4b/active_poles.json`
without modifying any Qwen3-1.7B file or `data/qwen4b/snippets_4b.jsonl`:

1. `results/qwen4b/pole_sparsity_diagnostics.json`:
   - Computes exact `max_act_dir`, `p99.9_thresh`, `p99.9_frac_of_max`, firing rates
     (`frac_pos`, `frac_ge_p25_abs`, `frac_ge_20pct_max`, `n_tokens_ge_20pct_max`),
     and sampled `p99.9_plus` vs. `top100` positive activation distributions for all 424
     active poles (including `info`, `PCR`, `insurance`), exporting the complete sorted
     list of all poles with `p99.9_thresh < 0.20 * max_act_dir`.
2. `data/qwen4b/snippets_4b_top100.jsonl` & `results/qwen4b/structural_checks_4b_top100.json`:
   - For each of the 424 active poles, collects its 100 strongest eligible interior tokens
     (`60 <= pos <= len-9`, `d > 0`, `|a| >= p25_abs`) strictly across 100 distinct
     documents (single strongest token per doc, descending by `d`, with 16-token suffix
     deduplication), samples 20 positives (seed=42), pairs with 20 strict Tier-0
     `dead_zone` distractors (`|a| < p25_abs` & whole-window `< p90`, distinct docs),
     and enforces exact nested A/B balance (`10/10` overall, `5/5` per split half, `3/3`
     in pairs `0..5`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qwen4b_eval import (
    FINEWEB_MIN_ROW_IDX,
    MODEL_ID_4B,
    PAIRS_FLOOR6_PER_POLE,
    PAIRS_P999_PER_POLE,
    SEED_4B,
    ensure_4b_dirs,
)
from qwen4b_eval.snippets_4b import _stable_rng_4b, nested_balanced_p999_slots
from w_audit.config import DEDUP_SUFFIX_LEN, WindowSpec
from w_audit.extract import QUAD_CODE_TO_NAME
from w_audit.snippets import (
    _select_strict_tier0_pairs,
    _suffix_key,
    compute_other_and_window_max_1d,
    render_marked_window,
)

TOP_POOL_SIZE_4B: int = 100
SPARSE_MAX_FRAC_THRESHOLD: float = 0.20


def select_top_k_distinct_doc_indices(
    d_n: np.ndarray,
    eligible_mask: np.ndarray,
    doc_ids: np.ndarray,
    token_ids: np.ndarray,
    top_k: int = TOP_POOL_SIZE_4B,
    suffix_len: int = DEDUP_SUFFIX_LEN,
) -> dict[str, Any]:
    """Select the `top_k` strongest eligible tokens strictly across `top_k` distinct documents.

    For each document, considers ONLY its single strongest eligible token (`argmax d_n`,
    tie-break lowest flat index), ranks documents descending by `d_n`, and walks down
    distinct documents skipping any 16-token suffix duplicates until `top_k` tokens are
    collected.
    """
    elig_idx = np.flatnonzero(eligible_mask)
    if elig_idx.size == 0:
        raise RuntimeError("No eligible interior tokens found for top-k selection.")

    elig_d = d_n[elig_idx].astype(np.float32, copy=False)
    # Deterministic descending sort by activation, tie-break ascending by flat_idx
    order = np.lexsort((elig_idx, -elig_d))
    sorted_idx = elig_idx[order]

    seen_docs: set[int] = set()
    used_suffixes: set[tuple[int, ...]] = set()
    pool_indices: list[int] = []
    pool_doc_ranks: list[int] = []
    doc_rank = 0
    suffix_dups_skipped = 0

    for raw_i in sorted_idx.tolist():
        flat_idx = int(raw_i)
        did = int(doc_ids[flat_idx])
        if did in seen_docs:
            continue
        seen_docs.add(did)
        doc_rank += 1

        sk = _suffix_key(token_ids, flat_idx, suffix_len)
        if sk in used_suffixes:
            suffix_dups_skipped += 1
            continue
        used_suffixes.add(sk)
        pool_indices.append(flat_idx)
        pool_doc_ranks.append(doc_rank)
        if len(pool_indices) >= top_k:
            break

    if len(pool_indices) < top_k:
        raise RuntimeError(
            f"Only found {len(pool_indices)}/{top_k} distinct-doc unique-suffix eligible tokens "
            f"(scanned {doc_rank} distinct docs, skipped {suffix_dups_skipped} suffix dups)."
        )

    pool_acts = np.array([float(d_n[i]) for i in pool_indices], dtype=np.float64)
    return {
        "indices": pool_indices,
        "doc_ranks": pool_doc_ranks,
        "pool_size": len(pool_indices),
        "max_doc_rank_scanned": doc_rank,
        "suffix_dups_skipped": suffix_dups_skipped,
        "max_act_dir": float(np.max(pool_acts)),
        "min_act_dir": float(np.min(pool_acts)),
        "median_act_dir": float(np.median(pool_acts)),
        "mean_act_dir": float(np.mean(pool_acts)),
    }


def verify_top100_snippets_4b(
    pair_rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    pool_meta_by_pole: Mapping[str, Mapping[str, Any]],
    pairs_per_pole: int = PAIRS_P999_PER_POLE,
    floor_n: int = PAIRS_FLOOR6_PER_POLE,
    min_allowed_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
    wspec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Verify every structural, eligibility, doc-disjointness, and nested A/B invariant."""
    ws = wspec if wspec is not None else WindowSpec()
    poles_by_id = {str(p["pole_id"]): p for p in catalog["poles"]}

    total_pairs = len(pair_rows)
    exact_win = 0
    exact_off = 0
    bos_leaks = 0
    same_doc_leaks = 0
    ulto85_doc_overlap = 0
    pos_ineligible = 0
    dist_abs_ge_p25 = 0
    dist_win_ge_p90 = 0
    min_doc_id_seen: int | None = None
    max_doc_id_seen: int | None = None

    by_pole: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    slot_counts = Counter()

    for row in pair_rows:
        pid = str(row["pole_id"])
        band = str(row["band"])
        if band != "top100":
            raise ValueError(f"Expected band='top100', got {band!r}")
        slot = str(row["pos_slot"])
        slot_counts[slot] += 1

        pole = poles_by_id[pid]
        p25_abs = float(pole["p25_abs"])
        p90 = float(pole["band_edges_dir"]["p90"])
        pos = row["pos"]
        dist = row["dist"]
        by_pole[pid].append(row)

        if int(pos["window_tokens"]) == ws.total and int(dist["window_tokens"]) == ws.total:
            exact_win += 1
        if (
            int(pos["target_offset"]) == ws.target_offset
            and int(dist["target_offset"]) == ws.target_offset
        ):
            exact_off += 1
        if int(pos["pos_in_chunk"]) <= 0 or int(dist["pos_in_chunk"]) <= 0:
            bos_leaks += 1

        p_doc = int(pos["doc_id"])
        d_doc = int(dist["doc_id"])
        if p_doc == d_doc:
            same_doc_leaks += 1
        for d in (p_doc, d_doc):
            if min_doc_id_seen is None or d < min_doc_id_seen:
                min_doc_id_seen = d
            if max_doc_id_seen is None or d > max_doc_id_seen:
                max_doc_id_seen = d
            if min_allowed_doc_id is not None and d < min_allowed_doc_id:
                ulto85_doc_overlap += 1

        d_pos = float(pos["act_dir"])
        if not (d_pos > 0.0 and abs(float(pos["act_val"])) >= p25_abs):
            pos_ineligible += 1
        if abs(float(dist["act_val"])) >= p25_abs:
            dist_abs_ge_p25 += 1
        if float(dist["window_max_act_dir"]) >= p90:
            dist_win_ge_p90 += 1

    n_poles = len(poles_by_id)
    full_poles = 0
    imbalanced_full = 0
    imbalanced_half_0 = 0
    imbalanced_half_1 = 0
    imbalanced_floor_6 = 0
    pos_same_doc_within_pole = 0
    dist_same_doc_within_pole = 0
    suffix_dup_within_pole_pos = 0
    pool_min_fracs: list[float] = []
    sampled_min_fracs: list[float] = []
    max_rank_scanned_all = 0
    total_suffix_dups_skipped = 0

    for pid in poles_by_id:
        rows = sorted(by_pole.get(pid, []), key=lambda r: int(r["pair_idx"]))
        if len(rows) == pairs_per_pole:
            full_poles += 1

        pos_docs = [int(r["pos"]["doc_id"]) for r in rows]
        dist_docs = [int(r["dist"]["doc_id"]) for r in rows]
        pos_same_doc_within_pole += len(pos_docs) - len(set(pos_docs))
        dist_same_doc_within_pole += len(dist_docs) - len(set(dist_docs))

        suf_list = [str(r["pos"]["suffix_16_hash"]) for r in rows if "suffix_16_hash" in r["pos"]]
        suffix_dup_within_pole_pos += len(suf_list) - len(set(suf_list))

        a_full = sum(1 for r in rows if r["pos_slot"] == "A")
        if a_full * 2 != len(rows):
            imbalanced_full += 1

        h0 = [r for r in rows if r.get("split_half") == 0]
        h1 = [r for r in rows if r.get("split_half") == 1]
        if len(h0) != len(rows) // 2 or sum(1 for r in h0 if r["pos_slot"] == "A") * 2 != len(h0):
            imbalanced_half_0 += 1
        if len(h1) != len(rows) - len(h0) or sum(1 for r in h1 if r["pos_slot"] == "A") * 2 != len(h1):
            imbalanced_half_1 += 1

        f6 = [r for r in rows if bool(r.get("in_floor_6"))]
        if len(f6) != min(floor_n, len(rows)) or sum(1 for r in f6 if r["pos_slot"] == "A") * 2 != len(f6):
            imbalanced_floor_6 += 1

        pm = pool_meta_by_pole[pid]
        pool_min_fracs.append(float(pm["top100_min_frac_of_max"]))
        sampled_min_fracs.append(float(pm["sampled_20_min_frac_of_max"]))
        max_rank_scanned_all = max(max_rank_scanned_all, int(pm["top100_max_doc_rank_scanned"]))
        total_suffix_dups_skipped += int(pm["top100_suffix_dups_skipped"])

    all_passed = bool(
        total_pairs == n_poles * pairs_per_pole
        and exact_win == total_pairs
        and exact_off == total_pairs
        and bos_leaks == 0
        and same_doc_leaks == 0
        and ulto85_doc_overlap == 0
        and pos_ineligible == 0
        and dist_abs_ge_p25 == 0
        and dist_win_ge_p90 == 0
        and pos_same_doc_within_pole == 0
        and dist_same_doc_within_pole == 0
        and suffix_dup_within_pole_pos == 0
        and full_poles == n_poles
        and imbalanced_full == 0
        and imbalanced_half_0 == 0
        and imbalanced_half_1 == 0
        and imbalanced_floor_6 == 0
    )

    arr_pool = np.asarray(pool_min_fracs, dtype=np.float64)
    arr_samp = np.asarray(sampled_min_fracs, dtype=np.float64)

    return {
        "all_hard_invariants_passed": all_passed,
        "band": "top100",
        "n_neurons": len(catalog["neurons"]),
        "n_active_poles": n_poles,
        "total_pairs": total_pairs,
        "target_pairs": n_poles * pairs_per_pole,
        "full_poles": full_poles,
        "expected_window_tokens": ws.total,
        "expected_target_offset": ws.target_offset,
        "window_length_exact_rate": exact_win / max(total_pairs, 1),
        "target_offset_exact_rate": exact_off / max(total_pairs, 1),
        "bos_pos0_leak_count": bos_leaks,
        "same_doc_leak_count": same_doc_leaks,
        "pos_same_doc_within_pole_count": pos_same_doc_within_pole,
        "dist_same_doc_within_pole_count": dist_same_doc_within_pole,
        "min_allowed_doc_id": min_allowed_doc_id,
        "min_observed_doc_id": min_doc_id_seen,
        "max_observed_doc_id": max_doc_id_seen,
        "ulto85_doc_overlap_count": ulto85_doc_overlap,
        "pos_ineligible_count": pos_ineligible,
        "dist_abs_ge_p25_count": dist_abs_ge_p25,
        "dist_window_ge_p90_count": dist_win_ge_p90,
        "suffix_dup_within_pole_pos_count": suffix_dup_within_pole_pos,
        "pos_slot_balance": {
            "slot_A_count": int(slot_counts["A"]),
            "slot_B_count": int(slot_counts["B"]),
            "slot_A_rate": slot_counts["A"] / max(total_pairs, 1),
        },
        "nested_ab_balance": {
            "imbalanced_full_poles": imbalanced_full,
            "imbalanced_half_0_poles": imbalanced_half_0,
            "imbalanced_half_1_poles": imbalanced_half_1,
            "imbalanced_floor_6_poles": imbalanced_floor_6,
        },
        "top100_pool_summary": {
            "max_doc_rank_scanned_across_poles": max_rank_scanned_all,
            "total_suffix_dups_skipped": total_suffix_dups_skipped,
            "pool_min_frac_of_max_stats": {
                "min": float(np.min(arr_pool)) if arr_pool.size else 0.0,
                "p10": float(np.quantile(arr_pool, 0.10)) if arr_pool.size else 0.0,
                "p25": float(np.quantile(arr_pool, 0.25)) if arr_pool.size else 0.0,
                "median": float(np.median(arr_pool)) if arr_pool.size else 0.0,
                "p75": float(np.quantile(arr_pool, 0.75)) if arr_pool.size else 0.0,
                "max": float(np.max(arr_pool)) if arr_pool.size else 0.0,
            },
            "sampled_20_min_frac_of_max_stats": {
                "min": float(np.min(arr_samp)) if arr_samp.size else 0.0,
                "p10": float(np.quantile(arr_samp, 0.10)) if arr_samp.size else 0.0,
                "p25": float(np.quantile(arr_samp, 0.25)) if arr_samp.size else 0.0,
                "median": float(np.median(arr_samp)) if arr_samp.size else 0.0,
                "p75": float(np.quantile(arr_samp, 0.75)) if arr_samp.size else 0.0,
                "max": float(np.max(arr_samp)) if arr_samp.size else 0.0,
            },
            "n_poles_pool_min_ge_20pct_max": int(np.sum(arr_pool >= SPARSE_MAX_FRAC_THRESHOLD)),
            "n_poles_pool_min_lt_20pct_max": int(np.sum(arr_pool < SPARSE_MAX_FRAC_THRESHOLD)),
        },
    }


def run_sparsity_and_top100(
    tokenizer: Any,
    base_dir: Path | None = None,
    seed: int = SEED_4B,
    top_pool_size: int = TOP_POOL_SIZE_4B,
    pairs_per_pole: int = PAIRS_P999_PER_POLE,
    floor_n: int = PAIRS_FLOOR6_PER_POLE,
    min_allowed_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Run Item (1) sparsity diagnostics and Item (2) percentile-free top100 builder."""
    t0 = time.time()
    dirs = ensure_4b_dirs(base_dir)
    cache_dir = dirs["cache"]
    active_poles_path = dirs["data"] / "active_poles.json"
    p999_snippets_path = dirs["data"] / "snippets_4b.jsonl"
    top100_snippets_path = dirs["data"] / "snippets_4b_top100.jsonl"
    sparsity_path = dirs["results"] / "pole_sparsity_diagnostics.json"
    checks_path = dirs["results"] / "structural_checks_4b_top100.json"

    catalog = json.loads(active_poles_path.read_text(encoding="utf-8"))
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

    n_total_tokens = int(len(token_ids))
    n_non_bos_tokens = int(np.sum(non_bos))
    n_interior_tokens = int(np.sum(interior_mask))
    dedup_len = min(DEDUP_SUFFIX_LEN, wspec.before + 1)

    # Load existing p99.9_plus sampled positives (read-only!) for Item (1) diagnostic
    p999_pos_by_pole: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if p999_snippets_path.is_file():
        with p999_snippets_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("band")) == "p99.9_plus":
                    p999_pos_by_pole[str(row["pole_id"])].append(row["pos"])

    def _build_snippet_obj(
        flat_idx: int,
        act_val: float,
        act_dir: float,
        quad_code: int,
        window_max_act_dir: float,
        other_max_act_dir: float,
        p90_dir: float,
        p97_dir: float,
        top100_doc_rank: int | None = None,
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
        if top100_doc_rank is not None:
            obj["top100_doc_rank"] = int(top100_doc_rank)
        if include_suffix_hash:
            sk = _suffix_key(token_ids, flat_idx, dedup_len)
            obj["suffix_16_hash"] = hashlib.sha256(
                ",".join(str(x) for x in sk).encode("ascii")
            ).hexdigest()[:16]
        return obj

    all_pair_rows: list[dict[str, Any]] = []
    sparsity_poles: list[dict[str, Any]] = []
    pool_meta_by_pole: dict[str, dict[str, Any]] = {}
    named_diagnostics: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for pole in catalog["poles"]:
        pole_id = str(pole["pole_id"])
        neuron_id = str(pole["neuron_id"])
        k = int(pole["k"])
        layer = int(pole["layer"])
        neuron = int(pole["neuron"])
        sign = int(pole["sign"])
        in_full_expl_50 = bool(pole.get("in_full_expl_50", False))
        named = pole.get("named_neuron")

        p25_abs = float(pole["p25_abs"])
        edges = pole["band_edges_dir"]
        p90 = float(edges["p90"])
        p97 = float(edges["p97"])
        p99_9 = float(edges["p99.9"])

        col = neuron_to_col[(layer, neuron)]
        a_n = acts[:, col].astype(np.float32)
        q_n = quads[:, col]
        d_n = float(sign) * a_n
        abs_n = np.abs(a_n)

        # ---- Item (1): Exact sparsity diagnostics on 1.536M corpus (non-BOS scope) ----
        d_nb = d_n[non_bos]
        abs_nb = abs_n[non_bos]
        max_act_dir = float(np.max(d_nb))
        max_act_dir_all = float(np.max(d_n))
        max_act_dir_int = float(np.max(d_n[interior_mask]))

        thresh_20pct = SPARSE_MAX_FRAC_THRESHOLD * max_act_dir
        p999_frac_of_max = float(p99_9 / max_act_dir) if max_act_dir > 0.0 else 0.0
        p999_frac_of_max_all = float(p99_9 / max_act_dir_all) if max_act_dir_all > 0.0 else 0.0

        frac_pos_dir = float(np.mean(d_nb > 0.0))
        frac_ge_p25_abs = float(np.mean((d_nb > 0.0) & (abs_nb >= p25_abs)))
        n_ge_20pct = int(np.sum(d_nb >= thresh_20pct))
        frac_ge_20pct = float(n_ge_20pct / max(n_non_bos_tokens, 1))
        n_int_ge_20pct = int(np.sum(d_n[interior_mask] >= thresh_20pct))
        n_all_ge_20pct = int(np.sum(d_n >= (SPARSE_MAX_FRAC_THRESHOLD * max_act_dir_all)))

        # ---- Item (2): Top-100 strongest eligible interior tokens across 100 distinct docs ----
        other_max_dir, win_max_dir = compute_other_and_window_max_1d(d_n, wspec)
        pos_guard = interior_mask & (d_n > 0.0) & (abs_n >= p25_abs)
        dz_tier0_mask = interior_mask & (abs_n < p25_abs) & (win_max_dir < p90)

        top_pool = select_top_k_distinct_doc_indices(
            d_n=d_n,
            eligible_mask=pos_guard,
            doc_ids=doc_ids,
            token_ids=token_ids,
            top_k=top_pool_size,
            suffix_len=dedup_len,
        )
        pool_indices = top_pool["indices"]
        pool_ranks = top_pool["doc_ranks"]

        # Deterministically sample 20 positives from the 100-doc strong pool (unbiased IID order)
        rng_pick = _stable_rng_4b(seed, pole_id, "sample_top100_20")
        chosen_pos_slots = rng_pick.permutation(len(pool_indices))[:pairs_per_pole].tolist()
        picked_indices = [pool_indices[j] for j in chosen_pos_slots]
        picked_ranks = [pool_ranks[j] for j in chosen_pos_slots]

        # Pair with 20 strict Tier-0 dead_zone distractors (distinct docs, whole-window < p90)
        rng_dz = _stable_rng_4b(seed, pole_id, "dist_dead_zone_top100")
        paired = _select_strict_tier0_pairs(
            tier0_mask=dz_tier0_mask,
            pos_indices=picked_indices,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            rng=rng_dz,
            min_spacing=wspec.total,
            allow_same_doc_fallback=False,
        )
        if len(paired) != pairs_per_pole:
            raise RuntimeError(
                f"Pole {pole_id}: only formed {len(paired)}/{pairs_per_pole} top100 pairs "
                f"(dz_cands={int(np.sum(dz_tier0_mask))})"
            )

        slots = nested_balanced_p999_slots(
            len(paired), seed=seed, pole_id=pole_id, floor_n=floor_n
        )
        half_cut = len(paired) // 2

        top100_sampled_pos_objs: list[dict[str, Any]] = []
        for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired, slots)):
            p_idx = picked_indices[pos_k]
            p_rank = picked_ranks[pos_k]
            pos_obj = _build_snippet_obj(
                flat_idx=p_idx,
                act_val=float(a_n[p_idx]),
                act_dir=float(d_n[p_idx]),
                quad_code=int(q_n[p_idx]),
                window_max_act_dir=float(win_max_dir[p_idx]),
                other_max_act_dir=float(other_max_dir[p_idx]),
                p90_dir=p90,
                p97_dir=p97,
                top100_doc_rank=p_rank,
                include_suffix_hash=True,
            )
            dist_obj = _build_snippet_obj(
                flat_idx=d_idx,
                act_val=float(a_n[d_idx]),
                act_dir=float(d_n[d_idx]),
                quad_code=int(q_n[d_idx]),
                window_max_act_dir=float(win_max_dir[d_idx]),
                other_max_act_dir=float(other_max_dir[d_idx]),
                p90_dir=p90,
                p97_dir=p97,
                top100_doc_rank=None,
                include_suffix_hash=False,
            )
            top100_sampled_pos_objs.append(pos_obj)
            all_pair_rows.append(
                {
                    "pair_id": f"{pole_id}__top100__dead_zone__{pair_idx}",
                    "pole_id": pole_id,
                    "neuron_id": neuron_id,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "band": "top100",
                    "distractor_type": "dead_zone",
                    "distractor_tier": 0,
                    "pair_idx": pair_idx,
                    "pos_slot": slot,
                    "in_floor_6": bool(pair_idx < floor_n),
                    "split_half": 0 if pair_idx < half_cut else 1,
                    "in_full_expl_50": in_full_expl_50,
                    "pos": pos_obj,
                    "dist": dist_obj,
                }
            )

        # Summarize p99.9+ vs top100 sampled positives for this pole
        def _summarize_sampled(pos_objs: Sequence[Mapping[str, Any]], include_tokens: bool) -> dict[str, Any]:
            if not pos_objs:
                return {
                    "n_pairs": 0,
                    "n_ge_20pct_max": 0,
                    "n_lt_20pct_max": 0,
                }
            vals = np.array([float(o["act_dir"]) for o in pos_objs], dtype=np.float64)
            n_ge = int(np.sum(vals >= thresh_20pct))
            res: dict[str, Any] = {
                "n_pairs": int(len(vals)),
                "min_pos_act_dir": float(np.min(vals)),
                "median_pos_act_dir": float(np.median(vals)),
                "mean_pos_act_dir": float(np.mean(vals)),
                "max_pos_act_dir": float(np.max(vals)),
                "min_frac_of_max": float(np.min(vals) / max_act_dir) if max_act_dir > 0.0 else 0.0,
                "median_frac_of_max": float(np.median(vals) / max_act_dir) if max_act_dir > 0.0 else 0.0,
                "n_ge_20pct_max": n_ge,
                "n_lt_20pct_max": int(len(vals) - n_ge),
            }
            if include_tokens:
                ranked_samples = sorted(pos_objs, key=lambda o: float(o["act_dir"]), reverse=True)
                res["pos_act_dirs_desc"] = [round(float(o["act_dir"]), 4) for o in ranked_samples]
                res["pos_samples"] = [
                    {
                        "act_dir": round(float(o["act_dir"]), 4),
                        "frac_of_max": round(float(o["act_dir"]) / max_act_dir, 4) if max_act_dir > 0.0 else 0.0,
                        "target_token_str": str(o["target_token_str"]),
                        "doc_id": int(o["doc_id"]),
                    }
                    for o in ranked_samples
                ]
            return res

        p999_samp_summary = _summarize_sampled(p999_pos_by_pole.get(pole_id, []), include_tokens=bool(named))
        top100_samp_summary = _summarize_sampled(top100_sampled_pos_objs, include_tokens=bool(named))

        top100_min_frac = float(top_pool["min_act_dir"] / max_act_dir) if max_act_dir > 0.0 else 0.0
        pool_acts_arr = np.array([float(d_n[i]) for i in pool_indices], dtype=np.float64)
        pool_n_ge_20pct = int(np.sum(pool_acts_arr >= thresh_20pct))

        pool_meta = {
            "top100_pool_size": int(top_pool["pool_size"]),
            "top100_max_doc_rank_scanned": int(top_pool["max_doc_rank_scanned"]),
            "top100_suffix_dups_skipped": int(top_pool["suffix_dups_skipped"]),
            "top100_max_act_dir": float(top_pool["max_act_dir"]),
            "top100_min_act_dir": float(top_pool["min_act_dir"]),
            "top100_median_act_dir": float(top_pool["median_act_dir"]),
            "top100_min_frac_of_max": top100_min_frac,
            "top100_n_ge_20pct_max": pool_n_ge_20pct,
            "sampled_20_min_frac_of_max": float(top100_samp_summary["min_frac_of_max"]),
            "sampled_20_median_frac_of_max": float(top100_samp_summary["median_frac_of_max"]),
            "sampled_20_n_ge_20pct_max": int(top100_samp_summary["n_ge_20pct_max"]),
        }
        pool_meta_by_pole[pole_id] = pool_meta

        ulto = pole.get("ulto85_scores") or {}
        diag_row = {
            "pole_id": pole_id,
            "neuron_id": neuron_id,
            "k": k,
            "layer": layer,
            "neuron": neuron,
            "sign": sign,
            "pole_name": str(pole["pole_name"]),
            "is_bipolar": bool(pole["is_bipolar"]),
            "is_dominant_pole": bool(pole.get("is_dominant_pole", False)),
            "named_neuron": named,
            "label": str(pole["label"]),
            "p25_abs": p25_abs,
            "p99.9_thresh": p99_9,
            "max_act_dir": max_act_dir,
            "max_act_dir_all_tokens": max_act_dir_all,
            "max_act_dir_interior": max_act_dir_int,
            "p99.9_frac_of_max": p999_frac_of_max,
            "p99.9_frac_of_max_all_tokens": p999_frac_of_max_all,
            "p99_9_lt_20pct_max": bool(p99_9 < thresh_20pct),
            "frac_pos": frac_pos_dir,
            "frac_ge_p25_abs": frac_ge_p25_abs,
            "frac_ge_20pct_max": frac_ge_20pct,
            "n_tokens_ge_20pct_max": n_ge_20pct,
            "n_interior_ge_20pct_max": n_int_ge_20pct,
            "n_tokens_ge_20pct_max_all_tokens": n_all_ge_20pct,
            "sim_cal_rec": ulto.get("sim_cal_rec"),
            "sim_cal_rec_active": ulto.get("sim_cal_rec_active"),
            "strong20_rec": ulto.get("strong20_rec"),
            "p99_9_plus_sampled": p999_samp_summary,
            "top100_pool": pool_meta,
            "top100_sampled": top100_samp_summary,
        }
        sparsity_poles.append(diag_row)
        if named is not None:
            named_diagnostics[str(named)].append(diag_row)

    # Verify top100 invariants before persisting
    checks = verify_top100_snippets_4b(
        pair_rows=all_pair_rows,
        catalog=catalog,
        pool_meta_by_pole=pool_meta_by_pole,
        pairs_per_pole=pairs_per_pole,
        floor_n=floor_n,
        min_allowed_doc_id=min_allowed_doc_id,
        wspec=wspec,
    )
    checks["elapsed_sec"] = round(time.time() - t0, 2)

    sparse_sorted = sorted(
        [r for r in sparsity_poles if r["p99_9_lt_20pct_max"]],
        key=lambda r: (float(r["p99.9_frac_of_max"]), str(r["pole_id"])),
    )
    fracs_arr = np.array([float(r["p99.9_frac_of_max"]) for r in sparsity_poles], dtype=np.float64)
    sparse_by_neuron: dict[int, list[bool]] = defaultdict(list)
    for r in sparsity_poles:
        sparse_by_neuron[int(r["k"])].append(bool(r["p99_9_lt_20pct_max"]))

    sparsity_doc = {
        "model_id": catalog.get("model_id", MODEL_ID_4B),
        "n_neurons": len(catalog["neurons"]),
        "n_active_poles": len(sparsity_poles),
        "n_total_tokens": n_total_tokens,
        "n_non_bos_tokens": n_non_bos_tokens,
        "n_interior_tokens": n_interior_tokens,
        "max_act_dir_scope": "non_bos",
        "sparse_threshold_frac_of_max": SPARSE_MAX_FRAC_THRESHOLD,
        "n_poles_lt_20pct_max": len(sparse_sorted),
        "frac_poles_lt_20pct_max": len(sparse_sorted) / max(len(sparsity_poles), 1),
        "n_poles_lt_20pct_max_all_tokens": sum(
            1 for r in sparsity_poles if float(r["p99.9_frac_of_max_all_tokens"]) < SPARSE_MAX_FRAC_THRESHOLD
        ),
        "n_neurons_any_pole_lt_20pct_max": sum(1 for vs in sparse_by_neuron.values() if any(vs)),
        "n_neurons_all_poles_lt_20pct_max": sum(1 for vs in sparse_by_neuron.values() if all(vs)),
        "p99_9_frac_of_max_stats": {
            "min": float(np.min(fracs_arr)) if fracs_arr.size else 0.0,
            "p10": float(np.quantile(fracs_arr, 0.10)) if fracs_arr.size else 0.0,
            "p25": float(np.quantile(fracs_arr, 0.25)) if fracs_arr.size else 0.0,
            "median": float(np.median(fracs_arr)) if fracs_arr.size else 0.0,
            "p75": float(np.quantile(fracs_arr, 0.75)) if fracs_arr.size else 0.0,
            "p90": float(np.quantile(fracs_arr, 0.90)) if fracs_arr.size else 0.0,
            "max": float(np.max(fracs_arr)) if fracs_arr.size else 0.0,
        },
        "named_neurons_diagnostic": dict(named_diagnostics),
        "sparse_poles_lt_20pct_max": sparse_sorted,
        "poles": sparsity_poles,
        "elapsed_sec": checks["elapsed_sec"],
    }

    with top100_snippets_path.open("w", encoding="utf-8") as f:
        for row in all_pair_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sparsity_path.write_text(json.dumps(sparsity_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    if not checks["all_hard_invariants_passed"]:
        raise AssertionError(
            f"Structural checks FAILED for snippets_4b_top100.jsonl: {json.dumps(checks, indent=2)}"
        )

    print(
        f"[top100_4b] COMPLETE in {checks['elapsed_sec']:.1f}s | "
        f"sparse(<20% max)={len(sparse_sorted)}/{len(sparsity_poles)} poles | "
        f"top100 pairs={checks['total_pairs']:,} ({checks['full_poles']}/{checks['n_active_poles']} poles) | "
        f"all_hard_invariants_passed={checks['all_hard_invariants_passed']}"
    )
    return {
        "sparsity": sparsity_doc,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute Qwen3-4B pole sparsity diagnostics and build percentile-free top100 snippets."
    )
    parser.add_argument("--model-id", default=MODEL_ID_4B)
    parser.add_argument("--seed", type=int, default=SEED_4B)
    parser.add_argument("--top-pool-size", type=int, default=TOP_POOL_SIZE_4B)
    parser.add_argument("--pairs-per-pole", type=int, default=PAIRS_P999_PER_POLE)
    parser.add_argument("--floor-n", type=int, default=PAIRS_FLOOR6_PER_POLE)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    run_sparsity_and_top100(
        tokenizer=tokenizer,
        seed=args.seed,
        top_pool_size=args.top_pool_size,
        pairs_per_pole=args.pairs_per_pole,
        floor_n=args.floor_n,
    )


if __name__ == "__main__":
    main()
