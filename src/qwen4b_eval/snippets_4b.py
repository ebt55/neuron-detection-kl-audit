"""Symmetric 69-token snippet cutting and nested-balanced pair builder for Qwen3-4B.

Guarantees for all 424 active poles (278 scored Ulto85 neurons):
- Exact 69-token windows (`60 before + <<target>> + 8 after`, `target_offset == 60`,
  strict interior `60 <= pos_in_chunk <= chunk_len - 9`, 0 `<bos>`, 0 truncation).
- Strict document disjointness: `pos.doc_id != dist.doc_id` on every pair, and all
  documents come strictly from FineWeb `000_00000.parquet` rows `60,000+` (`doc_id >= 60_000`).
- 16-token suffix deduplication (`token_ids[i-15 : i+1]`) across positives per pole.
- Strict Tier-0 `dead_zone` distractors (`|a| < p25_abs` AND whole 69-token window `< p90`).
- Two bands per active pole (`12,720` total pairs):
  1. `p99.9_plus`: **20 pairs/pole** (`8,480` pairs) with EXACT nested A/B balance:
     - `pair_idx 0..5` (`in_floor_6=True`): exact `3/3` A/B
     - `pair_idx 6..9`: exact `2/2` A/B  -> `0..9` (`split_half=0`): exact `5/5` A/B
     - `pair_idx 10..19` (`split_half=1`): exact `5/5` A/B
     - Total `0..19`: exact `10/10` A/B
  2. `p97_99`: **10 pairs/pole** (`4,240` pairs, strictly `p97 <= d_pos < p99`) with
     EXACT `5/5` A/B balance per pole.
- Saves `data/qwen4b/snippets_4b.jsonl` and verifies all hard invariants in
  `results/qwen4b/structural_checks_4b.json`.
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
    PAIRS_P97_99_PER_POLE,
    PAIRS_P999_PER_POLE,
    SEED_4B,
    ensure_4b_dirs,
)
from w_audit.config import DEDUP_SUFFIX_LEN, WindowSpec
from w_audit.extract import QUAD_CODE_TO_NAME
from w_audit.snippets import (
    _sample_distinct_doc_indices,
    _select_strict_tier0_pairs,
    _suffix_key,
    compute_other_and_window_max_1d,
    render_marked_window,
)


def _stable_rng_4b(seed: int, *parts: Any) -> np.random.Generator:
    key = ":".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def nested_balanced_p999_slots(
    n_pairs: int,
    seed: int,
    pole_id: str,
    floor_n: int = PAIRS_FLOOR6_PER_POLE,
) -> list[str]:
    """Return exact nested A/B slot assignment for `p99.9_plus`.

    For the canonical `n_pairs=20, floor_n=6`:
      - Block `[0..5]`   (`in_floor_6=True`):  exact `3/3` A/B
      - Block `[6..9]`:                        exact `2/2` A/B  -> `[0..9]` (`split_half=0`): exact `5/5` A/B
      - Block `[10..19]` (`split_half=1`):     exact `5/5` A/B
      - Total `[0..19]`:                       exact `10/10` A/B

    Works cleanly for any even `n_pairs` (and even `floor_n <= n_pairs // 2`).
    """
    if n_pairs < 0 or (n_pairs % 2) != 0:
        raise ValueError(f"n_pairs must be non-negative even, got {n_pairs}")
    if n_pairs == 0:
        return []

    def _shuffled_half(k: int, tag: str) -> list[str]:
        if k == 0:
            return []
        if k % 2 != 0:
            raise ValueError(f"Block size {k} for {tag} must be even")
        arr = ["A"] * (k // 2) + ["B"] * (k // 2)
        rng = _stable_rng_4b(seed, pole_id, "p99.9_plus_slots", tag)
        order = rng.permutation(k).tolist()
        return [arr[i] for i in order]

    half = n_pairs // 2
    if (half % 2 == 0) and (floor_n % 2 == 0) and (0 < floor_n <= half):
        # e.g., n=20 -> half=10 (even), floor_n=6 -> [0..5]=3/3, [6..9]=2/2, [10..19]=5/5
        block1 = _shuffled_half(floor_n, "floor6")
        block2 = _shuffled_half(half - floor_n, "half0_rest")
        block3 = _shuffled_half(n_pairs - half, "half1")
        return block1 + block2 + block3

    # General even fallback (e.g., tiny test n=6 or n=4)
    if floor_n % 2 == 0 and 0 < floor_n < n_pairs:
        return _shuffled_half(floor_n, "floor") + _shuffled_half(n_pairs - floor_n, "rest")
    return _shuffled_half(n_pairs, "all")


def balanced_even_slots(
    n_pairs: int,
    seed: int,
    pole_id: str,
    band: str,
) -> list[str]:
    """Return exact k/k 50/50 deterministic shuffle of 'A' and 'B'."""
    if n_pairs < 0 or (n_pairs % 2) != 0:
        raise ValueError(f"n_pairs must be non-negative even, got {n_pairs}")
    if n_pairs == 0:
        return []
    arr = ["A"] * (n_pairs // 2) + ["B"] * (n_pairs // 2)
    rng = _stable_rng_4b(seed, pole_id, band, "slots")
    order = rng.permutation(n_pairs).tolist()
    return [arr[i] for i in order]


def verify_snippets_4b(
    pair_rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    pairs_p999_target: int = PAIRS_P999_PER_POLE,
    pairs_floor6_target: int = PAIRS_FLOOR6_PER_POLE,
    pairs_p97_99_target: int = PAIRS_P97_99_PER_POLE,
    min_allowed_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
    wspec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Verify every structural, band, distractor, doc-overlap, and nested A/B invariant."""
    ws = wspec if wspec is not None else WindowSpec()
    poles_by_id = {str(p["pole_id"]): p for p in catalog["poles"]}

    total_pairs = len(pair_rows)
    exact_win = 0
    exact_off = 0
    bos_leaks = 0
    same_doc_leaks = 0
    ulto85_doc_overlap = 0
    dist_abs_ge_p25 = 0
    dist_win_ge_p90 = 0
    p999_out_of_band = 0
    p97_99_out_of_band = 0
    min_doc_id_seen: int | None = None
    max_doc_id_seen: int | None = None

    by_pole_band: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    pos_suffix_keys_by_pole: dict[str, list[str]] = defaultdict(list)
    slot_counts = Counter()

    for row in pair_rows:
        pid = str(row["pole_id"])
        band = str(row["band"])
        slot = str(row["pos_slot"])
        slot_counts[slot] += 1
        pole = poles_by_id[pid]
        edges = pole["band_edges_dir"]
        p25_abs = float(pole["p25_abs"])
        p90 = float(edges["p90"])
        p97 = float(edges["p97"])
        p99 = float(edges["p99"])
        p99_9 = float(edges["p99.9"])

        pos = row["pos"]
        dist = row["dist"]
        by_pole_band[(pid, band)].append(row)

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

        # Positive band bounds
        d_pos = float(pos["act_dir"])
        if band == "p99.9_plus":
            if not (d_pos >= p99_9 and d_pos > 0.0 and abs(float(pos["act_val"])) >= p25_abs):
                p999_out_of_band += 1
        elif band == "p97_99":
            if not (
                p97 <= d_pos < p99 and d_pos > 0.0 and abs(float(pos["act_val"])) >= p25_abs
            ):
                p97_99_out_of_band += 1
        else:
            raise ValueError(f"Unexpected band {band!r} in snippets_4b.jsonl")

        # Distractor dead-zone & whole-window < p90 checks
        if abs(float(dist["act_val"])) >= p25_abs:
            dist_abs_ge_p25 += 1
        if float(dist["window_max_act_dir"]) >= p90:
            dist_win_ge_p90 += 1

        if "suffix_16_hash" in pos:
            pos_suffix_keys_by_pole[pid].append(str(pos["suffix_16_hash"]))

    # Nested A/B balance & exact survival checks per pole
    n_poles = len(poles_by_id)
    p999_full_poles = 0
    p97_99_full_poles = 0
    p97_99_zero_clipped_poles = 0
    p97_99_unexpected_short_poles = 0
    p97_99_clipped_pole_details: list[dict[str, Any]] = []
    imbalanced_p999_full = 0
    imbalanced_floor_6 = 0
    imbalanced_half_0 = 0
    imbalanced_half_1 = 0
    imbalanced_p97_99 = 0
    p999_total_pairs = 0
    p97_99_total_pairs = 0
    floor_6_total_pairs = 0
    full_expl_50_p999_pairs = 0
    pos_same_doc_within_band = 0
    suffix_dup_within_pole_pos = 0

    for pid in poles_by_id:
        suf_list = pos_suffix_keys_by_pole.get(pid, [])
        suffix_dup_within_pole_pos += len(suf_list) - len(set(suf_list))

        # Check p99.9_plus
        rows_999 = sorted(by_pole_band.get((pid, "p99.9_plus"), []), key=lambda r: int(r["pair_idx"]))
        p999_total_pairs += len(rows_999)
        if len(rows_999) == pairs_p999_target:
            p999_full_poles += 1
        pos_docs_999 = [int(r["pos"]["doc_id"]) for r in rows_999]
        pos_same_doc_within_band += len(pos_docs_999) - len(set(pos_docs_999))

        a_full = sum(1 for r in rows_999 if r["pos_slot"] == "A")
        if a_full * 2 != len(rows_999):
            imbalanced_p999_full += 1

        f6 = [r for r in rows_999 if bool(r["in_floor_6"])]
        floor_6_total_pairs += len(f6)
        a_f6 = sum(1 for r in f6 if r["pos_slot"] == "A")
        if len(f6) != min(pairs_floor6_target, len(rows_999)) or a_f6 * 2 != len(f6):
            imbalanced_floor_6 += 1

        h0 = [r for r in rows_999 if r.get("split_half") == 0]
        h1 = [r for r in rows_999 if r.get("split_half") == 1]
        a_h0 = sum(1 for r in h0 if r["pos_slot"] == "A")
        a_h1 = sum(1 for r in h1 if r["pos_slot"] == "A")
        if len(h0) != len(rows_999) // 2 or a_h0 * 2 != len(h0):
            imbalanced_half_0 += 1
        if len(h1) != len(rows_999) - len(h0) or a_h1 * 2 != len(h1):
            imbalanced_half_1 += 1

        if bool(poles_by_id[pid].get("in_full_expl_50", False)):
            full_expl_50_p999_pairs += len(rows_999)

        # Check p97_99 (either full 10/10 or 0 when entire [p97, p99) is sign/dead-zone clipped)
        rows_97 = sorted(by_pole_band.get((pid, "p97_99"), []), key=lambda r: int(r["pair_idx"]))
        p97_99_total_pairs += len(rows_97)
        pole_meta = poles_by_id[pid]
        elig_97 = int((pole_meta.get("eligible_interior_counts") or {}).get("p97_99", -1))
        if len(rows_97) == pairs_p97_99_target:
            p97_99_full_poles += 1
        elif len(rows_97) == 0 and elig_97 == 0:
            p97_99_zero_clipped_poles += 1
            edges_m = pole_meta["band_edges_dir"]
            p97_99_clipped_pole_details.append(
                {
                    "pole_id": pid,
                    "k": int(pole_meta["k"]),
                    "layer": int(pole_meta["layer"]),
                    "neuron": int(pole_meta["neuron"]),
                    "sign": int(pole_meta["sign"]),
                    "frac_same_sign": float(
                        pole_meta["frac_pos"] if int(pole_meta["sign"]) > 0 else pole_meta["frac_neg"]
                    ),
                    "p25_abs": float(pole_meta["p25_abs"]),
                    "p97": float(edges_m["p97"]),
                    "p99": float(edges_m["p99"]),
                    "p99.9": float(edges_m["p99.9"]),
                    "reason": "entire [p97, p99) interval lies at d <= 0 or |a| < p25_abs (0 eligible interior tokens)",
                }
            )
        else:
            p97_99_unexpected_short_poles += 1

        pos_docs_97 = [int(r["pos"]["doc_id"]) for r in rows_97]
        pos_same_doc_within_band += len(pos_docs_97) - len(set(pos_docs_97))

        a_97 = sum(1 for r in rows_97 if r["pos_slot"] == "A")
        if a_97 * 2 != len(rows_97):
            imbalanced_p97_99 += 1

    all_passed = bool(
        total_pairs > 0
        and exact_win == total_pairs
        and exact_off == total_pairs
        and bos_leaks == 0
        and same_doc_leaks == 0
        and ulto85_doc_overlap == 0
        and dist_abs_ge_p25 == 0
        and dist_win_ge_p90 == 0
        and p999_out_of_band == 0
        and p97_99_out_of_band == 0
        and imbalanced_p999_full == 0
        and imbalanced_floor_6 == 0
        and imbalanced_half_0 == 0
        and imbalanced_half_1 == 0
        and imbalanced_p97_99 == 0
        and p999_full_poles == n_poles
        and p97_99_unexpected_short_poles == 0
        and (p97_99_full_poles + p97_99_zero_clipped_poles) == n_poles
        and suffix_dup_within_pole_pos == 0
    )

    return {
        "all_hard_invariants_passed": all_passed,
        "n_neurons": len(catalog["neurons"]),
        "n_active_poles": n_poles,
        "total_pairs": total_pairs,
        "expected_window_tokens": ws.total,
        "expected_target_offset": ws.target_offset,
        "window_length_exact_rate": exact_win / max(total_pairs, 1),
        "target_offset_exact_rate": exact_off / max(total_pairs, 1),
        "bos_pos0_leak_count": bos_leaks,
        "same_doc_leak_count": same_doc_leaks,
        "min_allowed_doc_id": min_allowed_doc_id,
        "min_observed_doc_id": min_doc_id_seen,
        "max_observed_doc_id": max_doc_id_seen,
        "ulto85_doc_overlap_count": ulto85_doc_overlap,
        "dist_abs_ge_p25_count": dist_abs_ge_p25,
        "dist_window_ge_p90_count": dist_win_ge_p90,
        "p99_9_plus_out_of_band_count": p999_out_of_band,
        "p97_99_out_of_band_count": p97_99_out_of_band,
        "suffix_dup_within_pole_pos_count": suffix_dup_within_pole_pos,
        "pos_same_doc_within_band_count": pos_same_doc_within_band,
        "pos_slot_balance": {
            "slot_A_count": int(slot_counts["A"]),
            "slot_B_count": int(slot_counts["B"]),
            "slot_A_rate": slot_counts["A"] / max(total_pairs, 1),
        },
        "nested_ab_balance": {
            "imbalanced_p99_9_full_poles": imbalanced_p999_full,
            "imbalanced_floor_6_poles": imbalanced_floor_6,
            "imbalanced_half_0_poles": imbalanced_half_0,
            "imbalanced_half_1_poles": imbalanced_half_1,
            "imbalanced_p97_99_poles": imbalanced_p97_99,
        },
        "surviving_counts": {
            "p99.9_plus": {
                "pairs": p999_total_pairs,
                "target_pairs": n_poles * pairs_p999_target,
                "full_poles": p999_full_poles,
                "total_poles": n_poles,
            },
            "p99.9_plus_floor_6": {
                "pairs": floor_6_total_pairs,
                "target_pairs": n_poles * pairs_floor6_target,
            },
            "p99.9_plus_full_expl_50": {
                "pairs": full_expl_50_p999_pairs,
                "n_poles": sum(1 for p in poles_by_id.values() if p.get("in_full_expl_50")),
            },
            "p97_99": {
                "pairs": p97_99_total_pairs,
                "max_possible_pairs": n_poles * pairs_p97_99_target,
                "full_poles": p97_99_full_poles,
                "clipped_zero_poles": p97_99_zero_clipped_poles,
                "unexpected_short_poles": p97_99_unexpected_short_poles,
                "total_poles": n_poles,
            },
        },
        "p97_99_clipped_ineligible_poles": p97_99_clipped_pole_details,
    }


def build_snippets_4b(
    tokenizer: Any,
    base_dir: Path | None = None,
    seed: int = SEED_4B,
    pairs_p999: int = PAIRS_P999_PER_POLE,
    pairs_floor6: int = PAIRS_FLOOR6_PER_POLE,
    pairs_p97_99: int = PAIRS_P97_99_PER_POLE,
    min_allowed_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Cut symmetric 69-token snippets and build `data/qwen4b/snippets_4b.jsonl`."""
    t0 = time.time()
    dirs = ensure_4b_dirs(base_dir)
    cache_dir = dirs["cache"]
    active_poles_path = dirs["data"] / "active_poles.json"
    snippets_path = dirs["data"] / "snippets_4b.jsonl"
    checks_path = dirs["results"] / "structural_checks_4b.json"

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

    dedup_len = min(DEDUP_SUFFIX_LEN, wspec.before + 1)

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

    all_pair_rows: list[dict[str, Any]] = []

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

        other_max_dir, win_max_dir = compute_other_and_window_max_1d(d_n, wspec)
        pos_guard = interior_mask & (d_n > 0.0) & (abs_n >= p25_abs)
        dz_tier0_mask = interior_mask & (abs_n < p25_abs) & (win_max_dir < p90)

        # ------------------------------------------------------------------
        # 1. Sample 20 p99.9_plus positives (distinct docs + 16-tok suffix dedup)
        # ------------------------------------------------------------------
        cands_999 = np.flatnonzero(pos_guard & (d_n >= p99_9))
        rng_999 = _stable_rng_4b(seed, pole_id, "pos_p99.9_plus")
        picked_999 = _sample_distinct_doc_indices(
            candidate_indices=cands_999,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            n_want=pairs_p999,
            rng=rng_999,
            min_spacing=wspec.total,
            token_ids=token_ids,
            dedup_within_sample=True,
            suffix_len=dedup_len,
            allow_same_doc_fallback=True,
        )
        # Deterministic shuffle so floor-6 (0..5) and halves (0..9, 10..19) are unbiased IID draws
        rng_shuf_999 = _stable_rng_4b(seed, pole_id, "shuffle_p99.9_plus")
        rng_shuf_999.shuffle(picked_999)

        used_pos_docs_999 = {int(doc_ids[idx]) for idx in picked_999}
        used_pos_suffixes = {_suffix_key(token_ids, idx, dedup_len) for idx in picked_999}

        # ------------------------------------------------------------------
        # 2. Sample 10 p97_99 positives (p97 <= d_pos < p99, distinct docs & suffix dedup)
        # ------------------------------------------------------------------
        cands_97 = np.flatnonzero(pos_guard & (d_n >= p97) & (d_n < p99))
        rng_97 = _stable_rng_4b(seed, pole_id, "pos_p97_99")
        picked_97 = _sample_distinct_doc_indices(
            candidate_indices=cands_97,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            n_want=pairs_p97_99,
            rng=rng_97,
            min_spacing=wspec.total,
            excluded_docs=used_pos_docs_999,
            token_ids=token_ids,
            forbidden_suffixes=used_pos_suffixes,
            dedup_within_sample=True,
            suffix_len=dedup_len,
            allow_same_doc_fallback=False,
        )
        # If excluding p99.9 docs left fewer than 10, top up from remaining docs (still suffix-deduped)
        if len(picked_97) < pairs_p97_99:
            already_docs_97 = {int(doc_ids[idx]) for idx in picked_97}
            already_sufs = used_pos_suffixes | {
                _suffix_key(token_ids, idx, dedup_len) for idx in picked_97
            }
            extra_97 = _sample_distinct_doc_indices(
                candidate_indices=cands_97,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=pairs_p97_99 - len(picked_97),
                rng=rng_97,
                min_spacing=wspec.total,
                excluded_docs=already_docs_97,
                token_ids=token_ids,
                forbidden_suffixes=already_sufs,
                dedup_within_sample=True,
                suffix_len=dedup_len,
                allow_same_doc_fallback=True,
            )
            picked_97.extend(extra_97)

        rng_shuf_97 = _stable_rng_4b(seed, pole_id, "shuffle_p97_99")
        rng_shuf_97.shuffle(picked_97)

        # ------------------------------------------------------------------
        # 3. Pair with strict Tier-0 dead_zone distractors (|a| < p25_abs & win_max < p90)
        # ------------------------------------------------------------------
        dz_used_docs: set[int] = set()

        # 3a. p99.9_plus (20 pairs, exact nested 3/3 floor-6, 5/5 halves, 10/10 total)
        rng_dz_999 = _stable_rng_4b(seed, pole_id, "dist_dead_zone_p99.9_plus")
        paired_999 = _select_strict_tier0_pairs(
            tier0_mask=dz_tier0_mask,
            pos_indices=picked_999,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            rng=rng_dz_999,
            used_docs_across_bands=dz_used_docs,
            min_spacing=wspec.total,
            allow_same_doc_fallback=False,
        )
        if len(paired_999) != pairs_p999:
            raise RuntimeError(
                f"Pole {pole_id}: only formed {len(paired_999)}/{pairs_p999} p99.9_plus pairs "
                f"(cands_999={len(cands_999)}, picked_999={len(picked_999)}, dz_cands={int(np.sum(dz_tier0_mask))})"
            )
        slots_999 = nested_balanced_p999_slots(
            len(paired_999), seed=seed, pole_id=pole_id, floor_n=pairs_floor6
        )
        half_cut = len(paired_999) // 2

        for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired_999, slots_999)):
            p_idx = picked_999[pos_k]
            pair_id = f"{pole_id}__p99.9_plus__dead_zone__{pair_idx}"
            all_pair_rows.append(
                {
                    "pair_id": pair_id,
                    "pole_id": pole_id,
                    "neuron_id": neuron_id,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "band": "p99.9_plus",
                    "distractor_type": "dead_zone",
                    "distractor_tier": 0,
                    "pair_idx": pair_idx,
                    "pos_slot": slot,
                    "in_floor_6": bool(pair_idx < pairs_floor6),
                    "split_half": 0 if pair_idx < half_cut else 1,
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

        # 3b. p97_99 (10 pairs, exact 5/5 A/B balance; skip cleanly if 0 eligible tokens)
        if len(cands_97) == 0:
            continue
        rng_dz_97 = _stable_rng_4b(seed, pole_id, "dist_dead_zone_p97_99")
        paired_97 = _select_strict_tier0_pairs(
            tier0_mask=dz_tier0_mask,
            pos_indices=picked_97,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            rng=rng_dz_97,
            used_docs_across_bands=dz_used_docs,
            min_spacing=wspec.total,
            allow_same_doc_fallback=False,
        )
        if len(paired_97) != pairs_p97_99:
            raise RuntimeError(
                f"Pole {pole_id}: only formed {len(paired_97)}/{pairs_p97_99} p97_99 pairs "
                f"(cands_97={len(cands_97)}, picked_97={len(picked_97)}, dz_cands={int(np.sum(dz_tier0_mask))})"
            )
        slots_97 = balanced_even_slots(len(paired_97), seed=seed, pole_id=pole_id, band="p97_99")

        for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired_97, slots_97)):
            p_idx = picked_97[pos_k]
            pair_id = f"{pole_id}__p97_99__dead_zone__{pair_idx}"
            all_pair_rows.append(
                {
                    "pair_id": pair_id,
                    "pole_id": pole_id,
                    "neuron_id": neuron_id,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "band": "p97_99",
                    "distractor_type": "dead_zone",
                    "distractor_tier": 0,
                    "pair_idx": pair_idx,
                    "pos_slot": slot,
                    "in_floor_6": False,
                    "split_half": None,
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

    # Verify all hard invariants BEFORE returning
    checks = verify_snippets_4b(
        pair_rows=all_pair_rows,
        catalog=catalog,
        pairs_p999_target=pairs_p999,
        pairs_floor6_target=pairs_floor6,
        pairs_p97_99_target=pairs_p97_99,
        min_allowed_doc_id=min_allowed_doc_id,
        wspec=wspec,
    )
    checks["elapsed_sec"] = round(time.time() - t0, 2)

    with snippets_path.open("w", encoding="utf-8") as f:
        for row in all_pair_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    if not checks["all_hard_invariants_passed"]:
        raise AssertionError(
            f"Structural checks FAILED for snippets_4b.jsonl: {json.dumps(checks, indent=2)}"
        )

    print(
        f"[snippets_4b] COMPLETE in {checks['elapsed_sec']:.1f}s | "
        f"total={checks['total_pairs']:,} pairs "
        f"(p99.9+={checks['surviving_counts']['p99.9_plus']['pairs']:,}, "
        f"p97-99={checks['surviving_counts']['p97_99']['pairs']:,}) | "
        f"all_hard_invariants_passed={checks['all_hard_invariants_passed']}"
    )
    return checks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cut symmetric 69-token Qwen3-4B evaluation pairs and verify invariants."
    )
    parser.add_argument("--model-id", default=MODEL_ID_4B)
    parser.add_argument("--seed", type=int, default=SEED_4B)
    parser.add_argument("--pairs-p999", type=int, default=PAIRS_P999_PER_POLE)
    parser.add_argument("--pairs-floor6", type=int, default=PAIRS_FLOOR6_PER_POLE)
    parser.add_argument("--pairs-p97-99", type=int, default=PAIRS_P97_99_PER_POLE)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    build_snippets_4b(
        tokenizer=tokenizer,
        seed=args.seed,
        pairs_p999=args.pairs_p999,
        pairs_floor6=args.pairs_floor6,
        pairs_p97_99=args.pairs_p97_99,
    )


if __name__ == "__main__":
    main()
