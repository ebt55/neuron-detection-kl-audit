"""Symmetric snippet window cutting and evaluation pair construction for Qwen3-1.7B evaluation.

Enforces strict windowing, balance, and distractor invariants:
- Strict symmetric-window (arXiv:2610.37676) invariant: every firing and distractor token satisfies
  `60 <= pos_in_chunk <= chunk_len - 9`, producing 100% exact 69-token windows
  (60 before + marked <<target>> + 8 after, target_offset == 60, 0% truncation, 0% <bos>).
- Weak-band guard: For weak-band positives (`p75_90`, `p90_97`), every non-target token in the
  69-token window (`j != target`) must satisfy `d_j < p90`, matching the distractor rule.
  Both `pos` and `dist` record exact `window_max_act_dir`, `other_max_act_dir`,
  `window_has_ge_p97`, and `other_has_ge_p90`.
- Slot balance: Exact 3/3 (or even `k/k` 50/50) Slot A / Slot B balance within EVERY surviving
  `(pole_id, band, distractor_type)` cell (odd survival counts drop 1 pair).
- Same-layer neighbour arm: Neighbour distractor arm (`nearest_wgate_tier0`) is strictly Tier 0 ONLY
  (zero fallback tiers!): nearest same-layer `w_gate`-cosine neighbour firing in its top
  band (`|a_m| >= p99.9(|a_m|)` & `g_m > 0`) while target is in dead zone (`|a_n| < p25`)
  AND target's entire 69-token window stays `< p90`.
- Strict doc-level train/eval split (`doc_id % 2 == 0` for 20 train_label examples & top-3
  tokens; `doc_id % 2 == 1` for eval positives & distractors) + 16-token suffix dedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d

from w_audit.config import (
    BANDS,
    DEDUP_SUFFIX_LEN,
    LABEL_EXAMPLES_PER_POLE,
    MODEL_ID,
    NEIGHBOUR_BANDS,
    P90_COMPARE_BANDS,
    PAIRS_PER_CELL,
    SEED,
    WindowSpec,
    ensure_dirs,
)
from w_audit.extract import QUAD_CODE_TO_NAME

WEAK_BANDS: tuple[str, ...] = ("p75_90", "p90_97")
NEIGHBOUR_DISTRACTOR_TYPE: str = "nearest_wgate_tier0"


def _stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    """Deterministic per-pole/per-cell RNG independent of Python hash randomization."""
    key = ":".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed_int)


def balanced_cell_pos_slots(
    n_pairs: int,
    seed: int,
    pole_id: str,
    band: str,
    distractor_type: str,
) -> list[str]:
    """Return exact k/k 50/50 deterministic shuffle of Slot 'A' and 'B' for one cell."""
    if n_pairs < 0 or (n_pairs % 2) != 0:
        raise ValueError(f"n_pairs must be non-negative even integer, got {n_pairs}")
    if n_pairs == 0:
        return []
    half = n_pairs // 2
    slots = ["A"] * half + ["B"] * half
    rng = _stable_rng(seed, pole_id, band, distractor_type, "pos_slot_balanced")
    order = rng.permutation(n_pairs).tolist()
    return [slots[i] for i in order]


def compute_other_and_window_max_1d(
    arr: np.ndarray,
    wspec: WindowSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact rolling max over non-target window tokens and full 69-token window.

    For any interior index i (before <= i <= len(arr) - after - 1):
      - other_max[i] = max(arr[i - before : i], arr[i + 1 : i + after + 1])
      - win_max[i]   = max(arr[i - before : i + after + 1])
    Uses origin=0 SciPy maximum_filter1d with exact slice shifts (no origin-bound overflow).
    """
    d = arr.astype(np.float32, copy=False)
    n = len(d)
    before = wspec.before
    after = wspec.after

    pre_max = np.full(n, -np.inf, dtype=np.float32)
    suf_max = np.full(n, -np.inf, dtype=np.float32)

    if before > 0 and n > before:
        pre_all = maximum_filter1d(d, size=before, origin=0)
        # pre_all[c] covers [c - before//2, c - before//2 + before - 1]
        # To cover [i - before, i - 1], set c = i - before + before//2 = i - (before - before//2)
        shift_pre = before - (before // 2)
        pre_max[before:] = pre_all[before - shift_pre : n - shift_pre]

    if after > 0 and n > after:
        suf_all = maximum_filter1d(d, size=after, origin=0)
        # suf_all[c] covers [c - after//2, c - after//2 + after - 1]
        # To cover [i + 1, i + after], set c = i + 1 + after//2
        shift_suf = 1 + (after // 2)
        suf_max[: n - after] = suf_all[shift_suf : n - after + shift_suf]

    other_max = np.maximum(pre_max, suf_max)
    win_max = np.maximum(d, other_max)
    return other_max, win_max


def compute_window_max_1d(arr: np.ndarray, wspec: WindowSpec) -> np.ndarray:
    """Return exact rolling max over [i - wspec.before, i + wspec.after] at every index i."""
    _, win_max = compute_other_and_window_max_1d(arr, wspec)
    return win_max


def render_marked_window(
    token_ids: np.ndarray,
    flat_idx: int,
    tokenizer: Any,
    wspec: WindowSpec,
) -> tuple[str, str, int]:
    """Decode symmetric window [idx - before, idx + after] with <<target>> marking.

    Returns (marked_text, target_token_str, extra_marker_count).
    """
    prefix_ids = token_ids[flat_idx - wspec.before : flat_idx].tolist()
    target_id = int(token_ids[flat_idx])
    suffix_ids = token_ids[flat_idx + 1 : flat_idx + wspec.after + 1].tolist()

    if len(prefix_ids) != wspec.before or len(suffix_ids) != wspec.after:
        raise AssertionError(
            f"Asymmetric window at flat_idx={flat_idx}: "
            f"prefix={len(prefix_ids)} (want {wspec.before}), "
            f"suffix={len(suffix_ids)} (want {wspec.after})"
        )

    prefix_text = tokenizer.decode(
        prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    target_text = tokenizer.decode(
        [target_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    suffix_text = tokenizer.decode(
        suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    extra_marker_count = (
        prefix_text.count(wspec.marker_open)
        + target_text.count(wspec.marker_open)
        + suffix_text.count(wspec.marker_open)
        + prefix_text.count(wspec.marker_close)
        + target_text.count(wspec.marker_close)
        + suffix_text.count(wspec.marker_close)
    )
    marked_text = f"{prefix_text}{wspec.marker_open}{target_text}{wspec.marker_close}{suffix_text}"
    return marked_text, target_text, int(extra_marker_count)


def _suffix_key(
    token_ids: np.ndarray,
    flat_idx: int,
    suffix_len: int = DEDUP_SUFFIX_LEN,
) -> tuple[int, ...]:
    """Return exact 16-token suffix ending at flat_idx (tokens[i-15 : i+1])."""
    start = flat_idx - suffix_len + 1
    if start < 0:
        raise ValueError(f"flat_idx={flat_idx} too small for suffix_len={suffix_len}")
    return tuple(int(x) for x in token_ids[start : flat_idx + 1].tolist())


def _sample_distinct_doc_indices(
    candidate_indices: np.ndarray,
    doc_ids: np.ndarray,
    chunk_ids: np.ndarray,
    pos_in_chunk: np.ndarray,
    n_want: int,
    rng: np.random.Generator,
    min_spacing: int = 69,
    excluded_docs: set[int] | None = None,
    prefer_max_score: np.ndarray | None = None,
    token_ids: np.ndarray | None = None,
    forbidden_suffixes: set[tuple[int, ...]] | None = None,
    dedup_within_sample: bool = False,
    suffix_len: int = DEDUP_SUFFIX_LEN,
    allow_same_doc_fallback: bool = True,
) -> list[int]:
    """Sample up to n_want flat indices preferring distinct documents."""
    if candidate_indices.size == 0 or n_want <= 0:
        return []

    excluded = excluded_docs or set()
    forbid_suf = forbidden_suffixes or set()
    used_suffixes: set[tuple[int, ...]] = set()

    by_doc: dict[int, list[int]] = defaultdict(list)
    for raw_idx in candidate_indices.tolist():
        idx = int(raw_idx)
        did = int(doc_ids[idx])
        if did in excluded:
            continue
        if token_ids is not None and forbid_suf:
            if _suffix_key(token_ids, idx, suffix_len) in forbid_suf:
                continue
        by_doc[did].append(idx)

    if not by_doc:
        return []

    chosen: list[int] = []
    chosen_chunks: dict[int, list[int]] = defaultdict(list)

    # Pass 1: At most 1 token per distinct document
    doc_list = list(by_doc.keys())
    rng.shuffle(doc_list)
    for did in doc_list:
        if len(chosen) >= n_want:
            break
        idxs = by_doc[did]
        if prefer_max_score is not None:
            idxs = sorted(idxs, key=lambda i: float(prefer_max_score[i]), reverse=True)
        else:
            rng.shuffle(idxs)
        for idx in idxs:
            if dedup_within_sample and token_ids is not None:
                sk = _suffix_key(token_ids, idx, suffix_len)
                if sk in used_suffixes:
                    continue
                used_suffixes.add(sk)
            chosen.append(idx)
            chosen_chunks[int(chunk_ids[idx])].append(int(pos_in_chunk[idx]))
            break

    if len(chosen) >= n_want or not allow_same_doc_fallback:
        return chosen[:n_want]

    # Pass 2: non-overlapping windows >= min_spacing
    chosen_set = set(chosen)
    flat_pool: list[int] = []
    for did in doc_list:
        for idx in by_doc[did]:
            if idx not in chosen_set:
                flat_pool.append(idx)
    rng.shuffle(flat_pool)

    for idx in flat_pool:
        if len(chosen) >= n_want:
            break
        cid = int(chunk_ids[idx])
        pos = int(pos_in_chunk[idx])
        if any(abs(pos - p) < min_spacing for p in chosen_chunks[cid]):
            continue
        if dedup_within_sample and token_ids is not None:
            sk = _suffix_key(token_ids, idx, suffix_len)
            if sk in used_suffixes:
                continue
            used_suffixes.add(sk)
        chosen.append(idx)
        chosen_set.add(idx)
        chosen_chunks[cid].append(pos)

    if len(chosen) >= n_want:
        return chosen[:n_want]

    # Pass 3 (last-resort fallback for tiny synthetic corpora)
    for idx in flat_pool:
        if len(chosen) >= n_want:
            break
        if idx in chosen_set:
            continue
        chosen.append(idx)
        chosen_set.add(idx)

    return chosen[:n_want]


def _select_strict_tier0_pairs(
    tier0_mask: np.ndarray,
    pos_indices: Sequence[int],
    doc_ids: np.ndarray,
    chunk_ids: np.ndarray,
    pos_in_chunk: np.ndarray,
    rng: np.random.Generator,
    used_docs_across_bands: set[int] | None = None,
    min_spacing: int = 69,
    allow_same_doc_fallback: bool = False,
) -> list[tuple[int, int]]:
    """Pair an even number of positives with strictly Tier-0 distractors (cell-balanced).

    - ZERO fallback tiers: only candidates where tier0_mask is True are eligible.
    - Guarantees dist.doc_id != pos.doc_id for every pair and distinct distractor docs within
      the cell (preferring unused docs across bands first).
    - Always returns an EVEN list of `(pos_idx_position_in_pos_indices, dist_flat_idx)` pairs
      so every surviving cell has an exact k/k 50/50 Slot A / Slot B split (dropping 1 pair if
      the surviving count is odd).
    """
    idxs = np.flatnonzero(tier0_mask)
    if idxs.size == 0 or len(pos_indices) < 2:
        return []

    global_used = used_docs_across_bands if used_docs_across_bands is not None else set()
    n_pos = len(pos_indices)
    pos_docs = [int(doc_ids[i]) for i in pos_indices]

    matched: list[int | None] = [None] * n_pos
    cell_used_docs: set[int] = set()
    cell_used_indices: set[int] = set()

    for allow_global_reuse in (False, True):
        for pair_i in range(n_pos):
            if matched[pair_i] is not None:
                continue
            pdoc = pos_docs[pair_i]
            excl = {pdoc} | cell_used_docs
            if not allow_global_reuse:
                excl = excl | global_used

            picked_idx: int | None = None
            # Fast O(1) uniform corpus-position probe for large pools
            if idxs.size > 512:
                probes = rng.integers(0, idxs.size, size=256)
                for p in probes.tolist():
                    cand = int(idxs[p])
                    if cand not in cell_used_indices and int(doc_ids[cand]) not in excl:
                        picked_idx = cand
                        break
            if picked_idx is None:
                picked = _sample_distinct_doc_indices(
                    candidate_indices=idxs,
                    doc_ids=doc_ids,
                    chunk_ids=chunk_ids,
                    pos_in_chunk=pos_in_chunk,
                    n_want=1,
                    rng=rng,
                    min_spacing=min_spacing,
                    excluded_docs=excl,
                    allow_same_doc_fallback=allow_same_doc_fallback,
                )
                if picked and picked[0] not in cell_used_indices:
                    picked_idx = picked[0]
            if picked_idx is not None:
                matched[pair_i] = picked_idx
                d_doc = int(doc_ids[picked_idx])
                cell_used_docs.add(d_doc)
                cell_used_indices.add(picked_idx)
                global_used.add(d_doc)

    # Optional same-doc non-overlapping fallback ONLY when requested (tiny synthetic pools);
    # strictly preserves dist.doc_id != pos.doc_id and Tier-0 mask.
    if allow_same_doc_fallback and any(m is None for m in matched):
        for pair_i in range(n_pos):
            if matched[pair_i] is not None:
                continue
            pdoc = pos_docs[pair_i]
            valid = [
                int(i)
                for i in idxs.tolist()
                if int(doc_ids[i]) != pdoc and int(i) not in cell_used_indices
            ]
            if valid:
                rng.shuffle(valid)
                d_idx = valid[0]
                matched[pair_i] = d_idx
                cell_used_indices.add(d_idx)

    paired = [(i, d_idx) for i, d_idx in enumerate(matched) if d_idx is not None]
    # Enforce strictly even pair count per cell (drop last pair if odd)
    even_len = 2 * (len(paired) // 2)
    return paired[:even_len]


def _compute_top_tokens(
    label_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[int], list[str], list[int]]:
    """Compute top-3 and top-10 trigger tokens strictly from the 20 train_label examples.

    Ranking rule:
      1. Highest frequency across the 20 train_label p99.9_plus snippets.
      2. Tie-break: highest mean act_dir.
      3. Tie-break: lowest token ID.
    """
    counts: dict[int, int] = defaultdict(int)
    act_sums: dict[int, float] = defaultdict(float)
    for row in label_rows:
        tid = int(row["target_token_id"])
        counts[tid] += 1
        act_sums[tid] += float(row["act_dir"])

    ranked = sorted(
        counts.keys(),
        key=lambda tid: (-counts[tid], -(act_sums[tid] / counts[tid]), tid),
    )
    top_3_ids = [int(tid) for tid in ranked[:3]]
    top_3_tokens = [
        tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for tid in top_3_ids
    ]
    top_10_ids = [int(tid) for tid in ranked[:10]]
    return top_3_ids, top_3_tokens, top_10_ids


def build_snippets(
    tokenizer: Any,
    base_dir: Path | None = None,
    seed: int = SEED,
    label_examples_per_pole: int = LABEL_EXAMPLES_PER_POLE,
    pairs_per_cell: int = PAIRS_PER_CELL,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Build data/label_examples.jsonl and data/snippets.jsonl and update data/poles.json top tokens."""
    if pairs_per_cell % 2 != 0:
        raise ValueError(f"pairs_per_cell must be even for exact k/k A/B balance, got {pairs_per_cell}")

    t0 = time.time()
    dirs = ensure_dirs(base_dir)
    cache_dir = dirs["cache"]
    poles_path = dirs["data"] / "poles.json"
    label_examples_path = dirs["data"] / "label_examples.jsonl"
    snippets_path = dirs["data"] / "snippets.jsonl"

    poles_doc = json.loads(poles_path.read_text(encoding="utf-8"))
    cache_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))

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

    is_train = (doc_ids % 2) == 0
    is_eval = ~is_train
    train_interior = interior_mask & is_train
    eval_interior = interior_mask & is_eval

    dedup_len = min(DEDUP_SUFFIX_LEN, wspec.before + 1)

    all_label_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    surviving_by_cell_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    neighbor_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _get_neighbor_arrays(layer: int, nb_neuron: int) -> tuple[np.ndarray, np.ndarray]:
        key = (layer, nb_neuron)
        if key not in neighbor_cache:
            nb_col = neuron_to_col[key]
            abs_m = np.abs(acts[:, nb_col].astype(np.float32))
            gate_open_m = quads[:, nb_col] < 2
            neighbor_cache[key] = (abs_m, gate_open_m)
        return neighbor_cache[key]

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
        neighbor_act_val: float | None = None,
        neighbor_gate_cosine: float | None = None,
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
        }
        if is_dist:
            obj["neighbor_act_val"] = (
                float(neighbor_act_val) if neighbor_act_val is not None else None
            )
            obj["neighbor_gate_cosine"] = (
                float(neighbor_gate_cosine) if neighbor_gate_cosine is not None else None
            )
        obj["extra_marker_count"] = int(extra_markers)
        obj["marked_text"] = marked_text
        return obj

    for pole in poles_doc["poles"]:
        pole_id = str(pole["pole_id"])
        layer = int(pole["layer"])
        neuron = int(pole["neuron"])
        sign = int(pole["sign"])
        p25_abs = float(pole["p25_abs"])
        edges = pole["band_edges_dir"]
        p75 = float(edges["p75"])
        p90 = float(edges["p90"])
        p97 = float(edges["p97"])
        p99_9 = float(edges["p99.9"])

        col = neuron_to_col[(layer, neuron)]
        a_n = acts[:, col].astype(np.float32)
        q_n = quads[:, col]
        d_n = float(sign) * a_n
        abs_n = np.abs(a_n)

        # Exact rolling max over non-target 68 tokens AND full 69-token window
        other_max_dir, win_max_dir = compute_other_and_window_max_1d(d_n, wspec)

        # ------------------------------------------------------------------
        # 1. Select 20 disjoint train_label p99.9_plus examples (§2.3, §4.2)
        #    (Deterministic — matches existing data/labels.jsonl 100% verbatim!)
        # ------------------------------------------------------------------
        train_top_mask = train_interior & (d_n >= p99_9) & (d_n > 0.0) & (abs_n >= p25_abs)
        train_top_candidates = np.flatnonzero(train_top_mask)
        rng_train = _stable_rng(seed, pole_id, "train_label")
        train_selected = _sample_distinct_doc_indices(
            candidate_indices=train_top_candidates,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            n_want=label_examples_per_pole,
            rng=rng_train,
            min_spacing=wspec.total,
            prefer_max_score=d_n,
            token_ids=token_ids,
            dedup_within_sample=True,
            suffix_len=dedup_len,
        )
        if len(train_selected) < label_examples_per_pole:
            raise RuntimeError(
                f"Pole {pole_id}: only found {len(train_selected)}/{label_examples_per_pole} "
                f"train_label p99.9_plus examples."
            )

        train_selected.sort(key=lambda idx: float(d_n[idx]), reverse=True)
        train_suffixes: set[tuple[int, ...]] = set()
        pole_label_rows: list[dict[str, Any]] = []

        for ex_idx, flat_idx in enumerate(train_selected):
            train_suffixes.add(_suffix_key(token_ids, flat_idx, dedup_len))
            marked_text, target_tok_str, _ = render_marked_window(
                token_ids, flat_idx, tokenizer, wspec
            )
            q_code = int(q_n[flat_idx])
            row = {
                "pole_id": pole_id,
                "example_idx": ex_idx,
                "doc_id": int(doc_ids[flat_idx]),
                "chunk_id": int(chunk_ids[flat_idx]),
                "pos_in_chunk": int(pos_in_chunk[flat_idx]),
                "target_token_id": int(token_ids[flat_idx]),
                "target_token_str": target_tok_str,
                "act_val": float(a_n[flat_idx]),
                "act_dir": float(d_n[flat_idx]),
                "gate_open": bool(q_code < 2),
                "quadrant": QUAD_CODE_TO_NAME[q_code],
                "marked_text": marked_text,
            }
            pole_label_rows.append(row)
            all_label_rows.append(row)

        top_3_ids, top_3_tokens, top_10_ids = _compute_top_tokens(pole_label_rows, tokenizer)
        pole["top_3_token_ids"] = top_3_ids
        pole["top_3_tokens"] = top_3_tokens
        pole["top_10_token_ids"] = top_10_ids

        # ------------------------------------------------------------------
        # 2. Sample up to 6 held-out eval positives per band (weak-band guard)
        # ------------------------------------------------------------------
        band_bounds: dict[str, tuple[float, float | None]] = {
            "p75_90": (p75, p90),
            "p90_97": (p90, p97),
            "p97_99.9": (p97, p99_9),
            "p99.9_plus": (p99_9, None),
        }
        eval_guard = eval_interior & (d_n > 0.0) & (abs_n >= p25_abs)
        eval_positives_by_band: dict[str, list[int]] = {}
        eval_pos_objs_by_band: dict[str, list[dict[str, Any]]] = {}

        for band in BANDS:
            lo, hi = band_bounds[band]
            bmask = eval_guard & (d_n >= lo)
            if hi is not None:
                bmask = bmask & (d_n < hi)
            # Weak-band positives require ALL other 68 tokens in window < p90
            if band in WEAK_BANDS:
                bmask = bmask & (other_max_dir < p90)

            cands = np.flatnonzero(bmask)
            rng_pos = _stable_rng(seed, pole_id, "eval_pos", band)
            picked = _sample_distinct_doc_indices(
                candidate_indices=cands,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=pairs_per_cell,
                rng=rng_pos,
                min_spacing=wspec.total,
                token_ids=token_ids,
                forbidden_suffixes=train_suffixes,
                suffix_len=dedup_len,
            )
            # Keep even count <= pairs_per_cell (even slot balance)
            even_pos_count = 2 * (len(picked) // 2)
            picked = picked[:even_pos_count]
            eval_positives_by_band[band] = picked
            eval_pos_objs_by_band[band] = [
                _build_snippet_obj(
                    flat_idx=idx,
                    act_val=float(a_n[idx]),
                    act_dir=float(d_n[idx]),
                    quad_code=int(q_n[idx]),
                    window_max_act_dir=float(win_max_dir[idx]),
                    other_max_act_dir=float(other_max_dir[idx]),
                    p90_dir=p90,
                    p97_dir=p97,
                    is_dist=False,
                )
                for idx in picked
            ]

        # ------------------------------------------------------------------
        # 3. Pair with strict Tier-0 dead_zone distractors across all 4 bands
        # ------------------------------------------------------------------
        in_dead_zone = eval_interior & (abs_n < p25_abs)
        dz_tier0_mask = in_dead_zone & (win_max_dir < p90)
        dz_used_docs: set[int] = set()
        for band in BANDS:
            rng_dz = _stable_rng(seed, pole_id, "dist_dead_zone", band)
            paired_dz = _select_strict_tier0_pairs(
                tier0_mask=dz_tier0_mask,
                pos_indices=eval_positives_by_band[band],
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                rng=rng_dz,
                used_docs_across_bands=dz_used_docs,
                min_spacing=wspec.total,
            )
            slots = balanced_cell_pos_slots(len(paired_dz), seed, pole_id, band, "dead_zone")
            surviving_by_cell_type["dead_zone"][band] += len(paired_dz)
            for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired_dz, slots)):
                pair_id = f"{pole_id}__{band}__dead_zone__{pair_idx}"
                all_pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "pole_id": pole_id,
                        "layer": layer,
                        "neuron": neuron,
                        "sign": sign,
                        "band": band,
                        "distractor_type": "dead_zone",
                        "distractor_tier": 0,
                        "pair_idx": pair_idx,
                        "pos_slot": slot,
                        "pos": eval_pos_objs_by_band[band][pos_k],
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
                            neighbor_act_val=None,
                            neighbor_gate_cosine=None,
                        ),
                    }
                )

        # ------------------------------------------------------------------
        # 4. Pair with strict Tier-0 goodfire_lt_p90 distractors (p99.9_plus)
        # ------------------------------------------------------------------
        gf_tier0_mask = eval_interior & (win_max_dir < p90)
        for band in P90_COMPARE_BANDS:
            rng_gf = _stable_rng(seed, pole_id, "dist_goodfire_lt_p90", band)
            paired_gf = _select_strict_tier0_pairs(
                tier0_mask=gf_tier0_mask,
                pos_indices=eval_positives_by_band[band],
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                rng=rng_gf,
                min_spacing=wspec.total,
            )
            slots = balanced_cell_pos_slots(len(paired_gf), seed, pole_id, band, "goodfire_lt_p90")
            surviving_by_cell_type["goodfire_lt_p90"][band] += len(paired_gf)
            for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired_gf, slots)):
                pair_id = f"{pole_id}__{band}__goodfire_lt_p90__{pair_idx}"
                all_pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "pole_id": pole_id,
                        "layer": layer,
                        "neuron": neuron,
                        "sign": sign,
                        "band": band,
                        "distractor_type": "goodfire_lt_p90",
                        "distractor_tier": 0,
                        "pair_idx": pair_idx,
                        "pos_slot": slot,
                        "pos": eval_pos_objs_by_band[band][pos_k],
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
                            neighbor_act_val=None,
                            neighbor_gate_cosine=None,
                        ),
                    }
                )

        # ------------------------------------------------------------------
        # 5. Pair with strictly Tier-0 ONLY nearest_wgate_tier0 distractors (Tier-0 only)
        #    (ZERO fallback tiers 1-5; exact Tier-0 filter; exact even k/k A/B balance)
        # ------------------------------------------------------------------
        nb_neuron = int(pole["neighbor_neuron"])
        nb_col = neuron_to_col[(layer, nb_neuron)]
        nb_gate_cos = float(pole["neighbor_gate_cosine"])
        abs_m, gate_open_m = _get_neighbor_arrays(layer, nb_neuron)
        q999_m = float(pole["neighbor_abs_quantiles"]["p99.9_abs"])

        nb_tier0_mask = (
            in_dead_zone
            & gate_open_m
            & (abs_m >= q999_m)
            & (win_max_dir < p90)
        )
        nb_used_docs: set[int] = set()
        for band in NEIGHBOUR_BANDS:
            rng_nb = _stable_rng(seed, pole_id, f"dist_{NEIGHBOUR_DISTRACTOR_TYPE}", band)
            paired_nb = _select_strict_tier0_pairs(
                tier0_mask=nb_tier0_mask,
                pos_indices=eval_positives_by_band[band],
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                rng=rng_nb,
                used_docs_across_bands=nb_used_docs,
                min_spacing=wspec.total,
                allow_same_doc_fallback=False,
            )
            slots = balanced_cell_pos_slots(
                len(paired_nb), seed, pole_id, band, NEIGHBOUR_DISTRACTOR_TYPE
            )
            surviving_by_cell_type[NEIGHBOUR_DISTRACTOR_TYPE][band] += len(paired_nb)
            for pair_idx, ((pos_k, d_idx), slot) in enumerate(zip(paired_nb, slots)):
                pair_id = f"{pole_id}__{band}__{NEIGHBOUR_DISTRACTOR_TYPE}__{pair_idx}"
                all_pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "pole_id": pole_id,
                        "layer": layer,
                        "neuron": neuron,
                        "sign": sign,
                        "band": band,
                        "distractor_type": NEIGHBOUR_DISTRACTOR_TYPE,
                        "distractor_tier": 0,
                        "pair_idx": pair_idx,
                        "pos_slot": slot,
                        "pos": eval_pos_objs_by_band[band][pos_k],
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
                            neighbor_act_val=float(acts[d_idx, nb_col]),
                            neighbor_gate_cosine=nb_gate_cos,
                        ),
                    }
                )

    poles_path.write_text(json.dumps(poles_doc, indent=2), encoding="utf-8")

    with label_examples_path.open("w", encoding="utf-8") as f:
        for row in all_label_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with snippets_path.open("w", encoding="utf-8") as f:
        for row in all_pair_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_poles": len(poles_doc["poles"]),
        "num_label_examples": len(all_label_rows),
        "num_snippet_pairs": len(all_pair_rows),
        "surviving_pairs_by_distractor_band": {
            dt: dict(sorted(bmap.items())) for dt, bmap in sorted(surviving_by_cell_type.items())
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"[snippets] Done: {json.dumps(summary)}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cut symmetric snippets and build evaluation pairs.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--label-examples", type=int, default=LABEL_EXAMPLES_PER_POLE)
    parser.add_argument("--pairs-per-cell", type=int, default=PAIRS_PER_CELL)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    build_snippets(
        tokenizer=tokenizer,
        seed=args.seed,
        label_examples_per_pole=args.label_examples,
        pairs_per_cell=args.pairs_per_cell,
    )


if __name__ == "__main__":
    main()
