"""Causal simulation patching & band KL evaluation on Qwen/Qwen3-1.7B (50 neurons / 100 poles) in exact IEEE-754 FP32.

Three clean stages:
1. `prepare_17b_inputs` (Step 3A, CPU):
   - Computes exact centered L2 (and L1) `EnergyShare` across 4-band (`p99.9+`, `p97_99.9`,
     `p75_97`, `lt_p75`), 5-band matching 1.7B's evaluated detection bands (`p99.9_plus`,
     `p97_99.9`, `p90_97`, `p75_90`, `lt_p75`), amplitude (`>=20% max`, `>=50% max`), and
     SwiGLU gate (`Q1+Q2` open vs `Q3+Q4` closed) from `data/cache/acts.npy` + `quads.npy`.
   - Builds shared `128 x 64 = 8,192`-token ordinary eval set (`e3_ordinary_17b.npz`) strictly
     from odd held-out docs (`doc_id % 2 == 1`).
   - Builds `data/qwen17b_causal/sim_windows_17b.jsonl` (`50 x 64 = 3,200` 64-token windows:
     `32` random ordinary + `32` active-stratified `max|a| >= 20% max`, exact Horvitz-Thompson
     stratum weights matching Ulto85 `evalsets.py`, zero train-doc overlap).
   - Builds `data/qwen17b_causal/fpr_probes_17b.jsonl` (`100` poles x `24` 69-token probes:
     `10+2` `core` for `terra@none`, plus `10+2` `luna_extra` for `luna@none`, strictly odd
     held-out docs, `0` `<bos>`, `0` dup docs within pole).
2. `run_17b_baseline_kl` (Step 3B, RTX 3080 exact IEEE-754 `float32`, `allow_tf32=False`):
   - 2-half IEEE-754 `float32` forward pass (`< 4.5 GB` peak VRAM, `zero_ablation_sanity_kl == 0.0`).
   - Measures `zero_kl`, `mean_kl`, `strong20_kl` (`strong20_rec`), `strong50_kl` (`strong50_rec`),
     4-band mean-ablation KL (`p99.9+`, `p97_99.9`, `p75_97`, `lt_p75`), and SwiGLU gate KL
     (`gate_open`, `gate_closed`) on all 50 neurons over `e3_ordinary_17b.npz`.
   - Writes `results/qwen17b_causal/energy_and_kl_17b.json`.
3. `run_17b_sim_patching` (Step 3C, RTX 3080 exact IEEE-754 `float32`, `allow_tf32=False`):
   - Loads simulator predictions from `results/qwen17b_causal/sim_calls_17b.jsonl`.
   - Fits BOTH Ulto85's exact 2-fold even/odd cross-fit linear calibration (`sim_cal_cv`, primary)
     AND in-sample linear calibration (`sim_cal_insample`) on non-BOS tokens.
   - Patches `mean`, `zero`, `strong20`, `strong50`, `sim_raw`, `sim_cal_cv`, `sim_cal_insample`
     in exact IEEE-754 FP32 on each neuron's `64 x 64` windows and computes exact `sim_cal_rec`,
     `sim_cal_rec_insample`, `sim_cal_rec_random`, `sim_cal_rec_active`, `sim_raw_rec`, and `corr`.
   - Writes `results/qwen17b_causal/causal_scores_17b.json`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen17b_causal import (
    E3_ORDINARY_N_WINS_17B,
    E3_ORDINARY_WIN_LEN_17B,
    FPR_ORD_CORE_17B,
    FPR_ORD_EXTRA_17B,
    FPR_TOP_CORE_17B,
    FPR_TOP_EXTRA_17B,
    LAYERS_17B,
    MODEL_ID_17B,
    SEED_17B,
    SIM_ACT_WINS_17B,
    SIM_N_WINS_PER_NEURON_17B,
    SIM_ORD_WINS_17B,
    SIM_WIN_LEN_17B,
    ensure_17b_causal_dirs,
)
from qwen4b_eval.band_kl_4b import _compute_chunked_kl_sum
from qwen4b_eval.extract_4b import (
    ParityGateError,
    _call_decoder_layer,
    _clear_cuda,
    _prepare_masks_and_rope,
    load_qwen4b_backbone_cpu,
)
from qwen4b_eval.gap_and_bands_4b import bootstrap_corr_stats
from w_audit.config import DEDUP_SUFFIX_LEN, WindowSpec
from w_audit.extract import QUAD_CODE_TO_NAME, encode_quadrant_torch
from w_audit.snippets import (
    _sample_distinct_doc_indices,
    compute_other_and_window_max_1d,
    render_marked_window,
)

BASELINE_KL_CONDS_17B: tuple[str, ...] = (
    "zero",
    "mean",
    "strong20",
    "strong50",
    "p99.9+",
    "p97_99.9",
    "p75_97",
    "lt_p75",
    "gate_open",
    "gate_closed",
)
FOUR_BAND_17B: tuple[str, ...] = ("p99.9+", "p97_99.9", "p75_97", "lt_p75")


def _stable_rng_17b(seed: int, *parts: Any) -> np.random.Generator:
    key = ":".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def make_neuron_id_17b(layer: int, neuron: int) -> str:
    return f"L{int(layer):02d}_N{int(neuron):04d}"


def _compute_two_half_ranges(n_layers: int) -> list[tuple[int, int]]:
    if n_layers <= 1:
        return [(0, n_layers)]
    mid = n_layers // 2
    return [(0, mid), (mid, n_layers)]


# ---------------------------------------------------------------------------
# Calibration Helpers (Exact Ulto85 2-Fold Even/Odd CV + In-Sample)
# ---------------------------------------------------------------------------


def calibrate_ratings_2fold_cv(
    ratings: np.ndarray,
    true_acts: np.ndarray,
    exclude_pos0: bool = True,
) -> np.ndarray:
    """Exact Ulto85 2-fold cross-fit linear calibration (fit even windows -> odd, odd -> even).

    `ratings` and `true_acts` have shape `[W, T]`. Fits OLS `a_hat = alpha * r + beta` on
    non-BOS positions (`pos >= 1` if `exclude_pos0`) of each half and applies to the other half.
    """
    r = np.asarray(ratings, dtype=np.float64)
    a = np.asarray(true_acts, dtype=np.float64)
    w, t = r.shape
    out = np.zeros_like(a, dtype=np.float64)
    par = (np.arange(w) % 2) == 0
    col_slice = slice(1, t) if (exclude_pos0 and t > 1) else slice(0, t)

    for tr in (par, ~par):
        te = ~tr
        if not np.any(tr) or not np.any(te):
            continue
        r_tr = r[tr, col_slice].ravel()
        a_tr = a[tr, col_slice].ravel()
        Z = np.stack([r_tr, np.ones_like(r_tr)], axis=1)
        coef, *_ = np.linalg.lstsq(Z, a_tr, rcond=None)
        out[te] = coef[0] * r[te] + coef[1]
    return out


def calibrate_ratings_insample(
    ratings: np.ndarray,
    true_acts: np.ndarray,
    exclude_pos0: bool = True,
) -> tuple[np.ndarray, float, float]:
    """In-sample OLS linear calibration `a_hat = alpha * r + beta` across all W windows."""
    r = np.asarray(ratings, dtype=np.float64)
    a = np.asarray(true_acts, dtype=np.float64)
    t = r.shape[1]
    col_slice = slice(1, t) if (exclude_pos0 and t > 1) else slice(0, t)
    r_tr = r[:, col_slice].ravel()
    a_tr = a[:, col_slice].ravel()
    Z = np.stack([r_tr, np.ones_like(r_tr)], axis=1)
    coef, *_ = np.linalg.lstsq(Z, a_tr, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    return alpha * r + beta, alpha, beta


# ---------------------------------------------------------------------------
# Step 3A: Corpus Energy Shares + Disjoint Eval Windows + FPR Probes (CPU)
# ---------------------------------------------------------------------------


def prepare_17b_inputs(
    tokenizer: Any,
    base_dir: Path | None = None,
    seed: int = SEED_17B,
    e3_n_wins: int = E3_ORDINARY_N_WINS_17B,
    e3_win_len: int = E3_ORDINARY_WIN_LEN_17B,
    sim_ord_wins: int = SIM_ORD_WINS_17B,
    sim_act_wins: int = SIM_ACT_WINS_17B,
    sim_win_len: int = SIM_WIN_LEN_17B,
    fpr_ord_core: int = FPR_ORD_CORE_17B,
    fpr_ord_extra: int = FPR_ORD_EXTRA_17B,
    fpr_top_core: int = FPR_TOP_CORE_17B,
    fpr_top_extra: int = FPR_TOP_EXTRA_17B,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Run Step 3A: compute 1.7B corpus energy shares and build disjoint eval & FPR datasets."""
    t0 = time.time()
    dirs = ensure_17b_causal_dirs(base_dir)
    data_root = dirs["data_root"]
    cache_root = dirs["cache_root"]
    out_data = dirs["data"]
    out_res = dirs["results"]

    poles_doc = json.loads((data_root / "poles.json").read_text(encoding="utf-8"))
    cache_meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))

    labels_by_pole: dict[str, dict[str, Any]] = {}
    labels_path = data_root / "labels.jsonl"
    if labels_path.is_file():
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    labels_by_pole[str(row["pole_id"])] = row

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

    acts = np.load(cache_root / "acts.npy", mmap_mode="r")
    quads = np.load(cache_root / "quads.npy", mmap_mode="r")
    token_ids = np.load(cache_root / "token_ids.npy", mmap_mode="r")
    doc_ids = np.load(cache_root / "doc_ids.npy", mmap_mode="r")
    chunk_ids = np.load(cache_root / "chunk_ids.npy", mmap_mode="r")
    pos_in_chunk = np.load(cache_root / "pos_in_chunk.npy", mmap_mode="r")
    chunk_starts = np.load(cache_root / "chunk_starts.npy", mmap_mode="r")
    chunk_lengths = np.load(cache_root / "chunk_lengths.npy", mmap_mode="r")

    col_to_neuron = [tuple(x) for x in cache_meta["col_to_neuron"]]
    neuron_to_col = {(int(L), int(n)): col for col, (L, n) in enumerate(col_to_neuron)}
    special_ids = set(int(x) for x in cache_meta.get("special_token_ids", []))

    chunk_len_at_pos = chunk_lengths[chunk_ids]
    non_bos = pos_in_chunk > 0
    if special_ids:
        non_bos = non_bos & (~np.isin(token_ids, list(special_ids)))
    is_eval_odd = (doc_ids % 2) == 1
    eval_interior = (
        non_bos
        & is_eval_odd
        & (pos_in_chunk >= wspec.before)
        & (pos_in_chunk <= (chunk_len_at_pos - wspec.after - 1))
    )
    n_non_bos = int(np.sum(non_bos))

    # Group 100 poles into 50 canonical neurons (sorted by (layer, neuron))
    poles_by_ln: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for p in poles_doc["poles"]:
        poles_by_ln[(int(p["layer"]), int(p["neuron"]))][int(p["sign"])] = dict(p)

    sorted_lns = sorted(poles_by_ln.keys())

    # ---- 1. Build odd-doc non-overlapping 64-token candidate windows (zero train overlap!) ----
    odd_win_starts: list[int] = []
    odd_win_doc_ids: list[int] = []
    odd_win_chunk_ids: list[int] = []
    odd_win_local_offsets: list[int] = []
    odd_chunk_first_win: dict[int, int] = {}

    n_chunks = int(len(chunk_starts))
    for cid in range(n_chunks):
        c_start = int(chunk_starts[cid])
        c_len = int(chunk_lengths[cid])
        did = int(doc_ids[c_start])
        if (did % 2) != 1 or c_len < sim_win_len:
            continue
        for off in range(0, c_len - sim_win_len + 1, sim_win_len):
            w_idx = len(odd_win_starts)
            if off == 0:
                odd_chunk_first_win[cid] = w_idx
            odd_win_starts.append(c_start + off)
            odd_win_doc_ids.append(did)
            odd_win_chunk_ids.append(cid)
            odd_win_local_offsets.append(off)

    odd_win_starts_arr = np.asarray(odd_win_starts, dtype=np.int64)
    odd_win_docs_arr = np.asarray(odd_win_doc_ids, dtype=np.int32)
    odd_win_cids_arr = np.asarray(odd_win_chunk_ids, dtype=np.int32)
    odd_win_offs_arr = np.asarray(odd_win_local_offsets, dtype=np.int32)
    n_odd_wins = len(odd_win_starts)

    # ---- 2. Select shared `e3_ordinary` set (`128 x 64 = 8,192` tok, distinct odd docs!) ----
    if e3_win_len == sim_win_len:
        # Pick 128 windows strictly from 128 distinct odd docs (prefer chunk start off=0)
        by_odd_doc: dict[int, list[int]] = defaultdict(list)
        for wi, did in enumerate(odd_win_doc_ids):
            if odd_win_local_offsets[wi] == 0:
                by_odd_doc[did].append(wi)
        rng_e3 = _stable_rng_17b(seed, "e3_ordinary_shared")
        odd_docs_sorted = sorted(by_odd_doc.keys())
        rng_e3.shuffle(odd_docs_sorted)
        e3_picked_w: list[int] = []
        for did in odd_docs_sorted:
            if len(e3_picked_w) >= e3_n_wins:
                break
            e3_picked_w.append(by_odd_doc[did][0])
        if len(e3_picked_w) < e3_n_wins:
            rem_ws = [wi for wi in range(n_odd_wins) if wi not in set(e3_picked_w)]
            rng_e3.shuffle(rem_ws)
            e3_picked_w.extend(rem_ws[: e3_n_wins - len(e3_picked_w)])

        e3_tok_ids = np.stack(
            [token_ids[odd_win_starts_arr[wi] : odd_win_starts_arr[wi] + e3_win_len] for wi in e3_picked_w],
            axis=0,
        ).astype(np.int32)
        e3_docs = np.array([odd_win_docs_arr[wi] for wi in e3_picked_w], dtype=np.int32)
        e3_cids = np.array([odd_win_cids_arr[wi] for wi in e3_picked_w], dtype=np.int32)
        e3_offs = np.array([odd_win_offs_arr[wi] for wi in e3_picked_w], dtype=np.int32)
    else:
        raise ValueError(f"Expected e3_win_len == sim_win_len ({sim_win_len}), got {e3_win_len}")

    assert np.all((e3_docs % 2) == 1), "Even train doc leaked into e3_ordinary!"

    # ---- 3. Compute exact corpus EnergyShare & select 64 sim windows per neuron ----
    neurons_out: dict[str, dict[str, Any]] = {}
    poles_out: dict[str, dict[str, Any]] = {}
    sim_windows_rows: list[dict[str, Any]] = []
    sim_meta_tok_ids: dict[str, np.ndarray] = {}
    sim_meta_weights: dict[str, np.ndarray] = {}
    sim_meta_is_active: dict[str, np.ndarray] = {}
    sim_meta_strata: dict[str, np.ndarray] = {}

    for k, (L, n_idx) in enumerate(sorted_lns):
        nid = make_neuron_id_17b(L, n_idx)
        col = neuron_to_col[(L, n_idx)]
        p_pos = poles_by_ln[(L, n_idx)][1]
        p_neg = poles_by_ln[(L, n_idx)][-1]

        a_all = acts[:, col].astype(np.float32)
        q_all = quads[:, col]
        a_nb = a_all[non_bos].astype(np.float64)
        q_nb = q_all[non_bos]
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
        mask_20 = abs_nb >= (0.20 * m_abs)
        mask_50 = abs_nb >= (0.50 * m_abs)
        energy_ge_20 = float(np.sum(l2[mask_20]) / denom_l2)
        energy_ge_50 = float(np.sum(l2[mask_50]) / denom_l2)
        energy_lt_20 = float(1.0 - energy_ge_20)
        l1_ge_20 = float(np.sum(l1[mask_20]) / denom_l1)
        l1_ge_50 = float(np.sum(l1[mask_50]) / denom_l1)

        gate_open = q_nb < 2
        energy_gate_open = float(np.sum(l2[gate_open]) / denom_l2)
        energy_gate_closed = float(np.sum(l2[~gate_open]) / denom_l2)

        # Exact 4-band (`p99.9+`, `p97_99.9`, `p75_97`, `lt_p75`) & 5-band (matching 1.7B det bands)
        def _pole_codes_4_and_5(sign: int, p_rec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            d = float(sign) * a_nb
            edges = p_rec["band_edges_dir"]
            p75 = float(edges["p75"])
            p90 = float(edges["p90"])
            p97 = float(edges["p97"])
            p99_9 = float(edges["p99.9"])
            # 4-band: 0=p99.9+, 1=p97_99.9, 2=p75_97, 3=lt_p75
            c4 = np.full(d.shape, 3, dtype=np.uint8)
            c4[(d >= p75) & (d < p97)] = 2
            c4[(d >= p97) & (d < p99_9)] = 1
            c4[d >= p99_9] = 0
            # 5-band (matching 1.7B detection bands): 0=p99.9_plus, 1=p97_99.9, 2=p90_97, 3=p75_90, 4=lt_p75
            c5 = np.full(d.shape, 4, dtype=np.uint8)
            c5[(d >= p75) & (d < p90)] = 3
            c5[(d >= p90) & (d < p97)] = 2
            c5[(d >= p97) & (d < p99_9)] = 1
            c5[d >= p99_9] = 0
            return c4, c5

        c4_pos, c5_pos = _pole_codes_4_and_5(1, p_pos)
        c4_neg, c5_neg = _pole_codes_4_and_5(-1, p_neg)
        n_c4 = np.where(a_nb >= 0.0, c4_pos, c4_neg)
        n_c5 = np.where(a_nb >= 0.0, c5_pos, c5_neg)

        def _agg_4(codes: np.ndarray, w: np.ndarray, den: float) -> dict[str, float]:
            return {
                "p99.9+": float(np.sum(w[codes == 0]) / den),
                "p97_99.9": float(np.sum(w[codes == 1]) / den),
                "p75_97": float(np.sum(w[codes == 2]) / den),
                "lt_p75": float(np.sum(w[codes == 3]) / den),
            }

        def _agg_5(codes: np.ndarray, w: np.ndarray, den: float) -> dict[str, float]:
            return {
                "p99.9_plus": float(np.sum(w[codes == 0]) / den),
                "p97_99.9": float(np.sum(w[codes == 1]) / den),
                "p90_97": float(np.sum(w[codes == 2]) / den),
                "p75_90": float(np.sum(w[codes == 3]) / den),
                "lt_p75": float(np.sum(w[codes == 4]) / den),
            }

        bands_4_l2 = _agg_4(n_c4, l2, denom_l2)
        bands_5_l2 = _agg_5(n_c5, l2, denom_l2)
        bands_4_l1 = _agg_4(n_c4, l1, denom_l1)
        bands_5_l1 = _agg_5(n_c5, l1, denom_l1)
        assert abs(sum(bands_4_l2.values()) - 1.0) < 1e-9
        assert abs(sum(bands_5_l2.values()) - 1.0) < 1e-9

        pos_pid = str(p_pos["pole_id"])
        neg_pid = str(p_neg["pole_id"])
        for sign, pid, p_rec, c4_p, c5_p in (
            (1, pos_pid, p_pos, c4_pos, c5_pos),
            (-1, neg_pid, p_neg, c4_neg, c5_neg),
        ):
            d_nb = float(sign) * a_nb
            max_d = float(np.max(d_nb))
            poles_out[pid] = {
                "pole_id": pid,
                "neuron_id": nid,
                "k": k,
                "layer": L,
                "neuron": n_idx,
                "sign": sign,
                "in_pilot": bool(p_rec.get("in_pilot", False)),
                "p25_abs": float(p_rec["p25_abs"]),
                "band_edges_dir": dict(p_rec["band_edges_dir"]),
                "max_act_dir": max_d,
                "own_label": str(labels_by_pole.get(pid, {}).get("own_label", "")),
                "energy_l2": {
                    "bands_4": _agg_4(c4_p, l2, denom_l2),
                    "bands_5": _agg_5(c5_p, l2, denom_l2),
                },
                "energy_l1": {
                    "bands_4": _agg_4(c4_p, l1, denom_l1),
                    "bands_5": _agg_5(c5_p, l1, denom_l1),
                },
            }

        # ---- Select 64 disjoint odd-doc 64-token simulation windows (32 ord + 32 active) ----
        # Vectorized window max(|a|) over non-BOS positions [1..63] across all odd candidate windows
        win_max_abs = np.empty(n_odd_wins, dtype=np.float32)
        for wi in range(n_odd_wins):
            st = int(odd_win_starts_arr[wi])
            win_max_abs[wi] = float(np.max(np.abs(a_all[st + 1 : st + sim_win_len])))

        is_win_active_all = win_max_abs >= (0.20 * m_abs)
        freq_active = float(np.mean(is_win_active_all))
        freq_inactive = float(1.0 - freq_active)

        rng_sim = _stable_rng_17b(seed, nid, "sim_windows_64")
        # 32 random ordinary windows (uniform random over ALL odd-doc windows, distinct docs first)
        perm_all = rng_sim.permutation(n_odd_wins).tolist()
        picked_ord: list[int] = []
        used_ws: set[int] = set()
        used_docs_ord: set[int] = set()
        for wi in perm_all:
            if len(picked_ord) >= sim_ord_wins:
                break
            did = int(odd_win_docs_arr[wi])
            if did not in used_docs_ord:
                used_docs_ord.add(did)
                used_ws.add(wi)
                picked_ord.append(wi)
        if len(picked_ord) < sim_ord_wins:
            for wi in perm_all:
                if len(picked_ord) >= sim_ord_wins:
                    break
                if wi not in used_ws:
                    used_ws.add(wi)
                    picked_ord.append(wi)

        # 32 active-stratified windows (max|a| >= 20% max, distinct from ordinary windows)
        act_cands = [int(wi) for wi in np.flatnonzero(is_win_active_all).tolist() if int(wi) not in used_ws]
        rng_sim.shuffle(act_cands)
        picked_act: list[int] = []
        used_docs_act: set[int] = set()
        for wi in act_cands:
            if len(picked_act) >= sim_act_wins:
                break
            did = int(odd_win_docs_arr[wi])
            if did not in used_docs_act:
                used_docs_act.add(did)
                used_ws.add(wi)
                picked_act.append(wi)
        if len(picked_act) < sim_act_wins:
            for wi in act_cands:
                if len(picked_act) >= sim_act_wins:
                    break
                if wi not in used_ws:
                    used_ws.add(wi)
                    picked_act.append(wi)
        active_fallback_used = 0
        if len(picked_act) < sim_act_wins:
            # Ultra-sparse fallback: take strongest remaining odd windows
            rem_order = np.argsort(-win_max_abs).tolist()
            for wi in rem_order:
                if len(picked_act) >= sim_act_wins:
                    break
                if wi not in used_ws:
                    used_ws.add(wi)
                    picked_act.append(wi)
                    active_fallback_used += 1

        chosen_ws = picked_ord + picked_act
        strata = ["ordinary"] * len(picked_ord) + ["active"] * len(picked_act)
        is_act_flags = [bool(is_win_active_all[wi]) for wi in chosen_ws]

        n_act_chosen = sum(1 for x in is_act_flags if x)
        n_inact_chosen = len(chosen_ws) - n_act_chosen
        if n_act_chosen > 0 and n_inact_chosen > 0:
            win_weights = [
                (freq_active / n_act_chosen) if is_a else (freq_inactive / n_inact_chosen)
                for is_a in is_act_flags
            ]
        else:
            win_weights = [1.0 / len(chosen_ws)] * len(chosen_ws)

        w_tok_mat = np.empty((len(chosen_ws), sim_win_len), dtype=np.int32)
        pos_lbl = str(labels_by_pole.get(pos_pid, {}).get("own_label", ""))
        neg_lbl = str(labels_by_pole.get(neg_pid, {}).get("own_label", ""))

        for w_local, (wi, strat, is_a, wt) in enumerate(zip(chosen_ws, strata, is_act_flags, win_weights)):
            st = int(odd_win_starts_arr[wi])
            tids = token_ids[st : st + sim_win_len].tolist()
            w_tok_mat[w_local] = tids
            toks_str = [
                tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for tid in tids
            ]
            full_text = tokenizer.decode(tids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            sim_windows_rows.append(
                {
                    "window_id": f"{nid}__w{w_local:02d}",
                    "neuron_id": nid,
                    "k": k,
                    "layer": L,
                    "neuron": n_idx,
                    "window_idx": w_local,
                    "stratum": strat,
                    "is_active": bool(is_a),
                    "freq_active": freq_active,
                    "freq_inactive": freq_inactive,
                    "weight_ordinary": float(wt),
                    "doc_id": int(odd_win_docs_arr[wi]),
                    "chunk_id": int(odd_win_cids_arr[wi]),
                    "start_in_chunk": int(odd_win_offs_arr[wi]),
                    "token_ids": [int(x) for x in tids],
                    "tokens_str": toks_str,
                    "text": full_text,
                    "pos_pole_id": pos_pid,
                    "neg_pole_id": neg_pid,
                    "pos_label": pos_lbl,
                    "neg_label": neg_lbl,
                    "max_abs_act_corpus": m_abs,
                    "mean_act_corpus": mean_a,
                }
            )

        sim_meta_tok_ids[nid] = w_tok_mat
        sim_meta_weights[nid] = np.asarray(win_weights, dtype=np.float64)
        sim_meta_is_active[nid] = np.asarray(is_act_flags, dtype=bool)
        sim_meta_strata[nid] = np.asarray([0 if s == "ordinary" else 1 for s in strata], dtype=np.uint8)

        neurons_out[nid] = {
            "k": k,
            "neuron_id": nid,
            "layer": L,
            "neuron": n_idx,
            "in_pilot": bool(p_pos.get("in_pilot", False)),
            "pos_pole_id": pos_pid,
            "neg_pole_id": neg_pid,
            "pos_label": pos_lbl,
            "neg_label": neg_lbl,
            "mean_act": mean_a,
            "var_act": float(tot_l2 / max(n_non_bos, 1)),
            "max_abs_act": m_abs,
            "p25_abs": float(p_pos["p25_abs"]),
            "pos_band_edges_dir": dict(p_pos["band_edges_dir"]),
            "neg_band_edges_dir": dict(p_neg["band_edges_dir"]),
            "energy_l2": {"bands_4": bands_4_l2, "bands_5": bands_5_l2},
            "energy_l1": {"bands_4": bands_4_l1, "bands_5": bands_5_l1},
            "energy_p99.9+": float(bands_4_l2["p99.9+"]),
            "energy_p97+": float(bands_4_l2["p99.9+"] + bands_4_l2["p97_99.9"]),
            "energy_lt_p97": float(bands_4_l2["p75_97"] + bands_4_l2["lt_p75"]),
            "energy_ge_20pct_max": energy_ge_20,
            "energy_ge_50pct_max": energy_ge_50,
            "energy_lt_20pct_max": energy_lt_20,
            "l1_ge_20pct_max": l1_ge_20,
            "l1_ge_50pct_max": l1_ge_50,
            "energy_gate_open": energy_gate_open,
            "energy_gate_closed": energy_gate_closed,
            "sim_windows_summary": {
                "n_windows": len(chosen_ws),
                "n_ordinary_stratum": len(picked_ord),
                "n_active_stratum": len(picked_act),
                "n_is_active_ge_20pct": n_act_chosen,
                "active_pool_total_odd": int(np.sum(is_win_active_all)),
                "active_fallback_used": active_fallback_used,
                "freq_active": freq_active,
                "freq_inactive": freq_inactive,
                "weight_sum": float(sum(win_weights)),
            },
        }

    # ---- 4. Build `fpr_probes_17b.jsonl` (100 poles x 24 = 2,400 probes, odd docs ONLY!) ----
    fpr_rows: list[dict[str, Any]] = []
    n_ord_want = fpr_ord_core + fpr_ord_extra
    n_top_want = fpr_top_core + fpr_top_extra
    dedup_len = min(DEDUP_SUFFIX_LEN, wspec.before + 1)

    for p in poles_doc["poles"]:
        pid = str(p["pole_id"])
        L = int(p["layer"])
        n_idx = int(p["neuron"])
        sign = int(p["sign"])
        nid = make_neuron_id_17b(L, n_idx)
        k = int(neurons_out[nid]["k"])
        col = neuron_to_col[(L, n_idx)]
        p25_abs = float(p["p25_abs"])
        p90 = float(p["band_edges_dir"]["p90"])
        p99_9 = float(p["band_edges_dir"]["p99.9"])
        own_label = str(labels_by_pole.get(pid, {}).get("own_label", ""))

        a_n = acts[:, col].astype(np.float32)
        q_n = quads[:, col]
        d_n = float(sign) * a_n
        abs_n = np.abs(a_n)
        max_act_dir = float(np.max(d_n[non_bos]))

        # 4a. 4 p99.9_plus firing probes (2 core + 2 luna_extra, distinct odd docs)
        top_mask = eval_interior & (d_n >= p99_9) & (d_n > 0.0) & (abs_n >= p25_abs)
        rng_top = _stable_rng_17b(seed, pid, "fpr_17b_top")
        top_idxs = _sample_distinct_doc_indices(
            candidate_indices=np.flatnonzero(top_mask),
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            pos_in_chunk=pos_in_chunk,
            n_want=n_top_want,
            rng=rng_top,
            min_spacing=wspec.total,
            token_ids=token_ids,
            dedup_within_sample=True,
            suffix_len=dedup_len,
            allow_same_doc_fallback=False,
        )
        if len(top_idxs) < n_top_want:
            raise RuntimeError(f"1.7B pole {pid}: only found {len(top_idxs)}/{n_top_want} odd-doc p99.9+ FPR probes")
        used_fpr_docs = {int(doc_ids[i]) for i in top_idxs}

        for i_top, f_idx in enumerate(top_idxs):
            marked_text, target_tok_str, _ = render_marked_window(token_ids, f_idx, tokenizer, wspec)
            q_code = int(q_n[f_idx])
            fpr_rows.append(
                {
                    "probe_id": f"{pid}__p999_firing__{i_top}",
                    "pole_id": pid,
                    "neuron_id": nid,
                    "k": k,
                    "layer": L,
                    "neuron": n_idx,
                    "sign": sign,
                    "probe_type": "top100_firing",  # Keep exact 4B key compatibility for surrogate_4b/sim_17b
                    "source_band": "p99.9_plus",
                    "ground_truth": "YES",
                    "tier": "core" if i_top < fpr_top_core else "luna_extra",
                    "probe_idx": i_top,
                    "doc_id": int(doc_ids[f_idx]),
                    "chunk_id": int(chunk_ids[f_idx]),
                    "pos_in_chunk": int(pos_in_chunk[f_idx]),
                    "window_tokens": int(wspec.total),
                    "target_offset": int(wspec.target_offset),
                    "target_token_id": int(token_ids[f_idx]),
                    "target_token_str": target_tok_str,
                    "act_val": float(a_n[f_idx]),
                    "act_dir": float(d_n[f_idx]),
                    "frac_of_max": float(d_n[f_idx]) / max_act_dir if max_act_dir > 0.0 else 0.0,
                    "quadrant": QUAD_CODE_TO_NAME[q_code],
                    "own_label": own_label,
                    "marked_text": marked_text,
                }
            )

        # 4b. 20 random ordinary non-firing probes (10 core + 10 luna_extra, distinct odd docs)
        ord_mask = eval_interior & (abs_n < (0.20 * max_act_dir)) & (d_n < p90)
        ord_cands = np.flatnonzero(ord_mask)
        rng_ord = _stable_rng_17b(seed, pid, "fpr_17b_ord")
        ord_idxs: list[int] = []
        if ord_cands.size > 0:
            probes = rng_ord.integers(0, ord_cands.size, size=min(4096, max(256, n_ord_want * 64)))
            for p_i in probes.tolist():
                cand = int(ord_cands[p_i])
                did = int(doc_ids[cand])
                if did not in used_fpr_docs:
                    used_fpr_docs.add(did)
                    ord_idxs.append(cand)
                    if len(ord_idxs) >= n_ord_want:
                        break
        if len(ord_idxs) < n_ord_want:
            extra = _sample_distinct_doc_indices(
                candidate_indices=ord_cands,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                pos_in_chunk=pos_in_chunk,
                n_want=n_ord_want - len(ord_idxs),
                rng=rng_ord,
                min_spacing=wspec.total,
                excluded_docs=used_fpr_docs,
                allow_same_doc_fallback=False,
            )
            ord_idxs.extend(extra)
        if len(ord_idxs) < n_ord_want:
            raise RuntimeError(f"1.7B pole {pid}: only found {len(ord_idxs)}/{n_ord_want} odd-doc ordinary FPR probes")

        for i_ord, f_idx in enumerate(ord_idxs):
            marked_text, target_tok_str, _ = render_marked_window(token_ids, f_idx, tokenizer, wspec)
            q_code = int(q_n[f_idx])
            fpr_rows.append(
                {
                    "probe_id": f"{pid}__ordinary_nonfiring__{i_ord}",
                    "pole_id": pid,
                    "neuron_id": nid,
                    "k": k,
                    "layer": L,
                    "neuron": n_idx,
                    "sign": sign,
                    "probe_type": "ordinary_nonfiring",
                    "source_band": "ordinary",
                    "ground_truth": "NO",
                    "tier": "core" if i_ord < fpr_ord_core else "luna_extra",
                    "probe_idx": i_ord,
                    "doc_id": int(doc_ids[f_idx]),
                    "chunk_id": int(chunk_ids[f_idx]),
                    "pos_in_chunk": int(pos_in_chunk[f_idx]),
                    "window_tokens": int(wspec.total),
                    "target_offset": int(wspec.target_offset),
                    "target_token_id": int(token_ids[f_idx]),
                    "target_token_str": target_tok_str,
                    "act_val": float(a_n[f_idx]),
                    "act_dir": float(d_n[f_idx]),
                    "frac_of_max": float(d_n[f_idx]) / max_act_dir if max_act_dir > 0.0 else 0.0,
                    "quadrant": QUAD_CODE_TO_NAME[q_code],
                    "own_label": own_label,
                    "marked_text": marked_text,
                }
            )

    # ---- 5. Verify hard invariants & write artifacts ----
    sim_even_leak = sum(1 for r in sim_windows_rows if (int(r["doc_id"]) % 2) == 0)
    fpr_even_leak = sum(1 for r in fpr_rows if (int(r["doc_id"]) % 2) == 0)
    fpr_bos_leak = sum(1 for r in fpr_rows if int(r["pos_in_chunk"]) <= 0)
    fpr_win_err = sum(
        1
        for r in fpr_rows
        if int(r["window_tokens"]) != wspec.total or int(r["target_offset"]) != wspec.target_offset
    )
    fpr_docs_by_pole: dict[str, list[int]] = defaultdict(list)
    for r in fpr_rows:
        fpr_docs_by_pole[str(r["pole_id"])].append(int(r["doc_id"]))
    fpr_dup_doc = sum(len(v) - len(set(v)) for v in fpr_docs_by_pole.values())

    e3_even_leak = int(np.sum((e3_docs % 2) == 0))
    all_passed = bool(
        e3_even_leak == 0
        and sim_even_leak == 0
        and fpr_even_leak == 0
        and fpr_bos_leak == 0
        and fpr_win_err == 0
        and fpr_dup_doc == 0
        and len(sim_windows_rows) == len(sorted_lns) * (sim_ord_wins + sim_act_wins)
        and len(fpr_rows) == len(poles_doc["poles"]) * (n_ord_want + n_top_want)
    )

    np.savez_compressed(
        out_data / "e3_ordinary_17b.npz",
        token_ids=e3_tok_ids,
        doc_ids=e3_docs,
        chunk_ids=e3_cids,
        start_offsets=e3_offs,
    )
    np.savez_compressed(
        out_data / "sim_windows_meta_17b.npz",
        **{f"tok__{nid}": mat for nid, mat in sim_meta_tok_ids.items()},
        **{f"w__{nid}": arr for nid, arr in sim_meta_weights.items()},
        **{f"act__{nid}": arr for nid, arr in sim_meta_is_active.items()},
        **{f"strat__{nid}": arr for nid, arr in sim_meta_strata.items()},
    )
    with (out_data / "sim_windows_17b.jsonl").open("w", encoding="utf-8") as f:
        for row in sim_windows_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_data / "fpr_probes_17b.jsonl").open("w", encoding="utf-8") as f:
        for row in fpr_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    neurons_catalog = {
        "model_id": MODEL_ID_17B,
        "seed": seed,
        "layers": sorted({L for L, _ in sorted_lns}),
        "n_neurons": len(neurons_out),
        "n_poles": len(poles_out),
        "n_non_bos_tokens": n_non_bos,
        "e3_ordinary_shape": [int(e3_tok_ids.shape[0]), int(e3_tok_ids.shape[1])],
        "per_neuron": neurons_out,
        "per_pole": poles_out,
    }
    (out_data / "neurons_17b.json").write_text(
        json.dumps(neurons_catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checks = {
        "all_hard_invariants_passed": all_passed,
        "n_neurons": len(neurons_out),
        "n_poles": len(poles_out),
        "e3_ordinary_n_windows": int(e3_tok_ids.shape[0]),
        "e3_ordinary_win_len": int(e3_tok_ids.shape[1]),
        "e3_ordinary_unique_odd_docs": len(set(e3_docs.tolist())),
        "e3_ordinary_even_doc_leaks": e3_even_leak,
        "sim_windows_total": len(sim_windows_rows),
        "sim_windows_even_doc_leaks": sim_even_leak,
        "fpr_probes_total": len(fpr_rows),
        "fpr_probes_core": sum(1 for r in fpr_rows if r["tier"] == "core"),
        "fpr_probes_luna_extra": sum(1 for r in fpr_rows if r["tier"] == "luna_extra"),
        "fpr_even_doc_leaks": fpr_even_leak,
        "fpr_bos_leaks": fpr_bos_leak,
        "fpr_window_errors": fpr_win_err,
        "fpr_within_pole_dup_docs": fpr_dup_doc,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_res / "structural_checks_17b_causal.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    if not all_passed:
        raise AssertionError(f"1.7B causal input checks FAILED: {json.dumps(checks, indent=2)}")

    print(
        f"[causal_17b:3A] COMPLETE in {checks['elapsed_sec']:.1f}s | "
        f"e3_ordinary={checks['e3_ordinary_n_windows']}x{checks['e3_ordinary_win_len']} ({checks['e3_ordinary_unique_odd_docs']} odd docs) | "
        f"sim_windows={checks['sim_windows_total']:,} | fpr_probes={checks['fpr_probes_total']:,} "
        f"(core={checks['fpr_probes_core']:,}, extra={checks['fpr_probes_luna_extra']:,}) | even_leaks=0"
    )
    return {"catalog": neurons_catalog, "checks": checks}


# ---------------------------------------------------------------------------
# Exact 2-Half IEEE-754 Float32 Engine for Qwen3-1.7B
# ---------------------------------------------------------------------------


def _run_17b_clean_and_patch_engine(
    backbone: nn.Module,
    input_ids_cpu: torch.Tensor,
    hook_layers: Sequence[int],
    neurons_by_layer: Mapping[int, Sequence[int]],
    compute_device: torch.device,
    micro_batch_size: int,
    kl_token_chunk: int,
    max_vram_gb: float,
    lm_head_weight_cpu: torch.Tensor,
) -> dict[str, Any]:
    """Run exact 2-half IEEE-754 FP32 clean pass + zero-ablation re-run sanity check."""
    is_cuda = compute_device.type == "cuda"
    cpu_dev = torch.device("cpu")
    orig_dtype = next(backbone.parameters()).dtype
    n_layers = len(backbone.layers)
    stage_ranges = _compute_two_half_ranges(n_layers)
    n_stages = len(stage_ranges)
    n_seqs, seq_len = int(input_ids_cpu.shape[0]), int(input_ids_cpu.shape[1])
    assert n_seqs % micro_batch_size == 0
    n_batches = n_seqs // micro_batch_size
    hidden_size = int(backbone.embed_tokens.weight.shape[1])

    rotary_mod = getattr(backbone, "rotary_emb", None)
    norm_mod = getattr(backbone, "norm", None)

    def _stage_mods(s_idx: int) -> list[nn.Module]:
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

    def _load(s_idx: int) -> list[nn.Module]:
        mods = _stage_mods(s_idx)
        for m in mods:
            if m is rotary_mod:
                m.to(device=compute_device)
            else:
                m.to(device=compute_device, dtype=torch.float32)
        return mods

    def _unload(mods: Sequence[nn.Module]) -> None:
        for m in mods:
            if m is rotary_mod:
                m.to(device=cpu_dev)
            else:
                m.to(device=cpu_dev, dtype=orig_dtype)
        _clear_cuda()

    peak_a = peak_r = 0.0

    def _check_vram() -> None:
        nonlocal peak_a, peak_r
        if is_cuda:
            torch.cuda.synchronize(compute_device)
            pa = torch.cuda.max_memory_allocated(compute_device) / (1024**3)
            pr = torch.cuda.max_memory_reserved(compute_device) / (1024**3)
            peak_a = max(peak_a, pa)
            peak_r = max(peak_r, pr)
            if pa >= max_vram_gb or pr >= max_vram_gb:
                raise ParityGateError(
                    f"1.7B FP32 VRAM limit exceeded: alloc={pa:.3f}GB res={pr:.3f}GB >= {max_vram_gb}GB"
                )

    clean_layer_in: dict[int, torch.Tensor] = {
        L: torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
        for L in hook_layers
    }
    stage_h = torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
    clean_final = torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
    clean_acts: dict[tuple[int, int], np.ndarray] = {}
    clean_quads: dict[tuple[int, int], np.ndarray] = {}
    for L in hook_layers:
        for n_idx in neurons_by_layer[L]:
            clean_acts[(L, int(n_idx))] = np.empty((n_seqs, seq_len), dtype=np.float32)
            clean_quads[(L, int(n_idx))] = np.empty((n_seqs, seq_len), dtype=np.uint8)

    for s_idx, (l_a, l_b) in enumerate(stage_ranges):
        if is_cuda:
            _clear_cuda()
            torch.cuda.reset_peak_memory_stats(compute_device)
        s_hooks = [L for L in hook_layers if l_a <= L < l_b]
        mods = _load(s_idx)
        handles: list[Any] = []
        buf: dict[int, dict[str, torch.Tensor]] = {L: {} for L in s_hooks}
        try:
            for L in s_hooks:
                idx_t = torch.tensor(list(neurons_by_layer[L]), dtype=torch.long, device=compute_device)
                mlp = backbone.layers[L].mlp

                def _mg(li: int, sel: torch.Tensor):
                    def _h(_m, _a, out):
                        buf[li]["g"] = out.detach().index_select(-1, sel)
                    return _h

                def _mu(li: int, sel: torch.Tensor):
                    def _h(_m, _a, out):
                        buf[li]["u"] = out.detach().index_select(-1, sel)
                    return _h

                def _ma(li: int, sel: torch.Tensor):
                    def _h(_m, args):
                        buf[li]["a"] = args[0].detach().index_select(-1, sel)
                    return _h

                handles.append(mlp.gate_proj.register_forward_hook(_mg(L, idx_t)))
                handles.append(mlp.up_proj.register_forward_hook(_mu(L, idx_t)))
                handles.append(mlp.down_proj.register_forward_pre_hook(_ma(L, idx_t)))

            with torch.no_grad():
                for b_idx in range(n_batches):
                    s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                    h = (
                        backbone.embed_tokens(input_ids_cpu[s0:s1].to(compute_device))
                        if s_idx == 0
                        else stage_h[s0:s1].to(compute_device, dtype=torch.float32)
                    )
                    pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                    for i in range(l_a, l_b):
                        if i in clean_layer_in:
                            clean_layer_in[i][s0:s1].copy_(h.detach().cpu())
                        h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                    if s_idx == n_stages - 1:
                        if norm_mod is not None:
                            h = norm_mod(h)
                        clean_final[s0:s1].copy_(h.detach().cpu())
                    else:
                        stage_h[s0:s1].copy_(h.detach().cpu())

                    for L in s_hooks:
                        a_np = buf[L]["a"].cpu().numpy()
                        q_np = encode_quadrant_torch(buf[L]["g"], buf[L]["u"]).cpu().numpy()
                        for j, n_idx in enumerate(neurons_by_layer[L]):
                            clean_acts[(L, int(n_idx))][s0:s1] = a_np[:, :, j]
                            clean_quads[(L, int(n_idx))][s0:s1] = q_np[:, :, j]
                        buf[L].clear()
                    del h, pos_ids, pos_embs, mask_map
            _check_vram()
        finally:
            for hh in handles:
                hh.remove()
            _unload(mods)

    # Zero-ablation re-run sanity check from first hooked layer
    first_L = hook_layers[0]
    first_s = 0 if first_L < stage_ranges[0][1] else 1
    test_h = clean_layer_in[first_L][:micro_batch_size].clone()
    for s_idx in range(first_s, n_stages):
        l_a, l_b = stage_ranges[s_idx]
        mods = _load(s_idx)
        try:
            with torch.no_grad():
                h = test_h.to(compute_device, dtype=torch.float32)
                pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                loop_a = first_L if s_idx == first_s else l_a
                for i in range(loop_a, l_b):
                    h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                if s_idx == n_stages - 1 and norm_mod is not None:
                    h = norm_mod(h)
                test_h = h.detach().cpu()
                del h, pos_ids, pos_embs, mask_map
            _check_vram()
        finally:
            _unload(mods)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(compute_device)
    lm_w = lm_head_weight_cpu.to(device=compute_device, dtype=torch.float32)
    with torch.no_grad():
        zero_kl = (
            _compute_chunked_kl_sum(
                clean_final[:micro_batch_size].to(compute_device),
                test_h.to(compute_device),
                lm_w,
                token_chunk=kl_token_chunk,
            )
            / float(micro_batch_size * (seq_len - 1))
        )
    _check_vram()
    del lm_w, test_h
    _clear_cuda()
    if zero_kl > 1e-10:
        raise ParityGateError(f"1.7B FP32 zero-ablation re-run KL failed: {zero_kl:.3e} > 1e-10")

    return {
        "stage_ranges": stage_ranges,
        "clean_layer_in": clean_layer_in,
        "clean_final": clean_final,
        "clean_acts": clean_acts,
        "clean_quads": clean_quads,
        "zero_ablation_sanity_kl": zero_kl,
        "peak_vram_alloc": peak_a,
        "peak_vram_res": peak_r,
    }


# ---------------------------------------------------------------------------
# Step 3B: Exact IEEE-754 FP32 strong20_rec, strong50_rec & 4-Band KLShare
# ---------------------------------------------------------------------------


def run_17b_baseline_kl(
    backbone: nn.Module,
    base_dir: Path | None = None,
    compute_device: str | torch.device = "cuda",
    micro_batch_size: int = 16,
    kl_token_chunk: int = 512,
    max_vram_gb: float = 6.0,
    neurons_per_group: int = 5,
    n_bootstrap: int = 1000,
    seed: int = SEED_17B,
) -> dict[str, Any]:
    """Run Step 3B exact IEEE-754 FP32 baseline & 4-band KL ablations for all 50 1.7B neurons."""
    t0 = time.time()
    dirs = ensure_17b_causal_dirs(base_dir)
    dev = torch.device(compute_device)
    is_cuda = dev.type == "cuda"
    cpu_dev = torch.device("cpu")

    prev_tf32_m = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    neurons_cat = json.loads((dirs["data"] / "neurons_17b.json").read_text(encoding="utf-8"))
    e3_npz = np.load(dirs["data"] / "e3_ordinary_17b.npz")
    input_ids_cpu = torch.from_numpy(e3_npz["token_ids"].astype(np.int64))
    n_seqs, seq_len = int(input_ids_cpu.shape[0]), int(input_ids_cpu.shape[1])
    n_batches = n_seqs // micro_batch_size
    total_non_bos = n_seqs * (seq_len - 1)

    orig_dtype = next(backbone.parameters()).dtype
    lm_head_weight_cpu = backbone.embed_tokens.weight.detach()
    rotary_mod = getattr(backbone, "rotary_emb", None)
    norm_mod = getattr(backbone, "norm", None)

    per_neuron_cat = neurons_cat["per_neuron"]
    sorted_nids = sorted(per_neuron_cat.keys(), key=lambda nid: int(per_neuron_cat[nid]["k"]))
    neurons_by_layer: dict[int, list[int]] = defaultdict(list)
    recs_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for nid in sorted_nids:
        rec = per_neuron_cat[nid]
        L = int(rec["layer"])
        neurons_by_layer[L].append(int(rec["neuron"]))
        recs_by_layer[L].append(rec)
    hook_layers = sorted(neurons_by_layer.keys())

    try:
        clean = _run_17b_clean_and_patch_engine(
            backbone=backbone,
            input_ids_cpu=input_ids_cpu,
            hook_layers=hook_layers,
            neurons_by_layer=neurons_by_layer,
            compute_device=dev,
            micro_batch_size=micro_batch_size,
            kl_token_chunk=kl_token_chunk,
            max_vram_gb=max_vram_gb,
            lm_head_weight_cpu=lm_head_weight_cpu,
        )
        stage_ranges = clean["stage_ranges"]
        n_stages = len(stage_ranges)
        clean_layer_in = clean["clean_layer_in"]
        clean_final = clean["clean_final"]
        clean_acts = clean["clean_acts"]
        clean_quads = clean["clean_quads"]
        peak_a = clean["peak_vram_alloc"]
        peak_r = clean["peak_vram_res"]
        zero_sanity_kl = clean["zero_ablation_sanity_kl"]

        print(
            f"[causal_17b:3B] Clean pass + zero-ablation gate PASSED: "
            f"peak_vram={peak_a:.3f}/{peak_r:.3f}GB, zero_kl={zero_sanity_kl:.2e} ({time.time() - t0:.1f}s)"
        )

        def _stage_mods(s_idx: int) -> list[nn.Module]:
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

        def _load(s_idx: int) -> list[nn.Module]:
            mods = _stage_mods(s_idx)
            for m in mods:
                if m is rotary_mod:
                    m.to(device=dev)
                else:
                    m.to(device=dev, dtype=torch.float32)
            return mods

        def _unload(mods: Sequence[nn.Module]) -> None:
            for m in mods:
                if m is rotary_mod:
                    m.to(device=cpu_dev)
                else:
                    m.to(device=cpu_dev, dtype=orig_dtype)
            _clear_cuda()

        def _check_vram() -> None:
            nonlocal peak_a, peak_r
            if is_cuda:
                torch.cuda.synchronize(dev)
                pa = torch.cuda.max_memory_allocated(dev) / (1024**3)
                pr = torch.cuda.max_memory_reserved(dev) / (1024**3)
                peak_a = max(peak_a, pa)
                peak_r = max(peak_r, pr)
                if pa >= max_vram_gb or pr >= max_vram_gb:
                    raise ParityGateError(f"1.7B VRAM exceeded: {pa:.3f}/{pr:.3f}GB >= {max_vram_gb}GB")

        # Build replacement value tensors `[n_seqs, seq_len]` (or mask + scalar) for all 10 conditions
        cond_specs_by_nid: dict[str, dict[str, tuple[np.ndarray, float]]] = {}
        eval_energy_4b_by_nid: dict[str, dict[str, float]] = {}
        eval_energy_ge20_by_nid: dict[str, float] = {}
        eval_energy_gc_by_nid: dict[str, float] = {}

        for nid in sorted_nids:
            rec = per_neuron_cat[nid]
            L, n_idx = int(rec["layer"]), int(rec["neuron"])
            a = clean_acts[(L, n_idx)].astype(np.float64)
            q = clean_quads[(L, n_idx)]
            abs_a = np.abs(a)
            m_abs = float(rec["max_abs_act"])
            mean_a = float(rec["mean_act"])

            # 4-band assignment (bipolar: sign(a) pole)
            def _pc4(sign: int, edges: Mapping[str, float]) -> np.ndarray:
                d = float(sign) * a
                c = np.full(a.shape, 3, dtype=np.uint8)
                c[(d >= float(edges["p75"])) & (d < float(edges["p97"]))] = 2
                c[(d >= float(edges["p97"])) & (d < float(edges["p99.9"]))] = 1
                c[d >= float(edges["p99.9"])] = 0
                return c

            b4 = np.where(
                a >= 0.0,
                _pc4(1, rec["pos_band_edges_dir"]),
                _pc4(-1, rec["neg_band_edges_dir"]),
            )
            nb = np.ones(a.shape, dtype=bool)
            nb[:, 0] = False
            gate_open = q < 2

            # Each cond: (boolean mask of tokens to replace, replacement scalar value)
            cond_specs_by_nid[nid] = {
                "zero": (nb.copy(), 0.0),
                "mean": (nb.copy(), mean_a),
                # Ulto85 exact strong20/strong50: keep |a| >= f*M, replace < f*M with 0.0
                "strong20": (nb & (abs_a < 0.20 * m_abs), 0.0),
                "strong50": (nb & (abs_a < 0.50 * m_abs), 0.0),
                "p99.9+": (nb & (b4 == 0), mean_a),
                "p97_99.9": (nb & (b4 == 1), mean_a),
                "p75_97": (nb & (b4 == 2), mean_a),
                "lt_p75": (nb & (b4 == 3), mean_a),
                "gate_open": (nb & gate_open, mean_a),
                "gate_closed": (nb & (~gate_open), mean_a),
            }

            l2_eval = (a[:, 1:] - mean_a) ** 2
            tot_e = float(np.sum(l2_eval))
            den = tot_e if tot_e > 0.0 else 1.0
            eval_energy_4b_by_nid[nid] = {
                b: float(np.sum(l2_eval[cond_specs_by_nid[nid][b][0][:, 1:]]) / den)
                for b in FOUR_BAND_17B
            }
            eval_energy_ge20_by_nid[nid] = float(np.sum(l2_eval[abs_a[:, 1:] >= 0.20 * m_abs]) / den)
            eval_energy_gc_by_nid[nid] = float(np.sum(l2_eval[~gate_open[:, 1:]]) / den)

        # Group neurons by starting stage (Stage 1 -> Stage 0)
        layer_to_stage = {L: s_i for s_i, (a, b) in enumerate(stage_ranges) for L in range(a, b)}
        recs_by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for L in hook_layers:
            recs_by_stage[layer_to_stage[L]].extend(recs_by_layer[L])

        kl_by_nid: dict[str, dict[str, float]] = {nid: {} for nid in sorted_nids}
        done = 0

        for start_s in sorted(recs_by_stage.keys(), reverse=True):
            s_recs = recs_by_stage[start_s]
            for g0 in range(0, len(s_recs), neurons_per_group):
                group = s_recs[g0 : g0 + neurons_per_group]
                pending: dict[tuple[str, str], list[tuple[int, list[int], torch.Tensor]]] = {}

                l_a, l_b = stage_ranges[start_s]
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats(dev)
                mods = _load(start_s)
                try:
                    for rec in group:
                        nid = str(rec["neuron_id"])
                        L, n_idx = int(rec["layer"]), int(rec["neuron"])
                        state: dict[str, Any] = {"bm": None, "val": None}

                        def _hook(_mod: nn.Module, args: tuple[Any, ...]):
                            x = args[0].clone()
                            x[:, :, n_idx] = torch.where(state["bm"], state["val"], x[:, :, n_idx])
                            return (x,)

                        hh = backbone.layers[L].mlp.down_proj.register_forward_pre_hook(_hook)
                        try:
                            with torch.no_grad():
                                for cond in BASELINE_KL_CONDS_17B:
                                    m2d, rep_val = cond_specs_by_nid[nid][cond]
                                    state["val"] = torch.tensor(rep_val, dtype=torch.float32, device=dev)
                                    items: list[tuple[int, list[int], torch.Tensor]] = []
                                    for b_idx in range(n_batches):
                                        s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                                        bm = m2d[s0:s1]
                                        if not bm.any():
                                            continue
                                        changed = np.flatnonzero(bm.any(axis=1)).tolist()
                                        h = clean_layer_in[L][s0:s1].to(dev, dtype=torch.float32)
                                        state["bm"] = torch.from_numpy(bm).to(dev)
                                        pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                        for i in range(L, l_b):
                                            h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                                        if start_s == n_stages - 1 and norm_mod is not None:
                                            h = norm_mod(h)
                                        items.append((b_idx, changed, h.detach().cpu()))
                                        del h, pos_ids, pos_embs, mask_map
                                        state["bm"] = None
                                    if not items:
                                        kl_by_nid[nid][cond] = 0.0
                                    else:
                                        pending[(nid, cond)] = items
                        finally:
                            hh.remove()
                    _check_vram()
                finally:
                    _unload(mods)

                for next_s in range(start_s + 1, n_stages):
                    nl_a, nl_b = stage_ranges[next_s]
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats(dev)
                    mods = _load(next_s)
                    try:
                        with torch.no_grad():
                            for key_nc, items in pending.items():
                                new_items: list[tuple[int, list[int], torch.Tensor]] = []
                                for b_idx, changed, h_cpu in items:
                                    h = h_cpu.to(dev, dtype=torch.float32)
                                    pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                    for i in range(nl_a, nl_b):
                                        h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                                    if next_s == n_stages - 1 and norm_mod is not None:
                                        h = norm_mod(h)
                                    new_items.append((b_idx, changed, h.detach().cpu()))
                                    del h, pos_ids, pos_embs, mask_map
                                pending[key_nc] = new_items
                        _check_vram()
                    finally:
                        _unload(mods)

                if pending:
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats(dev)
                    lm_w = lm_head_weight_cpu.to(device=dev, dtype=torch.float32)
                    try:
                        with torch.no_grad():
                            for (nid, cond), items in pending.items():
                                kl_sum = 0.0
                                for b_idx, changed, h_fin_cpu in items:
                                    s0 = b_idx * micro_batch_size
                                    abs_ch = [s0 + li for li in changed]
                                    c_h = clean_final[abs_ch].to(dev, dtype=torch.float32)
                                    a_h = h_fin_cpu[changed].to(dev, dtype=torch.float32)
                                    kl_sum += _compute_chunked_kl_sum(c_h, a_h, lm_w, token_chunk=kl_token_chunk)
                                    del c_h, a_h
                                kl_by_nid[nid][cond] = float(kl_sum / float(total_non_bos))
                        _check_vram()
                    finally:
                        del lm_w
                        _clear_cuda()

                del pending
                done += len(group)
                print(
                    f"[causal_17b:3B] Stage {start_s} group done ({done}/{len(sorted_nids)} neurons, "
                    f"{time.time() - t0:.1f}s, peak_vram={peak_a:.2f}/{peak_r:.2f}GB)"
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_m
        torch.backends.cudnn.allow_tf32 = prev_tf32_c

    # Assemble per-neuron results & correlations
    per_neuron_out: dict[str, dict[str, Any]] = {}
    for nid in sorted_nids:
        rec = per_neuron_cat[nid]
        kls = kl_by_nid[nid]
        mean_kl = float(kls["mean"])
        zero_kl = float(kls["zero"])
        sum_4b = sum(float(kls[b]) for b in FOUR_BAND_17B)
        kl_4b_norm = {b: (float(kls[b]) / sum_4b if sum_4b > 0.0 else 0.0) for b in FOUR_BAND_17B}
        kl_4b_vs_mean = {b: (float(kls[b]) / mean_kl if mean_kl > 0.0 else 0.0) for b in FOUR_BAND_17B}
        sum_g = float(kls["gate_open"] + kls["gate_closed"])
        kl_gc_norm = float(kls["gate_closed"] / sum_g) if sum_g > 0.0 else 0.0

        strong20_rec = float(1.0 - kls["strong20"] / mean_kl) if mean_kl > 0.0 else float("nan")
        strong50_rec = float(1.0 - kls["strong50"] / mean_kl) if mean_kl > 0.0 else float("nan")
        strong20_rec_zero = float(1.0 - kls["strong20"] / zero_kl) if zero_kl > 0.0 else float("nan")
        strong50_rec_zero = float(1.0 - kls["strong50"] / zero_kl) if zero_kl > 0.0 else float("nan")

        per_neuron_out[nid] = {
            "k": int(rec["k"]),
            "neuron_id": nid,
            "layer": int(rec["layer"]),
            "neuron": int(rec["neuron"]),
            "in_pilot": bool(rec["in_pilot"]),
            "kl_by_cond": kls,
            "strong20_rec": strong20_rec,
            "strong50_rec": strong50_rec,
            "strong20_rec_vs_zero": strong20_rec_zero,
            "strong50_rec_vs_zero": strong50_rec_zero,
            "kl_4band_additivity_ratio": float(sum_4b / mean_kl) if mean_kl > 0.0 else float("nan"),
            "kl_share_4band_norm": kl_4b_norm,
            "kl_share_4band_vs_mean": kl_4b_vs_mean,
            "kl_share_gate_closed_norm": kl_gc_norm,
            "corpus_energy_4band": rec["energy_l2"]["bands_4"],
            "corpus_energy_5band": rec["energy_l2"]["bands_5"],
            "corpus_energy_ge_20pct_max": float(rec["energy_ge_20pct_max"]),
            "corpus_energy_ge_50pct_max": float(rec["energy_ge_50pct_max"]),
            "corpus_energy_gate_closed": float(rec["energy_gate_closed"]),
            "eval_energy_4band": eval_energy_4b_by_nid[nid],
            "eval_energy_ge_20pct_max": eval_energy_ge20_by_nid[nid],
            "eval_energy_gate_closed": eval_energy_gc_by_nid[nid],
        }

    rows = [per_neuron_out[nid] for nid in sorted_nids]
    corrs: dict[str, Any] = {
        "corpus_energy_ge_20pct_max_vs_strong20_rec": bootstrap_corr_stats(
            [r["corpus_energy_ge_20pct_max"] for r in rows],
            [r["strong20_rec"] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed,
        ),
        "eval_energy_ge_20pct_max_vs_strong20_rec": bootstrap_corr_stats(
            [r["eval_energy_ge_20pct_max"] for r in rows],
            [r["strong20_rec"] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed + 1,
        ),
        "corpus_energy_ge_50pct_max_vs_strong50_rec": bootstrap_corr_stats(
            [r["corpus_energy_ge_50pct_max"] for r in rows],
            [r["strong50_rec"] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed + 2,
        ),
        "pooled_200_eval_energy_vs_kl_norm": {
            **bootstrap_corr_stats(
                [r["eval_energy_4band"][b] for r in rows for b in FOUR_BAND_17B],
                [r["kl_share_4band_norm"][b] for r in rows for b in FOUR_BAND_17B],
                n_resamples=n_bootstrap,
                seed=seed + 3,
            ),
            "bootstrap_unit": "flattened_neuron_band_pair",
        },
        "pooled_200_corpus_energy_vs_kl_norm": {
            **bootstrap_corr_stats(
                [r["corpus_energy_4band"][b] for r in rows for b in FOUR_BAND_17B],
                [r["kl_share_4band_norm"][b] for r in rows for b in FOUR_BAND_17B],
                n_resamples=n_bootstrap,
                seed=seed + 4,
            ),
            "bootstrap_unit": "flattened_neuron_band_pair",
        },
        "eval_energy_vs_kl_norm__gate_closed": bootstrap_corr_stats(
            [r["eval_energy_gate_closed"] for r in rows],
            [r["kl_share_gate_closed_norm"] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed + 5,
        ),
    }
    for i_b, b in enumerate(FOUR_BAND_17B):
        corrs[f"eval_energy_vs_kl_norm__{b}"] = bootstrap_corr_stats(
            [r["eval_energy_4band"][b] for r in rows],
            [r["kl_share_4band_norm"][b] for r in rows],
            n_resamples=n_bootstrap,
            seed=seed + 10 + i_b,
        )

    add_vals = [r["kl_4band_additivity_ratio"] for r in rows if math.isfinite(r["kl_4band_additivity_ratio"])]
    doc = {
        "model_id": MODEL_ID_17B,
        "compute_dtype": "torch.float32 (IEEE-754, allow_tf32=False)",
        "stage_ranges": [list(sr) for sr in stage_ranges],
        "n_neurons": len(rows),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "total_non_bos_tokens": total_non_bos,
        "peak_vram_allocated_gb": round(peak_a, 4),
        "peak_vram_reserved_gb": round(peak_r, 4),
        "max_vram_limit_gb": max_vram_gb,
        "vram_pass": (not is_cuda) or (peak_a < max_vram_gb and peak_r < max_vram_gb),
        "zero_ablation_sanity_kl": zero_sanity_kl,
        "kl_4band_additivity_ratio_median": float(np.median(add_vals)) if add_vals else float("nan"),
        "mean_kl_share_4band_norm": {
            b: float(np.mean([r["kl_share_4band_norm"][b] for r in rows])) for b in FOUR_BAND_17B
        },
        "mean_eval_energy_4band": {
            b: float(np.mean([r["eval_energy_4band"][b] for r in rows])) for b in FOUR_BAND_17B
        },
        "mean_corpus_energy_4band": {
            b: float(np.mean([r["corpus_energy_4band"][b] for r in rows])) for b in FOUR_BAND_17B
        },
        "median_strong20_rec": float(np.median([r["strong20_rec"] for r in rows])),
        "median_strong50_rec": float(np.median([r["strong50_rec"] for r in rows])),
        "correlations": corrs,
        "per_neuron": per_neuron_out,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = dirs["results"] / "energy_and_kl_17b.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[causal_17b:3B] COMPLETE in {doc['elapsed_sec']:.1f}s | "
        f"peak_vram={peak_a:.3f}/{peak_r:.3f}GB | additivity_med={doc['kl_4band_additivity_ratio_median']:.3f} | "
        f"E(>=20%) vs strong20_rec rho={corrs['corpus_energy_ge_20pct_max_vs_strong20_rec']['spearman_rho']:.3f} | "
        f"pooled_200(eval) rho={corrs['pooled_200_eval_energy_vs_kl_norm']['spearman_rho']:.3f}"
    )
    return doc


# ---------------------------------------------------------------------------
# Step 3C: Exact IEEE-754 FP32 Patching of Calibrated Simulator Ratings
# ---------------------------------------------------------------------------


def _extract_sim_ratings_by_window(sim_calls_path: Path, win_len: int = SIM_WIN_LEN_17B) -> dict[str, np.ndarray]:
    """Load `sim_calls_17b.jsonl` into `{window_id: float64[win_len]}` robustly across schema keys."""
    by_wid: dict[str, np.ndarray] = {}
    with sim_calls_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            wid = str(row["window_id"])
            raw = None
            for key in ("ratings", "predicted_ratings", "preds", "sim_ratings", "values"):
                if key in row and row[key] is not None:
                    raw = row[key]
                    break
            if raw is None:
                # Support sparse index->value dict (`sparse_ratings` or `predictions`)
                for key in ("sparse_ratings", "predictions"):
                    if isinstance(row.get(key), Mapping):
                        arr = np.zeros(win_len, dtype=np.float64)
                        for idx_s, val in row[key].items():
                            idx_i = int(idx_s)
                            if 0 <= idx_i < win_len:
                                arr[idx_i] = float(val)
                        by_wid[wid] = arr
                        raw = arr
                        break
            if raw is None:
                raise ValueError(f"Cannot extract ratings from sim_calls row for window_id={wid!r}")
            arr = np.asarray(raw, dtype=np.float64)
            if arr.shape != (win_len,):
                raise ValueError(f"window_id={wid}: expected shape ({win_len},), got {arr.shape}")
            by_wid[wid] = arr
    return by_wid


def run_17b_sim_patching(
    backbone: nn.Module,
    base_dir: Path | None = None,
    sim_calls_path: Path | None = None,
    compute_device: str | torch.device = "cuda",
    micro_batch_size: int = 16,
    kl_token_chunk: int = 512,
    max_vram_gb: float = 6.0,
    rating_scale: float = 5.0,
) -> dict[str, Any]:
    """Run Step 3C exact IEEE-754 FP32 patching of calibrated simulator ratings -> causal_scores_17b.json."""
    t0 = time.time()
    dirs = ensure_17b_causal_dirs(base_dir)
    dev = torch.device(compute_device)
    is_cuda = dev.type == "cuda"
    cpu_dev = torch.device("cpu")

    if sim_calls_path is None:
        sim_calls_path = dirs["results"] / "sim_calls_17b.jsonl"
    if not sim_calls_path.is_file():
        raise FileNotFoundError(f"Missing {sim_calls_path}; run the simulator calls step first.")

    prev_tf32_m = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    neurons_cat = json.loads((dirs["data"] / "neurons_17b.json").read_text(encoding="utf-8"))
    per_neuron_cat = neurons_cat["per_neuron"]
    sorted_nids = sorted(per_neuron_cat.keys(), key=lambda nid: int(per_neuron_cat[nid]["k"]))
    sim_meta = np.load(dirs["data"] / "sim_windows_meta_17b.npz")

    sample_mat = sim_meta[f"tok__{sorted_nids[0]}"]
    n_wins_per_neuron, win_len = int(sample_mat.shape[0]), int(sample_mat.shape[1])
    assert n_wins_per_neuron % micro_batch_size == 0
    n_batches = n_wins_per_neuron // micro_batch_size

    ratings_by_wid = _extract_sim_ratings_by_window(sim_calls_path, win_len=win_len)

    orig_dtype = next(backbone.parameters()).dtype
    lm_head_weight_cpu = backbone.embed_tokens.weight.detach()
    rotary_mod = getattr(backbone, "rotary_emb", None)
    norm_mod = getattr(backbone, "norm", None)
    n_layers = len(backbone.layers)
    stage_ranges = _compute_two_half_ranges(n_layers)
    n_stages = len(stage_ranges)
    layer_to_stage = {L: s_i for s_i, (a, b) in enumerate(stage_ranges) for L in range(a, b)}

    peak_a = peak_r = 0.0

    def _stage_mods(s_idx: int) -> list[nn.Module]:
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

    def _load(s_idx: int) -> list[nn.Module]:
        mods = _stage_mods(s_idx)
        for m in mods:
            if m is rotary_mod:
                m.to(device=dev)
            else:
                m.to(device=dev, dtype=torch.float32)
        return mods

    def _unload(mods: Sequence[nn.Module]) -> None:
        for m in mods:
            if m is rotary_mod:
                m.to(device=cpu_dev)
            else:
                m.to(device=cpu_dev, dtype=orig_dtype)
        _clear_cuda()

    def _check_vram() -> None:
        nonlocal peak_a, peak_r
        if is_cuda:
            torch.cuda.synchronize(dev)
            pa = torch.cuda.max_memory_allocated(dev) / (1024**3)
            pr = torch.cuda.max_memory_reserved(dev) / (1024**3)
            peak_a = max(peak_a, pa)
            peak_r = max(peak_r, pr)
            if pa >= max_vram_gb or pr >= max_vram_gb:
                raise ParityGateError(f"1.7B sim patching VRAM exceeded: {pa:.3f}/{pr:.3f}GB")

    def _per_window_kl(clean_h: torch.Tensor, abl_h: torch.Tensor, lm_w: torch.Tensor) -> np.ndarray:
        """Compute exact per-window mean KL(clean || ablated) over non-BOS tokens [1..T-1]."""
        B = int(clean_h.shape[0])
        Tm1 = int(clean_h.shape[1]) - 1
        c_flat = clean_h[:, 1:, :].reshape(-1, clean_h.shape[-1])
        a_flat = abl_h[:, 1:, :].reshape(-1, abl_h.shape[-1])
        tok_kls = torch.empty(B * Tm1, dtype=torch.float32, device=dev)
        for s0 in range(0, B * Tm1, kl_token_chunk):
            s1 = min(s0 + kl_token_chunk, B * Tm1)
            lc = F.linear(c_flat[s0:s1], lm_w).float()
            la = F.linear(a_flat[s0:s1], lm_w).float()
            log_pc = F.log_softmax(lc, dim=-1)
            log_pa = F.log_softmax(la, dim=-1)
            tok_kls[s0:s1] = (log_pc.exp() * (log_pc - log_pa)).sum(dim=-1).clamp_min(0.0)
            del lc, la, log_pc, log_pa
        return tok_kls.reshape(B, Tm1).mean(dim=1).cpu().numpy()

    patch_conds = ("zero", "mean", "strong20", "strong50", "sim_raw", "sim_cal_cv", "sim_cal_insample")
    per_neuron_scores: dict[str, dict[str, Any]] = {}
    zero_sanity_max = 0.0

    try:
        # Process neurons grouped by starting stage (Stage 0 first, then Stage 1) to minimize swaps
        # Actually, each neuron has its OWN 64 x 64 token windows, so we run all 50 neurons cleanly
        # in 2-stage passes:
        #   Pass 1 (Stage 0 ON GPU ONCE): compute clean_layer_in[L] (for L in Stage 0) AND
        #     clean_mid_h[nid] for ALL 50 neurons + clean_acts for Stage-0 neurons!
        #   Pass 2 (Stage 1 ON GPU ONCE): compute clean_layer_in[L] (for L in Stage 1),
        #     clean_final_h[nid] for ALL 50 neurons, clean_acts for Stage-1 neurons!
        hidden_size = int(backbone.embed_tokens.weight.shape[1])
        clean_in_by_nid: dict[str, torch.Tensor] = {}
        clean_mid_by_nid: dict[str, torch.Tensor] = {}
        clean_final_by_nid: dict[str, torch.Tensor] = {}
        clean_acts_by_nid: dict[str, np.ndarray] = {}

        # Stage 0 clean pass for ALL 50 neurons
        l0_a, l0_b = stage_ranges[0]
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        mods = _load(0)
        try:
            with torch.no_grad():
                for nid in sorted_nids:
                    rec = per_neuron_cat[nid]
                    L, n_idx = int(rec["layer"]), int(rec["neuron"])
                    in_stage0 = l0_a <= L < l0_b
                    toks = torch.from_numpy(sim_meta[f"tok__{nid}"].astype(np.int64))
                    mid_cpu = torch.empty((n_wins_per_neuron, win_len, hidden_size), dtype=torch.float32)
                    in_cpu = (
                        torch.empty((n_wins_per_neuron, win_len, hidden_size), dtype=torch.float32)
                        if in_stage0
                        else None
                    )
                    acts_np = np.empty((n_wins_per_neuron, win_len), dtype=np.float32) if in_stage0 else None

                    cap: dict[str, torch.Tensor] = {}
                    hh = None
                    if in_stage0:
                        def _cap_a(_m, args):
                            cap["a"] = args[0].detach()[:, :, n_idx]
                        hh = backbone.layers[L].mlp.down_proj.register_forward_pre_hook(_cap_a)
                    try:
                        for b_idx in range(n_batches):
                            s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                            h = backbone.embed_tokens(toks[s0:s1].to(dev))
                            pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                            for i in range(l0_a, l0_b):
                                if in_stage0 and i == L:
                                    assert in_cpu is not None
                                    in_cpu[s0:s1].copy_(h.detach().cpu())
                                h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                            mid_cpu[s0:s1].copy_(h.detach().cpu())
                            if in_stage0:
                                assert acts_np is not None
                                acts_np[s0:s1] = cap["a"].cpu().numpy()
                                cap.clear()
                            del h, pos_ids, pos_embs, mask_map
                    finally:
                        if hh is not None:
                            hh.remove()
                    clean_mid_by_nid[nid] = mid_cpu
                    if in_stage0:
                        assert in_cpu is not None and acts_np is not None
                        clean_in_by_nid[nid] = in_cpu
                        clean_acts_by_nid[nid] = acts_np
            _check_vram()
        finally:
            _unload(mods)

        # Stage 1 clean pass for ALL 50 neurons (if n_stages > 1)
        if n_stages > 1:
            l1_a, l1_b = stage_ranges[1]
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(dev)
            mods = _load(1)
            try:
                with torch.no_grad():
                    for nid in sorted_nids:
                        rec = per_neuron_cat[nid]
                        L, n_idx = int(rec["layer"]), int(rec["neuron"])
                        in_stage1 = l1_a <= L < l1_b
                        mid_cpu = clean_mid_by_nid[nid]
                        fin_cpu = torch.empty((n_wins_per_neuron, win_len, hidden_size), dtype=torch.float32)
                        in_cpu = (
                            torch.empty((n_wins_per_neuron, win_len, hidden_size), dtype=torch.float32)
                            if in_stage1
                            else None
                        )
                        acts_np = np.empty((n_wins_per_neuron, win_len), dtype=np.float32) if in_stage1 else None

                        cap = {}
                        hh = None
                        if in_stage1:
                            def _cap_a1(_m, args):
                                cap["a"] = args[0].detach()[:, :, n_idx]
                            hh = backbone.layers[L].mlp.down_proj.register_forward_pre_hook(_cap_a1)
                        try:
                            for b_idx in range(n_batches):
                                s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                                h = mid_cpu[s0:s1].to(dev, dtype=torch.float32)
                                pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                for i in range(l1_a, l1_b):
                                    if in_stage1 and i == L:
                                        assert in_cpu is not None
                                        in_cpu[s0:s1].copy_(h.detach().cpu())
                                    h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                                if norm_mod is not None:
                                    h = norm_mod(h)
                                fin_cpu[s0:s1].copy_(h.detach().cpu())
                                if in_stage1:
                                    assert acts_np is not None
                                    acts_np[s0:s1] = cap["a"].cpu().numpy()
                                    cap.clear()
                                del h, pos_ids, pos_embs, mask_map
                        finally:
                            if hh is not None:
                                hh.remove()
                        clean_final_by_nid[nid] = fin_cpu
                        if in_stage1:
                            assert in_cpu is not None and acts_np is not None
                            clean_in_by_nid[nid] = in_cpu
                            clean_acts_by_nid[nid] = acts_np
                _check_vram()
            finally:
                _unload(mods)
        else:
            for nid in sorted_nids:
                clean_final_by_nid[nid] = clean_mid_by_nid[nid]
        del clean_mid_by_nid

        # Build calibrated replacement arrays `[64, 64]` for all 50 neurons x 7 patch conditions
        patch_arrays_by_nid: dict[str, dict[str, np.ndarray]] = {}
        cal_meta_by_nid: dict[str, dict[str, Any]] = {}

        for nid in sorted_nids:
            rec = per_neuron_cat[nid]
            a_true = clean_acts_by_nid[nid].astype(np.float64)
            m_abs = float(rec["max_abs_act"])
            mean_a = float(rec["mean_act"])

            r_mat = np.stack(
                [ratings_by_wid[f"{nid}__w{wi:02d}"] for wi in range(n_wins_per_neuron)],
                axis=0,
            )
            sim_raw = (r_mat / float(rating_scale)) * m_abs
            sim_cal_cv = calibrate_ratings_2fold_cv(r_mat, a_true, exclude_pos0=True)
            sim_cal_in, alpha_in, beta_in = calibrate_ratings_insample(r_mat, a_true, exclude_pos0=True)

            # Keep pos=0 untouched (clean) across all patch conditions!
            def _keep_pos0(arr: np.ndarray) -> np.ndarray:
                out = arr.astype(np.float32, copy=True)
                out[:, 0] = a_true[:, 0].astype(np.float32)
                return out

            patch_arrays_by_nid[nid] = {
                "zero": _keep_pos0(np.zeros_like(a_true)),
                "mean": _keep_pos0(np.full_like(a_true, mean_a)),
                "strong20": _keep_pos0(a_true * (np.abs(a_true) >= 0.20 * m_abs)),
                "strong50": _keep_pos0(a_true * (np.abs(a_true) >= 0.50 * m_abs)),
                "sim_raw": _keep_pos0(sim_raw),
                "sim_cal_cv": _keep_pos0(sim_cal_cv),
                "sim_cal_insample": _keep_pos0(sim_cal_in),
            }

            # Correlation between raw ratings and true activations (non-BOS tokens)
            r_flat = r_mat[:, 1:].ravel()
            a_flat = a_true[:, 1:].ravel()
            cv_flat = sim_cal_cv[:, 1:].ravel()
            corr_raw = float(np.corrcoef(r_flat, a_flat)[0, 1]) if np.std(r_flat) > 1e-12 and np.std(a_flat) > 1e-12 else 0.0
            corr_cv = float(np.corrcoef(cv_flat, a_flat)[0, 1]) if np.std(cv_flat) > 1e-12 and np.std(a_flat) > 1e-12 else 0.0
            cal_meta_by_nid[nid] = {
                "corr": corr_raw,
                "corr_cal_cv": corr_cv,
                "alpha_insample": alpha_in,
                "beta_insample": beta_in,
                "ratings_nonzero_frac": float(np.mean(np.abs(r_flat) > 1e-9)),
            }

        # Run exact FP32 patching per starting stage in groups of 5 neurons
        nids_by_stage: dict[int, list[str]] = defaultdict(list)
        for nid in sorted_nids:
            nids_by_stage[layer_to_stage[int(per_neuron_cat[nid]["layer"])]].append(nid)

        per_win_kl_by_nid: dict[str, dict[str, np.ndarray]] = {nid: {} for nid in sorted_nids}
        done = 0
        for start_s in sorted(nids_by_stage.keys(), reverse=True):
            s_nids = nids_by_stage[start_s]
            for g0 in range(0, len(s_nids), 5):
                group_nids = s_nids[g0 : g0 + 5]
                pending_h: dict[tuple[str, str], torch.Tensor] = {}

                l_a, l_b = stage_ranges[start_s]
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats(dev)
                mods = _load(start_s)
                try:
                    for nid in group_nids:
                        rec = per_neuron_cat[nid]
                        L, n_idx = int(rec["layer"]), int(rec["neuron"])
                        h_in_cpu = clean_in_by_nid[nid]
                        state: dict[str, Any] = {"vals": None}

                        def _patch_hook(_mod: nn.Module, args: tuple[Any, ...]):
                            x = args[0].clone()
                            x[:, :, n_idx] = state["vals"]
                            return (x,)

                        hh = backbone.layers[L].mlp.down_proj.register_forward_pre_hook(_patch_hook)
                        try:
                            with torch.no_grad():
                                # Also run clean identity sanity check on first neuron of run
                                conds_to_run = ("__clean_sanity__",) + patch_conds if done == 0 and nid == group_nids[0] else patch_conds
                                for cond in conds_to_run:
                                    arr = (
                                        clean_acts_by_nid[nid]
                                        if cond == "__clean_sanity__"
                                        else patch_arrays_by_nid[nid][cond]
                                    )
                                    out_cpu = torch.empty((n_wins_per_neuron, win_len, hidden_size), dtype=torch.float32)
                                    for b_idx in range(n_batches):
                                        s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                                        h = h_in_cpu[s0:s1].to(dev, dtype=torch.float32)
                                        state["vals"] = torch.from_numpy(arr[s0:s1]).to(dev, dtype=torch.float32)
                                        pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                        for i in range(L, l_b):
                                            h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                                        if start_s == n_stages - 1 and norm_mod is not None:
                                            h = norm_mod(h)
                                        out_cpu[s0:s1].copy_(h.detach().cpu())
                                        del h, pos_ids, pos_embs, mask_map
                                        state["vals"] = None
                                    pending_h[(nid, cond)] = out_cpu
                        finally:
                            hh.remove()
                    _check_vram()
                finally:
                    _unload(mods)

                for next_s in range(start_s + 1, n_stages):
                    nl_a, nl_b = stage_ranges[next_s]
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats(dev)
                    mods = _load(next_s)
                    try:
                        with torch.no_grad():
                            for key_nc, h_cpu in list(pending_h.items()):
                                out_cpu = torch.empty_like(h_cpu)
                                for b_idx in range(n_batches):
                                    s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                                    h = h_cpu[s0:s1].to(dev, dtype=torch.float32)
                                    pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                    for i in range(nl_a, nl_b):
                                        h = _call_decoder_layer(backbone, i, h, mask_map, pos_embs, pos_ids)
                                    if next_s == n_stages - 1 and norm_mod is not None:
                                        h = norm_mod(h)
                                    out_cpu[s0:s1].copy_(h.detach().cpu())
                                    del h, pos_ids, pos_embs, mask_map
                                pending_h[key_nc] = out_cpu
                        _check_vram()
                    finally:
                        _unload(mods)

                # Score per-window exact float32 KL via standalone lm_head
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats(dev)
                lm_w = lm_head_weight_cpu.to(device=dev, dtype=torch.float32)
                try:
                    with torch.no_grad():
                        for (nid, cond), h_fin_cpu in pending_h.items():
                            win_kls = np.empty(n_wins_per_neuron, dtype=np.float64)
                            c_fin_all = clean_final_by_nid[nid]
                            for b_idx in range(n_batches):
                                s0, s1 = b_idx * micro_batch_size, (b_idx + 1) * micro_batch_size
                                win_kls[s0:s1] = _per_window_kl(
                                    c_fin_all[s0:s1].to(dev, dtype=torch.float32),
                                    h_fin_cpu[s0:s1].to(dev, dtype=torch.float32),
                                    lm_w,
                                )
                            if cond == "__clean_sanity__":
                                zero_sanity_max = max(zero_sanity_max, float(np.max(win_kls)))
                                if zero_sanity_max > 1e-10:
                                    raise ParityGateError(f"1.7B sim patch clean sanity failed: {zero_sanity_max:.3e}")
                            else:
                                per_win_kl_by_nid[nid][cond] = win_kls
                    _check_vram()
                finally:
                    del lm_w
                    _clear_cuda()

                del pending_h
                done += len(group_nids)
                print(
                    f"[causal_17b:3C] Stage {start_s} sim-patch group done ({done}/{len(sorted_nids)} neurons, "
                    f"{time.time() - t0:.1f}s, peak_vram={peak_a:.2f}/{peak_r:.2f}GB)"
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_m
        torch.backends.cudnn.allow_tf32 = prev_tf32_c

    # Aggregate weighted ordinary-text and active-window causal recovery scores
    for nid in sorted_nids:
        rec = per_neuron_cat[nid]
        w_ord = sim_meta[f"w__{nid}"].astype(np.float64)
        is_act = sim_meta[f"act__{nid}"].astype(bool)
        is_ord_strat = sim_meta[f"strat__{nid}"] == 0
        wkls = per_win_kl_by_nid[nid]

        kl_w = {c: float(np.sum(w_ord * wkls[c])) for c in patch_conds}
        kl_rand = {c: float(np.mean(wkls[c][is_ord_strat])) for c in patch_conds}
        act_mask = is_act if np.any(is_act) else (sim_meta[f"strat__{nid}"] == 1)
        kl_act = {c: float(np.mean(wkls[c][act_mask])) for c in patch_conds}

        def _rec(kl_c: float, kl_base: float) -> float:
            return float(1.0 - kl_c / kl_base) if kl_base > 0.0 else float("nan")

        per_neuron_scores[nid] = {
            "k": int(rec["k"]),
            "neuron_id": nid,
            "layer": int(rec["layer"]),
            "neuron": int(rec["neuron"]),
            "in_pilot": bool(rec["in_pilot"]),
            "sim_cal_rec": _rec(kl_w["sim_cal_cv"], kl_w["mean"]),
            "sim_cal_rec_cv": _rec(kl_w["sim_cal_cv"], kl_w["mean"]),
            "sim_cal_rec_insample": _rec(kl_w["sim_cal_insample"], kl_w["mean"]),
            "sim_cal_rec_random": _rec(kl_rand["sim_cal_cv"], kl_rand["mean"]),
            "sim_cal_rec_active": _rec(kl_act["sim_cal_cv"], kl_act["mean"]),
            "sim_cal_rec_active_insample": _rec(kl_act["sim_cal_insample"], kl_act["mean"]),
            "sim_raw_rec": _rec(kl_w["sim_raw"], kl_w["mean"]),
            "sim_raw_rec_active": _rec(kl_act["sim_raw"], kl_act["mean"]),
            "strong20_rec_eval64": _rec(kl_w["strong20"], kl_w["mean"]),
            "strong50_rec_eval64": _rec(kl_w["strong50"], kl_w["mean"]),
            "zero_rec": _rec(kl_w["zero"], kl_w["mean"]),
            "kl_weighted_ordinary": kl_w,
            "kl_random_ordinary": kl_rand,
            "kl_active": kl_act,
            **cal_meta_by_nid[nid],
        }

    srows = [per_neuron_scores[nid] for nid in sorted_nids]
    sim_cals = [r["sim_cal_rec"] for r in srows]
    sim_acts = [r["sim_cal_rec_active"] for r in srows]
    doc = {
        "model_id": MODEL_ID_17B,
        "compute_dtype": "torch.float32 (IEEE-754, allow_tf32=False)",
        "calibration_primary": "2-fold even/odd window cross-fit OLS (exact Ulto85 calibrate())",
        "n_neurons": len(srows),
        "n_windows_per_neuron": n_wins_per_neuron,
        "win_len": win_len,
        "peak_vram_allocated_gb": round(peak_a, 4),
        "peak_vram_reserved_gb": round(peak_r, 4),
        "max_vram_limit_gb": max_vram_gb,
        "vram_pass": (not is_cuda) or (peak_a < max_vram_gb and peak_r < max_vram_gb),
        "zero_ablation_sanity_kl": zero_sanity_max,
        "summary": {
            "median_sim_cal_rec": float(np.median(sim_cals)),
            "mean_sim_cal_rec": float(np.mean(sim_cals)),
            "median_sim_cal_rec_insample": float(np.median([r["sim_cal_rec_insample"] for r in srows])),
            "median_sim_cal_rec_active": float(np.median(sim_acts)),
            "mean_sim_cal_rec_active": float(np.mean(sim_acts)),
            "median_corr": float(np.median([r["corr"] for r in srows])),
            "pass_sim_cal_rec_gt_0": sum(1 for v in sim_cals if v > 0.0),
            "pass_sim_cal_rec_ge_0_25": sum(1 for v in sim_cals if v >= 0.25),
            "pass_sim_cal_rec_ge_0_50": sum(1 for v in sim_cals if v >= 0.50),
            "pass_sim_cal_rec_active_ge_0_50": sum(1 for v in sim_acts if v >= 0.50),
        },
        "per_neuron": per_neuron_scores,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = dirs["results"] / "causal_scores_17b.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[causal_17b:3C] COMPLETE in {doc['elapsed_sec']:.1f}s | "
        f"med sim_cal_rec={doc['summary']['median_sim_cal_rec']:.3f} "
        f"(>0:{doc['summary']['pass_sim_cal_rec_gt_0']}/50, >=0.25:{doc['summary']['pass_sim_cal_rec_ge_0_25']}/50, "
        f">=0.50:{doc['summary']['pass_sim_cal_rec_ge_0_50']}/50) | "
        f"med active={doc['summary']['median_sim_cal_rec_active']:.3f} (>=0.50:{doc['summary']['pass_sim_cal_rec_active_ge_0_50']}/50)"
    )
    return doc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-1.7B causal patching & band KL steps.")
    parser.add_argument("step", choices=["prepare", "baseline-kl", "patch-sims"])
    parser.add_argument("--model-id", default=MODEL_ID_17B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--kl-token-chunk", type=int, default=512)
    parser.add_argument("--max-vram-gb", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=SEED_17B)
    args = parser.parse_args(argv)

    if args.step == "prepare":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        prepare_17b_inputs(tokenizer=tok, seed=args.seed)
    elif args.step == "baseline-kl":
        backbone, _ = load_qwen4b_backbone_cpu(model_id=args.model_id, dtype=torch.bfloat16)
        run_17b_baseline_kl(
            backbone=backbone,
            compute_device=args.device,
            micro_batch_size=args.micro_batch_size,
            kl_token_chunk=args.kl_token_chunk,
            max_vram_gb=args.max_vram_gb,
            seed=args.seed,
        )
    elif args.step == "patch-sims":
        backbone, _ = load_qwen4b_backbone_cpu(model_id=args.model_id, dtype=torch.bfloat16)
        run_17b_sim_patching(
            backbone=backbone,
            compute_device=args.device,
            micro_batch_size=args.micro_batch_size,
            kl_token_chunk=args.kl_token_chunk,
            max_vram_gb=args.max_vram_gb,
        )


if __name__ == "__main__":
    main()
