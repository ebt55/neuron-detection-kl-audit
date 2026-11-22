"""Exact IEEE-754 float32 per-neuron band mean-ablation KL validation on Qwen/Qwen3-4B.

Why IEEE-754 float32 (`allow_tf32=False`) with fixed microbatching is mandatory:
- Single-neuron mean-ablation KL on Qwen3-4B is ~1e-6 to 2e-5 (matching Ulto85's `e3_scores.json`,
  which loaded Qwen3-4B in `dtype=torch.float32` in `src/common.py`).
- In `bfloat16` (7-bit mantissa, 1 ULP = 0.78% relative), single-neuron perturbation flips random
  ULPs and induces a ~4.5e-4 KL quantization noise floor (100x larger than true ~3e-6 KL!).
- Running our 4-stage GPU-swapped pipeline in exact IEEE-754 `float32` (`allow_tf32=False`, casting
  the active 8-10 layer stage from CPU `bf16` -> GPU `float32` -> CPU `bf16`, which is bit-exact
  lossless for `bf16` checkpoints!) keeps peak VRAM <= 4.9 GB (cleanly < 6.5 GB) while driving the
  non-ablation KL noise floor to EXACT `0.00e+00` (with fixed `[8, 128]` microbatching)!

Validates zero-cost activation-energy shares (`EnergyShare(B)`) against true downstream output
`KL(clean || ablated)` across 60 stratified neurons (6/layer x 10 layers, including `info`, `PCR`,
and `insurance`) over `64 x 128 = 8,192` disjoint FineWeb tokens (`rows 65,000+`).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen4b_eval import (
    HOOK_LAYERS_4B,
    MODEL_ID_4B,
    NAMED_NEURONS_4B,
    SEED_4B,
    ensure_4b_dirs,
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

KL_MIN_ROW_IDX_4B: int = 65_000  # Strictly disjoint from Ulto85 (0..60k) & extraction (60k..64.8k)
KL_N_SEQS_4B: int = 64
KL_SEQ_LEN_4B: int = 128  # 64 * 128 = 8,192 tokens
KL_NEURONS_PER_LAYER: int = 6

ABLATION_CONDITIONS: tuple[str, ...] = (
    "full",
    "p99.9+",
    "p97_99.9",
    "p75_97",
    "lt_p75",
    "ge_20pct_max",
    "lt_20pct_max",
    "gate_open",
    "gate_closed",
)
FOUR_BAND_CONDS: tuple[str, ...] = ("p99.9+", "p97_99.9", "p75_97", "lt_p75")


def _compute_stage_ranges(n_layers: int) -> list[tuple[int, int]]:
    """Partition `n_layers` into balanced FP32 GPU stages (<= 4.5 GiB weights each on 36L Qwen3-4B)."""
    if n_layers == 36:
        # Stage 0 (embed+8L = 4.46 GiB), Stage 1 (10L = 3.76 GiB),
        # Stage 2 (10L = 3.76 GiB), Stage 3 (8L+norm = 3.01 GiB); lm_head scored separately (1.45 GiB)
        return [(0, 8), (8, 18), (18, 28), (28, 36)]
    n_stages = min(4, max(1, n_layers))
    edges = np.linspace(0, n_layers, n_stages + 1, dtype=int).tolist()
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_stages) if edges[i] < edges[i + 1]]


def select_stratified_kl_neurons(
    gap_doc: Mapping[str, Any],
    layers: Sequence[int] = HOOK_LAYERS_4B,
    per_layer: int = KL_NEURONS_PER_LAYER,
    must_include_ks: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Select `per_layer` neurons per layer stratified across `sim_cal_rec`, including named neurons."""
    if must_include_ks is None:
        must_include_ks = [int(v["k"]) for v in NAMED_NEURONS_4B.values()]
    must_set = set(int(k) for k in must_include_ks)

    per_neuron = gap_doc["per_neuron"]
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for k_str, rec in per_neuron.items():
        by_layer[int(rec["layer"])].append(dict(rec))

    selected: list[dict[str, Any]] = []
    for L in sorted(int(x) for x in layers):
        pool = by_layer.get(L, [])
        if len(pool) < per_layer:
            raise ValueError(f"Layer {L} has only {len(pool)} neurons, need {per_layer}")

        forced = sorted([r for r in pool if int(r["k"]) in must_set], key=lambda r: int(r["k"]))
        rem = [r for r in pool if int(r["k"]) not in must_set]
        rem.sort(key=lambda r: (float(r.get("sim_cal_rec", 0.0)), int(r["k"])))

        n_need = per_layer - len(forced)
        if n_need < 0:
            raise ValueError(f"Layer {L}: more forced neurons ({len(forced)}) than per_layer ({per_layer})")

        picked_rem: list[dict[str, Any]] = []
        if n_need > 0:
            if n_need == 1:
                idxs = [len(rem) // 2]
            else:
                raw_idxs = np.linspace(0, len(rem) - 1, n_need)
                idxs = []
                used: set[int] = set()
                for val in raw_idxs:
                    cand = int(round(float(val)))
                    while cand in used and cand + 1 < len(rem):
                        cand += 1
                    while cand in used and cand - 1 >= 0:
                        cand -= 1
                    used.add(cand)
                    idxs.append(cand)
            picked_rem = [rem[i] for i in sorted(idxs)]

        selected.extend(sorted(forced + picked_rem, key=lambda r: int(r["k"])))

    return selected


def build_neuron_ablation_masks(
    acts_2d: np.ndarray,
    quads_2d: np.ndarray,
    neuron_gap_rec: Mapping[str, Any],
    per_pole_gap: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Build exact boolean `[N, T]` ablation masks for all 9 conditions (strictly non-BOS `pos >= 1`).

    Setting `mask[:, 0] = False` across all 9 conditions guarantees:
      - `pos=0` attention sink is never ablated into downstream tokens, AND
      - `full` == exact disjoint union of (`p99.9+`, `p97_99.9`, `p75_97`, `lt_p75`)
        == exact disjoint union of (`ge_20pct_max`, `lt_20pct_max`)
        == exact disjoint union of (`gate_open`, `gate_closed`).
    """
    a = acts_2d.astype(np.float64)
    abs_a = np.abs(a)
    m_abs = float(neuron_gap_rec["max_abs_act"])

    active_pids = list(neuron_gap_rec["active_pole_ids"])

    def _pole_4band_codes(pid: str) -> np.ndarray:
        pm = per_pole_gap[pid]
        sign = int(pm["sign"])
        d = float(sign) * a
        p75 = float(pm["p75"])
        p97 = float(pm["p97"])
        p99_9 = float(pm["p99.9"])
        c = np.full(a.shape, 3, dtype=np.uint8)
        c[(d >= p75) & (d < p97)] = 2
        c[(d >= p97) & (d < p99_9)] = 1
        c[d >= p99_9] = 0
        return c

    if len(active_pids) == 1:
        b4 = _pole_4band_codes(active_pids[0])
    else:
        pid_pos = [pid for pid in active_pids if int(per_pole_gap[pid]["sign"]) > 0][0]
        pid_neg = [pid for pid in active_pids if int(per_pole_gap[pid]["sign"]) < 0][0]
        b4 = np.where(a >= 0.0, _pole_4band_codes(pid_pos), _pole_4band_codes(pid_neg))

    gate_open = quads_2d < 2
    ge_20 = abs_a >= (0.20 * m_abs)

    non_bos = np.ones(a.shape, dtype=bool)
    non_bos[:, 0] = False

    return {
        "full": non_bos.copy(),
        "p99.9+": non_bos & (b4 == 0),
        "p97_99.9": non_bos & (b4 == 1),
        "p75_97": non_bos & (b4 == 2),
        "lt_p75": non_bos & (b4 == 3),
        "ge_20pct_max": non_bos & ge_20,
        "lt_20pct_max": non_bos & (~ge_20),
        "gate_open": non_bos & gate_open,
        "gate_closed": non_bos & (~gate_open),
    }


def _compute_chunked_kl_sum(
    clean_final_h: torch.Tensor,
    abl_final_h: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_chunk: int = 512,
) -> float:
    """Compute exact float32 sum of KL(p_clean || p_abl) over non-BOS positions (pos >= 1)."""
    c_flat = clean_final_h[:, 1:, :].reshape(-1, clean_final_h.shape[-1])
    a_flat = abl_final_h[:, 1:, :].reshape(-1, abl_final_h.shape[-1])
    n_tok = int(c_flat.shape[0])
    kl_total = 0.0
    for s0 in range(0, n_tok, token_chunk):
        s1 = min(s0 + token_chunk, n_tok)
        lc = F.linear(c_flat[s0:s1], lm_head_weight).float()
        la = F.linear(a_flat[s0:s1], lm_head_weight).float()
        log_pc = F.log_softmax(lc, dim=-1)
        log_pa = F.log_softmax(la, dim=-1)
        pc = log_pc.exp()
        kl_tok = (pc * (log_pc - log_pa)).sum(dim=-1).clamp_min(0.0)
        kl_total += float(kl_tok.sum().item())
        del lc, la, log_pc, log_pa, pc, kl_tok
    return kl_total


def run_band_kl_validation(
    backbone: nn.Module,
    input_ids_cpu: torch.Tensor,
    gap_doc: Mapping[str, Any],
    selected_neurons: Sequence[Mapping[str, Any]],
    lm_head_weight_cpu: torch.Tensor | None = None,
    compute_device: str | torch.device = "cuda",
    micro_batch_size: int = 8,
    kl_token_chunk: int = 512,
    max_vram_gb: float = 6.5,
    neurons_per_group: int = 3,
    n_bootstrap: int = 1000,
    seed: int = SEED_4B,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run exact IEEE-754 float32 4-stage mean-ablation band KL across selected neurons."""
    t0 = time.time()
    dev = torch.device(compute_device)
    is_cuda = dev.type == "cuda"
    cpu_dev = torch.device("cpu")

    # Enforce strict IEEE-754 float32 (NO TF32 10-bit rounding on Ampere!)
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
            f"n_seqs ({n_seqs}) must be divisible by micro_batch_size ({micro_batch_size}) "
            "for exact fixed-batch zero-noise parity."
        )
    n_batches = n_seqs // micro_batch_size
    hidden_size = int(backbone.embed_tokens.weight.shape[1])
    total_non_bos_tokens = n_seqs * (seq_len - 1)

    if lm_head_weight_cpu is None:
        lm_head_weight_cpu = backbone.embed_tokens.weight.detach()

    neurons_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in selected_neurons:
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
                # Never truncate rotary_emb.inv_freq to bf16; preserve its native float32 precision!
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
                    f"VRAM limit exceeded during FP32 band_kl_4b: alloc={alloc:.3f}GB, "
                    f"res={resv:.3f}GB >= {max_vram_gb}GB"
                )

    try:
        # ------------------------------------------------------------------
        # STEP 1: Clean 4-stage IEEE-754 FP32 pass caching layer inputs,
        #         clean_final_h (FP32), and clean neuron acts + SwiGLU quads
        # ------------------------------------------------------------------
        clean_layer_in: dict[int, torch.Tensor] = {
            L: torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
            for L in hook_layers
        }
        stage_h_cpu = torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
        clean_final_cpu = torch.empty((n_seqs, seq_len, hidden_size), dtype=torch.float32, device=cpu_dev)
        clean_acts_by_k: dict[int, np.ndarray] = {
            int(r["k"]): np.empty((n_seqs, seq_len), dtype=np.float32) for r in selected_neurons
        }
        clean_quads_by_k: dict[int, np.ndarray] = {
            int(r["k"]): np.empty((n_seqs, seq_len), dtype=np.uint8) for r in selected_neurons
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
        # STEP 2: Exact zero-mask re-run sanity check on first stage's first layer
        #         (verifies fixed-batch FP32 re-run gives EXACT 0.00e+00 KL!)
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
            f"[band_kl_4b] Exact IEEE-754 FP32 clean pass + zero-ablation re-run gate PASSED: "
            f"peak_vram_alloc={peak_vram_alloc:.3f}GB, peak_vram_res={peak_vram_res:.3f}GB, "
            f"zero_rerun_kl={zero_ablation_sanity_kl:.2e} ({time.time() - t0:.1f}s)"
        )

        # ------------------------------------------------------------------
        # STEP 3: Build ablation masks and run 4-stage FP32 mean ablations
        #         grouped by starting stage (fixed [micro_batch_size, seq_len]
        #         batches, skipping batches with 0 masked tokens -> exact 0.0 KL!)
        # ------------------------------------------------------------------
        per_pole_gap = gap_doc["per_pole"]
        masks_by_k: dict[int, dict[str, np.ndarray]] = {}
        kl_results_by_k: dict[int, dict[str, float]] = {int(r["k"]): {} for r in selected_neurons}
        n_abl_tok_by_k: dict[int, dict[str, int]] = {int(r["k"]): {} for r in selected_neurons}

        for rec in selected_neurons:
            k = int(rec["k"])
            masks_by_k[k] = build_neuron_ablation_masks(
                clean_acts_by_k[k], clean_quads_by_k[k], rec, per_pole_gap
            )
            for cond in ABLATION_CONDITIONS:
                n_abl_tok_by_k[k][cond] = int(np.sum(masks_by_k[k][cond][:, 1:]))

        neurons_by_start_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for L in hook_layers:
            neurons_by_start_stage[layer_to_stage[L]].extend(neurons_by_layer[L])

        done_neurons = 0
        # Process from deepest stage (Stage 3 -> Stage 0) so early progress prints fast
        for start_stage in sorted(neurons_by_start_stage.keys(), reverse=True):
            stage_recs = neurons_by_start_stage[start_stage]
            for g0 in range(0, len(stage_recs), neurons_per_group):
                group = stage_recs[g0 : g0 + neurons_per_group]

                # Track active fixed microbatches per (k, cond):
                # pending[(k, cond)] = list of (b_idx, changed_local_seq_idxs, h_cpu[B, T, D])
                pending: dict[tuple[int, str], list[tuple[int, list[int], torch.Tensor]]] = {}

                # Stage `start_stage`: apply exact mean-ablation pre-hook at neuron's layer L
                l_start, l_end = stage_ranges[start_stage]
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats(dev)
                mods = _load_stage_fp32(start_stage)
                try:
                    for rec in group:
                        k = int(rec["k"])
                        L = int(rec["layer"])
                        n_idx = int(rec["neuron"])
                        mean_scalar = torch.tensor(float(rec["mean_act"]), dtype=torch.float32, device=dev)
                        hook_state: dict[str, Any] = {"bm": None}

                        def _pre_hook(_mod: nn.Module, args: tuple[Any, ...]):
                            x = args[0].clone()
                            x[:, :, n_idx] = torch.where(hook_state["bm"], mean_scalar, x[:, :, n_idx])
                            return (x,)

                        h_handle = backbone.layers[L].mlp.down_proj.register_forward_pre_hook(_pre_hook)
                        try:
                            with torch.no_grad():
                                for cond in ABLATION_CONDITIONS:
                                    m2d = masks_by_k[k][cond]
                                    items: list[tuple[int, list[int], torch.Tensor]] = []
                                    for b_idx in range(n_batches):
                                        s0 = b_idx * micro_batch_size
                                        s1 = s0 + micro_batch_size
                                        bm_np = m2d[s0:s1]
                                        if not bm_np.any():
                                            continue
                                        changed_local = np.flatnonzero(bm_np.any(axis=1)).tolist()
                                        h = clean_layer_in[L][s0:s1].to(dev, dtype=torch.float32)
                                        hook_state["bm"] = torch.from_numpy(bm_np).to(dev)
                                        pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                        for i in range(L, l_end):
                                            h = _call_decoder_layer(
                                                backbone, i, h, mask_map, pos_embs, pos_ids
                                            )
                                        if start_stage == n_stages - 1 and norm_mod is not None:
                                            h = norm_mod(h)
                                        items.append((b_idx, changed_local, h.detach().cpu()))
                                        del h, pos_ids, pos_embs, mask_map
                                        hook_state["bm"] = None
                                    if not items:
                                        kl_results_by_k[k][cond] = 0.0
                                    else:
                                        pending[(k, cond)] = items
                        finally:
                            h_handle.remove()
                    _record_vram()
                finally:
                    _unload_stage(mods)

                # Propagate pending batches through remaining stages `start_stage + 1 .. n_stages - 1`
                for next_s in range(start_stage + 1, n_stages):
                    nl_start, nl_end = stage_ranges[next_s]
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats(dev)
                    mods = _load_stage_fp32(next_s)
                    try:
                        with torch.no_grad():
                            for key_kc, items in pending.items():
                                new_items: list[tuple[int, list[int], torch.Tensor]] = []
                                for b_idx, changed_local, h_cpu in items:
                                    h = h_cpu.to(dev, dtype=torch.float32)
                                    pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, h)
                                    for i in range(nl_start, nl_end):
                                        h = _call_decoder_layer(
                                            backbone, i, h, mask_map, pos_embs, pos_ids
                                        )
                                    if next_s == n_stages - 1 and norm_mod is not None:
                                        h = norm_mod(h)
                                    new_items.append((b_idx, changed_local, h.detach().cpu()))
                                    del h, pos_ids, pos_embs, mask_map
                                pending[key_kc] = new_items
                        _record_vram()
                    finally:
                        _unload_stage(mods)

                # Score exact float32 KL using standalone lm_head on ONLY changed sequences!
                if pending:
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats(dev)
                    lm_w_gpu = lm_head_weight_cpu.to(device=dev, dtype=torch.float32)
                    try:
                        with torch.no_grad():
                            for (k, cond), items in pending.items():
                                kl_sum = 0.0
                                for b_idx, changed_local, h_final_cpu in items:
                                    s0 = b_idx * micro_batch_size
                                    # Only score sequences in the batch that had >=1 ablated token
                                    # (unchanged sequences in the batch have exact 0.0 KL!)
                                    abs_changed = [s0 + li for li in changed_local]
                                    c_h = clean_final_cpu[abs_changed].to(dev, dtype=torch.float32)
                                    a_h = h_final_cpu[changed_local].to(dev, dtype=torch.float32)
                                    kl_sum += _compute_chunked_kl_sum(
                                        c_h, a_h, lm_w_gpu, token_chunk=kl_token_chunk
                                    )
                                    del c_h, a_h
                                kl_results_by_k[k][cond] = float(kl_sum / float(total_non_bos_tokens))
                        _record_vram()
                    finally:
                        del lm_w_gpu
                        _clear_cuda()

                del pending
                done_neurons += len(group)
                print(
                    f"[band_kl_4b] Stage {start_stage} group done "
                    f"({done_neurons}/{len(selected_neurons)} neurons, {time.time() - t0:.1f}s, "
                    f"peak_vram_alloc={peak_vram_alloc:.2f}GB, peak_vram_res={peak_vram_res:.2f}GB)"
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn

    # ------------------------------------------------------------------
    # STEP 4: Assemble per-neuron KL shares & EnergyShare vs KLShare correlations
    # ------------------------------------------------------------------
    per_neuron_kl: dict[str, dict[str, Any]] = {}
    for rec in sorted(selected_neurons, key=lambda r: int(r["k"])):
        k = int(rec["k"])
        kls = kl_results_by_k[k]
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

        # Exact eval-set (8,128 non-BOS tokens) L2 energy shares
        a_eval = clean_acts_by_k[k][:, 1:].astype(np.float64)
        mean_a = float(rec["mean_act"])
        l2_eval = (a_eval - mean_a) ** 2
        tot_eval_l2 = float(np.sum(l2_eval))
        denom_eval = tot_eval_l2 if tot_eval_l2 > 0.0 else 1.0
        eval_energy_4b = {
            b: float(np.sum(l2_eval[masks_by_k[k][b][:, 1:]]) / denom_eval) for b in FOUR_BAND_CONDS
        }
        eval_energy_ge20 = float(np.sum(l2_eval[masks_by_k[k]["ge_20pct_max"][:, 1:]]) / denom_eval)
        eval_energy_gc = float(np.sum(l2_eval[masks_by_k[k]["gate_closed"][:, 1:]]) / denom_eval)

        corpus_4b = rec["energy_l2"]["bands_4"]
        per_neuron_kl[str(k)] = {
            "k": k,
            "neuron_id": str(rec["neuron_id"]),
            "layer": int(rec["layer"]),
            "neuron": int(rec["neuron"]),
            "named_neuron": rec.get("named_neuron"),
            "failure_bucket": rec.get("failure_bucket"),
            "sim_cal_rec": float(rec["sim_cal_rec"]),
            "sim_cal_rec_active": float(rec["sim_cal_rec_active"]),
            "strong20_rec": float(rec["strong20_rec"]),
            "kl_by_cond": kls,
            "n_ablated_non_bos_tokens": n_abl_tok_by_k[k],
            "kl_4band_sum": sum_4b,
            "kl_4band_additivity_ratio": float(sum_4b / kl_full) if kl_full > 0.0 else float("nan"),
            "kl_share_4band_norm": kl_share_4b_norm,
            "kl_share_4band_vs_full": kl_share_4b_vs_full,
            "kl_share_ge_20pct_max_norm": kl_share_ge20_norm,
            "kl_share_ge_20pct_max_vs_full": kl_share_ge20_full,
            "kl_share_gate_closed_norm": kl_share_gc_norm,
            "kl_share_gate_closed_vs_full": kl_share_gc_full,
            "corpus_energy_4band": corpus_4b,
            "corpus_energy_ge_20pct_max": float(rec["energy_ge_20pct_max"]),
            "corpus_energy_gate_closed": float(rec["energy_gate_closed"]),
            "eval_energy_4band": eval_energy_4b,
            "eval_energy_ge_20pct_max": eval_energy_ge20,
            "eval_energy_gate_closed": eval_energy_gc,
        }

    rows = [per_neuron_kl[str(k)] for k in sorted(int(x) for x in per_neuron_kl)]
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
            seed=seed + 1,
        )

    corrs["pooled_240_corpus_energy_vs_kl_norm"] = {
        **bootstrap_corr_stats(
            [r["corpus_energy_4band"][b] for r in rows for b in FOUR_BAND_CONDS],
            [r["kl_share_4band_norm"][b] for r in rows for b in FOUR_BAND_CONDS],
            n_resamples=n_bootstrap,
            seed=seed + 2,
        ),
        "bootstrap_unit": "flattened_neuron_band_pair",
    }
    corrs["pooled_240_eval_energy_vs_kl_norm"] = {
        **bootstrap_corr_stats(
            [r["eval_energy_4band"][b] for r in rows for b in FOUR_BAND_CONDS],
            [r["kl_share_4band_norm"][b] for r in rows for b in FOUR_BAND_CONDS],
            n_resamples=n_bootstrap,
            seed=seed + 3,
        ),
        "bootstrap_unit": "flattened_neuron_band_pair",
    }
    corrs["corpus_energy_vs_kl_norm__ge_20pct_max"] = bootstrap_corr_stats(
        [r["corpus_energy_ge_20pct_max"] for r in rows],
        [r["kl_share_ge_20pct_max_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed + 4,
    )
    corrs["eval_energy_vs_kl_norm__ge_20pct_max"] = bootstrap_corr_stats(
        [r["eval_energy_ge_20pct_max"] for r in rows],
        [r["kl_share_ge_20pct_max_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed + 5,
    )
    corrs["corpus_energy_vs_kl_norm__gate_closed"] = bootstrap_corr_stats(
        [r["corpus_energy_gate_closed"] for r in rows],
        [r["kl_share_gate_closed_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed + 6,
    )
    corrs["eval_energy_vs_kl_norm__gate_closed"] = bootstrap_corr_stats(
        [r["eval_energy_gate_closed"] for r in rows],
        [r["kl_share_gate_closed_norm"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed + 7,
    )
    corrs["kl_ge_20pct_max_vs_ulto85_strong20_rec"] = bootstrap_corr_stats(
        [r["kl_share_ge_20pct_max_norm"] for r in rows],
        [r["strong20_rec"] for r in rows],
        n_resamples=n_bootstrap,
        seed=seed + 8,
    )

    add_ratios = [
        r["kl_4band_additivity_ratio"] for r in rows if math.isfinite(r["kl_4band_additivity_ratio"])
    ]
    named_kl = {str(r["named_neuron"]): r for r in rows if r.get("named_neuron")}

    doc = {
        "model_id": gap_doc.get("model_id", MODEL_ID_4B),
        "compute_dtype": "torch.float32 (IEEE-754, allow_tf32=False)",
        "stage_ranges": [list(sr) for sr in stage_ranges],
        "n_neurons": len(rows),
        "neurons_per_layer": {str(L): len(neurons_by_layer[L]) for L in hook_layers},
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "total_tokens": n_seqs * seq_len,
        "total_non_bos_tokens": total_non_bos_tokens,
        "peak_vram_allocated_gb": round(peak_vram_alloc, 4),
        "peak_vram_reserved_gb": round(peak_vram_res, 4),
        "max_vram_limit_gb": max_vram_gb,
        "vram_pass": (not is_cuda) or (peak_vram_alloc < max_vram_gb and peak_vram_res < max_vram_gb),
        "zero_ablation_sanity_kl": zero_ablation_sanity_kl,
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
        "correlations": corrs,
        "named_neurons": named_kl,
        "per_neuron": per_neuron_kl,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[band_kl_4b] COMPLETE in {doc['elapsed_sec']:.1f}s | n={len(rows)} neurons | "
        f"peak_vram={peak_vram_alloc:.3f}/{peak_vram_res:.3f}GB | "
        f"additivity_med={doc['kl_4band_additivity_ratio_median']:.3f} | "
        f"pooled_240(eval) rho={corrs['pooled_240_eval_energy_vs_kl_norm']['spearman_rho']:.3f}, "
        f"pooled_240(corpus) rho={corrs['pooled_240_corpus_energy_vs_kl_norm']['spearman_rho']:.3f} | "
        f"ge20(eval) rho={corrs['eval_energy_vs_kl_norm__ge_20pct_max']['spearman_rho']:.3f}"
    )
    return doc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run exact IEEE-754 FP32 60-neuron mean-ablation band KL on Qwen/Qwen3-4B."
    )
    parser.add_argument("--model-id", default=MODEL_ID_4B)
    parser.add_argument("--n-seqs", type=int, default=KL_N_SEQS_4B)
    parser.add_argument("--seq-len", type=int, default=KL_SEQ_LEN_4B)
    parser.add_argument("--min-row-idx", type=int, default=KL_MIN_ROW_IDX_4B)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--kl-token-chunk", type=int, default=512)
    parser.add_argument("--per-layer", type=int, default=KL_NEURONS_PER_LAYER)
    parser.add_argument("--neurons-per-group", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-vram-gb", type=float, default=6.5)
    parser.add_argument("--seed", type=int, default=SEED_4B)
    args = parser.parse_args(argv)

    dirs = ensure_4b_dirs()
    gap_path = dirs["results"] / "gap_decomposition_4b.json"
    if not gap_path.is_file():
        raise FileNotFoundError(
            f"Missing {gap_path}. Run `python -m qwen4b_eval.gap_and_bands_4b` first."
        )
    gap_doc = json.loads(gap_path.read_text(encoding="utf-8"))
    selected = select_stratified_kl_neurons(gap_doc, per_layer=args.per_layer)

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
        f"[band_kl_4b] Loaded {len(seq_chunks)} seqs x {args.seq_len} = {input_ids.numel():,} tok "
        f"(rows {min(doc_ids_used)}..{max(doc_ids_used)}) for {len(selected)} stratified neurons"
    )

    out_path = dirs["results"] / "band_kl_4b.json"
    doc = run_band_kl_validation(
        backbone=backbone,
        input_ids_cpu=input_ids,
        gap_doc=gap_doc,
        selected_neurons=selected,
        compute_device=args.device,
        micro_batch_size=args.micro_batch_size,
        kl_token_chunk=args.kl_token_chunk,
        max_vram_gb=args.max_vram_gb,
        neurons_per_group=args.neurons_per_group,
        seed=args.seed,
        output_path=out_path,
    )
    doc["fineweb_min_row_idx_allowed"] = args.min_row_idx
    doc["fineweb_min_doc_id_observed"] = int(min(doc_ids_used))
    doc["fineweb_max_doc_id_observed"] = int(max(doc_ids_used))
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
