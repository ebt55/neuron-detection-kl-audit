"""Local exact-bf16 Qwen/Qwen3-4B 2-half GPU-swapped activation extraction pipeline.

Features:
- Strictly isolated under `src/qwen4b_eval/`, `data/qwen4b/`, and `results/qwen4b/`.
- HARD PARITY GATE (`run_parity_gate`): compares our 2-half GPU-swapped `bf16` forward
  pass (`embed_tokens + layers 0..17` -> `bf16` hidden state in CPU RAM -> `layers 18..35`,
  with `lm_head` deleted) against a full 36-layer `bf16` CPU reference pass on `[2, 32]`
  across all 10 hooked layers `[0, 4, 8, 12, 16, 20, 24, 28, 32, 35]`, verifying:
    1. Peak GPU VRAM cleanly < 6.5 GB (on both parity batch and extraction microbatch),
    2. Exact hooked `down_proj` input == `silu(gate_proj) * up_proj` on GPU (max_abs_err < 1e-4),
    3. 2-half GPU activations match full 36-layer CPU `bf16` reference (cos > 0.9999 on all 10 layers).
- Streams 1,536,000 tokens (24 macro-chunks x 500 seqs x 128 tokens) strictly from
  `HuggingFaceFW/fineweb` `sample/10BT/000_00000.parquet` rows `60,000+` (100% document-disjoint
  from Ulto85 rows `0..60,000`).
- Records float16 activations + uint8 SwiGLU quadrant codes (`Q1`-`Q4`) for all 278 Ulto85
  neurons into `data/qwen4b/cache/` (~854 MB acts + ~427 MB quads).
- Computes exact signed global mirrored thresholds (`p25_abs`, `p90`, `p97`, `p99`, `p99.9`),
  dominant pole per neuron by `argmax(|p99.9(pos)|, |p99.9(neg)|)`, and quadrant shares,
  updating `data/qwen4b/active_poles.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen4b_eval import (
    FINEWEB_DATASET_ID,
    FINEWEB_MIN_ROW_IDX,
    FINEWEB_PARQUET_PATH,
    HOOK_LAYERS_4B,
    MODEL_ID_4B,
    NUM_MACRO_CHUNKS_4B,
    SEQ_LEN_4B,
    SEQS_PER_MACRO_CHUNK_4B,
    SPLIT_LAYER_IDX_4B,
    ensure_4b_dirs,
    make_pole_id_4b,
)
from w_audit.config import QUADRANTS, WindowSpec
from w_audit.extract import QUAD_CODE_TO_NAME, encode_quadrant_torch


class ParityGateError(RuntimeError):
    """Raised when the hard VRAM safety or numerical parity gate fails."""


# ---------------------------------------------------------------------------
# Model Loading & 2-Half Layer Swapping Helpers
# ---------------------------------------------------------------------------


def get_backbone_module(model: nn.Module) -> nn.Module:
    """Return base transformer backbone (deleting lm_head if present on a CausalLM)."""
    if hasattr(model, "lm_head"):
        delattr(model, "lm_head")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise AttributeError(
        f"Expected Qwen3Model/Qwen3ForCausalLM-style backbone with .layers, got {type(model).__name__}"
    )


def load_qwen4b_backbone_cpu(
    model_id: str = MODEL_ID_4B,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, Any]:
    """Load Qwen/Qwen3-4B in exact bf16 on CPU with lm_head explicitly deleted."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        full_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        full_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if hasattr(full_model, "lm_head"):
        del full_model.lm_head
    backbone = full_model.model
    del full_model
    gc.collect()

    backbone.eval()
    backbone.requires_grad_(False)
    return backbone, tokenizer


def _prepare_masks_and_rope(
    backbone: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    """Build exact position_ids, RoPE embeddings, and causal masks matching Qwen3Model.forward."""
    seq_len = int(hidden_states.shape[1])
    device = hidden_states.device
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    position_embeddings = None
    if hasattr(backbone, "rotary_emb") and backbone.rotary_emb is not None:
        position_embeddings = backbone.rotary_emb(hidden_states, position_ids)

    causal_mask_mapping: dict[str, Any] = {"full_attention": None}
    if hasattr(backbone, "config") and hasattr(backbone.config, "layer_types"):
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        mask_kwargs = {
            "config": backbone.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs)
        if getattr(backbone, "has_sliding_layers", False):
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )
    return position_ids, position_embeddings, causal_mask_mapping


def _call_decoder_layer(
    backbone: nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    causal_mask_mapping: dict[str, Any],
    position_embeddings: Any,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Invoke a single Qwen3 or synthetic decoder layer with exact arguments."""
    layer = backbone.layers[layer_idx]
    if hasattr(backbone, "config") and hasattr(backbone.config, "layer_types"):
        layer_type = backbone.config.layer_types[layer_idx]
        attn_mask = causal_mask_mapping[layer_type]
        out = layer(
            hidden_states,
            attention_mask=attn_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
    else:
        try:
            out = layer(
                hidden_states,
                attention_mask=causal_mask_mapping.get("full_attention"),
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
            )
        except TypeError:
            out = layer(hidden_states)

    if isinstance(out, tuple):
        return out[0]
    return out


def _move_half1(backbone: nn.Module, split_idx: int, device: torch.device) -> None:
    backbone.embed_tokens.to(device)
    if hasattr(backbone, "rotary_emb") and backbone.rotary_emb is not None:
        backbone.rotary_emb.to(device)
    for i in range(0, split_idx):
        backbone.layers[i].to(device)


def _move_half2(backbone: nn.Module, split_idx: int, device: torch.device) -> None:
    if hasattr(backbone, "rotary_emb") and backbone.rotary_emb is not None:
        backbone.rotary_emb.to(device)
    for i in range(split_idx, len(backbone.layers)):
        backbone.layers[i].to(device)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_two_half_forward(
    backbone: nn.Module,
    input_ids_cpu: torch.Tensor,
    hook_layers: Sequence[int],
    neurons_by_layer: Mapping[int, Sequence[int]] | None = None,
    split_idx: int | None = None,
    compute_device: torch.device | str = "cuda",
    micro_batch_size: int = 16,
    capture_gate_up: bool = True,
) -> dict[str, Any]:
    """Run exact 2-half swapped bf16 forward pass over `input_ids_cpu` [N, T].

    - Half 1 (`embed_tokens + layers[0:split_idx]`) runs on `compute_device`, storing exact
      `bf16` intermediate hidden states in CPU RAM and capturing hooked activations for
      `L < split_idx`.
    - Half 1 moves back to CPU and CUDA cache is cleared.
    - Half 2 (`layers[split_idx:]`) runs on `compute_device` from the saved `bf16` hidden
      states, capturing hooked activations for `L >= split_idx`, then moves back to CPU.

    Returns dict with:
      - `acts`: `{L: Tensor[N, T, K_L]}` (CPU, model dtype)
      - `quads`: `{L: Tensor[N, T, K_L]}` (CPU, uint8) if `capture_gate_up`
      - `hook_identity_max_err`: `{L: float}` if `capture_gate_up`
      - `quad_sign_mismatches`: `{L: int}` if `capture_gate_up`
      - `peak_vram_half1_allocated_gb`, `peak_vram_half1_reserved_gb`
      - `peak_vram_half2_allocated_gb`, `peak_vram_half2_reserved_gb`
    """
    dev = torch.device(compute_device)
    is_cuda = dev.type == "cuda"
    cpu_dev = torch.device("cpu")

    n_layers = len(backbone.layers)
    if split_idx is None:
        split_idx = n_layers // 2

    half1_layers = sorted(int(L) for L in hook_layers if int(L) < split_idx)
    half2_layers = sorted(int(L) for L in hook_layers if int(L) >= split_idx)

    n_seqs, seq_len = int(input_ids_cpu.shape[0]), int(input_ids_cpu.shape[1])
    model_dtype = next(backbone.parameters()).dtype
    hidden_size = int(backbone.embed_tokens.weight.shape[1])

    # Pre-allocate compact CPU outputs per hooked layer
    acts_out: dict[int, torch.Tensor] = {}
    quads_out: dict[int, torch.Tensor] = {}
    hook_err_by_layer: dict[int, float] = {int(L): 0.0 for L in hook_layers}
    quad_mismatch_by_layer: dict[int, int] = {int(L): 0 for L in hook_layers}

    for L in hook_layers:
        d_mlp = int(backbone.layers[L].mlp.gate_proj.out_features)
        k_l = len(neurons_by_layer[L]) if neurons_by_layer is not None else d_mlp
        acts_out[int(L)] = torch.empty((n_seqs, seq_len, k_l), dtype=model_dtype, device=cpu_dev)
        if capture_gate_up:
            quads_out[int(L)] = torch.empty((n_seqs, seq_len, k_l), dtype=torch.uint8, device=cpu_dev)

    mid_hidden_cpu = torch.empty((n_seqs, seq_len, hidden_size), dtype=model_dtype, device=cpu_dev)

    peak_h1_alloc = 0.0
    peak_h1_res = 0.0
    peak_h2_alloc = 0.0
    peak_h2_res = 0.0

    def _register_layer_hooks(
        target_layers: Sequence[int],
        buf: dict[int, dict[str, torch.Tensor]],
    ) -> list[Any]:
        handles: list[Any] = []
        for L in target_layers:
            mlp = backbone.layers[L].mlp
            idx_t = (
                torch.tensor(list(neurons_by_layer[L]), dtype=torch.long, device=dev)
                if neurons_by_layer is not None
                else None
            )

            def _mk_down_pre(layer_i: int, sel: torch.Tensor | None):
                def _hook(_mod: nn.Module, args: tuple[Any, ...]) -> None:
                    t = args[0].detach()
                    buf[layer_i]["down_in"] = t.index_select(-1, sel) if sel is not None else t
                return _hook

            handles.append(mlp.down_proj.register_forward_pre_hook(_mk_down_pre(L, idx_t)))

            if capture_gate_up:
                def _mk_gate(layer_i: int, sel: torch.Tensor | None):
                    def _hook(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
                        t = out.detach()
                        buf[layer_i]["gate"] = t.index_select(-1, sel) if sel is not None else t
                    return _hook

                def _mk_up(layer_i: int, sel: torch.Tensor | None):
                    def _hook(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
                        t = out.detach()
                        buf[layer_i]["up"] = t.index_select(-1, sel) if sel is not None else t
                    return _hook

                handles.append(mlp.gate_proj.register_forward_hook(_mk_gate(L, idx_t)))
                handles.append(mlp.up_proj.register_forward_hook(_mk_up(L, idx_t)))
        return handles

    def _flush_batch_buf(
        target_layers: Sequence[int],
        buf: dict[int, dict[str, torch.Tensor]],
        s0: int,
        s1: int,
    ) -> None:
        for L in target_layers:
            a_t = buf[L]["down_in"]
            acts_out[L][s0:s1].copy_(a_t.to(cpu_dev), non_blocking=False)
            if capture_gate_up:
                g_t = buf[L]["gate"]
                u_t = buf[L]["up"]
                # Verify hooked silu(gate) * up == down_proj input on compute device
                a_check = F.silu(g_t) * u_t
                err = float((a_check.float() - a_t.float()).abs().max().item())
                if err > hook_err_by_layer[L]:
                    hook_err_by_layer[L] = err
                q_t = encode_quadrant_torch(g_t, u_t)
                nz = a_t != 0
                if nz.any():
                    q_nz = q_t[nz]
                    a_nz = a_t[nz]
                    expect_pos = (q_nz == 0) | (q_nz == 2)
                    mismatches = int((expect_pos != (a_nz > 0)).sum().item())
                    quad_mismatch_by_layer[L] += mismatches
                quads_out[L][s0:s1].copy_(q_t.to(cpu_dev), non_blocking=False)
            buf[L].clear()

    # ----------------------------- HALF 1 ---------------------------------
    if is_cuda:
        _clear_cuda()
        torch.cuda.reset_peak_memory_stats(dev)

    handles_h1: list[Any] = []
    try:
        _move_half1(backbone, split_idx, dev)
        buf_h1: dict[int, dict[str, torch.Tensor]] = {L: {} for L in half1_layers}
        handles_h1 = _register_layer_hooks(half1_layers, buf_h1)

        with torch.no_grad():
            for s0 in range(0, n_seqs, micro_batch_size):
                s1 = min(s0 + micro_batch_size, n_seqs)
                batch_ids = input_ids_cpu[s0:s1].to(dev, non_blocking=False)
                hidden_states = backbone.embed_tokens(batch_ids)
                pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, hidden_states)
                for i in range(0, split_idx):
                    hidden_states = _call_decoder_layer(
                        backbone, i, hidden_states, mask_map, pos_embs, pos_ids
                    )
                mid_hidden_cpu[s0:s1].copy_(
                    hidden_states.detach().to(cpu_dev, dtype=model_dtype),
                    non_blocking=False,
                )
                _flush_batch_buf(half1_layers, buf_h1, s0, s1)
                del batch_ids, hidden_states, pos_ids, pos_embs, mask_map

        if is_cuda:
            torch.cuda.synchronize(dev)
            peak_h1_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
            peak_h1_res = torch.cuda.max_memory_reserved(dev) / (1024**3)
    finally:
        for h in handles_h1:
            h.remove()
        _move_half1(backbone, split_idx, cpu_dev)
        if is_cuda:
            _clear_cuda()

    # ----------------------------- HALF 2 ---------------------------------
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    handles_h2: list[Any] = []
    try:
        _move_half2(backbone, split_idx, dev)
        buf_h2: dict[int, dict[str, torch.Tensor]] = {L: {} for L in half2_layers}
        handles_h2 = _register_layer_hooks(half2_layers, buf_h2)

        with torch.no_grad():
            for s0 in range(0, n_seqs, micro_batch_size):
                s1 = min(s0 + micro_batch_size, n_seqs)
                hidden_states = mid_hidden_cpu[s0:s1].to(dev, dtype=model_dtype, non_blocking=False)
                pos_ids, pos_embs, mask_map = _prepare_masks_and_rope(backbone, hidden_states)
                for i in range(split_idx, n_layers):
                    hidden_states = _call_decoder_layer(
                        backbone, i, hidden_states, mask_map, pos_embs, pos_ids
                    )
                _flush_batch_buf(half2_layers, buf_h2, s0, s1)
                del hidden_states, pos_ids, pos_embs, mask_map

        if is_cuda:
            torch.cuda.synchronize(dev)
            peak_h2_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
            peak_h2_res = torch.cuda.max_memory_reserved(dev) / (1024**3)
    finally:
        for h in handles_h2:
            h.remove()
        _move_half2(backbone, split_idx, cpu_dev)
        del mid_hidden_cpu
        if is_cuda:
            _clear_cuda()

    return {
        "acts": acts_out,
        "quads": quads_out if capture_gate_up else None,
        "hook_identity_max_err": hook_err_by_layer,
        "quad_sign_mismatches": quad_mismatch_by_layer,
        "peak_vram_half1_allocated_gb": peak_h1_alloc,
        "peak_vram_half1_reserved_gb": peak_h1_res,
        "peak_vram_half2_allocated_gb": peak_h2_alloc,
        "peak_vram_half2_reserved_gb": peak_h2_res,
    }


def run_full_cpu_reference(
    backbone: nn.Module,
    input_ids_cpu: torch.Tensor,
    hook_layers: Sequence[int],
    neurons_by_layer: Mapping[int, Sequence[int]] | None = None,
) -> dict[int, torch.Tensor]:
    """Run full unsplit forward pass on CPU and return hooked down_proj inputs per layer."""
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    try:
        for L in hook_layers:
            mlp = backbone.layers[L].mlp
            idx_t = (
                torch.tensor(list(neurons_by_layer[L]), dtype=torch.long, device="cpu")
                if neurons_by_layer is not None
                else None
            )

            def _mk_pre(layer_i: int, sel: torch.Tensor | None):
                def _hook(_mod: nn.Module, args: tuple[Any, ...]) -> None:
                    t = args[0].detach().cpu()
                    captured[layer_i] = t.index_select(-1, sel) if sel is not None else t
                return _hook

            handles.append(mlp.down_proj.register_forward_pre_hook(_mk_pre(L, idx_t)))

        with torch.no_grad():
            try:
                backbone(input_ids=input_ids_cpu, use_cache=False)
            except TypeError:
                backbone(input_ids_cpu)
    finally:
        for h in handles:
            h.remove()
    return captured


def _compare_layer_activations(
    ref_acts: Mapping[int, torch.Tensor],
    test_acts: Mapping[int, torch.Tensor],
    hook_layers: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Compute exact max_abs_err, mean_abs_err, and float64 cosine similarity per layer."""
    metrics: dict[str, dict[str, float]] = {}
    for L in hook_layers:
        a = ref_acts[L].to(torch.float64).reshape(-1)
        b = test_acts[L].to(torch.float64).reshape(-1)
        diff = (a - b).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        denom = float(a.norm(p=2).item() * b.norm(p=2).item())
        cos = float(torch.dot(a, b).item() / denom) if denom > 0.0 else 1.0
        metrics[str(L)] = {
            "max_abs_err": max_abs,
            "mean_abs_err": mean_abs,
            "cosine_similarity": cos,
        }
    return metrics


def run_parity_gate(
    backbone: nn.Module,
    parity_input_ids: torch.Tensor,
    hook_layers: Sequence[int] = HOOK_LAYERS_4B,
    neurons_by_layer: Mapping[int, Sequence[int]] | None = None,
    split_idx: int | None = None,
    compute_device: torch.device | str = "cuda",
    vram_stress_batch: torch.Tensor | None = None,
    max_vram_gb: float = 6.5,
    hook_tol: float = 1e-4,
    min_cosine: float = 0.9995,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the HARD PARITY & VRAM GATE before full extraction.

    Verifies:
      1. Peak GPU VRAM cleanly < `max_vram_gb` (6.5 GB) on parity AND extraction microbatch.
      2. Exact hooked `down_proj` input == `silu(gate_proj) * up_proj` (< 1e-4, 0 quad mismatches).
      3. 2-half CPU split matches full CPU reference bit-for-bit (0.0 error — proves zero split error).
      4. 2-half GPU-swapped activations match full CPU bf16 reference (`cos > min_cosine` on ALL
         hooked layers, accounting for 7-bit-mantissa CUDA-vs-CPU bf16 GEMM reduction order).
    """
    t0 = time.time()
    dev = torch.device(compute_device)
    is_cuda = dev.type == "cuda"
    model_dtype = str(next(backbone.parameters()).dtype)
    if split_idx is None:
        split_idx = len(backbone.layers) // 2

    # 1. Reference full 36-layer bf16 forward pass on CPU (ALL neurons on hooked layers)
    t_ref = time.time()
    ref_cpu_all = run_full_cpu_reference(
        backbone=backbone,
        input_ids_cpu=parity_input_ids,
        hook_layers=hook_layers,
        neurons_by_layer=None,
    )
    ref_cpu_sec = round(time.time() - t_ref, 3)

    # 2. 2-half CPU split self-parity check (proves splitting/masking/RoPE introduces 0.0 error)
    two_half_cpu = run_two_half_forward(
        backbone=backbone,
        input_ids_cpu=parity_input_ids,
        hook_layers=hook_layers,
        neurons_by_layer=None,
        split_idx=split_idx,
        compute_device="cpu",
        micro_batch_size=int(parity_input_ids.shape[0]),
        capture_gate_up=False,
    )
    cpu_split_metrics = _compare_layer_activations(ref_cpu_all, two_half_cpu["acts"], hook_layers)
    max_cpu_split_err = max(m["max_abs_err"] for m in cpu_split_metrics.values())
    del two_half_cpu

    # 3. 2-half GPU-swapped bf16 forward pass on parity batch
    t_gpu = time.time()
    two_half_gpu = run_two_half_forward(
        backbone=backbone,
        input_ids_cpu=parity_input_ids,
        hook_layers=hook_layers,
        neurons_by_layer=None,
        split_idx=split_idx,
        compute_device=dev,
        micro_batch_size=int(parity_input_ids.shape[0]),
        capture_gate_up=True,
    )
    gpu_parity_sec = round(time.time() - t_gpu, 3)

    all_neuron_metrics = _compare_layer_activations(ref_cpu_all, two_half_gpu["acts"], hook_layers)

    tracked_metrics: dict[str, dict[str, float]] = {}
    if neurons_by_layer is not None:
        ref_tracked = {
            L: ref_cpu_all[L].index_select(-1, torch.tensor(list(neurons_by_layer[L]), dtype=torch.long))
            for L in hook_layers
        }
        gpu_tracked = {
            L: two_half_gpu["acts"][L].index_select(
                -1, torch.tensor(list(neurons_by_layer[L]), dtype=torch.long)
            )
            for L in hook_layers
        }
        tracked_metrics = _compare_layer_activations(ref_tracked, gpu_tracked, hook_layers)

    max_hook_id_err = max(two_half_gpu["hook_identity_max_err"].values())
    total_quad_mismatches = sum(two_half_gpu["quad_sign_mismatches"].values())
    worst_all_cos = min(m["cosine_similarity"] for m in all_neuron_metrics.values())
    worst_all_max_err = max(m["max_abs_err"] for m in all_neuron_metrics.values())
    worst_tracked_cos = (
        min(m["cosine_similarity"] for m in tracked_metrics.values())
        if tracked_metrics
        else worst_all_cos
    )

    peak_vram_alloc_gb = max(
        two_half_gpu["peak_vram_half1_allocated_gb"],
        two_half_gpu["peak_vram_half2_allocated_gb"],
    )
    peak_vram_res_gb = max(
        two_half_gpu["peak_vram_half1_reserved_gb"],
        two_half_gpu["peak_vram_half2_reserved_gb"],
    )

    # 4. Optional realistic extraction microbatch VRAM stress test
    stress_vram: dict[str, Any] | None = None
    if vram_stress_batch is not None and is_cuda:
        stress_res = run_two_half_forward(
            backbone=backbone,
            input_ids_cpu=vram_stress_batch,
            hook_layers=hook_layers,
            neurons_by_layer=neurons_by_layer,
            split_idx=split_idx,
            compute_device=dev,
            micro_batch_size=int(vram_stress_batch.shape[0]),
            capture_gate_up=True,
        )
        stress_alloc = max(
            stress_res["peak_vram_half1_allocated_gb"],
            stress_res["peak_vram_half2_allocated_gb"],
        )
        stress_resv = max(
            stress_res["peak_vram_half1_reserved_gb"],
            stress_res["peak_vram_half2_reserved_gb"],
        )
        peak_vram_alloc_gb = max(peak_vram_alloc_gb, stress_alloc)
        peak_vram_res_gb = max(peak_vram_res_gb, stress_resv)
        max_hook_id_err = max(max_hook_id_err, max(stress_res["hook_identity_max_err"].values()))
        total_quad_mismatches += sum(stress_res["quad_sign_mismatches"].values())
        stress_vram = {
            "batch_shape": list(vram_stress_batch.shape),
            "peak_half1_allocated_gb": round(stress_res["peak_vram_half1_allocated_gb"], 4),
            "peak_half1_reserved_gb": round(stress_res["peak_vram_half1_reserved_gb"], 4),
            "peak_half2_allocated_gb": round(stress_res["peak_vram_half2_allocated_gb"], 4),
            "peak_half2_reserved_gb": round(stress_res["peak_vram_half2_reserved_gb"], 4),
        }

    vram_ok = (not is_cuda) or (peak_vram_alloc_gb < max_vram_gb and peak_vram_res_gb < max_vram_gb)
    hook_ok = (max_hook_id_err < hook_tol) and (total_quad_mismatches == 0)
    cpu_split_ok = max_cpu_split_err == 0.0
    parity_ok = (worst_all_cos > min_cosine) and (worst_tracked_cos > min_cosine)
    passed = bool(vram_ok and hook_ok and cpu_split_ok and parity_ok)
    cuda_vs_cpu_min_cos = min(worst_all_cos, worst_tracked_cos)

    report: dict[str, Any] = {
        "passed": passed,
        "split_exact_pass": cpu_split_ok,
        "gpu_hook_pass": hook_ok,
        "vram_pass": vram_ok,
        "cuda_vs_cpu_bf16_min_cos": round(cuda_vs_cpu_min_cos, 6),
        "checks": {
            "vram_under_limit": vram_ok,
            "hook_identity_under_tol": hook_ok,
            "cpu_two_half_bit_exact": cpu_split_ok,
            "gpu_vs_cpu_ref_cosine_ok": parity_ok,
        },
        "model_dtype": model_dtype,
        "compute_device": str(dev),
        "split_layer_idx": split_idx,
        "num_layers": len(backbone.layers),
        "hook_layers": [int(L) for L in hook_layers],
        "parity_batch_shape": list(parity_input_ids.shape),
        "max_vram_limit_gb": max_vram_gb,
        "peak_vram_allocated_gb": round(peak_vram_alloc_gb, 4),
        "peak_vram_reserved_gb": round(peak_vram_res_gb, 4),
        "parity_batch_vram": {
            "peak_half1_allocated_gb": round(two_half_gpu["peak_vram_half1_allocated_gb"], 4),
            "peak_half1_reserved_gb": round(two_half_gpu["peak_vram_half1_reserved_gb"], 4),
            "peak_half2_allocated_gb": round(two_half_gpu["peak_vram_half2_allocated_gb"], 4),
            "peak_half2_reserved_gb": round(two_half_gpu["peak_vram_half2_reserved_gb"], 4),
        },
        "extraction_microbatch_vram": stress_vram,
        "hook_tol": hook_tol,
        "max_hook_identity_err": max_hook_id_err,
        "hook_identity_err_by_layer": {
            str(k): float(v) for k, v in two_half_gpu["hook_identity_max_err"].items()
        },
        "total_quad_sign_mismatches": total_quad_mismatches,
        "max_cpu_two_half_err": max_cpu_split_err,
        "min_cosine_required": min_cosine,
        "worst_all_neurons_cosine": worst_all_cos,
        "worst_all_neurons_max_abs_err": worst_all_max_err,
        "worst_tracked_neurons_cosine": worst_tracked_cos,
        "per_layer_all_neurons": all_neuron_metrics,
        "per_layer_tracked_neurons": tracked_metrics,
        "ref_cpu_sec": ref_cpu_sec,
        "gpu_parity_sec": gpu_parity_sec,
        "elapsed_sec": round(time.time() - t0, 3),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not passed:
        raise ParityGateError(
            f"HARD PARITY GATE FAILED: vram_ok={vram_ok} (alloc={peak_vram_alloc_gb:.3f}GB, "
            f"res={peak_vram_res_gb:.3f}GB), hook_ok={hook_ok} (max_err={max_hook_id_err:.2e}, "
            f"quad_mis={total_quad_mismatches}), cpu_split_ok={cpu_split_ok} "
            f"(err={max_cpu_split_err:.2e}), parity_ok={parity_ok} "
            f"(worst_all_cos={worst_all_cos:.6f}, worst_tracked_cos={worst_tracked_cos:.6f}, "
            f"worst_max_err={worst_all_max_err:.4e})"
        )
    return report


# ---------------------------------------------------------------------------
# Strictly Disjoint FineWeb Streaming (rows 60,000+ of 000_00000.parquet)
# ---------------------------------------------------------------------------


def iter_fineweb_disjoint_rows(
    min_row_idx: int = FINEWEB_MIN_ROW_IDX,
    dataset_id: str = FINEWEB_DATASET_ID,
    parquet_rel_path: str = FINEWEB_PARQUET_PATH,
) -> Iterator[tuple[int, str]]:
    """Stream `(parquet_row_idx, text)` strictly from rows >= `min_row_idx` (60,000+).

    Uses HTTP range reads via `fsspec` + `pyarrow.parquet`, skipping row groups entirely
    below `min_row_idx` without downloading them, and asserting `parquet_row_idx >= min_row_idx`
    on every yielded document.
    """
    import fsspec
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{parquet_rel_path}"
    fs = fsspec.filesystem("https")
    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        row_offset = 0
        for rg_idx in range(pf.num_row_groups):
            rg_rows = int(pf.metadata.row_group(rg_idx).num_rows)
            rg_end = row_offset + rg_rows
            if rg_end <= min_row_idx:
                row_offset = rg_end
                continue
            tbl = pf.read_row_groups([rg_idx], columns=["text"])
            texts = tbl["text"].to_pylist()
            start_local = max(0, min_row_idx - row_offset)
            for local_i in range(start_local, len(texts)):
                abs_row_idx = row_offset + local_i
                if abs_row_idx < min_row_idx:
                    raise AssertionError(
                        f"Overlap violation: yielded row {abs_row_idx} < {min_row_idx}"
                    )
                txt = texts[local_i]
                if isinstance(txt, str) and txt:
                    yield abs_row_idx, txt
            row_offset = rg_end


@dataclass
class FixedSeqChunk:
    doc_id: int  # Exact FineWeb parquet row index (>= 60,000)
    chunk_id: int
    token_ids: list[int]


def collect_fixed_length_sequences(
    docs_with_ids: Iterable[tuple[int, str]],
    tokenizer: Any,
    n_sequences: int,
    seq_len: int = SEQ_LEN_4B,
    max_seqs_per_doc: int = 4,
    min_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
) -> list[FixedSeqChunk]:
    """Tokenize documents into exact `seq_len`-token non-overlapping single-doc sequences.

    - Skips documents shorter than `seq_len` (zero padding, zero cross-doc concatenation).
    - Caps sequences per document at `max_seqs_per_doc` to maximize document diversity for
      rare-token neurons while preserving exact single-document provenance (`doc_id`).
    """
    chunks: list[FixedSeqChunk] = []
    chunk_id = 0
    for doc_id, text in docs_with_ids:
        if min_doc_id is not None and int(doc_id) < min_doc_id:
            raise AssertionError(f"Disjointness violation: doc_id={doc_id} < {min_doc_id}")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < seq_len:
            continue
        taken_from_doc = 0
        for start in range(0, len(ids) - seq_len + 1, seq_len):
            sub = ids[start : start + seq_len]
            chunks.append(FixedSeqChunk(doc_id=int(doc_id), chunk_id=chunk_id, token_ids=sub))
            chunk_id += 1
            taken_from_doc += 1
            if len(chunks) >= n_sequences or taken_from_doc >= max_seqs_per_doc:
                break
        if len(chunks) >= n_sequences:
            break

    if len(chunks) < n_sequences:
        raise RuntimeError(
            f"Corpus exhausted after {len(chunks)}/{n_sequences} sequences of length {seq_len}."
        )
    return chunks


# ---------------------------------------------------------------------------
# Signed Global Thresholds, Dominant Pole & Quadrant Shares
# ---------------------------------------------------------------------------


def compute_pole_thresholds_4b(
    acts_1d: np.ndarray,
    quads_1d: np.ndarray,
    non_bos_mask: np.ndarray,
    interior_mask: np.ndarray,
    sign: int,
) -> dict[str, Any]:
    """Compute exact mirrored global quantiles (p25_abs, p90, p97, p99, p99.9) & quadrant shares."""
    a_valid = acts_1d[non_bos_mask].astype(np.float32)
    q_valid = quads_1d[non_bos_mask]

    frac_pos = float(np.mean(a_valid > 0.0))
    frac_neg = float(np.mean(a_valid < 0.0))
    p25_abs = float(np.quantile(np.abs(a_valid), 0.25))

    d_valid = float(sign) * a_valid
    q90, q97, q99, q99_9 = (float(x) for x in np.quantile(d_valid, [0.90, 0.97, 0.99, 0.999]))
    band_edges_dir = {
        "p90": q90,
        "p97": q97,
        "p99": q99,
        "p99.9": q99_9,
    }

    n_valid = max(len(q_valid), 1)
    quadrant_shares_corpus = {
        QUAD_CODE_TO_NAME[c]: float(np.sum(q_valid == c) / n_valid) for c in range(4)
    }

    guard_valid = (d_valid > 0.0) & (np.abs(a_valid) >= p25_abs)
    band_bounds = {
        "p97_99": (q97, q99),
        "p99.9_plus": (q99_9, None),
    }
    quadrant_shares_by_band: dict[str, dict[str, float]] = {}
    for bname, (lo, hi) in band_bounds.items():
        bmask = guard_valid & (d_valid >= lo)
        if hi is not None:
            bmask = bmask & (d_valid < hi)
        cnt = int(np.sum(bmask))
        if cnt == 0:
            quadrant_shares_by_band[bname] = {qn: 0.0 for qn in QUADRANTS}
        else:
            qb = q_valid[bmask]
            quadrant_shares_by_band[bname] = {
                QUAD_CODE_TO_NAME[c]: float(np.sum(qb == c) / cnt) for c in range(4)
            }

    # Interior eligibility counts
    a_all = acts_1d.astype(np.float32)
    d_all = float(sign) * a_all
    abs_all = np.abs(a_all)
    guard_int = interior_mask & (d_all > 0.0) & (abs_all >= p25_abs)
    elig_p999 = int(np.sum(guard_int & (d_all >= q99_9)))
    elig_p97_99 = int(np.sum(guard_int & (d_all >= q97) & (d_all < q99)))
    elig_dz = int(np.sum(interior_mask & (abs_all < p25_abs)))

    return {
        "frac_pos": frac_pos,
        "frac_neg": frac_neg,
        "p25_abs": p25_abs,
        "band_edges_dir": band_edges_dir,
        "quadrant_shares_corpus": quadrant_shares_corpus,
        "quadrant_shares_by_band": quadrant_shares_by_band,
        "eligible_interior_counts": {
            "p99.9_plus": elig_p999,
            "p97_99": elig_p97_99,
            "dead_zone": elig_dz,
        },
    }


# ---------------------------------------------------------------------------
# Full Streaming Extraction Entrypoint
# ---------------------------------------------------------------------------


def run_extraction_4b(
    backbone: nn.Module,
    tokenizer: Any,
    catalog: dict[str, Any],
    docs_with_ids: Iterable[tuple[int, str]] | None = None,
    base_dir: Path | None = None,
    num_macro_chunks: int = NUM_MACRO_CHUNKS_4B,
    seqs_per_macro_chunk: int = SEQS_PER_MACRO_CHUNK_4B,
    seq_len: int = SEQ_LEN_4B,
    micro_batch_size: int = 16,
    max_seqs_per_doc: int = 4,
    split_idx: int | None = None,
    compute_device: str | torch.device = "cuda",
    min_doc_id: int | None = FINEWEB_MIN_ROW_IDX,
    run_gate: bool = True,
    max_vram_gb: float = 6.5,
    hook_tol: float = 1e-4,
    min_cosine: float = 0.9995,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Run parity gate + 24-chunk 2-half bf16 extraction + cache + active_poles.json update."""
    t0 = time.time()
    dirs = ensure_4b_dirs(base_dir)
    cache_dir = dirs["cache"]
    results_dir = dirs["results"]
    active_poles_path = dirs["data"] / "active_poles.json"
    parity_path = results_dir / "parity_check.json"
    wspec = window_spec if window_spec is not None else WindowSpec()

    # Build deterministic column order from catalog["neurons"] (278 neurons sorted by k)
    neurons_list = sorted(catalog["neurons"], key=lambda r: int(r["k"]))
    hook_layers = sorted({int(n["layer"]) for n in neurons_list})
    neurons_by_layer: dict[int, list[int]] = {L: [] for L in hook_layers}
    col_to_neuron: list[tuple[int, int]] = []
    col_to_k: list[int] = []
    neuron_to_col: dict[tuple[int, int], int] = {}

    for rec in neurons_list:
        L = int(rec["layer"])
        n_idx = int(rec["neuron"])
        k = int(rec["k"])
        col = len(col_to_neuron)
        neuron_to_col[(L, n_idx)] = col
        col_to_neuron.append((L, n_idx))
        col_to_k.append(k)
        neurons_by_layer[L].append(n_idx)

    col_slices_by_layer: dict[int, list[int]] = {
        L: [neuron_to_col[(L, n_idx)] for n_idx in neurons_by_layer[L]] for L in hook_layers
    }
    n_tracked = len(col_to_neuron)
    n_seqs_total = num_macro_chunks * seqs_per_macro_chunk
    total_tokens = n_seqs_total * seq_len

    # 1. Stream & tokenize strictly disjoint FineWeb rows 60,000+
    t_tok = time.time()
    if docs_with_ids is None:
        docs_with_ids = iter_fineweb_disjoint_rows(
            min_row_idx=FINEWEB_MIN_ROW_IDX if min_doc_id is None else min_doc_id
        )

    seq_chunks = collect_fixed_length_sequences(
        docs_with_ids=docs_with_ids,
        tokenizer=tokenizer,
        n_sequences=n_seqs_total,
        seq_len=seq_len,
        max_seqs_per_doc=max_seqs_per_doc,
        min_doc_id=min_doc_id,
    )
    unique_doc_ids = sorted({c.doc_id for c in seq_chunks})
    min_seen_doc = unique_doc_ids[0]
    max_seen_doc = unique_doc_ids[-1]
    print(
        f"[extract_4b] Tokenized {len(seq_chunks):,} seqs x {seq_len} = {total_tokens:,} tok "
        f"from {len(unique_doc_ids):,} unique docs (doc_id {min_seen_doc}..{max_seen_doc}) "
        f"in {time.time() - t_tok:.1f}s"
    )

    # 2. HARD PARITY & VRAM GATE before full extraction
    parity_report: dict[str, Any] | None = None
    if run_gate:
        parity_len = min(32, seq_len)
        parity_b = min(2, len(seq_chunks))
        parity_batch = torch.tensor(
            [seq_chunks[i].token_ids[:parity_len] for i in range(parity_b)],
            dtype=torch.long,
        )
        stress_b = min(micro_batch_size, len(seq_chunks))
        stress_batch = torch.tensor(
            [seq_chunks[i].token_ids for i in range(stress_b)],
            dtype=torch.long,
        )
        parity_report = run_parity_gate(
            backbone=backbone,
            parity_input_ids=parity_batch,
            hook_layers=hook_layers,
            neurons_by_layer=neurons_by_layer,
            split_idx=split_idx,
            compute_device=compute_device,
            vram_stress_batch=stress_batch,
            max_vram_gb=max_vram_gb,
            hook_tol=hook_tol,
            min_cosine=min_cosine,
            output_path=parity_path,
        )
        print(
            f"[extract_4b] HARD PARITY GATE PASSED: "
            f"peak_vram_alloc={parity_report['peak_vram_allocated_gb']:.3f} GB, "
            f"peak_vram_res={parity_report['peak_vram_reserved_gb']:.3f} GB, "
            f"max_hook_err={parity_report['max_hook_identity_err']:.2e}, "
            f"worst_all_cos={parity_report['worst_all_neurons_cosine']:.6f}, "
            f"worst_tracked_cos={parity_report['worst_tracked_neurons_cosine']:.6f}"
        )

    # 3. Pre-allocate host arrays
    token_ids_arr = np.empty(total_tokens, dtype=np.int32)
    doc_ids_arr = np.empty(total_tokens, dtype=np.int32)
    chunk_ids_arr = np.empty(total_tokens, dtype=np.int32)
    pos_in_chunk_arr = np.empty(total_tokens, dtype=np.int16)
    chunk_starts_arr = np.empty(n_seqs_total, dtype=np.int64)
    chunk_lengths_arr = np.full(n_seqs_total, seq_len, dtype=np.int32)
    acts_arr = np.empty((total_tokens, n_tracked), dtype=np.float16)
    quads_arr = np.empty((total_tokens, n_tracked), dtype=np.uint8)

    pos_template = np.arange(seq_len, dtype=np.int16)
    for i, ch in enumerate(seq_chunks):
        s0 = i * seq_len
        s1 = s0 + seq_len
        chunk_starts_arr[i] = s0
        token_ids_arr[s0:s1] = ch.token_ids
        doc_ids_arr[s0:s1] = ch.doc_id
        chunk_ids_arr[s0:s1] = ch.chunk_id
        pos_in_chunk_arr[s0:s1] = pos_template

    # 4. Run 2-half swapped bf16 forward pass chunk-by-chunk (24 macro-chunks x 64k tokens)
    t_fwd = time.time()
    max_run_vram_alloc = 0.0
    max_run_vram_res = 0.0
    max_run_hook_err = 0.0
    total_run_quad_mis = 0

    for mc in range(num_macro_chunks):
        seq_start = mc * seqs_per_macro_chunk
        seq_end = seq_start + seqs_per_macro_chunk
        tok_start = seq_start * seq_len
        tok_end = seq_end * seq_len

        mc_input_ids = torch.from_numpy(
            token_ids_arr[tok_start:tok_end].reshape(seqs_per_macro_chunk, seq_len).astype(np.int64)
        )
        res = run_two_half_forward(
            backbone=backbone,
            input_ids_cpu=mc_input_ids,
            hook_layers=hook_layers,
            neurons_by_layer=neurons_by_layer,
            split_idx=split_idx,
            compute_device=compute_device,
            micro_batch_size=micro_batch_size,
            capture_gate_up=True,
        )

        max_run_vram_alloc = max(
            max_run_vram_alloc,
            res["peak_vram_half1_allocated_gb"],
            res["peak_vram_half2_allocated_gb"],
        )
        max_run_vram_res = max(
            max_run_vram_res,
            res["peak_vram_half1_reserved_gb"],
            res["peak_vram_half2_reserved_gb"],
        )
        chunk_hook_err = max(res["hook_identity_max_err"].values())
        chunk_quad_mis = sum(res["quad_sign_mismatches"].values())
        max_run_hook_err = max(max_run_hook_err, chunk_hook_err)
        total_run_quad_mis += chunk_quad_mis

        if chunk_hook_err >= hook_tol or chunk_quad_mis > 0:
            raise ParityGateError(
                f"Runtime hook check failed on macro-chunk {mc}: "
                f"max_hook_err={chunk_hook_err:.2e}, quad_mismatches={chunk_quad_mis}"
            )

        for L in hook_layers:
            cols = col_slices_by_layer[L]
            a_f16 = (
                res["acts"][L]
                .float()
                .clamp(-65500.0, 65500.0)
                .to(torch.float16)
                .reshape(-1, len(cols))
                .numpy()
            )
            q_u8 = res["quads"][L].reshape(-1, len(cols)).numpy()
            acts_arr[tok_start:tok_end, cols] = a_f16
            quads_arr[tok_start:tok_end, cols] = q_u8

        del res, mc_input_ids
        elapsed = max(time.time() - t_fwd, 1e-6)
        print(
            f"[extract_4b] Macro-chunk {mc + 1:02d}/{num_macro_chunks} | "
            f"{tok_end:,}/{total_tokens:,} tok ({tok_end / elapsed:,.0f} tok/s) | "
            f"peak_vram_alloc={max_run_vram_alloc:.2f} GB | max_hook_err={max_run_hook_err:.2e}"
        )

    # 5. Compute masks, token counts, global mirrored thresholds, dominant poles & quadrant shares
    special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int) and val >= 0:
            special_ids.add(val)

    non_bos_mask = pos_in_chunk_arr > 0
    if special_ids:
        non_bos_mask = non_bos_mask & (~np.isin(token_ids_arr, list(special_ids)))

    chunk_len_at_pos = chunk_lengths_arr[chunk_ids_arr]
    interior_mask = (
        non_bos_mask
        & (pos_in_chunk_arr >= wspec.before)
        & (pos_in_chunk_arr <= (chunk_len_at_pos - wspec.after - 1))
    )

    vocab_size = max(int(token_ids_arr.max()) + 1, int(getattr(tokenizer, "vocab_size", 0) or 0))
    token_counts = np.bincount(token_ids_arr, minlength=vocab_size).astype(np.int64)

    neuron_by_k = {int(n["k"]): n for n in catalog["neurons"]}
    pole_by_id = {str(p["pole_id"]): p for p in catalog["poles"]}

    zero_p25_count = 0
    for rec in neurons_list:
        k = int(rec["k"])
        L = int(rec["layer"])
        n_idx = int(rec["neuron"])
        col = neuron_to_col[(L, n_idx)]
        acts_1d = acts_arr[:, col]
        quads_1d = quads_arr[:, col]

        pos_stats = compute_pole_thresholds_4b(acts_1d, quads_1d, non_bos_mask, interior_mask, +1)
        neg_stats = compute_pole_thresholds_4b(acts_1d, quads_1d, non_bos_mask, interior_mask, -1)

        if pos_stats["p25_abs"] <= 0.0:
            zero_p25_count += 1

        p99_9_pos = float(pos_stats["band_edges_dir"]["p99.9"])
        p99_9_neg = float(neg_stats["band_edges_dir"]["p99.9"])
        raw_dom_sign = 1 if abs(p99_9_pos) >= abs(p99_9_neg) else -1

        n_entry = neuron_by_k[k]
        active_signs = [int(s) for s in n_entry["active_signs"]]
        if len(active_signs) == 1:
            dom_sign = active_signs[0]
        else:
            dom_sign = raw_dom_sign
        dom_pole_id = make_pole_id_4b(L, n_idx, dom_sign)

        n_entry["p99_9_pos"] = p99_9_pos
        n_entry["p99_9_neg"] = p99_9_neg
        n_entry["abs_p99_9_pos"] = abs(p99_9_pos)
        n_entry["abs_p99_9_neg"] = abs(p99_9_neg)
        n_entry["raw_dominant_sign"] = raw_dom_sign
        n_entry["dominant_sign"] = dom_sign
        n_entry["dominant_pole_id"] = dom_pole_id
        n_entry["p25_abs"] = float(pos_stats["p25_abs"])
        n_entry["frac_pos"] = float(pos_stats["frac_pos"])
        n_entry["frac_neg"] = float(pos_stats["frac_neg"])

        for sign, st in ((1, pos_stats), (-1, neg_stats)):
            pid = make_pole_id_4b(L, n_idx, sign)
            if pid not in pole_by_id:
                continue
            p_entry = pole_by_id[pid]
            p_entry["is_dominant_pole"] = bool(pid == dom_pole_id)
            p_entry["dominant_pole_id"] = dom_pole_id
            p_entry["p99_9_pos"] = p99_9_pos
            p_entry["p99_9_neg"] = p99_9_neg
            p_entry["frac_pos"] = st["frac_pos"]
            p_entry["frac_neg"] = st["frac_neg"]
            p_entry["p25_abs"] = st["p25_abs"]
            p_entry["band_edges_dir"] = st["band_edges_dir"]
            p_entry["quadrant_shares_corpus"] = st["quadrant_shares_corpus"]
            p_entry["quadrant_shares_by_band"] = st["quadrant_shares_by_band"]
            p_entry["eligible_interior_counts"] = st["eligible_interior_counts"]

    # 6. Persist compact numpy cache under data/qwen4b/cache/
    np.save(cache_dir / "token_ids.npy", token_ids_arr)
    np.save(cache_dir / "doc_ids.npy", doc_ids_arr)
    np.save(cache_dir / "chunk_ids.npy", chunk_ids_arr)
    np.save(cache_dir / "pos_in_chunk.npy", pos_in_chunk_arr)
    np.save(cache_dir / "chunk_starts.npy", chunk_starts_arr)
    np.save(cache_dir / "chunk_lengths.npy", chunk_lengths_arr)
    np.save(cache_dir / "acts.npy", acts_arr)
    np.save(cache_dir / "quads.npy", quads_arr)
    np.save(cache_dir / "token_counts.npy", token_counts)

    elapsed_total = round(time.time() - t0, 2)
    cache_meta = {
        "model_id": catalog.get("model_id", MODEL_ID_4B),
        "dataset_id": FINEWEB_DATASET_ID,
        "parquet_rel_path": FINEWEB_PARQUET_PATH,
        "min_allowed_row_idx": min_doc_id,
        "min_observed_doc_id": int(min_seen_doc),
        "max_observed_doc_id": int(max_seen_doc),
        "n_unique_docs": len(unique_doc_ids),
        "num_macro_chunks": num_macro_chunks,
        "seqs_per_macro_chunk": seqs_per_macro_chunk,
        "seq_len": seq_len,
        "max_seqs_per_doc": max_seqs_per_doc,
        "total_tokens": total_tokens,
        "non_bos_tokens": int(np.sum(non_bos_mask)),
        "interior_tokens": int(np.sum(interior_mask)),
        "special_token_ids": sorted(special_ids),
        "window": wspec.to_dict(),
        "col_to_neuron": [[int(L), int(n)] for (L, n) in col_to_neuron],
        "col_to_k": [int(k) for k in col_to_k],
        "zero_p25_neuron_count": zero_p25_count,
        "peak_vram_allocated_gb": round(max_run_vram_alloc, 4),
        "peak_vram_reserved_gb": round(max_run_vram_res, 4),
        "max_runtime_hook_identity_err": max_run_hook_err,
        "total_runtime_quad_mismatches": total_run_quad_mis,
        "parity_check_passed": bool(parity_report["passed"]) if parity_report is not None else None,
        "elapsed_sec": elapsed_total,
    }
    (cache_dir / "meta.json").write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")

    # 7. Update data/qwen4b/active_poles.json
    catalog["extraction_complete"] = True
    catalog["corpus"] = {
        "dataset_id": FINEWEB_DATASET_ID,
        "parquet_rel_path": FINEWEB_PARQUET_PATH,
        "min_allowed_row_idx": min_doc_id,
        "min_observed_doc_id": int(min_seen_doc),
        "max_observed_doc_id": int(max_seen_doc),
        "n_unique_docs": len(unique_doc_ids),
        "total_tokens": total_tokens,
        "seq_len": seq_len,
        "num_macro_chunks": num_macro_chunks,
        "seqs_per_macro_chunk": seqs_per_macro_chunk,
        "max_seqs_per_doc": max_seqs_per_doc,
    }
    catalog["window"] = wspec.to_dict()
    catalog["extraction_summary"] = {
        "peak_vram_allocated_gb": round(max_run_vram_alloc, 4),
        "peak_vram_reserved_gb": round(max_run_vram_res, 4),
        "max_runtime_hook_identity_err": max_run_hook_err,
        "total_runtime_quad_mismatches": total_run_quad_mis,
        "zero_p25_neuron_count": zero_p25_count,
        "elapsed_sec": elapsed_total,
    }
    active_poles_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[extract_4b] COMPLETE in {elapsed_total:.1f}s | {total_tokens:,} tokens | "
        f"peak_vram_alloc={max_run_vram_alloc:.3f} GB | zero_p25={zero_p25_count}"
    )
    return {
        "cache_meta": cache_meta,
        "parity_report": parity_report,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run parity-gated exact-bf16 2-half GPU-swapped Qwen3-4B extraction."
    )
    parser.add_argument("--model-id", default=MODEL_ID_4B)
    parser.add_argument("--macro-chunks", type=int, default=NUM_MACRO_CHUNKS_4B)
    parser.add_argument("--seqs-per-chunk", type=int, default=SEQS_PER_MACRO_CHUNK_4B)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_4B)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--max-seqs-per-doc", type=int, default=4)
    parser.add_argument("--split-idx", type=int, default=SPLIT_LAYER_IDX_4B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-vram-gb", type=float, default=6.5)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args(argv)

    dirs = ensure_4b_dirs()
    active_poles_path = dirs["data"] / "active_poles.json"
    if not active_poles_path.is_file():
        raise FileNotFoundError(
            f"Missing {active_poles_path}. Run `python -m qwen4b_eval.fetch_ulto85` first."
        )
    catalog = json.loads(active_poles_path.read_text(encoding="utf-8"))

    backbone, tokenizer = load_qwen4b_backbone_cpu(model_id=args.model_id, dtype=torch.bfloat16)

    if args.parity_only:
        neurons_list = sorted(catalog["neurons"], key=lambda r: int(r["k"]))
        hook_layers = sorted({int(n["layer"]) for n in neurons_list})
        neurons_by_layer: dict[int, list[int]] = {L: [] for L in hook_layers}
        for rec in neurons_list:
            neurons_by_layer[int(rec["layer"])].append(int(rec["neuron"]))
        docs_it = iter_fineweb_disjoint_rows(min_row_idx=FINEWEB_MIN_ROW_IDX)
        sample_seqs = collect_fixed_length_sequences(
            docs_with_ids=docs_it,
            tokenizer=tokenizer,
            n_sequences=max(args.micro_batch_size, 2),
            seq_len=args.seq_len,
            max_seqs_per_doc=args.max_seqs_per_doc,
            min_doc_id=FINEWEB_MIN_ROW_IDX,
        )
        parity_batch = torch.tensor(
            [sample_seqs[0].token_ids[:32], sample_seqs[1].token_ids[:32]], dtype=torch.long
        )
        stress_batch = torch.tensor(
            [s.token_ids for s in sample_seqs[: args.micro_batch_size]], dtype=torch.long
        )
        report = run_parity_gate(
            backbone=backbone,
            parity_input_ids=parity_batch,
            hook_layers=hook_layers,
            neurons_by_layer=neurons_by_layer,
            split_idx=args.split_idx,
            compute_device=args.device,
            vram_stress_batch=stress_batch,
            max_vram_gb=args.max_vram_gb,
            output_path=dirs["results"] / "parity_check.json",
        )
        print(json.dumps(report, indent=2))
        return

    run_extraction_4b(
        backbone=backbone,
        tokenizer=tokenizer,
        catalog=catalog,
        num_macro_chunks=args.macro_chunks,
        seqs_per_macro_chunk=args.seqs_per_chunk,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        max_seqs_per_doc=args.max_seqs_per_doc,
        split_idx=args.split_idx,
        compute_device=args.device,
        max_vram_gb=args.max_vram_gb,
    )


if __name__ == "__main__":
    main()
