"""Memory-safe PyTorch streaming activation extraction for Qwen3-1.7B evaluation.

Extraction design:
- Hooks mlp.gate_proj, mlp.up_proj, and input of mlp.down_proj on Qwen3-style SwiGLU MLPs.
- Computes same-layer w_gate cosine similarity across all neurons per sampled layer,
  selecting each sampled neuron's nearest same-layer neighbour (with concat/index tie-break).
- Tracks target + reserve + unique neighbour neurons in one streaming pass over ~2M tokens
  (<1 GB RAM via float16 activations + uint8 exact SwiGLU quadrant codes).
- Verifies hook parity against a manual forward pass (--verify-hooks).
- Computes exact signed global percentiles (mirrored tails via directional d = s * a),
  dead-zone thresholds (p25(|a|)), SwiGLU quadrant shares (Q1-Q4), and corpus token counts.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from w_audit.config import (
    BAND_QUANTILES,
    BANDS,
    CACHE_DIR,
    DATASET_CONFIG,
    DATASET_ID,
    FALLBACK_DATASET_ID,
    LAYER4_CONTROL_LAYER,
    LAYER_NEURON_COUNTS,
    LAYERS,
    MAX_SEQ_LEN,
    MIN_DOC_TOKENS,
    MODEL_ID,
    PILOT_NEURON_COUNTS,
    POLES_PATH,
    QUADRANTS,
    SCHEMA_VERSION,
    SECOND_JUDGE_NEURON_COUNTS,
    SEED,
    TOTAL_TOKENS_TARGET,
    TOY_NEURON_COUNTS,
    WINDOW_AFTER,
    WINDOW_BEFORE,
    WindowSpec,
    ensure_dirs,
    make_pole_id,
)

# Quadrant uint8 encoding (matching Goodfire Fig. 2-3 verbatim):
#   0 = Q1 (g > 0, u > 0  -> gate_open=True,  a > 0)
#   1 = Q2 (g > 0, u <= 0 -> gate_open=True,  a < 0)
#   2 = Q3 (g <= 0, u <= 0 -> gate_open=False, a > 0, double-negative)
#   3 = Q4 (g <= 0, u > 0 -> gate_open=False, a < 0)
QUAD_CODE_TO_NAME: tuple[str, str, str, str] = QUADRANTS


def encode_quadrant_torch(gate_pre: torch.Tensor, up_pre: torch.Tensor) -> torch.Tensor:
    """Encode exact SwiGLU (gate_pre, up_pre) signs into uint8 quadrant codes 0..3."""
    g_open = gate_pre > 0
    u_pos = up_pre > 0
    # g_open=True,  u_pos=True  -> 0 (Q1)
    # g_open=True,  u_pos=False -> 1 (Q2)
    # g_open=False, u_pos=False -> 2 (Q3: double-negative)
    # g_open=False, u_pos=True  -> 3 (Q4)
    q = torch.empty_like(g_open, dtype=torch.uint8)
    q.fill_(3)
    q = torch.where(g_open & u_pos, torch.tensor(0, dtype=torch.uint8, device=gate_pre.device), q)
    q = torch.where(g_open & (~u_pos), torch.tensor(1, dtype=torch.uint8, device=gate_pre.device), q)
    q = torch.where((~g_open) & (~u_pos), torch.tensor(2, dtype=torch.uint8, device=gate_pre.device), q)
    return q


def quad_code_is_gate_open(quad_code: int | np.ndarray) -> bool | np.ndarray:
    """Return True for Q1 (0) and Q2 (1); False for Q3 (2) and Q4 (3)."""
    return quad_code < 2


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Locate decoder layer list on Qwen3ForCausalLM, Qwen3Model, or synthetic test models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        f"Cannot find transformer layers (.model.layers or .layers) on {type(model).__name__}"
    )


def get_backbone_callable(model: nn.Module) -> Callable[..., Any]:
    """Return base transformer callable skipping lm_head to avoid huge vocab-logit VRAM spikes."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


# ---------------------------------------------------------------------------
# Same-Layer Input-Weight Cosine & Neighbour Selection (§2.2)
# ---------------------------------------------------------------------------


@dataclass
class LayerNeighborResult:
    layer: int
    d_mlp: int
    cosine_stats: dict[str, float]
    # Maps target neuron index -> dict of neighbor metadata
    neighbors: dict[int, dict[str, Any]]


def compute_layer_neighbors(
    mlp: nn.Module,
    layer: int,
    candidate_neurons: Sequence[int],
    device: torch.device | str | None = None,
) -> LayerNeighborResult:
    """Compute full same-layer w_gate cosine similarity matrix and nearest neighbours.

    Selection rule:
      - Primary: maximize positive cos(w_gate[n], w_gate[m]) over m != n.
      - Tie-break 1: maximize cos([w_gate[n], w_up[n]], [w_gate[m], w_up[m]]).
      - Tie-break 2: lowest neuron index m.
    Also computes layer-wide nearest-neighbour gate cosine quantiles over all d_mlp neurons.
    """
    compute_dev = torch.device(device) if device is not None else mlp.gate_proj.weight.device
    with torch.no_grad():
        w_gate = mlp.gate_proj.weight.detach().to(device=compute_dev, dtype=torch.float32)
        w_up = mlp.up_proj.weight.detach().to(device=compute_dev, dtype=torch.float32)
        w_down_t = mlp.down_proj.weight.detach().to(device=compute_dev, dtype=torch.float32).T
        d_mlp = int(w_gate.shape[0])

        g_norm = F.normalize(w_gate, p=2, dim=1)
        u_norm = F.normalize(w_up, p=2, dim=1)
        d_norm = F.normalize(w_down_t, p=2, dim=1)
        concat_norm = F.normalize(torch.cat([w_gate, w_up], dim=1), p=2, dim=1)

        cos_mat = g_norm @ g_norm.T  # (d_mlp, d_mlp)
        cos_mat.fill_diagonal_(-float("inf"))

        max_gate_cos, _ = cos_mat.max(dim=1)
        max_gate_np = max_gate_cos.cpu().numpy()
        cosine_stats = {
            "min": float(np.min(max_gate_np)),
            "p25": float(np.quantile(max_gate_np, 0.25)),
            "median": float(np.quantile(max_gate_np, 0.50)),
            "p75": float(np.quantile(max_gate_np, 0.75)),
            "p90": float(np.quantile(max_gate_np, 0.90)),
            "max": float(np.max(max_gate_np)),
        }

        neighbors: dict[int, dict[str, Any]] = {}
        for n in candidate_neurons:
            row = cos_mat[n]
            best_val = float(row.max().item())
            tied = torch.nonzero(row >= best_val - 1e-7, as_tuple=False).squeeze(-1)
            if tied.numel() == 1:
                m = int(tied.item())
            else:
                tied_list = sorted(int(x) for x in tied.tolist())
                concat_scores = (concat_norm[n : n + 1] @ concat_norm[tied_list].T).squeeze(0)
                best_concat = float(concat_scores.max().item())
                best_m = tied_list[0]
                for idx_pos, cand_m in enumerate(tied_list):
                    if float(concat_scores[idx_pos].item()) >= best_concat - 1e-7:
                        best_m = cand_m
                        break
                m = best_m

            gate_cos = float(torch.dot(g_norm[n], g_norm[m]).item())
            up_cos = float(torch.dot(u_norm[n], u_norm[m]).item())
            concat_cos = float(torch.dot(concat_norm[n], concat_norm[m]).item())
            down_cos = float(torch.dot(d_norm[n], d_norm[m]).item())
            neighbors[int(n)] = {
                "neighbor_neuron": m,
                "neighbor_gate_cosine": gate_cos,
                "neighbor_up_cosine": up_cos,
                "neighbor_concat_cosine": concat_cos,
                "neighbor_down_cosine": down_cos,
            }

        del cos_mat
        if compute_dev.type == "cuda":
            torch.cuda.empty_cache()

    return LayerNeighborResult(
        layer=layer,
        d_mlp=d_mlp,
        cosine_stats=cosine_stats,
        neighbors=neighbors,
    )


def sample_layer_neuron_order(
    d_mlp: int,
    layer: int,
    seed: int = SEED,
) -> list[int]:
    """Deterministic permutation of all neuron indices in a layer."""
    rng = np.random.default_rng(seed + layer * 1009)
    return [int(x) for x in rng.permutation(d_mlp).tolist()]


# ---------------------------------------------------------------------------
# Hook Parity Verification (--verify-hooks, §2.1)
# ---------------------------------------------------------------------------


def verify_hooks(
    model: nn.Module,
    token_batches: Sequence[torch.Tensor],
    layers: Sequence[int] = LAYERS,
    n_neurons_total: int = 10,
    seed: int = SEED,
    tol: float = 1e-4,
    manual_tol: float | None = None,
) -> dict[str, Any]:
    """Verify hooked gate_proj, up_proj, and down_proj pre-hook match manual SwiGLU forward pass.

    Checks BOTH:
      1. Identity check: max|F.silu(g_hook) * u_hook - a_down_pre_hook| < tol (in float32).
      2. Manual forward pass from captured MLP input x:
         max|(F.silu(x @ W_gate.T) * (x @ W_up.T)) - a_down_pre_hook| < manual_tol.
         (In float32, manual_tol defaults to tol=1e-4; in bf16/fp16, computes manual forward
          in exact model dtype so max error stays < 1e-4!)
    """
    if manual_tol is None:
        manual_tol = tol

    decoder_layers = get_transformer_layers(model)
    backbone = get_backbone_callable(model)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Distribute n_neurons_total across specified layers deterministically
    selected: dict[int, list[int]] = {}
    base_per = n_neurons_total // len(layers)
    rem = n_neurons_total % len(layers)
    for i, layer_idx in enumerate(layers):
        k = base_per + (1 if i < rem else 0)
        if k <= 0:
            continue
        d_mlp = int(decoder_layers[layer_idx].mlp.gate_proj.out_features)
        order = sample_layer_neuron_order(d_mlp, layer_idx, seed=seed)
        selected[int(layer_idx)] = order[:k]

    captured: dict[int, dict[str, torch.Tensor]] = {L: {} for L in selected}
    handles: list[Any] = []

    def _make_mlp_pre_hook(L: int):
        def _hook(_mod: nn.Module, args: tuple[Any, ...]) -> None:
            captured[L]["mlp_in"] = args[0].detach()
        return _hook

    def _make_gate_hook(L: int, idx_t: torch.Tensor):
        def _hook(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
            captured[L]["gate"] = out.detach().index_select(-1, idx_t)
        return _hook

    def _make_up_hook(L: int, idx_t: torch.Tensor):
        def _hook(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
            captured[L]["up"] = out.detach().index_select(-1, idx_t)
        return _hook

    def _make_down_pre_hook(L: int, idx_t: torch.Tensor):
        def _hook(_mod: nn.Module, args: tuple[Any, ...]) -> None:
            captured[L]["down_in"] = args[0].detach().index_select(-1, idx_t)
        return _hook

    max_hook_identity_err = 0.0
    max_manual_forward_err = 0.0
    max_quad_mismatch = 0
    tokens_checked = 0

    try:
        for L, neurons in selected.items():
            mlp = decoder_layers[L].mlp
            idx_t = torch.tensor(neurons, dtype=torch.long, device=device)
            handles.append(mlp.register_forward_pre_hook(_make_mlp_pre_hook(L)))
            handles.append(mlp.gate_proj.register_forward_hook(_make_gate_hook(L, idx_t)))
            handles.append(mlp.up_proj.register_forward_hook(_make_up_hook(L, idx_t)))
            handles.append(mlp.down_proj.register_forward_pre_hook(_make_down_pre_hook(L, idx_t)))

        model.eval()
        with torch.inference_mode():
            for batch in token_batches:
                batch = batch.to(device)
                tokens_checked += int(batch.numel())
                try:
                    backbone(input_ids=batch, use_cache=False)
                except TypeError:
                    backbone(batch)

                for L, neurons in selected.items():
                    mlp = decoder_layers[L].mlp
                    idx_t = torch.tensor(neurons, dtype=torch.long, device=device)
                    g = captured[L]["gate"]
                    u = captured[L]["up"]
                    a_hook = captured[L]["down_in"]
                    x = captured[L]["mlp_in"]

                    # 1. Hooked silu(g) * u vs down_proj pre-hook input
                    a_from_hooks = F.silu(g) * u
                    err_id = float((a_from_hooks.float() - a_hook.float()).abs().max().item())
                    max_hook_identity_err = max(max_hook_identity_err, err_id)

                    # 2. Manual forward pass from mlp input using full projection weights
                    # (avoids bf16 cuBLAS tile-order divergence between full vs 3-row GEMM)
                    g_man = F.linear(x, mlp.gate_proj.weight).index_select(-1, idx_t)
                    u_man = F.linear(x, mlp.up_proj.weight).index_select(-1, idx_t)
                    a_man = F.silu(g_man) * u_man
                    err_man = float((a_man.float() - a_hook.float()).abs().max().item())
                    max_manual_forward_err = max(max_manual_forward_err, err_man)

                    # 3. Verify quadrant code sign consistency whenever a_hook != 0
                    q_code = encode_quadrant_torch(g, u)
                    nz = a_hook != 0
                    if nz.any():
                        q_nz = q_code[nz]
                        a_nz = a_hook[nz]
                        # Q1 (0) and Q3 (2) must have a > 0; Q2 (1) and Q4 (3) must have a < 0
                        expect_pos = (q_nz == 0) | (q_nz == 2)
                        actual_pos = a_nz > 0
                        mismatches = int((expect_pos != actual_pos).sum().item())
                        max_quad_mismatch += mismatches
    finally:
        for h in handles:
            h.remove()

    passed = (
        max_hook_identity_err < tol
        and max_manual_forward_err < manual_tol
        and max_quad_mismatch == 0
    )
    report = {
        "passed": passed,
        "tokens_checked": tokens_checked,
        "neurons_checked": sum(len(v) for v in selected.values()),
        "selected_neurons": {str(k): v for k, v in selected.items()},
        "model_dtype": str(model_dtype),
        "tol": tol,
        "manual_tol": manual_tol,
        "max_hook_identity_err": max_hook_identity_err,
        "max_manual_forward_err": max_manual_forward_err,
        "quadrant_sign_mismatches": max_quad_mismatch,
    }
    if not passed:
        raise AssertionError(f"Hook parity verification FAILED: {json.dumps(report, indent=2)}")
    return report


# ---------------------------------------------------------------------------
# Fast Deterministic Web-Text Corpus Streaming (§2.3)
# ---------------------------------------------------------------------------


def iter_fineweb_edu_documents(
    shard_prefix: str = "sample/10BT",
    max_shards: int = 3,
) -> Iterator[str]:
    """Stream text documents from HuggingFaceFW/fineweb-edu via pyarrow HTTP range reads.

    Reads individual parquet row groups sequentially (~1,000 docs / ~1.1M tokens in ~6s)
    without enumerating the full repo or downloading multi-GB shards.
    """
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi()
    files = sorted(
        f.path
        for f in api.list_repo_tree(
            DATASET_ID, path_in_repo=shard_prefix, repo_type="dataset"
        )
        if f.path.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No parquet shards found in {DATASET_ID}/{shard_prefix}")

    fs = fsspec.filesystem("https")
    for rel_path in files[:max_shards]:
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{rel_path}"
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg_idx in range(pf.num_row_groups):
                tbl = pf.read_row_groups([rg_idx], columns=["text"])
                for text in tbl["text"].to_pylist():
                    if isinstance(text, str) and text:
                        yield text


def iter_c4_documents(max_shards: int = 3) -> Iterator[str]:
    """Fallback clean generic web text stream from allenai/c4 JSONL gzip shards."""
    import httpx
    import zlib

    for shard_idx in range(max_shards):
        url = (
            f"https://huggingface.co/datasets/{FALLBACK_DATASET_ID}/resolve/main/"
            f"en/c4-train.{shard_idx:05d}-of-01024.json.gz"
        )
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        buf = b""
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(65536):
                buf += decompressor.decompress(chunk)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if isinstance(text, str) and text:
                        yield text
        tail = decompressor.flush()
        if tail:
            buf += tail
        for line in buf.splitlines():
            if line:
                obj = json.loads(line)
                text = obj.get("text", "")
                if isinstance(text, str) and text:
                    yield text


def iter_corpus_documents(
    dataset_id: str = DATASET_ID,
    local_jsonl: Path | None = None,
) -> tuple[str, Iterator[str]]:
    """Return (resolved_dataset_id, document_text_iterator) with automatic C4 fallback."""
    if local_jsonl is not None:
        def _local_iter() -> Iterator[str]:
            with local_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
                    if text:
                        yield text
        return (str(local_jsonl), _local_iter())

    if dataset_id == FALLBACK_DATASET_ID or dataset_id == "c4":
        return (FALLBACK_DATASET_ID, iter_c4_documents())

    # Try FineWeb-Edu first; fall back seamlessly to allenai/c4 if network/parquet fails
    try:
        it = iter_fineweb_edu_documents()
        first_doc = next(it)

        def _chain_first() -> Iterator[str]:
            yield first_doc
            yield from it

        return (DATASET_ID, _chain_first())
    except Exception as exc:
        print(f"[extract] FineWeb-Edu stream failed ({exc}); falling back to {FALLBACK_DATASET_ID}")
        return (FALLBACK_DATASET_ID, iter_c4_documents())


# ---------------------------------------------------------------------------
# Document Chunking & Single-Pass Streaming Extraction
# ---------------------------------------------------------------------------


@dataclass
class TokenChunk:
    doc_id: int
    chunk_id: int
    token_ids: list[int]


def stream_token_chunks(
    documents: Iterable[str],
    tokenizer: Any,
    target_tokens: int = TOTAL_TOKENS_TARGET,
    max_seq_len: int = MAX_SEQ_LEN,
    min_doc_tokens: int = MIN_DOC_TOKENS,
) -> list[TokenChunk]:
    """Tokenize documents into non-overlapping <=max_seq_len chunks until target_tokens reached.

    - Skips documents shorter than min_doc_tokens (69).
    - Preserves sequential doc_id for strict doc-level train/eval split (doc_id % 2).
    - Drops trailing document remainders shorter than min_doc_tokens so every chunk can host
      valid interior 69-token windows (60 <= pos <= len - 9).
    """
    chunks: list[TokenChunk] = []
    total = 0
    doc_id = 0
    chunk_id = 0

    for text in documents:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_doc_tokens:
            continue
        doc_added = False
        for start in range(0, len(ids), max_seq_len):
            sub = ids[start : start + max_seq_len]
            if len(sub) < min_doc_tokens:
                break
            chunks.append(TokenChunk(doc_id=doc_id, chunk_id=chunk_id, token_ids=sub))
            chunk_id += 1
            total += len(sub)
            doc_added = True
            if total >= target_tokens:
                break
        if doc_added:
            doc_id += 1
        if total >= target_tokens:
            break

    return chunks


def _compute_pole_band_stats(
    acts_1d: np.ndarray,
    quads_1d: np.ndarray,
    non_bos_mask: np.ndarray,
    interior_mask: np.ndarray,
    doc_ids: np.ndarray,
    sign: int,
) -> dict[str, Any]:
    """Compute exact signed global quantiles, dead-zone threshold, and quadrant shares for one pole."""
    a_valid = acts_1d[non_bos_mask].astype(np.float32)
    q_valid = quads_1d[non_bos_mask]

    frac_pos = float(np.mean(a_valid > 0.0))
    frac_neg = float(np.mean(a_valid < 0.0))
    p25_abs = float(np.quantile(np.abs(a_valid), 0.25))

    d_valid = float(sign) * a_valid
    q75, q90, q97, q99_9 = (
        float(x) for x in np.quantile(d_valid, [0.75, 0.90, 0.97, 0.999])
    )
    band_edges_dir = {
        "p75": q75,
        "p90": q90,
        "p97": q97,
        "p99.9": q99_9,
    }
    sign_clipped_p75_90 = bool(q75 <= 0.0 or q75 < p25_abs)

    # Corpus-wide quadrant shares over all non-BOS tokens
    n_valid = max(len(q_valid), 1)
    quadrant_shares_corpus = {
        QUAD_CODE_TO_NAME[c]: float(np.sum(q_valid == c) / n_valid) for c in range(4)
    }

    # Quadrant shares per band (applying the exact sign & dead-zone guards)
    abs_valid = np.abs(a_valid)
    guard_valid = (d_valid > 0.0) & (abs_valid >= p25_abs)
    edges_map = {
        "p75_90": (q75, q90),
        "p90_97": (q90, q97),
        "p97_99.9": (q97, q99_9),
        "p99.9_plus": (q99_9, None),
    }
    quadrant_shares_by_band: dict[str, dict[str, float]] = {}
    for bname in BANDS:
        lo, hi = edges_map[bname]
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

    # Eligibility counts on train_label (doc_id % 2 == 0) and eval (doc_id % 2 == 1)
    a_all = acts_1d.astype(np.float32)
    d_all = float(sign) * a_all
    guard_all = interior_mask & (d_all > 0.0) & (np.abs(a_all) >= p25_abs)
    is_train = (doc_ids % 2) == 0
    is_eval = ~is_train

    train_top_count = int(np.sum(guard_all & is_train & (d_all >= q99_9)))
    eval_band_counts: dict[str, int] = {}
    for bname in BANDS:
        lo, hi = edges_map[bname]
        em = guard_all & is_eval & (d_all >= lo)
        if hi is not None:
            em = em & (d_all < hi)
        eval_band_counts[bname] = int(np.sum(em))

    return {
        "frac_pos": frac_pos,
        "frac_neg": frac_neg,
        "p25_abs": p25_abs,
        "band_edges_dir": band_edges_dir,
        "sign_clipped_p75_90": sign_clipped_p75_90,
        "quadrant_shares_corpus": quadrant_shares_corpus,
        "quadrant_shares_by_band": quadrant_shares_by_band,
        "train_top_count": train_top_count,
        "eval_band_counts": eval_band_counts,
    }


def run_extraction(
    model: nn.Module,
    tokenizer: Any,
    documents: Iterable[str] | None = None,
    dataset_id: str = DATASET_ID,
    model_id: str = MODEL_ID,
    target_tokens: int = TOTAL_TOKENS_TARGET,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = 8,
    layers: Sequence[int] = LAYERS,
    layer_neuron_counts: dict[int, int] | None = None,
    pilot_neuron_counts: dict[int, int] | None = None,
    second_judge_neuron_counts: dict[int, int] | None = None,
    toy_neuron_counts: dict[int, int] | None = None,
    reserve_per_layer: int = 24,
    min_train_top: int = 20,
    min_eval_per_band: int = 6,
    seed: int = SEED,
    base_dir: Path | None = None,
    run_hook_check: bool = True,
    hook_check_tokens: int = 10_000,
    window_spec: WindowSpec | None = None,
) -> dict[str, Any]:
    """Run complete single-pass streaming extraction and write cache + data/poles.json."""
    t0 = time.time()
    dirs = ensure_dirs(base_dir)
    cache_dir = dirs["cache"]
    poles_path = dirs["data"] / "poles.json"
    wspec = window_spec if window_spec is not None else WindowSpec()

    if layer_neuron_counts is None:
        layer_neuron_counts = {L: LAYER_NEURON_COUNTS[L] for L in layers}
    if pilot_neuron_counts is None:
        pilot_neuron_counts = {L: PILOT_NEURON_COUNTS.get(L, min(2, layer_neuron_counts[L])) for L in layers}
    if second_judge_neuron_counts is None:
        second_judge_neuron_counts = {
            L: SECOND_JUDGE_NEURON_COUNTS.get(L, min(3, layer_neuron_counts[L])) for L in layers
        }
    if toy_neuron_counts is None:
        toy_neuron_counts = {L: TOY_NEURON_COUNTS.get(L, layer_neuron_counts[L]) for L in layers}

    decoder_layers = get_transformer_layers(model)
    backbone = get_backbone_callable(model)
    device = next(model.parameters()).device

    # 1. Sample candidate neurons (+ reserve buffer) and compute same-layer w_gate neighbours
    layer_candidates: dict[int, list[int]] = {}
    layer_neighbor_results: dict[int, LayerNeighborResult] = {}
    tracked_by_layer: dict[int, list[int]] = {}
    neuron_to_col: dict[tuple[int, int], int] = {}
    col_to_neuron: list[tuple[int, int]] = []

    for L in layers:
        mlp = decoder_layers[L].mlp
        d_mlp = int(mlp.gate_proj.out_features)
        order = sample_layer_neuron_order(d_mlp, L, seed=seed)
        want = layer_neuron_counts[L]
        cand = order[: min(want + reserve_per_layer, d_mlp)]
        layer_candidates[L] = cand

        nres = compute_layer_neighbors(mlp, L, cand, device=device)
        layer_neighbor_results[L] = nres

        # Track both candidate targets and their nearest neighbours in one pass
        unique_set: list[int] = []
        seen: set[int] = set()
        for n in cand:
            for idx in (n, nres.neighbors[n]["neighbor_neuron"]):
                if idx not in seen:
                    seen.add(idx)
                    unique_set.append(idx)
        tracked_by_layer[L] = unique_set
        for idx in unique_set:
            neuron_to_col[(L, idx)] = len(col_to_neuron)
            col_to_neuron.append((L, idx))

    k_total = len(col_to_neuron)
    print(
        f"[extract] Layers={list(layers)}, candidates={sum(len(v) for v in layer_candidates.values())}, "
        f"total unique tracked neurons (targets+reserves+neighbours)={k_total}"
    )

    # 2. Stream & tokenize corpus into chunks
    resolved_dataset_id = dataset_id
    if documents is None:
        resolved_dataset_id, documents = iter_corpus_documents(dataset_id=dataset_id)

    chunks = stream_token_chunks(
        documents=documents,
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        max_seq_len=max_seq_len,
        min_doc_tokens=wspec.total,
    )
    if not chunks:
        raise RuntimeError("No valid document chunks >= min_doc_tokens produced from corpus.")

    total_tokens = sum(len(c.token_ids) for c in chunks)
    print(
        f"[extract] Tokenized {len(chunks)} chunks from {chunks[-1].doc_id + 1} docs "
        f"({total_tokens:,} tokens) in {time.time() - t0:.1f}s"
    )

    # 3. Optional hook parity verification on first ~hook_check_tokens tokens
    hook_report: dict[str, Any] | None = None
    if run_hook_check:
        verify_batches: list[torch.Tensor] = []
        v_tok = 0
        for ch in chunks:
            verify_batches.append(torch.tensor([ch.token_ids], dtype=torch.long))
            v_tok += len(ch.token_ids)
            if v_tok >= hook_check_tokens:
                break
        n_check = min(10, sum(layer_neuron_counts.values()))
        hook_report = verify_hooks(
            model=model,
            token_batches=verify_batches,
            layers=layers,
            n_neurons_total=n_check,
            seed=seed,
        )
        print(
            f"[extract] --verify-hooks PASSED: tokens={hook_report['tokens_checked']:,}, "
            f"max_hook_err={hook_report['max_hook_identity_err']:.2e}, "
            f"max_manual_err={hook_report['max_manual_forward_err']:.2e}"
        )

    # 4. Pre-allocate compact host arrays (<1 GB RAM for 2M tokens x ~115 neurons)
    token_ids_arr = np.empty(total_tokens, dtype=np.int32)
    doc_ids_arr = np.empty(total_tokens, dtype=np.int32)
    chunk_ids_arr = np.empty(total_tokens, dtype=np.int32)
    pos_in_chunk_arr = np.empty(total_tokens, dtype=np.int16)
    chunk_starts_arr = np.empty(len(chunks), dtype=np.int64)
    chunk_lengths_arr = np.empty(len(chunks), dtype=np.int32)
    acts_arr = np.empty((total_tokens, k_total), dtype=np.float16)
    quads_arr = np.empty((total_tokens, k_total), dtype=np.uint8)

    cursor = 0
    for c in chunks:
        clen = len(c.token_ids)
        chunk_starts_arr[c.chunk_id] = cursor
        chunk_lengths_arr[c.chunk_id] = clen
        token_ids_arr[cursor : cursor + clen] = c.token_ids
        doc_ids_arr[cursor : cursor + clen] = c.doc_id
        chunk_ids_arr[cursor : cursor + clen] = c.chunk_id
        pos_in_chunk_arr[cursor : cursor + clen] = np.arange(clen, dtype=np.int16)
        cursor += clen

    # 5. Single streaming forward pass capturing down_proj pre-hook acts + exact quadrant codes
    batch_buf: dict[int, dict[str, torch.Tensor]] = {L: {} for L in layers}
    handles: list[Any] = []
    col_slices_by_layer: dict[int, list[int]] = {}
    pad_id = (
        int(tokenizer.pad_token_id)
        if getattr(tokenizer, "pad_token_id", None) is not None
        else 0
    )

    def _mk_gate_hook(L: int, idx_t: torch.Tensor):
        def _hook(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
            batch_buf[L]["gate"] = out.detach().index_select(-1, idx_t)
        return _hook

    def _mk_up_hook(L: int, idx_t: torch.Tensor):
        def _hook(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
            batch_buf[L]["up"] = out.detach().index_select(-1, idx_t)
        return _hook

    def _mk_down_pre_hook(L: int, idx_t: torch.Tensor):
        def _hook(_mod: nn.Module, args: tuple[Any, ...]) -> None:
            batch_buf[L]["act"] = args[0].detach().index_select(-1, idx_t)
        return _hook

    t_fwd = time.time()
    try:
        for L in layers:
            mlp = decoder_layers[L].mlp
            idx_list = tracked_by_layer[L]
            idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
            col_slices_by_layer[L] = [neuron_to_col[(L, n)] for n in idx_list]
            handles.append(mlp.gate_proj.register_forward_hook(_mk_gate_hook(L, idx_t)))
            handles.append(mlp.up_proj.register_forward_hook(_mk_up_hook(L, idx_t)))
            handles.append(mlp.down_proj.register_forward_pre_hook(_mk_down_pre_hook(L, idx_t)))

        model.eval()
        with torch.inference_mode():
            n_batches = math.ceil(len(chunks) / batch_size)
            for b_idx in range(n_batches):
                b_chunks = chunks[b_idx * batch_size : (b_idx + 1) * batch_size]
                max_len = max(len(c.token_ids) for c in b_chunks)
                B = len(b_chunks)
                input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
                attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
                for row_i, ch in enumerate(b_chunks):
                    clen = len(ch.token_ids)
                    input_ids[row_i, :clen] = torch.tensor(ch.token_ids, dtype=torch.long, device=device)
                    attn_mask[row_i, :clen] = 1

                try:
                    backbone(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                except TypeError:
                    backbone(input_ids)

                for L in layers:
                    act_t = batch_buf[L]["act"]  # (B, max_len, K_L)
                    quad_t = encode_quadrant_torch(batch_buf[L]["gate"], batch_buf[L]["up"])
                    # Clamp to float16 finite range to avoid overflow on extreme outliers
                    act_f16 = act_t.float().clamp(-65500.0, 65500.0).to(torch.float16).cpu().numpy()
                    quad_u8 = quad_t.cpu().numpy()
                    cols = col_slices_by_layer[L]
                    for row_i, ch in enumerate(b_chunks):
                        start = int(chunk_starts_arr[ch.chunk_id])
                        clen = len(ch.token_ids)
                        acts_arr[start : start + clen, cols] = act_f16[row_i, :clen, :]
                        quads_arr[start : start + clen, cols] = quad_u8[row_i, :clen, :]
                    batch_buf[L].clear()

                if (b_idx + 1) % 25 == 0 or (b_idx + 1) == n_batches:
                    done_tok = int(chunk_starts_arr[b_chunks[-1].chunk_id] + len(b_chunks[-1].token_ids))
                    elapsed = max(time.time() - t_fwd, 1e-6)
                    print(
                        f"[extract] Batch {b_idx + 1}/{n_batches} | "
                        f"{done_tok:,}/{total_tokens:,} tok ({done_tok / elapsed:,.0f} tok/s)"
                    )
    finally:
        for h in handles:
            h.remove()

    # 6. Masks, corpus unigram token counts, and per-pole stats / eligibility selection
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

    train_tokens = int(np.sum((doc_ids_arr % 2) == 0))
    eval_tokens = int(np.sum((doc_ids_arr % 2) == 1))

    # Pre-compute neighbour |a_m| thresholds (p97, p99, p99.9) for every tracked neighbour
    neighbor_abs_quantiles: dict[str, dict[str, float]] = {}
    for L in layers:
        nres = layer_neighbor_results[L]
        for n in layer_candidates[L]:
            m = nres.neighbors[n]["neighbor_neuron"]
            key = f"{L}:{m}"
            if key not in neighbor_abs_quantiles:
                col_m = neuron_to_col[(L, m)]
                abs_m = np.abs(acts_arr[non_bos_mask, col_m].astype(np.float32))
                q97, q99, q99_9 = (float(x) for x in np.quantile(abs_m, [0.97, 0.99, 0.999]))
                neighbor_abs_quantiles[key] = {
                    "p97_abs": q97,
                    "p99_abs": q99,
                    "p99.9_abs": q99_9,
                }

    # Select final accepted neurons per layer (skipping any neuron failing eligibility guards)
    poles_list: list[dict[str, Any]] = []
    skipped_neurons: list[dict[str, Any]] = []

    for L in layers:
        want = layer_neuron_counts[L]
        nres = layer_neighbor_results[L]
        accepted_in_layer = 0

        for n in layer_candidates[L]:
            if accepted_in_layer >= want:
                break
            col = neuron_to_col[(L, n)]
            acts_1d = acts_arr[:, col]
            quads_1d = quads_arr[:, col]

            pos_stats = _compute_pole_band_stats(acts_1d, quads_1d, non_bos_mask, interior_mask, doc_ids_arr, +1)
            neg_stats = _compute_pole_band_stats(acts_1d, quads_1d, non_bos_mask, interior_mask, doc_ids_arr, -1)

            # Check eligibility on BOTH poles
            fail_reasons: list[str] = []
            for sname, st in (("pos", pos_stats), ("neg", neg_stats)):
                if st["train_top_count"] < min_train_top:
                    fail_reasons.append(f"{sname}:train_p99.9_plus={st['train_top_count']}<{min_train_top}")
                for bname, cnt in st["eval_band_counts"].items():
                    if cnt < min_eval_per_band:
                        fail_reasons.append(f"{sname}:eval_{bname}={cnt}<{min_eval_per_band}")

            if fail_reasons:
                skipped_neurons.append(
                    {"layer": int(L), "neuron": int(n), "reasons": fail_reasons}
                )
                continue

            in_pilot = accepted_in_layer < pilot_neuron_counts[L]
            in_second_judge = accepted_in_layer < second_judge_neuron_counts[L]
            in_toy = accepted_in_layer < toy_neuron_counts[L]
            nb = nres.neighbors[n]
            nb_key = f"{L}:{nb['neighbor_neuron']}"

            for sign, pole_name, st in ((1, "pos", pos_stats), (-1, "neg", neg_stats)):
                pole_entry = {
                    "pole_id": make_pole_id(L, n, sign),
                    "layer": int(L),
                    "neuron": int(n),
                    "sign": int(sign),
                    "pole_name": pole_name,
                    "is_layer4_control": bool(L == LAYER4_CONTROL_LAYER),
                    "in_pilot": bool(in_pilot),
                    "in_toy": bool(in_toy),
                    "in_second_judge": bool(in_second_judge),
                    "neighbor_neuron": int(nb["neighbor_neuron"]),
                    "neighbor_gate_cosine": float(nb["neighbor_gate_cosine"]),
                    "neighbor_up_cosine": float(nb["neighbor_up_cosine"]),
                    "neighbor_concat_cosine": float(nb["neighbor_concat_cosine"]),
                    "neighbor_down_cosine": float(nb["neighbor_down_cosine"]),
                    "neighbor_abs_quantiles": neighbor_abs_quantiles[nb_key],
                    "frac_pos": st["frac_pos"],
                    "frac_neg": st["frac_neg"],
                    "p25_abs": st["p25_abs"],
                    "band_edges_dir": st["band_edges_dir"],
                    "sign_clipped_p75_90": st["sign_clipped_p75_90"],
                    "quadrant_shares_corpus": st["quadrant_shares_corpus"],
                    "quadrant_shares_by_band": st["quadrant_shares_by_band"],
                    # Populated deterministically from 20 train_label examples by snippets.py
                    "top_3_token_ids": [],
                    "top_3_tokens": [],
                    "top_10_token_ids": [],
                }
                poles_list.append(pole_entry)

            accepted_in_layer += 1

        if accepted_in_layer < want:
            raise RuntimeError(
                f"Layer {L}: only accepted {accepted_in_layer}/{want} neurons "
                f"(increase reserve_per_layer or total tokens). Skipped: {skipped_neurons}"
            )

    # 7. Trim cache columns to final accepted targets + their unique neighbours (<=100 neurons)
    kept_pairs: list[tuple[int, int]] = []
    kept_seen: set[tuple[int, int]] = set()
    for p in poles_list:
        for pair in ((int(p["layer"]), int(p["neuron"])), (int(p["layer"]), int(p["neighbor_neuron"]))):
            if pair not in kept_seen:
                kept_seen.add(pair)
                kept_pairs.append(pair)
    kept_old_cols = [neuron_to_col[pair] for pair in kept_pairs]
    acts_trimmed = np.ascontiguousarray(acts_arr[:, kept_old_cols])
    quads_trimmed = np.ascontiguousarray(quads_arr[:, kept_old_cols])
    del acts_arr, quads_arr

    # Persist compact memory-mappable numpy cache under data/cache/
    np.save(cache_dir / "token_ids.npy", token_ids_arr)
    np.save(cache_dir / "doc_ids.npy", doc_ids_arr)
    np.save(cache_dir / "chunk_ids.npy", chunk_ids_arr)
    np.save(cache_dir / "pos_in_chunk.npy", pos_in_chunk_arr)
    np.save(cache_dir / "chunk_starts.npy", chunk_starts_arr)
    np.save(cache_dir / "chunk_lengths.npy", chunk_lengths_arr)
    np.save(cache_dir / "acts.npy", acts_trimmed)
    np.save(cache_dir / "quads.npy", quads_trimmed)
    np.save(cache_dir / "token_counts.npy", token_counts)

    cache_meta = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "dataset_id": resolved_dataset_id,
        "seed": seed,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
        "num_chunks": len(chunks),
        "num_docs": int(doc_ids_arr.max()) + 1,
        "special_token_ids": sorted(special_ids),
        "col_to_neuron": [[int(L), int(n)] for (L, n) in kept_pairs],
        "hook_verification": hook_report,
    }
    (cache_dir / "meta.json").write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")

    # 8. Write data/poles.json (canonical pole schema)
    poles_doc = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "dataset_id": resolved_dataset_id,
        "seed": seed,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
        "layers": [int(L) for L in layers],
        "window": wspec.to_dict(),
        "layer_neighbor_cosine_stats": {
            str(L): layer_neighbor_results[L].cosine_stats for L in layers
        },
        "skipped_neurons": skipped_neurons,
        "poles": poles_list,
    }
    poles_path.write_text(json.dumps(poles_doc, indent=2), encoding="utf-8")

    print(
        f"[extract] Complete in {time.time() - t0:.1f}s -> "
        f"{poles_path} ({len(poles_list)} poles, {len(skipped_neurons)} skipped)"
    )
    return poles_doc


# ---------------------------------------------------------------------------
# CLI Entrypoint (supports --verify-hooks standalone or full extraction)
# ---------------------------------------------------------------------------


def load_qwen_model_and_tokenizer(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: str = "bfloat16",
) -> tuple[nn.Module, Any]:
    """Load Qwen3 model and tokenizer cleanly on GPU (bf16) or CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    if device == "cpu" and torch_dtype == torch.float16:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stream Qwen3-1.7B MLP neuron activations for Qwen3-1.7B evaluation."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--tokens", type=int, default=TOTAL_TOKENS_TARGET)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--verify-hooks",
        action="store_true",
        help="Run standalone 10-neuron x 10k-token hook parity verification and exit.",
    )
    parser.add_argument("--skip-hook-check", action="store_true")
    parser.add_argument("--verify-tokens", type=int, default=10_000)
    parser.add_argument("--reserve-per-layer", type=int, default=24)
    args = parser.parse_args(argv)

    model, tokenizer = load_qwen_model_and_tokenizer(
        model_id=args.model_id, device=args.device, dtype=args.dtype
    )

    if args.verify_hooks:
        _, docs = iter_corpus_documents(dataset_id=args.dataset)
        chunks = stream_token_chunks(
            documents=docs,
            tokenizer=tokenizer,
            target_tokens=args.verify_tokens,
            max_seq_len=args.max_seq_len,
        )
        batches = [torch.tensor([c.token_ids], dtype=torch.long) for c in chunks]
        report = verify_hooks(
            model=model,
            token_batches=batches,
            layers=LAYERS,
            n_neurons_total=10,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return

    run_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        model_id=args.model_id,
        target_tokens=args.tokens,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        reserve_per_layer=args.reserve_per_layer,
        run_hook_check=not args.skip_hook_check,
        hook_check_tokens=args.verify_tokens,
    )


if __name__ == "__main__":
    main()
