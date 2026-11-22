"""CPU unit tests for gap decomposition, multiband & FPR snippets,
and exact per-neuron band mean-ablation KL validation.

All tests run cleanly on CPU with zero GPU and zero network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen4b_eval.band_kl_4b import (
    ABLATION_CONDITIONS,
    FOUR_BAND_CONDS,
    build_neuron_ablation_masks,
    run_band_kl_validation,
    select_stratified_kl_neurons,
)
from qwen4b_eval.extract_4b import run_extraction_4b
from qwen4b_eval.fetch_ulto85 import build_active_poles_catalog
from qwen4b_eval.gap_and_bands_4b import (
    FAILURE_BUCKETS,
    assign_failure_bucket,
    build_multiband_and_fpr_snippets_4b,
    compute_gap_decomposition_4b,
)
from qwen4b_eval.snippets_4b import build_snippets_4b
from qwen4b_eval.top100_4b import run_sparsity_and_top100
from w_audit.config import WindowSpec


# ---------------------------------------------------------------------------
# Minimal Synthetic SwiGLU Transformer + Tokenizer (CPU-only)
# ---------------------------------------------------------------------------


class _TinySwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = _TinySwiGLUMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h = self.ln1(x)
        cumsum = torch.cumsum(h, dim=1)
        denom = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        x = x + 0.2 * self.attn_proj(cumsum / denom)
        x = x + self.mlp(self.ln2(x))
        return x


class _TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 16, d_mlp: int = 12, n_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_TinyBlock(d_model, d_mlp) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class _SimpleTok:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    vocab_size = 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [3 + (int(t[1:]) % 250) for t in text.split()]

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(f"t{int(i)}" for i in ids)


# ---------------------------------------------------------------------------
# 1. Failure-Bucket Precedence & Exhaustiveness Tests
# ---------------------------------------------------------------------------


def test_assign_failure_bucket_precedence_and_named_neurons() -> None:
    # info: high causal recovery -> pass_causal
    assert assign_failure_bucket(0.859, 0.958, 0.883) == "pass_causal"
    # PCR: sim_cal_rec < 0 AND sim_cal_rec_active >= 0.50 (even though strong20_rec < 0.25!)
    assert assign_failure_bucket(-0.570, 0.919, 0.144) == "overbroad_fp_damage"
    # insurance: sim_cal_rec < 0.50, active < 0.50, strong20_rec < 0.25 -> weak_mass_dominated
    assert assign_failure_bucket(-0.008, 0.040, 0.192) == "weak_mass_dominated"
    # active_sim_failure: strong20_rec >= 0.25 but active < 0.50
    assert assign_failure_bucket(0.10, 0.30, 0.45) == "active_sim_failure"
    # other_partial: sim in [0, 0.50), strong20 >= 0.25, active >= 0.50
    assert assign_failure_bucket(0.25, 0.70, 0.50) == "other_partial"


# ---------------------------------------------------------------------------
# 2. Stratified 60-Neuron KL Selector Test
# ---------------------------------------------------------------------------


def test_select_stratified_kl_neurons_includes_named_and_stratifies() -> None:
    per_neuron: dict[str, Any] = {}
    k = 0
    for L in (0, 32):
        for i in range(10):
            per_neuron[str(k)] = {
                "k": k,
                "layer": L,
                "neuron": i * 100,
                "sim_cal_rec": -0.5 + 0.15 * i,
            }
            k += 1

    # Force k=1 (L0) and k=12 (L32)
    selected = select_stratified_kl_neurons(
        {"per_neuron": per_neuron},
        layers=(0, 32),
        per_layer=4,
        must_include_ks=[1, 12],
    )
    assert len(selected) == 8
    sel_ks = {int(r["k"]) for r in selected}
    assert 1 in sel_ks and 12 in sel_ks
    assert sum(1 for r in selected if r["layer"] == 0) == 4
    assert sum(1 for r in selected if r["layer"] == 32) == 4


# ---------------------------------------------------------------------------
# 3. End-to-End Synthetic Gap, Multiband/FPR, and Band-KL Validation Test
# ---------------------------------------------------------------------------


def test_end_to_end_gap_multiband_fpr_and_band_kl_cpu(tmp_path: Path) -> None:
    torch.manual_seed(77)
    rng = np.random.default_rng(77)
    backbone = _TinyBackbone(vocab_size=256, d_model=16, d_mlp=12, n_layers=4).to(torch.float32)
    tok = _SimpleTok()

    e3_neurons = [[0, 1], [0, 4], [3, 2], [3, 7]]
    e3_explanations = {
        "0": {
            "layer": 0,
            "neuron": 1,
            "expl": {
                "label": "Uni pos 0",
                "positive_pole": "Active pos 0",
                "negative_pole": "Negligible.",
                "rubric": "R0",
            },
        },
        "1": {
            "layer": 0,
            "neuron": 4,
            "expl": {
                "label": "Bi 1",
                "positive_pole": "Active pos 1",
                "negative_pole": "Active neg 1",
                "rubric": "R1",
            },
        },
        "2": {
            "layer": 3,
            "neuron": 2,
            "expl": {
                "label": "Uni neg 2",
                "positive_pole": "Weak only.",
                "negative_pole": "Active neg 2",
                "rubric": "R2",
            },
        },
        "3": {
            "layer": 3,
            "neuron": 7,
            "expl": {
                "label": "Bi 3",
                "positive_pole": "Active pos 3",
                "negative_pole": "Active neg 3",
                "rubric": "R3",
            },
        },
    }
    e3_scores = {
        "neurons": [
            {
                "k": 0,
                "layer": 0,
                "neuron": 1,
                "M": 2.0,
                "sim_cal_rec": 0.70,
                "sim_cal_rec_active": 0.90,
                "strong20_rec": 0.80,
                "strong50_rec": 0.50,
            },
            {
                "k": 1,
                "layer": 0,
                "neuron": 4,
                "M": 2.0,
                "sim_cal_rec": -0.40,
                "sim_cal_rec_active": 0.75,
                "strong20_rec": 0.15,
                "strong50_rec": 0.05,
            },
            {
                "k": 2,
                "layer": 3,
                "neuron": 2,
                "M": 2.0,
                "sim_cal_rec": 0.05,
                "sim_cal_rec_active": 0.10,
                "strong20_rec": 0.10,
                "strong50_rec": 0.02,
            },
            {
                "k": 3,
                "layer": 3,
                "neuron": 7,
                "M": 2.0,
                "sim_cal_rec": 0.20,
                "sim_cal_rec_active": 0.30,
                "strong20_rec": 0.45,
                "strong50_rec": 0.20,
            },
        ]
    }
    ulto_dir = tmp_path / "data" / "qwen4b" / "ulto85"
    ulto_dir.mkdir(parents=True, exist_ok=True)
    (ulto_dir / "e3_scores.json").write_text(json.dumps(e3_scores), encoding="utf-8")

    catalog = build_active_poles_catalog(
        e3_neurons=e3_neurons,
        e3_explanations=e3_explanations,
        e3_scores=e3_scores,
        seed=42,
        per_layer_full_expl=1,
        verify_expected_counts=False,
    )

    docs_with_ids = [
        (60_000 + i, " ".join(f"t{int(x)}" for x in rng.integers(1, 240, size=128)))
        for i in range(260)
    ]
    wspec = WindowSpec(before=16, after=4, marker_open="<<", marker_close=">>")
    run_extraction_4b(
        backbone=backbone,
        tokenizer=tok,
        catalog=catalog,
        docs_with_ids=docs_with_ids,
        base_dir=tmp_path,
        num_macro_chunks=2,
        seqs_per_macro_chunk=120,
        seq_len=128,
        micro_batch_size=30,
        max_seqs_per_doc=1,
        split_idx=2,
        compute_device="cpu",
        min_doc_id=60_000,
        run_gate=False,
        window_spec=wspec,
    )
    build_snippets_4b(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=42,
        pairs_p999=8,
        pairs_floor6=4,
        pairs_p97_99=6,
        min_allowed_doc_id=60_000,
        window_spec=wspec,
    )
    run_sparsity_and_top100(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=42,
        top_pool_size=60,
        pairs_per_pole=8,
        floor_n=4,
        min_allowed_doc_id=60_000,
        window_spec=wspec,
    )

    # Mock metrics_4b.json so failure buckets for top100>=0.80 are exercised
    mock_metrics = {
        "per_neuron": {
            str(k): {
                "det_top100_terra_mean": 0.90,
                "det_p999_terra_mean": 0.85,
            }
            for k in range(4)
        }
    }
    (tmp_path / "results" / "qwen4b" / "metrics_4b.json").write_text(
        json.dumps(mock_metrics), encoding="utf-8"
    )

    # 1. Gap decomposition
    gap_doc = compute_gap_decomposition_4b(base_dir=tmp_path, seed=42, n_bootstrap=50)
    assert gap_doc["n_neurons"] == 4
    assert gap_doc["n_active_poles"] == 6
    for k_str, n_rec in gap_doc["per_neuron"].items():
        assert sum(n_rec["energy_l2"]["bands_4"].values()) == pytest.approx(1.0, abs=1e-9)
        assert sum(n_rec["energy_l2"]["bands_5"].values()) == pytest.approx(1.0, abs=1e-9)
        assert sum(n_rec["energy_l2"]["bands_6"].values()) == pytest.approx(1.0, abs=1e-9)
        assert sum(n_rec["energy_l1"]["bands_4"].values()) == pytest.approx(1.0, abs=1e-9)
        assert n_rec["energy_gate_open"] + n_rec["energy_gate_closed"] == pytest.approx(1.0, abs=1e-9)
        assert n_rec["energy_ge_20pct_max"] + n_rec["energy_lt_20pct_max"] == pytest.approx(1.0, abs=1e-9)

    fb = gap_doc["failure_buckets_top100_ge_0_80"]
    assert fb["n_total"] == 4
    assert sum(fb["buckets"][b]["count"] for b in FAILURE_BUCKETS) == 4
    assert fb["buckets"]["pass_causal"]["count"] == 1
    assert fb["buckets"]["overbroad_fp_damage"]["count"] == 1
    assert fb["buckets"]["weak_mass_dominated"]["count"] == 1
    assert fb["buckets"]["active_sim_failure"]["count"] == 1

    # 2. Multiband & FPR snippets
    mb_checks = build_multiband_and_fpr_snippets_4b(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=42,
        pairs_per_band=6,
        fpr_ord_core=4,
        fpr_ord_extra=4,
        fpr_top_core=2,
        fpr_top_extra=2,
        min_allowed_doc_id=60_000,
        window_spec=wspec,
    )
    assert mb_checks["all_hard_invariants_passed"] is True
    assert mb_checks["multiband"]["imbalanced_cells"] == 0
    assert mb_checks["multiband"]["bos_pos0_leak_count"] == 0
    assert mb_checks["multiband"]["same_doc_leak_count"] == 0
    assert mb_checks["multiband"]["dist_window_ge_p90_count"] == 0
    assert mb_checks["fpr"]["total_probes"] == 6 * (4 + 4 + 2 + 2)
    assert mb_checks["fpr"]["within_pole_dup_doc_count"] == 0

    # 3. Exact band KL ablation validation on CPU
    eval_ids = torch.randint(3, 240, (8, 32), dtype=torch.long)
    sel = select_stratified_kl_neurons(gap_doc, layers=(0, 3), per_layer=2, must_include_ks=[0, 2])
    assert len(sel) == 4
    kl_doc = run_band_kl_validation(
        backbone=backbone,
        input_ids_cpu=eval_ids,
        gap_doc=gap_doc,
        selected_neurons=sel,
        compute_device="cpu",
        micro_batch_size=4,
        kl_token_chunk=32,
        neurons_per_group=2,
        n_bootstrap=50,
        seed=42,
        output_path=tmp_path / "results" / "qwen4b" / "band_kl_4b.json",
    )
    assert kl_doc["n_neurons"] == 4
    assert kl_doc["zero_ablation_sanity_kl"] == pytest.approx(0.0, abs=1e-7)
    for k_str, n_kl in kl_doc["per_neuron"].items():
        assert n_kl["kl_by_cond"]["full"] > 0.0
        for cond in ABLATION_CONDITIONS:
            assert n_kl["kl_by_cond"][cond] >= 0.0
        assert sum(n_kl["kl_share_4band_norm"][b] for b in FOUR_BAND_CONDS) == pytest.approx(1.0, abs=1e-9)
