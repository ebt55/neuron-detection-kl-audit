"""CPU unit tests for full 278-neuron IEEE-754 float32 band KL pipeline & statistics.

Covers:
  1. Opposite-sign eval energy share for unipolar vs bipolar neurons (incl. p75 < 0 edge case).
  2. Neuron-clustered vs pair-level pooled bootstrap CI math.
  3. Permuted-neuron null (row permutation preserving per-band means & row sums).
  4. Anchor determinism gate (`< 1e-11` pass and `[anchor-mismatch]` error on `>= 1e-9`),
     incremental JSONL checkpoint resume (exact bit-for-bit equivalence to uninterrupted run),
     and graceful wall-clock cutoff pause.
  5. Full schema invariants on `assemble_band_kl_all278_doc`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen4b_eval.band_kl_4b import ABLATION_CONDITIONS, FOUR_BAND_CONDS, run_band_kl_validation
from qwen4b_eval.band_kl_4b_all278 import (
    IST_TZ,
    AnchorMismatchError,
    append_ckpt_jsonl,
    compute_opposite_sign_eval_energy,
    load_ckpt_jsonl,
    permuted_neuron_null,
    pooled_corr_with_clustered_and_pair_ci,
    run_band_kl_all278,
)
from qwen4b_eval.extract_4b import run_extraction_4b
from qwen4b_eval.fetch_ulto85 import build_active_poles_catalog
from qwen4b_eval.gap_and_bands_4b import compute_gap_decomposition_4b
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
# 1. Opposite-Sign Eval Energy Share Unit Tests
# ---------------------------------------------------------------------------


def test_opposite_sign_eval_energy_unipolar_vs_bipolar() -> None:
    acts = np.array([[2.0, 0.5, -1.0, -3.0]], dtype=np.float64)
    mean_act = 0.0
    l2 = acts**2  # [4.0, 0.25, 1.0, 9.0], total = 14.25

    # Unipolar positive neuron where lt_p75 includes [0.5, -1.0, -3.0]
    lt_mask = np.array([[False, True, True, True]], dtype=bool)
    opp_lt, opp_tot, opp_frac = compute_opposite_sign_eval_energy(acts, mean_act, lt_mask, [1])
    assert opp_lt == pytest.approx(10.0 / 14.25, abs=1e-12)
    assert opp_tot == pytest.approx(10.0 / 14.25, abs=1e-12)
    assert opp_frac == pytest.approx(10.0 / 10.25, abs=1e-12)

    # Edge case: p75 < 0 so -1.0 falls outside lt_p75
    lt_mask_neg_p75 = np.array([[False, False, False, True]], dtype=bool)
    opp_lt2, opp_tot2, opp_frac2 = compute_opposite_sign_eval_energy(
        acts, mean_act, lt_mask_neg_p75, [1]
    )
    assert opp_lt2 == pytest.approx(9.0 / 14.25, abs=1e-12)
    assert opp_tot2 == pytest.approx(10.0 / 14.25, abs=1e-12)
    assert opp_frac2 == pytest.approx(1.0, abs=1e-12)

    # Unipolar negative neuron: act_dir = -acts, so act_dir < 0 iff acts > 0
    lt_neg_pole = np.array([[True, True, False, False]], dtype=bool)
    opp_lt_n, opp_tot_n, opp_frac_n = compute_opposite_sign_eval_energy(
        acts, mean_act, lt_neg_pole, [-1]
    )
    assert opp_lt_n == pytest.approx(4.25 / 14.25, abs=1e-12)
    assert opp_tot_n == pytest.approx(4.25 / 14.25, abs=1e-12)
    assert opp_frac_n == pytest.approx(1.0, abs=1e-12)

    # Bipolar neuron: always identically zero
    assert compute_opposite_sign_eval_energy(acts, mean_act, lt_mask, [1, -1]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 2. Neuron-Clustered vs Pair-Level CI & Permuted-Neuron Null Tests
# ---------------------------------------------------------------------------


def test_pooled_clustered_vs_pair_level_ci_math() -> None:
    rng = np.random.default_rng(123)
    # Construct 30 clusters where all 4 bands of a neuron move together around a cluster slope
    # (high intra-cluster correlation => cluster bootstrap CI MUST be wider than pair-level CI)
    n_neurons = 30
    x_base = rng.uniform(0.05, 0.45, size=(n_neurons, 4))
    cluster_err = rng.normal(0.0, 0.12, size=(n_neurons, 1))
    y_mat = 0.8 * x_base + cluster_err + rng.normal(0.0, 0.01, size=(n_neurons, 4))

    res1 = pooled_corr_with_clustered_and_pair_ci(x_base, y_mat, n_resamples=500, seed=42)
    res2 = pooled_corr_with_clustered_and_pair_ci(x_base, y_mat, n_resamples=500, seed=42)

    assert res1["n"] == 120
    assert res1["n_neurons"] == 30
    assert res1["n_bands"] == 4
    assert res1["ci_type"] == "neuron_clustered"
    assert res1["spearman_ci_low"] == res1["neuron_clustered_ci"]["spearman_ci_low"]
    assert res1["spearman_ci_high"] == res1["neuron_clustered_ci"]["spearman_ci_high"]
    assert res1["neuron_clustered_ci"]["ci_type"] == "neuron_clustered"
    assert res1["pair_level_ci"]["ci_type"] == "pair_level"

    # Exact reproducibility
    assert res1["spearman_ci_low"] == res2["spearman_ci_low"]
    assert res1["pair_level_ci"]["pearson_ci_high"] == res2["pair_level_ci"]["pearson_ci_high"]

    # Cluster CI width strictly exceeds pair-level CI width under within-neuron correlation
    w_cluster = res1["neuron_clustered_ci"]["pearson_ci_high"] - res1["neuron_clustered_ci"]["pearson_ci_low"]
    w_pair = res1["pair_level_ci"]["pearson_ci_high"] - res1["pair_level_ci"]["pearson_ci_low"]
    assert w_cluster > w_pair


def test_permuted_neuron_null_preserves_band_means() -> None:
    rng = np.random.default_rng(7)
    # Strong per-band means [0.55, 0.25, 0.15, 0.05] plus neuron-specific signal
    band_means = np.array([0.55, 0.25, 0.15, 0.05])
    delta = rng.normal(0.0, 0.08, size=(60, 4))
    delta -= delta.mean(axis=1, keepdims=True)
    x_mat = np.clip(band_means + delta, 0.001, 0.999)
    x_mat /= x_mat.sum(axis=1, keepdims=True)
    y_mat = 0.85 * x_mat + 0.15 * band_means
    y_mat /= y_mat.sum(axis=1, keepdims=True)

    null_res = permuted_neuron_null(x_mat, y_mat, n_permutations=400, seed=42)
    assert null_res["n_neurons"] == 60
    assert null_res["n_bands"] == 4
    assert null_res["n_pairs"] == 240
    assert null_res["preserves_per_band_means"] is True
    assert null_res["spearman_mean"] > 0.4  # band-mean baseline alone induces positive rho
    assert null_res["observed_spearman_rho"] > null_res["spearman_ci_high"]
    assert null_res["spearman_p_empirical"] == pytest.approx(1.0 / 401.0, abs=1e-9)
    assert null_res["pearson_ci_low"] < null_res["pearson_mean"] < null_res["pearson_ci_high"]


# ---------------------------------------------------------------------------
# 3. Checkpoint Resume, Wall-Clock Cutoff, Anchor Gate & Schema Integration Test
# ---------------------------------------------------------------------------


def test_checkpoint_resume_cutoff_anchors_and_schema_cpu(tmp_path: Path) -> None:
    torch.manual_seed(91)
    rng = np.random.default_rng(91)
    backbone = _TinyBackbone(vocab_size=256, d_model=16, d_mlp=12, n_layers=4).to(torch.float32)
    tok = _SimpleTok()

    e3_neurons = [[0, 1], [0, 3], [0, 5], [3, 2], [3, 4], [3, 7]]
    e3_explanations = {}
    e3_scores_list = []
    for k, (L, n_idx) in enumerate(e3_neurons):
        bipolar = (k % 2 == 1)
        e3_explanations[str(k)] = {
            "layer": L,
            "neuron": n_idx,
            "expl": {
                "label": f"Neuron {k}",
                "positive_pole": f"Active pos {k}",
                "negative_pole": f"Active neg {k}" if bipolar else "Negligible.",
                "rubric": f"R{k}",
            },
        }
        e3_scores_list.append(
            {
                "k": k,
                "layer": L,
                "neuron": n_idx,
                "M": 2.0,
                "sim_cal_rec": -0.2 + 0.2 * k,
                "sim_cal_rec_active": 0.1 + 0.15 * k,
                "strong20_rec": 0.1 + 0.14 * k,
                "strong50_rec": 0.05 + 0.08 * k,
            }
        )
    e3_scores = {"neurons": e3_scores_list}
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
    active_poles_path = tmp_path / "data" / "qwen4b" / "active_poles.json"
    active_poles_path.write_text(json.dumps(catalog), encoding="utf-8")

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
    (tmp_path / "results" / "qwen4b" / "metrics_4b.json").write_text(
        json.dumps(
            {
                "per_neuron": {
                    str(k): {"det_top100_terra_mean": 0.90, "det_p999_terra_mean": 0.85}
                    for k in range(6)
                }
            }
        ),
        encoding="utf-8",
    )

    gap_doc = compute_gap_decomposition_4b(base_dir=tmp_path, seed=42, n_bootstrap=50)
    eval_ids = torch.randint(3, 240, (8, 32), dtype=torch.long)

    # Seed base_kl_doc with 2 neurons (k=0 in stage 0, k=3 in stage 1)
    base_sel = [gap_doc["per_neuron"]["0"], gap_doc["per_neuron"]["3"]]
    base_kl_doc = run_band_kl_validation(
        backbone=backbone,
        input_ids_cpu=eval_ids,
        gap_doc=gap_doc,
        selected_neurons=base_sel,
        compute_device="cpu",
        micro_batch_size=4,
        kl_token_chunk=32,
        neurons_per_group=1,
        n_bootstrap=40,
        seed=42,
    )

    # 1. Anchor mismatch gate test (>= 1e-9 raises AnchorMismatchError with [anchor-mismatch])
    bad_base = json.loads(json.dumps(base_kl_doc))
    bad_base["per_neuron"]["0"]["kl_by_cond"]["full"] += 2e-9
    with pytest.raises(AnchorMismatchError, match=r"\[anchor-mismatch\]"):
        run_band_kl_all278(
            backbone=backbone,
            input_ids_cpu=eval_ids,
            gap_doc=gap_doc,
            active_poles_doc=catalog,
            base_kl_doc=bad_base,
            anchor_ks=(0, 3),
            compute_device="cpu",
            micro_batch_size=4,
            kl_token_chunk=32,
            neurons_per_group=1,
            n_bootstrap=40,
            n_permutations=40,
        )

    # 2. Wall-clock cutoff test: stop gracefully after 1st group (2 neurons with group size 2)
    ckpt_path = tmp_path / "results" / "qwen4b" / "ckpt.jsonl"
    out_path = tmp_path / "results" / "qwen4b" / "all6.json"
    t_start = datetime(2026, 11, 22, 7, 29, 0, tzinfo=IST_TZ)
    cutoff_dt = datetime(2026, 11, 22, 7, 30, 0, tzinfo=IST_TZ)
    calls = {"n": 0}

    def _clock_advances() -> datetime:
        calls["n"] += 1
        # Allow group 1 (first check before group 1), then pass cutoff on 2nd check
        if calls["n"] == 1:
            return t_start
        return cutoff_dt + timedelta(seconds=5)

    partial = run_band_kl_all278(
        backbone=backbone,
        input_ids_cpu=eval_ids,
        gap_doc=gap_doc,
        active_poles_doc=catalog,
        base_kl_doc=base_kl_doc,
        anchor_ks=(0, 3),
        compute_device="cpu",
        micro_batch_size=4,
        kl_token_chunk=32,
        neurons_per_group=2,
        n_bootstrap=40,
        n_permutations=40,
        ckpt_path=ckpt_path,
        output_path=out_path,
        cutoff_dt=cutoff_dt,
        now_fn=_clock_advances,
    )
    assert partial["status"] == "partial_cutoff"
    assert partial["completed_new"] == 2
    assert partial["target_new"] == 4
    assert not out_path.exists()
    loaded_ckpt = load_ckpt_jsonl(ckpt_path)
    assert len(loaded_ckpt) == 2

    # Simulate truncated trailing crash line in ckpt_path; load_ckpt_jsonl must recover cleanly
    with ckpt_path.open("a", encoding="utf-8") as fh:
        fh.write('{"k": 99, "kl_by_cond": {"full": ')
    assert len(load_ckpt_jsonl(ckpt_path)) == 2
    # Rewrite clean ckpt before appending remaining via resume
    ckpt_path.unlink()
    append_ckpt_jsonl(ckpt_path, list(loaded_ckpt.values()))

    # 3. Resume from checkpoint and verify exact parity vs uninterrupted run
    resumed_doc = run_band_kl_all278(
        backbone=backbone,
        input_ids_cpu=eval_ids,
        gap_doc=gap_doc,
        active_poles_doc=catalog,
        base_kl_doc=base_kl_doc,
        anchor_ks=(0, 3),
        compute_device="cpu",
        micro_batch_size=4,
        kl_token_chunk=32,
        neurons_per_group=2,
        n_bootstrap=40,
        n_permutations=40,
        ckpt_path=ckpt_path,
        output_path=out_path,
        cutoff_dt=cutoff_dt + timedelta(hours=5),
        now_fn=lambda: t_start,
    )
    clean_doc = run_band_kl_all278(
        backbone=backbone,
        input_ids_cpu=eval_ids,
        gap_doc=gap_doc,
        active_poles_doc=catalog,
        base_kl_doc=base_kl_doc,
        anchor_ks=(0, 3),
        compute_device="cpu",
        micro_batch_size=4,
        kl_token_chunk=32,
        neurons_per_group=2,
        n_bootstrap=40,
        n_permutations=40,
    )

    assert resumed_doc["n_neurons"] == 6
    assert resumed_doc["n_reused_from_band_kl_4b"] == 2
    assert resumed_doc["n_newly_computed"] == 4
    assert resumed_doc["zero_ablation_sanity_kl"] == pytest.approx(0.0, abs=1e-10)
    assert resumed_doc["anchor_validation"]["validated"] is True
    assert resumed_doc["anchor_validation"]["passed_lt_1e_11"] is True

    # Resumed KL matches clean uninterrupted run bit-for-bit across all 6 neurons x 9 conds
    for k_str in resumed_doc["per_neuron"]:
        r1 = resumed_doc["per_neuron"][k_str]
        r2 = clean_doc["per_neuron"][k_str]
        for cond in ABLATION_CONDITIONS:
            assert r1["kl_by_cond"][cond] == pytest.approx(r2["kl_by_cond"][cond], abs=1e-15)
        assert r1["is_bipolar"] == (r1["n_active_poles"] == 2)
        if r1["is_bipolar"]:
            assert r1["opposite_sign_eval_energy_share"] == 0.0
        else:
            assert 0.0 <= r1["opposite_sign_eval_energy_share"] <= r1["eval_energy_4band"]["lt_p75"] + 1e-15

    # Aggregate schema checks
    uvb = resumed_doc["unipolar_vs_bipolar"]
    assert uvb["unipolar"]["n"] == 3
    assert uvb["bipolar"]["n"] == 3
    assert uvb["bipolar"]["mean_opposite_sign_eval_energy_share"] == 0.0
    assert "pooled_24_eval_energy_vs_kl_norm" in resumed_doc["correlations"]
    pooled = resumed_doc["correlations"]["pooled_24_eval_energy_vs_kl_norm"]
    assert pooled["neuron_clustered_ci"]["ci_type"] == "neuron_clustered"
    assert pooled["pair_level_ci"]["ci_type"] == "pair_level"
    assert "eval_energy_vs_kl_norm" in resumed_doc["permuted_neuron_null"]
    assert "corpus_energy_vs_kl_norm" in resumed_doc["permuted_neuron_null"]
    assert out_path.is_file()
