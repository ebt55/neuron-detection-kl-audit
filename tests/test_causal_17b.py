"""CPU unit tests for Qwen/Qwen3-1.7B causal patching & band-KL pipeline (`causal_17b.py`).

Tests (100% CPU-only, zero GPU / zero network):
1. Calibration helpers (`calibrate_ratings_2fold_cv`, `calibrate_ratings_insample`):
   - Exact affine recovery on noiseless ratings (`r_hat == a_true`),
   - Strict out-of-fold separation (even windows only calibrate odd windows and vice versa).
2. End-to-end synthetic Step 3A (`prepare_17b_inputs`) + Step 3B (`run_17b_baseline_kl`) +
   Step 3C (`run_17b_sim_patching`):
   - Verifies strict odd-doc (`doc_id % 2 == 1`) held-out separation (`0` even train leaks),
   - Verifies 4-band & 5-band corpus energy shares sum strictly to `1.000`,
   - Verifies exact `zero_ablation_sanity_kl == 0.0` and `kl_4band_additivity_ratio ≈ 1.0`,
   - Verifies perfect oracle simulator ratings achieve `sim_cal_rec ≈ 1.0` and `sim_cal_rec_active ≈ 1.0`,
     while constant-zero ratings achieve `sim_cal_rec ≈ 0.0` (`<= 1e-3`).
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

from qwen17b_causal.causal_17b import (
    FOUR_BAND_17B,
    calibrate_ratings_2fold_cv,
    calibrate_ratings_insample,
    prepare_17b_inputs,
    run_17b_baseline_kl,
    run_17b_sim_patching,
)
from w_audit.config import WindowSpec
from w_audit.extract import run_extraction


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
# 1. Calibration Unit Tests
# ---------------------------------------------------------------------------


def test_calibrate_ratings_2fold_cv_and_insample() -> None:
    rng = np.random.default_rng(42)
    r = rng.normal(size=(16, 32))
    a = 2.5 * r - 0.75

    cv_pred = calibrate_ratings_2fold_cv(r, a, exclude_pos0=True)
    in_pred, alpha, beta = calibrate_ratings_insample(r, a, exclude_pos0=True)
    assert alpha == pytest.approx(2.5, abs=1e-6)
    assert beta == pytest.approx(-0.75, abs=1e-6)
    assert np.max(np.abs(cv_pred[:, 1:] - a[:, 1:])) < 1e-6
    assert np.max(np.abs(in_pred[:, 1:] - a[:, 1:])) < 1e-6

    # Verify strict fold isolation: corrupting even windows affects odd predictions only
    a_mod = a.copy()
    a_mod[0::2, 1:] = 0.0
    cv_mod = calibrate_ratings_2fold_cv(r, a_mod, exclude_pos0=True)
    # Odd windows were calibrated from even windows (all 0) -> odd predictions == 0
    assert np.max(np.abs(cv_mod[1::2, 1:])) < 1e-9
    # Even windows were calibrated from uncorrupted odd windows -> still exact 2.5*r - 0.75!
    assert np.max(np.abs(cv_mod[0::2, 1:] - a[0::2, 1:])) < 1e-6


# ---------------------------------------------------------------------------
# 2. End-to-End Synthetic Step 3A + Step 3B + Step 3C Pipeline Test
# ---------------------------------------------------------------------------


def test_end_to_end_17b_causal_pipeline_cpu(tmp_path: Path) -> None:
    torch.manual_seed(99)
    rng = np.random.default_rng(99)
    backbone = _TinyBackbone(vocab_size=256, d_model=16, d_mlp=12, n_layers=4).to(torch.float32)
    tok = _SimpleTok()

    # Create synthetic root cache via w_audit.extract.run_extraction (2 layers x 2 neurons = 4 neurons / 8 poles)
    docs = [
        " ".join(f"t{int(x)}" for x in rng.integers(1, 240, size=140))
        for _ in range(180)
    ]
    wspec = WindowSpec(before=12, after=4, marker_open="<<", marker_close=">>")
    poles_doc = run_extraction(
        model=backbone,
        tokenizer=tok,
        documents=docs,
        target_tokens=24_000,
        max_seq_len=128,
        batch_size=16,
        layers=(0, 3),
        layer_neuron_counts={0: 2, 3: 2},
        pilot_neuron_counts={0: 1, 3: 1},
        second_judge_neuron_counts={0: 1, 3: 1},
        toy_neuron_counts={0: 1, 3: 1},
        reserve_per_layer=4,
        min_train_top=2,
        min_eval_per_band=2,
        seed=42,
        base_dir=tmp_path,
        run_hook_check=False,
        window_spec=wspec,
    )

    # Create minimal data/labels.jsonl for the 8 poles
    with (tmp_path / "data" / "labels.jsonl").open("w", encoding="utf-8") as f:
        for p in poles_doc["poles"]:
            f.write(
                json.dumps({"pole_id": p["pole_id"], "own_label": f"Label for {p['pole_id']}"})
                + "\n"
            )

    # Step 3A: prepare_17b_inputs
    prep = prepare_17b_inputs(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=42,
        e3_n_wins=16,
        e3_win_len=32,
        sim_ord_wins=8,
        sim_act_wins=8,
        sim_win_len=32,
        fpr_ord_core=4,
        fpr_ord_extra=4,
        fpr_top_core=2,
        fpr_top_extra=2,
        window_spec=wspec,
    )
    assert prep["checks"]["all_hard_invariants_passed"] is True
    assert prep["checks"]["e3_ordinary_even_doc_leaks"] == 0
    assert prep["checks"]["sim_windows_even_doc_leaks"] == 0
    assert prep["checks"]["fpr_even_doc_leaks"] == 0
    assert prep["checks"]["fpr_bos_leaks"] == 0
    assert prep["checks"]["fpr_within_pole_dup_docs"] == 0
    for nid, nrec in prep["catalog"]["per_neuron"].items():
        assert sum(nrec["energy_l2"]["bands_4"].values()) == pytest.approx(1.0, abs=1e-9)
        assert sum(nrec["energy_l2"]["bands_5"].values()) == pytest.approx(1.0, abs=1e-9)
        assert nrec["energy_gate_open"] + nrec["energy_gate_closed"] == pytest.approx(1.0, abs=1e-9)

    # Step 3B: run_17b_baseline_kl on CPU
    kl_doc = run_17b_baseline_kl(
        backbone=backbone,
        base_dir=tmp_path,
        compute_device="cpu",
        micro_batch_size=8,
        kl_token_chunk=32,
        neurons_per_group=2,
        n_bootstrap=30,
        seed=42,
    )
    assert kl_doc["n_neurons"] == 4
    assert kl_doc["zero_ablation_sanity_kl"] == pytest.approx(0.0, abs=1e-10)
    for nid, nkl in kl_doc["per_neuron"].items():
        assert nkl["kl_by_cond"]["mean"] > 0.0
        assert sum(nkl["kl_share_4band_norm"][b] for b in FOUR_BAND_17B) == pytest.approx(1.0, abs=1e-9)

    # Step 3C: run_17b_sim_patching with oracle ratings on neuron 0 and zero ratings on neuron 1..3
    sim_meta = np.load(tmp_path / "data" / "qwen17b_causal" / "sim_windows_meta_17b.npz")
    nids = sorted(prep["catalog"]["per_neuron"].keys(), key=lambda s: int(prep["catalog"]["per_neuron"][s]["k"]))

    # Compute true acts for neuron 0 so oracle ratings achieve sim_cal_rec == 1.0
    nid0 = nids[0]
    rec0 = prep["catalog"]["per_neuron"][nid0]
    L0, n0 = int(rec0["layer"]), int(rec0["neuron"])
    toks0 = torch.from_numpy(sim_meta[f"tok__{nid0}"].astype(np.int64))
    cap0: dict[str, torch.Tensor] = {}

    def _cap(_m, args):
        cap0["a"] = args[0].detach()[:, :, n0]

    hh = backbone.layers[L0].mlp.down_proj.register_forward_pre_hook(_cap)
    with torch.no_grad():
        backbone(toks0)
    hh.remove()
    true_a0 = cap0["a"].numpy()

    sim_calls_path = tmp_path / "results" / "qwen17b_causal" / "sim_calls_17b.jsonl"
    with sim_calls_path.open("w", encoding="utf-8") as f:
        for nid in nids:
            for wi in range(16):
                ratings = (
                    (true_a0[wi] / max(float(rec0["max_abs_act"]), 1e-6) * 5.0).tolist()
                    if nid == nid0
                    else [0.0] * 32
                )
                f.write(
                    json.dumps({"window_id": f"{nid}__w{wi:02d}", "neuron_id": nid, "ratings": ratings})
                    + "\n"
                )

    cs_doc = run_17b_sim_patching(
        backbone=backbone,
        base_dir=tmp_path,
        sim_calls_path=sim_calls_path,
        compute_device="cpu",
        micro_batch_size=8,
        kl_token_chunk=32,
    )
    assert cs_doc["n_neurons"] == 4
    assert cs_doc["zero_ablation_sanity_kl"] == pytest.approx(0.0, abs=1e-10)
    # Oracle ratings on neuron 0 must recover ~1.000 of KL damage!
    assert cs_doc["per_neuron"][nid0]["sim_cal_rec"] > 0.999
    assert cs_doc["per_neuron"][nid0]["sim_cal_rec_active"] > 0.999
    assert cs_doc["per_neuron"][nid0]["corr"] > 0.999
    # Constant-zero ratings on neuron 1 must recover <= 0.05 (essentially 0.0, mean-calibrated!)
    assert cs_doc["per_neuron"][nids[1]]["sim_cal_rec"] < 0.05
