"""CPU unit tests for the Qwen/Qwen3-4B Ulto85 extraction & snippet pipeline.

Tests (all fast, deterministic, zero GPU / zero network required):
1. Pole classification (`is_negligible_pole_text` & `build_active_poles_catalog`):
   - Verifies exact prefix filtering (`negligible`, `none`, `weak`, `no `, `minimal`, `virtually`),
   - Verifies stratified 5-per-layer sampling determinism (seed=42),
   - If real `data/qwen4b/ulto85/*.json` exist, verifies exact `278` neurons, `132` unipolar,
     `146` bipolar, `424` active poles, `0` both-negligible, `50` stratified full_expl neurons,
     and exact named neurons (`info`, `PCR`, `insurance`).
2. 2-half forward pass & parity gate helper (`run_two_half_forward`, `run_parity_gate`):
   - Builds a synthetic 6-layer SwiGLU transformer (`bf16` + `lm_head` deleted),
   - Verifies 2-half split matches full unsplit forward pass bit-for-bit (`max_abs_err == 0.0`,
     `cos == 1.0`), exact hooked `silu(gate)*up == down_proj_in` (`< 1e-4`), 0 quadrant sign
     mismatches, and proper `ParityGateError` raising on broken tolerances.
3. Exact nested A/B slot balancing (`nested_balanced_p999_slots`, `balanced_even_slots`):
   - Verifies `10/10` full, `3/3` `in_floor_6` (`0..5`), `5/5` first half (`0..9`),
     `5/5` second half (`10..19`) for `p99.9_plus`, AND `5/5` for `p97_99`.
4. End-to-end synthetic extraction + snippet cutting + structural verification:
   - Verifies `<bos>` exclusion, `0` same-doc leaks, strict `doc_id >= 60_000` FineWeb
     disjointness enforcement, `p99.9_plus` & `p97_99` exact bounds, `dead_zone` (`|a| < p25_abs`
     AND whole-window `< p90`), 16-token suffix dedup, dominant pole selection by `|p99.9|`,
     and `all_hard_invariants_passed == True`.
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

from qwen4b_eval import ULTO85_DIR
from qwen4b_eval.extract_4b import (
    ParityGateError,
    collect_fixed_length_sequences,
    get_backbone_module,
    run_extraction_4b,
    run_full_cpu_reference,
    run_parity_gate,
    run_two_half_forward,
)
from qwen4b_eval.fetch_ulto85 import (
    build_active_poles_catalog,
    is_negligible_pole_text,
)
from qwen4b_eval.snippets_4b import (
    balanced_even_slots,
    build_snippets_4b,
    nested_balanced_p999_slots,
)
from qwen4b_eval.top100_4b import (
    run_sparsity_and_top100,
    select_top_k_distinct_doc_indices,
)
from w_audit.config import WindowSpec


# ---------------------------------------------------------------------------
# Synthetic SwiGLU Transformer & Whitespace Tokenizer (CPU-only)
# ---------------------------------------------------------------------------


class _SyntheticSwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _SyntheticBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = _SyntheticSwiGLUMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Causal rolling-mean mixing so activations depend on context causally
        h = self.ln1(x)
        cumsum = torch.cumsum(h, dim=1)
        denom = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        x = x + 0.25 * self.attn_proj(cumsum / denom)
        x = x + self.mlp(self.ln2(x))
        return x


class _SyntheticBackbone(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 24, d_mlp: int = 16, n_layers: int = 6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_SyntheticBlock(d_model, d_mlp) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class _SyntheticCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 24, d_mlp: int = 16, n_layers: int = 6):
        super().__init__()
        self.model = _SyntheticBackbone(vocab_size, d_model, d_mlp, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


class _SimpleTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    vocab_size = 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for tok in text.split():
            if tok.startswith("t") and tok[1:].isdigit():
                out.append(3 + (int(tok[1:]) % 250))
            else:
                out.append(3 + (sum(ord(c) for c in tok) % 250))
        return out

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(f"t{int(i)}" for i in ids)


# ---------------------------------------------------------------------------
# 1. Pole Classification & Ulto85 Catalog Tests
# ---------------------------------------------------------------------------


def test_is_negligible_pole_text() -> None:
    for neg_case in [
        None,
        "",
        "   ",
        "Negligible. Values near zero.",
        "None observed.",
        "Weak only, about +1 to +2.",
        "No meaningful negative pole.",
        "Minimal scattered activations.",
        "Virtually zero across corpus.",
    ]:
        assert is_negligible_pole_text(neg_case) is True, f"Should be negligible: {neg_case!r}"

    for pos_case in [
        "The token 'info' as a standalone word.",
        "Names of health-insurance programs.",
        "Normal punctuation tokens (note 'no' without trailing space).",
        "Nothing-else? Wait: 'nothing' does not start with 'no '.",
    ]:
        assert is_negligible_pole_text(pos_case) is False, f"Should be active: {pos_case!r}"


def test_real_ulto85_catalog_if_present() -> None:
    req = [ULTO85_DIR / fn for fn in ("e3_neurons.json", "e3_explanations.json", "e3_scores.json", "e3_scores_opus.json")]
    if not all(p.is_file() for p in req):
        pytest.skip("Ulto85 JSON files not downloaded yet")

    cat = build_active_poles_catalog(
        e3_neurons=json.loads(req[0].read_text(encoding="utf-8")),
        e3_explanations=json.loads(req[1].read_text(encoding="utf-8")),
        e3_scores=json.loads(req[2].read_text(encoding="utf-8")),
        e3_scores_opus=json.loads(req[3].read_text(encoding="utf-8")),
        seed=42,
        per_layer_full_expl=5,
        verify_expected_counts=True,
    )
    assert cat["n_scored_neurons"] == 278
    assert cat["n_unipolar_neurons"] == 132
    assert cat["n_bipolar_neurons"] == 146
    assert cat["n_both_negligible_neurons"] == 0
    assert cat["n_active_poles"] == 424
    assert cat["n_full_expl_50_neurons"] == 50
    assert all(v == 5 for v in cat["full_expl_50_by_layer"].values())
    assert cat["named_neurons"]["info"]["k"] == 26
    assert (cat["named_neurons"]["info"]["layer"], cat["named_neurons"]["info"]["neuron"]) == (0, 9209)
    assert cat["named_neurons"]["PCR"]["k"] == 23
    assert (cat["named_neurons"]["PCR"]["layer"], cat["named_neurons"]["PCR"]["neuron"]) == (0, 8146)
    assert cat["named_neurons"]["insurance"]["k"] == 236
    assert (cat["named_neurons"]["insurance"]["layer"], cat["named_neurons"]["insurance"]["neuron"]) == (32, 4362)


# ---------------------------------------------------------------------------
# 2. Nested A/B Slot Balance Tests
# ---------------------------------------------------------------------------


def test_nested_balanced_p999_and_p97_99_slots() -> None:
    for pole_id in ("L00_N9209_pos", "L00_N8146_pos", "L32_N4362_pos", "L32_N4362_neg"):
        slots_20 = nested_balanced_p999_slots(20, seed=42, pole_id=pole_id, floor_n=6)
        assert len(slots_20) == 20
        assert set(slots_20) == {"A", "B"}
        # Exact 10/10 overall
        assert slots_20.count("A") == 10 and slots_20.count("B") == 10
        # Exact 3/3 in floor-6 (0..5)
        assert slots_20[:6].count("A") == 3 and slots_20[:6].count("B") == 3
        # Exact 5/5 in first half (0..9)
        assert slots_20[:10].count("A") == 5 and slots_20[:10].count("B") == 5
        # Exact 5/5 in second half (10..19)
        assert slots_20[10:20].count("A") == 5 and slots_20[10:20].count("B") == 5

        # Deterministic across repeated calls
        assert slots_20 == nested_balanced_p999_slots(20, seed=42, pole_id=pole_id, floor_n=6)

        slots_10 = balanced_even_slots(10, seed=42, pole_id=pole_id, band="p97_99")
        assert len(slots_10) == 10
        assert slots_10.count("A") == 5 and slots_10.count("B") == 5


# ---------------------------------------------------------------------------
# 3. 2-Half Parity Gate & lm_head Deletion Tests
# ---------------------------------------------------------------------------


def test_two_half_parity_gate_cpu_bf16(tmp_path: Path) -> None:
    torch.manual_seed(42)
    lm = _SyntheticCausalLM(vocab_size=128, d_model=24, d_mlp=16, n_layers=6).to(torch.bfloat16)
    assert hasattr(lm, "lm_head")
    backbone = get_backbone_module(lm)
    assert not hasattr(lm, "lm_head")

    batch = torch.randint(3, 120, (2, 32), dtype=torch.long)
    hook_layers = [0, 2, 3, 5]
    neurons_by_layer = {0: [1, 5], 2: [0, 7], 3: [2, 9], 5: [3, 12]}

    ref = run_full_cpu_reference(backbone, batch, hook_layers, neurons_by_layer=neurons_by_layer)
    res = run_two_half_forward(
        backbone=backbone,
        input_ids_cpu=batch,
        hook_layers=hook_layers,
        neurons_by_layer=neurons_by_layer,
        split_idx=3,
        compute_device="cpu",
        micro_batch_size=1,
        capture_gate_up=True,
    )
    for L in hook_layers:
        diff = (ref[L].float() - res["acts"][L].float()).abs().max().item()
        assert diff == 0.0
        assert res["hook_identity_max_err"][L] < 1e-4
        assert res["quad_sign_mismatches"][L] == 0

    parity_out = tmp_path / "parity_check.json"
    report = run_parity_gate(
        backbone=backbone,
        parity_input_ids=batch,
        hook_layers=hook_layers,
        neurons_by_layer=neurons_by_layer,
        split_idx=3,
        compute_device="cpu",
        output_path=parity_out,
    )
    assert report["passed"] is True
    assert report["max_cpu_two_half_err"] == 0.0
    assert report["worst_all_neurons_cosine"] > 0.99999
    assert report["total_quad_sign_mismatches"] == 0
    assert parity_out.is_file()

    # Verify ParityGateError triggers when threshold is impossible
    with pytest.raises(ParityGateError):
        run_parity_gate(
            backbone=backbone,
            parity_input_ids=batch,
            hook_layers=hook_layers,
            neurons_by_layer=neurons_by_layer,
            split_idx=3,
            compute_device="cpu",
            min_cosine=1.01,
        )


# ---------------------------------------------------------------------------
# 4. Disjoint Doc Guard + End-to-End Synthetic Extraction & Snippet Test
# ---------------------------------------------------------------------------


def test_disjoint_doc_id_guard_rejects_ulto85_overlap() -> None:
    tok = _SimpleTokenizer()
    bad_docs = [(59_999, " ".join(f"t{i}" for i in range(140)))]
    with pytest.raises(AssertionError, match="Disjointness violation"):
        collect_fixed_length_sequences(
            docs_with_ids=bad_docs,
            tokenizer=tok,
            n_sequences=1,
            seq_len=128,
            min_doc_id=60_000,
        )


def test_end_to_end_synthetic_extraction_and_snippets(tmp_path: Path) -> None:
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    backbone = _SyntheticBackbone(vocab_size=256, d_model=16, d_mlp=12, n_layers=4).to(torch.float32)
    tok = _SimpleTokenizer()

    # Build a synthetic catalog with 2 layers x 2 neurons (1 unipolar + 1 bipolar per layer = 6 active poles)
    e3_neurons = [[0, 1], [0, 4], [3, 2], [3, 7]]
    e3_explanations = {
        "0": {
            "layer": 0,
            "neuron": 1,
            "expl": {
                "label": "Unipolar pos neuron 0",
                "positive_pole": "Active positive pole 0",
                "negative_pole": "Negligible.",
                "rubric": "Rubric 0",
            },
        },
        "1": {
            "layer": 0,
            "neuron": 4,
            "expl": {
                "label": "Bipolar neuron 1",
                "positive_pole": "Active positive pole 1",
                "negative_pole": "Active negative pole 1",
                "rubric": "Rubric 1",
            },
        },
        "2": {
            "layer": 3,
            "neuron": 2,
            "expl": {
                "label": "Unipolar neg neuron 2",
                "positive_pole": "Weak only.",
                "negative_pole": "Active negative pole 2",
                "rubric": "Rubric 2",
            },
        },
        "3": {
            "layer": 3,
            "neuron": 7,
            "expl": {
                "label": "Bipolar neuron 3",
                "positive_pole": "Active positive pole 3",
                "negative_pole": "Active negative pole 3",
                "rubric": "Rubric 3",
            },
        },
    }
    e3_scores = {
        "neurons": [
            {"k": 0, "layer": 0, "neuron": 1, "sim_cal_rec": 0.5, "sim_cal_rec_active": 0.8},
            {"k": 1, "layer": 0, "neuron": 4, "sim_cal_rec": -0.2, "sim_cal_rec_active": 0.6},
            {"k": 2, "layer": 3, "neuron": 2, "sim_cal_rec": 0.1, "sim_cal_rec_active": 0.2},
            {"k": 3, "layer": 3, "neuron": 7, "sim_cal_rec": 0.7, "sim_cal_rec_active": 0.9},
        ]
    }
    catalog = build_active_poles_catalog(
        e3_neurons=e3_neurons,
        e3_explanations=e3_explanations,
        e3_scores=e3_scores,
        seed=42,
        per_layer_full_expl=1,
        verify_expected_counts=False,
    )
    assert catalog["n_scored_neurons"] == 4
    assert catalog["n_active_poles"] == 6
    assert catalog["n_full_expl_50_neurons"] == 2

    # Generate 240 distinct documents strictly with doc_id >= 60_000 (each 128 tokens)
    docs_with_ids = []
    for i in range(260):
        toks = rng.integers(1, 240, size=128)
        text = " ".join(f"t{int(x)}" for x in toks)
        docs_with_ids.append((60_000 + i, text))

    # Small window spec (before=16, after=4 -> total=21) allows plenty of interior tokens on 240 seqs
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
        run_gate=True,
        window_spec=wspec,
    )

    # Verify active_poles.json got updated with thresholds and dominant poles
    updated_cat = json.loads((tmp_path / "data" / "qwen4b" / "active_poles.json").read_text(encoding="utf-8"))
    assert updated_cat["extraction_complete"] is True
    assert updated_cat["corpus"]["min_observed_doc_id"] >= 60_000
    for n_rec in updated_cat["neurons"]:
        assert n_rec["dominant_pole_id"] in n_rec["active_pole_ids"]
        assert n_rec["p99_9_pos"] is not None and n_rec["p99_9_neg"] is not None

    # Build snippets with 20 p99.9_plus (floor_6=6) and 10 p97_99 per pole
    checks = build_snippets_4b(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=42,
        pairs_p999=20,
        pairs_floor6=6,
        pairs_p97_99=10,
        min_allowed_doc_id=60_000,
        window_spec=wspec,
    )
    assert checks["all_hard_invariants_passed"] is True
    assert checks["total_pairs"] == 6 * (20 + 10)
    assert checks["bos_pos0_leak_count"] == 0
    assert checks["same_doc_leak_count"] == 0
    assert checks["ulto85_doc_overlap_count"] == 0
    assert checks["min_observed_doc_id"] >= 60_000
    assert checks["dist_abs_ge_p25_count"] == 0
    assert checks["dist_window_ge_p90_count"] == 0
    assert checks["nested_ab_balance"]["imbalanced_p99_9_full_poles"] == 0
    assert checks["nested_ab_balance"]["imbalanced_floor_6_poles"] == 0
    assert checks["nested_ab_balance"]["imbalanced_half_0_poles"] == 0
    assert checks["nested_ab_balance"]["imbalanced_half_1_poles"] == 0
    assert checks["nested_ab_balance"]["imbalanced_p97_99_poles"] == 0

    # Also test Item (1) pole_sparsity_diagnostics.json & Item (2) snippets_4b_top100.jsonl
    p999_bytes_before = (tmp_path / "data" / "qwen4b" / "snippets_4b.jsonl").read_bytes()
    top_res = run_sparsity_and_top100(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=42,
        top_pool_size=60,
        pairs_per_pole=20,
        floor_n=6,
        min_allowed_doc_id=60_000,
        window_spec=wspec,
    )
    # Existing snippets_4b.jsonl must be 100% untouched
    assert (tmp_path / "data" / "qwen4b" / "snippets_4b.jsonl").read_bytes() == p999_bytes_before

    top_checks = top_res["checks"]
    assert top_checks["all_hard_invariants_passed"] is True
    assert top_checks["total_pairs"] == 6 * 20
    assert top_checks["full_poles"] == 6
    assert top_checks["bos_pos0_leak_count"] == 0
    assert top_checks["same_doc_leak_count"] == 0
    assert top_checks["pos_same_doc_within_pole_count"] == 0
    assert top_checks["dist_same_doc_within_pole_count"] == 0
    assert top_checks["ulto85_doc_overlap_count"] == 0
    assert top_checks["dist_abs_ge_p25_count"] == 0
    assert top_checks["dist_window_ge_p90_count"] == 0
    assert top_checks["suffix_dup_within_pole_pos_count"] == 0
    assert top_checks["nested_ab_balance"]["imbalanced_full_poles"] == 0
    assert top_checks["nested_ab_balance"]["imbalanced_half_0_poles"] == 0
    assert top_checks["nested_ab_balance"]["imbalanced_half_1_poles"] == 0
    assert top_checks["nested_ab_balance"]["imbalanced_floor_6_poles"] == 0

    sp_doc = top_res["sparsity"]
    assert sp_doc["n_active_poles"] == 6
    assert len(sp_doc["poles"]) == 6
    assert all(p["top100_pool"]["top100_pool_size"] == 60 for p in sp_doc["poles"])
    assert all(p["p99_9_plus_sampled"]["n_pairs"] == 20 for p in sp_doc["poles"])
    assert all(p["top100_sampled"]["n_pairs"] == 20 for p in sp_doc["poles"])


def test_select_top_k_distinct_doc_indices_single_per_doc_and_suffix_dedup() -> None:
    # 5 docs of length 20 each (total 100 tokens); suffix_len=4
    n_docs = 5
    seq_len = 20
    token_ids = np.arange(100, 100 + n_docs * seq_len, dtype=np.int32)
    doc_ids = np.repeat(np.arange(60_000, 60_000 + n_docs, dtype=np.int32), seq_len)
    eligible = np.zeros(n_docs * seq_len, dtype=bool)
    d_n = np.zeros(n_docs * seq_len, dtype=np.float32)

    # Doc 0 has TWO strong tokens (10.0 at idx 10, 9.5 at idx 15) -> only 10.0 may be chosen
    eligible[10] = True
    d_n[10] = 10.0
    eligible[15] = True
    d_n[15] = 9.5

    # Doc 1 has strong token 9.0 at idx 30 with IDENTICAL 4-token suffix to Doc 0 idx 10 -> skipped!
    token_ids[27:31] = token_ids[7:11]
    eligible[30] = True
    d_n[30] = 9.0

    # Docs 2, 3, 4 have unique suffixes with scores 8.0, 7.0, 6.0
    eligible[50] = True
    d_n[50] = 8.0
    eligible[70] = True
    d_n[70] = 7.0
    eligible[90] = True
    d_n[90] = 6.0

    res = select_top_k_distinct_doc_indices(
        d_n=d_n,
        eligible_mask=eligible,
        doc_ids=doc_ids,
        token_ids=token_ids,
        top_k=3,
        suffix_len=4,
    )
    # Must pick Doc 0 (idx 10, rank 1), skip Doc 1 (rank 2, suffix dup), pick Doc 2 (idx 50, rank 3) & Doc 3 (idx 70, rank 4)
    assert res["indices"] == [10, 50, 70]
    assert res["doc_ranks"] == [1, 3, 4]
    assert res["suffix_dups_skipped"] == 1
    assert res["max_doc_rank_scanned"] == 4
    assert res["max_act_dir"] == pytest.approx(10.0)
    assert res["min_act_dir"] == pytest.approx(7.0)

