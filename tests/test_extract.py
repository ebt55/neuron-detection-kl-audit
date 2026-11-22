"""Comprehensive CPU-only unit tests for w_audit extraction, snippets, and rule baselines.

Uses a lightweight synthetic SwiGLU Qwen3-style transformer on CPU (ZERO GPU memory allocated).
Verifies windowing, slot-balance, and distractor invariants:
  - Hook parity against manual forward pass (<1e-4 across 10 neurons x 10k tokens)
  - Exact Goodfire SwiGLU Q1/Q2/Q3/Q4 quadrant separation (Q3 double-negative)
  - Same-layer w_gate cosine neighbour selection & deterministic tie-breaking
  - Position-0 / <bos> masking and strict interior window eligibility (60 <= pos <= L - 9)
  - Exact SciPy non-target other_max and full window_max alignment over [i - 60, i + 8]
  - Weak-band guard: weak-band positives (`p75_90`, `p90_97`) have zero non-target tokens >= p90,
    all distractors have whole window < p90, and `window_has_ge_p97` rule baseline logged
  - Slot balance: exact k/k 50/50 A/B slot balance in 100% of surviving (pole, band, distractor) cells
  - Same-layer neighbour arm: `nearest_wgate_tier0` keeps strictly Tier 0 ONLY (0 fallback tiers) + cosine stats
  - Disjoint doc_id % 2 train/eval splits + 16-token suffix dedup + reused positives
  - Corpus rarity rule, top-token rule, window_has_ge_p97 rule, and symmetric-window structural checks
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from w_audit.baselines import (
    run_baselines,
    run_structural_checks,
    score_rarity_pair,
    score_top_token_pair,
    score_window_ge_p97_pair,
)
from w_audit.config import (
    BANDS,
    SEED,
    WindowSpec,
    classify_quadrant,
    format_token_only_label,
    make_pair_id,
    make_pole_id,
)
from w_audit.extract import (
    compute_layer_neighbors,
    encode_quadrant_torch,
    run_extraction,
    verify_hooks,
)
from w_audit.snippets import (
    NEIGHBOUR_DISTRACTOR_TYPE,
    WEAK_BANDS,
    balanced_cell_pos_slots,
    build_snippets,
    compute_other_and_window_max_1d,
    compute_window_max_1d,
    render_marked_window,
)


# ---------------------------------------------------------------------------
# Lightweight Synthetic SwiGLU Model & Tokenizer (100% CPU)
# ---------------------------------------------------------------------------


class SyntheticSwiGLUMLP(nn.Module):
    """Exact bias-free SwiGLU MLP matching Qwen3MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SyntheticDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.mlp = SyntheticSwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = torch.roll(x, shifts=1, dims=1)
        shifted[:, 0, :] = 0.0
        h = self.norm(0.8 * x + 0.2 * shifted)
        return x + self.mlp(h)


class SyntheticQwenBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [SyntheticDecoderLayer(hidden_size, intermediate_size) for _ in range(n_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class SyntheticQwenForCausalLM(nn.Module):
    """Mimics Qwen3ForCausalLM structure (model.layers[L].mlp.{gate,up,down}_proj)."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 24,
        intermediate_size: int = 32,
        n_layers: int = 4,
        seed: int = 123,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.model = SyntheticQwenBackbone(vocab_size, hidden_size, intermediate_size, n_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.4, generator=g)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, attention_mask=attention_mask, use_cache=use_cache))


class WhitespaceIntTokenizer:
    """Fast deterministic integer-word tokenizer for CPU unit tests."""

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self._id_to_str = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
        }
        for i in range(3, vocab_size):
            self._id_to_str[i] = f" tok{i}"
        self._str_to_id = {v.strip(): k for k, v in self._id_to_str.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._str_to_id[w] for w in text.split() if w in self._str_to_id]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        parts = []
        for tid in token_ids:
            if skip_special_tokens and tid in (0, 1, 2):
                continue
            parts.append(self._id_to_str.get(int(tid), f" tok{tid}"))
        return "".join(parts)


def _make_topic_clustered_documents(
    n_docs: int = 360,
    doc_len: int = 200,
    vocab_size: int = 256,
    n_topics: int = 12,
    seed: int = 99,
    inject_bos_spike: bool = False,
    inject_shared_suffix: bool = False,
) -> list[str]:
    """Generate realistic topic-clustered documents so whole-window <p90 Tier-0 windows exist."""
    rng = np.random.default_rng(seed)
    tok_pool = np.arange(3, vocab_size)
    topic_chunks = np.array_split(tok_pool, n_topics)

    shared_suffix = [f"tok{200 + i}" for i in range(16)]
    docs: list[str] = []
    for d in range(n_docs):
        # Pick 1 primary topic (~85% of tokens) + occasional topic blend (~15%)
        t_idx = d % n_topics
        primary = topic_chunks[t_idx]
        p_ranks = np.arange(1, len(primary) + 1, dtype=np.float64)
        p_probs = 1.0 / (p_ranks ** 1.1)
        p_probs /= p_probs.sum()

        toks = rng.choice(primary, size=doc_len, p=p_probs)
        # Inject ~8% rare cross-topic tokens so p99.9 spikes happen across distinct docs
        rare_mask = rng.random(doc_len) < 0.08
        toks[rare_mask] = rng.choice(tok_pool, size=int(rare_mask.sum()))
        words = [f"tok{int(t)}" for t in toks]
        if inject_bos_spike:
            words[0] = "<bos>"
        if inject_shared_suffix and d in (0, 1):
            words[75:91] = shared_suffix
        docs.append(" ".join(words))
    return docs


# ---------------------------------------------------------------------------
# 1. Hook Parity Verification Tests
# ---------------------------------------------------------------------------


def test_verify_hooks_matches_manual_forward_10_neurons_10k_tokens() -> None:
    model = SyntheticQwenForCausalLM(vocab_size=256, hidden_size=16, intermediate_size=32, n_layers=4)
    rng = torch.Generator().manual_seed(7)
    batches = [torch.randint(3, 256, (4, 125), generator=rng) for _ in range(20)]

    report = verify_hooks(
        model=model,
        token_batches=batches,
        layers=(0, 1, 2, 3),
        n_neurons_total=10,
        seed=SEED,
        tol=1e-4,
    )
    assert report["passed"] is True
    assert report["tokens_checked"] == 10_000
    assert report["neurons_checked"] == 10
    assert report["max_hook_identity_err"] < 1e-4
    assert report["max_manual_forward_err"] < 1e-4
    assert report["quadrant_sign_mismatches"] == 0


def test_verify_hooks_catches_perturbed_activation() -> None:
    model = SyntheticQwenForCausalLM(vocab_size=128, hidden_size=16, intermediate_size=24, n_layers=2)
    bad_handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
        lambda _m, args: (args[0] + 1.0,)
    )
    try:
        batches = [torch.randint(3, 128, (2, 32))]
        with pytest.raises(AssertionError, match="Hook parity verification FAILED"):
            verify_hooks(
                model=model,
                token_batches=batches,
                layers=(0, 1),
                n_neurons_total=4,
                tol=1e-4,
            )
    finally:
        bad_handle.remove()


# ---------------------------------------------------------------------------
# 2. Goodfire Exact SwiGLU Quadrant Separation Tests (Q1 / Q2 / Q3 / Q4)
# ---------------------------------------------------------------------------


def test_swiglu_quadrant_encoding_matches_goodfire_fig2_3() -> None:
    g = torch.tensor([+2.0, +2.0, -1.2, -1.2])
    u = torch.tensor([+3.0, -3.0, -3.0, +3.0])
    a = F.silu(g) * u

    assert float(a[0]) > 0.0  # Q1 positive
    assert float(a[1]) < 0.0  # Q2 negative (gate open)
    assert float(a[2]) > 0.0  # Q3 double-negative -> POSITIVE activation!
    assert float(a[3]) < 0.0  # Q4 negative (gate closed)

    codes = encode_quadrant_torch(g, u).tolist()
    assert codes == [0, 1, 2, 3]

    for i, exp in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        assert classify_quadrant(float(a[i]), bool(g[i] > 0)) == exp


# ---------------------------------------------------------------------------
# 3. Same-Layer Input-Weight Cosine & Deterministic Tie-Breaking Tests
# ---------------------------------------------------------------------------


def test_compute_layer_neighbors_and_tie_break() -> None:
    mlp = SyntheticSwiGLUMLP(hidden_size=4, intermediate_size=5)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.6, 0.8, 0.0, 0.0],
                ]
            )
        )
        mlp.up_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        mlp.down_proj.weight.copy_(torch.eye(4, 5))

    res = compute_layer_neighbors(mlp, layer=4, candidate_neurons=[0, 4], device="cpu")
    assert res.neighbors[0]["neighbor_neuron"] == 2
    assert abs(res.neighbors[0]["neighbor_gate_cosine"] - 1.0) < 1e-6
    assert abs(res.neighbors[0]["neighbor_concat_cosine"] - 1.0) < 1e-6
    assert res.neighbors[4]["neighbor_neuron"] == 0
    assert abs(res.neighbors[4]["neighbor_gate_cosine"] - 0.6) < 1e-6
    assert set(res.cosine_stats.keys()) == {"min", "p25", "median", "p75", "p90", "max"}


# ---------------------------------------------------------------------------
# 4. Exact Window Max, Non-Target Max, & Balanced Cell Slots
# ---------------------------------------------------------------------------


def test_compute_other_and_window_max_matches_naive_exact() -> None:
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(400).astype(np.float32)
    # Plant a single massive spike at index 150 so target != other_max is rigorously exercised
    arr[150] = 99.0
    wspec = WindowSpec(before=60, after=8)
    other_max, win_max = compute_other_and_window_max_1d(arr, wspec)
    win_max_wrapper = compute_window_max_1d(arr, wspec)
    assert np.allclose(win_max, win_max_wrapper)

    for i in range(wspec.before, len(arr) - wspec.after):
        naive_other = float(max(np.max(arr[i - wspec.before : i]), np.max(arr[i + 1 : i + wspec.after + 1])))
        naive_win = float(np.max(arr[i - wspec.before : i + wspec.after + 1]))
        assert abs(float(other_max[i]) - naive_other) < 1e-6
        assert abs(float(win_max[i]) - naive_win) < 1e-6

    # At the spike itself (i=150), win_max is 99.0 while other_max excludes index 150!
    assert float(win_max[150]) == 99.0
    assert float(other_max[150]) < 10.0


def test_balanced_cell_pos_slots_exact_50_50() -> None:
    for n_pairs in (0, 2, 4, 6):
        slots1 = balanced_cell_pos_slots(n_pairs, SEED, "L04_N0001_pos", "p99.9_plus", "dead_zone")
        slots2 = balanced_cell_pos_slots(n_pairs, SEED, "L04_N0001_pos", "p99.9_plus", "dead_zone")
        assert slots1 == slots2
        assert len(slots1) == n_pairs
        assert slots1.count("A") == n_pairs // 2
        assert slots1.count("B") == n_pairs // 2

    with pytest.raises(ValueError):
        balanced_cell_pos_slots(5, SEED, "L04_N0001_pos", "p99.9_plus", "dead_zone")


def test_render_marked_window_symmetric_and_counts_extra_markers() -> None:
    tok = WhitespaceIntTokenizer(vocab_size=128)
    tok._id_to_str[50] = " a<<b"
    wspec = WindowSpec(before=60, after=8)
    seq = np.arange(10, 10 + 100, dtype=np.int32)
    seq[40] = 50
    marked, target_str, extra = render_marked_window(seq, flat_idx=65, tokenizer=tok, wspec=wspec)
    assert f"<<{target_str}>>" in marked
    assert extra == 1


# ---------------------------------------------------------------------------
# 5. Full End-to-End CPU Pipeline Test (extract -> snippets -> baselines, v3)
# ---------------------------------------------------------------------------


def test_end_to_end_cpu_extraction_snippets_and_baselines(tmp_path: Path) -> None:
    tok = WhitespaceIntTokenizer(vocab_size=256)
    model = SyntheticQwenForCausalLM(
        vocab_size=256, hidden_size=24, intermediate_size=28, n_layers=4, seed=42
    )
    docs = _make_topic_clustered_documents(
        n_docs=400,
        doc_len=200,
        vocab_size=256,
        n_topics=12,
        seed=17,
        inject_bos_spike=True,
        inject_shared_suffix=True,
    )

    layers = (0, 1, 2, 3)
    layer_counts = {0: 2, 1: 2, 2: 2, 3: 2}  # 8 neurons -> 16 poles
    pilot_counts = {0: 1, 1: 1, 2: 1, 3: 1}  # 4 neurons -> 8 poles
    second_counts = {0: 2, 1: 1, 2: 1, 3: 1}  # 5 neurons -> 10 poles
    toy_counts = {0: 2, 1: 2, 2: 1, 3: 1}  # 6 neurons -> 12 poles

    poles_doc = run_extraction(
        model=model,
        tokenizer=tok,
        documents=docs,
        dataset_id="synthetic-test",
        model_id="synthetic-qwen3",
        target_tokens=76_000,
        max_seq_len=200,
        batch_size=8,
        layers=layers,
        layer_neuron_counts=layer_counts,
        pilot_neuron_counts=pilot_counts,
        second_judge_neuron_counts=second_counts,
        toy_neuron_counts=toy_counts,
        reserve_per_layer=6,
        min_train_top=8,
        min_eval_per_band=4,
        seed=SEED,
        base_dir=tmp_path,
        run_hook_check=True,
        hook_check_tokens=1_000,
    )

    assert len(poles_doc["poles"]) == 16

    for p in poles_doc["poles"]:
        if p["in_pilot"]:
            assert p["in_second_judge"]
        if p["in_second_judge"]:
            assert p["in_toy"]
        assert abs(sum(p["quadrant_shares_corpus"].values()) - 1.0) < 1e-5
        for bname in BANDS:
            qb = p["quadrant_shares_by_band"][bname]
            if p["sign"] == 1:
                assert qb["Q2"] == 0.0 and qb["Q4"] == 0.0
                assert abs((qb["Q1"] + qb["Q3"]) - 1.0) < 1e-5
            else:
                assert qb["Q1"] == 0.0 and qb["Q3"] == 0.0
                assert abs((qb["Q2"] + qb["Q4"]) - 1.0) < 1e-5

    snip_summary = build_snippets(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=SEED,
        label_examples_per_pole=8,
        pairs_per_cell=4,
    )
    assert snip_summary["num_poles"] == 16
    assert snip_summary["num_label_examples"] == 16 * 8
    assert snip_summary["num_snippet_pairs"] > 0

    poles_updated = json.loads((tmp_path / "data" / "poles.json").read_text(encoding="utf-8"))
    pole_by_id = {p["pole_id"]: p for p in poles_updated["poles"]}
    for p in poles_updated["poles"]:
        assert 1 <= len(p["top_3_token_ids"]) <= 3
        assert len(p["top_3_tokens"]) == len(p["top_3_token_ids"])
        tok_lbl = format_token_only_label(p["top_3_tokens"])
        assert "tokens such as:" in tok_lbl.lower()
        assert "strongly" not in tok_lbl.lower()

    label_rows = [
        json.loads(line)
        for line in (tmp_path / "data" / "label_examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_docs_used: set[int] = set()
    by_pole_examples: dict[str, list[dict[str, Any]]] = {}
    for row in label_rows:
        assert row["doc_id"] % 2 == 0
        assert row["pos_in_chunk"] >= 60
        assert row["target_token_id"] != tok.bos_token_id
        assert row["act_dir"] > 0.0
        train_docs_used.add(row["doc_id"])
        by_pole_examples.setdefault(row["pole_id"], []).append(row)

    for rows in by_pole_examples.values():
        acts_dir = [r["act_dir"] for r in rows]
        assert acts_dir == sorted(acts_dir, reverse=True)

    pair_rows = [
        json.loads(line)
        for line in (tmp_path / "data" / "snippets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eval_docs_used: set[int] = set()
    cell_slots: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for row in pair_rows:
        pole = pole_by_id[row["pole_id"]]
        pos = row["pos"]
        dist = row["dist"]
        p90_edge = float(pole["band_edges_dir"]["p90"])

        assert row["pair_id"] == make_pair_id(
            row["pole_id"], row["band"], row["distractor_type"], row["pair_idx"]
        )
        assert row["distractor_tier"] == 0
        cell_slots[(row["pole_id"], row["band"], row["distractor_type"])][row["pos_slot"]] += 1

        assert pos["doc_id"] % 2 == 1
        assert dist["doc_id"] % 2 == 1
        assert pos["doc_id"] != dist["doc_id"]
        eval_docs_used.add(pos["doc_id"])
        eval_docs_used.add(dist["doc_id"])

        assert pos["window_tokens"] == 69 and dist["window_tokens"] == 69
        assert pos["target_offset"] == 60 and dist["target_offset"] == 60
        assert pos["pos_in_chunk"] >= 60 and dist["pos_in_chunk"] >= 60
        assert pos["target_token_id"] != tok.bos_token_id
        assert dist["target_token_id"] != tok.bos_token_id

        # Weak-band guard: Weak-band positives have zero non-target tokens >= p90
        if row["band"] in WEAK_BANDS:
            assert pos["other_has_ge_p90"] is False
            assert pos["other_max_act_dir"] < p90_edge + 1e-5
            assert pos["window_has_ge_p97"] is False

        # All distractors have whole window < p90
        assert dist["window_max_act_dir"] < p90_edge + 1e-5
        assert dist["window_has_ge_p97"] is False

        if row["distractor_type"] == "dead_zone":
            assert abs(dist["act_val"]) < pole["p25_abs"] + 1e-6
            assert dist["neighbor_act_val"] is None
        elif row["distractor_type"] == "goodfire_lt_p90":
            assert row["band"] == "p99.9_plus"
            assert dist["neighbor_act_val"] is None
        elif row["distractor_type"] == NEIGHBOUR_DISTRACTOR_TYPE:
            assert row["band"] in ("p99.9_plus", "p90_97")
            assert abs(dist["act_val"]) < pole["p25_abs"] + 1e-6
            assert dist["neighbor_act_val"] is not None
            assert dist["neighbor_gate_cosine"] is not None
        else:
            raise AssertionError(f"Unexpected distractor_type: {row['distractor_type']}")

    # Slot balance: Verify exact 50/50 (k/k) A/B slot balance in EVERY surviving cell
    assert len(cell_slots) > 0
    for ckey, cnts in cell_slots.items():
        assert cnts["A"] == cnts["B"] and cnts["A"] > 0, f"Imbalanced cell {ckey}: {cnts}"

    # Verify 16-token suffix dedup + doc-level disjointness
    token_ids_cache = np.load(tmp_path / "data" / "cache" / "token_ids.npy")
    chunk_starts_cache = np.load(tmp_path / "data" / "cache" / "chunk_starts.npy")
    doc_ids_cache = np.load(tmp_path / "data" / "cache" / "doc_ids.npy")
    pos_in_chunk_cache = np.load(tmp_path / "data" / "cache" / "pos_in_chunk.npy")
    doc_pos_to_flat = {
        (int(doc_ids_cache[i]), int(pos_in_chunk_cache[i])): i
        for i in range(len(token_ids_cache))
    }
    train_suffix_by_pole: dict[str, set[tuple[int, ...]]] = {}
    for row in label_rows:
        fi = int(chunk_starts_cache[row["chunk_id"]]) + int(row["pos_in_chunk"])
        train_suffix_by_pole.setdefault(row["pole_id"], set()).add(
            tuple(token_ids_cache[fi - 15 : fi + 1].tolist())
        )
    for row in pair_rows:
        pos = row["pos"]
        fi = doc_pos_to_flat[(pos["doc_id"], pos["pos_in_chunk"])]
        assert tuple(token_ids_cache[fi - 15 : fi + 1].tolist()) not in train_suffix_by_pole[row["pole_id"]]
    assert train_docs_used.isdisjoint(eval_docs_used)

    # Run baselines + structural sanity checks
    struct = run_baselines(base_dir=tmp_path, strict=True)
    assert struct["all_hard_invariants_passed"] is True
    assert struct["window_length_exact_rate"] == 1.0
    assert struct["target_offset_exact_rate"] == 1.0
    assert struct["other_token_ge_p90_rate"] == 0.0
    assert struct["dist_window_ge_p90_rate"] == 0.0
    assert struct["exact_50_50_cell_balance_rate"] == 1.0
    assert struct["imbalanced_cells_count"] == 0
    assert struct["nearest_wgate_tier0_non_tier0_count"] == 0
    assert struct["marker_position_rule_accuracy"] == 0.5
    assert struct["token_length_rule_accuracy"] == 0.5
    assert struct["pos_slot_balance"]["slot_A_rate"] == 0.5
    assert struct["nearest_wgate_tier0_summary"]["sampled_50_neurons_wgate_cosine_stats"]["n"] == 8

    # Verify window_has_ge_p97 rule is in baselines.jsonl and ties at 0.500 on weak bands!
    baseline_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "baselines.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(baseline_rows) == len(pair_rows) * 4
    assert {r["rule"] for r in baseline_rows} == {"rarity", "top_token", "window_has_ge_p97", "top_10_token"}

    for b in WEAK_BANDS:
        ck = f"dead_zone:{b}"
        if ck in struct["rule_baseline_summary"]:
            assert struct["rule_baseline_summary"][ck]["window_has_ge_p97"] == 0.5
            assert struct["rule_tie_rate_summary"][ck]["window_has_ge_p97"] == 1.0


# ---------------------------------------------------------------------------
# 6. Direct Unit Tests for Rule Baselines & Structural Violation Catching
# ---------------------------------------------------------------------------


def test_rule_baseline_math_exact() -> None:
    diff, tie, acc = score_rarity_pair(pos_count=10, dist_count=1000, total_tokens=100_000)
    assert diff > 0.0 and not tie and acc == 1.0
    diff2, tie2, acc2 = score_rarity_pair(pos_count=500, dist_count=5, total_tokens=100_000)
    assert diff2 < 0.0 and not tie2 and acc2 == 0.0
    diff3, tie3, acc3 = score_rarity_pair(pos_count=42, dist_count=42, total_tokens=100_000)
    assert diff3 == 0.0 and tie3 and acc3 == 0.5

    assert score_top_token_pair(10, 99, [10, 20, 30]) == (1.0, False, 1.0)
    assert score_top_token_pair(99, 20, [10, 20, 30]) == (-1.0, False, 0.0)
    assert score_top_token_pair(10, 20, [10, 20, 30]) == (0.0, True, 0.5)
    assert score_top_token_pair(99, 88, [10, 20, 30]) == (0.0, True, 0.5)

    # window_has_ge_p97 rule
    assert score_window_ge_p97_pair(True, False) == (1.0, False, 1.0)
    assert score_window_ge_p97_pair(False, True) == (-1.0, False, 0.0)
    assert score_window_ge_p97_pair(False, False) == (0.0, True, 0.5)
    assert score_window_ge_p97_pair(True, True) == (0.0, True, 0.5)


def test_structural_checks_catch_window_balance_and_tier0_violations() -> None:
    pid = make_pole_id(4, 1, 1)
    bad_pair = {
        "pair_id": make_pair_id(pid, "p90_97", NEIGHBOUR_DISTRACTOR_TYPE, 0),
        "pole_id": pid,
        "layer": 4,
        "neuron": 1,
        "band": "p90_97",
        "distractor_type": NEIGHBOUR_DISTRACTOR_TYPE,
        "distractor_tier": 1,  # Tier violation: non-Tier-0 fallback!
        "pos_slot": "A",  # Slot balance violation: 1A / 0B imbalanced cell!
        "pos": {
            "doc_id": 1,
            "pos_in_chunk": 0,  # pos-0 / <bos> leak!
            "window_tokens": 55,  # truncated!
            "target_offset": 46,  # shifted offset!
            "other_has_ge_p90": True,  # Weak-band guard violation: weak-band context >= p90!
            "extra_marker_count": 0,
            "marked_text": "short <<tok>> end",
        },
        "dist": {
            "doc_id": 1,  # same-doc leak!
            "pos_in_chunk": 60,
            "window_tokens": 69,
            "target_offset": 60,
            "other_has_ge_p90": True,  # distractor window leak!
            "neighbor_gate_cosine": 0.25,
            "extra_marker_count": 0,
            "marked_text": "normal full length <<tok>> end",
        },
    }
    res = run_structural_checks([bad_pair])
    assert res["all_hard_invariants_passed"] is False
    assert res["truncation_rate_pos"] == 1.0
    assert res["bos_pos0_leak_count"] == 1
    assert res["same_doc_leak_count"] == 1
    assert res["weak_pos_other_token_ge_p90_count"] == 1
    assert res["dist_window_ge_p90_count"] == 1
    assert res["imbalanced_cells_count"] == 1
    assert res["exact_50_50_cell_balance_rate"] == 0.0
    assert res["nearest_wgate_tier0_non_tier0_count"] == 1
    assert res["marker_position_rule_accuracy"] != 0.5
