"""Comprehensive mock-API CPU test suite for the judge/label/analysis pipeline.

Zero live network calls, zero GPU/VRAM usage. Verifies:
- probe_limits.py: live header parsing, logprobs check, 0%-cache cost & ETA math
- labeler.py: resumable own-label cache, token_only formatting, cross-layer wrong-label
  near-duplicate rejection (embedding cosine > 0.85 + word-Jaccard > 0.50 fallback)
- judge.py: strict 1-pair-per-call enforcement, deterministic A/B randomization,
  logprob margin sign & imputation, JSONL resumability + truncated-line crash recovery,
  hard --max-usd spend abort guard, 429 exponential backoff, Ollama explicit
  "think": false JSON payload, and exact call counts for all 6 CLI run modes
- analyze.py & cli.py: exact margin AUROC, pole-clustered bootstrap CIs, Q3 double-negative
  weak-band slicing, Layer-4 pilot gate, first-position bias, second-judge agreement,
  PNG figure creation, summary tables, 30-pole manual audit worksheet, and CLI dispatch
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from w_audit import analyze, cli, judge, labeler, probe_limits
from w_audit.config import (
    BANDS,
    CONDITIONS,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    LAYER_NEURON_COUNTS,
    NEIGHBOUR_DISTRACTOR_TYPE,
    NO_LABEL_DESCRIPTION,
    PILOT_NEURON_COUNTS,
    SECOND_JUDGE_NEURON_COUNTS,
    TOY_NEURON_COUNTS,
    deterministic_pos_slot,
    make_pair_id,
    make_pole_id,
)


# ---------------------------------------------------------------------------
# Mock Helpers
# ---------------------------------------------------------------------------


def _make_top_lp(token: str, logprob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob)


def _make_chat_response(
    content: str,
    top_lps: list[SimpleNamespace] | None = None,
    prompt_tokens: int = 440,
    completion_tokens: int = 1,
) -> SimpleNamespace:
    lp_obj = (
        SimpleNamespace(content=[SimpleNamespace(top_logprobs=top_lps)])
        if top_lps is not None
        else None
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                logprobs=lp_obj,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class _MockRawResponse:
    def __init__(self, parsed: Any, headers: dict[str, str]) -> None:
        self._parsed = parsed
        self.headers = headers

    def parse(self) -> Any:
        return self._parsed


class _FakeRateLimitError(Exception):
    status_code = 429

    def __init__(self, msg: str = "429 Too Many Requests") -> None:
        super().__init__(msg)
        self.response = SimpleNamespace(headers={"retry-after": "0.01"})


def _build_synthetic_poles_and_snippets(
    n_neurons_per_layer: dict[int, int] | None = None,
    pairs_per_cell: int = 6,
    include_p90: bool = True,
    include_neighbour: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build synthetic poles.json, label_examples.jsonl, and snippets.jsonl."""
    if n_neurons_per_layer is None:
        n_neurons_per_layer = {4: 2, 11: 2, 18: 2, 24: 2}

    poles: list[dict[str, Any]] = []
    label_examples: list[dict[str, Any]] = []
    snippets: list[dict[str, Any]] = []

    for layer, count in n_neurons_per_layer.items():
        for idx in range(count):
            neuron = 100 + idx
            in_pilot = idx < min(count, PILOT_NEURON_COUNTS.get(layer, 1))
            in_sec = idx < min(count, SECOND_JUDGE_NEURON_COUNTS.get(layer, 1))
            in_toy = idx < min(count, TOY_NEURON_COUNTS.get(layer, count))
            for sign, pole_name in ((1, "pos"), (-1, "neg")):
                pid = make_pole_id(layer, neuron, sign)
                top3_ids = [1000 + layer * 10 + idx, 2000 + idx, 3000 + idx]
                top3_toks = [f" tok{layer}_{idx}", f" word{idx}", f" sym{sign:+d}"]
                poles.append(
                    {
                        "pole_id": pid,
                        "layer": layer,
                        "neuron": neuron,
                        "sign": sign,
                        "pole_name": pole_name,
                        "is_layer4_control": (layer == 4),
                        "in_pilot": in_pilot,
                        "in_toy": in_toy,
                        "in_second_judge": in_sec,
                        "neighbor_neuron": neuron + 500,
                        "neighbor_gate_cosine": 0.42,
                        "neighbor_up_cosine": 0.11,
                        "neighbor_concat_cosine": 0.27,
                        "neighbor_down_cosine": 0.09,
                        "frac_pos": 0.5,
                        "frac_neg": 0.5,
                        "p25_abs": 0.03,
                        "band_edges_dir": {
                            "p75": 0.05,
                            "p90": 0.15,
                            "p97": 0.45,
                            "p99.9": 1.60,
                        },
                        "sign_clipped_p75_90": False,
                        "quadrant_shares_corpus": {
                            "Q1": 0.15,
                            "Q2": 0.15,
                            "Q3": 0.35,
                            "Q4": 0.35,
                        },
                        "quadrant_shares_by_band": {
                            "p75_90": {"Q1": 0.5, "Q2": 0.0, "Q3": 0.5, "Q4": 0.0}
                            if sign > 0
                            else {"Q1": 0.0, "Q2": 0.5, "Q3": 0.0, "Q4": 0.5},
                            "p90_97": {"Q1": 0.75, "Q2": 0.0, "Q3": 0.25, "Q4": 0.0}
                            if sign > 0
                            else {"Q1": 0.0, "Q2": 0.75, "Q3": 0.0, "Q4": 0.25},
                            "p97_99.9": {"Q1": 0.95, "Q2": 0.0, "Q3": 0.05, "Q4": 0.0}
                            if sign > 0
                            else {"Q1": 0.0, "Q2": 0.95, "Q3": 0.0, "Q4": 0.05},
                            "p99.9_plus": {"Q1": 1.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0}
                            if sign > 0
                            else {"Q1": 0.0, "Q2": 1.0, "Q3": 0.0, "Q4": 0.0},
                        },
                        "top_3_token_ids": top3_ids,
                        "top_3_tokens": top3_toks,
                        "top_10_token_ids": top3_ids,
                    }
                )

                for ex_i in range(20):
                    tok_i = ex_i % 3
                    label_examples.append(
                        {
                            "pole_id": pid,
                            "example_idx": ex_i,
                            "doc_id": ex_i * 2,
                            "chunk_id": ex_i * 2,
                            "pos_in_chunk": 100,
                            "target_token_id": top3_ids[tok_i],
                            "target_token_str": top3_toks[tok_i],
                            "act_val": float(sign * (2.5 - ex_i * 0.05)),
                            "act_dir": float(2.5 - ex_i * 0.05),
                            "gate_open": True,
                            "quadrant": "Q1" if sign > 0 else "Q2",
                            "marked_text": f"prefix context {ex_i} <<{top3_toks[tok_i]}>> suffix tail",
                        }
                    )

                def _append_pairs(band: str, dist_type: str) -> None:
                    for p_i in range(pairs_per_cell):
                        pair_id = make_pair_id(pid, band, dist_type, p_i)
                        pos_slot = deterministic_pos_slot(pair_id)
                        # Put Q3 / Q4 gate-closed tokens in weak bands on half of pairs
                        is_weak = band in ("p75_90", "p90_97")
                        gate_open = not (is_weak and (p_i % 2 == 1))
                        if sign > 0:
                            quad = "Q1" if gate_open else "Q3"
                        else:
                            quad = "Q2" if gate_open else "Q4"
                        snippets.append(
                            {
                                "pair_id": pair_id,
                                "pole_id": pid,
                                "layer": layer,
                                "neuron": neuron,
                                "sign": sign,
                                "band": band,
                                "distractor_type": dist_type,
                                "distractor_tier": 0 if p_i % 3 else 1,
                                "pair_idx": p_i,
                                "pos_slot": pos_slot,
                                "pos": {
                                    "doc_id": 1001 + p_i * 2,
                                    "pos_in_chunk": 120,
                                    "window_tokens": 69,
                                    "target_offset": 60,
                                    "target_token_id": top3_ids[0],
                                    "target_token_str": top3_toks[0],
                                    "target_corpus_count": 200,
                                    "act_val": float(sign * (1.8 if band == "p99.9_plus" else 0.12)),
                                    "act_dir": float(1.8 if band == "p99.9_plus" else 0.12),
                                    "gate_open": gate_open,
                                    "quadrant": quad,
                                    "extra_marker_count": 0,
                                    "marked_text": f"POS_TEXT {pair_id} <<{top3_toks[0]}>> end",
                                },
                                "dist": {
                                    "doc_id": 5001 + p_i * 2,
                                    "pos_in_chunk": 95,
                                    "window_tokens": 69,
                                    "target_offset": 60,
                                    "target_token_id": 9999,
                                    "target_token_str": " the",
                                    "target_corpus_count": 50000,
                                    "act_val": 0.004,
                                    "act_dir": 0.004,
                                    "gate_open": False,
                                    "quadrant": "Q4",
                                    "neighbor_act_val": 2.1 if dist_type == NEIGHBOUR_DISTRACTOR_TYPE else None,
                                    "window_max_act_dir": 0.05,
                                    "extra_marker_count": 0,
                                    "marked_text": f"DIST_TEXT {pair_id} << the>> end",
                                },
                            }
                        )

                for band in BANDS:
                    _append_pairs(band, "dead_zone")
                if include_p90:
                    _append_pairs("p99.9_plus", "goodfire_lt_p90")
                if include_neighbour:
                    _append_pairs("p99.9_plus", NEIGHBOUR_DISTRACTOR_TYPE)
                    _append_pairs("p90_97", NEIGHBOUR_DISTRACTOR_TYPE)

    poles_doc = {
        "schema_version": "1.0",
        "model_id": "Qwen/Qwen3-1.7B",
        "dataset_id": "synthetic",
        "seed": 42,
        "total_tokens": 2_000_000,
        "train_tokens": 1_000_000,
        "eval_tokens": 1_000_000,
        "layers": sorted(n_neurons_per_layer.keys()),
        "window": {"before": 60, "after": 8, "total": 69, "marker_open": "<<", "marker_close": ">>"},
        "layer_neighbor_cosine_stats": {},
        "skipped_neurons": [],
        "poles": poles,
    }
    return poles_doc, label_examples, snippets


# ---------------------------------------------------------------------------
# 1. probe_limits.py Tests
# ---------------------------------------------------------------------------


def test_probe_limits_reads_raw_headers_logprobs_and_zero_cache_math() -> None:
    calls: list[dict[str, Any]] = []

    class _MockSyncRaw:
        def create(self, **kwargs: Any) -> _MockRawResponse:
            calls.append(kwargs)
            model = kwargs["model"]
            headers = {
                "x-ratelimit-limit-requests": "5000" if model == "gpt-4.1-mini" else "500",
                "x-ratelimit-remaining-requests": "4999" if model == "gpt-4.1-mini" else "499",
                "x-ratelimit-reset-requests": "12ms" if model == "gpt-4.1-mini" else "120ms",
                "x-ratelimit-limit-tokens": "2000000" if model == "gpt-4.1-mini" else "200000",
                "x-ratelimit-remaining-tokens": "1999980" if model == "gpt-4.1-mini" else "199980",
                "x-ratelimit-reset-tokens": "0s",
            }
            parsed = _make_chat_response(
                "A",
                top_lps=[_make_top_lp("A", -0.01), _make_top_lp("B", -4.6)],
                prompt_tokens=25,
                completion_tokens=1,
            )
            return _MockRawResponse(parsed, headers)

    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=_MockSyncRaw()))
    )
    report = probe_limits.run_probe(
        models=("gpt-4o-mini", "gpt-4.1-mini"),
        client=mock_client,
        avg_judge_input_tokens=450,
        avg_judge_output_tokens=2,
    )

    assert len(calls) == 2
    assert all(c["logprobs"] is True and c["top_logprobs"] == 5 for c in calls)
    assert report["probes"]["gpt-4o-mini"]["logprobs_supported"] is True
    assert report["probes"]["gpt-4o-mini"]["limit_requests"] == 500
    assert report["probes"]["gpt-4.1-mini"]["limit_requests"] == 5000
    # Tier 1 RPM=500 should flag the 10k RPD warning
    assert report["summary"]["tier1_10k_rpd_warning"] is True
    assert report["summary"]["assuming_prompt_cache_hit_rate"] == 0.0

    by_name = {w["name"]: w for w in report["workloads"]}
    # Exact 0%-cache math: 9,600 * (450 * 0.15 + 2 * 0.60) / 1e6 = 0.65952
    assert by_name["core_grid"]["estimated_cost_usd"] == pytest.approx(0.6595, abs=1e-4)
    assert by_name["neighbour_arm"]["num_calls"] == 4_800
    assert by_name["second_judge_3000"]["num_calls"] == 2_880

    text = probe_limits.format_probe_report(report)
    assert "gpt-4o-mini" in text and "gpt-4.1-mini" in text


# ---------------------------------------------------------------------------
# 2. labeler.py Tests
# ---------------------------------------------------------------------------


def test_wrong_label_cross_layer_and_near_duplicate_rejection_embeddings() -> None:
    own_records = [
        {"pole_id": "L04_N0001_pos", "layer": 4, "own_label": "Medical case studies: clinical patients"},
        {"pole_id": "L04_N0002_pos", "layer": 4, "own_label": "SQL joins: database table queries"},
        {"pole_id": "L18_N0010_pos", "layer": 18, "own_label": "Medical case reports: hospital patients"},
        {"pole_id": "L18_N0020_pos", "layer": 18, "own_label": "Python dictionary keys: string mapping lookups"},
    ]
    # L04_N0001_pos and L18_N0010_pos have cosine ~0.99 (> 0.85) -> MUST be rejected!
    # L04_N0001_pos and L18_N0020_pos have cosine 0.0 (<= 0.85) -> MUST be chosen!
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.99, 0.141, 0.0],  # cosine with row 0 is ~0.990
            [0.0, 1.0, 0.0],     # cosine with row 0 is 0.0
        ],
        dtype=np.float64,
    )
    assigned = labeler.assign_wrong_labels(
        own_records=own_records,
        embeddings=embeddings,
        max_embed_cosine=0.85,
        seed=42,
    )
    assert assigned["L04_N0001_pos"]["wrong_donor_layer"] != 4
    assert assigned["L04_N0001_pos"]["wrong_donor_pole_id"] == "L18_N0020_pos"
    assert assigned["L04_N0001_pos"]["wrong_embed_cosine"] <= 0.85
    assert assigned["L04_N0001_pos"]["wrong_similarity_metric"] == "text-embedding-3-small"
    # Every pole must get a donor from a strictly different layer
    for rec in own_records:
        pid = rec["pole_id"]
        assert assigned[pid]["wrong_donor_layer"] != rec["layer"]


def test_wrong_label_word_jaccard_fallback_rejects_near_duplicate() -> None:
    own_records = [
        {"pole_id": "L04_N0001_pos", "layer": 4, "own_label": "Medical patient case study clinical diagnosis"},
        {"pole_id": "L18_N0010_pos", "layer": 18, "own_label": "Medical patient case study clinical diagnosis terms"},
        {"pole_id": "L18_N0020_pos", "layer": 18, "own_label": "Astronomical telescope galaxy redshift spectroscopy"},
    ]
    a1 = labeler.assign_wrong_labels(own_records, embeddings=None, max_jaccard=0.50, seed=42)
    a2 = labeler.assign_wrong_labels(own_records, embeddings=None, max_jaccard=0.50, seed=42)
    assert a1 == a2  # strictly deterministic
    assert a1["L04_N0001_pos"]["wrong_donor_pole_id"] == "L18_N0020_pos"
    assert a1["L04_N0001_pos"]["wrong_donor_layer"] == 18
    assert a1["L04_N0001_pos"]["wrong_embed_cosine"] <= 0.50
    assert a1["L04_N0001_pos"]["wrong_similarity_metric"] == "word_jaccard_fallback"


async def test_labeler_resumable_cache_and_token_only_format(tmp_path: Path) -> None:
    poles_doc, label_examples, _ = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 1, 18: 1}, pairs_per_cell=2
    )
    poles_path = tmp_path / "poles.json"
    ex_path = tmp_path / "label_examples.jsonl"
    out_path = tmp_path / "labels.jsonl"
    cache_path = tmp_path / "own_labels_cache.jsonl"
    poles_path.write_text(json.dumps(poles_doc), encoding="utf-8")
    ex_path.write_text("\n".join(json.dumps(r) for r in label_examples) + "\n", encoding="utf-8")

    chat_calls: list[str] = []

    class _MockAsyncChat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            user_txt = kwargs["messages"][1]["content"]
            pole_id = user_txt.splitlines()[0].split(": ")[1].strip()
            chat_calls.append(pole_id)
            payload = {
                "title": f"Pattern for {pole_id}",
                "description": f"Fires specifically on unique marker tokens for {pole_id}.",
            }
            return _make_chat_response(
                json.dumps(payload), prompt_tokens=1500, completion_tokens=30
            )

    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=_MockAsyncChat()))

    def _mock_embed(texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float64)

    rows_1 = await labeler.generate_labels_async(
        poles_path=poles_path,
        examples_path=ex_path,
        output_path=out_path,
        cache_path=cache_path,
        client=mock_client,
        embed_fn=_mock_embed,
    )
    assert len(rows_1) == 4
    assert len(chat_calls) == 4
    for r in rows_1:
        assert r["wrong_donor_layer"] != r["layer"]
        assert r["token_only_label"].startswith("Tokens such as: ")
        assert "strongly" not in r["token_only_label"].lower()
        assert r["own_label"] == f"{r['own_title']}: {r['own_description']}"

    # Second run MUST resume from cache with 0 new chat API calls!
    rows_2 = await labeler.generate_labels_async(
        poles_path=poles_path,
        examples_path=ex_path,
        output_path=out_path,
        cache_path=cache_path,
        client=mock_client,
        embed_fn=_mock_embed,
    )
    assert len(chat_calls) == 4
    assert rows_2 == rows_1


async def test_labeler_repetition_loop_detection_and_frequency_penalty_retry() -> None:
    bad_v1 = (
        "Suffixes forming adjectives: triggers on suffixes such as "
        "'-able', '-ic', '-ic', '-ic', '-ic', '-ic', '-ic', '-ic'."
    )
    assert labeler.detect_repetition_loop(bad_v1) is not None
    assert labeler.detect_repetition_loop("word, word, word, word") is not None
    assert labeler.detect_repetition_loop("clean concise description of digits") is None

    seen_kwargs: list[dict[str, Any]] = []

    class _LoopThenCleanChat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            seen_kwargs.append(kwargs)
            if len(seen_kwargs) == 1:
                payload = {
                    "title": "Adjective suffixes",
                    "description": "Triggers on '-ic', '-ic', '-ic', '-ic', '-ic', '-ic'.",
                }
            else:
                payload = {
                    "title": "Adjective-forming morphological suffixes",
                    "description": "Occurs on '-able', '-ic', and '-al' derivational adjective endings.",
                }
            return _make_chat_response(json.dumps(payload), prompt_tokens=1600, completion_tokens=40)

    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=_LoopThenCleanChat()))
    pole = {"pole_id": "L11_N0278_pos", "layer": 11}
    examples = [
        {"act_dir": 2.0, "marked_text": "some context <<ic>> tail", "target_token_id": 1, "target_token_str": "ic"}
    ]
    rec = await labeler._generate_one_own_label(
        client=mock_client, pole=pole, examples=examples, model="gpt-4.1"
    )
    assert len(seen_kwargs) == 2
    assert "frequency_penalty" not in seen_kwargs[0]
    assert seen_kwargs[1]["frequency_penalty"] == pytest.approx(0.5)
    assert rec["repetition_retries"] == 1
    assert labeler.detect_repetition_loop(rec["own_label"]) is None


async def test_labeler_boilerplate_strip_and_065_wrong_donor_cutoff(tmp_path: Path) -> None:
    stripped = labeler.strip_label_boilerplate(
        "Mixed / polysemantic: This neuron pole strongly activates on numeric dates and punctuation."
    )
    assert "this neuron pole" not in stripped.lower()
    assert "mixed / polysemantic" not in stripped.lower()
    assert "numeric dates" in stripped.lower()

    # Verify default 0.65 cutoff rejects cos=0.72 and chooses cos=0.35
    own_records = [
        {"pole_id": "L04_N0001_pos", "layer": 4, "own_label": "Numeric dates and years"},
        {"pole_id": "L18_N0010_pos", "layer": 18, "own_label": "Numeric years and percentages"},
        {"pole_id": "L18_N0020_pos", "layer": 18, "own_label": "French culinary verbs"},
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.72, math.sqrt(1.0 - 0.72**2)],  # cosine with row 0 = 0.72 (> 0.65 -> reject!)
            [0.35, math.sqrt(1.0 - 0.35**2)],  # cosine with row 0 = 0.35 (<= 0.65 -> accept!)
        ],
        dtype=np.float64,
    )
    assigned = labeler.assign_wrong_labels(own_records, embeddings=embeddings)
    assert assigned["L04_N0001_pos"]["wrong_donor_pole_id"] == "L18_N0020_pos"
    assert assigned["L04_N0001_pos"]["wrong_embed_cosine"] <= 0.65
    assert assigned["L04_N0001_pos"]["wrong_threshold_relaxed"] is False

    # Verify generate_labels_async strips boilerplate BEFORE calling embed_fn
    poles_doc, label_examples, _ = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 1, 18: 1}, pairs_per_cell=2
    )
    poles_path = tmp_path / "poles.json"
    ex_path = tmp_path / "label_examples.jsonl"
    out_path = tmp_path / "labels.jsonl"
    poles_path.write_text(json.dumps(poles_doc), encoding="utf-8")
    ex_path.write_text("\n".join(json.dumps(r) for r in label_examples) + "\n", encoding="utf-8")

    class _BoilerplateChat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            user_txt = kwargs["messages"][1]["content"]
            pid = user_txt.splitlines()[0].split(": ")[1].strip()
            return _make_chat_response(
                json.dumps(
                    {
                        "title": f"Topic {pid}",
                        "description": f"This neuron pole strongly activates on unique feature {pid}.",
                    }
                ),
                prompt_tokens=1500,
                completion_tokens=30,
            )

    embedded_inputs: list[str] = []

    def _capture_embed(texts: list[str]) -> np.ndarray:
        embedded_inputs.extend(texts)
        return np.eye(len(texts), dtype=np.float64)

    await labeler.generate_labels_async(
        poles_path=poles_path,
        examples_path=ex_path,
        output_path=out_path,
        cache_path=tmp_path / "cache.jsonl",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_BoilerplateChat())),
        embed_fn=_capture_embed,
    )
    assert len(embedded_inputs) == 4
    assert all("this neuron pole strongly activates on" not in t.lower() for t in embedded_inputs)


# ---------------------------------------------------------------------------
# 3. judge.py Tests
# ---------------------------------------------------------------------------


async def test_judge_one_pair_per_call_ab_randomization_and_logprob_margin(tmp_path: Path) -> None:
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 1, 18: 1},
        pairs_per_cell=4,
        include_p90=False,
        include_neighbour=False,
    )
    # Build labels map
    labels_by_pole = {
        p["pole_id"]: {
            "pole_id": p["pole_id"],
            "layer": p["layer"],
            "own_label": f"Own desc for {p['pole_id']}",
            "wrong_label": f"Wrong desc for {p['pole_id']}",
            "token_only_label": 'Activates on tokens such as: " tok", " word", " sym"',
        }
        for p in poles_doc["poles"]
    }

    # Pick only p99.9_plus -> 4 poles * 4 pairs * 4 conditions = 64 single-pair calls
    top_snips = [s for s in snippets if s["band"] == "p99.9_plus"]
    tasks = judge.build_task_plan(
        poles_data=poles_doc,
        snippets=top_snips,
        labels_by_pole=labels_by_pole,
        mode="core",
        judge_model="gpt-4o-mini",
        judge_backend="openai",
    )
    assert len(tasks) == 64

    recorded_requests: list[dict[str, Any]] = []

    class _MockAsyncRaw:
        async def create(self, **kwargs: Any) -> _MockRawResponse:
            recorded_requests.append(kwargs)
            user_msg = kwargs["messages"][1]["content"]
            # Enforce STRICT 1-pair-per-call invariant!
            assert user_msg.count("Snippet A:") == 1
            assert user_msg.count("Snippet B:") == 1
            assert kwargs["logprobs"] is True
            assert kwargs["top_logprobs"] == 5
            assert kwargs["max_tokens"] == 1
            assert kwargs["temperature"] == 0.0

            # Always pick whichever slot contains POS_TEXT (high confidence)
            a_section = user_msg.split("Snippet A:\n", 1)[1].split("\n\nSnippet B:\n", 1)[0]
            ans = "A" if "POS_TEXT" in a_section else "B"
            other = "B" if ans == "A" else "A"
            parsed = _make_chat_response(
                ans,
                top_lps=[_make_top_lp(ans, -0.02), _make_top_lp(other, -4.02)],
                prompt_tokens=450,
                completion_tokens=1,
            )
            return _MockRawResponse(
                parsed,
                {
                    "x-ratelimit-remaining-requests": "4900",
                    "x-ratelimit-remaining-tokens": "1990000",
                },
            )

    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=_MockAsyncRaw()))
    )
    out_path = tmp_path / "judgements.jsonl"
    summary = await judge.run_judge_tasks_async(
        tasks=tasks,
        output_path=out_path,
        max_usd=10.0,
        concurrency=8,
        target_rpm=None,
        openai_client=mock_client,
    )

    assert summary["completed_this_run"] == 64
    assert len(recorded_requests) == 64  # strictly 1 API call per pair!

    # Unified prompt: ALL 4 conditions use ONE identical system prompt & ONE user template, ZERO 'strongly'
    assert all(r["messages"][0]["content"] == JUDGE_SYSTEM_PROMPT for r in recorded_requests)
    assert all("strongly" not in r["messages"][0]["content"].lower() for r in recorded_requests)
    assert all("strongly" not in r["messages"][1]["content"].lower() for r in recorded_requests)
    no_label_reqs = [
        r for r in recorded_requests if f"Neuron pole description:\n{NO_LABEL_DESCRIPTION}\n" in r["messages"][1]["content"]
    ]
    assert len(no_label_reqs) == 16
    # Verify legacy 'Activates on tokens such as:' was normalized to 'Tokens such as:'
    tok_reqs = [
        r for r in recorded_requests if "Neuron pole description:\nTokens such as:" in r["messages"][1]["content"]
    ]
    assert len(tok_reqs) == 16

    rows = labeler.load_jsonl(out_path)
    assert len(rows) == 64
    pos_slots = {r["pos_slot"] for r in rows}
    assert pos_slots == {"A", "B"}  # randomized across both slots
    for r in rows:
        assert r["outcome"] == "scored"
        assert r["correct"] == 1.0
        assert r["predicted_slot"] == r["pos_slot"]
        assert r["margin"] == pytest.approx(4.0, abs=1e-4)
        assert r["logprob_imputed"] is False


def test_unified_single_prompt_template_zero_strongly() -> None:
    cond_msgs = {
        "own": judge.build_judge_messages("own", "Numeric digits", "snip A", "snip B"),
        "wrong": judge.build_judge_messages("wrong", "French verbs", "snip A", "snip B"),
        "token_only": judge.build_judge_messages(
            "token_only", 'Activates on tokens such as: "1", "2", "3"', "snip A", "snip B"
        ),
        "no_label": judge.build_judge_messages("no_label", None, "snip A", "snip B"),
    }
    sys_prompts = {m[0]["content"] for m in cond_msgs.values()}
    assert len(sys_prompts) == 1  # strictly ONE identical system prompt across all 4 conditions
    assert "strongly" not in JUDGE_SYSTEM_PROMPT.lower()
    assert "strongly" not in JUDGE_USER_TEMPLATE.lower()
    for cond, msgs in cond_msgs.items():
        for msg in msgs:
            assert "strongly" not in msg["content"].lower(), f"'strongly' leaked in {cond}"
        # Strip description line and assert identical surrounding user prompt template
        user = msgs[1]["content"]
        assert user.startswith("Neuron pole description:\n")
        assert "\n\nSnippet A:\nsnip A\n\nSnippet B:\nsnip B\n\n" in user
    assert f"Neuron pole description:\n{NO_LABEL_DESCRIPTION}\n" in cond_msgs["no_label"][1]["content"]
    assert 'Neuron pole description:\nTokens such as: "1", "2", "3"\n' in cond_msgs["token_only"][1]["content"]


def test_unparseable_and_logprob_tie_excluded_from_accuracy() -> None:
    # Exact logprob tie -> excluded (NOT scored as 0 or 0.5)
    pred_t, out_t, corr_t = judge.classify_judge_outcome("A", "A", -0.693147, -0.693147)
    assert pred_t is None and out_t == "logprob_tie" and corr_t is None

    # Unparseable text without logprobs -> excluded
    pred_u, out_u, corr_u = judge.classify_judge_outcome("Cannot decide", "B", None, None)
    assert pred_u is None and out_u == "unparseable" and corr_u is None

    # Valid scored row
    pred_s, out_s, corr_s = judge.classify_judge_outcome("B", "B", -2.5, -0.05)
    assert pred_s == "B" and out_s == "scored" and corr_s == 1.0

    # Aggregation excludes unparseable and logprob_tie from accuracy numerator & denominator
    rows = [
        {"pole_id": "L04_N0001_pos", "outcome": "scored", "predicted_slot": "A", "correct": 1.0},
        {"pole_id": "L04_N0001_pos", "outcome": "scored", "predicted_slot": "A", "correct": 1.0},
        {"pole_id": "L04_N0001_pos", "outcome": "scored", "predicted_slot": "B", "correct": 1.0},
        {"pole_id": "L04_N0001_pos", "outcome": "scored", "predicted_slot": "B", "correct": 0.0},
        {"pole_id": "L04_N0001_pos", "outcome": "unparseable", "predicted_slot": None, "correct": None},
        {"pole_id": "L04_N0001_pos", "outcome": "unparseable", "predicted_slot": None, "correct": None},
        {"pole_id": "L04_N0001_pos", "outcome": "logprob_tie", "predicted_slot": None, "correct": None},
        {"pole_id": "L04_N0001_pos", "outcome": "logprob_tie", "predicted_slot": None, "correct": None},
    ]
    st = analyze.neuron_clustered_bootstrap_accuracy(rows, n_resamples=200, seed=42)
    assert st["mean"] == pytest.approx(0.75)  # exact 3/4, NOT 3/8!
    assert st["n_scored"] == 4
    assert st["n_unparseable"] == 2
    assert st["n_logprob_tie"] == 2


def test_judge_logprob_imputation_when_other_letter_missing() -> None:
    top_lps = [_make_top_lp("A", -0.001), _make_top_lp("C", -12.0)]
    lp_a, lp_b, margin_a, imputed = judge.extract_ab_logprobs(top_lps, pos_slot="A")
    assert imputed is True
    assert lp_a == pytest.approx(-0.001, abs=1e-5)
    assert lp_b == pytest.approx(-14.0, abs=1e-5)
    assert margin_a == pytest.approx(13.999, abs=1e-4)

    _, _, margin_b, _ = judge.extract_ab_logprobs(top_lps, pos_slot="B")
    assert margin_b == pytest.approx(-13.999, abs=1e-4)

    # GPT-6 pruned tail logprobs: [('B', 0.0)] > [('B', -0.0004)] > [('B', -0.13), ('A', -2.10)]
    _, _, m_cert, imp_cert = judge.extract_ab_logprobs([_make_top_lp("B", 0.0)], pos_slot="B")
    _, _, m_high, imp_high = judge.extract_ab_logprobs([_make_top_lp("B", -0.0004)], pos_slot="B")
    _, _, m_mod, imp_mod = judge.extract_ab_logprobs(
        [_make_top_lp("B", -0.13), _make_top_lp("A", -2.10)], pos_slot="B"
    )
    assert imp_cert is True and imp_high is True and imp_mod is False
    assert m_cert is not None and m_high is not None and m_mod is not None
    assert m_cert > m_high > m_mod > 0.0


async def test_gpt6_reasoning_kwargs_billing_and_both_orders_flip_and_resume(tmp_path: Path) -> None:
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer=LAYER_NEURON_COUNTS,
        pairs_per_cell=6,
        include_p90=False,
        include_neighbour=False,
    )
    labels_by_pole = {
        p["pole_id"]: {
            "pole_id": p["pole_id"],
            "layer": p["layer"],
            "own_label": "own",
            "wrong_label": "wrong",
            "token_only_label": "tok",
        }
        for p in poles_doc["poles"]
    }

    sol_tasks = judge.build_task_plan(
        poles_doc,
        snippets,
        labels_by_pole,
        mode="pilot",
        judge_model="gpt-6.2-sol",
        judge_backend="openai",
        bands_filter=["p99.9_plus"],
        both_orders=True,
        reasoning_effort="low",
    )
    assert len(sol_tasks) == 960  # 480 fwd + 480 rev
    assert {t.judge_model for t in sol_tasks} == {"gpt-6.2-sol@low"}
    assert {t.wire_model for t in sol_tasks} == {"gpt-6.2-sol"}
    assert sum(1 for t in sol_tasks if t.order == "fwd") == 480
    assert sum(1 for t in sol_tasks if t.order == "rev") == 480

    # Verify exact fwd <-> rev flip of pos_slot and snippets
    by_pco = {(t.pair_id, t.condition, t.order): t for t in sol_tasks}
    t_fwd = sol_tasks[0]
    t_rev = by_pco[(t_fwd.pair_id, t_fwd.condition, "rev")]
    assert t_fwd.pos_slot != t_rev.pos_slot
    assert t_fwd.snippet_a == t_rev.snippet_b
    assert t_fwd.snippet_b == t_rev.snippet_a

    # Verify wire kwargs for reasoning='low' vs reasoning='none'
    msgs = judge.build_judge_messages("own", "desc", "a", "b")
    sol_kw = judge._build_openai_request_kwargs(t_fwd, msgs)
    assert sol_kw["model"] == "gpt-6.2-sol"
    assert sol_kw["reasoning_effort"] == "low"
    assert sol_kw["max_completion_tokens"] >= 512
    assert "temperature" not in sol_kw
    assert "logprobs" not in sol_kw

    top_one = [s for s in snippets if s["band"] == "p99.9_plus"][:1]
    terra_task = judge.build_task_plan(
        poles_doc,
        top_one,
        labels_by_pole,
        mode="pilot",
        judge_model="gpt-6.1-terra",
        judge_backend="openai",
        bands_filter=["p99.9_plus"],
        reasoning_effort="none",
    )[0]
    terra_kw = judge._build_openai_request_kwargs(terra_task, msgs)
    assert terra_kw["model"] == "gpt-6.1-terra"
    assert terra_kw["reasoning_effort"] == "none"
    assert terra_kw["temperature"] == 0.0
    assert terra_kw["logprobs"] is True and terra_kw["top_logprobs"] == 5

    # Run 2 fwd + 2 rev mock calls with reasoning_tokens=60, visible=4 (total comp=64)
    seen_calls: list[dict[str, Any]] = []

    class _MockSolChat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            seen_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="A"), logprobs=None)],
                usage=SimpleNamespace(
                    prompt_tokens=380,
                    completion_tokens=64,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=60),
                ),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_MockSolChat()))
    out_path = tmp_path / "sol.jsonl"
    pair_slice = [t_fwd, t_rev]

    # Run fwd first, then run [fwd, rev] -> fwd skipped, rev executed cleanly!
    s1 = await judge.run_judge_tasks_async(
        tasks=[t_fwd], output_path=out_path, concurrency=1, target_rpm=None, openai_client=client
    )
    s2 = await judge.run_judge_tasks_async(
        tasks=pair_slice, output_path=out_path, concurrency=1, target_rpm=None, openai_client=client
    )
    assert s1["completed_this_run"] == 1
    assert s2["skipped_existing"] == 1 and s2["completed_this_run"] == 1
    assert len(seen_calls) == 2

    rows = labeler.load_jsonl(out_path)
    assert {r["order"] for r in rows} == {"fwd", "rev"}
    expected_cost = (380 * 2.0 + 64 * 10.0) / 1_000_000.0
    for r in rows:
        assert r["judge_model"] == "gpt-6.2-sol@low"
        assert r["api_model"] == "gpt-6.2-sol"
        assert r["reasoning_tokens"] == 60
        assert r["visible_output_tokens"] == 4
        assert r["cost_usd"] == pytest.approx(expected_cost, abs=1e-7)


def test_analyze_both_orders_consistency_and_reasoning_token_audit() -> None:
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 1, 18: 1},
        pairs_per_cell=4,
        include_p90=False,
        include_neighbour=False,
    )
    top_4 = [s for s in snippets if s["band"] == "p99.9_plus"][:4]
    judgements: list[dict[str, Any]] = []
    # Pair 0: both fwd & rev pick pos (consistent, both correct)
    # Pair 1: both fwd & rev pick dist (consistent, both wrong)
    # Pair 2: both fwd & rev always pick Slot A (inconsistent!)
    # Pair 3: both fwd & rev always pick Slot B (inconsistent!)
    for idx, s in enumerate(top_4):
        fwd_slot = s["pos_slot"]
        rev_slot = "B" if fwd_slot == "A" else "A"
        for order, slot in (("fwd", fwd_slot), ("rev", rev_slot)):
            if idx == 0:
                pred = slot
            elif idx == 1:
                pred = "B" if slot == "A" else "A"
            elif idx == 2:
                pred = "A"
            else:
                pred = "B"
            judgements.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": s["pole_id"],
                    "layer": s["layer"],
                    "band": s["band"],
                    "distractor_type": "dead_zone",
                    "condition": "own",
                    "judge_model": "gpt-6.2-sol@low",
                    "api_model": "gpt-6.2-sol",
                    "reasoning_effort": "low",
                    "order": order,
                    "judge_backend": "openai",
                    "pos_slot": slot,
                    "predicted_slot": pred,
                    "correct": 1.0 if pred == slot else 0.0,
                    "raw_response": pred,
                    "logprob_A": None,
                    "logprob_B": None,
                    "margin": None,
                    "logprob_imputed": False,
                    "prompt_tokens": 380,
                    "completion_tokens": 54,
                    "reasoning_tokens": 50,
                    "visible_output_tokens": 4,
                    "cost_usd": 0.0013,
                    "timestamp_utc": "2026-11-21T12:00:00Z",
                }
            )

    metrics = analyze.compute_all_metrics(
        poles_data=poles_doc,
        snippets=top_4,
        judgements=judgements,
        baselines=[],
        n_resamples=50,
        seed=42,
    )
    bo = metrics["both_orders_diagnostics"]["gpt-6.2-sol@low"]["by_scope"]["own"]
    assert bo["n_paired"] == 4
    assert bo["order_consistency_rate"] == pytest.approx(0.5)
    assert bo["mean_both_orders_acc"] == pytest.approx(0.5)

    ca = metrics["cost_audit"]["gpt-6.2-sol@low"]
    assert ca["calls"] == 8
    assert ca["has_logprobs"] is False
    assert ca["mean_reasoning_tokens"] == pytest.approx(50.0)
    assert ca["mean_visible_output_tokens"] == pytest.approx(4.0)
    assert ca["mean_cost_usd_per_call"] == pytest.approx(0.0013)

    md = analyze.render_summary_tables_md(metrics)
    assert "2d. Both-Orders" in md
    assert "Mean Reas Tok" in md


async def test_judge_resumability_and_truncated_crash_line_recovery(tmp_path: Path) -> None:
    tasks = [
        judge.JudgeTask(
            pair_id=f"L04_N0100_pos__p99.9_plus__dead_zone__{i}",
            pole_id="L04_N0100_pos",
            layer=4,
            band="p99.9_plus",
            distractor_type="dead_zone",
            condition="own",
            pos_slot="A",
            snippet_a=f"pos {i}",
            snippet_b=f"dist {i}",
            label_text="Some label",
            judge_model="gpt-4o-mini",
            judge_backend="openai",
        )
        for i in range(6)
    ]

    call_count = 0

    class _MockAsyncChat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            return _make_chat_response(
                "A",
                top_lps=[_make_top_lp("A", -0.1), _make_top_lp("B", -2.3)],
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_MockAsyncChat()))
    out_path = tmp_path / "judgements.jsonl"

    # Run first 3 tasks
    s1 = await judge.run_judge_tasks_async(
        tasks=tasks[:3], output_path=out_path, concurrency=2, target_rpm=None, openai_client=client
    )
    assert s1["completed_this_run"] == 3
    assert call_count == 3

    # Simulate crash leaving a truncated trailing partial JSON line
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write('{"pair_id": "L04_N0100_pos__p99.9_plus__dead_zone__3", "trunc')

    # Re-run all 6 tasks -> must skip the 3 valid completed keys and run only 3 new ones
    s2 = await judge.run_judge_tasks_async(
        tasks=tasks, output_path=out_path, concurrency=2, target_rpm=None, openai_client=client
    )
    assert s2["skipped_existing"] == 3
    assert s2["completed_this_run"] == 3
    assert call_count == 6
    assert len(judge.load_completed_keys(out_path)) == 6


async def test_judge_hard_usd_spend_cap_aborts_early(tmp_path: Path) -> None:
    tasks = [
        judge.JudgeTask(
            pair_id=f"L04_N0100_pos__p99.9_plus__dead_zone__{i}",
            pole_id="L04_N0100_pos",
            layer=4,
            band="p99.9_plus",
            distractor_type="dead_zone",
            condition="own",
            pos_slot="A",
            snippet_a="pos",
            snippet_b="dist",
            label_text="Label",
            judge_model="gpt-4.1-mini",  # $0.40 / 1M input
            judge_backend="openai",
        )
        for i in range(20)
    ]

    api_calls = 0

    class _ExpensiveMockChat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            nonlocal api_calls
            api_calls += 1
            # 500,000 prompt tokens * $0.40/1M = $0.20 per call
            return _make_chat_response(
                "A",
                top_lps=[_make_top_lp("A", -0.05), _make_top_lp("B", -3.0)],
                prompt_tokens=500_000,
                completion_tokens=1,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_ExpensiveMockChat()))
    out_path = tmp_path / "judgements.jsonl"

    with pytest.raises(judge.SpendLimitExceededError):
        await judge.run_judge_tasks_async(
            tasks=tasks,
            output_path=out_path,
            max_usd=0.50,  # allows at most 3 calls ($0.60 trips >= $0.50 with concurrency=1)
            concurrency=1,
            target_rpm=None,
            openai_client=client,
        )

    assert api_calls <= 3
    # Completed rows prior to abort MUST be safely persisted on disk
    saved = labeler.load_jsonl(out_path)
    assert len(saved) == api_calls


async def test_judge_429_backoff_and_recovery(tmp_path: Path) -> None:
    task = judge.JudgeTask(
        pair_id="L04_N0100_pos__p99.9_plus__dead_zone__0",
        pole_id="L04_N0100_pos",
        layer=4,
        band="p99.9_plus",
        distractor_type="dead_zone",
        condition="own",
        pos_slot="B",
        snippet_a="dist",
        snippet_b="pos",
        label_text="Label",
        judge_model="gpt-4o-mini",
        judge_backend="openai",
    )

    attempts = 0

    class _Flaky429Chat:
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise _FakeRateLimitError()
            return _make_chat_response(
                "B", top_lps=[_make_top_lp("B", -0.05), _make_top_lp("A", -3.0)]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Flaky429Chat()))
    out_path = tmp_path / "judgements.jsonl"
    summary = await judge.run_judge_tasks_async(
        tasks=[task],
        output_path=out_path,
        concurrency=1,
        target_rpm=6000.0,
        openai_client=client,
        base_backoff=0.01,
    )
    assert attempts == 3
    assert summary["completed_this_run"] == 1
    rows = labeler.load_jsonl(out_path)
    assert rows[0]["correct"] == 1.0


async def test_ollama_judge_explicit_think_false_wire_payload(tmp_path: Path) -> None:
    captured_payloads: list[dict[str, Any]] = []
    captured_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": "B"},
                "done": True,
                "prompt_eval_count": 410,
                "eval_count": 2,
            },
        )

    transport = httpx.MockTransport(_handler)
    tasks = [
        judge.JudgeTask(
            pair_id=f"L04_N0100_pos__p99.9_plus__dead_zone__{i}",
            pole_id="L04_N0100_pos",
            layer=4,
            band="p99.9_plus",
            distractor_type="dead_zone",
            condition="own",
            pos_slot="B" if i % 2 == 0 else "A",
            snippet_a="snippet a text",
            snippet_b="snippet b text",
            label_text="Test description",
            judge_model="qwen3.5:9b",
            judge_backend="ollama",
        )
        for i in range(4)
    ]

    out_path = tmp_path / "ollama_judgements.jsonl"
    async with httpx.AsyncClient(transport=transport) as http_client:
        summary = await judge.run_judge_tasks_async(
            tasks=tasks,
            output_path=out_path,
            ollama_url="http://localhost:11434",
            http_client=http_client,
        )

    assert summary["completed_this_run"] == 4
    assert len(captured_payloads) == 4
    for url, payload in zip(captured_urls, captured_payloads):
        assert url == "http://localhost:11434/api/chat"
        # MANDATORY: top-level "think": False so qwen3.5:9b never emits thinking traces!
        assert "think" in payload
        assert payload["think"] is False
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.0

    rows = labeler.load_jsonl(out_path)
    assert all(r["judge_backend"] == "ollama" for r in rows)
    assert all(r["cost_usd"] == 0.0 and r["margin"] is None for r in rows)
    assert sum(r["correct"] for r in rows) == 2.0


def test_judge_task_plan_exact_call_counts_all_modes() -> None:
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer=LAYER_NEURON_COUNTS,  # exact 50 neurons = 100 poles
        pairs_per_cell=6,
        include_p90=True,
        include_neighbour=True,
    )
    assert len(poles_doc["poles"]) == 100
    assert len(snippets) == 4_200  # 100 * (4*6 + 1*6 + 2*6)

    labels_by_pole = {
        p["pole_id"]: {
            "pole_id": p["pole_id"],
            "layer": p["layer"],
            "own_label": "own",
            "wrong_label": "wrong",
            "token_only_label": "tok",
        }
        for p in poles_doc["poles"]
    }

    assert len(judge.build_task_plan(poles_doc, snippets, labels_by_pole, "pilot", "gpt-4o-mini", "openai")) == 1_920
    assert len(judge.build_task_plan(poles_doc, snippets, labels_by_pole, "toy", "gpt-4o-mini", "openai")) == 3_240
    assert len(judge.build_task_plan(poles_doc, snippets, labels_by_pole, "core", "gpt-4o-mini", "openai")) == 9_600
    assert (
        len(
            judge.build_task_plan(
                poles_doc, snippets, labels_by_pole, "core", "gpt-4o-mini", "openai", include_p90_compare=True
            )
        )
        == 12_000
    )
    assert len(judge.build_task_plan(poles_doc, snippets, labels_by_pole, "p90_compare", "gpt-4o-mini", "openai")) == 2_400
    assert len(judge.build_task_plan(poles_doc, snippets, labels_by_pole, "neighbour", "gpt-4o-mini", "openai")) == 4_800
    assert len(judge.build_task_plan(poles_doc, snippets, labels_by_pole, "second_judge", "gpt-4.1-mini", "openai")) == 2_880
    assert (
        len(
            judge.build_task_plan(
                poles_doc, snippets, labels_by_pole, "second_judge", "gpt-4.1-mini", "openai", lite=True
            )
        )
        == 1_440
    )
    # --bands filter: pilot + p99.9_plus only -> 20 poles * 1 band * 4 conds * 6 pairs = 480 calls
    pilot_top = judge.build_task_plan(
        poles_doc,
        snippets,
        labels_by_pole,
        "pilot",
        "gpt-4.1-mini",
        "openai",
        bands_filter=["p99.9_plus"],
    )
    assert len(pilot_top) == 480
    assert {t.band for t in pilot_top} == {"p99.9_plus"}


# ---------------------------------------------------------------------------
# 4. analyze.py & cli.py Tests
# ---------------------------------------------------------------------------


def test_no_auroc_and_neuron_clustered_bootstrap_math() -> None:
    # AUROC functions completely removed from analyze.py
    assert not hasattr(analyze, "compute_margin_auroc")
    assert not hasattr(analyze, "pole_clustered_bootstrap_auroc")

    # Neuron-clustered bootstrap MUST resample whole neurons (layer, neuron),
    # pulling both pos and neg poles together!
    # Create 20 neurons (40 poles): 10 neurons with BOTH pos & neg = 1.0,
    # 10 neurons with BOTH pos & neg = 0.0 -> mean = 0.50.
    rows: list[dict[str, Any]] = []
    poles_by_id: dict[str, dict[str, Any]] = {}
    for n_idx in range(20):
        val = 1.0 if n_idx < 10 else 0.0
        for sign, pname in ((1, "pos"), (-1, "neg")):
            pid = make_pole_id(4, n_idx, sign)
            poles_by_id[pid] = {"pole_id": pid, "layer": 4, "neuron": n_idx}
            for _ in range(6):
                rows.append({"pole_id": pid, "outcome": "scored", "predicted_slot": "A", "correct": val})

    st1 = analyze.neuron_clustered_bootstrap_accuracy(rows, poles_by_id=poles_by_id, n_resamples=2000, seed=42)
    st2 = analyze.neuron_clustered_bootstrap_accuracy(rows, poles_by_id=poles_by_id, n_resamples=2000, seed=42)
    assert st1 == st2
    assert st1["mean"] == pytest.approx(0.50)
    assert st1["n_neurons"] == 20
    assert st1["n_poles"] == 40
    assert st1["n_scored"] == 240

    # Compare against naive pole-level bootstrap (treating 40 poles as independent):
    # With perfect within-neuron correlation, neuron-cluster SE = sqrt(2) * pole-level SE!
    pole_vals = np.array([1.0] * 20 + [0.0] * 20)
    rng = np.random.default_rng(42)
    naive_boot = pole_vals[rng.integers(0, 40, size=(2000, 40))].mean(axis=1)
    naive_lo, naive_hi = np.percentile(naive_boot, [2.5, 97.5])
    neuron_width = st1["ci_high"] - st1["ci_low"]
    naive_width = float(naive_hi - naive_lo)
    assert neuron_width > naive_width * 1.25  # theoretically ~1.414x wider!

    # Cohen's kappa exact check
    assert analyze.cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == pytest.approx(1.0)
    assert analyze.cohens_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"]) == pytest.approx(-1.0)


def test_full_analysis_pipeline_figure_tables_q3_second_judge_and_worksheet(tmp_path: Path) -> None:
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 3, 11: 3, 18: 2, 24: 2},  # 10 neurons = 20 poles
        pairs_per_cell=6,
        include_p90=True,
        include_neighbour=True,
    )
    labels = [
        {
            "pole_id": p["pole_id"],
            "layer": p["layer"],
            "own_title": f"Title {p['pole_id']}",
            "own_description": f"Description for {p['pole_id']}",
            "own_label": f"Title {p['pole_id']}: Description for {p['pole_id']}",
            "wrong_label": "Unrelated label: different layer",
            "wrong_donor_pole_id": "L24_N0100_pos" if p["layer"] != 24 else "L04_N0100_pos",
            "wrong_donor_layer": 24 if p["layer"] != 24 else 4,
            "wrong_embed_cosine": 0.12,
            "token_only_label": 'Activates on tokens such as: " a", " b", " c"',
            "top_3_token_ids": p["top_3_token_ids"],
            "top_3_tokens": p["top_3_tokens"],
            "label_model": "gpt-4.1-mini",
            "prompt_tokens": 1800,
            "completion_tokens": 35,
        }
        for p in poles_doc["poles"]
    ]

    rng = np.random.default_rng(123)
    judgements: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []

    band_own_prob = {"p75_90": 0.58, "p90_97": 0.72, "p97_99.9": 0.85, "p99.9_plus": 0.94}
    for s in snippets:
        pid = s["pole_id"]
        band = s["band"]
        dist_type = s["distractor_type"]
        pos_slot = s["pos_slot"]
        other_slot = "B" if pos_slot == "A" else "A"
        is_q3 = s["pos"]["quadrant"] == "Q3"

        # Rule baselines
        baselines.append(
            {
                "pair_id": s["pair_id"],
                "pole_id": pid,
                "band": band,
                "distractor_type": dist_type,
                "rule": "rarity",
                "score_diff": 2.0,
                "is_tie": False,
                "correct": 1.0 if (band == "p99.9_plus" and rng.random() < 0.65) else float(rng.random() < 0.52),
            }
        )
        baselines.append(
            {
                "pair_id": s["pair_id"],
                "pole_id": pid,
                "band": band,
                "distractor_type": dist_type,
                "rule": "top_token",
                "score_diff": 1.0,
                "is_tie": False,
                "correct": 1.0 if (band == "p99.9_plus" and rng.random() < 0.70) else 0.5,
            }
        )

        # Primary judge (gpt-4o-mini) across all 4 conditions
        for cond in CONDITIONS:
            if cond == "own":
                p_corr = 0.52 if is_q3 else band_own_prob[band]
                if s["layer"] == 4 and band == "p99.9_plus":
                    p_corr = 0.96
                if dist_type == NEIGHBOUR_DISTRACTOR_TYPE:
                    p_corr -= 0.12
            elif cond == "token_only":
                p_corr = 0.75 if band == "p99.9_plus" else 0.54
            elif cond == "no_label":
                p_corr = 0.58 if band == "p99.9_plus" else 0.50
            else:
                p_corr = 0.50

            is_corr = bool(rng.random() < p_corr)
            pred = pos_slot if is_corr else other_slot
            margin = float(rng.uniform(0.5, 4.0) if is_corr else rng.uniform(-3.0, -0.2))
            lp_pos = -0.05 if is_corr else -2.5
            lp_dist = lp_pos - margin
            judgements.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": pid,
                    "layer": s["layer"],
                    "band": band,
                    "distractor_type": dist_type,
                    "condition": cond,
                    "judge_model": "gpt-4o-mini",
                    "judge_backend": "openai",
                    "pos_slot": pos_slot,
                    "predicted_slot": pred,
                    "correct": 1.0 if is_corr else 0.0,
                    "raw_response": pred,
                    "logprob_A": lp_pos if pos_slot == "A" else lp_dist,
                    "logprob_B": lp_dist if pos_slot == "A" else lp_pos,
                    "margin": margin,
                    "logprob_imputed": False,
                    "prompt_tokens": 450,
                    "completion_tokens": 1,
                    "cost_usd": 0.000068,
                    "timestamp_utc": "2026-11-21T12:00:00Z",
                }
            )

            # Second judge (gpt-4.1-mini) on dead_zone subset
            if dist_type == "dead_zone" and s["pair_idx"] < 4:
                agree = bool(rng.random() < 0.88)
                pred2 = pred if agree else ("B" if pred == "A" else "A")
                corr2 = 1.0 if pred2 == pos_slot else 0.0
                judgements.append(
                    {
                        "pair_id": s["pair_id"],
                        "pole_id": pid,
                        "layer": s["layer"],
                        "band": band,
                        "distractor_type": dist_type,
                        "condition": cond,
                        "judge_model": "gpt-4.1-mini",
                        "judge_backend": "openai",
                        "pos_slot": pos_slot,
                        "predicted_slot": pred2,
                        "correct": corr2,
                        "raw_response": pred2,
                        "logprob_A": -0.1,
                        "logprob_B": -2.0,
                        "margin": 1.9 if corr2 == 1.0 else -1.9,
                        "logprob_imputed": False,
                        "prompt_tokens": 450,
                        "completion_tokens": 1,
                        "cost_usd": 0.000182,
                        "timestamp_utc": "2026-11-21T12:05:00Z",
                    }
                )

    poles_path = tmp_path / "poles.json"
    snips_path = tmp_path / "snippets.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    judge_path = tmp_path / "judgements.jsonl"
    base_path = tmp_path / "baselines.jsonl"
    fig_path = tmp_path / "figures" / "w_band_accuracy.png"
    tables_path = tmp_path / "results" / "summary_tables.md"
    ws_path = tmp_path / "audit" / "raw_30_poles_worksheet.md"
    metrics_path = tmp_path / "results" / "metrics.json"

    poles_path.write_text(json.dumps(poles_doc), encoding="utf-8")
    snips_path.write_text("\n".join(json.dumps(r) for r in snippets) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    judge_path.write_text("\n".join(json.dumps(r) for r in judgements) + "\n", encoding="utf-8")
    base_path.write_text("\n".join(json.dumps(r) for r in baselines) + "\n", encoding="utf-8")

    metrics = analyze.run_analysis(
        poles_path=poles_path,
        snippets_path=snips_path,
        labels_path=labels_path,
        judgements_path=judge_path,
        baselines_path=base_path,
        structural_path=tmp_path / "nonexistent_structural.json",
        figure_path=fig_path,
        tables_path=tables_path,
        worksheet_path=ws_path,
        metrics_json_path=metrics_path,
        n_resamples=200,
        seed=42,
    )

    # Verify PNG figure exists and has valid PNG magic header
    assert fig_path.is_file()
    assert fig_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # Verify summary tables contain every required section & ZERO AUROC
    tables_md = tables_path.read_text(encoding="utf-8")
    assert "Main 4-Band × Method Detection Audit" in tables_md
    assert "Neuron-Clustered Bootstrap CI" in tables_md
    assert "auroc" not in tables_md.lower()
    assert "goodfire" not in tables_md.lower() or "0.917" not in analyze.__file__
    assert "0.917" not in Path(analyze.__file__).read_text(encoding="utf-8")
    assert "0.927" not in Path(analyze.__file__).read_text(encoding="utf-8")
    assert "Layer-4 Positive Control & Pilot Gate Diagnostic" in tables_md
    assert "SwiGLU Quadrant Slice" in tables_md
    assert "`Q3`" in tables_md
    assert "Dead-Zone vs. Goodfire `<p90` Distractor" in tables_md
    assert f"(`{NEIGHBOUR_DISTRACTOR_TYPE}`)" in tables_md
    assert "Second-Judge Agreement" in tables_md
    assert "gpt-4.1-mini" in tables_md
    assert "First-Position Bias" in tables_md
    assert "Unparseable" in tables_md and "Logprob Ties" in tables_md
    assert "GO (" in metrics["pilot_gate"]["verdict"]
    assert metrics["quadrant_slice"]["p75_90"]["Q3"]["own"]["n_scored"] > 0
    assert math.isfinite(metrics["second_judge_agreement"]["gpt-4.1-mini"]["overall"]["exact_agreement"])

    # Verify 30-pole manual audit worksheet (20 poles available -> all 20 rendered)
    ws_md = ws_path.read_text(encoding="utf-8")
    assert "Manual Audit Worksheet: 20 Stratified Qwen3-1.7B Neuron Poles" in ws_md
    assert "[LAYER-4 POSITIVE CONTROL]" in ws_md
    assert "Manual Reading Notes" in ws_md
    assert "FIRING:" in ws_md and "DISTRACTOR:" in ws_md


def test_parse_ab_choice_rejects_unrelated_words_and_uses_logprob_fallback() -> None:
    # Unrelated words containing 'a'/'b' MUST NOT be silently misparsed!
    for bad in ("Cannot", "Both", "Answer", "Neither", "A or B", ""):
        assert judge.parse_ab_choice(bad) is None

    # Valid explicit answers parse cleanly
    assert judge.parse_ab_choice("A") == "A"
    assert judge.parse_ab_choice(" b ") == "B"
    assert judge.parse_ab_choice("Snippet B.") == "B"
    assert judge.parse_ab_choice("Snippet A is stronger") == "A"

    # Unparseable text falls back cleanly when logprobs are present
    assert judge.parse_ab_choice("Cannot", logprob_a=-0.1, logprob_b=-2.5) == "A"
    assert judge.parse_ab_choice("Both", logprob_a=-3.0, logprob_b=-0.2) == "B"


def test_pilot_gate_floor_warning_and_empty_baselines_zero_warnings() -> None:
    import warnings

    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 2, 18: 2},
        pairs_per_cell=6,
        include_p90=False,
        include_neighbour=False,
    )
    # Construct judgements where L4 own = 0.833 and L4 no_label = 0.833 (gap = 0.0 < +0.10!)
    judgements: list[dict[str, Any]] = []
    for s in snippets:
        if s["band"] != "p99.9_plus":
            continue
        for cond in CONDITIONS:
            is_corr = (s["pair_idx"] < 5) if cond in ("own", "no_label") else (s["pair_idx"] < 3)
            pos_slot = s["pos_slot"]
            pred = pos_slot if is_corr else ("B" if pos_slot == "A" else "A")
            judgements.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": s["pole_id"],
                    "layer": s["layer"],
                    "band": s["band"],
                    "distractor_type": s["distractor_type"],
                    "condition": cond,
                    "judge_model": "gpt-4o-mini",
                    "judge_backend": "openai",
                    "pos_slot": pos_slot,
                    "predicted_slot": pred,
                    "correct": 1.0 if is_corr else 0.0,
                    "raw_response": pred,
                    "logprob_A": -0.1,
                    "logprob_B": -2.0,
                    "margin": 1.9 if is_corr else -1.9,
                    "logprob_imputed": False,
                    "prompt_tokens": 450,
                    "completion_tokens": 1,
                    "cost_usd": 0.000068,
                    "timestamp_utc": "2026-11-21T12:00:00Z",
                }
            )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        metrics = analyze.compute_all_metrics(
            poles_data=poles_doc,
            snippets=snippets,
            judgements=judgements,
            baselines=[],  # empty baselines must NOT trigger All-NaN RuntimeWarning!
            n_resamples=100,
            seed=42,
        )
        md = analyze.render_summary_tables_md(metrics)

    assert "STOP / FLOOR WARNING" in metrics["pilot_gate"]["verdict"]
    assert metrics["pilot_gate"]["l4_floor_gap"] == pytest.approx(0.0, abs=1e-6)
    assert "STOP / FLOOR WARNING" in md
    assert len(caught) == 0


def test_analyze_condition3_three_way_cohort_split_and_multi_judge_top_band(tmp_path: Path) -> None:
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer=LAYER_NEURON_COUNTS,  # exact 50 neurons = 100 poles (20 pilot + 80 heldout)
        pairs_per_cell=6,
        include_p90=False,
        include_neighbour=False,
    )
    poles_by_id = {p["pole_id"]: p for p in poles_doc["poles"]}
    judgements: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []

    for s in snippets:
        pid = s["pole_id"]
        in_pilot = bool(poles_by_id[pid]["in_pilot"])
        pos_slot = s["pos_slot"]
        other_slot = "B" if pos_slot == "A" else "A"
        for rule in ("rarity", "top_token"):
            baselines.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": pid,
                    "band": s["band"],
                    "distractor_type": "dead_zone",
                    "rule": rule,
                    "score_diff": 1.0,
                    "is_tie": False,
                    "correct": 1.0 if (s["pair_idx"] < 4 if in_pilot else s["pair_idx"] < 3) else 0.0,
                }
            )
        for cond in CONDITIONS:
            # Make pilot_20 own = 5/6 (0.8333) and heldout_80 own = 4/6 (0.6667) -> exact check!
            if cond == "own":
                is_corr = s["pair_idx"] < (5 if in_pilot else 4)
            else:
                is_corr = s["pair_idx"] < 3
            pred = pos_slot if is_corr else other_slot
            judgements.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": pid,
                    "layer": s["layer"],
                    "band": s["band"],
                    "distractor_type": "dead_zone",
                    "condition": cond,
                    "judge_model": "gpt-4o-mini",
                    "judge_backend": "openai",
                    "pos_slot": pos_slot,
                    "predicted_slot": pred,
                    "correct": 1.0 if is_corr else 0.0,
                    "raw_response": pred,
                    "logprob_A": -0.1,
                    "logprob_B": -2.0,
                    "margin": 1.9 if is_corr else -1.9,
                    "logprob_imputed": False,
                    "prompt_tokens": 400,
                    "completion_tokens": 1,
                    "cost_usd": 0.00006,
                    "timestamp_utc": "2026-11-21T12:00:00Z",
                }
            )
            # Second judge on pilot p99.9_plus only (scores 6/6 = 1.0 on own)
            if in_pilot and s["band"] == "p99.9_plus":
                is_corr2 = True if cond == "own" else (s["pair_idx"] < 3)
                pred2 = pos_slot if is_corr2 else other_slot
                judgements.append(
                    {
                        "pair_id": s["pair_id"],
                        "pole_id": pid,
                        "layer": s["layer"],
                        "band": s["band"],
                        "distractor_type": "dead_zone",
                        "condition": cond,
                        "judge_model": "gpt-4.1-mini",
                        "judge_backend": "openai",
                        "pos_slot": pos_slot,
                        "predicted_slot": pred2,
                        "correct": 1.0 if is_corr2 else 0.0,
                        "raw_response": pred2,
                        "logprob_A": -0.05,
                        "logprob_B": -3.0,
                        "margin": 2.95 if is_corr2 else -2.95,
                        "logprob_imputed": False,
                        "prompt_tokens": 400,
                        "completion_tokens": 1,
                        "cost_usd": 0.00016,
                        "timestamp_utc": "2026-11-21T12:05:00Z",
                    }
                )

    metrics = analyze.compute_all_metrics(
        poles_data=poles_doc,
        snippets=snippets,
        judgements=judgements,
        baselines=baselines,
        n_resamples=100,
        seed=42,
    )
    assert metrics["cohort_counts"]["heldout_80"] == {"total_poles": 80, "judged_poles": 80}
    assert metrics["cohort_counts"]["pilot_20"] == {"total_poles": 20, "judged_poles": 20}
    assert metrics["cohort_counts"]["all_100"] == {"total_poles": 100, "judged_poles": 100}

    assert metrics["cohort_tables"]["pilot_20"]["p99.9_plus"]["own"]["acc"]["mean"] == pytest.approx(5 / 6)
    assert metrics["cohort_tables"]["heldout_80"]["p99.9_plus"]["own"]["acc"]["mean"] == pytest.approx(4 / 6)
    assert metrics["cohort_tables"]["all_100"]["p99.9_plus"]["own"]["acc"]["mean"] == pytest.approx(
        0.2 * (5 / 6) + 0.8 * (4 / 6)
    )
    # Per-judge top-band comparison captures both gpt-4o-mini and gpt-4.1-mini
    assert metrics["per_judge_top_band"]["gpt-4.1-mini"]["layer4_control"]["own"]["acc"]["mean"] == pytest.approx(1.0)

    md = analyze.render_summary_tables_md(metrics)
    assert "1.1 Held-Out 80 Non-Pilot Poles" in md
    assert "1.2 Pilot 20 Poles" in md
    assert "1.3 All 100 Poles Pooled" in md
    assert "2b. Per-Judge Top-Band" in md

    # 2-panel plot (heldout_80 + pilot_20) succeeds cleanly
    fig_path = tmp_path / "two_panel.png"
    analyze.plot_band_accuracy(metrics, fig_path)
    assert fig_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_help_and_subcommand_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    for sub in ("probe", "extract", "snippets", "baselines", "label", "judge", "analyze"):
        assert sub in out

    for sub in ("probe", "label", "judge", "analyze"):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([sub, "--help"])
        assert exc_info.value.code == 0
        sub_out = capsys.readouterr().out
        assert len(sub_out) > 20
        if sub == "judge":
            assert "--pilot" in sub_out and "--max-usd" in sub_out
            assert "--bands" in sub_out and "--judge-model" in sub_out


def test_reporting_floors_below_045_third_judge_limitations_and_context_ablation(
    tmp_path: Path,
) -> None:
    """Verify reporting locks: <0.45 flags, third judge label, Limitations, P1-P5, exploratory ablation."""
    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 3, 11: 3, 18: 2, 24: 2},
        pairs_per_cell=6,
        include_p90=True,
        include_neighbour=True,
    )
    # Mark half pilot, half heldout cleanly
    for idx, p in enumerate(poles_doc["poles"]):
        p["in_pilot"] = idx < 8
    labels = [
        {
            "pole_id": p["pole_id"],
            "layer": p["layer"],
            "own_label": f"Label {p['pole_id']}",
            "wrong_label": "Wrong label",
            "wrong_donor_pole_id": "L24_N0100_pos",
            "wrong_donor_layer": 24,
            "wrong_embed_cosine": 0.35,
            "token_only_label": 'Tokens such as: " a", " b", " c"',
            "top_3_tokens": p["top_3_tokens"],
        }
        for p in poles_doc["poles"]
    ]
    judgements: list[dict[str, Any]] = []
    unfilt_snips: list[dict[str, Any]] = []
    unfilt_judgements: list[dict[str, Any]] = []
    for s in snippets:
        pid = s["pole_id"]
        band = s["band"]
        dt = s["distractor_type"]
        pos_slot = s["pos_slot"]
        other_slot = "B" if pos_slot == "A" else "A"
        for cond in CONDITIONS:
            # Make wrong score 0.333 (< 0.45!) and no_label 0.417 (< 0.45!) on p75_90
            if cond == "own":
                corr = 1.0 if (band in ("p97_99.9", "p99.9_plus") and s["pair_idx"] < 5) else (1.0 if s["pair_idx"] < 3 else 0.0)
            elif cond == "token_only":
                corr = 1.0 if (band == "p99.9_plus" and s["pair_idx"] < 5) else (1.0 if s["pair_idx"] < 3 else 0.0)
            elif cond == "no_label":
                corr = 1.0 if s["pair_idx"] < 2 else 0.0  # 0.333 < 0.45!
            else:
                corr = 1.0 if s["pair_idx"] < 2 else 0.0  # 0.333 < 0.45!
            pred = pos_slot if corr == 1.0 else other_slot
            for jm in ("gpt-6.1-terra@none", "gpt-6-luna@none"):
                judgements.append(
                    {
                        "pair_id": s["pair_id"],
                        "pole_id": pid,
                        "layer": s["layer"],
                        "band": band,
                        "distractor_type": dt,
                        "condition": cond,
                        "judge_model": jm,
                        "judge_backend": "openai",
                        "order": "fwd",
                        "pos_slot": pos_slot,
                        "predicted_slot": pred,
                        "outcome": "scored",
                        "correct": corr,
                        "raw_response": pred,
                        "prompt_tokens": 370,
                        "completion_tokens": 4,
                        "reasoning_tokens": 0,
                        "visible_output_tokens": 4,
                        "cost_usd": 0.00039,
                    }
                )
            if dt == "dead_zone":
                for order in ("fwd", "rev"):
                    judgements.append(
                        {
                            "pair_id": s["pair_id"],
                            "pole_id": pid,
                            "layer": s["layer"],
                            "band": band,
                            "distractor_type": dt,
                            "condition": cond,
                            "judge_model": "gpt-6.2-sol@low",
                            "judge_backend": "openai",
                            "order": order,
                            "pos_slot": pos_slot if order == "fwd" else other_slot,
                            "predicted_slot": (pos_slot if corr == 1.0 else other_slot) if order == "fwd" else (other_slot if corr == 1.0 else pos_slot),
                            "outcome": "scored",
                            "correct": corr,
                            "raw_response": "A",
                            "prompt_tokens": 370,
                            "completion_tokens": 4,
                            "reasoning_tokens": 0,
                            "visible_output_tokens": 4,
                            "cost_usd": 0.00078,
                        }
                    )
        if dt == "dead_zone" and band in ("p75_90", "p90_97"):
            us = dict(s)
            us["pair_id"] = f"unfilt_{s['pair_id']}"
            us["pos"] = dict(s["pos"], other_has_ge_p90=True, window_has_ge_p97=True)
            unfilt_snips.append(us)
            for cond in CONDITIONS:
                corr_u = 1.0 if (cond == "own" and s["pair_idx"] < 5) else (1.0 if s["pair_idx"] < 2 else 0.0)
                unfilt_judgements.append(
                    {
                        "pair_id": us["pair_id"],
                        "pole_id": pid,
                        "layer": s["layer"],
                        "band": band,
                        "distractor_type": "dead_zone",
                        "condition": cond,
                        "judge_model": "gpt-6.1-terra@none",
                        "judge_backend": "openai",
                        "order": "fwd",
                        "pos_slot": pos_slot,
                        "predicted_slot": pos_slot if corr_u == 1.0 else other_slot,
                        "outcome": "scored",
                        "correct": corr_u,
                        "raw_response": "A",
                        "prompt_tokens": 370,
                        "completion_tokens": 4,
                        "reasoning_tokens": 0,
                        "visible_output_tokens": 4,
                        "cost_usd": 0.00039,
                    }
                )

    survival_doc = {
        "weak_band_survival": {
            "p75_90": {
                "all_100": {
                    "pooled_total_candidates": 1000,
                    "pooled_surviving_clean": 31,
                    "pooled_survival_rate": 0.031,
                    "per_pole_survival_rate": {"mean": 0.031, "median": 0.025, "p10": 0.008, "p90": 0.061, "min": 0.004, "max": 0.146},
                }
            },
            "p90_97": {
                "all_100": {
                    "pooled_total_candidates": 500,
                    "pooled_surviving_clean": 6,
                    "pooled_survival_rate": 0.012,
                    "per_pole_survival_rate": {"mean": 0.012, "median": 0.012, "p10": 0.005, "p90": 0.019, "min": 0.003, "max": 0.029},
                }
            },
        }
    }

    poles_p = tmp_path / "poles.json"
    snips_p = tmp_path / "snippets.jsonl"
    labels_p = tmp_path / "labels.jsonl"
    judge_p = tmp_path / "judgements.jsonl"
    base_p = tmp_path / "baselines.jsonl"
    surv_p = tmp_path / "weak_band_survival.json"
    usnips_p = tmp_path / "snippets_unfiltered.jsonl"
    ujudge_p = tmp_path / "judgements_unfiltered.jsonl"
    fig_p = tmp_path / "figures" / "w_band_accuracy.png"
    ctx_fig_p = tmp_path / "figures" / "context_ablation.png"
    tables_p = tmp_path / "results" / "summary_tables.md"
    ws_p = tmp_path / "audit" / "worksheet.md"

    poles_p.write_text(json.dumps(poles_doc), encoding="utf-8")
    snips_p.write_text("\n".join(json.dumps(r) for r in snippets) + "\n", encoding="utf-8")
    labels_p.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    judge_p.write_text("\n".join(json.dumps(r) for r in judgements) + "\n", encoding="utf-8")
    base_p.write_text("", encoding="utf-8")
    surv_p.write_text(json.dumps(survival_doc), encoding="utf-8")
    usnips_p.write_text("\n".join(json.dumps(r) for r in unfilt_snips) + "\n", encoding="utf-8")
    ujudge_p.write_text("\n".join(json.dumps(r) for r in unfilt_judgements) + "\n", encoding="utf-8")

    metrics = analyze.run_analysis(
        poles_path=poles_p,
        snippets_path=snips_p,
        labels_path=labels_p,
        judgements_path=judge_p,
        baselines_path=base_p,
        structural_path=tmp_path / "none.json",
        figure_path=fig_p,
        tables_path=tables_p,
        worksheet_path=ws_p,
        metrics_json_path=tmp_path / "metrics.json",
        primary_judge="gpt-6.1-terra@none",
        n_resamples=100,
        seed=42,
        weak_band_survival_path=surv_p,
        unfiltered_judgements_path=ujudge_p,
        unfiltered_snippets_path=usnips_p,
        context_figure_path=ctx_fig_p,
    )

    md = tables_p.read_text(encoding="utf-8")
    assert "Floors Below 0.45 Flagged" in md
    assert "[<0.45]" in md
    assert len(metrics["floors_below_045"]) > 0
    assert "third judge" in md.lower()
    assert "reasoning check" not in md.lower()
    assert "Nearest Same-Layer Neuron (Descriptive, Tier 0 Only)" in md
    assert "## 8. Limitations" in md
    assert "3.10%" in md and "1.20%" in md
    assert "EXPLORATORY CONTEXT ABLATION: Unfiltered Weak-Band Positive Windows" in md
    assert "9.1 Held-Out 80" in md and "9.2 Pilot 20" in md
    assert ctx_fig_p.is_file() and ctx_fig_p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    pv = metrics["prediction_checks"]
    assert pv["status"] == "EVALUATED"
    assert pv["n_total"] >= 18
    assert pv["n_held"] + pv["n_failed"] == pv["n_total"]
    for row in pv["sub_predictions"]:
        assert row["outcome"] in ("HELD", "FAILED")
    assert "Exact Paired Differences on Identical Pairs" in md
    assert "Within-Pole Matched Comparison" in md
    assert "Per-Pole Distribution of `own` Accuracy at `p99.9+`" in md
    assert "[HELDOUT_80]" in ws_p.read_text(encoding="utf-8")
    assert "[PILOT_20]" in ws_p.read_text(encoding="utf-8")


def test_examples_as_label_runner_prompt_quote_cap_isolation_and_section10(
    tmp_path: Path,
) -> None:
    """Verify Step (2) examples_as_label prompt, pre-run quote abort, file guard, and Section 10 math."""
    import asyncio
    from w_audit import examples_as_label as eal

    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 2, 11: 2},
        pairs_per_cell=6,
        include_p90=False,
        include_neighbour=False,
    )
    for idx, p in enumerate(poles_doc["poles"]):
        p["in_pilot"] = idx < 4

    examples_by_pole: dict[str, list[dict[str, Any]]] = {}
    for p in poles_doc["poles"]:
        pid = p["pole_id"]
        examples_by_pole[pid] = [
            {
                "pole_id": pid,
                "example_idx": i,
                "act_dir": 3.0 - 0.05 * i,
                "marked_text": f"Example {i} context <<tok{i}>> suffix for {pid}.",
            }
            for i in range(20)
        ]

    plan = eal.build_examples_task_plan(snippets=snippets, examples_by_pole=examples_by_pole)
    # 8 poles x 2 bands (p97_99.9, p99.9_plus) x 6 pairs = 96 tasks
    assert len(plan) == 96
    for task, msgs in plan:
        assert task.condition == "examples_as_label"
        assert task.band in ("p97_99.9", "p99.9_plus")
        assert task.distractor_type == "dead_zone"
        assert "strongly" not in msgs[0]["content"].lower()
        assert "strongly" not in msgs[1]["content"].lower()
        assert "[1] Example 0 context" in msgs[1]["content"]
        assert "[20] Example 19 context" in msgs[1]["content"]

    quote = eal.quote_examples_plan_cost(plan, api_model="gpt-6.1-terra")
    assert quote["n_calls"] == 96
    assert quote["est_total_usd_0pct_cache"] > 0.0

    # Refuses to write into frozen judgements.jsonl
    with pytest.raises(ValueError, match="judgements.jsonl"):
        asyncio.run(
            eal.run_examples_tasks_async(
                plan=plan[:4],
                output_path=tmp_path / "judgements.jsonl",
                max_usd=4.0,
            )
        )

    # Aborts before any live call if pre-run conservative quote exceeds max_usd
    with pytest.raises(judge.SpendLimitExceededError):
        asyncio.run(
            eal.run_examples_tasks_async(
                plan=plan,
                output_path=tmp_path / "judgements_examples_as_label_exploratory.jsonl",
                max_usd=0.0001,
            )
        )

    # Mock OpenAI execution into strictly separate output file + resumability
    class _FakeExCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            return _make_chat_response(
                content="A",
                top_lps=[_make_top_lp("A", -0.05), _make_top_lp("B", -3.0)],
                prompt_tokens=1800,
                completion_tokens=4,
            )

    class _FakeExClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=_FakeExCompletions())

    fake_client = _FakeExClient()
    out_p = tmp_path / "judgements_examples_as_label_exploratory.jsonl"
    res1 = asyncio.run(
        eal.run_examples_tasks_async(
            plan=plan[:12],
            output_path=out_p,
            max_usd=4.0,
            concurrency=4,
            target_rpm=10_000.0,
            openai_client=fake_client,
        )
    )
    assert res1["completed_this_run"] == 12
    assert res1["skipped_existing"] == 0
    assert fake_client.chat.completions.calls == 12

    # Resume skips all 12 already written rows
    res2 = asyncio.run(
        eal.run_examples_tasks_async(
            plan=plan[:12],
            output_path=out_p,
            max_usd=4.0,
            concurrency=4,
            target_rpm=10_000.0,
            openai_client=fake_client,
        )
    )
    assert res2["completed_this_run"] == 0
    assert res2["skipped_existing"] == 12
    assert fake_client.chat.completions.calls == 12

    # Verify Section 10 renders paired examples - own & per-layer breakdown cleanly
    ex_rows = labeler.load_jsonl(out_p)
    clean_rows = []
    for task, _ in plan[:12]:
        for cond in ("own", "token_only", "no_label", "wrong"):
            clean_rows.append(
                {
                    "pair_id": task.pair_id,
                    "pole_id": task.pole_id,
                    "layer": task.layer,
                    "band": task.band,
                    "distractor_type": "dead_zone",
                    "condition": cond,
                    "judge_model": "gpt-6.1-terra@none",
                    "judge_backend": "openai",
                    "order": "fwd",
                    "pos_slot": task.pos_slot,
                    "predicted_slot": "B",
                    "outcome": "scored",
                    "correct": 1.0 if task.pos_slot == "B" else 0.0,
                    "raw_response": "B",
                    "prompt_tokens": 370,
                    "completion_tokens": 4,
                    "reasoning_tokens": 0,
                    "visible_output_tokens": 4,
                    "cost_usd": 0.00039,
                }
            )
    metrics = analyze.compute_all_metrics(
        poles_data=poles_doc,
        snippets=snippets,
        judgements=clean_rows,
        baselines=[],
        primary_judge="gpt-6.1-terra@none",
        n_resamples=100,
        seed=42,
        examples_judgements=ex_rows,
    )
    assert metrics["examples_as_label"] is not None
    assert metrics["examples_as_label"]["n_calls"] == 12
    md = analyze.render_summary_tables_md(metrics)
    assert "10. EXPLORATORY ARM: 20 Top-Band Examples as Label" in md
    assert "Paired `examples - own`" in md


def test_p97_p99_subband_split_cache_offline_fallback_and_main_figure(
    tmp_path: Path,
) -> None:
    """Verify exact p99 cache threshold computation, sub-band split partition/math, offline fallback, and main figure."""
    import numpy as np

    poles_doc, _, snippets = _build_synthetic_poles_and_snippets(
        n_neurons_per_layer={4: 2, 11: 2},
        pairs_per_cell=6,
    )
    # Split neurons cleanly into pilot (first 2 poles / 1 neuron) and heldout (remaining 6 poles / 3 neurons)
    for idx, p in enumerate(poles_doc["poles"]):
        p["in_pilot"] = idx < 2

    # Build a synthetic cache matching extract.py's non_bos_mask + quantile logic
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    neurons = sorted({(int(p["layer"]), int(p["neuron"])) for p in poles_doc["poles"]})
    col_to_neuron = [[L, n] for L, n in neurons]
    n_tok = 2000
    rng = np.random.default_rng(123)
    acts = rng.normal(loc=0.0, scale=1.0, size=(n_tok, len(neurons))).astype(np.float16)
    pos_in_chunk = np.arange(n_tok, dtype=np.int32) % 200
    token_ids = np.arange(n_tok, dtype=np.int32) % 500 + 10
    token_ids[5] = 151643  # special token
    non_bos = (pos_in_chunk > 0) & (~np.isin(token_ids, [151643, 151645]))

    # Sync poles' band_edges_dir to exact quantiles of acts so max_abs_err == 0.0
    col_map = {(L, n): i for i, (L, n) in enumerate(neurons)}
    p99_expected: dict[str, tuple[float, float, float]] = {}
    for p in poles_doc["poles"]:
        col = col_map[(int(p["layer"]), int(p["neuron"]))]
        d_valid = float(p["sign"]) * acts[non_bos, col].astype(np.float32)
        q75, q90, q97, q99, q999 = (
            float(x) for x in np.quantile(d_valid, [0.75, 0.90, 0.97, 0.99, 0.999])
        )
        p["band_edges_dir"] = {"p75": q75, "p90": q90, "p97": q97, "p99.9": q999}
        p99_expected[p["pole_id"]] = (q97, q99, q999)

    np.save(cache_dir / "acts.npy", acts)
    np.save(cache_dir / "pos_in_chunk.npy", pos_in_chunk)
    np.save(cache_dir / "token_ids.npy", token_ids)
    (cache_dir / "meta.json").write_text(
        json.dumps({"special_token_ids": [151643, 151645], "col_to_neuron": col_to_neuron}),
        encoding="utf-8",
    )

    # Set dead_zone p97_99.9 snippets: pair_idx 0..3 -> p97_99 (< p99), pair_idx 4..5 -> p99_99.9 (>= p99)
    judgements: list[dict[str, Any]] = []
    examples_rows: list[dict[str, Any]] = []
    for s in snippets:
        if s["distractor_type"] == "dead_zone" and s["band"] == "p97_99.9":
            q97, q99, q999 = p99_expected[s["pole_id"]]
            if int(s["pair_idx"]) < 4:
                s["pos"]["act_dir"] = 0.5 * (q97 + q99)
            else:
                s["pos"]["act_dir"] = 0.5 * (q99 + q999)
        if s["distractor_type"] != "dead_zone":
            continue
        is_hi = s["band"] == "p97_99.9" and int(s["pair_idx"]) >= 4
        for jm in ("gpt-6.1-terra@none", "gpt-6-luna@none"):
            for cond in CONDITIONS:
                if cond == "own":
                    corr = 1.0 if (is_hi or s["band"] == "p99.9_plus") else (1.0 if int(s["pair_idx"]) < 2 else 0.0)
                elif cond == "token_only":
                    corr = 1.0 if (is_hi and int(s["pair_idx"]) == 4) or (s["band"] == "p99.9_plus") else 0.0
                else:
                    corr = 0.5 if int(s["pair_idx"]) % 2 == 0 else 0.0
                    corr = 1.0 if int(s["pair_idx"]) % 2 == 0 else 0.0
                pred = s["pos_slot"] if corr == 1.0 else ("B" if s["pos_slot"] == "A" else "A")
                judgements.append(
                    {
                        "pair_id": s["pair_id"],
                        "pole_id": s["pole_id"],
                        "layer": s["layer"],
                        "band": s["band"],
                        "distractor_type": "dead_zone",
                        "condition": cond,
                        "judge_model": jm,
                        "judge_backend": "openai",
                        "order": "fwd",
                        "pos_slot": s["pos_slot"],
                        "predicted_slot": pred,
                        "outcome": "scored",
                        "correct": corr,
                        "raw_response": pred,
                        "prompt_tokens": 370,
                        "completion_tokens": 4,
                        "reasoning_tokens": 0,
                        "visible_output_tokens": 4,
                        "cost_usd": 0.00039,
                    }
                )
        if s["band"] in ("p97_99.9", "p99.9_plus"):
            # examples_as_label scores 1.0 everywhere so paired examples - own > 0 on p97_99 and == 0 on p99_99.9
            examples_rows.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": s["pole_id"],
                    "layer": s["layer"],
                    "band": s["band"],
                    "distractor_type": "dead_zone",
                    "condition": "examples_as_label",
                    "judge_model": "gpt-6.1-terra@none",
                    "judge_backend": "openai",
                    "order": "fwd",
                    "pos_slot": s["pos_slot"],
                    "predicted_slot": s["pos_slot"],
                    "outcome": "scored",
                    "correct": 1.0,
                    "raw_response": s["pos_slot"],
                    "prompt_tokens": 1800,
                    "completion_tokens": 4,
                    "reasoning_tokens": 0,
                    "visible_output_tokens": 4,
                    "cost_usd": 0.00182,
                }
            )

    # Write inputs and run end-to-end via run_analysis with cache_dir
    poles_p = tmp_path / "poles.json"
    snips_p = tmp_path / "snippets.jsonl"
    labels_p = tmp_path / "labels.jsonl"
    judg_p = tmp_path / "judgements.jsonl"
    base_p = tmp_path / "baselines.jsonl"
    ex_p = tmp_path / "examples.jsonl"
    subband_p = tmp_path / "p97_p99_subband_split.json"
    main_fig_p = tmp_path / "main_band_accuracy.png"

    poles_p.write_text(json.dumps(poles_doc), encoding="utf-8")
    snips_p.write_text("\n".join(json.dumps(s) for s in snippets) + "\n", encoding="utf-8")
    labels_p.write_text(
        "\n".join(
            json.dumps(
                {
                    "pole_id": p["pole_id"],
                    "layer": p["layer"],
                    "own_label": f"Label {p['pole_id']}",
                    "wrong_label": "Wrong",
                    "wrong_donor_pole_id": "L11_N0001_pos",
                    "wrong_donor_layer": 11,
                    "wrong_embed_cosine": 0.3,
                    "token_only_label": 'Tokens such as: "a", "b", "c"',
                    "top_3_tokens": p["top_3_tokens"],
                }
            )
            for p in poles_doc["poles"]
        )
        + "\n",
        encoding="utf-8",
    )
    judg_p.write_text("\n".join(json.dumps(r) for r in judgements) + "\n", encoding="utf-8")
    base_p.write_text("", encoding="utf-8")
    ex_p.write_text("\n".join(json.dumps(r) for r in examples_rows) + "\n", encoding="utf-8")

    metrics = analyze.run_analysis(
        poles_path=poles_p,
        snippets_path=snips_p,
        labels_path=labels_p,
        judgements_path=judg_p,
        baselines_path=base_p,
        structural_path=tmp_path / "no_struct.json",
        figure_path=tmp_path / "w_band.png",
        main_figure_path=main_fig_p,
        tables_path=tmp_path / "summary.md",
        worksheet_path=tmp_path / "worksheet.md",
        metrics_json_path=tmp_path / "metrics.json",
        primary_judge="gpt-6.1-terra@none",
        n_resamples=100,
        seed=42,
        examples_judgements_path=ex_p,
        cache_dir=cache_dir,
        subband_json_path=subband_p,
    )

    sb = metrics["p97_p99_subband_split"]
    assert sb is not None
    assert sb["verification"]["all_strictly_ordered"] is True
    assert sb["verification"]["max_abs_err_vs_poles_json"] == pytest.approx(0.0, abs=1e-6)
    assert sb["verification"]["out_of_range_count"] == 0
    # Heldout: 6 poles * 6 pairs = 36 -> 24 in p97_99 + 12 in p99_99.9
    h_pc = sb["verification"]["partition_checks"]["heldout_80"]
    assert (h_pc["n_p97_99"], h_pc["n_p99_99.9"], h_pc["n_total"]) == (24, 12, 36)
    assert h_pc["exact_sum_match"] is True
    # Pilot: 2 poles * 6 pairs = 12 -> 8 in p97_99 + 4 in p99_99.9
    p_pc = sb["verification"]["partition_checks"]["pilot_20"]
    assert (p_pc["n_p97_99"], p_pc["n_p99_99.9"], p_pc["n_total"]) == (8, 4, 12)

    h_terra = sb["cohorts"]["heldout_80"]["gpt-6.1-terra@none"]
    assert h_terra["p97_99"]["conditions"]["own"]["mean"] == pytest.approx(0.5)
    assert h_terra["p97_99"]["conditions"]["examples_as_label"]["mean"] == pytest.approx(1.0)
    assert h_terra["p97_99"]["paired_examples_minus_own"]["mean"] == pytest.approx(0.5)
    assert h_terra["p99_99.9"]["conditions"]["own"]["mean"] == pytest.approx(1.0)
    assert h_terra["p99_99.9"]["conditions"]["token_only"]["mean"] == pytest.approx(0.5)
    assert h_terra["p99_99.9"]["paired_own_minus_token_only"]["mean"] == pytest.approx(0.5)
    assert h_terra["p99_99.9"]["paired_examples_minus_own"]["mean"] == pytest.approx(0.0)
    assert "gpt-6-luna@none" in sb["cohorts"]["heldout_80"]

    # Offline fallback without acts.npy reads saved p97_p99_subband_split.json identically
    assert subband_p.is_file()
    offline_thr = analyze.load_or_compute_p99_thresholds(
        poles_doc["poles"], cache_dir=tmp_path / "nonexistent_cache", subband_json_path=subband_p
    )
    assert offline_thr is not None
    assert offline_thr["n_poles"] == len(poles_doc["poles"])
    for pid, (q97, q99, q999) in p99_expected.items():
        assert offline_thr["pole_thresholds"][pid]["p99"] == pytest.approx(q99)

    # Main figure PNG exists and is valid PNG
    assert main_fig_p.is_file()
    assert main_fig_p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert main_fig_p.stat().st_size > 15_000

    # Summary markdown includes Section 11 cleanly
    md_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "11. Zero-Cost Sub-Band Split of `p97–99.9`" in md_text
    assert "`p97–99`" in md_text and "`p99–99.9`" in md_text
