"""Mock-API CPU pytest suite for exploratory Qwen3-4B (Ulto85) judge & analysis pipeline."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qwen4b_eval import analyze_4b, judge_4b
from w_audit.config import JUDGE_SYSTEM_PROMPT, NO_LABEL_DESCRIPTION
from w_audit.judge import SpendLimitExceededError


def _make_fake_chat_response(
    content: str = "A",
    lp_a: float = -0.05,
    lp_b: float = -3.0,
    prompt_tokens: int = 410,
    completion_tokens: int = 4,
) -> Any:
    top_lps = [
        SimpleNamespace(token="A", logprob=lp_a),
        SimpleNamespace(token="B", logprob=lp_b),
    ]
    parsed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(token=content, logprob=lp_a, top_logprobs=top_lps)]
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    return SimpleNamespace(parse=lambda: parsed, headers={})


class _FakeAsyncOpenAI:
    def __init__(self, answer_fn=None) -> None:
        self.calls = 0
        self.seen_models: list[str] = []
        self.seen_messages: list[list[dict[str, str]]] = []
        self._answer_fn = answer_fn
        parent = self

        class _WithRaw:
            async def create(self, **kwargs: Any) -> Any:
                parent.calls += 1
                parent.seen_models.append(str(kwargs.get("model")))
                msgs = kwargs.get("messages", [])
                parent.seen_messages.append(msgs)
                ans = "A"
                lp_a, lp_b = -0.05, -3.0
                if parent._answer_fn is not None:
                    ans, lp_a, lp_b = parent._answer_fn(kwargs)
                return _make_fake_chat_response(content=ans, lp_a=lp_a, lp_b=lp_b)

        class _Completions:
            with_raw_response = _WithRaw()

        self.chat = SimpleNamespace(completions=_Completions())


def _build_synthetic_4b_fixture(tmp_path: Path) -> dict[str, Any]:
    """Create synthetic 10-neuron (6 unipolar + 4 bipolar = 14 active poles) fixture incl. info/PCR/insurance."""
    neurons_spec = [
        # (k, layer, neuron, short, active_signs, sim_cal, sim_act, s20, corr, mono, in_50, in_opus)
        (26, 0, 9209, "info", [1], 0.86, 0.96, 0.88, 0.97, 5, True, True),
        (23, 0, 8146, "PCR", [1], -0.57, 0.92, 0.14, 0.94, 4, True, True),
        (236, 32, 4362, "insurance", [1, -1], -0.01, 0.04, 0.19, 0.10, 2, True, True),
        (10, 4, 100, "n10", [1], 0.60, 0.72, 0.55, 0.75, 4, True, True),
        (11, 8, 200, "n11", [-1], -1.80, 0.20, 0.10, 0.30, 2, False, True),  # outlier < -0.85!
        (12, 12, 300, "n12", [1, -1], 0.42, 0.65, 0.40, 0.62, 3, True, False),
        (13, 16, 400, "n13", [1], 0.10, 0.35, 0.25, 0.40, 3, False, False),
        (14, 20, 500, "n14", [1, -1], 0.75, 0.85, 0.70, 0.82, 4, False, False),
        (15, 24, 600, "n15", [-1], -0.20, 0.52, 0.22, 0.45, 2, False, False),
        (16, 35, 700, "n16", [1, -1], 0.30, 0.58, 0.35, 0.51, 3, True, False),
    ]

    active_poles: list[dict[str, Any]] = []
    explanations: dict[str, Any] = {}
    scores_list: list[dict[str, Any]] = []
    opus_list: list[dict[str, Any]] = []
    snippets: list[dict[str, Any]] = []

    for k, ly, nn, short, signs, sc_rec, sc_act, s20, corr, mono, in_50, in_opus in neurons_spec:
        explanations[str(k)] = {
            "layer": ly,
            "neuron": nn,
            "M": 2.0,
            "expl": {
                "label": f"Summary concept for {short} L{ly}",
                "positive_pole": f"Positive feature {short} domain alpha_{k}" if 1 in signs else "Negligible.",
                "negative_pole": f"Negative feature {short} domain beta_{k}" if -1 in signs else "Negligible.",
                "rubric": f"Score 8 on {short} tokens.",
            },
        }
        sc_row = {
            "k": k,
            "layer": ly,
            "neuron": nn,
            "label": explanations[str(k)]["expl"]["label"],
            "sim_cal_rec": sc_rec,
            "sim_cal_rec_active": sc_act,
            "strong20_rec": s20,
            "corr": corr,
            "mono": mono,
        }
        scores_list.append(sc_row)
        if in_opus:
            opus_row = dict(sc_row)
            opus_row["sim_cal_rec"] = sc_rec + 0.04
            opus_row["sim_cal_rec_active"] = min(1.0, sc_act - 0.02)
            opus_list.append(opus_row)

        for s_idx, sign in enumerate(signs):
            pole_tag = "pos" if sign > 0 else "neg"
            pid = f"L{ly:02d}_N{nn:04d}_{pole_tag}"
            p999_mag = 5.0 if s_idx == 0 else 3.0  # first sign is dominant by |p99.9|
            active_poles.append(
                {
                    "pole_id": pid,
                    "k": k,
                    "layer": ly,
                    "neuron": nn,
                    "sign": sign,
                    "pole_text": (
                        explanations[str(k)]["expl"]["positive_pole"]
                        if sign > 0
                        else explanations[str(k)]["expl"]["negative_pole"]
                    ),
                    "in_full_expl_50": in_50,
                    "is_dominant": s_idx == 0,
                    "band_edges_dir": {"p90": 1.0, "p97": 2.0, "p99": 3.0, "p99.9": p999_mag},
                }
            )
            # 20 p99.9_plus pairs: 0..5 exact 3/3 A/B, 0..9 exact 5/5 A/B, 10..19 exact 5/5 A/B
            slots_20 = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 2
            for pair_idx, slot in enumerate(slots_20):
                snippets.append(
                    {
                        "pair_id": f"{pid}__p99.9_plus__dead_zone__{pair_idx}",
                        "pole_id": pid,
                        "k": k,
                        "layer": ly,
                        "neuron": nn,
                        "sign": sign,
                        "band": "p99.9_plus",
                        "distractor_type": "dead_zone",
                        "pair_idx": pair_idx,
                        "in_floor_6": pair_idx < 6,
                        "pos_slot": slot,
                        "pos": {"marked_text": f"Firing context for <<{short}_{pole_tag}>> p999 {pair_idx}"},
                        "dist": {"marked_text": f"Dead zone background <<other>> {pair_idx}"},
                    }
                )
            # 10 p97_99 pairs: exact 5/5 A/B
            slots_10 = ["A", "B"] * 5
            for pair_idx, slot in enumerate(slots_10):
                snippets.append(
                    {
                        "pair_id": f"{pid}__p97_99__dead_zone__{pair_idx}",
                        "pole_id": pid,
                        "k": k,
                        "layer": ly,
                        "neuron": nn,
                        "sign": sign,
                        "band": "p97_99",
                        "distractor_type": "dead_zone",
                        "pair_idx": pair_idx,
                        "in_floor_6": False,
                        "pos_slot": slot,
                        "pos": {"marked_text": f"Mid band <<{short}_{pole_tag}>> p97_99 {pair_idx}"},
                        "dist": {"marked_text": f"Dead zone background <<other>> {pair_idx}"},
                    }
                )

    paths = {
        "active_poles": tmp_path / "active_poles.json",
        "explanations": tmp_path / "e3_explanations.json",
        "scores": tmp_path / "e3_scores.json",
        "opus_scores": tmp_path / "e3_scores_opus.json",
        "snippets": tmp_path / "snippets_4b.jsonl",
        "wrong_donors": tmp_path / "wrong_donors_4b.json",
        "judgements": tmp_path / "judgements_4b.jsonl",
        "metrics": tmp_path / "metrics_4b.json",
        "summary": tmp_path / "summary_4b.md",
        "figure": tmp_path / "qwen4b_detection_vs_causal.png",
    }
    paths["active_poles"].write_text(json.dumps({"poles": active_poles}, indent=2), encoding="utf-8")
    paths["explanations"].write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    paths["scores"].write_text(json.dumps({"neurons": scores_list}, indent=2), encoding="utf-8")
    paths["opus_scores"].write_text(json.dumps({"neurons": opus_list}, indent=2), encoding="utf-8")
    paths["snippets"].write_text(
        "\n".join(json.dumps(s) for s in snippets) + "\n", encoding="utf-8"
    )
    return {
        "paths": paths,
        "active_poles": active_poles,
        "explanations": explanations,
        "snippets": snippets,
        "scores": {"neurons": scores_list},
        "opus_scores": {"neurons": opus_list},
    }


def test_judge_4b_isolation_wrong_donors_plan_counts_resume_and_spend_cap(tmp_path: Path) -> None:
    fx = _build_synthetic_4b_fixture(tmp_path)
    paths = fx["paths"]

    # 1. Protected 1.7B filenames must be strictly refused
    with pytest.raises(ValueError, match="Refusing"):
        asyncio.run(
            judge_4b.run_4b_judge_async(
                plan=[], output_path=tmp_path / "judgements.jsonl", max_usd=5.0
            )
        )

    # 2. Cross-layer wrong-donor matching with orthogonal mock embeddings (cos <= 0.65, zero same-layer)
    def _mock_embed(texts: list[str]) -> np.ndarray:
        n = len(texts)
        mat = np.eye(n, dtype=np.float64)
        return mat

    wrong_map = asyncio.run(
        judge_4b.build_or_load_wrong_donors_4b_async(
            active_poles=fx["active_poles"],
            explanations_by_k=fx["explanations"],
            output_path=paths["wrong_donors"],
            embed_fn=_mock_embed,
        )
    )
    assert len(wrong_map) == 14
    poles_by_id = {p["pole_id"]: p for p in fx["active_poles"]}
    for pid, donor in wrong_map.items():
        assert int(donor["wrong_donor_layer"]) != int(poles_by_id[pid]["layer"])
        assert float(donor["wrong_embed_cosine"]) <= 0.65

    # 3. Exact task counts across all 5 sweeps:
    # 14 poles: p999_own_terra=280, p999_own_luna=280, p999_floors_terra=14*6*2=168,
    # p97_99_own_terra=14*10=140, p999_full_expl_terra=9 active poles in_full_expl_50 * 20 = 180
    plan = judge_4b.build_4b_task_plan(
        active_poles=fx["active_poles"],
        snippets=fx["snippets"],
        wrong_donors=wrong_map,
        explanations_by_k=fx["explanations"],
        sweeps=judge_4b.ALL_SWEEPS,
    )
    n_fe_poles = sum(1 for p in fx["active_poles"] if p["in_full_expl_50"])
    assert n_fe_poles == 9
    expected_total = 14 * 20 + 14 * 20 + 14 * 6 * 2 + 14 * 10 + n_fe_poles * 20
    assert len(plan) == expected_total

    # Verify unified prompt (zero "strongly"), exact no_label, and exact full_explanation format
    for task, msgs, _meta in plan:
        assert msgs[0]["content"] == JUDGE_SYSTEM_PROMPT
        assert "strongly" not in msgs[0]["content"].lower()
        assert "strongly" not in msgs[1]["content"].lower()
        if task.condition == "no_label":
            assert NO_LABEL_DESCRIPTION in msgs[1]["content"]
        elif task.condition == "full_explanation":
            assert "Summary:" in msgs[1]["content"]
            assert "Positive pole:" in msgs[1]["content"]
            assert "Negative pole:" in msgs[1]["content"]
            assert "Rubric:" in msgs[1]["content"]

    # 4. Run tiny slice + verify resume skips 100% + spend cap aborts
    fake_client = _FakeAsyncOpenAI()
    res1 = asyncio.run(
        judge_4b.run_4b_judge_async(
            plan=plan[:20],
            output_path=paths["judgements"],
            max_usd=5.0,
            concurrency=4,
            target_rpm=10_000.0,
            openai_client=fake_client,
            progress_every=0,
        )
    )
    assert res1["completed_this_run"] == 20
    assert res1["skipped_existing"] == 0
    assert fake_client.calls == 20

    # Resume skips all 20
    res2 = asyncio.run(
        judge_4b.run_4b_judge_async(
            plan=plan[:20],
            output_path=paths["judgements"],
            max_usd=5.0,
            concurrency=4,
            target_rpm=10_000.0,
            openai_client=fake_client,
            progress_every=0,
        )
    )
    assert res2["completed_this_run"] == 0
    assert res2["skipped_existing"] == 20
    assert fake_client.calls == 20

    # Spend cap abort when prior_usd already exceeds cap
    with pytest.raises(SpendLimitExceededError):
        asyncio.run(
            judge_4b.run_4b_judge_async(
                plan=plan[20:25],
                output_path=paths["judgements"],
                max_usd=0.0001,
                concurrency=1,
                target_rpm=10_000.0,
                openai_client=fake_client,
                progress_every=0,
            )
        )


def test_analyze_4b_full_metrics_reliability_quadrants_full_expl_named_and_two_panel_figure(
    tmp_path: Path,
) -> None:
    fx = _build_synthetic_4b_fixture(tmp_path)
    paths = fx["paths"]

    # Generate deterministic synthetic judgements covering all 5 sweeps cleanly
    target_acc_by_k = {
        26: 1.00,  # info
        23: 0.90,  # PCR
        236: 0.80,  # insurance (bipolar: pos=0.90, neg=0.70 -> mean=0.80, dom=0.90)
        10: 0.85,
        11: 0.40,  # outlier y=-1.80
        12: 0.75,
        13: 0.55,
        14: 0.90,
        15: 0.50,
        16: 0.70,
    }
    judgements: list[dict[str, Any]] = []
    poles_by_id = {p["pole_id"]: p for p in fx["active_poles"]}

    for s in fx["snippets"]:
        pid = s["pole_id"]
        pole = poles_by_id[pid]
        k = int(pole["k"])
        pair_idx = int(s["pair_idx"])
        pos_slot = s["pos_slot"]
        other_slot = "B" if pos_slot == "A" else "A"

        base_p = target_acc_by_k[k]
        if k == 236:
            # Make pos (dominant) 0.90 and neg 0.70 -> neuron mean = 0.80, dominant = 0.90
            base_p = 0.90 if pole["sign"] > 0 else 0.70

        def _make_row(jm: str, cond: str, correct: float, cost: float) -> dict[str, Any]:
            pred = pos_slot if correct == 1.0 else other_slot
            return {
                "pair_id": s["pair_id"],
                "pole_id": pid,
                "k": k,
                "layer": int(pole["layer"]),
                "neuron": int(pole["neuron"]),
                "sign": int(pole["sign"]),
                "band": s["band"],
                "distractor_type": "dead_zone",
                "condition": cond,
                "judge_model": jm,
                "order": "fwd",
                "pos_slot": pos_slot,
                "predicted_slot": pred,
                "outcome": "scored",
                "correct": correct,
                "raw_response": pred,
                "prompt_tokens": 410 if cond != "full_explanation" else 820,
                "completion_tokens": 4,
                "reasoning_tokens": 0,
                "visible_output_tokens": 4,
                "cost_usd": cost,
                "pair_idx": pair_idx,
                "in_floor_6": bool(s["in_floor_6"]),
                "in_full_expl_50": bool(pole["in_full_expl_50"]),
            }

        if s["band"] == "p99.9_plus":
            # Distribute hits evenly across halves (0..9 and 10..19) so split-half reliability is high
            half_idx = pair_idx % 10
            k_hits_half = int(round(base_p * 10))
            corr_own = 1.0 if half_idx < k_hits_half else 0.0
            judgements.append(_make_row(analyze_4b.PRIMARY_JUDGE_4B, "own", corr_own, 0.00043))
            judgements.append(_make_row(analyze_4b.SECONDARY_JUDGE_4B, "own", corr_own, 0.000043))
            if s["in_floor_6"]:
                corr_floor = 1.0 if pair_idx < 3 else 0.0  # 0.50 floor
                judgements.append(_make_row(analyze_4b.PRIMARY_JUDGE_4B, "wrong", corr_floor, 0.00041))
                judgements.append(_make_row(analyze_4b.PRIMARY_JUDGE_4B, "no_label", corr_floor, 0.00037))
            if pole["in_full_expl_50"]:
                # Make full_explanation strictly higher on bipolar poles (+0.10) and equal on unipolar
                k_hits_fe = min(10, k_hits_half + (1 if k in (236, 12, 16) else 0))
                corr_fe = 1.0 if half_idx < k_hits_fe else 0.0
                judgements.append(_make_row(analyze_4b.PRIMARY_JUDGE_4B, "full_explanation", corr_fe, 0.00084))
        elif s["band"] == "p97_99":
            k_hits_mid = max(0, int(round((base_p - 0.15) * 10)))
            corr_mid = 1.0 if pair_idx < k_hits_mid else 0.0
            judgements.append(_make_row(analyze_4b.PRIMARY_JUDGE_4B, "own", corr_mid, 0.00043))

    paths["judgements"].write_text(
        "\n".join(json.dumps(r) for r in judgements) + "\n", encoding="utf-8"
    )

    metrics = analyze_4b.run_analysis_4b(
        active_poles_path=paths["active_poles"],
        scores_path=paths["scores"],
        opus_scores_path=paths["opus_scores"],
        snippets_path=paths["snippets"],
        judgements_path=paths["judgements"],
        wrong_donors_path=paths["wrong_donors"],
        metrics_path=paths["metrics"],
        summary_path=paths["summary"],
        figure_path=paths["figure"],
        n_resamples=100,
        n_random_splits=20,
        seed=42,
    )

    assert metrics["n_neurons"] == 10
    assert metrics["n_active_poles"] == 14
    assert metrics["n_unipolar_neurons"] == 6
    assert metrics["n_bipolar_neurons"] == 4

    # Named neurons check (including insurance bipolar mean=0.80 vs dominant=0.90)
    named_by_short = {r["short_name"]: r for r in metrics["named_neurons"]}
    assert set(named_by_short) == {"info", "PCR", "insurance"}
    assert named_by_short["info"]["det_p999_terra_mean"] == pytest.approx(1.0)
    assert named_by_short["PCR"]["det_p999_terra_mean"] == pytest.approx(0.90)
    assert named_by_short["insurance"]["det_p999_terra_mean"] == pytest.approx(0.80)
    assert named_by_short["insurance"]["det_p999_terra_dom"] == pytest.approx(0.90)

    # Split-half reliability (high since halves match identically)
    sh_terra = metrics["split_half_reliability"][analyze_4b.PRIMARY_JUDGE_4B]
    assert sh_terra["deterministic_halves"]["spearman_brown_r"] == pytest.approx(1.0)
    assert sh_terra["random_balanced_splits"]["spearman_brown_r"]["mean"] > 0.80

    # Causal Opus reliability on 5 overlapping neurons
    opus = metrics["causal_opus_reliability"]
    assert opus["n_overlap"] == 5
    assert opus["metrics"]["sim_cal_rec"]["spearman_rho"] == pytest.approx(1.0)
    assert opus["metrics"]["sim_cal_rec"]["median_abs_diff"] == pytest.approx(0.04)

    # 2x2 pass-rate quadrant check (sum of 4 cells == n_total == 10)
    for row in metrics["pass_matrices"]["mean_active_poles"]:
        assert row["both_pass"] + row["det_only"] + row["causal_only"] + row["both_fail"] == 10

    # 50-neuron full_explanation paired comparison: bipolar > 0, unipolar == 0
    fe = metrics["full_explanation_50"]
    assert fe["unipolar"]["paired_full_minus_pole"]["mean"] == pytest.approx(0.0)
    assert fe["bipolar"]["paired_full_minus_pole"]["mean"] > 0.05

    # Two-panel PNG figure exists, valid PNG, and clipped outlier was handled cleanly
    assert paths["figure"].is_file()
    assert paths["figure"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths["figure"].stat().st_size > 20_000

    # Summary markdown includes expected sections
    md = paths["summary"].read_text(encoding="utf-8")
    assert "info" in md and "PCR" in md and "insurance" in md
    assert "Measurement Reliability" in md
    assert "full_explanation" in md


def test_top100_sweep_strong20_floor_pass_rates_sparsity_and_all_four_figures(
    tmp_path: Path,
) -> None:
    """Verify top100 sweep planning/isolation, above-no_label pass criteria, strong20_rec, sparsity table, and all 4 PNGs."""
    fx = _build_synthetic_4b_fixture(tmp_path)
    paths = fx["paths"]
    poles_by_id = {p["pole_id"]: p for p in fx["active_poles"]}

    # Build top100 snippets (20 pairs per pole, band='top100', exact 5/5 halves)
    top100_snips: list[dict[str, Any]] = []
    for s in fx["snippets"]:
        if s["band"] != "p99.9_plus":
            continue
        t = dict(s)
        t["band"] = "top100"
        t["pair_id"] = str(s["pair_id"]).replace("__p99.9_plus__", "__top100__")
        top100_snips.append(t)

    top100_snips_p = tmp_path / "snippets_4b_top100.jsonl"
    top100_snips_p.write_text("\n".join(json.dumps(r) for r in top100_snips) + "\n", encoding="utf-8")

    # 1. Verify top100_own_terra plan count (14 poles * 20 = 280) and CLI isolation guard
    plan = judge_4b.build_4b_task_plan(
        active_poles=fx["active_poles"],
        snippets=top100_snips,
        wrong_donors={},
        explanations_by_k=fx["explanations"],
        sweeps=(judge_4b.TOP100_SWEEP,),
    )
    assert len(plan) == 14 * 20
    assert all(t.band == "top100" and t.condition == "own" for t, _, _ in plan)

    with pytest.raises(ValueError, match="Refusing to write top100_own_terra"):
        judge_4b.main(
            [
                "--sweeps",
                judge_4b.TOP100_SWEEP,
                "--snippets",
                str(top100_snips_p),
                "--output",
                str(tmp_path / "judgements_4b.jsonl"),
                "--dry-run",
            ]
        )

    # 2. Synthetic core + top100 judgements and sparsity diagnostics
    core_j: list[dict[str, Any]] = []
    top100_j: list[dict[str, Any]] = []
    for s in fx["snippets"]:
        pid = s["pole_id"]
        pole = poles_by_id[pid]
        k = int(pole["k"])
        pidx = int(s["pair_idx"])
        pos_slot = s["pos_slot"]
        other_slot = "B" if pos_slot == "A" else "A"
        if s["band"] == "p99.9_plus":
            corr = 1.0 if (pidx % 10) < (6 if k == 26 else 8) else 0.0
            core_j.append(
                {
                    "pair_id": s["pair_id"],
                    "pole_id": pid,
                    "k": k,
                    "layer": int(pole["layer"]),
                    "neuron": int(pole["neuron"]),
                    "sign": int(pole["sign"]),
                    "band": "p99.9_plus",
                    "distractor_type": "dead_zone",
                    "condition": "own",
                    "judge_model": analyze_4b.PRIMARY_JUDGE_4B,
                    "order": "fwd",
                    "pos_slot": pos_slot,
                    "predicted_slot": pos_slot if corr == 1.0 else other_slot,
                    "outcome": "scored",
                    "correct": corr,
                    "raw_response": pos_slot if corr == 1.0 else other_slot,
                    "prompt_tokens": 500,
                    "completion_tokens": 4,
                    "reasoning_tokens": 0,
                    "visible_output_tokens": 4,
                    "cost_usd": 0.00052,
                    "pair_idx": pidx,
                }
            )
            if s["in_floor_6"]:
                for fl_cond in ("wrong", "no_label"):
                    fl_corr = 1.0 if pidx < 3 else 0.0
                    core_j.append(
                        {
                            "pair_id": s["pair_id"],
                            "pole_id": pid,
                            "k": k,
                            "layer": int(pole["layer"]),
                            "neuron": int(pole["neuron"]),
                            "sign": int(pole["sign"]),
                            "band": "p99.9_plus",
                            "distractor_type": "dead_zone",
                            "condition": fl_cond,
                            "judge_model": analyze_4b.PRIMARY_JUDGE_4B,
                            "order": "fwd",
                            "pos_slot": pos_slot,
                            "predicted_slot": pos_slot if fl_corr == 1.0 else other_slot,
                            "outcome": "scored",
                            "correct": fl_corr,
                            "raw_response": pos_slot if fl_corr == 1.0 else other_slot,
                            "prompt_tokens": 350,
                            "completion_tokens": 4,
                            "reasoning_tokens": 0,
                            "visible_output_tokens": 4,
                            "cost_usd": 0.00037,
                            "pair_idx": pidx,
                        }
                    )
    for s in top100_snips:
        pid = s["pole_id"]
        pole = poles_by_id[pid]
        k = int(pole["k"])
        pidx = int(s["pair_idx"])
        pos_slot = s["pos_slot"]
        other_slot = "B" if pos_slot == "A" else "A"
        # info (k=26) rises to 1.0 under top100; others 0.9 except k=11 (0.4)
        hits = 10 if k == 26 else (4 if k == 11 else 9)
        corr = 1.0 if (pidx % 10) < hits else 0.0
        top100_j.append(
            {
                "pair_id": s["pair_id"],
                "pole_id": pid,
                "k": k,
                "layer": int(pole["layer"]),
                "neuron": int(pole["neuron"]),
                "sign": int(pole["sign"]),
                "band": "top100",
                "distractor_type": "dead_zone",
                "condition": "own",
                "judge_model": analyze_4b.PRIMARY_JUDGE_4B,
                "order": "fwd",
                "pos_slot": pos_slot,
                "predicted_slot": pos_slot if corr == 1.0 else other_slot,
                "outcome": "scored",
                "correct": corr,
                "raw_response": pos_slot if corr == 1.0 else other_slot,
                "prompt_tokens": 500,
                "completion_tokens": 4,
                "reasoning_tokens": 0,
                "visible_output_tokens": 4,
                "cost_usd": 0.00052,
                "pair_idx": pidx,
            }
        )

    paths["judgements"].write_text("\n".join(json.dumps(r) for r in core_j) + "\n", encoding="utf-8")
    top100_j_p = tmp_path / "judgements_4b_top100.jsonl"
    top100_j_p.write_text("\n".join(json.dumps(r) for r in top100_j) + "\n", encoding="utf-8")

    sp_p = tmp_path / "pole_sparsity_diagnostics.json"
    info_diag = {
        "pole_id": "L00_N9209_pos",
        "k": 26,
        "layer": 0,
        "p99.9_thresh": 0.1475,
        "max_act_dir": 4.2188,
        "p99.9_frac_of_max": 0.0350,
        "frac_pos": 0.12,
        "frac_ge_20pct_max": 0.000073,
        "n_tokens_ge_20pct_max": 112,
        "sim_cal_rec": 0.86,
        "p99_9_plus_sampled": {"n_sampled": 20, "n_ge_20pct_max": 1},
        "top100_sampled": {"n_sampled": 20, "n_ge_20pct_max": 9},
    }
    sp_p.write_text(
        json.dumps(
            {
                "n_active_poles": 14,
                "n_poles_lt_20pct_max": 1,
                "frac_poles_lt_20pct_max": 1.0 / 14.0,
                "n_neurons_any_pole_lt_20pct_max": 1,
                "p99_9_frac_of_max_stats": {"min": 0.035, "p10": 0.05, "p25": 0.10, "median": 0.25, "p75": 0.30, "max": 0.45},
                "named_neurons_diagnostic": {"info": [info_diag]},
                "sparse_poles_lt_20pct_max": [info_diag],
            }
        ),
        encoding="utf-8",
    )

    t100_fig = tmp_path / "qwen4b_top100_two_panel.png"
    s20_fig = tmp_path / "qwen4b_detection_vs_strong20.png"
    plain_fig = tmp_path / "qwen4b_main_plain.png"

    metrics = analyze_4b.run_analysis_4b(
        active_poles_path=paths["active_poles"],
        scores_path=paths["scores"],
        opus_scores_path=paths["opus_scores"],
        snippets_path=paths["snippets"],
        judgements_path=paths["judgements"],
        wrong_donors_path=paths["wrong_donors"],
        metrics_path=paths["metrics"],
        summary_path=paths["summary"],
        figure_path=paths["figure"],
        top100_snippets_path=top100_snips_p,
        top100_judgements_path=top100_j_p,
        sparsity_diag_path=sp_p,
        top100_two_panel_figure_path=t100_fig,
        strong20_figure_path=s20_fig,
        main_plain_figure_path=plain_fig,
        n_resamples=80,
        n_random_splits=15,
        seed=42,
    )

    # Check top100 metrics, split-half, above-no_label pass criteria, strong20_rec, and sparsity enrichment
    assert "top100_own_terra" in metrics["sweep_audit"]["sweeps"]
    assert "top100_terra" in metrics["split_half_reliability"]
    assert "top100_terra_mean" in metrics["correlations"]
    assert "top100_mean_active_poles" in metrics["pass_matrices"]

    det_criteria_seen = {r["det_criterion"] for r in metrics["pass_matrices"]["top100_mean_active_poles"]}
    causal_seen = {r["causal_criterion"] for r in metrics["pass_matrices"]["top100_mean_active_poles"]}
    assert "> per-neuron no_label" in det_criteria_seen
    assert "own - no_label >= 0.20" in det_criteria_seen
    assert any(c.startswith("> global no_label") for c in det_criteria_seen)
    assert "strong20_rec >= 0.25" in causal_seen and "strong20_rec >= 0.50" in causal_seen

    named_by_short = {r["short_name"]: r for r in metrics["named_neurons"]}
    assert named_by_short["info"]["det_p999_terra_mean"] == pytest.approx(0.60)
    assert named_by_short["info"]["det_top100_terra_mean"] == pytest.approx(1.00)

    assert metrics["sparsity_analysis"]["sparse_poles_lt_20pct_max"][0]["det_top100_terra"] == pytest.approx(1.00)

    # All 4 figures generated cleanly as valid PNGs
    for fp in (paths["figure"], t100_fig, s20_fig, plain_fig):
        assert fp.is_file() and fp.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and fp.stat().st_size > 20_000

    md = paths["summary"].read_text(encoding="utf-8")
    assert "Pole Sparsity Diagnostic" in md
    assert "strong20_rec >= 0.50" in md
    assert "> per-neuron no_label" in md

