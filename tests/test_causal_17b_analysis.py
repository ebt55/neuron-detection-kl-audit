"""CPU unit tests for Qwen3-1.7B causal simulator, FPR runner, transfer, and cross-model analysis."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qwen17b_causal.analyze_17b_causal import (
    analyze_17b_causal,
    fit_4b_transfer_and_predict_17b,
)
from qwen17b_causal.simulate_17b import (
    assert_17b_causal_output_path,
    build_fpr_17b_task_plan,
    build_sim_17b_messages,
    build_sim_17b_task_plan,
    format_numbered_tokens,
    group_17b_neuron_labels,
    parse_sparse_ratings,
    run_fpr_17b_async,
    run_sim_17b_async,
)
from w_audit.judge import SpendLimitExceededError
from w_audit.labeler import load_jsonl


def _fake_sim_client(reply: str = '{"2": 4, "5": -2}') -> Any:
    calls: list[dict[str, Any]] = []

    async def _create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(
                prompt_tokens=600,
                completion_tokens=20,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    client.calls = calls
    return client


def _fake_fpr_client(reply: str = "NO") -> Any:
    calls: list[dict[str, Any]] = []
    alt = "YES" if reply == "NO" else "NO"

    async def _create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        lps = [
            SimpleNamespace(token=reply, logprob=-0.05),
            SimpleNamespace(token=alt, logprob=-3.0),
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=reply),
                    logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=lps)]),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=300,
                completion_tokens=1,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    client.calls = calls
    return client


def test_17b_causal_isolation_guard_refuses_frozen_and_extractor_files() -> None:
    for bad in (
        "results/judgements.jsonl",
        "results/qwen4b/judgements_4b.jsonl",
        "results/qwen4b/surrogate_metrics_4b.json",
        "results/qwen17b_causal/energy_and_kl_17b.json",
        "results/qwen17b_causal/causal_scores_17b.json",
        "data/poles.json",
        "data/labels.jsonl",
        "README.md",
        "README.md",
    ):
        with pytest.raises(ValueError, match="protected"):
            assert_17b_causal_output_path(Path(bad))

    for ok in (
        "results/qwen17b_causal/sim_calls_17b.jsonl",
        "results/qwen17b_causal/fpr_17b.jsonl",
        "results/qwen17b_causal/metrics_17b_causal.json",
        "results/qwen17b_causal/summary_17b_causal.md",
        "figures/cross_model_17b_vs_4b.png",
    ):
        assert_17b_causal_output_path(Path(ok))


def test_parse_sparse_ratings_handles_all_formats_and_clamping() -> None:
    # Standard sparse JSON with clamping and out-of-range ignore
    r, sp, st = parse_sparse_ratings('{"0": 8, "3": -7, "7": 2.5, "99": 4, "4": 0}', seq_len=8)
    assert st == "ok"
    assert len(r) == 8
    assert r[0] == 5.0 and r[3] == -5.0 and r[7] == 2.5 and r[4] == 0.0
    assert sp == {"0": 5.0, "3": -5.0, "7": 2.5}

    # Empty JSON -> empty_ok
    r2, sp2, st2 = parse_sparse_ratings("{}", seq_len=8)
    assert st2 == "empty_ok" and all(v == 0.0 for v in r2) and sp2 == {}

    # Markdown fence + wrapper dict
    r3, sp3, st3 = parse_sparse_ratings('```json\n{"activations": {"1": 3, "6": -1}}\n```', seq_len=8)
    assert st3 == "ok" and r3[1] == 3.0 and r3[6] == -1.0

    # Truncated JSON -> fallback_regex recovery
    r4, sp4, st4 = parse_sparse_ratings('{"1": 4, "5": -2, "6":', seq_len=8)
    assert st4 == "fallback_regex" and r4[1] == 4.0 and r4[5] == -2.0

    # Pure garbage -> parse_error
    r5, sp5, st5 = parse_sparse_ratings("cannot answer", seq_len=8)
    assert st5 == "parse_error" and all(v == 0.0 for v in r5)

    # Numbered token formatter escapes newlines cleanly
    fmt = format_numbered_tokens(["Hello", "\nworld", "\t!"])
    assert "0:Hello" in fmt and "1:\\nworld" in fmt and "2:\\t!" in fmt
    msgs = build_sim_17b_messages("Pos desc", "Neg desc", ["Hello", " world"])
    assert len(msgs) == 2 and "Pos desc" in msgs[1]["content"] and "Neg desc" in msgs[1]["content"]


def test_sim_and_fpr_17b_runners_resume_tiers_and_caps(tmp_path: Path) -> None:
    poles = [
        {"pole_id": "L04_N0001_pos", "layer": 4, "neuron": 1, "sign": 1},
        {"pole_id": "L04_N0001_neg", "layer": 4, "neuron": 1, "sign": -1},
    ]
    labels = {
        "L04_N0001_pos": {"pole_id": "L04_N0001_pos", "own_label": "Positive feature"},
        "L04_N0001_neg": {"pole_id": "L04_N0001_neg", "own_label": "Negative feature"},
    }
    nlabels = group_17b_neuron_labels(poles, labels)
    wins = [
        {
            "window_id": "L04_N0001__w00",
            "neuron_id": "L04_N0001",
            "layer": 4,
            "neuron": 1,
            "window_idx": 0,
            "stratum": "ordinary",
            "tokens_str": ["a", " b", " c", " d", " e", " f", " g", " h"],
            "text": "a b c d e f g h",
        }
    ]
    sim_plan = build_sim_17b_task_plan(wins, nlabels)
    assert len(sim_plan) == 1

    sim_out = tmp_path / "sim_calls_17b.jsonl"
    fpr_out = tmp_path / "fpr_17b.jsonl"
    c_sim = _fake_sim_client('{"2": 4, "5": -2}')
    res1 = asyncio.run(
        run_sim_17b_async(
            sim_plan,
            output_path=sim_out,
            companion_fpr_path=fpr_out,
            max_usd=3.5,
            combined_max_usd=4.5,
            openai_client=c_sim,
            progress_every=0,
        )
    )
    assert res1["completed_this_run"] == 1
    assert res1["parse_errors_this_run"] == 0
    rows = load_jsonl(sim_out)
    assert rows[0]["ratings"] == [0.0, 0.0, 4.0, 0.0, 0.0, -2.0, 0.0, 0.0]

    # Resume skips completed window
    c_sim2 = _fake_sim_client()
    res2 = asyncio.run(
        run_sim_17b_async(
            sim_plan,
            output_path=sim_out,
            companion_fpr_path=fpr_out,
            openai_client=c_sim2,
            progress_every=0,
        )
    )
    assert res2["completed_this_run"] == 0 and res2["skipped_existing"] == 1 and len(c_sim2.calls) == 0

    # FPR probes tier filtering & combined spend cap
    probes = [
        {
            "probe_id": "L04_N0001_pos__ord__0",
            "pole_id": "L04_N0001_pos",
            "probe_type": "ordinary_nonfiring",
            "ground_truth": "NO",
            "tier": "core",
            "probe_idx": 0,
            "marked_text": "ordinary <<word>> here.",
        },
        {
            "probe_id": "L04_N0001_pos__ord__1",
            "pole_id": "L04_N0001_pos",
            "probe_type": "ordinary_nonfiring",
            "ground_truth": "NO",
            "tier": "luna_extra",
            "probe_idx": 1,
            "marked_text": "extra <<word>> here.",
        },
    ]
    fpr_plan = build_fpr_17b_task_plan(poles, labels, probes)
    assert len(fpr_plan) == 3  # 1 terra core + 2 luna all

    with pytest.raises(SpendLimitExceededError):
        asyncio.run(
            run_fpr_17b_async(
                fpr_plan,
                output_path=fpr_out,
                companion_sim_path=sim_out,
                max_usd=1.0,
                combined_max_usd=res1["cumulative_self_usd"] * 0.5,
                openai_client=_fake_fpr_client("NO"),
                progress_every=0,
            )
        )

    res_fpr = asyncio.run(
        run_fpr_17b_async(
            fpr_plan,
            output_path=fpr_out,
            companion_sim_path=sim_out,
            max_usd=1.0,
            combined_max_usd=4.5,
            openai_client=_fake_fpr_client("NO"),
            progress_every=0,
        )
    )
    assert res_fpr["completed_this_run"] == 3


def test_zero_shot_4b_to_17b_transfer_is_strictly_invariant_to_17b_data() -> None:
    rng = np.random.default_rng(77)
    X4 = rng.normal(size=(60, 4))
    y4 = 1.2 * X4[:, 0] - 0.5 * X4[:, 1] + 0.3 * rng.normal(size=60)
    lays4 = [i % 5 for i in range(60)]
    X17_a = rng.normal(size=(20, 4))
    X17_b = X17_a * 50.0 - 100.0

    for rmode in (False, True):
        out_a = fit_4b_transfer_and_predict_17b(X4, y4, lays4, X17_a, rank_mode=rmode, seed=42)
        out_b = fit_4b_transfer_and_predict_17b(X4, y4, lays4, X17_b, rank_mode=rmode, seed=42)
        assert out_a["best_alpha_4b_cv"] == out_b["best_alpha_4b_cv"]
        assert out_a["std_weights_4b"] == out_b["std_weights_4b"]
        assert len(out_a["preds_17b"]) == 20


def test_analyze_17b_causal_end_to_end_cpu(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    n_neurons = 20
    poles: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []
    fpr_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    ekl_per_n: dict[str, Any] = {}
    cs_per_n: dict[str, Any] = {}

    for i in range(n_neurons):
        lay = (4, 11, 18, 24)[i % 4]
        nid = f"L{lay:02d}_N{100 + i:04d}"
        e20 = float(rng.uniform(0.05, 0.90))
        e999 = float(np.clip(e20 * 0.55, 0.02, 0.75))
        e97 = float(np.clip((1.0 - e999) * 0.35, 0.02, 0.40))
        e75 = float(np.clip((1.0 - e999 - e97) * 0.45, 0.02, 0.40))
        elt = max(0.0, 1.0 - (e999 + e97 + e75))
        b4 = {"p99.9+": e999, "p97_99.9": e97, "p75_97": e75, "lt_p75": elt}
        s20 = float(np.clip(0.85 * e20 + rng.normal(0, 0.08), -0.1, 1.0))
        sim_v = float(0.65 * e20 + rng.normal(0, 0.12))
        act_v = float(np.clip(0.5 + 0.4 * e20 + rng.normal(0, 0.1), -0.2, 1.0))

        ekl_per_n[nid] = {
            "neuron_id": nid,
            "layer": lay,
            "neuron": 100 + i,
            "corpus_energy_4band": b4,
            "eval_energy_4band": b4,
            "kl_share_4band_norm": b4,
            "corpus_energy_ge_20pct_max": e20,
            "corpus_energy_ge_50pct_max": e20 * 0.5,
            "eval_energy_ge_20pct_max": e20,
            "kl_share_ge_20pct_max_norm": e20,
            "corpus_energy_gate_closed": 0.3,
            "eval_energy_gate_closed": 0.3,
            "kl_share_gate_closed_norm": 0.3,
            "strong20_rec": s20,
            "strong50_rec": s20 * 0.7,
        }
        cs_per_n[nid] = {
            "neuron_id": nid,
            "layer": lay,
            "neuron": 100 + i,
            "sim_cal_rec": sim_v,
            "sim_cal_rec_cv": sim_v - 0.02,
            "sim_cal_rec_active": act_v,
            "corr": 0.4,
            "sim_raw_rec": sim_v - 0.1,
        }
        sim_rows.append(
            {
                "window_id": f"{nid}__w00",
                "judge_model": "gpt-6.1-terra@none",
                "parse_status": "ok",
                "reasoning_tokens": 0,
                "cost_usd": 0.0008,
            }
        )

        for sign, pname in ((1, "pos"), (-1, "neg")):
            pid = f"{nid}_{pname}"
            poles.append({"pole_id": pid, "neuron_id": nid, "layer": lay, "neuron": 100 + i, "sign": sign})
            for jm in ("gpt-6.1-terra@none", "gpt-6-luna@none"):
                for band in ("p99.9_plus", "p97_99.9", "p90_97", "p75_90"):
                    for pi in range(2):
                        det_rows.append(
                            {
                                "pair_id": f"{pid}__{band}__{pi}",
                                "pole_id": pid,
                                "band": band,
                                "distractor_type": "dead_zone",
                                "condition": "own",
                                "order": "fwd",
                                "judge_model": jm,
                                "outcome": "scored",
                                "correct": 1.0 if (band == "p99.9_plus" and (pi == 0 or i % 3 != 0)) else (1.0 if pi == 0 else 0.0),
                            }
                        )
                fpr_rows.append(
                    {
                        "probe_id": f"{pid}__ord__0",
                        "pole_id": pid,
                        "probe_type": "ordinary_nonfiring",
                        "ground_truth": "NO",
                        "judge_model": jm,
                        "outcome": "scored",
                        "predicted_yes": 0.0 if act_v > 0.5 else 1.0,
                        "p_yes": float(np.clip(0.8 - 0.7 * act_v, 0.01, 0.99)),
                        "yes_margin": -2.0 * act_v,
                        "reasoning_tokens": 0,
                        "cost_usd": 0.0001,
                    }
                )
                fpr_rows.append(
                    {
                        "probe_id": f"{pid}__top__0",
                        "pole_id": pid,
                        "probe_type": "top100_firing",
                        "ground_truth": "YES",
                        "judge_model": jm,
                        "outcome": "scored",
                        "predicted_yes": 1.0,
                        "p_yes": 0.95,
                        "yes_margin": 3.0,
                        "reasoning_tokens": 0,
                        "cost_usd": 0.0001,
                    }
                )

    # Minimal 4B metrics doc for zero-shot transfer
    pn4: dict[str, Any] = {}
    for k in range(30):
        e20_4 = float(rng.uniform(0.05, 0.95))
        pn4[str(k)] = {
            "layer": (k % 5) * 6,
            "sim_cal_rec": 0.5 * e20_4 + float(rng.normal(0, 0.1)),
            "sim_cal_rec_active": 0.6 * e20_4 + float(rng.normal(0, 0.1)),
            "s2_wdet_clamped_terra": e20_4 * 0.6,
            "s2_wdet_clamped_luna": e20_4 * 0.5,
            "fpr_ord_terra": 0.05,
            "fpr_ord_luna": 0.05,
            "tpr_top_terra": 0.9,
            "tpr_top_luna": 0.85,
            "b1_energy_ge_20pct_max": e20_4,
            "s3_one_minus_pyes_ord_terra": 0.9 - 0.2 * (1 - e20_4),
            "s3_one_minus_pyes_ord_luna": 0.85 - 0.2 * (1 - e20_4),
            "s3_pyes_margin_gap_terra": 2.0 + e20_4,
            "s3_pyes_margin_gap_luna": 1.8 + e20_4,
            "b0_p999_terra": 0.85,
            "b0_p999_luna": 0.80,
        }
    m4b = {
        "benchmarks": {
            ck: {
                "sim_cal_rec": {"spearman_rho": 0.2, "spearman_ci_low": 0.1, "spearman_ci_high": 0.3},
                "sim_cal_rec_active": {"spearman_rho": 0.4, "spearman_ci_low": 0.3, "spearman_ci_high": 0.5},
            }
            for ck in ("b0_p999_terra", "b1_energy_ge_20pct_max", "s2_wdet_clamped_terra", "s3_one_minus_pyes_ord_terra", "s3_pyes_margin_gap_luna")
        },
        "gap_decomposition_summary": {"correlations": {}},
        "per_neuron": pn4,
    }

    p_poles = tmp_path / "p.json"
    p_jud = tmp_path / "j.jsonl"
    p_ekl = tmp_path / "ekl.json"
    p_cs = tmp_path / "cs.json"
    p_sim = tmp_path / "sim.jsonl"
    p_fpr = tmp_path / "fpr.jsonl"
    p_m4b = tmp_path / "m4b.json"

    p_poles.write_text(json.dumps({"poles": poles}), encoding="utf-8")
    p_jud.write_text("\n".join(json.dumps(r) for r in det_rows) + "\n", encoding="utf-8")
    p_ekl.write_text(json.dumps({"n_neurons": n_neurons, "per_neuron": ekl_per_n}), encoding="utf-8")
    p_cs.write_text(json.dumps({"n_neurons": n_neurons, "per_neuron": cs_per_n}), encoding="utf-8")
    p_sim.write_text("\n".join(json.dumps(r) for r in sim_rows) + "\n", encoding="utf-8")
    p_fpr.write_text("\n".join(json.dumps(r) for r in fpr_rows) + "\n", encoding="utf-8")
    p_m4b.write_text(json.dumps(m4b), encoding="utf-8")

    out_m = tmp_path / "out_metrics_17b.json"
    out_s = tmp_path / "out_summary_17b.md"
    out_f = tmp_path / "out_fig_17b.png"

    doc = analyze_17b_causal(
        poles_path=p_poles,
        judgements_17b_path=p_jud,
        energy_kl_17b_path=p_ekl,
        causal_scores_17b_path=p_cs,
        sim_calls_17b_path=p_sim,
        fpr_17b_path=p_fpr,
        surrogate_4b_metrics_path=p_m4b,
        band_kl_4b_path=tmp_path / "missing_kl.json",
        output_metrics_path=out_m,
        output_summary_path=out_s,
        figure_path=out_f,
        n_bootstrap=60,
        seed=42,
    )
    assert doc["n_neurons"] == n_neurons
    assert doc["causal_17b_spend"]["within_budget"] is True
    assert "transfer_4b_to_17b_rank_ridge_active_terra" in doc["benchmarks_17b"]
    assert out_m.is_file() and out_s.is_file()
    assert "Ulto85/neurons-kinda-clean" in out_s.read_text(encoding="utf-8")
    assert out_f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
