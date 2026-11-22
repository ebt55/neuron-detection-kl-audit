"""CPU unit tests for Qwen3-4B surrogate runner, math, CV, and figures."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qwen4b_eval.surrogate_4b import (
    FPR_SYSTEM_PROMPT,
    FPR_USER_TEMPLATE,
    analyze_surrogates_4b,
    assert_surrogate_output_path,
    build_fpr_messages,
    build_fpr_task_plan,
    build_multiband_task_plan,
    classify_yesno_outcome,
    compute_wdet_score,
    exact_auroc,
    expected_precision_at_k,
    extract_yesno_logprobs,
    make_stratified_folds,
    parse_yesno_choice,
    partial_spearman_corr,
    run_fpr_judge_async,
    run_multiband_judge_async,
    run_nested_oof_ridge_cv,
)
from w_audit.judge import SpendLimitExceededError
from w_audit.labeler import load_jsonl


# ---------------------------------------------------------------------------
# Helpers: Fake OpenAI Async Client
# ---------------------------------------------------------------------------


def _make_fake_chat_response(
    content: str,
    top_logprobs: list[dict[str, Any]],
    prompt_tokens: int = 320,
    completion_tokens: int = 1,
    reasoning_tokens: int = 0,
) -> Any:
    lp_items = [SimpleNamespace(token=d["token"], logprob=d["logprob"]) for d in top_logprobs]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=lp_items)]),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


class _FakeOpenAIClient:
    """Deterministic mock AsyncOpenAI client for multiband (A/B) and FPR (YES/NO) calls."""

    def __init__(self, mode: str = "ab", reply: str = "A") -> None:
        self.mode = mode
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            self.calls.append(dict(kwargs))
            if self.mode == "ab":
                alt = "B" if self.reply == "A" else "A"
                return _make_fake_chat_response(
                    self.reply,
                    [{"token": self.reply, "logprob": -0.05}, {"token": alt, "logprob": -3.1}],
                    prompt_tokens=500,
                    completion_tokens=1,
                )
            alt = "NO" if self.reply == "YES" else "YES"
            return _make_fake_chat_response(
                self.reply,
                [{"token": self.reply, "logprob": -0.08}, {"token": alt, "logprob": -2.6}],
                prompt_tokens=300,
                completion_tokens=1,
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


# ---------------------------------------------------------------------------
# 1. Isolation & Prompt Invariants
# ---------------------------------------------------------------------------


def test_isolation_guard_refuses_protected_files() -> None:
    for bad in (
        "results/judgements.jsonl",
        "results/qwen4b/judgements_4b.jsonl",
        "results/qwen4b/judgements_4b_top100.jsonl",
        "results/qwen4b/metrics_4b.json",
        "results/qwen4b/summary_4b.md",
        "results/qwen4b/gap_decomposition_4b.json",
        "results/qwen4b/band_kl_4b.json",
        "README.md",
        "README.md",
    ):
        with pytest.raises(ValueError, match="protected"):
            assert_surrogate_output_path(Path(bad))

    # Allowed surrogate outputs pass silently
    for ok in (
        "results/qwen4b/judgements_4b_multiband.jsonl",
        "results/qwen4b/judgements_4b_fpr.jsonl",
        "results/qwen4b/surrogate_metrics_4b.json",
        "results/qwen4b/surrogate_summary_4b.md",
        "figures/qwen4b_gap_decomposition.png",
        "figures/qwen4b_surrogate_comparison.png",
    ):
        assert_surrogate_output_path(Path(ok))


def test_fpr_prompt_has_zero_strongly_and_exact_structure() -> None:
    assert "strongly" not in FPR_SYSTEM_PROMPT.lower()
    assert "strongly" not in FPR_USER_TEMPLATE.lower()
    msgs = build_fpr_messages("Fires on digits", "The value is <<42>> today.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Fires on digits" in msgs[1]["content"]
    assert "The value is <<42>> today." in msgs[1]["content"]
    assert "strongly" not in msgs[1]["content"].lower()


# ---------------------------------------------------------------------------
# 2. YES/NO Logprob Extraction, Parser, and Outcome Classifier
# ---------------------------------------------------------------------------


def test_yesno_logprobs_and_classification() -> None:
    # Case & leading-space logsumexp aggregation
    lp_y, lp_n, ym, py, imp = extract_yesno_logprobs(
        [
            {"token": "YES", "logprob": -0.3},
            {"token": " yes", "logprob": -2.0},
            {"token": "NO", "logprob": -1.5},
        ]
    )
    assert lp_y is not None and lp_n is not None and ym is not None and py is not None
    assert not imp
    assert lp_y > -0.3  # logsumexp combines YES + " yes"
    assert abs(ym - (lp_y - lp_n)) < 1e-5
    assert 0.5 < py < 1.0

    # Pruned top_logprobs (only NO present with logprob 0.0 -> YES imputed cleanly)
    lp_y2, lp_n2, ym2, py2, imp2 = extract_yesno_logprobs([{"token": "NO", "logprob": 0.0}])
    assert imp2
    assert lp_n2 == 0.0
    assert lp_y2 is not None and lp_y2 < -10.0
    assert ym2 is not None and ym2 < -10.0
    assert py2 is not None and py2 < 1e-4

    # Strict parser rejects substrings inside words ("YESTERDAY", "NOTHING")
    assert parse_yesno_choice("YESTERDAY", None, None) is None
    assert parse_yesno_choice("NOTHING", None, None) is None
    assert parse_yesno_choice("YES or NO", None, None) is None
    assert parse_yesno_choice("yes.", None, None) == "YES"
    assert parse_yesno_choice("  No! ", None, None) == "NO"

    # Exact logprob tie -> excluded
    pred, out, pyes, corr = classify_yesno_outcome("YES", "YES", -0.693147, -0.693147)
    assert pred is None and out == "logprob_tie" and pyes is None and corr is None

    # Scored ordinary negative probe where model predicts YES -> false positive (predicted_yes=1, correct=0)
    pred_fp, out_fp, pyes_fp, corr_fp = classify_yesno_outcome("YES", "NO", -0.1, -2.5)
    assert pred_fp == "YES" and out_fp == "scored" and pyes_fp == 1.0 and corr_fp == 0.0


# ---------------------------------------------------------------------------
# 3. Mock Async Runners (Multiband & FPR): Resume, Tier Filtering, Combined Cap
# ---------------------------------------------------------------------------


def _sample_poles() -> list[dict[str, Any]]:
    return [
        {
            "pole_id": "L00_N0001_pos",
            "k": 1,
            "layer": 0,
            "neuron": 1,
            "sign": 1,
            "pole_text": "Fires on uppercase tokens.",
        },
        {
            "pole_id": "L04_N0002_neg",
            "k": 2,
            "layer": 4,
            "neuron": 2,
            "sign": -1,
            "pole_text": "Fires on digits.",
        },
    ]


def test_multiband_and_fpr_runners_resume_tier_and_combined_cap(tmp_path: Path) -> None:
    poles = _sample_poles()
    mb_snips = [
        {
            "pair_id": "L00_N0001_pos__p99_99.9__dead_zone__0",
            "pole_id": "L00_N0001_pos",
            "band": "p99_99.9",
            "distractor_type": "dead_zone",
            "pair_idx": 0,
            "pos_slot": "A",
            "pos": {"marked_text": "Hello <<WORLD>> now."},
            "dist": {"marked_text": "quiet <<the>> text."},
        },
        {
            "pair_id": "L04_N0002_neg__p75_97__dead_zone__0",
            "pole_id": "L04_N0002_neg",
            "band": "p75_97",
            "distractor_type": "dead_zone",
            "pair_idx": 0,
            "pos_slot": "B",
            "pos": {"marked_text": "Count <<42>> items."},
            "dist": {"marked_text": "quiet <<cat>> naps."},
        },
    ]
    s4b_snips = [
        {
            "pair_id": "L00_N0001_pos__p97_99__dead_zone__0",
            "pole_id": "L00_N0001_pos",
            "band": "p97_99",
            "distractor_type": "dead_zone",
            "pair_idx": 0,
            "pos_slot": "A",
            "pos": {"marked_text": "Big <<DOG>> runs."},
            "dist": {"marked_text": "small <<and>> soft."},
        }
    ]
    fpr_snips = [
        {
            "probe_id": "L00_N0001_pos__ordinary_nonfiring__0",
            "pole_id": "L00_N0001_pos",
            "probe_type": "ordinary_nonfiring",
            "ground_truth": "NO",
            "tier": "core",
            "probe_idx": 0,
            "marked_text": "ordinary <<cat>> text.",
        },
        {
            "probe_id": "L00_N0001_pos__ordinary_nonfiring__1",
            "pole_id": "L00_N0001_pos",
            "probe_type": "ordinary_nonfiring",
            "ground_truth": "NO",
            "tier": "luna_extra",
            "probe_idx": 1,
            "marked_text": "extra <<dog>> text.",
        },
        {
            "probe_id": "L00_N0001_pos__top100_firing__0",
            "pole_id": "L00_N0001_pos",
            "probe_type": "top100_firing",
            "ground_truth": "YES",
            "tier": "core",
            "probe_idx": 0,
            "marked_text": "loud <<APPLE>> token.",
        },
    ]

    mb_plan = build_multiband_task_plan(poles, mb_snips, s4b_snips)
    # 2 terra multiband + 2 luna multiband + 1 luna p97_99 = 5 tasks
    assert len(mb_plan) == 5

    mb_out = tmp_path / "judgements_4b_multiband.jsonl"
    fpr_out = tmp_path / "judgements_4b_fpr.jsonl"

    client_ab = _FakeOpenAIClient(mode="ab", reply="A")
    res1 = asyncio.run(
        run_multiband_judge_async(
            mb_plan,
            output_path=mb_out,
            companion_spend_paths=(fpr_out,),
            max_usd=8.0,
            concurrency=2,
            openai_client=client_ab,
            progress_every=0,
        )
    )
    assert res1["completed_this_run"] == 5
    assert len(load_jsonl(mb_out)) == 5

    # Resume skips all 5 completed tasks with 0 API calls
    client_ab_2 = _FakeOpenAIClient(mode="ab", reply="A")
    res2 = asyncio.run(
        run_multiband_judge_async(
            mb_plan,
            output_path=mb_out,
            companion_spend_paths=(fpr_out,),
            max_usd=8.0,
            openai_client=client_ab_2,
            progress_every=0,
        )
    )
    assert res2["completed_this_run"] == 0
    assert res2["skipped_existing"] == 5
    assert len(client_ab_2.calls) == 0

    # FPR tier filtering: terra gets only 2 core probes; luna gets all 3 probes -> 5 total
    fpr_plan = build_fpr_task_plan(poles, fpr_snips)
    assert len(fpr_plan) == 5
    terra_tasks = [t for t, _, _ in fpr_plan if t.judge_model == "gpt-6.1-terra@none"]
    luna_tasks = [t for t, _, _ in fpr_plan if t.judge_model == "gpt-6-luna@none"]
    assert len(terra_tasks) == 2
    assert all(t.tier != "luna_extra" for t in terra_tasks)
    assert len(luna_tasks) == 3

    # Combined spend cap: setting max_usd below already-spent multiband cost raises SpendLimitExceededError
    with pytest.raises(SpendLimitExceededError):
        asyncio.run(
            run_fpr_judge_async(
                fpr_plan,
                output_path=fpr_out,
                companion_spend_paths=(mb_out,),
                max_usd=res1["cumulative_self_usd"] * 0.5,
                openai_client=_FakeOpenAIClient(mode="yesno", reply="NO"),
                progress_every=0,
            )
        )

    # With ample cap, FPR completes cleanly and records exact logprobs & outcomes
    client_yn = _FakeOpenAIClient(mode="yesno", reply="NO")
    res_fpr = asyncio.run(
        run_fpr_judge_async(
            fpr_plan,
            output_path=fpr_out,
            companion_spend_paths=(mb_out,),
            max_usd=8.0,
            concurrency=2,
            openai_client=client_yn,
            progress_every=0,
        )
    )
    assert res_fpr["completed_this_run"] == 5
    rows_fpr = load_jsonl(fpr_out)
    assert all(r["outcome"] == "scored" for r in rows_fpr)
    assert all(r["reasoning_tokens"] == 0 for r in rows_fpr)
    assert all(r["predicted_answer"] == "NO" for r in rows_fpr)
    assert all(r["yes_margin"] < 0 for r in rows_fpr)


# ---------------------------------------------------------------------------
# 4. Statistical Primitives (WDet, AUROC, Prec@k, Partial Spearman, Leak-Free CV)
# ---------------------------------------------------------------------------


def test_wdet_auroc_and_tie_invariant_precision_at_k() -> None:
    w = {"p99.9+": 0.5, "p99_99.9": 0.2, "p97_99": 0.1, "p75_97": 0.1, "lt_p75": 0.1}
    a = {"p99.9_plus": 1.0, "p99_99.9": 0.75, "p97_99": 0.4, "p75_97": 0.5, "p25_75": None}
    bmap = (
        ("p99.9+", "p99.9_plus"),
        ("p99_99.9", "p99_99.9"),
        ("p97_99", "p97_99"),
        ("p75_97", "p75_97"),
        ("lt_p75", "p25_75"),
    )
    # Clamped: 0.5*1.0 + 0.2*0.5 + 0.1*0.0 + 0.1*0.0 + 0.1*0.0 = 0.60
    assert abs(compute_wdet_score(w, a, bmap, mode="clamped") - 0.60) < 1e-9
    # Signed: 0.5*1.0 + 0.2*0.5 + 0.1*(-0.2) + 0.1*0.0 + 0.1*0.0 = 0.58
    assert abs(compute_wdet_score(w, a, bmap, mode="signed") - 0.58) < 1e-9
    # Raw: 0.5*1.0 + 0.2*0.75 + 0.1*0.4 + 0.1*0.5 + 0.1*0.5 = 0.79
    assert abs(compute_wdet_score(w, a, bmap, mode="raw") - 0.79) < 1e-9

    assert exact_auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert exact_auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert exact_auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)

    # Tie-invariant precision@2: 4 tied at score 1.0 (2 pos, 2 neg) -> expected prec@2 = 0.5 regardless of ordering
    p_a = expected_precision_at_k([1.0, 1.0, 1.0, 1.0, 0.0], [1, 1, 0, 0, 1], k=2)
    p_b = expected_precision_at_k([1.0, 1.0, 1.0, 1.0, 0.0], [0, 0, 1, 1, 1], k=2)
    assert p_a == pytest.approx(0.5)
    assert p_b == pytest.approx(0.5)


def test_partial_spearman_removes_confounder_and_detects_unique_signal() -> None:
    rng = np.random.default_rng(123)
    n = 200
    z = rng.normal(size=n)
    # Purely confounded X and Y (conditionally independent given Z)
    x_conf = 2.0 * z + 0.5 * rng.normal(size=n)
    y_conf = 2.0 * z + 0.5 * rng.normal(size=n)
    res_conf = partial_spearman_corr(x_conf, y_conf, z, n_resamples=200, seed=7)
    assert abs(res_conf["partial_spearman_rho"]) < 0.20

    # Genuine extra signal U shared by X and Y orthogonal to Z
    u = rng.normal(size=n)
    x_sig = z + 1.5 * u + 0.2 * rng.normal(size=n)
    y_sig = z + 1.5 * u + 0.2 * rng.normal(size=n)
    res_sig = partial_spearman_corr(x_sig, y_sig, z, n_resamples=200, seed=7)
    assert res_sig["partial_spearman_rho"] > 0.70
    assert res_sig["ci_low"] > 0.55
    assert res_sig["partial_spearman_p"] < 1e-10


def test_nested_oof_ridge_cv_zero_test_leakage_invariance() -> None:
    rng = np.random.default_rng(99)
    n = 60
    layers = [i % 6 for i in range(n)]
    folds = make_stratified_folds(layers, n_folds=5, seed=42)
    X = rng.normal(size=(n, 3))
    y = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)

    for rank_mode in (False, True):
        res_orig = run_nested_oof_ridge_cv(X, y, folds=folds, rank_mode=rank_mode)

        # Perturb fold 0 test rows massively -> fold 0 weights & alpha MUST be bit-for-bit identical!
        X_mod = X.copy()
        y_mod = y.copy()
        te0 = folds == 0
        X_mod[te0] = X_mod[te0] * 100.0 + 50.0
        y_mod[te0] = -999.0
        res_mod = run_nested_oof_ridge_cv(X_mod, y_mod, folds=folds, rank_mode=rank_mode)

        assert res_orig["chosen_alphas"][0] == res_mod["chosen_alphas"][0]
        assert res_orig["fold_weights"][0] == res_mod["fold_weights"][0]
        assert res_orig["fold_spearman_mean"] > 0.5


# ---------------------------------------------------------------------------
# 5. End-to-End Synthetic Analysis + 60-Neuron KL + Figure Generation
# ---------------------------------------------------------------------------


def test_analyze_surrogates_end_to_end_with_kl_and_figures(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)
    n_neurons = 30
    poles: list[dict[str, Any]] = []
    ulto_neurons: list[dict[str, Any]] = []
    gap_per_neuron: dict[str, Any] = {}
    kl_per_neuron: dict[str, Any] = {}

    det_rows: list[dict[str, Any]] = []
    fpr_rows: list[dict[str, Any]] = []

    named_map = {26: "info", 23: "PCR", 10: "insurance"}
    ks = [26, 23, 10] + [i for i in range(100, 100 + n_neurons - 3)]

    for idx, k in enumerate(ks):
        layer = (idx % 5) * 6
        pid = f"L{layer:02d}_N{k:04d}_pos"
        poles.append(
            {
                "pole_id": pid,
                "k": k,
                "layer": layer,
                "neuron": k,
                "sign": 1,
                "pole_text": f"Feature {k}",
            }
        )
        e20 = float(rng.uniform(0.05, 0.95))
        e999 = float(np.clip(e20 * 0.6, 0.02, 0.80))
        e99 = float(np.clip((1.0 - e999) * 0.25, 0.01, 0.30))
        e97 = float(np.clip((1.0 - e999 - e99) * 0.25, 0.01, 0.30))
        e75 = float(np.clip((1.0 - e999 - e99 - e97) * 0.40, 0.01, 0.40))
        elt75 = max(0.0, 1.0 - (e999 + e99 + e97 + e75))
        s20 = float(np.clip(0.8 * e20 + rng.normal(0, 0.1), -0.1, 1.0))
        fpr_true = float(rng.uniform(0.0, 0.6))
        sim = float(0.6 * e20 - 0.5 * fpr_true + rng.normal(0, 0.15))
        if k == 26:
            sim, act, s20 = 0.85, 0.95, 0.88
            bucket = "pass_causal"
        elif k == 23:
            sim, act, s20 = -0.57, 0.92, 0.14
            bucket = "overbroad_fp_damage"
        elif k == 10:
            sim, act, s20 = -0.01, 0.04, 0.19
            bucket = "weak_mass_dominated"
        else:
            act = float(np.clip(sim + 0.3, -0.2, 1.0))
            if sim >= 0.5:
                bucket = "pass_causal"
            elif sim < 0 and act >= 0.5:
                bucket = "overbroad_fp_damage"
            elif s20 < 0.25:
                bucket = "weak_mass_dominated"
            elif act < 0.5:
                bucket = "active_sim_failure"
            else:
                bucket = "other_partial"

        ulto_neurons.append(
            {
                "k": k,
                "layer": layer,
                "neuron": k,
                "sim_cal_rec": sim,
                "sim_cal_rec_active": act,
                "strong20_rec": s20,
                "strong50_rec": s20 * 0.7,
            }
        )
        b4 = {"p99.9+": e999, "p97_99.9": e99 + e97, "p75_97": e75, "lt_p75": elt75}
        b5 = {"p99.9+": e999, "p99_99.9": e99, "p97_99": e97, "p75_97": e75, "lt_p75": elt75}
        b6 = {
            "top100_thresh": e999 * 0.5,
            "rest_of_p99.9+": e999 * 0.5,
            "p99_99.9": e99,
            "p97_99": e97,
            "p75_97": e75,
            "bulk_lt_p75": elt75,
        }
        bp25 = {
            "p99.9+": e999,
            "p99_99.9": e99,
            "p97_99": e97,
            "p75_97": e75,
            "p25_75": elt75 * 0.5,
            "dead_or_opp": elt75 * 0.5,
        }
        gap_per_neuron[str(k)] = {
            "k": k,
            "neuron_id": f"L{layer:02d}_N{k:04d}",
            "layer": layer,
            "neuron": k,
            "is_bipolar": False,
            "named_neuron": named_map.get(k),
            "failure_bucket": bucket,
            "energy_l2": {"bands_4": b4, "bands_5": b5, "bands_6": b6, "bands_with_p25_75": bp25},
            "energy_l1": {"bands_4": b4, "bands_5": b5, "bands_6": b6, "bands_with_p25_75": bp25},
            "energy_p99.9+": e999,
            "energy_p97+": e999 + e99 + e97,
            "energy_lt_p97": e75 + elt75,
            "energy_ge_50pct_max": e20 * 0.5,
            "energy_ge_20pct_max": e20,
            "energy_lt_20pct_max": 1.0 - e20,
            "energy_gate_open": 0.7,
            "energy_gate_closed": 0.3,
            "sim_cal_rec": sim,
            "sim_cal_rec_active": act,
            "strong20_rec": s20,
            "strong50_rec": s20 * 0.7,
            "det_top100_terra_mean": 0.9,
            "det_p999_terra_mean": 0.85,
        }
        if idx < 12:
            kl_per_neuron[str(k)] = {
                "kl_shares_4band": b4,
                "kl_share_ge_20pct_max": e20,
            }

        for jm in ("gpt-6.1-terra@none", "gpt-6-luna@none"):
            for band in ("top100", "p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75"):
                if band == "top100" and jm != "gpt-6.1-terra@none":
                    continue
                for pi in range(4):
                    det_rows.append(
                        {
                            "pair_id": f"{pid}__{band}__{pi}",
                            "pole_id": pid,
                            "band": band,
                            "condition": "own",
                            "judge_model": jm,
                            "order": "fwd",
                            "outcome": "scored",
                            "correct": 1.0 if (pi < 3 or band in ("top100", "p99.9_plus")) else 0.0,
                            "cost_usd": 0.0002,
                            "reasoning_tokens": 0,
                        }
                    )
            for pi in range(4):
                pred_yes = 1.0 if (pi / 4.0) < fpr_true else 0.0
                fpr_rows.append(
                    {
                        "probe_id": f"{pid}__ord__{pi}",
                        "pole_id": pid,
                        "probe_type": "ordinary_nonfiring",
                        "ground_truth": "NO",
                        "judge_model": jm,
                        "outcome": "scored",
                        "predicted_yes": pred_yes,
                        "correct": 1.0 - pred_yes,
                        "p_yes": 0.8 if pred_yes else 0.1,
                        "yes_margin": 1.5 if pred_yes else -2.0,
                        "cost_usd": 0.0001,
                        "reasoning_tokens": 0,
                    }
                )
            for pi in range(2):
                fpr_rows.append(
                    {
                        "probe_id": f"{pid}__top__{pi}",
                        "pole_id": pid,
                        "probe_type": "top100_firing",
                        "ground_truth": "YES",
                        "judge_model": jm,
                        "outcome": "scored",
                        "predicted_yes": 1.0,
                        "correct": 1.0,
                        "p_yes": 0.95,
                        "yes_margin": 3.0,
                        "cost_usd": 0.0001,
                        "reasoning_tokens": 0,
                    }
                )

    # Build minimal bucket summary inside gap_doc
    buckets_summary: dict[str, Any] = {}
    for bname in ("pass_causal", "overbroad_fp_damage", "weak_mass_dominated", "active_sim_failure", "other_partial"):
        brows = [v for v in gap_per_neuron.values() if v["failure_bucket"] == bname]
        buckets_summary[bname] = {
            "count": len(brows),
            "share": len(brows) / n_neurons,
            "median_energy_p99_9_plus": float(np.median([r["energy_p99.9+"] for r in brows])) if brows else None,
            "median_energy_lt_p97": float(np.median([r["energy_lt_p97"] for r in brows])) if brows else None,
            "median_energy_ge_20pct_max": float(np.median([r["energy_ge_20pct_max"] for r in brows])) if brows else None,
            "median_strong20_rec": float(np.median([r["strong20_rec"] for r in brows])) if brows else None,
            "median_energy_gate_closed": 0.3 if brows else None,
            "median_sim_cal_rec": float(np.median([r["sim_cal_rec"] for r in brows])) if brows else None,
            "mean_bands_4_energy": {
                bk: float(np.mean([r["energy_l2"]["bands_4"][bk] for r in brows])) if brows else None
                for bk in ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
            },
            "named_neurons": [r["named_neuron"] for r in brows if r.get("named_neuron")],
        }

    gap_doc = {
        "n_neurons": n_neurons,
        "correlations": {
            "energy_ge_20pct_max_vs_strong20_rec": {
                "n": n_neurons,
                "spearman_rho": 0.82,
                "spearman_ci_low": 0.70,
                "spearman_ci_high": 0.91,
                "pearson_r": 0.80,
                "pearson_ci_low": 0.68,
                "pearson_ci_high": 0.90,
                "r_squared": 0.64,
            }
        },
        "failure_buckets_top100_ge_0_80": {"n_total": n_neurons, "buckets": buckets_summary},
        "per_neuron": gap_per_neuron,
    }

    ap_path = tmp_path / "active_poles_test.json"
    sc_path = tmp_path / "scores_test.json"
    gd_path = tmp_path / "gap_test.json"
    kl_path = tmp_path / "kl_test.json"
    j4b_path = tmp_path / "j4b_test.jsonl"
    jtop_path = tmp_path / "jtop_test.jsonl"
    jmb_path = tmp_path / "jmb_test.jsonl"
    jfpr_path = tmp_path / "jfpr_test.jsonl"

    ap_path.write_text(json.dumps({"poles": poles}), encoding="utf-8")
    sc_path.write_text(json.dumps({"neurons": ulto_neurons}), encoding="utf-8")
    gd_path.write_text(json.dumps(gap_doc), encoding="utf-8")
    kl_path.write_text(json.dumps({"n_neurons": len(kl_per_neuron), "per_neuron": kl_per_neuron}), encoding="utf-8")
    j4b_path.write_text("\n".join(json.dumps(r) for r in det_rows if r["band"] in ("p99.9_plus", "p97_99")) + "\n", encoding="utf-8")
    jtop_path.write_text("\n".join(json.dumps(r) for r in det_rows if r["band"] == "top100") + "\n", encoding="utf-8")
    jmb_path.write_text("\n".join(json.dumps(r) for r in det_rows if r["band"] in ("p99_99.9", "p75_97", "p25_75")) + "\n", encoding="utf-8")
    jfpr_path.write_text("\n".join(json.dumps(r) for r in fpr_rows) + "\n", encoding="utf-8")

    out_m = tmp_path / "out_surrogate_metrics.json"
    out_s = tmp_path / "out_surrogate_summary.md"
    fig_g = tmp_path / "out_gap.png"
    fig_s = tmp_path / "out_surr.png"

    doc = analyze_surrogates_4b(
        active_poles_path=ap_path,
        scores_path=sc_path,
        gap_decomp_path=gd_path,
        judgements_4b_path=j4b_path,
        judgements_top100_path=jtop_path,
        judgements_multiband_path=jmb_path,
        judgements_fpr_path=jfpr_path,
        band_kl_path=kl_path,
        output_metrics_path=out_m,
        output_summary_path=out_s,
        figure_gap_path=fig_g,
        figure_surrogate_path=fig_s,
        n_bootstrap=80,
        seed=42,
    )

    assert doc["n_neurons"] == n_neurons
    assert doc["surrogate_spend"]["within_budget"] is True
    assert doc["kl_60_neuron_validation"] is not None
    assert doc["kl_60_neuron_validation"]["n_neurons"] == 12
    assert "s2_wdet_clamped_terra" in doc["benchmarks"]
    assert "s4_cv_rank_ridge_full_terra" in doc["benchmarks"]
    assert out_m.is_file() and out_s.is_file()
    assert "Ulto85/neurons-kinda-clean" in out_s.read_text(encoding="utf-8")
    # Valid PNG magic headers
    assert fig_g.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert fig_s.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
