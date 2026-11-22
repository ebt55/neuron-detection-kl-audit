"""CPU unit tests for qwen4b_eval.effect_weighted."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from qwen4b_eval.effect_weighted import (
    FOUR_BANDS,
    GE_P75_BANDS,
    compute_neuron_ewl,
    compute_neuron_ewl_over_floor,
    merge_4b_p97_99_9_accuracy,
    neuron_bootstrap_mean,
    pooled_energy_kl_analysis,
    run_effect_weighted,
)


def test_compute_neuron_ewl_exact_missing_and_renorm() -> None:
    kl = {"p99.9+": 0.30, "p97_99.9": 0.35, "p75_97": 0.15, "lt_p75": 0.20}
    accs = {"p99.9+": 0.85, "p97_99.9": 0.65, "p75_97": 0.55, "lt_p75": 0.50}

    # Exact manual EWL: 0.30*(0.70) + 0.35*(0.30) + 0.15*(0.10) + 0.20*(0.0) = 0.21 + 0.105 + 0.015 = 0.33
    full = compute_neuron_ewl(kl, accs, FOUR_BANDS, renormalize=False)
    assert math.isclose(full, 0.33, rel_tol=1e-12, abs_tol=1e-12)

    # Renormalised over >= p75 (denominator = 0.80): 0.33 / 0.80 = 0.4125
    ge75 = compute_neuron_ewl(kl, accs, GE_P75_BANDS, renormalize=True)
    assert math.isclose(ge75, 0.33 / 0.80, rel_tol=1e-12, abs_tol=1e-12)

    # Missing band (None) contributes 0 lift (equivalent to acc = 0.50)
    accs_missing = {"p99.9+": 0.85, "p97_99.9": 0.65, "p75_97": None, "lt_p75": None}
    full_missing = compute_neuron_ewl(kl, accs_missing, FOUR_BANDS, renormalize=False)
    assert math.isclose(full_missing, 0.30 * 0.70 + 0.35 * 0.30, rel_tol=1e-12, abs_tol=1e-12)

    # Lift over no_label floor (raw and 2x scaled)
    floor = {"p99.9+": 0.55, "p97_99.9": 0.45, "p75_97": 0.50, "lt_p75": None}
    raw_nl = compute_neuron_ewl_over_floor(kl, accs, floor, GE_P75_BANDS, renormalize=False, scale=1.0)
    expected_raw = 0.30 * (0.85 - 0.55) + 0.35 * (0.65 - 0.45) + 0.15 * (0.55 - 0.50)
    assert math.isclose(raw_nl, expected_raw, rel_tol=1e-12, abs_tol=1e-12)
    scaled_nl = compute_neuron_ewl_over_floor(kl, accs, floor, GE_P75_BANDS, renormalize=True, scale=2.0)
    assert math.isclose(scaled_nl, 2.0 * expected_raw / 0.80, rel_tol=1e-12, abs_tol=1e-12)


def test_merge_4b_p97_99_9_accuracy_energy_equal_and_missing() -> None:
    # Energy-weighted merge: (0.75 * 0.80 + 0.25 * 0.60) / 1.0 = 0.75
    assert math.isclose(merge_4b_p97_99_9_accuracy(0.80, 0.60, 0.75, 0.25, mode="energy"), 0.75)
    # Equal-weight merge: 0.70
    assert math.isclose(merge_4b_p97_99_9_accuracy(0.80, 0.60, 0.75, 0.25, mode="equal"), 0.70)
    # Missing one sub-band falls back cleanly to detected sub-band
    assert math.isclose(merge_4b_p97_99_9_accuracy(0.80, None, 0.75, 0.25, mode="energy"), 0.80)
    assert math.isclose(merge_4b_p97_99_9_accuracy(None, 0.60, 0.75, 0.25, mode="energy"), 0.60)
    assert merge_4b_p97_99_9_accuracy(None, None, 0.75, 0.25, mode="energy") is None


def test_pooled_energy_kl_clustered_vs_pair_ci_and_permuted_neuron_null() -> None:
    rng = np.random.default_rng(7)
    n_neurons = 48
    # Strong per-band mean differences + genuine within-band neuron-level correlation
    band_means = np.array([0.40, 0.35, 0.18, 0.07])
    z = rng.normal(0.0, 0.06, size=(n_neurons, 4))
    E = np.clip(band_means[None, :] + z, 0.01, None)
    E /= E.sum(axis=1, keepdims=True)
    K = np.clip(E + rng.normal(0.0, 0.015, size=(n_neurons, 4)), 0.005, None)
    K /= K.sum(axis=1, keepdims=True)

    res = pooled_energy_kl_analysis(E, K, n_bootstrap=300, n_permutations=200, seed=123)
    assert res["n_neurons"] == n_neurons
    assert res["n_pairs"] == 4 * n_neurons

    # Within-band Spearman matches scipy exactly
    for bi, bname in enumerate(FOUR_BANDS):
        expected_rho = float(stats.spearmanr(E[:, bi], K[:, bi]).statistic)
        assert math.isclose(res["within_band"][bname]["spearman_rho"], expected_rho, abs_tol=1e-12)
        assert res["within_band"][bname]["bootstrap_unit"] == "neuron"

    pooled = res["pooled"]
    cl = pooled["neuron_clustered_bootstrap_ci"]
    pr = pooled["pair_level_bootstrap_ci"]
    null = pooled["permuted_neuron_null"]

    assert cl["bootstrap_unit"] == "neuron_cluster_all_4_bands"
    assert pr["bootstrap_unit"] == "flattened_neuron_band_pair"
    assert cl["spearman_ci_low"] < pooled["spearman_rho"] < cl["spearman_ci_high"]
    assert pr["spearman_ci_low"] < pooled["spearman_rho"] < pr["spearman_ci_high"]

    # Permuted-neuron null preserves band means (> 0.5 baseline) yet obs strictly beats null
    assert null["spearman_mean"] > 0.50
    assert pooled["spearman_rho"] > null["spearman_q975"]
    assert null["spearman_p_empirical_ge_obs"] < 0.02

    # When within-band signal is pure noise (only band means shared), obs lies inside permuted null
    E_only_means = np.clip(band_means[None, :] + rng.normal(0.0, 0.03, size=(n_neurons, 4)), 0.01, None)
    E_only_means /= E_only_means.sum(axis=1, keepdims=True)
    K_indep = np.clip(band_means[None, :] + rng.normal(0.0, 0.03, size=(n_neurons, 4)), 0.01, None)
    K_indep /= K_indep.sum(axis=1, keepdims=True)
    res_null_case = pooled_energy_kl_analysis(E_only_means, K_indep, n_bootstrap=100, n_permutations=150, seed=99)
    assert res_null_case["pooled"]["permuted_neuron_null"]["spearman_p_empirical_ge_obs"] > 0.05


def test_neuron_bootstrap_mean_deterministic_and_handles_singleton() -> None:
    s1 = neuron_bootstrap_mean([0.2, 0.4, 0.6, 0.8], n_bootstrap=500, seed=42)
    s2 = neuron_bootstrap_mean([0.2, 0.4, 0.6, 0.8], n_bootstrap=500, seed=42)
    assert s1 == s2
    assert math.isclose(s1["mean"], 0.5)
    assert s1["ci_low"] < 0.5 < s1["ci_high"]

    single = neuron_bootstrap_mean([0.42], n_bootstrap=100, seed=42)
    assert single["n"] == 1 and math.isclose(single["ci_low"], 0.42) and math.isclose(single["ci_high"], 0.42)


def test_end_to_end_cpu_synthetic_and_offline_missing_snippets(tmp_path: Path) -> None:
    """Synthetic end-to-end test verifying bipolar filter, figure, reading sheet, and omitted-snippets mode."""
    # Build 12 synthetic Qwen3-4B neurons (including k=26 info and k=23 PCR) across layers
    specs = [
        (26, 0, 9209, False, 1.00, +0.86, +0.96, +0.88, "info"),
        (276, 35, 7008, False, 1.00, +0.91, +0.92, +0.65, None),
        (59, 8, 2134, False, 1.00, +0.58, +0.71, +0.76, None),
        (225, 32, 146, False, 1.00, +0.43, +0.85, +0.32, None),
        (23, 0, 8146, False, 1.00, -0.57, +0.92, +0.14, "PCR"),
        (17, 0, 7325, False, 1.00, -7.11, +0.22, +0.40, None),
        (93, 12, 3077, False, 1.00, -1.07, +0.52, +0.25, None),
        (256, 35, 617, False, 1.00, -0.80, +0.04, +0.21, None),
        (32, 4, 2686, True, 0.45, +0.01, +0.04, +0.05, None),
        (82, 8, 8930, True, 0.75, +0.03, +0.55, +0.35, None),
        (117, 16, 1674, False, 0.55, -0.001, -0.004, +0.02, None),
        (265, 35, 3526, True, 0.55, -0.72, -0.005, +0.01, None),
    ]

    poles = []
    gap_per = {}
    kl_per = {}
    e3_neurons = []
    e3_scores = []
    j4b_rows = []
    jtop_rows = []
    jmb_rows = []
    jfpr_rows = []
    stop_rows = []
    s4b_rows = []

    for idx, (k, layer, neuron, is_bi, t100_acc, sim, act, s20, named) in enumerate(specs):
        e3_neurons.append([layer, neuron])
        e3_scores.append({"k": k, "layer": layer, "neuron": neuron, "sim_cal_rec": sim})
        nid = f"L{layer:02d}_N{neuron:04d}"
        signs = [1, -1] if is_bi else [1]
        apids = [f"{nid}_{'pos' if s > 0 else 'neg'}" for s in signs]
        for s, pid in zip(signs, apids):
            poles.append(
                {
                    "pole_id": pid,
                    "neuron_id": nid,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": s,
                    "is_bipolar": is_bi,
                    "n_active_poles": len(signs),
                    "pole_text": f"fires on synthetic token word{k}",
                    "label": f"synthetic neuron {k}",
                }
            )
            # Top100 judgements & snippets (20 pairs)
            n_corr = int(round(20 * t100_acc))
            for pi in range(20):
                pair_id = f"{pid}__top100__{pi}"
                corr = pi < n_corr
                pos_slot = "A" if pi % 2 == 0 else "B"
                pred_slot = pos_slot if corr else ("B" if pos_slot == "A" else "A")
                jtop_rows.append(
                    {
                        "pair_id": pair_id, "pole_id": pid, "k": k, "layer": layer, "neuron": neuron,
                        "band": "top100", "condition": "own", "judge_model": "gpt-6.1-terra@none",
                        "outcome": "scored", "correct": corr, "pos_slot": pos_slot, "predicted_slot": pred_slot,
                    }
                )
                stop_rows.append(
                    {
                        "pair_id": pair_id, "pole_id": pid, "k": k, "band": "top100", "pair_idx": pi,
                        "pos": {"target_token_str": f" word{k}", "act_dir": 4.0 - 0.05 * pi, "marked_text": f"ctx << word{k}>> tail"},
                    }
                )
            # p99.9+ and p97_99 judgements
            for pi in range(6):
                for bname, acc_b in (("p99.9_plus", min(1.0, t100_acc)), ("p97_99", 0.60)):
                    pair_id = f"{pid}__{bname}__{pi}"
                    corr = pi < int(round(6 * acc_b))
                    j4b_rows.append(
                        {
                            "pair_id": pair_id, "pole_id": pid, "k": k, "layer": layer, "neuron": neuron,
                            "band": bname, "condition": "own", "judge_model": "gpt-6.1-terra@none",
                            "outcome": "scored", "correct": corr, "pos_slot": "A", "predicted_slot": "A" if corr else "B",
                        }
                    )
                    if bname == "p97_99":
                        s4b_rows.append(
                            {
                                "pair_id": pair_id, "pole_id": pid, "k": k, "band": "p97_99", "pair_idx": pi,
                                "pos": {"target_token_str": " mid", "act_dir": 1.2, "marked_text": "ctx << mid>> tail"},
                            }
                        )
            # Multiband judgements (leave p25_75 missing for neuron index 0 to test missing-cell path)
            for bname, acc_b in (("p99_99.9", 0.70), ("p75_97", 0.55), ("p25_75", 0.50)):
                if bname == "p25_75" and idx == 0:
                    continue
                for pi in range(4):
                    corr = pi < int(round(4 * acc_b))
                    jmb_rows.append(
                        {
                            "pair_id": f"{pid}__{bname}__{pi}", "pole_id": pid, "k": k, "layer": layer, "neuron": neuron,
                            "band": bname, "condition": "own", "judge_model": "gpt-6.1-terra@none",
                            "outcome": "scored", "correct": corr, "pos_slot": "A", "predicted_slot": "A" if corr else "B",
                        }
                    )
            # FPR rows
            for pi in range(4):
                jfpr_rows.append(
                    {
                        "probe_id": f"{pid}__ord__{pi}", "pole_id": pid, "k": k,
                        "judge_model": "gpt-6.1-terra@none", "probe_type": "ordinary_nonfiring",
                        "logprob_imputed": pi < 3,
                    }
                )

        lt75_val = 0.03 if is_bi else 0.30
        rem = 1.0 - lt75_val
        kl_sh = {"p99.9+": rem * 0.40, "p97_99.9": rem * 0.40, "p75_97": rem * 0.20, "lt_p75": lt75_val}
        gap_per[str(k)] = {
            "k": k, "neuron_id": nid, "layer": layer, "neuron": neuron,
            "is_bipolar": is_bi, "n_active_poles": len(signs), "active_pole_ids": apids,
            "dominant_pole_id": apids[0], "named_neuron": named, "max_abs_act": 5.0,
            "sim_cal_rec": sim, "sim_cal_rec_active": act, "strong20_rec": s20,
            "energy_l2": {
                "bands_4": kl_sh,
                "bands_5": {"p99.9+": kl_sh["p99.9+"], "p99_99.9": kl_sh["p97_99.9"] * 0.6, "p97_99": kl_sh["p97_99.9"] * 0.4, "p75_97": kl_sh["p75_97"], "lt_p75": lt75_val},
                "bands_with_p25_75": {"p25_75": 0.01, "dead_or_opp": max(0.0, lt75_val - 0.01)},
            },
        }
        kl_per[str(k)] = {
            "k": k, "neuron_id": nid, "layer": layer, "neuron": neuron,
            "is_bipolar": is_bi, "n_active_poles": len(signs),
            "opposite_sign_eval_energy_share": 0.0 if is_bi else 0.28,
            "kl_share_4band_norm": kl_sh,
            "eval_energy_4band": kl_sh,
            "corpus_energy_4band": kl_sh,
            "eval_energy_ge_20pct_max": 0.5 + 0.01 * idx,
            "corpus_energy_ge_20pct_max": 0.45 + 0.01 * idx,
            "kl_share_ge_20pct_max_norm": 0.48 + 0.01 * idx,
            "eval_energy_gate_closed": 0.3 + 0.02 * idx,
            "corpus_energy_gate_closed": 0.32 + 0.02 * idx,
            "kl_share_gate_closed_norm": 0.29 + 0.02 * idx,
        }

    # Add 1 extra unscored neuron to Ulto85 catalog
    e3_neurons.append([12, 1597])

    # Synthetic 1.7B fixtures (6 neurons: 2 pilot + 4 heldout)
    ek17_per = {}
    c17_per = {}
    poles17 = []
    j17_rows = []
    for ni in range(6):
        nid = f"L04_N{ni:04d}"
        in_pilot = ni < 2
        kl17 = {"p99.9+": 0.41, "p97_99.9": 0.43, "p75_97": 0.15, "lt_p75": 0.01}
        ek17_per[nid] = {
            "k": ni, "neuron_id": nid, "layer": 4, "neuron": ni, "in_pilot": in_pilot,
            "strong20_rec": 0.30 if ni < 4 else 0.10,
            "kl_share_4band_norm": kl17,
            "eval_energy_4band": kl17,
            "corpus_energy_4band": kl17,
            "corpus_energy_5band": {"p99.9_plus": 0.41, "p97_99.9": 0.43, "p90_97": 0.10, "p75_90": 0.05, "lt_p75": 0.01},
            "eval_energy_gate_closed": 0.5 + 0.05 * ni,
            "corpus_energy_gate_closed": 0.48 + 0.05 * ni,
            "kl_share_gate_closed_norm": 0.52 + 0.05 * ni,
        }
        c17_per[nid] = {
            "k": ni, "neuron_id": nid, "in_pilot": in_pilot,
            "sim_cal_rec": -0.03 + 0.01 * ni, "sim_cal_rec_active": 0.04 + 0.02 * ni,
            "strong20_rec_eval64": 0.35 if ni < 4 else 0.15, "corr": 0.08,
        }
        for s_tag in ("pos", "neg"):
            pid = f"{nid}_{s_tag}"
            poles17.append({"pole_id": pid, "in_pilot": in_pilot})
            for b, own_c, nl_c in (("p99.9_plus", 0.80, 0.55), ("p97_99.9", 0.60, 0.45), ("p90_97", 0.55, 0.50), ("p75_90", 0.55, 0.48)):
                for pi in range(5):
                    for cond, acc_v in (("own", own_c), ("no_label", nl_c)):
                        j17_rows.append(
                            {
                                "pair_id": f"{pid}__{b}__{cond}__{pi}", "pole_id": pid, "band": b,
                                "distractor_type": "dead_zone", "condition": cond, "order": "fwd",
                                "judge_model": "gpt-6.1-terra@none", "outcome": "scored",
                                "correct": pi < int(round(5 * acc_v)),
                            }
                        )

    sub17 = {
        "cohorts": {
            "heldout_80": {"gpt-6.1-terra@none": {"p97_99": {"conditions": {"own": {"mean": 0.594}}}, "p99_99.9": {"conditions": {"own": {"mean": 0.679}}}}},
            "all_100": {"gpt-6.1-terra@none": {"p97_99": {"conditions": {"own": {"mean": 0.600}}}, "p99_99.9": {"conditions": {"own": {"mean": 0.680}}}}},
        }
    }

    def _wjson(name: str, obj: object) -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    def _wjsonl(name: str, rows: list[dict]) -> Path:
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return p

    p_kl = _wjson("band_kl.json", {"zero_ablation_sanity_kl": 0.0, "kl_4band_additivity_ratio_median": 1.0, "per_neuron": kl_per})
    p_gap = _wjson("gap.json", {"per_neuron": gap_per})
    p_ap = _wjson("active_poles.json", {"poles": poles})
    p_e3n = _wjson("e3_neurons.json", e3_neurons)
    p_e3s = _wjson("e3_scores.json", {"neurons": e3_scores})
    p_j4b = _wjsonl("j4b.jsonl", j4b_rows)
    p_jtop = _wjsonl("jtop.jsonl", jtop_rows)
    p_jmb = _wjsonl("jmb.jsonl", jmb_rows)
    p_jfpr = _wjsonl("jfpr.jsonl", jfpr_rows)
    p_s4b = _wjsonl("s4b.jsonl", s4b_rows)
    p_stop = _wjsonl("stop.jsonl", stop_rows)
    p_ek17 = _wjson("ek17.json", {"zero_ablation_sanity_kl": 0.0, "per_neuron": ek17_per})
    p_c17 = _wjson("c17.json", {"per_neuron": c17_per})
    p_j17 = _wjsonl("j17.jsonl", j17_rows)
    p_sub17 = _wjson("sub17.json", sub17)
    p_poles17 = _wjson("poles17.json", {"poles": poles17})

    out_json = tmp_path / "effect_weighted.json"
    out_fig = tmp_path / "main_fig.png"
    out_sheet = tmp_path / "reading_sheet.md"

    payload = run_effect_weighted(
        kl_path=p_kl, kl_n60_path=p_kl, gap_path=p_gap, active_poles_path=p_ap,
        ulto_neurons_path=p_e3n, ulto_scores_path=p_e3s,
        judgements_4b_path=p_j4b, judgements_top100_path=p_jtop,
        judgements_multiband_path=p_jmb, judgements_fpr_path=p_jfpr,
        snippets_4b_path=p_s4b, snippets_top100_path=p_stop,
        ek_17b_path=p_ek17, causal_17b_path=p_c17, judgements_17b_path=p_j17,
        subband_17b_path=p_sub17, poles_17b_path=p_poles17,
        output_json_path=out_json, figure_path=out_fig, reading_sheet_path=out_sheet,
        n_bootstrap=100, n_permutations=80, seed=42,
    )

    assert out_json.is_file()
    assert out_fig.is_file() and out_fig.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out_sheet.is_file()
    sheet_txt = out_sheet.read_text(encoding="utf-8")
    assert "`k=26`" in sheet_txt and "`k=23`" in sheet_txt and "`k=17`" in sheet_txt

    # Verify bipolar-only subgroup has exactly 3 neurons & much smaller <p75 KL share than unipolar
    assert payload["qwen3_4b"]["n_bipolar_in_kl"] == 3
    assert payload["qwen3_4b"]["n_unipolar_in_kl"] == 9
    assert payload["qwen3_4b"]["bipolar_only"]["bands"]["lt_p75"]["kl_share"]["mean"] < 0.05
    assert payload["qwen3_4b"]["unipolar_only"]["bands"]["lt_p75"]["kl_share"]["mean"] > 0.25
    assert payload["qwen3_4b"]["all_kl_neurons"]["bands"]["lt_p75"]["n_missing_neurons"] == 1
    assert payload["statistical_disclosures"]["base_rates_and_detection_vs_causal_4b"]["unscored_ulto85_neurons"] == [[12, 1597]]
    assert payload["statistical_disclosures"]["top100_token_in_label_lexical_overlap"]["source"] == "computed_live_from_snippets_4b_top100"

    # Re-run WITHOUT snippet JSONL files (simulating slim export offline notebook mode)
    out_fig2 = tmp_path / "main_fig_offline.png"
    payload2 = run_effect_weighted(
        kl_path=p_kl, kl_n60_path=p_kl, gap_path=p_gap, active_poles_path=p_ap,
        ulto_neurons_path=p_e3n, ulto_scores_path=p_e3s,
        judgements_4b_path=p_j4b, judgements_top100_path=p_jtop,
        judgements_multiband_path=p_jmb, judgements_fpr_path=p_jfpr,
        snippets_4b_path=tmp_path / "missing_s4b.jsonl",
        snippets_top100_path=tmp_path / "missing_stop.jsonl",
        ek_17b_path=p_ek17, causal_17b_path=p_c17, judgements_17b_path=p_j17,
        subband_17b_path=p_sub17, poles_17b_path=p_poles17,
        output_json_path=out_json, figure_path=out_fig2, reading_sheet_path=out_sheet,
        n_bootstrap=100, n_permutations=80, seed=42,
    )
    assert out_fig2.is_file() and out_fig2.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload2["reading_sheet"] is None
    assert payload2["statistical_disclosures"]["top100_token_in_label_lexical_overlap"]["n_pairs"] == 300
