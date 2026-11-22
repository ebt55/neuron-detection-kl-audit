"""CPU unit tests for weak-band clean-window survival & unfiltered context ablation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from w_audit.config import SEED
from w_audit.context_ablation import (
    SURVIVAL_REPORT_FILENAME,
    UNFILTERED_SNIPPETS_FILENAME,
    build_weak_band_survival_and_unfiltered_ablation,
)
from w_audit.extract import run_extraction
from w_audit.snippets import WEAK_BANDS, build_snippets
from test_extract import (
    SyntheticQwenForCausalLM,
    WhitespaceIntTokenizer,
    _make_topic_clustered_documents,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_weak_band_survival_and_unfiltered_context_ablation_cpu(tmp_path: Path) -> None:
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
    second_counts = {0: 2, 1: 1, 2: 1, 3: 1}
    toy_counts = {0: 2, 1: 2, 2: 1, 3: 1}

    run_extraction(
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
        run_hook_check=False,
    )
    build_snippets(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=SEED,
        label_examples_per_pole=8,
        pairs_per_cell=4,
    )

    # Record SHA-256 of frozen core files to prove context_ablation NEVER mutates them!
    poles_hash_before = _sha256(tmp_path / "data" / "poles.json")
    labels_ex_hash_before = _sha256(tmp_path / "data" / "label_examples.jsonl")
    clean_snips_hash_before = _sha256(tmp_path / "data" / "snippets.jsonl")

    report = build_weak_band_survival_and_unfiltered_ablation(
        tokenizer=tok,
        base_dir=tmp_path,
        seed=SEED,
        pairs_per_cell=4,
    )

    # Hard rule: zero modification to frozen artifacts
    assert _sha256(tmp_path / "data" / "poles.json") == poles_hash_before
    assert _sha256(tmp_path / "data" / "label_examples.jsonl") == labels_ex_hash_before
    assert _sha256(tmp_path / "data" / "snippets.jsonl") == clean_snips_hash_before

    # Verify both new standalone outputs exist
    survival_path = tmp_path / "results" / SURVIVAL_REPORT_FILENAME
    unfiltered_path = tmp_path / "data" / UNFILTERED_SNIPPETS_FILENAME
    assert survival_path.exists()
    assert unfiltered_path.exists()

    # Check weak_band_survival.json schema and math invariants
    for band in WEAK_BANDS:
        bsec = report["weak_band_survival"][band]
        assert set(bsec.keys()) >= {"all_100", "heldout_80", "pilot_20", "by_layer", "per_pole"}
        assert bsec["all_100"]["n_poles"] == 16
        assert bsec["pilot_20"]["n_poles"] == 8
        assert bsec["heldout_80"]["n_poles"] == 8
        assert set(bsec["by_layer"].keys()) == {"0", "1", "2", "3"}

        # Exact sum consistency: pooled_surviving + pooled_ge90 == pooled_total
        tot = bsec["all_100"]["pooled_total_candidates"]
        surv = bsec["all_100"]["pooled_surviving_clean"]
        assert tot > 0
        assert 0 <= surv <= tot
        assert abs(bsec["all_100"]["pooled_survival_rate"] - (surv / tot)) < 1e-12
        assert abs(
            (bsec["all_100"]["pooled_survival_rate"] + bsec["all_100"]["pooled_other_ge_p90_rate"])
            - 1.0
        ) < 1e-12
        dist_stats = bsec["all_100"]["per_pole_survival_rate"]
        assert set(dist_stats.keys()) >= {"mean", "median", "p10", "p90", "min", "max"}
        assert dist_stats["min"] <= dist_stats["p10"] <= dist_stats["median"] <= dist_stats["p90"] <= dist_stats["max"]

    # Check unfiltered snippet file invariants
    ab = report["unfiltered_ablation_summary"]
    assert ab["all_invariants_passed"] is True
    assert ab["exact_50_50_cell_balance_rate"] == 1.0
    assert ab["imbalanced_cells_count"] == 0
    assert ab["slot_A_rate"] == 0.5
    assert ab["dist_window_ge_p90_count"] == 0
    assert ab["bos_pos0_leak_count"] == 0
    assert ab["same_doc_or_split_leak_count"] == 0

    rows = [json.loads(l) for l in unfiltered_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == ab["total_pairs"]
    assert len(rows) > 0

    # Verify exact k/k cell balance, odd eval docs, distinct docs, and unfiltered positives actually include other>=p90
    cell_slots: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    any_pos_other_ge_p90 = False
    for r in rows:
        assert r["band"] in WEAK_BANDS
        assert r["distractor_type"] == "dead_zone"
        assert r["distractor_tier"] == 0
        assert r["ablation_arm"] == "unfiltered_weak_context"
        assert "_unfiltered__" in r["pair_id"]
        assert r["pos"]["doc_id"] % 2 == 1
        assert r["dist"]["doc_id"] % 2 == 1
        assert r["pos"]["doc_id"] != r["dist"]["doc_id"]
        assert r["pos"]["window_tokens"] == 69 and r["dist"]["window_tokens"] == 69
        assert r["pos"]["target_offset"] == 60 and r["dist"]["target_offset"] == 60
        assert r["pos"]["pos_in_chunk"] >= 60 and r["dist"]["pos_in_chunk"] >= 60
        if r["pos"]["other_has_ge_p90"]:
            any_pos_other_ge_p90 = True
        cell_slots[(r["pole_id"], r["band"])][r["pos_slot"]] += 1

    assert any_pos_other_ge_p90 is True
    for ckey, cnt in cell_slots.items():
        assert cnt["A"] == cnt["B"] and cnt["A"] > 0, f"Imbalanced unfiltered cell {ckey}: {cnt}"
