"""Download MIT-licensed Ulto85/neurons-kinda-clean results and build active_poles.json.

Steps:
1. Download `e3_neurons.json`, `e3_explanations.json`, `e3_scores.json`, and
   `e3_scores_opus.json` into `data/qwen4b/ulto85/`.
2. Classify all 278 scored neurons into active non-negligible poles (excluding poles
   starting with `negligible`/`none`/`weak`/`no `/`minimal`/`virtually`):
   verified 132 unipolar + 146 bipolar = 424 active poles, 0 both-negligible.
3. Verify exact `(k, layer, neuron)` for named neurons:
   - `info`: k=26, L0 N9209
   - `PCR`: k=23, L0 N8146
   - `insurance`: k=236, L32 N4362
4. Deterministically sample 50 neurons stratified across the 10 layers (5 per layer,
   seed=42) for the `full_explanation` arm (`in_full_expl_50=True`).
5. Save `data/qwen4b/active_poles.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
import numpy as np

from qwen4b_eval import (
    HOOK_LAYERS_4B,
    MODEL_ID_4B,
    NAMED_NEURONS_4B,
    NEGLIGIBLE_PREFIXES,
    SCHEMA_VERSION_4B,
    SEED_4B,
    ULTO85_BASE_URL,
    ULTO85_FILES,
    ensure_4b_dirs,
    make_neuron_id_4b,
    make_pole_id_4b,
)


def is_negligible_pole_text(text: str | None) -> bool:
    """Return True if a pole description is empty or starts with a negligible prefix."""
    if text is None:
        return True
    stripped = str(text).strip()
    if not stripped:
        return True
    low = stripped.lower()
    return any(low.startswith(prefix) for prefix in NEGLIGIBLE_PREFIXES)


def format_full_explanation(expl: Mapping[str, Any]) -> str:
    """Format the 4-field Ulto85 full neuron explanation string."""
    label = str(expl.get("label", "")).strip()
    pos_pole = str(expl.get("positive_pole", "")).strip()
    neg_pole = str(expl.get("negative_pole", "")).strip()
    rubric = str(expl.get("rubric", "")).strip()
    return (
        f"Summary: {label}\n"
        f"Positive pole: {pos_pole}\n"
        f"Negative pole: {neg_pole}\n"
        f"Rubric: {rubric}"
    )


def download_ulto85_files(
    ulto85_dir: Path,
    base_url: str = ULTO85_BASE_URL,
    files: Sequence[str] = ULTO85_FILES,
    force: bool = False,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """Download MIT-licensed Ulto85 result JSONs into `ulto85_dir`."""
    ulto85_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fname in files:
        dst = ulto85_dir / fname
        if force or not dst.is_file() or dst.stat().st_size == 0:
            url = f"{base_url.rstrip('/')}/{fname}"
            resp = httpx.get(url, follow_redirects=True, timeout=timeout)
            resp.raise_for_status()
            # Validate JSON before writing
            json.loads(resp.text)
            dst.write_bytes(resp.content)
        paths[fname] = dst
    return paths


def sample_stratified_full_expl_neurons(
    scored_neurons: Sequence[Mapping[str, Any]],
    layers: Sequence[int] = HOOK_LAYERS_4B,
    per_layer: int = 5,
    seed: int = SEED_4B,
) -> set[int]:
    """Deterministically sample `per_layer` neuron `k` indices per layer (50 total)."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for rec in scored_neurons:
        by_layer[int(rec["layer"])].append(int(rec["k"]))

    rng = np.random.default_rng(seed)
    selected_ks: set[int] = set()
    for layer in sorted(int(L) for L in layers):
        ks = sorted(by_layer[layer])
        if len(ks) < per_layer:
            raise ValueError(
                f"Layer {layer} has only {len(ks)} scored neurons, need {per_layer}."
            )
        chosen = rng.choice(ks, size=per_layer, replace=False).tolist()
        selected_ks.update(int(x) for x in chosen)
    return selected_ks


def build_active_poles_catalog(
    e3_neurons: Sequence[Sequence[int]],
    e3_explanations: Mapping[str, Mapping[str, Any]],
    e3_scores: Mapping[str, Any],
    e3_scores_opus: Mapping[str, Any] | None = None,
    seed: int = SEED_4B,
    per_layer_full_expl: int = 5,
    verify_expected_counts: bool = True,
) -> dict[str, Any]:
    """Classify all scored neurons into active poles and build the canonical catalog."""
    scored_list: list[dict[str, Any]] = list(e3_scores["neurons"])
    scored_list.sort(key=lambda r: int(r["k"]))

    opus_by_k: dict[int, dict[str, Any]] = {}
    if e3_scores_opus is not None:
        for rec in e3_scores_opus.get("neurons", []):
            opus_by_k[int(rec["k"])] = dict(rec)

    # Verify consistency across e3_neurons, e3_explanations, and e3_scores
    scored_by_k: dict[int, dict[str, Any]] = {}
    for rec in scored_list:
        k = int(rec["k"])
        layer = int(rec["layer"])
        neuron = int(rec["neuron"])
        if 0 <= k < len(e3_neurons):
            n_layer, n_idx = int(e3_neurons[k][0]), int(e3_neurons[k][1])
            if (n_layer, n_idx) != (layer, neuron):
                raise ValueError(
                    f"Mismatch for k={k}: e3_neurons={(n_layer, n_idx)} vs e3_scores={(layer, neuron)}"
                )
        expl_rec = e3_explanations.get(str(k))
        if expl_rec is None:
            raise KeyError(f"Missing explanation entry for k={k}")
        if (int(expl_rec["layer"]), int(expl_rec["neuron"])) != (layer, neuron):
            raise ValueError(
                f"Mismatch for k={k}: e3_explanations={(expl_rec['layer'], expl_rec['neuron'])} "
                f"vs e3_scores={(layer, neuron)}"
            )
        scored_by_k[k] = rec

    # Verify named neurons if full dataset
    named_by_k: dict[int, str] = {}
    if verify_expected_counts:
        for name, spec in NAMED_NEURONS_4B.items():
            k = int(spec["k"])
            if k not in scored_by_k:
                raise AssertionError(f"Named neuron {name!r} (k={k}) missing from e3_scores.")
            rec = scored_by_k[k]
            if (int(rec["layer"]), int(rec["neuron"])) != (int(spec["layer"]), int(spec["neuron"])):
                raise AssertionError(
                    f"Named neuron {name!r} (k={k}) expected L{spec['layer']} N{spec['neuron']}, "
                    f"got L{rec['layer']} N{rec['neuron']}"
                )
            named_by_k[k] = name
    else:
        for name, spec in NAMED_NEURONS_4B.items():
            k = int(spec["k"])
            if k in scored_by_k:
                rec = scored_by_k[k]
                if (int(rec["layer"]), int(rec["neuron"])) == (int(spec["layer"]), int(spec["neuron"])):
                    named_by_k[k] = name

    present_layers = sorted({int(r["layer"]) for r in scored_list})
    full_expl_ks = sample_stratified_full_expl_neurons(
        scored_list,
        layers=present_layers,
        per_layer=per_layer_full_expl,
        seed=seed,
    )

    neurons_out: list[dict[str, Any]] = []
    poles_out: list[dict[str, Any]] = []
    n_unipolar = 0
    n_bipolar = 0
    n_both_negligible = 0
    named_summary: dict[str, dict[str, Any]] = {}
    full_expl_by_layer: Counter[int] = Counter()

    score_fields = (
        "M",
        "mean",
        "mono",
        "freq",
        "n_windows",
        "corr",
        "corr_cal_weighted",
        "zero_rec",
        "zero_rec_active",
        "strong20_rec",
        "strong20_rec_active",
        "strong50_rec",
        "strong50_rec_active",
        "sim_raw_rec",
        "sim_raw_rec_active",
        "sim_cal_rec",
        "sim_cal_rec_active",
    )

    for rec in scored_list:
        k = int(rec["k"])
        layer = int(rec["layer"])
        neuron = int(rec["neuron"])
        neuron_id = make_neuron_id_4b(layer, neuron)
        expl = e3_explanations[str(k)]["expl"]

        label = str(expl.get("label", "")).strip()
        pos_text = str(expl.get("positive_pole", "")).strip()
        neg_text = str(expl.get("negative_pole", "")).strip()
        rubric = str(expl.get("rubric", "")).strip()
        full_expl_text = format_full_explanation(expl)

        pos_active = not is_negligible_pole_text(pos_text)
        neg_active = not is_negligible_pole_text(neg_text)

        active_signs: list[int] = []
        if pos_active:
            active_signs.append(1)
        if neg_active:
            active_signs.append(-1)

        if len(active_signs) == 2:
            n_bipolar += 1
        elif len(active_signs) == 1:
            n_unipolar += 1
        else:
            n_both_negligible += 1

        in_full_expl_50 = k in full_expl_ks
        if in_full_expl_50:
            full_expl_by_layer[layer] += 1

        named = named_by_k.get(k)
        ulto_scores = {f: rec.get(f) for f in score_fields if f in rec}
        opus_rec = opus_by_k.get(k)
        opus_scores = (
            {f: opus_rec.get(f) for f in score_fields if f in opus_rec}
            if opus_rec is not None
            else None
        )

        active_pole_ids = [make_pole_id_4b(layer, neuron, s) for s in active_signs]

        neuron_entry: dict[str, Any] = {
            "k": k,
            "neuron_id": neuron_id,
            "layer": layer,
            "neuron": neuron,
            "is_bipolar": len(active_signs) == 2,
            "n_active_poles": len(active_signs),
            "active_signs": active_signs,
            "active_pole_ids": active_pole_ids,
            "dominant_pole_id": active_pole_ids[0] if len(active_pole_ids) == 1 else None,
            "dominant_sign": active_signs[0] if len(active_signs) == 1 else None,
            "raw_dominant_sign": None,
            "p99_9_pos": None,
            "p99_9_neg": None,
            "in_full_expl_50": in_full_expl_50,
            "named_neuron": named,
            "label": label,
            "positive_pole": pos_text,
            "negative_pole": neg_text,
            "rubric": rubric,
            "full_explanation_text": full_expl_text,
            "ulto85_scores": ulto_scores,
            "opus_scores": opus_scores,
        }
        neurons_out.append(neuron_entry)

        if named is not None:
            named_summary[named] = {
                "k": k,
                "layer": layer,
                "neuron": neuron,
                "neuron_id": neuron_id,
                "is_bipolar": len(active_signs) == 2,
                "active_pole_ids": active_pole_ids,
                "in_full_expl_50": in_full_expl_50,
                "sim_cal_rec": ulto_scores.get("sim_cal_rec"),
                "sim_cal_rec_active": ulto_scores.get("sim_cal_rec_active"),
                "strong20_rec": ulto_scores.get("strong20_rec"),
                "corr": ulto_scores.get("corr"),
                "mono": ulto_scores.get("mono"),
            }

        for sign in active_signs:
            pole_id = make_pole_id_4b(layer, neuron, sign)
            pole_name = "pos" if sign > 0 else "neg"
            pole_text = pos_text if sign > 0 else neg_text
            poles_out.append(
                {
                    "pole_id": pole_id,
                    "neuron_id": neuron_id,
                    "k": k,
                    "layer": layer,
                    "neuron": neuron,
                    "sign": sign,
                    "pole_name": pole_name,
                    "is_bipolar": len(active_signs) == 2,
                    "n_active_poles": len(active_signs),
                    # Populated/confirmed by extract_4b.py via argmax(|p99.9(pos)|, |p99.9(neg)|)
                    "is_dominant_pole": len(active_signs) == 1,
                    "in_full_expl_50": in_full_expl_50,
                    "named_neuron": named,
                    "pole_text": pole_text,
                    "label": label,
                    "positive_pole": pos_text,
                    "negative_pole": neg_text,
                    "rubric": rubric,
                    "full_explanation_text": full_expl_text,
                    "ulto85_scores": ulto_scores,
                    "opus_scores": opus_scores,
                    # Populated by extract_4b.py
                    "frac_pos": None,
                    "frac_neg": None,
                    "p25_abs": None,
                    "band_edges_dir": None,
                    "quadrant_shares_corpus": None,
                    "quadrant_shares_by_band": None,
                }
            )

    if verify_expected_counts:
        assert len(neurons_out) == 278, f"Expected 278 scored neurons, got {len(neurons_out)}"
        assert n_unipolar == 132, f"Expected 132 unipolar neurons, got {n_unipolar}"
        assert n_bipolar == 146, f"Expected 146 bipolar neurons, got {n_bipolar}"
        assert n_both_negligible == 0, f"Expected 0 both-negligible neurons, got {n_both_negligible}"
        assert len(poles_out) == 424, f"Expected 424 active poles, got {len(poles_out)}"
        assert sum(full_expl_by_layer.values()) == 50, (
            f"Expected 50 full_expl neurons, got {sum(full_expl_by_layer.values())}"
        )
        for L in HOOK_LAYERS_4B:
            assert full_expl_by_layer[L] == 5, (
                f"Expected 5 full_expl neurons in layer {L}, got {full_expl_by_layer[L]}"
            )

    n_full_expl_poles = sum(1 for p in poles_out if p["in_full_expl_50"])
    n_full_expl_uni = sum(1 for n in neurons_out if n["in_full_expl_50"] and not n["is_bipolar"])
    n_full_expl_bi = sum(1 for n in neurons_out if n["in_full_expl_50"] and n["is_bipolar"])

    return {
        "schema_version": SCHEMA_VERSION_4B,
        "model_id": MODEL_ID_4B,
        "source_repo": "Ulto85/neurons-kinda-clean",
        "source_license": "MIT",
        "seed": seed,
        "layers": present_layers,
        "negligible_prefixes": list(NEGLIGIBLE_PREFIXES),
        "n_e3_neurons_total": len(e3_neurons),
        "n_scored_neurons": len(neurons_out),
        "n_opus_overlap_neurons": len(opus_by_k),
        "n_unipolar_neurons": n_unipolar,
        "n_bipolar_neurons": n_bipolar,
        "n_both_negligible_neurons": n_both_negligible,
        "n_active_poles": len(poles_out),
        "n_full_expl_50_neurons": sum(full_expl_by_layer.values()),
        "n_full_expl_50_unipolar": n_full_expl_uni,
        "n_full_expl_50_bipolar": n_full_expl_bi,
        "n_full_expl_50_poles": n_full_expl_poles,
        "full_expl_50_by_layer": {str(L): int(full_expl_by_layer[L]) for L in present_layers},
        "named_neurons": named_summary,
        "extraction_complete": False,
        "neurons": neurons_out,
        "poles": poles_out,
    }


def fetch_and_build(
    base_dir: Path | None = None,
    seed: int = SEED_4B,
    force_download: bool = False,
) -> dict[str, Any]:
    """Download Ulto85 JSONs and write `data/qwen4b/active_poles.json`."""
    dirs = ensure_4b_dirs(base_dir)
    paths = download_ulto85_files(dirs["ulto85"], force=force_download)

    e3_neurons = json.loads(paths["e3_neurons.json"].read_text(encoding="utf-8"))
    e3_explanations = json.loads(paths["e3_explanations.json"].read_text(encoding="utf-8"))
    e3_scores = json.loads(paths["e3_scores.json"].read_text(encoding="utf-8"))
    e3_scores_opus = json.loads(paths["e3_scores_opus.json"].read_text(encoding="utf-8"))

    catalog = build_active_poles_catalog(
        e3_neurons=e3_neurons,
        e3_explanations=e3_explanations,
        e3_scores=e3_scores,
        e3_scores_opus=e3_scores_opus,
        seed=seed,
        per_layer_full_expl=5,
        verify_expected_counts=True,
    )

    out_path = dirs["data"] / "active_poles.json"
    out_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return catalog


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch MIT-licensed Ulto85 Qwen3-4B neuron metadata and classify active poles."
    )
    parser.add_argument("--seed", type=int, default=SEED_4B)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    catalog = fetch_and_build(seed=args.seed, force_download=args.force_download)
    summary = {
        "n_scored_neurons": catalog["n_scored_neurons"],
        "n_unipolar_neurons": catalog["n_unipolar_neurons"],
        "n_bipolar_neurons": catalog["n_bipolar_neurons"],
        "n_both_negligible_neurons": catalog["n_both_negligible_neurons"],
        "n_active_poles": catalog["n_active_poles"],
        "n_opus_overlap_neurons": catalog["n_opus_overlap_neurons"],
        "n_full_expl_50_neurons": catalog["n_full_expl_50_neurons"],
        "n_full_expl_50_unipolar": catalog["n_full_expl_50_unipolar"],
        "n_full_expl_50_bipolar": catalog["n_full_expl_50_bipolar"],
        "n_full_expl_50_poles": catalog["n_full_expl_50_poles"],
        "named_neurons": catalog["named_neurons"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
