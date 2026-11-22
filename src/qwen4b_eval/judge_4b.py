"""1-pair-per-call OpenAI judge runner for exploratory Qwen3-4B (Ulto85) audit.

Strictly isolated from Qwen3-1.7B files:
- Reads inputs from `data/qwen4b/`
- Writes wrong-donor mapping to `data/qwen4b/wrong_donors_4b.json`
- Writes judgements strictly to `results/qwen4b/judgements_4b.jsonl`

Reuses the exact unified prompt template from `w_audit.config` (zero "strongly"),
exact boilerplate-stripped cross-layer wrong-donor matching (`cos <= 0.65`) from
`w_audit.labeler`, and exact single-pair OpenAI execution + outcome classification
(`scored` / `unparseable` / `logprob_tie`) from `w_audit.judge`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from w_audit.config import (
    DEFAULT_EMBED_MODEL,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    NO_LABEL_DESCRIPTION,
    SEED,
    WRONG_LABEL_EMBED_COSINE_MAX,
    WRONG_LABEL_JACCARD_MAX,
)
from w_audit.judge import (
    AdaptiveRateLimiter,
    JudgeTask,
    SpendLimitExceededError,
    _call_openai_once,
    load_completed_keys,
    sanitize_jsonl_tail,
)
from w_audit.labeler import (
    assign_wrong_labels,
    fetch_openai_embeddings,
    load_jsonl,
    strip_label_boilerplate,
)
from w_audit.probe_limits import (
    MODEL_PRICING_PER_M,
    compute_call_cost_usd,
    load_dotenv_file,
)

DEFAULT_ACTIVE_POLES_PATH: Path = Path("data/qwen4b/active_poles.json")
DEFAULT_EXPLANATIONS_PATH: Path = Path("data/qwen4b/ulto85/e3_explanations.json")
DEFAULT_SNIPPETS_4B_PATH: Path = Path("data/qwen4b/snippets_4b.jsonl")
DEFAULT_SNIPPETS_4B_TOP100_PATH: Path = Path("data/qwen4b/snippets_4b_top100.jsonl")
DEFAULT_WRONG_DONORS_4B_PATH: Path = Path("data/qwen4b/wrong_donors_4b.json")
DEFAULT_JUDGEMENTS_4B_PATH: Path = Path("results/qwen4b/judgements_4b.jsonl")
DEFAULT_JUDGEMENTS_4B_TOP100_PATH: Path = Path("results/qwen4b/judgements_4b_top100.jsonl")
DEFAULT_MAX_USD_4B: float = 15.0

ALL_SWEEPS: tuple[str, ...] = (
    "p999_own_terra",
    "p999_own_luna",
    "p999_floors_terra",
    "p97_99_own_terra",
    "p999_full_expl_terra",
)
TOP100_SWEEP: str = "top100_own_terra"
VALID_SWEEPS: tuple[str, ...] = ALL_SWEEPS + (TOP100_SWEEP,)

PROTECTED_17B_FILENAMES: frozenset[str] = frozenset(
    {
        "judgements.jsonl",
        "judgements_unfiltered_context_ablation.jsonl",
        "judgements_examples_as_label_exploratory.jsonl",
        "labels.jsonl",
        "snippets.jsonl",
        "poles.json",
    }
)


def _assert_isolated_output_path(path: Path) -> None:
    """Refuse writes targeting any frozen Qwen3-1.7B or exploratory root file."""
    if path.name in PROTECTED_17B_FILENAMES:
        raise ValueError(
            f"Refusing to write Qwen3-4B output into protected 1.7B file: {path}"
        )


def load_active_poles(active_poles_path: Path) -> list[dict[str, Any]]:
    """Load active_poles.json whether stored as a list or a wrapper dict."""
    raw = json.loads(active_poles_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(p) for p in raw]
    if isinstance(raw, dict):
        for key in ("poles", "active_poles", "items"):
            if isinstance(raw.get(key), list):
                return [dict(p) for p in raw[key]]
    raise ValueError(f"Unrecognized active_poles.json structure in {active_poles_path}")


def resolve_pole_sign(pole: Mapping[str, Any]) -> int:
    """Resolve signed pole direction (+1 or -1) cleanly across schema aliases."""
    if "sign" in pole and pole["sign"] is not None:
        s = int(pole["sign"])
        if s in (1, -1):
            return s
    pole_name = str(pole.get("pole", pole.get("direction", ""))).lower()
    if pole_name in ("pos", "positive", "+1", "+"):
        return 1
    if pole_name in ("neg", "negative", "-1", "-"):
        return -1
    pid = str(pole.get("pole_id", ""))
    if pid.endswith("_pos"):
        return 1
    if pid.endswith("_neg"):
        return -1
    raise ValueError(f"Cannot determine pole sign for {pole!r}")


def resolve_pole_own_text(
    pole: Mapping[str, Any],
    explanations_by_k: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Return the pole's own natural-language description (positive_pole / negative_pole)."""
    for key in ("pole_text", "own_label", "pole_description", "description", "text"):
        val = pole.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    sign = resolve_pole_sign(pole)
    field = "positive_pole" if sign > 0 else "negative_pole"
    val = pole.get(field)
    if isinstance(val, str) and val.strip():
        return val.strip()

    if explanations_by_k is not None and "k" in pole:
        k_entry = explanations_by_k.get(str(int(pole["k"])))
        if k_entry:
            expl = k_entry.get("expl", k_entry)
            val = expl.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()

    raise ValueError(f"Missing own pole text for pole_id={pole.get('pole_id')!r}")


def format_full_explanation_text(
    pole: Mapping[str, Any],
    explanations_by_k: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Format full Ulto85 neuron explanation for the 50-neuron full_explanation arm."""
    expl: Mapping[str, Any] = {}
    if explanations_by_k is not None and "k" in pole:
        k_entry = explanations_by_k.get(str(int(pole["k"])))
        if k_entry:
            expl = k_entry.get("expl", k_entry)

    label = str(pole.get("label") or expl.get("label") or "").strip()
    pos_pole = str(pole.get("positive_pole") or expl.get("positive_pole") or "").strip()
    neg_pole = str(pole.get("negative_pole") or expl.get("negative_pole") or "").strip()
    rubric = str(pole.get("rubric") or expl.get("rubric") or "").strip()

    if not (label or pos_pole or neg_pole or rubric):
        raise ValueError(
            f"Missing full explanation fields for pole_id={pole.get('pole_id')!r} (k={pole.get('k')!r})"
        )

    return (
        f"Summary: {label}\n"
        f"Positive pole: {pos_pole}\n"
        f"Negative pole: {neg_pole}\n"
        f"Rubric: {rubric}"
    )


def build_4b_messages(
    condition: str,
    description_text: str | None,
    snippet_a: str,
    snippet_b: str,
) -> list[dict[str, str]]:
    """Build 1-pair-per-call chat messages using the exact unified w_audit prompt template."""
    if not isinstance(snippet_a, str) or not isinstance(snippet_b, str):
        raise TypeError("Judge strictly accepts exactly one pair (snippet_a, snippet_b).")
    if condition == "no_label":
        desc = NO_LABEL_DESCRIPTION
    else:
        if not description_text or not description_text.strip():
            raise ValueError(f"Non-empty description required for condition={condition!r}")
        desc = description_text.strip()

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                description=desc,
                snippet_a=snippet_a,
                snippet_b=snippet_b,
            ),
        },
    ]


async def build_or_load_wrong_donors_4b_async(
    active_poles: Sequence[Mapping[str, Any]],
    explanations_by_k: Mapping[str, Mapping[str, Any]] | None = None,
    output_path: Path = DEFAULT_WRONG_DONORS_4B_PATH,
    embed_model: str = DEFAULT_EMBED_MODEL,
    max_embed_cosine: float = WRONG_LABEL_EMBED_COSINE_MAX,
    max_jaccard: float = WRONG_LABEL_JACCARD_MAX,
    seed: int = SEED,
    use_embeddings: bool = True,
    openai_client: Any = None,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build or load cross-layer boilerplate-stripped wrong-donor mapping (cos <= 0.65)."""
    _assert_isolated_output_path(output_path)
    pole_ids = {str(p["pole_id"]) for p in active_poles}

    if not force and output_path.is_file():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        donors = cached.get("donors", cached)
        if isinstance(donors, dict) and pole_ids.issubset(set(donors.keys())):
            return {pid: dict(donors[pid]) for pid in sorted(pole_ids)}

    own_records = [
        {
            "pole_id": str(p["pole_id"]),
            "layer": int(p["layer"]),
            "own_label": resolve_pole_own_text(p, explanations_by_k),
        }
        for p in sorted(active_poles, key=lambda r: str(r["pole_id"]))
    ]

    embeddings: np.ndarray | None = None
    embed_tokens = 0
    embed_cost_usd = 0.0
    owned_client = False
    client = openai_client

    if use_embeddings and len(own_records) > 0:
        texts = [strip_label_boilerplate(str(r["own_label"])) for r in own_records]
        if embed_fn is not None:
            embeddings = np.asarray(embed_fn(texts), dtype=np.float64)
        else:
            if client is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set; required for text-embedding-3-small wrong-donor matching."
                    )
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=api_key, timeout=60.0)
                owned_client = True
            try:
                emb_chunks: list[np.ndarray] = []
                embed_tokens = 0
                for start in range(0, len(texts), 200):
                    chunk_emb, chunk_toks = await fetch_openai_embeddings(
                        client, texts[start : start + 200], embed_model=embed_model
                    )
                    emb_chunks.append(np.asarray(chunk_emb, dtype=np.float64))
                    embed_tokens += int(chunk_toks)
                embeddings = np.vstack(emb_chunks) if emb_chunks else np.zeros((0, 0), dtype=np.float64)
                embed_cost_usd = compute_call_cost_usd(
                    embed_model, embed_tokens, 0, MODEL_PRICING_PER_M
                )
            finally:
                if owned_client and hasattr(client, "close"):
                    await client.close()

    wrong_map = assign_wrong_labels(
        own_records=own_records,
        embeddings=embeddings,
        max_embed_cosine=max_embed_cosine,
        max_jaccard=max_jaccard,
        seed=seed,
    )

    # Verify strict cross-layer invariant and summarize cosines
    cosines = [float(v["wrong_embed_cosine"]) for v in wrong_map.values()]
    relaxed_count = sum(1 for v in wrong_map.values() if v.get("wrong_threshold_relaxed"))
    same_layer_count = sum(
        1
        for r in own_records
        if int(wrong_map[str(r["pole_id"])]["wrong_donor_layer"]) == int(r["layer"])
    )
    if same_layer_count > 0:
        raise RuntimeError(f"Same-layer wrong donor violation: {same_layer_count}")

    payload = {
        "n_poles": len(wrong_map),
        "embed_model": embed_model if embeddings is not None else "word_jaccard_fallback",
        "max_embed_cosine_cutoff": max_embed_cosine,
        "seed": seed,
        "embed_tokens": embed_tokens,
        "embed_cost_usd": round(embed_cost_usd, 8),
        "same_layer_violations": same_layer_count,
        "threshold_relaxed_count": relaxed_count,
        "cosine_stats": {
            "min": float(np.min(cosines)) if cosines else 0.0,
            "median": float(np.median(cosines)) if cosines else 0.0,
            "p90": float(np.percentile(cosines, 90)) if cosines else 0.0,
            "max": float(np.max(cosines)) if cosines else 0.0,
        },
        "donors": wrong_map,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return wrong_map


def _snippet_marked_pair(snippet: Mapping[str, Any]) -> tuple[Literal["A", "B"], str, str]:
    """Extract (pos_slot, snippet_a, snippet_b) preserving exact snippet A/B order."""
    pos_slot: Literal["A", "B"] = "A" if str(snippet["pos_slot"]) == "A" else "B"
    if isinstance(snippet.get("pos"), Mapping) and isinstance(snippet.get("dist"), Mapping):
        pos_txt = str(snippet["pos"]["marked_text"])
        dist_txt = str(snippet["dist"]["marked_text"])
    else:
        pos_txt = str(snippet["pos_marked_text"])
        dist_txt = str(snippet["dist_marked_text"])
    snippet_a = pos_txt if pos_slot == "A" else dist_txt
    snippet_b = dist_txt if pos_slot == "A" else pos_txt
    return pos_slot, snippet_a, snippet_b


def _is_floor_6_snippet(snippet: Mapping[str, Any]) -> bool:
    """Return True if snippet belongs to the exact 3/3-balanced 6-pair floor subset."""
    if "in_floor_6" in snippet:
        return bool(snippet["in_floor_6"])
    return int(snippet.get("pair_idx", 999)) < 6


def build_4b_task_plan(
    active_poles: Sequence[Mapping[str, Any]],
    snippets: Sequence[Mapping[str, Any]],
    wrong_donors: Mapping[str, Mapping[str, Any]],
    explanations_by_k: Mapping[str, Mapping[str, Any]] | None = None,
    sweeps: Sequence[str] = ALL_SWEEPS,
) -> list[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]]:
    """Build deterministic single-pair task plan across requested 4B sweeps."""
    for sw in sweeps:
        if sw not in VALID_SWEEPS:
            raise ValueError(f"Unknown 4B sweep {sw!r}; valid: {VALID_SWEEPS}")

    poles_by_id = {str(p["pole_id"]): p for p in active_poles}
    own_text_by_pole = {
        pid: resolve_pole_own_text(p, explanations_by_k) for pid, p in poles_by_id.items()
    }
    full_expl_pole_ids = {
        pid for pid, p in poles_by_id.items() if bool(p.get("in_full_expl_50", False))
    }
    full_text_by_pole = {
        pid: format_full_explanation_text(poles_by_id[pid], explanations_by_k)
        for pid in full_expl_pole_ids
    }

    p999_snips = [
        s
        for s in snippets
        if str(s.get("band")) == "p99.9_plus" and str(s.get("pole_id")) in poles_by_id
    ]
    p97_99_snips = [
        s
        for s in snippets
        if str(s.get("band")) == "p97_99" and str(s.get("pole_id")) in poles_by_id
    ]

    plan: list[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    def _add_task(
        s: Mapping[str, Any],
        condition: str,
        api_model: str,
        desc_text: str | None,
    ) -> None:
        pid = str(s["pole_id"])
        pole = poles_by_id[pid]
        pos_slot, snip_a, snip_b = _snippet_marked_pair(s)
        judge_model_tag = f"{api_model}@none"
        task = JudgeTask(
            pair_id=str(s["pair_id"]),
            pole_id=pid,
            layer=int(pole["layer"]),
            band=str(s["band"]),
            distractor_type=str(s.get("distractor_type", "dead_zone")),
            condition=condition,
            pos_slot=pos_slot,
            snippet_a=snip_a,
            snippet_b=snip_b,
            label_text=desc_text,
            judge_model=judge_model_tag,
            judge_backend="openai",
            order="fwd",
            reasoning_effort="none",
            api_model=api_model,
            request_logprobs=True,
            max_completion_tokens=16,
        )
        if task.key in seen_keys:
            return
        seen_keys.add(task.key)
        messages = build_4b_messages(
            condition=condition,
            description_text=desc_text,
            snippet_a=snip_a,
            snippet_b=snip_b,
        )
        extra_meta = {
            "k": int(pole["k"]) if "k" in pole else None,
            "neuron": int(pole["neuron"]),
            "sign": resolve_pole_sign(pole),
            "pair_idx": int(s.get("pair_idx", -1)),
            "in_floor_6": _is_floor_6_snippet(s),
            "in_full_expl_50": bool(pole.get("in_full_expl_50", False)),
        }
        plan.append((task, messages, extra_meta))

    for sw in sweeps:
        if sw == "p999_own_terra":
            for s in p999_snips:
                pid = str(s["pole_id"])
                _add_task(s, "own", "gpt-6.1-terra", own_text_by_pole[pid])
        elif sw == "p999_own_luna":
            for s in p999_snips:
                pid = str(s["pole_id"])
                _add_task(s, "own", "gpt-6-luna", own_text_by_pole[pid])
        elif sw == "p999_floors_terra":
            for s in p999_snips:
                if not _is_floor_6_snippet(s):
                    continue
                pid = str(s["pole_id"])
                donor = wrong_donors.get(pid)
                if not donor:
                    raise KeyError(f"Missing wrong donor for pole_id={pid!r}")
                _add_task(s, "wrong", "gpt-6.1-terra", str(donor["wrong_label"]))
                _add_task(s, "no_label", "gpt-6.1-terra", None)
        elif sw == "p97_99_own_terra":
            for s in p97_99_snips:
                pid = str(s["pole_id"])
                _add_task(s, "own", "gpt-6.1-terra", own_text_by_pole[pid])
        elif sw == "p999_full_expl_terra":
            for s in p999_snips:
                pid = str(s["pole_id"])
                if pid not in full_expl_pole_ids:
                    continue
                _add_task(s, "full_explanation", "gpt-6.1-terra", full_text_by_pole[pid])
        elif sw == TOP100_SWEEP:
            for s in snippets:
                pid = str(s.get("pole_id", ""))
                if pid not in poles_by_id:
                    continue
                _add_task(s, "own", "gpt-6.1-terra", own_text_by_pole[pid])

    return plan


def estimate_prompt_tokens_conservative(messages: Sequence[Mapping[str, str]]) -> int:
    """Conservative offline token count estimate (~20% headroom above GPT BPE)."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return int(math.ceil(total_chars / 3.3)) + 32


def quote_4b_plan_cost(
    plan: Sequence[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]],
    assumed_completion_tokens: int = 8,
) -> dict[str, Any]:
    """Compute conservative 0%-cache pre-run USD quote broken down by model & condition."""
    by_bucket: dict[str, dict[str, Any]] = {}
    tot_usd = 0.0
    tot_in = 0
    tot_out = 0
    for task, msgs, _ in plan:
        in_tok = estimate_prompt_tokens_conservative(msgs)
        out_tok = assumed_completion_tokens
        cost = compute_call_cost_usd(task.wire_model, in_tok, out_tok, MODEL_PRICING_PER_M)
        tot_usd += cost
        tot_in += in_tok
        tot_out += out_tok
        bkey = f"{task.judge_model}:{task.band}:{task.condition}"
        if bkey not in by_bucket:
            by_bucket[bkey] = {"calls": 0, "est_prompt_tokens": 0, "est_usd": 0.0}
        by_bucket[bkey]["calls"] += 1
        by_bucket[bkey]["est_prompt_tokens"] += in_tok
        by_bucket[bkey]["est_usd"] += cost

    for b in by_bucket.values():
        b["est_usd"] = round(b["est_usd"], 6)
        b["est_mean_prompt_tokens"] = round(b["est_prompt_tokens"] / max(b["calls"], 1), 1)

    return {
        "n_calls": len(plan),
        "est_total_prompt_tokens": tot_in,
        "est_total_completion_tokens": tot_out,
        "est_total_usd_0pct_cache": round(tot_usd, 6),
        "by_bucket": by_bucket,
    }


def _existing_spend_usd(output_path: Path) -> float:
    """Sum already-paid USD inside output_path for cumulative cap safety."""
    if not output_path.is_file():
        return 0.0
    return sum(float(r.get("cost_usd", 0.0) or 0.0) for r in load_jsonl(output_path))


async def run_4b_judge_async(
    plan: Sequence[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]],
    output_path: Path = DEFAULT_JUDGEMENTS_4B_PATH,
    max_usd: float = DEFAULT_MAX_USD_4B,
    concurrency: int = 32,
    target_rpm: float = 3500.0,
    openai_client: Any = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Execute Qwen3-4B single-pair judge tasks into resumable append-only JSONL."""
    _assert_isolated_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)

    completed_keys = load_completed_keys(output_path)
    prior_usd = _existing_spend_usd(output_path)
    pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed_keys]

    quote = quote_4b_plan_cost(pending)
    if not pending:
        return {
            "total_planned": len(plan),
            "skipped_existing": len(plan),
            "completed_this_run": 0,
            "prior_usd": round(prior_usd, 6),
            "spent_usd_this_run": 0.0,
            "cumulative_usd": round(prior_usd, 6),
            "aborted_for_spend_cap": False,
            "pre_run_quote": quote,
        }

    if prior_usd >= max_usd:
        raise SpendLimitExceededError(
            f"Prior spend ${prior_usd:.4f} already meets/exceeds --max-usd ${max_usd:.2f}"
        )

    client = openai_client
    close_client = False
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment or .env.")
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        close_client = True

    limiter = AdaptiveRateLimiter(target_rpm=target_rpm)
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    state_lock = asyncio.Lock()

    spent_this_run = 0.0
    prompt_toks = 0
    comp_toks = 0
    reas_toks = 0
    vis_toks = 0
    completed_now = 0
    aborted = False

    try:
        with output_path.open("a", encoding="utf-8") as fh:

            async def _worker(
                task: JudgeTask,
                messages: list[dict[str, str]],
                extra_meta: dict[str, Any],
            ) -> None:
                nonlocal spent_this_run, prompt_toks, comp_toks, reas_toks, vis_toks, completed_now, aborted
                async with sem:
                    async with state_lock:
                        if aborted or (prior_usd + spent_this_run) >= max_usd:
                            aborted = True
                            return
                    row = await _call_openai_once(
                        client=client,
                        task=task,
                        messages=messages,
                        limiter=limiter,
                    )
                    row.update(extra_meta)
                    async with write_lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                    async with state_lock:
                        spent_this_run += float(row["cost_usd"])
                        prompt_toks += int(row["prompt_tokens"])
                        comp_toks += int(row["completion_tokens"])
                        reas_toks += int(row.get("reasoning_tokens", 0))
                        vis_toks += int(row.get("visible_output_tokens", 0))
                        completed_now += 1
                        if progress_every > 0 and (
                            completed_now % progress_every == 0 or completed_now == len(pending)
                        ):
                            print(
                                f"[judge_4b] {completed_now:,}/{len(pending):,} done | "
                                f"run=${spent_this_run:.4f} | cum=${prior_usd + spent_this_run:.4f}",
                                flush=True,
                            )
                        if (prior_usd + spent_this_run) >= max_usd:
                            aborted = True

            await asyncio.gather(*(_worker(t, m, meta) for t, m, meta in pending))
    finally:
        if close_client and hasattr(client, "close"):
            await client.close()

    return {
        "total_planned": len(plan),
        "skipped_existing": len(plan) - len(pending),
        "completed_this_run": completed_now,
        "prior_usd": round(prior_usd, 6),
        "spent_usd_this_run": round(spent_this_run, 6),
        "cumulative_usd": round(prior_usd + spent_this_run, 6),
        "prompt_tokens_this_run": prompt_toks,
        "completion_tokens_this_run": comp_toks,
        "reasoning_tokens_this_run": reas_toks,
        "visible_output_tokens_this_run": vis_toks,
        "aborted_for_spend_cap": aborted,
        "pre_run_quote": quote,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 1-pair-per-call OpenAI detection judge sweeps for exploratory Qwen3-4B (Ulto85) audit."
    )
    parser.add_argument("--active-poles", type=Path, default=DEFAULT_ACTIVE_POLES_PATH)
    parser.add_argument("--explanations", type=Path, default=DEFAULT_EXPLANATIONS_PATH)
    parser.add_argument("--snippets", type=Path, default=DEFAULT_SNIPPETS_4B_PATH)
    parser.add_argument("--wrong-donors", type=Path, default=DEFAULT_WRONG_DONORS_4B_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JUDGEMENTS_4B_PATH)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=list(VALID_SWEEPS),
        default=list(ALL_SWEEPS),
        help="Subset of 4B sweeps to run (default: all 5 core sweeps).",
    )
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD_4B)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--target-rpm", type=float, default=3500.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-wrong-donors", action="store_true")
    args = parser.parse_args(argv)
    if TOP100_SWEEP in args.sweeps and args.output.name == DEFAULT_JUDGEMENTS_4B_PATH.name:
        raise ValueError(
            "Refusing to write top100_own_terra into judgements_4b.jsonl; pass --output results/qwen4b/judgements_4b_top100.jsonl."
        )

    load_dotenv_file(args.env_file)
    active_poles = load_active_poles(args.active_poles)
    explanations_by_k = (
        json.loads(args.explanations.read_text(encoding="utf-8"))
        if args.explanations.is_file()
        else None
    )
    snippets = load_jsonl(args.snippets)

    need_wrong = "p999_floors_terra" in args.sweeps
    wrong_donors: dict[str, dict[str, Any]] = {}
    if need_wrong or args.wrong_donors.is_file():
        wrong_donors = asyncio.run(
            build_or_load_wrong_donors_4b_async(
                active_poles=active_poles,
                explanations_by_k=explanations_by_k,
                output_path=args.wrong_donors,
                seed=args.seed,
                force=args.force_wrong_donors,
            )
        )

    plan = build_4b_task_plan(
        active_poles=active_poles,
        snippets=snippets,
        wrong_donors=wrong_donors,
        explanations_by_k=explanations_by_k,
        sweeps=args.sweeps,
    )
    completed = load_completed_keys(args.output)
    pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed]
    quote = quote_4b_plan_cost(pending)
    print(
        json.dumps(
            {
                "sweeps": list(args.sweeps),
                "total_planned": len(plan),
                "already_completed": len(plan) - len(pending),
                "pending": len(pending),
                "max_usd": args.max_usd,
                "pre_run_quote": quote,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    summary = asyncio.run(
        run_4b_judge_async(
            plan=plan,
            output_path=args.output,
            max_usd=args.max_usd,
            concurrency=args.concurrency,
            target_rpm=args.target_rpm,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
