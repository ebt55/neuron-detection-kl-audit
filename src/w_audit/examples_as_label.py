"""Exploratory runner: 20 top-band examples as label.

Isolated from the frozen main evaluation sweep.

Evaluates whether giving the judge (`gpt-6.1-terra@none`) the pole's 20 disjoint
top-band `train_label` examples directly (in place of the GPT-4.1 natural-language
summary label) improves or matches `own` detection accuracy on `p97_99.9` and
`p99.9_plus` (`dead_zone`, exact same 1,200 pairs and exact same `pos_slot` A/B
presentation order from `data/snippets.jsonl`).

Outputs STRICTLY to `results/judgements_examples_as_label_exploratory.jsonl`
(NEVER writes to `results/judgements.jsonl`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from w_audit.config import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
)
from w_audit.judge import (
    AdaptiveRateLimiter,
    JudgeTask,
    SpendLimitExceededError,
    _call_openai_once,
    load_completed_keys,
    sanitize_jsonl_tail,
)
from w_audit.labeler import load_jsonl
from w_audit.probe_limits import (
    MODEL_PRICING_PER_M,
    compute_call_cost_usd,
    load_dotenv_file,
)

EXPLORATORY_BANDS: tuple[str, ...] = ("p97_99.9", "p99.9_plus")
EXPLORATORY_CONDITION: str = "examples_as_label"
DEFAULT_EXPLORATORY_OUTPUT: Path = Path("results/judgements_examples_as_label_exploratory.jsonl")
DEFAULT_MAX_USD: float = 4.0

EXAMPLES_HEADER: str = (
    "No summary label provided. Below are 20 top-band examples where this neuron "
    "pole activates at the marked <<...>> token:"
)


def format_examples_as_label_description(
    examples: Sequence[Mapping[str, Any]],
) -> str:
    """Format a pole's 20 disjoint top-band train_label snippets sorted by act_dir desc."""
    ordered = sorted(
        examples,
        key=lambda r: float(r.get("act_dir", abs(float(r.get("act_val", 0.0))))),
        reverse=True,
    )
    if not ordered:
        raise ValueError("examples_as_label requires non-empty train_label examples.")
    lines = [EXAMPLES_HEADER]
    for idx, ex in enumerate(ordered, start=1):
        marked = str(ex["marked_text"]).replace("\r\n", "\n")
        lines.append(f"[{idx}] {marked}")
    return "\n".join(lines)


def build_examples_as_label_messages(
    examples: Sequence[Mapping[str, Any]],
    snippet_a: str,
    snippet_b: str,
) -> list[dict[str, str]]:
    """Build 1-pair-per-call messages using the exact unified judge prompt template."""
    if not isinstance(snippet_a, str) or not isinstance(snippet_b, str):
        raise TypeError("Judge strictly accepts exactly one pair (snippet_a, snippet_b).")
    description = format_examples_as_label_description(examples)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                description=description,
                snippet_a=snippet_a,
                snippet_b=snippet_b,
            ),
        },
    ]


def estimate_prompt_tokens_conservative(messages: Sequence[Mapping[str, str]]) -> int:
    """Conservative offline token count estimate (~20% headroom over GPT BPE ~3.9 chars/tok)."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    # 3.2 chars/token + 32 message framing tokens is conservatively above real OpenAI counts
    return int(math.ceil(total_chars / 3.2)) + 32


def build_examples_task_plan(
    snippets: Sequence[Mapping[str, Any]],
    examples_by_pole: Mapping[str, Sequence[Mapping[str, Any]]],
    api_model: str = "gpt-6.1-terra",
    reasoning_effort: str = "none",
    bands: Sequence[str] = EXPLORATORY_BANDS,
) -> list[tuple[JudgeTask, list[dict[str, str]]]]:
    """Build exact 1,200 single-pair tasks preserving data/snippets.jsonl pair_id & pos_slot."""
    band_set = set(bands)
    judge_model_tag = f"{api_model}@{reasoning_effort}"
    plan: list[tuple[JudgeTask, list[dict[str, str]]]] = []

    for s in snippets:
        if str(s["distractor_type"]) != "dead_zone":
            continue
        if str(s["band"]) not in band_set:
            continue
        pid = str(s["pole_id"])
        ex_list = examples_by_pole.get(pid)
        if not ex_list:
            raise KeyError(f"Missing label_examples for pole_id={pid!r}")

        pos_slot: Literal["A", "B"] = "A" if str(s["pos_slot"]) == "A" else "B"
        pos_txt = str(s["pos"]["marked_text"])
        dist_txt = str(s["dist"]["marked_text"])
        snippet_a = pos_txt if pos_slot == "A" else dist_txt
        snippet_b = dist_txt if pos_slot == "A" else pos_txt

        messages = build_examples_as_label_messages(
            examples=ex_list,
            snippet_a=snippet_a,
            snippet_b=snippet_b,
        )
        task = JudgeTask(
            pair_id=str(s["pair_id"]),
            pole_id=pid,
            layer=int(s["layer"]),
            band=str(s["band"]),
            distractor_type="dead_zone",
            condition=EXPLORATORY_CONDITION,
            pos_slot=pos_slot,
            snippet_a=snippet_a,
            snippet_b=snippet_b,
            label_text=None,
            judge_model=judge_model_tag,
            judge_backend="openai",
            order="fwd",
            reasoning_effort=reasoning_effort,
            api_model=api_model,
            request_logprobs=True,
            max_completion_tokens=16,
        )
        plan.append((task, messages))

    return plan


def quote_examples_plan_cost(
    plan: Sequence[tuple[JudgeTask, list[dict[str, str]]]],
    api_model: str = "gpt-6.1-terra",
    assumed_completion_tokens: int = 16,
) -> dict[str, Any]:
    """Compute conservative 0%-cache pre-run USD quote over all planned prompts."""
    prompt_toks = [estimate_prompt_tokens_conservative(msgs) for _, msgs in plan]
    tot_in = sum(prompt_toks)
    tot_out = len(plan) * assumed_completion_tokens
    est_usd = compute_call_cost_usd(api_model, tot_in, tot_out, MODEL_PRICING_PER_M)
    return {
        "n_calls": len(plan),
        "api_model": api_model,
        "est_total_prompt_tokens": tot_in,
        "est_mean_prompt_tokens": (tot_in / len(plan)) if plan else 0.0,
        "assumed_completion_tokens_per_call": assumed_completion_tokens,
        "est_total_usd_0pct_cache": round(est_usd, 6),
        "est_usd_per_call": round((est_usd / len(plan)) if plan else 0.0, 8),
    }


async def run_examples_tasks_async(
    plan: Sequence[tuple[JudgeTask, list[dict[str, str]]]],
    output_path: Path = DEFAULT_EXPLORATORY_OUTPUT,
    max_usd: float = DEFAULT_MAX_USD,
    concurrency: int = 20,
    target_rpm: float = 480.0,
    openai_client: Any = None,
) -> dict[str, Any]:
    """Execute exploratory examples_as_label calls into a strictly separate JSONL file."""
    if output_path.name == "judgements.jsonl":
        raise ValueError(
            "Refusing to write exploratory examples_as_label rows into frozen judgements.jsonl!"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)
    completed_keys = load_completed_keys(output_path)
    pending = [(t, m) for t, m in plan if t.key not in completed_keys]

    quote = quote_examples_plan_cost(
        pending,
        api_model=pending[0][0].wire_model if pending else "gpt-6.1-terra",
    )
    if quote["est_total_usd_0pct_cache"] > max_usd:
        raise SpendLimitExceededError(
            f"Pre-run conservative 0%-cache quote ${quote['est_total_usd_0pct_cache']:.4f} "
            f"exceeds hard --max-usd ${max_usd:.2f}; aborting before any live API call."
        )

    if not pending:
        return {
            "total_planned": len(plan),
            "skipped_existing": len(plan),
            "completed_this_run": 0,
            "spent_usd_this_run": 0.0,
            "prompt_tokens_this_run": 0,
            "completion_tokens_this_run": 0,
            "reasoning_tokens_this_run": 0,
            "visible_output_tokens_this_run": 0,
            "aborted_for_spend_cap": False,
            "pre_run_quote": quote,
        }

    client = openai_client
    close_client = False
    if client is None:
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        close_client = True

    limiter = AdaptiveRateLimiter(target_rpm=target_rpm)
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    state_lock = asyncio.Lock()

    spent_usd = 0.0
    prompt_toks = 0
    comp_toks = 0
    reas_toks = 0
    vis_toks = 0
    completed_now = 0
    aborted = False

    try:
        with output_path.open("a", encoding="utf-8") as fh:
            async def _worker(task: JudgeTask, messages: list[dict[str, str]]) -> None:
                nonlocal spent_usd, prompt_toks, comp_toks, reas_toks, vis_toks, completed_now, aborted
                async with sem:
                    async with state_lock:
                        if aborted or spent_usd >= max_usd:
                            aborted = True
                            return
                    row = await _call_openai_once(
                        client=client,
                        task=task,
                        messages=messages,
                        limiter=limiter,
                    )
                    async with write_lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                    async with state_lock:
                        spent_usd += float(row["cost_usd"])
                        prompt_toks += int(row["prompt_tokens"])
                        comp_toks += int(row["completion_tokens"])
                        reas_toks += int(row.get("reasoning_tokens", 0))
                        vis_toks += int(row.get("visible_output_tokens", 0))
                        completed_now += 1
                        if spent_usd >= max_usd:
                            aborted = True

            await asyncio.gather(*(_worker(t, m) for t, m in pending))
    finally:
        if close_client and hasattr(client, "close"):
            await client.close()

    return {
        "total_planned": len(plan),
        "skipped_existing": len(plan) - len(pending),
        "completed_this_run": completed_now,
        "spent_usd_this_run": round(spent_usd, 6),
        "prompt_tokens_this_run": prompt_toks,
        "completion_tokens_this_run": comp_toks,
        "reasoning_tokens_this_run": reas_toks,
        "visible_output_tokens_this_run": vis_toks,
        "aborted_for_spend_cap": aborted,
        "pre_run_quote": quote,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory runner: score p97_99.9 and p99.9_plus "
            "dead_zone pairs using 20 disjoint top-band train_label examples as label."
        )
    )
    parser.add_argument("--snippets", type=Path, default=Path("data/snippets.jsonl"))
    parser.add_argument("--label-examples", type=Path, default=Path("data/label_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPLORATORY_OUTPUT)
    parser.add_argument("--model", type=str, default="gpt-6.1-terra")
    parser.add_argument("--reasoning-effort", type=str, default="none")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--target-rpm", type=float, default=480.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv_file(args.env_file)
    snippets = load_jsonl(args.snippets)
    raw_examples = load_jsonl(args.label_examples)
    examples_by_pole: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_examples:
        examples_by_pole[str(row["pole_id"])].append(row)

    plan = build_examples_task_plan(
        snippets=snippets,
        examples_by_pole=examples_by_pole,
        api_model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    completed = load_completed_keys(args.output) if args.output.is_file() else set()
    pending = [(t, m) for t, m in plan if t.key not in completed]
    quote = quote_examples_plan_cost(pending, api_model=args.model)

    print(
        f"[examples_as_label] model={args.model}@{args.reasoning_effort} "
        f"planned={len(plan):,} existing={len(plan) - len(pending):,} pending={len(pending):,} "
        f"est_mean_prompt_tok={quote['est_mean_prompt_tokens']:.1f} "
        f"est_usd_0pct_cache=${quote['est_total_usd_0pct_cache']:.4f} max_usd=${args.max_usd:.2f}"
    )
    if args.dry_run:
        if quote["est_total_usd_0pct_cache"] > args.max_usd:
            print("ABORT: conservative 0%-cache quote exceeds --max-usd!")
            return 2
        return 0

    summary = asyncio.run(
        run_examples_tasks_async(
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
