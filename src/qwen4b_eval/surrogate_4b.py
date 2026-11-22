"""Multi-band detection, YES/NO precision, and causal surrogate benchmark for Qwen3-4B.

Strictly isolated from Qwen3-1.7B outputs and prior Qwen3-4B JSONLs:
- Reads inputs from `data/qwen4b/` and `results/qwen4b/` (read-only)
- Writes strictly to:
  - `results/qwen4b/judgements_4b_multiband.jsonl`
  - `results/qwen4b/judgements_4b_fpr.jsonl`
  - `results/qwen4b/surrogate_metrics_4b.json`
  - `results/qwen4b/surrogate_summary_4b.md`
  - `figures/qwen4b_gap_decomposition.png`
  - `figures/qwen4b_surrogate_comparison.png`

All tables, summaries, and figure titles are strictly factual (zero interpretive prose).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr, t as student_t

from qwen4b_eval.analyze_4b import (
    NAMED_NEURONS,
    PRIMARY_JUDGE_4B,
    SECONDARY_JUDGE_4B,
    ULTO85_CREDIT_FOOTER,
    _dedup_judgements_4b,
    _fmt_ci,
    _fmt_num,
    _fmt_signed,
    _safe_pearson,
    _safe_spearman,
    bootstrap_corr_ci,
)
from qwen4b_eval.judge_4b import (
    DEFAULT_ACTIVE_POLES_PATH,
    DEFAULT_EXPLANATIONS_PATH,
    DEFAULT_JUDGEMENTS_4B_PATH,
    DEFAULT_JUDGEMENTS_4B_TOP100_PATH,
    DEFAULT_SNIPPETS_4B_PATH,
    PROTECTED_17B_FILENAMES,
    _snippet_marked_pair,
    build_4b_messages,
    estimate_prompt_tokens_conservative,
    load_active_poles,
    quote_4b_plan_cost,
    resolve_pole_own_text,
    resolve_pole_sign,
)
from w_audit.analyze import DEFAULT_BOOTSTRAP_B, classify_row_outcome
from w_audit.config import SEED
from w_audit.judge import (
    LOGPROB_TIE_EPS,
    AdaptiveRateLimiter,
    JudgeTask,
    SpendLimitExceededError,
    _call_openai_once,
    _extract_retry_after,
    _is_rate_limit_error,
    _logsumexp,
    load_completed_keys,
    sanitize_jsonl_tail,
)
from w_audit.labeler import load_jsonl
from w_audit.probe_limits import (
    MODEL_PRICING_PER_M,
    compute_call_cost_usd,
    load_dotenv_file,
)

# ---------------------------------------------------------------------------
# Paths & Isolation Guards
# ---------------------------------------------------------------------------

DEFAULT_SCORES_4B_PATH: Path = Path("data/qwen4b/ulto85/e3_scores.json")
DEFAULT_SNIPPETS_MULTIBAND_PATH: Path = Path("data/qwen4b/snippets_4b_multiband.jsonl")
DEFAULT_SNIPPETS_FPR_PATH: Path = Path("data/qwen4b/snippets_4b_fpr.jsonl")
DEFAULT_GAP_DECOMP_PATH: Path = Path("results/qwen4b/gap_decomposition_4b.json")
DEFAULT_BAND_KL_PATH: Path = Path("results/qwen4b/band_kl_4b.json")

DEFAULT_JUDGEMENTS_MULTIBAND_PATH: Path = Path("results/qwen4b/judgements_4b_multiband.jsonl")
DEFAULT_JUDGEMENTS_FPR_PATH: Path = Path("results/qwen4b/judgements_4b_fpr.jsonl")
DEFAULT_SURROGATE_METRICS_PATH: Path = Path("results/qwen4b/surrogate_metrics_4b.json")
DEFAULT_SURROGATE_SUMMARY_PATH: Path = Path("results/qwen4b/surrogate_summary_4b.md")
DEFAULT_FIGURE_GAP_PATH: Path = Path("figures/qwen4b_gap_decomposition.png")
DEFAULT_FIGURE_SURROGATE_PATH: Path = Path("figures/qwen4b_surrogate_comparison.png")

DEFAULT_SURROGATE_MAX_USD: float = 8.0

PROTECTED_IMMUTABLE_FILENAMES: frozenset[str] = PROTECTED_17B_FILENAMES | frozenset(
    {
        "judgements_4b.jsonl",
        "judgements_4b_top100.jsonl",
        "metrics_4b.json",
        "summary_4b.md",
        "README.md",
        "active_poles.json",
        "wrong_donors_4b.json",
        "snippets_4b.jsonl",
        "snippets_4b_top100.jsonl",
        "snippets_4b_multiband.jsonl",
        "snippets_4b_fpr.jsonl",
        "gap_decomposition_4b.json",
        "band_kl_4b.json",
    }
)

MULTIBAND_SWEEPS: tuple[str, ...] = (
    "multiband_own_terra",
    "multiband_own_luna",
    "p97_99_own_luna",
)

FPR_SWEEPS: tuple[str, ...] = (
    "fpr_core_terra",
    "fpr_all_luna",
)

# Exact single-snippet YES/NO prompt template (zero "strongly")
FPR_SYSTEM_PROMPT: str = (
    "You evaluate neuron pole descriptions on single text snippets. "
    "Respond with only YES or NO."
)
FPR_USER_TEMPLATE: str = (
    "Neuron pole description: {description}\n\n"
    "Text snippet (target token marked inside <<...>>):\n"
    "{snippet}\n\n"
    "Does this neuron pole activate on the marked <<...>> token according to the "
    "description? Answer with only YES or NO."
)

FAILURE_BUCKET_ORDER: tuple[str, ...] = (
    "pass_causal",
    "overbroad_fp_damage",
    "weak_mass_dominated",
    "active_sim_failure",
    "other_partial",
)

FAILURE_BUCKET_LABELS: dict[str, str] = {
    "pass_causal": "Pass causal\n(sim>=0.50)",
    "overbroad_fp_damage": "Overbroad FP\n(sim<0, act>=0.50)",
    "weak_mass_dominated": "Weak-mass dom.\n(strong20<0.25)",
    "active_sim_failure": "Active-sim fail\n(act<0.50)",
    "other_partial": "Other partial\n(0<=sim<0.50)",
}

RIDGE_ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def assert_surrogate_output_path(path: Path) -> None:
    """Refuse writes targeting any frozen 1.7B or prior 4B artifact."""
    if path.name in PROTECTED_IMMUTABLE_FILENAMES:
        raise ValueError(f"Refusing to write surrogate output into protected file: {path}")


def _sum_existing_spend(paths: Sequence[Path]) -> float:
    """Sum exact cost_usd across one or more append-only JSONL files."""
    seen_paths: set[Path] = set()
    total = 0.0
    for p in paths:
        rp = p.resolve()
        if rp in seen_paths or not p.is_file():
            continue
        seen_paths.add(rp)
        for row in load_jsonl(p):
            total += float(row.get("cost_usd", 0.0) or 0.0)
    return total


# ---------------------------------------------------------------------------
# Part 1: Multi-Band 2AFC Detection Runner
# ---------------------------------------------------------------------------


def build_multiband_task_plan(
    active_poles: Sequence[Mapping[str, Any]],
    multiband_snippets: Sequence[Mapping[str, Any]],
    snippets_4b: Sequence[Mapping[str, Any]] | None = None,
    explanations_by_k: Mapping[str, Mapping[str, Any]] | None = None,
    sweeps: Sequence[str] = MULTIBAND_SWEEPS,
) -> list[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]]:
    """Build deterministic 1-pair-per-call multi-band detection plan."""
    for sw in sweeps:
        if sw not in MULTIBAND_SWEEPS:
            raise ValueError(f"Unknown multiband sweep {sw!r}; valid: {MULTIBAND_SWEEPS}")

    poles_by_id = {str(p["pole_id"]): p for p in active_poles}
    own_text_by_pole = {
        pid: resolve_pole_own_text(p, explanations_by_k) for pid, p in poles_by_id.items()
    }

    mb_valid = [s for s in multiband_snippets if str(s.get("pole_id")) in poles_by_id]
    p97_99_snips = (
        [
            s
            for s in snippets_4b
            if str(s.get("band")) == "p97_99" and str(s.get("pole_id")) in poles_by_id
        ]
        if snippets_4b is not None
        else []
    )

    plan: list[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    def _add(s: Mapping[str, Any], api_model: str) -> None:
        pid = str(s["pole_id"])
        pole = poles_by_id[pid]
        pos_slot, snip_a, snip_b = _snippet_marked_pair(s)
        desc = own_text_by_pole[pid]
        judge_model_tag = f"{api_model}@none"
        task = JudgeTask(
            pair_id=str(s["pair_id"]),
            pole_id=pid,
            layer=int(pole["layer"]),
            band=str(s["band"]),
            distractor_type=str(s.get("distractor_type", "dead_zone")),
            condition="own",
            pos_slot=pos_slot,
            snippet_a=snip_a,
            snippet_b=snip_b,
            label_text=desc,
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
            condition="own",
            description_text=desc,
            snippet_a=snip_a,
            snippet_b=snip_b,
        )
        extra_meta = {
            "k": int(pole["k"]) if "k" in pole else None,
            "neuron": int(pole["neuron"]),
            "sign": resolve_pole_sign(pole),
            "pair_idx": int(s.get("pair_idx", -1)),
            "pos_window_tier": s.get("pos_window_tier"),
        }
        plan.append((task, messages, extra_meta))

    for sw in sweeps:
        if sw == "multiband_own_terra":
            for s in mb_valid:
                _add(s, "gpt-6.1-terra")
        elif sw == "multiband_own_luna":
            for s in mb_valid:
                _add(s, "gpt-6-luna")
        elif sw == "p97_99_own_luna":
            for s in p97_99_snips:
                _add(s, "gpt-6-luna")

    return plan


async def run_multiband_judge_async(
    plan: Sequence[tuple[JudgeTask, list[dict[str, str]], dict[str, Any]]],
    output_path: Path = DEFAULT_JUDGEMENTS_MULTIBAND_PATH,
    companion_spend_paths: Sequence[Path] = (DEFAULT_JUDGEMENTS_FPR_PATH,),
    max_usd: float = DEFAULT_SURROGATE_MAX_USD,
    concurrency: int = 32,
    target_rpm: float = 3500.0,
    openai_client: Any = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Execute multi-band 2AFC detection calls with hard combined USD cap."""
    assert_surrogate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)

    completed_keys = load_completed_keys(output_path)
    prior_self_usd = _sum_existing_spend([output_path])
    prior_combined_usd = _sum_existing_spend([output_path, *companion_spend_paths])
    pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed_keys]
    quote = quote_4b_plan_cost(pending)

    if not pending:
        return {
            "total_planned": len(plan),
            "skipped_existing": len(plan),
            "completed_this_run": 0,
            "prior_self_usd": round(prior_self_usd, 6),
            "prior_combined_usd": round(prior_combined_usd, 6),
            "spent_usd_this_run": 0.0,
            "cumulative_self_usd": round(prior_self_usd, 6),
            "cumulative_combined_usd": round(prior_combined_usd, 6),
            "aborted_for_spend_cap": False,
            "pre_run_quote": quote,
        }

    if prior_combined_usd >= max_usd:
        raise SpendLimitExceededError(
            f"Prior combined surrogate-sweep spend ${prior_combined_usd:.4f} >= --max-usd ${max_usd:.2f}"
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
    completed_now = 0
    aborted = False

    try:
        with output_path.open("a", encoding="utf-8") as fh:

            async def _worker(
                task: JudgeTask,
                messages: list[dict[str, str]],
                extra_meta: dict[str, Any],
            ) -> None:
                nonlocal spent_this_run, prompt_toks, comp_toks, reas_toks, completed_now, aborted
                async with sem:
                    async with state_lock:
                        if aborted or (prior_combined_usd + spent_this_run) >= max_usd:
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
                        completed_now += 1
                        if progress_every > 0 and (
                            completed_now % progress_every == 0 or completed_now == len(pending)
                        ):
                            print(
                                f"[multiband_4b] {completed_now:,}/{len(pending):,} done | "
                                f"run=${spent_this_run:.4f} | combined=${prior_combined_usd + spent_this_run:.4f}",
                                flush=True,
                            )
                        if (prior_combined_usd + spent_this_run) >= max_usd:
                            aborted = True

            await asyncio.gather(*(_worker(t, m, meta) for t, m, meta in pending))
    finally:
        if close_client and hasattr(client, "close"):
            await client.close()

    return {
        "total_planned": len(plan),
        "skipped_existing": len(plan) - len(pending),
        "completed_this_run": completed_now,
        "prior_self_usd": round(prior_self_usd, 6),
        "prior_combined_usd": round(prior_combined_usd, 6),
        "spent_usd_this_run": round(spent_this_run, 6),
        "cumulative_self_usd": round(prior_self_usd + spent_this_run, 6),
        "cumulative_combined_usd": round(prior_combined_usd + spent_this_run, 6),
        "prompt_tokens_this_run": prompt_toks,
        "completion_tokens_this_run": comp_toks,
        "reasoning_tokens_this_run": reas_toks,
        "aborted_for_spend_cap": aborted,
        "pre_run_quote": quote,
    }


# ---------------------------------------------------------------------------
# Part 2: Single-Snippet YES/NO Precision Audit Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FprProbeTask:
    """Single-snippet YES/NO precision task."""

    probe_id: str
    pole_id: str
    k: int
    layer: int
    neuron: int
    sign: int
    probe_type: Literal["top100_firing", "ordinary_nonfiring"]
    ground_truth: Literal["YES", "NO"]
    tier: str
    probe_idx: int
    snippet_text: str
    description_text: str
    judge_model: str
    api_model: str
    reasoning_effort: str = "none"
    max_completion_tokens: int = 8

    @property
    def key(self) -> tuple[str, str]:
        return (self.probe_id, self.judge_model)


def build_fpr_messages(description_text: str, snippet_text: str) -> list[dict[str, str]]:
    """Build single-snippet YES/NO chat messages (strictly zero 'strongly')."""
    if not description_text or not description_text.strip():
        raise ValueError("Non-empty pole description required for YES/NO precision probe.")
    if not snippet_text or not snippet_text.strip():
        raise ValueError("Non-empty snippet text required for YES/NO precision probe.")
    return [
        {"role": "system", "content": FPR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FPR_USER_TEMPLATE.format(
                description=description_text.strip(),
                snippet=snippet_text,
            ),
        },
    ]


def extract_yesno_logprobs(
    top_logprobs: Sequence[Any] | None,
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    """Extract (logprob_YES, logprob_NO, yes_margin, p_yes, imputed) from top-5 logprobs."""
    if not top_logprobs:
        return None, None, None, None, False

    yes_lps: list[float] = []
    no_lps: list[float] = []
    all_lps: list[float] = []

    for item in top_logprobs:
        if isinstance(item, Mapping):
            tok = str(item.get("token", ""))
            lp = float(item["logprob"])
        else:
            tok = str(getattr(item, "token", ""))
            lp = float(getattr(item, "logprob"))
        all_lps.append(lp)
        norm = tok.strip().upper()
        if norm == "YES":
            yes_lps.append(lp)
        elif norm == "NO":
            no_lps.append(lp)

    if not all_lps or (not yes_lps and not no_lps):
        return None, None, None, None, True

    imputed = False
    rem_prob = max(1e-9, 1.0 - sum(math.exp(v) for v in all_lps))
    floor_lp = min(min(all_lps) - 2.0, math.log(rem_prob))

    if yes_lps:
        lp_yes = _logsumexp(yes_lps)
    else:
        lp_yes = floor_lp
        imputed = True

    if no_lps:
        lp_no = _logsumexp(no_lps)
    else:
        lp_no = floor_lp
        imputed = True

    yes_margin = lp_yes - lp_no
    if yes_margin >= 0.0:
        z = math.exp(-yes_margin)
        p_yes = 1.0 / (1.0 + z)
    else:
        z = math.exp(yes_margin)
        p_yes = z / (1.0 + z)

    return (
        round(lp_yes, 6),
        round(lp_no, 6),
        round(yes_margin, 6),
        round(p_yes, 6),
        imputed,
    )


def parse_yesno_choice(
    raw_text: str,
    logprob_yes: float | None = None,
    logprob_no: float | None = None,
    tie_eps: float = LOGPROB_TIE_EPS,
) -> Literal["YES", "NO"] | None:
    """Strictly parse YES or NO from model text, falling back to YES/NO logprobs."""
    cleaned = raw_text.strip().upper().rstrip(".!:")
    if cleaned in ("YES", "NO"):
        return cleaned  # type: ignore[return-value]

    matches = set(re.findall(r"\b(YES|NO)\b", cleaned))
    if len(matches) == 1:
        return next(iter(matches))  # type: ignore[return-value]

    if logprob_yes is not None and logprob_no is not None:
        if abs(logprob_yes - logprob_no) <= tie_eps:
            return None
        return "YES" if logprob_yes > logprob_no else "NO"
    return None


def classify_yesno_outcome(
    raw_text: str,
    ground_truth: Literal["YES", "NO"],
    logprob_yes: float | None = None,
    logprob_no: float | None = None,
    tie_eps: float = LOGPROB_TIE_EPS,
) -> tuple[
    Literal["YES", "NO"] | None,
    Literal["scored", "unparseable", "logprob_tie"],
    float | None,
    float | None,
]:
    """Classify single-snippet YES/NO response into (pred, outcome, predicted_yes, correct)."""
    if (
        logprob_yes is not None
        and logprob_no is not None
        and abs(logprob_yes - logprob_no) <= tie_eps
    ):
        return None, "logprob_tie", None, None

    pred = parse_yesno_choice(raw_text, logprob_yes, logprob_no, tie_eps=tie_eps)
    if pred is None:
        return None, "unparseable", None, None

    predicted_yes = 1.0 if pred == "YES" else 0.0
    correct = 1.0 if pred == ground_truth else 0.0
    return pred, "scored", predicted_yes, correct


def load_completed_fpr_keys(fpr_judgements_path: Path) -> set[tuple[str, str]]:
    """Load already-scored (probe_id, judge_model) keys for crash-safe resume."""
    keys: set[tuple[str, str]] = set()
    for row in load_jsonl(fpr_judgements_path):
        pid = row.get("probe_id")
        jm = row.get("judge_model")
        if pid is not None and jm is not None:
            keys.add((str(pid), str(jm)))
    return keys


def build_fpr_task_plan(
    active_poles: Sequence[Mapping[str, Any]],
    fpr_snippets: Sequence[Mapping[str, Any]],
    explanations_by_k: Mapping[str, Mapping[str, Any]] | None = None,
    sweeps: Sequence[str] = FPR_SWEEPS,
) -> list[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]]:
    """Build deterministic single-snippet YES/NO precision task plan."""
    for sw in sweeps:
        if sw not in FPR_SWEEPS:
            raise ValueError(f"Unknown FPR sweep {sw!r}; valid: {FPR_SWEEPS}")

    poles_by_id = {str(p["pole_id"]): p for p in active_poles}
    own_text_by_pole = {
        pid: resolve_pole_own_text(p, explanations_by_k) for pid, p in poles_by_id.items()
    }

    plan: list[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]] = []
    seen_keys: set[tuple[str, str]] = set()

    def _resolve_probe_type(row: Mapping[str, Any]) -> tuple[Literal["top100_firing", "ordinary_nonfiring"], Literal["YES", "NO"]]:
        pt = str(row.get("probe_type", "")).lower()
        gt = str(row.get("ground_truth", "")).upper()
        if pt in ("top100_firing", "top100", "positive", "pos") or gt == "YES":
            return "top100_firing", "YES"
        return "ordinary_nonfiring", "NO"

    def _add(row: Mapping[str, Any], api_model: str) -> None:
        pid = str(row["pole_id"])
        if pid not in poles_by_id:
            return
        pole = poles_by_id[pid]
        desc = own_text_by_pole[pid]
        ptype, gt = _resolve_probe_type(row)
        tier = str(row.get("tier", "core"))
        probe_idx = int(row.get("probe_idx", 0))
        probe_id = str(row.get("probe_id") or f"{pid}__{ptype}__{probe_idx}")
        snip_txt = str(row["marked_text"])
        jm_tag = f"{api_model}@none"

        task = FprProbeTask(
            probe_id=probe_id,
            pole_id=pid,
            k=int(pole["k"]),
            layer=int(pole["layer"]),
            neuron=int(pole["neuron"]),
            sign=resolve_pole_sign(pole),
            probe_type=ptype,
            ground_truth=gt,
            tier=tier,
            probe_idx=probe_idx,
            snippet_text=snip_txt,
            description_text=desc,
            judge_model=jm_tag,
            api_model=api_model,
            reasoning_effort="none",
            max_completion_tokens=8,
        )
        if task.key in seen_keys:
            return
        seen_keys.add(task.key)
        messages = build_fpr_messages(desc, snip_txt)
        extra_meta = {
            "doc_id": row.get("doc_id"),
            "act_val": row.get("act_val"),
            "act_dir": row.get("act_dir"),
            "frac_of_max": row.get("frac_of_max"),
            "quadrant": row.get("quadrant"),
        }
        plan.append((task, messages, extra_meta))

    for sw in sweeps:
        if sw == "fpr_core_terra":
            for r in fpr_snippets:
                if str(r.get("tier", "core")).lower() == "luna_extra":
                    continue
                _add(r, "gpt-6.1-terra")
        elif sw == "fpr_all_luna":
            for r in fpr_snippets:
                _add(r, "gpt-6-luna")

    return plan


def quote_fpr_plan_cost(
    plan: Sequence[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]],
    assumed_completion_tokens: int = 4,
) -> dict[str, Any]:
    """Conservative 0%-cache pre-run USD quote for YES/NO precision probes."""
    by_bucket: dict[str, dict[str, Any]] = {}
    tot_usd = 0.0
    tot_in = 0
    tot_out = 0
    for task, msgs, _ in plan:
        in_tok = estimate_prompt_tokens_conservative(msgs)
        out_tok = assumed_completion_tokens
        cost = compute_call_cost_usd(task.api_model, in_tok, out_tok, MODEL_PRICING_PER_M)
        tot_usd += cost
        tot_in += in_tok
        tot_out += out_tok
        bkey = f"{task.judge_model}:{task.probe_type}:{task.tier}"
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


async def _call_openai_fpr_once(
    client: Any,
    task: FprProbeTask,
    messages: list[dict[str, str]],
    limiter: AdaptiveRateLimiter,
    max_retries: int = 6,
    base_backoff: float = 0.5,
) -> dict[str, Any]:
    """Execute one OpenAI single-snippet YES/NO call with retries and logprobs."""
    req_kwargs: dict[str, Any] = {
        "model": task.api_model,
        "messages": messages,
        "reasoning_effort": task.reasoning_effort,
        "temperature": 0.0,
        "max_completion_tokens": int(task.max_completion_tokens),
        "logprobs": True,
        "top_logprobs": 5,
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        await limiter.acquire()
        try:
            completions = client.chat.completions
            if hasattr(completions, "with_raw_response"):
                raw_resp = await completions.with_raw_response.create(**req_kwargs)
                headers = {k.lower(): str(v) for k, v in dict(raw_resp.headers).items()}
                parsed = raw_resp.parse()
                await limiter.note_headers(headers)
            else:
                parsed = await completions.create(**req_kwargs)

            choice = parsed.choices[0]
            raw_content = (choice.message.content or "").strip()

            top_lps = None
            lp_obj = getattr(choice, "logprobs", None)
            if lp_obj is not None and getattr(lp_obj, "content", None):
                top_lps = getattr(lp_obj.content[0], "top_logprobs", None)

            lp_yes, lp_no, yes_margin, p_yes, imputed = extract_yesno_logprobs(top_lps)
            pred, outcome, predicted_yes, correct = classify_yesno_outcome(
                raw_content, task.ground_truth, lp_yes, lp_no
            )

            usage = getattr(parsed, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
            visible_output_tokens = max(0, completion_tokens - reasoning_tokens)
            cost_usd = compute_call_cost_usd(
                task.api_model, prompt_tokens, completion_tokens, MODEL_PRICING_PER_M
            )

            return {
                "probe_id": task.probe_id,
                "pole_id": task.pole_id,
                "k": task.k,
                "layer": task.layer,
                "neuron": task.neuron,
                "sign": task.sign,
                "probe_type": task.probe_type,
                "ground_truth": task.ground_truth,
                "tier": task.tier,
                "probe_idx": task.probe_idx,
                "judge_model": task.judge_model,
                "api_model": task.api_model,
                "reasoning_effort": task.reasoning_effort,
                "judge_backend": "openai",
                "predicted_answer": pred,
                "predicted_yes": predicted_yes,
                "outcome": outcome,
                "correct": correct,
                "logprob_YES": lp_yes,
                "logprob_NO": lp_no,
                "yes_margin": yes_margin,
                "p_yes": p_yes,
                "logprob_imputed": imputed,
                "raw_response": raw_content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_output_tokens": visible_output_tokens,
                "cost_usd": round(cost_usd, 8),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _is_rate_limit_error(exc):
                ra = _extract_retry_after(exc)
                await limiter.note_rate_limit(ra)
                sleep_s = max(ra or 0.0, base_backoff * (2**attempt))
            else:
                sleep_s = base_backoff * (2**attempt)
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(sleep_s, 30.0))

    raise RuntimeError(
        f"OpenAI FPR call failed after {max_retries} retries for {task.probe_id} ({task.judge_model}): {last_err}"
    ) from last_err


async def run_fpr_judge_async(
    plan: Sequence[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]],
    output_path: Path = DEFAULT_JUDGEMENTS_FPR_PATH,
    companion_spend_paths: Sequence[Path] = (DEFAULT_JUDGEMENTS_MULTIBAND_PATH,),
    max_usd: float = DEFAULT_SURROGATE_MAX_USD,
    concurrency: int = 32,
    target_rpm: float = 3500.0,
    openai_client: Any = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Execute single-snippet YES/NO precision calls with hard combined USD cap."""
    assert_surrogate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)

    completed_keys = load_completed_fpr_keys(output_path)
    prior_self_usd = _sum_existing_spend([output_path])
    prior_combined_usd = _sum_existing_spend([output_path, *companion_spend_paths])
    pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed_keys]
    quote = quote_fpr_plan_cost(pending)

    if not pending:
        return {
            "total_planned": len(plan),
            "skipped_existing": len(plan),
            "completed_this_run": 0,
            "prior_self_usd": round(prior_self_usd, 6),
            "prior_combined_usd": round(prior_combined_usd, 6),
            "spent_usd_this_run": 0.0,
            "cumulative_self_usd": round(prior_self_usd, 6),
            "cumulative_combined_usd": round(prior_combined_usd, 6),
            "aborted_for_spend_cap": False,
            "pre_run_quote": quote,
        }

    if prior_combined_usd >= max_usd:
        raise SpendLimitExceededError(
            f"Prior combined surrogate-sweep spend ${prior_combined_usd:.4f} >= --max-usd ${max_usd:.2f}"
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
    completed_now = 0
    aborted = False

    try:
        with output_path.open("a", encoding="utf-8") as fh:

            async def _worker(
                task: FprProbeTask,
                messages: list[dict[str, str]],
                extra_meta: dict[str, Any],
            ) -> None:
                nonlocal spent_this_run, prompt_toks, comp_toks, reas_toks, completed_now, aborted
                async with sem:
                    async with state_lock:
                        if aborted or (prior_combined_usd + spent_this_run) >= max_usd:
                            aborted = True
                            return
                    row = await _call_openai_fpr_once(
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
                        completed_now += 1
                        if progress_every > 0 and (
                            completed_now % progress_every == 0 or completed_now == len(pending)
                        ):
                            print(
                                f"[fpr_4b] {completed_now:,}/{len(pending):,} done | "
                                f"run=${spent_this_run:.4f} | combined=${prior_combined_usd + spent_this_run:.4f}",
                                flush=True,
                            )
                        if (prior_combined_usd + spent_this_run) >= max_usd:
                            aborted = True

            await asyncio.gather(*(_worker(t, m, meta) for t, m, meta in pending))
    finally:
        if close_client and hasattr(client, "close"):
            await client.close()

    return {
        "total_planned": len(plan),
        "skipped_existing": len(plan) - len(pending),
        "completed_this_run": completed_now,
        "prior_self_usd": round(prior_self_usd, 6),
        "prior_combined_usd": round(prior_combined_usd, 6),
        "spent_usd_this_run": round(spent_this_run, 6),
        "cumulative_self_usd": round(prior_self_usd + spent_this_run, 6),
        "cumulative_combined_usd": round(prior_combined_usd + spent_this_run, 6),
        "prompt_tokens_this_run": prompt_toks,
        "completion_tokens_this_run": comp_toks,
        "reasoning_tokens_this_run": reas_toks,
        "aborted_for_spend_cap": aborted,
        "pre_run_quote": quote,
    }


# ---------------------------------------------------------------------------
# Part 3: Statistical Primitives (Partial Spearman, AUROC, Precision@k, 5-Fold CV)
# ---------------------------------------------------------------------------


def exact_auroc(scores: Sequence[float], labels: Sequence[bool | int]) -> float:
    """Exact rank-based AUROC (handles ties identically to Mann-Whitney U)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_precision_at_k(
    scores: Sequence[float],
    labels: Sequence[bool | int],
    k: int,
) -> float:
    """Exact tie-invariant expected precision@k under uniform random tie-breaking."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    n = len(s)
    if n == 0 or k <= 0:
        return float("nan")
    k_eff = min(int(k), n)
    # Sort descending by unique score values
    uniq_vals = np.unique(s)[::-1]
    rem = k_eff
    exp_pos = 0.0
    for val in uniq_vals:
        if rem <= 0:
            break
        in_grp = s == val
        grp_n = int(np.sum(in_grp))
        grp_pos = int(np.sum(y[in_grp] == 1))
        if grp_n <= rem:
            exp_pos += float(grp_pos)
            rem -= grp_n
        else:
            exp_pos += float(rem) * (float(grp_pos) / float(grp_n))
            rem = 0
    return float(exp_pos / float(k_eff))


def partial_spearman_corr(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute exact partial Spearman rho(X, Y | Z) with t-test p and bootstrap 95% CI."""
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    az = np.asarray(z, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    ax, ay, az = ax[mask], ay[mask], az[mask]
    n = int(len(ax))

    def _compute_once(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> float:
        if len(vx) < 4 or float(np.ptp(vx)) <= 1e-12 or float(np.ptp(vy)) <= 1e-12:
            return float("nan")
        rx = rankdata(vx, method="average")
        ry = rankdata(vy, method="average")
        if float(np.ptp(vz)) <= 1e-12:
            r, _ = pearsonr(rx, ry)
            return float(r)
        rz = rankdata(vz, method="average")
        rz_c = rz - np.mean(rz)
        denom = float(np.dot(rz_c, rz_c))
        if denom <= 1e-12:
            r, _ = pearsonr(rx, ry)
            return float(r)
        rx_c = rx - np.mean(rx)
        ry_c = ry - np.mean(ry)
        ex = rx_c - (float(np.dot(rx_c, rz_c)) / denom) * rz_c
        ey = ry_c - (float(np.dot(ry_c, rz_c)) / denom) * rz_c
        if float(np.ptp(ex)) <= 1e-12 or float(np.ptp(ey)) <= 1e-12:
            return float("nan")
        r, _ = pearsonr(ex, ey)
        return float(np.clip(r, -1.0, 1.0))

    rho_partial = _compute_once(ax, ay, az)
    if not math.isfinite(rho_partial) or n <= 3:
        return {
            "n": n,
            "partial_spearman_rho": float("nan"),
            "partial_spearman_p": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    if abs(rho_partial) >= 1.0 - 1e-12:
        p_val = 0.0
    else:
        t_stat = rho_partial * math.sqrt((n - 3) / max(1e-15, 1.0 - rho_partial * rho_partial))
        p_val = float(2.0 * student_t.sf(abs(t_stat), df=n - 3))

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(n_resamples, n))
    boots: list[float] = []
    for idx in idx_mat:
        val = _compute_once(ax[idx], ay[idx], az[idx])
        if math.isfinite(val):
            boots.append(val)

    return {
        "n": n,
        "partial_spearman_rho": float(rho_partial),
        "partial_spearman_p": p_val,
        "ci_low": float(np.percentile(boots, 2.5)) if boots else float("nan"),
        "ci_high": float(np.percentile(boots, 97.5)) if boots else float("nan"),
    }


def bootstrap_paired_spearman_diff(
    x_cand: Sequence[float],
    x_base: Sequence[float],
    y: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired neuron bootstrap for Delta_rho = rho(x_cand, y) - rho(x_base, y)."""
    ac = np.asarray(x_cand, dtype=np.float64)
    ab = np.asarray(x_base, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(ac) & np.isfinite(ab) & np.isfinite(ay)
    ac, ab, ay = ac[mask], ab[mask], ay[mask]
    n = int(len(ay))
    rho_c, _ = _safe_spearman(ac, ay)
    rho_b, _ = _safe_spearman(ab, ay)
    if not (math.isfinite(rho_c) and math.isfinite(rho_b)) or n < 3:
        return {
            "n": n,
            "delta_rho": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(n_resamples, n))
    diffs: list[float] = []
    for idx in idx_mat:
        rc, _ = _safe_spearman(ac[idx], ay[idx])
        rb, _ = _safe_spearman(ab[idx], ay[idx])
        if math.isfinite(rc) and math.isfinite(rb):
            diffs.append(rc - rb)

    return {
        "n": n,
        "delta_rho": float(rho_c - rho_b),
        "ci_low": float(np.percentile(diffs, 2.5)) if diffs else float("nan"),
        "ci_high": float(np.percentile(diffs, 97.5)) if diffs else float("nan"),
    }


def make_stratified_folds(
    layers: Sequence[int],
    n_folds: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    """Deterministic layer-stratified k-fold assignment across neurons."""
    n = len(layers)
    folds = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_layer: dict[int, list[int]] = defaultdict(list)
    for idx, lay in enumerate(layers):
        by_layer[int(lay)].append(idx)

    cursor = 0
    for lay in sorted(by_layer):
        idxs = list(by_layer[lay])
        rng.shuffle(idxs)
        for idx in idxs:
            folds[idx] = cursor % n_folds
            cursor += 1
    return folds


def _fit_ridge_closed_form(X_s: np.ndarray, y_c: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form L2 ridge solution (X^T X + alpha I)^-1 X^T y."""
    p = X_s.shape[1]
    gram = X_s.T @ X_s + float(alpha) * np.eye(p, dtype=np.float64)
    rhs = X_s.T @ y_c
    return np.linalg.solve(gram, rhs)


def _train_only_ecdf_rank_transform(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map train features to average ranks and eval features strictly via train ECDF."""
    n_tr, p = X_train.shape
    R_tr = np.zeros_like(X_train, dtype=np.float64)
    R_ev = np.zeros_like(X_eval, dtype=np.float64)
    denom = float(n_tr + 1)
    for j in range(p):
        col_tr = X_train[:, j]
        R_tr[:, j] = rankdata(col_tr, method="average") / denom
        sorted_tr = np.sort(col_tr)
        left = np.searchsorted(sorted_tr, X_eval[:, j], side="left")
        right = np.searchsorted(sorted_tr, X_eval[:, j], side="right")
        R_ev[:, j] = (0.5 * (left + right) + 0.5) / denom
    return R_tr, R_ev


def _safe_nanmean(vals: Sequence[float]) -> float:
    fin = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(fin)) if fin else float("nan")


def _safe_nanstd(vals: Sequence[float], ddof: int = 1) -> float:
    fin = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(fin) <= ddof:
        return 0.0 if fin else float("nan")
    return float(np.std(fin, ddof=ddof))


def run_nested_oof_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    rank_mode: bool = False,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
) -> dict[str, Any]:
    """Strictly leak-free outer 5-fold CV with inner 4-fold CV for ridge penalty alpha.

    All standardization, empirical-CDF rank mapping, and alpha selection happen strictly
    inside each outer training fold; zero test-fold information touches training.
    """
    n, p = X.shape
    oof_preds = np.full(n, np.nan, dtype=np.float64)
    n_folds = int(np.max(folds)) + 1
    chosen_alphas: list[float] = []
    fold_weights: list[list[float]] = []
    fold_rhos: list[float] = []
    fold_rs: list[float] = []

    for f in range(n_folds):
        te_mask = folds == f
        tr_mask = ~te_mask
        X_tr_raw = X[tr_mask].copy()
        y_tr_raw = y[tr_mask].copy()
        X_te_raw = X[te_mask].copy()
        y_te_raw = y[te_mask].copy()

        # Impute any rare non-finite value strictly with train-fold column median
        for j in range(p):
            tr_fin = np.isfinite(X_tr_raw[:, j])
            med_j = float(np.median(X_tr_raw[tr_fin, j])) if np.any(tr_fin) else 0.0
            X_tr_raw[~tr_fin, j] = med_j
            te_fin = np.isfinite(X_te_raw[:, j])
            X_te_raw[~te_fin, j] = med_j

        # Inner CV folds formed by the remaining outer fold IDs
        inner_fold_ids = sorted(int(u) for u in np.unique(folds[tr_mask]))
        tr_inner_folds = folds[tr_mask]

        best_alpha = float(alpha_grid[0])
        best_score = -float("inf") if rank_mode else float("inf")

        for alpha in alpha_grid:
            inner_preds = np.zeros(len(y_tr_raw), dtype=np.float64)
            inner_targets = np.zeros(len(y_tr_raw), dtype=np.float64)
            for inf in inner_fold_ids:
                iv = tr_inner_folds == inf
                it = ~iv
                if rank_mode:
                    Xi_tr, Xi_va = _train_only_ecdf_rank_transform(X_tr_raw[it], X_tr_raw[iv])
                    yi_tr = rankdata(y_tr_raw[it], method="average") / float(np.sum(it) + 1)
                else:
                    Xi_tr, Xi_va = X_tr_raw[it], X_tr_raw[iv]
                    yi_tr = y_tr_raw[it]

                mu = np.mean(Xi_tr, axis=0)
                sd = np.std(Xi_tr, axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                Xi_tr_s = (Xi_tr - mu) / sd
                Xi_va_s = (Xi_va - mu) / sd
                y_mu = float(np.mean(yi_tr))
                w = _fit_ridge_closed_form(Xi_tr_s, yi_tr - y_mu, alpha)
                inner_preds[iv] = y_mu + Xi_va_s @ w
                inner_targets[iv] = y_tr_raw[iv]

            if rank_mode:
                rho_in, _ = _safe_spearman(inner_preds, inner_targets)
                sc = rho_in if math.isfinite(rho_in) else -1.0
                if sc > best_score:
                    best_score = sc
                    best_alpha = float(alpha)
            else:
                mse_in = float(np.mean((inner_preds - inner_targets) ** 2))
                if mse_in < best_score:
                    best_score = mse_in
                    best_alpha = float(alpha)

        # Fit final outer-fold model strictly on outer train fold
        if rank_mode:
            X_tr_f, X_te_f = _train_only_ecdf_rank_transform(X_tr_raw, X_te_raw)
            y_tr_f = rankdata(y_tr_raw, method="average") / float(len(y_tr_raw) + 1)
        else:
            X_tr_f, X_te_f = X_tr_raw, X_te_raw
            y_tr_f = y_tr_raw

        mu_f = np.mean(X_tr_f, axis=0)
        sd_f = np.std(X_tr_f, axis=0)
        sd_f = np.where(sd_f < 1e-12, 1.0, sd_f)
        X_tr_s = (X_tr_f - mu_f) / sd_f
        X_te_s = (X_te_f - mu_f) / sd_f
        y_mu_f = float(np.mean(y_tr_f))
        w_f = _fit_ridge_closed_form(X_tr_s, y_tr_f - y_mu_f, best_alpha)
        preds_te = y_mu_f + X_te_s @ w_f

        oof_preds[te_mask] = preds_te
        chosen_alphas.append(best_alpha)
        fold_weights.append([round(float(v), 6) for v in w_f])
        f_rho, _ = _safe_spearman(preds_te, y_te_raw)
        f_r, _ = _safe_pearson(preds_te, y_te_raw)
        fold_rhos.append(f_rho)
        fold_rs.append(f_r)

    mean_weights = [
        round(float(np.mean([fw[j] for fw in fold_weights])), 6) for j in range(p)
    ]
    return {
        "oof_preds": oof_preds,
        "chosen_alphas": chosen_alphas,
        "fold_weights": fold_weights,
        "mean_std_weights": mean_weights,
        "fold_spearman_rhos": fold_rhos,
        "fold_pearson_rs": fold_rs,
        "fold_spearman_mean": _safe_nanmean(fold_rhos),
        "fold_spearman_sd": _safe_nanstd(fold_rhos, ddof=1),
        "fold_pearson_mean": _safe_nanmean(fold_rs),
        "fold_pearson_sd": _safe_nanstd(fold_rs, ddof=1),
    }


def evaluate_surrogate_vector(
    name: str,
    family: str,
    scores: Sequence[float],
    y_primary: Sequence[float],
    y_active: Sequence[float],
    z_energy20: Sequence[float],
    folds: np.ndarray,
    is_oof_cv: bool = False,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluate one surrogate vector across correlations, partial rho, CV folds, AUROC, and prec@k."""
    sc = np.asarray(scores, dtype=np.float64)
    yp = np.asarray(y_primary, dtype=np.float64)
    ya = np.asarray(y_active, dtype=np.float64)
    ze = np.asarray(z_energy20, dtype=np.float64)

    corr_prim = bootstrap_corr_ci(sc, yp, n_resamples=n_bootstrap, seed=seed)
    corr_act = bootstrap_corr_ci(sc, ya, n_resamples=n_bootstrap, seed=seed + 11)
    partial_prim = partial_spearman_corr(sc, yp, ze, n_resamples=n_bootstrap, seed=seed + 23)
    partial_act = partial_spearman_corr(sc, ya, ze, n_resamples=n_bootstrap, seed=seed + 37)
    delta_vs_e20 = bootstrap_paired_spearman_diff(
        sc, ze, yp, n_resamples=n_bootstrap, seed=seed + 41
    )

    # 5-fold mean +- SD on identical folds
    n_folds = int(np.max(folds)) + 1
    f_rhos: list[float] = []
    f_rs: list[float] = []
    for f in range(n_folds):
        m = folds == f
        frho, _ = _safe_spearman(sc[m], yp[m])
        fr, _ = _safe_pearson(sc[m], yp[m])
        f_rhos.append(frho)
        f_rs.append(fr)

    # Classification metrics (identifying causal pass >=0.25, >=0.50, and <=0 failures)
    y_ge_25 = yp >= 0.25
    y_ge_50 = yp >= 0.50
    y_le_0 = yp <= 0.0
    y_lt_0 = yp < 0.0
    neg_sc = -sc

    clf = {
        "pass_ge_0_25": {
            "n_pos": int(np.sum(y_ge_25)),
            "auroc": exact_auroc(sc, y_ge_25),
            "precision_at_20": expected_precision_at_k(sc, y_ge_25, 20),
            "precision_at_50": expected_precision_at_k(sc, y_ge_25, 50),
        },
        "pass_ge_0_50": {
            "n_pos": int(np.sum(y_ge_50)),
            "auroc": exact_auroc(sc, y_ge_50),
            "precision_at_20": expected_precision_at_k(sc, y_ge_50, 20),
            "precision_at_50": expected_precision_at_k(sc, y_ge_50, 50),
        },
        "fail_le_0_00": {
            "n_pos": int(np.sum(y_le_0)),
            "auroc": exact_auroc(neg_sc, y_le_0),
            "precision_at_20": expected_precision_at_k(neg_sc, y_le_0, 20),
            "precision_at_50": expected_precision_at_k(neg_sc, y_le_0, 50),
        },
        "fail_lt_0_00": {
            "n_pos": int(np.sum(y_lt_0)),
            "auroc": exact_auroc(neg_sc, y_lt_0),
            "precision_at_20": expected_precision_at_k(neg_sc, y_lt_0, 20),
            "precision_at_50": expected_precision_at_k(neg_sc, y_lt_0, 50),
        },
    }

    return {
        "name": name,
        "family": family,
        "is_oof_cv": bool(is_oof_cv),
        "n_valid": int(np.sum(np.isfinite(sc) & np.isfinite(yp))),
        "sim_cal_rec": corr_prim,
        "sim_cal_rec_active": corr_act,
        "partial_above_energy_ge_20pct_sim_cal_rec": partial_prim,
        "partial_above_energy_ge_20pct_sim_cal_rec_active": partial_act,
        "delta_rho_vs_energy_ge_20pct_sim_cal_rec": delta_vs_e20,
        "cv5_fold_spearman_mean": _safe_nanmean(f_rhos),
        "cv5_fold_spearman_sd": _safe_nanstd(f_rhos, ddof=1),
        "cv5_fold_pearson_mean": _safe_nanmean(f_rs),
        "cv5_fold_pearson_sd": _safe_nanstd(f_rs, ddof=1),
        "classification": clf,
    }


# ---------------------------------------------------------------------------
# Part 4: Data Aggregation (Detection Bands, FPR/TPR, WDet, WDet_KL)
# ---------------------------------------------------------------------------


def _dedup_fpr_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate FPR rows by canonical (probe_id, judge_model)."""
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["probe_id"]), str(row["judge_model"]))
        dedup[key] = dict(row)
    return list(dedup.values())


def _canonical_band_name(band: str) -> str:
    """Normalize band labels across snippets_4b / multiband / top100."""
    b = str(band).strip()
    if b in ("p99.9+", "p99.9_plus"):
        return "p99.9_plus"
    if b in ("top100", "top100_thresh"):
        return "top100"
    return b


def _aggregate_band_accuracy(
    judgement_rows: Sequence[Mapping[str, Any]],
    poles_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[int, str, str], float],
    dict[str, dict[str, Any]],
]:
    """Compute per-pole and per-neuron (mean across active poles) own-label accuracy by (band, judge_model)."""
    pole_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    cell_audit: dict[str, dict[str, Any]] = {}

    for row in judgement_rows:
        if str(row.get("condition", "own")) != "own":
            continue
        jm = str(row["judge_model"])
        band = _canonical_band_name(str(row["band"]))
        pid = str(row["pole_id"])
        outcome = classify_row_outcome(row)
        audit_key = f"{jm}:{band}"
        if audit_key not in cell_audit:
            cell_audit[audit_key] = {
                "judge_model": jm,
                "band": band,
                "n_total": 0,
                "n_scored": 0,
                "n_unparseable": 0,
                "n_logprob_tie": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }
        ca = cell_audit[audit_key]
        ca["n_total"] += 1
        ca["reasoning_tokens"] += int(row.get("reasoning_tokens", 0) or 0)
        ca["cost_usd"] += float(row.get("cost_usd", 0.0) or 0.0)
        if outcome == "scored":
            ca["n_scored"] += 1
            pole_scores[(pid, band, jm)].append(float(row["correct"]))
        elif outcome == "logprob_tie":
            ca["n_logprob_tie"] += 1
        else:
            ca["n_unparseable"] += 1

    for ca in cell_audit.values():
        ca["cost_usd"] = round(ca["cost_usd"], 6)

    pole_acc: dict[tuple[str, str, str], float] = {
        k_t: float(np.mean(vals)) for k_t, vals in pole_scores.items() if vals
    }

    # Per-neuron mean across active poles (matching analyze_4b.py convention)
    neuron_acc: dict[tuple[int, str, str], float] = {}
    all_bands_j = {(b, jm) for _, b, jm in pole_acc}
    for k, plist in poles_by_k.items():
        pids = [str(p["pole_id"]) for p in plist]
        for band, jm in all_bands_j:
            vals = [pole_acc[(pid, band, jm)] for pid in pids if (pid, band, jm) in pole_acc]
            if vals:
                neuron_acc[(k, band, jm)] = float(np.mean(vals))

    # Add mean accuracy across neurons into cell_audit
    for audit_key, ca in cell_audit.items():
        jm, band = ca["judge_model"], ca["band"]
        n_vals = [
            neuron_acc[(k, band, jm)]
            for k in poles_by_k
            if (k, band, jm) in neuron_acc
        ]
        p_vals = [
            pole_acc[(pid, band, jm)]
            for plist in poles_by_k.values()
            for p in plist
            for pid in [str(p["pole_id"])]
            if (pid, band, jm) in pole_acc
        ]
        ca["n_poles_covered"] = len(p_vals)
        ca["n_neurons_covered"] = len(n_vals)
        ca["mean_pole_accuracy"] = float(np.mean(p_vals)) if p_vals else float("nan")
        ca["mean_neuron_accuracy"] = float(np.mean(n_vals)) if n_vals else float("nan")

    return pole_acc, neuron_acc, cell_audit


def _aggregate_fpr_metrics(
    fpr_rows: Sequence[Mapping[str, Any]],
    poles_by_k: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[int, str], dict[str, float]],
    dict[str, dict[str, Any]],
]:
    """Aggregate per-pole and per-neuron FPR_ord, TPR_top, yes_margin, p_yes, BalancedAcc."""
    pole_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    audit: dict[str, dict[str, Any]] = {}

    for row in fpr_rows:
        jm = str(row["judge_model"])
        ptype = str(row["probe_type"])
        pid = str(row["pole_id"])
        outcome = str(row.get("outcome", "unparseable"))
        akey = f"{jm}:{ptype}"
        if akey not in audit:
            audit[akey] = {
                "judge_model": jm,
                "probe_type": ptype,
                "n_total": 0,
                "n_scored": 0,
                "n_unparseable": 0,
                "n_logprob_tie": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }
        a = audit[akey]
        a["n_total"] += 1
        a["reasoning_tokens"] += int(row.get("reasoning_tokens", 0) or 0)
        a["cost_usd"] += float(row.get("cost_usd", 0.0) or 0.0)
        if outcome == "scored":
            a["n_scored"] += 1
            pole_buckets[(pid, jm, ptype)].append(dict(row))
        elif outcome == "logprob_tie":
            a["n_logprob_tie"] += 1
        else:
            a["n_unparseable"] += 1

    for a in audit.values():
        a["cost_usd"] = round(a["cost_usd"], 6)

    judges = sorted({str(r["judge_model"]) for r in fpr_rows})
    pole_metrics: dict[tuple[str, str], dict[str, float]] = {}

    def _mean_field(rows_sub: Sequence[Mapping[str, Any]], field: str) -> float:
        vals = [float(r[field]) for r in rows_sub if r.get(field) is not None and math.isfinite(float(r[field]))]
        return float(np.mean(vals)) if vals else float("nan")

    for plist in poles_by_k.values():
        for p in plist:
            pid = str(p["pole_id"])
            for jm in judges:
                ord_rows = pole_buckets.get((pid, jm, "ordinary_nonfiring"), [])
                top_rows = pole_buckets.get((pid, jm, "top100_firing"), [])
                if not ord_rows and not top_rows:
                    continue
                fpr_ord = _mean_field(ord_rows, "predicted_yes")
                tpr_top = _mean_field(top_rows, "predicted_yes")
                py_ord = _mean_field(ord_rows, "p_yes")
                py_top = _mean_field(top_rows, "p_yes")
                ym_ord = _mean_field(ord_rows, "yes_margin")
                ym_top = _mean_field(top_rows, "yes_margin")
                bal_acc = (
                    0.5 * (tpr_top + (1.0 - fpr_ord))
                    if math.isfinite(tpr_top) and math.isfinite(fpr_ord)
                    else float("nan")
                )
                tpr_m_fpr = (
                    tpr_top - fpr_ord
                    if math.isfinite(tpr_top) and math.isfinite(fpr_ord)
                    else float("nan")
                )
                ym_gap = (
                    ym_top - ym_ord
                    if math.isfinite(ym_top) and math.isfinite(ym_ord)
                    else float("nan")
                )
                py_gap = (
                    py_top - py_ord
                    if math.isfinite(py_top) and math.isfinite(py_ord)
                    else float("nan")
                )
                pole_metrics[(pid, jm)] = {
                    "n_ord_scored": float(len(ord_rows)),
                    "n_top_scored": float(len(top_rows)),
                    "fpr_ord": fpr_ord,
                    "tpr_top": tpr_top,
                    "one_minus_fpr_ord": 1.0 - fpr_ord if math.isfinite(fpr_ord) else float("nan"),
                    "tpr_minus_fpr": tpr_m_fpr,
                    "balanced_acc_yesno": bal_acc,
                    "p_yes_ord": py_ord,
                    "p_yes_top": py_top,
                    "p_yes_gap": py_gap,
                    "yes_margin_ord": ym_ord,
                    "yes_margin_top": ym_top,
                    "yes_margin_gap": ym_gap,
                }

    # Aggregate to neuron level (mean across active poles, plus max FPR_ord across active poles)
    neuron_metrics: dict[tuple[int, str], dict[str, float]] = {}
    metric_fields = (
        "fpr_ord",
        "tpr_top",
        "one_minus_fpr_ord",
        "tpr_minus_fpr",
        "balanced_acc_yesno",
        "p_yes_ord",
        "p_yes_top",
        "p_yes_gap",
        "yes_margin_ord",
        "yes_margin_top",
        "yes_margin_gap",
    )
    for k, plist in poles_by_k.items():
        pids = [str(p["pole_id"]) for p in plist]
        for jm in judges:
            pms = [pole_metrics[(pid, jm)] for pid in pids if (pid, jm) in pole_metrics]
            if not pms:
                continue
            agg: dict[str, float] = {}
            for fld in metric_fields:
                vals = [pm[fld] for pm in pms if math.isfinite(pm[fld])]
                agg[fld] = float(np.mean(vals)) if vals else float("nan")
            fprs = [pm["fpr_ord"] for pm in pms if math.isfinite(pm["fpr_ord"])]
            agg["fpr_ord_max_pole"] = float(np.max(fprs)) if fprs else float("nan")
            neuron_metrics[(k, jm)] = agg

    return pole_metrics, neuron_metrics, audit


def _lift_transform(acc: float | None, mode: Literal["clamped", "signed", "raw"]) -> float:
    """Transform band accuracy in [0, 1] into WDet band credit."""
    if acc is None or not math.isfinite(acc):
        return 0.0 if mode != "raw" else 0.5
    if mode == "clamped":
        return max(0.0, 2.0 * float(acc) - 1.0)
    if mode == "signed":
        return 2.0 * float(acc) - 1.0
    return float(acc)


def compute_wdet_score(
    band_weights: Mapping[str, float],
    band_accs: Mapping[str, float | None],
    band_map: Sequence[tuple[str, str]],
    mode: Literal["clamped", "signed", "raw"] = "clamped",
) -> float:
    """Zero-parameter energy-weighted multi-band detection WDet = sum_B w(B) * phi(acc(B))."""
    total = 0.0
    for w_key, acc_key in band_map:
        w = float(band_weights.get(w_key, 0.0))
        acc = band_accs.get(acc_key)
        total += w * _lift_transform(acc, mode=mode)
    return float(total)


def _extract_kl_4band_shares(
    kl_neuron_entry: Mapping[str, Any],
) -> tuple[dict[str, float] | None, dict[str, float] | None, float | None]:
    """Flexibly extract normalized 4-band KL shares, unnormalized KL/full ratios, and kl_ge_20pct_max."""
    # Support clean nested dicts or flat keys from band_kl_4b.json
    band_keys = ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
    norm_shares: dict[str, float] | None = None
    ratio_full: dict[str, float] | None = None

    for k_dict in (
        "kl_share_4band_norm",
        "kl_shares_4band",
        "kl_shares_norm",
        "kl_shares_bands_4",
        "band_kl_shares",
    ):
        sub = kl_neuron_entry.get(k_dict)
        if isinstance(sub, Mapping) and all(bk in sub for bk in band_keys):
            norm_shares = {bk: float(sub[bk]) for bk in band_keys}
            break

    for k_dict in (
        "kl_share_4band_vs_full",
        "kl_over_full_4band",
        "kl_frac_of_full",
        "kl_ratio_full_bands_4",
    ):
        sub = kl_neuron_entry.get(k_dict)
        if isinstance(sub, Mapping) and all(bk in sub for bk in band_keys):
            ratio_full = {bk: float(sub[bk]) for bk in band_keys}
            break

    raw_kl = None
    for k_dict in ("kl_by_cond", "kl_by_band", "band_kl", "kl", "kl_values"):
        sub = kl_neuron_entry.get(k_dict)
        if isinstance(sub, Mapping):
            raw_kl = sub
            break

    if norm_shares is None and raw_kl is not None:
        # Try aliases like p99.9+ vs p99_9_plus
        alias_map = {
            "p99.9+": ("p99.9+", "p99.9_plus", "p99_9_plus"),
            "p97_99.9": ("p97_99.9", "p97_99_9"),
            "p75_97": ("p75_97",),
            "lt_p75": ("lt_p75", "bulk_lt_p75"),
        }
        vals: dict[str, float] = {}
        ok = True
        for bk, aliases in alias_map.items():
            found = None
            for a in aliases:
                if a in raw_kl and raw_kl[a] is not None:
                    found = max(0.0, float(raw_kl[a]))
                    break
            if found is None:
                ok = False
                break
            vals[bk] = found
        if ok:
            s_tot = sum(vals.values())
            if s_tot > 0.0:
                norm_shares = {bk: vals[bk] / s_tot for bk in band_keys}
            full_kl = raw_kl.get("full")
            if full_kl is not None and float(full_kl) > 0.0 and ratio_full is None:
                ratio_full = {bk: vals[bk] / float(full_kl) for bk in band_keys}

    kl_ge_20: float | None = None
    for k_direct in (
        "kl_share_ge_20pct_max_norm",
        "kl_share_ge_20pct_max_vs_full",
        "kl_share_ge_20pct_max",
        "kl_ge_20pct_max_share",
        "kl_ratio_ge_20pct_max",
    ):
        if k_direct in kl_neuron_entry and kl_neuron_entry[k_direct] is not None:
            kl_ge_20 = float(kl_neuron_entry[k_direct])
            break
    if kl_ge_20 is None and raw_kl is not None:
        g20 = raw_kl.get("ge_20pct_max")
        l20 = raw_kl.get("lt_20pct_max")
        full = raw_kl.get("full")
        if g20 is not None and l20 is not None and (float(g20) + float(l20)) > 0.0:
            kl_ge_20 = max(0.0, float(g20)) / (max(0.0, float(g20)) + max(0.0, float(l20)))
        elif g20 is not None and full is not None and float(full) > 0.0:
            kl_ge_20 = max(0.0, float(g20)) / float(full)

    return norm_shares, ratio_full, kl_ge_20


def analyze_surrogates_4b(
    active_poles_path: Path = DEFAULT_ACTIVE_POLES_PATH,
    scores_path: Path = DEFAULT_SCORES_4B_PATH,
    gap_decomp_path: Path = DEFAULT_GAP_DECOMP_PATH,
    judgements_4b_path: Path = DEFAULT_JUDGEMENTS_4B_PATH,
    judgements_top100_path: Path = DEFAULT_JUDGEMENTS_4B_TOP100_PATH,
    judgements_multiband_path: Path = DEFAULT_JUDGEMENTS_MULTIBAND_PATH,
    judgements_fpr_path: Path = DEFAULT_JUDGEMENTS_FPR_PATH,
    band_kl_path: Path | None = DEFAULT_BAND_KL_PATH,
    output_metrics_path: Path = DEFAULT_SURROGATE_METRICS_PATH,
    output_summary_path: Path = DEFAULT_SURROGATE_SUMMARY_PATH,
    figure_gap_path: Path = DEFAULT_FIGURE_GAP_PATH,
    figure_surrogate_path: Path = DEFAULT_FIGURE_SURROGATE_PATH,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run complete surrogate evaluation, JSON/MD outputs, and 2 PNG figures."""
    for out_p in (output_metrics_path, output_summary_path, figure_gap_path, figure_surrogate_path):
        assert_surrogate_output_path(out_p)

    active_poles = load_active_poles(active_poles_path)
    poles_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in active_poles:
        poles_by_k[int(p["k"])].append(p)

    gap_doc = json.loads(gap_decomp_path.read_text(encoding="utf-8"))
    gap_per_neuron: dict[str, dict[str, Any]] = gap_doc["per_neuron"]
    ks_sorted = sorted(int(k) for k in gap_per_neuron.keys())

    ulto_raw = json.loads(scores_path.read_text(encoding="utf-8"))
    ulto_by_k = {int(r["k"]): r for r in ulto_raw["neurons"]}

    # Load and deduplicate all detection judgements (read-only!)
    det_rows_raw: list[dict[str, Any]] = []
    for jp in (judgements_4b_path, judgements_top100_path, judgements_multiband_path):
        if jp.is_file():
            det_rows_raw.extend(load_jsonl(jp))
    det_rows = _dedup_judgements_4b(det_rows_raw)
    pole_acc, neuron_acc, det_cell_audit = _aggregate_band_accuracy(det_rows, poles_by_k)

    # Load and deduplicate FPR YES/NO judgements
    fpr_rows_raw = load_jsonl(judgements_fpr_path) if judgements_fpr_path.is_file() else []
    fpr_rows = _dedup_fpr_rows(fpr_rows_raw)
    pole_fpr, neuron_fpr, fpr_cell_audit = _aggregate_fpr_metrics(fpr_rows, poles_by_k)

    # Exact API spend audit (strictly multiband + fpr JSONLs)
    mb_rows_only = _dedup_judgements_4b(load_jsonl(judgements_multiband_path)) if judgements_multiband_path.is_file() else []
    mb_spend_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in mb_rows_only)
    fpr_spend_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in fpr_rows)
    surrogate_total_spend_usd = mb_spend_usd + fpr_spend_usd

    mb_spend_by_model: dict[str, float] = defaultdict(float)
    for r in mb_rows_only:
        mb_spend_by_model[str(r["judge_model"])] += float(r.get("cost_usd", 0.0) or 0.0)
    fpr_spend_by_model: dict[str, float] = defaultdict(float)
    for r in fpr_rows:
        fpr_spend_by_model[str(r["judge_model"])] += float(r.get("cost_usd", 0.0) or 0.0)

    # Build per-neuron feature table (n = 278)
    layers_vec: list[int] = []
    y_sim: list[float] = []
    y_active: list[float] = []
    y_strong20: list[float] = []
    missing_band_cells: Counter[str] = Counter()

    per_neuron_table: dict[str, dict[str, Any]] = {}

    bands_5_map_p999 = (
        ("p99.9+", "p99.9_plus"),
        ("p99_99.9", "p99_99.9"),
        ("p97_99", "p97_99"),
        ("p75_97", "p75_97"),
        ("lt_p75", "p25_75"),
    )
    bands_5_map_top100 = (
        ("p99.9+", "top100"),
        ("p99_99.9", "p99_99.9"),
        ("p97_99", "p97_99"),
        ("p75_97", "p75_97"),
        ("lt_p75", "p25_75"),
    )
    bands_6_map_split = (
        ("top100_thresh", "top100"),
        ("rest_of_p99.9+", "p99.9_plus"),
        ("p99_99.9", "p99_99.9"),
        ("p97_99", "p97_99"),
        ("p75_97", "p75_97"),
        ("bulk_lt_p75", "p25_75"),
    )
    bands_p25_only_map = (
        ("p99.9+", "p99.9_plus"),
        ("p99_99.9", "p99_99.9"),
        ("p97_99", "p97_99"),
        ("p75_97", "p75_97"),
        ("p25_75", "p25_75"),
    )

    for k in ks_sorted:
        g = gap_per_neuron[str(k)]
        u = ulto_by_k.get(k, {})
        layers_vec.append(int(g["layer"]))
        sim_v = float(g.get("sim_cal_rec", u.get("sim_cal_rec", float("nan"))))
        act_v = float(g.get("sim_cal_rec_active", u.get("sim_cal_rec_active", float("nan"))))
        s20_v = float(g.get("strong20_rec", u.get("strong20_rec", float("nan"))))
        y_sim.append(sim_v)
        y_active.append(act_v)
        y_strong20.append(s20_v)

        w5_l2 = g["energy_l2"]["bands_5"]
        w6_l2 = g["energy_l2"]["bands_6"]
        wp25_l2 = g["energy_l2"]["bands_with_p25_75"]
        w5_l1 = g["energy_l1"]["bands_5"]

        accs_terra: dict[str, float | None] = {}
        accs_luna: dict[str, float | None] = {}
        for b in ("top100", "p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75"):
            vt = neuron_acc.get((k, b, PRIMARY_JUDGE_4B))
            vl = neuron_acc.get((k, b, SECONDARY_JUDGE_4B))
            accs_terra[b] = vt
            accs_luna[b] = vl
            if b != "top100":
                if vt is None:
                    missing_band_cells[f"{PRIMARY_JUDGE_4B}:{b}"] += 1
                if vl is None:
                    missing_band_cells[f"{SECONDARY_JUDGE_4B}:{b}"] += 1

        # Unweighted 5-band mean detection across available bands
        t_5_p999 = [accs_terra[b] for b in ("p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75") if accs_terra[b] is not None]
        t_5_top100 = [accs_terra[b] for b in ("top100", "p99_99.9", "p97_99", "p75_97", "p25_75") if accs_terra[b] is not None]
        l_5_p999 = [accs_luna[b] for b in ("p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75") if accs_luna[b] is not None]

        # WDet variants (zero parameters!)
        wdet_clamped_terra = compute_wdet_score(w5_l2, accs_terra, bands_5_map_p999, mode="clamped")
        wdet_clamped_top100_terra = compute_wdet_score(w5_l2, accs_terra, bands_5_map_top100, mode="clamped")
        wdet_6band_clamped_terra = compute_wdet_score(w6_l2, accs_terra, bands_6_map_split, mode="clamped")
        wdet_p25only_clamped_terra = compute_wdet_score(wp25_l2, accs_terra, bands_p25_only_map, mode="clamped")
        wdet_signed_terra = compute_wdet_score(w5_l2, accs_terra, bands_5_map_p999, mode="signed")
        wdet_raw_acc_terra = compute_wdet_score(w5_l2, accs_terra, bands_5_map_p999, mode="raw")
        wdet_l1_clamped_terra = compute_wdet_score(w5_l1, accs_terra, bands_5_map_p999, mode="clamped")

        wdet_clamped_luna = compute_wdet_score(w5_l2, accs_luna, bands_5_map_p999, mode="clamped")
        wdet_p25only_clamped_luna = compute_wdet_score(wp25_l2, accs_luna, bands_p25_only_map, mode="clamped")
        wdet_signed_luna = compute_wdet_score(w5_l2, accs_luna, bands_5_map_p999, mode="signed")
        wdet_raw_acc_luna = compute_wdet_score(w5_l2, accs_luna, bands_5_map_p999, mode="raw")

        # FPR / TPR precision metrics
        ft = neuron_fpr.get((k, PRIMARY_JUDGE_4B), {})
        fl = neuron_fpr.get((k, SECONDARY_JUDGE_4B), {})

        fpr_t = float(ft.get("fpr_ord", float("nan")))
        tpr_t = float(ft.get("tpr_top", float("nan")))
        fpr_l = float(fl.get("fpr_ord", float("nan")))
        tpr_l = float(fl.get("tpr_top", float("nan")))

        row_n: dict[str, Any] = {
            "k": k,
            "neuron_id": str(g["neuron_id"]),
            "layer": int(g["layer"]),
            "neuron": int(g["neuron"]),
            "is_bipolar": bool(g["is_bipolar"]),
            "named_neuron": g.get("named_neuron"),
            "failure_bucket": str(g["failure_bucket"]),
            "sim_cal_rec": sim_v,
            "sim_cal_rec_active": act_v,
            "strong20_rec": s20_v,
            "strong50_rec": float(g.get("strong50_rec", u.get("strong50_rec", float("nan")))),
            # B0: Top-band detection
            "b0_top100_terra": float(accs_terra["top100"]) if accs_terra["top100"] is not None else float("nan"),
            "b0_p999_terra": float(accs_terra["p99.9_plus"]) if accs_terra["p99.9_plus"] is not None else float("nan"),
            "b0_p999_luna": float(accs_luna["p99.9_plus"]) if accs_luna["p99.9_plus"] is not None else float("nan"),
            # B1: Zero-label structural energy
            "b1_energy_ge_20pct_max": float(g["energy_ge_20pct_max"]),
            "b1_energy_ge_50pct_max": float(g["energy_ge_50pct_max"]),
            "b1_energy_p999": float(g["energy_p99.9+"]),
            "b1_energy_p97_plus": float(g["energy_p97+"]),
            "b1_one_minus_gate_closed_energy": float(1.0 - float(g["energy_gate_closed"])),
            # S1: Unweighted 5-band detection
            "s1_unweighted_5band_p999_terra": float(np.mean(t_5_p999)) if t_5_p999 else float("nan"),
            "s1_unweighted_5band_top100_terra": float(np.mean(t_5_top100)) if t_5_top100 else float("nan"),
            "s1_unweighted_5band_p999_luna": float(np.mean(l_5_p999)) if l_5_p999 else float("nan"),
            # S2: Zero-parameter WDet
            "s2_wdet_clamped_terra": wdet_clamped_terra,
            "s2_wdet_clamped_top100_terra": wdet_clamped_top100_terra,
            "s2_wdet_6band_clamped_terra": wdet_6band_clamped_terra,
            "s2_wdet_p25only_clamped_terra": wdet_p25only_clamped_terra,
            "s2_wdet_signed_terra": wdet_signed_terra,
            "s2_wdet_raw_acc_terra": wdet_raw_acc_terra,
            "s2_wdet_l1_clamped_terra": wdet_l1_clamped_terra,
            "s2_wdet_clamped_luna": wdet_clamped_luna,
            "s2_wdet_p25only_clamped_luna": wdet_p25only_clamped_luna,
            "s2_wdet_signed_luna": wdet_signed_luna,
            "s2_wdet_raw_acc_luna": wdet_raw_acc_luna,
            # S3: Ordinary-text precision / FPR_ord / TPR_top
            "fpr_ord_terra": fpr_t,
            "tpr_top_terra": tpr_t,
            "s3_one_minus_fpr_ord_terra": float(ft.get("one_minus_fpr_ord", float("nan"))),
            "s3_tpr_minus_fpr_terra": float(ft.get("tpr_minus_fpr", float("nan"))),
            "s3_balanced_acc_yesno_terra": float(ft.get("balanced_acc_yesno", float("nan"))),
            "s3_pyes_margin_gap_terra": float(ft.get("yes_margin_gap", float("nan"))),
            "s3_one_minus_pyes_ord_terra": (
                1.0 - float(ft["p_yes_ord"]) if ft.get("p_yes_ord") is not None and math.isfinite(float(ft["p_yes_ord"])) else float("nan")
            ),
            "fpr_ord_luna": fpr_l,
            "tpr_top_luna": tpr_l,
            "s3_one_minus_fpr_ord_luna": float(fl.get("one_minus_fpr_ord", float("nan"))),
            "s3_tpr_minus_fpr_luna": float(fl.get("tpr_minus_fpr", float("nan"))),
            "s3_balanced_acc_yesno_luna": float(fl.get("balanced_acc_yesno", float("nan"))),
            "s3_pyes_margin_gap_luna": float(fl.get("yes_margin_gap", float("nan"))),
            "s3_one_minus_pyes_ord_luna": (
                1.0 - float(fl["p_yes_ord"]) if fl.get("p_yes_ord") is not None and math.isfinite(float(fl["p_yes_ord"])) else float("nan")
            ),
            # S4: Zero-parameter precision-penalized WDet
            "s4_wdet_minus_fpr_terra": wdet_clamped_terra - fpr_t if math.isfinite(fpr_t) else float("nan"),
            "s4_wdet_times_1m_fpr_terra": wdet_clamped_terra * (1.0 - fpr_t) if math.isfinite(fpr_t) else float("nan"),
            "s4_wdet_top100_minus_fpr_terra": wdet_clamped_top100_terra - fpr_t if math.isfinite(fpr_t) else float("nan"),
            "s4_wdet_top100_times_1m_fpr_terra": wdet_clamped_top100_terra * (1.0 - fpr_t) if math.isfinite(fpr_t) else float("nan"),
            "s4_wdet_minus_fpr_luna": wdet_clamped_luna - fpr_l if math.isfinite(fpr_l) else float("nan"),
            "s4_wdet_times_1m_fpr_luna": wdet_clamped_luna * (1.0 - fpr_l) if math.isfinite(fpr_l) else float("nan"),
            # Raw band accuracies for full auditability
            "band_accs_terra": {b: accs_terra[b] for b in ("top100", "p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75")},
            "band_accs_luna": {b: accs_luna[b] for b in ("p99.9_plus", "p99_99.9", "p97_99", "p75_97", "p25_75")},
            "fpr_details_terra": ft,
            "fpr_details_luna": fl,
        }
        per_neuron_table[str(k)] = row_n

    rows_ordered = [per_neuron_table[str(k)] for k in ks_sorted]
    y_sim_arr = np.asarray(y_sim, dtype=np.float64)
    y_act_arr = np.asarray(y_active, dtype=np.float64)
    z_e20_arr = np.asarray([r["b1_energy_ge_20pct_max"] for r in rows_ordered], dtype=np.float64)
    folds = make_stratified_folds(layers_vec, n_folds=5, seed=seed)

    # Strictly Out-Of-Fold 5-Fold CV models (S4 learned + B1 structural CV reference)
    cv_specs: list[tuple[str, str, list[str], bool]] = [
        (
            "s4_cv_ridge_full_terra",
            "S4_cv",
            ["s2_wdet_clamped_terra", "fpr_ord_terra", "tpr_top_terra", "b1_energy_ge_20pct_max"],
            False,
        ),
        (
            "s4_cv_rank_ridge_full_terra",
            "S4_cv",
            ["s2_wdet_clamped_terra", "fpr_ord_terra", "tpr_top_terra", "b1_energy_ge_20pct_max"],
            True,
        ),
        (
            "s4_cv_ridge_label_only_terra",
            "S4_cv",
            ["s2_wdet_clamped_terra", "fpr_ord_terra", "tpr_top_terra"],
            False,
        ),
        (
            "s4_cv_rank_ridge_label_only_terra",
            "S4_cv",
            ["s2_wdet_clamped_terra", "fpr_ord_terra", "tpr_top_terra"],
            True,
        ),
        (
            "s4_cv_ridge_full_luna",
            "S4_cv",
            ["s2_wdet_clamped_luna", "fpr_ord_luna", "tpr_top_luna", "b1_energy_ge_20pct_max"],
            False,
        ),
        (
            "s4_cv_rank_ridge_full_luna",
            "S4_cv",
            ["s2_wdet_clamped_luna", "fpr_ord_luna", "tpr_top_luna", "b1_energy_ge_20pct_max"],
            True,
        ),
        (
            "b1_cv_ridge_energy_only",
            "B1_structural_cv",
            [
                "b1_energy_ge_20pct_max",
                "b1_energy_p999",
                "b1_energy_p97_plus",
                "b1_one_minus_gate_closed_energy",
            ],
            False,
        ),
        (
            "b1_cv_rank_ridge_energy_only",
            "B1_structural_cv",
            [
                "b1_energy_ge_20pct_max",
                "b1_energy_p999",
                "b1_energy_p97_plus",
                "b1_one_minus_gate_closed_energy",
            ],
            True,
        ),
    ]

    cv_model_details: dict[str, dict[str, Any]] = {}
    for cname, _, feat_cols, rank_mode in cv_specs:
        X_mat = np.column_stack(
            [np.asarray([r[c] for r in rows_ordered], dtype=np.float64) for c in feat_cols]
        )
        # Skip CV model cleanly if any required feature column is entirely NaN (e.g. partial test)
        if any(not np.any(np.isfinite(X_mat[:, j])) for j in range(X_mat.shape[1])):
            oof_vec = np.full(len(rows_ordered), np.nan, dtype=np.float64)
            cv_model_details[cname] = {"features": feat_cols, "rank_mode": rank_mode, "skipped_missing_features": True}
        else:
            res = run_nested_oof_ridge_cv(X_mat, y_sim_arr, folds=folds, rank_mode=rank_mode)
            oof_vec = res["oof_preds"]
            cv_model_details[cname] = {
                "features": feat_cols,
                "rank_mode": rank_mode,
                "chosen_alphas": res["chosen_alphas"],
                "mean_std_weights": dict(zip(feat_cols, res["mean_std_weights"])),
                "fold_spearman_rhos": res["fold_spearman_rhos"],
                "fold_spearman_mean": res["fold_spearman_mean"],
                "fold_spearman_sd": res["fold_spearman_sd"],
            }
        for idx, r in enumerate(rows_ordered):
            r[cname] = float(oof_vec[idx])

    # Evaluate ALL candidates in deterministic benchmark order
    candidate_specs: list[tuple[str, str, bool]] = [
        # B0
        ("b0_top100_terra", "B0_top_band", False),
        ("b0_p999_terra", "B0_top_band", False),
        ("b0_p999_luna", "B0_top_band", False),
        # B1
        ("b1_energy_ge_20pct_max", "B1_zero_label", False),
        ("b1_energy_ge_50pct_max", "B1_zero_label", False),
        ("b1_energy_p999", "B1_zero_label", False),
        ("b1_energy_p97_plus", "B1_zero_label", False),
        ("b1_one_minus_gate_closed_energy", "B1_zero_label", False),
        ("b1_cv_ridge_energy_only", "B1_zero_label_cv", True),
        ("b1_cv_rank_ridge_energy_only", "B1_zero_label_cv", True),
        # S1
        ("s1_unweighted_5band_p999_terra", "S1_unweighted_multiband", False),
        ("s1_unweighted_5band_top100_terra", "S1_unweighted_multiband", False),
        ("s1_unweighted_5band_p999_luna", "S1_unweighted_multiband", False),
        # S2
        ("s2_wdet_clamped_terra", "S2_wdet", False),
        ("s2_wdet_clamped_top100_terra", "S2_wdet", False),
        ("s2_wdet_6band_clamped_terra", "S2_wdet", False),
        ("s2_wdet_p25only_clamped_terra", "S2_wdet", False),
        ("s2_wdet_signed_terra", "S2_wdet", False),
        ("s2_wdet_raw_acc_terra", "S2_wdet", False),
        ("s2_wdet_l1_clamped_terra", "S2_wdet", False),
        ("s2_wdet_clamped_luna", "S2_wdet", False),
        ("s2_wdet_p25only_clamped_luna", "S2_wdet", False),
        ("s2_wdet_signed_luna", "S2_wdet", False),
        ("s2_wdet_raw_acc_luna", "S2_wdet", False),
        # S3
        ("s3_one_minus_fpr_ord_terra", "S3_precision", False),
        ("s3_tpr_minus_fpr_terra", "S3_precision", False),
        ("s3_balanced_acc_yesno_terra", "S3_precision", False),
        ("s3_pyes_margin_gap_terra", "S3_precision", False),
        ("s3_one_minus_pyes_ord_terra", "S3_precision", False),
        ("s3_one_minus_fpr_ord_luna", "S3_precision", False),
        ("s3_tpr_minus_fpr_luna", "S3_precision", False),
        ("s3_balanced_acc_yesno_luna", "S3_precision", False),
        ("s3_pyes_margin_gap_luna", "S3_precision", False),
        ("s3_one_minus_pyes_ord_luna", "S3_precision", False),
        # S4 zero-param
        ("s4_wdet_minus_fpr_terra", "S4_zero_param", False),
        ("s4_wdet_times_1m_fpr_terra", "S4_zero_param", False),
        ("s4_wdet_top100_minus_fpr_terra", "S4_zero_param", False),
        ("s4_wdet_top100_times_1m_fpr_terra", "S4_zero_param", False),
        ("s4_wdet_minus_fpr_luna", "S4_zero_param", False),
        ("s4_wdet_times_1m_fpr_luna", "S4_zero_param", False),
        # S4 5-fold OOF CV
        ("s4_cv_ridge_full_terra", "S4_oof_cv", True),
        ("s4_cv_rank_ridge_full_terra", "S4_oof_cv", True),
        ("s4_cv_ridge_label_only_terra", "S4_oof_cv", True),
        ("s4_cv_rank_ridge_label_only_terra", "S4_oof_cv", True),
        ("s4_cv_ridge_full_luna", "S4_oof_cv", True),
        ("s4_cv_rank_ridge_full_luna", "S4_oof_cv", True),
    ]

    benchmarks: dict[str, dict[str, Any]] = {}
    for idx_c, (cname, family, is_oof) in enumerate(candidate_specs):
        vec = [r[cname] for r in rows_ordered]
        benchmarks[cname] = evaluate_surrogate_vector(
            name=cname,
            family=family,
            scores=vec,
            y_primary=y_sim_arr,
            y_active=y_act_arr,
            z_energy20=z_e20_arr,
            folds=folds,
            is_oof_cv=is_oof,
            n_bootstrap=n_bootstrap,
            seed=seed + 100 * (idx_c + 1),
        )
        if cname in cv_model_details:
            benchmarks[cname]["cv_details"] = cv_model_details[cname]

    # Direct negative-recovery FPR_ord diagnostic (does FPR_ord predict sim_cal_rec < 0?)
    fpr_neg_recovery_diag: dict[str, Any] = {}
    y_neg_lt0 = y_sim_arr < 0.0
    y_overbroad = np.asarray(
        [r["failure_bucket"] == "overbroad_fp_damage" for r in rows_ordered], dtype=bool
    )
    for jm_short, fld_fpr, fld_py in (
        ("terra", "fpr_ord_terra", "s3_one_minus_pyes_ord_terra"),
        ("luna", "fpr_ord_luna", "s3_one_minus_pyes_ord_luna"),
    ):
        fpr_v = np.asarray([r[fld_fpr] for r in rows_ordered], dtype=np.float64)
        py_v = 1.0 - np.asarray([r[fld_py] for r in rows_ordered], dtype=np.float64)
        fin = np.isfinite(fpr_v)
        fpr_neg_recovery_diag[jm_short] = {
            "n_valid": int(np.sum(fin)),
            "mean_fpr_ord_all": float(np.nanmean(fpr_v)) if np.any(fin) else float("nan"),
            "median_fpr_ord_all": float(np.nanmedian(fpr_v)) if np.any(fin) else float("nan"),
            "mean_fpr_ord_sim_lt_0": float(np.nanmean(fpr_v[y_neg_lt0])) if np.any(fin & y_neg_lt0) else float("nan"),
            "mean_fpr_ord_sim_ge_0": float(np.nanmean(fpr_v[~y_neg_lt0])) if np.any(fin & (~y_neg_lt0)) else float("nan"),
            "mean_fpr_ord_overbroad_bucket": (
                float(np.nanmean(fpr_v[y_overbroad])) if np.any(fin & y_overbroad) else float("nan")
            ),
            "fpr_ord_auroc_sim_lt_0": exact_auroc(fpr_v, y_neg_lt0),
            "fpr_ord_prec20_sim_lt_0": expected_precision_at_k(fpr_v, y_neg_lt0, 20),
            "fpr_ord_auroc_overbroad_bucket": exact_auroc(fpr_v, y_overbroad),
            "p_yes_ord_auroc_sim_lt_0": exact_auroc(py_v, y_neg_lt0),
            "p_yes_ord_auroc_overbroad_bucket": exact_auroc(py_v, y_overbroad),
            "fpr_ord_vs_sim_cal_rec": bootstrap_corr_ci(fpr_v, y_sim_arr, n_resamples=n_bootstrap, seed=seed + 700),
            "by_bucket_median_fpr_ord": {
                b: (
                    float(np.nanmedian([r[fld_fpr] for r in rows_ordered if r["failure_bucket"] == b]))
                    if any(r["failure_bucket"] == b and math.isfinite(r[fld_fpr]) for r in rows_ordered)
                    else float("nan")
                )
                for b in FAILURE_BUCKET_ORDER
            },
        }

    # Named neurons summary (info, PCR, insurance)
    named_surrogate_summary: dict[str, dict[str, Any]] = {}
    for label_name, k_val, _, _ in NAMED_NEURONS:
        r = per_neuron_table.get(str(k_val))
        if r is None:
            continue
        named_surrogate_summary[label_name] = {
            "k": k_val,
            "neuron_id": r["neuron_id"],
            "failure_bucket": r["failure_bucket"],
            "sim_cal_rec": r["sim_cal_rec"],
            "sim_cal_rec_active": r["sim_cal_rec_active"],
            "strong20_rec": r["strong20_rec"],
            "b0_top100_terra": r["b0_top100_terra"],
            "b0_p999_terra": r["b0_p999_terra"],
            "b1_energy_ge_20pct_max": r["b1_energy_ge_20pct_max"],
            "s1_unweighted_5band_p999_terra": r["s1_unweighted_5band_p999_terra"],
            "s2_wdet_clamped_terra": r["s2_wdet_clamped_terra"],
            "s2_wdet_clamped_luna": r["s2_wdet_clamped_luna"],
            "fpr_ord_terra": r["fpr_ord_terra"],
            "tpr_top_terra": r["tpr_top_terra"],
            "fpr_ord_luna": r["fpr_ord_luna"],
            "tpr_top_luna": r["tpr_top_luna"],
            "s4_wdet_minus_fpr_terra": r["s4_wdet_minus_fpr_terra"],
            "s4_wdet_times_1m_fpr_terra": r["s4_wdet_times_1m_fpr_terra"],
            "s4_cv_rank_ridge_full_terra": r.get("s4_cv_rank_ridge_full_terra"),
            "band_accs_terra": r["band_accs_terra"],
            "band_accs_luna": r["band_accs_luna"],
        }

    # Optional 60-neuron WDet_KL validation
    kl_validation: dict[str, Any] | None = None
    if band_kl_path is not None and band_kl_path.is_file():
        kl_doc = json.loads(band_kl_path.read_text(encoding="utf-8"))
        kl_per_n = kl_doc.get("per_neuron", {})
        kl_ks = sorted(int(k) for k in kl_per_n.keys() if str(k) in per_neuron_table)
        if len(kl_ks) >= 5:
            kl_rows: list[dict[str, Any]] = []
            for k in kl_ks:
                kn = kl_per_n[str(k)]
                rn = per_neuron_table[str(k)]
                gn = gap_per_neuron[str(k)]
                norm_sh, ratio_full, kl_ge20 = _extract_kl_4band_shares(kn)
                if norm_sh is None:
                    continue
                # Construct 4-band accuracies matching KL bands:
                # p97_99.9 combines p99_99.9 and p97_99 weighted by their relative L2 energy (fallback 50/50)
                at = rn["band_accs_terra"]
                al = rn["band_accs_luna"]
                w5 = gn["energy_l2"]["bands_5"]
                w99 = float(w5["p99_99.9"])
                w97 = float(w5["p97_99"])
                denom_mid = w99 + w97

                def _mid_acc(acc_d: Mapping[str, float | None]) -> float | None:
                    a99 = acc_d.get("p99_99.9")
                    a97 = acc_d.get("p97_99")
                    if a99 is not None and a97 is not None:
                        if denom_mid > 1e-12:
                            return (w99 * a99 + w97 * a97) / denom_mid
                        return 0.5 * (a99 + a97)
                    return a99 if a99 is not None else a97

                acc4_t = {
                    "p99.9+": at.get("p99.9_plus"),
                    "p97_99.9": _mid_acc(at),
                    "p75_97": at.get("p75_97"),
                    "lt_p75": at.get("p25_75"),
                }
                acc4_l = {
                    "p99.9+": al.get("p99.9_plus"),
                    "p97_99.9": _mid_acc(al),
                    "p75_97": al.get("p75_97"),
                    "lt_p75": al.get("p25_75"),
                }
                b4_map = (
                    ("p99.9+", "p99.9+"),
                    ("p97_99.9", "p97_99.9"),
                    ("p75_97", "p75_97"),
                    ("lt_p75", "lt_p75"),
                )
                wdet_kl_t = compute_wdet_score(norm_sh, acc4_t, b4_map, mode="clamped")
                wdet_kl_l = compute_wdet_score(norm_sh, acc4_l, b4_map, mode="clamped")
                wdet_e4_t = compute_wdet_score(gn["energy_l2"]["bands_4"], acc4_t, b4_map, mode="clamped")
                wdet_kl_full_t = (
                    compute_wdet_score(ratio_full, acc4_t, b4_map, mode="clamped")
                    if ratio_full is not None
                    else float("nan")
                )
                fpr_t = rn["fpr_ord_terra"]
                kl_rows.append(
                    {
                        "k": k,
                        "layer": rn["layer"],
                        "failure_bucket": rn["failure_bucket"],
                        "sim_cal_rec": rn["sim_cal_rec"],
                        "sim_cal_rec_active": rn["sim_cal_rec_active"],
                        "strong20_rec": rn["strong20_rec"],
                        "b0_top100_terra": rn["b0_top100_terra"],
                        "b1_energy_ge_20pct_max": rn["b1_energy_ge_20pct_max"],
                        "kl_ge_20pct_max": kl_ge20 if kl_ge20 is not None else float("nan"),
                        "kl_shares_4band": norm_sh,
                        "energy_shares_4band": gn["energy_l2"]["bands_4"],
                        "wdet_energy_5band_terra": rn["s2_wdet_clamped_terra"],
                        "wdet_energy_4band_terra": wdet_e4_t,
                        "wdet_kl_norm_terra": wdet_kl_t,
                        "wdet_kl_norm_luna": wdet_kl_l,
                        "wdet_kl_ratio_full_terra": wdet_kl_full_t,
                        "wdet_kl_minus_fpr_terra": wdet_kl_t - fpr_t if math.isfinite(fpr_t) else float("nan"),
                        "wdet_energy_minus_fpr_terra": rn["s4_wdet_minus_fpr_terra"],
                    }
                )

            if len(kl_rows) >= 5:
                y_kl_sim = [r["sim_cal_rec"] for r in kl_rows]
                y_kl_act = [r["sim_cal_rec_active"] for r in kl_rows]
                kl_corr_table = {}
                for field in (
                    "b0_top100_terra",
                    "b1_energy_ge_20pct_max",
                    "kl_ge_20pct_max",
                    "wdet_energy_5band_terra",
                    "wdet_energy_4band_terra",
                    "wdet_kl_norm_terra",
                    "wdet_kl_norm_luna",
                    "wdet_kl_ratio_full_terra",
                    "wdet_energy_minus_fpr_terra",
                    "wdet_kl_minus_fpr_terra",
                ):
                    vals = [r[field] for r in kl_rows]
                    kl_corr_table[field] = {
                        "sim_cal_rec": bootstrap_corr_ci(vals, y_kl_sim, n_resamples=n_bootstrap, seed=seed + 901),
                        "sim_cal_rec_active": bootstrap_corr_ci(vals, y_kl_act, n_resamples=n_bootstrap, seed=seed + 902),
                    }
                # EnergyShare vs KLShare band-by-band correlation + WDet_energy vs WDet_KL
                band_energy_vs_kl = {}
                for bk in ("p99.9+", "p97_99.9", "p75_97", "lt_p75"):
                    band_energy_vs_kl[bk] = bootstrap_corr_ci(
                        [r["energy_shares_4band"][bk] for r in kl_rows],
                        [r["kl_shares_4band"][bk] for r in kl_rows],
                        n_resamples=n_bootstrap,
                        seed=seed + 910,
                    )
                kl_by_bucket: dict[str, Any] = {}
                for bname in FAILURE_BUCKET_ORDER:
                    brows = [r for r in kl_rows if r["failure_bucket"] == bname]
                    if brows:
                        kl_by_bucket[bname] = {
                            "count": len(brows),
                            "mean_kl_shares_4band": {
                                bk: float(np.mean([r["kl_shares_4band"][bk] for r in brows]))
                                for bk in ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
                            },
                            "mean_energy_shares_4band": {
                                bk: float(np.mean([r["energy_shares_4band"][bk] for r in brows]))
                                for bk in ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
                            },
                        }
                kl_validation = {
                    "n_neurons": len(kl_rows),
                    "wdet_energy_vs_wdet_kl": bootstrap_corr_ci(
                        [r["wdet_energy_4band_terra"] for r in kl_rows],
                        [r["wdet_kl_norm_terra"] for r in kl_rows],
                        n_resamples=n_bootstrap,
                        seed=seed + 920,
                    ),
                    "energy_ge_20pct_vs_kl_ge_20pct": bootstrap_corr_ci(
                        [r["b1_energy_ge_20pct_max"] for r in kl_rows],
                        [r["kl_ge_20pct_max"] for r in kl_rows],
                        n_resamples=n_bootstrap,
                        seed=seed + 921,
                    ),
                    "band_energy_vs_kl_shares": band_energy_vs_kl,
                    "correlations": kl_corr_table,
                    "by_bucket": kl_by_bucket,
                    "source_meta": {
                        k_m: kl_doc.get(k_m)
                        for k_m in (
                            "n_neurons",
                            "n_tokens_eval",
                            "total_non_bos_tokens",
                            "peak_vram_gb",
                            "peak_vram_allocated_gb",
                            "peak_vram_reserved_gb",
                            "elapsed_sec",
                        )
                        if k_m in kl_doc
                    },
                }

    # Identify best zero-parameter cheap surrogate & best overall cheap surrogate on sim_cal_rec
    def _rank_key(cname: str) -> float:
        v = benchmarks[cname]["sim_cal_rec"]["spearman_rho"]
        return float(v) if v is not None and math.isfinite(v) else -999.0

    llm_zero_param_names = [
        c for c, fam, is_oof in candidate_specs
        if not is_oof and fam in ("S1_unweighted_multiband", "S2_wdet", "S3_precision", "S4_zero_param")
    ]
    all_llm_names = [
        c for c, fam, _ in candidate_specs
        if fam in ("S1_unweighted_multiband", "S2_wdet", "S3_precision", "S4_zero_param", "S4_oof_cv")
    ]
    best_zero_param = max(llm_zero_param_names, key=_rank_key)
    best_overall_llm = max(all_llm_names, key=_rank_key)

    metrics_doc: dict[str, Any] = {
        "n_neurons": len(ks_sorted),
        "n_active_poles": len(active_poles),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "surrogate_spend": {
            "multiband_jsonl_calls": len(mb_rows_only),
            "multiband_spend_usd": round(mb_spend_usd, 6),
            "multiband_by_model_usd": {k: round(v, 6) for k, v in sorted(mb_spend_by_model.items())},
            "fpr_jsonl_calls": len(fpr_rows),
            "fpr_spend_usd": round(fpr_spend_usd, 6),
            "fpr_by_model_usd": {k: round(v, 6) for k, v in sorted(fpr_spend_by_model.items())},
            "combined_spend_usd": round(surrogate_total_spend_usd, 6),
            "budget_cap_usd": DEFAULT_SURROGATE_MAX_USD,
            "within_budget": bool(surrogate_total_spend_usd <= DEFAULT_SURROGATE_MAX_USD),
        },
        "detection_band_audit": det_cell_audit,
        "fpr_probe_audit": fpr_cell_audit,
        "missing_neuron_band_cells": dict(missing_band_cells),
        "best_zero_param_surrogate": best_zero_param,
        "best_overall_llm_surrogate": best_overall_llm,
        "benchmarks": benchmarks,
        "fpr_negative_recovery_diagnostic": fpr_neg_recovery_diag,
        "named_neurons": named_surrogate_summary,
        "gap_decomposition_summary": {
            "correlations": gap_doc.get("correlations", {}),
            "failure_buckets_top100_ge_0_80": gap_doc.get("failure_buckets_top100_ge_0_80", {}),
            "failure_buckets_p99_9_plus_ge_0_80": gap_doc.get("failure_buckets_p99_9_plus_ge_0_80", {}),
            "failure_buckets_all_278": gap_doc.get("failure_buckets_all_278", {}),
        },
        "kl_60_neuron_validation": kl_validation,
        "per_neuron": per_neuron_table,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(
        json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write factual markdown summary (zero interpretive prose)
    summary_md = render_surrogate_summary_md(metrics_doc)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_md, encoding="utf-8")

    # Render both 250-DPI factual figures
    plot_gap_decomposition_figure(
        gap_doc=gap_doc,
        kl_validation=kl_validation,
        output_path=figure_gap_path,
    )
    plot_surrogate_comparison_figure(
        metrics_doc=metrics_doc,
        output_path=figure_surrogate_path,
    )

    return metrics_doc


# ---------------------------------------------------------------------------
# Part 5: Strictly Factual Markdown Summary
# ---------------------------------------------------------------------------


def render_surrogate_summary_md(metrics_doc: Mapping[str, Any]) -> str:
    """Render strictly factual surrogate tables (zero interpretive prose)."""
    lines: list[str] = []
    spend = metrics_doc["surrogate_spend"]
    gap_sum = metrics_doc["gap_decomposition_summary"]
    benches = metrics_doc["benchmarks"]
    kl_val = metrics_doc.get("kl_60_neuron_validation")

    lines.append("# Qwen3-4B (Ulto85) Gap Decomposition & Surrogate Benchmark")
    lines.append("")
    lines.append(f"- Dataset: `Ulto85/neurons-kinda-clean` (MIT, `Qwen/Qwen3-4B`), `n = {metrics_doc['n_neurons']}` neurons (`{metrics_doc['n_active_poles']}` active poles).")
    lines.append(f"- Bootstrap resamples: `B = {metrics_doc['n_bootstrap']}` (per-neuron bootstrap, `seed = {metrics_doc['seed']}`).")
    lines.append(
        f"- Surrogate-sweep API spend: `${spend['combined_spend_usd']:.4f}` "
        f"(multiband: `${spend['multiband_spend_usd']:.4f}` across `{spend['multiband_jsonl_calls']:,}` calls; "
        f"FPR YES/NO: `${spend['fpr_spend_usd']:.4f}` across `{spend['fpr_jsonl_calls']:,}` calls; "
        f"cap `${spend['budget_cap_usd']:.2f}`)."
    )
    lines.append("")

    # Section 1: Energy-Share Correlations & Failure Buckets
    lines.append("## 1. Activation-Energy Shares & `top100 >= 0.80` Failure Buckets")
    lines.append("")
    lines.append("| Pair | n | Spearman ρ [95% CI] | Pearson r [95% CI] | R² |")
    lines.append("|---|---:|---:|---:|---:|")
    gcorrs = gap_sum.get("correlations", {})
    for key_lbl, ckey in (
        ("`energy_ge_20pct_max` vs `strong20_rec`", "energy_ge_20pct_max_vs_strong20_rec"),
        ("`l1_ge_20pct_max` vs `strong20_rec`", "l1_ge_20pct_max_vs_strong20_rec"),
        ("`energy_ge_50pct_max` vs `strong50_rec`", "energy_ge_50pct_max_vs_strong50_rec"),
        ("`energy_ge_20pct_max` vs `sim_cal_rec`", "energy_ge_20pct_max_vs_sim_cal_rec"),
        ("`energy_p99.9+` vs `sim_cal_rec`", "energy_p99_9_plus_vs_sim_cal_rec"),
        ("`energy_p97+` vs `sim_cal_rec`", "energy_p97_plus_vs_sim_cal_rec"),
        ("`energy_gate_open` vs `sim_cal_rec`", "energy_gate_open_vs_sim_cal_rec"),
        ("`energy_ge_20pct_max` vs `sim_cal_rec_active`", "energy_ge_20pct_max_vs_sim_cal_rec_active"),
    ):
        c = gcorrs.get(ckey)
        if not c:
            continue
        lines.append(
            f"| {key_lbl} | {c['n']} | "
            f"{_fmt_ci(c['spearman_rho'], c['spearman_ci_low'], c['spearman_ci_high'])} | "
            f"{_fmt_ci(c['pearson_r'], c['pearson_ci_low'], c['pearson_ci_high'])} | "
            f"{_fmt_num(c.get('r_squared'))} |"
        )
    lines.append("")

    fb = gap_sum.get("failure_buckets_top100_ge_0_80", {})
    if fb:
        lines.append(f"### Mutually Exclusive Buckets for `top100 >= 0.80` (`n = {fb.get('n_total', 0)}`)")
        lines.append("")
        lines.append("| Bucket | n | Share | Med `E(p99.9+)` | Med `E(<p97)` | Med `E(>=20%)` | Med `strong20` | Med `E(gate_closed)` | Med `sim_cal_rec` | Named |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for bname in FAILURE_BUCKET_ORDER:
            b = fb.get("buckets", {}).get(bname, {})
            named_s = ", ".join(b.get("named_neurons", [])) or "—"
            lines.append(
                f"| `{bname}` | {b.get('count', 0)} | {100.0 * float(b.get('share', 0.0)):.1f}% | "
                f"{_fmt_num(b.get('median_energy_p99_9_plus'))} | {_fmt_num(b.get('median_energy_lt_p97'))} | "
                f"{_fmt_num(b.get('median_energy_ge_20pct_max'))} | {_fmt_num(b.get('median_strong20_rec'))} | "
                f"{_fmt_num(b.get('median_energy_gate_closed'))} | {_fmt_num(b.get('median_sim_cal_rec'))} | {named_s} |"
            )
        lines.append("")

    # Section 2: Multi-Band Detection & FPR Call Audit
    lines.append("## 2. Multi-Band & Precision Data / API Audit")
    lines.append("")
    lines.append("### 2.1 Multi-Band 2AFC Detection Cells (`own` label)")
    lines.append("")
    lines.append("| Judge | Band | Total | Scored | Unp | Tie | Reas Tok | Poles | Neurons | Mean Neuron Acc | Cost USD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for akey in sorted(metrics_doc["detection_band_audit"]):
        ca = metrics_doc["detection_band_audit"][akey]
        lines.append(
            f"| `{ca['judge_model']}` | `{ca['band']}` | {ca['n_total']:,} | {ca['n_scored']:,} | "
            f"{ca['n_unparseable']} | {ca['n_logprob_tie']} | {ca['reasoning_tokens']} | "
            f"{ca['n_poles_covered']} | {ca['n_neurons_covered']} | {_fmt_num(ca['mean_neuron_accuracy'])} | "
            f"${ca['cost_usd']:.4f} |"
        )
    lines.append("")
    if metrics_doc.get("missing_neuron_band_cells"):
        lines.append(f"- Missing neuron×band cells (assigned lift `0.0`): `{metrics_doc['missing_neuron_band_cells']}`")
        lines.append("")

    lines.append("### 2.2 Single-Snippet YES/NO Precision Audit (`judgements_4b_fpr.jsonl`)")
    lines.append("")
    lines.append("| Judge | Probe Type | Total | Scored | Unp | Tie | Reas Tok | Cost USD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for akey in sorted(metrics_doc["fpr_probe_audit"]):
        a = metrics_doc["fpr_probe_audit"][akey]
        lines.append(
            f"| `{a['judge_model']}` | `{a['probe_type']}` | {a['n_total']:,} | {a['n_scored']:,} | "
            f"{a['n_unparseable']} | {a['n_logprob_tie']} | {a['reasoning_tokens']} | ${a['cost_usd']:.4f} |"
        )
    lines.append("")

    # Section 3: Main Surrogate Benchmark Table (n=278)
    lines.append("## 3. Causal Surrogate Benchmark (`n = 278` Neurons)")
    lines.append("")
    lines.append("| Candidate | Family | `sim_cal_rec` ρ [95% CI] | `sim_cal_rec` r | Partial ρ \\| `E(>=20%)` [95% CI] | p (partial) | Δρ vs `E(>=20%)` [95% CI] | `active` ρ | 5-Fold ρ (mean±SD) | AUC `>=0.25` | AUC `>=0.50` | P@20 `>=0.25` | P@50 `>=0.25` | AUC `<=0` |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cname, bm in benches.items():
        cs = bm["sim_cal_rec"]
        ca = bm["sim_cal_rec_active"]
        pp = bm["partial_above_energy_ge_20pct_sim_cal_rec"]
        dp = bm["delta_rho_vs_energy_ge_20pct_sim_cal_rec"]
        clf = bm["classification"]
        p_str = f"{pp['partial_spearman_p']:.2e}" if pp["partial_spearman_p"] is not None and math.isfinite(pp["partial_spearman_p"]) else "—"
        lines.append(
            f"| `{cname}` | `{bm['family']}` | "
            f"{_fmt_ci(cs['spearman_rho'], cs['spearman_ci_low'], cs['spearman_ci_high'])} | "
            f"{_fmt_num(cs['pearson_r'])} | "
            f"{_fmt_ci(pp['partial_spearman_rho'], pp['ci_low'], pp['ci_high'], signed=True)} | "
            f"{p_str} | "
            f"{_fmt_ci(dp['delta_rho'], dp['ci_low'], dp['ci_high'], signed=True)} | "
            f"{_fmt_num(ca['spearman_rho'])} | "
            f"{_fmt_num(bm['cv5_fold_spearman_mean'])}±{_fmt_num(bm['cv5_fold_spearman_sd'])} | "
            f"{_fmt_num(clf['pass_ge_0_25']['auroc'])} | "
            f"{_fmt_num(clf['pass_ge_0_50']['auroc'])} | "
            f"{_fmt_num(clf['pass_ge_0_25']['precision_at_20'])} | "
            f"{_fmt_num(clf['pass_ge_0_25']['precision_at_50'])} | "
            f"{_fmt_num(clf['fail_le_0_00']['auroc'])} |"
        )
    lines.append("")

    # Section 4: Negative-Recovery FPR Diagnostic
    lines.append("## 4. Ordinary-Text `FPR_ord` Negative-Recovery Diagnostic")
    lines.append("")
    lines.append("| Judge | Mean `FPR_ord` (all) | Mean `FPR_ord` (`sim<0`) | Mean `FPR_ord` (`sim>=0`) | Mean `FPR_ord` (`overbroad_fp`) | AUROC (`sim<0`) | P@20 (`sim<0`) | AUROC (`overbroad_fp`) | `p_yes_ord` AUROC (`sim<0`) | ρ(`FPR_ord`, `sim_cal_rec`) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for jm_short, d in metrics_doc["fpr_negative_recovery_diagnostic"].items():
        rc = d["fpr_ord_vs_sim_cal_rec"]
        lines.append(
            f"| `{jm_short}` | {_fmt_num(d['mean_fpr_ord_all'])} | {_fmt_num(d['mean_fpr_ord_sim_lt_0'])} | "
            f"{_fmt_num(d['mean_fpr_ord_sim_ge_0'])} | {_fmt_num(d['mean_fpr_ord_overbroad_bucket'])} | "
            f"{_fmt_num(d['fpr_ord_auroc_sim_lt_0'])} | {_fmt_num(d['fpr_ord_prec20_sim_lt_0'])} | "
            f"{_fmt_num(d['fpr_ord_auroc_overbroad_bucket'])} | {_fmt_num(d['p_yes_ord_auroc_sim_lt_0'])} | "
            f"{_fmt_ci(rc['spearman_rho'], rc['spearman_ci_low'], rc['spearman_ci_high'], signed=True)} |"
        )
    lines.append("")

    # Section 5: Named Neurons
    lines.append("## 5. Named Neurons (`info`, `PCR`, `insurance`)")
    lines.append("")
    lines.append("| Neuron | k | Bucket | `sim_cal_rec` | `active` | `strong20` | `top100` | `E(>=20%)` | `WDet (terra)` | `WDet (luna)` | `FPR_ord (terra)` | `TPR_top (terra)` | `WDet - FPR (terra)` | `CV rank-ridge` |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nname, nr in metrics_doc["named_neurons"].items():
        lines.append(
            f"| `{nname}` | {nr['k']} | `{nr['failure_bucket']}` | {_fmt_num(nr['sim_cal_rec'])} | "
            f"{_fmt_num(nr['sim_cal_rec_active'])} | {_fmt_num(nr['strong20_rec'])} | "
            f"{_fmt_num(nr['b0_top100_terra'])} | {_fmt_num(nr['b1_energy_ge_20pct_max'])} | "
            f"{_fmt_num(nr['s2_wdet_clamped_terra'])} | {_fmt_num(nr['s2_wdet_clamped_luna'])} | "
            f"{_fmt_num(nr['fpr_ord_terra'])} | {_fmt_num(nr['tpr_top_terra'])} | "
            f"{_fmt_num(nr['s4_wdet_minus_fpr_terra'])} | {_fmt_num(nr.get('s4_cv_rank_ridge_full_terra'))} |"
        )
    lines.append("")

    # Section 6: 60-Neuron Band KL Validation
    if kl_val is not None:
        lines.append(f"## 6. 60-Neuron `WDet_KL` vs `WDet_energy` Validation (`n = {kl_val['n_neurons']}`)")
        lines.append("")
        we_kl = kl_val["wdet_energy_vs_wdet_kl"]
        e20_kl = kl_val["energy_ge_20pct_vs_kl_ge_20pct"]
        lines.append(
            f"- `WDet_energy (4-band)` vs `WDet_KL (norm)`: "
            f"Spearman ρ = {_fmt_ci(we_kl['spearman_rho'], we_kl['spearman_ci_low'], we_kl['spearman_ci_high'])}, "
            f"Pearson r = {_fmt_num(we_kl['pearson_r'])}."
        )
        lines.append(
            f"- `EnergyShare(>=20% max)` vs `KLShare(>=20% max)`: "
            f"Spearman ρ = {_fmt_ci(e20_kl['spearman_rho'], e20_kl['spearman_ci_low'], e20_kl['spearman_ci_high'])}, "
            f"Pearson r = {_fmt_num(e20_kl['pearson_r'])}."
        )
        lines.append("")
        lines.append("| Metric (`n = 60` subset) | `sim_cal_rec` ρ [95% CI] | `sim_cal_rec` r | `sim_cal_rec_active` ρ |")
        lines.append("|---|---:|---:|---:|")
        for fld, cd in kl_val["correlations"].items():
            cs = cd["sim_cal_rec"]
            ca = cd["sim_cal_rec_active"]
            lines.append(
                f"| `{fld}` | {_fmt_ci(cs['spearman_rho'], cs['spearman_ci_low'], cs['spearman_ci_high'])} | "
                f"{_fmt_num(cs['pearson_r'])} | {_fmt_num(ca['spearman_rho'])} |"
            )
        lines.append("")

    lines.append(f"*{ULTO85_CREDIT_FOOTER}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 6: Clean 250-DPI Factual Figures (Zero Interpretive Prose)
# ---------------------------------------------------------------------------


def plot_gap_decomposition_figure(
    gap_doc: Mapping[str, Any],
    kl_validation: Mapping[str, Any] | None,
    output_path: Path = DEFAULT_FIGURE_GAP_PATH,
) -> None:
    """Plot gap decomposition figure (Panel A: band energy/KL by bucket; Panel B: Energy vs strong20_rec)."""
    assert_surrogate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(14.4, 6.1), dpi=250)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.05], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel A: Mean 4-band activation-energy share (and KL share when available) by bucket ----
    fb = gap_doc["failure_buckets_top100_ge_0_80"]
    buckets_dict = fb["buckets"]
    active_buckets = [b for b in FAILURE_BUCKET_ORDER if int(buckets_dict.get(b, {}).get("count", 0)) > 0]

    band_keys = ("p99.9+", "p97_99.9", "p75_97", "lt_p75")
    band_labels = ("p99.9+ (top 0.1%)", "p97–99.9 (2.9%)", "p75–97 (22%)", "<p75 (bulk 75%)")
    band_colors = ("#1b4965", "#3b8ea5", "#e9c46a", "#d97757")

    has_kl = kl_validation is not None and bool(kl_validation.get("by_bucket"))
    kl_by_bucket = kl_validation.get("by_bucket", {}) if has_kl and kl_validation else {}

    x_centers = np.arange(len(active_buckets), dtype=np.float64)
    bar_w = 0.34 if has_kl else 0.56

    for bi, bname in enumerate(active_buckets):
        mb = buckets_dict[bname]["mean_bands_4_energy"]
        bottom = 0.0
        x_pos = x_centers[bi] - (bar_w * 0.54 if has_kl else 0.0)
        for bk, blbl, bcol in zip(band_keys, band_labels, band_colors):
            val = float(mb[bk]) if mb.get(bk) is not None else 0.0
            ax_a.bar(
                x_pos,
                val,
                width=bar_w,
                bottom=bottom,
                color=bcol,
                edgecolor="white",
                linewidth=0.7,
                label=blbl if bi == 0 else None,
            )
            if val >= 0.095:
                txt_col = "white" if bk in ("p99.9+", "p97_99.9") else "#1a1a1a"
                ax_a.text(
                    x_pos,
                    bottom + val * 0.5,
                    f"{100.0 * val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=txt_col,
                    fontweight="bold",
                )
            bottom += val

        if has_kl and bname in kl_by_bucket and int(kl_by_bucket[bname].get("count", 0)) >= 1:
            mk = kl_by_bucket[bname]["mean_kl_shares_4band"]
            x_kl = x_centers[bi] + bar_w * 0.54
            bot_kl = 0.0
            for bk, bcol in zip(band_keys, band_colors):
                val = float(mk[bk])
                ax_a.bar(
                    x_kl,
                    val,
                    width=bar_w,
                    bottom=bot_kl,
                    color=bcol,
                    edgecolor="#222222",
                    linewidth=0.6,
                    alpha=0.72,
                    hatch="///",
                )
                if val >= 0.11:
                    ax_a.text(
                        x_kl,
                        bot_kl + val * 0.5,
                        f"{100.0 * val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color="#111111",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.78, edgecolor="none"),
                    )
                bot_kl += val
            kl_cnt = int(kl_by_bucket[bname]["count"])
            ax_a.text(
                x_kl,
                1.018,
                f"KL\n(n={kl_cnt})",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#333333",
            )
        if has_kl:
            ax_a.text(
                x_pos,
                1.018,
                "E",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#222222",
                fontweight="bold",
            )

    xtick_lbls = [
        f"{FAILURE_BUCKET_LABELS[b]}\n(n={buckets_dict[b]['count']})"
        for b in active_buckets
    ]
    ax_a.set_xticks(x_centers)
    ax_a.set_xticklabels(xtick_lbls, fontsize=8.0)
    ax_a.set_ylim(0.0, 1.15 if has_kl else 1.07)
    ax_a.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax_a.set_ylabel("Mean share of variance / KL across bands")
    title_a = f"A. Band energy share by bucket (top100 >= 0.80, n = {fb['n_total']})"
    if has_kl and kl_validation:
        title_a += f"\nSolid = L2 energy (n={fb['n_total']}); hatched = true KL (n={kl_validation['n_neurons']})"
    ax_a.set_title(title_a, fontsize=10.0, fontweight="bold", pad=8)
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=8.3,
    )
    ax_a.grid(axis="y", linestyle=":", alpha=0.45)
    ax_a.set_axisbelow(True)

    # ---- Panel B: EnergyShare(|a| >= 20% max) vs Ulto85 strong20_rec (n = 278) ----
    per_n = gap_doc["per_neuron"]
    ks = sorted(int(k) for k in per_n)
    x_e20 = np.asarray([float(per_n[str(k)]["energy_ge_20pct_max"]) for k in ks], dtype=np.float64)
    y_s20 = np.asarray([float(per_n[str(k)]["strong20_rec"]) for k in ks], dtype=np.float64)
    lays = np.asarray([int(per_n[str(k)]["layer"]) for k in ks], dtype=np.float64)

    y_lo_clip = -0.25
    y_hi_clip = 1.15
    clipped_lo = y_s20 < y_lo_clip
    clipped_hi = y_s20 > y_hi_clip
    in_bounds = (~clipped_lo) & (~clipped_hi)

    sc = ax_b.scatter(
        x_e20[in_bounds],
        y_s20[in_bounds],
        c=lays[in_bounds],
        cmap="viridis",
        s=24,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
        vmin=0,
        vmax=35,
        zorder=3,
    )
    if np.any(clipped_lo):
        ax_b.scatter(
            x_e20[clipped_lo],
            np.full(int(np.sum(clipped_lo)), y_lo_clip + 0.015),
            c=lays[clipped_lo],
            cmap="viridis",
            marker="v",
            s=30,
            alpha=0.9,
            edgecolors="#111111",
            linewidths=0.5,
            vmin=0,
            vmax=35,
            zorder=4,
        )
    if np.any(clipped_hi):
        ax_b.scatter(
            x_e20[clipped_hi],
            np.full(int(np.sum(clipped_hi)), y_hi_clip - 0.015),
            c=lays[clipped_hi],
            cmap="viridis",
            marker="^",
            s=30,
            alpha=0.9,
            edgecolors="#111111",
            linewidths=0.5,
            vmin=0,
            vmax=35,
            zorder=4,
        )

    # Diagonal y = x reference line
    ax_b.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.0, alpha=0.65, zorder=2)
    ax_b.axhline(0.25, linestyle=":", color="#888888", linewidth=0.9, alpha=0.65, zorder=2)

    # Mark named neurons cleanly
    named_offsets = {
        "info": (-0.20, -0.10),
        "PCR": (0.09, -0.09),
        "insurance": (0.10, 0.12),
    }
    for nname, k_val, _, _ in NAMED_NEURONS:
        nr = per_n.get(str(k_val))
        if not nr:
            continue
        nx = float(nr["energy_ge_20pct_max"])
        ny = float(np.clip(float(nr["strong20_rec"]), y_lo_clip + 0.02, y_hi_clip - 0.02))
        ax_b.scatter(
            [nx],
            [ny],
            s=62,
            facecolors="none",
            edgecolors="#c1121f",
            linewidths=1.8,
            zorder=5,
        )
        dx, dy = named_offsets.get(nname, (0.08, 0.08))
        tx = float(np.clip(nx + dx, 0.06, 0.84))
        ty = float(np.clip(ny + dy, y_lo_clip + 0.08, y_hi_clip - 0.10))
        ax_b.annotate(
            f"{nname} (k={k_val})",
            xy=(nx, ny),
            xytext=(tx, ty),
            fontsize=8.2,
            fontweight="bold",
            color="#8b0000",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#c1121f", alpha=0.92, linewidth=0.8),
            arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.0, shrinkA=2, shrinkB=4),
            zorder=6,
        )

    c_s20 = gap_doc["correlations"]["energy_ge_20pct_max_vs_strong20_rec"]
    stats_txt = (
        f"n = {c_s20['n']}\n"
        f"Spearman ρ = {c_s20['spearman_rho']:.3f} [{c_s20['spearman_ci_low']:.3f}, {c_s20['spearman_ci_high']:.3f}]\n"
        f"Pearson r = {c_s20['pearson_r']:.3f} (R² = {c_s20['r_squared']:.3f})"
    )
    if np.any(clipped_lo) or np.any(clipped_hi):
        stats_txt += f"\nClipped edge markers: {int(np.sum(clipped_lo | clipped_hi))}"
    ax_b.text(
        0.03,
        0.96,
        stats_txt,
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95),
        zorder=6,
    )

    cbar = fig.colorbar(sc, ax=ax_b, pad=0.02, fraction=0.045)
    cbar.set_label("Layer", fontsize=8.8)
    cbar.ax.tick_params(labelsize=8.0)

    ax_b.set_xlim(-0.03, 1.04)
    ax_b.set_ylim(y_lo_clip, y_hi_clip)
    ax_b.set_xlabel("Activation variance share with |a| >= 20% max (0-label)")
    ax_b.set_ylabel("Ulto85 strong20_rec (causal ablation recovery)")
    ax_b.set_title("B. Zero-label EnergyShare(>=20% max) vs. strong20_rec (n = 278)", fontsize=10.0, fontweight="bold", pad=8)
    ax_b.grid(True, linestyle=":", alpha=0.45)
    ax_b.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        ULTO85_CREDIT_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.06, right=0.95, top=0.88, bottom=0.22)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_surrogate_comparison_figure(
    metrics_doc: Mapping[str, Any],
    output_path: Path = DEFAULT_FIGURE_SURROGATE_PATH,
) -> None:
    """Plot surrogate comparison (Panel A: Spearman rho bars + CIs; Panel B: best surrogate vs sim_cal_rec)."""
    assert_surrogate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    benches = metrics_doc["benchmarks"]
    bar_spec: list[tuple[str, str, str]] = [
        ("b0_p999_terra", "B0: p99.9+ (terra)", "#9e9e9e"),
        ("b0_top100_terra", "B0: top100 (terra)", "#7a7a7a"),
        ("b1_energy_p999", "B1: Energy p99.9+ (0-label)", "#6c757d"),
        ("b1_energy_ge_20pct_max", "B1: Energy >=20% max (0-label)", "#495057"),
        ("s1_unweighted_5band_p999_terra", "S1: Unweighted 5-band (terra)", "#5c80bc"),
        ("s2_wdet_clamped_terra", "S2: WDet p99.9+ (terra)", "#2a6f97"),
        ("s2_wdet_clamped_top100_terra", "S2: WDet top100 (terra)", "#1d5b79"),
        ("s2_wdet_clamped_luna", "S2: WDet p99.9+ (luna <$1)", "#468faf"),
        ("s3_one_minus_fpr_ord_terra", "S3: 1 - FPR_ord (terra)", "#e76f51"),
        ("s3_tpr_minus_fpr_terra", "S3: TPR_top - FPR_ord (terra)", "#f4a261"),
        ("s4_wdet_minus_fpr_terra", "S4: WDet - FPR_ord (terra, 0-p)", "#2a9d8f"),
        ("s4_wdet_top100_times_1m_fpr_terra", "S4: WDet(top)*(1-FPR) (terra)", "#21867a"),
        ("s4_wdet_minus_fpr_luna", "S4: WDet - FPR_ord (luna, 0-p)", "#52b788"),
        ("s4_cv_ridge_full_terra", "S4: 5-fold OOF ridge (terra)", "#1d3557"),
        ("s4_cv_rank_ridge_full_luna", "S4: 5-fold OOF rank-ridge (luna)", "#0b2545"),
    ]

    present = [
        (key, lbl, col)
        for key, lbl, col in bar_spec
        if key in benches
        and benches[key]["sim_cal_rec"]["spearman_rho"] is not None
        and math.isfinite(float(benches[key]["sim_cal_rec"]["spearman_rho"]))
    ]

    fig = plt.figure(figsize=(14.8, 6.4), dpi=250)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.05], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel A: Spearman rho [95% CI] bar chart ----
    y_pos = np.arange(len(present))[::-1]
    rhos = [float(benches[k]["sim_cal_rec"]["spearman_rho"]) for k, _, _ in present]
    lo_errs = [
        max(0.0, float(benches[k]["sim_cal_rec"]["spearman_rho"]) - float(benches[k]["sim_cal_rec"]["spearman_ci_low"]))
        if math.isfinite(float(benches[k]["sim_cal_rec"]["spearman_ci_low"]))
        else 0.0
        for k, _, _ in present
    ]
    hi_errs = [
        max(0.0, float(benches[k]["sim_cal_rec"]["spearman_ci_high"]) - float(benches[k]["sim_cal_rec"]["spearman_rho"]))
        if math.isfinite(float(benches[k]["sim_cal_rec"]["spearman_ci_high"]))
        else 0.0
        for k, _, _ in present
    ]
    colors = [col for _, _, col in present]
    labels = [lbl for _, lbl, _ in present]

    ax_a.barh(
        y_pos,
        rhos,
        height=0.66,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_a.errorbar(
        rhos,
        y_pos,
        xerr=[lo_errs, hi_errs],
        fmt="none",
        ecolor="#1a1a1a",
        elinewidth=1.1,
        capsize=2.6,
        zorder=4,
    )

    e20_rho = float(benches["b1_energy_ge_20pct_max"]["sim_cal_rec"]["spearman_rho"])
    ax_a.axvline(0.0, color="#222222", linewidth=0.8, zorder=2)
    if math.isfinite(e20_rho):
        ax_a.axvline(
            e20_rho,
            color="#495057",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            zorder=2,
        )

    max_right = max(
        (r + hi for r, hi in zip(rhos, hi_errs)),
        default=0.6,
    )
    min_left = min(
        (r - lo for r, lo in zip(rhos, lo_errs)),
        default=0.0,
    )
    x_lo = min(-0.08, min_left - 0.06)
    x_hi = max(0.65, max_right + 0.16)
    ax_a.set_xlim(x_lo, x_hi)

    for yp_i, r_val, hi_e in zip(y_pos, rhos, hi_errs):
        x_txt = max(r_val + hi_e, 0.0) + 0.015
        ax_a.text(
            x_txt,
            yp_i,
            f"{r_val:+.3f}",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#111111",
        )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels, fontsize=8.3)
    ax_a.set_xlabel("Spearman ρ with Ulto85 sim_cal_rec (95% neuron bootstrap CI)")
    ax_a.set_title(
        f"A. Cheap surrogate rank correlation with sim_cal_rec (n = {metrics_doc['n_neurons']})\n"
        f"Dashed line = zero-label EnergyShare(>=20% max) (ρ = {e20_rho:.3f})",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax_a.grid(axis="x", linestyle=":", alpha=0.45)
    ax_a.set_axisbelow(True)

    # ---- Panel B: Best cheap surrogate vs sim_cal_rec scatter ----
    best_key = str(metrics_doc.get("best_overall_llm_surrogate", "s2_wdet_clamped_terra"))
    # Prefer a finite candidate
    if not math.isfinite(float(benches[best_key]["sim_cal_rec"]["spearman_rho"])):
        best_key = "s2_wdet_clamped_terra"
    per_n = metrics_doc["per_neuron"]
    ks = sorted(int(k) for k in per_n)
    x_best = np.asarray([float(per_n[str(k)][best_key]) for k in ks], dtype=np.float64)
    y_sim = np.asarray([float(per_n[str(k)]["sim_cal_rec"]) for k in ks], dtype=np.float64)
    lays = np.asarray([int(per_n[str(k)]["layer"]) for k in ks], dtype=np.float64)

    fin = np.isfinite(x_best) & np.isfinite(y_sim)
    x_best, y_sim, lays = x_best[fin], y_sim[fin], lays[fin]
    ks_fin = [k for k, ok in zip(ks, fin) if ok]

    y_lo_clip = -0.80
    y_hi_clip = 1.08
    clipped_lo = y_sim < y_lo_clip
    clipped_hi = y_sim > y_hi_clip
    in_bounds = (~clipped_lo) & (~clipped_hi)

    sc = ax_b.scatter(
        x_best[in_bounds],
        y_sim[in_bounds],
        c=lays[in_bounds],
        cmap="viridis",
        s=25,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
        vmin=0,
        vmax=35,
        zorder=3,
    )
    if np.any(clipped_lo):
        ax_b.scatter(
            x_best[clipped_lo],
            np.full(int(np.sum(clipped_lo)), y_lo_clip + 0.025),
            c=lays[clipped_lo],
            cmap="viridis",
            marker="v",
            s=32,
            alpha=0.92,
            edgecolors="#111111",
            linewidths=0.55,
            vmin=0,
            vmax=35,
            zorder=4,
        )
    if np.any(clipped_hi):
        ax_b.scatter(
            x_best[clipped_hi],
            np.full(int(np.sum(clipped_hi)), y_hi_clip - 0.025),
            c=lays[clipped_hi],
            cmap="viridis",
            marker="^",
            s=32,
            alpha=0.92,
            edgecolors="#111111",
            linewidths=0.55,
            vmin=0,
            vmax=35,
            zorder=4,
        )

    ax_b.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.55, zorder=2)
    ax_b.axhline(0.50, color="#666666", linestyle=":", linewidth=0.9, alpha=0.65, zorder=2)

    x_min, x_max = float(np.min(x_best)), float(np.max(x_best))
    x_span = max(x_max - x_min, 1e-6)
    ax_b.set_xlim(x_min - 0.06 * x_span, x_max + 0.08 * x_span)
    ax_b.set_ylim(y_lo_clip, y_hi_clip)

    # Mark info, PCR, insurance cleanly
    k_to_pos = {k: i for i, k in enumerate(ks_fin)}
    named_rel_offsets = {
        "info": (-0.36, -0.28),
        "PCR": (0.14, -0.09),
        "insurance": (-0.28, 0.25),
    }
    for nname, k_val, _, _ in NAMED_NEURONS:
        if k_val not in k_to_pos:
            continue
        pi = k_to_pos[k_val]
        nx = float(x_best[pi])
        ny = float(np.clip(float(y_sim[pi]), y_lo_clip + 0.03, y_hi_clip - 0.03))
        ax_b.scatter(
            [nx],
            [ny],
            s=64,
            facecolors="none",
            edgecolors="#c1121f",
            linewidths=1.8,
            zorder=5,
        )
        rx, ry = named_rel_offsets.get(nname, (0.10, 0.12))
        tx = float(np.clip(nx + rx * x_span, x_min + 0.04 * x_span, x_max - 0.22 * x_span))
        ty = float(np.clip(ny + ry, y_lo_clip + 0.10, y_hi_clip - 0.14))
        ax_b.annotate(
            f"{nname} (k={k_val})",
            xy=(nx, ny),
            xytext=(tx, ty),
            fontsize=8.0,
            fontweight="bold",
            color="#8b0000",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#c1121f", alpha=0.92, linewidth=0.8),
            arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.0, shrinkA=2, shrinkB=4),
            zorder=6,
        )

    bm_best = benches[best_key]
    cs = bm_best["sim_cal_rec"]
    pp = bm_best["partial_above_energy_ge_20pct_sim_cal_rec"]
    clf = bm_best["classification"]
    box_txt = (
        f"Surrogate: {best_key}\n"
        f"Spearman ρ = {cs['spearman_rho']:.3f} [{cs['spearman_ci_low']:.3f}, {cs['spearman_ci_high']:.3f}]\n"
        f"Partial ρ | E(>=20%) = {pp['partial_spearman_rho']:+.3f}\n"
        f"AUROC (>=0.25 / >=0.50) = {clf['pass_ge_0_25']['auroc']:.3f} / {clf['pass_ge_0_50']['auroc']:.3f}"
    )
    if np.any(clipped_lo) or np.any(clipped_hi):
        box_txt += f"\nClipped at y={y_lo_clip:.2f}: {int(np.sum(clipped_lo | clipped_hi))} neurons"
    ax_b.text(
        0.025,
        0.97,
        box_txt,
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95),
        zorder=6,
    )

    cbar = fig.colorbar(sc, ax=ax_b, pad=0.02, fraction=0.045)
    cbar.set_label("Layer", fontsize=8.8)
    cbar.ax.tick_params(labelsize=8.0)

    ax_b.set_xlabel(f"Surrogate score ({best_key})")
    ax_b.set_ylabel("Ulto85 sim_cal_rec (full-distribution causal recovery)")
    ax_b.set_title(
        f"B. {best_key} vs. sim_cal_rec (n = {len(ks_fin)})",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax_b.grid(True, linestyle=":", alpha=0.45)
    ax_b.set_axisbelow(True)

    fig.text(
        0.5,
        0.012,
        ULTO85_CREDIT_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.14)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-band detection, YES/NO precision audit, and causal surrogate evaluation for Qwen3-4B."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_mb = sub.add_parser("run-multiband", help="Run multi-band 2AFC detection sweeps into judgements_4b_multiband.jsonl.")
    p_mb.add_argument("--active-poles", type=Path, default=DEFAULT_ACTIVE_POLES_PATH)
    p_mb.add_argument("--explanations", type=Path, default=DEFAULT_EXPLANATIONS_PATH)
    p_mb.add_argument("--multiband-snippets", type=Path, default=DEFAULT_SNIPPETS_MULTIBAND_PATH)
    p_mb.add_argument("--snippets-4b", type=Path, default=DEFAULT_SNIPPETS_4B_PATH)
    p_mb.add_argument("--output", type=Path, default=DEFAULT_JUDGEMENTS_MULTIBAND_PATH)
    p_mb.add_argument("--companion-fpr-output", type=Path, default=DEFAULT_JUDGEMENTS_FPR_PATH)
    p_mb.add_argument("--sweeps", nargs="+", choices=list(MULTIBAND_SWEEPS), default=list(MULTIBAND_SWEEPS))
    p_mb.add_argument("--max-usd", type=float, default=DEFAULT_SURROGATE_MAX_USD)
    p_mb.add_argument("--concurrency", type=int, default=32)
    p_mb.add_argument("--target-rpm", type=float, default=3500.0)
    p_mb.add_argument("--env-file", type=Path, default=Path(".env"))
    p_mb.add_argument("--dry-run", action="store_true")

    p_fpr = sub.add_parser("run-fpr", help="Run single-snippet YES/NO precision audit into judgements_4b_fpr.jsonl.")
    p_fpr.add_argument("--active-poles", type=Path, default=DEFAULT_ACTIVE_POLES_PATH)
    p_fpr.add_argument("--explanations", type=Path, default=DEFAULT_EXPLANATIONS_PATH)
    p_fpr.add_argument("--fpr-snippets", type=Path, default=DEFAULT_SNIPPETS_FPR_PATH)
    p_fpr.add_argument("--output", type=Path, default=DEFAULT_JUDGEMENTS_FPR_PATH)
    p_fpr.add_argument("--companion-multiband-output", type=Path, default=DEFAULT_JUDGEMENTS_MULTIBAND_PATH)
    p_fpr.add_argument("--sweeps", nargs="+", choices=list(FPR_SWEEPS), default=list(FPR_SWEEPS))
    p_fpr.add_argument("--max-usd", type=float, default=DEFAULT_SURROGATE_MAX_USD)
    p_fpr.add_argument("--concurrency", type=int, default=32)
    p_fpr.add_argument("--target-rpm", type=float, default=3500.0)
    p_fpr.add_argument("--env-file", type=Path, default=Path(".env"))
    p_fpr.add_argument("--dry-run", action="store_true")

    p_an = sub.add_parser("analyze", help="Compute surrogate metrics, summary markdown, and 2 PNG figures.")
    p_an.add_argument("--active-poles", type=Path, default=DEFAULT_ACTIVE_POLES_PATH)
    p_an.add_argument("--scores", type=Path, default=DEFAULT_SCORES_4B_PATH)
    p_an.add_argument("--gap-decomp", type=Path, default=DEFAULT_GAP_DECOMP_PATH)
    p_an.add_argument("--judgements-4b", type=Path, default=DEFAULT_JUDGEMENTS_4B_PATH)
    p_an.add_argument("--judgements-top100", type=Path, default=DEFAULT_JUDGEMENTS_4B_TOP100_PATH)
    p_an.add_argument("--judgements-multiband", type=Path, default=DEFAULT_JUDGEMENTS_MULTIBAND_PATH)
    p_an.add_argument("--judgements-fpr", type=Path, default=DEFAULT_JUDGEMENTS_FPR_PATH)
    p_an.add_argument("--band-kl", type=Path, default=DEFAULT_BAND_KL_PATH)
    p_an.add_argument("--output-metrics", type=Path, default=DEFAULT_SURROGATE_METRICS_PATH)
    p_an.add_argument("--output-summary", type=Path, default=DEFAULT_SURROGATE_SUMMARY_PATH)
    p_an.add_argument("--figure-gap", type=Path, default=DEFAULT_FIGURE_GAP_PATH)
    p_an.add_argument("--figure-surrogate", type=Path, default=DEFAULT_FIGURE_SURROGATE_PATH)
    p_an.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_B)
    p_an.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args(argv)

    if args.command == "run-multiband":
        load_dotenv_file(args.env_file)
        active_poles = load_active_poles(args.active_poles)
        explanations_by_k = (
            json.loads(args.explanations.read_text(encoding="utf-8"))
            if args.explanations.is_file()
            else None
        )
        mb_snips = load_jsonl(args.multiband_snippets)
        s4b_snips = load_jsonl(args.snippets_4b) if "p97_99_own_luna" in args.sweeps else None
        plan = build_multiband_task_plan(
            active_poles=active_poles,
            multiband_snippets=mb_snips,
            snippets_4b=s4b_snips,
            explanations_by_k=explanations_by_k,
            sweeps=args.sweeps,
        )
        completed = load_completed_keys(args.output)
        pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed]
        quote = quote_4b_plan_cost(pending)
        prior_combined = _sum_existing_spend([args.output, args.companion_fpr_output])
        print(
            json.dumps(
                {
                    "command": "run-multiband",
                    "sweeps": list(args.sweeps),
                    "total_planned": len(plan),
                    "already_completed": len(plan) - len(pending),
                    "pending": len(pending),
                    "prior_combined_usd": round(prior_combined, 6),
                    "max_usd": args.max_usd,
                    "pre_run_quote": quote,
                },
                indent=2,
            )
        )
        if args.dry_run:
            return 0
        summary = asyncio.run(
            run_multiband_judge_async(
                plan=plan,
                output_path=args.output,
                companion_spend_paths=(args.companion_fpr_output,),
                max_usd=args.max_usd,
                concurrency=args.concurrency,
                target_rpm=args.target_rpm,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "run-fpr":
        load_dotenv_file(args.env_file)
        active_poles = load_active_poles(args.active_poles)
        explanations_by_k = (
            json.loads(args.explanations.read_text(encoding="utf-8"))
            if args.explanations.is_file()
            else None
        )
        fpr_snips = load_jsonl(args.fpr_snippets)
        plan = build_fpr_task_plan(
            active_poles=active_poles,
            fpr_snippets=fpr_snips,
            explanations_by_k=explanations_by_k,
            sweeps=args.sweeps,
        )
        completed = load_completed_fpr_keys(args.output)
        pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed]
        quote = quote_fpr_plan_cost(pending)
        prior_combined = _sum_existing_spend([args.output, args.companion_multiband_output])
        print(
            json.dumps(
                {
                    "command": "run-fpr",
                    "sweeps": list(args.sweeps),
                    "total_planned": len(plan),
                    "already_completed": len(plan) - len(pending),
                    "pending": len(pending),
                    "prior_combined_usd": round(prior_combined, 6),
                    "max_usd": args.max_usd,
                    "pre_run_quote": quote,
                },
                indent=2,
            )
        )
        if args.dry_run:
            return 0
        summary = asyncio.run(
            run_fpr_judge_async(
                plan=plan,
                output_path=args.output,
                companion_spend_paths=(args.companion_multiband_output,),
                max_usd=args.max_usd,
                concurrency=args.concurrency,
                target_rpm=args.target_rpm,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "analyze":
        doc = analyze_surrogates_4b(
            active_poles_path=args.active_poles,
            scores_path=args.scores,
            gap_decomp_path=args.gap_decomp,
            judgements_4b_path=args.judgements_4b,
            judgements_top100_path=args.judgements_top100,
            judgements_multiband_path=args.judgements_multiband,
            judgements_fpr_path=args.judgements_fpr,
            band_kl_path=args.band_kl,
            output_metrics_path=args.output_metrics,
            output_summary_path=args.output_summary,
            figure_gap_path=args.figure_gap,
            figure_surrogate_path=args.figure_surrogate,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "n_neurons": doc["n_neurons"],
                    "surrogate_spend": doc["surrogate_spend"],
                    "best_zero_param_surrogate": doc["best_zero_param_surrogate"],
                    "best_overall_llm_surrogate": doc["best_overall_llm_surrogate"],
                    "has_kl_validation": doc["kl_60_neuron_validation"] is not None,
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
