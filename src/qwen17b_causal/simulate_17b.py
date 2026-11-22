"""Causal simulator & YES/NO precision runner for Qwen/Qwen3-1.7B (n=50 neurons).

Strictly isolated under `results/qwen17b_causal/`:
- Reads frozen `Qwen3-1.7B` labels from `data/labels.jsonl` and `data/poles.json` (read-only)
- Reads disjoint odd-doc eval windows from `data/qwen17b_causal/sim_windows_17b.jsonl`
- Reads disjoint odd-doc precision probes from `data/qwen17b_causal/fpr_probes_17b.jsonl`
- Writes strictly to:
  - `results/qwen17b_causal/sim_calls_17b.jsonl` (hard `--max-usd 3.50`, combined `<= $4.50`)
  - `results/qwen17b_causal/fpr_17b.jsonl` (hard `--max-usd 1.00`, combined `<= $4.50`)

Zero existing `Qwen3-1.7B` or `Qwen3-4B` files are ever modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from qwen4b_eval.judge_4b import estimate_prompt_tokens_conservative
from qwen4b_eval.surrogate_4b import (
    FprProbeTask,
    _call_openai_fpr_once,
    _sum_existing_spend,
    build_fpr_messages,
    load_completed_fpr_keys,
    quote_fpr_plan_cost,
)
from w_audit.judge import (
    AdaptiveRateLimiter,
    SpendLimitExceededError,
    _extract_retry_after,
    _is_rate_limit_error,
    sanitize_jsonl_tail,
)
from w_audit.labeler import load_jsonl
from w_audit.probe_limits import (
    MODEL_PRICING_PER_M,
    compute_call_cost_usd,
    load_dotenv_file,
)

# ---------------------------------------------------------------------------
# Paths & Strict Isolation Guards
# ---------------------------------------------------------------------------

DEFAULT_POLES_17B_PATH: Path = Path("data/poles.json")
DEFAULT_LABELS_17B_PATH: Path = Path("data/labels.jsonl")
DEFAULT_SIM_WINDOWS_17B_PATH: Path = Path("data/qwen17b_causal/sim_windows_17b.jsonl")
DEFAULT_FPR_PROBES_17B_PATH: Path = Path("data/qwen17b_causal/fpr_probes_17b.jsonl")

DEFAULT_SIM_CALLS_17B_PATH: Path = Path("results/qwen17b_causal/sim_calls_17b.jsonl")
DEFAULT_FPR_17B_PATH: Path = Path("results/qwen17b_causal/fpr_17b.jsonl")

DEFAULT_SIM_MAX_USD: float = 3.50
DEFAULT_FPR_MAX_USD: float = 1.00
DEFAULT_CAUSAL_17B_MAX_USD: float = 4.50

SIM_API_MODEL: str = "gpt-6.1-terra"
SIM_JUDGE_TAG: str = "gpt-6.1-terra@none"

FPR_17B_SWEEPS: tuple[str, ...] = ("fpr_core_terra", "fpr_all_luna")

PROTECTED_CAUSAL_17B_FILENAMES: frozenset[str] = frozenset(
    {
        "judgements.jsonl",
        "judgements_unfiltered_context_ablation.jsonl",
        "judgements_examples_as_label_exploratory.jsonl",
        "labels.jsonl",
        "snippets.jsonl",
        "poles.json",
        "README.md",
        "README.md",
        "judgements_4b.jsonl",
        "judgements_4b_top100.jsonl",
        "judgements_4b_multiband.jsonl",
        "judgements_4b_fpr.jsonl",
        "metrics_4b.json",
        "summary_4b.md",
        "surrogate_metrics_4b.json",
        "surrogate_summary_4b.md",
        "gap_decomposition_4b.json",
        "band_kl_4b.json",
        "energy_and_kl_17b.json",
        "causal_scores_17b.json",
        "sim_windows_17b.jsonl",
        "fpr_probes_17b.jsonl",
    }
)

# Concise, neutral bipolar sparse JSON simulation prompt (zero hype)
SIM_17B_SYSTEM_PROMPT: str = (
    "Simulate a bipolar MLP neuron on a 64-token window given its positive (+1..+5) "
    "and negative (-1..-5) pole descriptions. For each token index 0..63, predict an "
    "integer from -5 to +5 (+1..+5 matches positive pole, -1..-5 matches negative pole, "
    "0 neither). Return ONLY a compact JSON object of NONZERO indices, e.g. "
    '{"14":4,"29":-2}, or {} if none activate. Omit zeros; no prose.'
)

SIM_17B_USER_TEMPLATE: str = (
    "Positive pole (+1..+5): {pos_label}\n"
    "Negative pole (-1..-5): {neg_label}\n\n"
    "Text:\n{passage_text}\n\n"
    "Tokens (index:token, 0..{last_idx}):\n{numbered_tokens}\n\n"
    "Sparse JSON:"
)


def assert_17b_causal_output_path(path: Path) -> None:
    """Refuse writes targeting any frozen 1.7B, 4B, or extractor-owned artifact."""
    if path.name in PROTECTED_CAUSAL_17B_FILENAMES:
        raise ValueError(f"Refusing to write 1.7B causal output into protected file: {path}")


# ---------------------------------------------------------------------------
# Frozen Qwen3-1.7B Pole & Label Loaders
# ---------------------------------------------------------------------------


def load_poles_17b(poles_path: Path = DEFAULT_POLES_17B_PATH) -> list[dict[str, Any]]:
    """Load frozen Qwen3-1.7B poles from data/poles.json."""
    raw = json.loads(poles_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(p) for p in raw]
    if isinstance(raw, dict) and isinstance(raw.get("poles"), list):
        return [dict(p) for p in raw["poles"]]
    raise ValueError(f"Unrecognized poles.json format in {poles_path}")


def load_labels_17b(labels_path: Path = DEFAULT_LABELS_17B_PATH) -> dict[str, dict[str, Any]]:
    """Load frozen gpt-4.1 pole descriptions keyed by pole_id."""
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(labels_path):
        pid = str(row["pole_id"])
        out[pid] = dict(row)
    return out


def resolve_17b_pole_label_text(label_row: Mapping[str, Any]) -> str:
    """Extract canonical natural-language pole description from data/labels.jsonl."""
    for key in ("own_label", "own_description", "description", "label"):
        val = label_row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise ValueError(f"Missing label text in 1.7B label row: {label_row!r}")


def group_17b_neuron_labels(
    poles: Sequence[Mapping[str, Any]],
    labels_by_pole: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map neuron_id ('L04_N2467') -> {layer, neuron, pos_pole_id, neg_pole_id, pos_label, neg_label}."""
    by_nid: dict[str, dict[str, Any]] = {}
    for p in poles:
        layer = int(p["layer"])
        neuron = int(p["neuron"])
        nid = str(p.get("neuron_id") or f"L{layer:02d}_N{neuron:04d}")
        pid = str(p["pole_id"])
        sign = int(p["sign"])
        if nid not in by_nid:
            by_nid[nid] = {
                "neuron_id": nid,
                "layer": layer,
                "neuron": neuron,
                "pos_pole_id": None,
                "neg_pole_id": None,
                "pos_label": "",
                "neg_label": "",
            }
        lrow = labels_by_pole.get(pid)
        ltxt = resolve_17b_pole_label_text(lrow) if lrow is not None else ""
        if sign > 0:
            by_nid[nid]["pos_pole_id"] = pid
            by_nid[nid]["pos_label"] = ltxt
        else:
            by_nid[nid]["neg_pole_id"] = pid
            by_nid[nid]["neg_label"] = ltxt
    return by_nid


# ---------------------------------------------------------------------------
# Part 1: Sparse JSON Bipolar Activation Simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimWindowTask:
    """Single 64-token causal simulation task for one Qwen3-1.7B neuron."""

    window_id: str
    neuron_id: str
    layer: int
    neuron: int
    stratum: str
    window_idx: int
    seq_len: int
    token_strs: tuple[str, ...]
    passage_text: str
    pos_label: str
    neg_label: str
    judge_model: str = SIM_JUDGE_TAG
    api_model: str = SIM_API_MODEL
    reasoning_effort: str = "none"
    max_completion_tokens: int = 220

    @property
    def key(self) -> tuple[str, str]:
        return (self.window_id, self.judge_model)


def format_numbered_tokens(token_strs: Sequence[str]) -> str:
    """Format token sequence compactly and unambiguously as `0:tok0\\n1:tok1\\n...`."""
    lines = []
    for i, t in enumerate(token_strs):
        esc = str(t).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        lines.append(f"{i}:{esc}")
    return "\n".join(lines)


def build_sim_17b_messages(
    pos_label: str,
    neg_label: str,
    token_strs: Sequence[str],
    passage_text: str | None = None,
) -> list[dict[str, str]]:
    """Build exact 1-window sparse JSON bipolar simulation prompt."""
    if not token_strs:
        raise ValueError("token_strs must be non-empty.")
    text = passage_text if (passage_text is not None and passage_text.strip()) else "".join(token_strs)
    pos_t = pos_label.strip() or "(inactive / no description)"
    neg_t = neg_label.strip() or "(inactive / no description)"
    numbered = format_numbered_tokens(token_strs)
    return [
        {"role": "system", "content": SIM_17B_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SIM_17B_USER_TEMPLATE.format(
                pos_label=pos_t,
                neg_label=neg_t,
                passage_text=text,
                last_idx=len(token_strs) - 1,
                numbered_tokens=numbered,
            ),
        },
    ]


def parse_sparse_ratings(
    raw_text: str,
    seq_len: int,
    clamp_abs: float = 5.0,
) -> tuple[list[float], dict[str, float], Literal["ok", "empty_ok", "fallback_regex", "parse_error"]]:
    """Deterministically parse sparse JSON index->rating output into exact length-`seq_len` float array."""
    ratings = [0.0] * int(seq_len)
    sparse: dict[str, float] = {}
    cleaned = raw_text.strip()
    if not cleaned:
        return ratings, sparse, "parse_error"

    # Strip optional markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    body = fence_match.group(1).strip() if fence_match else cleaned

    parsed_obj: Any = None
    try:
        parsed_obj = json.loads(body)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        b_start = body.find("{")
        b_end = body.rfind("}")
        if b_start != -1 and b_end > b_start:
            try:
                parsed_obj = json.loads(body[b_start : b_end + 1])
            except json.JSONDecodeError:
                parsed_obj = None

    def _record_pair(k_raw: Any, v_raw: Any) -> None:
        try:
            idx = int(str(k_raw).strip())
            val = float(v_raw)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        if 0 <= idx < seq_len:
            clamped = float(max(-clamp_abs, min(clamp_abs, val)))
            ratings[idx] = clamped
            if abs(clamped) > 0.0:
                sparse[str(idx)] = clamped

    if isinstance(parsed_obj, Mapping):
        # Unwrap optional wrapper dict like {"activations": {...}} or {"ratings": [...]}
        inner: Any = parsed_obj
        if len(parsed_obj) == 1:
            only_k, only_v = next(iter(parsed_obj.items()))
            if not str(only_k).strip().isdigit() and isinstance(only_v, (Mapping, list)):
                inner = only_v
        if isinstance(inner, Mapping):
            for k_raw, v_raw in inner.items():
                _record_pair(k_raw, v_raw)
            status: Literal["ok", "empty_ok"] = "empty_ok" if len(sparse) == 0 else "ok"
            return ratings, sparse, status
        if isinstance(inner, list):
            if len(inner) == seq_len and all(isinstance(x, (int, float)) for x in inner):
                for idx, v_raw in enumerate(inner):
                    _record_pair(idx, v_raw)
            else:
                for item in inner:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        _record_pair(item[0], item[1])
                    elif isinstance(item, Mapping):
                        idx_k = item.get("index", item.get("idx", item.get("i")))
                        val_k = item.get("rating", item.get("val", item.get("activation", item.get("score"))))
                        _record_pair(idx_k, val_k)
            return ratings, sparse, ("empty_ok" if len(sparse) == 0 else "ok")

    if isinstance(parsed_obj, list):
        if len(parsed_obj) == seq_len and all(isinstance(x, (int, float)) for x in parsed_obj):
            for idx, v_raw in enumerate(parsed_obj):
                _record_pair(idx, v_raw)
            return ratings, sparse, ("empty_ok" if len(sparse) == 0 else "ok")
        for item in parsed_obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                _record_pair(item[0], item[1])
        if sparse or len(parsed_obj) == 0:
            return ratings, sparse, ("empty_ok" if len(sparse) == 0 else "ok")

    # Fallback regex recovery for truncated JSON objects like `{"3": 2, "15": -4, "28":`
    matches = re.findall(r'"?(\d{1,3})"?\s*:\s*(-?\d+(?:\.\d+)?)', body)
    if matches:
        for k_s, v_s in matches:
            _record_pair(k_s, v_s)
        return ratings, sparse, "fallback_regex"

    return ratings, sparse, "parse_error"


def load_completed_sim_keys(sim_calls_path: Path) -> set[tuple[str, str]]:
    """Load already-scored (window_id, judge_model) keys for crash-safe resume."""
    keys: set[tuple[str, str]] = set()
    for row in load_jsonl(sim_calls_path):
        wid = row.get("window_id")
        jm = row.get("judge_model")
        if wid is not None and jm is not None:
            keys.add((str(wid), str(jm)))
    return keys


def build_sim_17b_task_plan(
    sim_windows: Sequence[Mapping[str, Any]],
    neuron_labels: Mapping[str, Mapping[str, Any]],
    api_model: str = SIM_API_MODEL,
) -> list[tuple[SimWindowTask, list[dict[str, str]], dict[str, Any]]]:
    """Build deterministic 1-window-per-call causal simulation task plan."""
    jm_tag = f"{api_model}@none"
    plan: list[tuple[SimWindowTask, list[dict[str, str]], dict[str, Any]]] = []
    seen_keys: set[tuple[str, str]] = set()

    for idx_row, w in enumerate(sim_windows):
        layer = int(w["layer"])
        neuron = int(w["neuron"])
        nid = str(w.get("neuron_id") or f"L{layer:02d}_N{neuron:04d}")
        if nid not in neuron_labels:
            raise KeyError(f"Missing frozen 1.7B labels for neuron_id={nid!r}")
        nl = neuron_labels[nid]

        stratum = str(w.get("stratum", w.get("split", "ordinary")))
        w_idx = int(w.get("window_idx", w.get("seq_idx", idx_row)))
        wid = str(w.get("window_id") or f"{nid}__{stratum}__{w_idx}")

        toks_raw = w.get("tokens_str", w.get("token_strs", w.get("tokens")))
        if not isinstance(toks_raw, list) or len(toks_raw) == 0:
            raise ValueError(f"Window {wid!r} missing non-empty tokens_str list.")
        tok_tuple = tuple(str(t) for t in toks_raw)
        seq_len = int(w.get("seq_len", len(tok_tuple)))
        if seq_len != len(tok_tuple):
            raise ValueError(f"Window {wid!r}: seq_len={seq_len} != len(token_strs)={len(tok_tuple)}")
        passage = str(w.get("text", w.get("window_text", "".join(tok_tuple))))

        task = SimWindowTask(
            window_id=wid,
            neuron_id=nid,
            layer=layer,
            neuron=neuron,
            stratum=stratum,
            window_idx=w_idx,
            seq_len=seq_len,
            token_strs=tok_tuple,
            passage_text=passage,
            pos_label=str(nl["pos_label"]),
            neg_label=str(nl["neg_label"]),
            judge_model=jm_tag,
            api_model=api_model,
            reasoning_effort="none",
            max_completion_tokens=220,
        )
        if task.key in seen_keys:
            continue
        seen_keys.add(task.key)
        msgs = build_sim_17b_messages(
            pos_label=task.pos_label,
            neg_label=task.neg_label,
            token_strs=task.token_strs,
            passage_text=task.passage_text,
        )
        extra_meta = {
            "doc_id": w.get("doc_id"),
            "chunk_id": w.get("chunk_id"),
            "start_pos": w.get("start_pos"),
            "seq_idx_in_eval": w.get("seq_idx_in_eval"),
        }
        plan.append((task, msgs, extra_meta))

    return plan


def quote_sim_17b_plan_cost(
    plan: Sequence[tuple[SimWindowTask, list[dict[str, str]], dict[str, Any]]],
    assumed_completion_tokens: int = 45,
) -> dict[str, Any]:
    """Conservative 0%-cache pre-run USD quote for 1.7B causal simulations."""
    by_stratum: dict[str, dict[str, Any]] = {}
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
        skey = f"{task.judge_model}:{task.stratum}"
        if skey not in by_stratum:
            by_stratum[skey] = {"calls": 0, "est_prompt_tokens": 0, "est_usd": 0.0}
        by_stratum[skey]["calls"] += 1
        by_stratum[skey]["est_prompt_tokens"] += in_tok
        by_stratum[skey]["est_usd"] += cost

    for b in by_stratum.values():
        b["est_usd"] = round(b["est_usd"], 6)
        b["est_mean_prompt_tokens"] = round(b["est_prompt_tokens"] / max(b["calls"], 1), 1)

    return {
        "n_calls": len(plan),
        "est_total_prompt_tokens": tot_in,
        "est_total_completion_tokens": tot_out,
        "assumed_completion_tokens_per_call": assumed_completion_tokens,
        "est_total_usd_0pct_cache": round(tot_usd, 6),
        "by_stratum": by_stratum,
    }


async def _call_openai_sim_once(
    client: Any,
    task: SimWindowTask,
    messages: list[dict[str, str]],
    limiter: AdaptiveRateLimiter,
    use_json_mode: bool = True,
    max_retries: int = 6,
    base_backoff: float = 0.5,
) -> dict[str, Any]:
    """Execute one OpenAI sparse JSON activation-simulation call with retries."""
    req_kwargs: dict[str, Any] = {
        "model": task.api_model,
        "messages": messages,
        "reasoning_effort": task.reasoning_effort,
        "temperature": 0.0,
        "max_completion_tokens": int(task.max_completion_tokens),
    }
    if use_json_mode:
        req_kwargs["response_format"] = {"type": "json_object"}

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
            ratings, sparse, parse_status = parse_sparse_ratings(raw_content, task.seq_len)

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
                "window_id": task.window_id,
                "neuron_id": task.neuron_id,
                "layer": task.layer,
                "neuron": task.neuron,
                "stratum": task.stratum,
                "window_idx": task.window_idx,
                "seq_len": task.seq_len,
                "judge_model": task.judge_model,
                "api_model": task.api_model,
                "reasoning_effort": task.reasoning_effort,
                "ratings": ratings,
                "sparse_ratings": sparse,
                "n_nonzero": len(sparse),
                "parse_status": parse_status,
                "raw_response": raw_content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_output_tokens": visible_output_tokens,
                "cost_usd": round(cost_usd, 8),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            # Fallback if a backend ever rejects response_format=json_object
            if use_json_mode and "response_format" in str(exc).lower():
                req_kwargs.pop("response_format", None)
                use_json_mode = False
                continue
            if _is_rate_limit_error(exc):
                ra = _extract_retry_after(exc)
                await limiter.note_rate_limit(ra)
                sleep_s = max(ra or 0.0, base_backoff * (2**attempt))
            else:
                sleep_s = base_backoff * (2**attempt)
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(sleep_s, 30.0))

    raise RuntimeError(
        f"OpenAI 1.7B sim call failed after {max_retries} retries for {task.window_id}: {last_err}"
    ) from last_err


async def run_sim_17b_async(
    plan: Sequence[tuple[SimWindowTask, list[dict[str, str]], dict[str, Any]]],
    output_path: Path = DEFAULT_SIM_CALLS_17B_PATH,
    companion_fpr_path: Path = DEFAULT_FPR_17B_PATH,
    max_usd: float = DEFAULT_SIM_MAX_USD,
    combined_max_usd: float = DEFAULT_CAUSAL_17B_MAX_USD,
    concurrency: int = 40,
    target_rpm: float = 4000.0,
    openai_client: Any = None,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Run crash-safe resumable 1.7B causal simulation sweep with hard self + combined USD caps."""
    assert_17b_causal_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)

    completed_keys = load_completed_sim_keys(output_path)
    prior_self_usd = _sum_existing_spend([output_path])
    prior_combined_usd = _sum_existing_spend([output_path, companion_fpr_path])
    pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed_keys]
    quote = quote_sim_17b_plan_cost(pending)

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

    if prior_self_usd >= max_usd or prior_combined_usd >= combined_max_usd:
        raise SpendLimitExceededError(
            f"Prior spend self=${prior_self_usd:.4f} (cap ${max_usd:.2f}) / "
            f"combined=${prior_combined_usd:.4f} (cap ${combined_max_usd:.2f}) already at cap."
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
    parse_errors = 0
    fallback_regex = 0
    aborted = False

    try:
        with output_path.open("a", encoding="utf-8") as fh:

            async def _worker(
                task: SimWindowTask,
                messages: list[dict[str, str]],
                extra_meta: dict[str, Any],
            ) -> None:
                nonlocal spent_this_run, prompt_toks, comp_toks, reas_toks
                nonlocal completed_now, parse_errors, fallback_regex, aborted
                async with sem:
                    async with state_lock:
                        if (
                            aborted
                            or (prior_self_usd + spent_this_run) >= max_usd
                            or (prior_combined_usd + spent_this_run) >= combined_max_usd
                        ):
                            aborted = True
                            return
                    row = await _call_openai_sim_once(
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
                        if row["parse_status"] == "parse_error":
                            parse_errors += 1
                        elif row["parse_status"] == "fallback_regex":
                            fallback_regex += 1
                        completed_now += 1
                        if progress_every > 0 and (
                            completed_now % progress_every == 0 or completed_now == len(pending)
                        ):
                            print(
                                f"[sim_17b] {completed_now:,}/{len(pending):,} done | "
                                f"run=${spent_this_run:.4f} | parse_err={parse_errors} | "
                                f"fallback={fallback_regex}",
                                flush=True,
                            )
                        if (
                            (prior_self_usd + spent_this_run) >= max_usd
                            or (prior_combined_usd + spent_this_run) >= combined_max_usd
                        ):
                            aborted = True

            await asyncio.gather(*(_worker(t, m, meta) for t, m, meta in pending))
    finally:
        if close_client and hasattr(client, "close"):
            await client.close()

    return {
        "total_planned": len(plan),
        "skipped_existing": len(plan) - len(pending),
        "completed_this_run": completed_now,
        "parse_errors_this_run": parse_errors,
        "fallback_regex_this_run": fallback_regex,
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
# Part 2: 1.7B Single-Snippet YES/NO Precision Audit Runner
# ---------------------------------------------------------------------------


def build_fpr_17b_task_plan(
    poles: Sequence[Mapping[str, Any]],
    labels_by_pole: Mapping[str, Mapping[str, Any]],
    fpr_probes: Sequence[Mapping[str, Any]],
    sweeps: Sequence[str] = FPR_17B_SWEEPS,
) -> list[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]]:
    """Build deterministic 1.7B YES/NO precision task plan (reusing exact 4B prompt)."""
    for sw in sweeps:
        if sw not in FPR_17B_SWEEPS:
            raise ValueError(f"Unknown 1.7B FPR sweep {sw!r}; valid: {FPR_17B_SWEEPS}")

    poles_by_id = {str(p["pole_id"]): p for p in poles}
    label_text_by_pole = {
        pid: resolve_17b_pole_label_text(labels_by_pole[pid])
        for pid in poles_by_id
        if pid in labels_by_pole
    }

    # Assign stable integer k per neuron_id sorted by (layer, neuron)
    unique_neurons = sorted({(int(p["layer"]), int(p["neuron"])) for p in poles})
    neuron_to_k = {pair: i for i, pair in enumerate(unique_neurons)}

    plan: list[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]] = []
    seen_keys: set[tuple[str, str]] = set()

    def _resolve_ptype(row: Mapping[str, Any]) -> tuple[Literal["top100_firing", "ordinary_nonfiring"], Literal["YES", "NO"]]:
        pt = str(row.get("probe_type", "")).lower()
        gt = str(row.get("ground_truth", "")).upper()
        if pt in ("top100_firing", "p999_firing", "p99.9_plus_firing", "positive", "pos", "top") or gt == "YES":
            return "top100_firing", "YES"
        return "ordinary_nonfiring", "NO"

    def _add(row: Mapping[str, Any], api_model: str) -> None:
        pid = str(row["pole_id"])
        if pid not in poles_by_id or pid not in label_text_by_pole:
            return
        pole = poles_by_id[pid]
        desc = label_text_by_pole[pid]
        ptype, gt = _resolve_ptype(row)
        tier = str(row.get("tier", "core"))
        probe_idx = int(row.get("probe_idx", 0))
        probe_id = str(row.get("probe_id") or f"{pid}__{ptype}__{probe_idx}")
        snip_txt = str(row["marked_text"])
        jm_tag = f"{api_model}@none"
        layer = int(pole["layer"])
        neuron = int(pole["neuron"])
        nid = str(pole.get("neuron_id") or f"L{layer:02d}_N{neuron:04d}")

        task = FprProbeTask(
            probe_id=probe_id,
            pole_id=pid,
            k=int(row.get("k", neuron_to_k[(layer, neuron)])),
            layer=layer,
            neuron=neuron,
            sign=int(pole["sign"]),
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
        msgs = build_fpr_messages(desc, snip_txt)
        extra_meta = {
            "neuron_id": nid,
            "orig_probe_type": str(row.get("probe_type", ptype)),
            "doc_id": row.get("doc_id"),
            "act_val": row.get("act_val"),
            "act_dir": row.get("act_dir"),
            "frac_of_max": row.get("frac_of_max"),
            "quadrant": row.get("quadrant"),
        }
        plan.append((task, msgs, extra_meta))

    for sw in sweeps:
        if sw == "fpr_core_terra":
            for r in fpr_probes:
                if str(r.get("tier", "core")).lower() == "luna_extra":
                    continue
                _add(r, "gpt-6.1-terra")
        elif sw == "fpr_all_luna":
            for r in fpr_probes:
                _add(r, "gpt-6-luna")

    return plan


async def run_fpr_17b_async(
    plan: Sequence[tuple[FprProbeTask, list[dict[str, str]], dict[str, Any]]],
    output_path: Path = DEFAULT_FPR_17B_PATH,
    companion_sim_path: Path = DEFAULT_SIM_CALLS_17B_PATH,
    max_usd: float = DEFAULT_FPR_MAX_USD,
    combined_max_usd: float = DEFAULT_CAUSAL_17B_MAX_USD,
    concurrency: int = 40,
    target_rpm: float = 4000.0,
    openai_client: Any = None,
    progress_every: int = 600,
) -> dict[str, Any]:
    """Run crash-safe resumable 1.7B YES/NO precision audit with hard self + combined USD caps."""
    assert_17b_causal_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)

    completed_keys = load_completed_fpr_keys(output_path)
    prior_self_usd = _sum_existing_spend([output_path])
    prior_combined_usd = _sum_existing_spend([output_path, companion_sim_path])
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

    if prior_self_usd >= max_usd or prior_combined_usd >= combined_max_usd:
        raise SpendLimitExceededError(
            f"Prior spend self=${prior_self_usd:.4f} (cap ${max_usd:.2f}) / "
            f"combined=${prior_combined_usd:.4f} (cap ${combined_max_usd:.2f}) already at cap."
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
                        if (
                            aborted
                            or (prior_self_usd + spent_this_run) >= max_usd
                            or (prior_combined_usd + spent_this_run) >= combined_max_usd
                        ):
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
                                f"[fpr_17b] {completed_now:,}/{len(pending):,} done | "
                                f"run=${spent_this_run:.4f} | combined=${prior_combined_usd + spent_this_run:.4f}",
                                flush=True,
                            )
                        if (
                            (prior_self_usd + spent_this_run) >= max_usd
                            or (prior_combined_usd + spent_this_run) >= combined_max_usd
                        ):
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
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-1.7B causal simulations (sparse JSON) and YES/NO precision probes."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("run-sim", help="Run 3,200 bipolar sparse JSON causal simulations into sim_calls_17b.jsonl.")
    p_sim.add_argument("--poles", type=Path, default=DEFAULT_POLES_17B_PATH)
    p_sim.add_argument("--labels", type=Path, default=DEFAULT_LABELS_17B_PATH)
    p_sim.add_argument("--sim-windows", type=Path, default=DEFAULT_SIM_WINDOWS_17B_PATH)
    p_sim.add_argument("--output", type=Path, default=DEFAULT_SIM_CALLS_17B_PATH)
    p_sim.add_argument("--companion-fpr-output", type=Path, default=DEFAULT_FPR_17B_PATH)
    p_sim.add_argument("--api-model", type=str, default=SIM_API_MODEL)
    p_sim.add_argument("--max-usd", type=float, default=DEFAULT_SIM_MAX_USD)
    p_sim.add_argument("--combined-max-usd", type=float, default=DEFAULT_CAUSAL_17B_MAX_USD)
    p_sim.add_argument("--concurrency", type=int, default=40)
    p_sim.add_argument("--target-rpm", type=float, default=4000.0)
    p_sim.add_argument("--env-file", type=Path, default=Path(".env"))
    p_sim.add_argument("--dry-run", action="store_true")

    p_fpr = sub.add_parser("run-fpr", help="Run 3,600 single-snippet YES/NO precision probes into fpr_17b.jsonl.")
    p_fpr.add_argument("--poles", type=Path, default=DEFAULT_POLES_17B_PATH)
    p_fpr.add_argument("--labels", type=Path, default=DEFAULT_LABELS_17B_PATH)
    p_fpr.add_argument("--fpr-probes", type=Path, default=DEFAULT_FPR_PROBES_17B_PATH)
    p_fpr.add_argument("--output", type=Path, default=DEFAULT_FPR_17B_PATH)
    p_fpr.add_argument("--companion-sim-output", type=Path, default=DEFAULT_SIM_CALLS_17B_PATH)
    p_fpr.add_argument("--sweeps", nargs="+", choices=list(FPR_17B_SWEEPS), default=list(FPR_17B_SWEEPS))
    p_fpr.add_argument("--max-usd", type=float, default=DEFAULT_FPR_MAX_USD)
    p_fpr.add_argument("--combined-max-usd", type=float, default=DEFAULT_CAUSAL_17B_MAX_USD)
    p_fpr.add_argument("--concurrency", type=int, default=40)
    p_fpr.add_argument("--target-rpm", type=float, default=4000.0)
    p_fpr.add_argument("--env-file", type=Path, default=Path(".env"))
    p_fpr.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    load_dotenv_file(args.env_file)
    poles = load_poles_17b(args.poles)
    labels_by_pole = load_labels_17b(args.labels)

    if args.command == "run-sim":
        neuron_labels = group_17b_neuron_labels(poles, labels_by_pole)
        sim_windows = load_jsonl(args.sim_windows)
        plan = build_sim_17b_task_plan(sim_windows, neuron_labels, api_model=args.api_model)
        completed = load_completed_sim_keys(args.output)
        pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed]
        quote = quote_sim_17b_plan_cost(pending)
        prior_self = _sum_existing_spend([args.output])
        prior_combined = _sum_existing_spend([args.output, args.companion_fpr_output])
        print(
            json.dumps(
                {
                    "command": "run-sim",
                    "api_model": args.api_model,
                    "total_planned": len(plan),
                    "already_completed": len(plan) - len(pending),
                    "pending": len(pending),
                    "prior_self_usd": round(prior_self, 6),
                    "prior_combined_usd": round(prior_combined, 6),
                    "max_usd": args.max_usd,
                    "combined_max_usd": args.combined_max_usd,
                    "pre_run_quote": quote,
                },
                indent=2,
            )
        )
        if args.dry_run:
            return 0
        summary = asyncio.run(
            run_sim_17b_async(
                plan=plan,
                output_path=args.output,
                companion_fpr_path=args.companion_fpr_output,
                max_usd=args.max_usd,
                combined_max_usd=args.combined_max_usd,
                concurrency=args.concurrency,
                target_rpm=args.target_rpm,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "run-fpr":
        fpr_probes = load_jsonl(args.fpr_probes)
        plan = build_fpr_17b_task_plan(poles, labels_by_pole, fpr_probes, sweeps=args.sweeps)
        completed = load_completed_fpr_keys(args.output)
        pending = [(t, m, meta) for t, m, meta in plan if t.key not in completed]
        quote = quote_fpr_plan_cost(pending)
        prior_self = _sum_existing_spend([args.output])
        prior_combined = _sum_existing_spend([args.output, args.companion_sim_output])
        print(
            json.dumps(
                {
                    "command": "run-fpr",
                    "sweeps": list(args.sweeps),
                    "total_planned": len(plan),
                    "already_completed": len(plan) - len(pending),
                    "pending": len(pending),
                    "prior_self_usd": round(prior_self, 6),
                    "prior_combined_usd": round(prior_combined, 6),
                    "max_usd": args.max_usd,
                    "combined_max_usd": args.combined_max_usd,
                    "pre_run_quote": quote,
                },
                indent=2,
            )
        )
        if args.dry_run:
            return 0
        summary = asyncio.run(
            run_fpr_17b_async(
                plan=plan,
                output_path=args.output,
                companion_sim_path=args.companion_sim_output,
                max_usd=args.max_usd,
                combined_max_usd=args.combined_max_usd,
                concurrency=args.concurrency,
                target_rpm=args.target_rpm,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
