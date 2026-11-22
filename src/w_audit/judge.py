"""Async 1-pair-per-call detection judge harness for Qwen3-1.7B evaluation.

Supports OpenAI (gpt-4o-mini, gpt-4.1-mini with logprobs=True, top_logprobs=5)
and local Ollama (http://localhost:11434/api/chat with explicit "think": false).

Core invariants:
- STRICTLY 1 snippet pair per API call (no multi-pair batching / no leakage).
- Deterministic A/B order randomization via sha256(f"{SEED}:{pair_id}") LSB.
- Hard --max-usd spend abort guard (default $25.00, 0% prompt caching).
- Adaptive rate-limit pacing + exponential backoff on 429.
- Crash-safe append-only resumable JSONL keyed by (pair_id, condition, judge_model).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import httpx

from w_audit.config import (
    BANDS as ALL_BANDS,
    CONDITIONS as ALL_CONDITIONS,
    DEFAULT_JUDGE_MODEL as DEFAULT_PRIMARY_MODEL,
    DEFAULT_MAX_USD,
    DEFAULT_OLLAMA_BASE_URL as DEFAULT_OLLAMA_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SECOND_JUDGE_MODEL as DEFAULT_SECOND_OPENAI_MODEL,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    NEIGHBOUR_BANDS,
    NEIGHBOUR_DISTRACTOR_TYPE,
    NO_LABEL_DESCRIPTION,
    SEED,
    TOY_BANDS,
    TOY_CONDITIONS,
    deterministic_pos_slot as compute_pos_slot,
)
from w_audit.labeler import get_label_for_condition, load_jsonl
from w_audit.probe_limits import (
    MODEL_PRICING_PER_M,
    compute_call_cost_usd,
    load_dotenv_file,
    strip_model_tag,
)

# Backwards-compatible aliases (all point to the single unified prompt template)
SYSTEM_PROMPT_LABELED: str = JUDGE_SYSTEM_PROMPT
USER_PROMPT_LABELED_TEMPLATE: str = JUDGE_USER_TEMPLATE
SYSTEM_PROMPT_NO_LABEL: str = JUDGE_SYSTEM_PROMPT
USER_PROMPT_NO_LABEL_TEMPLATE: str = JUDGE_USER_TEMPLATE

LITE_SECOND_JUDGE_CONDITIONS: tuple[str, ...] = ("own", "no_label")
LOGPROB_TIE_EPS: float = 1e-9


class SpendLimitExceededError(RuntimeError):
    """Raised when live spend exceeds --max-usd."""


@dataclass(frozen=True)
class JudgeTask:
    pair_id: str
    pole_id: str
    layer: int
    band: str
    distractor_type: str
    condition: str
    pos_slot: Literal["A", "B"]
    snippet_a: str
    snippet_b: str
    label_text: str | None
    judge_model: str
    judge_backend: Literal["openai", "ollama"]
    order: Literal["fwd", "rev"] = "fwd"
    reasoning_effort: str | None = None
    api_model: str | None = None
    request_logprobs: bool = True
    max_completion_tokens: int | None = None

    @property
    def wire_model(self) -> str:
        return self.api_model if self.api_model else strip_model_tag(self.judge_model)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.pair_id, self.condition, self.judge_model, self.order)


def _normalize_token_only_description(text: str) -> str:
    """Normalize legacy 'Activates on tokens such as: ...' to 'Tokens such as: ...'."""
    prefix = "Activates on tokens such as: "
    if text.startswith(prefix):
        return "Tokens such as: " + text[len(prefix) :]
    return text


def build_judge_messages(
    condition: str,
    label_text: str | None,
    snippet_a: str,
    snippet_b: str,
) -> list[dict[str, str]]:
    """Build exact 1-pair-per-call chat messages using ONE unified template.

    All four conditions ('own', 'wrong', 'no_label', 'token_only') share the
    exact same system prompt and user prompt template with ZERO occurrences of
    'strongly'. Only the {description} slot changes:
      - own        -> pole's own description
      - wrong      -> cross-layer donor description
      - token_only -> 'Tokens such as: "t1", "t2", "t3"'
      - no_label   -> '(none provided)'
    """
    if not isinstance(snippet_a, str) or not isinstance(snippet_b, str):
        raise TypeError("Judge strictly accepts exactly one pair (snippet_a, snippet_b).")

    if condition not in ("own", "wrong", "no_label", "token_only"):
        raise ValueError(f"Unsupported judge condition: {condition!r}")

    if condition == "no_label":
        description = NO_LABEL_DESCRIPTION
    else:
        if not label_text:
            raise ValueError(f"Condition {condition!r} requires a non-empty label_text.")
        description = (
            _normalize_token_only_description(label_text)
            if condition == "token_only"
            else label_text
        )

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


def build_ollama_payload(
    model: str,
    messages: Sequence[Mapping[str, str]],
    num_predict: int = 4,
) -> dict[str, Any]:
    """Build Ollama /api/chat payload with explicit top-level 'think': False."""
    return {
        "model": model,
        "messages": [dict(m) for m in messages],
        "think": False,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": int(num_predict),
        },
    }


def _logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    total = sum(math.exp(v - m) for v in values)
    return min(0.0, m + math.log(total))


def extract_ab_logprobs(
    top_logprobs: Sequence[Any] | None,
    pos_slot: Literal["A", "B"],
) -> tuple[float | None, float | None, float | None, bool]:
    """Extract (logprob_A, logprob_B, margin, imputed) from top-5 logprobs."""
    if not top_logprobs:
        return None, None, None, False

    a_lps: list[float] = []
    b_lps: list[float] = []
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
        if norm == "A":
            a_lps.append(lp)
        elif norm == "B":
            b_lps.append(lp)

    if not all_lps or (not a_lps and not b_lps):
        return None, None, None, True

    imputed = False
    # Probability-mass upper bound for missing A/B handles GPT-6 pruned top_logprobs
    # (e.g. [('B', 0.0)] -> remaining mass <= 1e-9) while preserving min(lps)-2.0 on 5-token lists.
    rem_prob = max(1e-9, 1.0 - sum(math.exp(v) for v in all_lps))
    floor_lp = min(min(all_lps) - 2.0, math.log(rem_prob))
    if a_lps:
        lp_a = _logsumexp(a_lps)
    else:
        lp_a = floor_lp
        imputed = True

    if b_lps:
        lp_b = _logsumexp(b_lps)
    else:
        lp_b = floor_lp
        imputed = True

    margin = (lp_a - lp_b) if pos_slot == "A" else (lp_b - lp_a)
    return round(lp_a, 6), round(lp_b, 6), round(margin, 6), imputed


def parse_ab_choice(
    raw_text: str,
    logprob_a: float | None = None,
    logprob_b: float | None = None,
    tie_eps: float = LOGPROB_TIE_EPS,
) -> Literal["A", "B"] | None:
    """Parse A or B decision strictly from raw model text, falling back to logprobs.

    Never scans raw substrings inside unrelated words (e.g. "Cannot", "Both",
    "Answer" return None when logprobs are unavailable). Exact logprob ties
    (|lp_A - lp_B| <= tie_eps) also return None rather than breaking toward A.
    """
    cleaned = raw_text.strip().upper()
    if cleaned in ("A", "B"):
        return cleaned  # type: ignore[return-value]

    matches = set(re.findall(r"\b([AB])\b", cleaned))
    if len(matches) == 1:
        m = re.match(r"^(?:SNIPPET\s+)?([AB])\b", cleaned)
        if m:
            return m.group(1)  # type: ignore[return-value]
        return next(iter(matches))  # type: ignore[return-value]

    if logprob_a is not None and logprob_b is not None:
        if abs(logprob_a - logprob_b) <= tie_eps:
            return None
        return "A" if logprob_a > logprob_b else "B"
    return None


def classify_judge_outcome(
    raw_text: str,
    pos_slot: Literal["A", "B"],
    logprob_a: float | None = None,
    logprob_b: float | None = None,
    tie_eps: float = LOGPROB_TIE_EPS,
) -> tuple[Literal["A", "B"] | None, Literal["scored", "unparseable", "logprob_tie"], float | None]:
    """Classify a judge response (exclude unparseable & logprob ties from accuracy).

    Returns (predicted_slot, outcome, correct) where correct is None whenever
    outcome is 'unparseable' or 'logprob_tie'.
    """
    if (
        logprob_a is not None
        and logprob_b is not None
        and abs(logprob_a - logprob_b) <= tie_eps
    ):
        return None, "logprob_tie", None

    predicted = parse_ab_choice(raw_text, logprob_a, logprob_b, tie_eps=tie_eps)
    if predicted is None:
        return None, "unparseable", None

    correct = 1.0 if predicted == pos_slot else 0.0
    return predicted, "scored", correct


def sanitize_jsonl_tail(path: Path) -> None:
    """Truncate any corrupted partial trailing line left by a hard crash."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    last_nl = raw.rfind(b"\n")
    tail = raw[last_nl + 1 :].decode("utf-8", errors="replace").strip()
    try:
        json.loads(tail)
        # Valid JSON line missing trailing newline: add newline cleanly
        with path.open("ab") as fh:
            fh.write(b"\n")
    except json.JSONDecodeError:
        # Truncated crash fragment: truncate file back to last complete line
        with path.open("r+b") as fh:
            fh.truncate(last_nl + 1 if last_nl >= 0 else 0)


def load_completed_keys(judgements_path: Path) -> set[tuple[str, str, str, str]]:
    """Load already-scored (pair_id, condition, judge_model, order) keys for resume."""
    keys: set[tuple[str, str, str, str]] = set()
    for row in load_jsonl(judgements_path):
        pid = row.get("pair_id")
        cond = row.get("condition")
        model = row.get("judge_model")
        order = str(row.get("order") or "fwd")
        if pid is not None and cond is not None and model is not None:
            keys.add((str(pid), str(cond), str(model), order))
    return keys


def _parse_reset_seconds(value: str | None) -> float | None:
    if not value:
        return None
    s = value.strip().lower()
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", s):
        matched = True
        v = float(amount)
        if unit == "ms":
            total += v / 1000.0
        elif unit == "s":
            total += v
        elif unit == "m":
            total += v * 60.0
        elif unit == "h":
            total += v * 3600.0
    return total if matched else None


class AdaptiveRateLimiter:
    """Smooth token/request pacer that adapts dynamically to 429s and headers."""

    def __init__(
        self,
        target_rpm: float | None = 3_500.0,
        min_rpm: float = 60.0,
    ) -> None:
        self.max_rpm = target_rpm
        self.current_rpm = target_rpm
        self.min_rpm = min_rpm
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0
        self._clean_streak = 0

    async def acquire(self) -> None:
        if self.current_rpm is None or self.current_rpm <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            interval = 60.0 / max(self.current_rpm, 1.0)
            wait = self._next_allowed - now
            if wait > 0:
                self._next_allowed += interval
            else:
                self._next_allowed = now + interval
                wait = 0.0
        if wait > 0:
            await asyncio.sleep(wait)

    async def note_headers(self, headers: Mapping[str, str]) -> None:
        if self.current_rpm is None:
            return
        low = {k.lower(): str(v) for k, v in headers.items()}
        rem_req = low.get("x-ratelimit-remaining-requests")
        rem_tok = low.get("x-ratelimit-remaining-tokens")
        reset_req = _parse_reset_seconds(low.get("x-ratelimit-reset-requests"))
        reset_tok = _parse_reset_seconds(low.get("x-ratelimit-reset-tokens"))

        async with self._lock:
            pause = 0.0
            if rem_req is not None and rem_req.isdigit() and int(rem_req) <= 5:
                pause = max(pause, reset_req or 0.5)
            if rem_tok is not None and rem_tok.isdigit() and int(rem_tok) <= 5_000:
                pause = max(pause, reset_tok or 0.5)
            if pause > 0:
                self._next_allowed = max(self._next_allowed, time.monotonic() + pause)
            else:
                self._clean_streak += 1
                if (
                    self.max_rpm is not None
                    and self.current_rpm < self.max_rpm
                    and self._clean_streak >= 20
                ):
                    self.current_rpm = min(self.max_rpm, self.current_rpm * 1.1)
                    self._clean_streak = 0

    async def note_rate_limit(self, retry_after: float | None = None) -> None:
        async with self._lock:
            self._clean_streak = 0
            if self.current_rpm is not None:
                self.current_rpm = max(self.min_rpm, self.current_rpm * 0.5)
            if retry_after and retry_after > 0:
                self._next_allowed = max(
                    self._next_allowed, time.monotonic() + retry_after
                )


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc)


def _extract_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except ValueError:
                return _parse_reset_seconds(str(ra))
    return None


def _build_openai_request_kwargs(
    task: JudgeTask,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Build exact OpenAI Chat Completions kwargs for legacy or GPT-6 reasoning models."""
    kwargs: dict[str, Any] = {
        "model": task.wire_model,
        "messages": messages,
    }
    if task.reasoning_effort is not None:
        kwargs["reasoning_effort"] = task.reasoning_effort
        if task.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = int(task.max_completion_tokens)
        elif task.reasoning_effort == "none":
            kwargs["max_completion_tokens"] = 16
        else:
            # Never cap reasoning-enabled models at 1-4 tokens; give ample headroom
            kwargs["max_completion_tokens"] = 1024

        if task.reasoning_effort == "none":
            kwargs["temperature"] = 0.0
            if task.request_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5
        elif task.request_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
    else:
        kwargs["temperature"] = 0.0
        if task.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = int(task.max_completion_tokens)
        else:
            kwargs["max_tokens"] = 1
        if task.request_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
    return kwargs


async def _call_openai_once(
    client: Any,
    task: JudgeTask,
    messages: list[dict[str, str]],
    limiter: AdaptiveRateLimiter,
    max_retries: int = 6,
    base_backoff: float = 0.5,
) -> dict[str, Any]:
    """Execute one OpenAI single-pair judge call with retries, logprobs, and reasoning accounting."""
    req_kwargs = _build_openai_request_kwargs(task, messages)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        await limiter.acquire()
        try:
            completions = client.chat.completions
            headers: dict[str, str] = {}
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

            lp_a, lp_b, margin, imputed = extract_ab_logprobs(top_lps, task.pos_slot)
            predicted, outcome, correct = classify_judge_outcome(
                raw_content, task.pos_slot, lp_a, lp_b
            )

            usage = getattr(parsed, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
            visible_output_tokens = max(0, completion_tokens - reasoning_tokens)
            # OpenAI bills reasoning_tokens inside completion_tokens at the output rate.
            cost_usd = compute_call_cost_usd(
                task.wire_model, prompt_tokens, completion_tokens, MODEL_PRICING_PER_M
            )

            return {
                "pair_id": task.pair_id,
                "pole_id": task.pole_id,
                "layer": task.layer,
                "band": task.band,
                "distractor_type": task.distractor_type,
                "condition": task.condition,
                "judge_model": task.judge_model,
                "api_model": task.wire_model,
                "reasoning_effort": task.reasoning_effort,
                "order": task.order,
                "judge_backend": "openai",
                "pos_slot": task.pos_slot,
                "predicted_slot": predicted,
                "outcome": outcome,
                "correct": correct,
                "raw_response": raw_content,
                "logprob_A": lp_a,
                "logprob_B": lp_b,
                "margin": margin,
                "logprob_imputed": imputed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_output_tokens": visible_output_tokens,
                "cost_usd": round(cost_usd, 8),
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        except Exception as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                break
            if _is_rate_limit_error(exc):
                ra = _extract_retry_after(exc)
                await limiter.note_rate_limit(ra)
                delay = ra if (ra is not None and ra > 0) else min(30.0, base_backoff * (2**attempt))
                delay += random.uniform(0.0, min(0.25, delay * 0.2))
            else:
                delay = min(15.0, base_backoff * (2**attempt))
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"OpenAI judge call failed for {task.key} after {max_retries} attempts: {last_err}"
    ) from last_err


async def _call_ollama_once(
    http_client: httpx.AsyncClient,
    ollama_url: str,
    task: JudgeTask,
    messages: list[dict[str, str]],
    max_retries: int = 3,
    base_backoff: float = 0.5,
) -> dict[str, Any]:
    """Execute one local Ollama /api/chat call with explicit 'think': False."""
    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    payload = build_ollama_payload(model=task.wire_model, messages=messages, num_predict=4)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await http_client.post(endpoint, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            raw_content = str(data.get("message", {}).get("content", "")).strip()
            predicted, outcome, correct = classify_judge_outcome(
                raw_content, task.pos_slot, None, None
            )
            prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
            completion_tokens = int(data.get("eval_count", 0) or 0)

            return {
                "pair_id": task.pair_id,
                "pole_id": task.pole_id,
                "layer": task.layer,
                "band": task.band,
                "distractor_type": task.distractor_type,
                "condition": task.condition,
                "judge_model": task.judge_model,
                "api_model": task.wire_model,
                "reasoning_effort": task.reasoning_effort,
                "order": task.order,
                "judge_backend": "ollama",
                "pos_slot": task.pos_slot,
                "predicted_slot": predicted,
                "outcome": outcome,
                "correct": correct,
                "raw_response": raw_content,
                "logprob_A": None,
                "logprob_B": None,
                "margin": None,
                "logprob_imputed": False,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": 0,
                "visible_output_tokens": completion_tokens,
                "cost_usd": 0.0,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        except Exception as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                break
            await asyncio.sleep(base_backoff * (2**attempt))

    raise RuntimeError(
        f"Ollama judge call failed for {task.key} after {max_retries} attempts: {last_err}"
    ) from last_err


def build_task_plan(
    poles_data: Mapping[str, Any],
    snippets: Sequence[Mapping[str, Any]],
    labels_by_pole: Mapping[str, Mapping[str, Any]],
    mode: Literal["pilot", "toy", "core", "p90_compare", "neighbour", "second_judge"],
    judge_model: str,
    judge_backend: Literal["openai", "ollama"],
    include_p90_compare: bool = False,
    lite: bool = False,
    bands_filter: Sequence[str] | None = None,
    both_orders: bool = False,
    reasoning_effort: str | None = None,
    request_logprobs: bool | None = None,
    max_completion_tokens: int | None = None,
    seed: int = SEED,
) -> list[JudgeTask]:
    """Construct the exact list of 1-pair-per-call JudgeTasks for a sweep mode."""
    poles_by_id = {str(p["pole_id"]): p for p in poles_data["poles"]}

    api_model = strip_model_tag(judge_model)
    if reasoning_effort is not None and "@" not in judge_model:
        tagged_model = f"{api_model}@{reasoning_effort}"
    else:
        tagged_model = judge_model
        if reasoning_effort is None and "@" in judge_model:
            reasoning_effort = judge_model.split("@", 1)[1].strip()

    if request_logprobs is None:
        eff_request_logprobs = (
            judge_backend == "openai" and reasoning_effort in (None, "none")
        )
    else:
        eff_request_logprobs = bool(request_logprobs)

    if mode == "pilot":
        allowed_poles = {pid for pid, p in poles_by_id.items() if p.get("in_pilot")}
        allowed_bands = set(ALL_BANDS)
        allowed_distractors = {"dead_zone"}
        conditions = ALL_CONDITIONS
    elif mode == "toy":
        allowed_poles = {pid for pid, p in poles_by_id.items() if p.get("in_toy")}
        allowed_bands = set(TOY_BANDS)
        allowed_distractors = {"dead_zone"}
        conditions = TOY_CONDITIONS
    elif mode == "core":
        allowed_poles = set(poles_by_id.keys())
        allowed_bands = set(ALL_BANDS)
        allowed_distractors = {"dead_zone", "goodfire_lt_p90"} if include_p90_compare else {"dead_zone"}
        conditions = ALL_CONDITIONS
    elif mode == "p90_compare":
        allowed_poles = set(poles_by_id.keys())
        allowed_bands = {"p99.9_plus"}
        allowed_distractors = {"goodfire_lt_p90"}
        conditions = ALL_CONDITIONS
    elif mode == "neighbour":
        allowed_poles = set(poles_by_id.keys())
        allowed_bands = set(NEIGHBOUR_BANDS)
        allowed_distractors = {NEIGHBOUR_DISTRACTOR_TYPE}
        conditions = ALL_CONDITIONS
    elif mode == "second_judge":
        allowed_poles = {pid for pid, p in poles_by_id.items() if p.get("in_second_judge")}
        allowed_bands = set(ALL_BANDS)
        allowed_distractors = {"dead_zone"}
        conditions = LITE_SECOND_JUDGE_CONDITIONS if lite else ALL_CONDITIONS
    else:
        raise ValueError(f"Unknown judge mode: {mode!r}")

    if bands_filter is not None:
        unknown = [b for b in bands_filter if b not in ALL_BANDS]
        if unknown:
            raise ValueError(f"Unknown band(s) in bands_filter: {unknown!r}")
        allowed_bands = allowed_bands & set(bands_filter)

    orders: tuple[Literal["fwd", "rev"], ...] = ("fwd", "rev") if both_orders else ("fwd",)

    tasks: list[JudgeTask] = []
    for snip in snippets:
        pole_id = str(snip["pole_id"])
        band = str(snip["band"])
        dist_type = str(snip["distractor_type"])
        if pole_id not in allowed_poles:
            continue
        if band not in allowed_bands:
            continue
        if dist_type not in allowed_distractors:
            continue
        if dist_type == "goodfire_lt_p90" and band != "p99.9_plus":
            continue

        pair_id = str(snip["pair_id"])
        layer = int(snip["layer"])
        pos_slot_raw = snip.get("pos_slot")
        fwd_pos_slot: Literal["A", "B"] = (
            pos_slot_raw if pos_slot_raw in ("A", "B") else compute_pos_slot(pair_id, seed)
        )

        pos_text = str(snip["pos"]["marked_text"])
        dist_text = str(snip["dist"]["marked_text"])

        label_row = labels_by_pole.get(pole_id)
        for cond in conditions:
            if cond == "no_label":
                label_text = None
            else:
                if label_row is None:
                    raise KeyError(
                        f"Missing label entry in labels.jsonl for pole {pole_id!r}"
                    )
                label_text = get_label_for_condition(label_row, cond)

            for order in orders:
                if order == "fwd":
                    slot = fwd_pos_slot
                else:
                    slot = "B" if fwd_pos_slot == "A" else "A"

                if slot == "A":
                    snippet_a, snippet_b = pos_text, dist_text
                else:
                    snippet_a, snippet_b = dist_text, pos_text

                tasks.append(
                    JudgeTask(
                        pair_id=pair_id,
                        pole_id=pole_id,
                        layer=layer,
                        band=band,
                        distractor_type=dist_type,
                        condition=cond,
                        pos_slot=slot,
                        snippet_a=snippet_a,
                        snippet_b=snippet_b,
                        label_text=label_text,
                        judge_model=tagged_model,
                        judge_backend=judge_backend,
                        order=order,
                        reasoning_effort=reasoning_effort,
                        api_model=api_model,
                        request_logprobs=eff_request_logprobs,
                        max_completion_tokens=max_completion_tokens,
                    )
                )

    return tasks


async def run_judge_tasks_async(
    tasks: Sequence[JudgeTask],
    output_path: Path,
    max_usd: float = DEFAULT_MAX_USD,
    concurrency: int = 32,
    target_rpm: float | None = 3_500.0,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    openai_client: Any | None = None,
    http_client: httpx.AsyncClient | None = None,
    env_path: Path | str = ".env",
    base_backoff: float = 0.5,
) -> dict[str, Any]:
    """Execute JudgeTasks with resume, adaptive pacing, and hard USD spend cap."""
    completed_keys = load_completed_keys(output_path)
    pending = [t for t in tasks if t.key not in completed_keys]

    summary: dict[str, Any] = {
        "total_planned": len(tasks),
        "skipped_existing": len(tasks) - len(pending),
        "completed_this_run": 0,
        "spent_usd_this_run": 0.0,
        "prompt_tokens_this_run": 0,
        "completion_tokens_this_run": 0,
        "reasoning_tokens_this_run": 0,
        "visible_output_tokens_this_run": 0,
        "aborted_for_spend_cap": False,
    }
    if not pending:
        return summary

    needs_openai = any(t.judge_backend == "openai" for t in pending)
    needs_ollama = any(t.judge_backend == "ollama" for t in pending)

    owned_openai = False
    owned_http = False

    if needs_openai and openai_client is None:
        load_dotenv_file(env_path)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in environment or .env. "
                "Add OPENAI_API_KEY=... to .env before running OpenAI judge calls."
            )
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key=api_key)
        owned_openai = True

    if needs_ollama and http_client is None:
        http_client = httpx.AsyncClient()
        owned_http = True

    effective_concurrency = 1 if (needs_ollama and not needs_openai) else max(1, concurrency)
    sem = asyncio.Semaphore(effective_concurrency)
    limiter = AdaptiveRateLimiter(target_rpm=target_rpm)
    file_lock = asyncio.Lock()
    spend_lock = asyncio.Lock()
    abort_event = asyncio.Event()
    spend_error: list[SpendLimitExceededError] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_jsonl_tail(output_path)

    try:
        with output_path.open("a", encoding="utf-8") as out_fh:

            async def _run_one(task: JudgeTask) -> None:
                if abort_event.is_set():
                    return
                async with sem:
                    if abort_event.is_set():
                        return
                    async with spend_lock:
                        if summary["spent_usd_this_run"] >= max_usd:
                            abort_event.set()
                            if not spend_error:
                                spend_error.append(
                                    SpendLimitExceededError(
                                        f"Spend cap reached before dispatching {task.key}: "
                                        f"${summary['spent_usd_this_run']:.4f} >= ${max_usd:.2f}"
                                    )
                                )
                            return

                    messages = build_judge_messages(
                        condition=task.condition,
                        label_text=task.label_text,
                        snippet_a=task.snippet_a,
                        snippet_b=task.snippet_b,
                    )

                    if task.judge_backend == "openai":
                        row = await _call_openai_once(
                            client=openai_client,
                            task=task,
                            messages=messages,
                            limiter=limiter,
                            base_backoff=base_backoff,
                        )
                    else:
                        assert http_client is not None
                        row = await _call_ollama_once(
                            http_client=http_client,
                            ollama_url=ollama_url,
                            task=task,
                            messages=messages,
                            base_backoff=base_backoff,
                        )

                    # Crash-safe append immediately
                    async with file_lock:
                        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_fh.flush()

                    async with spend_lock:
                        summary["completed_this_run"] += 1
                        summary["prompt_tokens_this_run"] += int(row["prompt_tokens"])
                        summary["completion_tokens_this_run"] += int(row["completion_tokens"])
                        summary["reasoning_tokens_this_run"] += int(row.get("reasoning_tokens", 0))
                        summary["visible_output_tokens_this_run"] += int(
                            row.get("visible_output_tokens", row["completion_tokens"])
                        )
                        summary["spent_usd_this_run"] += float(row["cost_usd"])
                        if summary["spent_usd_this_run"] >= max_usd:
                            summary["aborted_for_spend_cap"] = True
                            abort_event.set()
                            if not spend_error:
                                spend_error.append(
                                    SpendLimitExceededError(
                                        f"Hard spend cap exceeded: "
                                        f"${summary['spent_usd_this_run']:.4f} >= ${max_usd:.2f} "
                                        f"(saved {summary['completed_this_run']} rows safely to {output_path})"
                                    )
                                )

            coros = [asyncio.create_task(_run_one(t)) for t in pending]
            await asyncio.gather(*coros)
    finally:
        if owned_openai and openai_client is not None and hasattr(openai_client, "close"):
            await openai_client.close()
        if owned_http and http_client is not None:
            await http_client.aclose()

    summary["spent_usd_this_run"] = round(summary["spent_usd_this_run"], 6)
    if spend_error:
        raise spend_error[0]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 1-pair-per-call detection judge sweeps."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--pilot", action="store_true", help="20 pilot poles (incl. L4 controls), 1,920 calls."
    )
    mode_group.add_argument(
        "--toy", action="store_true", help="60 poles x 3 bands x 3 conds fallback, 3,240 calls."
    )
    mode_group.add_argument(
        "--core", action="store_true", help="100 poles x 4 bands x 4 conds (9,600 calls)."
    )
    mode_group.add_argument(
        "--p90-compare",
        action="store_true",
        help="100 poles top-band Goodfire <p90 distractor comparison (2,400 calls).",
    )
    mode_group.add_argument(
        "--neighbour",
        action="store_true",
        help="100 poles x 2 bands same-layer neighbour distractor arm (4,800 calls).",
    )
    mode_group.add_argument(
        "--second-judge",
        action="store_true",
        help="30 poles second-judge verification (2,880 calls; or 1,440 with --lite).",
    )

    parser.add_argument(
        "--include-p90-compare",
        action="store_true",
        help="When running --core, also include the 2,400 top-band <p90 comparison calls.",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="When running --second-judge, run 2 conditions (own, no_label = 1,440 calls).",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=list(ALL_BANDS),
        default=None,
        help="Optional subset of activation bands to evaluate (e.g., --bands p99.9_plus).",
    )
    parser.add_argument(
        "--backend", choices=("openai", "ollama"), default="openai"
    )
    parser.add_argument(
        "--model",
        "--judge-model",
        dest="model",
        type=str,
        default=None,
        help="Override judge model (defaults: gpt-4o-mini; gpt-4.1-mini for --second-judge openai; qwen3.5:9b for ollama).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default=None,
        help="Optional reasoning_effort for GPT-6 / reasoning models (tags judge_model as <model>@<effort>).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional max_completion_tokens override (defaults to 1024 when reasoning enabled, 16 when effort=none).",
    )
    parser.add_argument(
        "--both-orders",
        action="store_true",
        help="Score every pair in both A/B presentation orders (order='fwd' and order='rev').",
    )
    parser.add_argument(
        "--no-logprobs",
        action="store_true",
        help="Disable logprobs=True on OpenAI requests (auto-disabled when reasoning_effort != none).",
    )
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--poles", type=Path, default=Path("data/poles.json"))
    parser.add_argument("--snippets", type=Path, default=Path("data/snippets.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/judgements.jsonl")
    )
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--target-rpm", type=float, default=3_500.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact task count, resume skip count, and 0%%-cache USD estimate without calling APIs.",
    )
    args = parser.parse_args(argv)

    if args.pilot:
        mode: Literal["pilot", "toy", "core", "p90_compare", "neighbour", "second_judge"] = "pilot"
    elif args.toy:
        mode = "toy"
    elif args.core:
        mode = "core"
    elif args.p90_compare:
        mode = "p90_compare"
    elif args.neighbour:
        mode = "neighbour"
    else:
        mode = "second_judge"

    if args.model:
        judge_model = args.model
    elif args.backend == "ollama":
        judge_model = DEFAULT_OLLAMA_MODEL
    elif mode == "second_judge":
        judge_model = DEFAULT_SECOND_OPENAI_MODEL
    else:
        judge_model = DEFAULT_PRIMARY_MODEL

    poles_data = json.loads(args.poles.read_text(encoding="utf-8"))
    snippets = load_jsonl(args.snippets)
    labels_by_pole = {
        str(row["pole_id"]): row for row in load_jsonl(args.labels)
    }

    req_lp: bool | None = False if args.no_logprobs else None
    tasks = build_task_plan(
        poles_data=poles_data,
        snippets=snippets,
        labels_by_pole=labels_by_pole,
        mode=mode,
        judge_model=judge_model,
        judge_backend=args.backend,
        include_p90_compare=args.include_p90_compare,
        lite=args.lite,
        bands_filter=args.bands,
        both_orders=args.both_orders,
        reasoning_effort=args.reasoning_effort,
        request_logprobs=req_lp,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed,
    )

    tagged_display = tasks[0].judge_model if tasks else judge_model
    existing = load_completed_keys(args.output)
    pending_count = sum(1 for t in tasks if t.key not in existing)
    est_out = 4 if args.reasoning_effort in (None, "none") else 64
    est_usd = (
        0.0
        if args.backend == "ollama"
        else compute_call_cost_usd(
            tagged_display, pending_count * 380, pending_count * est_out, MODEL_PRICING_PER_M
        )
    )
    print(
        f"[judge:{mode}] backend={args.backend} model={tagged_display} "
        f"both_orders={args.both_orders} planned={len(tasks):,} "
        f"existing={len(tasks) - pending_count:,} pending={pending_count:,} "
        f"est_usd_0pct_cache=${est_usd:.3f} max_usd=${args.max_usd:.2f}"
    )
    if args.dry_run:
        return 0

    summary = asyncio.run(
        run_judge_tasks_async(
            tasks=tasks,
            output_path=args.output,
            max_usd=args.max_usd,
            concurrency=args.concurrency,
            target_rpm=args.target_rpm,
            ollama_url=args.ollama_url,
            env_path=args.env_file,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
