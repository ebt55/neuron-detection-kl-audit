"""Live OpenAI rate-limit, logprobs support, and sweep ETA/cost probe.

Makes one tiny API call each to gpt-4o-mini and gpt-4.1-mini using the OpenAI
SDK `.with_raw_response` interface to inspect live `x-ratelimit-*` headers and
verify `logprobs=True` support, then computes 0%-cache cost and ETA estimates
for every evaluation sweep tier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from w_audit.config import DEFAULT_LABEL_MODEL, MODEL_PRICING_USD_PER_1M

# Standard-tier OpenAI pricing (USD per 1M tokens, 0% prompt caching assumed
# because single-pair detection prompts are < 1,024 static prefix tokens).
MODEL_PRICING_PER_M: dict[str, tuple[float, float]] = dict(MODEL_PRICING_USD_PER_1M)

DEFAULT_PROBE_MODELS: tuple[str, ...] = ("gpt-4o-mini", "gpt-4.1-mini")

# Documented Tier 1 RPD warning threshold when headers only report per-minute RPM.
TIER1_RPD_WARNING_CALLS: int = 10_000


@dataclass
class ModelProbeResult:
    model: str
    ok: bool
    latency_ms: float
    logprobs_supported: bool
    sample_output: str
    top_logprobs: list[dict[str, float]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_effort: str | None = None
    limit_requests: int | None = None
    remaining_requests: int | None = None
    reset_requests: str | None = None
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    reset_tokens: str | None = None
    raw_ratelimit_headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class WorkloadEstimate:
    name: str
    model: str
    num_calls: int
    avg_input_tokens: int
    avg_output_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    effective_rpm: float | None
    eta_minutes: float | None
    fits_rpm_tpm_within_hours: float | None
    exceeds_tier1_10k_rpd: bool
    notes: str = ""


def load_dotenv_file(env_path: Path | str = ".env") -> dict[str, str]:
    """Load key=value pairs from a .env file into os.environ without overwriting."""
    loaded: dict[str, str] = {}
    path = Path(env_path)
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _parse_int_header(headers: Mapping[str, str], key: str) -> int | None:
    raw = headers.get(key)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def strip_model_tag(model: str) -> str:
    """Return raw API model ID by stripping any '@effort' suffix (e.g. 'gpt-6.2-sol@low')."""
    return model.split("@", 1)[0].strip()


def compute_call_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Mapping[str, tuple[float, float]] = MODEL_PRICING_PER_M,
) -> float:
    """Compute USD cost assuming 0% prompt caching (output_tokens includes reasoning_tokens)."""
    base_model = strip_model_tag(model)
    in_price, out_price = pricing.get(base_model, pricing.get(model, (0.40, 1.60)))
    return (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price


def probe_single_model(
    client: Any,
    model: str,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> ModelProbeResult:
    """Run one tiny chat completion using `.with_raw_response` to read live headers."""
    api_model = strip_model_tag(model)
    if reasoning_effort is None and "@" in model:
        reasoning_effort = model.split("@", 1)[1].strip()

    messages = [
        {
            "role": "system",
            "content": "Respond with a single capital letter: A or B.",
        },
        {"role": "user", "content": "Reply with A."},
    ]

    # Try with logprobs=True first; if a reasoning model rejects logprobs/temperature, retry cleanly.
    attempts: list[dict[str, Any]] = []
    if reasoning_effort is not None:
        max_ct = max_completion_tokens or (16 if reasoning_effort == "none" else 512)
        base = {
            "model": api_model,
            "messages": messages,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_ct,
        }
        if reasoning_effort == "none":
            attempts.append({**base, "temperature": 0.0, "logprobs": True, "top_logprobs": 5})
        attempts.append({**base, "logprobs": True, "top_logprobs": 5})
        attempts.append(dict(base))
    else:
        attempts.append(
            {
                "model": api_model,
                "messages": messages,
                "max_tokens": 2,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
            }
        )

    last_exc: Exception | None = None
    t0 = time.perf_counter()
    for kwargs in attempts:
        try:
            raw_resp = client.chat.completions.with_raw_response.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            headers = {k.lower(): str(v) for k, v in dict(raw_resp.headers).items()}
            rl_headers = {k: v for k, v in headers.items() if "ratelimit" in k}
            parsed = raw_resp.parse()

            choice = parsed.choices[0]
            content = (choice.message.content or "").strip()
            top_lp_list: list[dict[str, float]] = []
            lp_supported = False
            lp_obj = getattr(choice, "logprobs", None)
            if lp_obj is not None and getattr(lp_obj, "content", None):
                first_tok = lp_obj.content[0]
                candidates = getattr(first_tok, "top_logprobs", None) or []
                if candidates:
                    lp_supported = True
                    top_lp_list = [
                        {str(c.token): float(c.logprob)} for c in candidates
                    ]

            usage = getattr(parsed, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)

            return ModelProbeResult(
                model=model,
                ok=True,
                latency_ms=round(latency_ms, 1),
                logprobs_supported=lp_supported,
                sample_output=content,
                top_logprobs=top_lp_list,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                reasoning_effort=reasoning_effort,
                limit_requests=_parse_int_header(headers, "x-ratelimit-limit-requests"),
                remaining_requests=_parse_int_header(headers, "x-ratelimit-remaining-requests"),
                reset_requests=headers.get("x-ratelimit-reset-requests"),
                limit_tokens=_parse_int_header(headers, "x-ratelimit-limit-tokens"),
                remaining_tokens=_parse_int_header(headers, "x-ratelimit-remaining-tokens"),
                reset_tokens=headers.get("x-ratelimit-reset-tokens"),
                raw_ratelimit_headers=rl_headers,
            )
        except Exception as exc:  # pragma: no cover - fallback/network guard
            last_exc = exc
            continue

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ModelProbeResult(
        model=model,
        ok=False,
        latency_ms=round(latency_ms, 1),
        logprobs_supported=False,
        sample_output="",
        reasoning_effort=reasoning_effort,
        error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "Unknown probe failure",
    )


def _infer_rpm_from_probe(probe: ModelProbeResult | None) -> float | None:
    """Interpret x-ratelimit-limit-requests as RPM (guards against RPD headers)."""
    if probe is None or not probe.ok or probe.limit_requests is None:
        return None
    reset = (probe.reset_requests or "").lower()
    # OpenAI usually reports RPM in x-ratelimit-limit-requests (reset in ms/s/m),
    # unless the RPD bucket is currently binding (reset contains 'h').
    if "h" in reset and probe.limit_requests >= 5_000:
        return probe.limit_requests / (24.0 * 60.0)
    return float(probe.limit_requests)


def estimate_workloads(
    probe_results: Mapping[str, ModelProbeResult],
    avg_judge_input_tokens: int = 450,
    avg_judge_output_tokens: int = 2,
    safety_factor: float = 0.80,
) -> list[WorkloadEstimate]:
    """Calculate exact 0%-cache costs and ETAs for all evaluation sweeps."""
    specs: list[tuple[str, str, int, int, int, str]] = [
        (
            "label_generation",
            DEFAULT_LABEL_MODEL,
            100,
            1_800,
            65,
            f"100 poles x 20 top-band snippets ({DEFAULT_LABEL_MODEL}, 1 call/pole)",
        ),
        (
            "pilot_20poles",
            "gpt-4o-mini",
            1_920,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "20 poles (incl. L4 control) x 4 bands x 4 conds x 6 pairs",
        ),
        (
            "toy_60poles",
            "gpt-4o-mini",
            3_240,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "60 poles x 3 bands x 3 conds x 6 pairs (fallback)",
        ),
        (
            "core_grid",
            "gpt-4o-mini",
            9_600,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "100 poles x 4 bands x 4 conds x 6 pairs",
        ),
        (
            "core_top_p90_cmp",
            "gpt-4o-mini",
            2_400,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "100 poles x top band (<p90 distractor) x 4 conds x 6 pairs",
        ),
        (
            "neighbour_arm",
            "gpt-4o-mini",
            4_800,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "100 poles x 2 bands x 4 conds x 6 pairs",
        ),
        (
            "second_judge_1500",
            "gpt-4.1-mini",
            1_500,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "30 poles (subset ~1,500 calls)",
        ),
        (
            "second_judge_3000",
            "gpt-4.1-mini",
            2_880,
            avg_judge_input_tokens,
            avg_judge_output_tokens,
            "30 poles x 4 bands x 4 conds x 6 pairs (~3,000 calls)",
        ),
    ]

    estimates: list[WorkloadEstimate] = []
    for name, model, n_calls, in_tok, out_tok, notes in specs:
        tot_in = n_calls * in_tok
        tot_out = n_calls * out_tok
        cost = compute_call_cost_usd(model, tot_in, tot_out)

        probe = probe_results.get(model)
        eff_rpm: float | None = None
        eta_min: float | None = None
        eta_hrs: float | None = None

        rpm = _infer_rpm_from_probe(probe)
        tpm = probe.limit_tokens if (probe and probe.ok) else None
        toks_per_call = max(1, in_tok + out_tok)

        caps: list[float] = []
        if rpm is not None and rpm > 0:
            caps.append(rpm * safety_factor)
        if tpm is not None and tpm > 0:
            caps.append((tpm / toks_per_call) * safety_factor)

        if caps:
            eff_rpm = min(caps)
            eta_min = n_calls / max(eff_rpm, 1e-6)
            eta_hrs = eta_min / 60.0

        estimates.append(
            WorkloadEstimate(
                name=name,
                model=model,
                num_calls=n_calls,
                avg_input_tokens=in_tok,
                avg_output_tokens=out_tok,
                total_input_tokens=tot_in,
                total_output_tokens=tot_out,
                estimated_cost_usd=round(cost, 4),
                effective_rpm=round(eff_rpm, 1) if eff_rpm is not None else None,
                eta_minutes=round(eta_min, 1) if eta_min is not None else None,
                fits_rpm_tpm_within_hours=round(eta_hrs, 2) if eta_hrs is not None else None,
                exceeds_tier1_10k_rpd=(n_calls > TIER1_RPD_WARNING_CALLS),
                notes=notes,
            )
        )
    return estimates


def summarize_combined_plan(
    estimates: Sequence[WorkloadEstimate],
    probe_results: Mapping[str, ModelProbeResult] | None = None,
) -> dict[str, Any]:
    """Summarize total calls, cost, and RPD risk for core + neighbour + 2nd judge."""
    by_name = {e.name: e for e in estimates}
    full_keys = [
        "label_generation",
        "core_grid",
        "core_top_p90_cmp",
        "neighbour_arm",
        "second_judge_3000",
    ]
    selected = [by_name[k] for k in full_keys if k in by_name]

    calls_by_model: dict[str, int] = {}
    cost_total = 0.0
    eta_total_min = 0.0
    any_eta = False
    for item in selected:
        calls_by_model[item.model] = calls_by_model.get(item.model, 0) + item.num_calls
        cost_total += item.estimated_cost_usd
        if item.eta_minutes is not None:
            eta_total_min += item.eta_minutes
            any_eta = True

    mini_core_only = by_name["core_grid"].num_calls if "core_grid" in by_name else 0
    mini_core_plus_nb = mini_core_only + (
        by_name["neighbour_arm"].num_calls if "neighbour_arm" in by_name else 0
    )
    mini_all = calls_by_model.get("gpt-4o-mini", 0)

    mini_probe = probe_results.get("gpt-4o-mini") if probe_results else None
    live_mini_rpm = _infer_rpm_from_probe(mini_probe)
    live_mini_tpm = mini_probe.limit_tokens if (mini_probe and mini_probe.ok) else None
    inferred_tier = (
        "Tier 2+ (5,000+ RPM / 2M+ TPM — no Tier-1 10k RPD cap)"
        if (live_mini_rpm is not None and live_mini_rpm >= 2_000)
        else (
            "Tier 1 (<=500 RPM — 10k RPD cap likely applies)"
            if (live_mini_rpm is not None and live_mini_rpm <= 600)
            else "Unknown (check Platform > Settings > Limits)"
        )
    )

    rpd_warning = (
        mini_core_plus_nb > TIER1_RPD_WARNING_CALLS
        and (live_mini_rpm is None or live_mini_rpm <= 600)
    )
    if rpd_warning:
        recommendation = (
            "Core (9,600) fits under a 10k RPD cap, but core + neighbour "
            f"({mini_core_plus_nb:,}) / core + p90 + neighbour ({mini_all:,}) exceeds "
            "10,000 RPD on Tier 1 gpt-4o-mini. Run core on Day 1 and neighbour after "
            "UTC midnight, or route overflow to gpt-4.1-mini."
        )
    elif live_mini_rpm is not None and live_mini_rpm >= 2_000:
        recommendation = (
            f"Full sweep ({sum(e.num_calls for e in selected):,} calls, "
            f"~${cost_total + 0.001:.2f} at 0% cache, ~{eta_total_min:.1f} min) "
            f"fits comfortably within live limits ({int(live_mini_rpm):,} RPM / "
            f"{live_mini_tpm:,} TPM)."
        )
    else:
        recommendation = "Combined call volume fits within observed limits."

    return {
        "assuming_prompt_cache_hit_rate": 0.0,
        "inferred_tier": inferred_tier,
        "total_calls_full_plan": sum(e.num_calls for e in selected),
        "calls_by_model": calls_by_model,
        "core_plus_neighbour_4o_mini_calls": mini_core_plus_nb,
        "total_estimated_cost_usd": round(cost_total, 4),
        "embedding_dedup_cost_usd": 0.001,
        "total_eta_minutes_sequential": round(eta_total_min, 1) if any_eta else None,
        "tier1_10k_rpd_warning": rpd_warning,
        "recommendation": recommendation,
    }


def run_probe(
    models: Sequence[str] = DEFAULT_PROBE_MODELS,
    client: Any | None = None,
    env_path: Path | str = ".env",
    avg_judge_input_tokens: int = 450,
    avg_judge_output_tokens: int = 2,
) -> dict[str, Any]:
    """Execute live probe calls and return structured results + workload estimates."""
    if client is None:
        load_dotenv_file(env_path)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in environment or .env. "
                "Add OPENAI_API_KEY=... to .env before running live probes."
            )
        from openai import OpenAI  # local import so unit tests never need network

        client = OpenAI(api_key=api_key)

    probes: dict[str, ModelProbeResult] = {}
    for model in models:
        probes[model] = probe_single_model(client, model)

    estimates = estimate_workloads(
        probes,
        avg_judge_input_tokens=avg_judge_input_tokens,
        avg_judge_output_tokens=avg_judge_output_tokens,
    )
    summary = summarize_combined_plan(estimates, probe_results=probes)

    return {
        "probes": {m: asdict(res) for m, res in probes.items()},
        "workloads": [asdict(e) for e in estimates],
        "summary": summary,
    }


def format_probe_report(report: Mapping[str, Any]) -> str:
    """Render a concise human-readable report of live headers, logprobs, and ETAs."""
    lines: list[str] = []
    lines.append("# OpenAI Live Rate-Limit & Sweep Feasibility Probe (0% Prompt Cache)")
    lines.append("")
    lines.append("## 1. Live Model Probe Results")
    lines.append(
        "| Model | OK | Logprobs | Limit Req | Rem Req | Reset Req | Limit Tok | Rem Tok | Latency (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|")
    for model, info in report["probes"].items():
        lines.append(
            f"| `{model}` | {info['ok']} | {info['logprobs_supported']} | "
            f"{info['limit_requests']} | {info['remaining_requests']} | "
            f"{info['reset_requests']} | {info['limit_tokens']} | "
            f"{info['remaining_tokens']} | {info['latency_ms']} |"
        )
        if info.get("error"):
            lines.append(f"  - ERROR for `{model}`: {info['error']}")

    lines.append("")
    lines.append("## 2. Workload Cost & ETA Estimates (0% Prompt Caching, 80% Rate-Limit Safety)")
    lines.append(
        "| Workload | Model | Calls | Avg In/Out | Est. USD | Eff. RPM | ETA (min) | >10k RPD? |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for w in report["workloads"]:
        eff_rpm = f"{w['effective_rpm']:.1f}" if w["effective_rpm"] is not None else "n/a"
        eta_min = f"{w['eta_minutes']:.1f}" if w["eta_minutes"] is not None else "n/a"
        lines.append(
            f"| `{w['name']}` | `{w['model']}` | {w['num_calls']:,} | "
            f"{w['avg_input_tokens']}/{w['avg_output_tokens']} | "
            f"${w['estimated_cost_usd']:.3f} | {eff_rpm} | {eta_min} | "
            f"{'YES' if w['exceeds_tier1_10k_rpd'] else 'no'} |"
        )

    s = report["summary"]
    lines.append("")
    lines.append("## 3. Combined Full-Plan Verdict")
    if s.get("inferred_tier"):
        lines.append(f"- Inferred tier: **{s['inferred_tier']}**")
    lines.append(f"- Total full-plan calls: **{s['total_calls_full_plan']:,}** (`{s['calls_by_model']}`)")
    lines.append(
        f"- Total estimated USD (0% caching + ~$0.001 embeddings): "
        f"**${s['total_estimated_cost_usd'] + s['embedding_dedup_cost_usd']:.3f}**"
    )
    if s["total_eta_minutes_sequential"] is not None:
        eta_m = s["total_eta_minutes_sequential"]
        lines.append(
            f"- Estimated sequential runtime at 80% rate-limit cap: "
            f"**{eta_m:.1f} min ({math.ceil(eta_m) / 60:.2f} hr)**"
        )
    lines.append(f"- Tier-1 10k RPD warning: **{s['tier1_10k_rpd_warning']}**")
    lines.append(f"- Recommendation: {s['recommendation']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe live OpenAI rate limits and logprobs support for evaluation sweeps."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_PROBE_MODELS),
        help="Models to probe (default: gpt-4o-mini gpt-4.1-mini).",
    )
    parser.add_argument(
        "--avg-input-tokens",
        type=int,
        default=450,
        help="Expected average input tokens per single-pair judge call (default: 450).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON probe report.",
    )
    args = parser.parse_args(argv)

    report = run_probe(
        models=args.models,
        env_path=args.env_file,
        avg_judge_input_tokens=args.avg_input_tokens,
    )
    print(format_probe_report(report))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
