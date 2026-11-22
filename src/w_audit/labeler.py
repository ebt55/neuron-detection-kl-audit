"""Pole label generation and 4-condition label construction for Qwen3-1.7B evaluation.

Generates one concise natural-language label per neuron pole from its 20
disjoint train_label top-band (p99.9+) snippets using gpt-4.1-mini (resumable
JSONL cache), then builds all four evaluation label conditions:
  1. own        - pole's own generated label ("{title}: {description}")
  2. wrong      - another pole's label from a DIFFERENT layer, rejecting
                  near-duplicates via OpenAI text-embedding-3-small cosine > 0.85
                  (with deterministic word-Jaccard > 0.50 offline fallback)
  3. no_label   - explanation omitted (handled by neutral prompt in judge.py)
  4. token_only - deterministic top-3 trigger token list from top-band examples
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from w_audit.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LABEL_MODEL,
    LABELER_SYSTEM_PROMPT,
    LABEL_EXAMPLES_PATH,
    LABELS_PATH,
    POLES_PATH,
    SEED,
    WRONG_LABEL_EMBED_COSINE_MAX,
    WRONG_LABEL_JACCARD_MAX,
    format_token_only_label,
)
from w_audit.probe_limits import (
    MODEL_PRICING_PER_M,
    compute_call_cost_usd,
    load_dotenv_file,
)

DEFAULT_MAX_EMBED_COSINE: float = WRONG_LABEL_EMBED_COSINE_MAX
DEFAULT_MAX_JACCARD: float = WRONG_LABEL_JACCARD_MAX


class LabelerSpendLimitError(RuntimeError):
    """Raised when label generation would exceed --max-usd."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, ignoring blank lines and trailing corrupted fragments."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                # Tolerate truncated final line after an interrupted write
                if line_no == 0:
                    raise
                continue
    return rows


def compute_top_k_tokens(
    examples: Sequence[Mapping[str, Any]],
    k: int = 3,
) -> tuple[list[int], list[str]]:
    """Derive deterministic top-k distinct target tokens from train_label examples.

    Matches specification: most frequent distinct marked target
    tokens across the 20 train_label p99.9+ snippets, breaking ties by highest
    mean act_dir, then lowest target_token_id.
    """
    counts: Counter[int] = Counter()
    act_sums: dict[int, float] = defaultdict(float)
    tok_strs: dict[int, str] = {}

    for ex in examples:
        tid = int(ex["target_token_id"])
        counts[tid] += 1
        act_sums[tid] += float(ex.get("act_dir", abs(float(ex.get("act_val", 0.0)))))
        if tid not in tok_strs:
            tok_strs[tid] = str(ex["target_token_str"])

    ranked = sorted(
        counts.keys(),
        key=lambda tid: (-counts[tid], -(act_sums[tid] / counts[tid]), tid),
    )
    top_ids = ranked[:k]
    top_strs = [tok_strs[tid] for tid in top_ids]
    return top_ids, top_strs


def build_labeler_user_prompt(
    pole_id: str,
    examples: Sequence[Mapping[str, Any]],
) -> str:
    """Format a pole's 20 top-band train_label snippets sorted by act_dir desc."""
    ordered = sorted(
        examples,
        key=lambda r: float(r.get("act_dir", abs(float(r.get("act_val", 0.0))))),
        reverse=True,
    )
    lines = [
        f"Neuron pole: {pole_id}",
        f"Top {len(ordered)} strongly activating snippets (target token marked with <<...>>):",
        "",
    ]
    for idx, ex in enumerate(ordered, start=1):
        act_dir = float(ex.get("act_dir", abs(float(ex.get("act_val", 0.0)))))
        marked = str(ex["marked_text"]).replace("\r\n", "\n")
        lines.append(f"[{idx}] (act={act_dir:.3f}) {marked}")
    lines.append("")
    lines.append('Return ONLY a JSON object with keys "title" and "description".')
    return "\n".join(lines)


def parse_label_json(raw_text: str) -> tuple[str, str]:
    """Parse {"title": ..., "description": ...} from model output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse JSON label from response: {raw_text!r}")
        data = json.loads(match.group(0))

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    if not title or not description:
        raise ValueError(f"Label JSON missing non-empty title/description: {data!r}")
    return title, description


_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bthis\s+(?:mlp\s+)?neuron(?:\s+pole)?\s+(?:strongly\s+)?(?:activates|fires|responds)(?:\s+strongly)?\s+(?:on|to|when)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:mlp\s+)?neuron\s+pole\s+(?:for|activating|firing|responding|that\s+(?:strongly\s+)?(?:activates|fires|responds))(?:\s+strongly)?(?:\s+(?:on|to|when))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:strongly\s+)?(?:activates|fires|responds)\s+(?:strongly\s+)?(?:on|to|when)\s+(?:tokens?\s+(?:representing|such\s+as|that\s+are|including)\s*)?",
        re.IGNORECASE,
    ),
    re.compile(r"\bon\s+the\s+marked\s+<<token>>\b", re.IGNORECASE),
    re.compile(r"\bmarked\s+<<token>>\b", re.IGNORECASE),
    re.compile(r"<<token>>", re.IGNORECASE),
    re.compile(r"\btoken\s+activations?\b", re.IGNORECASE),
    re.compile(r"\bneuron\s+pole\b", re.IGNORECASE),
    re.compile(r"\bmixed\s*/\s*polysemantic\s*:\s*", re.IGNORECASE),
)


def strip_label_boilerplate(text: str) -> str:
    """Strip recurring autointerp template boilerplate before similarity embedding."""
    cleaned = text
    for pat in _BOILERPLATE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;,.-")
    return cleaned if len(cleaned) >= 3 else text.strip()


def detect_repetition_loop(text: str, min_consecutive: int = 4) -> str | None:
    """Return a reason string if text contains a degenerate repetition loop, else None.

    Catches >=4 consecutive repeated tokens/quoted-items/clauses (e.g. "'-ic', '-ic',
    '-ic', '-ic'"), identical quoted token spam (>=4 times), or >=8 repeated characters.
    """
    if not text or not text.strip():
        return "empty label text"

    # 1. Long single-character run (e.g. '........' or '--------')
    char_run = re.search(r"(.)\1{7,}", text)
    if char_run:
        return f"character run {char_run.group(1)!r} x>=8"

    # 2. Quoted-item repetition anywhere or consecutively (e.g. '-ic', '-ic', ...)
    quoted_items = re.findall(r"['\"]([^'\"]{1,30})['\"]", text)
    if quoted_items:
        norm_q = [q.strip().lower() for q in quoted_items if q.strip()]
        # Consecutive identical quoted items
        run_len = 1
        for i in range(1, len(norm_q)):
            if norm_q[i] == norm_q[i - 1]:
                run_len += 1
                if run_len >= min_consecutive:
                    return f"consecutive quoted token {norm_q[i]!r} x{run_len}"
            else:
                run_len = 1
        # Overall excessive repetition of the same quoted token
        q_counts = Counter(norm_q)
        top_tok, top_cnt = q_counts.most_common(1)[0]
        if top_cnt >= min_consecutive:
            return f"repeated quoted token {top_tok!r} x{top_cnt}"

    # 3. Consecutive repeated comma/semicolon-separated clauses
    clauses = [
        re.sub(r"\s+", " ", c.strip(" '\"()[]")).lower()
        for c in re.split(r"[,;]+", text)
    ]
    clauses = [c for c in clauses if c]
    run_len = 1
    for i in range(1, len(clauses)):
        if clauses[i] == clauses[i - 1]:
            run_len += 1
            if run_len >= min_consecutive:
                return f"consecutive repeated clause {clauses[i]!r} x{run_len}"
        else:
            run_len = 1

    # 4. Consecutive repeated 1..6-word n-grams
    words = re.findall(r"[a-z0-9'-]+", text.lower())
    for n in range(1, 7):
        if len(words) < n * min_consecutive:
            continue
        for start in range(0, len(words) - n * min_consecutive + 1):
            gram = tuple(words[start : start + n])
            reps = 1
            pos = start + n
            while pos + n <= len(words) and tuple(words[pos : pos + n]) == gram:
                reps += 1
                pos += n
            if reps >= min_consecutive:
                return f"consecutive repeated {n}-gram {' '.join(gram)!r} x{reps}"

    return None


def word_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Deterministic fallback lexical similarity on boilerplate-stripped words."""
    tokens_a = set(re.findall(r"[a-z0-9]+", strip_label_boilerplate(text_a).lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", strip_label_boilerplate(text_b).lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for row vectors."""
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = arr / norms
    return np.clip(normed @ normed.T, -1.0, 1.0)


def assign_wrong_labels(
    own_records: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray | None = None,
    max_embed_cosine: float = DEFAULT_MAX_EMBED_COSINE,
    max_jaccard: float = DEFAULT_MAX_JACCARD,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Assign each pole a cross-layer wrong label rejecting near-duplicates.

    Args:
        own_records: Sequence of dicts with keys 'pole_id', 'layer', 'own_label'.
        embeddings: Optional (N, D) float array aligned with own_records. If
            provided, candidates with cosine > max_embed_cosine are rejected.
            Otherwise falls back deterministically to word-Jaccard > max_jaccard.
        max_embed_cosine: Upper bound on embedding cosine similarity (default 0.85).
        max_jaccard: Upper bound on fallback word-Jaccard similarity (default 0.50).
        seed: Global deterministic seed.

    Returns:
        Mapping from pole_id -> dict with wrong_label, wrong_donor_pole_id,
        wrong_donor_layer, wrong_embed_cosine, wrong_similarity_metric,
        wrong_threshold_relaxed.
    """
    ordered = sorted(own_records, key=lambda r: str(r["pole_id"]))
    n = len(ordered)
    if n == 0:
        return {}

    # Align embeddings with sorted order if provided
    orig_index = {str(r["pole_id"]): idx for idx, r in enumerate(own_records)}
    use_embed = embeddings is not None and len(embeddings) == len(own_records)
    if use_embed:
        perm = [orig_index[str(r["pole_id"])] for r in ordered]
        sim_matrix = cosine_similarity_matrix(np.asarray(embeddings)[perm])
        threshold = max_embed_cosine
        metric_name = "text-embedding-3-small"
    else:
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                s = word_jaccard_similarity(
                    str(ordered[i]["own_label"]),
                    str(ordered[j]["own_label"]),
                )
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s
            sim_matrix[i, i] = 1.0
        threshold = max_jaccard
        metric_name = "word_jaccard_fallback"

    donor_usage: Counter[str] = Counter()
    assignments: dict[str, dict[str, Any]] = {}

    for i, target in enumerate(ordered):
        target_id = str(target["pole_id"])
        target_layer = int(target["layer"])

        cross_layer_indices = [
            j for j, cand in enumerate(ordered) if int(cand["layer"]) != target_layer
        ]
        if not cross_layer_indices:
            raise ValueError(
                f"Pole {target_id} (layer {target_layer}) has no cross-layer donor "
                "candidates; at least 2 distinct layers are required."
            )

        # Deterministic per-pole candidate permutation
        h = hashlib.sha256(f"{seed}:wrong:{target_id}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        shuffled = list(cross_layer_indices)
        rng.shuffle(shuffled)
        rank_map = {j: pos for pos, j in enumerate(shuffled)}

        eligible = [j for j in shuffled if float(sim_matrix[i, j]) <= threshold]
        relaxed = False
        if eligible:
            # Prefer least-used donors to distribute wrong labels evenly
            chosen_j = min(
                eligible,
                key=lambda j: (donor_usage[str(ordered[j]["pole_id"])], rank_map[j]),
            )
        else:
            # Degenerate fallback: pick lowest-similarity cross-layer donor
            relaxed = True
            chosen_j = min(
                shuffled,
                key=lambda j: (float(sim_matrix[i, j]), rank_map[j]),
            )

        donor = ordered[chosen_j]
        donor_id = str(donor["pole_id"])
        donor_usage[donor_id] += 1

        assignments[target_id] = {
            "wrong_label": str(donor["own_label"]),
            "wrong_donor_pole_id": donor_id,
            "wrong_donor_layer": int(donor["layer"]),
            "wrong_embed_cosine": round(float(sim_matrix[i, chosen_j]), 6),
            "wrong_similarity_metric": metric_name,
            "wrong_threshold_relaxed": relaxed,
        }

    return assignments


def get_label_for_condition(
    label_row: Mapping[str, Any],
    condition: str,
) -> str | None:
    """Return the exact label text for a condition, or None for no_label."""
    if condition == "own":
        return str(label_row["own_label"])
    if condition == "wrong":
        return str(label_row["wrong_label"])
    if condition == "token_only":
        return str(label_row["token_only_label"])
    if condition == "no_label":
        return None
    raise ValueError(f"Unknown label condition: {condition!r}")


async def _generate_one_own_label(
    client: Any,
    pole: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    model: str,
    max_api_retries: int = 5,
    max_repetition_retries: int = 2,
) -> dict[str, Any]:
    """Call OpenAI chat completions to generate one pole's own label.

    Validates every generated label with detect_repetition_loop; if a loop is
    found, retries up to max_repetition_retries (2x) with frequency_penalty=0.5+.
    """
    pole_id = str(pole["pole_id"])
    user_prompt = build_labeler_user_prompt(pole_id, examples)
    base_messages = [
        {"role": "system", "content": LABELER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_loop_reason: str | None = None

    for rep_attempt in range(max_repetition_retries + 1):
        messages = list(base_messages)
        extra_kwargs: dict[str, Any] = {}
        if rep_attempt > 0:
            extra_kwargs["frequency_penalty"] = 0.5 if rep_attempt == 1 else 0.8
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response contained a repetition loop ({last_loop_reason}). "
                        "Rewrite the JSON object concisely with NO repeated tokens or clauses. "
                        'Return ONLY valid JSON with keys "title" and "description".'
                    ),
                }
            )

        last_err: Exception | None = None
        parsed_ok = False
        title = ""
        description = ""
        for api_attempt in range(max_api_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=220,
                    response_format={"type": "json_object"},
                    **extra_kwargs,
                )
                usage = getattr(resp, "usage", None)
                total_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                total_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                raw_content = resp.choices[0].message.content or ""
                title, description = parse_label_json(raw_content)
                parsed_ok = True
                break
            except Exception as exc:
                last_err = exc
                if api_attempt + 1 >= max_api_retries:
                    break
                await asyncio.sleep(min(20.0, (2.0**api_attempt) * 0.5))

        if not parsed_ok:
            raise RuntimeError(
                f"Failed to generate label for {pole_id} after {max_api_retries} attempts: {last_err}"
            ) from last_err

        loop_reason = detect_repetition_loop(title) or detect_repetition_loop(description)
        if loop_reason is None:
            return {
                "pole_id": pole_id,
                "layer": int(pole["layer"]),
                "own_title": title,
                "own_description": description,
                "own_label": f"{title}: {description}",
                "label_model": model,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "repetition_retries": rep_attempt,
            }
        last_loop_reason = loop_reason

    raise ValueError(
        f"Label for {pole_id} still contained repetition loop after "
        f"{max_repetition_retries} retries: {last_loop_reason}"
    )


async def fetch_openai_embeddings(
    client: Any,
    texts: Sequence[str],
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> tuple[np.ndarray, int]:
    """Fetch embeddings for all own labels via OpenAI embeddings API."""
    resp = await client.embeddings.create(model=embed_model, input=list(texts))
    items = sorted(resp.data, key=lambda d: int(d.index))
    matrix = np.asarray([item.embedding for item in items], dtype=np.float64)
    usage = getattr(resp, "usage", None)
    total_tokens = int(getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    return matrix, total_tokens


async def generate_labels_async(
    poles_path: Path,
    examples_path: Path,
    output_path: Path,
    cache_path: Path | None = None,
    label_model: str = DEFAULT_LABEL_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    subset: str = "all",
    max_embed_cosine: float = DEFAULT_MAX_EMBED_COSINE,
    max_jaccard: float = DEFAULT_MAX_JACCARD,
    max_usd: float = 5.0,
    concurrency: int = 10,
    use_embeddings: bool = True,
    seed: int = SEED,
    force: bool = False,
    client: Any | None = None,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    env_path: Path | str = ".env",
) -> list[dict[str, Any]]:
    """Run resumable own-label generation + cross-layer wrong-label assignment."""
    poles_payload = json.loads(poles_path.read_text(encoding="utf-8"))
    all_poles: list[dict[str, Any]] = list(poles_payload["poles"])

    if subset == "pilot":
        poles = [p for p in all_poles if p.get("in_pilot")]
    elif subset == "toy":
        poles = [p for p in all_poles if p.get("in_toy")]
    elif subset == "second_judge":
        poles = [p for p in all_poles if p.get("in_second_judge")]
    elif subset == "all":
        poles = all_poles
    else:
        raise ValueError(f"Unknown subset: {subset!r}")

    examples_by_pole: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(examples_path):
        examples_by_pole[str(row["pole_id"])].append(row)

    if cache_path is None:
        cache_path = output_path.parent / "own_labels_cache.jsonl"

    # Load resumable own-label cache ONLY for matching label_model & valid non-loop labels
    cached_own: dict[str, dict[str, Any]] = {}
    if not force:
        for src in (output_path, cache_path):
            for row in load_jsonl(src):
                pid = str(row.get("pole_id", ""))
                row_model = str(row.get("label_model", ""))
                own_lbl = str(row.get("own_label", ""))
                if (
                    pid
                    and row_model == label_model
                    and own_lbl
                    and row.get("own_title")
                    and detect_repetition_loop(own_lbl) is None
                ):
                    cached_own[pid] = {
                        "pole_id": pid,
                        "layer": int(row["layer"]),
                        "own_title": str(row["own_title"]),
                        "own_description": str(row["own_description"]),
                        "own_label": own_lbl,
                        "label_model": row_model,
                        "prompt_tokens": int(row.get("prompt_tokens", 0)),
                        "completion_tokens": int(row.get("completion_tokens", 0)),
                        "repetition_retries": int(row.get("repetition_retries", 0)),
                    }

    missing_poles = [p for p in poles if str(p["pole_id"]) not in cached_own]

    owned_client = False
    if (missing_poles or (use_embeddings and embed_fn is None)) and client is None:
        load_dotenv_file(env_path)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in environment or .env. "
                "Add OPENAI_API_KEY=... to .env before running label generation."
            )
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        owned_client = True

    spent_usd = 0.0
    spend_lock = asyncio.Lock()
    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, concurrency))

    try:
        if missing_poles:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                raw_tail = cache_path.read_bytes()
                if not raw_tail.endswith(b"\n"):
                    last_nl = raw_tail.rfind(b"\n")
                    frag = raw_tail[last_nl + 1 :].decode("utf-8", errors="replace").strip()
                    try:
                        json.loads(frag)
                        with cache_path.open("ab") as fix_fh:
                            fix_fh.write(b"\n")
                    except json.JSONDecodeError:
                        with cache_path.open("r+b") as fix_fh:
                            fix_fh.truncate(last_nl + 1 if last_nl >= 0 else 0)
            with cache_path.open("a", encoding="utf-8") as cache_fh:

                async def _worker(pole_obj: dict[str, Any]) -> None:
                    nonlocal spent_usd
                    pid = str(pole_obj["pole_id"])
                    exs = examples_by_pole.get(pid, [])
                    if not exs:
                        raise ValueError(f"No train_label examples found for pole {pid}")
                    async with sem:
                        async with spend_lock:
                            if spent_usd >= max_usd:
                                raise LabelerSpendLimitError(
                                    f"Labeler spend ${spent_usd:.4f} reached cap ${max_usd:.2f}"
                                )
                        rec = await _generate_one_own_label(
                            client=client,
                            pole=pole_obj,
                            examples=exs,
                            model=label_model,
                        )
                        call_usd = compute_call_cost_usd(
                            label_model,
                            rec["prompt_tokens"],
                            rec["completion_tokens"],
                            MODEL_PRICING_PER_M,
                        )
                        async with spend_lock:
                            spent_usd += call_usd
                            if spent_usd > max_usd:
                                # Persist this completed call before aborting remaining work
                                async with file_lock:
                                    cache_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    cache_fh.flush()
                                    cached_own[pid] = rec
                                raise LabelerSpendLimitError(
                                    f"Labeler spend ${spent_usd:.4f} exceeded cap ${max_usd:.2f}"
                                )
                        async with file_lock:
                            cache_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            cache_fh.flush()
                            cached_own[pid] = rec

                tasks = [asyncio.create_task(_worker(p)) for p in missing_poles]
                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    raise

        own_records = [cached_own[str(p["pole_id"])] for p in poles]

        # Strip residual boilerplate BEFORE embedding for cross-layer near-duplicate rejection
        embeddings: np.ndarray | None = None
        if use_embeddings and len(own_records) > 0:
            texts = [strip_label_boilerplate(str(r["own_label"])) for r in own_records]
            if embed_fn is not None:
                embeddings = np.asarray(embed_fn(texts), dtype=np.float64)
            elif client is not None and hasattr(client, "embeddings"):
                try:
                    embeddings, emb_toks = await fetch_openai_embeddings(
                        client, texts, embed_model=embed_model
                    )
                    spent_usd += compute_call_cost_usd(
                        embed_model, emb_toks, 0, MODEL_PRICING_PER_M
                    )
                except Exception:
                    embeddings = None

        wrong_map = assign_wrong_labels(
            own_records=own_records,
            embeddings=embeddings,
            max_embed_cosine=max_embed_cosine,
            max_jaccard=max_jaccard,
            seed=seed,
        )

        # Combine into canonical data/labels.jsonl schema (canonical labels schema)
        final_rows: list[dict[str, Any]] = []
        for pole in poles:
            pid = str(pole["pole_id"])
            own = cached_own[pid]
            wrong = wrong_map[pid]

            if pole.get("top_3_token_ids") and pole.get("top_3_tokens"):
                top_3_ids = [int(x) for x in pole["top_3_token_ids"][:3]]
                top_3_toks = [str(x) for x in pole["top_3_tokens"][:3]]
            else:
                top_3_ids, top_3_toks = compute_top_k_tokens(
                    examples_by_pole.get(pid, []), k=3
                )

            row = {
                "pole_id": pid,
                "layer": int(pole["layer"]),
                "own_title": own["own_title"],
                "own_description": own["own_description"],
                "own_label": own["own_label"],
                "wrong_label": wrong["wrong_label"],
                "wrong_donor_pole_id": wrong["wrong_donor_pole_id"],
                "wrong_donor_layer": wrong["wrong_donor_layer"],
                "wrong_embed_cosine": wrong["wrong_embed_cosine"],
                "wrong_similarity_metric": wrong["wrong_similarity_metric"],
                "wrong_threshold_relaxed": wrong["wrong_threshold_relaxed"],
                "token_only_label": format_token_only_label(top_3_toks),
                "top_3_token_ids": top_3_ids,
                "top_3_tokens": top_3_toks,
                "label_model": own["label_model"],
                "prompt_tokens": own["prompt_tokens"],
                "completion_tokens": own["completion_tokens"],
                "repetition_retries": own.get("repetition_retries", 0),
            }
            final_rows.append(row)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as out_fh:
            for row in final_rows:
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(output_path)
        return final_rows
    finally:
        if owned_client and client is not None and hasattr(client, "close"):
            await client.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Generate {DEFAULT_LABEL_MODEL} pole labels and build all 4 label conditions."
    )
    parser.add_argument("--poles", type=Path, default=Path("data/poles.json"))
    parser.add_argument(
        "--examples", type=Path, default=Path("data/label_examples.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument(
        "--cache", type=Path, default=Path("data/own_labels_cache.jsonl")
    )
    parser.add_argument("--model", type=str, default=DEFAULT_LABEL_MODEL)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--subset",
        choices=("all", "pilot", "toy", "second_judge"),
        default="all",
    )
    parser.add_argument(
        "--max-embed-cosine", type=float, default=DEFAULT_MAX_EMBED_COSINE
    )
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)

    rows = asyncio.run(
        generate_labels_async(
            poles_path=args.poles,
            examples_path=args.examples,
            output_path=args.output,
            cache_path=args.cache,
            label_model=args.model,
            embed_model=args.embed_model,
            subset=args.subset,
            max_embed_cosine=args.max_embed_cosine,
            max_usd=args.max_usd,
            concurrency=args.concurrency,
            use_embeddings=not args.no_embeddings,
            seed=args.seed,
            force=args.force,
            env_path=args.env_file,
        )
    )
    sims = sorted(float(r["wrong_embed_cosine"]) for r in rows)
    p_tok = sum(int(r.get("prompt_tokens", 0)) for r in rows)
    c_tok = sum(int(r.get("completion_tokens", 0)) for r in rows)
    est_usd = compute_call_cost_usd(args.model, p_tok, c_tok, MODEL_PRICING_PER_M)
    over_cut = sum(1 for s in sims if s > args.max_embed_cosine)
    same_layer = sum(1 for r in rows if int(r["wrong_donor_layer"]) == int(r["layer"]))
    rep_loops = sum(
        1 for r in rows if detect_repetition_loop(str(r["own_label"])) is not None
    )
    rep_retries = sum(int(r.get("repetition_retries", 0)) for r in rows)
    if sims:
        med = float(np.median(sims))
        p90 = float(np.percentile(sims, 90))
        sim_str = f"min={sims[0]:.3f} med={med:.3f} p90={p90:.3f} max={sims[-1]:.3f}"
    else:
        sim_str = "n/a"
    print(
        f"Wrote {len(rows)} pole labels ({args.model}) to {args.output} | "
        f"wrong_cos: {sim_str} | over_{args.max_embed_cosine:.2f}={over_cut} | "
        f"same_layer={same_layer} | rep_loops={rep_loops} (retries={rep_retries}) | "
        f"tokens={p_tok:,}in/{c_tok:,}out (~${est_usd:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
