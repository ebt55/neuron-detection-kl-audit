"""Shared configuration, constants, schemas, and deterministic helpers for the two-snippet neuron detection audit.

All conventions are locked to the evaluation specification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

# ---------------------------------------------------------------------------
# Repository & Artifact Paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"
AUDIT_DIR: Path = REPO_ROOT / "audit"

POLES_PATH: Path = DATA_DIR / "poles.json"
LABEL_EXAMPLES_PATH: Path = DATA_DIR / "label_examples.jsonl"
SNIPPETS_PATH: Path = DATA_DIR / "snippets.jsonl"
LABELS_PATH: Path = DATA_DIR / "labels.jsonl"

BASELINES_PATH: Path = RESULTS_DIR / "baselines.jsonl"
STRUCTURAL_CHECKS_PATH: Path = RESULTS_DIR / "structural_checks.json"
JUDGEMENTS_PATH: Path = RESULTS_DIR / "judgements.jsonl"
RATE_LIMITS_PATH: Path = RESULTS_DIR / "rate_limits.json"
SUMMARY_TABLES_PATH: Path = RESULTS_DIR / "summary_tables.md"
MAIN_FIGURE_PATH: Path = FIGURES_DIR / "w_band_accuracy.png"
WORKSHEET_PATH: Path = AUDIT_DIR / "raw_30_poles_worksheet.md"

# ---------------------------------------------------------------------------
# Experiment Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.0"
SEED: int = 42

MODEL_ID: str = "Qwen/Qwen3-1.7B"
DATASET_ID: str = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG: str = "sample-10BT"
FALLBACK_DATASET_ID: str = "allenai/c4"

TOTAL_TOKENS_TARGET: int = 2_000_000
MAX_SEQ_LEN: int = 512
MIN_DOC_TOKENS: int = 69

# Snippet Windowing (Goodfire Table 4: 60 before + marked target + 8 after)
WINDOW_BEFORE: int = 60
WINDOW_AFTER: int = 8
WINDOW_TOTAL: int = WINDOW_BEFORE + 1 + WINDOW_AFTER  # 69
TARGET_OFFSET: int = WINDOW_BEFORE  # 60 (0-indexed)
MARKER_OPEN: str = "<<"
MARKER_CLOSE: str = ">>"
DEDUP_SUFFIX_LEN: int = 16

LABEL_EXAMPLES_PER_POLE: int = 20
PAIRS_PER_CELL: int = 6

# Sampled Layers & Nested Deterministic Subsets
LAYERS: tuple[int, ...] = (4, 11, 18, 24)
LAYER4_CONTROL_LAYER: int = 4

# Neurons per layer (each neuron yields 2 poles: pos & neg)
LAYER_NEURON_COUNTS: dict[int, int] = {4: 13, 11: 13, 18: 12, 24: 12}  # 50 neurons = 100 poles
PILOT_NEURON_COUNTS: dict[int, int] = {4: 4, 11: 2, 18: 2, 24: 2}  # 10 neurons = 20 poles
SECOND_JUDGE_NEURON_COUNTS: dict[int, int] = {4: 4, 11: 4, 18: 4, 24: 3}  # 15 neurons = 30 poles
TOY_NEURON_COUNTS: dict[int, int] = {4: 8, 11: 8, 18: 7, 24: 7}  # 30 neurons = 60 poles

# Canonical Enums
BandName = Literal["p75_90", "p90_97", "p97_99.9", "p99.9_plus"]
DistractorType = Literal["dead_zone", "goodfire_lt_p90", "nearest_wgate_tier0"]
ConditionName = Literal["own", "wrong", "no_label", "token_only"]
RuleName = Literal["rarity", "top_token", "top_10_token", "window_has_ge_p97"]
QuadrantName = Literal["Q1", "Q2", "Q3", "Q4"]
JudgeBackend = Literal["openai", "ollama"]
OutcomeType = Literal["scored", "unparseable", "logprob_tie"]

BANDS: tuple[BandName, ...] = ("p75_90", "p90_97", "p97_99.9", "p99.9_plus")
TOY_BANDS: tuple[BandName, ...] = ("p75_90", "p90_97", "p99.9_plus")
NEIGHBOUR_BANDS: tuple[BandName, ...] = ("p99.9_plus", "p90_97")
P90_COMPARE_BANDS: tuple[BandName, ...] = ("p99.9_plus",)

NEIGHBOUR_DISTRACTOR_TYPE: DistractorType = "nearest_wgate_tier0"
DISTRACTOR_TYPES: tuple[DistractorType, ...] = (
    "dead_zone",
    "goodfire_lt_p90",
    "nearest_wgate_tier0",
)
CONDITIONS: tuple[ConditionName, ...] = ("own", "wrong", "no_label", "token_only")
TOY_CONDITIONS: tuple[ConditionName, ...] = ("own", "wrong", "no_label")
RULES: tuple[RuleName, ...] = ("rarity", "top_token", "top_10_token", "window_has_ge_p97")
PRIMARY_RULES: tuple[RuleName, ...] = ("rarity", "top_token", "window_has_ge_p97")
QUADRANTS: tuple[QuadrantName, ...] = ("Q1", "Q2", "Q3", "Q4")

# Band quantile boundaries on directional activation d = s * a
BAND_QUANTILES: dict[BandName, tuple[float, float | None]] = {
    "p75_90": (0.75, 0.90),
    "p90_97": (0.90, 0.97),
    "p97_99.9": (0.97, 0.999),
    "p99.9_plus": (0.999, None),
}

# Models & Cost Defaults
DEFAULT_LABEL_MODEL: str = "gpt-4.1"
DEFAULT_JUDGE_MODEL: str = "gpt-4o-mini"
DEFAULT_SECOND_JUDGE_MODEL: str = "gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL: str = "qwen3.5:9b"
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_EMBED_MODEL: str = "text-embedding-3-small"
WRONG_LABEL_EMBED_COSINE_MAX: float = 0.65
WRONG_LABEL_JACCARD_MAX: float = 0.50
DEFAULT_MAX_USD: float = 25.0

# OpenAI Standard-tier pricing (USD per 1M tokens, 0% assumed prompt caching)
# Note: Reasoning tokens are billed inside completion_tokens at the output rate.
MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    # model_name: (input_usd_per_1M, output_usd_per_1M)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-6.2-sol": (2.00, 10.00),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-luna": (0.10, 0.50),
    "text-embedding-3-small": (0.02, 0.00),
}

# ---------------------------------------------------------------------------
# Prompt Templates (labeler and unified judge)
# ---------------------------------------------------------------------------

LABELER_SYSTEM_PROMPT: str = (
    "You are an interpretability researcher writing a concise, accurate description "
    "of an MLP neuron pole in a language model.\n"
    "You will be given 20 text snippets where this neuron pole fires strongly on the "
    "marked <<token>>.\n"
    "Write a JSON object with exactly two keys:\n"
    '- "title": a short, specific descriptive title (3 to 10 words). Do NOT include '
    'generic framing like "Neuron pole for..." or "Token activations...". If the 20 '
    'snippets show no single coherent pattern, start the title with '
    '"Mixed / polysemantic:" followed by the 2-3 main clusters.\n'
    '- "description": 1 to 2 concise sentences stating the concrete lexical, '
    "morphological, syntactic, or semantic triggers on or immediately around the "
    'marked <<token>>. FORBIDDEN boilerplate: do NOT start with or include generic '
    'filler like "This neuron pole strongly activates on...", "This neuron pole '
    'fires on...", or "This neuron responds to...". Name concrete token shapes, '
    "suffixes, grammatical constructions, or domains directly. Do not repeat any "
    "token or phrase in a list, and do not copy multi-word quotes from the snippets.\n"
    "Respond with valid JSON only."
)

# Unified Single Judge Prompt Template for ALL 4 conditions:
# - Identical system & user template across own, wrong, no_label, token_only.
# - ZERO occurrences of "strongly" anywhere in judge prompts.
NO_LABEL_DESCRIPTION: str = "(none provided)"

JUDGE_SYSTEM_PROMPT: str = (
    "You are evaluating which of two short text snippets causes an MLP neuron pole "
    "in a language model to fire on its marked token.\n"
    "You will be given a description of the neuron pole (which may be '(none provided)'), "
    "followed by two short text snippets labeled A and B. In each snippet, exactly one "
    "target token is marked with <<...>>.\n"
    "In exactly one snippet, the neuron pole fires on the marked <<token>>. In the "
    "other snippet, it does not fire.\n"
    "Based on the description (if provided) and the two snippets, decide whether snippet A "
    "or snippet B makes the neuron pole fire on its marked <<token>>.\n"
    "Respond with a single letter: A or B. Do not include any explanation or extra text."
)

JUDGE_USER_TEMPLATE: str = (
    "Neuron pole description:\n"
    "{description}\n\n"
    "Snippet A:\n"
    "{snippet_a}\n\n"
    "Snippet B:\n"
    "{snippet_b}\n\n"
    "Which snippet (A or B) makes the neuron pole fire on the marked <<token>>? "
    "Reply with only A or B."
)

# Backwards-compatible aliases pointing to the single unified template
JUDGE_LABELED_SYSTEM_PROMPT: str = JUDGE_SYSTEM_PROMPT
JUDGE_LABELED_USER_TEMPLATE: str = JUDGE_USER_TEMPLATE
JUDGE_NO_LABEL_SYSTEM_PROMPT: str = JUDGE_SYSTEM_PROMPT
JUDGE_NO_LABEL_USER_TEMPLATE: str = JUDGE_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Canonical ID, Quadrant, Slot, and Formatting Helpers
# ---------------------------------------------------------------------------


def make_pole_id(layer: int, neuron: int, sign: int) -> str:
    """Return canonical pole ID, e.g. 'L04_N0123_pos' or 'L04_N0123_neg'."""
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")
    pole_name = "pos" if sign > 0 else "neg"
    return f"L{layer:02d}_N{neuron:04d}_{pole_name}"


def make_pair_id(
    pole_id: str,
    band: str,
    distractor_type: str,
    pair_idx: int,
) -> str:
    """Return canonical pair ID: '{pole_id}__{band}__{distractor_type}__{pair_idx}'."""
    return f"{pole_id}__{band}__{distractor_type}__{pair_idx}"


def deterministic_pos_slot(pair_id: str, seed: int = SEED) -> Literal["A", "B"]:
    """Deterministically assign firing snippet to Slot 'A' or 'B' from sha256 LSB."""
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    return "A" if (digest[-1] & 1) == 0 else "B"


def classify_quadrant(act: float, gate_open: bool) -> QuadrantName:
    """Classify SwiGLU token into Goodfire's exact Fig. 2-3 quadrants (Q1-Q4).

    Q1: g > 0, u > 0 -> gate_open=True,  a > 0 (positive pole, gate open)
    Q2: g > 0, u < 0 -> gate_open=True,  a < 0 (negative pole, gate open)
    Q3: g < 0, u < 0 -> gate_open=False, a > 0 (positive pole, double-negative)
    Q4: g < 0, u > 0 -> gate_open=False, a < 0 (negative pole, gate closed)
    """
    if gate_open:
        return "Q1" if act >= 0.0 else "Q2"
    return "Q3" if act >= 0.0 else "Q4"


def format_token_only_label(top_tokens: Sequence[str]) -> str:
    """Format deterministic token-only description (no 'strongly')."""
    formatted = ", ".join(json.dumps(tok, ensure_ascii=False) for tok in top_tokens)
    return f"Tokens such as: {formatted}"


def ensure_dirs(base_dir: Path | None = None) -> dict[str, Path]:
    """Create and return standard project directories under base_dir (default REPO_ROOT)."""
    root = base_dir if base_dir is not None else REPO_ROOT
    dirs = {
        "root": root,
        "data": root / "data",
        "cache": root / "data" / "cache",
        "results": root / "results",
        "figures": root / "figures",
        "audit": root / "audit",
    }
    for key in ("data", "cache", "results", "figures", "audit"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


@dataclass(frozen=True)
class WindowSpec:
    before: int = WINDOW_BEFORE
    after: int = WINDOW_AFTER
    marker_open: str = MARKER_OPEN
    marker_close: str = MARKER_CLOSE

    @property
    def total(self) -> int:
        return self.before + 1 + self.after

    @property
    def target_offset(self) -> int:
        return self.before

    def to_dict(self) -> dict[str, int | str]:
        return {
            "before": self.before,
            "after": self.after,
            "total": self.total,
            "marker_open": self.marker_open,
            "marker_close": self.marker_close,
        }
