"""Exploratory Qwen/Qwen3-4B (Ulto85/neurons-kinda-clean) extraction & evaluation package.

Strictly isolated from the Qwen3-1.7B evaluation codebase (`src/w_audit/*`,
root `data/`, root `results/`, and `figures/main_band_accuracy.png`).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Isolated Paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_4B_DIR: Path = REPO_ROOT / "data" / "qwen4b"
ULTO85_DIR: Path = DATA_4B_DIR / "ulto85"
CACHE_4B_DIR: Path = DATA_4B_DIR / "cache"
RESULTS_4B_DIR: Path = REPO_ROOT / "results" / "qwen4b"

ACTIVE_POLES_PATH: Path = DATA_4B_DIR / "active_poles.json"
SNIPPETS_4B_PATH: Path = DATA_4B_DIR / "snippets_4b.jsonl"
PARITY_CHECK_PATH: Path = RESULTS_4B_DIR / "parity_check.json"
STRUCTURAL_CHECKS_4B_PATH: Path = RESULTS_4B_DIR / "structural_checks_4b.json"

# ---------------------------------------------------------------------------
# Locked Exploratory Qwen3-4B Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION_4B: str = "1.0"
SEED_4B: int = 42

MODEL_ID_4B: str = "Qwen/Qwen3-4B"
HOOK_LAYERS_4B: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28, 32, 35)
SPLIT_LAYER_IDX_4B: int = 18  # Half 1: layers 0..17, Half 2: layers 18..35

FINEWEB_DATASET_ID: str = "HuggingFaceFW/fineweb"
FINEWEB_PARQUET_PATH: str = "sample/10BT/000_00000.parquet"
FINEWEB_MIN_ROW_IDX: int = 60_000  # Strictly disjoint from Ulto85 rows 0..60,000

SEQ_LEN_4B: int = 128
NUM_MACRO_CHUNKS_4B: int = 24
SEQS_PER_MACRO_CHUNK_4B: int = 500  # 500 * 128 = 64,000 tokens/chunk; 24 * 64,000 = 1,536,000
TOTAL_TOKENS_4B: int = NUM_MACRO_CHUNKS_4B * SEQS_PER_MACRO_CHUNK_4B * SEQ_LEN_4B  # 1,536,000

PAIRS_P999_PER_POLE: int = 20
PAIRS_FLOOR6_PER_POLE: int = 6
PAIRS_P97_99_PER_POLE: int = 10

ULTO85_BASE_URL: str = (
    "https://raw.githubusercontent.com/Ulto85/neurons-kinda-clean/main/results/"
)
ULTO85_FILES: tuple[str, ...] = (
    "e3_neurons.json",
    "e3_explanations.json",
    "e3_scores.json",
    "e3_scores_opus.json",
)

NEGLIGIBLE_PREFIXES: tuple[str, ...] = (
    "negligible",
    "none",
    "weak",
    "no ",
    "minimal",
    "virtually",
)

# Canonical Ulto85 named neurons verified from e3_scores.json / e3_explanations.json
NAMED_NEURONS_4B: dict[str, dict[str, int]] = {
    "info": {"k": 26, "layer": 0, "neuron": 9209},
    "PCR": {"k": 23, "layer": 0, "neuron": 8146},
    "insurance": {"k": 236, "layer": 32, "neuron": 4362},
}


def make_neuron_id_4b(layer: int, neuron: int) -> str:
    """Return canonical neuron ID, e.g. 'L00_N9209' or 'L32_N4362'."""
    return f"L{int(layer):02d}_N{int(neuron):04d}"


def make_pole_id_4b(layer: int, neuron: int, sign: int) -> str:
    """Return canonical pole ID, e.g. 'L00_N9209_pos' or 'L32_N4362_neg'."""
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")
    pole_name = "pos" if sign > 0 else "neg"
    return f"L{int(layer):02d}_N{int(neuron):04d}_{pole_name}"


def ensure_4b_dirs(base_dir: Path | None = None) -> dict[str, Path]:
    """Create and return isolated directories for Qwen3-4B artifacts."""
    root = base_dir if base_dir is not None else REPO_ROOT
    dirs = {
        "root": root,
        "data": root / "data" / "qwen4b",
        "ulto85": root / "data" / "qwen4b" / "ulto85",
        "cache": root / "data" / "qwen4b" / "cache",
        "results": root / "results" / "qwen4b",
    }
    for key in ("data", "ulto85", "cache", "results"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs
