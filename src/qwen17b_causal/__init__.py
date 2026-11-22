"""Isolated Qwen/Qwen3-1.7B causal simulation & KL validation package (n = 50 neurons / 100 poles).

Strictly read-only on existing Qwen3-1.7B (`data/*` root, `results/*` root, `src/w_audit/*`,
frozen labels) and Qwen3-4B (`data/qwen4b/*`, `results/qwen4b/*`) artifacts. All outputs live
strictly under `data/qwen17b_causal/` and `results/qwen17b_causal/`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_17B_CAUSAL_DIR: Path = REPO_ROOT / "data" / "qwen17b_causal"
RESULTS_17B_CAUSAL_DIR: Path = REPO_ROOT / "results" / "qwen17b_causal"

MODEL_ID_17B: str = "Qwen/Qwen3-1.7B"
LAYERS_17B: tuple[int, ...] = (4, 11, 18, 24)
SEED_17B: int = 42

E3_ORDINARY_N_WINS_17B: int = 128
E3_ORDINARY_WIN_LEN_17B: int = 64  # 128 * 64 = 8,192 tokens
SIM_N_WINS_PER_NEURON_17B: int = 64  # 32 ordinary + 32 active-stratified
SIM_WIN_LEN_17B: int = 64
SIM_ORD_WINS_17B: int = 32
SIM_ACT_WINS_17B: int = 32

FPR_ORD_CORE_17B: int = 10
FPR_ORD_EXTRA_17B: int = 10
FPR_TOP_CORE_17B: int = 2
FPR_TOP_EXTRA_17B: int = 2


def ensure_17b_causal_dirs(base_dir: Path | None = None) -> dict[str, Path]:
    """Create and return isolated directories for Qwen3-1.7B causal simulation & KL validation."""
    root = base_dir if base_dir is not None else REPO_ROOT
    dirs = {
        "root": root,
        "data_root": root / "data",
        "cache_root": root / "data" / "cache",
        "data": root / "data" / "qwen17b_causal",
        "results": root / "results" / "qwen17b_causal",
    }
    for key in ("data", "results"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs
