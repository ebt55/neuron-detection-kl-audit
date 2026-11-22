"""Unified CLI entrypoint (`w-audit`) for two-snippet neuron detection audit.

Subcommands:
  probe      Check live OpenAI rate limits, logprobs support, and 0%-cache ETA/cost.
  extract    Stream Qwen3-1.7B MLP activations, gate signs, and neighbour stats.
  snippets   Cut symmetric 69-token marked snippets and build evaluation pairs.
  baselines  Compute explanation-free rules (rarity, top_token) + Chen checks.
  label      Generate gpt-4.1-mini pole labels and build all 4 label conditions.
  judge      Run 1-pair-per-call OpenAI or Ollama (think=false) detection sweeps.
  analyze    Compute neuron-clustered CIs, binary accuracy slices, figure, tables, and audit sheet.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "probe": (
        "w_audit.probe_limits",
        "Probe live OpenAI rate limits, logprobs support, and 0%-cache ETA/cost.",
    ),
    "extract": (
        "w_audit.extract",
        "Stream Qwen3-1.7B activations, gate signs, and nearest same-layer neighbours.",
    ),
    "snippets": (
        "w_audit.snippets",
        "Cut symmetric 69-token marked snippets and build evaluation pairs.",
    ),
    "baselines": (
        "w_audit.baselines",
        "Run explanation-free no-LLM rules (rarity, top_token) and Chen structural checks.",
    ),
    "label": (
        "w_audit.labeler",
        "Generate gpt-4.1-mini labels and construct own/wrong/no_label/token_only arms.",
    ),
    "judge": (
        "w_audit.judge",
        "Run 1-pair-per-call detection judge sweeps (OpenAI or Ollama with think=false).",
    ),
    "analyze": (
        "w_audit.analyze",
        "Compute neuron-clustered bootstrap CIs, binary accuracy slices, figure, tables, and 30-pole worksheet.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    epilog_lines = ["Available subcommands:"]
    for name, (_, desc) in SUBCOMMANDS.items():
        epilog_lines.append(f"  {name:10s}  {desc}")
    epilog_lines.append("")
    epilog_lines.append("Run `w-audit <subcommand> --help` for subcommand options.")

    parser = argparse.ArgumentParser(
        prog="w-audit",
        description="Audit of two-snippet neuron detection on Qwen3-1.7B.",
        epilog="\n".join(epilog_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "command",
        choices=list(SUBCOMMANDS.keys()),
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded directly to the selected subcommand.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    cmd = raw_argv[0]
    rest = raw_argv[1:]
    if cmd not in SUBCOMMANDS:
        build_parser().error(
            f"Unknown subcommand {cmd!r}. Choose from: {', '.join(SUBCOMMANDS.keys())}"
        )

    module_name, _ = SUBCOMMANDS[cmd]
    mod = importlib.import_module(module_name)
    ret = mod.main(rest)
    return int(ret) if isinstance(ret, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
