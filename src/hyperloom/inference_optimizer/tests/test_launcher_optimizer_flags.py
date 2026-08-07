# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every optimizer flag a shipped launcher builds must still be accepted by the CLI.

The launchers assemble the command as shell text, so a flag removed from argparse
stays invisible until a real run reaches it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer"

# (asset, invocations carrying flags); the count stops a scanner that matches
# nothing from passing the flag check vacuously.
LAUNCHERS = (
    (PACKAGE_ROOT / "assets" / "slurm" / "_incontainer.sh.in", 2),
    (PACKAGE_ROOT / "assets" / "slurm" / "run_hyperloom.sbatch", 0),
    (PACKAGE_ROOT / "tools" / "robustness_monitor.sh.example", 2),
)

_CLI_MODULE = "hyperloom.inference_optimizer.cli"
_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")


def _optimizer_flag_sets(text: str) -> list[set[str]]:
    """Flags per ``optimize`` invocation, backslash continuations joined."""
    lines = text.splitlines()
    found: list[set[str]] = []
    idx = 0
    while idx < len(lines):
        if _CLI_MODULE in lines[idx]:
            block: list[str] = []
            while idx < len(lines):
                block.append(lines[idx])
                if not lines[idx].rstrip().endswith("\\"):
                    break
                idx += 1
            joined = " ".join(block)
            flags = set(_FLAG_RE.findall(joined))
            if " optimize" in joined and flags:
                found.append(flags)
        idx += 1
    return found


def _known_optimize_flags() -> set[str]:
    parser = _build_parser()
    subcommands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    optimize = subcommands.choices["optimize"]
    return {opt for action in (*optimize._actions, *parser._actions) for opt in action.option_strings}


@pytest.mark.parametrize(("launcher", "expected"), LAUNCHERS, ids=lambda v: getattr(v, "name", v))
def test_launcher_optimizer_flags_are_accepted_by_the_cli(launcher: Path, expected: int) -> None:
    known = _known_optimize_flags()
    flag_sets = _optimizer_flag_sets(launcher.read_text(encoding="utf-8"))
    assert len(flag_sets) == expected
    for flags in flag_sets:
        for flag in sorted(flags):
            assert flag in known, f"{launcher.name} passes {flag}, which the optimize CLI does not accept"
