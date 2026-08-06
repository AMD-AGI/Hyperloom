# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every optimizer flag a shipped launcher builds must still be accepted by the CLI.

The launchers assemble the ``optimize`` command line as shell text, so a removed
flag surfaces only when a real run reaches argparse. These tests parse the
command out of each asset and check it against the live parser.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer"

LAUNCHERS = (
    PACKAGE_ROOT / "assets" / "slurm" / "_incontainer.sh.in",
    PACKAGE_ROOT / "assets" / "slurm" / "run_hyperloom.sbatch",
    PACKAGE_ROOT / "tools" / "robustness_monitor.sh.example",
)

_CLI_MODULE = "hyperloom.inference_optimizer.cli"
_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")


def _optimizer_commands(text: str) -> list[str]:
    """Return each ``optimize`` invocation, backslash continuations joined."""
    lines = text.splitlines()
    commands: list[str] = []
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
            if " optimize" in joined:
                commands.append(joined)
        idx += 1
    return commands


def _known_optimize_flags() -> set[str]:
    parser = _build_parser()
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert subparsers, "CLI parser exposes no subcommands"
    optimize = subparsers[0].choices["optimize"]
    flags = {opt for action in optimize._actions for opt in action.option_strings}
    flags |= {opt for action in parser._actions for opt in action.option_strings}
    return flags


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_optimizer_flags_are_accepted_by_the_cli(launcher: Path) -> None:
    known = _known_optimize_flags()
    for command in _optimizer_commands(launcher.read_text(encoding="utf-8")):
        for flag in _FLAG_RE.findall(command):
            assert flag in known, f"{launcher.name} passes {flag}, which the optimize CLI does not accept"


def _flag_sets(launcher: Path) -> list[set[str]]:
    """Flags per invocation, dropping the bare module references that carry none."""
    found = [set(_FLAG_RE.findall(cmd)) for cmd in _optimizer_commands(launcher.read_text(encoding="utf-8"))]
    return [flags for flags in found if flags]


def test_launcher_scan_finds_the_commands_it_is_meant_to_guard():
    """A scanner that silently matches nothing would pass the check above vacuously."""
    incontainer, _sbatch, monitor = LAUNCHERS

    # The python carrier execs the optimizer; the claude carrier embeds the same
    # command in its prompt. Both must stay in sync with the parser.
    incontainer_flags = _flag_sets(incontainer)
    assert len(incontainer_flags) == 2
    for flags in incontainer_flags:
        assert {"--model", "--framework", "--launch-info-file"} <= flags

    monitor_flags = _flag_sets(monitor)
    assert monitor_flags, "monitor resume command not found"
    assert any({"--resume", "--resume-from"} <= flags for flags in monitor_flags)


def test_retired_kernel_backend_flags_are_gone_from_every_launcher():
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        for retired in ("--kernel-codex", "--kernel-claude", "--kernel-prompt", "HL_KERNEL_BACKEND"):
            assert retired not in text, f"{launcher.name} still references retired {retired}"
