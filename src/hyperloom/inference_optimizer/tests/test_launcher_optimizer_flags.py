# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every optimizer flag a shipped launcher builds must still be accepted by the CLI.

The launchers assemble the command as shell text, so a flag removed from argparse
stays invisible until a real run reaches it.

Accepting the *name* is not the whole contract: a flag the launcher gives a value
to must be one argparse consumes a value for, or the value is left behind as a
stray positional and the run still exits 2 before it starts.
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

# Retired with the kernel LLM role. The parser keeps them as hidden no-ops for
# out-of-tree callers, so they are "known" flags and the acceptance check below
# can no longer notice a launcher reintroducing one.
RETIRED = ("--kernel-codex", "--kernel-claude", "--kernel-prompt", "HL_KERNEL_BACKEND")

_CLI_MODULE = "hyperloom.inference_optimizer.cli"
_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")
# Tokens that may follow a flag without being its value.
_SHELL_OPERATORS = frozenset({">", ">>", "<", "|", "&", "&&", "||", ";", "2>&1"})


def _optimizer_invocations(text: str) -> list[list[str]]:
    """Tokens per ``optimize`` invocation, backslash continuations joined."""
    lines = text.splitlines()
    found: list[list[str]] = []
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
            if " optimize" in joined and _FLAG_RE.search(joined):
                found.append(joined.replace("\\", " ").split())
        idx += 1
    return found


def _optimize_actions() -> dict[str, argparse.Action]:
    parser = _build_parser()
    subcommands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    optimize = subcommands.choices["optimize"]
    return {opt: action for action in (*optimize._actions, *parser._actions) for opt in action.option_strings}


def _non_value_tokens() -> frozenset[str]:
    """Tokens that follow a flag but belong to the shell or name the subcommand."""
    parser = _build_parser()
    subcommands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return frozenset(subcommands.choices) | _SHELL_OPERATORS


def _wants_a_value(action: argparse.Action) -> bool | None:
    """``True``/``False`` when the arity is fixed, ``None`` when either shape parses."""
    if action.nargs == 0:
        return False
    if action.nargs in ("?", "*"):
        return None
    return True


@pytest.mark.parametrize(("launcher", "expected"), LAUNCHERS, ids=lambda v: getattr(v, "name", v))
def test_launcher_optimizer_flags_are_accepted_by_the_cli(launcher: Path, expected: int) -> None:
    actions = _optimize_actions()
    invocations = _optimizer_invocations(launcher.read_text(encoding="utf-8"))
    assert len(invocations) == expected
    for tokens in invocations:
        for flag in sorted(set(_FLAG_RE.findall(" ".join(tokens)))):
            assert flag in actions, f"{launcher.name} passes {flag}, which the optimize CLI does not accept"


@pytest.mark.parametrize(("launcher", "expected"), LAUNCHERS, ids=lambda v: getattr(v, "name", v))
def test_launcher_flag_values_match_the_cli_arity(launcher: Path, expected: int) -> None:
    """A launcher that writes ``--flag value`` needs a flag argparse takes a value for.

    The name check above passes either way, so an arity drift exits 2 at every
    launch with the value reported as an unrecognized argument.
    """
    actions = _optimize_actions()
    non_values = _non_value_tokens()
    invocations = _optimizer_invocations(launcher.read_text(encoding="utf-8"))
    assert len(invocations) == expected
    for tokens in invocations:
        # Only whole tokens, so a flag quoted inside a pgrep pattern is skipped.
        for pos, token in enumerate(tokens):
            action = actions.get(token)
            wants = _wants_a_value(action) if action is not None else None
            if wants is None:
                continue
            nxt = tokens[pos + 1] if pos + 1 < len(tokens) else None
            given = nxt is not None and not nxt.startswith("-") and nxt not in non_values
            assert given == wants, (
                f"{launcher.name} passes {token} "
                f"{'with' if given else 'without'} a value, but the CLI takes "
                f"{'one' if wants else 'none'} — argparse exits 2 at launch"
            )


def test_retired_kernel_backend_flags_are_gone_from_every_launcher() -> None:
    for launcher, _expected in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        for retired in RETIRED:
            assert retired not in text, f"{launcher.name} still references retired {retired}"


def test_retired_kernel_flags_parse_as_no_ops() -> None:
    """The shims exist so an out-of-tree caller does not exit 2.

    ``--kernel-prompt`` took a path, so accepting the flag while rejecting its
    value would not be acceptance — argparse reports the path as an
    unrecognized argument and the container dies at startup.
    """
    args = _build_parser().parse_args(
        [
            "optimize",
            "--model",
            "/models/m",
            "--framework",
            "sglang",
            "--gpu-type",
            "mi355x",
            "--kernel-codex",
            "--kernel-claude",
            "--kernel-prompt",
            "/tmp/kernel_prompt.md",
        ]
    )
    assert args.kernel_prompt == "/tmp/kernel_prompt.md"
