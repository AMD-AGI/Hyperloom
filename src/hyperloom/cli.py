# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified ``hyperloom <command>`` entry point; forwards the remaining argv to each command's ``main``."""

from __future__ import annotations

import importlib
import sys

_IO = "hyperloom.inference_optimizer"

# command -> (module whose ``main(argv)`` runs it, argv prefix, one-line help)
_COMMANDS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "setup": (f"{_IO}.setup", (), "Install and configure Hyperloom on this host"),
    "check": (f"{_IO}.tools.preflight_optimizer", (), "Check host readiness for a model before optimizing"),
    "optimize": (f"{_IO}.cli", ("optimize",), "Run a multi-agent optimization session"),
    "recover": (f"{_IO}.cli", ("recover",), "Rebuild the artifacts of a crashed session"),
    "quantize": ("hyperloom.agents.quantization.cli", (), "Quantize a model with AMD Quark"),
    "multi-node": (f"{_IO}.multi_node.cli", (), "Operate the multi-node cluster of a session"),
}

_SESSION_COMMANDS: dict[str, tuple[str, str]] = {
    "breakdown": (f"{_IO}.tools.dump_session_breakdown", "Write session_breakdown.json for a session"),
    "report": (f"{_IO}.tools.dump_session_report", "Render a session breakdown as markdown"),
    "backfill": (f"{_IO}.tools.backfill_langfuse", "Replay a session's trace into Langfuse"),
    "events": (f"{_IO}.tools.event_counts", "Count coordinator events of a session"),
    "state": (f"{_IO}.tools.read_optimizer_state", "Summarize a session's state.json"),
}


def _usage() -> str:
    lines = ["usage: hyperloom <command> [args...]", "", "commands:"]
    lines += [f"  {name:<12}{help_}" for name, (_, _, help_) in _COMMANDS.items()]
    lines.append(f"  {'session':<12}Inspect a session: {', '.join(_SESSION_COMMANDS)}")
    lines += ["", "Run 'hyperloom <command> --help' for the options of a command."]
    return "\n".join(lines)


def _session_usage() -> str:
    lines = ["usage: hyperloom session <command> [args...]", "", "commands:"]
    lines += [f"  {name:<12}{help_}" for name, (_, help_) in _SESSION_COMMANDS.items()]
    return "\n".join(lines)


def _run(module: str, argv: list[str]) -> int:
    return importlib.import_module(module).main(argv)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(_usage())
        return 0
    if not args or (args[0] not in _COMMANDS and args[0] != "session"):
        print(_usage(), file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    if command != "session":
        module, prefix, _ = _COMMANDS[command]
        return _run(module, [*prefix, *rest])
    if rest and rest[0] in ("-h", "--help"):
        print(_session_usage())
        return 0
    if not rest or rest[0] not in _SESSION_COMMANDS:
        print(_usage(), file=sys.stderr)
        return 2
    return _run(_SESSION_COMMANDS[rest[0]][0], rest[1:])
