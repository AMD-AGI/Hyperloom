# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the unified ``hyperloom <subcommand>`` entry point."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hyperloom import cli


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    recorded: list[tuple[str, list[str]]] = []

    def fake_import(name: str) -> types.SimpleNamespace:
        def main(argv: list[str]) -> int:
            recorded.append((name, list(argv)))
            return 0

        return types.SimpleNamespace(main=main)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return recorded


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["setup", "--check-only"], ("hyperloom.inference_optimizer.setup", ["--check-only"])),
        (["check", "/models/x"], ("hyperloom.inference_optimizer.tools.preflight_optimizer", ["/models/x"])),
        (["optimize", "--model", "m"], ("hyperloom.inference_optimizer.cli", ["optimize", "--model", "m"])),
        (["recover", "--session-dir", "s"], ("hyperloom.inference_optimizer.cli", ["recover", "--session-dir", "s"])),
        (["quantize", "--workspace", "w"], ("hyperloom.agents.quantization.cli", ["--workspace", "w"])),
        (["multi-node", "verify"], ("hyperloom.inference_optimizer.multi_node.cli", ["verify"])),
        (
            ["session", "breakdown", "--session-dir", "s"],
            ("hyperloom.inference_optimizer.tools.dump_session_breakdown", ["--session-dir", "s"]),
        ),
        (
            ["session", "report", "-i", "b.json"],
            ("hyperloom.inference_optimizer.tools.dump_session_report", ["-i", "b.json"]),
        ),
        (
            ["session", "backfill", "--session-dir", "s"],
            ("hyperloom.inference_optimizer.tools.backfill_langfuse", ["--session-dir", "s"]),
        ),
        (["session", "events", "s"], ("hyperloom.inference_optimizer.tools.event_counts", ["s"])),
        (["session", "state", "s"], ("hyperloom.inference_optimizer.tools.read_optimizer_state", ["s"])),
    ],
)
def test_forwards_remaining_argv(
    calls: list[tuple[str, list[str]]], argv: list[str], expected: tuple[str, list[str]]
) -> None:
    assert cli.main(argv) == 0
    assert calls == [expected]


@pytest.mark.parametrize("argv", [[], ["--check-only"], ["nope"], ["session"], ["session", "nope"]])
def test_unknown_or_missing_subcommand_prints_usage(
    calls: list[tuple[str, list[str]]], argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2
    assert calls == []
    assert "usage: hyperloom" in capsys.readouterr().err


def test_help_lists_commands(calls: list[tuple[str, list[str]]], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("setup", "check", "optimize", "recover", "quantize", "session", "multi-node"):
        assert f"  {name}" in out


def test_python_dash_m_hyperloom_runs_the_dispatcher() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "hyperloom", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).resolve().parents[1])},
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("usage: hyperloom <command>")


def test_inference_optimizer_cli_package_is_not_runnable_with_dash_m() -> None:
    src = Path(cli.__file__).resolve().parents[1]
    cli_pkg = src / "hyperloom" / "inference_optimizer" / "cli"
    assert not (cli_pkg / "__main__.py").is_file()
    proc = subprocess.run(
        [sys.executable, "-m", "hyperloom.inference_optimizer.cli", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        check=False,
    )
    assert proc.returncode != 0
    assert "__main__" in proc.stderr.lower()
