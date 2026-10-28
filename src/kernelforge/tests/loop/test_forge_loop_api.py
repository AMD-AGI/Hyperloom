# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a caller launches forge-loop and reads the result it publishes."""

from __future__ import annotations

import sys

import pytest

from kernelforge.loop import api


def test_the_command_runs_this_interpreter_and_package():
    assert api.forge_loop_command("--max-hours", "1") == [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "forge-loop",
        "--max-hours",
        "1",
    ]


def test_the_command_falls_back_to_the_console_script(monkeypatch):
    monkeypatch.setattr(api.sys, "executable", "")
    monkeypatch.setattr(api.shutil, "which", lambda name: "/usr/bin/kernelforge")

    assert api.forge_loop_command() == ["/usr/bin/kernelforge", "forge-loop"]


def test_a_published_result_reads_back_from_mixed_output():
    published = api.format_forge_loop_result({"improved": True, "best_iteration": 3})
    output = f"  [agent] working\n{published}\n  trailing log\n"

    assert api.parse_forge_loop_result(output) == {"improved": True, "best_iteration": 3}


def test_the_last_published_result_wins():
    first = api.format_forge_loop_result({"improved": False})
    last = api.format_forge_loop_result({"improved": True})

    assert api.parse_forge_loop_result(f"{first}\n{last}\n") == {"improved": True}


@pytest.mark.parametrize(
    "output",
    [
        "no result here",
        f'{api.RESULT_SENTINEL}{{"improved": true}}',
    ],
)
def test_output_without_a_complete_result_has_none_and_says_nothing(output, caplog):
    with caplog.at_level("WARNING", logger=api.__name__):
        assert api.parse_forge_loop_result(output) is None

    assert caplog.records == []


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "JSON list, not an object"),
    ],
)
def test_a_malformed_result_has_none_and_is_reported(payload, reason, caplog):
    output = f"{api.RESULT_SENTINEL}{payload}{api.RESULT_SENTINEL}"

    with caplog.at_level("WARNING", logger=api.__name__):
        assert api.parse_forge_loop_result(output) is None

    assert reason in caplog.text
