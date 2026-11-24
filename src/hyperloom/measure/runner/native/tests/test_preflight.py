# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The torch-free preflight of a native recipe: what the machine lacks is the machine's problem, what it has is recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.runner.native.preflight import native_preflight


def _tool(directory: Path, name: str, body: str) -> str:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[str, ...]]:
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "0")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    return {
        "cmake": (_tool(bin_dir, "cmake", 'echo "cmake version 3.28.3"; echo "CMake suite"'), "--version"),
        "hip_clang": (_tool(bin_dir, "clang++", 'echo; echo "AMD clang version 24.0.0git"'), "--version"),
    }


def test_each_tool_is_recorded_by_the_first_line_it_prints(tools: dict, tmp_path: Path) -> None:
    checked = native_preflight(tools, 0.0, tmp_path / "scratch", "gfx1100")

    assert checked.problem is None
    assert dict(checked.toolchain) == {"cmake": "cmake version 3.28.3", "hip_clang": "AMD clang version 24.0.0git"}


@pytest.mark.parametrize("script", [None, "exit 3"], ids=["not-there", "fails-its-version"])
def test_a_tool_that_does_not_run_is_the_machines_problem(tools: dict, tmp_path: Path, script: str | None) -> None:
    ninja = _tool(tmp_path, "ninja", script) if script else "no-such-ninja-tool"

    checked = native_preflight({**tools, "ninja": (ninja, "--version")}, 0.0, tmp_path, "gfx1100")

    assert checked.problem == f"preflight_toolchain: ninja ({ninja}) does not run on this machine"
    assert checked.toolchain == {}


def test_too_little_disk_is_the_machines_problem(tools: dict, tmp_path: Path) -> None:
    checked = native_preflight(tools, 1e9, tmp_path, "gfx1100")

    assert checked.problem is not None and checked.problem.startswith("preflight_disk: insufficient disk")
