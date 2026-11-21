# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lane of a workspace that is imported as a package imports itself by that package's name."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kernelforge.tests.utils.test_lane_aiter_cache import (
    _campaign,
    _implementer_through,
    _lane_factory,
    _recording_claude_backend,
    _run_lanes,
)

_PRINT_VALUE = "from served.src import kernel; print(kernel.VALUE)"


def _value_imported_by(env: dict[str, str]) -> str:
    """What ``served.src.kernel`` holds for a process started with ``env``."""
    return subprocess.run(
        [sys.executable, "-c", _PRINT_VALUE], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


async def test_a_lane_session_imports_the_lane_by_the_name_its_workspace_is_imported_as(tmp_path, monkeypatch) -> None:
    """Otherwise a driver importing the package by name times the canonical workspace in every lane."""
    config, workspace = _campaign(tmp_path)
    import_root = tmp_path / "imports"
    import_root.mkdir()
    (import_root / "served").symlink_to(workspace, target_is_directory=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(import_root), str(elsewhere))))
    spawned: list[dict[str, str]] = []

    await _run_lanes(
        tmp_path,
        _lane_factory(config, workspace, _implementer_through(_recording_claude_backend(spawned))),
        ("1", "2"),
    )

    for lane_id, value in (("1", "VALUE = 1\n"), ("2", "VALUE = 2\n")):
        (tmp_path / "lanes" / lane_id / "src" / "kernel.py").write_text(value)
    assert sorted(_value_imported_by(env) for env in spawned) == ["1", "2"]
    assert _value_imported_by(dict(os.environ)) == "0"
    assert all(env["PYTHONPATH"].split(os.pathsep)[1:] == [str(import_root), str(elsewhere)] for env in spawned)
    # Beside the lane copies, where the round's fan-out removes them, and not inside one, where it would be an edit.
    assert {Path(env["PYTHONPATH"].split(os.pathsep)[0]).parent for env in spawned} == {tmp_path / "lanes"}


async def test_a_lane_of_a_workspace_no_import_path_names_inherits_the_path(tmp_path, monkeypatch) -> None:
    config, workspace = _campaign(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    spawned: list[dict[str, str]] = []

    await _run_lanes(
        tmp_path,
        _lane_factory(config, workspace, _implementer_through(_recording_claude_backend(spawned))),
        ("1",),
    )

    assert [env["PYTHONPATH"] for env in spawned] == [str(tmp_path)]
