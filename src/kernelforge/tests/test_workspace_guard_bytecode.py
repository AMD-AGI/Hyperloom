# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bytecode a session compiles beside a protected module is put back, never kept and never a rejection."""

import subprocess
from pathlib import Path

import pytest

from kernelforge.agent_backends.base import AgentRunSpec
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError

_BYTECODE = "scripts/__pycache__/task_measure.cpython-312.pyc"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def spec(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Forge Test")
    _git(tmp_path, "config", "user.email", "forge@example.invalid")
    (tmp_path / "kernel.py").write_text("VALUE = 1\n")
    (tmp_path / "driver.py").write_text("ORACLE = 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "task_measure.py").write_text("MEASURE = 1\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n*.dat\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "baseline")
    return AgentRunSpec(
        system_prompt="", user_prompt="", cwd=str(tmp_path), target_files=["kernel.py"], driver_script="driver.py"
    )


def test_bytecode_a_session_compiles_is_removed_and_the_turn_is_kept(spec):
    root = Path(spec.cwd)
    guard = WorkspaceGuard(spec)
    guard.prepare()
    (root / "kernel.py").write_text("VALUE = 2\n")
    (root / _BYTECODE).parent.mkdir()
    (root / _BYTECODE).write_bytes(b"compiled by running the driver")

    assert guard.verify() == ["kernel.py"]
    assert not (root / "scripts" / "__pycache__").exists()
    assert (root / "kernel.py").read_text() == "VALUE = 2\n"


def test_bytecode_a_session_rewrites_gets_its_snapshot_back(spec):
    root = Path(spec.cwd)
    (root / _BYTECODE).parent.mkdir()
    (root / _BYTECODE).write_bytes(b"compiled before the turn")
    guard = WorkspaceGuard(spec)
    guard.prepare()
    (root / _BYTECODE).write_bytes(b"planted by the session")

    assert guard.verify() == []
    assert (root / _BYTECODE).read_bytes() == b"compiled before the turn"


def test_any_other_protected_ignored_file_still_rejects_the_turn(spec):
    root = Path(spec.cwd)
    guard = WorkspaceGuard(spec)
    guard.prepare()
    (root / "scripts" / "reference.dat").write_bytes(b"a different oracle")

    with pytest.raises(WorkspaceSafetyError, match="protected ignored files changed"):
        guard.verify()
    assert not (root / "scripts" / "reference.dat").exists()
