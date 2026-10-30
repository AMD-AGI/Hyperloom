# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace guard enforces the protected-path rules its caller supplies, and none of its own."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
from kernelforge.agent_backends.session_resume import is_retryable_api_error
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError
from kernelforge.utils.protected_paths import ProtectedPathRules


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Forge Test")
    _git(tmp_path, "config", "user.email", "forge@example.invalid")
    for name in ("kernel.py", "config.yaml", "secret.txt"):
        (tmp_path / name).write_text(f"{name}\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "baseline")
    return tmp_path


def _writable(repo: Path, rules: ProtectedPathRules | None) -> AgentRunSpec:
    return AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(repo),
        target_files=["kernel.py"],
        protected_path_rules=rules,
    )


def test_a_writable_session_without_rules_is_refused_before_it_starts(repo: Path) -> None:
    guard = WorkspaceGuard(_writable(repo, None))

    with pytest.raises(WorkspaceSafetyError, match="protected-path rules") as refused:
        guard.prepare()

    assert not guard.prepared
    # The caller's omission, not the agent's doing, and no retry can supply what the caller left out.
    assert refused.value.agent_safety_rejection is False
    assert not is_retryable_api_error(refused.value)


def test_a_read_only_session_needs_no_rules(repo: Path) -> None:
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(repo),
        writable=False,
        tool_policy=AgentToolPolicy(read=True, search=True, write=False, shell=False),
    )
    guard = WorkspaceGuard(spec)

    guard.prepare()

    assert guard.skipped


def test_the_guard_protects_what_the_callers_rules_name_and_nothing_else(repo: Path) -> None:
    guard = WorkspaceGuard(_writable(repo, ProtectedPathRules(globs=("secret.txt",))))
    guard.prepare()
    (repo / "secret.txt").write_text("changed\n")

    with pytest.raises(WorkspaceSafetyError, match="protected tracked files changed: secret.txt"):
        guard.verify()
    assert (repo / "secret.txt").read_text() == "secret.txt\n"

    # config.yaml is on Forge's list, not this caller's: the guard carries no list of its own.
    guard = WorkspaceGuard(_writable(repo, ProtectedPathRules(globs=("secret.txt",))))
    guard.prepare()
    (repo / "config.yaml").write_text("changed\n")

    assert guard.verify() == ["config.yaml"]
