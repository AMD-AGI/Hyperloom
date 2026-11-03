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
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS
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


def test_changed_bytecode_under_a_protected_dir_is_deleted_and_a_changed_source_still_reported(repo: Path) -> None:
    tests = repo / "test"
    tests.mkdir()
    (tests / "layer_ut_utils.py").write_text("VALUE = 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "test sources")
    cache = tests / "__pycache__"
    cache.mkdir()
    rewritten, untouched, created = (
        cache / name
        for name in ("layer_ut_utils.cpython-312.pyc", "helpers.cpython-312.pyc", "conftest.cpython-312.pyc")
    )
    rewritten.write_bytes(b"stale")
    untouched.write_bytes(b"cached")

    guard = WorkspaceGuard(_writable(repo, FORGE_PROTECTED_PATHS))
    guard.prepare()
    rewritten.write_bytes(b"forged")
    created.write_bytes(b"new")

    assert guard.verify() == []
    assert not rewritten.exists() and not created.exists()
    assert untouched.read_bytes() == b"cached"

    guard = WorkspaceGuard(_writable(repo, FORGE_PROTECTED_PATHS))
    guard.prepare()
    (tests / "layer_ut_utils.py").write_text("VALUE = 2\n")

    with pytest.raises(WorkspaceSafetyError, match="protected tracked files changed: test/layer_ut_utils.py"):
        guard.verify()
    assert (tests / "layer_ut_utils.py").read_text() == "VALUE = 1\n"

    # Only the interpreter's cache is rebuilt from source; a .pyc beside the sources may be the module itself.
    sourceless = tests / "legacy.pyc"
    sourceless.write_bytes(b"module")
    guard = WorkspaceGuard(_writable(repo, FORGE_PROTECTED_PATHS))
    guard.prepare()
    sourceless.write_bytes(b"forged")

    with pytest.raises(WorkspaceSafetyError, match="protected ignored files changed: .*legacy.pyc"):
        guard.verify()
    assert sourceless.read_bytes() == b"module"


def test_bytecode_behind_a_linked_cache_dir_is_reported_not_deleted(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tests = repo / "test"
    tests.mkdir()
    (tests / "layer_ut_utils.py").write_text("VALUE = 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "test sources")
    (tests / "__pycache__").mkdir()
    (tests / "__pycache__" / "layer_ut_utils.cpython-312.pyc").write_bytes(b"stale")
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "__pycache__"
    elsewhere.mkdir()
    (elsewhere / "layer_ut_utils.cpython-312.pyc").write_bytes(b"not the session's")

    guard = WorkspaceGuard(_writable(repo, FORGE_PROTECTED_PATHS))
    guard.prepare()
    (tests / "__pycache__" / "layer_ut_utils.cpython-312.pyc").unlink()
    (tests / "__pycache__").rmdir()
    (tests / "__pycache__").symlink_to(elsewhere)

    with pytest.raises(WorkspaceSafetyError, match="protected ignored files changed: .*layer_ut_utils.cpython-312.pyc"):
        guard.verify()
    assert (elsewhere / "layer_ut_utils.cpython-312.pyc").read_bytes() == b"not the session's"
