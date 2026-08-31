# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for framework git/subprocess helpers: rev-parse error+success
branches, repo-id normalization, and same-repo gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _git as gitmod
from hyperloom.orchestrator.actions.executors import _patch_source_pr as fp


class _CP:
    """Minimal CompletedProcess stand-in."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _seq_runner(results: list[Any]):
    """Build a subprocess.run replacement that returns/raises queued items."""
    calls: list[list[str]] = []

    def _run(cmd, *a, **k):
        calls.append(list(cmd))
        item = results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def test_git_head_sha_success(monkeypatch) -> None:
    monkeypatch.setattr(gitmod.subprocess, "run", _seq_runner([_CP(0, "deadbeef\n")]))
    sha, err = fp._git_head_sha(Path("/repo"))
    assert sha == "deadbeef" and err == ""


def test_git_head_sha_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(gitmod.subprocess, "run", _seq_runner([FileNotFoundError("git")]))
    sha, err = fp._git_head_sha(Path("/repo"))
    assert sha is None and "spawn failed" in err


def test_git_head_sha_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(gitmod.subprocess, "run", _seq_runner([_CP(1, "", "fatal: no head")]))
    sha, err = fp._git_head_sha(Path("/repo"))
    assert sha is None and err == "fatal: no head"


def test_run_git_success(monkeypatch) -> None:
    monkeypatch.setattr(gitmod.subprocess, "run", _seq_runner([_CP(0, "out", "")]))
    ok, out, err = fp._run_git(["status"])
    assert ok is True and out == "out"


def test_run_git_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(gitmod.subprocess, "run", _seq_runner([FileNotFoundError("git")]))
    ok, out, err = fp._run_git(["status"])
    assert ok is False and "spawn/timeout failed" in err


def test_run_git_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(gitmod.subprocess, "run", _seq_runner([_CP(2, "partial", "boom")]))
    ok, out, err = fp._run_git(["status"])
    assert ok is False and out == "partial" and err == "boom"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ""),
        ("Owner/Name", "owner/name"),
        ("https://github.com/Owner/Name.git", "owner/name"),
        ("git@github.com:Owner/Name.git", "owner/name"),
        ("https://github.com/Owner/Name/", "owner/name"),
        ("https://example.com/a/b/c/d", "c/d"),
        ("single", "single"),
    ],
)
def test_normalize_repo_id(raw: str, expected: str) -> None:
    assert fp._normalize_repo_id(raw) == expected


def test_candidate_is_same_repo_no_candidate_repo() -> None:
    assert fp._candidate_is_same_repo({}, Path("/repo")) is True
    assert fp._candidate_is_same_repo({"repo": "noslash"}, Path("/repo")) is True


def test_candidate_is_same_repo_origin_unreadable(monkeypatch) -> None:
    monkeypatch.setattr(fp, "_run_git", lambda *a, **k: (False, "", "no origin"))
    assert fp._candidate_is_same_repo({"repo": "Owner/Name"}, Path("/repo")) is True


def test_candidate_is_same_repo_non_github_origin(monkeypatch) -> None:
    monkeypatch.setattr(fp, "_run_git", lambda *a, **k: (True, "/local/path/repo\n", ""))
    assert fp._candidate_is_same_repo({"repo": "Owner/Name"}, Path("/repo")) is True


def test_candidate_is_same_repo_matching_github(monkeypatch) -> None:
    monkeypatch.setattr(
        fp,
        "_run_git",
        lambda *a, **k: (True, "https://github.com/Owner/Name.git\n", ""),
    )
    assert fp._candidate_is_same_repo({"repo": "Owner/Name"}, Path("/repo")) is True


def test_candidate_is_same_repo_differing_github(monkeypatch) -> None:
    monkeypatch.setattr(
        fp,
        "_run_git",
        lambda *a, **k: (True, "https://github.com/Other/Repo.git\n", ""),
    )
    assert fp._candidate_is_same_repo({"repo": "Owner/Name"}, Path("/repo")) is False
