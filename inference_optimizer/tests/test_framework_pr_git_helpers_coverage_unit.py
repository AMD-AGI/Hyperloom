# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for framework_pr git/subprocess helpers: rev-parse / reset-hard /
commit error+success branches, repo-id normalization, and same-repo gating."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import framework_pr as fp


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


# -- _git_head_sha ---------------------------------------------------------
def test_git_head_sha_success(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(0, "deadbeef\n")]))
    sha, err = fp._git_head_sha(Path("/repo"))
    assert sha == "deadbeef" and err == ""


def test_git_head_sha_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([FileNotFoundError("git")]))
    sha, err = fp._git_head_sha(Path("/repo"))
    assert sha is None and "spawn failed" in err


def test_git_head_sha_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(1, "", "fatal: no head")]))
    sha, err = fp._git_head_sha(Path("/repo"))
    assert sha is None and err == "fatal: no head"


# -- _git_reset_hard -------------------------------------------------------
def test_git_reset_hard_success(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(0), _CP(0)]))
    ok, err = fp._git_reset_hard(Path("/repo"), "sha")
    assert ok is True and err == ""


def test_git_reset_hard_reset_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        fp.subprocess, "run", _seq_runner([subprocess.TimeoutExpired("git", 60)]),
    )
    ok, err = fp._git_reset_hard(Path("/repo"), "sha")
    assert ok is False and "reset --hard spawn failed" in err


def test_git_reset_hard_reset_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(1, "", "bad sha")]))
    ok, err = fp._git_reset_hard(Path("/repo"), "sha")
    assert ok is False and err == "bad sha"


def test_git_reset_hard_clean_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        fp.subprocess, "run",
        _seq_runner([_CP(0), FileNotFoundError("git")]),
    )
    ok, err = fp._git_reset_hard(Path("/repo"), "sha")
    assert ok is False and "clean -fd spawn failed" in err


def test_git_reset_hard_clean_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(
        fp.subprocess, "run", _seq_runner([_CP(0), _CP(1, "", "clean failed")]),
    )
    ok, err = fp._git_reset_hard(Path("/repo"), "sha")
    assert ok is False and err == "clean failed"


# -- _git_commit_keep ------------------------------------------------------
def test_git_commit_keep_success(monkeypatch) -> None:
    # add -A ok, commit ok, rev-parse returns new sha
    monkeypatch.setattr(
        fp.subprocess, "run",
        _seq_runner([_CP(0), _CP(0), _CP(0, "newsha\n")]),
    )
    sha, err = fp._git_commit_keep(Path("/repo"), "msg")
    assert sha == "newsha" and err == ""


def test_git_commit_keep_add_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([FileNotFoundError("git")]))
    sha, err = fp._git_commit_keep(Path("/repo"), "msg")
    assert sha is None and "add -A spawn failed" in err


def test_git_commit_keep_add_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(1, "", "add failed")]))
    sha, err = fp._git_commit_keep(Path("/repo"), "msg")
    assert sha is None and err == "add failed"


def test_git_commit_keep_commit_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        fp.subprocess, "run",
        _seq_runner([_CP(0), subprocess.TimeoutExpired("git", 60)]),
    )
    sha, err = fp._git_commit_keep(Path("/repo"), "msg")
    assert sha is None and "commit spawn failed" in err


def test_git_commit_keep_commit_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(
        fp.subprocess, "run", _seq_runner([_CP(0), _CP(1, "", "nothing to commit")]),
    )
    sha, err = fp._git_commit_keep(Path("/repo"), "msg")
    assert sha is None and err == "nothing to commit"


def test_git_commit_keep_head_unreadable(monkeypatch) -> None:
    monkeypatch.setattr(
        fp.subprocess, "run",
        _seq_runner([_CP(0), _CP(0), _CP(1, "", "")]),
    )
    sha, err = fp._git_commit_keep(Path("/repo"), "msg")
    assert sha is None and err  # surfaces the rev-parse error/fallback


# -- _run_git --------------------------------------------------------------
def test_run_git_success(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(0, "out", "")]))
    ok, out, err = fp._run_git(["status"])
    assert ok is True and out == "out"


def test_run_git_spawn_failure(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([FileNotFoundError("git")]))
    ok, out, err = fp._run_git(["status"])
    assert ok is False and "spawn/timeout failed" in err


def test_run_git_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(fp.subprocess, "run", _seq_runner([_CP(2, "partial", "boom")]))
    ok, out, err = fp._run_git(["status"])
    assert ok is False and out == "partial" and err == "boom"


# -- _normalize_repo_id ----------------------------------------------------
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


# -- _candidate_is_same_repo ----------------------------------------------
def test_candidate_is_same_repo_no_candidate_repo() -> None:
    # missing/invalid candidate repo -> fail open (True)
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
        fp, "_run_git",
        lambda *a, **k: (True, "https://github.com/Owner/Name.git\n", ""),
    )
    assert fp._candidate_is_same_repo({"repo": "Owner/Name"}, Path("/repo")) is True


def test_candidate_is_same_repo_differing_github(monkeypatch) -> None:
    monkeypatch.setattr(
        fp, "_run_git",
        lambda *a, **k: (True, "https://github.com/Other/Repo.git\n", ""),
    )
    assert fp._candidate_is_same_repo({"repo": "Owner/Name"}, Path("/repo")) is False
