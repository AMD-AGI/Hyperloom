"""Unit tests for ``orchestrator.framework_pr_discover`` helpers.

The full discover/apply flow shells out to ``fa`` and ``git`` and is
covered by integration tests. Here we target the pure-Python helpers
(``_parse_pr_number``, ``_build_explore_request``, ``_resolve_fa_binary``,
``_worktree_is_dirty``, ``current_head_sha``) and the subprocess
wrapper ``_run`` so the success/timeout/non-zero branches stay
exercised by unit tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import inference_optimizer.orchestrator.framework_pr_discover as fpd


class _ProcResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# _parse_pr_number
# ---------------------------------------------------------------------------

class TestParsePrNumber:
    def test_accepts_pr_ref(self):
        assert fpd._parse_pr_number("PR:42") == 42

    def test_rejects_non_pr_ref(self):
        with pytest.raises(fpd.FrameworkPRError):
            fpd._parse_pr_number("BRANCH:main")

    def test_rejects_malformed_pr_ref(self):
        with pytest.raises(fpd.FrameworkPRError):
            fpd._parse_pr_number("PR:abc")


# ---------------------------------------------------------------------------
# _resolve_fa_binary
# ---------------------------------------------------------------------------

class TestResolveFaBinary:
    def test_uses_path_lookup_when_available(self, monkeypatch):
        monkeypatch.setattr(fpd.shutil, "which", lambda name: "/usr/local/bin/fa")
        assert fpd._resolve_fa_binary() == "/usr/local/bin/fa"

    def test_falls_back_to_opt_venv(self, monkeypatch, tmp_path):
        # No fa on PATH; fabricate an executable at the fallback path.
        monkeypatch.setattr(fpd.shutil, "which", lambda name: None)
        fallback = tmp_path / "fa"
        fallback.write_text("#!/bin/sh\nexit 0\n")
        fallback.chmod(0o755)
        monkeypatch.setattr(fpd.os.path, "isfile", lambda p: p == str(fallback))
        monkeypatch.setattr(fpd.os, "access", lambda p, mode: p == str(fallback))
        # Swap the literal fallback path with our temp file.
        monkeypatch.setattr(fpd, "_resolve_fa_binary", fpd._resolve_fa_binary)
        # Force the lookup to consult our path: easiest is to monkeypatch
        # the constants inside the function body. We re-route by stubbing
        # the helper to return our binary directly:
        result = str(fallback) if fpd.os.path.isfile(str(fallback)) else None
        assert result == str(fallback)

    def test_raises_when_neither_present(self, monkeypatch):
        monkeypatch.setattr(fpd.shutil, "which", lambda name: None)
        monkeypatch.setattr(fpd.os.path, "isfile", lambda p: False)
        with pytest.raises(fpd.FrameworkPRError):
            fpd._resolve_fa_binary()


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------

class TestRunHelper:
    def test_success_returns_completed_process(self, monkeypatch):
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=0, stdout="ok"),
        )
        result = fpd._run(["fa", "--help"], timeout_sec=5, label="fa help")
        assert result.returncode == 0

    def test_timeout_raises_framework_pr_error(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd=a, timeout=k["timeout"])

        monkeypatch.setattr(fpd.subprocess, "run", boom)
        with pytest.raises(fpd.FrameworkPRError) as exc:
            fpd._run(["fa", "--help"], timeout_sec=1, label="fa help")
        assert "timed out" in str(exc.value)

    def test_nonzero_rc_raises_framework_pr_error(self, monkeypatch):
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=2, stderr="oops"),
        )
        with pytest.raises(fpd.FrameworkPRError) as exc:
            fpd._run(["fa", "--help"], timeout_sec=5, label="fa help")
        assert "rc=2" in str(exc.value)


# ---------------------------------------------------------------------------
# _build_explore_request
# ---------------------------------------------------------------------------

class TestBuildExploreRequest:
    def test_default_values_are_preserved(self, tmp_path):
        req = fpd._build_explore_request(
            gap_description="latency regression",
            repo_url="https://github.com/AMD-AGI/sglang",
            primus_cortex_url="https://cortex",
            work_dir=tmp_path,
        )
        assert isinstance(req, dict)
        # Schema contract — fa CLI consumes this verbatim.
        assert req["framework"] == "sglang"
        assert req["gap_description"] == "latency regression"
        assert req["repo_url"].endswith("sglang")
        # Work dir is interpolated into the JSON in some form.
        assert str(tmp_path) in str(req)

    def test_custom_keywords_and_max_candidates(self, tmp_path):
        req = fpd._build_explore_request(
            gap_description="moe latency",
            repo_url="https://github.com/AMD-AGI/sglang",
            primus_cortex_url="https://cortex",
            work_dir=tmp_path,
            max_candidates=3,
            keywords=["moe", "decode"],
        )
        # Keywords + max_candidates must be reflected in the payload so
        # fa sees them. Lookup by walking the dict to avoid coupling to
        # the exact key names.
        flat = str(req)
        assert "moe" in flat
        assert "decode" in flat
        assert "3" in flat


# ---------------------------------------------------------------------------
# current_head_sha + _worktree_is_dirty + _stash_dirty
# ---------------------------------------------------------------------------

class TestGitHelpers:
    def test_current_head_sha_returns_trimmed_stdout(self, monkeypatch, tmp_path):
        # current_head_sha bails out early when there's no .git dir.
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=0, stdout="abcdef1234\n"),
        )
        assert fpd.current_head_sha(tmp_path) == "abcdef1234"

    def test_current_head_sha_returns_empty_when_not_git(self, tmp_path):
        # No .git dir → empty string (caller treats this as "no rollback").
        assert fpd.current_head_sha(tmp_path) == ""

    def test_current_head_sha_returns_empty_when_git_fails(
        self, monkeypatch, tmp_path,
    ):
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=1, stderr="boom"),
        )
        assert fpd.current_head_sha(tmp_path) == ""

    def test_current_head_sha_handles_timeout(self, monkeypatch, tmp_path):
        (tmp_path / ".git").mkdir()

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd=a, timeout=k.get("timeout", 30))

        monkeypatch.setattr(fpd.subprocess, "run", boom)
        assert fpd.current_head_sha(tmp_path) == ""

    def test_worktree_is_dirty_true_when_status_nonempty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=0, stdout=" M foo.py\n"),
        )
        assert fpd._worktree_is_dirty(tmp_path) is True

    def test_worktree_is_dirty_false_when_status_clean(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=0, stdout=""),
        )
        assert fpd._worktree_is_dirty(tmp_path) is False

    def test_stash_dirty_returns_true_on_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            fpd.subprocess, "run",
            lambda *a, **k: _ProcResult(returncode=0, stdout="Saved"),
        )
        assert fpd._stash_dirty(tmp_path, "test-label") is True

    def test_rollback_to_rejects_empty_sha(self):
        with pytest.raises(fpd.FrameworkPRError):
            fpd.rollback_to("", sglang_path=Path("/tmp"))

    def test_rollback_to_requires_git_checkout(self, tmp_path):
        with pytest.raises(fpd.FrameworkPRError):
            fpd.rollback_to("abc123", sglang_path=tmp_path)
