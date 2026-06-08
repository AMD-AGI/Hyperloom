"""Tests for the InferenceX preflight clone + read-only guard.

Regression coverage for the baseline failures where a brain-launched run
skipped install.sh's ``ensure_inferencex`` and the preflight then fell
back to a read-only ``/wekafs/hyperloom/InferenceX`` mount, so Magpie's
``_prepare_benchmark_scripts`` died with ``[Errno 30] Read-only file
system`` before the server booted.

The fix:
  * removes the hard-coded read-only host candidates from detection;
  * clones a fresh writable checkout when none is found;
  * hard-fails when the clone fails or the resolved path is not writable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer import cli


# ---------------------------------------------------------------------------
# _clone_inferencex
# ---------------------------------------------------------------------------
def test_clone_inferencex_sha_ref_uses_shallow_fetch(tmp_path, monkeypatch):
    """A 40-hex INFERENCEX_REF triggers init + shallow fetch + checkout."""
    monkeypatch.setenv("INFERENCEX_REF", "a" * 40)
    monkeypatch.setenv("INFERENCEX_REPO", "https://example.invalid/X.git")
    dest = tmp_path / "InferenceX"
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    with patch("inference_optimizer.cli.subprocess.run", side_effect=fake_run):
        out = cli._clone_inferencex(dest)

    assert out == str(dest)
    # init + fetch + checkout (not a --branch clone)
    assert any(c[:2] == ["git", "init"] for c in calls)
    assert any("fetch" in c for c in calls)
    assert not any("clone" in c for c in calls)


def test_clone_inferencex_branch_ref_uses_clone(tmp_path, monkeypatch):
    """A non-hex ref (branch/tag) uses ``git clone --branch``."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    with patch("inference_optimizer.cli.subprocess.run", side_effect=fake_run):
        out = cli._clone_inferencex(dest)

    assert out == str(dest)
    assert any("clone" in c and "--branch" in c for c in calls)


def test_clone_inferencex_failure_returns_none(tmp_path, monkeypatch):
    """A git failure soft-degrades to None (caller decides to hard-fail)."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"

    def boom(cmd, *a, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    with patch("inference_optimizer.cli.subprocess.run", side_effect=boom):
        out = cli._clone_inferencex(dest)

    assert out is None


# ---------------------------------------------------------------------------
# detection no longer includes read-only host mounts
# ---------------------------------------------------------------------------
def test_detection_candidates_exclude_wekafs_host_mounts():
    """The removed read-only fallbacks must not reappear in the source."""
    src = Path(cli.__file__).read_text(encoding="utf-8")
    # The preflight detection loop must not hard-code these read-only mounts.
    assert 'Path("/wekafs/hyperloom/InferenceX")' not in src
    assert 'Path("/opt/hyperloom/InferenceX")' not in src
    assert (
        'Path("/wekafs/fully-local/inference_optimization/InferenceX")'
        not in src
    )
