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
        # The checkout step materializes the validity marker the post-clone
        # check looks for (benchmarks/benchmark_lib.sh).
        if "checkout" in cmd:
            (dest / "benchmarks").mkdir(parents=True, exist_ok=True)
            (dest / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
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
        if "clone" in cmd:
            (dest / "benchmarks").mkdir(parents=True, exist_ok=True)
            (dest / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
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


def test_clone_inferencex_removes_partial_dir_on_failure(tmp_path, monkeypatch):
    """A bare ``git init`` that then fails to fetch leaves a stub dir; the
    clone must delete it so a later preflight does not mistake the stub for
    a valid checkout and skip re-cloning."""
    monkeypatch.setenv("INFERENCEX_REF", "a" * 40)
    dest = tmp_path / "InferenceX"

    def init_then_fail(cmd, *a, **kw):
        if "init" in cmd:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / ".git").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(128, cmd)  # fetch fails

    with patch("inference_optimizer.cli.subprocess.run", side_effect=init_then_fail):
        out = cli._clone_inferencex(dest)

    assert out is None
    assert not dest.exists()


def test_clone_inferencex_rejects_checkout_without_marker(tmp_path, monkeypatch):
    """git reports success but the tree lacks benchmarks/benchmark_lib.sh →
    treat as failure and clean up, never return a half-checkout."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"

    def fake_run(cmd, *a, **kw):
        dest.mkdir(parents=True, exist_ok=True)  # empty, no marker
        return subprocess.CompletedProcess(cmd, 0)

    with patch("inference_optimizer.cli.subprocess.run", side_effect=fake_run):
        out = cli._clone_inferencex(dest)

    assert out is None
    assert not dest.exists()


def test_inferencex_checkout_ok_requires_benchmark_lib(tmp_path):
    """The validity check rejects a bare dir and accepts a real checkout."""
    stub = tmp_path / "stub"
    stub.mkdir()
    assert cli._inferencex_checkout_ok(stub) is False

    good = tmp_path / "good"
    (good / "benchmarks").mkdir(parents=True)
    (good / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    assert cli._inferencex_checkout_ok(good) is True


def test_preflight_detects_checkout_via_validity_not_isdir():
    """Detection and post-clone guards must use the validity helper, not a
    bare ``is_dir()`` that would accept a half-cloned stub."""
    src = Path(cli.__file__).read_text(encoding="utf-8")
    assert "_inferencex_checkout_ok(candidate)" in src
    assert "_inferencex_checkout_ok(inferencex_path)" in src


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


# ---------------------------------------------------------------------------
# validated path overwrites a stale/broken INFERENCEX_PATH env
# ---------------------------------------------------------------------------
def test_validated_inferencex_path_overwrites_env_not_setdefault():
    """A stale/broken INFERENCEX_PATH that triggers the clone must be
    overwritten with the validated path. ``setdefault`` would leave the bad
    value in place, so Magpie would still read the broken mount."""
    src = Path(cli.__file__).read_text(encoding="utf-8")
    # The final export must be an unconditional assignment, never setdefault.
    assert 'os.environ["INFERENCEX_PATH"] = inferencex_path' in src
    assert 'os.environ.setdefault("INFERENCEX_PATH"' not in src


def test_auto_detected_inferencex_candidates_must_be_writable():
    """Auto-detected read-only checkouts are skipped so preflight can clone."""
    src = Path(cli.__file__).read_text(encoding="utf-8")
    assert "if os.access(candidate, os.W_OK):" in src
    assert "skipping non-writable auto-detected" in src
