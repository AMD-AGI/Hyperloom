# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Smoke tests for the shared _nogit_patch module.

Verifies the extracted helpers in isolation (no git, no GPU, no gateway).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _nogit_patch as ng
from hyperloom.orchestrator.actions.executors import integrate_patch as ip


# ---------------------------------------------------------------------------
# Public surface re-exported via integrate_patch (backward-compat guard)
# ---------------------------------------------------------------------------

def test_reexport_from_integrate_patch():
    """All names extracted from integrate_patch must still be importable from it."""
    assert ip._P_LEVELS is ng._P_LEVELS
    assert ip._PATCH_DEV_NULL is ng._PATCH_DEV_NULL
    assert ip._strip_path_prefix is ng._strip_path_prefix
    assert ip._is_within is ng._is_within
    assert ip._is_git_tree is ng._is_git_tree
    assert ip._apply_patch_no_git is ng._apply_patch_no_git
    assert ip._revert_patches_no_git is ng._revert_patches_no_git


# ---------------------------------------------------------------------------
# _strip_path_prefix
# ---------------------------------------------------------------------------

def test_strip_path_prefix_zero():
    assert ng._strip_path_prefix("a/b/c.py", 0) == "a/b/c.py"


def test_strip_path_prefix_one():
    assert ng._strip_path_prefix("a/b/c.py", 1) == "b/c.py"


def test_strip_path_prefix_beyond():
    # More levels than parts → basename
    assert ng._strip_path_prefix("a/b.py", 5) == "b.py"


def test_strip_path_prefix_p_levels_sane():
    # _P_LEVELS starts at 1 and covers git-native default first.
    assert ng._P_LEVELS[0] == 1


# ---------------------------------------------------------------------------
# _is_within
# ---------------------------------------------------------------------------

def test_is_within_same():
    p = Path("/foo/bar")
    assert ng._is_within(p, p)


def test_is_within_child():
    assert ng._is_within(Path("/foo/bar/baz.py"), Path("/foo/bar"))


def test_is_within_sibling():
    assert not ng._is_within(Path("/foo/other"), Path("/foo/bar"))


def test_is_within_parent():
    assert not ng._is_within(Path("/foo"), Path("/foo/bar"))


# ---------------------------------------------------------------------------
# _is_git_tree
# ---------------------------------------------------------------------------

class _CP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_is_git_tree_true(monkeypatch):
    monkeypatch.setattr(
        ng.subprocess,
        "run",
        lambda *a, **k: _CP(0, "true\n"),
    )
    assert ng._is_git_tree(Path("/some/repo")) is True


def test_is_git_tree_false_nonzero(monkeypatch):
    monkeypatch.setattr(
        ng.subprocess,
        "run",
        lambda *a, **k: _CP(128, ""),
    )
    assert ng._is_git_tree(Path("/not/a/repo")) is False


def test_is_git_tree_false_on_filenotfound(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(ng.subprocess, "run", _raise)
    assert ng._is_git_tree(Path("/wherever")) is False


def test_is_git_tree_false_on_timeout(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired("git", 10)
    monkeypatch.setattr(ng.subprocess, "run", _raise)
    assert ng._is_git_tree(Path("/wherever")) is False


# ---------------------------------------------------------------------------
# _apply_patch_no_git + _revert_patches_no_git — round-trip using real patch CLI
# ---------------------------------------------------------------------------

SIMPLE_DIFF = """\
--- a/target.py
+++ b/target.py
@@ -1 +1 @@
-original
+patched
"""


def test_apply_and_revert_roundtrip(tmp_path):
    """Apply a one-liner patch then revert it via backup; file ends up unchanged."""
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")

    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    backup_root = tmp_path / "backups"

    ok, err, backups = ng._apply_patch_no_git(tmp_path, patch_file, backup_root)
    if not ok:
        pytest.skip(f"patch CLI unavailable or dry-run failed: {err}")

    assert target.read_text() == "patched\n"
    assert backups  # at least one backup record

    ng._revert_patches_no_git(backups)
    assert target.read_text() == "original\n"


def test_apply_patch_no_git_dry_run_failure(tmp_path, monkeypatch):
    """All strip levels fail dry-run → returns (False, …, [])."""
    patch_file = tmp_path / "bad.patch"
    patch_file.write_text("not a patch\n", encoding="utf-8")

    # Make patch always fail
    import subprocess as _sp

    class _FailCP:
        returncode = 1
        stdout = ""
        stderr = "reject"

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _FailCP())
    monkeypatch.setattr(ng.subprocess, "run", lambda *a, **k: _FailCP())

    ok, err, backups = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert backups == []


def test_revert_patches_no_git_missing_backup_is_noop(tmp_path):
    """A revert record with no backup_path and no existing target is a no-op."""
    records = [{"target": str(tmp_path / "ghost.py"), "existed": False, "backup_path": None}]
    ng._revert_patches_no_git(records)  # must not raise
