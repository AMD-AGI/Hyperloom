# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the non-git patch apply/revert path in FrameworkAgentExecutor.

Mirrors the style of test_framework_agent_git_helpers_coverage_unit.py but
exercises the _is_git_tree=False branch introduced in the nongit-patch-symmetry
plan.  No GPU / gateway / real framework required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import framework_agent as fp
from hyperloom.orchestrator.actions.executors import _nogit_patch as ng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Executor:
    """Minimal FrameworkAgentExecutor stand-in exposing just _revert_patches."""

    def __init__(self) -> None:
        self._nogit_patch_backups: list[dict[str, Any]] = []

    def _revert_patches(
        self,
        framework_root: Path | None,
        applied: list[Path],
        *,
        pre_apply_sha: str | None,
    ) -> list[Path]:
        return fp.FrameworkAgentExecutor._revert_patches(
            self,  # type: ignore[arg-type]
            framework_root,
            applied,
            pre_apply_sha=pre_apply_sha,
        )


SIMPLE_DIFF = """\
--- a/target.py
+++ b/target.py
@@ -1 +1 @@
-original
+patched
"""


# ---------------------------------------------------------------------------
# _revert_patches — non-git branch
# ---------------------------------------------------------------------------

def test_revert_patches_uses_nogit_backups_when_not_git_tree(tmp_path, monkeypatch):
    """When _nogit_patch_backups is populated and the root is not a git tree,
    _revert_patches must call _revert_patches_no_git and return all applied."""
    target = tmp_path / "target.py"
    target.write_text("patched\n", encoding="utf-8")

    bak = tmp_path / "target.py.bak"
    bak.write_text("original\n", encoding="utf-8")

    exe = _Executor()
    exe._nogit_patch_backups = [
        {"target": str(target), "existed": True, "backup_path": str(bak)},
    ]

    # Ensure _is_git_tree returns False (non-git directory)
    monkeypatch.setattr(ng, "_is_git_tree", lambda p: False)
    # Also patch the import inside framework_agent module namespace
    monkeypatch.setattr(fp, "_is_git_tree", lambda p: False)

    applied = [tmp_path / "fix.patch"]
    reverted = exe._revert_patches(tmp_path, applied, pre_apply_sha=None)

    assert reverted == applied
    assert target.read_text() == "original\n"


def test_revert_patches_git_tree_uses_git_reset(tmp_path, monkeypatch):
    """When _is_git_tree returns True, _revert_patches uses git reset --hard."""
    monkeypatch.setattr(fp, "_is_git_tree", lambda p: True)

    reset_calls: list[str] = []

    def _fake_reset(root, sha):
        reset_calls.append(sha)
        return True, ""

    monkeypatch.setattr(fp, "_git_reset_hard", _fake_reset)

    exe = _Executor()
    exe._nogit_patch_backups = []

    applied = [tmp_path / "fix.patch"]
    reverted = exe._revert_patches(tmp_path, applied, pre_apply_sha="deadbeef")
    assert reverted == applied
    assert reset_calls == ["deadbeef"]


def test_revert_patches_no_pre_apply_sha_on_git_tree_logs_error(tmp_path, monkeypatch, caplog):
    """If we somehow have a git tree but no pre_apply_sha, return [] and log."""
    import logging
    monkeypatch.setattr(fp, "_is_git_tree", lambda p: True)

    exe = _Executor()
    exe._nogit_patch_backups = []

    with caplog.at_level(logging.ERROR, logger="hyperloom.orchestrator.actions.executors.framework_agent"):
        reverted = exe._revert_patches(tmp_path, [tmp_path / "p.patch"], pre_apply_sha=None)

    assert reverted == []
    assert any("no pre_apply_sha" in r.message for r in caplog.records)


def test_revert_patches_none_framework_root_is_noop():
    """framework_root=None is always a no-op."""
    exe = _Executor()
    assert exe._revert_patches(None, [Path("/some.patch")], pre_apply_sha="sha") == []


def test_revert_patches_empty_applied_is_noop(tmp_path):
    """No applied patches → no-op regardless of git state."""
    exe = _Executor()
    assert exe._revert_patches(tmp_path, [], pre_apply_sha="sha") == []


# ---------------------------------------------------------------------------
# _apply_patch_no_git integration: apply into a non-git dir, then revert
# ---------------------------------------------------------------------------

def test_nogit_apply_revert_via_executor_roundtrip(tmp_path):
    """End-to-end: apply a patch via _apply_patch_no_git, collect backups,
    then revert via _revert_patches with non-git path — file must be restored."""
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")

    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    backup_root = tmp_path / "bak"
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, backup_root)
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")

    assert target.read_text() == "patched\n"

    exe = _Executor()
    exe._nogit_patch_backups = backups

    # Monkeypatch _is_git_tree so the revert takes the nogit branch
    import hyperloom.orchestrator.actions.executors.framework_agent as fa_mod
    original_fn = fa_mod._is_git_tree
    try:
        fa_mod._is_git_tree = lambda p: False
        reverted = exe._revert_patches(tmp_path, [patch_file], pre_apply_sha=None)
    finally:
        fa_mod._is_git_tree = original_fn

    assert reverted == [patch_file]
    assert target.read_text() == "original\n"


# ---------------------------------------------------------------------------
# is_git_tree import (smoke: framework_agent module exports it via _nogit_patch)
# ---------------------------------------------------------------------------

def test_framework_agent_imports_is_git_tree():
    """_is_git_tree must be importable from the framework_agent module namespace."""
    assert hasattr(fp, "_is_git_tree")
    assert callable(fp._is_git_tree)
