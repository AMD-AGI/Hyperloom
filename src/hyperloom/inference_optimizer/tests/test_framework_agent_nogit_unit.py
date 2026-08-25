# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the patch revert paths in FrameworkAgentExecutor.

Covers both undo ledgers: the non-git per-file backups and the git-tree
snapshot. No GPU / gateway / real framework required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import framework_agent as fp
from hyperloom.orchestrator.actions.executors import _nogit_patch as ng


class _Executor:
    """Minimal FrameworkAgentExecutor stand-in exposing just _revert_patches."""

    def __init__(self) -> None:
        self._nogit_patch_backups: list[dict[str, Any]] = []
        self._git_snapshot_manifest: dict | None = None

    def _revert_patches(
        self,
        framework_root: Path | None,
        applied: list[Path],
    ) -> list[Path]:
        return fp.FrameworkAgentExecutor._revert_patches(
            self,  # type: ignore[arg-type]
            framework_root,
            applied,
        )


SIMPLE_DIFF = """\
--- a/target.py
+++ b/target.py
@@ -1 +1 @@
-original
+patched
"""

# Two headers for the same file, so GNU patch dry-runs clean against the
# on-disk content but fails the second hunk once the first has been written.
# Both files are mutated before the failure, which is what leaves a partially
# applied tree behind while ``applied`` is still empty.
OVERLAP_DIFF = """\
--- a/a.txt
+++ b/a.txt
@@ -1,3 +1,3 @@
 L1
-L2
+L2-PATCHED
 L3
--- a/a.txt
+++ b/a.txt
@@ -1,3 +1,3 @@
 L1
-L2
+L2-SECOND
 L3
--- a/b.txt
+++ b/b.txt
@@ -1,3 +1,3 @@
 B1
-B2
+B2-PATCHED
 B3
"""


def test_revert_patches_uses_nogit_backups_when_not_git_tree(tmp_path, monkeypatch):
    """A populated backup ledger on a non-git root restores and reports applied."""
    target = tmp_path / "target.py"
    target.write_text("patched\n", encoding="utf-8")

    bak = tmp_path / "target.py.bak"
    bak.write_text("original\n", encoding="utf-8")

    exe = _Executor()
    exe._nogit_patch_backups = [
        {"target": str(target), "existed": True, "backup_path": str(bak)},
    ]

    monkeypatch.setattr(ng, "_is_git_tree", lambda p: False)
    monkeypatch.setattr(fp, "_is_git_tree", lambda p: False)

    applied = [tmp_path / "fix.patch"]
    reverted = exe._revert_patches(tmp_path, applied)

    assert reverted == applied
    assert target.read_text() == "original\n"


def test_revert_patches_git_tree_restores_snapshot(tmp_path, monkeypatch):
    """A git tree reverts through the snapshot, never a whole-tree reset."""
    monkeypatch.setattr(fp, "_is_git_tree", lambda p: True)

    restored: list[dict] = []

    def _fake_restore(manifest):
        restored.append(manifest)
        return {"ok": True, "errors": []}

    monkeypatch.setattr(fp, "_restore_patch_snapshot", _fake_restore)

    exe = _Executor()
    exe._git_snapshot_manifest = {"repo_path": str(tmp_path), "paths": []}

    applied = [tmp_path / "fix.patch"]
    assert exe._revert_patches(tmp_path, applied) == applied
    assert restored == [exe._git_snapshot_manifest]


def test_revert_patches_reports_failed_snapshot_restore(tmp_path, monkeypatch, caplog):
    """A snapshot restore that did not fully verify must not report success."""
    import logging

    monkeypatch.setattr(fp, "_is_git_tree", lambda p: True)
    monkeypatch.setattr(
        fp,
        "_restore_patch_snapshot",
        lambda manifest: {"ok": False, "errors": ["pkg/mod.py:OSError:boom"]},
    )

    exe = _Executor()
    exe._git_snapshot_manifest = {"repo_path": str(tmp_path), "paths": []}

    with caplog.at_level(logging.ERROR, logger="hyperloom.orchestrator.actions.executors.framework_agent"):
        reverted = exe._revert_patches(tmp_path, [tmp_path / "p.patch"])

    assert reverted == []
    assert any("snapshot restore incomplete" in r.message for r in caplog.records)


def test_revert_patches_none_framework_root_is_noop():
    """framework_root=None is always a no-op."""
    exe = _Executor()
    assert exe._revert_patches(None, [Path("/some.patch")]) == []


def test_empty_ledger_does_not_touch_the_tree(tmp_path, monkeypatch):
    """Both ledgers empty → nothing is restored, so no git operation may fire."""
    monkeypatch.setattr(fp, "_is_git_tree", lambda p: True)

    def _unexpected(manifest):
        raise AssertionError("empty ledger must not attempt a restore")

    monkeypatch.setattr(fp, "_restore_patch_snapshot", _unexpected)

    exe = _Executor()
    assert exe._revert_patches(tmp_path, []) == []


def test_partial_apply_residue_is_reverted(tmp_path, monkeypatch):
    """A patch that mutates the tree then fails is fully rolled back.

    ``applied`` stays empty because no patch completed, so only the backup
    ledger can tell that a restore is owed.
    """
    (tmp_path / "a.txt").write_text("L1\nL2\nL3\n")
    (tmp_path / "b.txt").write_text("B1\nB2\nB3\n")

    patch_file = tmp_path / "overlap.patch"
    patch_file.write_text(OVERLAP_DIFF)

    ok, _err, backups, _fb = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")

    assert ok is False, "overlapping headers must fail the real apply"
    assert (tmp_path / "a.txt").read_text() != "L1\nL2\nL3\n", "tree must be mutated pre-failure"
    assert backups, "backups must be recorded even when the apply fails"

    exe = _Executor()
    exe._nogit_patch_backups = list(backups)

    monkeypatch.setattr(fp, "_is_git_tree", lambda p: False)
    monkeypatch.setattr(ng, "_is_git_tree", lambda p: False)

    exe._revert_patches(tmp_path, [])

    assert (tmp_path / "a.txt").read_text() == "L1\nL2\nL3\n"
    assert (tmp_path / "b.txt").read_text() == "B1\nB2\nB3\n"


def test_nogit_apply_revert_via_executor_roundtrip(tmp_path, monkeypatch):
    """Apply through the non-git channel, then revert restores the original."""
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")

    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")

    assert target.read_text() == "patched\n"

    exe = _Executor()
    exe._nogit_patch_backups = backups

    monkeypatch.setattr(fp, "_is_git_tree", lambda p: False)
    reverted = exe._revert_patches(tmp_path, [patch_file])

    assert reverted == [patch_file]
    assert target.read_text() == "original\n"


def test_framework_agent_imports_is_git_tree():
    """_is_git_tree must be importable from the framework_agent module namespace."""
    assert hasattr(fp, "_is_git_tree")
    assert callable(fp._is_git_tree)
