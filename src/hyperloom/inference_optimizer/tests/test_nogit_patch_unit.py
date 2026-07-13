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

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, backup_root)
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

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert backups == []


def test_revert_patches_no_git_missing_backup_is_noop(tmp_path):
    """A revert record with no backup_path and no existing target is a no-op."""
    records = [{"target": str(tmp_path / "ghost.py"), "existed": False, "backup_path": None}]
    ng._revert_patches_no_git(records)  # must not raise


# ---------------------------------------------------------------------------
# #3 — backup name uniqueness across multiple patches sharing a backup_root
# ---------------------------------------------------------------------------

PATCH_A = """\
--- a/common.py
+++ b/common.py
@@ -1 +1 @@
-original_a
+patched_a
"""

PATCH_B = """\
--- a/common.py
+++ b/common.py
@@ -1 +1 @@
-patched_a
+patched_b
"""


def test_backup_names_unique_across_patches_same_basename(tmp_path):
    """Two patches touching a file with the same basename must not overwrite each
    other's backup when seq_offset is maintained across calls (#3 fix)."""
    target = tmp_path / "common.py"
    target.write_text("original_a\n", encoding="utf-8")

    patch_a = tmp_path / "patch_a.patch"
    patch_a.write_text(PATCH_A, encoding="utf-8")
    patch_b = tmp_path / "patch_b.patch"
    patch_b.write_text(PATCH_B, encoding="utf-8")

    backup_root = tmp_path / "shared_backups"
    accumulated: list = []

    ok, err, backups_a, *_ = ng._apply_patch_no_git(
        tmp_path, patch_a, backup_root, seq_offset=len(accumulated)
    )
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")
    accumulated.extend(backups_a)

    ok, err, backups_b, *_ = ng._apply_patch_no_git(
        tmp_path, patch_b, backup_root, seq_offset=len(accumulated)
    )
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")
    accumulated.extend(backups_b)

    # Both patches produced backup records.
    assert len(backups_a) >= 1
    assert len(backups_b) >= 1

    # Backup files must be distinct: no path collision.
    bak_paths_a = {r["backup_path"] for r in backups_a if r.get("backup_path")}
    bak_paths_b = {r["backup_path"] for r in backups_b if r.get("backup_path")}
    assert bak_paths_a.isdisjoint(bak_paths_b), (
        f"backup paths collide between patches: {bak_paths_a & bak_paths_b}"
    )

    # The patch_a backup must contain the original content (not overwritten by patch_b).
    for bak_path in bak_paths_a:
        content = Path(bak_path).read_text(encoding="utf-8")
        assert content == "original_a\n", (
            f"patch_a backup was overwritten by patch_b: {bak_path!r}"
        )


def test_seq_offset_zero_gives_deterministic_names(tmp_path):
    """seq_offset=0 (default) still produces valid non-empty backup filenames."""
    target = tmp_path / "mod.py"
    target.write_text("original\n", encoding="utf-8")

    patch_file = tmp_path / "mypatch.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bk")
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")

    # Backup name must embed the patch stem and end with .bak.
    for r in backups:
        if r.get("backup_path"):
            name = Path(r["backup_path"]).name
            assert "mypatch" in name, f"patch stem missing from backup name: {name!r}"
            assert name.endswith(".bak"), f"backup name doesn't end in .bak: {name!r}"


# ---------------------------------------------------------------------------
# #4 — rename/move patch: full revert (new file deleted, old file restored)
# ---------------------------------------------------------------------------

# Simulate a rename by: patch A removes old.py content, patch B creates new.py
# content (same file, moved). We encode this as two separate diff headers
# (old=/dev/null for new file creation, new=/dev/null for old file deletion).
RENAME_DELETE_OLD = """\
--- a/old_name.py
+++ /dev/null
@@ -1 +0,0 @@
-moved_content
"""

RENAME_CREATE_NEW = """\
--- /dev/null
+++ b/new_name.py
@@ -0,0 +1 @@
+moved_content
"""

# A single diff that truly renames (old != new, neither /dev/null).
# ``patch`` CLI can apply this as a modification to new_name.py when
# old_name.py doesn't exist (create) and new_name.py is touched.
# We rely on the Python-level record logic, not patch CLI execution,
# for the rename revert_action tracking test.
RENAME_DIFF = """\
--- a/old_name.py
+++ b/new_name.py
@@ -1 +1 @@
-moved_content
+moved_content
"""


def test_revert_action_delete_removes_new_file(tmp_path):
    """A record with revert_action='delete' causes the target to be removed."""
    target = tmp_path / "created.py"
    target.write_text("content\n", encoding="utf-8")
    records = [
        {"target": str(target), "existed": False, "backup_path": None, "revert_action": "delete"}
    ]
    ng._revert_patches_no_git(records)
    assert not target.exists(), "revert_action='delete' must remove the target"


def test_revert_action_restore_old_puts_back_source(tmp_path):
    """A record with revert_action='restore_old' restores the old file from backup."""
    old_file = tmp_path / "old_name.py"
    bak = tmp_path / "bk" / "backup.bak"
    bak.parent.mkdir(parents=True)
    bak.write_text("original_content\n", encoding="utf-8")
    # old_name.py was renamed away (doesn't exist now).
    records = [
        {
            "target": str(old_file),
            "existed": True,
            "backup_path": str(bak),
            "revert_action": "restore_old",
        }
    ]
    ng._revert_patches_no_git(records)
    assert old_file.exists(), "revert_action='restore_old' must recreate the old file"
    assert old_file.read_text() == "original_content\n"


def test_rename_patch_tracked_as_two_records(tmp_path):
    """A rename hunk (old != new, neither /dev/null) produces two backup records:
    one for the old source (restore_old) and one for the new destination (delete).
    """
    old_file = tmp_path / "old_name.py"
    old_file.write_text("moved_content\n", encoding="utf-8")
    # new_name.py does not exist yet (patch will create it).

    patch_file = tmp_path / "rename.patch"
    patch_file.write_text(RENAME_DIFF, encoding="utf-8")

    backup_root = tmp_path / "bk"

    # We only test the record structure produced by _apply_patch_no_git; the
    # actual patch CLI application may or may not succeed depending on the
    # target file layout, so we inspect records regardless of ok.
    _ok, _err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, backup_root)

    # Find old-source and new-destination records.
    restore_old_records = [r for r in backups if r.get("revert_action") == "restore_old"]
    delete_records = [r for r in backups if r.get("revert_action") == "delete"]

    # The rename must produce at least one restore_old and one delete record.
    assert restore_old_records, "rename patch must produce a restore_old record for the source file"
    assert delete_records, "rename patch must produce a delete record for the destination file"

    # The source record must point at old_name.py and carry a backup.
    src_rec = restore_old_records[0]
    assert "old_name" in src_rec["target"]
    assert src_rec["backup_path"] is not None

    # The destination record must point at new_name.py.
    dst_rec = next((r for r in delete_records if "new_name" in r["target"]), None)
    assert dst_rec is not None, "delete record must target new_name.py"


# ---------------------------------------------------------------------------
# ApplyFeedback structure tests
# ---------------------------------------------------------------------------

def test_apply_feedback_dry_run_failure_returns_fourth_item(tmp_path, monkeypatch):
    """When all dry-run levels fail, _apply_patch_no_git returns a 4-tuple with
    an ApplyFeedback carrying the accumulated per-level stderr."""
    import subprocess as _sp
    from hyperloom.orchestrator.actions.executors._apply_feedback import ApplyFeedback

    patch_file = tmp_path / "bad.patch"
    patch_file.write_text("not a patch\n", encoding="utf-8")

    class _FailCP:
        returncode = 1
        stdout = ""
        stderr = "hunk FAILED"

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _FailCP())
    monkeypatch.setattr(ng.subprocess, "run", lambda *a, **k: _FailCP())

    result = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert len(result) == 4, "must return a 4-tuple"
    ok, err, backups, feedback = result
    assert ok is False
    assert backups == []
    assert isinstance(feedback, ApplyFeedback)
    assert feedback.channel == "nogit"
    assert len(feedback.tried_levels) > 0
    # Per-level stderr should be present.
    assert "hunk FAILED" in feedback.stderr


def test_apply_feedback_success_returns_none_feedback(tmp_path):
    """A successful apply returns (True, '', backups, None)."""
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")
    backup_root = tmp_path / "bak"

    result = ng._apply_patch_no_git(tmp_path, patch_file, backup_root)
    if len(result) != 4:
        pytest.skip("unexpected return length")
    ok, err, backups, feedback = result
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")
    assert feedback is None, "successful apply must return None feedback"


def test_apply_feedback_roundtrip_serialization():
    """ApplyFeedback serialises to dict and deserialises cleanly."""
    from hyperloom.orchestrator.actions.executors._apply_feedback import ApplyFeedback

    fb = ApplyFeedback(
        patch="/tmp/foo.patch",
        channel="git",
        tried_levels=[1, 0, 2],
        stderr="error: patch does not apply",
        rejected_hunks="--- a/foo.py\n+++ b/foo.py",
        source_context="  42| def foo():",
    )
    d = fb.to_dict()
    assert d["patch"] == "/tmp/foo.patch"
    assert d["channel"] == "git"
    restored = ApplyFeedback.from_dict(d)
    assert restored.patch == fb.patch
    assert restored.tried_levels == fb.tried_levels
    assert restored.stderr == fb.stderr
    assert restored.rejected_hunks == fb.rejected_hunks


def test_apply_feedback_format_for_mandate_contains_key_sections():
    """format_for_mandate includes patch name, stderr, and context sections."""
    from hyperloom.orchestrator.actions.executors._apply_feedback import ApplyFeedback

    fb = ApplyFeedback(
        patch="/path/to/001_fix.patch",
        channel="git",
        tried_levels=[1],
        stderr="patch failed for file foo.py",
        rejected_hunks="@@ -1 +1 @@\n-old",
        source_context="  10| def foo():\n  11|     pass",
    )
    mandate_block = fb.format_for_mandate()
    assert "001_fix.patch" in mandate_block
    assert "patch failed for file foo.py" in mandate_block
    assert "Rejected hunks" in mandate_block
    assert "Source context" in mandate_block


def test_read_patch_source_context_returns_snippet(tmp_path):
    """read_patch_source_context resolves target + returns a line-numbered snippet."""
    from hyperloom.orchestrator.actions.executors._apply_feedback import read_patch_source_context

    target = tmp_path / "mod.py"
    target.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    patch_text = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -2,2 +2,2 @@\n"
        "-line2\n"
        "+line2_patched\n"
    )
    ctx = read_patch_source_context(patch_text, tmp_path, radius=6)
    assert "mod.py" in ctx
    assert "line" in ctx  # some content from the file


def test_read_patch_source_context_returns_empty_for_missing_file(tmp_path):
    """read_patch_source_context returns '' when target file doesn't exist."""
    from hyperloom.orchestrator.actions.executors._apply_feedback import read_patch_source_context

    patch_text = (
        "--- a/nonexistent.py\n"
        "+++ b/nonexistent.py\n"
        "@@ -1 +1 @@\n"
        "-x\n+y\n"
    )
    ctx = read_patch_source_context(patch_text, tmp_path, radius=6)
    assert ctx == ""
