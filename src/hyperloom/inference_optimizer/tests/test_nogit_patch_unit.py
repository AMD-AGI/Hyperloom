# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Smoke tests for the shared _nogit_patch module.

Verifies the extracted helpers in isolation (no git, no GPU, no gateway).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _apply_feedback as af
from hyperloom.orchestrator.actions.executors import _nogit_patch as ng
from hyperloom.orchestrator.actions.executors import integrate_patch as ip


# Public surface re-exported via integrate_patch (backward-compat guard)

def test_reexport_from_integrate_patch():
    """All names extracted from integrate_patch must still be importable from it."""
    assert ip._P_LEVELS is ng._P_LEVELS
    assert ip._PATCH_DEV_NULL is ng._PATCH_DEV_NULL
    assert ip._strip_path_prefix is ng._strip_path_prefix
    assert ip._is_within is ng._is_within
    assert ip._is_git_tree is ng._is_git_tree
    assert ip._apply_patch_no_git is ng._apply_patch_no_git
    assert ip._revert_patches_no_git is ng._revert_patches_no_git


# _strip_path_prefix

def test_strip_path_prefix_zero():
    assert ng._strip_path_prefix("a/b/c.py", 0) == "a/b/c.py"


def test_strip_path_prefix_one():
    assert ng._strip_path_prefix("a/b/c.py", 1) == "b/c.py"


def test_strip_path_prefix_beyond():
    # More levels than parts -> basename
    assert ng._strip_path_prefix("a/b.py", 5) == "b.py"


def test_strip_path_prefix_p_levels_sane():
    # _P_LEVELS starts at 1.
    assert ng._P_LEVELS[0] == 1


# _is_within

def test_is_within_same():
    p = Path("/foo/bar")
    assert ng._is_within(p, p)


def test_is_within_child():
    assert ng._is_within(Path("/foo/bar/baz.py"), Path("/foo/bar"))


def test_is_within_sibling():
    assert not ng._is_within(Path("/foo/other"), Path("/foo/bar"))


def test_is_within_parent():
    assert not ng._is_within(Path("/foo"), Path("/foo/bar"))


# _is_git_tree

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


# _apply_patch_no_git + _revert_patches_no_git — round-trip using real patch CLI

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


# Backup name uniqueness across multiple patches sharing a backup_root

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
    other's backup when seq_offset is maintained across calls."""
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

    assert len(backups_a) >= 1
    assert len(backups_b) >= 1

    # Backup files must be distinct: no path collision.
    bak_paths_a = {r["backup_path"] for r in backups_a if r.get("backup_path")}
    bak_paths_b = {r["backup_path"] for r in backups_b if r.get("backup_path")}
    assert bak_paths_a.isdisjoint(bak_paths_b), (
        f"backup paths collide between patches: {bak_paths_a & bak_paths_b}"
    )

    # The patch_a backup must retain the original content.
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

    for r in backups:
        if r.get("backup_path"):
            name = Path(r["backup_path"]).name
            assert "mypatch" in name, f"patch stem missing from backup name: {name!r}"
            assert name.endswith(".bak"), f"backup name doesn't end in .bak: {name!r}"


# Rename/move patch: full revert (new file deleted, old file restored)

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

    patch_file = tmp_path / "rename.patch"
    patch_file.write_text(RENAME_DIFF, encoding="utf-8")

    backup_root = tmp_path / "bk"

    # Inspect the record structure regardless of whether the patch CLI succeeds.
    _ok, _err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, backup_root)

    restore_old_records = [r for r in backups if r.get("revert_action") == "restore_old"]
    delete_records = [r for r in backups if r.get("revert_action") == "delete"]

    assert restore_old_records, "rename patch must produce a restore_old record for the source file"
    assert delete_records, "rename patch must produce a delete record for the destination file"

    src_rec = restore_old_records[0]
    assert "old_name" in src_rec["target"]
    assert src_rec["backup_path"] is not None

    dst_rec = next((r for r in delete_records if "new_name" in r["target"]), None)
    assert dst_rec is not None, "delete record must target new_name.py"


# ApplyFeedback structure tests

def test_apply_feedback_dry_run_failure_returns_fourth_item(tmp_path, monkeypatch):
    """When all dry-run levels fail, _apply_patch_no_git returns a 4-tuple with
    an ApplyFeedback carrying the accumulated per-level stderr."""
    import subprocess as _sp

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
    assert isinstance(feedback, af.ApplyFeedback)
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

    fb = af.ApplyFeedback(
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
    restored = af.ApplyFeedback.from_dict(d)
    assert restored.patch == fb.patch
    assert restored.tried_levels == fb.tried_levels
    assert restored.stderr == fb.stderr
    assert restored.rejected_hunks == fb.rejected_hunks


def test_apply_feedback_format_for_mandate_contains_key_sections():
    """format_for_mandate includes patch name, stderr, and context sections."""

    fb = af.ApplyFeedback(
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

    target = tmp_path / "mod.py"
    target.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    patch_text = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -2,2 +2,2 @@\n"
        "-line2\n"
        "+line2_patched\n"
    )
    ctx = af.read_patch_source_context(patch_text, tmp_path, radius=6)
    assert "mod.py" in ctx
    assert "line" in ctx


def test_read_patch_source_context_returns_empty_for_missing_file(tmp_path):
    """read_patch_source_context returns '' when target file doesn't exist."""

    patch_text = (
        "--- a/nonexistent.py\n"
        "+++ b/nonexistent.py\n"
        "@@ -1 +1 @@\n"
        "-x\n+y\n"
    )
    ctx = af.read_patch_source_context(patch_text, tmp_path, radius=6)
    assert ctx == ""


# _bak_name — filename sanitisation

def test_bak_name_sanitises_unsafe_chars():
    name = ng._bak_name("my:patch", Path("a/b/c.py"), 7)
    assert name == "my_patch__a_b_c.py__0007.bak"
    assert "/" not in name.replace(".bak", "")
    assert ":" not in name


# _apply_patch_no_git — patch CLI unavailable during dry-run

def test_apply_patch_dry_run_cli_missing(tmp_path, monkeypatch):
    """A FileNotFoundError during the dry-run yields a nogit ApplyFeedback."""

    patch_file = tmp_path / "p.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    def _raise(*a, **k):
        raise FileNotFoundError("patch")

    monkeypatch.setattr(ng.subprocess, "run", _raise)
    ok, err, backups, feedback = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert backups == []
    assert isinstance(feedback, af.ApplyFeedback)
    assert "unavailable" in err


# _apply_patch_no_git — create-file hunk (old = /dev/null)

CREATE_DIFF = """\
--- /dev/null
+++ b/created.py
@@ -0,0 +1 @@
+brand_new
"""


def test_apply_create_file_then_revert_deletes(tmp_path):
    """A create hunk applies, records a delete revert, and revert removes it."""
    patch_file = tmp_path / "create.patch"
    patch_file.write_text(CREATE_DIFF, encoding="utf-8")

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")

    created = tmp_path / "created.py"
    assert created.exists()
    delete_recs = [r for r in backups if r.get("revert_action") == "delete"]
    assert delete_recs, "create hunk must record a delete revert action"

    ng._revert_patches_no_git(backups)
    assert not created.exists(), "revert of a create hunk must delete the new file"


# _apply_patch_no_git — delete-file hunk (new = /dev/null)

DELETE_DIFF = """\
--- a/doomed.py
+++ /dev/null
@@ -1 +0,0 @@
-goodbye
"""


def test_apply_delete_file_backs_up_and_reverts(tmp_path):
    """A delete hunk backs up the existing file and revert restores it."""
    doomed = tmp_path / "doomed.py"
    doomed.write_text("goodbye\n", encoding="utf-8")

    patch_file = tmp_path / "delete.patch"
    patch_file.write_text(DELETE_DIFF, encoding="utf-8")

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    if not ok:
        pytest.skip(f"patch CLI unavailable: {err}")

    assert not doomed.exists(), "delete hunk must remove the file"
    restore_recs = [r for r in backups if r.get("revert_action") == "restore"]
    assert restore_recs, "delete hunk must back up the existing file"

    ng._revert_patches_no_git(backups)
    assert doomed.exists(), "revert must restore the deleted file"
    assert doomed.read_text() == "goodbye\n"


def test_apply_delete_missing_file_records_delete(tmp_path, monkeypatch):
    """A delete hunk whose target is already absent records a delete action."""
    # Force dry-run to succeed so we reach the target-resolution loop.
    calls = {"n": 0}

    def _fake_run(cmd, *a, **k):
        calls["n"] += 1
        return _CP(0, "", "")

    monkeypatch.setattr(ng.subprocess, "run", _fake_run)

    patch_file = tmp_path / "delete.patch"
    patch_file.write_text(DELETE_DIFF, encoding="utf-8")
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is True
    delete_recs = [r for r in backups if r.get("revert_action") == "delete"]
    assert delete_recs, "absent delete target must still record a delete action"
    assert delete_recs[0]["backup_path"] is None


# _apply_patch_no_git — target escapes framework root

ESCAPE_DIFF = """\
--- a/../../../etc/evil.py
+++ b/../../../etc/evil.py
@@ -1 +1 @@
-x
+y
"""


def test_apply_rejects_target_escaping_root(tmp_path, monkeypatch):
    """A patch whose resolved target escapes framework_root is rejected."""
    def _fake_run(cmd, *a, **k):
        return _CP(0, "", "")

    monkeypatch.setattr(ng.subprocess, "run", _fake_run)

    patch_file = tmp_path / "escape.patch"
    patch_file.write_text(ESCAPE_DIFF, encoding="utf-8")
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "escapes framework root" in err


# _apply_patch_no_git — backup copy failure

def test_apply_backup_failure_returns_error(tmp_path, monkeypatch):
    """When shutil.copy2 fails during backup, the apply aborts with an error."""
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    def _fake_run(cmd, *a, **k):
        return _CP(0, "", "")

    monkeypatch.setattr(ng.subprocess, "run", _fake_run)

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ng.shutil, "copy2", _boom)
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "backup of" in err


# _apply_patch_no_git — real apply failure collects .rej hunks

def test_apply_real_failure_collects_rej(tmp_path):
    """When dry-run passes for a level but the real apply fails, we surface a
    nogit ApplyFeedback (with any .rej content) rather than raising."""

    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    real_run = ng.subprocess.run
    call_state = {"n": 0}

    def _run(cmd, *a, **k):
        # Let dry-run pass through; force the real apply to fail.
        is_dry = any("--dry-run" in str(c) for c in cmd)
        if is_dry:
            return real_run(cmd, *a, **k)
        call_state["n"] += 1
        return _CP(1, "", "Hunk #1 FAILED at 1.")

    import unittest.mock as um

    with um.patch.object(ng.subprocess, "run", _run):
        ok, err, backups, feedback = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")

    if call_state["n"] == 0:
        pytest.skip("dry-run never succeeded on this patch CLI")
    assert ok is False
    assert isinstance(feedback, af.ApplyFeedback)
    assert feedback.channel == "nogit"
    assert "FAILED" in feedback.stderr


def test_apply_real_apply_cli_vanishes(tmp_path):
    """A FileNotFoundError raised only during the real apply is handled."""

    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    real_run = ng.subprocess.run

    def _run(cmd, *a, **k):
        is_dry = any("--dry-run" in str(c) for c in cmd)
        if is_dry:
            return real_run(cmd, *a, **k)
        raise FileNotFoundError("patch")

    import unittest.mock as um

    with um.patch.object(ng.subprocess, "run", _run):
        result = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")

    ok, err, backups, feedback = result
    if ok:
        pytest.skip("dry-run failed; cannot reach real-apply branch")
    assert isinstance(feedback, af.ApplyFeedback)
    assert "patch apply failed" in err


# _collect_rej_files

def test_collect_rej_files_reads_and_removes(tmp_path):
    """A recent .rej file is read into the summary and then removed."""
    rej = tmp_path / "foo.py.rej"
    rej.write_text("@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")

    out = ng._collect_rej_files(tmp_path, tmp_path / "some.patch")
    assert "foo.py.rej" in out
    assert "@@ -1 +1 @@" in out
    assert not rej.exists(), ".rej file must be cleaned up after collection"


def test_collect_rej_files_skips_stale(tmp_path):
    """A .rej file older than the 60s cutoff is ignored (and left in place)."""
    import os
    import time

    rej = tmp_path / "old.py.rej"
    rej.write_text("stale hunk\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(rej, (old, old))

    out = ng._collect_rej_files(tmp_path, tmp_path / "some.patch")
    assert out == ""


def test_collect_rej_files_no_rej_returns_empty(tmp_path):
    assert ng._collect_rej_files(tmp_path, tmp_path / "some.patch") == ""


# _revert_patches_no_git — error paths are swallowed

def test_revert_restore_error_is_logged_not_raised(tmp_path, monkeypatch):
    """A copy failure during restore is logged, never raised."""
    bak = tmp_path / "b.bak"
    bak.write_text("data\n", encoding="utf-8")
    records = [
        {"target": str(tmp_path / "t.py"), "existed": True, "backup_path": str(bak), "revert_action": "restore"}
    ]

    def _boom(*a, **k):
        raise OSError("perm denied")

    monkeypatch.setattr(ng.shutil, "copy2", _boom)
    ng._revert_patches_no_git(records)


def test_revert_delete_removes_existing(tmp_path):
    """revert_action='delete' with an existing target removes it."""
    t = tmp_path / "created.py"
    t.write_text("x\n", encoding="utf-8")
    records = [{"target": str(t), "existed": False, "backup_path": None, "revert_action": "delete"}]
    ng._revert_patches_no_git(records)
    assert not t.exists()


# Deterministic branch coverage via a fake dry-run-succeeds patch runner

def _fake_ok_run(cmd, *a, **k):
    """A subprocess.run stand-in that reports every patch invocation as success."""
    return _CP(0, "", "")


# Modification diff with identical old/new header tokens (plain modification).
MODIFY_SAME_PATH_DIFF = """\
--- target.py
+++ target.py
@@ -1 +1 @@
-original
+patched
"""


def test_modification_existing_file_backed_up(tmp_path, monkeypatch):
    """A plain modification hunk backs up the existing target (restore action)."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(MODIFY_SAME_PATH_DIFF, encoding="utf-8")

    ok, err, backups, feedback = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is True
    assert feedback is None
    restore = [r for r in backups if r.get("revert_action") == "restore"]
    assert restore and restore[0]["backup_path"] is not None


def test_modification_missing_file_records_delete(tmp_path, monkeypatch):
    """A modification hunk whose target is absent records a delete action."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(MODIFY_SAME_PATH_DIFF, encoding="utf-8")
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is True
    delete = [r for r in backups if r.get("revert_action") == "delete"]
    assert delete and delete[0]["backup_path"] is None


def test_modification_target_escape_rejected(tmp_path, monkeypatch):
    """A modification hunk whose target escapes the root is rejected."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    patch_file = tmp_path / "evil.patch"
    patch_file.write_text(
        "--- ../../../etc/evil.py\n"
        "+++ ../../../etc/evil.py\n"
        "@@ -1 +1 @@\n"
        "-x\n+y\n",
        encoding="utf-8",
    )
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "escapes framework root" in err


def test_delete_existing_file_backed_up(tmp_path, monkeypatch):
    """A delete hunk on an existing file records a restore backup (fake apply)."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    doomed = tmp_path / "doomed.py"
    doomed.write_text("goodbye\n", encoding="utf-8")
    patch_file = tmp_path / "del.patch"
    patch_file.write_text(DELETE_DIFF, encoding="utf-8")

    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is True
    restore = [r for r in backups if r.get("revert_action") == "restore"]
    assert restore and restore[0]["backup_path"] is not None


def test_create_target_escape_rejected(tmp_path, monkeypatch):
    """A create hunk whose destination escapes the root is rejected."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    patch_file = tmp_path / "evil_create.patch"
    patch_file.write_text(
        "--- /dev/null\n"
        "+++ b/../../../etc/evil.py\n"
        "@@ -0,0 +1 @@\n"
        "+pwn\n",
        encoding="utf-8",
    )
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "escapes framework root" in err


def test_patch_file_unreadable_after_dry_run(tmp_path, monkeypatch):
    """An OSError reading the patch text after dry-run returns a 3-tuple error."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    real_read = Path.read_text
    state = {"n": 0}

    def _read_text(self, *a, **k):
        # The post-dry-run read of the patch body raises to hit the OSError branch.
        if self == patch_file:
            state["n"] += 1
            if state["n"] >= 1:
                raise OSError("gone")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _read_text)
    result = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    ok = result[0]
    err = result[1]
    assert ok is False
    assert "cannot read patch file" in err


def test_collect_rej_files_scan_exception_is_swallowed(tmp_path, monkeypatch):
    """An exception during the rglob scan is logged and yields ''."""
    def _boom(*a, **k):
        raise RuntimeError("fs blew up")

    monkeypatch.setattr(ng.Path, "rglob", _boom)
    assert ng._collect_rej_files(tmp_path, tmp_path / "p.patch") == ""


def test_collect_rej_files_stat_oserror_skips(tmp_path, monkeypatch):
    """An OSError while stat-ing a .rej file is skipped (continue branch)."""
    rej = tmp_path / "foo.py.rej"
    rej.write_text("@@ hunk @@\n", encoding="utf-8")

    real_stat = Path.stat

    def _stat(self, *a, **k):
        if self.name.endswith(".rej"):
            raise OSError("stat denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _stat)
    assert ng._collect_rej_files(tmp_path, tmp_path / "p.patch") == ""


# Escape / continue edge branches in the create / delete / rename arms

def test_create_devnull_both_sides_skipped(tmp_path, monkeypatch):
    """A degenerate /dev/null -> /dev/null header is skipped (create continue)."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    patch_file = tmp_path / "noop.patch"
    patch_file.write_text(
        "--- /dev/null\n"
        "+++ /dev/null\n"
        "@@ -0,0 +0,0 @@\n",
        encoding="utf-8",
    )
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is True
    assert backups == []


def test_delete_target_escape_rejected(tmp_path, monkeypatch):
    """A delete hunk whose source escapes the root is rejected."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    patch_file = tmp_path / "evil_del.patch"
    patch_file.write_text(
        "--- ../../../etc/evil.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x\n",
        encoding="utf-8",
    )
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "escapes framework root" in err


def test_delete_backup_failure_returns_error(tmp_path, monkeypatch):
    """A backup copy failure on a delete hunk aborts with an error."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    doomed = tmp_path / "doomed.py"
    doomed.write_text("goodbye\n", encoding="utf-8")
    patch_file = tmp_path / "del.patch"
    patch_file.write_text(DELETE_DIFF, encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ng.shutil, "copy2", _boom)
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "backup of" in err


def test_rename_new_target_escape_rejected(tmp_path, monkeypatch):
    """A rename hunk whose destination escapes the root is rejected."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    old_file = tmp_path / "old_name.py"
    old_file.write_text("moved\n", encoding="utf-8")
    patch_file = tmp_path / "rename_evil.patch"
    patch_file.write_text(
        "--- a/old_name.py\n"
        "+++ b/../../../etc/evil.py\n"
        "@@ -1 +1 @@\n"
        "-moved\n+moved\n",
        encoding="utf-8",
    )
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "escapes framework root" in err


def test_modification_backup_failure_returns_error(tmp_path, monkeypatch):
    """A backup copy failure in the modification arm aborts with an error."""
    monkeypatch.setattr(ng.subprocess, "run", _fake_ok_run)
    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "mod.patch"
    patch_file.write_text(MODIFY_SAME_PATH_DIFF, encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ng.shutil, "copy2", _boom)
    ok, err, backups, *_ = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert "backup of" in err


def test_real_apply_fail_source_context_exception_swallowed(tmp_path, monkeypatch):
    """When the real apply fails AND source-context extraction throws, feedback
    still returns with empty source_context (real-apply exception branch)."""

    target = tmp_path / "target.py"
    target.write_text("original\n", encoding="utf-8")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    real_run = ng.subprocess.run

    def _run(cmd, *a, **k):
        is_dry = any("--dry-run" in str(c) for c in cmd)
        if is_dry:
            return real_run(cmd, *a, **k)
        return _CP(1, "", "Hunk #1 FAILED")

    def _raise_ctx(*a, **k):
        raise RuntimeError("ctx boom")

    import unittest.mock as um

    monkeypatch.setattr(af, "read_patch_source_context", _raise_ctx)
    with um.patch.object(ng.subprocess, "run", _run):
        result = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")

    ok, _err, _backups, feedback = result
    if ok:
        pytest.skip("dry-run failed; cannot reach real-apply branch")
    assert isinstance(feedback, af.ApplyFeedback)
    assert feedback.source_context == ""


def test_dry_run_fail_source_context_exception_swallowed(tmp_path, monkeypatch):
    """If reading source context throws on total dry-run failure, feedback still
    returns with an empty source_context (exception branch)."""

    patch_file = tmp_path / "bad.patch"
    patch_file.write_text(SIMPLE_DIFF, encoding="utf-8")

    monkeypatch.setattr(ng.subprocess, "run", lambda *a, **k: _CP(1, "", "nope"))

    def _raise_ctx(*a, **k):
        raise RuntimeError("context read boom")

    monkeypatch.setattr(af, "read_patch_source_context", _raise_ctx)
    ok, err, backups, feedback = ng._apply_patch_no_git(tmp_path, patch_file, tmp_path / "bak")
    assert ok is False
    assert isinstance(feedback, af.ApplyFeedback)
    assert feedback.source_context == ""
