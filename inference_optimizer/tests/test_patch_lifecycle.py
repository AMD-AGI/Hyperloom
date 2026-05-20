"""_patch_lifecycle helper tests (P3 PR-H)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from inference_optimizer.orchestrator._patch_lifecycle import (
    BackupRef,
    PatchApplyError,
    VerdictInputs,
    apply_patch,
    backup_files,
    decide_verdict,
    generate_patch_id,
    rollback_backup,
)


# ---------------------------------------------------------------------------
# patch_id
# ---------------------------------------------------------------------------
def test_generate_patch_id_default_prefix_format():
    pid = generate_patch_id()
    assert re.match(r"^fw-\d{8}-[0-9a-f]{8}$", pid), pid


def test_generate_patch_id_custom_prefix():
    pid = generate_patch_id("kn")
    assert pid.startswith("kn-")


def test_generate_patch_id_unique_per_call():
    a = {generate_patch_id() for _ in range(50)}
    assert len(a) == 50


# ---------------------------------------------------------------------------
# backup_files + rollback_backup
# ---------------------------------------------------------------------------
def _seed_source(tmp_path: Path, name: str = "scheduler.py", body: str = "x = 1\n") -> Path:
    src_dir = tmp_path / "framework_source" / "vllm" / "engine"
    src_dir.mkdir(parents=True)
    p = src_dir / name
    p.write_text(body)
    return p


def test_backup_files_writes_snapshot_under_session_dir(tmp_path: Path):
    src = _seed_source(tmp_path)
    ref = backup_files("fw-test-001", [src], session_dir=tmp_path)
    assert isinstance(ref, BackupRef)
    assert ref.patch_id == "fw-test-001"
    assert ref.backup_root.exists()
    # Snapshot is reachable + identical to source.
    rel = src.as_posix().lstrip("/")
    snap = ref.backup_root / rel
    assert snap.read_text() == src.read_text()


def test_backup_files_raises_when_file_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        backup_files("fw-x", [tmp_path / "nope.py"], session_dir=tmp_path)


def test_rollback_backup_restores_modified_file(tmp_path: Path):
    src = _seed_source(tmp_path, body="original\n")
    ref = backup_files("fw-test", [src], session_dir=tmp_path)
    # Mutate the source.
    src.write_text("MUTATED\n")
    status = rollback_backup(ref)
    assert status[str(src)] == "restored"
    assert src.read_text() == "original\n"


def test_rollback_backup_records_missing_snapshot_as_failure(tmp_path: Path):
    src = _seed_source(tmp_path)
    ref = backup_files("fw-test", [src], session_dir=tmp_path)
    # Delete the snapshot post-backup.
    (ref.backup_root / src.as_posix().lstrip("/")).unlink()
    status = rollback_backup(ref)
    assert status[str(src)] == "missing_backup"


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------
def _write_minimal_patch(tmp_path: Path, src: Path) -> Path:
    """Build a unified diff that flips ``x = 1`` to ``x = 2`` in ``src``."""
    # Use the absolute path so apply_patch can resolve from cwd=/.
    rel = src.as_posix().lstrip("/")
    patch = tmp_path / "p.diff"
    patch.write_text(
        "--- a/" + rel + "\n"
        "+++ b/" + rel + "\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    return patch


@pytest.mark.skipif(
    shutil.which("git") is None and shutil.which("patch") is None,
    reason="needs git or patch on PATH",
)
def test_apply_patch_modifies_file_when_valid(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_minimal_patch(tmp_path, src)
    # Run from /, since the diff carries absolute paths.
    apply_patch(patch, cwd=Path("/"))
    assert "x = 2" in src.read_text()


def test_apply_patch_raises_when_patch_missing(tmp_path: Path):
    with pytest.raises(PatchApplyError, match="not found"):
        apply_patch(tmp_path / "nope.diff")


@pytest.mark.skipif(
    shutil.which("git") is None and shutil.which("patch") is None,
    reason="needs git or patch on PATH",
)
def test_apply_patch_raises_on_bad_diff(tmp_path: Path):
    src = _seed_source(tmp_path)
    rel = src.as_posix().lstrip("/")
    patch = tmp_path / "bad.diff"
    # Reference a line that doesn't exist (the source has "x = 1", not "y = 9").
    patch.write_text(
        "--- a/" + rel + "\n"
        "+++ b/" + rel + "\n"
        "@@ -1 +1 @@\n"
        "-y = 9\n"
        "+y = 10\n"
    )
    with pytest.raises(PatchApplyError):
        apply_patch(patch, cwd=Path("/"))


# ---------------------------------------------------------------------------
# decide_verdict (3-gate)
# ---------------------------------------------------------------------------
def test_decide_verdict_keep_when_all_gates_pass():
    out = decide_verdict(VerdictInputs(
        baseline_tput=100.0,
        baseline_accuracy=0.9,
        tput_after=110.0,
        accuracy_after=0.895,
        min_throughput_gain_pct=3.0,
        max_accuracy_drop_pct=1.0,
    ))
    assert out.verdict == "KEEP"
    assert out.gain_pct == pytest.approx(10.0, abs=0.01)


def test_decide_verdict_revert_when_gain_below_threshold():
    out = decide_verdict(VerdictInputs(
        baseline_tput=100.0,
        baseline_accuracy=0.9,
        tput_after=101.0,
        accuracy_after=0.9,
    ))
    assert out.verdict == "REVERT"
    assert "gain" in out.reason


def test_decide_verdict_revert_when_accuracy_drop_too_large():
    out = decide_verdict(VerdictInputs(
        baseline_tput=100.0,
        baseline_accuracy=0.9,
        tput_after=110.0,
        accuracy_after=0.85,  # 5.5% drop
        max_accuracy_drop_pct=1.0,
    ))
    assert out.verdict == "REVERT"
    assert "accuracy drop" in out.reason


def test_decide_verdict_revert_when_bench_failed():
    out = decide_verdict(VerdictInputs(
        baseline_tput=100.0,
        baseline_accuracy=0.9,
        tput_after=999.0,  # ignored when bench_ok=False
        accuracy_after=1.0,
        bench_ok=False,
        bench_reason="timeout",
    ))
    assert out.verdict == "REVERT"
    assert out.reason == "timeout"


def test_decide_verdict_needs_review_when_accuracy_missing():
    out = decide_verdict(VerdictInputs(
        baseline_tput=100.0,
        baseline_accuracy=0.9,
        tput_after=110.0,
        accuracy_after=None,
    ))
    assert out.verdict == "NEEDS_REVIEW"
    assert "missing" in out.reason


# ---------------------------------------------------------------------------
# Multi-patch LIFO rollback (design §9.2)
# ---------------------------------------------------------------------------
def test_multi_patch_rollback_lifo(tmp_path: Path):
    """Two patches touch the same file. Rollback in reverse order
    restores the original; rollback in forward order leaves an
    intermediate state."""
    src = _seed_source(tmp_path, body="v0\n")
    # P1: write v1
    ref1 = backup_files("fw-1", [src], session_dir=tmp_path)
    src.write_text("v1\n")
    # P2: write v2
    ref2 = backup_files("fw-2", [src], session_dir=tmp_path)
    src.write_text("v2\n")

    # LIFO: undo p2 first -> "v1"
    rollback_backup(ref2)
    assert src.read_text() == "v1\n"
    # Then undo p1 -> "v0"
    rollback_backup(ref1)
    assert src.read_text() == "v0\n"
