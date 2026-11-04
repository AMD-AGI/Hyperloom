# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""apply_snapshot's contract: every descriptor's final state, or none of them."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom_common.unified_diff import PatchOp, PatchTarget
from hyperloom.orchestrator.kernel import apply_kernel_patch as akp


def _repo_with_kernel(tmp_path: Path) -> tuple[Path, Path]:
    """A repo holding ``kernel.py`` and an empty snapshot directory beside it."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    return repo_root, snapshot


def _apply(tmp_path: Path, repo_root: Path, snapshot: Path, descriptors: list[PatchTarget]) -> dict:
    return akp.apply_snapshot(
        descriptors=descriptors,
        snapshot_dir=snapshot,
        repo_root=repo_root,
        backup_dir=tmp_path / "backups",
    )


def test_a_delete_descriptor_removes_the_live_file_and_records_how_to_put_it_back(tmp_path: Path) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)

    result = _apply(tmp_path, repo_root, snapshot, [PatchTarget(PatchOp.DELETE, "kernel.py")])

    assert result["status"] == "ok"
    assert not (repo_root / "kernel.py").exists()
    [entry] = result["source_backups"]
    assert entry["disposition"] == "deleted"
    assert Path(entry["backup_path"]).read_text(encoding="utf-8") == "VALUE = 1\n"


def test_a_snapshot_symlink_lands_as_a_symlink(tmp_path: Path) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)
    try:
        (snapshot / "alias.py").symlink_to("kernel.py")
    except OSError:
        pytest.skip("this platform does not allow creating symlinks")

    result = _apply(tmp_path, repo_root, snapshot, [PatchTarget(PatchOp.WRITE, "alias.py")])

    assert result["status"] == "ok"
    assert os.readlink(repo_root / "alias.py") == "kernel.py"


def test_a_snapshot_symlink_pointing_outside_the_repo_is_refused(tmp_path: Path) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    try:
        (snapshot / "alias.py").symlink_to(outside)
    except OSError:
        pytest.skip("this platform does not allow creating symlinks")

    result = _apply(tmp_path, repo_root, snapshot, [PatchTarget(PatchOp.WRITE, "alias.py")])

    assert result["status"] == "failed"
    assert "outside authorized" in result["error"]
    assert not (repo_root / "alias.py").exists()


@pytest.mark.skipif(os.name != "posix", reason="mode bits are only meaningful on POSIX")
def test_a_descriptor_mode_is_applied_to_the_written_file(tmp_path: Path) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)
    (snapshot / "build.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    result = _apply(tmp_path, repo_root, snapshot, [PatchTarget(PatchOp.WRITE, "build.sh", mode="100755")])

    assert result["status"] == "ok"
    assert (repo_root / "build.sh").stat().st_mode & 0o777 == 0o755


def test_a_missing_snapshot_source_touches_nothing(tmp_path: Path) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)

    result = _apply(tmp_path, repo_root, snapshot, [PatchTarget(PatchOp.WRITE, "kernel.py")])

    assert result["status"] == "failed"
    assert result["path"] == "kernel.py"
    assert "snapshot missing content" in result["error"]
    assert (repo_root / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.parametrize("rel_path", ("/etc/passwd", "../outside.py"))
def test_a_destination_outside_the_repo_fails_pre_flight(tmp_path: Path, rel_path: str) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)

    result = _apply(tmp_path, repo_root, snapshot, [PatchTarget(PatchOp.WRITE, rel_path)])

    assert result["status"] == "failed"
    assert result["path"] == "<pre-flight>"


def test_a_write_that_fails_halfway_restores_every_already_touched_path(tmp_path: Path) -> None:
    repo_root, snapshot = _repo_with_kernel(tmp_path)
    (snapshot / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    (snapshot / "fresh.py").write_text("VALUE = 3\n", encoding="utf-8")
    # A directory where the last descriptor asserts a file: its write fails
    # after the two before it have already landed.
    (repo_root / "occupied").mkdir()
    (snapshot / "occupied").write_text("VALUE = 4\n", encoding="utf-8")

    result = _apply(
        tmp_path,
        repo_root,
        snapshot,
        [
            PatchTarget(PatchOp.WRITE, "kernel.py"),
            PatchTarget(PatchOp.WRITE, "fresh.py"),
            PatchTarget(PatchOp.WRITE, "occupied"),
        ],
    )

    assert result["status"] == "failed"
    assert result["path"] == "occupied"
    assert "repo restored" in result["error"]
    assert (repo_root / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo_root / "fresh.py").exists()
    assert (repo_root / "occupied").is_dir()
