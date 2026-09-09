# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import ControllerLayout, load_task
from kernelforge.kernel_rewrite_controller import publisher
from kernelforge.kernel_rewrite_controller.paths import operator_directory_name


def _publication(task_dir: Path, best_commit: str, patch: str):
    task = load_task(task_dir, record_state=False).task
    assert task is not None
    return publisher.publication_from_task(
        task,
        best_commit=best_commit,
        patch=patch,
        manifest={
            "correctness_passed": True,
            "total_improved": True,
            "mean_case_speedup": 1.2,
            "iteration": 2,
            "changed_files": [task.kernel_path],
        },
    )


def test_publish_operator_result_exposes_one_complete_operator_directory(
    tmp_path: Path,
    task_dir: Path,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    publication = _publication(task_dir, "b" * 40, "first patch\n")

    destination = publisher.publish_operator_result(layout, publication)

    assert destination.is_symlink()
    assert (destination / "change.patch").read_text(encoding="utf-8") == "first patch\n"
    assert "**Best commit:** `" + "b" * 40 + "`" in (destination / "report.md").read_text(encoding="utf-8")
    metadata = json.loads((destination / "publication.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["repo_root"] == str(publication.repo_root)
    assert metadata["kernel_path"] == publication.kernel_path
    assert metadata["identity"]["kernel_name"] == "fused_moe"
    assert metadata["micro_validated"] is True
    assert [path.name for path in publisher.published_operator_dirs(layout)] == [
        operator_directory_name(publication.operator_id)
    ]


def test_publication_records_the_scope_git_reports_not_the_manifest_claim(
    tmp_path: Path,
    task_dir: Path,
) -> None:
    """What integration stages has to come from the patch, not from the optimizer.

    ``source_files`` is orientation for forge-loop rather than an edit allowlist,
    so the manifest's account of what it edited cannot bound the patch.
    """
    layout = ControllerLayout(tmp_path / "output")
    task = load_task(task_dir, record_state=False).task
    assert task is not None
    publication = publisher.publication_from_task(
        task,
        best_commit="c" * 40,
        patch="patch\n",
        manifest={"changed_files": [task.kernel_path]},
        changed_files=(task.kernel_path, "sglang/srt/layers/helper.py"),
    )

    destination = publisher.publish_operator_result(layout, publication)

    metadata = json.loads((destination / "publication.json").read_text(encoding="utf-8"))
    assert metadata["changed_files"] == [task.kernel_path, "sglang/srt/layers/helper.py"]
    assert "sglang/srt/layers/helper.py" in (destination / "report.md").read_text(encoding="utf-8")


def test_new_keep_atomically_replaces_the_public_operator_result(
    tmp_path: Path,
    task_dir: Path,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    first = _publication(task_dir, "b" * 40, "first patch\n")
    second = _publication(task_dir, "c" * 40, "second patch\n")
    destination = publisher.publish_operator_result(layout, first)
    first_target = destination.resolve()

    publisher.publish_operator_result(layout, second)

    assert destination.resolve() != first_target
    assert (destination / "change.patch").read_text(encoding="utf-8") == "second patch\n"
    # Only one operator is exposed, and the superseded version survives: the
    # version store is content-addressed and immutable, and the pointer swap is
    # atomic where removing the old tree would not be, so a reader that already
    # resolved it keeps what it resolved.
    assert len(publisher.published_operator_dirs(layout)) == 1
    assert (first_target / "change.patch").read_text(encoding="utf-8") == "first patch\n"


def test_failure_while_writing_staging_keeps_the_previous_result(
    tmp_path: Path,
    task_dir: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    first = _publication(task_dir, "b" * 40, "first patch\n")
    second = _publication(task_dir, "c" * 40, "second patch\n")
    destination = publisher.publish_operator_result(layout, first)
    real_write = publisher.atomic_write_text

    def _fail_report(path, text):
        if Path(path).name == publisher.REPORT_FILENAME:
            raise OSError("injected report failure")
        return real_write(path, text)

    monkeypatch.setattr(publisher, "atomic_write_text", _fail_report)

    with pytest.raises(OSError, match="injected report failure"):
        publisher.publish_operator_result(layout, second)

    assert (destination / "change.patch").read_text(encoding="utf-8") == "first patch\n"
    assert not any(path.name.startswith(".") and "tmp" in path.name for path in layout.patches_root.iterdir())


def test_pointer_swap_failure_keeps_the_previous_public_pointer(
    tmp_path: Path,
    task_dir: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    first = _publication(task_dir, "b" * 40, "first patch\n")
    second = _publication(task_dir, "c" * 40, "second patch\n")
    destination = publisher.publish_operator_result(layout, first)
    real_replace = os.replace

    def _fail_pointer(source, target):
        if Path(source).is_symlink():
            raise OSError("injected pointer failure")
        return real_replace(source, target)

    monkeypatch.setattr(publisher.os, "replace", _fail_pointer)

    with pytest.raises(OSError, match="injected pointer failure"):
        publisher.publish_operator_result(layout, second)

    assert (destination / "change.patch").read_text(encoding="utf-8") == "first patch\n"


def test_republishing_the_same_commit_is_idempotent(tmp_path: Path, task_dir: Path) -> None:
    """A recovery pass that re-reads the same checkpoint must not fail on it."""
    layout = ControllerLayout(tmp_path / "output")
    publication = _publication(task_dir, "b" * 40, "same patch\n")

    first = publisher.publish_operator_result(layout, publication)
    second = publisher.publish_operator_result(layout, publication)

    assert first.resolve() == second.resolve()
    assert (second / "change.patch").read_text(encoding="utf-8") == "same patch\n"


def test_a_version_directory_that_disagrees_with_its_commit_is_refused(
    tmp_path: Path,
    task_dir: Path,
) -> None:
    """One commit means one patch. A version that says otherwise is corrupt.

    The version directory is keyed by best commit and treated as immutable, so a
    second publication claiming the same commit with different bytes cannot be
    reconciled -- publishing either one would misreport what the commit contains.
    """
    layout = ControllerLayout(tmp_path / "output")
    publisher.publish_operator_result(layout, _publication(task_dir, "b" * 40, "original\n"))

    with pytest.raises(publisher.PublicationError, match="patch conflicts with existing version"):
        publisher.publish_operator_result(layout, _publication(task_dir, "b" * 40, "rewritten\n"))


def test_an_incomplete_version_directory_is_refused(tmp_path: Path, task_dir: Path) -> None:
    """A half-written version is not a version, even if the pointer survived."""
    layout = ControllerLayout(tmp_path / "output")
    publication = _publication(task_dir, "b" * 40, "patch\n")
    destination = publisher.publish_operator_result(layout, publication)
    (destination.resolve() / "report.md").unlink()

    with pytest.raises(publisher.PublicationError, match="incomplete operator publication"):
        publisher.publish_operator_result(layout, publication)


def test_unreadable_version_metadata_is_refused(tmp_path: Path, task_dir: Path) -> None:
    layout = ControllerLayout(tmp_path / "output")
    publication = _publication(task_dir, "b" * 40, "patch\n")
    destination = publisher.publish_operator_result(layout, publication)
    (destination.resolve() / "publication.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(publisher.PublicationError, match="invalid operator publication metadata"):
        publisher.publish_operator_result(layout, publication)


def test_a_plain_directory_where_the_pointer_belongs_is_refused(tmp_path: Path, task_dir: Path) -> None:
    """The public path must stay a symlink, or the swap stops being atomic."""
    layout = ControllerLayout(tmp_path / "output")
    publication = _publication(task_dir, "b" * 40, "patch\n")
    squatter = layout.patch_dir(publication.operator_id)
    squatter.mkdir(parents=True)

    with pytest.raises(publisher.PublicationError, match="not an atomic pointer"):
        publisher.publish_operator_result(layout, publication)
