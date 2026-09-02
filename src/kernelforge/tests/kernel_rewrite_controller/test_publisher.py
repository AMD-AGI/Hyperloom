# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import ControllerLayout, load_task
from kernelforge.kernel_rewrite_controller import publisher


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
    assert [path.name for path in publisher.published_operator_dirs(layout)] == [publication.operator_id]


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
    assert not first_target.exists()
    assert len(publisher.published_operator_dirs(layout)) == 1


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
