"""Tests for attempt-scoped producer state inside a caller's workspace."""

from __future__ import annotations

import os

import pytest

from kernelforge.rewrite_by_flydsl import attempt as attempt_module
from kernelforge.rewrite_by_flydsl.attempt import (
    create_attempt_workspace,
    export_import_path,
)


def test_an_attempt_gets_its_own_directory_under_the_workspace(tmp_path):
    attempt = create_attempt_workspace(tmp_path)

    assert attempt.root.is_dir()
    assert attempt.root.parent.name == ".forge_rewrite"
    assert attempt.root.parent.parent == tmp_path.resolve()
    assert attempt.relative_root == f".forge_rewrite/{attempt.attempt_id}"


def test_two_attempts_never_share_a_directory(tmp_path):
    first = create_attempt_workspace(tmp_path)
    second = create_attempt_workspace(tmp_path)

    assert first.root != second.root


def test_a_rerun_cannot_inherit_the_previous_candidate(tmp_path):
    first = create_attempt_workspace(tmp_path)
    first.candidate_path("kernel.py").write_text("stale port\n")

    second = create_attempt_workspace(tmp_path)

    assert second.candidate_path("kernel.py").exists() is False


def test_the_declared_temporary_path_is_workspace_relative(tmp_path):
    attempt = create_attempt_workspace(tmp_path)

    assert attempt.temporary_paths == [attempt.relative_root]
    for path in attempt.temporary_paths:
        assert not os.path.isabs(path)
        assert (tmp_path / path).resolve() == attempt.root


def test_the_candidate_may_sit_in_a_subdirectory_of_the_attempt(tmp_path):
    attempt = create_attempt_workspace(tmp_path)

    candidate = attempt.candidate_path("flydsl/kernel.py")

    assert candidate.parent.parent == attempt.root


@pytest.mark.parametrize("name", ["../kernel.py", "a/../../kernel.py", "", "   "])
def test_a_candidate_name_that_escapes_the_attempt_is_rejected(tmp_path, name):
    attempt = create_attempt_workspace(tmp_path)

    with pytest.raises(ValueError):
        attempt.candidate_path(name)


def test_an_absolute_candidate_name_cannot_leave_the_attempt(tmp_path):
    attempt = create_attempt_workspace(tmp_path)

    with pytest.raises(ValueError):
        attempt.candidate_path("/etc/kernel.py")


def test_the_attempt_directory_becomes_importable(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing/entry")
    attempt = create_attempt_workspace(tmp_path)

    export_import_path(attempt)

    entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(attempt.root)
    assert "/existing/entry" in entries


def test_exporting_the_import_path_twice_adds_one_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    attempt = create_attempt_workspace(tmp_path)

    export_import_path(attempt)
    export_import_path(attempt)

    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(attempt.root)]


def test_the_attempt_root_is_a_producer_owned_path(tmp_path):
    from kernelforge.loop import path_ownership
    from kernelforge.rewrite_by_flydsl import protocol

    attempt = create_attempt_workspace(tmp_path)

    assert attempt_module.ATTEMPT_ROOT_DIR in path_ownership.PRODUCER_PATH_PATTERNS
    assert protocol.is_producer_owned_path(f"{attempt.relative_root}/kernel.py") is True
