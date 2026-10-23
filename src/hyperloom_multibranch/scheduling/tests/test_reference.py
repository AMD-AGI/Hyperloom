# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The reference material a run is given: what ``NAME=DIR`` has to spell, and what the copy leaves in the session."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.scheduling.reference import ReferenceSource, copy_references, reference_source
from hyperloom_multibranch.scheduling.tests.checkouts import KNOWLEDGE_TEXT, knowledge_checkout


def test_a_source_carries_the_commit_its_checkout_is_on(tmp_path: Path) -> None:
    commit = knowledge_checkout(tmp_path / "GEAK", "tuning/gemm.md")

    source = reference_source(f"geak-tuning={tmp_path / 'GEAK' / 'tuning'}")

    assert source == ReferenceSource("geak-tuning", str((tmp_path / "GEAK" / "tuning").resolve()), commit)


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ("/shared_nfs/ref/GEAK", "is not NAME=DIR"),
        ("GEAK Tuning=/shared_nfs/ref", "is not NAME=DIR"),
        ("geak-tuning=", "which is no readable directory"),
        ("geak-tuning=/nowhere-at-all", "which is no readable directory"),
    ],
)
def test_a_source_that_is_not_a_named_readable_directory_is_refused(spec: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        reference_source(spec)


def test_a_directory_in_no_git_checkout_cannot_be_pinned(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()

    with pytest.raises(ValueError, match="is in no git checkout"):
        reference_source(f"geak-tuning={loose}")


def test_the_copy_holds_what_the_source_held_under_its_name(tmp_path: Path) -> None:
    knowledge_checkout(tmp_path / "GEAK", "tuning/gemm.md", "tuning/deep/attention.md")
    source = reference_source(f"geak-tuning={tmp_path / 'GEAK' / 'tuning'}")
    into = tmp_path / "session" / "MB-reference"
    into.mkdir(parents=True)

    copy_references((source,), into)

    assert (into / "geak-tuning" / "gemm.md").read_text(encoding="utf-8") == KNOWLEDGE_TEXT
    assert (into / "geak-tuning" / "deep" / "attention.md").exists()


def test_a_source_that_vanished_between_the_command_line_and_the_copy_raises(tmp_path: Path) -> None:
    source = ReferenceSource("geak-tuning", str(tmp_path / "gone"), "c" * 40)

    with pytest.raises(OSError):
        copy_references((source,), tmp_path / "MB-reference")
