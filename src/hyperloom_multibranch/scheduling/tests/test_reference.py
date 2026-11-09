# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The reference material a run is given: what ``NAME=DIR`` has to spell, and what the copy leaves in the session."""

from __future__ import annotations

import os
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
        ("/data/ref/GEAK", "is not NAME=DIR"),
        ("GEAK Tuning=/data/ref", "is not NAME=DIR"),
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


def test_a_link_out_of_the_source_or_to_nothing_is_left_out_of_the_copy_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secrets = tmp_path / "operator" / ".ssh"
    secrets.mkdir(parents=True)
    (secrets / "id_rsa").write_text("PRIVATE-KEY", encoding="utf-8")
    knowledge_checkout(tmp_path / "GEAK", "tuning/gemm.md", "notes.md")
    tuning = tmp_path / "GEAK" / "tuning"
    (tuning / "innocent.md").symlink_to(secrets / "id_rsa")
    (tuning / "relative.md").symlink_to(Path("..", "..", "operator", ".ssh", "id_rsa"))
    (tuning / "keys").symlink_to(secrets)
    (tuning / "sibling.md").symlink_to(Path("..", "notes.md"))
    (tuning / "dangling.md").symlink_to("nowhere.md")
    into = tmp_path / "MB-reference"
    into.mkdir()

    with caplog.at_level("WARNING", logger="hyperloom_multibranch.scheduling.reference"):
        copy_references((reference_source(f"geak-tuning={tuning}"),), into)

    copy = into / "geak-tuning"
    assert sorted(entry.name for entry in copy.iterdir()) == ["gemm.md"]
    assert not any(b"PRIVATE-KEY" in path.read_bytes() for path in copy.rglob("*") if path.is_file())
    assert [record.getMessage() for record in caplog.records] == [
        f"reference geak-tuning: 5 entries left out of the copy, because each is a link out of {tuning.resolve()} or "
        "to nothing, or no regular file: dangling.md, innocent.md, keys, relative.md, sibling.md"
    ]


def test_a_link_within_the_source_points_at_its_place_in_the_copy(tmp_path: Path) -> None:
    knowledge_checkout(tmp_path / "GEAK", "tuning/gemm.md", "tuning/deep/attention.md")
    tuning = tmp_path / "GEAK" / "tuning"
    (tuning / "alias.md").symlink_to("gemm.md")
    (tuning / "absolute.md").symlink_to(tuning / "deep" / "attention.md")
    (tuning / "deep" / "up.md").symlink_to(Path("..", "alias.md"))
    (tuning / "shallow").symlink_to("deep")
    into = tmp_path / "MB-reference"
    into.mkdir()

    copy_references((reference_source(f"geak-tuning={tuning}"),), into)

    copy = (into / "geak-tuning").resolve()
    for name, target in (
        ("alias.md", "gemm.md"),
        ("absolute.md", "deep/attention.md"),
        ("deep/up.md", "gemm.md"),
        ("shallow", "deep"),
    ):
        link = copy / name
        assert link.is_symlink() and not Path(link.readlink()).is_absolute()
        assert link.resolve() == copy / target
    assert (copy / "shallow" / "attention.md").read_text(encoding="utf-8") == KNOWLEDGE_TEXT


def test_a_fifo_in_the_source_is_left_out_rather_than_failing_the_copy(tmp_path: Path) -> None:
    knowledge_checkout(tmp_path / "GEAK", "tuning/gemm.md")
    tuning = tmp_path / "GEAK" / "tuning"
    os.mkfifo(tuning / "pipe")
    into = tmp_path / "MB-reference"
    into.mkdir()

    copy_references((reference_source(f"geak-tuning={tuning}"),), into)

    assert sorted(entry.name for entry in (into / "geak-tuning").iterdir()) == ["gemm.md"]
