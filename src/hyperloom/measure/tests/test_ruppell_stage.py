# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Staging a ruppell tree into the runner cache: the entry a sandbox checks, written whole or not at all."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from hyperloom.measure import ruppell, ruppell_stage
from hyperloom.measure.runner.cache_layout import tree_digest
from hyperloom.measure.ruppell_stage import StageError, stage
from hyperloom.measure.tests.ruppell_fixtures import COMMIT, archive, tree_files


def test_an_entry_is_the_tree_with_its_commit_and_a_manifest_written_last(tmp_path):
    entry, digest = stage(tmp_path, COMMIT, archive())

    assert entry == tmp_path / "tools" / f"ruppell-{COMMIT}"
    source = entry / "source"
    assert (source / ".commit").read_text() == COMMIT + "\n", "ruppell reads its commit from .commit without .git"
    assert (source / "runtime" / "build.sh").stat().st_mode & 0o777 == 0o755
    assert (source / "python" / "ruppell" / "__main__.py").stat().st_mode & 0o777 == 0o644
    manifest = json.loads((entry / "manifest").read_text())
    assert (
        (manifest["commit"], manifest["source"]["sha256"]) == (COMMIT, digest) == (COMMIT, tree_digest(source).sha256)
    )
    assert ruppell.entry_problem(entry, COMMIT, digest) == ""
    assert ruppell.tree_sha256(source) == digest, "the backend's digest is the runner cache's"
    assert [p.name for p in (tmp_path / "tools").iterdir()] == [entry.name], "no unfinished directory is left"


def test_staging_the_same_tree_again_leaves_the_entry_and_another_tree_is_refused(tmp_path):
    entry, digest = stage(tmp_path, COMMIT, archive())
    before = (entry / "manifest").read_bytes()

    assert stage(tmp_path, COMMIT, archive()) == (entry, digest)
    assert (entry / "manifest").read_bytes() == before
    changed = {**tree_files(), "README.md": ("another tree\n", 0o644)}
    with pytest.raises(StageError, match="exists and is not this one"):
        stage(tmp_path, COMMIT, archive(files=changed))


def test_an_entry_whose_source_changed_after_staging_is_reported(tmp_path):
    entry, digest = stage(tmp_path, COMMIT, archive())
    (entry / "source" / "runtime" / "build.sh").write_text("#!/bin/bash\nexit 0\n")

    assert "is not the tree its manifest records" in ruppell.entry_problem(entry, COMMIT, digest)
    assert "holds" in ruppell.entry_problem(entry, "2" * 40, digest)
    assert "no readable manifest" in ruppell.entry_problem(tmp_path / "nothing", COMMIT, digest)


def test_the_pinned_commit_must_be_the_pinned_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(ruppell_stage, "RUPPELL_COMMIT", COMMIT)
    monkeypatch.setattr(ruppell_stage, "RUPPELL_TREE_SHA256", "0" * 64)

    with pytest.raises(StageError, match="not the pinned"):
        stage(tmp_path, COMMIT, archive())
    assert not (tmp_path / "tools" / f"ruppell-{COMMIT}").exists()


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (archive(commit="2" * 40), "the archive is of"),
        (archive(files={"../escape": ("x", 0o644)}), "unsafe path"),
        (archive(files={"/etc/x": ("x", 0o644)}, top=""), "unsafe path|one top directory"),
    ],
)
def test_an_archive_that_is_not_the_commits_tree_is_refused(tmp_path, raw, error):
    with pytest.raises(StageError, match=error):
        stage(tmp_path, COMMIT, raw)
    assert not (tmp_path / "tools" / f"ruppell-{COMMIT}").exists()


def test_a_link_in_the_archive_is_refused(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        link = tarfile.TarInfo("ruppell/escape")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    with pytest.raises(StageError, match="not a file or a directory"):
        stage(tmp_path, COMMIT, buffer.getvalue())


def test_the_command_stages_a_file_and_prints_the_entry_and_its_digest(tmp_path, capsys):
    tarball = tmp_path / "r.tar.gz"
    tarball.write_bytes(archive())

    assert ruppell_stage.main(["--commit", COMMIT, "--archive", str(tarball), "--cache", str(tmp_path / "cache")]) == 0
    entry, digest = capsys.readouterr().out.split()
    assert ruppell.entry_problem(Path(entry), COMMIT, digest) == ""
    assert ruppell_stage.main(["--commit", "abc", "--archive", str(tarball), "--cache", str(tmp_path / "c")]) == 1
    assert "not a full commit hash" in capsys.readouterr().err
