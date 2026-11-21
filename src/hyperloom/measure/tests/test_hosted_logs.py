# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted attempt keeps of its runner's files: nested logs at their paths, within bounds, never through a link."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from hyperloom.measure.hosted_logs import (
    INDEX_FILE,
    PROGRESS_FILE,
    SaveLimits,
    SeenProgress,
    save_runner_files,
    write_progress,
)
from hyperloom.sandbox.socket import DirEntry, SandboxGone

ROOT = "/out/files"


class _Tree:
    """A runner's ``/out`` as a directory on disk, read the way a sandbox reads it (``list_dir`` and ``copy_out``)."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.copied: list[str] = []
        self.gone_after: int | None = None

    def _real(self, path: str) -> Path:
        return self.base / path.lstrip("/")

    def list_dir(self, path: str) -> list[str]:
        return sorted(os.listdir(self._real(path)))

    def copy_out(self, path: str, dest: Path) -> int:
        if self.gone_after is not None and len(self.copied) >= self.gone_after:
            raise SandboxGone("ended")
        self.copied.append(path)
        with self._real(path).open("rb") as source, dest.open("wb") as out:
            shutil.copyfileobj(source, out)
        return dest.stat().st_size


class _ListingTree(_Tree):
    def list_entries(self, path: str) -> list[DirEntry]:
        entries = []
        for name in self.list_dir(path):
            facts = os.lstat(self._real(path) / name)
            if stat.S_ISLNK(facts.st_mode):
                kind = "link"
            elif stat.S_ISDIR(facts.st_mode):
                kind = "dir"
            else:
                kind = "file"
            entries.append(DirEntry(name, kind, facts.st_size))  # type: ignore[arg-type]
        return entries


def _write(base: Path, relative: str, data: bytes) -> None:
    path = base / "out" / "files" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _index(output: Path) -> dict:
    return json.loads((output / INDEX_FILE).read_text())


def test_nested_logs_and_documents_keep_their_relative_paths_and_traces_stay_behind(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    tree = {
        "server.log": b"top\n",
        "variant_00_test/suite_run_0/server.log": b"boot\n",
        "variant_00_test/suite_run_0/client/bench.log": b"client\n",
        "variant_00_test/suite_run_0/phase_marks.json": b"{}\n",
        "variant_00_test/suite_run_0/launch.json": b"{}\n",
        "variant_00_test/suite_run_0/server_boot_active_at": b"1.0\n",
        "variant_00_test/suite_run_0/torch_trace/rank0.json": b"{}\n",
        "variant_00_test/suite_run_0/model.bin": b"weights",
    }
    for relative, data in tree.items():
        _write(base, relative, data)
    save_runner_files(_ListingTree(base), ROOT, output)
    assert (output / "hosted-server.log").read_bytes() == b"top\n", "a top-level log keeps the name it always had"
    nested = output / "hosted-variant_00_test" / "suite_run_0"
    for name in ("server.log", "client/bench.log", "phase_marks.json", "launch.json", "server_boot_active_at"):
        assert (nested / name).read_bytes() == tree[f"variant_00_test/suite_run_0/{name}"], name
    assert not (nested / "torch_trace").exists() and not (nested / "model.bin").exists()
    saved = {entry["path"] for entry in _index(output)["saved"]}
    assert saved == {relative for relative in tree if "torch_trace" not in relative and "model.bin" not in relative}


def test_a_link_is_never_followed_out_of_the_tree(tmp_path: Path) -> None:
    base, output, outside = tmp_path / "sandbox", tmp_path / "attempt", tmp_path / "outside"
    _write(base, "round/server.log", b"ours\n")
    outside.mkdir()
    (outside / "secret.log").write_bytes(b"secret\n")
    (base / "out" / "files" / "round" / "dir.log").symlink_to(outside, target_is_directory=True)
    (base / "out" / "files" / "round" / "file.log").symlink_to(outside / "secret.log")
    tree = _ListingTree(base)
    save_runner_files(tree, ROOT, output)
    assert tree.copied == [f"{ROOT}/round/server.log"]
    assert not list(output.rglob("secret.log")) and not (output / "hosted-round" / "file.log").exists()
    skipped = {entry["path"]: entry["reason"] for entry in _index(output)["skipped"]}
    assert skipped == {"round/dir.log": "a link, not followed", "round/file.log": "a link, not followed"}


def test_a_long_file_keeps_its_tail_after_a_line_that_says_it_was_cut(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "round/server.log", b"head\n" + b"x" * 100 + b"\nthe last line\n")
    save_runner_files(_ListingTree(base), ROOT, output, SaveLimits(file_bytes=20))
    kept = (output / "hosted-round" / "server.log").read_bytes()
    assert kept == b"[hyperloom: truncated, kept the last 20 of 120 bytes]\n" + b"xxxxx\nthe last line\n"
    assert _index(output)["truncated"] == [{"path": "round/server.log", "size": 120, "kept_bytes": 20}]


def test_the_total_bound_leaves_later_files_behind_and_logs_go_first(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a.json", b"j" * 10)
    _write(base, "round/server.log", b"s" * 10)
    _write(base, "z.log", b"z" * 10)
    save_runner_files(_ListingTree(base), ROOT, output, SaveLimits(total_bytes=25))
    index = _index(output)
    assert [entry["path"] for entry in index["saved"]] == ["round/server.log", "z.log"]
    assert index["skipped"] == [{"path": "a.json", "size": 10, "reason": "total_bytes reached"}]
    assert index["saved_bytes"] == 20


def test_a_file_too_large_to_fetch_is_listed_not_copied(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "big.log", b"b" * 50)
    tree = _ListingTree(base)
    save_runner_files(tree, ROOT, output, SaveLimits(file_bytes=10, fetch_bytes=40))
    assert tree.copied == [] and not (output / "hosted-big.log").exists()
    assert _index(output)["skipped"] == [{"path": "big.log", "size": 50, "reason": "larger than fetch_bytes"}]


def test_the_walk_stops_at_its_depth_and_entry_bounds(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a/b/c/deep.log", b"deep\n")
    _write(base, "a/shallow.log", b"shallow\n")
    save_runner_files(_ListingTree(base), ROOT, output, SaveLimits(depth=2))
    index = _index(output)
    assert [entry["path"] for entry in index["saved"]] == ["a/shallow.log"]
    assert {"path": "a/b/c", "reason": "deeper than the depth limit"} in index["skipped"]
    save_runner_files(_ListingTree(base), ROOT, tmp_path / "again", SaveLimits(entries=1))
    assert _index(tmp_path / "again")["listing_truncated"] == "stopped after 1 entries"


def test_a_runner_file_cannot_take_the_name_of_the_index_or_the_progress(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "saved-files.json", b"forged")
    _write(base, "progress.json", b"forged")
    save_runner_files(_ListingTree(base), ROOT, output)
    assert json.loads((output / INDEX_FILE).read_text())["saved"] == []
    assert not (output / PROGRESS_FILE).exists()


def test_a_sandbox_that_ends_mid_copy_still_leaves_the_index_of_what_was_kept(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a.log", b"a\n")
    _write(base, "b.log", b"b\n")
    tree = _ListingTree(base)
    tree.gone_after = 1
    with pytest.raises(SandboxGone):
        save_runner_files(tree, ROOT, output)
    assert [entry["path"] for entry in _index(output)["saved"]] == ["a.log"]
    assert not (output / "hosted-b.log").exists()


def test_a_sandbox_that_cannot_tell_a_link_keeps_only_the_top_level_files(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "server.log", b"top\n")
    _write(base, "round/server.log", b"nested\n")
    save_runner_files(_Tree(base), ROOT, output)
    assert (output / "hosted-server.log").read_bytes() == b"top\n"
    assert not (output / "hosted-round").exists()


def test_a_missing_files_root_is_a_missing_file(tmp_path: Path) -> None:
    (tmp_path / "sandbox" / "out").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        save_runner_files(_ListingTree(tmp_path / "sandbox"), ROOT, tmp_path / "attempt")


def test_the_last_progress_read_is_written_with_when_it_was_read(tmp_path: Path) -> None:
    seen = SeenProgress()
    seen.note("req-1", None, 5.0)
    assert seen.get("req-1") is None
    seen.note("req-1", {"phase": "running", "server_ready": False, "last_log_age_sec": 3.0}, 100.0)
    write_progress(tmp_path, seen.get("req-1"), 160.0)
    document = json.loads((tmp_path / PROGRESS_FILE).read_text())
    assert document == {
        "progress": {"phase": "running", "server_ready": False, "last_log_age_sec": 3.0},
        "read_unix": 100.0,
        "read_sec_ago": 60.0,
    }
    seen.forget("req-1")
    write_progress(tmp_path, seen.get("req-1"), 170.0)
    assert json.loads((tmp_path / PROGRESS_FILE).read_text())["progress"] is None
