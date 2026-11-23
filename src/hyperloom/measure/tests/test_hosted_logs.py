# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted attempt keeps of its runner's files: nested logs at their paths, within bounds, never through a link."""

from __future__ import annotations

import json
import os
import shutil
import stat
from types import SimpleNamespace
from pathlib import Path

import pytest

from hyperloom.measure import hosted_logs
from hyperloom.measure.hosted_logs import INDEX_FILE, PROGRESS_FILE, SeenProgress, save_runner_files, write_progress
from hyperloom.sandbox.socket import DirEntry, SandboxGone

ROOT = "/out/files"


class _Tree:
    """A runner's ``/out`` as a directory on disk, read the way a sandbox reads it (``list_entries`` and ``copy_out``)."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.copied: list[str] = []
        self.gone_after: int | None = None
        self.grows: dict[str, bytes] = {}  # appended to a file after it was listed, before it is copied
        self.on_copy = lambda: None

    def _real(self, path: str) -> Path:
        return self.base / path.lstrip("/")

    def list_entries(self, path: str) -> list[DirEntry]:
        entries = []
        for name in sorted(os.listdir(self._real(path))):
            facts = os.lstat(self._real(path) / name)
            if stat.S_ISLNK(facts.st_mode):
                kind = "link"
            elif stat.S_ISDIR(facts.st_mode):
                kind = "dir"
            else:
                kind = "file" if stat.S_ISREG(facts.st_mode) else "other"
            entries.append(DirEntry(name, kind, facts.st_size))  # type: ignore[arg-type]
        return entries

    def copy_out(self, path: str, dest: Path) -> int:
        if self.gone_after is not None and len(self.copied) >= self.gone_after:
            raise SandboxGone("ended")
        self.copied.append(path)
        with self._real(path).open("ab") as growing:
            growing.write(self.grows.get(path, b""))
        with self._real(path).open("rb") as source, dest.open("wb") as out:
            shutil.copyfileobj(source, out)
        self.on_copy()
        return dest.stat().st_size


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
    save_runner_files(_Tree(base), ROOT, output)
    assert (output / "hosted-server.log").read_bytes() == b"top\n", "a top-level log keeps the name it always had"
    nested = output / "hosted-variant_00_test" / "suite_run_0"
    for name in ("server.log", "client/bench.log", "phase_marks.json", "launch.json", "server_boot_active_at"):
        assert (nested / name).read_bytes() == tree[f"variant_00_test/suite_run_0/{name}"], name
    assert not (nested / "torch_trace").exists() and not (nested / "model.bin").exists()
    saved = {entry["path"] for entry in _index(output)["saved"]}
    assert saved == {relative for relative in tree if "torch_trace" not in relative and "model.bin" not in relative}


def test_the_index_lists_trace_directories_other_files_and_special_files_left_behind(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "round/server.log", b"boot\n")
    _write(base, "round/core.bin", b"core")
    _write(base, "round/torch_trace/rank0.json", b"{}\n")
    os.mkfifo(base / "out" / "files" / "round" / "pipe.log")
    save_runner_files(_Tree(base), ROOT, output)
    assert _index(output)["skipped"] == [
        {"path": "round/core.bin", "size": 4, "reason": "not a diagnostic file"},
        {"path": "round/pipe.log", "reason": "not a regular file"},
        {"path": "round/torch_trace", "reason": "a profiler trace directory"},
    ]


def test_a_link_is_never_followed_out_of_the_tree(tmp_path: Path) -> None:
    base, output, outside = tmp_path / "sandbox", tmp_path / "attempt", tmp_path / "outside"
    _write(base, "round/server.log", b"ours\n")
    outside.mkdir()
    (outside / "secret.log").write_bytes(b"secret\n")
    (base / "out" / "files" / "round" / "dir.log").symlink_to(outside, target_is_directory=True)
    (base / "out" / "files" / "round" / "file.log").symlink_to(outside / "secret.log")
    tree = _Tree(base)
    save_runner_files(tree, ROOT, output)
    assert tree.copied == [f"{ROOT}/round/server.log"]
    assert not list(output.rglob("secret.log")) and not (output / "hosted-round" / "file.log").exists()
    skipped = {entry["path"]: entry["reason"] for entry in _index(output)["skipped"]}
    assert skipped == {"round/dir.log": "a link, not followed", "round/file.log": "a link, not followed"}


def test_a_long_file_keeps_its_tail_after_a_line_that_says_it_was_cut(tmp_path: Path, monkeypatch) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "round/server.log", b"head\n" + b"x" * 100 + b"\nthe last line\n")
    monkeypatch.setattr(hosted_logs, "FILE_BYTES", 20)
    save_runner_files(_Tree(base), ROOT, output)
    kept = (output / "hosted-round" / "server.log").read_bytes()
    assert kept == b"[hyperloom: truncated, kept the last 20 of 120 bytes]\n" + b"xxxxx\nthe last line\n"
    assert _index(output)["truncated"] == [{"path": "round/server.log", "size": 120, "kept_bytes": 20}]


def test_the_total_bound_leaves_later_files_behind_and_server_logs_go_first(tmp_path: Path, monkeypatch) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a.json", b"j" * 10)
    _write(base, "round/server.log", b"s" * 10)
    _write(base, "z.log", b"z" * 10)
    monkeypatch.setattr(hosted_logs, "TOTAL_BYTES", 25)
    save_runner_files(_Tree(base), ROOT, output)
    index = _index(output)
    assert [entry["path"] for entry in index["saved"]] == ["round/server.log", "a.json"]
    assert index["skipped"] == [{"path": "z.log", "size": 10, "reason": "total_bytes reached"}]
    assert index["saved_bytes"] == 20


def test_a_failing_rounds_server_log_is_kept_when_earlier_rounds_client_logs_would_fill_the_total(
    tmp_path: Path, monkeypatch
) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "variant_00_test/suite_run_0/client/bench.log", b"0" * 10)
    _write(base, "variant_00_test/suite_run_1/client/bench.log", b"1" * 10)
    _write(base, "variant_00_test/suite_run_2/server.log", b"the failing round\n")
    monkeypatch.setattr(hosted_logs, "TOTAL_BYTES", 30)
    save_runner_files(_Tree(base), ROOT, output)
    assert (output / "hosted-variant_00_test" / "suite_run_2" / "server.log").read_bytes() == b"the failing round\n"
    index = _index(output)
    assert [entry["path"] for entry in index["saved"]] == [
        "variant_00_test/suite_run_2/server.log",
        "variant_00_test/suite_run_0/client/bench.log",
    ]
    assert [entry["path"] for entry in index["skipped"]] == ["variant_00_test/suite_run_1/client/bench.log"]


def test_a_file_that_grew_since_its_listing_is_cut_to_what_is_left_of_the_total(tmp_path: Path, monkeypatch) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a.log", b"a" * 10)
    _write(base, "b.log", b"b" * 4)
    tree = _Tree(base)
    tree.grows[f"{ROOT}/b.log"] = b"B" * 6
    monkeypatch.setattr(hosted_logs, "TOTAL_BYTES", 15)
    save_runner_files(tree, ROOT, output)
    index = _index(output)
    assert index["truncated"] == [{"path": "b.log", "size": 10, "kept_bytes": 5}]
    assert index["saved_bytes"] == 15
    assert (output / "hosted-b.log").read_bytes() == b"[hyperloom: truncated, kept the last 5 of 10 bytes]\nBBBBB"


def test_a_file_too_large_to_fetch_is_listed_not_copied(tmp_path: Path, monkeypatch) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "big.log", b"b" * 50)
    tree = _Tree(base)
    monkeypatch.setattr(hosted_logs, "FILE_BYTES", 10)
    monkeypatch.setattr(hosted_logs, "FETCH_BYTES", 40)
    save_runner_files(tree, ROOT, output)
    assert tree.copied == [] and not (output / "hosted-big.log").exists()
    assert _index(output)["skipped"] == [{"path": "big.log", "size": 50, "reason": "larger than fetch_bytes"}]


def test_the_walk_stops_at_its_depth_and_entry_bounds(tmp_path: Path, monkeypatch) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a/b/c/deep.log", b"deep\n")
    _write(base, "a/shallow.log", b"shallow\n")
    monkeypatch.setattr(hosted_logs, "DEPTH", 2)
    save_runner_files(_Tree(base), ROOT, output)
    index = _index(output)
    assert [entry["path"] for entry in index["saved"]] == ["a/shallow.log"]
    assert {"path": "a/b/c", "reason": "deeper than the depth limit"} in index["skipped"]
    monkeypatch.setattr(hosted_logs, "ENTRIES", 1)
    save_runner_files(_Tree(base), ROOT, tmp_path / "again")
    assert _index(tmp_path / "again")["listing_truncated"] == "stopped after 1 entries"


def test_a_runner_file_cannot_take_the_name_of_the_index_or_the_progress(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "saved-files.json", b"forged")
    _write(base, "progress.json", b"forged")
    save_runner_files(_Tree(base), ROOT, output)
    assert json.loads((output / INDEX_FILE).read_text())["saved"] == []
    assert not (output / PROGRESS_FILE).exists()


def test_a_sandbox_that_ends_mid_copy_still_leaves_the_index_of_what_was_kept(tmp_path: Path) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    _write(base, "a.log", b"a\n")
    _write(base, "b.log", b"b\n")
    tree = _Tree(base)
    tree.gone_after = 1
    with pytest.raises(SandboxGone):
        save_runner_files(tree, ROOT, output)
    assert [entry["path"] for entry in _index(output)["saved"]] == ["a.log"]
    assert not (output / "hosted-b.log").exists()


def test_the_save_stops_at_its_deadline_and_still_lists_what_it_kept(tmp_path: Path, monkeypatch) -> None:
    base, output = tmp_path / "sandbox", tmp_path / "attempt"
    for name in ("server.log", "b.log", "c.log"):
        _write(base, name, b"x\n")
    clock = [1000.0]
    monkeypatch.setattr(hosted_logs, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(hosted_logs, "DEADLINE_SEC", 120.0)
    tree = _Tree(base)
    tree.on_copy = lambda: clock.__setitem__(0, clock[0] + 121.0)  # the first copy takes the whole deadline
    save_runner_files(tree, ROOT, output)
    index = _index(output)
    assert tree.copied == [f"{ROOT}/server.log"]
    assert [entry["path"] for entry in index["saved"]] == ["server.log"]
    assert index["stopped"] == "stopped at the 120 s deadline; 2 files found were not copied"


def test_a_missing_files_root_is_a_missing_file(tmp_path: Path) -> None:
    (tmp_path / "sandbox" / "out").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        save_runner_files(_Tree(tmp_path / "sandbox"), ROOT, tmp_path / "attempt")


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
