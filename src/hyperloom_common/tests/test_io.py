# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_common.io import (
    DURABLE,
    JsonFormat,
    JsonLineFormat,
    WriteOptions,
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    fsync_tree,
    fsync_tree_directories,
)


def _tree(root: Path) -> Path:
    (root / "nested" / "deeper").mkdir(parents=True)
    (root / "top.json").write_text("{}")
    (root / "nested" / "mid.txt").write_text("mid")
    (root / "nested" / "deeper" / "leaf.txt").write_text("leaf")
    return root


def _opened_under(root: Path, walk: Callable[[Path], None]) -> set[str]:
    """The paths under ``root`` that ``walk(root)`` opened.

    ``os.open`` is patched for the whole process, so a finalizer or thread left over from another test can open its own
    paths while the walk runs; only the paths under ``root`` are the walk's.
    """
    opened: list[str] = []
    real_open = os.open

    def _record(path, flags, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "open", _record)
        walk(root)
    return {path for path in opened if Path(path).is_relative_to(root)}


def test_a_durable_write_replaces_prior_content_in_one_step(tmp_path):
    target = tmp_path / "out" / "result.json"
    atomic_write_text(target, '{"a": 1}', options=DURABLE)
    atomic_write_bytes(target, b'{"a": 2}', options=DURABLE)

    assert target.read_text() == '{"a": 2}'
    assert [p.name for p in target.parent.iterdir()] == ["result.json"]


def test_a_durable_json_write_publishes_one_stable_layout(tmp_path):
    target = tmp_path / "out" / "state.json"
    atomic_write_json(
        target, {"b": [1, 2], "a": {"z": 1, "y": 2}}, fmt=JsonFormat(trailing_newline=True), options=DURABLE
    )

    assert target.read_text() == '{\n  "a": {\n    "y": 2,\n    "z": 1\n  },\n  "b": [\n    1,\n    2\n  ]\n}\n'
    assert [p.name for p in target.parent.iterdir()] == ["state.json"]


@pytest.mark.parametrize("options", [WriteOptions(fsync=True), WriteOptions(fsync_dir=True)], ids=["file", "directory"])
def test_a_failed_fsync_fails_the_write_it_was_asked_to_make_durable(tmp_path, monkeypatch, options):
    target = tmp_path / "state.json"
    target.write_text("old")

    def _fail(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", _fail)
    with pytest.raises(OSError, match="fsync failed"):
        atomic_write_text(target, "new", options=options)
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_fsync_tree_visits_every_file_and_directory(tmp_path):
    root = _tree(tmp_path / "bundle")

    assert _opened_under(root, fsync_tree) == {
        str(root),
        str(root / "top.json"),
        str(root / "nested"),
        str(root / "nested" / "mid.txt"),
        str(root / "nested" / "deeper"),
        str(root / "nested" / "deeper" / "leaf.txt"),
    }


def test_fsync_tree_directories_skips_the_files(tmp_path):
    root = _tree(tmp_path / "bundle")

    assert _opened_under(root, fsync_tree_directories) == {
        str(root),
        str(root / "nested"),
        str(root / "nested" / "deeper"),
    }


@pytest.mark.parametrize("walk_tree", [fsync_tree, fsync_tree_directories])
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1000)() == 0, reason="root can enumerate a directory with no read bit"
)
def test_a_directory_that_cannot_be_enumerated_fails_the_durability_claim(tmp_path, walk_tree):
    """The caller renames this tree into place next, so a skipped subtree is data loss."""
    root = _tree(tmp_path / "bundle")
    unreadable = root / "nested"
    original_mode = stat.S_IMODE(unreadable.stat().st_mode)
    unreadable.chmod(0o000)
    try:
        with pytest.raises(OSError):
            walk_tree(root)
    finally:
        unreadable.chmod(original_mode)


def test_atomic_write_text_fsync(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write_text(p, "hello", options=WriteOptions(make_parents=True, fsync=True))
    assert p.read_text() == "hello"


def test_atomic_write_json_fsync(tmp_path):
    p = tmp_path / "f.json"
    atomic_write_json(p, {"b": 1, "a": 2}, options=WriteOptions(make_parents=True, fsync=True))
    assert json.loads(p.read_text()) == {"a": 2, "b": 1}
    assert p.read_text().index('"a"') < p.read_text().index('"b"')


def test_atomic_write_json_ensure_ascii_false_and_mode(tmp_path):
    p = tmp_path / "unicode.json"
    atomic_write_json(
        p,
        {"msg": "café"},
        fmt=JsonFormat(sort_keys=False, ensure_ascii=False),
        options=WriteOptions(make_parents=True, mode=0o640),
    )
    assert p.read_text(encoding="utf-8") == '{\n  "msg": "café"\n}'
    # Group/other bits stripped: 0o640 clamped to owner-only.
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_atomic_write_text_mode(tmp_path):
    p = tmp_path / "mode.txt"
    atomic_write_text(p, "x", options=WriteOptions(mode=0o600))
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_atomic_write_text_preserve_mode_keeps_existing_bits(tmp_path):
    p = tmp_path / "script.sh"
    p.write_text("old")
    p.chmod(0o755)
    atomic_write_text(p, "new", options=WriteOptions(preserve_mode=True))
    assert p.read_text() == "new"
    assert stat.S_IMODE(p.stat().st_mode) == 0o755


def test_atomic_write_bytes_preserve_mode_new_file_is_owner_only(tmp_path):
    p = tmp_path / "fresh.bin"
    atomic_write_bytes(p, b"x", options=WriteOptions(preserve_mode=True))
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_atomic_write_bytes_mode_wins_over_preserve_mode(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"old")
    p.chmod(0o644)
    atomic_write_bytes(p, b"new", options=WriteOptions(mode=0o600, preserve_mode=True))
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


class TestAppendJsonl:
    def test_appends_lines(self, tmp_path):
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"i": 1})
        append_jsonl(p, {"i": 2})
        lines = p.read_text().splitlines()
        assert lines == ['{"i": 1}', '{"i": 2}']

    def test_make_parents(self, tmp_path):
        p = tmp_path / "deep" / "log.jsonl"
        append_jsonl(p, {"x": 1}, make_parents=True)
        assert p.exists()

    def test_fsync(self, tmp_path):
        p = tmp_path / "log.jsonl"
        append_jsonl(p, [1, 2], fsync=True)
        assert p.read_text() == "[1, 2]\n"

    def test_line_format(self, tmp_path):
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"b": 1, "a": "café"}, fmt=JsonLineFormat(ensure_ascii=False, sort_keys=True))
        assert p.read_text(encoding="utf-8") == '{"a": "café", "b": 1}\n'
