# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import stat

import pytest

from hyperloom.common.io import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
    tail_lines,
)


def test_atomic_write_text_fsync(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write_text(p, "hello", make_parents=True, fsync=True)
    assert p.read_text() == "hello"


def test_atomic_write_bytes_make_parents_and_fsync(tmp_path):
    p = tmp_path / "nested" / "dir" / "blob.bin"
    atomic_write_bytes(p, b"\x00\x01\x02", make_parents=True, fsync=True)
    assert p.read_bytes() == b"\x00\x01\x02"


def test_atomic_write_bytes_mode_is_clamped(tmp_path):
    p = tmp_path / "secret.bin"
    atomic_write_bytes(p, b"data", mode=0o644)
    # Group/other bits stripped down to owner-only.
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_atomic_write_bytes_reraises_and_unlinks_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "boom.bin"

    def _boom(*a, **k):
        raise OSError("replace failed")

    import hyperloom.common.io as io_mod

    monkeypatch.setattr(io_mod.os, "replace", _boom)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_bytes(p, b"x")
    # No temp files left behind in the directory.
    leftovers = [q for q in tmp_path.iterdir() if q.name.startswith(".boom.bin")]
    assert leftovers == []


def test_atomic_write_json_fsync(tmp_path):
    p = tmp_path / "f.json"
    atomic_write_json(p, {"b": 1, "a": 2}, fsync=True)
    assert json.loads(p.read_text()) == {"a": 2, "b": 1}
    # sort_keys default True
    assert p.read_text().index('"a"') < p.read_text().index('"b"')


def test_atomic_write_json_ensure_ascii_false_and_mode(tmp_path):
    p = tmp_path / "unicode.json"
    atomic_write_json(
        p,
        {"msg": "你好"},
        ensure_ascii=False,
        sort_keys=False,
        mode=0o640,
    )
    assert p.read_text(encoding="utf-8") == '{\n  "msg": "你好"\n}'
    # Group/other bits are stripped: a requested 0o640 is clamped to owner-only.
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_atomic_write_text_mode(tmp_path):
    p = tmp_path / "mode.txt"
    atomic_write_text(p, "x", mode=0o600)
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


class TestReadJsonl:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"i": 1})
        append_jsonl(p, {"i": 2})
        assert read_jsonl(p) == [{"i": 1}, {"i": 2}]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text('{"a": 1}\n\n  \n{"b": 2}\n')
        assert read_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_missing_file_default(self, tmp_path):
        assert read_jsonl(tmp_path / "nope.jsonl") == []
        sentinel = ["fallback"]
        assert read_jsonl(tmp_path / "nope.jsonl", default=sentinel) is sentinel

    def test_malformed_raises(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{not json}\n")
        with pytest.raises(json.JSONDecodeError):
            read_jsonl(p)


class TestTailLines:
    def test_tail(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("\n".join(str(i) for i in range(100)) + "\n")
        assert tail_lines(p, 3) == ["97", "98", "99"]

    def test_more_than_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("a\nb\n")
        assert tail_lines(p, 10) == ["a", "b"]

    def test_zero_or_negative(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("a\nb\n")
        assert tail_lines(p, 0) == []
        assert tail_lines(p, -1) == []

    def test_missing_file(self, tmp_path):
        assert tail_lines(tmp_path / "nope.txt", 5) == []
