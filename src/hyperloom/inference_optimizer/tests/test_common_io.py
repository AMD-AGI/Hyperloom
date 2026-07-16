# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import stat

from hyperloom.common.io import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
)


def test_atomic_write_text_fsync(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write_text(p, "hello", make_parents=True, fsync=True)
    assert p.read_text() == "hello"


def test_atomic_write_json_fsync(tmp_path):
    p = tmp_path / "f.json"
    atomic_write_json(p, {"b": 1, "a": 2}, fsync=True)
    assert json.loads(p.read_text()) == {"a": 2, "b": 1}
    assert p.read_text().index('"a"') < p.read_text().index('"b"')


def test_atomic_write_json_ensure_ascii_false_and_mode(tmp_path):
    p = tmp_path / "unicode.json"
    atomic_write_json(
        p,
        {"msg": "café"},
        ensure_ascii=False,
        sort_keys=False,
        mode=0o640,
    )
    assert p.read_text(encoding="utf-8") == '{\n  "msg": "café"\n}'
    # Group/other bits stripped: 0o640 clamped to owner-only.
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
