# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for _io_utils.py shared helpers."""

from __future__ import annotations

from hyperloom.orchestrator.trace_analysis import _io_utils as io


def test_append_log_creates_parents_and_appends(tmp_path):
    log = tmp_path / "logs" / "run.log"
    io.append_log(log, "first  ")
    io.append_log(log, "second")
    assert log.read_text(encoding="utf-8") == "first\nsecond\n"


def test_write_text_creates_parents(tmp_path):
    path = tmp_path / "deep" / "out.txt"
    io.write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_read_last_lines_missing_returns_empty(tmp_path):
    assert io.read_last_lines(tmp_path / "nope.log") == []


def test_read_last_lines_limit(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(str(i) for i in range(10)), encoding="utf-8")
    assert io.read_last_lines(log, limit=3) == ["7", "8", "9"]
