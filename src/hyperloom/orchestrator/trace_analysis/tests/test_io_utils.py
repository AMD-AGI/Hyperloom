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


def test_safe_float_variants():
    assert io.safe_float(None) == 0.0
    assert io.safe_float("") == 0.0
    assert io.safe_float(True, default=None) is None
    assert io.safe_float("1.5") == 1.5
    assert io.safe_float(3) == 3.0
    assert io.safe_float("bad", default=-1.0) == -1.0
    assert io.safe_float("1,234.5%", default=None, strip_percent=True, strip_commas=True) == 1234.5


# Byte-consistency contract: the kernel-agent ``_io_utils`` mirror must stay behaviourally aligned with
# ``hyperloom.common`` for the primitives it duplicates.


def test_safe_float_matches_common_coerce_for_shared_cases():
    from hyperloom.common.coerce import to_float

    for value in ("1.5", 3, "bad", None, "", True):
        assert io.safe_float(value, default=0.0) == to_float(value, default=0.0)
