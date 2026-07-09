# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for _io_utils.py shared helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


def _load_module():
    """Load _io_utils.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "io_utils_under_test",
        _TOOLS_DIR / "_io_utils.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


io = _load_module()


def test_utc_now_is_iso8601_utc():
    value = io.utc_now()
    assert value.endswith("+00:00")
    assert "T" in value


def test_atomic_write_json_creates_parents_and_roundtrips(tmp_path):
    path = tmp_path / "nested" / "out.json"
    io.atomic_write_json(path, {"b": 2, "a": 1})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"a": 1, "b": 2}


def test_append_log_creates_parents_and_appends(tmp_path):
    log = tmp_path / "logs" / "run.log"
    io.append_log(log, "first  ")
    io.append_log(log, "second")
    assert log.read_text(encoding="utf-8") == "first\nsecond\n"


def test_read_last_lines_missing_returns_empty(tmp_path):
    assert io.read_last_lines(tmp_path / "nope.log") == []


def test_read_last_lines_limit(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(str(i) for i in range(10)), encoding="utf-8")
    assert io.read_last_lines(log, limit=3) == ["7", "8", "9"]


def test_kernel_row_matches_empty_target_matches_any():
    assert io.kernel_row_matches({"name": "x"}, "") is True


def test_kernel_row_matches_by_matched_name_and_name():
    assert io.kernel_row_matches({"matched_kernel_name": "k"}, "k") is True
    assert io.kernel_row_matches({"name": "k"}, "k") is True
    assert io.kernel_row_matches({"name": "other"}, "k") is False


def test_safe_float_variants():
    assert io.safe_float(None) == 0.0
    assert io.safe_float("") == 0.0
    assert io.safe_float("1.5") == 1.5
    assert io.safe_float(3) == 3.0
    assert io.safe_float("bad", default=-1.0) == -1.0


def test_source_text_looks_complete_python():
    assert io.source_text_looks_complete("import torch\n", ".py") is True
    # Valid syntax but no top-level marker -> rejected.
    assert io.source_text_looks_complete("x = 1\n", ".py") is False
    # Syntax error -> rejected.
    assert io.source_text_looks_complete("def (:\n", ".py") is False


def test_source_text_looks_complete_compiled_and_rejections():
    assert io.source_text_looks_complete("#include <cuda.h>\n", ".cu") is True
    # Fenced text rejected regardless of suffix.
    assert io.source_text_looks_complete("```\n#include <x>\n```", ".cu") is False
    # Empty rejected.
    assert io.source_text_looks_complete("   ", ".cu") is False
    # Unknown suffix rejected.
    assert io.source_text_looks_complete("void f(){}", ".txt") is False
    # Compiled suffix without any marker rejected.
    assert io.source_text_looks_complete("just some prose", ".cpp") is False
