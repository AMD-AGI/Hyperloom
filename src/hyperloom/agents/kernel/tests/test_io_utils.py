# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for _io_utils.py shared helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        (" On ", True),
        ("0", False),
        ("no", False),
        ("", False),
        (None, False),
    ],
)
def test_truthy_variants(value, expected):
    assert io.truthy(value) is expected


# ---------------------------------------------------------------------------
# Byte-consistency contract: the standalone kernel-agent ``_io_utils`` mirror
# must stay behaviourally aligned with ``hyperloom.common`` for the primitives
# it duplicates (the tools cannot import ``common`` at runtime, so this test is
# the guard that the two copies do not drift).
# ---------------------------------------------------------------------------


def test_truthy_matches_common_env_bool_vocabulary():
    from hyperloom.common.env import _TRUE_TOKENS

    # ``truthy`` accepts exactly the ``env_bool`` true-token vocabulary for
    # string inputs (case-insensitive, stripped).
    for token in _TRUE_TOKENS:
        assert io.truthy(token) is True
        assert io.truthy(token.upper()) is True
    assert io.truthy("maybe") is False


def test_atomic_write_json_bytes_match_common(tmp_path):
    from hyperloom.common.io import atomic_write_json as common_write

    payload = {"b": 2, "a": 1, "nested": {"y": 2, "x": 1}}
    kernel_path = tmp_path / "kernel.json"
    common_path = tmp_path / "common.json"
    io.atomic_write_json(kernel_path, payload)
    # ``_io_utils`` writes indent=2 + sort_keys + trailing newline.
    common_write(common_path, payload, indent=2, sort_keys=True, trailing_newline=True)
    assert kernel_path.read_bytes() == common_path.read_bytes()


def test_safe_float_matches_common_coerce_for_shared_cases():
    from hyperloom.common.coerce import to_float

    # For the inputs both accept, results agree (both reject bool -> default).
    for value in ("1.5", 3, "bad", None, ""):
        assert io.safe_float(value, default=0.0) == to_float(value, default=0.0)
