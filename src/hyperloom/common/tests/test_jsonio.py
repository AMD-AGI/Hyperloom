# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for hyperloom.common.jsonio."""

from __future__ import annotations

from hyperloom.common import jsonio


def test_iter_sse_objects_skips_malformed_and_non_data_events() -> None:
    raw = 'not json\n\ndata: {bad}\n\ndata: {"id":"1","result":{"ok":true}}\n\n'
    assert list(jsonio.iter_sse_objects(raw)) == [{"id": "1", "result": {"ok": True}}]


def test_iter_sse_plain_json() -> None:
    assert list(jsonio.iter_sse_objects('{"a": 1}')) == [{"a": 1}]
