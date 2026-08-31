###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the streaming Kineto reader primitives (_trace_reader).

Exercises ``stream_events`` (and its transparent-gzip open path) against
hand-authored payloads: truncation recovery, malformed-object resync, UTF-8
chunk-boundary handling, and the prefix/per-event size caps.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _trace_reader as reader  # noqa: E402


def test_truncated_gzip_records_eof_error_without_raising():
    """A gzip EOF must be reported through the stream error channel."""
    payload = json.dumps(
        {
            "traceEvents": [
                {"cat": "kernel", "name": "first"},
                {"cat": "kernel", "name": "second-" + "x" * 4096},
            ]
        }
    ).encode("utf-8")
    compressed = gzip.compress(payload)
    errors: list[str] = []
    with gzip.GzipFile(fileobj=io.BytesIO(compressed[:-32]), mode="rb") as fh:
        list(reader.stream_events(fh, bufsize=32, errors=errors))
    assert any("EOFError" in error for error in errors)


def test_bad_gzip_records_error_without_raising():
    """An invalid gzip header must fail soft like a truncated stream."""
    errors: list[str] = []
    with gzip.GzipFile(fileobj=io.BytesIO(b"not-a-gzip-stream"), mode="rb") as fh:
        assert list(reader.stream_events(fh, bufsize=8, errors=errors)) == []
    assert any("BadGzipFile" in error for error in errors)


def test_trace_events_null_does_not_capture_a_later_array():
    """A non-array traceEvents value must not redirect parsing elsewhere."""
    payload = b'{"traceEvents": null, "other": [{"cat": "kernel", "name": "wrong-array"}]}'
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=16, errors=errors)) == []
    assert errors == ["traceEvents value is not an array"]


def test_trace_events_text_value_before_key_is_ignored():
    """A string value named traceEvents must not shadow the real member."""
    good = {"cat": "kernel", "name": "real-array"}
    payload = json.dumps({"label": "traceEvents", "traceEvents": [good]}).encode("utf-8")
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=9, errors=errors)) == [good]
    assert errors == []


def test_utf8_character_split_across_chunks_is_preserved():
    """Incremental decoding must preserve a split multibyte code point."""
    marker = chr(0x20AC)
    expected = "kernel-" + marker + "-suffix"
    payload = json.dumps(
        {"traceEvents": [{"cat": "kernel", "name": expected}]},
        ensure_ascii=False,
    ).encode("utf-8")
    marker_start = payload.index(marker.encode("utf-8"))
    errors: list[str] = []
    events = list(
        reader.stream_events(
            io.BytesIO(payload),
            bufsize=marker_start + 1,
            errors=errors,
        )
    )
    assert events[0]["name"] == expected
    assert errors == []


def test_complete_malformed_object_resyncs_to_later_event():
    """A balanced bad object must not hide a later valid event."""
    good = {"cat": "kernel", "name": "recovered"}
    payload = b'{"traceEvents": [{"cat": "kernel", "name": invalid},' + json.dumps(good).encode("utf-8") + b"]}"
    errors: list[str] = []
    events = list(reader.stream_events(io.BytesIO(payload), bufsize=11, errors=errors))
    assert events == [good]
    assert len(errors) == 1
    assert "malformed after 0 event(s)" in errors[0]


def test_trace_prefix_growth_is_bounded(monkeypatch):
    """A missing late traceEvents key must not grow the prefix indefinitely."""
    monkeypatch.setattr(reader, "_MAX_TRACE_PREFIX_CHARS", 32)
    payload = b'{"padding":"' + b"x" * 64 + b'","traceEvents":[]}'
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=8, errors=errors)) == []
    assert len(errors) == 1
    assert "prefix exceeds" in errors[0]


def test_single_event_growth_is_bounded(monkeypatch):
    """An unclosed oversized object must stop at the per-event limit."""
    monkeypatch.setattr(reader, "_MAX_EVENT_CHARS", 64)
    payload = b'{"traceEvents":[{"cat":"kernel","name":"' + b"x" * 128
    errors: list[str] = []
    assert list(reader.stream_events(io.BytesIO(payload), bufsize=8, errors=errors)) == []
    assert len(errors) == 1
    assert "object exceeds" in errors[0]


def test_trace_events_opener_may_cross_a_read_boundary():
    """A chunk ending after the key must refill instead of raising ValueError."""
    payload = b'{"traceEvents"  : [{"cat": "kernel"}]}'
    errors: list[str] = []
    events = list(
        reader.stream_events(
            io.BytesIO(payload),
            bufsize=len('{"traceEvents"'),
            errors=errors,
        )
    )
    assert events == [{"cat": "kernel"}]
    assert errors == []


def test_gzip_open_is_transparent(tmp_path):
    """``_open_trace_binary`` must decompress a ``.gz`` trace transparently."""
    good = {"cat": "kernel", "name": "gz-kernel"}
    tf = tmp_path / "trace.json.gz"
    with gzip.open(tf, "wb") as fh:
        fh.write(json.dumps({"traceEvents": [good]}).encode("utf-8"))
    with reader._open_trace_binary(tf) as fh:
        assert list(reader.stream_events(fh)) == [good]
