###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Streaming, low-memory reader for Kineto torch-profiler traces.

The agent trace-launcher path (``_trace_launcher_resolver``) uses these two
primitives to walk a trace's ``traceEvents`` array without loading it whole:

* ``_open_trace_binary`` transparently decompresses ``.gz`` inputs.
* ``stream_events`` parses the array element-by-element with the C-accelerated
  ``json.JSONDecoder.raw_decode`` so peak memory stays flat regardless of trace
  size (no full ``json.load``). Complete leading events remain recoverable from
  a truncated file.
"""

from __future__ import annotations

import codecs
import gzip
import json
from pathlib import Path
from typing import Iterator

_DECODER = json.JSONDecoder()

# Hard caps keep a corrupt trace from turning the streaming reader into an
# unbounded accumulator. Kineto prefixes are normally a few KiB and individual
# events a few KiB, so these retain generous headroom for embedded metadata.
_MAX_TRACE_PREFIX_CHARS = 16 * 1024 * 1024
_MAX_EVENT_CHARS = 64 * 1024 * 1024


def _open_trace_binary(path: Path):
    """Open a trace file, transparently decompressing ``.gz``.

    Returns:
        A binary file object positioned at the start of the (decompressed)
        JSON stream. Caller is responsible for closing it.
    """
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def stream_events(
    fileobj,
    bufsize: int = 8 * 1024 * 1024,
    *,
    errors: list[str] | None = None,
) -> Iterator[dict]:
    """Yield each object inside the ``traceEvents`` array, one at a time.

    Locates the ``traceEvents`` array, then emits balanced ``{...}`` elements
    via ``raw_decode``. Complete leading events remain recoverable from a
    truncated file, while ``errors`` distinguishes that recovery from a clean
    end of the array.

    Args:
        fileobj: A binary, possibly-decompressing file object.
        bufsize: Read/refill chunk size in bytes (also the buffer-trim
            threshold).
        errors: Optional output list for structural stream errors.

    Yields:
        Parsed trace-event dicts.
    """

    def _record(message: str) -> None:
        """Append one structural error when the caller requested diagnostics."""
        if errors is not None:
            errors.append(message)

    chunk_size = max(1, int(bufsize))
    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    eof = False

    def _refill(max_bytes: int | None = None) -> bool:
        """Decode one bounded chunk and turn gzip failures into stream errors."""
        nonlocal buf, eof
        if eof:
            return False
        read_size = chunk_size
        if max_bytes is not None:
            read_size = max(1, min(read_size, max_bytes))
        try:
            chunk = fileobj.read(read_size)
        except (EOFError, gzip.BadGzipFile) as exc:
            _record(f"trace input error: {type(exc).__name__}: {exc}")
            eof = True
            tail = utf8_decoder.decode(b"", final=True)
            if tail:
                buf += tail
            return bool(tail)
        if not chunk:
            eof = True
            tail = utf8_decoder.decode(b"", final=True)
            if tail:
                buf += tail
            return bool(tail)
        buf += utf8_decoder.decode(chunk, final=False)
        return True

    def _trim_consumed() -> None:
        """Discard parsed input once it exceeds one refill chunk."""
        nonlocal buf, pos
        if pos > chunk_size:
            buf = buf[pos:]
            pos = 0

    key = '"traceEvents"'
    search_from = 0
    while True:
        key_pos = buf.find(key, search_from)
        if key_pos < 0:
            if eof:
                _record("traceEvents array not found")
                return
            if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
                _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
                return
            search_from = max(0, len(buf) - len(key) + 1)
            _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
            continue

        pos = key_pos + len(key)
        while pos >= len(buf):
            if eof:
                break
            if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
                _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
                return
            _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
        while pos < len(buf) and buf[pos] in " \t\r\n":
            pos += 1
            if pos == len(buf) and not eof:
                if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
                    _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
                    return
                _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
        if pos < len(buf) and buf[pos] == ":":
            break
        search_from = key_pos + len(key)

    pos += 1
    while True:
        while pos < len(buf) and buf[pos] in " \t\r\n":
            pos += 1
        if pos < len(buf):
            break
        if eof:
            _record("traceEvents array opener not found")
            return
        if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
            _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
            return
        _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
    if buf[pos] != "[":
        _record("traceEvents value is not an array")
        return

    buf = buf[pos + 1 :]
    pos = 0
    emitted = 0
    while True:
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf):
                break
            if eof:
                _record(f"traceEvents array unterminated after {emitted} event(s)")
                return
            _trim_consumed()
            _refill()

        if buf[pos] == "]":
            return

        if buf[pos] != "{":
            invalid_start = pos
            scan = pos
            curly_depth = 0
            square_depth = 0
            in_string = False
            escaped = False
            boundary: int | None = None
            while boundary is None:
                while scan < len(buf):
                    char = buf[scan]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif char == "\\":
                            escaped = True
                        elif char == '"':
                            in_string = False
                    elif char == '"':
                        in_string = True
                    elif char == "{":
                        curly_depth += 1
                    elif char == "}":
                        curly_depth = max(0, curly_depth - 1)
                    elif char == "[":
                        square_depth += 1
                    elif char == "]":
                        if curly_depth == 0 and square_depth == 0:
                            boundary = scan
                            break
                        square_depth = max(0, square_depth - 1)
                    elif char == "," and curly_depth == 0 and square_depth == 0:
                        boundary = scan + 1
                        break
                    scan += 1
                if boundary is not None:
                    break
                buffered = len(buf) - invalid_start
                if buffered >= _MAX_EVENT_CHARS:
                    _record(f"traceEvents element exceeds {_MAX_EVENT_CHARS} characters after {emitted} event(s)")
                    return
                if eof:
                    _record(f"traceEvents array unterminated after {emitted} event(s)")
                    return
                _refill(_MAX_EVENT_CHARS - buffered)
            _record(f"traceEvents element malformed after {emitted} event(s): expected an object")
            pos = boundary
            _trim_consumed()
            continue

        object_start = pos
        scan = pos
        depth = 0
        in_string = False
        escaped = False
        object_end: int | None = None
        while object_end is None:
            while scan < len(buf):
                char = buf[scan]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        object_end = scan + 1
                        break
                scan += 1
            if object_end is not None:
                break
            buffered = len(buf) - object_start
            if buffered >= _MAX_EVENT_CHARS:
                _record(f"traceEvents object exceeds {_MAX_EVENT_CHARS} characters after {emitted} event(s)")
                return
            if eof:
                _record(f"traceEvents object truncated after {emitted} event(s): unterminated object")
                return
            _refill(_MAX_EVENT_CHARS - buffered)

        if object_end - object_start > _MAX_EVENT_CHARS:
            _record(f"traceEvents object exceeds {_MAX_EVENT_CHARS} characters after {emitted} event(s)")
            return
        try:
            obj, decoded_end = _DECODER.raw_decode(buf, object_start)
        except json.JSONDecodeError as exc:
            _record(f"traceEvents object malformed after {emitted} event(s): {exc.msg}")
            pos = object_end
            _trim_consumed()
            continue
        if decoded_end != object_end or not isinstance(obj, dict):
            _record(f"traceEvents object malformed after {emitted} event(s): invalid object boundary")
            pos = object_end
            _trim_consumed()
            continue
        yield obj
        emitted += 1
        pos = object_end
        _trim_consumed()
