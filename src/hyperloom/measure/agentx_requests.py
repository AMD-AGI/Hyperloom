# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a measured AgentX round's per-request export says about the requests its client marked failed.

The replay client (aiperf) writes one JSON line per request to ``aiperf_artifacts/profile_export.jsonl``. The runner's
signed bundle keeps the file like a server log, so what is read here is a file the runner signed, and it can be read
again from the archive. A request the client counted failed has an ``error`` in its line; that is the client's own
decision and no threshold of ours. Each such request is classed by what the server did:

* ``transport``: the server did not serve it (a connection error, a timeout, a cancellation, an HTTP error status or an
  error event in the stream). Every error that is not the verdict below is counted here, so an exception of the client's
  own while it processed a reply that did arrive is too: the export does not tell it from the others;
* ``no content``: the server answered and the client found nothing in the answer to use (aiperf's own verdict, an
  ``InvalidInferenceResultError``). It is ``empty`` when the server produced zero output tokens and ``short`` when it
  produced one or more.

The client as released does not export an output token count for a request it counted failed. The runner patches the
client it runs (``measure.runner.aiperf_patch``) so that the line of a reply it found nothing in carries the server's
count as ``error_osl``: ``empty`` and ``short`` are told apart, exactly, when every no-content failure's line carries
one. Otherwise they are not computed and the result says so (an export of a client the runner did not patch, or a reply
that gave no count): a count that cannot be read is never written as zero.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NamedTuple

from hyperloom.common.outside_facts import (
    AIPERF_ERROR_CLASS_KEY,
    AIPERF_ERROR_KEY,
    AIPERF_ERROR_OUTPUT_LENGTH_METRIC,
    AIPERF_EXPORT_DIR,
    AIPERF_EXPORT_FILE,
    AIPERF_INVALID_REPLY_ERROR,
    AIPERF_METADATA_KEY,
    AIPERF_METRICS_KEY,
    AIPERF_PHASE_KEY,
    AIPERF_PROFILING_PHASE,
)
from hyperloom.contracts.measurement import RequestClasses
from hyperloom.measure.socket import RoundOutcome

__all__ = ["export_files", "read_request_classes", "read_round_classes"]

_MEASURE = "measure"
_EXPORT_PATTERNS = (f"{AIPERF_EXPORT_DIR}/{AIPERF_EXPORT_FILE}", f"*/{AIPERF_EXPORT_DIR}/{AIPERF_EXPORT_FILE}")
#: A request's line is a few kilobytes; a line past this is not one, and is never read whole.
_MAX_LINE_BYTES = 1 << 20
_NO_EXPORT = "the round left no per-request export"
_MANY_EXPORTS = "the round left {count} per-request exports, so none of them is the round's"
_UNREADABLE = "the export could not be read ({kind})"
_OUTSIDE = "the export is not below the test's output"
_BAD_LINES = "{count} line(s) of the export are not a request record, so no count of it can be trusted"
_NO_REQUESTS = "the export holds no profiling request"
#: Recorded in a result and compared when the result is reverified: a change of wording breaks the reverification of
#: every result recorded with the old one.
_UNSPLIT = (
    "the client's export carries no output token count for a request it counted failed, so {count} of the "
    "{no_content} no-content failures cannot be told empty from short"
)


class _Line(NamedTuple):
    """One request of the export: whether it is a profiling one, its error (``None`` when the client counted it
    successful) and the output token count of a failed request (``None`` when the line holds none)."""

    profiling: bool
    error: Mapping[str, object] | None
    output_tokens: int | None


@dataclass
class _Tally:
    """What one export held, request by request."""

    total: int = 0
    transport: int = 0
    no_content: int = 0
    empty: int = 0
    short: int = 0
    unreadable: int = 0

    @property
    def failed(self) -> int:
        return self.transport + self.no_content

    @property
    def split(self) -> int:
        """The no-content failures whose output token count the export gave."""
        return self.empty + self.short


def export_files(round_dir: Path) -> list[Path]:
    """The per-request exports below a round's directory, in the places the replay client puts one: the round's own
    artifact directory, or one level down where an AgentX round nests its benchmark."""
    found: list[Path] = []
    for pattern in _EXPORT_PATTERNS:
        found.extend(path for path in sorted(Path(round_dir).glob(pattern)) if path.is_file())
    return found


def _output_tokens(payload: Mapping[str, object]) -> int | None:
    metrics = payload.get(AIPERF_METRICS_KEY)
    entry = metrics.get(AIPERF_ERROR_OUTPUT_LENGTH_METRIC) if isinstance(metrics, Mapping) else None
    value = entry.get("value") if isinstance(entry, Mapping) else entry
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _parse(raw: bytes) -> _Line | None:
    """The request a line holds, or ``None`` when the line is not one."""
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    metadata = payload.get(AIPERF_METADATA_KEY) if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or not isinstance(payload.get(AIPERF_METRICS_KEY), dict):
        return None
    error = payload.get(AIPERF_ERROR_KEY)
    if error in (None, {}):
        error = None
    elif not isinstance(error, dict):
        return None
    phase = metadata.get(AIPERF_PHASE_KEY)
    return _Line(phase is None or phase == AIPERF_PROFILING_PHASE, error, _output_tokens(payload))


def _add(tally: _Tally, raw: bytes | None) -> None:
    line = None if raw is None else _parse(raw)
    if line is None:
        tally.unreadable += 1
    elif line.profiling:
        tally.total += 1
        if line.error is not None:
            _add_failed(tally, line.error, line.output_tokens)


def _add_failed(tally: _Tally, error: Mapping[str, object], output_tokens: int | None) -> None:
    if error.get(AIPERF_ERROR_CLASS_KEY) != AIPERF_INVALID_REPLY_ERROR:
        tally.transport += 1
        return
    tally.no_content += 1
    if output_tokens == 0:
        tally.empty += 1
    elif output_tokens is not None:
        tally.short += 1


def _lines(handle: BinaryIO) -> Iterator[bytes | None]:
    """The non-blank lines of the export; ``None`` stands for one too long to be a request's, drained unread."""
    while raw := handle.readline(_MAX_LINE_BYTES + 1):
        if len(raw) > _MAX_LINE_BYTES:
            while raw and not raw.endswith(b"\n"):
                raw = handle.readline(_MAX_LINE_BYTES)
            yield None
        elif raw.strip():
            yield raw


def _tally(path: Path) -> _Tally:
    tally = _Tally()
    with path.open("rb") as handle:
        for raw in _lines(handle):
            _add(tally, raw)
    return tally


def _refusal(point: int, concurrency: int | None, reason: str, export: str = "") -> RequestClasses:
    return RequestClasses(point, export, concurrency, reason=reason)


def _classes(tally: _Tally, point: int, concurrency: int | None, export: str) -> RequestClasses:
    """The counts of ``tally`` for the round at ``point``, or why there are none."""
    if tally.unreadable:
        return _refusal(point, concurrency, _BAD_LINES.format(count=tally.unreadable), export)
    if not tally.total:
        return _refusal(point, concurrency, _NO_REQUESTS, export)
    counts = {
        "requests_total": tally.total,
        "failed_total": tally.failed,
        "failed_transport": tally.transport,
        "failed_no_content": tally.no_content,
    }
    unsplit = tally.no_content - tally.split
    if unsplit:
        reason = _UNSPLIT.format(count=unsplit, no_content=tally.no_content)
        return RequestClasses(point, export, concurrency, **counts, reason=reason)
    return RequestClasses(point, export, concurrency, **counts, failed_empty=tally.empty, failed_short=tally.short)


def _relative(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def read_request_classes(round_dir: Path, root: Path, point: int, concurrency: int | None) -> RequestClasses:
    """The classes of the failed requests of one round, read from the export below ``round_dir``.

    ``root`` is the test's output directory: the export is named relative to it. A round whose export is missing,
    ambiguous or unreadable has no count and says why. Nothing here raises for a file that cannot be read.
    """
    try:
        exports = export_files(round_dir)
        if len(exports) != 1:
            reason = _NO_EXPORT if not exports else _MANY_EXPORTS.format(count=len(exports))
            return _refusal(point, concurrency, reason)
        pointer = _relative(exports[0], root)
        if pointer is None:
            return _refusal(point, concurrency, _OUTSIDE)
        return _classes(_tally(exports[0]), point, concurrency, pointer)
    except OSError as exc:
        return _refusal(point, concurrency, _UNREADABLE.format(kind=type(exc).__name__))


def read_round_classes(
    rounds: Sequence[RoundOutcome], root: Path, concurrencies: Sequence[int | None]
) -> tuple[RequestClasses, ...]:
    """One entry for each measured round of a test, in round order; ``concurrencies`` is each point's load."""
    return tuple(
        read_request_classes(
            Path(item.output_dir),
            root,
            item.point,
            concurrencies[item.point] if item.point < len(concurrencies) else None,
        )
        for item in rounds
        if item.phase == _MEASURE
    )
