# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the time of a session's tests went, read from the session directory in one command.

``python -m hyperloom_multibranch.toolbox.phase_report <session_dir>`` reads the toolbox's submit log
(``<session_dir>/toolbox/submissions.jsonl``), takes every measured test's ``phases`` and prints, for each kind of test,
how many seconds each phase took: its total, its median over the tests that ran it and its share of the tests' wall
time. A phase the engine's log splits (``server_boot.weights_load``) is also summed into its parent (``server_boot``),
and what no phase covers is the ``unaccounted`` row, so the rows add up to the wall time. ``--json`` prints the same as
a document.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from hyperloom.contracts.measurement import MeasurementError, TestResult
from hyperloom.measure.phases import PHASE_ORDER
from hyperloom_multibranch.toolbox.submission_log import ACCEPTED, OUTCOME

__all__ = ["PhaseRow", "aggregate", "main", "read_session"]

#: Where the toolbox keeps its submit log below the session directory.
SUBMISSIONS = "toolbox/submissions.jsonl"
_PART_SEPARATOR = "."
#: The row for the seconds of a test's wall time that none of its phases covers.
_UNACCOUNTED = "unaccounted"
#: A kind of test the log names no kind for.
_UNKNOWN_KIND = "unknown"


@dataclass(frozen=True)
class PhaseRow:
    """One phase of one kind of test: over how many tests, and how long it took."""

    kind: str
    phase: str
    tests: int
    total_seconds: float
    median_seconds: float
    gpu_seconds: float
    share: float


@dataclass(frozen=True)
class SessionPhases:
    """The measured tests of a session, by kind, with the ones whose phases could not be used counted apart."""

    results: list[tuple[str, TestResult]]
    without_phases: int
    unreadable: int


def read_session(session_dir: Path) -> SessionPhases:
    """The measured results of ``session_dir``'s submit log with the kind of test that made each.

    A result with no phases (one a measurement before phases existed made, or one no measurement made) and a result that
    cannot be read are counted, not skipped silently.
    """
    log = Path(session_dir) / SUBMISSIONS
    kinds: dict[str, str] = {}
    outcomes: dict[str, object] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # the torn last line of a crash, which the toolbox ignores as well
        request_id = str(record.get("request_id"))
        if record.get("type") == ACCEPTED:
            kinds[request_id] = str((record.get("request") or {}).get("kind") or _UNKNOWN_KIND)
        elif record.get("type") == OUTCOME and "result" in (record.get("outcome") or {}):
            outcomes[request_id] = record["outcome"]["result"]
    found: list[tuple[str, TestResult]] = []
    without, unreadable = 0, 0
    for request_id, document in outcomes.items():
        try:
            result = TestResult.from_document(document)
        except MeasurementError:
            unreadable += 1
            continue
        if result.phases:
            found.append((kinds.get(request_id, _UNKNOWN_KIND), result))
        else:
            without += 1
    return SessionPhases(found, without, unreadable)


def aggregate(results: Iterable[tuple[str, TestResult]]) -> list[PhaseRow]:
    """The phase rows of every kind of test, the kinds in name order and each kind's phases in the order tests run them."""
    by_kind: dict[str, list[TestResult]] = defaultdict(list)
    for kind, result in results:
        by_kind[kind].append(result)
    return [row for kind in sorted(by_kind) for row in _kind_rows(kind, by_kind[kind])]


def _kind_rows(kind: str, results: Sequence[TestResult]) -> list[PhaseRow]:
    """One row for each phase and each part of a split phase, and the unaccounted one, of the tests of one kind."""
    seconds: dict[str, list[float]] = defaultdict(list)
    gpu: dict[str, float] = defaultdict(float)
    wall = 0.0
    for result in results:
        per_test: dict[str, float] = defaultdict(float)
        for phase in result.phases:
            names = [phase.name]
            if _PART_SEPARATOR in phase.name:
                names.append(phase.name.partition(_PART_SEPARATOR)[0])
            for name in names:
                per_test[name] += phase.seconds
                gpu[name] += phase.gpu_seconds
        per_test[_UNACCOUNTED] = result.unaccounted_seconds or 0.0
        wall += sum(phase.seconds for phase in result.phases) + (result.unaccounted_seconds or 0.0)
        for name, total in per_test.items():
            seconds[name].append(total)
    ordered = sorted(seconds, key=_rank)
    return [
        PhaseRow(
            kind,
            name,
            len(seconds[name]),
            sum(seconds[name]),
            statistics.median(seconds[name]),
            gpu[name],
            sum(seconds[name]) / wall if wall > 0 else 0.0,
        )
        for name in ordered
    ]


def _rank(name: str) -> tuple[int, str]:
    """Phases in the order a test goes through them, a part right after the phase it splits, what no phase covers last."""
    parent = name.partition(_PART_SEPARATOR)[0]
    known = PHASE_ORDER.index(parent) if parent in PHASE_ORDER else len(PHASE_ORDER)
    return known * 2 + (_PART_SEPARATOR in name) + 4 * (name == _UNACCOUNTED), name


def render_text(session: SessionPhases, rows: Sequence[PhaseRow]) -> str:
    """The rows as one table for each kind of test, with a line for what the log held no phases of."""
    lines = [f"{len(session.results)} measured tests with phases"]
    if session.without_phases or session.unreadable:
        lines.append(f"({session.without_phases} without phases, {session.unreadable} unreadable: not counted)")
    kind = None
    for row in rows:
        if row.kind != kind:
            kind = row.kind
            tests = sum(1 for k, _ in session.results if k == kind)
            lines += [
                "",
                f"{kind} ({tests} tests)",
                f"  {'phase':<28}{'tests':>6}{'total s':>11}{'median s':>10}{'share':>7}",
            ]
        label = f"  {row.phase}" if _PART_SEPARATOR in row.phase else row.phase
        lines.append(
            f"  {label:<28}{row.tests:>6}{row.total_seconds:>11.1f}{row.median_seconds:>10.1f}{row.share:>7.1%}"
        )
    return "\n".join([*lines, "", "an indented phase is a part of the one above it, already in its seconds"])


def render_json(session: SessionPhases, rows: Sequence[PhaseRow]) -> str:
    document: Mapping[str, object] = {
        "tests": len(session.results),
        "without_phases": session.without_phases,
        "unreadable": session.unreadable,
        "rows": [asdict(row) for row in rows],
    }
    return json.dumps(document, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase_report", description=__doc__.split("\n\n")[0])
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="print a JSON document instead of the table")
    args = parser.parse_args(argv)
    try:
        session = read_session(args.session_dir)
    except OSError as exc:
        print(f"phase_report: cannot read {args.session_dir / SUBMISSIONS}: {exc.strerror}", file=sys.stderr)
        return 2
    rows = aggregate(session.results)
    print(render_json(session, rows) if args.json else render_text(session, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
