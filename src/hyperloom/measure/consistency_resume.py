# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resume the consistency check across machine losses (design 12.4 stage 1, 4.4.5).

``python -m hyperloom.measure.consistency_resume --checklist .. --config .. --resume FILE`` (the arguments of
``hyperloom.measure.consistency``, plus the ones below) keeps one line per finished unit in ``FILE``, a
``consistency.jsonl`` that lives somewhere that survives the machine. A unit is one repetition of one test: the local
measurement and the hosted measurement back to back on the same machine, so units that ran on different machines are each
a valid same-machine comparison, and the summary over all units is what the graduation criteria judge. A run starts by
reading ``FILE``, skips every (test id, repetition) already recorded as complete and appends only new ones.

Lines (all carry ``node_name``, ``run_id`` and ``pod``):

* ``"type": "unit"``: a finished unit. It has the fields of an item line of ``consistency.jsonl`` plus ``rep`` (from 1),
  ``attempts`` and ``duration_sec``. It is written only after both sides ended, in one append, so an interrupted unit
  leaves no line.
* ``"type": "attempt_failed"``: one side could not be measured (infrastructure) or the two sides ran on different nodes.
  The unit is tried again; after ``max_attempts`` infrastructure failures it is recorded as a ``unit`` that did not
  compare, which counts against the verdict. A cross-node pair is never recorded as a unit.

The summary (``FILE`` plus ``.summary.json``, and printed) is computed over all ``unit`` lines the way ``consistency
--repeats N`` computes it: per test, each side's mean over its repetitions, one difference per test. It adds ``units_expected``,
``units_complete``, ``nodes`` (the number of distinct nodes that contributed) and ``node_names``. While a unit is missing
the verdict is ``INCOMPLETE``: no threshold verdict is given on part of the units.

One run at a time. A run holds ``FILE`` plus ``.lock`` (the node, the pid and a heartbeat the run touches every 30 s). A
second run, which is what submitting the same command twice starts, waits for it and then stops with exit code 12 and
measures nothing. A lock is taken over when its holder is a dead process on this node, or when it has not been touched for
120 s: the machine that held it was lost, and a run that was lost leaves nothing else behind.

Time. ``lifetime_sec`` (default: unlimited) is how long this machine is expected to live; a new unit starts only when
the time left covers the estimate of one unit, 1.25 times the longest unit already recorded, or ``unit_budget_sec`` while
none is. The first unit of a run always starts.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.measure.consistency import (
    HOSTED,
    LOCAL,
    ChecklistItem,
    Criteria,
    ItemReport,
    Side,
    aggregate_repetitions,
    build_backends,
    build_parser,
    check_repetition,
    items_from_arguments,
    summarize,
)
from hyperloom.measure.socket import MeasurementBackend
from hyperloom.sandbox.node import node_name

__all__ = [
    "INCOMPLETE",
    "Unit",
    "plan_units",
    "read_lines",
    "append_line",
    "resume_summary",
    "run_resumable",
    "RunLock",
    "main",
]

INCOMPLETE = "INCOMPLETE"
UNIT, ATTEMPT_FAILED = "unit", "attempt_failed"
SUMMARY_SUFFIX = ".summary.json"
ATTEMPTS_DIR = "attempts"
DEFAULT_MAX_ATTEMPTS = 3
LOCK_SUFFIX = ".lock"
HEARTBEAT_SEC = 30.0
STALE_AFTER_SEC = 120.0
LOCKED_RC = 12
ESTIMATE_FACTOR = 1.25
_UTF8 = "utf-8"
_FILE_MODE = 0o600


@dataclasses.dataclass(frozen=True)
class Unit:
    """One repetition of one test: the smallest piece of the check that is a valid comparison by itself."""

    item: ChecklistItem
    rep: int

    @property
    def key(self) -> tuple[str, int]:
        return self.item.id, self.rep


def plan_units(items: Sequence[ChecklistItem]) -> list[Unit]:
    """Every (test, repetition), repetition-major: after each pass over the tests every test has one more reading."""
    deepest = max((item.repeats for item in items), default=0)
    return [Unit(item, rep) for rep in range(1, deepest + 1) for item in items if rep <= item.repeats]


# -- the file ---------------------------------------------------------------------------------------------------------


def read_lines(path: Path) -> list[dict[str, Any]]:
    """The JSON object lines of the file; a torn or foreign line (a write cut short by a loss) is skipped."""
    try:
        text = Path(path).read_text(encoding=_UTF8, errors="replace")
    except FileNotFoundError:
        return []
    lines = []
    for raw in text.splitlines():
        try:
            document = json.loads(raw)
        except ValueError:
            continue
        if isinstance(document, dict):
            lines.append(document)
    return lines


def append_line(path: Path, document: Mapping[str, Any]) -> None:
    """Append one line with a single write and flush it to disk; a line that does not start on a fresh line gets one."""
    data = (json.dumps(document, sort_keys=True) + "\n").encode(_UTF8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, _FILE_MODE)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            data = b"\n" + data
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclasses.dataclass(frozen=True)
class LockProbe:
    """How a lock decides that its holder is gone: the age limit, the clock and the liveness check of a pid."""

    stale_after_sec: float = STALE_AFTER_SEC
    now: Callable[[], float] = time.time
    pid_alive: Callable[[int], bool] = _pid_alive


class RunLock:
    """The right to measure into one file; see the module text. ``acquire`` never blocks, ``wait`` polls it."""

    def __init__(self, path: Path, *, node: str, pid: int | None = None, probe: LockProbe = LockProbe()) -> None:
        self.path, self._node, self._pid = Path(path), node, pid if pid is not None else os.getpid()
        self._stale_after, self._now, self._pid_alive = probe.stale_after_sec, probe.now, probe.pid_alive
        self._beat = threading.Event()

    def _stale(self) -> bool:
        try:
            age = self._now() - self.path.stat().st_mtime
        except OSError:
            return True
        try:
            holder = json.loads(self.path.read_text(encoding=_UTF8))
        except (OSError, ValueError):
            holder = {}
        dead_here = holder.get("node") == self._node and not self._pid_alive(int(holder.get("pid") or 0))
        return dead_here or age > self._stale_after

    def acquire(self) -> bool:
        """Take the lock; a stale one is moved aside first (``rename`` has one winner when two runs race for it)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
            except FileExistsError:
                if not self._stale():
                    return False
                try:
                    os.rename(self.path, self.path.with_name(f"{self.path.name}.stale.{self._node}.{self._pid}"))
                except OSError:
                    return False
                continue
            with os.fdopen(fd, "w", encoding=_UTF8) as handle:
                handle.write(json.dumps({"node": self._node, "pid": self._pid}))
            return True
        return False

    def wait(self, seconds: float, *, poll_sec: float = 10.0, sleep: Callable[[float], None] = time.sleep) -> bool:
        """Try to acquire for up to ``seconds``."""
        deadline = self._now() + seconds
        while not self.acquire():
            if self._now() >= deadline:
                return False
            sleep(poll_sec)
        return True

    def touch(self) -> None:
        try:
            os.utime(self.path)
        except OSError:
            pass

    def start_heartbeat(self, every_sec: float = HEARTBEAT_SEC) -> None:
        def beat() -> None:
            while not self._beat.wait(every_sec):
                self.touch()

        threading.Thread(target=beat, name="consistency-lock-heartbeat", daemon=True).start()

    def release(self) -> None:
        self._beat.set()
        self.path.unlink(missing_ok=True)


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding=_UTF8)
    os.replace(temporary, path)


# -- reading back -----------------------------------------------------------------------------------------------------


def _nodes_of(document: Mapping[str, Any]) -> set[str]:
    sides = (document.get(LOCAL) or {}, document.get(HOSTED) or {})
    names = [s.get("node_name") for s in sides] + [document.get("node_name")]
    return {str(n) for n in names if n}


def _crosses_nodes(document: Mapping[str, Any]) -> bool:
    return len(_nodes_of(document)) > 1


def complete_units(lines: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    """The first recorded ``unit`` line of each (test, repetition); a hand-made cross-node line is not one."""
    found: dict[tuple[str, int], Mapping[str, Any]] = {}
    for line in lines:
        if line.get("type") == UNIT and not _crosses_nodes(line):
            found.setdefault((str(line.get("id")), int(line.get("rep", 0))), line)
    return found


def _side_of(document: Mapping[str, Any]) -> Side:
    return Side(
        document.get("returncode"),
        throughput=document.get("output_throughput"),
        problem=document.get("problem"),
        node_name=document.get("node_name"),
    )


def _report_of(document: Mapping[str, Any]) -> ItemReport:
    return ItemReport(
        str(document["id"]),
        _side_of(document[LOCAL]),
        _side_of(document[HOSTED]),
        document.get("throughput_diff_pct"),
        document.get("completed_equal"),
        document.get("error_rate_equal"),
        document.get("rebuilt_files_equal"),
        bool(document.get("signature_verified")),
        tuple(document.get("problems") or ()),
    )


def _test_reports(units: Sequence[Unit], present: Sequence[Mapping[str, Any]]) -> list[ItemReport]:
    """One report per test, as ``consistency --repeats N`` makes it: the unit itself for a test of one repetition, else
    the means of its repetitions compared (``aggregate_repetitions``)."""
    repeats = {unit.item.id: unit.item.repeats for unit in units}
    by_test: dict[str, dict[int, ItemReport]] = {}
    for document in present:
        by_test.setdefault(str(document["id"]), {})[int(document["rep"])] = _report_of(document)
    reports = []
    for test_id, reps in sorted(by_test.items()):
        if repeats.get(test_id, 1) <= 1:
            reports.extend(reps.values())
        else:
            reports.append(aggregate_repetitions(test_id, reps))
    return reports


def resume_summary(
    units: Sequence[Unit], lines: Sequence[Mapping[str, Any]], criteria: Criteria = Criteria()
) -> dict[str, Any]:
    """The summary document over all complete units of ``lines``; ``INCOMPLETE`` while a planned unit is missing.

    It means what the summary of ``consistency --repeats N`` means: each test's local and hosted readings are averaged
    over its repetitions and the thresholds judge one difference per test. Every unit was measured on one node, so a
    test's repetitions may come from several; ``nodes`` says how many contributed. Completed requests, rebuilt files and
    signatures are still checked for every unit.
    """
    done = complete_units(lines)
    wanted = {unit.key for unit in units}
    present = [done[key] for key in sorted(done) if key in wanted]
    summary = summarize(_test_reports(units, present), criteria).to_document()
    names = sorted({name for d in present for name in _nodes_of(d)})
    missing = sorted(wanted - set(done))
    if missing:
        summary["verdict"] = INCOMPLETE
        summary["failures"] = [f"missing units: {len(missing)} of {len(wanted)}", *summary["failures"]]
    return {
        **summary,
        "units_expected": len(wanted),
        "units_complete": len(present),
        "nodes": len(names),
        "node_names": names,
        "runs": len({d.get("run_id") for d in present}),
    }


# -- running ----------------------------------------------------------------------------------------------------------


def _estimate(done: Mapping[Any, Mapping[str, Any]], unit_budget_sec: float) -> float:
    longest = max((float(d.get("duration_sec") or 0) for d in done.values()), default=0.0)
    return longest * ESTIMATE_FACTOR if longest else unit_budget_sec


def _unit_document(
    unit: Unit, report: ItemReport, who: Mapping[str, Any], attempts: int, seconds: float
) -> dict[str, Any]:
    return {
        **report.to_document(),
        "type": UNIT,
        "rep": unit.rep,
        "attempts": attempts,
        "duration_sec": round(seconds, 3),
        **who,
    }


def _attempt_document(unit: Unit, report: ItemReport, who: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "type": ATTEMPT_FAILED,
        "id": unit.item.id,
        "rep": unit.rep,
        "reason": reason,
        "problems": list(report.problems),
        **who,
    }


def _infrastructure_failed(report: ItemReport) -> bool:
    return not (report.local.usable and report.hosted.usable)


@dataclasses.dataclass(frozen=True)
class _Tally:
    """How many attempts a unit has had before this one, how many it may have, and how long this one took."""

    prior_attempts: int
    max_attempts: int
    seconds: float


def _record(path: Path, unit: Unit, report: ItemReport, who: Mapping[str, Any], tally: _Tally) -> bool:
    """Write the line this unit's outcome earns; returns whether the unit is now complete."""
    here = {who["node_name"], *(s.node_name for s in (report.local, report.hosted) if s.node_name)}
    if len(here) > 1:
        append_line(path, _attempt_document(unit, report, who, f"cross_node: {sorted(here)}"))
        return False
    attempts = tally.prior_attempts + 1
    if _infrastructure_failed(report) and attempts < tally.max_attempts:
        append_line(path, _attempt_document(unit, report, who, "side_failed"))
        return False
    append_line(path, _unit_document(unit, report, who, attempts, tally.seconds))
    return True


@dataclasses.dataclass(frozen=True)
class ResumeFiles:
    """Where a run keeps its finished units and its per-attempt reports."""

    jsonl: Path
    report_dir: Path


@dataclasses.dataclass(frozen=True)
class ResumeSettings:
    """How a run judges and paces itself; every field has the default of the command line."""

    criteria: Criteria = Criteria()
    lifetime_sec: float = 0.0
    unit_budget_sec: float = 600.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    run_id: str | None = None
    clock: Callable[[], float] = time.monotonic


def run_resumable(
    items: Sequence[ChecklistItem],
    backends: Callable[[], tuple[MeasurementBackend, Any]],
    files: ResumeFiles,
    settings: ResumeSettings = ResumeSettings(),
) -> dict[str, Any]:
    """Measure the units not yet recorded in ``files.jsonl``, one after the other, and return the summary over all.

    ``backends`` is called once, only when a unit has to run, and returns the local and the hosted backend.
    """
    jsonl, report_dir, clock = files.jsonl, files.report_dir, settings.clock
    lifetime_sec, unit_budget_sec = settings.lifetime_sec, settings.unit_budget_sec
    started = clock()
    run = settings.run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    who = {"node_name": node_name(), "run_id": run, "pod": os.environ.get("POD_NAME") or os.environ.get("HOSTNAME")}
    units = plan_units(items)
    pair: tuple[MeasurementBackend, Any] | None = None
    ran = 0
    for unit in units:
        lines = read_lines(jsonl)
        done = complete_units(lines)
        if unit.key in done:
            continue
        if ran and lifetime_sec > 0 and clock() - started + _estimate(done, unit_budget_sec) > lifetime_sec:
            break
        pair = pair or backends()
        prior = sum(1 for x in lines if x.get("type") == ATTEMPT_FAILED and (x.get("id"), x.get("rep")) == unit.key)
        began = clock()
        # A request id of the hosted journal must not repeat: an attempt of an earlier run has its own entries there.
        fresh = dataclasses.replace(unit.item, id=f"{unit.item.id}.{run}")
        report = check_repetition(fresh, report_dir / ATTEMPTS_DIR, pair[0], pair[1], unit.rep)
        report = dataclasses.replace(report, id=unit.item.id)
        _record(jsonl, unit, report, who, _Tally(prior, settings.max_attempts, clock() - began))
        ran += 1
    return resume_summary(units, read_lines(jsonl), settings.criteria)


def finish(jsonl: Path, summary: Mapping[str, Any]) -> int:
    """Store and print the summary; the exit code is 0 for ``PASS``, 1 for ``FAIL``/``NO_DATA``, 10 while incomplete."""
    text = json.dumps(summary, indent=2, sort_keys=True)
    _write_atomic(Path(jsonl).with_name(Path(jsonl).name + SUMMARY_SUFFIX), text + "\n")
    print(text)
    return {"PASS": 0, INCOMPLETE: 10}.get(str(summary["verdict"]), 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser("python -m hyperloom.measure.consistency_resume")
    parser.add_argument(
        "--resume", type=Path, required=True, help="the jsonl of finished units, on storage that survives"
    )
    parser.add_argument("--lifetime-sec", type=float, default=0.0, help="expected life of this machine (0: unlimited)")
    parser.add_argument(
        "--unit-budget-sec", type=float, default=600.0, help="estimate of one unit before any is recorded"
    )
    parser.add_argument("--lock-wait-sec", type=float, default=STALE_AFTER_SEC + 30, help="wait for another run")
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="infrastructure failures per unit"
    )
    args = parser.parse_args(argv)
    items = items_from_arguments(args)
    if items is None:
        return 2
    lock = RunLock(Path(f"{args.resume}{LOCK_SUFFIX}"), node=node_name())
    if not lock.wait(args.lock_wait_sec):
        print(
            f"{args.resume} is being measured by another run (lock {lock.path}); nothing was measured", file=sys.stderr
        )
        return LOCKED_RC
    lock.start_heartbeat()
    built: list[Any] = []

    def backends() -> tuple[MeasurementBackend, Any]:
        built.append(build_backends(args.config))
        return built[0].local, built[0].hosted

    try:
        summary = run_resumable(
            items,
            backends,
            ResumeFiles(args.resume, args.report_dir),
            ResumeSettings(
                criteria=Criteria(args.median_pct, args.p90_pct),
                lifetime_sec=args.lifetime_sec,
                unit_budget_sec=args.unit_budget_sec,
                max_attempts=args.max_attempts,
            ),
        )
    finally:
        for backend in built:
            backend.close()
        lock.release()
    return finish(args.resume, summary)


if __name__ == "__main__":
    sys.exit(main())
