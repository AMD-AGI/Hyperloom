# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The arena's timestamped session log, read as forge-rewrite's PORT step and each forge-loop iteration.

The arena logs every line it and KernelForge print as ``<date> <time>,<ms> - <LEVEL> - <message>``, KernelForge's
own lines under ``[FORGE]``. An iteration's fields come from its own lines: its header, the merge's text, the agent's
rationale, what it measured, and its decision line. Decision labels are KernelForge's own: each label of
``_decision_label`` in ``kernelforge/loop/runner.py`` (commit 1c14e79e1) is printed in one line shape.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from kernelforge.knowledge.kb_store.identity.kernel_recipe import DEFAULT_SCHEME_NAME, KERNEL_CANONICAL_DIMENSIONS
from kernelforge.loop.merge_candidates import MERGE_PLAN_PREFIX
from kernelforge.run_export.archive import first_sentences, rationale_text
from kernelforge.run_export.units import OBJECTIVE, CandidateOrigin, Run

#: Lines a printed message spans are logged at most this far apart.
_CONTINUATION_SECONDS = 2.0
_RATIONALE = "Rationale: "
#: KernelForge prints only this much of the agent's rationale (``kernelforge/loop/runner.py``).
_RATIONALE_CHARS = 200
_AGENT = "agent"
_FORGE_REWRITE = "forge-rewrite"
#: The tags of decision lines. The KEEP and CRASH tags are also the labels of the decisions their lines print.
_KEEP = "KEEP"
_CRASH = "CRASH"
_DECISION_TAGS = (_KEEP, "REVERT", _CRASH)

_LINE = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d),(\d{3}) - [A-Z]+ - (.*)$")
_FORGE = re.compile(r"^\[FORGE(?: STDERR)?\] ?(.*)$")
_TAG = re.compile(r"^\s*\[((?:[a-z][a-z0-9 _-]*)|KEEP|REVERT|CRASH|SKIP)\]\s?(.*)$")
_HEADER = re.compile(r"^--- Iteration (\d+)(?: \(best mean case speedup: ([-+0-9.eE]+)x, remaining: \d+ min\))? ---$")
_BENCH = re.compile(
    r"^pristine-relative scores=\[(?P<scores>[^\]]*)\]; sigma=(?P<sigma>[^;]+); (?P<note>.*?)"
    r"mean score=(?P<mean>[-+0-9.eE]+)x; required=(?P<required>[-+0-9.eE]+)x; raw mean=(?P<raw>\S+) ms"
)
_NEW_BEST = re.compile(r"^mean case speedup=([-+0-9.eE]+)x — NEW BEST")
_REVERT_PERF = re.compile(r"^mean case speedup=(?:([-+0-9.eE]+)x|None) not better than best=")
_VALIDATE = re.compile(r"^Stage (\d+) (.+?): (PASS|FAIL|TIMEOUT|ERROR)(?: SNR=([-+0-9.eE]+)dB)?$")
_MERGE_PLAN = re.compile(rf"^{re.escape(MERGE_PLAN_PREFIX)} (\d+)\+(\d+): ")
_LANE = re.compile(r"^lane (\d+)$")
_COMMITTED = re.compile(r"^Committed verified best: ([0-9a-f]{7,40})$")
_KB_PUBLISH = re.compile(r"^KEEP KB publish \(([0-9a-f]{7,40})\): (\S+)")
#: A record as the KB store names it, ``<canonical id>/<session id>``. forge-rewrite prints the reason no record was
#: written in the same place (``kernelforge/rewrite_by_flydsl/runner.py``), and no reason has this shape.
_KB_RECORD = re.compile(rf"^{re.escape(DEFAULT_SCHEME_NAME)}(?::[^:/\s]+){{{len(KERNEL_CANONICAL_DIMENSIONS)}}}/\S+$")
_KB_WRITTEN = re.compile(r"^experience written: (\S+)")
_SESSION_END = re.compile(r"^backend=\S+ reason=(\S+)")
_SOURCE_BASELINE = re.compile(r"^source baseline: ([-+0-9.eE]+) ms \(full suite, cases=\[(.*)\]\)$")
_PORT_OK = re.compile(r"^PORT OK \(attempt (\d+), SNR=([^)]*)\)$")
_PORT_FAILED = re.compile(r"^PORT FAILED after (\d+) attempts$")
_INTERIM = re.compile(
    r"^interim \(port only\): flydsl=([-+0-9.eE]+) ms vs source=([-+0-9.eE]+) ms -> speedup=([-+0-9.eE]+)x"
)
_EXCEPTION = re.compile(r"^[\w.]+(?:Error|Exception)\b")
_TRACEBACK = "Traceback (most recent call last):"
_NO_CROSS_CUTTING_MOVE = "round partition gave no lane its largest cross-cutting move:"
#: The tags of the lines KernelForge prints while it plans the round an iteration measures.
_PLANNING_TAGS = frozenset({"supervisor", "search-policy"})

#: Plain lines the forge-loop prints that end the iteration loop; the last iteration's lines stop at the first.
_LOOP_END = (
    "BUDGET EXHAUSTED after",
    "Autonomous loop complete",
    "GATE MET at iteration",
    "ROOFLINE TARGET MET at iteration",
    "ORCHESTRATION FAILED repeatedly",
    "FORCE STOP:",
    "DEVICE CONTENDED:",
    _TRACEBACK,
    "__FORGE_RESULT__",
)
#: What an iteration unit takes from an iteration. Its other keys only decide the decision label, or go to provenance.
_ITERATION_KEYS = (
    "started_at",
    "completed_at",
    "stop",
    "stop_line",
    "best_mean_case_speedup",
    "reasoning",
    "alternatives",
    "plan",
    "change_kind",
    "approach",
    "decision",
    "mean_case_speedup",
)


@dataclass(frozen=True)
class LogLine:
    """One timestamped log line: ``forge`` when KernelForge printed it, with its bracketed ``tag`` and the ``rest``."""

    at: str
    day: str
    seconds: float
    forge: bool
    body: str
    tag: str | None
    rest: str


def _timestamped(text: str) -> list[str]:
    return [line for line in text.splitlines() if _LINE.match(line)]


def _entry(line: str) -> LogLine:
    match = _LINE.match(line)
    assert match is not None
    day, clock, millis, message = match.groups()
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    forge = _FORGE.match(message)
    body = forge.group(1) if forge else message
    tag = _TAG.match(body) if forge else None
    return LogLine(
        at=f"{day}T{clock}.{millis}",
        day=day,
        seconds=hours * 3600 + minutes * 60 + seconds + int(millis) / 1000.0,
        forge=forge is not None,
        body=body,
        tag=tag.group(1) if tag else None,
        rest=tag.group(2) if tag else body.strip(),
    )


def timeline(texts: list[str]) -> list[LogLine]:
    """The timestamped lines of one session's logs, once each, in log order.

    Lines both logs hold are taken once. Where they differ, both sides' lines are kept, ordered by their timestamps,
    so a line one log lost is still read from the other.
    """

    sources = [_timestamped(text) for text in texts]
    merged = sources[0] if sources else []
    for other in sources[1:]:
        combined: list[str] = []
        matcher = difflib.SequenceMatcher(None, merged, other, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                combined.extend(merged[i1:i2])
            else:
                combined.extend(sorted(merged[i1:i2] + other[j1:j2], key=lambda line: line[:23]))
        merged = combined
    return [_entry(line) for line in merged]


def _continuation(lines: list[LogLine], index: int, *, limit: int | None = None) -> tuple[str, int]:
    """One printed message that the arena logged as several lines; returns its text and its last line's index."""

    first = lines[index]
    text = first.rest
    last = index
    for later in range(index + 1, len(lines)):
        line = lines[later]
        close = line.day == first.day and abs(line.seconds - first.seconds) <= _CONTINUATION_SECONDS
        if not line.forge or line.tag is not None or not close:
            break
        body = line.body.strip()
        if _HEADER.match(body) or body.startswith(_LOOP_END):
            break
        if limit is not None and len(text) + 1 + len(line.body) > limit:
            break
        text = f"{text}\n{line.body}"
        last = later
    return text, last


@dataclass
class Setup:
    """A session log's session-level lines: the arena run, forge-rewrite's port and source timing, and the span of
    the forge-loop's iterations. Each ``int`` is an index into ``lines``."""

    lines: list[LogLine]
    run_name: str | None = None
    evaluation_start: int | None = None
    forge_rewrite: bool = False
    port_start: int | None = None
    port_summary: str | None = None
    source_ms: float | None = None
    warm_start: str | None = None
    port_decision: str | None = None
    port_end: int | None = None
    port_kb_record: str | None = None
    interim_speedup: float | None = None
    interim_end: int | None = None
    loop_start: int | None = None
    loop_end: int | None = None

    @classmethod
    def read(cls, lines: list[LogLine]) -> Setup:
        setup = cls(lines)
        for index, line in enumerate(lines):
            if not line.forge:
                setup._arena_line(index, line.body)
                continue
            if line.tag == _FORGE_REWRITE:
                setup._forge_rewrite_line(index, line.rest)
            if line.body == "Starting autonomous iteration loop":
                setup.loop_start = index if setup.loop_start is None else setup.loop_start
            elif line.body.startswith(_LOOP_END):
                setup._end_loop(index)
        return setup

    def at(self, index: int) -> str:
        return self.lines[index].at

    @property
    def loop_stop(self) -> int:
        """The index of the first line past the forge-loop's iterations."""

        return next((index for index in (self.loop_end, self.evaluation_start) if index is not None), len(self.lines))

    def _arena_line(self, index: int, body: str) -> None:
        if body.startswith("NEW RUN: "):
            self.run_name = body.split(": ", 1)[1].strip() if self.run_name is None else self.run_name
        elif body.startswith("Starting centralized kernel evaluation") and self.evaluation_start is None:
            self.evaluation_start = index

    def _forge_rewrite_line(self, index: int, rest: str) -> None:
        self.forge_rewrite = True
        self.port_start = index if self.port_start is None else self.port_start
        if rest.startswith("op="):
            self.port_summary = rest if self.port_summary is None else self.port_summary
        elif (match := _SOURCE_BASELINE.match(rest)) is not None:
            self.source_ms = float(match.group(1))
        elif rest.startswith("KB warm-start accepted: "):
            self.warm_start = rest
        elif _PORT_OK.match(rest) is not None:
            self.port_decision, self.port_end = "PORT OK", index
        elif _PORT_FAILED.match(rest) is not None:
            self.port_decision, self.port_end = "PORT FAILED", index
        elif rest.startswith("PORT KB publish: "):
            self.port_kb_record = _published_record(rest.split(": ", 1)[1])
        elif (match := _INTERIM.match(rest)) is not None:
            self.interim_speedup, self.interim_end = float(match.group(3)), index
        elif rest.startswith("optimize cutoff reached"):
            self._end_loop(index)

    def _end_loop(self, index: int) -> None:
        if self.loop_start is not None and self.loop_end is None:
            self.loop_end = index


def port_unit(run: Run, setup: Setup, files: list[str]) -> dict[str, Any]:
    """Seq 0: forge-rewrite's agent port of the source operator, or the KB kernel it accepted in its place."""

    unit = run.unit("port", "port", 0, files, kb_record=setup.port_kb_record)
    unit["objective"] = OBJECTIVE
    unit["started_at"] = None if setup.port_start is None else setup.at(setup.port_start)
    unit["change_kind"] = CandidateOrigin.KB_WARM_START if setup.warm_start else CandidateOrigin.PORT
    unit["approach"] = setup.warm_start or setup.port_summary
    unit["source_ms"] = run.source_ms
    if setup.port_end is not None:
        unit["decision"] = setup.port_decision
        end = setup.port_end
        if setup.interim_end is not None:
            unit["mean_case_speedup"] = setup.interim_speedup
            end = setup.interim_end
        unit["completed_at"] = setup.at(end)
    return unit


class _IterationReader:
    """One forge-loop iteration, read from its header line up to its decision line."""

    def __init__(self, lines: list[LogLine], header_index: int, planning_from: int) -> None:
        header = _HEADER.match(lines[header_index].body.strip())
        assert header is not None
        started = next(
            (
                index
                for index in range(planning_from, header_index)
                if lines[index].forge and lines[index].tag in _PLANNING_TAGS
            ),
            header_index,
        )
        self.lines = lines
        self.header_index = header_index
        self.item: dict[str, Any] = {"seq": int(header.group(1)), "started_at": lines[started].at}
        if header.group(2) is not None:
            self.item["best_mean_case_speedup"] = float(header.group(2))
        self.alternatives: list[str] = []
        self.decision_index: int | None = None
        self._validation: str | None = None
        self._validated = False
        self._in_session = False

    @property
    def end(self) -> int:
        """Where the next iteration's planning lines can start: past this one's decision line, or at its header."""

        return self.header_index if self.decision_index is None else self.decision_index + 1

    def read(self, block_end: int) -> None:
        index = self.header_index + 1
        while index < block_end and self.decision_index is None:
            if self.lines[index].forge:
                index = self._line(index)
            index += 1

    def _line(self, index: int) -> int:
        """Read one of KernelForge's lines; returns the index of the last line its message spans."""

        tag, rest = self.lines[index].tag, self.lines[index].rest
        if tag is None:
            return self._untagged(index)
        if tag == "merge":
            return self._merge(index, rest)
        if tag == _AGENT:
            return self._agent(index, rest)
        if tag in _DECISION_TAGS:
            self._decide(index, self._label(tag, rest))
        elif tag == "session-end" and self._in_session:
            self.item["session_end_reason"] = (_SESSION_END.match(rest) or [None, None])[1]
        elif tag == "validate" and (match := _VALIDATE.match(rest)) is not None:
            self._validated, self._validation = True, match.group(3)
        elif tag == "bench" and (match := _BENCH.match(rest)) is not None:
            self.item["mean_case_speedup"] = float(match.group("mean"))
        elif tag == "orchestration" and rest.startswith("failed ("):
            if self.item.get("change_kind") is CandidateOrigin.SESSION:
                self._decide(index, "ORCHESTRATION_ERROR")
        elif (lane := _LANE.match(tag)) is not None and rest == "measuring queued candidate":
            self.item["change_kind"] = CandidateOrigin.LANE
            self.item["lane_number"] = int(lane.group(1))
        return index

    def _untagged(self, index: int) -> int:
        if self.lines[index].body.strip().startswith(_NO_CROSS_CUTTING_MOVE):
            text, index = _continuation(self.lines, index)
            self.alternatives.append(text)
        return index

    def _merge(self, index: int, rest: str) -> int:
        if rest.startswith("declined: "):
            self.alternatives.append(f"merge {rest}")
        elif _MERGE_PLAN.match(rest) is not None:
            self.item["change_kind"] = CandidateOrigin.MERGE
            text, index = _continuation(self.lines, index)
            self.item["approach"] = text.splitlines()[0]
            self.item["plan"] = first_sentences(text)
        return index

    def _agent(self, index: int, rest: str) -> int:
        if rest.startswith("Querying agent for kernel modification"):
            self.item["change_kind"] = CandidateOrigin.SESSION
            self._in_session = True
        elif rest.startswith(_RATIONALE):
            text, index = _continuation(self.lines, index, limit=len(_RATIONALE) + _RATIONALE_CHARS)
            self.item["reasoning"] = rationale_text(text[len(_RATIONALE) :])
        elif rest.startswith("ERROR: "):
            self.item["agent_error"] = True
        elif (match := _COMMITTED.match(rest)) is not None:
            self.item["commit"] = match.group(1)
        elif rest.startswith("No tracked source changes"):
            self._decide(index, self._label(_AGENT, rest))
        return index

    def _decide(self, index: int, label: str | None) -> None:
        if label is not None:
            self.item["decision"] = label
            self.decision_index = index

    def _label(self, tag: str, rest: str) -> str | None:
        """KernelForge's decision label for one decision line, by the line shape its runner prints for that label."""

        if tag == _KEEP and _NEW_BEST.match(rest):
            return _KEEP
        if tag == _CRASH:
            return None
        if tag == _AGENT:
            if self.item.get("agent_error"):
                return "AGENT_ERROR"
            if self.item.get("session_end_reason") == "api_error":
                return "API_ERROR"
            return "NO_CHANGES"
        if _REVERT_PERF.match(rest):
            return "REVERT_PERF"
        if rest.startswith("Protected integrity violation"):
            return "REVERT_INTEGRITY"
        if rest.startswith(("Workspace still contended", "Device still contended")):
            return "REVERT_CONTENDED"
        if rest.startswith(("Iteration crashed", "Commit failed after validation")):
            return _CRASH
        if rest.startswith("Validation failed"):
            if not self._validated:
                return "BUILD_FAILED"
            return {
                "TIMEOUT": "REVERT_VALIDATION_TIMEOUT",
                "ERROR": "REVERT_VALIDATION_ERROR",
            }.get(self._validation or "", "REVERT_VALIDATION")
        return None

    def iteration(self, publishes: dict[str, str]) -> dict[str, Any]:
        """The iteration as its lines show it, up to its decision line."""

        item = self.item
        if self.decision_index is not None:
            item["completed_at"] = self.lines[self.decision_index].at
        item["alternatives"] = self.alternatives or None
        commit = item.get("commit")
        if commit and item.get("decision") == _KEEP:
            item["kb_record"] = next((record for sha, record in publishes.items() if sha.startswith(commit)), None)
        return item


def _stop(lines: list[LogLine], header_index: int, block_end: int, loop_end: int) -> dict[str, str]:
    """How the forge-loop stopped before an iteration logged a decision, with the line that shows it."""

    for index in range(header_index, min(block_end + 1, len(lines))):
        body = lines[index].body.strip()
        if lines[index].forge and body.startswith("ROUND REFUSED FOR BUDGET at iteration"):
            return {"stop": "budget-refused", "stop_line": body}
    if block_end == loop_end and loop_end < len(lines):
        body = lines[loop_end].body.strip()
        if lines[loop_end].tag == _FORGE_REWRITE:
            return {"stop": "cut-off", "stop_line": body}
        if body.startswith(_TRACEBACK):
            crash = next(
                (
                    lines[index].body.strip()
                    for index in range(loop_end, min(loop_end + 80, len(lines)))
                    if lines[index].forge and _EXCEPTION.match(lines[index].body.strip())
                ),
                body,
            )
            return {"stop": "crashed", "stop_line": crash}
        if not body.startswith(("BUDGET EXHAUSTED", "Autonomous loop complete")):
            return {"stop": "cut-off", "stop_line": body}
    if block_end >= len(lines):
        return {"stop": "log-ended", "stop_line": lines[-1].body.strip()}
    return {"stop": "no-decision", "stop_line": lines[header_index].body.strip()}


def _kb_publishes(lines: list[LogLine]) -> dict[str, str]:
    """The KB record forge-rewrite published for each kept commit, by the commit's hash."""

    publishes: dict[str, str] = {}
    for line in lines:
        match = _KB_PUBLISH.match(line.rest) if line.forge and line.tag == _FORGE_REWRITE else None
        if match and (record := _published_record(match.group(2))):
            publishes[match.group(1)] = record
    return publishes


def _published_record(payload: str) -> str | None:
    """The KB record a publication line names, or None when the line names why no record was written."""

    payload = payload.strip()
    return payload if _KB_RECORD.match(payload) else None


def _kb_written(lines: list[LogLine]) -> str | None:
    """The first KB record the forge-loop wrote among ``lines``."""

    return next(
        (
            match.group(1)
            for line in lines
            if line.forge and line.tag == "kb" and (match := _KB_WRITTEN.match(line.rest))
        ),
        None,
    )


def iterations(setup: Setup) -> list[dict[str, Any]]:
    """Each forge-loop iteration of the session, from its header to its decision line or to how the loop stopped."""

    if setup.loop_start is None:
        return []
    lines, stop = setup.lines, setup.loop_stop
    headers = [
        index
        for index in range(setup.loop_start, stop)
        if lines[index].forge and lines[index].tag is None and _HEADER.match(lines[index].body.strip())
    ]
    publishes = _kb_publishes(lines)
    found: list[dict[str, Any]] = []
    planning_from = setup.loop_start
    for position, header_index in enumerate(headers):
        block_end = headers[position + 1] if position + 1 < len(headers) else stop
        reader = _IterationReader(lines, header_index, planning_from)
        reader.read(block_end)
        iteration = reader.iteration(publishes)
        if reader.decision_index is None:
            iteration.update(_stop(lines, header_index, block_end, stop))
        if iteration.get("decision") == _KEEP and not iteration.get("kb_record"):
            iteration["kb_record"] = _kb_written(lines[header_index:block_end])
        found.append(iteration)
        planning_from = reader.end
    return found


def iteration_unit(run: Run, iteration: dict[str, Any], files: list[str]) -> dict[str, Any]:
    """Seq N: forge-loop iteration N, the one candidate it measured and KernelForge's decision on it."""

    unit = run.unit(
        f"iteration-{iteration['seq']}",
        "iteration",
        iteration["seq"],
        files,
        commit=iteration.get("commit"),
        kb_record=iteration.get("kb_record"),
    )
    unit["objective"] = OBJECTIVE
    unit["source_ms"] = run.source_ms
    unit.update({key: iteration[key] for key in _ITERATION_KEYS if key in iteration})
    return unit
