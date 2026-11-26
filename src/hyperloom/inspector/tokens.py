# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Token consumption over time: the LLM-call ledger merged with live specialist logs.

``reports/trace/llm_calls.jsonl`` is the source of truth, but a Claude CLI specialist lands there only when it ends,
every turn stamped with the same completion time. While it runs, its stream-JSON ``process.log`` already carries each
turn's input and cache usage, so those turns show up as live events and are swapped for the ledger rows of the same
``(task_id, turn)`` once the runner writes them; nothing is counted twice. A ledger row takes its model from the log's
turn, which names the model that answered it: the ledger puts the specialist's model on every turn, a sub-agent's on
another model too, and older runners put none. Output tokens are the exception: the CLI's per-message output counts
are placeholders and the real total comes on the closing ``result`` line, which the ledger puts on the last turn.
Until then a live turn's output is pending. A specialist that ends without that line (cancelled, or timed out) leaves
its turns' output unreported, unless the runner recovered its ledger rows from the Claude CLI's own session record.

Each event is placed when its request was sent: a ledger row at ``ts`` less its latency, a specialist turn at the
timestamp of the tool result it answers (the task's start for the first turn).

The ``result`` line's ``modelUsage`` also counts the CLI's side calls (WebSearch, WebFetch and the like), which never
appear as assistant messages and so never reach the ledger; what it reports beyond the turns becomes a ``result``
event per model at the end of the specialist. The ledger leaves a side call's output in the session total on the last
turn; when the side call ran on another model, that output moves to its model's event, so it counts toward that
model. OpenAI-protocol rows count cached input inside ``input_tokens`` and reasoning inside
``output_tokens``; both are taken out so the five kinds add up the same way for every provider. Which rows those are
is told by their writer (:func:`_openai_usage`), not by the model's name alone: a model such as Kimi is served over
either protocol.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.inspector.collectors import ENDED_STATES, parse_ts, resolve_inside
from hyperloom.inspector.graph import agent_node, task_node
from hyperloom.inspector.model_names import normalize
from hyperloom.inspector.token_terms import NO_MODEL, RECENT_SEC, TERMS

# Anthropic semantics: ``input`` excludes the cached prefix, so the five kinds add up to what was processed.
KINDS = ("input", "cache_write", "cache_read", "output", "reasoning")
_INPUT, _CACHE_WRITE, _CACHE_READ, _OUTPUT, _REASONING = KINDS
# The kinds that make up a call's input.
_INPUT_KINDS = (_INPUT, _CACHE_WRITE, _CACHE_READ)
# The count fields of a usage record (a ledger row, or the usage of a Claude CLI message), by the kind they count.
_USAGE_INPUT = "input_tokens"
_USAGE_CACHE_WRITE = "cache_creation_input_tokens"
_USAGE_CACHE_READ = "cache_read_input_tokens"
_USAGE_OUTPUT = "output_tokens"
_LEDGER_KEYS = {
    "input": _USAGE_INPUT,
    "cache_write": _USAGE_CACHE_WRITE,
    "cache_read": _USAGE_CACHE_READ,
    "output": _USAGE_OUTPUT,
    "reasoning": "reasoning_output_tokens",
}
_USAGE_KEYS = (_USAGE_INPUT, _USAGE_OUTPUT, _USAGE_CACHE_WRITE, _USAGE_CACHE_READ)
_SIDE_KEYS = (_USAGE_INPUT, _USAGE_CACHE_WRITE, _USAGE_CACHE_READ)
_MODEL_USAGE = {
    "input_tokens": "inputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
    "output_tokens": "outputTokens",
    "web_searches": "webSearchRequests",
}
# Which rows count by the OpenAI protocol (cached input inside input, reasoning inside output) is the writer's: the
# proposal scorer only talks it; the critic talks it or the Anthropic one, and only its OpenAI path reports reasoning;
# roles and specialists on a Codex backend leave only their model to tell, a Codex one (normalized id prefixes).
_OPENAI_WRITERS = ("proposal_scorer",)
_OPENAI_WHEN_REASONING = ("critic",)
_CODEX_MODELS = ("gpt-", "o1", "o3", "o4", "codex")
_STEPS = (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)
MAX_BUCKETS = 180
# Where an event comes from: the ledger (reports/trace/llm_calls.jsonl), a running specialist's process.log not yet in
# the ledger, or a Claude CLI specialist's closing result line beyond its turns; and the status of a call that failed.
_SOURCE_LEDGER = "ledger"
_SOURCE_LIVE = "live"
_SOURCE_RESULT = "result"
_STATUS_ERROR = "error"
# A specialist's task kind, and the component its LLM calls are recorded under.
_SPECIALIST = "specialist"
# The field of an event (and of a call) that names its model; a call that names none lacks it.
_MODEL = "model"
_CHUNK = 32 << 20


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _result_output(result: dict) -> int | None:
    """The session's output total off a ``result`` line: ``modelUsage`` summed, else ``usage.output_tokens``."""
    per_model = result.get("modelUsage")
    if isinstance(per_model, dict):
        counts = [_int(m.get("outputTokens")) for m in per_model.values() if isinstance(m, dict)]
        counts = [c for c in counts if c is not None]
        if counts:
            return sum(counts)
    usage = result.get("usage")
    return _int(usage.get("output_tokens")) if isinstance(usage, dict) else None


def _model_usage(result: dict) -> dict[str, dict[str, int]]:
    per_model = result.get("modelUsage")
    if not isinstance(per_model, dict):
        return {}
    return {
        str(model): {k: _int(u.get(src)) or 0 for k, src in _MODEL_USAGE.items()}
        for model, u in per_model.items()
        if isinstance(u, dict)
    }


def _openai_usage(row: dict) -> bool:
    """Whether the ledger row's writer counted it by the OpenAI protocol."""
    component = row.get("component") or row.get("role")
    if component in _OPENAI_WRITERS:
        return True
    if component in _OPENAI_WHEN_REASONING:
        return row.get("reasoning_output_tokens") is not None
    return normalize(row.get("model")).startswith(_CODEX_MODELS)


def _openai_split(counts: dict, row: dict) -> dict:
    """OpenAI-protocol usage nests cached input in ``input`` and reasoning in ``output``; take them out."""
    if not _openai_usage(row):
        return counts
    out = dict(counts)
    for whole, part in ((_INPUT, _CACHE_READ), (_OUTPUT, _REASONING)):
        if out[part] and out[whole] is not None and out[part] <= out[whole]:
            out[whole] -= out[part]
    return out


class _LiveTurn(NamedTuple):
    """A specialist turn as its live log has it: the turn's usage, and the parser of that log."""

    usage: dict
    parser: StreamTurns


class _CallAt(NamedTuple):
    """Where an LLM call falls on the timeline: when its request was sent, who made it, and its task and turn."""

    t: float
    component: str
    tid: Any
    turn: int | None


class _LedgerRow(NamedTuple):
    """One ledger row, read once: the row, who made the call, its task and turn, when it completed, and its turn in
    the live log when the log has it."""

    row: dict
    component: str
    tid: Any
    turn: int | None
    ts: float | None
    live: _LiveTurn | None


def _ledger_counts(r: _LedgerRow) -> dict:
    """The ledger row's token counts, with the input side taken from its live turn's log when the log counts one."""
    counts = _openai_split({k: _int(r.row.get(src)) for k, src in _LEDGER_KEYS.items()}, r.row)
    if r.live is not None:
        # The input side is the turn's as its log has it (the one row the runner writes for a single turn is the
        # session's, which may count side calls), unless the log streamed zeroed usage and the row was recovered.
        logged = {k: _int(r.live.usage.get(_LEDGER_KEYS[k])) for k in _INPUT_KINDS}
        counts.update(logged if any(logged.values()) else {})
        counts["output"] = r.live.parser.own_output(r.turn, counts["output"])
    return counts


class StreamTurns:
    """Per-turn usage of one Claude CLI stream-JSON log, read incrementally.

    Mirrors ``parse_claude_stream_json_turn_usages``, which the runner uses for the ledger: a turn is a distinct
    assistant message id counted with the usage of its first line, and the ``result`` line's output total replaces
    the per-turn placeholders. Only complete lines are read; a rewritten (shorter) file starts over.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.mtime: float | None = None
        self._reset()

    def _reset(self) -> None:
        self.offset = 0
        self.turns: list[dict] = []
        self.saw_id = False
        self.done = False
        self.done_t: float | None = None
        self.duration: float | None = None
        self.session_output: int | None = None
        self.session_input = 0
        self.model_usage: dict[str, dict[str, int]] = {}
        self._ids: set[str] = set()
        self.last_ts: float | None = None
        self._skip = False

    def read(self) -> bool:
        """Consume new complete lines; True when the turns changed."""
        try:
            st = self.path.stat()
        except OSError:
            return False
        self.mtime = st.st_mtime
        reset = st.st_size < self.offset
        if reset:
            self._reset()
        if st.st_size == self.offset:
            return reset
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read(min(st.st_size - self.offset, _CHUNK))
        except OSError:
            return reset
        end = chunk.rfind(b"\n")
        if end < 0:
            if len(chunk) >= _CHUNK:  # one line longer than a chunk: drop it rather than stall
                self.offset += len(chunk)
                self._skip = True
            return reset
        self.offset += end + 1
        lines = chunk[:end].split(b"\n")
        if self._skip:
            lines, self._skip = lines[1:], False
        before = (len(self.turns), self.done, self.session_output)
        for line in lines:
            self._line(line)
        if self.done and self.done_t is None:
            self.done_t = st.st_mtime
        return reset or (len(self.turns), self.done, self.session_output) != before

    def _line(self, raw: bytes) -> None:
        raw = raw.strip()
        if not raw.startswith(b"{"):
            return
        try:
            obj = json.loads(raw)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        kind = obj.get("type")
        if kind == "result":
            out = _result_output(obj)
            if out is not None:
                self.session_output = out
            usage = obj.get("usage")
            self.session_input = sum(_int(usage.get(k)) or 0 for k in _SIDE_KEYS) if isinstance(usage, dict) else 0
            self.model_usage = _model_usage(obj)
            duration = _int(obj.get("duration_ms"))
            self.duration = duration / 1000.0 if duration and duration > 0 else None
            self.done = True
            return
        if kind == "assistant":
            self._assistant(obj.get("message"))
        ts = parse_ts(obj.get("timestamp"))
        if ts is not None and (self.last_ts is None or ts > self.last_ts):
            self.last_ts = ts

    def _assistant(self, message: Any) -> None:
        if not isinstance(message, dict) or not isinstance(message.get("usage"), dict):
            return
        counts = {k: _int(message["usage"].get(k)) for k in _USAGE_KEYS}
        if all(v is None for v in counts.values()):
            return
        mid = message.get("id")
        if isinstance(mid, str) and mid:
            self.saw_id = True
            if mid in self._ids:
                return
            self._ids.add(mid)
        model = message.get("model")
        self.turns.append({"t": self.last_ts, "model": model if isinstance(model, str) else None, **counts})

    def usages(self) -> list[dict]:
        """Per-turn rows as the runner will write them; output is ``None`` until the ``result`` line."""
        if len(self.turns) > 1 and not self.saw_id:
            return []  # the runner can't split such a log either; it writes one cumulative row
        if self.session_input and not any(t[k] for t in self.turns for k in _SIDE_KEYS):
            return []  # every turn reports zero usage, the result line the real counts: one cumulative row too
        rows = [dict(t) for t in self.turns]
        if not self.done:
            for r in rows:
                r["output_tokens"] = None
            return rows
        if rows and self._total_on_last():
            for r in rows[:-1]:
                r["output_tokens"] = None
            rows[-1]["output_tokens"] = self.session_output
        return rows

    def _total_on_last(self) -> bool:
        observed = sum(t["output_tokens"] or 0 for t in self.turns)
        return not (self.session_output is not None and observed == self.session_output)

    def side_usage(self) -> list[dict]:
        """Per model, what ``modelUsage`` counts beyond the turns: input and cache tokens, web searches, and the
        output of a model other than the last turn's.

        Those are the CLI's side calls, which never show up as assistant messages. Their output is in the session
        total the runner puts on the last turn; a side model's share moves to its row (see ``own_output``), so it is
        counted toward that model. Turns of a model the result does not name count against its largest entry.
        """
        if not self.done or not self.model_usage or not self.usages():
            return []
        usage = self.model_usage
        main = max(usage, key=lambda m: sum(usage[m][k] for k in _SIDE_KEYS))
        seen = {m: dict.fromkeys(_SIDE_KEYS, 0) for m in usage}
        for turn in self.turns:
            acc = seen[turn["model"] if turn["model"] in usage else main]
            for k in _SIDE_KEYS:
                acc[k] += turn[k] or 0
        last = self.turns[-1]["model"] if self.turns[-1]["model"] in usage else main
        moved = self._total_on_last()
        rows = []
        for model, u in usage.items():
            extra = {k: max(0, u[k] - seen[model][k]) for k in _SIDE_KEYS}
            extra["output_tokens"] = u["output_tokens"] if moved and model != last else 0
            if any(extra.values()) or u["web_searches"]:
                rows.append({"model": model, **extra, "web_searches": u["web_searches"]})
        return rows

    def own_output(self, turn: int | None, output: int | None) -> int | None:
        """A turn's output without what ``side_usage`` moved off the session total on the last turn."""
        if turn != len(self.turns) or output is None or output != self.session_output:
            return output
        return output - sum(r["output_tokens"] for r in self.side_usage())


def _started(task: dict) -> float | None:
    """The task's last ``running`` transition, else its creation (the timeline bar's start)."""
    start = parse_ts(task.get("created_at"))
    for tr in task.get("history") or []:
        ts = parse_ts(tr.get("ts")) if isinstance(tr, dict) else None
        if ts is not None and tr.get("to") == "running" and (start is None or ts >= start):
            start = ts
    return start


def _phase_starts(history: Any) -> list[tuple[float, str]]:
    rows = []
    for r in history if isinstance(history, list) else []:
        if not isinstance(r, dict) or r.get("from_phase") == r.get("to_phase"):
            continue
        t = parse_ts(r.get("ts_unix")) or parse_ts(r.get("ts"))
        if t is not None:
            rows.append((t, str(r.get("to_phase") or "")))
    return sorted(rows)


def _empty() -> dict:
    return {
        **dict.fromkeys(KINDS, 0),
        "total": 0,
        "calls": 0,
        "errors": 0,
        "web_searches": 0,
    }


def _add(acc: dict, ev: dict) -> None:
    for k in KINDS:
        acc[k] += ev[k] or 0
    acc["total"] += ev["total"]
    acc["calls"] += ev["source"] != _SOURCE_RESULT  # side calls, not one call of their own
    acc["errors"] += ev["status"] == _STATUS_ERROR
    acc["web_searches"] += ev["web_searches"]


def _totals(events: list[dict]) -> dict:
    """Sums per kind plus ``calls``, ``errors``, ``web_searches``, then ``live`` (not yet in the ledger),
    ``output_pending`` and ``output_unreported``."""
    acc = _empty()
    for ev in events:
        _add(acc, ev)
    acc["live"] = sum(e["source"] == _SOURCE_LIVE for e in events)
    acc["output_pending"] = sum(e["output_pending"] for e in events)
    acc["output_unreported"] = sum(e["output_unreported"] for e in events)
    return acc


def _rollup(events: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ev in events:
        _add(out.setdefault(str(ev.get(key) or NO_MODEL), _empty()), ev)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["total"]))


def _bucket_rows(events: list[dict], t0: float, step: int) -> tuple[list[str], list[list]]:
    """The series' node names and its ``[bucket, node index, *KINDS, calls]`` rows."""
    nodes: dict[str, int] = {}
    cells: dict[tuple[int, int], list[float]] = {}
    calls = len(KINDS)
    for ev in events:
        key = (int((ev["t"] - t0) // step), nodes.setdefault(str(ev["node"]), len(nodes)))
        cell = cells.setdefault(key, [0] * (calls + 1))
        for j, k in enumerate(KINDS):
            cell[j] += ev[k] or 0
        cell[calls] += ev["source"] != _SOURCE_RESULT
    rows = [[i, n, *cell] for (i, n), cell in sorted(cells.items())]
    return list(nodes), rows


def _rollups(events: list[dict]) -> dict[str, dict]:
    """The *events*' ``totals``, and their totals ``by_component``, ``by_node`` and ``by_model``."""
    return {
        "totals": _totals(events),
        "by_component": _rollup(events, "component"),
        "by_node": _rollup(events, "node"),
        "by_model": _rollup(events, _MODEL),
    }


def _step(span: float) -> int:
    return next((s for s in _STEPS if span / s <= MAX_BUCKETS), _STEPS[-1])


class TokenTracker:
    """Merges ledger rows with live specialist turns into time-ordered token events; not thread-safe on its own
    (the watcher calls it under its lock)."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self._ledger: list[dict] = []
        self._logs: dict[str, StreamTurns] = {}
        self._tasks: dict[str, tuple[str, float | None, bool]] = {}  # node, start, ended
        self._phases: list[tuple[float, str]] = []
        self._events: list[dict] = []
        self._times: list[float] = []
        self._dirty = True

    # ----- inputs --------------------------------------------------------------------------------------------

    def add_ledger(self, row: Any) -> None:
        if isinstance(row, dict):
            self._ledger.append(row)
            self._dirty = True

    def latest_write(self) -> float | None:
        """When a running specialist's log last changed: a long turn writes nothing else to the session."""
        return max((p.mtime for p in self._logs.values() if p.mtime is not None and not p.done), default=None)

    def poll(self, tasks: list[dict], phase_history: Any = None) -> bool:
        """Track the tasks' nodes and starts and read new specialist log lines; True when the events changed."""
        changed = False
        meta = {}
        for task in tasks:
            tid = task.get("task_id")
            if not isinstance(tid, str) or not tid:
                continue
            meta[tid] = (task_node(task), _started(task), task.get("state") in ENDED_STATES)
            if task.get("kind") == _SPECIALIST and tid not in self._logs and "/" not in tid:
                path = resolve_inside(self.session_dir, f"runs/specialist/{tid}/process.log")
                if path is not None:
                    self._logs[tid] = StreamTurns(path)
        if meta != self._tasks:
            self._tasks, changed = meta, True
        phases = _phase_starts(phase_history)
        if phases != self._phases:
            self._phases, changed = phases, True
        for parser in self._logs.values():
            changed |= parser.read()
        if not (changed or self._dirty):
            return False
        self._rebuild()
        return True

    # ----- events --------------------------------------------------------------------------------------------

    def events(self) -> list[dict]:
        """All events, oldest first, each with running ``cumulative`` totals."""
        if self._dirty:
            self._rebuild()
        return self._events

    def _rebuild(self) -> None:
        live = {
            (tid, i + 1): _LiveTurn(u, parser)
            for tid, parser in self._logs.items()
            for i, u in enumerate(parser.usages())
        }
        events, finished = self._ledger_events(live)
        events += self._live_events(live)
        events += self._result_events(finished)
        events.sort(key=lambda e: e["t"])
        self._cumulate(events)
        self._events, self._times = events, [e["t"] for e in events]
        self._dirty = False

    def _ledger_time(self, r: _LedgerRow) -> float | None:
        """When the ledger row's request was sent: its live turn's time, the task's start for a specialist's first
        turn, else the row's completion less its latency."""
        if r.live is not None:
            t = self._turn_time(r.tid, r.turn, r.live.parser)
        else:
            t = self._start(r.tid) if r.component == _SPECIALIST and r.turn == 1 else None
        if t is None and r.ts is not None:
            latency = _int(r.row.get("latency_ms")) if r.component != _SPECIALIST else None
            t = r.ts - latency / 1000.0 if latency and latency > 0 else r.ts
        return t

    def _ledger_events(self, live: dict[tuple[str, int], _LiveTurn]) -> tuple[list[dict], dict]:
        """The ledger's rows as events, and when each specialist's rows were written. A specialist row takes the
        time and model of its turn in the live log, and that turn is removed from *live*; a first turn no log line
        matches (a Codex specialist's) is placed at the task's start, when it was sent."""
        events = []
        finished: dict[Any, float] = {}
        for row in self._ledger:
            component = str(row.get("component") or row.get("role") or "llm")
            tid, turn = row.get("task_id"), _int(row.get("turn"))
            match = live.pop((tid, turn), None) if component == _SPECIALIST and turn else None
            r = _LedgerRow(row, component, tid, turn, parse_ts(row.get("ts")), match)
            if component == _SPECIALIST and r.ts is not None:
                finished[tid] = max(r.ts, finished.get(tid, r.ts))
            t = self._ledger_time(r)
            if t is None:
                continue
            counts = _ledger_counts(r)
            if match is not None and match.usage["model"]:
                row = {**row, "model": match.usage["model"]}
            status = _STATUS_ERROR if row.get("status") == _STATUS_ERROR else "ok"
            ev = self._event(_CallAt(t, component, tid, turn), row, counts, status, _SOURCE_LEDGER)
            ev["output_unreported"] = match is not None and not match.parser.done and counts["output"] is None
            events.append(ev)
        return events, finished

    def _live_events(self, live: dict[tuple[str, int], _LiveTurn]) -> list[dict]:
        """Specialists' turns the ledger has not got: their output is pending while the task runs, and was never
        reported when it ended without its log's result line (cancelled, say)."""
        events = []
        for (tid, turn), (usage, parser) in live.items():
            t = self._turn_time(tid, turn, parser) or parser.mtime
            if t is None:
                continue
            counts = {k: usage.get(src) for k, src in _LEDGER_KEYS.items()}
            counts["output"] = parser.own_output(turn, counts["output"])
            ev = self._event(_CallAt(t, _SPECIALIST, tid, turn), usage, counts, "ok", _SOURCE_LIVE)
            ended = self._tasks.get(tid, ("", None, False))[2]
            ev["output_pending"] = not parser.done and not ended
            ev["output_unreported"] = not parser.done and ended
            events.append(ev)
        return events

    def _result_events(self, finished: dict) -> list[dict]:
        """What each finished specialist's result line counts beyond its turns, as events at its end."""
        events = []
        for tid, parser in self._logs.items():
            if not parser.done:
                continue
            t = self._end_time(tid, parser, finished.get(tid))
            for side in parser.side_usage() if t is not None else ():
                counts = {k: side.get(src) for k, src in _LEDGER_KEYS.items()}
                ev = self._event(_CallAt(t, _SPECIALIST, tid, None), side, counts, "ok", _SOURCE_RESULT)
                events.append(ev)
        return events

    def _cumulate(self, events: list[dict]) -> None:
        """Each event's running totals, in place."""
        cum: dict[str, Any] = dict.fromkeys((*KINDS, "total"), 0)
        for ev in events:
            for k in KINDS:
                cum[k] += ev[k] or 0
            cum["total"] += ev["total"]
            ev["cumulative"] = dict(cum)

    def _end_time(self, tid: str, parser: StreamTurns, ledger_ts: float | None) -> float | None:
        """When a finished specialist ended, from the session rather than file times (a copied session has new
        ones): its ledger rows' completion stamp, else its start plus the CLI's duration, else its last stamp."""
        start = self._start(tid)
        if ledger_ts is not None:
            return ledger_ts
        if start is not None and parser.duration is not None:
            return start + parser.duration
        return parser.last_ts or start or parser.done_t

    def _turn_time(self, tid: Any, turn: int | None, parser: StreamTurns) -> float | None:
        i = (turn or 0) - 1
        t = parser.turns[i]["t"] if 0 <= i < len(parser.turns) else None
        return t if t is not None else self._start(tid)

    def _start(self, tid: Any) -> float | None:
        return self._tasks[tid][1] if tid in self._tasks else None

    def _event(self, at: _CallAt, row: dict, counts: dict, status: str, source: str) -> dict:
        # Same node as the harness graph: the task's node, else the component's agent node.
        node = self._tasks[at.tid][0] if at.tid in self._tasks else agent_node(at.component)
        phase = row.get("phase")
        i = bisect.bisect_right(self._phases, (at.t, "\uffff")) - 1
        if i >= 0:
            phase = self._phases[i][1]
        return {
            "t": round(at.t, 3),
            "component": at.component,
            "node": node,
            "model": row.get("model"),
            "task_id": at.tid,
            "turn": at.turn,
            "call_id": row.get("call_id"),
            "phase": phase,
            **counts,
            "total": sum(v or 0 for v in counts.values()),
            "web_searches": _int(row.get("web_searches")) or 0,
            "status": status,
            "source": source,
            "output_pending": False,
            "output_unreported": False,
        }

    def _upto(self, clock: float | None) -> list[dict]:
        events = self.events()
        return events if clock is None else events[: bisect.bisect_right(self._times, clock)]

    # ----- views ---------------------------------------------------------------------------------------------

    def view(self, now: float, upto: float | None = None, start: float | None = None, end: float | None = None) -> dict:
        """Totals, rollups and a bucketed series for the page, up to *upto* (``None``: everything).

        The series spans *start* (default: the first event) to *end* (default: *now*), stretched to cover every
        event. ``series.rows`` holds one ``[bucket, node index, *KINDS, calls]`` row per non-empty
        (bucket, node) pair; buckets are ``step`` seconds from ``series.start``, at most ``MAX_BUCKETS`` of them.
        """
        events = self._upto(upto)
        pending = sorted({str(e["task_id"]) for e in events if e["output_pending"]})
        unreported = sorted({str(e["task_id"]) for e in events if e["output_unreported"]})
        recent = _totals([e for e in events if e["t"] > now - RECENT_SEC])
        end = now if end is None else end
        t0 = min(x for x in (start, events[0]["t"] if events else None, end) if x is not None)
        t1 = max(end, events[-1]["t"]) if events else end
        step = _step(max(t1 - t0, 1.0))
        nodes, rows = _bucket_rows(events, t0, step)
        rollups = _rollups(events)
        return {
            **rollups,
            "pending_tasks": pending,
            "unreported_tasks": unreported,
            "recent": {"window_sec": RECENT_SEC, **{k: recent[k] for k in ("total", "calls")}},
            "series": {
                "start": t0,
                "end": t1,
                "step": step,
                "kinds": list(KINDS),
                "nodes": nodes,
                "rows": rows,
            },
            "terms": TERMS,
        }
