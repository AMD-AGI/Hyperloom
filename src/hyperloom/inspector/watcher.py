# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Polls a session directory and keeps what the page needs, in live or replay mode.

Two kinds of data come out of here:

* **Streams** are append-only and time-stamped (bus events, LLM calls, journal
  entries, critic verdicts, resource samples). Each is kept sorted by time; a
  client holds a per-stream count and receives the items released since then.
  In replay a stream releases an item once the virtual clock passes its time,
  so a backward scrub shrinks the released prefix and the client gets a reset.
* **Views** are recomputed at the clock on every request (run info, tasks with
  health, the harness graph, timeline bars and phase bands, badges).

Nothing here writes into the session directory, and every payload is
redacted before it leaves the process.
"""

from __future__ import annotations

import bisect
import glob
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.inspector import artifacts as artifacts_mod
from hyperloom.inspector import derive, graph
from hyperloom.inspector.collectors import (
    ENDED_STATES,
    ChangedFiles,
    CoordinatorDb,
    JsonlTail,
    container_memory,
    first_exception,
    gpu_metrics,
    newest_mtime,
    parse_ts,
    read_json,
    resolve_inside,
    specialist_turn,
)
from hyperloom.inspector.history import MetricsHistory
from hyperloom.inspector.redact import redact
from hyperloom.inspector.session_layout import STATE_FILE, coordinator_db, variant_dir_name

log = logging.getLogger(__name__)

_METRICS_STREAM = "metrics"
STREAMS = ("event", "llm_call", "journal", "review", _METRICS_STREAM)
# Folders of a session directory and the task kind and state the watcher looks for.
_REPORTS_DIR = "reports"
_RUNS_DIR = "runs"
_BASELINE_KIND = "baseline"
_RUNNING_STATE = "running"
_TS_KEYS = ("ts_unix", "ts", "recorded_unix", "completed_at", "started_at", "start_time", "created_at")
_METRIC_SAMPLES = 4000
_METRIC_EVERY_SEC = 5.0
_INDEX_EVERY_SEC = 10.0
_ACTIVE_SEC = 600.0


def item_time(rec: dict) -> float | None:
    for key in _TS_KEYS:
        t = parse_ts(rec.get(key))
        if t is not None:
            return t
    return None


class Clock:
    """Wall time in live mode; in replay, virtual time that runs at *speed* from the session start."""

    def __init__(self, replay: bool = False, speed: float = 1.0) -> None:
        self.replay = replay
        self.speed = speed
        self.start: float | None = None
        self.end: float | None = None
        self._anchor_virtual = 0.0
        self._anchor_wall = time.monotonic()
        self.paused = False

    def now(self) -> float:
        if not self.replay:
            return time.time()
        if self.start is None:
            return 0.0
        t = self._anchor_virtual
        if not self.paused:
            t += (time.monotonic() - self._anchor_wall) * self.speed
        return min(t, self.end) if self.end is not None else t

    def set_range(self, start: float | None, end: float | None) -> None:
        if not self.replay or start is None:
            return
        if self.start is None:
            self._anchor_virtual, self._anchor_wall = start, time.monotonic()
        self.start, self.end = start, end

    def seek(self, t: float | None = None, speed: float | None = None, paused: bool | None = None) -> None:
        current = self.now()
        if speed is not None and speed > 0:
            self.speed = speed
        if paused is not None:
            self.paused = paused
        self._anchor_virtual = current if t is None else max(self.start or t, t)
        self._anchor_wall = time.monotonic()

    def info(self) -> dict:
        return {
            "mode": "replay" if self.replay else "live",
            "now": self.now(),
            "start": self.start,
            "end": self.end,
            "speed": self.speed,
            "paused": self.paused,
        }


class Stream:
    """Time-sorted item list with release-by-clock.

    Cursors (``released``, ``upto``, ``between``) are absolute indices: trimming the head bumps ``dropped``, so a
    client's cursor stays valid after old items are gone.
    """

    def __init__(self) -> None:
        self.times: list[float] = []
        self.items: list[dict] = []
        self.dropped = 0

    def add(self, rec: dict, t: float | None) -> None:
        t = t if t is not None else (self.times[-1] if self.times else 0.0)
        i = bisect.bisect_right(self.times, t)
        self.times.insert(i, t)
        self.items.insert(i, {**rec, "_t": t})

    def released(self, clock: float) -> int:
        return self.dropped + bisect.bisect_right(self.times, clock)

    def upto(self, n: int) -> list[dict]:
        return self.items[: max(0, n - self.dropped)]

    def between(self, a: int, b: int) -> list[dict]:
        return self.items[max(0, a - self.dropped) : max(0, b - self.dropped)]

    def trim(self, keep: int) -> None:
        extra = len(self.items) - keep
        if extra > 0:
            del self.items[:extra], self.times[:extra]
            self.dropped += extra


@dataclass(frozen=True)
class WatchOptions:
    """How a :class:`Watcher` reads the session: on the wall clock, or with *replay* on a virtual clock running at
    *speed*; it polls every *poll_sec* seconds, and a task whose log is older than *stall_after_sec* seconds is
    stalled. *sample_metrics* says whether it samples resource use itself; ``None`` means yes live, no in replay."""

    replay: bool = False
    speed: float = 1.0
    poll_sec: float = 1.5
    stall_after_sec: float = 900.0
    sample_metrics: bool | None = None


class Watcher:
    """Owns the collectors and answers snapshot/stream queries; thread-safe."""

    def __init__(
        self,
        session_dir: Path,
        options: WatchOptions | None = None,
        history: MetricsHistory | None = None,
    ) -> None:
        options = options or WatchOptions()
        self.session_dir = Path(session_dir)
        self.clock = Clock(replay=options.replay, speed=options.speed)
        self.poll_sec = options.poll_sec
        self.stall_after_sec = options.stall_after_sec
        self.sample_metrics = (not options.replay) if options.sample_metrics is None else options.sample_metrics
        self.lock = threading.RLock()
        self.version = 0
        self.streams = {name: Stream() for name in STREAMS}
        db_path, self.db_problem = coordinator_db(self.session_dir)
        self.db = CoordinatorDb(db_path)
        reports = self.session_dir / _REPORTS_DIR
        self._llm_tail = JsonlTail(reports / "trace" / "llm_calls.jsonl")
        self._timeline_files = ChangedFiles(reports / "sbd_v6" / "timeline", "*.json")
        self._reviews = ChangedFiles(self.session_dir / "critic-workdir", "*/review.json")
        self.state: dict = {}
        self.manifest: dict = {}
        self.pod_history: list[dict] = []
        self.tasks: list[dict] = []
        self.results: dict[str, Any] = {}
        self.unit_errors: dict[tuple[str, str], str | None] = {}
        self.timeline: dict[str, dict] = {}
        self.journal_seen = 0
        self.artifact_index = artifacts_mod.ArtifactIndex()
        self._last_index = 0.0
        self._last_sample = 0.0
        self.history = history
        if history is not None:
            for rec in history.load():
                self.streams[_METRICS_STREAM].add({k: v for k, v in rec.items() if k != "t"}, rec["t"])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ----- polling -------------------------------------------------------------------------------------------

    def start(self) -> None:
        self.poll()
        self._thread = threading.Thread(target=self._run, name="inspector-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            try:
                self.poll()
            except Exception:  # a bad read must never kill the watcher
                log.debug("inspector: poll failed", exc_info=True)

    def poll(self) -> None:
        with self.lock:
            changed = self._poll_files()
            changed |= self._poll_db()
            if self.sample_metrics:
                changed |= self._sample_metrics()
            now = time.monotonic()
            if not self._last_index or (changed and now - self._last_index > _INDEX_EVERY_SEC):
                self.artifact_index = artifacts_mod.index(self.session_dir, self.task_nodes())
                self._last_index = now
            self._update_range()
            if changed:
                self.version += 1

    def _poll_files(self) -> bool:
        changed = False
        state = read_json(self.session_dir / STATE_FILE)
        if isinstance(state, dict) and state != self.state:
            self.state, changed = state, True
        if not self.manifest:
            self.manifest = read_json(self.session_dir / "manifest.json") or {}
        history = _read_jsonl(self.session_dir / "runtime" / "pod_history.jsonl")
        if len(history) != len(self.pod_history):
            self.pod_history, changed = history, True
        for rec in self._llm_tail.read_new():
            self.streams["llm_call"].add(rec, item_time(rec))
            changed = True
        for path, rec in self._timeline_files.read_changed():
            if isinstance(rec, dict):
                self.timeline[path.name] = rec
                changed = True
        for path, rec in self._reviews.read_changed():
            for verdict in (rec or {}).get("review_verdicts") or []:
                t = newest_mtime(path)
                self.streams["review"].add({**verdict, "turn": path.parent.name}, t)
                changed = True
        changed |= self._poll_journal()
        return changed

    def _poll_journal(self) -> bool:
        journal = read_json(self.session_dir / _REPORTS_DIR / "optimization_journal.json")
        entries = derive.throughput_series(journal if isinstance(journal, dict) else None)
        if len(entries) <= self.journal_seen:
            return False
        for e in entries[self.journal_seen :]:
            self.streams["journal"].add(e, parse_ts(e.get("ts")))
        self.journal_seen = len(entries)
        return True

    def _poll_db(self) -> bool:
        """New bus events and the task table, from the database Hyperloom keeps them in now: it moves out of the
        session once a leg holds a credential. Nothing is read while that one cannot be (``db_problem`` says why)."""
        db_path, problem = coordinator_db(self.session_dir)
        changed, self.db_problem = problem != self.db_problem, problem
        if db_path != self.db.path:
            self.db = CoordinatorDb(db_path)
        if problem:
            return changed
        for ev in self.db.new_events():
            payload = ev.get("payload")
            if ev.get("topic") == "delegated_result" and isinstance(payload, dict) and payload.get("task_id"):
                self.results[payload["task_id"]] = {"result": payload.get("result"), "t": parse_ts(ev.get("ts"))}
            self.streams["event"].add(ev, parse_ts(ev.get("ts")))
            changed = True
        tasks = self.db.tasks()
        if tasks != self.tasks:
            self.tasks, changed = tasks, True
        return changed

    def _sample_metrics(self) -> bool:
        """Memory and GPU use of this machine, every few seconds; none while the optimizer is known to run on another
        host, whose resources they would not be."""
        now = time.time()
        if now - self._last_sample < _METRIC_EVERY_SEC:
            return False
        self._last_sample = now
        if self.optimizer_status().get("host"):
            return False
        sample = {"memory": container_memory(), "gpus": gpu_metrics()}
        stream = self.streams[_METRICS_STREAM]
        stream.add(sample, now)
        stream.trim(_METRIC_SAMPLES)
        if self.history is not None:
            self.history.append({"t": now, **sample})
        return True

    def _update_range(self) -> None:
        # Resource samples can come from the history file and outlive the run, so they don't set the range.
        timed = [s for name, s in self.streams.items() if s.times and name != _METRICS_STREAM]
        times = [s.times[0] for s in timed]
        times += [t for t in (parse_ts(x.get("created_at")) for x in self.tasks) if t is not None]
        start = parse_ts(self.state.get("start_ts")) or (min(times) if times else None)
        last = self._last_activity()
        self.clock.set_range(start, last if last is not None else start)

    def _last_activity(self) -> float | None:
        """The run's last recorded activity: its streams but resources, and its tasks' updates."""
        ends = [s.times[-1] for name, s in self.streams.items() if s.times and name != _METRICS_STREAM]
        ends += [t for t in (parse_ts(x.get("updated_at")) for x in self.tasks) if t is not None]
        return max(ends, default=None)

    # ----- views ---------------------------------------------------------------------------------------------

    def task_nodes(self) -> dict[str, str]:
        return {t["task_id"]: graph.task_node(t) for t in self.tasks}

    def tasks_at(self, clock: float) -> list[dict]:
        """Tasks as of *clock*, each with ``node``, ``label``, ``health`` and ``reason``."""
        reasons = {r.get("task_id"): r.get("reason") for r in self.state.get("specialist_rounds") or []}
        out = []
        for task in self.tasks:
            t = derive.task_state_at(task, clock, parse_ts) if self.clock.replay else dict(task)
            if t is None:
                continue
            res = self.results.get(t["task_id"])
            result = res["result"] if res and (res["t"] is None or res["t"] <= clock) else None
            log_age = self._log_age(t) if not self.clock.replay and t.get("state") == _RUNNING_STATE else None
            reason = reasons.get(t["task_id"]) if t.get("state") == "succeeded" else None
            health = derive.task_health(t, result, reason, log_age, self.stall_after_sec)
            node = graph.task_node(t)
            label = graph.specialist_domain(t) if t.get("kind") == "specialist" else t.get("kind")
            out.append({**t, **health, "node": node, "label": label, "progress": self._progress(t, result, clock)})
        return out

    def _progress(self, task: dict, result: Any, clock: float) -> dict:
        outcomes = result.get("per_variant_outcomes") if isinstance(result, dict) else None
        progress = derive.task_progress(task.get("history"), clock if self.clock.replay else None, parse_ts, outcomes)
        for unit in progress.get("units") or []:
            if unit["status"] == "failed":
                unit["error"] = self._unit_error(task, str(unit.get("label") or ""))
        return progress

    def _unit_error(self, task: dict, label: str) -> str | None:
        """First exception in a failed variant's (or baseline round's) server log; cached once found. A variant's is
        below its grid slot, ``v<NN>_<its directory name>``."""
        key = (task["task_id"], label)
        if key in self.unit_errors or not label:
            return self.unit_errors.get(key)
        base = self.session_dir / _RUNS_DIR / str(task.get("kind")) / task["task_id"]
        slot, name = glob.escape(variant_dir_name(label)), glob.escape(label)
        logs = sorted({*base.glob(f"v*_{slot}/**/server.log"), *base.glob(f"{name}_round/**/server.log")})
        error = next((e for e in map(first_exception, logs) if e), None)
        if error or task.get("state") != _RUNNING_STATE:
            self.unit_errors[key] = error
        return error

    def _log_age(self, task: dict) -> float | None:
        base = self.session_dir / _RUNS_DIR
        candidates = [base / str(task.get("kind")) / task["task_id"], base / "specialist" / task["task_id"]]
        mtimes = [m for m in (newest_mtime(p) for p in candidates if p.exists()) if m is not None]
        return time.time() - max(mtimes) if mtimes else None

    def run_info(self) -> dict:
        summary = derive.state_summary(self.state)
        m = self.manifest
        return {
            "session_dir": str(self.session_dir),
            "session_id": summary.get("session_id") or self.session_dir.name,
            "model": summary.get("model_name") or m.get("model_name"),
            "manifest": {
                k: m.get(k)
                for k in (
                    "objective",
                    "workload",
                    "stack_fingerprint",
                    "dependencies",
                    "gfx_arch",
                    "host",
                    "max_minutes",
                    "reference_script",
                )
                if k in m
            },
            "pod_history": self.pod_history,
            "optimizer": self.optimizer_status(),
            "coordinator_db": {"path": str(self.db.path), "problem": self.db_problem},
        }

    def optimizer_status(self) -> dict:
        """Is an optimizer process alive for this session? Only answerable in live mode on the same host."""
        if self.clock.replay:
            return {"alive": None, "reason": "replay"}
        owner = read_json(self.session_dir / "runtime" / "optimizer.lock")
        last = self.pod_history[-1] if self.pod_history else {}
        owner = owner if isinstance(owner, dict) else last
        host, pid = owner.get("hostname"), owner.get("pid")
        if host and host != socket.gethostname():
            return {"alive": None, "reason": f"owner host {host} is not this host", "pid": pid, "host": host}
        alive = bool(pid) and os.path.exists(f"/proc/{pid}")
        return {"alive": alive, "pid": pid}

    def summary_at(self, clock: float) -> dict:
        """state.json summary; in replay, rolled back to *clock*.

        state.json holds each field's last value only. In replay, phase, budget used and best throughput are
        worked out for the clock; the baseline, the stop and the validated gain appear once the clock passes them;
        tick and crash count, which have no history, are left out.
        """
        summary = derive.state_summary(self.state)
        if not self.clock.replay:
            summary["current_best"] = self._best_at(summary.get("current_best") or {}, None)
            return summary
        return self._replay_summary(summary, clock)

    def _replay_summary(self, summary: dict, clock: float) -> dict:
        summary.update(_phases_at(summary, clock))
        summary["current_best"] = self._best_at({}, clock)
        summary["specialist_rounds"] = [
            r for r in summary.get("specialist_rounds") or [] if (parse_ts(r.get("completed_at")) or 0) <= clock
        ]
        start, used = parse_ts(summary.get("start_ts")), summary.get("elapsed_charged_sec")
        if start is not None and isinstance(used, (int, float)):
            summary["elapsed_charged_sec"] = max(0.0, min(float(used), clock - start))
        if not self._baseline_by(clock):
            summary["baseline_tput"] = None
        stop = parse_ts(summary.get("stop_ts"))
        if stop is None or stop > clock:
            summary["stop_ts"] = summary["stop_reason"] = summary["cumulative_gain_validated"] = None
        summary["tick"] = summary["crash_count"] = None
        return summary

    def _baseline_by(self, clock: float) -> bool:
        """Whether the baseline is measured by *clock*: its journal entry or its task's result is released."""
        stream = self.streams["journal"]
        if any(j.get("kind") == _BASELINE_KIND for j in stream.upto(stream.released(clock))):
            return True
        kinds = {t["task_id"]: t.get("kind") for t in self.tasks}
        return any(
            kinds.get(task_id) == _BASELINE_KIND and (r["t"] is None or r["t"] <= clock)
            for task_id, r in self.results.items()
        )

    def _best_at(self, current: dict, clock: float | None) -> dict:
        """Best tok/s from *current*, else from results and journal entries released by *clock* (None: all)."""
        kinds = {t["task_id"]: t.get("kind") or "task" for t in self.tasks}
        results = [
            (kinds.get(task_id, "task"), r["result"])
            for task_id, r in self.results.items()
            if clock is None or r["t"] is None or r["t"] <= clock
        ]
        stream = self.streams["journal"]
        series = stream.items if clock is None else stream.upto(stream.released(clock))
        return derive.best_throughput(current, results, series)

    def timeline_view(self, tasks: list[dict], clock: float) -> dict:
        bars = []
        for t in tasks:
            start, end = _task_span(t, clock)
            current = (t.get("progress") or {}).get("current") or {}
            bars.append(
                {
                    "task_id": t["task_id"],
                    "node": t["node"],
                    "label": t.get("label"),
                    "start": start,
                    "end": end,
                    "state": t.get("state"),
                    "health": t.get("health"),
                    "step": current.get("label") if t.get("state") == _RUNNING_STATE else None,
                }
            )
        history = self.state.get("phase_history") or []
        phases = _phase_bands(history, clock)
        steps = []
        for name, rec in sorted(self.timeline.items()):
            start = parse_ts(rec.get("start_time"))
            if start is None or start > clock:
                continue
            end = parse_ts(rec.get("end_time"))
            steps.append(
                {
                    "id": rec.get("id") or name,
                    "kind": rec.get("kind"),
                    "status": rec.get("status"),
                    "start": start,
                    "end": end if end is not None and end <= clock else None,
                }
            )
        out = {"bars": bars, "phases": phases, "steps": steps, "end": self._run_end(clock)}
        # The scrubber's chapters span the whole recording, so unlike the bands they aren't cut at the clock.
        if self.clock.replay and self.clock.end is not None:
            out["chapters"] = _phase_bands(history, self.clock.end)
        return out

    def snapshot(self, include_streams: bool = True) -> dict:
        """Everything the page needs, at the current clock; redacted. Streams are optional (SSE view ticks)."""
        with self.lock:
            clock = self.clock.now()
            tasks = self.tasks_at(clock)
            released = {name: s.released(clock) for name, s in self.streams.items()}
            summary = self.summary_at(clock)
            events = self.streams["event"].upto(released["event"])
            calls = self.streams["llm_call"].upto(released["llm_call"])
            artifacts = self._artifacts_at(clock)
            memory = self._latest_memory(released[_METRICS_STREAM])
            # A run that has ended shows how long each agent was quiet before it stopped, not until now.
            end = self._run_end(clock)
            agents = derive.agent_staleness(summary, clock if end is None else end, self.stall_after_sec)
            view = {
                "version": self.version,
                "clock": self.clock.info(),
                "run": self.run_info(),
                "summary": summary,
                "budget": derive.budget(summary),
                "tasks": tasks,
                "graph": graph.build_graph(tasks, events, calls, artifacts),
                "timeline": self.timeline_view(tasks, clock),
                "llm": derive.llm_summary(calls),
                "badges": self._db_badges() + derive.health_badges(summary, tasks, memory, agents),
                "agents": agents,
                "artifact_count": len(artifacts),
                "artifacts_cut_at": artifacts_mod.MAX_FILES if self.artifact_index.truncated else None,
                "released": released,
            }
            if include_streams:
                view["streams"] = {name: self.streams[name].upto(n) for name, n in released.items()}
            return redact(view)

    def _db_badges(self) -> list[dict]:
        """An error badge when the session's tasks cannot be read (its database is in a private ledger elsewhere)."""
        return [{"level": "error", "text": self.db_problem}] if self.db_problem else []

    def _run_end(self, clock: float) -> float | None:
        """Live, the run's last activity once it has gone quiet (its optimizer not known to be alive, and nothing new
        for ``_ACTIVE_SEC``); None while it is active, and in replay."""
        if self.clock.replay:
            return None
        last = self._last_activity()
        active = self.optimizer_status().get("alive") or last is None or clock - last < _ACTIVE_SEC
        return None if active else last

    def delta(self, cursors: dict[str, int]) -> dict:
        """Stream items released since *cursors*; ``reset`` when the clock moved back past one."""
        with self.lock:
            clock = self.clock.now()
            out: dict[str, Any] = {"clock": self.clock.info(), "version": self.version, "items": {}}
            for name, stream in self.streams.items():
                n = stream.released(clock)
                have = cursors.get(name, 0)
                if n < have:
                    return {"reset": True, "clock": out["clock"]}
                if n > have:
                    out["items"][name] = stream.between(have, n)
                cursors[name] = n
            return redact(out)

    def artifacts_at(self) -> list[dict]:
        with self.lock:
            return self._artifacts_at(self.clock.now())

    def _artifacts_at(self, clock: float) -> list[dict]:
        """In replay, a file appears once the clock passes its mtime; files written after the last recorded event
        (or copied without their mtimes) all appear when the replay reaches its end."""
        if not self.clock.replay or (self.clock.end is not None and clock >= self.clock.end):
            return list(self.artifact_index)
        return [a for a in self.artifact_index if a["mtime"] <= clock]

    def _latest_memory(self, released: int) -> dict:
        items = self.streams[_METRICS_STREAM].upto(released)
        return items[-1]["memory"] if items else {}

    def conversation(self, call_id: str = "", task_id: str = "", turn: str = "") -> dict | None:
        """The prompt/response record for one LLM call, redacted; scans conversations.jsonl.

        Orchestration and critic calls carry a ``call_id``. Specialist calls don't, so they match on
        ``task_id`` plus ``turn``; only their first turn is recorded, so later turns come from the specialist's
        ``process.log``.
        """
        if not call_id and not task_id:
            return None
        path = self.session_dir / _REPORTS_DIR / "trace" / "conversations.jsonl"
        needle = json.dumps(call_id or task_id)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if needle not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and _conversation_matches(rec, call_id, task_id, turn):
                        return redact(rec)
        except FileNotFoundError:
            pass
        return self._specialist_turn(task_id, turn) if task_id and turn.isdigit() else None

    def _specialist_turn(self, task_id: str, turn: str) -> dict | None:
        """Fallback for specialist turns without a recorded conversation: that turn of its ``process.log``."""
        rel = f"runs/specialist/{task_id}/process.log"
        path = resolve_inside(self.session_dir, rel)
        found = specialist_turn(path, int(turn)) if path is not None and "/" not in task_id else None
        return redact({"source": rel, "task_id": task_id, **found}) if found else None


def _phases_at(summary: dict, clock: float) -> dict:
    """The summary fields the clock rolls back, to update it with: its phase history as it stood at *clock*, and its
    phase when there is a phase history to tell it."""
    history = summary.get("phase_history") or []
    rows = [r for r in history if (_row_time(r) or 0) <= clock]
    out: dict = {"phase_history": rows}
    if rows:
        out["phase"] = rows[-1].get("to_phase") or summary.get("phase")
    elif history:
        out["phase"] = history[0].get("from_phase") or None
    return out


def _conversation_matches(rec: dict, call_id: str, task_id: str, turn: str) -> bool:
    if call_id:
        return rec.get("call_id") == call_id
    if rec.get("task_id") != task_id:
        return False
    return not turn or str(rec.get("turn")) == turn


def _read_jsonl(path: Path) -> list[dict]:
    return JsonlTail(path).read_new()


def _row_time(row: dict) -> float | None:
    return parse_ts(row.get("ts_unix")) or parse_ts(row.get("ts"))


def _task_span(task: dict, clock: float) -> tuple[float | None, float | None]:
    """(start, end) of a task bar: first ``running`` transition (else creation) to its terminal transition."""
    start = parse_ts(task.get("created_at"))
    end = None
    for tr in task.get("history") or []:
        ts = parse_ts(tr.get("ts"))
        if ts is None or ts > clock:
            continue
        if tr.get("to") == _RUNNING_STATE and (start is None or ts >= start):
            start = ts
        if tr.get("to") in ENDED_STATES:
            end = ts
    return start, end


def _phase_bands(history: list[dict], clock: float) -> list[dict]:
    """Background bands, one per phase transition up to *clock*."""
    rows = [(t, r) for r in history if (t := _row_time(r)) is not None and t <= clock]
    rows = [(t, r) for t, r in rows if r.get("from_phase") != r.get("to_phase")]
    bands = []
    for i, (t, r) in enumerate(rows):
        stop = rows[i + 1][0] if i + 1 < len(rows) else None
        bands.append({"phase": r.get("to_phase"), "start": t, "end": stop, "reason": r.get("reason")})
    return bands
