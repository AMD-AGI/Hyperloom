# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Polls a session directory and keeps what a client needs.

Two kinds of data come out of here:

* **Streams** are append-only and time-stamped (bus events, LLM calls, journal
  entries, critic verdicts). Each is kept sorted by time; a
  client holds a per-stream count and receives the items released since then.
* **Views** are recomputed at the clock on every request (run info, tasks with
  health, badges).

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
    ChangedFiles,
    CoordinatorDb,
    JsonlTail,
    first_exception,
    newest_mtime,
    parse_ts,
    read_json,
    resolve_inside,
    specialist_turn,
)
from hyperloom.inspector.redact import redact
from hyperloom.inspector.session_layout import STATE_FILE, coordinator_db, variant_dir_name

log = logging.getLogger(__name__)

STREAMS = ("event", "llm_call", "journal", "review")
# Folders of a session directory and the task state the watcher looks for.
_REPORTS_DIR = "reports"
_RUNS_DIR = "runs"
_RUNNING_STATE = "running"
_TS_KEYS = ("ts_unix", "ts", "recorded_unix", "completed_at", "started_at", "start_time", "created_at")
_INDEX_EVERY_SEC = 10.0


def item_time(rec: dict) -> float | None:
    for key in _TS_KEYS:
        t = parse_ts(rec.get(key))
        if t is not None:
            return t
    return None


class Clock:
    """Wall time."""

    def now(self) -> float:
        return time.time()

    def info(self) -> dict:
        return {
            "mode": "live",
            "now": self.now(),
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
    """How a :class:`Watcher` reads the session: it polls every *poll_sec* seconds, and a task whose log is older than
    *stall_after_sec* seconds is stalled."""

    poll_sec: float = 1.5
    stall_after_sec: float = 900.0


class Watcher:
    """Owns the collectors and answers snapshot/stream queries; thread-safe."""

    def __init__(
        self,
        session_dir: Path,
        options: WatchOptions | None = None,
    ) -> None:
        options = options or WatchOptions()
        self.session_dir = Path(session_dir)
        self.clock = Clock()
        self.poll_sec = options.poll_sec
        self.stall_after_sec = options.stall_after_sec
        self.lock = threading.RLock()
        self.version = 0
        self.streams = {name: Stream() for name in STREAMS}
        db_path, self.db_problem = coordinator_db(self.session_dir)
        self.db = CoordinatorDb(db_path)
        reports = self.session_dir / _REPORTS_DIR
        self._llm_tail = JsonlTail(reports / "trace" / "llm_calls.jsonl")
        self._reviews = ChangedFiles(self.session_dir / "critic-workdir", "*/review.json")
        self.state: dict = {}
        self.manifest: dict = {}
        self.pod_history: list[dict] = []
        self.tasks: list[dict] = []
        self.results: dict[str, Any] = {}
        self.unit_errors: dict[tuple[str, str], str | None] = {}
        self.journal_seen = 0
        self.artifact_index = artifacts_mod.ArtifactIndex()
        self._last_index = 0.0
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
            now = time.monotonic()
            if not self._last_index or (changed and now - self._last_index > _INDEX_EVERY_SEC):
                self.artifact_index = artifacts_mod.index(self.session_dir, self.task_nodes())
                self._last_index = now
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

    # ----- views ---------------------------------------------------------------------------------------------

    def task_nodes(self) -> dict[str, str]:
        return {t["task_id"]: graph.task_node(t) for t in self.tasks}

    def tasks_at(self, clock: float) -> list[dict]:
        """Tasks as of *clock*, each with ``node``, ``label``, ``health`` and ``reason``."""
        reasons = {r.get("task_id"): r.get("reason") for r in self.state.get("specialist_rounds") or []}
        out = []
        for task in self.tasks:
            t = dict(task)
            res = self.results.get(t["task_id"])
            result = res["result"] if res and (res["t"] is None or res["t"] <= clock) else None
            log_age = self._log_age(t) if t.get("state") == _RUNNING_STATE else None
            reason = reasons.get(t["task_id"]) if t.get("state") == "succeeded" else None
            health = derive.task_health(t, result, reason, log_age, self.stall_after_sec)
            node = graph.task_node(t)
            label = graph.specialist_domain(t) if t.get("kind") == "specialist" else t.get("kind")
            out.append({**t, **health, "node": node, "label": label, "progress": self._progress(t, result)})
        return out

    def _progress(self, task: dict, result: Any) -> dict:
        outcomes = result.get("per_variant_outcomes") if isinstance(result, dict) else None
        progress = derive.task_progress(task.get("history"), parse_ts, outcomes)
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
        """Is an optimizer process alive for this session? Only answerable on the same host."""
        owner = read_json(self.session_dir / "runtime" / "optimizer.lock")
        last = self.pod_history[-1] if self.pod_history else {}
        owner = owner if isinstance(owner, dict) else last
        host, pid = owner.get("hostname"), owner.get("pid")
        if host and host != socket.gethostname():
            return {"alive": None, "reason": f"owner host {host} is not this host", "pid": pid}
        alive = bool(pid) and os.path.exists(f"/proc/{pid}")
        return {"alive": alive, "pid": pid}

    def summary_at(self) -> dict:
        """state.json summary."""
        summary = derive.state_summary(self.state)
        summary["current_best"] = self._best_at(summary.get("current_best") or {})
        return summary

    def _best_at(self, current: dict) -> dict:
        """Best tok/s from *current*, else from results and journal entries."""
        kinds = {t["task_id"]: t.get("kind") or "task" for t in self.tasks}
        results = [(kinds.get(task_id, "task"), r["result"]) for task_id, r in self.results.items()]
        series = self.streams["journal"].items
        return derive.best_throughput(current, results, series)

    def snapshot(self, include_streams: bool = True) -> dict:
        """Everything a client needs, at the current clock; redacted. Streams are optional."""
        with self.lock:
            clock = self.clock.now()
            tasks = self.tasks_at(clock)
            released = {name: s.released(clock) for name, s in self.streams.items()}
            summary = self.summary_at()
            calls = self.streams["llm_call"].upto(released["llm_call"])
            artifacts = self._artifacts_at()
            view = {
                "version": self.version,
                "clock": self.clock.info(),
                "run": self.run_info(),
                "summary": summary,
                "budget": derive.budget(summary),
                "tasks": tasks,
                "llm": derive.llm_summary(calls),
                "badges": self._db_badges(),
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

    def delta(self, cursors: dict[str, int]) -> dict:
        """Stream items released since *cursors*."""
        with self.lock:
            clock = self.clock.now()
            out: dict[str, Any] = {"clock": self.clock.info(), "version": self.version, "items": {}}
            for name, stream in self.streams.items():
                n = stream.released(clock)
                have = cursors.get(name, 0)
                if n > have:
                    out["items"][name] = stream.between(have, n)
                cursors[name] = n
            return redact(out)

    def artifacts_at(self) -> list[dict]:
        with self.lock:
            return self._artifacts_at()

    def _artifacts_at(self) -> list[dict]:
        return list(self.artifact_index)

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


def _conversation_matches(rec: dict, call_id: str, task_id: str, turn: str) -> bool:
    if call_id:
        return rec.get("call_id") == call_id
    if rec.get("task_id") != task_id:
        return False
    return not turn or str(rec.get("turn")) == turn


def _read_jsonl(path: Path) -> list[dict]:
    return JsonlTail(path).read_new()
