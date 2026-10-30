# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The query tools the campaign's host front end calls over MCP, for the legacy ``optimize`` flow (design 4.12).

Every tool is registered with scope ``host`` only. Handlers run on the MCP server's request threads, so they read
what the run has already written to disk -- ``state.json`` (written atomically), the reports under the session
directory, the events segments and the coordinator database opened read-only -- and never touch the Coordinator's
objects. A source that cannot be read is reported as an error or as an explicit ``null`` with a warning, never as an
empty answer.

``request_stop`` records the stop in ``<session>/hosted/control/stop.json`` and then wakes the process with SIGTERM.
The file is the request: ``Coordinator.stop_requested`` reads it on every tick, and the hosted entry reads it before
every leg and ends a run that has it with a final ``exit.json`` instead of starting ``optimize``. The signal only
hastens the stop. While ``SignalDrain`` is armed it records the signal too; outside the tick loop the entry ignores a
SIGTERM that arrives while the file exists, so the stop then takes effect at the next tick or the next start.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.common.io import atomic_write_json
from hyperloom.hosted.events import segment_name, segment_numbers
from hyperloom.hosted.mcp_server import HOST_KIND, CallContext, McpServer, ToolError
from hyperloom.inference_optimizer.session.optimization_journal import read_journal_entries
from hyperloom.inference_optimizer.session.paths import db_path_for, stop_request_path
from hyperloom.inference_optimizer.session.session_paths import llm_calls_path
from hyperloom.orchestrator.state.shared_state import SharedState

STATE_FILE = "state.json"
MEASUREMENT_SANDBOXES = Path("hosted") / "measurement_sandboxes.jsonl"
EVENT_BEST_UPDATED = "hyperloom.best_updated"
IN_FLIGHT_STATES = ("queued", "running")
DEFAULT_CANDIDATE_LIMIT = 20
MAX_CANDIDATE_LIMIT = 200
MAX_REASON_CHARS = 500
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
)
_PATCH_FIELDS = (
    "stack_entry_id",
    "action",
    "variant_name",
    "operation_kind",
    "candidate_extra_server_args",
    "candidate_extra_envs",
    "tput",
    "ts",
)
_OBJECT = "object"
_STRING = "string"
_NO_ARGS = {"type": _OBJECT, "properties": {}, "additionalProperties": False}

Stopper = Callable[[], None]


def _sigterm_self() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def _read_state(session_dir: Path) -> dict[str, Any]:
    try:
        state = json.loads((session_dir / STATE_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ToolError("the run has not written its state yet") from None
    except (OSError, ValueError) as exc:
        raise ToolError(f"state.json cannot be read: {type(exc).__name__}") from None
    if not isinstance(state, dict):
        raise ToolError("state.json is not a JSON object")
    return state


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _lift_pct(tput: Any, baseline: Any) -> float | None:
    """Percent over the baseline, ``None`` without a positive baseline and a measured value."""
    value, base = _number(tput), _number(baseline)
    if value is None or base is None or base <= 0 or value <= 0:
        return None
    return round((value / base - 1.0) * 100.0, 3)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any] | None]:
    """The objects of a JSON Lines file; ``None`` stands for a line that is not one. A missing file has no rows."""
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError:
        return
    with handle:
        for line in handle:
            if not line.endswith("\n") or not line.strip():
                continue  # an unfinished write is not a row
            try:
                row = json.loads(line)
            except ValueError:
                yield None
                continue
            yield row if isinstance(row, dict) else None


def _events_newest_first(events_dir: Path | None) -> Iterator[dict[str, Any]]:
    if events_dir is None or not events_dir.is_dir():
        return
    for number in reversed(segment_numbers(events_dir)):
        rows = [row for row in _read_jsonl(events_dir / segment_name(number)) if row is not None]
        yield from reversed(rows)


def _best_updates(events_dir: Path | None) -> Iterator[dict[str, Any]]:
    for event in _events_newest_first(events_dir):
        data = event.get("data")
        if event.get("type") == EVENT_BEST_UPDATED and isinstance(data, dict):
            yield {**data, "time": event.get("time")}


def _sandbox_record(session_dir: Path, request_id: str | None) -> dict[str, Any] | None:
    if not request_id:
        return None
    found = None
    for row in _read_jsonl(session_dir / MEASUREMENT_SANDBOXES):
        if row is not None and row.get("request_id") == request_id:
            found = row
    return found


def _in_flight(session_dir: Path) -> list[dict[str, Any]]:
    path = db_path_for(session_dir)
    if not path.is_file():
        return []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as db:
        rows = db.execute(
            "SELECT task_id, kind, state, created_at, updated_at FROM tasks WHERE state IN (?, ?) ORDER BY created_at",
            IN_FLIGHT_STATES,
        ).fetchall()
    return [
        {"task_id": task_id, "action": kind, "state": state, "created_at": created, "updated_at": updated}
        for task_id, kind, state, created, updated in rows
    ]


def _task_timings(session_dir: Path, task_ids: list[str]) -> dict[str, tuple[str, float | None]]:
    """``{task id: (kind, seconds from creation to last update)}`` for the settled tasks among ``task_ids``."""
    path = db_path_for(session_dir)
    if not task_ids or not path.is_file():
        return {}
    marks = ",".join("?" * len(task_ids))
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as db:
        rows = db.execute(
            f"SELECT task_id, kind, created_at, updated_at FROM tasks WHERE task_id IN ({marks})", task_ids
        ).fetchall()
    out: dict[str, tuple[str, float | None]] = {}
    for task_id, kind, created, updated in rows:
        begun, ended = to_unix(created), to_unix(updated)
        out[task_id] = (kind, round(ended - begun, 3) if begun is not None and ended is not None else None)
    return out


def _best_summary(state: dict[str, Any]) -> dict[str, Any] | None:
    best = state.get("current_best")
    if not isinstance(best, dict) or not best:
        return None
    return {
        "metric": "output_throughput",
        "value": best.get("tput"),
        "lift_pct": _lift_pct(best.get("tput"), state.get("baseline_tput")),
        "variant_name": best.get("variant_name"),
        "action": best.get("action"),
    }


class HostTools:
    """The handlers of the host tools over one run's directories."""

    def __init__(
        self,
        session_dir: Path,
        events_dir: Path | None = None,
        *,
        stop: Stopper = _sigterm_self,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session = Path(session_dir)
        self._events = None if events_dir is None else Path(events_dir)
        self._stop = stop
        self._clock = clock

    def get_status(self, _args: dict[str, Any], _ctx: CallContext) -> dict[str, Any]:
        state = _read_state(self._session)
        warnings: list[str] = []
        try:
            in_flight: list[dict[str, Any]] | None = _in_flight(self._session)
        except sqlite3.Error as exc:
            in_flight = None
            warnings.append(f"in-flight tasks cannot be read: {type(exc).__name__}")
        total_min = _number(state.get("max_minutes")) or 0.0
        return {
            "session_id": state.get("session_id"),
            "phase": state.get("phase") or None,
            "macro_round": state.get("macro_cycle"),
            "tick": state.get("tick"),
            "elapsed_sec": round(SharedState.elapsed_sec_of(state, now_unix=self._clock()), 1),
            "total_sec": total_min * 60.0 if total_min > 0 else None,
            "stop_reason": state.get("stop_reason") or None,
            "stopped": bool(state.get("stop_ts")),
            "best": _best_summary(state),
            "in_flight": in_flight,
            "warnings": warnings,
        }

    def get_best(self, _args: dict[str, Any], _ctx: CallContext) -> dict[str, Any]:
        state = _read_state(self._session)
        summary = _best_summary(state)
        if summary is None:
            return {"best": None, "baseline_tput": state.get("baseline_tput")}
        best = state["current_best"]
        gains = state.get("gain_per_stack_entry") or []
        patches = []
        for index, entry in enumerate(state.get("optimization_stack") or []):
            if isinstance(entry, dict):
                patch = {key: entry[key] for key in _PATCH_FIELDS if key in entry}
                patch["gain_pct"] = gains[index] if index < len(gains) else None
                patches.append(patch)
        update = next(_best_updates(self._events), None)
        request_id = (update or {}).get("request_id")
        measurement = state.get("current_best_measurement")
        return {
            **summary,
            "baseline_tput": state.get("baseline_tput"),
            "variant": {
                "extra_server_args": best.get("extra_server_args"),
                "extra_envs": best.get("extra_envs") or {},
            },
            "patches": patches,
            "measurement": {
                "request_id": request_id,
                "sandbox": _sandbox_record(self._session, request_id),
                "launch_identity": (measurement or {}).get("declared_launch_identity"),
                "workspace": (measurement or {}).get("benchmark_workspace"),
            },
        }

    def list_candidates(self, args: dict[str, Any], _ctx: CallContext) -> dict[str, Any]:
        limit = int(args.get("limit", DEFAULT_CANDIDATE_LIMIT))
        phase = str(args.get("phase") or "").strip().upper()
        entries = [e for e in read_journal_entries(self._session) if not phase or e.phase.upper() == phase]
        recent = entries[::-1][:limit]
        timings = _task_timings(self._session, [e.task_id for e in recent if e.task_id])
        rows = []
        for entry in recent:
            kind, duration = timings.get(entry.task_id, (None, None))
            rows.append(
                {
                    "task_id": entry.task_id or None,
                    "action": kind or entry.kind,
                    "decision": entry.outcome,
                    "lift_pct": entry.gain_pct,
                    "duration_sec": duration,
                    "phase": entry.phase,
                    "change": entry.change,
                    "settled_at": entry.ts or None,
                }
            )
        return {"candidates": rows, "total": len(entries)}

    def get_measurement(self, args: dict[str, Any], _ctx: CallContext) -> dict[str, Any]:
        request_id = args["request_id"]
        updates = list(_best_updates(self._events))
        update = next((u for u in updates if u.get("request_id") == request_id), None)
        sandbox = _sandbox_record(self._session, request_id)
        if update is None and sandbox is None:
            raise ToolError(f"no measurement is recorded for request {request_id!r}")
        out: dict[str, Any] = {"request_id": request_id, "best_update": update, "sandbox": sandbox}
        if update is not None and update is updates[0]:
            measurement = _read_state(self._session).get("current_best_measurement") or {}
            out["current_best_measurement"] = {
                key: measurement.get(key)
                for key in (
                    "tput",
                    "benchmark_workspace",
                    "server_log_path",
                    "identity_verification_status",
                )
            }
            out["current_best_measurement"]["launch_identity"] = measurement.get("declared_launch_identity")
        return out

    def get_costs(self, _args: dict[str, Any], _ctx: CallContext) -> dict[str, Any]:
        totals = dict.fromkeys(TOKEN_FIELDS, 0)
        calls = failed = skipped = 0
        for row in _read_jsonl(llm_calls_path(self._session)):
            if row is None:
                skipped += 1
            elif row.get("status") == "error":
                failed += 1
            else:
                calls += 1
                for name in TOKEN_FIELDS:
                    value = row.get(name)
                    totals[name] += value if isinstance(value, int) and not isinstance(value, bool) else 0
        return {
            "tokens": totals,
            "llm_calls": calls,
            "llm_calls_failed": failed,
            "skipped_lines": skipped,
            # The legacy flow measures on the run's own GPUs and starts no sub-runs; it keeps neither figure.
            "gpu_seconds": None,
            "subrun_count": None,
        }

    def request_stop(self, args: dict[str, Any], _ctx: CallContext) -> dict[str, Any]:
        reason = args["reason"].strip()[:MAX_REASON_CHARS]
        if not reason:
            raise ToolError("a stop reason is required")
        path = stop_request_path(self._session)
        try:
            atomic_write_json(
                path, {"reason": reason, "requested_at": datetime.now(timezone.utc).isoformat(), "by": HOST_KIND}
            )
        except OSError as exc:
            raise ToolError(f"the stop request could not be written: {type(exc).__name__}") from None
        self._stop()
        return {"accepted": True}


_LIST_SCHEMA = {
    "type": _OBJECT,
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_CANDIDATE_LIMIT},
        "phase": {"type": _STRING},
    },
    "additionalProperties": False,
}
_MEASUREMENT_SCHEMA = {
    "type": _OBJECT,
    "properties": {"request_id": {"type": _STRING, "minLength": 1}},
    "required": ["request_id"],
    "additionalProperties": False,
}
_STOP_SCHEMA = {
    "type": _OBJECT,
    "properties": {"reason": {"type": _STRING, "minLength": 1}},
    "required": ["reason"],
    "additionalProperties": False,
}


def _tool_table(
    tools: HostTools,
) -> list[tuple[str, str, dict[str, Any], Callable[[dict[str, Any], CallContext], Any]]]:
    return [
        (
            "get_status",
            "Phase, macro round, time used and total, stop reason, best, tasks in flight.",
            _NO_ARGS,
            tools.get_status,
        ),
        (
            "get_best",
            "Current best: metric, lift over baseline, variant, patches, measurement.",
            _NO_ARGS,
            tools.get_best,
        ),
        ("list_candidates", "Recently settled candidates, newest first.", _LIST_SCHEMA, tools.list_candidates),
        (
            "get_measurement",
            "What is recorded about one measurement request.",
            _MEASUREMENT_SCHEMA,
            tools.get_measurement,
        ),
        (
            "get_costs",
            "Token totals from the LLM call ledger; GPU seconds and sub-runs when recorded.",
            _NO_ARGS,
            tools.get_costs,
        ),
        (
            "request_stop",
            "Ask the run to stop through its normal operator-stop path.",
            _STOP_SCHEMA,
            tools.request_stop,
        ),
    ]


def register_host_tools(
    server: McpServer,
    session_dir: Path,
    events_dir: Path | None = None,
    *,
    stop: Stopper = _sigterm_self,
    clock: Callable[[], float] = time.time,
) -> HostTools:
    """Offer the host tools on ``server`` for the run whose session directory is ``session_dir``."""
    tools = HostTools(session_dir, events_dir, stop=stop, clock=clock)
    for name, description, schema, handler in _tool_table(tools):
        server.register(name, schema=schema, handler=handler, scopes={HOST_KIND}, description=description)
    return tools


__all__ = ["HostTools", "register_host_tools"]
