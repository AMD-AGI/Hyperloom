# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session artifacts written by Hyperloom's own writers, for the Inspector's contract tests.

Each helper calls the producer that writes the artifact in a real ``hyperloom optimize`` session (the message bus,
the task registry, the LLM-call ledger, the journal, the session lock, the manifest, ``SharedState``, the baseline
timeline recorder) and never spells the artifact's format itself. A contract test reads what a helper wrote with the
Inspector's reader, so a change to a producer's format fails that test instead of passing a hand-written fixture.

Every helper reaches its producer through an :class:`Upstream` module, imported on first use: some take seconds to
import cold, and the rest of the Inspector suite should not pay for them at collection. A producer name that is gone
fails the test with its own message ("re-point this fixture"), unlike a broken contract, whose message
:func:`contract_message` writes.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

#: The session id the helpers stamp on ledger and conversation rows.
SESSION_ID = "contract-session"

#: The model the helpers' sessions are made for.
MODEL = "/models/Qwen3-8B"


class Upstream:
    """An upstream module the helpers below take producers from, imported when a name is first read from it.

    A name that cannot be read (the module or the name is gone) fails the calling test as a fixture to re-point, not as
    a broken contract: the producer moved or was renamed, and what the Inspector reads may well be unchanged.
    """

    def __init__(self, module: str) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(importlib.import_module(self.module), name)
        except (ImportError, AttributeError) as exc:
            pytest.fail(
                f"upstream producer {self.module}.{name} is gone ({exc}): re-point this fixture in "
                "src/hyperloom/inspector/tests/producers.py; no Inspector contract is known to be broken",
                pytrace=False,
            )


_PATHS = Upstream("hyperloom.inference_optimizer.session.paths")
_CONNECTION = Upstream("hyperloom.orchestrator.bus.storage.connection")
_MESSAGE_BUS = Upstream("hyperloom.orchestrator.bus.message_bus")
_TASK_REGISTRY = Upstream("hyperloom.orchestrator.state.task_registry")
_LLM_TRACE = Upstream("hyperloom.inference_optimizer.trace.llm_trace")


def contract_message(producer: str, inspector_file: str, detail: str) -> str:
    """An assertion message that names the producer that changed and the Inspector file to update."""
    return (
        f"{detail}\nProducer: {producer}. Inspector reader to update: src/hyperloom/inspector/{inspector_file} "
        "(then this contract test)."
    )


def watch(session_dir: Path, *, replay: bool = False) -> Any:
    """The Inspector's watcher over *session_dir*, polled once."""
    from hyperloom.inspector.watcher import Watcher, WatchOptions

    watcher = Watcher(session_dir, WatchOptions(replay=replay))
    watcher.poll()
    return watcher


class RecordingDict(dict):
    """A dict that records every key read from it, so a test can list the keys an Inspector reader asks for."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def get(self, key: Any, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.read.add(key)
        return super().__contains__(key)


def new_session(root: Path) -> Path:
    """A session for :data:`MODEL` made as ``hyperloom optimize`` makes one (``paths.make_session_dir``), in a workspace
    at *root*; the environment it sets is put back."""
    with mock.patch.dict(os.environ, {_PATHS.ENV_USER_DATA_PATH: str(root)}):
        return _PATHS.make_session_dir(MODEL)


def create_coordinator_db(db_path: Path) -> None:
    """An empty coordinator database at *db_path* with the schema ``schema.ensure_schema`` makes (the Coordinator's
    ``SqliteConnection`` runs it; that connection is not used here, as it leaves the database in WAL mode)."""
    ensure_schema = Upstream("hyperloom.orchestrator.bus.storage.schema").ensure_schema
    with closing(sqlite3.connect(db_path)) as conn:
        ensure_schema(conn)
        conn.commit()


# ----- coordinator database: the bus and the task registry -----------------------------------------------------


@dataclass
class Coordinator:
    """The session's coordinator database opened as the Coordinator opens it, with its bus and task registry."""

    path: Path
    db: Any
    bus: Any
    tasks: Any


@asynccontextmanager
async def coordinator_db(session_dir: Path, state: Any = None):
    """Open the coordinator database as ``Coordinator._open_persistence`` does: where ``state_file.database_target``
    puts it for *state* (a fresh :func:`new_state` when none is given: the session's shared database), with
    Hyperloom's connection (which runs ``ensure_schema``)."""
    database_target = Upstream("hyperloom.orchestrator.state.state_file").database_target
    path, file_mode = database_target(session_dir, new_state() if state is None else state)
    db = _CONNECTION.SqliteConnection(path, file_mode=file_mode)
    try:
        yield Coordinator(path=path, db=db, bus=_MESSAGE_BUS.MessageBus(db), tasks=_TASK_REGISTRY.TaskRegistry(db))
    finally:
        db.close()


@dataclass
class TaskRun:
    """One delegated task as the dispatcher runs it: created, moved through its states, its progress noted."""

    kind: str
    params: dict = field(default_factory=dict)
    states: tuple[str, ...] = ("running", "succeeded")
    progress: list[dict] = field(default_factory=list)
    task_id: str | None = None


async def run_task(coord: Coordinator, run: TaskRun) -> Any:
    """Create the task with ``TaskRegistry.create_or_return_existing``, record its progress notes with
    ``record_progress`` once running, and move it through ``transition``; returns the final ``Task``."""
    key = f"contract-{run.kind}-{run.task_id or len(run.params)}"
    task, _ = await coord.tasks.create_or_return_existing(
        _TASK_REGISTRY.TaskSpec(kind=run.kind, params=run.params, idempotency_key=key, task_id=run.task_id)
    )
    for i, state in enumerate(run.states):
        task = await coord.tasks.transition(task.task_id, state)
        if i == 0:
            for note in run.progress:
                await coord.tasks.record_progress(task.task_id, note=note)
    return await coord.tasks.get(task.task_id)


async def announce_result(coord: Coordinator, task: Any, result: dict, error: str | None = None) -> None:
    """Put the ``delegated_result`` announcement on the bus with ``MessageBus.append_and_seq``.

    The payload is built as ``orchestrator/loop/dispatcher.py`` builds it after a task settles; that dict is a literal
    there, with no function to call.
    """
    announcement = {"task_id": task.task_id, "kind": task.kind, "state": task.state, "result": result, "error": error}
    await coord.bus.append_and_seq(_MESSAGE_BUS.Message.new("coordinator", "*", "delegated_result", announcement))


def variant_progress_notes(names: list[str], statuses: list[str], throughputs: list[float | None]) -> list[dict]:
    """The grid runner's progress notes for variants that landed (``_grid_runner._variant_progress_note``)."""
    grid_base = Upstream("hyperloom.orchestrator.actions.executors.grid._grid_base")
    progress_note = Upstream("hyperloom.orchestrator.actions.executors.grid._grid_runner")._variant_progress_note
    grid = [grid_base.GridVariant(name=n, extra_server_args="", extra_envs={}) for n in names]
    landed = [
        grid_base.VariantResult(name=n, extra_server_args="", extra_envs={}, status=s, output_throughput=t)
        for n, s, t in zip(names, statuses, throughputs)
    ]
    return [progress_note(grid, landed, i) for i in range(len(names))]


# ----- ledgers under reports/trace -------------------------------------------------------------------------------


def append_ledger_row(session_dir: Path, **fields: Any) -> None:
    """One ``reports/trace/llm_calls.jsonl`` row through ``append_llm_call`` (closed schema validated)."""
    record = _LLM_TRACE.LLMCallRecord(session_id=SESSION_ID, **fields)
    _LLM_TRACE.append_llm_call(session_dir=session_dir, record=record)


def append_conversation_row(session_dir: Path, **fields: Any) -> None:
    """One ``reports/trace/conversations.jsonl`` row through ``append_conversation`` (closed schema validated)."""
    trace = Upstream("hyperloom.inference_optimizer.trace.conversation_trace")
    trace.append_conversation(session_dir=session_dir, record=trace.ConversationRecord(session_id=SESSION_ID, **fields))


# ----- session files ---------------------------------------------------------------------------------------------


def write_journal(session_dir: Path, entries: list[dict]) -> Any:
    """``reports/optimization_journal.json`` through ``Journal.load_or_create`` and ``append_entry``."""
    module = Upstream("hyperloom.inference_optimizer.session.optimization_journal")
    journal = module.Journal.load_or_create(
        session_dir, module.Journal(session_id=SESSION_ID, model="m", hardware="MI355X")
    )
    for entry in entries:
        journal.append_entry(module.JournalEntry(**entry))
    return journal


def acquire_lock(session_dir: Path) -> Any:
    """``runtime/optimizer.lock`` and ``runtime/pod_history.jsonl`` through ``SessionLock.acquire``; release it."""
    return Upstream("hyperloom.inference_optimizer.session.lock").SessionLock(session_dir).acquire()


def write_manifest(session_dir: Path, model: str) -> dict:
    """``manifest.json`` through ``manifest.write_manifest``, for a run of *model*."""
    write = Upstream("hyperloom.inference_optimizer.session.manifest").write_manifest
    args = argparse.Namespace(model=model, max_hours=2.0, reference_script="recipe.sh")
    return write(session_dir, args=args, session_id=SESSION_ID)


def new_state(**fields: Any) -> Any:
    """A ``SharedState`` as the Coordinator starts one (about 8 s to import cold)."""
    return Upstream("hyperloom.orchestrator.state.shared_state").SharedState(session_id=SESSION_ID, **fields)


def record_phase(state: Any, to_phase: str, reason: str, ts_unix: float) -> dict:
    """One ``phase_history`` row through ``machine_state.record_phase_transition``, stamped at *ts_unix*."""
    from datetime import datetime, timezone

    machine = Upstream("hyperloom.orchestrator.phases.machine_state")
    ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat(timespec="seconds")
    clock = machine.PhaseClock(ts=ts, ts_unix=ts_unix)
    return machine.record_phase_transition(state, to_phase=to_phase, reason=reason, at=clock)


def record_baseline(session_dir: Path, task_id: str, measured: dict) -> None:
    """The baseline's ``reports/sbd_v6/timeline`` event, through the recorder a dispatched baseline gets
    (``make_baseline_recorder`` and ``finish``), in a session bound as startup binds it."""
    event = Upstream("hyperloom.inference_optimizer.breakdown.recorder.baseline_event")
    sink_type = Upstream("hyperloom.inference_optimizer.breakdown.recorder.event_sink").EventSink
    session_scope = Upstream("hyperloom.inference_optimizer.session.session_binding").session_scope
    with session_scope(session_dir):
        request = event.BaselineRequest(task_id=task_id, task_kind="baseline", framework="vllm", params={})
        sink = sink_type(event.baseline_event_id("prelude", 0), producer=event.PRODUCER)
        event.make_baseline_recorder(sink, request).finish(measured)


# ----- specialists ------------------------------------------------------------------------------------------------


def specialist_backend_error(**ending: Any) -> str:
    """The failure reason the specialist runner records for a subprocess that ended as *ending*
    (``SpecialistSubprocessResult`` fields), through ``runner._subprocess_backend_error``."""
    result = Upstream("hyperloom.orchestrator.specialists.subprocess_types").SpecialistSubprocessResult(**ending)
    return Upstream("hyperloom.orchestrator.specialists.runner")._subprocess_backend_error(result)


def producer_says_failed(runner_status: str, reason: str) -> bool:
    """Whether ``failure_class.classify_specialist_failure`` calls the specialist's ending a failure."""
    failure_class = Upstream("hyperloom.orchestrator.specialists.failure_class")
    ftype, _retry = failure_class.classify_specialist_failure(runner_status, reason)
    kinds = failure_class.SpecialistFailureType
    return ftype in (kinds.TIMEOUT, kinds.STALE_HEARTBEAT, kinds.CRASH)
