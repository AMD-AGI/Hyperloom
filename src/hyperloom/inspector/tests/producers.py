# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session artifacts written by Hyperloom's own writers, for the Inspector's contract tests.

Each helper calls the producer that writes the artifact in a real ``hyperloom optimize`` session (the message bus,
the task registry, the manifest, ``SharedState``) and never
spells the artifact's format itself. A contract test reads what a helper wrote with the Inspector's reader, so a
change to a producer's format fails that test instead of passing a hand-written fixture.

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

#: The session id the helpers stamp.
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


def write_manifest(session_dir: Path, model: str) -> dict:
    """``manifest.json`` through ``manifest.write_manifest``, for a run of *model*."""
    write = Upstream("hyperloom.inference_optimizer.session.manifest").write_manifest
    args = argparse.Namespace(model=model, max_hours=2.0, reference_script="recipe.sh")
    return write(session_dir, args=args, session_id=SESSION_ID)


def new_state(**fields: Any) -> Any:
    """A ``SharedState`` as the Coordinator starts one (about 8 s to import cold)."""
    return Upstream("hyperloom.orchestrator.state.shared_state").SharedState(session_id=SESSION_ID, **fields)
