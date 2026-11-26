# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 1 (session reader): the session files the Inspector finds and reads are
the ones Hyperloom's own writers produce.

Each test writes a session artifact with the real producer (see :mod:`.producers`) or compares the Inspector's
names with the producer's constants, then reads it the way the Inspector does. A failure means a producer changed
what it writes: update the Inspector reader the failure message names, then this test. A test marked
``xfail(strict=True)`` pins a known break; it starts failing ("XPASS(strict)") once the Inspector is fixed, and then
the marker comes off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract


async def test_coordinator_db_written_by_bus_and_registry_is_read(tmp_path: Path) -> None:
    from dataclasses import fields

    from hyperloom.inspector.collectors import CoordinatorDb
    from hyperloom.orchestrator.bus.message_bus import Message
    from hyperloom.orchestrator.state.task_registry import Task

    session = producers.new_session(tmp_path)
    async with producers.coordinator_db(session) as coord:
        task = await producers.run_task(coord, producers.TaskRun(kind="baseline", params={"domain": "d"}))
        await producers.announce_result(coord, task, {"status": "succeeded", "output_throughput": 10.0})
        db = CoordinatorDb(session / "storage" / "coordinator.db")
        events, tasks = db.new_events(), db.tasks()
    where = "collectors.py (CoordinatorDb)"
    assert [t["task_id"] for t in tasks] == [task.task_id], contract_message("TaskRegistry", where, f"tasks: {tasks}")
    assert set(tasks[0]) <= {f.name for f in fields(Task)}, contract_message("Task / tasks DDL", where, tasks[0])
    assert tasks[0]["params"] == {"domain": "d"} and tasks[0]["history"], contract_message("Task", where, tasks[0])
    assert set(events[0]) == {f.name for f in fields(Message)}, contract_message("Message / events DDL", where, events)
    payload = events[0]["payload"]
    assert payload["task_id"] == task.task_id and payload["result"]["output_throughput"] == 10.0, contract_message(
        "MessageBus.append_and_seq", where, f"payload not decoded: {payload!r}"
    )


@pytest.mark.parametrize("ledger_dir", ["named", "tmpdir"])
@pytest.mark.parametrize("recorded", ["manifest", "state", "none"])
def test_the_inspector_s_private_db_path_is_hyperloom_s(
    tmp_path: Path, monkeypatch, ledger_dir: str, recorded: str
) -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    if ledger_dir == "named":
        monkeypatch.setenv(paths.ENV_LEDGER_DIR, str(tmp_path / "ledgers"))
    else:
        monkeypatch.delenv(paths.ENV_LEDGER_DIR, raising=False)
    session = producers.new_session(tmp_path)
    if recorded == "manifest":
        producers.write_manifest(session, "/models/Qwen3-8B")
    elif recorded == "state":
        producers.new_state().save(session)
    pairs = {
        "paths.private_db_path_for": (session_layout.private_db_path(session), paths.private_db_path_for(session)),
        "paths.ledger_mark_path": (session / session_layout.LEDGER_MARK, paths.ledger_mark_path(session)),
        "paths.ENV_LEDGER_DIR": (session_layout.ENV_LEDGER_DIR, paths.ENV_LEDGER_DIR),
    }
    for producer, (inspector, hyperloom) in pairs.items():
        assert inspector == hyperloom, contract_message(producer, "session_layout.py", f"{inspector} != {hyperloom}")


def test_a_relative_ledger_dir_is_refused_as_hyperloom_refuses_it(tmp_path: Path, monkeypatch) -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    monkeypatch.setenv(paths.ENV_LEDGER_DIR, "ledgers")
    with pytest.raises(ValueError):
        paths.private_db_path_for(tmp_path)
    with pytest.raises(ValueError):
        session_layout.private_db_path(tmp_path)


def test_the_inspector_s_private_subdirs_are_hyperloom_s() -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    assert session_layout.PRIVATE_SESSION_SUBDIRS == paths.PRIVATE_SESSION_SUBDIRS, contract_message(
        "paths.PRIVATE_SESSION_SUBDIRS", "session_layout.py", session_layout.PRIVATE_SESSION_SUBDIRS
    )


def test_the_inspector_s_session_files_are_hyperloom_s() -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    assert session_layout.SESSION_FILES == paths.SESSION_FILES, contract_message(
        "paths.SESSION_FILES", "session_layout.py", session_layout.SESSION_FILES
    )


_VARIANT_NAMES = ["mnbt_8192", "max-num-seqs=64 tp=2", "a/b", "ünï.cödé", "x" * 80]


@pytest.mark.parametrize("name", _VARIANT_NAMES)
def test_variant_dir_names_are_the_grid_runner_s(name: str) -> None:
    from hyperloom.orchestrator.actions.executors.grid._grid_base import _safe

    from hyperloom.inspector.session_layout import variant_dir_name

    assert variant_dir_name(name) == _safe(name), contract_message(
        "_grid_base._safe", "session_layout.py (variant_dir_name)", variant_dir_name(name)
    )
