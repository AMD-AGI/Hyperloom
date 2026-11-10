# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that holds a credential, resumed by the same account in a new container.

The coordinator database of such a session is kept outside the session directory, below the container's ``TMPDIR``,
so a run resumed after a preemption finds none there. These tests resume the same session with the database kept (the
same ``TMPDIR``, or a new ``TMPDIR`` and ``HYPERLOOM_LEDGER_DIR`` on a disk that outlives the container), and read what
the resume recovers: the half-applied integrate window the stopped leg left, the KEEP it had recorded on the bus but not
yet put on the stack, and the proposals nobody had judged. Without the database, and with the session having run, the
resume is refused, naming ``HYPERLOOM_LEDGER_DIR``: the leases and tasks of the earlier legs cannot be inspected, so
work they left running would go unseen (fail closed, as the guard did on the shared database)."""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.resume_guard import ResumeBlocked, ensure_resume_safe
from hyperloom.inference_optimizer.session.paths import (
    ENV_LEDGER_DIR,
    db_path_for,
    private_db_path_for,
    shared_db_path_for,
)
from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.bus.storage.connection import DatabaseNotPrivate
from hyperloom.orchestrator.loop._promote_outcome import IntegrateRecoveryIncomplete
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskSpec

_SECRET = "hf_live_value_123"
_KEPT = {
    "task_id": "ti-orphan",
    "kind": "integrate_patch",
    "state": "succeeded",
    "result": {
        "status": "kept",
        "specialist_task_id": "spec-orphan",
        "output_throughput": 123.0,
        "source_phase": "FRAMEWORK_AGENT",
        "domain": "serving_specialist",
        "provenance": "specialist:serving_specialist",
        "framework_agent_authoring": True,
        "source_manifest": "/session/optimization_stack/src/spec-orphan/manifest.json",
        "target_files": ["vllm/model.py"],
    },
}


def _coordinator(session: Path) -> Coordinator:
    plan = ScriptedPlan(
        turns=[], default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "observation", "body_md": "ok"})
    )
    return Coordinator(session, backends={name: MockBackend(plan, name=name) for name in ("orchestration", "critic")})


async def _stopped_leg(session: Path) -> Coordinator:
    """A leg with a credential in its pins that has a KEEP recorded but not on the stack and a proposal nobody had judged,
    and is about to be lost (its coordinator is returned open)."""
    state = SharedState()
    state.operator_extra_env = {"HF_TOKEN": _SECRET}
    state.save(session)
    coord = _coordinator(session)
    coord.shared_state.tick = 3  # the leg ran: a tick was counted
    coord.shared_state.save(session)
    await coord.bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", _KEPT))
    await coord.bus.append_and_seq(
        Message.new("orchestration", "*", "proposal", {"action_name": "explore", "predicted_gain_pct": 2.0})
    )
    return coord


async def _lost_with_an_integrate_window_open(session: Path) -> None:
    """The stopped leg dies with an integrate window open too: a task row that reads running and the sentinel."""
    coord = await _stopped_leg(session)
    task, _ = await coord.tasks.create_or_return_existing(
        TaskSpec(kind="integrate_patch", params={}, idempotency_key="ti-roll", task_id="ti-roll")
    )
    await coord.tasks.transition(task.task_id, "running")
    coord.shared_state.pending_integrate = {
        "task_id": "ti-roll",
        "framework_source_root": "/tmp/framework",
        "patches": ["/tmp/p.diff"],
    }
    coord.shared_state.save(session)
    coord.db.raw.close()


async def _resumed_leg(session: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """What the resume recovers: whether it was refused, the fixes it made, the stack and the proposals it rebuilt."""
    coord = _coordinator(session)
    monkeypatch.setattr(
        coord.writeback,
        "_resume_rollback_pending_integrate",
        lambda pending, _gone_scope=None: {"reversed": list(pending["patches"]), "failed": []},
    )
    await coord.writeback.replay_for_resume()
    proposals = len(coord.state.pending_proposals)
    try:
        report = await coord.writeback._resume_consistency_pass()
    except IntegrateRecoveryIncomplete as exc:
        outcome: dict[str, object] = {"refused": str(exc), "fixes": []}
    else:
        outcome = {"refused": None, "fixes": sorted(fix["kind"] for fix in report["fixes"])}
    coord.db.raw.close()
    return {**outcome, "proposals": proposals, "stack": len(coord.shared_state.optimization_stack or [])}


_KEEP_REPLAYED = {
    "refused": None,
    "fixes": ["queued_resume_stack_rebench", "replayed_orphaned_keep"],
    "proposals": 1,
    "stack": 1,
}


_RECOVERED = {**_KEEP_REPLAYED, "fixes": [*_KEEP_REPLAYED["fixes"], "rolled_back_pending_integrate"]}


def _container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> Path:
    """A container's own temporary directory."""
    path = tmp_path / name
    path.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(path))
    return path


@pytest.mark.asyncio
async def test_a_resume_in_the_same_container_recovers_what_the_stopped_leg_left(session_dir, tmp_path, monkeypatch):
    _container(tmp_path, monkeypatch, "container")
    await _lost_with_an_integrate_window_open(session_dir)
    assert await _resumed_leg(session_dir, monkeypatch) == _RECOVERED


@pytest.mark.asyncio
async def test_a_resume_in_a_new_container_on_the_ledger_directory_an_operator_named_recovers_it_too(
    session_dir, tmp_path, monkeypatch
):
    ledgers = tmp_path / "ledgers"  # a disk that outlives the container, a directory of this account's
    ledgers.mkdir(mode=0o700)
    monkeypatch.setenv(ENV_LEDGER_DIR, str(ledgers))
    _container(tmp_path, monkeypatch, "container-1")
    await _lost_with_an_integrate_window_open(session_dir)
    database = db_path_for(session_dir)
    assert ledgers in database.parents and stat.S_IMODE(os.stat(database).st_mode) == 0o600

    _container(tmp_path, monkeypatch, "container-2")  # the preempted run starts again on a new pod
    assert db_path_for(session_dir) == database
    assert await _resumed_leg(session_dir, monkeypatch) == _RECOVERED


@pytest.mark.asyncio
async def test_a_resume_in_a_new_container_without_a_ledger_is_refused_naming_the_ledger_directory(
    session_dir, tmp_path, monkeypatch
):
    """The session holds a credential and has run, this account has no database for it, and none is left in the session
    directory: the guard refuses (fail closed), and so does the coordinator should it be reached without the guard."""
    monkeypatch.delenv(ENV_LEDGER_DIR, raising=False)
    first = _container(tmp_path, monkeypatch, "container-1")
    await _lost_with_an_integrate_window_open(session_dir)
    assert first in private_db_path_for(session_dir).parents

    second = _container(tmp_path, monkeypatch, "container-2")
    with pytest.raises(ResumeBlocked) as guard:
        ensure_resume_safe(session_dir, owner_scope="host:2")
    with pytest.raises(ResumeBlocked) as coordinator:
        _coordinator(session_dir)
    for refusal in (guard.value, coordinator.value):
        said = str(refusal)
        assert "HYPERLOOM_LEDGER_DIR" in said and "start a new session" in said, said
        assert "\n" not in said, "one line"
    assert list(second.iterdir()) == [], "nothing was made: no empty ledger"


@pytest.mark.parametrize("named", ["unsafe-group-writable", "unsafe-wider"])
@pytest.mark.asyncio
async def test_a_named_ledger_directory_that_is_not_safe_is_refused_and_nothing_is_made(
    session_dir, tmp_path, monkeypatch, named
):
    ledgers = tmp_path / named
    ledgers.mkdir(mode=0o700)
    ledgers.chmod(0o2775 if named == "unsafe-group-writable" else 0o755)
    monkeypatch.setenv(ENV_LEDGER_DIR, str(ledgers))
    _container(tmp_path, monkeypatch, "container")
    before = sorted(path.name for path in ledgers.rglob("*"))
    state = SharedState()
    state.operator_extra_env = {"HF_TOKEN": _SECRET}
    state.save(session_dir)
    if named == "unsafe-wider":
        (ledgers / private_db_path_for(session_dir).parent.name).mkdir(mode=0o755)
        before = sorted(path.name for path in ledgers.rglob("*"))
    with pytest.raises(DatabaseNotPrivate) as raised:
        _coordinator(session_dir)
    assert str(ledgers) in str(raised.value)
    assert sorted(path.name for path in ledgers.rglob("*")) == before


@pytest.mark.asyncio
async def test_a_credential_added_to_a_session_that_ran_without_one_is_resumed_on_an_empty_ledger_with_a_warning(
    session_dir, tmp_path, monkeypatch, caplog
):
    """A database is in the session directory, which the guard inspects: the resume goes on, on a ledger of this
    account's that starts empty, and a warning says the earlier database is not read."""
    _container(tmp_path, monkeypatch, "container")
    first = _coordinator(session_dir)  # no credential: its database is in the session directory
    first.shared_state.tick = 3
    first.shared_state.save(session_dir)
    first.db.raw.close()
    state = SharedState.load_or_init(session_dir)
    state.operator_extra_env = {"HF_TOKEN": _SECRET}
    state.save(session_dir)

    ensure_resume_safe(session_dir, owner_scope="host:1")
    caplog.set_level(logging.WARNING)
    resumed = _coordinator(session_dir)
    assert private_db_path_for(session_dir).exists()
    resumed.db.raw.close()
    assert any("is not read" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_first_leg_with_a_credential_is_not_refused_for_having_no_ledger(session_dir, tmp_path, monkeypatch):
    _container(tmp_path, monkeypatch, "container")
    state = SharedState()
    state.operator_extra_env = {"HF_TOKEN": _SECRET}
    state.save(session_dir)  # nothing ran yet: no tick, no window, no stack
    ensure_resume_safe(session_dir, owner_scope="host:1")
    _coordinator(session_dir).db.raw.close()


@pytest.mark.asyncio
async def test_a_session_without_a_credential_resumed_in_a_new_container_keeps_its_database_and_is_not_refused(
    session_dir, tmp_path, monkeypatch
):
    _container(tmp_path, monkeypatch, "container-1")
    first = _coordinator(session_dir)
    first.shared_state.tick = 3
    first.shared_state.save(session_dir)
    first.db.raw.close()
    _container(tmp_path, monkeypatch, "container-2")
    ensure_resume_safe(session_dir, owner_scope="host:2")
    resumed = _coordinator(session_dir)  # its database is in the session directory, where a new container finds it
    assert resumed.db.db_path == shared_db_path_for(session_dir)
    resumed.db.raw.close()


async def _a_leg_whose_worker_exit_was_never_proven(session: Path) -> None:
    """A leg with a credential whose ledger holds a GPU lease: what the resume guard refuses to resume past."""
    coord = await _stopped_leg(session)
    coord.db.raw.execute("INSERT INTO gpu_leases (gpu_id, holder_id, task_id) VALUES (0, 'worker', 'ti-live')")
    coord.db.raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [True, False], ids=["ledger-directory-named", "default-ledger"])
async def test_the_resume_guard_reads_the_leases_of_a_kept_ledger_and_refuses_when_a_new_default_one_has_none(
    session_dir, tmp_path, monkeypatch, named
):
    """Where main's guard read the leases in the shared database and refused while work was live, the guard reads the
    ledger this account has: kept across containers by ``HYPERLOOM_LEDGER_DIR``, it still refuses while work is live; the
    default one is gone with the container, and the guard refuses the resume, naming the variable, rather than admit it
    with nothing to inspect."""
    monkeypatch.delenv(ENV_LEDGER_DIR, raising=False)
    if named:
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir(mode=0o700)
        monkeypatch.setenv(ENV_LEDGER_DIR, str(ledgers))
    _container(tmp_path, monkeypatch, "container-1")
    await _a_leg_whose_worker_exit_was_never_proven(session_dir)
    _container(tmp_path, monkeypatch, "container-2")
    if named:
        with pytest.raises(ResumeBlocked, match="worker exit is unproven"):
            ensure_resume_safe(session_dir, owner_scope="host:2")
    else:
        with pytest.raises(ResumeBlocked, match="HYPERLOOM_LEDGER_DIR"):  # nothing to inspect: refused, not admitted
            ensure_resume_safe(session_dir, owner_scope="host:2")


def test_a_session_without_a_credential_that_ran_is_not_refused_for_lacking_a_ledger(
    session_dir, tmp_path, monkeypatch
):
    """The refusal is for a credential session, whose database cannot be in the session directory: one that holds none
    is admitted as before, whatever is left of its database."""
    monkeypatch.delenv(ENV_LEDGER_DIR, raising=False)
    _container(tmp_path, monkeypatch, "container")
    state = SharedState()
    state.tick = 3
    state.save(session_dir)
    assert not shared_db_path_for(session_dir).exists() and not private_db_path_for(session_dir).exists()
    ensure_resume_safe(session_dir, owner_scope="host:1")
