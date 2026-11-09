# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that holds a credential, resumed by the same account in a new container.

The coordinator database of such a session is kept outside the session directory, below the container's ``TMPDIR``,
so a run resumed after a preemption finds none there: an empty one is made. These tests resume the same session with
the database kept (the same ``TMPDIR``, or a new ``TMPDIR`` and ``HYPERLOOM_LEDGER_DIR`` on a disk that outlives the
container) and with a new empty one, and read what the resume recovers: the half-applied integrate window the stopped
leg left, the KEEP it had recorded on the bus but not yet put on the stack, and the proposals nobody had judged."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import ENV_LEDGER_DIR, db_path_for, private_db_path_for
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
async def test_a_resume_in_a_new_container_without_a_named_ledger_directory_starts_an_empty_ledger(
    session_dir, tmp_path, monkeypatch
):
    monkeypatch.delenv(ENV_LEDGER_DIR, raising=False)
    first = _container(tmp_path, monkeypatch, "container-1")
    await _lost_with_an_integrate_window_open(session_dir)
    assert first in private_db_path_for(session_dir).parents

    _container(tmp_path, monkeypatch, "container-2")
    seen = await _resumed_leg(session_dir, monkeypatch)
    assert seen["proposals"] == 0 and seen["stack"] == 0, "the KEEP on the bus and the open proposal are gone"
    assert seen["refused"] is not None and "pending integrate recovery is incomplete" in str(seen["refused"])


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
async def test_a_resume_in_a_new_container_loses_a_recorded_keep_and_the_open_proposals_with_the_ledger(
    session_dir, tmp_path, monkeypatch
):
    """Without an integrate window to refuse on, the resume goes on, minus what only the ledger held: the KEEP on the bus
    is not put on the stack (its gain is found again or not at all) and the proposal nobody judged is gone."""
    monkeypatch.delenv(ENV_LEDGER_DIR, raising=False)
    _container(tmp_path, monkeypatch, "container-1")
    (await _stopped_leg(session_dir)).db.raw.close()
    _container(tmp_path, monkeypatch, "container-2")  # a new container: the ledger of the first is not there
    assert await _resumed_leg(session_dir, monkeypatch) == {"refused": None, "fixes": [], "proposals": 0, "stack": 0}
