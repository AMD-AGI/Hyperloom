# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestRequest, Validity
from hyperloom.contracts.recipe import ConfigDelta, apply
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import judgement, result
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion
from hyperloom_multibranch.contracts.toolbox import Backpressure
from hyperloom_multibranch.rulegate.rules import RETRACTION_ID
from hyperloom_multibranch.specialist.agent_import import KeptChange, Retraction
from hyperloom_multibranch.specialist import arbor_adapter, keep_requeue
from hyperloom_multibranch.specialist.autonomous_session import AGENT_DIR, MANIFEST_FILE, START_TABLES_FILE, KeepWatch
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionFinished, SessionRefused
from hyperloom_multibranch.tests.agent_rig import (
    BRANCH,
    CHAMPION_ROWS,
    CONFIG_FILE,
    LAUNCH,
    REASONS,
    SESSION_DIR,
    TABLE_ENV,
    Component,
    _aiter,
    _arbor_repo,
    _replayed,
    _until,
)
from hyperloom_multibranch.tests.test_specialist_autonomous_session import Rig, _failed, _imported, _rig, fast_session  # noqa: F401


async def test_an_arbor_run_on_the_adapter_s_config_has_its_keeps_imported_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    agent = Component(keeps=(256, 512), gap=0.1, latest_each=False, tail=0.6)
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, agent), start_seconds=0.2)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    assert rig.sandbox.prompts == [] and rig.sandbox.started == [LAUNCH]
    [request] = rig.toolbox.requests
    assert "+max_num_seqs = 512" in rig.patch(request)
    checkout = tmp_path / BRANCH / "session" / AGENT_DIR / "src"
    assert (checkout / SESSION_DIR / "keeps" / "LATEST").read_text() == "0002"
    assert (await rig.notices())[-1] == f"{NOTICE} the agent finished its run: done"


async def test_arbor_starts_with_the_start_s_kernel_tables_and_a_keep_that_keeps_them_imports_no_table_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    # The keep runs with the flags MB gave it, the TP MB names on vLLM included.
    agent = Component(keeps=(256,), gap=0.1, flags="--max-num-seqs 128 --tensor-parallel-size 1", tail=0.6)
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    source = _arbor_repo(tmp_path, agent)
    rig = await _rig(tmp_path, start=start, start_seconds=0.2, aiter=_aiter(tmp_path), source=source)

    request = await _imported(rig)

    config = json.loads((tmp_path / BRANCH / "session" / AGENT_DIR / "src" / CONFIG_FILE).read_text())
    given = json.loads((tmp_path / BRANCH / "session" / START_TABLES_FILE).read_text())[TABLE_ENV]["value"]
    assert config["server_env"] == {arbor_adapter.ACCURACY_FLOOR_ENV: "0.5", TABLE_ENV: given}
    assert Path(given.split(os.pathsep)[-1]).read_text() == CHAMPION_ROWS
    assert (request.delta.add_kernels, request.delta.remove_kernels) == ((), ())
    assert request.delta.config == ConfigDelta() and "+max_num_seqs = 256" in rig.patch(request)


async def test_an_arbor_started_with_too_little_time_ends_its_branch_as_finished_after_one_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    agent = Component(keeps=(), tail=0.0, reason="deadline")
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, agent), start_seconds=5.0)

    await asyncio.wait_for(rig.session.preflight(), 20)
    with pytest.raises(SessionFinished, match="the agent finished its run: deadline"):
        await asyncio.wait_for(rig.session.run(), 10)

    assert rig.sandbox.started == [LAUNCH]
    assert (await rig.notices())[-1] == f"{NOTICE} the agent finished its run: deadline"


async def test_an_arbor_restarted_after_a_crash_is_resumed_by_its_stored_manifest_without_onboarding_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, Component(keeps=(), exit=3, tail=0.6)), start_seconds=0.2)
    config = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / CONFIG_FILE

    await rig.session.preflight()
    written = config.read_text()
    with pytest.raises(SessionCrashed, match="the agent exited 3"):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    resumed = rig.restart()
    with pytest.raises(SessionCrashed, match="the agent exited 3"):
        await asyncio.wait_for(resumed.run(), 10)
    await resumed.stop()

    stored = json.loads((tmp_path / BRANCH / "session" / MANIFEST_FILE).read_text())
    assert (stored["launch"], stored["resume"]) == (LAUNCH, LAUNCH)
    assert rig.sandbox.started == [LAUNCH, LAUNCH] and rig.sandbox.prompts == []
    assert config.read_text() == written
    assert any(note == f"{NOTICE} the agent resumed with `{LAUNCH}`" for note in await rig.notices())


async def test_an_arbor_that_refuses_its_config_ends_the_branch_for_good_with_its_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())
    rig = await _rig(tmp_path, source=_arbor_repo(tmp_path, Component(tail=0.6), REASONS), start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionRefused) as refused:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert str(refused.value) == f"the agent refused its config: {'; '.join(REASONS)}"
    assert isinstance(refused.value, SessionFinished) and not isinstance(refused.value, SessionCrashed)
    assert (await rig.notices())[-1] == f"{NOTICE} {refused.value}"


FINDING = {"iteration": 7, "keep": 1, "finding": "iter7-agentx-283-selection-invalid"}
#: The launch an Arbor branch is started with and keeps, so a revert of a keep's source undoes all of it.
STEADY = "--tp 1"


def _retracting(keeps: tuple[int, ...]) -> Component:
    """An agent whose keep 2 reverts keep 1's commit and says it retracts keep 1."""
    started = {"server_flags": STEADY, "server_env": {}}
    return Component(keeps=keeps, retracts=FINDING, retracting_keep=2, flags=STEADY, started=started, tail=1.0)


async def _landed_in_turn(rig: Rig, tmp_path: Path, landed: int) -> list[TestRequest]:
    """Every import of a component agent's run, each measured once the one before was judged: the first ``landed`` of
    them committed on the branch, the rest recorded only."""

    gate = LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate"))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    while not running.done():
        count = len(rig.toolbox.requests)
        await _until(lambda: len(rig.toolbox.requests) > count or running.done(), "the next import")
        if len(rig.toolbox.requests) > count:
            request = rig.toolbox.requests[count]
            request_id = request.request_id or ""
            await rig.toolbox_blackboard.write_result(result(request_id))
            made = None
            if count < landed:
                recipe = apply(await rig.repo.read_recipe(request.parent_commit), request.delta)
                made = await rig.repo.commit(BRANCH, request.parent_commit, recipe, request_id)
            conclusion = Conclusion.COMMITTED if made else Conclusion.RECORDED_ONLY
            await gate.write_judgement(request_id, judgement(-3.0, (request_id,)), conclusion, made)
    with pytest.raises(SessionFinished):
        await running
    return rig.toolbox.requests


async def test_a_retraction_of_a_keep_that_landed_is_imported_with_both_checks_and_names_what_it_retracts(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, _retracting((256, 512)), start_seconds=0.2)

    first, retraction = await _landed_in_turn(rig, tmp_path, landed=1)

    [attachment] = retraction.attachments
    assert attachment.id == RETRACTION_ID
    landed = (await rig.toolbox_blackboard.record(first.request_id or "")).commit_id
    assert json.loads(rig.toolbox.attachments[attachment.sha256]) == {**FINDING, "retracted_commit": landed}
    assert (retraction.consistency_check, retraction.long_context_check) == (True, True)
    assert first.attachments == ()
    named = "the agent's iteration 7 (its keep 1), found invalid by iter7-agentx-283-selection-invalid"
    assert f"{NOTICE} keep 2 retracts {named}: r2 is not held to the speed gate" in await rig.notices()


async def test_a_retraction_that_does_more_than_undo_its_keep_is_held_to_the_speed_gate(tmp_path: Path) -> None:
    agent = replace(_retracting((256, 512)), flags=f"{STEADY} --max-num-seqs {{seqs}}")
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    _, retraction = await _landed_in_turn(rig, tmp_path, landed=1)

    assert (retraction.attachments, retraction.consistency_check) == ((), None)
    held = "r2 is held to the speed gate: it also changes --max-num-seqs"
    assert any(note.endswith(held) for note in await rig.notices())


async def test_a_retraction_of_a_keep_that_did_not_land_changes_nothing_on_the_head(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, _retracting((256, 512)), start_seconds=0.2)

    [first] = await _landed_in_turn(rig, tmp_path, landed=0)

    assert first.attachments == ()
    assert f"{NOTICE} keep 2 changes nothing on the branch head" in await rig.notices()


async def test_a_retraction_goes_alone_after_the_keep_it_was_made_on_and_before_the_keep_after_it(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, _retracting((256, 512, 768)), start_seconds=0.2)
    rig.toolbox.refusals = [Backpressure(BRANCH, 1, 1)] * 40

    first, retraction, after = await _landed_in_turn(rig, tmp_path, landed=2)

    assert "+max_num_seqs = 256" in rig.patch(first) and "-max_num_seqs = 256" in rig.patch(retraction)
    assert "+max_num_seqs = 768" in rig.patch(after)
    assert [len(request.attachments) for request in (first, retraction, after)] == [0, 1, 0]


async def test_a_retraction_whose_import_measured_nothing_goes_once_more_as_the_retraction_it_went_as(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(keep_requeue, "REQUEUE_SECONDS", 0.3)
    rig = await _rig(tmp_path, _retracting((256, 512)), start_seconds=0.2)
    gate = LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate"))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    first = rig.toolbox.requests[0]
    await rig.toolbox_blackboard.write_result(result("r1"))
    recipe = apply(await rig.repo.read_recipe(first.parent_commit), first.delta)
    made = await rig.repo.commit(BRANCH, first.parent_commit, recipe, "r1")
    await gate.write_judgement("r1", judgement(-3.0, ("r1",)), Conclusion.COMMITTED, made)
    await _until(lambda: len(rig.toolbox.requests) == 2, "the retraction's import")
    await rig.toolbox_blackboard.write_result(_failed("r2", Validity.MEASUREMENT_FAILED))
    await _until(lambda: len(rig.toolbox.requests) == 3, "the requeued retraction")
    await rig.toolbox_blackboard.write_result(result("r3"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    _, retraction, again = rig.toolbox.requests
    assert rig.patch(again) == rig.patch(retraction) and again.parent_commit == retraction.parent_commit
    assert [attachment.id for attachment in again.attachments] == [RETRACTION_ID]
    assert again.attachments == retraction.attachments
    assert (again.consistency_check, again.long_context_check) == (True, True)
    assert any(note.endswith("r3 is not held to the speed gate") for note in await rig.notices())


def test_the_keeps_waiting_at_a_crash_are_read_back_by_the_run_after_it(tmp_path: Path) -> None:
    retraction = Retraction(7, 1, "iter7-selection-invalid", 1)
    later = KeptChange(3, "patch", "--max-num-seqs 768")
    watch = KeepWatch("a" * 40, seq=3, pending=KeptChange(2, "patch", "", retracts=retraction), waiting=[later])

    watch.save(tmp_path / "watch.json")

    loaded = KeepWatch.load(tmp_path / "watch.json")
    assert loaded.pending is not None and (loaded.pending.seq, loaded.pending.retracts) == (2, retraction)
    assert [kept.seq for kept in loaded.waiting] == [3]


async def test_a_snapshot_that_retracts_nothing_is_imported_as_before(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Component(keeps=(256,)), start_seconds=0.2)

    request = await _imported(rig)

    assert (request.attachments, request.consistency_check, request.long_context_check) == ((), None, None)
    assert rig.toolbox.attachments == {}
