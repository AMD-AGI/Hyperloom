# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session records the Claw session that hosts it in ``MB-manifest.json`` as it starts, so a collector of running
sessions can tell whose a directory is before CLOSE; a run Claw did not host records none. The id follows the leg that
runs: a ``run --resume`` or ``final-only`` writes the id it is given to the manifest and the breakdown, and one given
none keeps the recorded id. The manifest is the one record of the id: the toolbox's ``state.json`` and
``retest_view.json`` do not carry it."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import freeze_base, frozen_base
from hyperloom.orchestrator.state.retest_view import RETEST_VIEW
from hyperloom_multibranch.blackboard.knowledge.breakdown import (
    FILENAME,
    SESSION_VARIABLES,
    claw_session_id,
    leg_claw_session_id,
)
from hyperloom_multibranch.scheduling import lifecycle
from hyperloom_multibranch.scheduling.manifest import read_manifest
from hyperloom_multibranch.scheduling.peers import PeerSetup, Peers, PeersUnavailable
from hyperloom_multibranch.scheduling.resumption import deadline_after_downtime, persisted_run, resumable_run
from hyperloom_multibranch.scheduling.session_base import freeze_session_base
from hyperloom_multibranch.tests.test_skeleton import (
    SCRIPTS,
    Stage,
    StuckStopToolbox,
    _recovery,
    _resumed,
    _script,
    _signalling,
    _stage,
)

CLAW = SESSION_VARIABLES["claw_session_id"]


async def _resume(stage: Stage) -> Stage:
    """Resume the session ``stage`` stopped, as ``run --resume`` does."""

    persisted = await asyncio.to_thread(resumable_run, stage.layout.root)
    resumed = _resumed(stage, persisted.settings)
    with resumed._polls_on_virtual_time():
        deadline = deadline_after_downtime(resumed.layout, resumed.time.now())
        await resumed._session().resume(persisted.ended, deadline)
    return resumed


def _breakdown_claw_session(stage: Stage) -> object:
    """The Claw session the ``session_breakdown.json`` CLOSE wrote in ``stage``'s directory names."""

    breakdown = json.loads((stage.layout.reports / FILENAME).read_text())
    return breakdown["metadata"]["session"]["claw_session_id"]


async def test_a_session_claw_hosts_records_its_claw_session_in_the_manifest_as_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CLAW, "claw-sess-1")
    stage = _stage(tmp_path)
    _script(stage)
    await stage.run()

    assert json.loads(stage.layout.manifest.read_text())["claw_session_id"] == "claw-sess-1"
    assert read_manifest(stage.layout.manifest).claw_session_id == "claw-sess-1"


@pytest.mark.parametrize("given", [None, "", "  "], ids=["unset", "empty", "blank"])
async def test_a_session_claw_did_not_host_records_no_claw_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: str | None
) -> None:
    if given is None:
        monkeypatch.delenv(CLAW, raising=False)
    else:
        monkeypatch.setenv(CLAW, given)
    stage = _stage(tmp_path)
    _script(stage)
    await stage.run()

    assert json.loads(stage.layout.manifest.read_text())["claw_session_id"] is None


async def test_a_manifest_an_earlier_build_wrote_reads_with_no_claw_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CLAW, "claw-sess-1")
    stage = _stage(tmp_path)
    _script(stage)
    await stage.run()
    manifest = json.loads(stage.layout.manifest.read_text())
    del manifest["claw_session_id"]
    stage.layout.manifest.write_text(json.dumps(manifest))

    assert read_manifest(stage.layout.manifest).claw_session_id is None


def test_the_claw_session_is_read_from_the_variable_the_breakdown_reads() -> None:
    assert CLAW == "CLAW_SESSION_ID"
    assert claw_session_id({CLAW: " claw-1 "}) == "claw-1"
    assert claw_session_id({CLAW: ""}) is None and claw_session_id({}) is None


def _given(monkeypatch: pytest.MonkeyPatch, claw: str | None) -> None:
    """Run what follows under the Claw session ``claw``, or under none."""

    if claw is None:
        monkeypatch.delenv(CLAW, raising=False)
    else:
        monkeypatch.setenv(CLAW, claw)


#: (the id the session started under, the id the next leg is given, the id that leg records): the id follows the leg
#: that runs, and a leg given none keeps the recorded one.
LEGS = [
    ("claw-first", "claw-other", "claw-other"),
    ("claw-first", None, "claw-first"),
    (None, "claw-late", "claw-late"),
    ("claw-first", "claw-first", "claw-first"),
]
LEG_IDS = ["another", "none", "none-recorded", "same"]


@pytest.mark.parametrize(("started", "given", "named"), LEGS, ids=LEG_IDS)
async def test_a_resumed_session_records_the_claw_session_of_the_leg_that_runs_in_the_manifest_and_the_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, started: str | None, given: str | None, named: str
) -> None:
    _given(monkeypatch, started)
    stage = _stage(tmp_path)
    stage.steps = _signalling(stage)
    await stage.run()
    assert read_manifest(stage.layout.manifest).claw_session_id == started
    (stage.layout.reports / FILENAME).unlink()
    _given(monkeypatch, given)

    await _resume(stage)

    assert json.loads(stage.layout.manifest.read_text())["claw_session_id"] == named
    assert _breakdown_claw_session(stage) == named


@pytest.mark.parametrize(("started", "given", "named"), LEGS, ids=LEG_IDS)
async def test_a_final_only_records_the_claw_session_of_the_leg_that_runs_in_the_manifest_and_the_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, started: str | None, given: str | None, named: str
) -> None:
    _given(monkeypatch, started)
    stage = _stage(tmp_path)
    stage.scripts = SCRIPTS
    stage.toolbox_type = StuckStopToolbox
    await stage.run()
    assert read_manifest(stage.layout.manifest).claw_session_id == started
    (stage.layout.reports / FILENAME).unlink()
    persisted = persisted_run(stage.layout.root)
    _given(monkeypatch, given)

    await _recovery(stage, persisted.settings).finish(persisted.ended)

    assert json.loads(stage.layout.manifest.read_text())["claw_session_id"] == named
    assert _breakdown_claw_session(stage) == named


async def test_a_resumed_session_run_by_no_claw_session_records_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _given(monkeypatch, None)
    stage = _stage(tmp_path)
    stage.steps = _signalling(stage)
    await stage.run()

    await _resume(stage)

    assert json.loads(stage.layout.manifest.read_text())["claw_session_id"] is None
    assert _breakdown_claw_session(stage) is None


@pytest.mark.parametrize(("recorded", "given", "named"), LEGS, ids=LEG_IDS)
def test_the_leg_rule_takes_the_given_claw_session_else_the_recorded_one(
    recorded: str | None, given: str | None, named: str
) -> None:
    environ = {} if given is None else {CLAW: f" {given} "}
    assert leg_claw_session_id(recorded, environ) == named
    assert leg_claw_session_id(recorded, {CLAW: "  "}) == recorded


def _toolbox_claw_sessions(stage: Stage) -> tuple[object, object]:
    """The Claw session the toolbox's ``state.json`` and its ``retest_view.json`` in ``stage``'s directory name."""

    toolbox = stage.layout.toolbox
    state = json.loads((toolbox / "state.json").read_text())
    view = json.loads((toolbox / RETEST_VIEW).read_text())
    return state["claw_session_id"], view["claw_session_id"]


def _freeze_base(setup: PeerSetup) -> None:
    """Freeze the session's base as the local peers do on every leg (:func:`freeze_session_base`), its launch record
    written first so that nothing is resolved."""

    toolbox = setup.layout.toolbox
    if frozen_base(toolbox) is None:
        freeze_base(toolbox, base_launch())
    freeze_session_base(toolbox, setup.workload, "")


def _refused_after_the_freeze(setup: PeerSetup) -> Peers:
    """Peers that freeze the session's base and are then refused, as the local peers are when the staging cannot be
    published."""

    _freeze_base(setup)
    raise PeersUnavailable("the inference_suite staging the runners read cannot be published")


async def test_a_leg_refused_after_the_base_freeze_and_a_later_final_only_leave_one_record_of_the_claw_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _given(monkeypatch, "claw-first")
    stage = _stage(tmp_path)
    stage.scripts = SCRIPTS
    stage.toolbox_type = StuckStopToolbox
    await stage.run()
    persisted = persisted_run(stage.layout.root)
    _given(monkeypatch, "claw-refused")
    refused = _recovery(stage, persisted.settings)
    refused.factory = _refused_after_the_freeze  # type: ignore[method-assign]

    with pytest.raises(PeersUnavailable):
        await refused.finish(persisted.ended)

    assert read_manifest(stage.layout.manifest).claw_session_id == "claw-first"
    assert _breakdown_claw_session(stage) == "claw-first"
    assert _toolbox_claw_sessions(stage) == ("", "")
    (stage.layout.reports / FILENAME).unlink()
    _given(monkeypatch, "claw-other")
    closing = _recovery(stage, persisted.settings)
    build = closing.factory

    def freezing(setup: PeerSetup) -> Peers:
        _freeze_base(setup)
        return build(setup)

    closing.factory = freezing  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO, logger=lifecycle.__name__):
        await closing.finish(persisted.ended)

    assert read_manifest(stage.layout.manifest).claw_session_id == "claw-other"
    assert _breakdown_claw_session(stage) == "claw-other"
    assert _toolbox_claw_sessions(stage) == ("", "")
    assert "changes from claw-first to claw-other" in caplog.text
