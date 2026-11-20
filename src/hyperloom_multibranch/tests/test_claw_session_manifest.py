# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session records the Claw session that hosts it in ``MB-manifest.json`` as it starts, so a collector of running
sessions can tell whose a directory is before CLOSE; a run Claw did not host records none, and a resumed session keeps
the id it recorded."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.blackboard.knowledge.breakdown import SESSION_VARIABLES, claw_session_id
from hyperloom_multibranch.scheduling import lifecycle
from hyperloom_multibranch.scheduling.manifest import read_manifest
from hyperloom_multibranch.scheduling.resumption import deadline_after_downtime, resumable_run
from hyperloom_multibranch.tests.test_skeleton import Stage, _resumed, _script, _signalling, _stage

CLAW = SESSION_VARIABLES["claw_session_id"]


async def _resume(stage: Stage) -> Stage:
    """Resume the session ``stage`` stopped, as ``run --resume`` does."""

    persisted = await asyncio.to_thread(resumable_run, stage.layout.root)
    resumed = _resumed(stage, persisted.settings)
    with resumed._polls_on_virtual_time() as patch:
        patch.setattr(lifecycle, "run_image_check", resumed.image_check)
        deadline = deadline_after_downtime(resumed.layout, resumed.time.now())
        await resumed._session().resume(persisted.ended, deadline)
    return resumed


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


async def test_a_resumed_session_keeps_the_claw_session_it_recorded_and_warns_when_given_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(CLAW, "claw-first")
    stage = _stage(tmp_path)
    stage.steps = _signalling(stage)
    await stage.run()
    monkeypatch.setenv(CLAW, "claw-other")

    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.scheduling.manifest"):
        await _resume(stage)

    assert read_manifest(stage.layout.manifest).claw_session_id == "claw-first"
    assert any("'claw-other'" in record.getMessage() for record in caplog.records)


async def test_a_resumed_session_given_no_claw_session_keeps_the_one_it_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CLAW, "claw-first")
    stage = _stage(tmp_path)
    stage.steps = _signalling(stage)
    await stage.run()
    monkeypatch.delenv(CLAW)

    await _resume(stage)

    assert read_manifest(stage.layout.manifest).claw_session_id == "claw-first"


async def test_a_resumed_session_that_recorded_no_claw_session_takes_the_one_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CLAW, raising=False)
    stage = _stage(tmp_path)
    stage.steps = _signalling(stage)
    await stage.run()
    monkeypatch.setenv(CLAW, "claw-late")

    await _resume(stage)

    assert read_manifest(stage.layout.manifest).claw_session_id == "claw-late"


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
