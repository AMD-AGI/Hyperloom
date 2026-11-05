# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.blackboard.tests.factories import result
from hyperloom_multibranch.specialist import arbor_adapter
from hyperloom_multibranch.specialist.autonomous_session import AGENT_DIR, MANIFEST_FILE, START_TABLES_FILE
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
from hyperloom_multibranch.tests.test_specialist_autonomous_session import _imported, _rig


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
