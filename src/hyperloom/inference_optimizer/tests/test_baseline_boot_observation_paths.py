# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline round names its persisted boot observation on every result.

The enablement gate exists for the case where a server still fails to boot, so
the failure branches -- not just the success one -- have to carry the artifact
the gate reads.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors import baseline as bl
from hyperloom.orchestrator.bringup import load_boot_observation
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task

_BOOT_FAILURE = (
    "2026-09-02 10:00:00 server_args=Namespace(model='m')\n"
    "2026-09-02 10:00:04 Loading weights\n"
    "2026-09-02 10:00:11 Traceback (most recent call last):\n"
    '2026-09-02 10:00:11   File "/opt/vllm/vllm/model_executor/loader.py", line 88, in load\n'
    "2026-09-02 10:00:11 KeyError: 'model.layers.0.mlp.gate_up_proj.weight'\n"
)


def _ctx() -> RunnerContext:
    task = Task(
        task_id="baseline-1",
        kind="baseline",
        state="queued",
        params={},
        idempotency_key="baseline-1",
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


async def _round(session: Path, out: Path) -> dict:
    ex = bl.BaselineExecutor(session_dir=session, magpie_python="/usr/bin/python3")
    cfg = session / "bench.yaml"
    cfg.write_text("benchmark:\n  framework: vllm\n  model: /tmp/m\n", encoding="utf-8")
    return await ex._run_single_benchmark(
        config_path=cfg,
        output_dir=out,
        timeout_sec=30,
        override_result_dir=None,
        resolved_model="/tmp/m",
        materialized_config_path=cfg,
        inferencex_path="",
        effective_extra_server_args="",
        params={"framework": "vllm"},
        ctx=_ctx(),
    )


@pytest.fixture()
def slot(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    out = tmp_path / "round"
    out.mkdir()
    monkeypatch.setattr(bl, "harvest_leaked_artifacts", lambda *a, **k: [])
    monkeypatch.setattr(bl, "snapshot_workspaces", lambda *a, **k: set())
    monkeypatch.setattr(bl, "select_run_workspace", lambda *a, **k: None)
    return session, out


def _write_log(out: Path) -> None:
    (out / "server.log").write_text(_BOOT_FAILURE, encoding="utf-8")


@pytest.mark.asyncio
async def test_nonzero_exit_result_names_a_real_observation(slot, monkeypatch) -> None:
    session, out = slot

    def _run(cmd, **kwargs):
        _write_log(out)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="magpie: benchmark failed")

    monkeypatch.setattr(bl, "launch", _run)
    result = await _round(session, out)

    assert result["status"] == "failed"
    loaded = load_boot_observation(result["boot_observation_path"])
    assert loaded.degraded == ""
    # The server's own log was classified, not the wrapper's one-line error.
    assert loaded.observation.evidence_ref == "server_log"
    assert loaded.observation.stage_reached is LadderStage.WEIGHTS_LOADING
    assert loaded.observation.stage_failed is LadderStage.WEIGHTS_LOADED
    assert loaded.observation.terminal_frame.line == 88
    # Measured on the server child's own clock, not the wrapper's wall-clock.
    assert loaded.observation.server_elapsed_sec == 11.0


@pytest.mark.asyncio
async def test_timeout_result_names_a_real_observation(slot, monkeypatch) -> None:
    session, out = slot

    def _run(cmd, **kwargs):
        _write_log(out)
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(bl, "launch", _run)
    result = await _round(session, out)

    assert result["error_class"] == "timeout"
    loaded = load_boot_observation(result["boot_observation_path"])
    assert loaded.degraded == ""
    assert loaded.observation.evidence_ref == "server_log"
    assert loaded.observation.terminal_frame.line == 88


@pytest.mark.asyncio
async def test_attempts_without_a_server_log_still_get_distinct_artifacts(slot, monkeypatch) -> None:
    """Three no-log attempts in one slot must not overwrite one another.

    The Magpie-never-created-a-workspace branch produces no ``server.log`` at
    all. An attempt index counted off retained log directories would stay at 0
    for every one of them, so the before half of a progress comparison would be
    replaced by the after half.
    """
    session, out = slot

    def _run(cmd, **kwargs):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="magpie: no benchmark workspace")

    monkeypatch.setattr(bl, "launch", _run)

    paths = [(await _round(session, out))["boot_observation_path"] for _ in range(3)]

    assert all(paths), paths
    assert len(set(paths)) == 3
    for path in paths:
        assert Path(path).is_file()
