# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for the after-kernel-opt rocprof roofline flow in
``kernel_request_handlers``: env gating, state resolution, tool/subprocess
failure branches, the happy path, and background scheduling."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.shared_state import SharedState

log = logging.getLogger("test")


def _state_with_test_command(session_dir: Path, kernel_id: str, cmd: str) -> None:
    """Persist a SharedState carrying a kernel_opt_attempt test_command."""
    state = SharedState.load_or_init(session_dir)
    state.kernel_opt_attempts = {kernel_id: {"test_command": cmd}}
    state.save(session_dir)


@pytest.mark.asyncio
async def test_run_after_kernel_opt_rocprof_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "0")
    out = await krh._run_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out == {"status": "skipped", "reason": "disabled_by_env"}


@pytest.mark.asyncio
async def test_run_after_kernel_opt_rocprof_no_test_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "1")
    out = await krh._run_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out["status"] == "skipped"
    assert out["reason"] == "no_test_command_in_state"


@pytest.mark.asyncio
async def test_run_after_kernel_opt_rocprof_tool_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "1")
    _state_with_test_command(tmp_path, "k1", "python harness_x --correctness")

    def _boom(_name):
        raise FileNotFoundError("no tool")

    monkeypatch.setattr(krh, "_kernel_agent_tool_path", _boom)
    out = await krh._run_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out == {
        "status": "skipped", "reason": "rocprof_roofline_tool_unavailable",
    }


@pytest.mark.asyncio
async def test_run_after_kernel_opt_rocprof_subprocess_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "1")
    _state_with_test_command(tmp_path, "k1", "python harness_x --correctness")
    monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda _n: tmp_path / "tool.py")
    monkeypatch.setattr(krh, "_lookup_kernel_roofline_name", lambda *a, **k: "")

    async def _raise(*_a, **_k):
        raise RuntimeError("subprocess blew up")

    monkeypatch.setattr(krh, "_run_subprocess", _raise)
    out = await krh._run_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out["status"] == "failed"
    assert "RuntimeError" in out["reason"]


@pytest.mark.asyncio
async def test_run_after_kernel_opt_rocprof_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "1")
    _state_with_test_command(tmp_path, "k1", "python harness_x --correctness")
    monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda _n: tmp_path / "tool.py")
    monkeypatch.setattr(krh, "_lookup_kernel_roofline_name", lambda *a, **k: "gemm_kernel")

    async def _fake_subprocess(cmd, *, timeout_sec):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        out_json.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        # confirm the target-kernel flag is forwarded
        assert "--target-kernel" in cmd
        return 0, "stdout", ""

    monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)
    out = await krh._run_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out["status"] == "ok"
    assert out["json_path"].endswith("after.json")


@pytest.mark.asyncio
async def test_schedule_after_kernel_opt_rocprof_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "off")
    out = krh._schedule_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out == {"status": "skipped", "reason": "disabled_by_env"}


@pytest.mark.asyncio
async def test_schedule_after_kernel_opt_rocprof_scheduled(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE", "1")
    out = krh._schedule_after_kernel_opt_rocprof(
        kernel_id="k1", session_dir=tmp_path, log=log,
    )
    assert out == {"status": "scheduled", "reason": "background_task"}
    # Drain the spawned background task (it short-circuits on no_test_command).
    await asyncio.sleep(0)
    pending = [t for t in krh._BACKGROUND_ROCPROF_TASKS]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
