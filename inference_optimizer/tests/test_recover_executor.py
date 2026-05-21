"""Unit tests for :class:`RecoverExecutor`.

The executor coordinates three independently-failing side effects
(rocm-smi probes, pgrep/kill, optional gpureset). Tests inject pure
in-process stubs for each and assert the result-dict shape Robustness +
Coordinator rely on, plus the on-disk ``result.json`` audit trail.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors.recover import (
    RecoverExecutor,
    recover_executor,
)
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# RunnerContext stand-in (mirrors test_target_analysis_executor.py)
# ---------------------------------------------------------------------------
@dataclass
class _Ctx:
    task: Task
    lease: Any = None
    extra: dict[str, Any] | None = None


def _ctx(
    workspace: Path | None,
    params: dict[str, Any] | None = None,
) -> _Ctx:
    return _Ctx(
        task=Task(
            task_id="t-recover-1",
            kind="recover",
            state="running",
            params=params or {},
            idempotency_key="recover-test-1",
        ),
        extra={"workspace": str(workspace) if workspace else None},
    )


def _healthy_probe(num_gpus: int = 4, free_mb: float = 180_000.0) -> list[dict]:
    """Helper: ``_probe_gpu_free_mb`` return value with all GPUs healthy."""
    return [
        {
            "gpu_id": i,
            "vram_used_mb": 196_608.0 - free_mb,
            "vram_total_mb": 196_608.0,
            "free_mb": free_mb,
        }
        for i in range(num_gpus)
    ]


def _leaked_probe(num_gpus: int = 4, free_mb: float = 0.0) -> list[dict]:
    return [
        {
            "gpu_id": i,
            "vram_used_mb": 196_608.0 - free_mb,
            "vram_total_mb": 196_608.0,
            "free_mb": free_mb,
        }
        for i in range(num_gpus)
    ]


# ---------------------------------------------------------------------------
# No-op path — GPU already healthy
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_stale_owners_returns_succeeded(tmp_path, monkeypatch):
    """Healthy GPUs + no stale owners -> state=succeeded, no kills."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _healthy_probe())
    # _kill_stale_owners would only run when force_gpu_cleanup=True; even
    # then with no pgrep matches it returns []. We patch the discovery
    # to be empty so the unit test never reaches subprocess.
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])

    out = await exe(_ctx(workspace, params={
        "reason": "smoke",
        "force_gpu_cleanup": True,
    }))

    assert out["state"] == "succeeded"
    assert out["killed_pids"] == []
    assert out["gpureset_attempted"] is False
    assert out["force_gpu_cleanup"] is True
    assert out["allow_reset_env"] is False
    # post == mid because no gpureset.
    assert out["mid_free_mb_per_gpu"] == out["post_free_mb_per_gpu"]
    # result.json was written.
    assert (workspace / "result.json").exists()


@pytest.mark.asyncio
async def test_force_cleanup_false_skips_kill_stage(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _healthy_probe())
    calls: list[str] = []

    def _should_not_be_called():
        calls.append("kill")
        return []

    monkeypatch.setattr(exe, "_kill_stale_owners", _should_not_be_called)

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": False}))
    assert out["state"] == "succeeded"
    assert calls == []
    assert out["killed_pids"] == []


# ---------------------------------------------------------------------------
# Happy path — kills stale owners and GPUs recover
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kills_stale_owners_and_recovers(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    # Pre: leaked. Post: healthy (matches the "soft cleanup succeeded" path).
    probes = iter([_leaked_probe(), _healthy_probe(), _healthy_probe()])
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: next(probes))

    discovered = [
        {
            "pid": 4242,
            "cmd": "python -m vllm.entrypoints.openai.api_server --port 8000",
            "pattern": "vllm.entrypoints",
        },
        {
            "pid": 4243,
            "cmd": "/sbin/Magpie --workload sglang_mi300x",
            "pattern": "Magpie",
        },
    ]
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: list(discovered))

    sent_signals: list[tuple[int, signal.Signals]] = []

    def _send(pid, sig):
        sent_signals.append((pid, sig))
        return True

    monkeypatch.setattr(exe, "_send_signal", _send)
    # All TERMed PIDs immediately exit (no SIGKILL fall-through).
    monkeypatch.setattr(exe, "_pid_alive", lambda pid: False)
    # Skip the real 5-second wait.
    import inference_optimizer.orchestrator.action_executors.recover as recmod
    monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

    out = await exe(_ctx(workspace, params={
        "reason": "gpu_memory_leaked",
        "force_gpu_cleanup": True,
    }))

    assert out["state"] == "succeeded"
    assert {pid for pid, _ in sent_signals} == {4242, 4243}
    assert all(sig == signal.SIGTERM for _, sig in sent_signals)
    assert [e["signal"] for e in out["killed_pids"]] == ["TERM", "TERM"]
    assert out["gpureset_attempted"] is False

    # result.json content matches the returned dict.
    persisted = json.loads((workspace / "result.json").read_text())
    assert persisted["state"] == "succeeded"
    assert persisted["killed_pids"][0]["pattern"] in {"vllm.entrypoints", "Magpie"}


@pytest.mark.asyncio
async def test_sigkill_fallthrough_when_pid_still_alive(tmp_path, monkeypatch):
    """TERM->wait->KILL: PIDs still alive after SIGTERM get SIGKILLed."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    probes = iter([_leaked_probe(), _healthy_probe(), _healthy_probe()])
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: next(probes))
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [
        {"pid": 7777, "cmd": "EngineCore worker", "pattern": "EngineCore"},
    ])
    sent: list[tuple[int, signal.Signals]] = []

    def _send(pid, sig):
        sent.append((pid, sig))
        return True

    monkeypatch.setattr(exe, "_send_signal", _send)
    # PID stays alive after TERM -> triggers SIGKILL.
    monkeypatch.setattr(exe, "_pid_alive", lambda pid: True)
    import inference_optimizer.orchestrator.action_executors.recover as recmod
    monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    # Both TERM and KILL were dispatched for the same PID.
    assert (7777, signal.SIGTERM) in sent
    assert (7777, signal.SIGKILL) in sent
    assert out["killed_pids"][0]["signal"] == "KILL"


# ---------------------------------------------------------------------------
# gpureset env-gate (the "soft_then_hard_gated" choice)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_env_gate_blocks_gpureset_by_default(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.delenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", raising=False)
    exe = RecoverExecutor()

    # Soft cleanup did NOT free VRAM — leak persists at mid-probe.
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _leaked_probe())
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])
    # Even if the env were set, this branch must NOT run; assert via spy.
    called: dict[str, bool] = {}

    def _spy():
        called["gpureset"] = True
        return {"returncode": 0}

    monkeypatch.setattr(exe, "_try_rocm_smi_gpureset", _spy)

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert out["state"] == "needs_review"
    assert out["error_class"] == "gpu_unhealthy_after_soft_cleanup"
    assert out["gpureset_attempted"] is False
    assert out["allow_reset_env"] is False
    assert "gpureset" not in called


@pytest.mark.asyncio
async def test_env_gate_allows_gpureset_and_recovers(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", "1")
    exe = RecoverExecutor()

    # pre: leaked, mid: still leaked, post: healthy (gpureset worked).
    probes = iter([
        _leaked_probe(), _leaked_probe(), _healthy_probe(),
    ])
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: next(probes))
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])

    monkeypatch.setattr(
        exe, "_try_rocm_smi_gpureset",
        lambda: {"returncode": 0, "stdout": "GPU reset OK", "stderr": ""},
    )

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert out["state"] == "succeeded"
    assert out["gpureset_attempted"] is True
    assert out["allow_reset_env"] is True
    assert out["gpureset_result"]["returncode"] == 0


@pytest.mark.asyncio
async def test_gpureset_returncode_nonzero_persists_failure(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", "1")
    exe = RecoverExecutor()
    # pre+mid+post all leaked: gpureset failed (returncode=1).
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _leaked_probe())
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])
    monkeypatch.setattr(
        exe, "_try_rocm_smi_gpureset",
        lambda: {
            "returncode": 1,
            "stdout": "",
            "stderr": "ERROR: requires root",
        },
    )

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert out["state"] == "needs_review"
    assert out["error_class"] == "gpu_unhealthy_after_gpureset"
    assert out["gpureset_attempted"] is True
    assert out["gpureset_result"]["returncode"] == 1
    # The dict was still persisted — operators need the audit trail.
    persisted = json.loads((workspace / "result.json").read_text())
    assert persisted["error_class"] == "gpu_unhealthy_after_gpureset"


@pytest.mark.asyncio
async def test_gpureset_skipped_when_mid_probe_healthy(tmp_path, monkeypatch):
    """If soft cleanup already cleared VRAM, gpureset must not run even
    when the env gate is open."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", "1")
    exe = RecoverExecutor()

    probes = iter([_leaked_probe(), _healthy_probe(), _healthy_probe()])
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: next(probes))
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])

    monkeypatch.setattr(
        exe, "_try_rocm_smi_gpureset",
        lambda: pytest.fail("gpureset should not run when mid-probe healthy"),
    )

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert out["state"] == "succeeded"
    assert out["gpureset_attempted"] is False


# ---------------------------------------------------------------------------
# CSV parser unit tests
# ---------------------------------------------------------------------------
def test_parse_rocm_smi_vram_csv_basic():
    text = (
        "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        "card0,206158430208,205678182400\n"
        "card1,206158430208,5242880\n"
    )
    out = RecoverExecutor._parse_rocm_smi_vram_csv(text)
    assert [g["gpu_id"] for g in out] == [0, 1]
    # 206158430208 / 1024^2 = 196608.0 MiB total
    assert out[0]["vram_total_mb"] == pytest.approx(196608.0, rel=1e-3)
    # card0 leaked: ~458 MiB free
    assert out[0]["free_mb"] == pytest.approx(458.0, abs=1.0)
    # card1 healthy: ~196603 MiB free
    assert out[1]["free_mb"] == pytest.approx(196603.0, abs=2.0)


def test_parse_rocm_smi_vram_csv_handles_garbage_and_blank_blocks():
    text = (
        "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        "card0,nope,205678182400\n"
        "\n"
        "noheader,row\n"
        "\n"
        "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        "card1,206158430208,5242880\n"
    )
    out = RecoverExecutor._parse_rocm_smi_vram_csv(text)
    # card0 had a non-numeric total → no total field, free_mb not computed
    ids = [g["gpu_id"] for g in out]
    assert ids == [0, 1]
    assert "free_mb" not in out[0]
    assert "free_mb" in out[1]


# ---------------------------------------------------------------------------
# Workspace handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_result_json_omitted_when_no_workspace(monkeypatch):
    """When SubAgentRunner does not pre-mkdir a workspace, the executor
    still returns a complete dict but skips the on-disk audit."""
    exe = RecoverExecutor()
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _healthy_probe())
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])

    out = await exe(_ctx(workspace=None, params={"force_gpu_cleanup": True}))
    assert out["state"] == "succeeded"
    assert "result_path" not in out
    assert "workspace" not in out


@pytest.mark.asyncio
async def test_result_json_has_expected_keys(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _healthy_probe())
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])
    await exe(_ctx(workspace, params={
        "reason": "test", "force_gpu_cleanup": True,
    }))
    persisted = json.loads((workspace / "result.json").read_text())
    expected_keys = {
        "state", "reason", "force_gpu_cleanup", "allow_reset_env",
        "killed_pids", "pre_free_mb_per_gpu", "mid_free_mb_per_gpu",
        "post_free_mb_per_gpu", "gpureset_attempted", "gpureset_result",
    }
    assert expected_keys.issubset(persisted.keys())


# ---------------------------------------------------------------------------
# Module-level callable
# ---------------------------------------------------------------------------
def test_module_callable_exists():
    from inference_optimizer.orchestrator.action_executors.recover import (
        RecoverExecutor as _Cls,
        recover_executor as _instance,
    )
    assert isinstance(_instance, _Cls)
    # Must match the singleton imported by ``cli.py``.
    assert recover_executor is _instance


# ---------------------------------------------------------------------------
# gpureset subprocess error paths (real ``_try_rocm_smi_gpureset``)
# ---------------------------------------------------------------------------
def test_try_rocm_smi_gpureset_handles_missing_binary(monkeypatch):
    exe = RecoverExecutor()
    import inference_optimizer.orchestrator.action_executors.recover as recmod
    monkeypatch.setattr(recmod.shutil, "which", lambda b: None)
    out = exe._try_rocm_smi_gpureset()
    assert out == {"error": "rocm-smi not on PATH"}


def test_try_rocm_smi_gpureset_handles_timeout(monkeypatch):
    exe = RecoverExecutor()
    import inference_optimizer.orchestrator.action_executors.recover as recmod
    monkeypatch.setattr(
        recmod.shutil, "which",
        lambda b: "/usr/bin/rocm-smi" if b == "rocm-smi" else None,
    )

    def _raise(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["rocm-smi"], timeout=30.0)

    monkeypatch.setattr(recmod.subprocess, "run", _raise)
    out = exe._try_rocm_smi_gpureset()
    assert out["error"] == "timeout"
    assert out["timeout_s"] == RecoverExecutor.GPURESET_TIMEOUT_S


def test_try_rocm_smi_gpureset_returns_stdout_stderr(monkeypatch):
    exe = RecoverExecutor()
    import inference_optimizer.orchestrator.action_executors.recover as recmod
    monkeypatch.setattr(
        recmod.shutil, "which",
        lambda b: "/usr/bin/rocm-smi" if b == "rocm-smi" else None,
    )

    class _Result:
        returncode = 1
        stdout = "tried gpureset"
        stderr = "permission denied"

    monkeypatch.setattr(recmod.subprocess, "run", lambda *a, **kw: _Result())
    out = exe._try_rocm_smi_gpureset()
    assert out == {
        "returncode": 1,
        "stdout": "tried gpureset",
        "stderr": "permission denied",
    }
