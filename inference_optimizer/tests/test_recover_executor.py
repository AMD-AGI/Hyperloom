# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :class:`RecoverExecutor`.

Stubs the three side effects (rocm-smi probes, pgrep/kill, optional gpureset)
and asserts the result-dict shape plus the on-disk ``result.json`` audit trail.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors.recover import (
    RecoverExecutor,
    _env_gate_allows_gpureset,
    recover_executor,
)
from inference_optimizer.orchestrator.task_registry import Task


# RunnerContext stand-in (mirrors test_target_analysis_executor.py)
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
    """``_probe_gpu_free_mb`` return value with all GPUs healthy."""
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


# No-op path — GPU already healthy
@pytest.mark.asyncio
async def test_no_stale_owners_returns_succeeded(tmp_path, monkeypatch):
    """Healthy GPUs + no stale owners -> state=succeeded, no kills."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _healthy_probe())
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
    assert out["mid_free_mb_per_gpu"] == out["post_free_mb_per_gpu"]
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


# Happy path — kills stale owners and GPUs recover
@pytest.mark.asyncio
async def test_kills_stale_owners_and_recovers(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

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
    monkeypatch.setattr(exe, "_pid_alive", lambda pid: False)
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
    monkeypatch.setattr(exe, "_pid_alive", lambda pid: True)
    import inference_optimizer.orchestrator.action_executors.recover as recmod
    monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert (7777, signal.SIGTERM) in sent
    assert (7777, signal.SIGKILL) in sent
    assert out["killed_pids"][0]["signal"] == "KILL"


# gpureset env-gate (the "soft_then_hard_gated" choice)
@pytest.mark.asyncio
async def test_env_gate_blocks_gpureset_by_default(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.delenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", raising=False)
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", lambda: _leaked_probe())
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])
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
    persisted = json.loads((workspace / "result.json").read_text())
    assert persisted["error_class"] == "gpu_unhealthy_after_gpureset"


@pytest.mark.asyncio
async def test_gpureset_skipped_when_mid_probe_healthy(tmp_path, monkeypatch):
    """If soft cleanup cleared VRAM, gpureset must not run even with the env gate open."""
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


# CSV parser unit tests
def test_parse_rocm_smi_vram_csv_basic():
    text = (
        "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        "card0,206158430208,205678182400\n"
        "card1,206158430208,5242880\n"
    )
    out = RecoverExecutor._parse_rocm_smi_vram_csv(text)
    assert [g["gpu_id"] for g in out] == [0, 1]
    assert out[0]["vram_total_mb"] == pytest.approx(196608.0, rel=1e-3)
    assert out[0]["free_mb"] == pytest.approx(458.0, abs=1.0)
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
    ids = [g["gpu_id"] for g in out]
    assert ids == [0, 1]
    assert "free_mb" not in out[0]
    assert "free_mb" in out[1]


# Workspace handling
@pytest.mark.asyncio
async def test_result_json_omitted_when_no_workspace(monkeypatch):
    """Without a pre-made workspace the executor returns a complete dict but skips the on-disk audit."""
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


# Module-level callable
def test_module_callable_exists():
    from inference_optimizer.orchestrator.action_executors.recover import (
        RecoverExecutor as _Cls,
        recover_executor as _instance,
    )
    assert isinstance(_instance, _Cls)
    assert recover_executor is _instance


# gpureset subprocess error paths (real ``_try_rocm_smi_gpureset``)
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


# ===========================================================================
# Fine-grained helper-method unit tests (formerly test_recover_executor_units.py)
# ===========================================================================


def _unit_ctx(workspace: Path | None = None, **params: Any) -> SimpleNamespace:
    extra = {"workspace": str(workspace)} if workspace is not None else {}
    return SimpleNamespace(
        task=SimpleNamespace(task_id="recover-t1", params=params),
        extra=extra,
    )


def _csv(*rows: tuple[int, int]) -> str:
    """Build a synthetic rocm-smi --showmeminfo vram --csv payload."""
    parts = ["device,VRAM Total Memory (B),VRAM Total Used Memory (B)"]
    for gpu_id, used_bytes in rows:
        total = 192 * 1024 * 1024 * 1024  # 192 GiB
        parts.append(f"card{gpu_id},{total},{used_bytes}")
    return "\n".join(parts) + "\n"


class _ProcResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# env gate
# ---------------------------------------------------------------------------

class TestEnvGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", raising=False)
        assert _env_gate_allows_gpureset() is False

    def test_explicit_one_enables(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_RECOVER_ALLOW_GPU_RESET", "1")
        assert _env_gate_allows_gpureset() is True


# ---------------------------------------------------------------------------
# _parse_rocm_smi_vram_csv
# ---------------------------------------------------------------------------

class TestParseRocmSmiVramCsv:
    def test_parses_two_cards(self):
        ex = RecoverExecutor()
        total = 192 * 1024 * 1024 * 1024
        text = (
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            f"card0,{total},{total}\n"
            f"card1,{total},{int(total * 0.1)}\n"
        )
        rows = ex._parse_rocm_smi_vram_csv(text)
        assert [r["gpu_id"] for r in rows] == [0, 1]
        assert rows[0]["free_mb"] == pytest.approx(0.0, abs=1.0)
        assert rows[1]["free_mb"] > rows[0]["free_mb"]

    def test_ignores_non_card_lines(self):
        ex = RecoverExecutor()
        text = (
            "header noise\n"
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            "card0,123,not-a-number\n"
            "card1,500,200\n"
        )
        rows = ex._parse_rocm_smi_vram_csv(text)
        rows_with_free = [r for r in rows if "free_mb" in r]
        assert any(r["gpu_id"] == 1 for r in rows_with_free)

    def test_empty_input_returns_empty(self):
        assert RecoverExecutor()._parse_rocm_smi_vram_csv("") == []


# ---------------------------------------------------------------------------
# _all_recovered
# ---------------------------------------------------------------------------

class TestAllRecovered:
    def test_empty_treated_as_unhealthy(self):
        assert RecoverExecutor()._all_recovered([]) is False

    def test_all_healthy(self):
        ex = RecoverExecutor()
        gpus = [{"gpu_id": 0, "free_mb": 1024.0}, {"gpu_id": 1, "free_mb": 2048.0}]
        assert ex._all_recovered(gpus) is True

    def test_partial_unhealthy(self):
        ex = RecoverExecutor()
        gpus = [{"gpu_id": 0, "free_mb": 100.0}, {"gpu_id": 1, "free_mb": 2048.0}]
        assert ex._all_recovered(gpus) is False

    def test_non_numeric_free_mb_treated_as_unhealthy(self):
        ex = RecoverExecutor()
        gpus = [{"gpu_id": 0}, {"gpu_id": 1, "free_mb": 9999.0}]
        assert ex._all_recovered(gpus) is False


# ---------------------------------------------------------------------------
# probe + signal helpers
# ---------------------------------------------------------------------------

class TestProbeGpuFreeMb:
    def test_returns_empty_when_rocm_smi_missing(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: None,
        )
        assert RecoverExecutor()._probe_gpu_free_mb() == []

    def test_returns_empty_on_nonzero_rc(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            lambda *a, **k: _ProcResult(returncode=1, stderr="boom"),
        )
        assert RecoverExecutor()._probe_gpu_free_mb() == []

    def test_returns_empty_on_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd=a, timeout=1)

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            boom,
        )
        assert RecoverExecutor()._probe_gpu_free_mb() == []

    def test_returns_parsed_rows_on_success(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            lambda *a, **k: _ProcResult(returncode=0, stdout=_csv((0, 0), (1, 100))),
        )
        rows = RecoverExecutor()._probe_gpu_free_mb()
        assert [r["gpu_id"] for r in rows] == [0, 1]


class TestSendAndAlive:
    def test_send_signal_handles_missing_pid(self, monkeypatch):
        ex = RecoverExecutor()

        def raise_lookup(pid, sig):
            raise ProcessLookupError()

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.os.kill",
            raise_lookup,
        )
        assert ex._send_signal(12345, signal.SIGTERM) is False

    def test_send_signal_handles_permission_error(self, monkeypatch):
        ex = RecoverExecutor()

        def raise_perm(pid, sig):
            raise PermissionError("not allowed")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.os.kill",
            raise_perm,
        )
        assert ex._send_signal(12345, signal.SIGTERM) is False

    def test_send_signal_success(self, monkeypatch):
        ex = RecoverExecutor()
        sent = []
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.os.kill",
            lambda pid, sig: sent.append((pid, sig)),
        )
        assert ex._send_signal(1, signal.SIGTERM) is True
        assert sent == [(1, signal.SIGTERM)]

    def test_pid_alive_handles_missing_pid(self, monkeypatch):
        def raise_lookup(pid, sig):
            raise ProcessLookupError()

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.os.kill",
            raise_lookup,
        )
        assert RecoverExecutor._pid_alive(99999) is False

    def test_pid_alive_handles_permission_error(self, monkeypatch):
        def raise_perm(pid, sig):
            raise PermissionError()

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.os.kill",
            raise_perm,
        )
        assert RecoverExecutor._pid_alive(1) is False


# ---------------------------------------------------------------------------
# _discover_stale_pids
# ---------------------------------------------------------------------------

class TestDiscoverStalePids:
    def test_no_pgrep_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: None,
        )
        assert RecoverExecutor()._discover_stale_pids() == []

    def test_parses_pgrep_output(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/pgrep",
        )
        scripted = {
            "sglang.launch_server": "12345 python sglang.launch_server --port 8000",
            "vllm.entrypoints": "12347 python vllm.entrypoints.api_server\n0 invalid",
            "EngineCore": "",
        }

        def fake_run(cmd, *args, **kwargs):
            pattern = cmd[-1]
            output = scripted.get(pattern, "")
            rc = 0 if output else 1
            return _ProcResult(returncode=rc, stdout=output)

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            fake_run,
        )
        ex = RecoverExecutor()
        ex.OWNER_PATTERNS = ("sglang.launch_server", "vllm.entrypoints", "EngineCore")
        out = ex._discover_stale_pids()
        pids = sorted(o["pid"] for o in out)
        assert pids == [12345, 12347]

    def test_skips_own_pid(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/pgrep",
        )
        own_pid = os.getpid()
        output = f"{own_pid} sglang.launch_server self\n12000 sglang.launch_server other"
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            lambda *a, **k: _ProcResult(returncode=0, stdout=output),
        )
        ex = RecoverExecutor()
        ex.OWNER_PATTERNS = ("sglang.launch_server",)
        out = ex._discover_stale_pids()
        assert [o["pid"] for o in out] == [12000]


# ---------------------------------------------------------------------------
# _try_rocm_smi_gpureset (unit-level coverage of subprocess branches)
# ---------------------------------------------------------------------------

class TestTryRocmSmiGpureset:
    def test_no_rocm_smi(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: None,
        )
        out = RecoverExecutor()._try_rocm_smi_gpureset()
        assert out == {"error": "rocm-smi not on PATH"}

    def test_success_truncates_outputs(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )
        big_stdout = "ok\n" * 1500
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            lambda *a, **k: _ProcResult(returncode=0, stdout=big_stdout, stderr=""),
        )
        out = RecoverExecutor()._try_rocm_smi_gpureset()
        assert out["returncode"] == 0
        assert len(out["stdout"]) <= 2000

    def test_timeout_reports_error(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(
                cmd=a, timeout=k["timeout"], stderr=b"timeout-stderr",
            )

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            boom,
        )
        out = RecoverExecutor()._try_rocm_smi_gpureset()
        assert out["returncode"] is None
        assert out["error"] == "timeout"
        assert out["timeout_s"] == RecoverExecutor.GPURESET_TIMEOUT_S

    def test_launch_failure_returns_error_dict(self, monkeypatch):
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )

        def boom(*a, **k):
            raise FileNotFoundError("missing rocm-smi binary")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.subprocess.run",
            boom,
        )
        out = RecoverExecutor()._try_rocm_smi_gpureset()
        assert "launch_failed" in out["error"]


# ---------------------------------------------------------------------------
# _workspace_dir + _write_result_json
# ---------------------------------------------------------------------------

class TestWorkspaceHelpers:
    def test_workspace_dir_returns_none_when_missing(self):
        assert RecoverExecutor()._workspace_dir(SimpleNamespace(extra=None)) is None

    def test_workspace_dir_invalid_type_returns_none(self):
        ctx = SimpleNamespace(extra={"workspace": object()})
        assert RecoverExecutor()._workspace_dir(ctx) is None

    def test_write_result_json_creates_file(self, tmp_path):
        target = tmp_path / "ws"
        RecoverExecutor()._write_result_json(target, {"state": "ok"})
        assert (target / "result.json").is_file()

    def test_write_result_json_swallows_oserror(self, tmp_path, monkeypatch):
        target = tmp_path / "broken"

        class _BadPath(Path):
            def mkdir(self, *args, **kwargs):
                raise OSError("readonly")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.Path",
            _BadPath,
        )
        RecoverExecutor()._write_result_json(target, {"state": "x"})
