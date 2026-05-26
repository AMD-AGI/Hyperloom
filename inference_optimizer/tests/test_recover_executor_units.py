"""Unit tests for ``orchestrator.action_executors.recover.RecoverExecutor``.

The executor coordinates subprocess kills + rocm-smi probes. We stub the
two subprocess primitives so the tests can drive every branch (no force,
empty pgrep, partial recovery, gpureset success/timeout/launch error)
without touching any actual GPU or process.

Companion to ``test_recover_executor`` which exercises higher-level
behaviour through the full RunnerContext shape — this file targets the
fine-grained helper methods that today have no direct tests.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors.recover import (
    RecoverExecutor,
    _env_gate_allows_gpureset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(workspace: Path | None = None, **params: Any) -> SimpleNamespace:
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
        # card0 fully utilized, card1 mostly free.
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
        # Non-numeric `Total Used Memory` is skipped for that card, but
        # the card row is still recorded with whatever fields parsed.
        rows = ex._parse_rocm_smi_vram_csv(text)
        # Only card1 has both vram_total and vram_used parsed → free_mb present.
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
        # pgrep -a -f -- pattern output: "<pid> <cmdline>"
        scripted = {
            "sglang.launch_server": "12345 python sglang.launch_server --port 8000",
            "vllm.entrypoints": "12347 python vllm.entrypoints.api_server\n0 invalid",
            "EngineCore": "",  # no matches → rc=1 path
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
# _try_rocm_smi_gpureset
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
        big_stdout = "ok\n" * 1500  # > 2000 chars
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
        # ``Path(value)`` raises TypeError for arbitrary objects.
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

        # Force the call through a path that fails to mkdir.
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.recover.Path",
            _BadPath,
        )
        # Should not raise.
        RecoverExecutor()._write_result_json(target, {"state": "x"})
