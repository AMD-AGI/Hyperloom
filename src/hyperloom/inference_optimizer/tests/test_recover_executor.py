# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :class:`RecoverExecutor`.

Stubs the two side effects (rocm-smi probes, pgrep/kill) and asserts the
result-dict shape plus the on-disk ``result.json`` audit trail.
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

from hyperloom.orchestrator.actions.executors.recover import (
    RecoverExecutor,
    recover_executor,
)
from hyperloom.orchestrator.state.task_registry import Task


# RunnerContext stand-in.
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


@pytest.mark.asyncio
async def test_no_stale_owners_returns_succeeded(tmp_path, monkeypatch):
    """Healthy GPUs + no stale owners -> state=succeeded, no kills."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", _healthy_probe)
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])

    out = await exe(
        _ctx(
            workspace,
            params={
                "reason": "smoke",
                "force_gpu_cleanup": True,
            },
        )
    )

    assert out["state"] == "succeeded"
    assert out["killed_pids"] == []
    assert out["force_gpu_cleanup"] is True
    assert (workspace / "result.json").exists()


@pytest.mark.asyncio
async def test_force_cleanup_false_skips_kill_stage(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", _healthy_probe)
    calls: list[str] = []

    def _should_not_be_called():
        calls.append("kill")
        return []

    monkeypatch.setattr(exe, "_kill_stale_owners", _should_not_be_called)

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": False}))
    assert out["state"] == "succeeded"
    assert calls == []
    assert out["killed_pids"] == []


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
    monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

    out = await exe(
        _ctx(
            workspace,
            params={
                "reason": "gpu_memory_leaked",
                "force_gpu_cleanup": True,
            },
        )
    )

    assert out["state"] == "succeeded"
    assert {pid for pid, _ in sent_signals} == {4242, 4243}
    assert all(sig == signal.SIGTERM for _, sig in sent_signals)
    assert [e["signal"] for e in out["killed_pids"]] == ["TERM", "TERM"]

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
    monkeypatch.setattr(
        exe,
        "_discover_stale_pids",
        lambda: [
            {"pid": 7777, "cmd": "EngineCore worker", "pattern": "EngineCore"},
        ],
    )
    sent: list[tuple[int, signal.Signals]] = []

    def _send(pid, sig):
        sent.append((pid, sig))
        return True

    monkeypatch.setattr(exe, "_send_signal", _send)
    monkeypatch.setattr(exe, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(exe, "_pid_cmdline", lambda pid: "EngineCore worker")
    monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert (7777, signal.SIGTERM) in sent
    assert (7777, signal.SIGKILL) in sent
    assert out["killed_pids"][0]["signal"] == "KILL"


def test_sigkill_skips_pid_reused_after_sigterm(monkeypatch):
    """The KILL phase revalidates ownership after its grace period."""
    exe = RecoverExecutor()
    monkeypatch.setattr(
        exe,
        "_discover_stale_pids",
        lambda: [{"pid": 7777, "cmd": "EngineCore worker", "pattern": "EngineCore"}],
    )
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(exe, "_send_signal", lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr(exe, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(exe, "_pid_cmdline", lambda _pid: "python unrelated.py")
    monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

    out = exe._kill_stale_owners()

    assert sent == [(7777, signal.SIGTERM)]
    assert out[0]["signal"] == "TERM"


# Soft-cleanup failure path: leaked VRAM after kills -> needs_review.
@pytest.mark.asyncio
async def test_soft_cleanup_leaves_vram_leaked_needs_review(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    exe = RecoverExecutor()

    monkeypatch.setattr(exe, "_probe_gpu_free_mb", _leaked_probe)
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])

    out = await exe(_ctx(workspace, params={"force_gpu_cleanup": True}))
    assert out["state"] == "needs_review"
    assert out["error_class"] == "gpu_unhealthy_after_soft_cleanup"
    persisted = json.loads((workspace / "result.json").read_text())
    assert persisted["error_class"] == "gpu_unhealthy_after_soft_cleanup"


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


@pytest.mark.asyncio
async def test_result_json_omitted_when_no_workspace(monkeypatch):
    """Without a pre-made workspace the executor returns a complete dict but skips the on-disk audit."""
    exe = RecoverExecutor()
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", _healthy_probe)
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
    monkeypatch.setattr(exe, "_probe_gpu_free_mb", _healthy_probe)
    monkeypatch.setattr(exe, "_discover_stale_pids", lambda: [])
    await exe(
        _ctx(
            workspace,
            params={
                "reason": "test",
                "force_gpu_cleanup": True,
            },
        )
    )
    persisted = json.loads((workspace / "result.json").read_text())
    expected_keys = {
        "state",
        "reason",
        "force_gpu_cleanup",
        "killed_pids",
        "pre_free_mb_per_gpu",
        "mid_free_mb_per_gpu",
    }
    assert expected_keys.issubset(persisted.keys())


def test_module_callable_exists():
    from hyperloom.orchestrator.actions.executors.recover import (
        RecoverExecutor as _Cls,
        recover_executor as _instance,
    )

    assert isinstance(_instance, _Cls)
    assert recover_executor is _instance


# ===========================================================================
# Fine-grained helper-method unit tests
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


class TestProbeGpuFreeMb:
    def test_returns_empty_when_rocm_smi_missing(self, monkeypatch):
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.shutil.which",
            lambda name: None,
        )
        assert RecoverExecutor()._probe_gpu_free_mb() == []

    def test_returns_empty_on_nonzero_rc(self, monkeypatch):
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.subprocess.run",
            lambda *a, **k: _ProcResult(returncode=1, stderr="boom"),
        )
        assert RecoverExecutor()._probe_gpu_free_mb() == []

    def test_returns_empty_on_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )

        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd=a, timeout=1)

        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.subprocess.run",
            boom,
        )
        assert RecoverExecutor()._probe_gpu_free_mb() == []

    def test_returns_parsed_rows_on_success(self, monkeypatch):
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.shutil.which",
            lambda name: "/usr/bin/rocm-smi",
        )
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.subprocess.run",
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
            "hyperloom.orchestrator.actions.executors.recover.os.kill",
            raise_lookup,
        )
        assert ex._send_signal(12345, signal.SIGTERM) is False

    def test_send_signal_handles_permission_error(self, monkeypatch):
        ex = RecoverExecutor()

        def raise_perm(pid, sig):
            raise PermissionError("not allowed")

        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.os.kill",
            raise_perm,
        )
        assert ex._send_signal(12345, signal.SIGTERM) is False

    def test_send_signal_success(self, monkeypatch):
        ex = RecoverExecutor()
        sent = []
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.os.kill",
            lambda pid, sig: sent.append((pid, sig)),
        )
        assert ex._send_signal(1, signal.SIGTERM) is True
        assert sent == [(1, signal.SIGTERM)]

    def test_pid_alive_handles_missing_pid(self, monkeypatch):
        def raise_lookup(pid, sig):
            raise ProcessLookupError()

        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.os.kill",
            raise_lookup,
        )
        assert RecoverExecutor._pid_alive(99999) is False

    def test_pid_alive_handles_permission_error(self, monkeypatch):
        def raise_perm(pid, sig):
            raise PermissionError()

        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.recover.os.kill",
            raise_perm,
        )
        assert RecoverExecutor._pid_alive(1) is False


class TestDiscoverStalePids:
    def test_no_session_dir_returns_empty(self):
        assert RecoverExecutor()._discover_stale_pids() == []

    def test_parses_session_pidfiles(self, tmp_path):
        runs = tmp_path / "runs" / "baseline"
        runs.mkdir(parents=True)
        (runs / "sglang_8000.pid").write_text("12345 12345\n", encoding="utf-8")
        (runs / "vllm_8001.pid").write_text("12347\n", encoding="utf-8")
        (runs / "empty.pid").write_text("\n", encoding="utf-8")
        (runs / "bad.pid").write_text("notapid rest\n", encoding="utf-8")
        ex = RecoverExecutor()
        ex._active_session_dir = tmp_path
        out = ex._discover_stale_pids()
        pids = sorted(o["pid"] for o in out)
        assert pids == [12345, 12347]
        assert all(o["pattern"] == "session_pidfile" for o in out)

    def test_skips_own_pid(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        own_pid = os.getpid()
        (runs / "self.pid").write_text(f"{own_pid}\n", encoding="utf-8")
        (runs / "other.pid").write_text("12000\n", encoding="utf-8")
        ex = RecoverExecutor()
        ex._active_session_dir = tmp_path
        out = ex._discover_stale_pids()
        assert [o["pid"] for o in out] == [12000]


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
            "hyperloom.orchestrator.actions.executors.recover.Path",
            _BadPath,
        )
        RecoverExecutor()._write_result_json(target, {"state": "x"})


# ===========================================================================
# Additional focused branch coverage
# ===========================================================================


from hyperloom.orchestrator.actions.executors import recover as recmod  # noqa: E402


class TestWriteResultJsonOSError:
    def test_oserror_on_write_text_is_swallowed(self, tmp_path, monkeypatch):
        """mkdir succeeds but write_text raises OSError -> logged and swallowed."""
        target = tmp_path / "ws"

        real_write_text = Path.write_text

        def _boom(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)
        # Must not raise.
        RecoverExecutor()._write_result_json(target, {"state": "x"})
        # Directory was created before the failing write.
        assert target.is_dir()
        monkeypatch.setattr(Path, "write_text", real_write_text)


class TestIsMultiNodeSandbox:
    def test_true_when_is_multi_node_true(self, monkeypatch):
        import hyperloom.orchestrator.actions.executors._multi_node_env as mne

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        assert recmod._is_multi_node_sandbox() is True

    def test_false_when_is_multi_node_false(self, monkeypatch):
        import hyperloom.orchestrator.actions.executors._multi_node_env as mne

        monkeypatch.setattr(mne, "is_multi_node", lambda: False)
        assert recmod._is_multi_node_sandbox() is False

    def test_exception_defaults_to_single_node(self, monkeypatch):
        """A failure in is_multi_node() is swallowed -> single-node."""
        import hyperloom.orchestrator.actions.executors._multi_node_env as mne

        def _boom():
            raise RuntimeError("cannot determine node topology")

        monkeypatch.setattr(mne, "is_multi_node", _boom)
        assert recmod._is_multi_node_sandbox() is False


class TestMultiNodeShortCircuit:
    @pytest.mark.asyncio
    async def test_cpu_only_sandbox_short_circuits_to_success(self, tmp_path, monkeypatch):
        """Multi-node sandbox skips local probes and returns cpu_only success."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(recmod, "_is_multi_node_sandbox", lambda: True)
        exe = RecoverExecutor()

        # Guard: none of the local GPU boundaries may be touched.
        monkeypatch.setattr(
            exe,
            "_probe_gpu_free_mb",
            lambda: pytest.fail("must not probe GPUs in multi-node sandbox"),
        )
        monkeypatch.setattr(
            exe,
            "_discover_stale_pids",
            lambda: pytest.fail("must not kill in multi-node sandbox"),
        )

        out = await exe(
            _ctx(
                workspace,
                params={"reason": "gpu_memory_leaked", "force_gpu_cleanup": True},
            )
        )
        assert out["state"] == "succeeded"
        assert out["cpu_only_sandbox"] is True
        assert out["killed_pids"] == []
        assert out["pre_free_mb_per_gpu"] == []
        assert out["mid_free_mb_per_gpu"] == []
        assert out["reason"] == "gpu_memory_leaked"
        assert out["force_gpu_cleanup"] is True
        assert out["workspace"] == str(workspace)
        assert out["result_path"] == str(workspace / "result.json")
        persisted = json.loads((workspace / "result.json").read_text())
        assert persisted["cpu_only_sandbox"] is True
        assert persisted["state"] == "succeeded"

    @pytest.mark.asyncio
    async def test_cpu_only_sandbox_without_workspace(self, monkeypatch):
        """Multi-node short-circuit with no workspace omits the on-disk audit keys."""
        monkeypatch.setattr(recmod, "_is_multi_node_sandbox", lambda: True)
        exe = RecoverExecutor()
        out = await exe(_ctx(workspace=None, params={"force_gpu_cleanup": False}))
        assert out["state"] == "succeeded"
        assert out["cpu_only_sandbox"] is True
        assert "workspace" not in out
        assert "result_path" not in out


class TestParseRocmSmiCsvEdgeCases:
    def test_blank_cells_line_is_skipped(self):
        """A line that splits to all-empty cells doesn't crash."""
        text = "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n,,\ncard0,206158430208,5242880\n"
        rows = RecoverExecutor._parse_rocm_smi_vram_csv(text)
        assert [r["gpu_id"] for r in rows] == [0]

    def test_card_with_non_integer_index_is_skipped(self):
        """``cardX`` where X is not an int is skipped."""
        text = (
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            "cardX,206158430208,5242880\n"
            "card2,206158430208,5242880\n"
        )
        rows = RecoverExecutor._parse_rocm_smi_vram_csv(text)
        assert [r["gpu_id"] for r in rows] == [2]


class TestKillStaleOwnersNoneSignalled:
    def test_recorded_process_group_is_signalled(self, monkeypatch):
        """A lifecycle pidfile entry reaps the full server process group."""
        exe = RecoverExecutor()
        monkeypatch.setattr(
            exe,
            "_discover_stale_pids",
            lambda: [
                {
                    "pid": 111,
                    "pgid": 222,
                    "cmd": "python -m vllm.entrypoints.openai.api_server",
                    "pattern": "session_pidfile",
                }
            ],
        )
        sent: list[tuple[int, signal.Signals]] = []
        monkeypatch.setattr(exe, "_send_group_signal", lambda pgid, sig: sent.append((pgid, sig)) or True)
        monkeypatch.setattr(exe, "_process_group_alive", lambda _pgid: False)
        monkeypatch.setattr(recmod.time, "sleep", lambda _s: None)

        out = exe._kill_stale_owners()

        assert sent == [(222, signal.SIGTERM)]
        assert out[0]["signal"] == "TERM"

    def test_unrecognized_pidfile_owner_is_not_signalled(self, monkeypatch):
        """A recycled PID with an unrelated cmdline is ignored."""
        exe = RecoverExecutor()
        monkeypatch.setattr(
            exe,
            "_discover_stale_pids",
            lambda: [{"pid": 111, "cmd": "python unrelated.py", "pattern": "session_pidfile"}],
        )
        monkeypatch.setattr(
            exe,
            "_send_signal",
            lambda _pid, _sig: pytest.fail("unrecognized owner must not be signalled"),
        )

        assert exe._kill_stale_owners() == []

    def test_all_sigterm_fail_returns_empty(self, monkeypatch):
        """If every SIGTERM fails to deliver, no wait/KILL and returns []."""
        exe = RecoverExecutor()
        monkeypatch.setattr(
            exe,
            "_discover_stale_pids",
            lambda: [{"pid": 111, "cmd": "EngineCore", "pattern": "EngineCore"}],
        )
        # Every signal delivery fails (dead / forbidden).
        monkeypatch.setattr(exe, "_send_signal", lambda pid, sig: False)
        # Sleep + alive-check must never run when nothing was TERMed.
        monkeypatch.setattr(
            recmod.time,
            "sleep",
            lambda _s: pytest.fail("must not wait when nothing was signalled"),
        )
        monkeypatch.setattr(
            exe,
            "_pid_alive",
            lambda pid: pytest.fail("must not re-check when nothing was signalled"),
        )
        assert exe._kill_stale_owners() == []


class TestDiscoverStalePidsBranches:
    def test_missing_runs_dir_returns_empty(self, tmp_path):
        ex = RecoverExecutor()
        ex._active_session_dir = tmp_path
        assert ex._discover_stale_pids() == []

    def test_duplicate_pid_is_deduped(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "a.pid").write_text("5001 5001\n", encoding="utf-8")
        (runs / "b.pid").write_text("5001\n", encoding="utf-8")
        ex = RecoverExecutor()
        ex._active_session_dir = tmp_path
        found = ex._discover_stale_pids()
        assert [o["pid"] for o in found] == [5001]

    def test_dead_pidfile_is_removed_during_cleanup(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        pid_file = runs / "dead.pid"
        pid_file.write_text("2147483646 2147483646\n", encoding="utf-8")
        ex = RecoverExecutor()
        ex._active_session_dir = tmp_path

        assert ex._kill_stale_owners() == []
        assert not pid_file.exists()


class TestPidAliveTrue:
    def test_pid_alive_returns_true_when_kill_succeeds(self, monkeypatch):
        """os.kill(pid, 0) not raising -> process alive."""
        monkeypatch.setattr(recmod.os, "kill", lambda pid, sig: None)
        assert RecoverExecutor._pid_alive(1) is True
