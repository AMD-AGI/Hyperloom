"""Regression tests for the baseline cold-start "warmup artifact".

Root cause (see hyperloom_models_jun1.csv rows tagged ``warmup``):
``BaselineExecutor`` used to run Magpie exactly once and take that
throughput as ``baseline_tput``. The baseline action is the FIRST step of
the optimization flow, so the server is always freshly booted for the
model under test. The first benchmark window is contaminated by one-time
cold-start costs (kernel JIT / torch.compile first-compile, CUDA/HIP-graph
first-capture, KV-cache cold allocation, GPU not yet at boost clocks). The
client-side ``--num-warmups`` (hardcoded ``2 * CONC``) is far too small to
absorb them, so the measured baseline lands well below the real hot-state
value. Every later optimization round runs against an already-hot server,
so ``gain = final/baseline - 1`` is inflated into a fictitious
1600%-2160% "improvement".

Fix: run the benchmark TWICE against the SAME persistent server via
Magpie's ``server_lifecycle`` reuse protocol. Round 1 boots the server and
pays every cold cost; round 2 re-attaches to the now-hot server (client
only, no restart) and its throughput is the clean baseline.

These tests mock ``run_with_session_kill`` so the first (warmup) round
returns a cold throughput and the second (measured) round returns a hot
throughput, then assert the executor reports the HOT value as the
baseline and that each round's Magpie config carries the right
``server_lifecycle`` block.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors.baseline import (
    BaselineExecutor,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_warmup")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    # Keep TP clamp deterministic on CPU-only CI (no GPU visible).
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "8")


def _write_yaml(path: Path, *, framework: str = "vllm") -> None:
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "fp8",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 64, "ISL": 1024, "OSL": 1024},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float) -> Path:
    ws = slot / "benchmark_vllm_20260602_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "vllm",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": tput / 1024,
            "output_throughput": tput,
            "total_token_throughput": tput * 2,
            "completed_requests": 64,
            "duration_seconds": 25.0,
        },
        "latency": {
            "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
            "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
        },
    }))
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-warmup", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


# Cold first round / hot second round throughputs, modelled on the
# MaralGPT-Maral-7B-alpha-1 CSV row (baseline 270.9 -> final 4701.6).
_COLD_TPUT = 270.9
_HOT_TPUT = 4701.6


def _cold_then_hot_fake_run(captured: list | None = None):
    """Return a ``run_with_session_kill`` stand-in that emits a cold
    throughput on its first call and a hot throughput thereafter.

    Each call materializes a ``benchmark_*`` workspace under the
    ``--output-dir`` slot the executor passed, and (if ``captured`` is
    given) records the parsed ``--benchmark-config`` YAML so tests can
    assert the per-round ``server_lifecycle`` block.
    """
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        if captured is not None:
            cfg_idx = cmd.index("--benchmark-config")
            cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
            captured.append(cfg)
        tput = _COLD_TPUT if state["calls"] == 0 else _HOT_TPUT
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    return fake_run, state


def _executor(base: Path, tmp_path: Path) -> BaselineExecutor:
    return BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )


def test_baseline_discards_cold_first_round_via_lifecycle(tmp_path):
    """The executor reports the HOT (second-round) throughput as the
    baseline, runs two rounds, and each round carries a server_lifecycle
    block sharing one pid_dir so round 2 reuses round 1's hot server."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    # Reported baseline is the HOT value, not the cold one.
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    # Discarded cold value surfaced for auditability.
    assert result.get("warmup_round_tput") == pytest.approx(_COLD_TPUT)
    assert "baseline_double_run_discarded_first" in result["nonfatal_warnings"]

    # Both rounds requested server_lifecycle on a built-in vllm script.
    assert len(captured) == 2
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["enabled"] is True and measure_lc["enabled"] is True
    # Round 1 persists the server; round 2 tears it down.
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    # Shared pid_dir + port so round 2 re-attaches to round 1's server.
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir)
    assert captured[0]["benchmark"]["envs"]["PORT"] == (
        captured[1]["benchmark"]["envs"]["PORT"]
    )
    assert captured[0]["benchmark"]["benchmark_script"] == "vllm_mi300x.sh"


def test_baseline_single_round_when_double_run_disabled(tmp_path, monkeypatch):
    """``INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN=0`` reverts to the legacy
    single-round behaviour (which reproduces the artifact) — an operator
    escape hatch. No server_lifecycle is injected."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "warmup_round_tput" not in result
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_single_round_when_script_not_builtin(tmp_path):
    """When the resolved benchmark script is NOT a Magpie built-in
    (server_lifecycle unsupported), the guard falls back to one round
    even with double-run enabled."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    # A model-specific InferenceX script (not in MAGPIE_BUILTIN_SCRIPTS).
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
        "benchmark_script": "dsr1_fp8_mi300x.sh",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_warmup_round_failure_short_circuits(tmp_path):
    """If the warmup round fails (server never came up), the executor
    returns the failure immediately and does NOT run a second round. The
    finally teardown runs but is a no-op (no pid file)."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        state["calls"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "boom: server crashed")

    executor = _executor(base, tmp_path)
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert state["calls"] == 1
    assert "baseline_warmup_round_failed" in result.get("nonfatal_warnings", [])


def test_atom_engages_double_run_like_vllm_sglang(tmp_path):
    """Atom is a Magpie built-in (phase-aware) framework per
    AMD-AGI/Magpie#34, so an atom baseline engages the lifecycle
    double-run just like vllm/sglang."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="atom")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert captured[0]["benchmark"]["benchmark_script"] == "atom_mi300x.sh"
    assert captured[0]["benchmark"]["server_lifecycle"]["enabled"] is True


def test_double_run_runtime_anchor_is_full_warmup_round(tmp_path):
    """The overtime-kill anchor (``subprocess_runtime_sec`` ->
    ``baseline_runtime_sec``) must reflect round 1's FULL server-boot +
    client wall-clock, NOT round 2's client-only reuse time.

    Otherwise the ExploreExecutor's soft-kill deadline
    (``baseline_runtime_sec * kill_ratio``) is anchored to the tiny
    client-only round-2 time and would kill normal full-run variants as
    KILLED_OVERTIME. Round 1 (full run) sleeps longer than round 2
    (reuse) here so the assertion is deterministic.
    """
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # Round 1 (full boot + client) is slow; round 2 (client-only
        # reuse) is fast. The executor measures wall-clock around this
        # call, so distinct sleeps make the anchor check deterministic.
        if state["calls"] == 0:
            time.sleep(0.6)
            tput = _COLD_TPUT
        else:
            tput = _HOT_TPUT
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path)
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    # Anchor reflects the slow round-1 full run...
    assert result["subprocess_runtime_sec"] >= 0.5
    # ...and round 2's client-only time is kept separately and is smaller.
    assert "measure_round_runtime_sec" in result
    assert result["measure_round_runtime_sec"] < result["subprocess_runtime_sec"]


def test_teardown_lifecycle_server_removes_state_files(tmp_path):
    """The defensive teardown unlinks stale pid/meta files (no live
    server in the unit env) without raising."""
    executor = _executor(tmp_path / "base.yaml", tmp_path)
    _write_yaml(tmp_path / "base.yaml", framework="vllm")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    # No real process: use a PID that is essentially never alive.
    (pid_dir / "vllm_8888.pid").write_text("2147483646")
    (pid_dir / "vllm_8888.json").write_text("{}")

    executor._teardown_lifecycle_server(
        pid_dir=pid_dir, framework="vllm", port=8888,
    )

    assert not (pid_dir / "vllm_8888.pid").exists()
    assert not (pid_dir / "vllm_8888.json").exists()
