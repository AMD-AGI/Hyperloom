# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the baseline cold-start "warmup artifact"."""

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


# Cold/hot throughputs modelled on the MaralGPT-Maral-7B-alpha-1 CSV row.
_COLD_TPUT = 270.9
_HOT_TPUT = 4701.6


def _cold_then_hot_fake_run(captured: list | None = None):
    """Return a ``run_with_session_kill`` stand-in that emits a cold
    throughput on its first call and a hot throughput thereafter."""
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
    """The executor reports the HOT (second-round) throughput as the baseline."""
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
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert result.get("warmup_round_tput") == pytest.approx(_COLD_TPUT)
    assert "baseline_double_run_discarded_first" in result["nonfatal_warnings"]

    assert len(captured) == 2
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["enabled"] is True and measure_lc["enabled"] is True
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir)
    assert captured[0]["benchmark"]["envs"]["PORT"] == (
        captured[1]["benchmark"]["envs"]["PORT"]
    )
    assert captured[0]["benchmark"]["benchmark_script"] == "vllm_mi300x.sh"


def test_baseline_single_round_when_double_run_disabled(tmp_path, monkeypatch):
    """``INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN=0`` reverts to legacy single-round."""
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
    """A non-builtin benchmark script falls back to one round even with double-run on."""
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
    """A failed warmup round returns immediately and does NOT run a second round."""
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


def test_baseline_no_workspace_persists_stderr_to_file(tmp_path):
    """When Magpie exits nonzero before creating a benchmark_* workspace
    (no server.log ever written), the executor must persist the captured
    stderr to ``baseline_stderr.log`` so the failure leaves an on-disk
    artifact that survives the NFS clone / S3 archive."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"
    crash_text = "torch.OutOfMemoryError: HIP out of memory (workspace_buffer)"

    def fake_run(cmd, *args, **kwargs):
        # Nonzero exit, no benchmark_* workspace created.
        return subprocess.CompletedProcess(cmd, 1, "", crash_text)

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
    assert result["error_class"] == "subprocess_nonzero"
    log_path = result.get("stderr_log_path")
    assert log_path is not None, result
    saved = Path(log_path)
    assert saved.exists() and saved.name == "baseline_stderr.log"
    assert crash_text in saved.read_text(encoding="utf-8")


def test_baseline_classifies_vllm_engine_init_as_server_init_dead(
    tmp_path, monkeypatch,
):
    """#524: a vLLM engine-core bootstrap failure — server.log carries
    ``Engine core initialization failed`` while Magpie exits nonzero without a
    benchmark_* workspace — is classified ``server_init_dead`` with the
    server.log root cause surfaced in ``error`` (not a generic
    ``subprocess_nonzero`` from Magpie's wrapper noise)."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "server.log").write_text(
            "(APIServer pid=16160)   File '.../vllm/v1/engine/utils.py', "
            "line 1057, in wait_for_engine_startup\n"
            "(APIServer pid=16160) RuntimeError: Engine core initialization "
            "failed. See root cause above. Failed core proc(s): {}\n",
            encoding="utf-8",
        )
        # Magpie exits nonzero, no benchmark_* workspace produced.
        return subprocess.CompletedProcess(cmd, 1, "", "magpie wrapper noise")

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
    assert result["error_class"] == "server_init_dead", result
    assert "Engine core initialization failed" in result["error"]


def test_baseline_server_dead_returncode_classifies_server_init_dead(
    tmp_path, monkeypatch,
):
    """#524: when the liveness watchdog reaps a hung server
    (``SERVER_DEAD_RETURNCODE``), baseline classifies it ``server_init_dead``
    even when no server.log marker is independently visible."""
    from inference_optimizer.orchestrator.action_executors._subprocess_kill import (
        SERVER_DEAD_RETURNCODE,
    )

    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, SERVER_DEAD_RETURNCODE, "", "")

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
    assert result["error_class"] == "server_init_dead", result


def test_ensure_local_inferencex_noop_for_local_path(tmp_path, monkeypatch):
    """#523: a checkout already on a local filesystem is returned unchanged
    (no needless copy)."""
    from inference_optimizer.orchestrator.action_executors import baseline as bl

    src = tmp_path / "InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: False)

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_mirrors_network_path(tmp_path, monkeypatch):
    """#523: a checkout on a (simulated) network mount is mirrored to local
    disk and the returned path points at the local copy, not the original."""
    from inference_optimizer.orchestrator.action_executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched lib")
    (src / "utils").mkdir()
    (src / "utils" / "marker.txt").write_text("payload")

    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT", str(local_root),
    )

    dest = bl._ensure_local_inferencex(str(src))

    assert dest != str(src)
    assert str(local_root) in dest
    # Mirror is complete (benchmark_lib.sh + the rest of the patched tree).
    assert (Path(dest) / "benchmarks" / "benchmark_lib.sh").read_text() == (
        "# patched lib"
    )
    assert (Path(dest) / "utils" / "marker.txt").read_text() == "payload"


def test_ensure_local_inferencex_disabled_by_env(tmp_path, monkeypatch):
    """#523: the relocation can be opted out of via env even on a network
    mount (escape hatch for multi-node / shared-mount setups)."""
    from inference_optimizer.orchestrator.action_executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX", "1")

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_baseline_points_magpie_at_local_inferencex(tmp_path, monkeypatch):
    """#523 end-to-end (unit): when INFERENCEX_PATH is on a network mount,
    the Magpie subprocess env's MAGPIE_INFERENCEX_PATH is rewritten to the
    local mirror so Magpie's ``cd <inferencex>`` lands on stable local disk."""
    from inference_optimizer.orchestrator.action_executors import baseline as bl

    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"

    ix_src = tmp_path / "wekafs_InferenceX"
    (ix_src / "benchmarks").mkdir(parents=True)
    (ix_src / "benchmarks" / "benchmark_lib.sh").write_text("# patched")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv("INFERENCEX_PATH", str(ix_src))
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT", str(local_root),
    )

    seen: dict = {}

    def fake_run(cmd, *args, **kwargs):
        seen["env"] = kwargs.get("env")
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=_HOT_TPUT)
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
    magpie_ix = seen["env"]["MAGPIE_INFERENCEX_PATH"]
    assert magpie_ix != str(ix_src), seen["env"]
    assert str(local_root) in magpie_ix


def test_baseline_anchors_server_cwd_to_output_dir(tmp_path, monkeypatch):
    """The Magpie *parent* subprocess cwd is anchored to the stable task
    output_dir (never the default ``/tmp``) as defence-in-depth. (The actual
    #523 cuda-graph fix is the local-InferenceX mirror — see
    ``test_baseline_points_magpie_at_local_inferencex`` — because Magpie
    re-roots the server via ``cd <inferencex>``.)"""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    seen: dict = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=_HOT_TPUT)
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
    assert seen["cwd"] is not None
    assert seen["cwd"] != "/tmp"
    assert str(output_dir) in seen["cwd"]


def test_atom_engages_double_run_like_vllm_sglang(tmp_path):
    """AMD-AGI/Magpie#34 — atom baseline engages the lifecycle double-run like vllm/sglang."""
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
    """The overtime-kill anchor must reflect round 1's FULL run, not round 2's reuse time."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
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
    assert result["subprocess_runtime_sec"] >= 0.5
    assert "measure_round_runtime_sec" in result
    assert result["measure_round_runtime_sec"] < result["subprocess_runtime_sec"]


def test_teardown_lifecycle_server_removes_state_files(tmp_path):
    """The defensive teardown unlinks stale pid/meta files without raising."""
    executor = _executor(tmp_path / "base.yaml", tmp_path)
    _write_yaml(tmp_path / "base.yaml", framework="vllm")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    # PID that is essentially never alive.
    (pid_dir / "vllm_8888.pid").write_text("2147483646")
    (pid_dir / "vllm_8888.json").write_text("{}")

    executor._teardown_lifecycle_server(
        pid_dir=pid_dir, framework="vllm", port=8888,
    )

    assert not (pid_dir / "vllm_8888.pid").exists()
    assert not (pid_dir / "vllm_8888.json").exists()
