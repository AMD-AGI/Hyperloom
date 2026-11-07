# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the baseline cold-start \"warmup artifact\"."""

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

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    stamp_server_ready,
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
            "model": "/path/models/Qwen-Qwen3-8B",
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


def _launch_kwargs(kwargs: dict) -> dict:
    """A launch's keywords with its RoundWatch gates spelled out; a Ray round arrives already flat."""
    watch = kwargs.get("watch")
    return {**kwargs, **vars(watch)} if watch is not None else dict(kwargs)


def _fake_workspace(slot: Path, *, tput: float) -> Path:
    ws = slot / "benchmark_vllm_20260602_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "/path/models/Qwen-Qwen3-8B",
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
            }
        )
    )
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-warmup", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


_COLD_TPUT = 270.9
_HOT_TPUT = 4701.6


def _cold_then_hot_fake_run(
    captured: list | None = None,
    *,
    launches: list | None = None,
    clock: _AClockOnlyThePassesMove | None = None,
    boot_sec: float = 0.0,
    benchmark_sec: float = 0.0,
):
    """Return a ``run_with_session_kill`` stand-in that emits a cold throughput on its first call and a hot throughput thereafter."""
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        kwargs = _launch_kwargs(kwargs)
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        if launches is not None:
            launches.append(kwargs)
        if captured is not None:
            cfg_idx = cmd.index("--benchmark-config")
            cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
            captured.append(cfg)
        tput = _COLD_TPUT if state["calls"] == 0 else _HOT_TPUT
        if clock is not None:
            server_log_path = kwargs.get("server_log_path")
            if state["calls"] == 0:
                clock.advance(boot_sec)
                if server_log_path:
                    Path(server_log_path).parent.mkdir(parents=True, exist_ok=True)
                    stamp_server_ready(server_log_path, boot_sec)
            clock.advance(benchmark_sec)
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    return fake_run, state


def _unbounded_shared_state(**fields) -> SimpleNamespace:
    """A session state whose budget never bounds a round, plus whatever the case sets."""
    return SimpleNamespace(session_budget_usable_sec=lambda: None, **fields)


def _executor(
    base: Path,
    tmp_path: Path,
) -> BaselineExecutor:
    return BaselineExecutor(
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=_unbounded_shared_state(),
    )


# The workload every case in the gate class below is priced against, and the figures the pricing derives from it.
# 900 - 550: the part of the cold round that was not the benchmark.
# One further measured variant: its own boot, then its own benchmark.
# A round's first pass is the cold one, measured whole rather than rebuilt from its halves -- rebuilding it as
# boot-plus-hot would drop the compile it paid and under-price the round by 150s.


@pytest.fixture
def deferred_accuracy_keep_policy(tmp_path, monkeypatch):
    # Keep the built-in lifecycle script while exercising the interactivity objective.
    monkeypatch.setenv("HYPERLOOM_AGENTX", "0")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "intvty_v1")
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "5")
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    executor = _executor(base, tmp_path)
    shared = executor.shared_state
    shared.framework = "vllm"
    shared.benchmark_mode = "synthetic"
    shared.baseline_tput = 100.0
    shared.baseline_perf = {
        "output_throughput": 100.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": 100.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    shared.current_best = {"action": "baseline", "tput": 100.0, **shared.baseline_perf}
    shared.optimization_stack = []
    output_dir = tmp_path / "ws"
    params = {
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "gpu_type": "mi300x",
        "baseline_double_run": True,
        "defer_accuracy_until_after_measure": True,
        "post_measure_accuracy_min_tput": 101.0,
        "post_measure_accuracy_keep_policy": {
            "base_tput": 100.0,
            "keep_threshold_pct": 1.0,
            "stack_incremental_keep_threshold_pct": 0.5,
        },
    }
    measurement = {
        "output_throughput": 90.0,
        "total_token_throughput": 1100.0,
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": 100.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    captured: list = []
    inner, calls = _cold_then_hot_fake_run(captured)

    def fake_run(cmd, *args, **kwargs):
        completed = inner(cmd, *args, **kwargs)
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        workspace = next(slot.glob("benchmark_*"))
        axes = (
            measurement
            if slot.name == "measure_round"
            else {
                "output_throughput": 9999.0,
                "total_token_throughput": 99999.0,
                "e2e_norm_intvty_p90": 999.0,
                "e2e_norm_intvty_p50": 999.0,
                "duration_seconds": 25.0,
                "request_error_rate": 0.0,
            }
        )
        report_path = workspace / "benchmark_report.json"
        report = json.loads(report_path.read_text())
        report["model"] = f"model-{slot.name}"
        report["throughput"].update(
            output_throughput=axes["output_throughput"],
            request_throughput=axes["output_throughput"] / 1024,
            total_token_throughput=axes.get("total_token_throughput"),
        )
        report_path.write_text(json.dumps(report), encoding="utf-8")
        (workspace / "inferencex_result.json").write_text(json.dumps(axes), encoding="utf-8")
        if captured[-1]["benchmark"]["envs"]["RUN_EVAL"] == "true":
            (slot / "results_gsm8k.json").write_text(
                json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}),
                encoding="utf-8",
            )
        return completed

    def run_case():
        with patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ):
            return _run(executor(_make_ctx(params)))

    return SimpleNamespace(
        shared=shared,
        params=params,
        measurement=measurement,
        captured=captured,
        calls=calls,
        output_dir=output_dir,
        run=run_case,
    )


def test_teardown_lifecycle_server_removes_state_files(tmp_path):
    """The defensive teardown unlinks stale pid/meta files without raising."""
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (pid_dir / "vllm_8888.pid").write_text("2147483646")
    (pid_dir / "vllm_8888.json").write_text("{}")

    sl.teardown_lifecycle_server(
        pid_dir=pid_dir,
        framework="vllm",
        port=8888,
    )

    assert not (pid_dir / "vllm_8888.pid").exists()
    assert not (pid_dir / "vllm_8888.json").exists()


def test_teardown_lifecycle_server_skips_signal_on_pid_reuse(tmp_path, monkeypatch):
    """A pid that does not look like a Hyperloom server must not receive a signal."""
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (pid_dir / "vllm_8888.pid").write_text("99999")
    (pid_dir / "vllm_8888.json").write_text("{}")

    signals_sent: list[int] = []
    monkeypatch.setattr(sl, "signal_group", lambda pgid, sig, **_: signals_sent.append(sig))
    monkeypatch.setattr(sl, "_looks_like_server_process", lambda pid: False)

    sl.teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=8888)

    assert signals_sent == [], "must not signal a pid that failed the identity check"
    assert not (pid_dir / "vllm_8888.pid").exists(), "stale pid file must still be removed"
    assert not (pid_dir / "vllm_8888.json").exists(), "stale meta file must still be removed"


# Every output slot a multi-node baseline round launches a benchmark process into, in launch order.


# ``--max-hours`` defaults to 2.0 (``cli/parser.py``), so this is the session shape almost every run has.


class _AClockOnlyThePassesMove:
    """The wall clock the executor prices rounds by, moved by hand."""

    def __init__(self) -> None:
        self._now = time.time()

    def __call__(self) -> float:
        """Answer as ``time.time`` does."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Spend ``seconds``, as a pass that ran that long does."""
        self._now += float(seconds)


# _classify_subprocess_error unit tests

from hyperloom.orchestrator.actions.executors.benchmark_run.bringup import (
    _classify_subprocess_error,
)


def test_classify_fast_exit_unknown_backend():
    assert (
        _classify_subprocess_error(5.0, "ValueError: Unknown attention backend: 'ROCM_FLASH'") == "fast_exit_arg_error"
    )


def test_classify_fast_exit_unrecognized_args():
    assert _classify_subprocess_error(2.0, "error: unrecognized arguments: --bogus-flag") == "fast_exit_arg_error"


def test_classify_slow_failure_not_arg_error():
    """A generic runtime ValueError does not establish argument validation."""
    assert _classify_subprocess_error(120.0, "ValueError: some runtime error") == "subprocess_nonzero"


_CACHE_CONFIG_ERROR = (
    "pydantic_core._pydantic_core.ValidationError: 1 validation error for CacheConfig\n"
    "gpu_memory_utilization\n"
    "Input should be less than or equal to 1 [type=less_than_equal, input_value=1.5, input_type=float]"
)


@pytest.mark.parametrize("elapsed", [8.0, 30.0, 31.08, 31.09, 70.74])
def test_cache_config_validation_is_not_a_worker_crash(elapsed):
    assert _classify_subprocess_error(elapsed, _CACHE_CONFIG_ERROR) == "fast_exit_arg_error"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            "ValidationError: 1 validation error for ModelConfig\nModel architecture is not registered",
            "subprocess_nonzero",
        ),
        ("ValidationError: 1 validation error for Sample\nmissing field", "subprocess_nonzero"),
        ("CacheConfig initialized\nRuntimeError: worker process died", "subprocess_nonzero"),
    ],
)
def test_cache_config_validation_preserves_other_failure_routes(error, expected):
    assert _classify_subprocess_error(70.74, error) == expected


def test_classify_fast_exit_without_pattern():
    """A fast exit without arg-error patterns stays subprocess_nonzero."""
    assert _classify_subprocess_error(3.0, "Segmentation fault (core dumped)") == "subprocess_nonzero"


def test_classify_fast_runtime_value_error_not_arg_error():
    """A generic fast runtime ValueError is not enough for arg-error routing."""
    assert _classify_subprocess_error(3.0, "ValueError: tensor shape mismatch during warmup") == "subprocess_nonzero"


def test_classify_value_error_with_argv_dump_not_arg_error():
    """A command/argv dump containing flags is not arg validation by itself."""
    assert (
        _classify_subprocess_error(
            3.0,
            "ValueError: tensor shape mismatch during warmup\nargv: vllm serve --model /models/foo --tp 8",
        )
        == "subprocess_nonzero"
    )


def test_classify_subprocess_error_none_tail_does_not_crash():
    # A slow failure with no captured stderr must not raise.
    assert _classify_subprocess_error(600.0, None) == "subprocess_nonzero"


def test_classify_kv_cache_oom_after_weight_load():
    # KV-cache OOM must be detected regardless of elapsed time.
    tail = (
        "ValueError: Loaded weights leave no GPU memory for the KV cache "
        "under --mem-fraction-static=0.7. Raise --mem-fraction-static above 0.737"
    )
    assert _classify_subprocess_error(600.0, tail) == "kv_cache_oom"


def test_classify_kv_cache_oom_fast_exit():
    tail = "no GPU memory for the KV cache"
    assert _classify_subprocess_error(3.0, tail) == "kv_cache_oom"


def test_classify_non_kv_oom_still_nonzero():
    assert _classify_subprocess_error(600.0, "some other runtime failure") == "subprocess_nonzero"
