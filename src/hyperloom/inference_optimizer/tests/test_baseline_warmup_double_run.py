# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the baseline cold-start \"warmup artifact\"."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.benchmark_run.records import MEASURE_ROUND_DROPPED_WARNING
from hyperloom.orchestrator.actions.executors._grid_runner import (
    SESSION_TIME_EXHAUSTED_CLASS,
)
from hyperloom.orchestrator.actions.executors._inferencex_patcher import BENCH_SEED_ENV
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    SESSION_TIME_EXHAUSTED_RETURNCODE,
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
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=_unbounded_shared_state(),
    )


def _run_mi300x_baseline(tmp_path, fake_run, *, framework: str = "vllm", **params) -> dict:
    """Run the baseline on mi300x into ``tmp_path/ws`` with Magpie replaced by ``fake_run``."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework=framework)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x", **params})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        return _run(_executor(base, tmp_path)(ctx))


def _prelude_shared_state(*, usable_sec: float, phase: str = "PRELUDE") -> SimpleNamespace:
    """A session state with an explicit clock, as the budget policy reads it."""
    return SimpleNamespace(
        baseline_double_run=True,
        phase=phase,
        max_minutes=180,
        session_budget_usable_sec=lambda: usable_sec,
    )


def test_a_measured_round_that_fails_on_its_own_is_still_a_failure(tmp_path):
    """Only the run's clock earns the fallback."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    calls = {"n": 0}

    def fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        calls["n"] += 1
        if calls["n"] == 1:
            _fake_workspace(slot, tput=_COLD_TPUT)
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.CompletedProcess(cmd, 1, "", "CUDA error")

    executor = BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=_prelude_shared_state(usable_sec=10_000.0),
    )
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] != SESSION_TIME_EXHAUSTED_CLASS
    assert MEASURE_ROUND_DROPPED_WARNING not in (result.get("nonfatal_warnings") or [])


# The workload every case in the gate class below is priced against, and the figures the pricing derives from it.
_COLD_ROUND_SEC = 900.0
_COLD_POST_READY_SEC = 550.0
_HOT_ROUND_SEC = 400.0
# 900 - 550: the part of the cold round that was not the benchmark.
# One further measured variant: its own boot, then its own benchmark.
# A round's first pass is the cold one, measured whole rather than rebuilt from its halves -- rebuilding it as
# boot-plus-hot would drop the compile it paid and under-price the round by 150s.


class TestARoundThatCannotFinishIsNotIgnited:
    """The gate in front of a round, and the two things it must be asked with."""

    def test_a_refused_round_carries_nothing_that_could_replace_the_anchor(self, tmp_path):
        """The property the whole gate rests on, asserted rather than assumed."""
        result, _calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=1400.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
        )

        assert result.get("output_throughput") in (None, 0, 0.0)
        assert not result.get("nonfatal_warnings"), "a refused round volunteered a warning to promote"


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


def test_a_native_agentx_warmup_keeps_its_seed(tmp_path, monkeypatch):
    """The native AgentX client is not the patched one, so its rounds are left exactly as they were."""
    from hyperloom.orchestrator.actions.executors import benchmark_backend

    config = tmp_path / "variant.yaml"
    config.write_text("benchmark:\n  framework: sglang\n  agentx:\n    enabled: true\n", encoding="utf-8")
    env = {BENCH_SEED_ENV: "7"}
    benchmark_backend.apply_benchmark_round_seed(config, env, warmup=True)
    assert env == {}


_STALE_FULL_REPORT = {
    "throughput": {"output_throughput": 9999.0, "completed_requests": 64, "duration_seconds": 25.0},
    "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
}


def _seed_stale_workspace(stale: Path, report: dict) -> None:
    """A successful vllm workspace from a prior attempt, dated well before the current run."""
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps({"success": True, "framework": "vllm", **report}),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))


def test_baseline_rejects_stale_workspace_on_crash(tmp_path, monkeypatch):
    """A stale benchmark_* workspace from a prior attempt must not be adopted as a successful result when the current subprocess crashes (rc=1)."""
    _seed_stale_workspace(tmp_path / "ws" / "benchmark_vllm_20260101_000000", _STALE_FULL_REPORT)

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "HIP out of memory")

    result = _run_mi300x_baseline(tmp_path, fake_run, baseline_double_run=False)

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_rejects_stale_workspace_on_silent_exit(tmp_path, monkeypatch):
    """A stale benchmark_* workspace must not be adopted when the subprocess exits 0 without producing any new workspace (silent no-op)."""
    _seed_stale_workspace(tmp_path / "ws" / "benchmark_vllm_20260101_000000", _STALE_FULL_REPORT)

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    result = _run_mi300x_baseline(tmp_path, fake_run, baseline_double_run=False)

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_rejects_stale_workspace_when_the_run_produced_none(tmp_path, monkeypatch):
    """A crashed run adopts no workspace, however high the stale one sorts."""
    _seed_stale_workspace(tmp_path / "ws" / "benchmark_vllm_29991231_235959", _STALE_FULL_REPORT)

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    result = _run_mi300x_baseline(tmp_path, fake_run, baseline_double_run=False)

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


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
_MEASURED_ROUND_SLOT = "measured_round"


# ``--max-hours`` defaults to 2.0 (``cli/parser.py``), so this is the session shape almost every run has.


class _BudgetedState:
    """A session state whose budget accounting moves as the passes spend it."""

    def __init__(
        self,
        *,
        remaining_sec: float | None,
        cold_round_sec: float = 0.0,
        cold_post_ready_sec: float = 0.0,
        hot_round_sec: float = 0.0,
        phase: str = "PRELUDE",
        double_run: bool = False,
    ) -> None:
        self.baseline_double_run = double_run
        self.baseline_runtime_sec = cold_round_sec
        self.baseline_post_ready_runtime_sec = cold_post_ready_sec
        self.baseline_warm_runtime_sec = hot_round_sec
        self.phase = phase
        self.max_minutes = 0.0 if remaining_sec is None else remaining_sec / 60.0
        self._deadline = None if remaining_sec is None else time.monotonic() + remaining_sec

    def charge(self, seconds: float) -> None:
        """Spend ``seconds`` of the session, as a pass that ran that long does."""
        if self._deadline is not None:
            self._deadline -= seconds

    def grid_session_deadline_sec(self) -> float | None:
        return self._deadline

    def session_budget_usable_sec(self) -> float | None:
        return None if self._deadline is None else self._deadline - time.monotonic()


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


def _capturing_fake_run(
    returncode: int = 0,
    *,
    produces_workspace: bool = True,
    pass_duration_sec: float = 0.0,
    state: _BudgetedState | None = None,
    charge_sec: float | None = None,
    clock: _AClockOnlyThePassesMove | None = None,
    boot_sec: float = 0.0,
    benchmark_sec: float = 0.0,
):
    """A ``run_with_session_kill`` stand-in that records how each round was launched."""
    calls: list[dict] = []

    def fake_run(cmd, *args, **kwargs):
        # The memoized interpreter probe is not a benchmark round.
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        kwargs = _launch_kwargs(kwargs)
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        calls.append({"round_slot": slot.name, **kwargs})
        granted = float(kwargs.get("timeout") or 0.0)
        deadline = kwargs.get("session_deadline_sec")
        ran_sec = min(granted, pass_duration_sec) if pass_duration_sec else 0.0
        reaped_by_the_session = False
        if deadline is not None and pass_duration_sec:
            until_deadline = max(0.0, deadline - time.monotonic())
            reaped_by_the_session = until_deadline < ran_sec
            ran_sec = min(ran_sec, until_deadline)
        if clock is not None:
            clock.advance(boot_sec)
            server_log_path = kwargs.get("server_log_path")
            if server_log_path:
                Path(server_log_path).parent.mkdir(parents=True, exist_ok=True)
                stamp_server_ready(server_log_path, boot_sec)
            clock.advance(benchmark_sec)
        if state is not None:
            state.charge(charge_sec if charge_sec is not None else ran_sec)
        if pass_duration_sec and ran_sec < pass_duration_sec:
            if reaped_by_the_session:
                return subprocess.CompletedProcess(cmd, SESSION_TIME_EXHAUSTED_RETURNCODE, "", "")
            raise subprocess.TimeoutExpired(cmd, granted)
        if produces_workspace:
            _fake_workspace(slot, tput=_HOT_TPUT)
        return subprocess.CompletedProcess(cmd, returncode, "ok", "")

    return fake_run, calls


def _run_baseline_under_budget(
    tmp_path,
    *,
    remaining_sec: float | None,
    timeout_sec: int = 7200,
    returncode: int = 0,
    produces_workspace: bool = True,
    cold_round_sec: float = 0.0,
    cold_post_ready_sec: float = 0.0,
    hot_round_sec: float = 0.0,
    pass_duration_sec: float = 0.0,
    phase: str = "PRELUDE",
    double_run: bool = False,
    executor_cls=BaselineExecutor,
) -> tuple[dict, list[dict]]:
    """Run one baseline round against a session with ``remaining_sec`` left."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    state = _BudgetedState(
        remaining_sec=remaining_sec,
        cold_round_sec=cold_round_sec,
        cold_post_ready_sec=cold_post_ready_sec,
        hot_round_sec=hot_round_sec,
        phase=phase,
        double_run=double_run,
    )
    fake_run, calls = _capturing_fake_run(
        returncode,
        produces_workspace=produces_workspace,
        pass_duration_sec=pass_duration_sec,
        state=state,
    )
    executor = executor_cls(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx_params: dict = {
        "output_dir": str(tmp_path / _MEASURED_ROUND_SLOT),
        "timeout_sec": timeout_sec,
        "gpu_type": "mi300x",
    }
    if not double_run:
        ctx_params["baseline_double_run"] = False
    ctx = _make_ctx(ctx_params)
    # The live state arrives on the context, the way the coordinator passes it.
    ctx.extra["shared_state"] = state
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))
    return result, calls


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
