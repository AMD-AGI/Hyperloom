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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.baseline import (
    BASELINE_DEFAULT_TIMEOUT_SEC,
    MEASURE_ROUND_DROPPED_WARNING,
    BaselineExecutor,
)
from hyperloom.orchestrator.actions.executors.profile import (
    PROFILE_DEFAULT_TIMEOUT_SEC,
    ProfileExecutor,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    GridVariant,
    _SESSION_KILL_GRACE_SEC,
    run_grid,
)
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    stamp_server_ready,
)
from hyperloom.orchestrator.actions.stop_attribution import STOPPED_BY_THE_RUN
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.trace.task_progress import progress_scope

from .conftest import (
    chatty_child,
    enable_multi_node,
    launches_by_round_slot,
    suppression_window_s,
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
    clock: _AClockOnlyThePassesMove | None = None,
    boot_sec: float = 0.0,
    benchmark_sec: float = 0.0,
):
    """Return a ``run_with_session_kill`` stand-in that emits a cold throughput on its first call and a hot throughput thereafter."""
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
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


def _executor(
    base: Path,
    tmp_path: Path,
    *,
    baseline_double_run: bool = True,
) -> BaselineExecutor:
    return BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=SimpleNamespace(baseline_double_run=baseline_double_run),
    )


def test_baseline_discards_cold_first_round_via_lifecycle(tmp_path, monkeypatch):
    """The double-run reports the HOT second-round throughput."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert result.get("warmup_round_tput") == pytest.approx(_COLD_TPUT)
    assert "baseline_double_run_discarded_first" in result["nonfatal_warnings"]
    # The hot pass reuses the warmup server, so its identity evidence must be carried from the server-owning warmup
    # slot.
    assert result["launch_evidence_path"].endswith("warmup_round/launch_evidence.json")
    assert result["launch_evidence"]["warm_reuse"]["reused_ready_server"] is True
    assert result["launch_evidence"]["warm_reuse"]["provenance"] == "warmup_round"

    assert len(captured) == 2
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["enabled"] is True and measure_lc["enabled"] is True
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir)
    assert captured[0]["benchmark"]["envs"]["PORT"] == (captured[1]["benchmark"]["envs"]["PORT"])
    assert captured[0]["benchmark"]["benchmark_script"] == "vllm_mi300x.sh"


def _run_capturing_rounds(executor, ctx, notes):
    """Run ``executor`` and record which round notes existed at each launch."""
    at_launch: list[list[str]] = []
    inner, _state = _cold_then_hot_fake_run()

    def fake_run(cmd, *args, **kwargs):
        at_launch.append([n["label"] for n in notes])
        return inner(cmd, *args, **kwargs)

    with (
        progress_scope(_sink_into(notes)),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        return _run(executor(ctx)), at_launch


def _sink_into(notes: list):
    """Build an ambient progress sink that appends every note to ``notes``."""

    async def _sink(**note):
        notes.append(note)

    return _sink


def test_each_double_run_round_reports_before_it_blocks(tmp_path):
    """A round that boots a server and never returns must still have said it started."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    notes: list[dict] = []
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    result, at_launch = _run_capturing_rounds(executor, ctx, notes)

    assert result["status"] == "succeeded"
    assert at_launch == [["warmup"], ["warmup", "warmup", "measure"]]
    assert [(n["label"], n["status"]) for n in notes] == [
        ("warmup", "started"),
        ("warmup", "succeeded"),
        ("measure", "started"),
    ]


def test_the_single_round_path_reports_too(tmp_path):
    """The non-double-run baseline used to report nothing at all."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    notes: list[dict] = []
    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    result, at_launch = _run_capturing_rounds(executor, ctx, notes)

    assert result["status"] == "succeeded"
    assert at_launch == [["single"]]


def test_a_round_is_handed_the_liveness_callback_its_heartbeat_needs(tmp_path):
    """A round outlives its start report; only child output can extend it."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    seen: list = []
    inner, _state = _cold_then_hot_fake_run()

    def fake_run(cmd, *args, **kwargs):
        seen.append(kwargs.get("on_output"))
        return inner(cmd, *args, **kwargs)

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert [callable(cb) for cb in seen] == [True]


def _cadence_ctx(tmp_path) -> SimpleNamespace:
    """A single-round baseline context for the cadence tests."""
    return _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})


def test_a_round_keeps_reporting_while_its_benchmark_blocks(tmp_path, progress_cadence):
    """A round blocks for the better part of an hour; entry markers cannot cover that."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    inner, _state = _cold_then_hot_fake_run()
    executor = _executor(base, tmp_path, baseline_double_run=False)

    with (
        progress_scope(progress_cadence.sink()),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=chatty_child(progress_cadence, inner, blocks_for_s=600.0, line_every_s=30.0),
        ),
    ):
        result = _run(executor(_cadence_ctx(tmp_path)))

    assert result["status"] == "succeeded"
    assert progress_cadence.widest_silence() < suppression_window_s()


def test_the_multi_node_warmup_pass_keeps_reporting_too(tmp_path, monkeypatch, progress_cadence):
    """The discarded MN warmup is a full benchmark pass and blocks just as long."""
    enable_multi_node(monkeypatch)
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    inner, state = _cold_then_hot_fake_run()
    executor = _executor(base, tmp_path, baseline_double_run=False)

    with (
        progress_scope(progress_cadence.sink()),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=chatty_child(progress_cadence, inner, blocks_for_s=600.0, line_every_s=30.0),
        ),
    ):
        result = _run(executor(_cadence_ctx(tmp_path)))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2  # the discarded warmup pass, then the measured one
    assert progress_cadence.widest_silence() < suppression_window_s()


def test_a_failing_warmup_round_still_reported_that_it_started(tmp_path):
    """The failure path returns early; only the entry report covers it."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    notes: list[dict] = []
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    with (
        progress_scope(_sink_into(notes)),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, "", "boom"),
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert [(n["label"], n["status"]) for n in notes] == [("warmup", "started")]


def _prelude_shared_state(*, usable_sec: float, phase: str = "PRELUDE") -> SimpleNamespace:
    """A session state with an explicit clock, as the budget policy reads it."""
    return SimpleNamespace(
        baseline_double_run=True,
        phase=phase,
        max_minutes=180,
        session_budget_usable_sec=lambda: usable_sec,
    )


def _a_session_the_passes_spend(
    *,
    usable_sec: float,
    clock: _AClockOnlyThePassesMove,
    **measured: float,
) -> SimpleNamespace:
    """A PRELUDE session whose remaining budget falls as the passes spend it."""
    started = clock()
    return SimpleNamespace(
        baseline_double_run=True,
        phase="PRELUDE",
        max_minutes=180,
        session_budget_usable_sec=lambda: usable_sec - (clock() - started),
        **measured,
    )


def _run_double_run_baseline(
    tmp_path,
    shared_state,
    *,
    clock: _AClockOnlyThePassesMove | None = None,
    boot_sec: float = 0.0,
    benchmark_sec: float = 0.0,
) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    fake_run, state = _cold_then_hot_fake_run(
        clock=clock,
        boot_sec=boot_sec,
        benchmark_sec=benchmark_sec,
    )
    executor = BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=shared_state,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})
    with (
        _passes_time_the_executor_believes(clock or _AClockOnlyThePassesMove()),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))
    result["_rounds_run"] = state["calls"]
    return result


def test_a_budget_that_cannot_pay_for_the_measured_round_keeps_the_cold_warmup(tmp_path):
    """The session clock cannot pay for the hot pass and a use for it; the cold one ran."""
    clock = _AClockOnlyThePassesMove()

    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(usable_sec=1200.0),
        clock=clock,
        boot_sec=350.0,
        benchmark_sec=550.0,
    )

    assert result["status"] == "succeeded"
    assert result["_rounds_run"] == 1, "the measured round ran on a budget that cannot pay for it"
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert MEASURE_ROUND_DROPPED_WARNING in result["nonfatal_warnings"]
    dropped = result["measure_round_dropped"]
    assert dropped["bound"] == "session_usable"
    assert dropped["priced_by"] == "warmup_post_ready"
    assert dropped["measure_round_sec"] == pytest.approx(550.0, abs=1.0)
    assert dropped["one_more_measurement_sec"] == pytest.approx(900.0, abs=1.0)
    assert dropped["measure_round_sec"] < 1200.0, (
        "the pass alone did not fit, so this case does not show what it claims to"
    )


def test_a_round_admitted_before_ignition_is_not_refused_after_its_cold_pass(tmp_path):
    """The two gates price the same second pass, so they must reach the same answer."""
    clock = _AClockOnlyThePassesMove()
    state = _a_session_the_passes_spend(
        usable_sec=2100.0,
        clock=clock,
        baseline_runtime_sec=900.0,
        baseline_post_ready_runtime_sec=550.0,
        baseline_warm_runtime_sec=400.0,
    )

    result = _run_double_run_baseline(
        tmp_path,
        state,
        clock=clock,
        boot_sec=350.0,
        benchmark_sec=550.0,
    )

    assert result["_rounds_run"] == 2, "a cold pass was spent on a round the gate after it was always going to refuse"
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)


def test_a_warmup_that_overran_its_prediction_still_drops_the_hot_pass(tmp_path):
    """Agreeing with the earlier gate is not the same as admitting everything."""
    clock = _AClockOnlyThePassesMove()
    state = _a_session_the_passes_spend(
        usable_sec=2100.0,
        clock=clock,
        baseline_runtime_sec=900.0,
        baseline_post_ready_runtime_sec=550.0,
        baseline_warm_runtime_sec=400.0,
    )

    result = _run_double_run_baseline(
        tmp_path,
        state,
        clock=clock,
        boot_sec=700.0,
        benchmark_sec=550.0,
    )

    assert result["_rounds_run"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert result["measure_round_dropped"]["priced_by"] == "session_hot_pass"


def test_a_rebaselines_hot_pass_needs_only_its_own_wall_clock(tmp_path):
    """The same 1200s that drops the hot pass in PRELUDE runs it here."""
    clock = _AClockOnlyThePassesMove()

    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(usable_sec=1200.0, phase="FRAMEWORK_AGENT"),
        clock=clock,
        boot_sec=350.0,
        benchmark_sec=550.0,
    )

    assert result["_rounds_run"] == 2, "the measurement the round exists for was dropped"
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert MEASURE_ROUND_DROPPED_WARNING not in (result.get("nonfatal_warnings") or [])


def test_a_rebaseline_that_cannot_cover_its_hot_pass_keeps_the_cold_warmup(tmp_path):
    """A later phase asks a narrower question, not no question."""
    clock = _AClockOnlyThePassesMove()

    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(usable_sec=400.0, phase="FRAMEWORK_AGENT"),
        clock=clock,
        boot_sec=350.0,
        benchmark_sec=550.0,
    )

    assert result["_rounds_run"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert result["measure_round_dropped"]["one_more_measurement_sec"] == pytest.approx(0.0)


def test_a_rounds_boot_is_priced_even_though_no_cap_bounded_it(tmp_path):
    """The gate prices the round on what it spent, not on what its cap allowed."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    clock = _AClockOnlyThePassesMove()
    state = _BudgetedState(remaining_sec=3600.0, double_run=True)
    fake_run, calls = _capturing_fake_run(
        state=state,
        clock=clock,
        boot_sec=1400.0,
        benchmark_sec=1200.0,
    )
    executor = BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=state,
    )
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})
    with (
        _passes_time_the_executor_believes(clock),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert [c["round_slot"] for c in calls] == ["warmup_round"], (
        f"the measured round ran on a budget the round had already spent: {[c['round_slot'] for c in calls]}"
    )
    assert float(calls[0]["timeout"]) <= 10.0, "the cap was not the small one this case rests on"
    assert result["status"] == "succeeded"
    assert MEASURE_ROUND_DROPPED_WARNING in result["nonfatal_warnings"]
    dropped = result["measure_round_dropped"]
    assert dropped["one_more_measurement_sec"] == pytest.approx(2600.0, abs=1.0)
    assert dropped["expected_cost_sec"] == pytest.approx(3800.0, abs=1.0)


def _warmup_then_reaped_fake_run(tmp_path):
    """A double run whose warmup lands and whose measured pass the clock takes."""
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        state["calls"] += 1
        if state["calls"] == 1:
            _fake_workspace(slot, tput=_COLD_TPUT)
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.CompletedProcess(cmd, SESSION_TIME_EXHAUSTED_RETURNCODE, "", "")

    return fake_run, state


def test_a_measured_round_the_clock_takes_mid_flight_keeps_the_cold_warmup(tmp_path):
    """The GPU time behind the warmup's figure is spent either way."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    fake_run, state = _warmup_then_reaped_fake_run(tmp_path)
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

    assert state["calls"] == 2, "the measured round did not run, so it cannot have been reaped"
    assert result["status"] == "succeeded"
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert MEASURE_ROUND_DROPPED_WARNING in result["nonfatal_warnings"]
    assert result["measure_round_dropped"]["reason"] == "measure_round_reaped_by_the_run"


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


def test_measured_round_survives_a_budget_that_still_covers_it(tmp_path):
    """The guard must not turn every double-run into a single one."""
    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(usable_sec=10_000.0),
        clock=_AClockOnlyThePassesMove(),
        boot_sec=350.0,
        benchmark_sec=550.0,
    )

    assert result["_rounds_run"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert "budget_shortfall" not in result


def test_a_double_run_reports_the_boot_split_of_the_pass_that_paid_it(tmp_path):
    """The round's total and the part of it that was the benchmark, from one pass."""
    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(usable_sec=10_000.0),
        clock=_AClockOnlyThePassesMove(),
        boot_sec=350.0,
        benchmark_sec=550.0,
    )

    assert result["subprocess_runtime_sec"] == pytest.approx(900.0, abs=1.0)
    assert result["post_ready_runtime_sec"] == pytest.approx(550.0, abs=1.0)
    assert result["measure_round_runtime_sec"] == pytest.approx(550.0, abs=1.0)
    boot_sec = result["subprocess_runtime_sec"] - result["post_ready_runtime_sec"]
    assert boot_sec == pytest.approx(350.0, abs=1.0)


# The workload every case in the gate class below is priced against, and the figures the pricing derives from it.
_COLD_ROUND_SEC = 900.0
_COLD_POST_READY_SEC = 550.0
_HOT_ROUND_SEC = 400.0
# 900 - 550: the part of the cold round that was not the benchmark.
_BOOT_SEC = 350.0
# One further measured variant: its own boot, then its own benchmark.
_ONE_MORE_SEC = _BOOT_SEC + _HOT_ROUND_SEC
# A round's first pass is the cold one, measured whole rather than rebuilt from its halves -- rebuilding it as
# boot-plus-hot would drop the compile it paid and under-price the round by 150s.
_SINGLE_ROUND_SEC = _COLD_ROUND_SEC
_DOUBLE_ROUND_SEC = _COLD_ROUND_SEC + _HOT_ROUND_SEC


class TestARoundThatCannotFinishIsNotIgnited:
    """The gate in front of a round, and the two things it must be asked with."""

    def test_a_first_round_is_not_judged_at_all(self, tmp_path):
        """Nothing measured yet, so nothing to refuse it with."""
        result, calls = _run_baseline_under_budget(tmp_path, remaining_sec=3600.0)

        assert result["status"] == "succeeded"
        assert calls, "a first baseline was refused on a prediction the session cannot have"

    def test_a_round_larger_than_what_is_left_boots_nothing(self, tmp_path):
        """750s for the round and 750s for a variant to use it: 1400s cannot."""
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=1400.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
        )

        assert calls == [], "GPU time was spent on a round the session cannot finish"
        assert result["status"] == "failed"
        assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
        assert result["returncode"] is None, "a round that never launched reported a returncode"
        assert result["error"] == STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS].never_started
        shortfall = result["budget_shortfall"]
        assert shortfall["expected_cost_sec"] == pytest.approx(_SINGLE_ROUND_SEC + _ONE_MORE_SEC)
        assert shortfall["round_sec"] == pytest.approx(_SINGLE_ROUND_SEC)
        assert shortfall["one_more_measurement_sec"] == pytest.approx(_ONE_MORE_SEC)
        assert shortfall["affordable_sec"] == pytest.approx(1400.0, abs=1.0)

    def test_a_round_the_budget_covers_is_ignited(self, tmp_path):
        """The gate must not turn a merely expensive round into a refused one."""
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=_SINGLE_ROUND_SEC + _ONE_MORE_SEC + 60.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
        )

        assert result["status"] == "succeeded"
        assert calls

    def test_a_round_the_budget_covers_with_nothing_left_to_use_it_is_refused(self, tmp_path):
        """The requirement that is not about finishing the round."""
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=1000.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
        )

        assert calls == [], "a round ran that nothing could be measured against"
        assert result["budget_shortfall"]["round_sec"] == pytest.approx(_SINGLE_ROUND_SEC)
        assert result["budget_shortfall"]["round_sec"] < 1000.0, (
            "the round itself did not fit, so this case is not the one it claims to be"
        )

    def test_a_rebaseline_in_a_later_phase_needs_no_successor(self, tmp_path):
        """The same 1000s that refuses a PRELUDE round admits this one."""
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=1000.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
            phase="FRAMEWORK_AGENT",
        )

        assert calls, "the round that validates the stack was refused for lack of a successor"
        assert result["status"] == "succeeded"

    def test_a_rebaseline_larger_than_what_is_left_is_still_refused(self, tmp_path):
        """Dropping the successor does not drop the round's own cost."""
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=_SINGLE_ROUND_SEC - 100.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
            phase="FRAMEWORK_AGENT",
        )

        assert calls == [], "GPU time was spent on a round the session cannot finish"
        assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
        shortfall = result["budget_shortfall"]
        assert shortfall["one_more_measurement_sec"] == pytest.approx(0.0)
        assert shortfall["expected_cost_sec"] == pytest.approx(_SINGLE_ROUND_SEC)

    def test_a_double_run_pays_for_the_second_pass_but_not_a_second_boot(self, tmp_path):
        """One budget, two answers: it covers a single-pass round, not a double one."""
        budget_sec = _SINGLE_ROUND_SEC + _ONE_MORE_SEC + 60.0
        for side in ("single", "double"):
            (tmp_path / side).mkdir()

        fits, fitting_calls = _run_baseline_under_budget(
            tmp_path / "single",
            remaining_sec=budget_sec,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
        )
        refused, refused_calls = _run_baseline_under_budget(
            tmp_path / "double",
            remaining_sec=budget_sec,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
            double_run=True,
        )

        assert fits["status"] == "succeeded" and fitting_calls
        assert refused_calls == [], "the round was priced on one pass while planning to run two"
        shortfall = refused["budget_shortfall"]
        assert shortfall["round_sec"] == pytest.approx(_DOUBLE_ROUND_SEC)
        assert shortfall["expected_cost_sec"] == pytest.approx(_DOUBLE_ROUND_SEC + _ONE_MORE_SEC)

    def test_a_session_with_no_hot_figure_prices_the_variant_from_the_cold_pass(self, tmp_path):
        """The state a previous cold-anchor drop leaves, and the one to catch."""
        one_more_sec = _BOOT_SEC + _COLD_POST_READY_SEC

        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=_SINGLE_ROUND_SEC + one_more_sec - 1.0,
            cold_round_sec=_COLD_ROUND_SEC,
            cold_post_ready_sec=_COLD_POST_READY_SEC,
        )

        assert calls == []
        shortfall = result["budget_shortfall"]
        assert shortfall["one_more_measurement_sec"] == pytest.approx(one_more_sec)
        assert shortfall["one_more_measurement_sec"] > _ONE_MORE_SEC, (
            "the fallback did not over-predict, so it cannot be the post-ready segment"
        )

    def test_a_round_whose_boot_was_never_measured_is_still_judged(self, tmp_path):
        """A round with no split is priced at whole cold rounds, not waved through."""
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=100.0,
            cold_round_sec=_COLD_ROUND_SEC,
            hot_round_sec=_HOT_ROUND_SEC,
        )

        assert calls == [], "a workload with no boot boundary was exempted from the gate"
        shortfall = result["budget_shortfall"]
        assert shortfall["round_sec"] == pytest.approx(_COLD_ROUND_SEC)
        assert shortfall["one_more_measurement_sec"] == pytest.approx(_COLD_ROUND_SEC)

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


def test_deferred_accuracy_skips_eval_when_hot_throughput_regresses(
    tmp_path,
):
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT + 1,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert all(cfg["benchmark"]["envs"]["RUN_EVAL"] == "false" for cfg in captured)
    assert result["accuracy_stage"]["status"] == "skipped"
    assert result["accuracy_stage"]["reason"] == "throughput_below_threshold"


def test_deferred_accuracy_reuses_hot_server_after_throughput_passes(
    tmp_path,
):
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        captured.append(cfg)
        tput = _COLD_TPUT if state["calls"] == 0 else _HOT_TPUT
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        if cfg["benchmark"]["envs"].get("RUN_EVAL") == "true":
            (slot / "results_gsm8k.json").write_text(
                json.dumps(
                    {
                        "results": {
                            "gsm8k": {
                                "exact_match,strict-match": 0.9,
                            }
                        }
                    }
                )
            )
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT - 1,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 3
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in captured] == ["false", "false", "true"]
    assert [cfg["benchmark"]["server_lifecycle"]["cleanup"] for cfg in captured] == [False, False, True]
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["accuracy_stage"]["status"] == "succeeded"


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
    shared.baseline_perf = {"output_throughput": 100.0, "total_throughput": 1000.0, "e2e_norm_intvty_p90": 100.0}
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
            else {"output_throughput": 9999.0, "total_token_throughput": 99999.0, "e2e_norm_intvty_p90": 999.0}
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


@pytest.mark.parametrize(
    "output_tput,total_tput,intvty,stack,run_accuracy,gain_pct",
    [
        pytest.param(90.0, 1100.0, 110.0, False, True, 10.0, id="intvty-win-output-drop"),
        pytest.param(100.1, 1507.5, 150.75, True, False, 0.5, id="stack-output-floor-does-not-apply"),
        pytest.param(100.1, 1509.0, 150.9, True, False, 0.6, id="stack-below-primary"),
        pytest.param(100.1, 1507.35, 152.985, True, False, 1.99, id="stack-below-agentx-floor"),
        pytest.param(100.1, 1500.0, 153.0, True, True, 2.0, id="stack-exact-agentx-floor"),
        pytest.param(110.0, 1010.0, 101.0, False, False, 1.0, id="primary-below-agentx-floor"),
        pytest.param(90.0, 950.0, 102.0, False, True, 2.0, id="exact-floor-and-throughput-guard"),
        pytest.param(110.0, 949.9, 110.0, False, False, 10.0, id="throughput-guard-breach"),
        pytest.param(110.0, 1100.0, 90.0, False, False, -10.0, id="interactivity-tradeoff-recorded"),
        pytest.param(110.0, 990.0, 100.0, False, False, 0.0, id="flat-interactivity-recorded"),
        pytest.param(110.0, 900.0, 90.0, False, False, -10.0, id="both-axes-regress"),
    ],
)
def test_deferred_accuracy_keep_policy_uses_graded_performance(
    deferred_accuracy_keep_policy, output_tput, total_tput, intvty, stack, run_accuracy, gain_pct
):
    case = deferred_accuracy_keep_policy
    case.measurement.update(
        output_throughput=output_tput, total_token_throughput=total_tput, e2e_norm_intvty_p90=intvty
    )
    reference = 150.0 if stack else 100.0
    if stack:
        case.shared.current_best.update(action="integrate", total_throughput=1500.0, e2e_norm_intvty_p90=reference)
        case.shared.optimization_stack = [{"kernel_id": "kept-kernel"}]

    result = case.run()

    assert result["status"] == "succeeded", result
    assert case.calls["calls"] == (3 if run_accuracy else 2), result
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in case.captured] == (
        ["false", "false", "true"] if run_accuracy else ["false", "false"]
    )
    assert len({cfg["benchmark"]["envs"]["PORT"] for cfg in case.captured}) == 1
    assert {cfg["benchmark"]["server_lifecycle"]["pid_dir"] for cfg in case.captured} == {str(case.output_dir)}
    assert result["valid_measurement"] is True
    assert result["output_throughput"] == pytest.approx(output_tput)
    assert result["request_throughput"] == pytest.approx(output_tput / 1024)
    assert result["total_token_throughput"] == pytest.approx(total_tput)
    assert result["e2e_norm_intvty_p90"] == pytest.approx(intvty)
    assert result["model"] == "model-measure_round"
    workspace = Path(result["workspace"])
    assert workspace.parent == case.output_dir / "measure_round"
    assert Path(result["report_path"]) == workspace / "benchmark_report.json"
    assert Path(result["raw_result_path"]) == workspace / "inferencex_result.json"
    assert Path(result["launch_evidence_path"]).parent == case.output_dir / "warmup_round"
    assert result["launch_evidence"]["warm_reuse"]["provenance"] == "warmup_round"
    stage = result["accuracy_stage"]
    if run_accuracy:
        assert [cfg["benchmark"]["server_lifecycle"]["cleanup"] for cfg in case.captured] == [False, False, True]
        assert result["accuracy"] == pytest.approx(0.9)
        assert stage["status"] == "succeeded"
        assert Path(stage["workspace"]).parent == case.output_dir / "accuracy_round"
    else:
        assert result.get("accuracy") is None
        assert stage["status"] == "skipped"
        both_axes_regress = intvty < reference * 0.95 and total_tput < (1500.0 if stack else 1000.0) * 0.95
        assert stage["reason"] == ("intvty_regression" if both_axes_regress else "performance_keep_not_eligible")
        assert stage["graded_objective"] == "e2e_norm_intvty_p90"
        assert stage["candidate"] == pytest.approx(intvty)
        assert stage["reference"] == pytest.approx(reference)
        assert stage["gain_pct"] == pytest.approx(gain_pct)
        assert stage["stack_incremental_gain_pct"] == pytest.approx(gain_pct)


@pytest.mark.parametrize("output_tput", [1507.5, 1509.0], ids=["exact-floor", "below-primary"])
def test_deferred_accuracy_keep_policy_allows_output_stack_gain(
    deferred_accuracy_keep_policy, monkeypatch, output_tput
):
    case = deferred_accuracy_keep_policy
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC")
    case.shared.current_best.update(action="integrate", tput=1500.0, output_throughput=1500.0)
    case.shared.optimization_stack = [{"kernel_id": "kept-kernel"}]
    case.params["post_measure_accuracy_keep_policy"]["base_tput"] = 1500.0
    case.params["post_measure_accuracy_min_tput"] = 1515.0
    case.measurement["output_throughput"] = output_tput

    result = case.run()

    assert result["status"] == "succeeded", result
    assert case.calls["calls"] == 3, result
    assert result["output_throughput"] == pytest.approx(output_tput)
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["accuracy_stage"]["status"] == "succeeded"


@pytest.mark.parametrize(
    "fallback",
    ["output-override", "synthetic", "candidate-total", "candidate-intvty", "reference-total", "reference-intvty"],
)
@pytest.mark.parametrize("output_wins", [True, False], ids=["explicit-base-wins", "explicit-base-rejects"])
def test_deferred_accuracy_keep_policy_preserves_output_fallback(
    deferred_accuracy_keep_policy, monkeypatch, fallback, output_wins
):
    case = deferred_accuracy_keep_policy
    case.measurement.update(output_throughput=102.0, total_token_throughput=900.0, e2e_norm_intvty_p90=80.0)
    reference_tput = 120.0 if output_wins else 80.0
    case.shared.current_best.update(tput=reference_tput, output_throughput=reference_tput)
    base_tput = 100.0 if output_wins else 120.0
    run_accuracy = output_wins and fallback in {"output-override", "synthetic"}
    case.params["post_measure_accuracy_keep_policy"]["base_tput"] = base_tput
    if fallback == "output-override":
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    elif fallback == "synthetic":
        monkeypatch.delenv("HYPERLOOM_PERF_METRIC")
    elif fallback == "candidate-total":
        case.measurement.pop("total_token_throughput")
    elif fallback == "candidate-intvty":
        case.measurement.pop("e2e_norm_intvty_p90")
    elif fallback == "reference-total":
        case.shared.current_best.pop("total_throughput")
    else:
        case.shared.current_best.pop("e2e_norm_intvty_p90")

    result = case.run()

    assert result["status"] == "succeeded", result
    assert case.calls["calls"] == (3 if run_accuracy else 2), result
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in case.captured] == (
        ["false", "false", "true"] if run_accuracy else ["false", "false"]
    )
    assert result["output_throughput"] == pytest.approx(102.0)
    stage = result["accuracy_stage"]
    if run_accuracy:
        assert stage["status"] == "succeeded"
        assert result["accuracy"] == pytest.approx(0.9)
    else:
        assert result.get("accuracy") is None
        assert stage["status"] == "skipped"
        assert stage["reason"] == "performance_keep_not_eligible"
        assert stage["graded_objective"] == "output_throughput"
        assert stage["candidate"] == pytest.approx(102.0)
        assert stage["reference"] == pytest.approx(reference_tput)
        assert stage["gain_pct"] == pytest.approx(2.0 if output_wins else -15.0)
        assert stage["degrade_reason"] == (
            "candidate_axes_missing"
            if fallback.startswith("candidate-")
            else "current_best_axes_missing"
            if fallback.startswith("reference-")
            else ""
        )


@pytest.mark.parametrize("missing_from", ["candidate", "reference"])
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize(
    "output,stack",
    [
        pytest.param(200.0, False, id="large-output-gain"),
        pytest.param(50.0, False, id="output-regression"),
        pytest.param(100.75, True, id="stack-output-gain"),
    ],
)
def test_deferred_accuracy_skips_incomparable_performance(
    deferred_accuracy_keep_policy, missing_from, missing_axis, output, stack
):
    case = deferred_accuracy_keep_policy
    case.measurement["output_throughput"] = output
    if stack:
        case.shared.current_best["action"] = "integrate"
        case.shared.optimization_stack = [{"kernel_id": "kept-kernel"}]
    if missing_from == "candidate":
        case.measurement.pop("total_token_throughput" if missing_axis == "total" else "e2e_norm_intvty_p90")
    else:
        case.shared.current_best.pop("total_throughput" if missing_axis == "total" else "e2e_norm_intvty_p90")

    result = case.run()

    assert result["status"] == "succeeded", result
    assert case.calls["calls"] == 2, result
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in case.captured] == ["false", "false"]
    assert result["valid_measurement"] is True
    assert result["output_throughput"] == output
    assert result.get("accuracy") is None
    stage = result["accuracy_stage"]
    assert stage["status"] == "skipped"
    assert stage["reason"] == "performance_keep_not_eligible"
    assert stage["graded_objective"] == "output_throughput"
    assert stage["candidate"] == output
    assert stage["reference"] == 100.0
    assert stage["gain_pct"] == pytest.approx(output - 100.0)
    assert stage["stack_incremental_gain_pct"] == pytest.approx(output - 100.0)
    assert stage["degrade_reason"] == (
        "candidate_axes_missing" if missing_from == "candidate" else "current_best_axes_missing"
    )


@pytest.mark.parametrize("with_policy", [False, True], ids=["legacy", "keep-policy"])
def test_deferred_accuracy_is_cancelled_by_no_eval(tmp_path, with_policy):
    """The staged accuracy round is an eval, so ``--no-eval`` drops it."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT - 1,
        }
    )
    ctx.extra["shared_state"] = SimpleNamespace(eval_disabled=True, baseline_double_run=True)
    if with_policy:
        ctx.task.params["post_measure_accuracy_keep_policy"] = {
            "base_tput": _HOT_TPUT - 1,
            "keep_threshold_pct": 1.0,
            "stack_incremental_keep_threshold_pct": 0.5,
        }

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in captured] == ["false", "false"]
    assert "accuracy_stage" not in result


@pytest.mark.parametrize("with_policy", [False, True], ids=["legacy", "keep-policy"])
def test_deferred_accuracy_single_round_keeps_eval_enabled(tmp_path, with_policy):
    """Ineligible lifecycle fallback must retain accuracy in its only round."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        captured.append(cfg)
        _fake_workspace(slot, tput=_HOT_TPUT)
        (slot / "results_gsm8k.json").write_text(
            json.dumps(
                {
                    "results": {
                        "gsm8k": {
                            "exact_match,strict-match": 0.9,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT - 1,
        }
    )
    if with_policy:
        ctx.task.params["post_measure_accuracy_keep_policy"] = {
            "base_tput": _HOT_TPUT + 1,
            "keep_threshold_pct": 1.0,
            "stack_incremental_keep_threshold_pct": 0.5,
        }

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert len(captured) == 1
    assert captured[0]["benchmark"]["envs"]["RUN_EVAL"] == "true"
    assert result["accuracy"] == pytest.approx(0.9)
    assert "accuracy_stage" not in result


def test_baseline_double_run_by_default(tmp_path, monkeypatch):
    """Baseline defaults to cold+hot rounds to match EXPLORE warm-decision."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", raising=False)
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert result.get("warmup_round_tput") == pytest.approx(_COLD_TPUT)
    assert "baseline_double_run_discarded_first" in result["nonfatal_warnings"]
    assert captured[0]["benchmark"]["server_lifecycle"]["cleanup"] is False
    assert captured[1]["benchmark"]["server_lifecycle"]["cleanup"] is True


def test_replay_warm_recipe_double_run_forces_warmup_eval(tmp_path):
    """Warm replay evaluates in the warmup round and measures in the second."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=SimpleNamespace(baseline_double_run=True),
    )
    task = SimpleNamespace(
        task_id="t-replay-warm",
        kind="replay_warm_recipe",
        params={
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
        },
    )
    ctx = SimpleNamespace(task=task, extra={})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert captured[0]["benchmark"]["envs"]["RUN_EVAL"] == "true"
    assert captured[1]["benchmark"]["envs"]["RUN_EVAL"] == "false"


def test_replay_warm_recipe_honours_no_eval(tmp_path):
    """``--no-eval`` outranks the replay's forced warmup eval."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    shared = SimpleNamespace(baseline_double_run=True, eval_disabled=True)
    executor = BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=shared,
    )
    task = SimpleNamespace(
        task_id="t-replay-no-eval",
        kind="replay_warm_recipe",
        params={
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
        },
    )
    ctx = SimpleNamespace(task=task, extra={"shared_state": shared})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in captured] == [
        "false",
        "false",
    ]


def test_baseline_double_run_can_be_disabled_by_task_param(tmp_path, monkeypatch):
    """Focused callers may explicitly opt out of the default cold+hot baseline."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "baseline_double_run": False,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_double_run_loads_persisted_session_opt_out(tmp_path):
    """A fresh executor process can recover a session-level opt-out from SharedState."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = SharedState.load_or_init(session_dir)
    state.baseline_double_run = False
    state.save(session_dir)

    executor = BaselineExecutor(
        magpie_python=sys.executable,
        session_dir=session_dir,
        shared_state=None,
    )

    assert executor._double_run_enabled() is False


def test_run_grid_discards_cold_first_round_via_lifecycle(tmp_path, monkeypatch):
    """The shared grid runner reports the HOT measured round when lifecycle reuse is eligible."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "grid"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = _run(
            run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant(name="candidate")],
                output_root=output_dir,
                magpie_python=sys.executable,
                variant_timeout_sec=10,
                gpu_type="mi300x",
            )
        )

    assert state["calls"] == 2
    assert len(results) == 1
    result = results[0]
    assert result.status == "succeeded"
    assert result.output_throughput == pytest.approx(_HOT_TPUT)
    assert "run_grid_warmup_discarded_first" in result.nonfatal_warnings

    assert len(captured) == 2
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir / "variant_00_candidate")


def test_run_grid_single_round_when_warmup_disabled(tmp_path, monkeypatch):
    """``INFERENCE_OPTIMIZER_RUN_GRID_WARMUP=0`` keeps the legacy single-round grid path."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "grid"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = _run(
            run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant(name="candidate")],
                output_root=output_dir,
                magpie_python=sys.executable,
                variant_timeout_sec=10,
                gpu_type="mi300x",
            )
        )

    assert state["calls"] == 1
    assert len(results) == 1
    result = results[0]
    assert result.status == "succeeded"
    assert result.output_throughput == pytest.approx(_COLD_TPUT)
    assert "run_grid_warmup_discarded_first" not in result.nonfatal_warnings
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_single_round_when_script_not_builtin(tmp_path):
    """A non-builtin benchmark script falls back to one round even with double-run on."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "benchmark_script": "dsr1_fp8_mi300x.sh",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_warmup_round_failure_short_circuits(tmp_path, monkeypatch):
    """A failed warmup round returns immediately and does NOT run a second round."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        state["calls"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "boom: server crashed")

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert state["calls"] == 1
    assert "baseline_warmup_round_failed" in result.get("nonfatal_warnings", [])


def test_baseline_no_workspace_persists_stderr_to_file(tmp_path):
    """When Magpie exits nonzero before creating a benchmark_* workspace, the executor must persist the captured stderr to ``baseline_stderr.log`` so the failure leaves an on-disk artifact that survives the NFS clone / S3 archive."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"
    crash_text = "torch.OutOfMemoryError: HIP out of memory (workspace_buffer)"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", crash_text)

    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
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
    tmp_path,
    monkeypatch,
):
    """A vLLM engine-core bootstrap failure (server.log carries ``Engine core initialization failed`` while Magpie exits nonzero without a benchmark_* workspace) is classified ``server_init_dead`` with the server.log root cause surfaced in ``error``."""
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
        return subprocess.CompletedProcess(cmd, 1, "", "magpie wrapper noise")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "server_init_dead", result
    assert "Engine core initialization failed" in result["error"]


def test_baseline_server_dead_returncode_classifies_server_init_dead(
    tmp_path,
    monkeypatch,
):
    """When the liveness watchdog reaps a hung server (``SERVER_DEAD_RETURNCODE``), baseline classifies it ``server_init_dead`` even when no server.log marker is independently visible."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        SERVER_DEAD_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, SERVER_DEAD_RETURNCODE, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "server_init_dead", result


def test_baseline_invalid_measurement_with_server_death_marker_is_dead(
    tmp_path,
    monkeypatch,
):
    """When Magpie creates a benchmark_* workspace with no valid measurement, a server.log death marker takes precedence — the failure is classified ``server_init_dead`` and the real engine fault is surfaced in ``error``."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # Workspace exists but has no report, so the measurement is invalid.
        (slot / "benchmark_vllm_20260602_010101").mkdir(parents=True)
        (slot / "server.log").write_text(
            "(APIServer pid=42) RuntimeError: Engine core initialization "
            "failed. See root cause above. Failed core proc(s): {} "
            "OPENAI_API_KEY=ak-invalid-measurement-secret\n",
            encoding="utf-8",
        )
        # Classification must be driven by the server.log marker, not returncode.
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "server_init_dead", result
    assert "Engine core initialization failed" in result["error"]
    assert "invalid-measurement-secret" not in result["error"]
    assert "[REDACTED]" in result["error"]


def test_baseline_clears_stale_server_log_before_run(tmp_path, monkeypatch):
    """A stale server.log death marker in a reused output_dir must NOT bias a fresh attempt's classification."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "server.log").write_text(
        "(APIServer pid=1) RuntimeError: Engine core initialization failed. Failed core proc(s): {}\n",
        encoding="utf-8",
    )

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # Boots fine, produces no report, does NOT rewrite a death marker.
        (slot / "benchmark_vllm_20260602_010101").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] != "server_init_dead", result
    assert result["error_class"] == "no_report", result


def test_baseline_nonzero_rc_with_valid_measurement_fails(tmp_path):
    """A parseable measurement must not launder a non-zero exit code into success."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        ws = slot / "benchmark_vllm_20260602_010101"
        ws.mkdir(parents=True)
        (ws / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "vllm",
                    "throughput": {
                        "output_throughput": 1200.0,
                        "request_throughput": 120.0,
                        "completed_requests": 640,
                        "duration_seconds": 120.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 1, "stdout tail", "server exited 1")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result["error_class"] == "magpie_nonzero_after_valid_measurement", result
    assert result["returncode"] == 1


def test_baseline_rejects_stale_workspace_on_crash(tmp_path, monkeypatch):
    """A stale benchmark_* workspace from a prior attempt must not be adopted as a successful result when the current subprocess crashes (rc=1)."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "throughput": {
                    "output_throughput": 9999.0,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
            }
        ),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "HIP out of memory")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_rejects_stale_workspace_on_silent_exit(tmp_path, monkeypatch):
    """A stale benchmark_* workspace must not be adopted when the subprocess exits 0 without producing any new workspace (silent no-op)."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "throughput": {
                    "output_throughput": 9999.0,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
            }
        ),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_rejects_stale_workspace_when_the_run_produced_none(tmp_path, monkeypatch):
    """A crashed run adopts no workspace, however high the stale one sorts."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_29991231_235959"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "throughput": {
                    "output_throughput": 9999.0,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
            }
        ),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_picks_fresh_workspace_sorting_before_a_stale_one(tmp_path, monkeypatch):
    """The fresh workspace wins even when the stale one sorts last."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_29991231_235959"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "throughput": {"output_throughput": 9999.0, "completed_requests": 64},
            }
        ),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]), tput=4000.0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded", result
    assert result.get("output_throughput") == pytest.approx(4000.0)


def test_baseline_fresh_workspace_succeeds_despite_stale_peer(tmp_path, monkeypatch):
    """A new workspace with valid throughput produced by the current run succeeds even when an older stale workspace is present in the same output_dir."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "throughput": {"output_throughput": 1.0, "completed_requests": 1},
            }
        ),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]), tput=4000.0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded", result
    assert result.get("output_throughput") == pytest.approx(4000.0)


def test_baseline_anchors_server_cwd_to_output_dir(tmp_path, monkeypatch):
    """The Magpie parent subprocess cwd is anchored to the stable task output_dir (never the default ``/tmp``) as defence-in-depth."""
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

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert seen["cwd"] is not None
    assert seen["cwd"] != "/tmp"
    assert str(output_dir) in seen["cwd"]


def test_atom_engages_double_run_like_vllm_sglang(tmp_path, monkeypatch):
    """Atom baseline engages the lifecycle double-run like vllm/sglang."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="atom")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert captured[0]["benchmark"]["benchmark_script"] == "atom_mi300x.sh"
    assert captured[0]["benchmark"]["server_lifecycle"]["enabled"] is True


def test_double_run_runtime_anchor_is_full_warmup_round(tmp_path, monkeypatch):
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

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["subprocess_runtime_sec"] >= 0.5
    assert "measure_round_runtime_sec" in result
    assert result["measure_round_runtime_sec"] < result["subprocess_runtime_sec"]


def test_pre_start_cleanup_unlinks_meta_and_kills_unconditionally(tmp_path, monkeypatch):
    """Pre-start cleanup no longer probes port health: it unconditionally (a) unlinks stale pid/json without sending signals to potentially-recycled PIDs, and (b) invokes _kill_stale_servers() -- Hyperloom's own scheduling (gpu_research_lane, capacity 1) guarantees nothing matching should be alive at this point, so no extra evidence is required before reaping."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    pid_file = output_dir / "vllm_8888.pid"
    meta_file = output_dir / "vllm_8888.json"
    pid_file.write_text("2147483646")
    meta_file.write_text("{}")

    executor = _executor(tmp_path / "base.yaml", tmp_path)
    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
        side_effect=fake_kill,
    ):
        _run(
            executor._pre_start_cleanup(
                pid_dir=output_dir,
                framework="vllm",
                port=8888,
            )
        )

    assert kill_calls["n"] == 1
    assert not pid_file.exists()
    assert not meta_file.exists()


def test_pre_start_cleanup_skipped_under_pytest(tmp_path):
    """Direct guard: _kill_stale_servers must NOT fire while ``PYTEST_CURRENT_TEST`` is set (pytest always sets it for a running test), mirroring the same guard on the per-launch preclean in ``_grid_runner.py``."""
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    pid_file = output_dir / "vllm_8888.pid"
    meta_file = output_dir / "vllm_8888.json"
    pid_file.write_text("2147483646")
    meta_file.write_text("{}")

    executor = _executor(tmp_path / "base.yaml", tmp_path)
    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
        side_effect=fake_kill,
    ):
        _run(
            executor._pre_start_cleanup(
                pid_dir=output_dir,
                framework="vllm",
                port=8888,
            )
        )

    assert kill_calls["n"] == 0, "must be a no-op while PYTEST_CURRENT_TEST is set"
    assert not pid_file.exists()
    assert not meta_file.exists()


def test_pre_start_cleanup_failure_does_not_break_double_run(tmp_path, monkeypatch):
    """The pre-start cleanup is best-effort: a raising _kill_stale_servers() must not propagate out of _pre_start_cleanup() itself."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)

    def boom():
        raise RuntimeError("proc scan blew up")

    executor = _executor(tmp_path / "base.yaml", tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
        side_effect=boom,
    ):
        _run(
            executor._pre_start_cleanup(
                pid_dir=output_dir,
                framework="vllm",
                port=8888,
            )
        )
    # No exception propagated past _pre_start_cleanup: that's the assertion.


def test_pre_start_cleanup_skipped_when_round_is_not_affordable(tmp_path):
    """The pre-start cleanup must not pay its cost for a round the budget gate is about to refuse: it now runs right before the round actually boots (after the affordability check and the Ray lease construction), not up front where an unaffordable round would still have paid for a scan it gets no benefit from (review on AMD-AGI/Hyperloom#1354)."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    cleanup_calls: list[dict] = []

    async def fake_cleanup(**kwargs):
        cleanup_calls.append(kwargs)

    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with (
        patch.object(type(executor), "_pre_start_cleanup", side_effect=fake_cleanup),
        patch.object(
            type(executor),
            "_round_affordable_before_ignition",
            return_value=(False, {"expected_cost_sec": 999.0, "affordable_sec": 0.0, "bound": "session"}),
        ),
    ):
        result = _run(executor(ctx))

    assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
    assert cleanup_calls == []


@pytest.mark.parametrize("baseline_double_run", [True, False])
def test_pre_start_cleanup_called_once_regardless_of_double_run(tmp_path, baseline_double_run):
    """The pre-start deep clean must run exactly once before the round(s) boot, whether this is a double-run or a single round."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    cleanup_calls: list[dict] = []

    async def fake_cleanup(**kwargs):
        cleanup_calls.append(kwargs)

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=baseline_double_run)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "baseline_double_run": baseline_double_run,
        }
    )

    with (
        patch.object(type(executor), "_pre_start_cleanup", side_effect=fake_cleanup),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert len(cleanup_calls) == 1
    assert state["calls"] == (2 if baseline_double_run else 1)


def test_teardown_lifecycle_server_removes_state_files(tmp_path):
    """The defensive teardown unlinks stale pid/meta files without raising."""
    executor = _executor(tmp_path / "base.yaml", tmp_path)
    _write_yaml(tmp_path / "base.yaml", framework="vllm")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (pid_dir / "vllm_8888.pid").write_text("2147483646")
    (pid_dir / "vllm_8888.json").write_text("{}")

    executor._teardown_lifecycle_server(
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
    monkeypatch.setattr(sl, "_signal_group", lambda pgid, sig: signals_sent.append(sig))
    monkeypatch.setattr(sl, "_looks_like_server_process", lambda pid: False)

    sl.teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=8888)

    assert signals_sent == [], "must not signal a pid that failed the identity check"
    assert not (pid_dir / "vllm_8888.pid").exists(), "stale pid file must still be removed"
    assert not (pid_dir / "vllm_8888.json").exists(), "stale meta file must still be removed"


# Every output slot a multi-node baseline round launches a benchmark process into, in launch order.
_MEASURED_ROUND_SLOT = "measured_round"
_BASELINE_ROUND_SLOTS = ("mn_warmup", _MEASURED_ROUND_SLOT)


# ``--max-hours`` defaults to 2.0 (``cli/parser.py``), so this is the session shape almost every run has.
_DEFAULT_SESSION_MINUTES = 120.0


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


@contextmanager
def _passes_time_the_executor_believes(clock: _AClockOnlyThePassesMove):
    """Run the block with ``clock`` standing in for the wall clock."""
    with patch("time.time", clock):
        yield


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


class _CapturingLease:
    """A serving-lease stand-in that records how a round reached the Ray actor."""

    def __init__(self) -> None:
        self._run, self.calls = _capturing_fake_run()

    def run_session_kill(self, cmd, **kwargs) -> tuple[int | None, str, str]:
        """Record the launch and answer as the actor does, with a bare triple."""
        proc = self._run(cmd, **kwargs)
        return proc.returncode, proc.stdout, proc.stderr

    def close(self) -> None:
        """The executor closes the lease it was given; nothing is held here."""


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
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / _MEASURED_ROUND_SLOT),
            "timeout_sec": timeout_sec,
            "gpu_type": "mi300x",
        }
    )
    # The live state arrives on the context, the way the coordinator passes it.
    ctx.extra["shared_state"] = state
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))
    return result, calls


def _mn_warmup_cap_sec(calls: list[dict]) -> int | None:
    """The cap the multi-node warmup pass was granted, or ``None`` when it never ran."""
    launch = launches_by_round_slot(calls).get("mn_warmup")
    return None if launch is None else int(launch["timeout"])


def _launch_one_grid_variant_under_budget(
    tmp_path,
    *,
    remaining_sec: float,
    variant_timeout_sec: int,
    variant_expected_sec: float,
) -> list[dict]:
    """Run one grid variant against the same budget a baseline round would get."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    fake_run, calls = _capturing_fake_run()
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        _run(
            run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant(name="candidate")],
                output_root=tmp_path / "out",
                magpie_python=sys.executable,
                variant_timeout_sec=variant_timeout_sec,
                session_deadline_sec=time.monotonic() + remaining_sec,
                variant_expected_sec=variant_expected_sec,
            )
        )
    return calls


class TestTheSessionBudgetReachesTheBaselineRound:
    """The arm #1146 names as the largest hole, and the one that motivated it."""

    @pytest.mark.parametrize("round_slot", _BASELINE_ROUND_SLOTS)
    def test_the_deadline_reaches_the_reaper(self, tmp_path, monkeypatch, round_slot):
        """Parameterized over the passes a round launches, not just the measured one."""
        enable_multi_node(monkeypatch)
        _result, calls = _run_baseline_under_budget(tmp_path, remaining_sec=3600.0)

        assert launches_by_round_slot(calls)[round_slot]["session_deadline_sec"] is not None

    def test_no_pass_of_a_round_is_launched_without_the_deadline(self, tmp_path, monkeypatch):
        """The net for a pass added later, which no per-slot test would know about."""
        enable_multi_node(monkeypatch)
        _result, calls = _run_baseline_under_budget(tmp_path, remaining_sec=3600.0)

        assert set(launches_by_round_slot(calls)) >= set(_BASELINE_ROUND_SLOTS)
        assert [c["round_slot"] for c in calls if c.get("session_deadline_sec") is None] == []

    def test_the_ray_path_is_handed_the_budget_as_a_duration(self, tmp_path, monkeypatch):
        """The fourth launch site, and the one a parameterization cannot reach."""
        from hyperloom.orchestrator.actions.executors import _ray_serving

        lease = _CapturingLease()
        monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **_kwargs: lease)
        result, _calls = _run_baseline_under_budget(tmp_path, remaining_sec=3600.0)

        assert result["status"] == "succeeded"
        launch = launches_by_round_slot(lease.calls)[_MEASURED_ROUND_SLOT]
        remaining = launch.get("session_remaining_sec")
        assert remaining is not None, f"the budget did not cross the process boundary: {sorted(launch)}"
        assert 0 < remaining <= 3600.0

    def test_the_hang_backstop_is_clamped_to_what_is_left(self, tmp_path):
        """A cap larger than the budget outlives the session it belongs to."""
        _result, calls = _run_baseline_under_budget(tmp_path, remaining_sec=120.0, timeout_sec=7200)

        assert 1 <= calls[0]["timeout"] <= 120 + _SESSION_KILL_GRACE_SEC

    def test_an_unbounded_budget_leaves_the_cap_alone(self, tmp_path):
        """No session context means no budget to respect, not a budget of zero."""
        _result, calls = _run_baseline_under_budget(tmp_path, remaining_sec=None, timeout_sec=7200)

        assert calls[0]["timeout"] == 7200
        assert calls[0]["session_deadline_sec"] is None

    def test_a_budget_kill_is_not_recorded_as_a_broken_model(self, tmp_path):
        """A reaped round leaves exactly what a broken server leaves behind."""
        result, _calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=1.0,
            returncode=SESSION_TIME_EXHAUSTED_RETURNCODE,
            produces_workspace=False,
        )

        assert result["status"] == "failed"
        assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS

    def test_a_cancel_is_told_apart_from_a_spent_budget(self, tmp_path):
        """A resume meets the spent budget again and does not meet the shutdown."""
        result, _calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=3600.0,
            returncode=ORCHESTRATOR_CANCELLED_RETURNCODE,
            produces_workspace=False,
        )

        assert result["error_class"] == ORCHESTRATOR_CANCELLED_CLASS

    def test_a_cancelled_multi_node_warmup_does_not_go_on_to_the_measured_round(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The discarded warmup is a full pass, so a cancel there ends the round."""
        base = tmp_path / "base.yaml"
        _write_yaml(base, framework="vllm")
        enable_multi_node(monkeypatch)
        launched: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            slot = Path(cmd[cmd.index("--output-dir") + 1])
            launched.append(slot.name)
            if slot.name == "mn_warmup":
                return subprocess.CompletedProcess(cmd, ORCHESTRATOR_CANCELLED_RETURNCODE, "", "")
            _fake_workspace(slot, tput=_HOT_TPUT)
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        executor = BaselineExecutor(
            magpie_python=sys.executable,
            default_config_path=base,
            session_dir=tmp_path,
        )
        ctx = _make_ctx(
            {
                "output_dir": str(tmp_path / "ws"),
                "timeout_sec": 600,
                "gpu_type": "mi300x",
            }
        )
        with patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ):
            result = _run(executor(ctx))

        assert launched == ["mn_warmup"], f"the measured round ran after the cancel: {launched}"
        assert result["error_class"] == ORCHESTRATOR_CANCELLED_CLASS

    def test_a_first_baseline_on_a_default_session_runs_both_of_its_passes(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The regime several rounds of this mechanism have failed in, pinned directly."""
        enable_multi_node(monkeypatch)
        pass_sec = 600.0
        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=_DEFAULT_SESSION_MINUTES * 60.0,
            timeout_sec=BASELINE_DEFAULT_TIMEOUT_SEC,
            pass_duration_sec=pass_sec,
        )
        launches = launches_by_round_slot(calls)

        assert result["status"] == "succeeded", f"a first baseline was refused a default session: {result}"
        assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
        assert set(launches) >= set(_BASELINE_ROUND_SLOTS), f"the round did not run both passes: {list(launches)}"
        remaining_sec = _DEFAULT_SESSION_MINUTES * 60.0
        warmup = _mn_warmup_cap_sec(calls)
        assert warmup is not None and warmup >= remaining_sec, (
            f"the warmup's cap was moved in front of the session deadline, so the "
            f"pass can now be killed by its own timeout before the watchdog reaches "
            f"it: {warmup}s against {remaining_sec}s left"
        )

    def test_a_multi_node_round_that_cannot_pay_for_both_passes_launches_neither(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A multi-node round is two client passes, and the figure covers one."""
        enable_multi_node(monkeypatch)

        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=1500.0,
            cold_round_sec=600.0,
        )

        assert calls == [], "a multi-node pair ran on a budget that covers one pass"
        assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
        assert result["returncode"] is None
        assert result["budget_shortfall"]["round_sec"] == pytest.approx(1200.0)

    def test_a_multi_node_round_the_budget_covers_runs_both_passes(self, tmp_path, monkeypatch):
        """The gate must not refuse the pairs that fit, only the ones that cannot."""
        enable_multi_node(monkeypatch)

        result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=3600.0,
            cold_round_sec=600.0,
        )

        assert set(launches_by_round_slot(calls)) >= set(_BASELINE_ROUND_SLOTS)
        assert result["status"] == "succeeded"

    def test_the_profile_arm_gets_all_of_it(self, tmp_path):
        """Profile is the same executor with a four-hour default -- longer than any session budget it could be given."""
        _result, calls = _run_baseline_under_budget(
            tmp_path,
            remaining_sec=120.0,
            timeout_sec=PROFILE_DEFAULT_TIMEOUT_SEC,
            executor_cls=ProfileExecutor,
        )

        assert calls[0]["session_deadline_sec"] is not None
        assert 1 <= calls[0]["timeout"] <= 120 + _SESSION_KILL_GRACE_SEC


# _classify_subprocess_error unit tests

from hyperloom.orchestrator.actions.executors.baseline import (
    _classify_subprocess_error,
)


def test_classify_fast_exit_unknown_backend():
    assert (
        _classify_subprocess_error(5.0, "ValueError: Unknown attention backend: 'ROCM_FLASH'") == "fast_exit_arg_error"
    )


def test_classify_fast_exit_unrecognized_args():
    assert _classify_subprocess_error(2.0, "error: unrecognized arguments: --bogus-flag") == "fast_exit_arg_error"


def test_classify_slow_failure_not_arg_error():
    """A slow failure (>30s) with the same stderr pattern must NOT be classified as arg error — it could be a real inference crash."""
    assert _classify_subprocess_error(120.0, "ValueError: some runtime error") == "subprocess_nonzero"


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
