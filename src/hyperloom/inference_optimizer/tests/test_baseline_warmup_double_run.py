# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the baseline cold-start "warmup artifact".

Covers the cold+hot double-run and its server-lifecycle reuse, the pre-start /
teardown cleanup around the reused port, the local InferenceX mirror, the
subprocess-failure classifier, and the session wall-clock budget's reach into
the round (the deadline the reaper is handed, the clamp on the hang backstop,
and how a round the run stopped is told apart from one that failed).
"""

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
    _stamp_server_ready,
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
    """Return a ``run_with_session_kill`` stand-in that emits a cold throughput
    on its first call and a hot throughput thereafter.

    Given a ``clock``, the first call spends it in the two parts a cold pass
    spends it in and announces the server ready between them; later calls
    re-attach, so they spend only the benchmark and announce nothing.
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
        if clock is not None:
            server_log_path = kwargs.get("server_log_path")
            if state["calls"] == 0:
                clock.advance(boot_sec)
                if server_log_path:
                    Path(server_log_path).parent.mkdir(parents=True, exist_ok=True)
                    _stamp_server_ready(server_log_path, boot_sec)
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
    """A session state with an explicit clock, as the budget policy reads it.

    Only the usable remainder, because that is all the policy reads. An earlier
    version also carried a phase ledger and a phase start, from when preparation
    answered to a share of its own; a double that still offers them invites a
    reader to believe they decide something.

    The phase is offered because the gates ask it one thing: whether the round's
    worth depends on a variant following it.
    """
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
    """A PRELUDE session whose remaining budget falls as the passes spend it.

    The fixed-remainder double cannot show the two gates disagreeing, because the
    second one is asked after the warmup has spent its share and a budget that
    never moves hides exactly that. This reads the same clock the passes move.
    """
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
    """The session clock cannot pay for the hot pass and a use for it; the cold one ran.

    Nothing is predicted before the warmup on a first baseline, so the round
    starts and the warmup's GPU time is spent before the shortfall is known.
    Refusing to keep its figure would throw that away and leave the session with
    no anchor at all, which is strictly worse than the cold anchor a single-round
    baseline would have produced. So the warmup is promoted and marked: the
    number is depressed, and the marker is what tells a reader of the session's
    later gains that their denominator is.

    The warmup boots for 350s and benchmarks for 550s, so the hot pass that would
    follow costs 550s and a variant to read against it costs 900s. 1200s covers
    the pass alone with room to spare and the pair not at all.
    """
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
    """The two gates price the same second pass, so they must reach the same answer.

    A gate before ignition that admits what the gate after the cold pass will
    certainly refuse spends a whole cold pass to learn something it already knew.
    The disagreement is in the ruler: this session has measured a 400s hot pass,
    and pricing the pass to come at the warmup's 550s post-ready segment instead
    -- a segment that also paid the first request's compile -- demands 300s more
    than ignition was allowed to require.

    2100s is inside that band. Ignition needs the 1300s round and a 750s variant;
    after the warmup spends 900s, the hot pass and its variant need 1150s of the
    1200s left, while the post-ready ruler would have called for 1450s.
    """
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
    """Agreeing with the earlier gate is not the same as admitting everything.

    Once the two price the same work, the only rounds left for this gate to
    refuse are the ones that cost more than they were admitted on -- which is
    precisely what a gate asked after the pass, against the clock rather than
    against a prediction, exists for. This round was admitted at 2050s and its
    warmup then took 1250s instead of 900s, leaving 850s where 1150s is needed.
    """
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
    """The same 1200s that drops the hot pass in PRELUDE runs it here.

    In PRELUDE the hot pass buys a denominator, so a session that cannot follow
    it with a variant gains nothing by running it. A re-baseline's hot pass is
    the measurement the session came for, and covering it is the whole question.
    """
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
    """A later phase asks a narrower question, not no question.

    550s of benchmarking does not fit in 400s, so the pass would be reaped
    mid-flight and the warmup's figure lost with it.
    """
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
    """The gate prices the round on what it spent, not on what its cap allowed.

    A warmup can keep well inside its own timeout and still leave the round unable
    to pay for what should follow: the server boot in front of the pass is
    wall-clock the cap never bounded. Asking after the pass, against the clock
    rather than against the cap, is what catches that -- and is why nothing is
    predicted before the pass on a first baseline instead.

    The cap here is 10 seconds and the pass spends 2600 of them, 1400 of which is
    the boot. A gate priced on the cap would have waved through a round costing
    three hundred times what it was allowed.
    """
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
    """A double run whose warmup lands and whose measured pass the clock takes.

    The regime a prediction cannot rule out: the gate before the measured pass
    admitted it on what the warmup had just cost, and the pass overran that.
    """
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
    """The GPU time behind the warmup's figure is spent either way.

    Reporting the round as failed would discard it and leave the session with
    nothing to show for a pass that ran to completion, so the warmup is kept and
    marked exactly as a refused measured round keeps it. A session that reaches
    this state has already paid for a cold anchor; what it must not do is pay
    again for nothing.
    """
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
    """Only the run's clock earns the fallback.

    A pass that broke for a reason of its own is a fact about the configuration,
    and the warmup having succeeded does not make the round's figure comparable.
    Promoting a cold anchor here would bury a real failure under a warning.
    """
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
    """The round's total and the part of it that was the benchmark, from one pass.

    Round 1 boots for 350s and benchmarks for 550s; round 2 re-attaches and
    benchmarks for 550s more. The split belongs to round 1, whose 900s total is
    also what the round reports: their difference is published as what booting
    this workload costs, so two rounds cannot each supply one of them. Round 2's
    own split says nothing, having never booted.
    """
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


# The workload every case in the gate class below is priced against, and the
# figures the pricing derives from it. A round that boots for 350s and then
# benchmarks is what a variant costs too, because a variant's config differs in
# the knobs that decide how a server comes up and it has to bring up its own.
_COLD_ROUND_SEC = 900.0
_COLD_POST_READY_SEC = 550.0
_HOT_ROUND_SEC = 400.0
# 900 - 550: the part of the cold round that was not the benchmark.
_BOOT_SEC = 350.0
# One further measured variant: its own boot, then its own benchmark. The
# benchmark is priced hot because the variant runs on a JIT cache this session
# has already populated.
_ONE_MORE_SEC = _BOOT_SEC + _HOT_ROUND_SEC
# A round's first pass is the cold one, measured whole rather than rebuilt from
# its halves -- rebuilding it as boot-plus-hot would drop the compile it paid and
# under-price the round by 150s. A double run adds a second benchmark, which
# re-attaches and so buys no second boot.
_SINGLE_ROUND_SEC = _COLD_ROUND_SEC
_DOUBLE_ROUND_SEC = _COLD_ROUND_SEC + _HOT_ROUND_SEC


class TestARoundThatCannotFinishIsNotIgnited:
    """The gate in front of a round, and the two things it must be asked with.

    A round is refused before it boots only on what earlier rounds measured, so
    the session's first one is never refused: it has nothing to be judged by, and
    a gate that guessed would either refuse every first baseline or wave every
    one through. From the second on -- and on a resumed session's first, which
    carries the earlier leg's figures -- the answer is available before a second
    of GPU time is spent.

    What must fit is the round *and one further measured variant*. A baseline is
    not a result; it is the denominator results are read against and the anchor
    their overtime kill uses. A round no variant can follow buys neither, so the
    wall-clock it would spend produces nothing.

    Priced on this workload: boot 350s, benchmark 400s, so one variant costs 750s,
    a single-pass round costs 750s and a double-run round 1150s.
    """

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
        """The gate must not turn a merely expensive round into a refused one.

        Given a margin over the 1500s the round and its use need, rather than
        exactly that: the headroom is read from a live clock, so a case pinned to
        the boundary would decide on how long the test itself took to get there.
        """
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
        """The requirement that is not about finishing the round.

        1000s covers the 750s round with room to spare, and the round would run
        to completion. It is still refused, because what it produces is a
        denominator, and 250s buys no variant to read against it. Wall-clock
        spent on a number nothing is ever compared to is wall-clock wasted, and
        a session that stops here keeps the anchor it already had.
        """
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
        """The same 1000s that refuses a PRELUDE round admits this one.

        A re-baseline re-measures the stack the session has assembled, and that
        measurement is what the session is for. Requiring a variant after it
        would refuse the round that validates the run's own answer, at the point
        in the budget where it is most likely to be the last thing left.
        """
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
        """Dropping the successor does not drop the round's own cost.

        A later phase is a narrower question, not an absent one: a round that
        cannot finish inside the budget burns a boot and a compile for a number
        the reaper takes away before it lands.
        """
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
        """One budget, two answers: it covers a single-pass round, not a double one.

        The difference between them is one benchmark and no second boot, because
        the second pass re-attaches to the server the first left running. Both
        sides run against the same figure so the refusal can only come from that.
        """
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
        """The state a previous cold-anchor drop leaves, and the one to catch.

        A round whose measured pass was dropped for budget promotes its cold
        number and has no hot number to write. Going inert on such a session would
        exempt exactly the one that already ran out of budget once, so the cold
        round's post-ready segment stands in for the variant's benchmark. It
        over-predicts, having also paid the first request's compile, which is why
        the hot figure wins whenever one exists.
        """
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
        """A round with no split is priced at whole cold rounds, not waved through.

        Multi-node and scriptable workloads never report a boot boundary -- one
        brings its server up outside the round, the other runs no server at all --
        so a gate that goes inert without the split exempts them permanently. Both
        terms fall back to what the session did measure: the cold round's own
        wall-clock, which is a boot and a benchmark, and so is what a variant
        costs too.
        """
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
        """The property the whole gate rests on, asserted rather than assumed.

        A refusal is only cheap if what the session already measured survives it.
        The anchor is held in session state, not in this result, so the refused
        round must come back with no throughput of its own to promote over it.
        """
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


def test_deferred_accuracy_is_cancelled_by_no_eval(tmp_path):
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

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in captured] == ["false", "false"]
    assert "accuracy_stage" not in result


def test_deferred_accuracy_single_round_keeps_eval_enabled(tmp_path):
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
    """``--no-eval`` outranks the replay's forced warmup eval.

    The flag is the operator saying no eval runs this session. Forcing one on
    the warmup round would spend the time the flag was passed to save, and do
    it silently -- the baseline path on the same executor already honours the
    flag, so a replay that did not would be the odd one out.
    """
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
    """When Magpie exits nonzero before creating a benchmark_* workspace, the
    executor must persist the captured stderr to ``baseline_stderr.log`` so the
    failure leaves an on-disk artifact that survives the NFS clone / S3 archive."""
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
    """A vLLM engine-core bootstrap failure (server.log carries ``Engine core
    initialization failed`` while Magpie exits nonzero without a benchmark_*
    workspace) is classified ``server_init_dead`` with the server.log root cause
    surfaced in ``error``."""
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
    """When the liveness watchdog reaps a hung server
    (``SERVER_DEAD_RETURNCODE``), baseline classifies it ``server_init_dead``
    even when no server.log marker is independently visible."""
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
    """When Magpie creates a benchmark_* workspace with no valid measurement, a
    server.log death marker takes precedence — the failure is classified
    ``server_init_dead`` and the real engine fault is surfaced in ``error``."""
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
    """A stale server.log death marker in a reused output_dir must NOT bias a
    fresh attempt's classification. The executor clears the prior log before
    launching, so an attempt that boots but yields no report is classified by
    its own outcome (``no_report``), never ``server_init_dead``."""
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
    """A stale benchmark_* workspace from a prior attempt must not be adopted
    as a successful result when the current subprocess crashes (rc=1)."""
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
    """A stale benchmark_* workspace must not be adopted when the subprocess
    exits 0 without producing any new workspace (silent no-op)."""
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
    """The fresh workspace wins even when the stale one sorts last.

    Every other case here creates a fresh name that also sorts last, so they
    pass against the lexicographic-last glob this replaced.
    """
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
    """A new workspace with valid throughput produced by the current run succeeds
    even when an older stale workspace is present in the same output_dir."""
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


def test_ensure_local_inferencex_noop_for_local_path(tmp_path, monkeypatch):
    """A checkout already on a local filesystem is returned unchanged."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    monkeypatch.setattr(bl, "is_network_fs", lambda p: False)

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_mirrors_network_path(tmp_path, monkeypatch):
    """A checkout on a simulated network mount is mirrored to local disk and the
    returned path points at the local copy, not the original."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched lib")
    (src / "utils").mkdir()
    (src / "utils" / "marker.txt").write_text("payload")

    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    dest = bl._ensure_local_inferencex(str(src))

    assert dest != str(src)
    assert str(local_root) in dest
    assert (Path(dest) / "benchmarks" / "benchmark_lib.sh").read_text() == ("# patched lib")
    assert (Path(dest) / "utils" / "marker.txt").read_text() == "payload"


def test_ensure_local_inferencex_isolates_per_task_mirrors(
    tmp_path,
    monkeypatch,
):
    """Callers can include a task/output-dir key in the mirror hash so two
    overlapping baselines sharing one wekafs checkout never rmtree/replace a
    directory that another server is currently ``cd``-ed into."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched lib")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    dest_a = bl._ensure_local_inferencex(str(src), mirror_key="task-a")
    dest_b = bl._ensure_local_inferencex(str(src), mirror_key="task-b")

    assert dest_a != dest_b
    assert (Path(dest_a) / "benchmarks" / "benchmark_lib.sh").is_file()
    assert (Path(dest_b) / "benchmarks" / "benchmark_lib.sh").is_file()


def test_ensure_local_inferencex_disabled_by_env(tmp_path, monkeypatch):
    """The relocation can be opted out of via env even on a network mount."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    monkeypatch.setattr(bl, "is_network_fs", lambda p: True)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX", "1")

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_falls_back_on_copy_failure(
    tmp_path,
    monkeypatch,
):
    """When the mirror copy itself fails (e.g. local disk full), the helper
    degrades to the original network-mount path instead of raising, so the run
    still proceeds rather than aborting."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    def _boom(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(bl.shutil, "copytree", _boom)

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_falls_back_when_mirror_incomplete(
    tmp_path,
    monkeypatch,
):
    """If the copy lands but the mirror is missing the load-bearing
    ``benchmarks/benchmark_lib.sh``, the helper rejects it and returns the
    original path rather than handing Magpie a broken ``cd`` target."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "utils").mkdir(parents=True)
    (src / "utils" / "marker.txt").write_text("payload")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    assert bl._ensure_local_inferencex(str(src)) == str(src)
    assert not [p for p in local_root.iterdir() if p.is_dir()]


def test_baseline_points_magpie_at_local_inferencex(tmp_path, monkeypatch):
    """When INFERENCEX_PATH is on a network mount, the local mirror is what
    Magpie actually ``cd``-s into. Asserts both channels:

    * the materialized YAML's ``benchmark.inferencex_path`` (the field Magpie's
      ``_build_local_command`` honours — the real ``cd`` target), and
    * the ``MAGPIE_INFERENCEX_PATH`` env fallback.
    """
    from hyperloom.orchestrator.actions.executors import baseline as bl

    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"

    ix_src = tmp_path / "wekafs_InferenceX"
    (ix_src / "benchmarks").mkdir(parents=True)
    # This test is about which InferenceX dir Magpie cd-s into, but the launch
    # path runs the real patcher, which refuses to start an eval whose patches
    # cannot be applied. So the stub has to carry the anchors a checkout carries.
    (ix_src / "benchmarks" / "benchmark_lib.sh").write_text(
        "# patched\n"
        "run_eval() {\n"
        '    export EVAL_RESULT_DIR="$results_dir"\n'
        "}\n"
        "append_lm_eval_summary() {\n"
        '    mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2\n'
        "}\n"
    )
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "is_network_fs", lambda p: True)
    monkeypatch.setenv("INFERENCEX_PATH", str(ix_src))
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    seen: dict = {}

    def fake_run(cmd, *args, **kwargs):
        seen["env"] = kwargs.get("env")
        cfg_idx = cmd.index("--benchmark-config")
        seen["materialized_cfg"] = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
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
    yaml_ix = seen["materialized_cfg"]["benchmark"]["inferencex_path"]
    assert yaml_ix != str(ix_src), seen["materialized_cfg"]
    assert str(local_root) in yaml_ix
    magpie_ix = seen["env"]["MAGPIE_INFERENCEX_PATH"]
    assert magpie_ix != str(ix_src), seen["env"]
    assert str(local_root) in magpie_ix
    # Relocation is task-local; process-wide env stays the original source path.
    assert os.environ["INFERENCEX_PATH"] == str(ix_src)


def test_baseline_anchors_server_cwd_to_output_dir(tmp_path, monkeypatch):
    """The Magpie parent subprocess cwd is anchored to the stable task
    output_dir (never the default ``/tmp``) as defence-in-depth."""
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


def test_double_run_pre_start_cleanup_kills_zombie_and_clears_stale_meta(
    tmp_path,
    monkeypatch,
):
    """When the reuse port is occupied by a zombie (healthy but no metadata),
    pre-start cleanup must (a) unlink stale pid/json without sending signals to
    potentially-recycled PIDs, and (b) invoke _kill_stale_servers() to reap the
    zombie listener."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "vllm_8888.pid").write_text("2147483646")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
        lambda: 8888,
    )

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

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

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert kill_calls["n"] == 1
    assert not (output_dir / "vllm_8888.pid").exists()
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir)


def test_pre_start_cleanup_no_kill_when_port_free(tmp_path, monkeypatch):
    """When the port is NOT occupied (no zombie), _kill_stale_servers must
    NOT fire — avoids killing unrelated servers sharing the pod."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "vllm_8888.pid").write_text("2147483646")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
        lambda: 8888,
    )

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

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

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=False,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert kill_calls["n"] == 0
    assert not (output_dir / "vllm_8888.pid").exists()
    assert state["calls"] == 2


def test_pre_start_cleanup_no_kill_when_metadata_existed(tmp_path, monkeypatch):
    """A healthy port with matching metadata is not a zombie signal.

    The global stale-server reaper must not fire for a likely legitimate
    server. File preservation is covered by the direct pre-start test below;
    this full double-run path later removes files in final teardown.
    """
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "vllm_8888.pid").write_text("2147483646")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
        lambda: 8888,
    )
    (output_dir / "vllm_8888.json").write_text("{}")

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

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

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert kill_calls["n"] == 0
    assert state["calls"] == 2


def test_pre_start_cleanup_preserves_metadata_when_reuse_target_healthy(
    tmp_path,
):
    """Direct guard: do not create port-occupied/no-metadata state."""
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

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
    ):
        executor._pre_start_cleanup(
            pid_dir=output_dir,
            framework="vllm",
            port=8888,
        )

    assert kill_calls["n"] == 0
    assert pid_file.exists()
    assert meta_file.exists()


def test_pre_start_cleanup_failure_does_not_break_double_run(tmp_path, monkeypatch):
    """The pre-start cleanup is best-effort: a raising _kill_stale_servers()
    must not abort the run — the double-run proceeds and still succeeds."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    def boom():
        raise RuntimeError("proc scan blew up")

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

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=boom,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2


def test_pre_start_cleanup_skipped_when_single_round(tmp_path, monkeypatch):
    """Single-round (double-run disabled) keeps legacy behaviour: the
    pre-start deep clean is a double-run-only concern and must not fire."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

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

    with (
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert kill_calls["n"] == 0


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


# Every output slot a multi-node baseline round launches a benchmark process
# into, in launch order. The discarded client warmup is a full pass and costs the
# same wall-clock as the measured round it precedes, so both need the deadline.
# ``mn_warmup`` is production's name for the warmup slot; the measured round runs
# in the task's own output dir, which these tests name.
_MEASURED_ROUND_SLOT = "measured_round"
_BASELINE_ROUND_SLOTS = ("mn_warmup", _MEASURED_ROUND_SLOT)


# ``--max-hours`` defaults to 2.0 (``cli/parser.py``), so this is the session
# shape almost every run has. It matters here because PRELUDE's share of it is
# 48 minutes while a baseline round's declared cap is 130 minutes warm and 150
# cold: any rule that prices the pair at the declared cap refuses every default
# session, and any rule that prices it at nothing lets the warmup eat the round.
_DEFAULT_SESSION_MINUTES = 120.0


class _BudgetedState:
    """A session state whose budget accounting moves as the passes spend it.

    Production reads one session through two accessors -- the deadline the round
    reaper is handed and the usable-seconds figure the phase policy reads -- and
    a double that lets them drift cannot see the regime where they disagree.
    Both are derived from one deadline here.

    A pass calls :meth:`charge` for the wall-clock it burned, which is the only
    way a test can reach the case this whole mechanism exists for: a round whose
    first pass leaves the second one nothing.

    It carries no phase clock on purpose. An earlier version set
    ``phase_started_unix`` to zero, which the budget policy of the day read as a
    preparation phase running since the epoch, and a test meant to show a
    round's overheads exhausting the session passed on that arithmetic instead.
    The policy answers to the session clock alone, so that is all this offers.
    """

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
    """The wall clock the executor prices rounds by, moved by hand.

    A round's price is wall-clock, so a test that needs a pass to cost twenty
    minutes can sleep for them, assert on nothing, or hand the executor a clock it
    can move. Only the third is both quick and about the thing under test.

    Installed over ``time.time``, which is what the executor times rounds with;
    ``time.monotonic``, which the budget deadline runs on, is left alone so the
    two cannot be confused for each other.
    """

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
    """A ``run_with_session_kill`` stand-in that records how each round was launched.

    Every record carries the ``round_slot`` the round wrote into, so a launch can
    be looked up by which pass it was rather than by the order it happened in.

    ``pass_duration_sec`` makes the double honour the cap it was handed the way a
    real benchmark pass does, and it charges the budget for what it ran: a pass
    granted less than the workload takes is killed rather than reporting, and it
    is killed by whichever of the two limits it meets first -- the session
    watchdog, which comes back with the sentinel returncode that says the run ran
    out of time, or its own hard cap, which raises ``TimeoutExpired``.

    ``charge_sec`` charges the session more than the pass itself ran, which is
    what a round with a server restart and a teardown around the pass costs.

    ``boot_sec`` and ``benchmark_sec`` spend ``clock`` in the two parts a real
    pass spends it in, announcing the server ready between them exactly as the
    gate loop does. A pass that models one duration cannot reach the pricing at
    all, which is built on telling the two apart.
    """
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
                _stamp_server_ready(server_log_path, boot_sec)
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
    """A serving-lease stand-in that records how a round reached the Ray actor.

    The Ray path is the same round through a different door, and the door matters
    here: the lease is handed what is left of the budget as a duration, because
    the absolute deadline is a ``time.monotonic()`` instant that means nothing in
    the actor's process. So it is a launch site of its own, with its own way of
    losing the reaper.
    """

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
    """The cap the multi-node warmup pass was granted, or ``None`` when it never ran.

    Both are outcomes a round is allowed to have -- a skipped warmup is what a
    single-node baseline does -- so they are told apart here rather than by a
    ``KeyError`` at the assertion.
    """
    launch = launches_by_round_slot(calls).get("mn_warmup")
    return None if launch is None else int(launch["timeout"])


def _launch_one_grid_variant_under_budget(
    tmp_path,
    *,
    remaining_sec: float,
    variant_timeout_sec: int,
    variant_expected_sec: float,
) -> list[dict]:
    """Run one grid variant against the same budget a baseline round would get.

    The other arm that benches on the GPU, driven through its own entry point so
    the two can be compared on what they grant the passes they both run.
    """
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
    """The arm #1146 names as the largest hole, and the one that motivated it.

    A baseline is admitted on a catalogue cost of five minutes and given a
    two-hour hang backstop (four, cold), so the round that runs first and
    longest was the one round no wall-clock defence covered: no deadline
    reached the reaper, and nothing clamped the cap to what was left.
    """

    @pytest.mark.parametrize("round_slot", _BASELINE_ROUND_SLOTS)
    def test_the_deadline_reaches_the_reaper(self, tmp_path, monkeypatch, round_slot):
        """Parameterized over the passes a round launches, not just the measured one.

        The reaper is the only thing that attributes a budget kill correctly, and
        it only knows about the deadline it was handed. A pass launched without
        one runs until its own hard cap and comes back looking like a variant
        that timed out.
        """
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
        """The fourth launch site, and the one a parameterization cannot reach.

        Production runs a single-node round through a Ray lease, which is handed a
        remaining duration rather than the deadline, by a different call. The
        reaper in the actor's process has nothing else to go on.
        """
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
        """A reaped round leaves exactly what a broken server leaves behind.

        No workspace, no report, a non-zero returncode -- so without this branch
        the run that ran out of time is filed as a fact about the model.
        """
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
        """The discarded warmup is a full pass, so a cancel there ends the round.

        Running the measured round anyway spends a second pass of GPU time the
        run has already been told to stop spending -- and grades the baseline on
        a round started after the stop.
        """
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
        """The regime several rounds of this mechanism have failed in, pinned directly.

        A session's first baseline has measured nothing, so every rule that
        shortens a pass by predicting the next one is guessing here. Each guess
        tried so far killed a round that fit: the cold pass pays weight load and
        graph capture, which the same file's cold-start cap sizes at up to 9000s,
        so any share-of-the-budget cap lands under it on a default session.

        No pass is shortened now. The warmup is granted the round's own cap, and
        this ten-minute workload runs both passes and yields the warm anchor the
        round exists to produce.
        """
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
        """A multi-node round is two client passes, and the figure covers one.

        The server comes up outside the round, so both passes are the same shape
        and the recorded wall-clock -- taken after the warmup -- is one of them.
        Pricing the round at that one figure admits a pair that cannot fit: the
        warmup spends its half and the measured pass meets the deadline, leaving
        the round with no anchor and the GPU time gone.

        1500s is the band only this gate catches: the generic gate before it
        prices the round at one 600s pass and a variant at another, admits at
        1200s, and has no way to know a second pass is coming. The pair plus a
        variant needs 1800s.
        """
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
        """Profile is the same executor with a four-hour default -- longer than
        any session budget it could be given."""
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
    """A slow failure (>30s) with the same stderr pattern must NOT be
    classified as arg error — it could be a real inference crash."""
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
