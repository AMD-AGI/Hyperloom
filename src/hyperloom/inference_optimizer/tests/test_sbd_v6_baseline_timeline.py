# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``baseline`` event.

The event this replaces was projected from V5, and the projection's defect is
what most of these tests pin: V5 stamps the measurement's rows when it
completes, so the projected event's window collapsed onto its own end and it
sorted onto the timeline at the moment it finished rather than the moment it
began.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    PRODUCER,
    ROUND_ACCURACY,
    ROUND_MEASURE,
    ROUND_SINGLE,
    ROUND_WARMUP,
    RUN_AFTER_EVAL_FAILURE,
    RUN_INITIAL,
    assemble_baseline_action,
    baseline_event_id,
    make_baseline_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.assembler import baseline_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.actions.executors.baseline import SBD_INNER_STEP_PARAM, BaselineExecutor


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "baseline"]


def _actions(session_dir: Path, index: int = 0) -> list[dict[str, Any]]:
    return _events(session_dir)[index]["ext"]["actions"]


def _recorder(
    *,
    task_id: str = "t-1",
    task_kind: str = "baseline",
    reason: str = "",
    phase: str = "prelude",
    macro_cycle: int = 0,
    establishes_quality_ref: bool = True,
    params: dict[str, Any] | None = None,
):
    """Build a recorder the way a dispatched baseline gets one."""
    recorder = make_baseline_recorder(
        make_sink(baseline_event_id(phase, macro_cycle), producer=PRODUCER),
        task_id=task_id,
        task_kind=task_kind,
        reason=reason,
        framework="sglang",
        establishes_quality_ref=establishes_quality_ref,
        params=params or {"config_path": "/cfg.yaml", "output_dir": "/w", "timeout_sec": 7800},
    )
    assert recorder is not None
    return recorder


def _measured(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "succeeded",
        "output_throughput": 15630.28,
        "ttft_mean_ms": 138.89,
        "e2el_mean_ms": 4185.47,
        "tpot_mean_ms": 12.5,
        "report_path": "/w/measure_round/benchmark_report.json",
        "workspace": "/w/measure_round/benchmark_sglang_1",
        "materialized_config": "/w/baseline.with_envs.yaml",
        "subprocess_runtime_sec": 241.0,
        "post_ready_runtime_sec": 120.0,
        "run_eval_disabled": False,
    }
    result.update(overrides)
    return result


def _failed(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "failed",
        "error_class": "server_init_dead",
        "error": "server engine/worker init failed; see server.log",
        "returncode": 250,
    }
    result.update(overrides)
    return result


def test_the_event_is_on_the_timeline_before_the_measurement_finishes(tmp_path: Path) -> None:
    """The capability the projection could not have: a live baseline is visible."""
    _recorder()

    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "running"
    assert events[0]["id"] == "prelude:0:baseline"
    assert events[0]["start_time"]


def test_the_window_starts_when_the_action_started_not_when_it_ended(tmp_path: Path) -> None:
    """The whole point of recording baseline rather than projecting it.

    The projected event took both ends of its window from completion stamps, so
    a measurement that ran for minutes was published as an instant at its own
    end, and it sorted onto the timeline behind actions that began after it.
    """
    recorder = _recorder()
    opened = _events(tmp_path)[0]["start_time"]
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=index,
        label=ROUND_SINGLE,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=242.0,
        timeout_sec=7800,
        result=_measured(),
    )
    recorder.end_run(run_index=index, result=_measured())
    recorder.finish(_measured())

    # The closing write reuses the opening write's start, so the window the
    # timeline publishes begins where the action began.
    event = _events(tmp_path)[0]
    assert event["start_time"] == opened
    assert event["end_time"] >= event["start_time"]
    action = _actions(tmp_path)[0]
    assert action["start_time"] == opened
    # Each round carries the window it ran in, taken when it started rather
    # than read back off a completion stamp.
    round_row = action["runs"][0]["rounds"][0]
    assert round_row["start_time"] == "2026-09-02T15:07:09+00:00"
    assert round_row["duration_sec"] == pytest.approx(242.0)
    assert round_row["timeout_sec"] == 7800


def test_a_measured_baseline_closes_succeeded_with_its_numbers(tmp_path: Path) -> None:
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=index,
        label=ROUND_SINGLE,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=242.0,
        result=_measured(),
    )
    recorder.end_run(run_index=index, result=_measured())
    recorder.finish(_measured())

    event = _events(tmp_path)[0]
    assert event["status"] == "succeeded"
    action = _actions(tmp_path)[0]
    assert action["status"] == "succeeded"
    # Named as the V5 section named it, which is what a consumer selects on.
    assert action["measurement"]["throughput_tok_s_per_gpu"] == pytest.approx(15630.28)
    assert action["measurement"]["ttft_mean_ms"] == pytest.approx(138.89)
    assert action["measurement"]["benchmark_report_path"].endswith("benchmark_report.json")
    assert action["timing"]["subprocess_runtime_sec"] == pytest.approx(241.0)
    assert action["request"]["establishes_quality_ref"] is True
    assert action["failure"] is None


def test_the_discarded_warmup_is_recorded_beside_the_pass_that_counted(tmp_path: Path) -> None:
    """The cold number is the only thing the adopted one can be weighed against."""
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=index,
        label=ROUND_WARMUP,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=180.0,
        result=_measured(output_throughput=9000.0),
    )
    recorder.record_round(
        run_index=index,
        label=ROUND_MEASURE,
        started_at="2026-09-02T15:10:31+00:00",
        duration_sec=40.0,
        result=_measured(),
    )
    recorder.end_run(run_index=index, result=_measured(warmup_round_tput=9000.0))
    recorder.finish(_measured(warmup_round_tput=9000.0))

    rounds = _actions(tmp_path)[0]["runs"][0]["rounds"]
    assert [row["label"] for row in rounds] == [ROUND_WARMUP, ROUND_MEASURE]
    assert rounds[0]["measurement"]["throughput_tok_s_per_gpu"] == pytest.approx(9000.0)
    assert rounds[1]["measurement"]["throughput_tok_s_per_gpu"] == pytest.approx(15630.28)
    assert _actions(tmp_path)[0]["warmup_round_tput"] == pytest.approx(9000.0)


def test_rounds_that_start_in_the_same_second_keep_the_order_they_ran_in(tmp_path: Path) -> None:
    """Start stamps are ISO seconds, so they cannot be the only ordering key.

    A round that fails fast can start and finish inside the same second as the
    next one. Ordering on the stamp alone leaves that tie to the next declared
    key, and on the label it resolves alphabetically -- ``measure`` ahead of
    the ``warmup`` that booted the server it re-attached to.
    """
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    same_second = "2026-09-02T15:07:09+00:00"
    for label in (ROUND_WARMUP, ROUND_MEASURE, ROUND_ACCURACY):
        recorder.record_round(
            run_index=index,
            label=label,
            started_at=same_second,
            duration_sec=0.4,
            result=_measured(),
        )
    recorder.end_run(run_index=index, result=_measured())
    recorder.finish(_measured())

    rounds = _actions(tmp_path)[0]["runs"][0]["rounds"]
    assert [row["label"] for row in rounds] == [ROUND_WARMUP, ROUND_MEASURE, ROUND_ACCURACY]
    # The ordinal that fixed the order is recording-side bookkeeping and does
    # not reach the wire; the array's own order carries it.
    assert all("ordinal" not in row for row in rounds)


def test_each_pass_keeps_its_own_rounds_and_says_why_it_ran(tmp_path: Path) -> None:
    """A salvage retry re-runs the rounds, and which pass a round belonged to matters."""
    recorder = _recorder()
    first = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=first,
        label=ROUND_SINGLE,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=30.0,
        result=_failed(error_class="subprocess_nonzero"),
    )
    recorder.end_run(run_index=first, result=_failed(error_class="subprocess_nonzero"))
    second = recorder.begin_run(attempt_reason=RUN_AFTER_EVAL_FAILURE)
    recorder.record_round(
        run_index=second,
        label=ROUND_SINGLE,
        started_at="2026-09-02T15:09:09+00:00",
        duration_sec=240.0,
        result=_measured(),
    )
    recorder.end_run(run_index=second, result=_measured())
    recorder.finish(_measured())

    runs = _actions(tmp_path)[0]["runs"]
    assert [row["run_index"] for row in runs] == [1, 2]
    assert [row["attempt_reason"] for row in runs] == [RUN_INITIAL, RUN_AFTER_EVAL_FAILURE]
    assert [row["status"] for row in runs] == ["failed", "succeeded"]
    assert runs[0]["rounds"][0]["failure"]["error_class"] == "subprocess_nonzero"
    assert runs[1]["rounds"][0]["failure"] is None
    # A run that failed and was recovered from does not make the action failed.
    assert _events(tmp_path)[0]["status"] == "succeeded"


def test_a_pass_refused_before_it_booted_still_leaves_a_row(tmp_path: Path) -> None:
    """The case a round-only model would drop: nothing ran, and that is the fact."""
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    refused = _failed(error_class="session_time_exhausted", error="the run's clock refused the round")
    recorder.end_run(run_index=index, result=refused)
    recorder.finish(refused)

    action = _actions(tmp_path)[0]
    assert action["status"] == "skipped"
    assert _events(tmp_path)[0]["status"] == "skipped"
    assert action["runs"][0]["rounds"] == []
    assert action["runs"][0]["error_class"] == "session_time_exhausted"


def test_a_baseline_standing_on_its_cold_warmup_closes_degraded(tmp_path: Path) -> None:
    """The number is usable and knowingly depressed, which is neither pass nor fail."""
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=index,
        label=ROUND_WARMUP,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=180.0,
        result=_measured(output_throughput=9000.0),
    )
    cold = _measured(
        output_throughput=9000.0,
        measure_round_dropped={"reason": "measure_round_reaped_by_the_run"},
    )
    recorder.end_run(run_index=index, result=cold)
    recorder.finish(cold)

    assert _events(tmp_path)[0]["status"] == "degraded"
    action = _actions(tmp_path)[0]
    assert action["status"] == "degraded"
    assert action["cold_anchor"]["reason"] == "measure_round_reaped_by_the_run"


def test_a_failed_baseline_names_the_class_it_failed_with(tmp_path: Path) -> None:
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=index,
        label=ROUND_SINGLE,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=30.0,
        result=_failed(),
    )
    recorder.end_run(run_index=index, result=_failed())
    recorder.finish(_failed())

    event = _events(tmp_path)[0]
    assert event["status"] == "failed"
    failure = _actions(tmp_path)[0]["failure"]
    assert failure["error_class"] == "server_init_dead"
    assert failure["returncode"] == 250
    assert _actions(tmp_path)[0]["measurement"]["throughput_tok_s_per_gpu"] is None


def test_an_executor_that_raised_is_not_left_running(tmp_path: Path) -> None:
    """Otherwise a crash and a killed session both read as a dangling event."""
    recorder = _recorder()
    recorder.finish_crashed(RuntimeError("boom"))

    event = _events(tmp_path)[0]
    assert event["status"] == "failed"
    failure = _actions(tmp_path)[0]["failure"]
    assert failure["error_class"] == "RuntimeError"
    assert "boom" in failure["message"]


def test_closing_twice_keeps_the_first_verdict(tmp_path: Path) -> None:
    recorder = _recorder()
    recorder.finish(_measured())
    recorder.finish_crashed(RuntimeError("late"))

    assert _events(tmp_path)[0]["status"] == "succeeded"


def test_two_baselines_in_one_cycle_are_one_event_with_two_actions(tmp_path: Path) -> None:
    """A failure streak re-dispatches, and N timeline entries per cycle is noise."""
    first = _recorder(task_id="t-1")
    first.finish(_failed())
    second = _recorder(task_id="t-2")
    second.finish(_measured())

    events = _events(tmp_path)
    assert len(events) == 1
    actions = events[0]["ext"]["actions"]
    assert [action["task_id"] for action in actions] == ["t-1", "t-2"]
    # The event takes the worst of them, so the failure the retry recovered
    # from cannot be read off the event as though it had not happened.
    assert events[0]["status"] == "failed"


def test_a_success_is_not_erased_by_a_sibling_that_was_skipped(tmp_path: Path) -> None:
    """``skipped`` ranks below ``succeeded``: a refused retry unmakes no anchor."""
    measured = _recorder(task_id="t-1")
    measured.finish(_measured())
    refused = _recorder(task_id="t-2")
    refused.finish(_failed(error_class="session_time_exhausted"))

    assert _events(tmp_path)[0]["status"] == "succeeded"


def test_baselines_in_different_cycles_are_different_events(tmp_path: Path) -> None:
    _recorder(task_id="t-1", phase="prelude", macro_cycle=0).finish(_measured())
    _recorder(task_id="t-2", phase="explore", macro_cycle=1).finish(_measured())

    assert sorted(event["id"] for event in _events(tmp_path)) == [
        "explore:1:baseline",
        "prelude:0:baseline",
    ]


def test_one_action_can_be_assembled_on_its_own(tmp_path: Path) -> None:
    _recorder(task_id="t-1").finish(_measured())
    _recorder(task_id="t-2").finish(_failed())

    action = assemble_baseline_action(baseline_event_parts(), event="prelude:0:baseline", task_id="t-2")
    assert action is not None
    assert action["status"] == "failed"
    assert assemble_baseline_action(baseline_event_parts(), event="prelude:0:baseline", task_id="t-9") is None


def test_a_killed_session_leaves_an_interrupted_event_with_its_rows(tmp_path: Path) -> None:
    """Finalize publishes what was recorded and refuses to judge it."""
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    recorder.record_round(
        run_index=index,
        label=ROUND_WARMUP,
        started_at="2026-09-02T15:07:09+00:00",
        duration_sec=180.0,
        result=_measured(output_throughput=9000.0),
    )

    assert finalize_events(tmp_path) == ["prelude:0:baseline"]

    event = _events(tmp_path)[0]
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    action = event["ext"]["actions"][0]
    # The action was never closed, so it is still running inside an event that
    # says nothing judged it.
    assert action["status"] == "running"
    assert action["in_flight_run_index"] == 1
    assert action["runs"][0]["rounds"][0]["label"] == ROUND_WARMUP


def test_assembling_the_same_rows_twice_gives_the_same_event(tmp_path: Path) -> None:
    recorder = _recorder()
    index = recorder.begin_run(attempt_reason=RUN_INITIAL)
    for label in (ROUND_WARMUP, ROUND_MEASURE):
        recorder.record_round(
            run_index=index,
            label=label,
            started_at="2026-09-02T15:07:09+00:00",
            duration_sec=1.0,
            result=_measured(),
        )
    recorder.end_run(run_index=index, result=_measured())
    recorder.finish(_measured())

    parts = baseline_event_parts()
    once = assemble_baseline_action(parts, event="prelude:0:baseline", task_id="t-1")
    twice = assemble_baseline_action(parts, event="prelude:0:baseline", task_id="t-1")
    assert once == twice


def test_no_sink_declines_rather_than_guessing_an_event(tmp_path: Path) -> None:
    assert make_baseline_recorder(None, task_id="t-1") is None
    assert _events(tmp_path) == []


# ---------------------------------------------------------------------------
# the executor wiring
# ---------------------------------------------------------------------------
def _executor_ctx(tmp_path: Path, **params: Any):
    """Build the context a dispatched baseline arrives with."""
    return SimpleNamespace(
        task=SimpleNamespace(task_id="t-exec", kind="baseline", params=params),
        lease=None,
        extra={"shared_state": SimpleNamespace(phase="PRELUDE", macro_cycle=0, framework="sglang")},
    )


@pytest.mark.asyncio
async def test_the_executor_records_the_rounds_it_actually_ran(tmp_path: Path) -> None:
    """The wiring, not the recorder: rounds land under the pass that ran them."""
    executor = object.__new__(BaselineExecutor)
    executor.shared_state = None

    async def _run_once(ctx, *, recorder=None, run_index=0, **_kwargs):
        for label in (ROUND_WARMUP, ROUND_MEASURE):
            await executor._run_reported_round(
                label=label,
                config_path=Path("/cfg.yaml"),
                output_dir=tmp_path / label,
                recorder=recorder,
                run_index=run_index,
                timeout_sec=7800,
            )
        return _measured()

    async def _benchmark(**_kwargs):
        return _measured()

    executor._run_once = _run_once  # type: ignore[method-assign]
    executor._run_single_benchmark = _benchmark  # type: ignore[method-assign]
    executor._maybe_stop_on_missing_baseline_accuracy = lambda *_a: None  # type: ignore[method-assign]
    executor._is_moe_runner_rooted_failure = lambda _r: False  # type: ignore[method-assign]
    executor._resolve_shared_state = lambda state=None: state  # type: ignore[method-assign]

    result = await executor(_executor_ctx(tmp_path, config_path="/cfg.yaml"))

    assert result["status"] == "succeeded"
    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["id"] == "prelude:0:baseline"
    assert events[0]["status"] == "succeeded"
    action = events[0]["ext"]["actions"][0]
    assert action["task_id"] == "t-exec"
    assert action["request"]["task_kind"] == "baseline"
    assert [row["label"] for row in action["runs"][0]["rounds"]] == [ROUND_WARMUP, ROUND_MEASURE]
    assert action["runs"][0]["attempt_reason"] == RUN_INITIAL


@pytest.mark.asyncio
async def test_an_inner_step_measurement_leaves_no_event_of_its_own(tmp_path: Path) -> None:
    """The kernel phase measures through this executor and records it itself."""
    executor = object.__new__(BaselineExecutor)
    executor.shared_state = None

    async def _run_once(_ctx, *, recorder=None, run_index=0, **_kwargs):
        assert recorder is None
        return _measured()

    executor._run_once = _run_once  # type: ignore[method-assign]
    executor._maybe_stop_on_missing_baseline_accuracy = lambda *_a: None  # type: ignore[method-assign]
    executor._is_moe_runner_rooted_failure = lambda _r: False  # type: ignore[method-assign]
    executor._resolve_shared_state = lambda state=None: state  # type: ignore[method-assign]

    await executor(_executor_ctx(tmp_path, **{SBD_INNER_STEP_PARAM: True}))

    assert _events(tmp_path) == []


@pytest.mark.asyncio
async def test_an_executor_raise_closes_the_event_it_opened(tmp_path: Path) -> None:
    executor = object.__new__(BaselineExecutor)
    executor.shared_state = None

    async def _run_once(_ctx, *, recorder=None, run_index=0, **_kwargs):
        raise FileNotFoundError("baseline config not found")

    executor._run_once = _run_once  # type: ignore[method-assign]
    executor._resolve_shared_state = lambda state=None: state  # type: ignore[method-assign]

    with pytest.raises(FileNotFoundError):
        await executor(_executor_ctx(tmp_path))

    event = _events(tmp_path)[0]
    assert event["status"] == "failed"
    action = event["ext"]["actions"][0]
    assert action["failure"]["error_class"] == "FileNotFoundError"
    # The pass that raised is closed too, rather than left reading "running".
    assert action["runs"][0]["status"] == "failed"
