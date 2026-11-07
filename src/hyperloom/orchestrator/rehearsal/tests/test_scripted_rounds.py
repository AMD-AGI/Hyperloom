# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``ScriptedRounds``: a written-down round played as ``run_round``, leaving the run_dir a real round leaves."""

from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import inference_suite
from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors._subprocess_kill import SESSION_TIME_EXHAUSTED_RETURNCODE, RoundWatch
from hyperloom.orchestrator.bringup import observe_bringup
from hyperloom.orchestrator.rehearsal import (
    DIED_SILENTLY,
    HANG,
    READY,
    STAGE_FAILED,
    LaunchAttempt,
    LaunchScenario,
    ScenarioError,
    ScenarioExhausted,
    ScriptedLaunchBackend,
    ScriptedRounds,
    VirtualClock,
    installed_clock,
)
from inference_suite import RunDirError
from inference_suite.contracts import (
    Deadlines,
    FailedStep,
    Lifecycle,
    LifecycleMode,
    ProfileConfig,
    Report,
    RunPaths,
    RunStatus,
    ScriptableWorkload,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.tests.test_compose import _launch
from inference_suite.tests.test_config_contract import _FIXED, _SHA, _config, _eval

_HARD_TIMEOUT = 1800.0
_METRICS = {"output_throughput": 1234.5, "completed_requests": 640}


def _fixed_config(run_dir: Path, **changes: object) -> SuiteConfig:
    config = _config(
        workload=WorkloadKind.FIXED,
        agentx=None,
        fixed=_FIXED,
        profile=ProfileConfig(False),
        deadlines=Deadlines(
            boot_sec=_HARD_TIMEOUT, round_sec=_HARD_TIMEOUT, eval_sec=600, silence_sec=3600, pre_ready_silence_sec=3600
        ),
        paths=RunPaths(run_dir=str(run_dir), tool_cache="/s/tools", tool_manifest_sha256=_SHA),
    )
    return replace(config, **changes)


def _rounds(*attempts: LaunchAttempt, **fields: object) -> ScriptedRounds:
    return ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), **fields)


def _verdict(result: inference_suite.RoundResult):
    layout = result.layout
    log = layout.server_log.read_text(encoding="utf-8") if layout.server_log.exists() else ""
    return observe_bringup(server_log=log, wrapper_stderr=result.report.failure_detail or "").observation


def _played_on_the_launch_backend(attempt: LaunchAttempt, slot: Path):
    backend = ScriptedLaunchBackend(scenario=LaunchScenario(attempts=(attempt,)), clock=VirtualClock())
    log = slot / "server.log"
    try:
        done = backend.run(["magpie"], watch=RoundWatch(server_log_path=str(log), timeout=_HARD_TIMEOUT))
    except subprocess.TimeoutExpired:
        return None, backend.clock.elapsed
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    observed = observe_bringup(server_log=text, wrapper_stderr=done.stderr).observation
    return (done.returncode == 0, observed.stage_failed), backend.clock.elapsed


@pytest.mark.parametrize(
    "attempt",
    [
        LaunchAttempt(outcome=READY, duration_sec=120.0),
        LaunchAttempt(outcome=STAGE_FAILED, stage="argv_parse", duration_sec=45.0),
        LaunchAttempt(outcome=STAGE_FAILED, stage="import"),
        LaunchAttempt(outcome=STAGE_FAILED, stage="config_validate", duration_sec=180.0),
        LaunchAttempt(outcome=STAGE_FAILED, stage="weights_loading"),
        LaunchAttempt(outcome=STAGE_FAILED, stage="engine_init"),
        LaunchAttempt(outcome=DIED_SILENTLY, returncode=139),
        LaunchAttempt(outcome=HANG, duration_sec=10.0),
    ],
    ids=lambda attempt: f"{attempt.outcome}-{attempt.stage}",
)
def test_an_existing_scenario_plays_to_the_same_result_on_the_round_double(tmp_path, attempt):
    """The classification, the time charged and whether the round served all match the launch double's."""
    launched, launch_elapsed = _played_on_the_launch_backend(attempt, tmp_path / "launch")

    rounds = _rounds(attempt)
    result = rounds.run_round(_fixed_config(tmp_path / "round"))

    assert rounds.clock.elapsed == pytest.approx(launch_elapsed)
    if launched is None:
        assert result.report.failed_step is FailedStep.ROUND_TIMEOUT
        return
    served, stage_failed = launched
    assert (result.report.run_status is RunStatus.COMPLETED) == served
    assert _verdict(result).stage_failed == stage_failed


def test_a_blocker_stack_peels_one_wall_per_round_and_then_serves(tmp_path):
    scenario = LaunchScenario.from_dict({"blockers": ["argv_parse", "import", "engine_init"]})
    rounds = ScriptedRounds(scenario=scenario, clock=VirtualClock())

    reached = [
        _verdict(rounds.run_round(_fixed_config(tmp_path / f"r{index}"))).stage_failed
        for index in range(len(scenario.attempts))
    ]

    assert reached == [LadderStage.ARGV_PARSE, LadderStage.IMPORT, LadderStage.ENGINE_INIT, None]
    assert rounds.exhausted
    assert [call.report.failed_step for call in rounds.calls] == [
        FailedStep.FLAG_UNKNOWN,
        FailedStep.SERVER_EXITED_BEFORE_READY,
        FailedStep.SERVER_EXITED_BEFORE_READY,
        None,
    ]


def test_a_served_fixed_round_leaves_the_run_dir_a_real_round_leaves(tmp_path):
    launch = _launch()
    attempt = LaunchAttempt(outcome=READY, duration_sec=90.0, metrics=_METRICS, artifacts={"extra.json": {"a": 1}})
    rounds = _rounds(attempt, launch=launch)

    result = rounds.run_round(_fixed_config(tmp_path / "run"))

    layout = result.layout
    report = Report.from_document(json.loads(layout.report.read_text(encoding="utf-8")))
    assert report == result.report == rounds.calls[0].report
    assert report.run_status is RunStatus.COMPLETED
    assert report.fixed is not None and report.fixed.values["output_throughput"] == 1234.5
    assert report.codes.client == 0
    assert report.launch_digest == launch.digest()
    assert layout.launch.exists() and layout.config.exists() and (layout.root / "extra.json").exists()
    assert report.server is not None and report.server.reused is False and report.server.log_range is None
    stamp = layout.read_ready_stamp()
    assert stamp is not None and stamp.boot_sec == pytest.approx(90.0)
    assert report.ended_unix - report.started_unix == pytest.approx(90.0)


def test_a_reuse_round_finds_its_server_ready_at_once_and_keeps_the_copied_log_range(tmp_path):
    lifecycle = Lifecycle(LifecycleMode.REUSE, pid_dir=str(tmp_path / "pids"))
    result = _rounds(LaunchAttempt(outcome=READY, metrics=_METRICS)).run_round(
        _fixed_config(tmp_path / "run", lifecycle=lifecycle)
    )

    server = result.report.server
    assert server is not None and server.reused is True
    assert server.log_range is not None and server.log_range.end == result.layout.server_log.stat().st_size
    stamp = result.layout.read_ready_stamp()
    assert stamp is not None and stamp.boot_sec == 0.0


def test_an_external_round_runs_its_client_against_a_server_it_neither_boots_nor_logs(tmp_path):
    """Multi-node: the server is someone else's, so the round keeps no server log and no ready stamp."""
    config = _fixed_config(tmp_path / "run", lifecycle=Lifecycle(LifecycleMode.EXTERNAL))
    result = _rounds(LaunchAttempt(outcome=READY, metrics=_METRICS)).run_round(config)

    assert result.report.run_status is RunStatus.COMPLETED
    assert result.report.server is not None and result.report.server.reused is True
    assert not result.layout.server_log.exists()
    assert result.layout.read_ready_stamp() is None


def test_an_external_round_cannot_play_a_boot_wall(tmp_path):
    config = _fixed_config(tmp_path / "run", lifecycle=Lifecycle(LifecycleMode.EXTERNAL))

    with pytest.raises(ScenarioError):
        _rounds(LaunchAttempt(outcome=STAGE_FAILED, stage="engine_init")).run_round(config)


def test_a_scriptable_round_keeps_its_script_result_and_records_no_server_or_launch(tmp_path):
    workload = ScriptableWorkload(script="xdit", params={"XDIT_NUM_STEPS": 28})
    config = _fixed_config(tmp_path / "run", workload=WorkloadKind.SCRIPTABLE, fixed=None, scriptable=workload)
    result = _rounds(LaunchAttempt(outcome=READY, metrics={"latency_s": 3.5}), launch=_launch()).run_round(config)

    scriptable = result.report.scriptable
    assert scriptable is not None and scriptable.result == {"latency_s": 3.5}
    assert (result.layout.root / scriptable.result_file) == result.layout.inferencex_result
    assert result.layout.inferencex_result.exists()
    assert result.report.server is None and result.report.launch_digest is None
    assert not result.layout.launch.exists() and not result.layout.server_log.exists()


def test_an_unknown_flag_is_refused_before_a_launch_is_composed(tmp_path):
    result = _rounds(LaunchAttempt(outcome=STAGE_FAILED, stage="argv_parse"), launch=_launch()).run_round(
        _fixed_config(tmp_path / "run")
    )

    assert result.report.failed_step is FailedStep.FLAG_UNKNOWN
    assert result.report.server is None and result.report.launch_digest is None
    assert not result.layout.launch.exists() and not result.layout.server_log.exists()


def test_a_client_that_fails_on_a_ready_server_reports_its_code_after_the_ready_stamp(tmp_path):
    result = _rounds(LaunchAttempt(outcome=READY, returncode=2)).run_round(_fixed_config(tmp_path / "run"))

    assert result.report.failed_step is FailedStep.CLIENT_FAILED
    assert result.report.run_status is RunStatus.FAILED
    assert result.report.codes.client == 2
    assert result.layout.read_ready_stamp() is not None


@pytest.mark.parametrize(
    ("attempt", "deadlines", "step", "elapsed"),
    [
        (
            LaunchAttempt(outcome=HANG, stage="graph_capture"),
            Deadlines(boot_sec=1800, round_sec=3600, eval_sec=600, silence_sec=600, pre_ready_silence_sec=900),
            FailedStep.SERVER_SILENT_BEFORE_READY,
            900.0,
        ),
        (
            LaunchAttempt(outcome=HANG, stage="graph_capture"),
            Deadlines(boot_sec=1200, round_sec=3600, eval_sec=600, silence_sec=600, pre_ready_silence_sec=1500),
            FailedStep.SERVER_BOOT_TIMEOUT,
            1200.0,
        ),
        (
            LaunchAttempt(outcome=HANG, duration_sec=100.0),
            Deadlines(boot_sec=1800, round_sec=3600, eval_sec=600, silence_sec=600),
            FailedStep.SERVER_STALLED,
            700.0,
        ),
    ],
    ids=["silent-boot", "boot-budget", "stalled-after-ready"],
)
def test_a_hang_is_stopped_by_the_first_gate_that_trips(tmp_path, attempt, deadlines, step, elapsed):
    rounds = _rounds(attempt)
    result = rounds.run_round(_fixed_config(tmp_path / "run", deadlines=deadlines))

    assert result.report.failed_step is step
    assert rounds.clock.elapsed == pytest.approx(elapsed)


def test_the_session_deadline_cuts_a_round_short_as_the_launch_double_does(tmp_path):
    rounds = _rounds(LaunchAttempt(outcome=READY, duration_sec=600.0, metrics=_METRICS))
    launch = ScriptedLaunchBackend(LaunchScenario(attempts=(LaunchAttempt(outcome=READY, duration_sec=600.0),)))

    with installed_clock(rounds.clock):
        deadline = int(rounds.clock.wall()) + 60
        result = rounds.run_round(
            _fixed_config(tmp_path / "run", deadlines=replace(_deadlines(), deadline_unix=deadline))
        )
    with installed_clock(launch.clock):
        done = launch.run(["magpie"], watch=RoundWatch(session_deadline_sec=launch.clock.monotonic() + 60.0))

    assert done.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
    assert result.report.failed_step is FailedStep.DEADLINE_EXHAUSTED
    assert result.report.fixed is None
    assert rounds.clock.elapsed == pytest.approx(launch.clock.elapsed) == pytest.approx(60.0)


def _deadlines() -> Deadlines:
    return Deadlines(boot_sec=1800, round_sec=3600, eval_sec=600, silence_sec=600)


def test_a_deadline_off_a_clock_this_double_does_not_read_is_refused(tmp_path):
    rounds = _rounds(LaunchAttempt(outcome=READY))

    with pytest.raises(ScenarioError):
        rounds.run_round(_fixed_config(tmp_path / "run", deadlines=replace(_deadlines(), deadline_unix=2_000_000_000)))

    assert not rounds.calls and not (tmp_path / "run").exists()


def test_a_cancel_stops_the_round_where_it_was_asked(tmp_path):
    answers = iter(["", "", "operator stop"])
    rounds = _rounds(LaunchAttempt(outcome=READY, duration_sec=400.0, metrics=_METRICS))

    result = rounds.run_round(_fixed_config(tmp_path / "run"), cancel=lambda: next(answers))

    assert result.report.run_status is RunStatus.STOPPED
    assert (result.report.failed_step, result.report.failure_detail) == (FailedStep.CANCELLED, "operator stop")
    assert rounds.clock.elapsed == pytest.approx(200.0)


def test_a_run_dir_that_holds_something_is_refused_as_run_round_refuses_it(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "left.txt").write_text("x")

    with pytest.raises(RunDirError):
        _rounds(LaunchAttempt(outcome=READY)).run_round(_fixed_config(tmp_path / "run"))


def test_one_round_past_the_end_of_the_scenario_is_an_error_not_a_repeat(tmp_path):
    rounds = _rounds(LaunchAttempt(outcome=READY))
    rounds.run_round(_fixed_config(tmp_path / "r1"))

    with pytest.raises(ScenarioExhausted):
        rounds.run_round(_fixed_config(tmp_path / "r2"))


@pytest.mark.parametrize(
    "changes",
    [
        {"workload": WorkloadKind.AGENTX, "fixed": None, "agentx": _config().agentx},
        {"eval": _eval()},
        {"profile": ProfileConfig(True, num_prompts=8, delay_iters=2, max_iters=4)},
    ],
    ids=["agentx", "eval", "profile"],
)
def test_a_round_the_double_does_not_script_is_refused_before_it_plays(tmp_path, changes):
    rounds = _rounds(LaunchAttempt(outcome=READY))

    with pytest.raises(ScenarioError):
        rounds.run_round(_fixed_config(tmp_path / "run", **changes))

    assert not rounds.calls


def test_the_double_takes_the_call_run_round_takes():
    real = inspect.signature(inference_suite.run_round)
    scripted = inspect.signature(ScriptedRounds.run_round)

    assert list(scripted.parameters)[1:] == list(real.parameters)
    assert scripted.return_annotation == real.return_annotation
