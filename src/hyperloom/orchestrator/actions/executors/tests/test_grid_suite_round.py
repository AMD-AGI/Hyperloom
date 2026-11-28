# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A grid whose session froze a base runs its rounds on inference_suite, played here by ``ScriptedRounds``.

The contract: the warmup, the measured, a caller's boot and a caller's reuse round are each one round with its own
run_dir; a launch that names no server leaves it to the grid, which tests a variant whose point warms up on one server
it holds -- a discarded bench that never evaluates, then the measured bench and its eval, both from the variant's one
config and on one port -- and runs any other variant as one round that owns its server; the server a variant's test
held is stopped once on its slot however the rounds ended, a caller's last reuse round takes the caller's server down,
a failed step and a refusal map to their error classes, a variant the session budget cannot admit is skipped
unlaunched, and a session that froze no base runs nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.state.tests._session_base import freeze_base
from hyperloom.orchestrator.actions.executors.grid import _grid_base, _grid_runner
from hyperloom.orchestrator.actions.executors.serving import _server_lifecycle
from hyperloom.orchestrator.actions.executors._suite import held_test
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors.tests.test_baseline_suite_round import _CAPACITY_SERVER_LOG
from hyperloom.orchestrator.actions.executors.grid._grid_runner import GridLaunch, GridRecipe, GridStop, GridVariant
from hyperloom.orchestrator.actions.executors.grid._grid_runner import run_grid
from hyperloom.orchestrator.actions.executors._suite.config import MEASURED_SEED
from hyperloom.orchestrator.actions.executors._suite.rounds import RUN_DIR_PREFIX, TOOL_CACHE_ENV
from hyperloom.orchestrator.actions.executors._suite.config import WARMUP_SEED
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _MODEL, _base
from hyperloom.orchestrator.rehearsal import READY, STAGE_FAILED, LaunchAttempt, LaunchScenario, ScriptedRounds
from hyperloom.orchestrator.rehearsal import VirtualClock, installed_clock
from hyperloom.orchestrator.rehearsal.scenario import ScenarioError
from hyperloom.orchestrator.state.failure_evidence import ARGV_INVALID, ENGINE_CAPACITY
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS
from inference_suite.contracts import FailedStep, LifecycleMode, RunLayout, WorkloadKind
from inference_suite.runtime.process import Reaped
from inference_suite.tests.test_compose import _launch

_HOT = {"output_throughput": 4701.6, "completed_requests": 64}
_COLD = {"output_throughput": 270.9, "completed_requests": 64}
_SCRUBBED_ENV = "GRID_SUITE_TEST_UNSET"


def _point(path: Path) -> Path:
    envs = {"TP": 2, "CONC": 8, "ISL": 256, "OSL": 256, "NUM_PROMPTS": 64, "NUM_WARMUPS": 2, "RUN_EVAL": "false"}
    bench = {
        "framework": "sglang",
        "model": _MODEL,
        "precision": "bf16",
        "run_mode": "local",
        "benchmark_script": "sglang_mi300x.sh",
        "envs": envs,
        "timeout_seconds": 600,
        "profiler": {"torch_profiler": {"enabled": False}},
        "gpu_selection": {"auto": False},
    }
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    session = tmp_path / "session"
    session.mkdir()
    freeze_base(session, _base("sglang"))
    cache = tmp_path / "tools"
    cache.mkdir()
    (cache / "staging_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(TOOL_CACHE_ENV, str(cache))
    monkeypatch.setenv(_SCRUBBED_ENV, "1")
    monkeypatch.setattr(_grid_runner, "_resolve_session_dir", lambda: session)
    monkeypatch.setattr(suite_rounds, "_resolve_probe_python", lambda framework: sys.executable)
    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: None)
    stopped: list[Path] = []

    def record_stop(pid_dir):
        stopped.append(Path(pid_dir))
        return Reaped((), ())

    monkeypatch.setattr(_grid_runner, "stop", record_stop)
    monkeypatch.setattr(_grid_base, "stop", record_stop)
    monkeypatch.setattr(held_test, "stop", record_stop)
    return SimpleNamespace(point=_point(tmp_path / "point.yaml"), out=tmp_path / "out", stopped=stopped)


@pytest.fixture
def unheld(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that cannot hold the point's server from one round to the next, so its point does not warm up."""
    refusal = {"eligible": False, "framework": "sglang", "port": 0, "reason": "the host holds no server"}
    monkeypatch.setattr(_server_lifecycle, "resolve_lifecycle_params", lambda _path: refusal)


def _play(monkeypatch, suite: SimpleNamespace, *attempts: LaunchAttempt, launch: GridLaunch = GridLaunch(), **variant):
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    grid = [GridVariant("vA", **variant)]
    results = asyncio.run(run_grid(grid=grid, output_root=suite.out, recipe=GridRecipe(suite.point), launch=launch))
    return results[0], rounds.calls, suite.out / "variant_00_vA"


def test_the_warmup_boots_the_server_the_measured_round_reuses_and_the_variant_reaps_it(monkeypatch, suite):
    row, calls, slot = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT),
    )

    assert row.status == "succeeded", row
    assert row.output_throughput == pytest.approx(_HOT["output_throughput"])
    assert "run_grid_warmup_discarded_first" in row.nonfatal_warnings
    assert f"warmup_round_tput:{_COLD['output_throughput']:.1f}" in row.nonfatal_warnings
    warmup, measured = (call.config for call in calls)
    assert (warmup.lifecycle.mode, warmup.lifecycle.pid_dir) == (LifecycleMode.BOOT, str(slot))
    assert (measured.lifecycle.mode, measured.lifecycle.pid_dir) == (LifecycleMode.REUSE, str(slot))
    assert (warmup.fixed.seed, measured.fixed.seed) == (WARMUP_SEED, MEASURED_SEED)
    assert warmup.server.port == measured.server.port
    assert row.workspace == str(slot / f"{RUN_DIR_PREFIX}0")
    assert RunLayout(slot / "warmup_round" / f"{RUN_DIR_PREFIX}0").report.is_file()
    assert suite.stopped == [slot]


def test_both_rounds_of_a_variants_test_run_from_its_one_config_which_names_no_server(monkeypatch, suite):
    _, _, slot = _play(
        monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_COLD), LaunchAttempt(outcome=READY, metrics=_HOT)
    )

    assert sorted(path.relative_to(slot) for path in slot.rglob("*.yaml") if RUN_DIR_PREFIX not in str(path)) == [
        Path("config.yaml")
    ]
    bench = yaml.safe_load((slot / "config.yaml").read_text(encoding="utf-8"))["benchmark"]
    assert "server_lifecycle" not in bench and "PORT" not in bench["envs"]


def test_a_failed_warmup_fails_the_variant_stops_its_server_and_runs_no_measured_round(monkeypatch, suite):
    row, calls, slot = _play(monkeypatch, suite, LaunchAttempt(outcome=STAGE_FAILED, stage="engine_init"))

    assert (row.status, row.error_class) == ("failed", "warmup_round_failed"), row
    assert row.nonfatal_warnings == ["run_grid_warmup_round_failed"]
    assert row.workspace == str(slot / "warmup_round" / f"{RUN_DIR_PREFIX}0")
    assert [call.config.lifecycle.mode for call in calls] == [LifecycleMode.BOOT]
    assert suite.stopped == [slot]


def test_a_variants_server_is_stopped_when_its_measured_round_raises(monkeypatch, suite):
    rounds = ScriptedRounds(
        scenario=LaunchScenario(attempts=(LaunchAttempt(outcome=READY, metrics=_COLD),)),
        clock=VirtualClock(),
        launch=_launch(),
    )

    def run_round(config, *, cancel=None):
        if config.lifecycle.mode is LifecycleMode.REUSE:
            raise RuntimeError("the round broke")
        return rounds.run_round(config, cancel=cancel)

    monkeypatch.setattr(suite_rounds, "run_round", run_round)

    with pytest.raises(RuntimeError, match="the round broke"):
        asyncio.run(run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point)))

    assert suite.stopped == [suite.out / "variant_00_vA"]


def test_the_pids_that_outlive_a_stopped_variants_server_are_named_on_its_row(monkeypatch, suite):
    monkeypatch.setattr(suite_rounds, "cancel_scope_listener", _cancelled_scope)
    monkeypatch.setattr(held_test, "stop", lambda pid_dir: Reaped((), (4242,)))

    row, _, _ = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT))

    assert row.status == "skipped"
    assert row.nonfatal_warnings == ["server_survived_teardown:4242"]


_SCORE = {"exact_match,strict-match": 0.9}
_EVALUATING = {"RUN_EVAL": "true"}


def _sequence(calls) -> list[tuple[LifecycleMode, int, bool]]:
    return [(call.config.lifecycle.mode, call.config.fixed.seed, call.config.eval.enabled) for call in calls]


def test_a_variants_own_pair_evaluates_once_after_its_measured_bench(monkeypatch, suite):
    """The default launch, as the integrate bench runs it: the discarded bench leaves the server no eval to carry."""
    row, calls, slot = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT, eval_scores=_SCORE),
        extra_envs=_EVALUATING,
    )

    assert row.status == "succeeded", row
    assert _sequence(calls) == [
        (LifecycleMode.BOOT, WARMUP_SEED, False),
        (LifecycleMode.REUSE, MEASURED_SEED, True),
    ]
    assert [path.relative_to(slot).parts[0] for path in slot.rglob("results*.json")] == [f"{RUN_DIR_PREFIX}0"]


@pytest.mark.parametrize("phase_sec", [None, 60.0], ids=["session-only", "phase-ends-first"])
def test_a_phase_deadline_that_ends_first_bounds_only_the_discarded_warmup(monkeypatch, suite, phase_sec):
    """The measured round keeps the session's deadline: a phase change waits for its booking."""
    attempts = (LaunchAttempt(outcome=READY, metrics=_COLD), LaunchAttempt(outcome=READY, metrics=_HOT))
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    clock = rounds.clock
    session_sec, started = 600.0, clock.wall()
    phase = None if phase_sec is None else Deadline.after(phase_sec, now=clock.monotonic())
    stop = GridStop(session_deadline_sec=clock.monotonic() + session_sec, phase_deadline=phase)

    with installed_clock(clock):
        [row] = asyncio.run(
            run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point), stop=stop)
        )

    assert row.status == "succeeded", row
    warmup, measured = (call.config.deadlines.deadline_unix for call in rounds.calls)
    assert measured == int(started + session_sec)
    assert warmup == int(started + (session_sec if phase_sec is None else phase_sec))


def test_a_variant_whose_point_does_not_warm_up_is_one_round_that_owns_its_server(monkeypatch, suite, unheld):
    row, calls, slot = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT))

    assert row.status == "succeeded", row
    (config,) = (call.config for call in calls)
    assert (config.lifecycle.mode, config.fixed.seed) == (LifecycleMode.OWN, MEASURED_SEED)
    assert "run_grid_warmup_discarded_first" not in row.nonfatal_warnings
    assert row.report_path == str(RunLayout(slot / f"{RUN_DIR_PREFIX}0").report)
    assert suite.stopped == [slot]


@pytest.mark.parametrize("extra_envs", [{}, {"RUN_EVAL": "true", "NUM_PROMPTS": "1"}])
def test_a_callers_boot_round_boots_its_server_and_leaves_it_up(monkeypatch, suite, tmp_path, extra_envs):
    """A boot-only launch runs no load and no eval, whatever the point's client knobs say and without a phase env."""
    pid_dir = tmp_path / "arm"
    lifecycle = {"cleanup": False, "pid_dir": str(pid_dir), "port": 30001}
    row, calls, slot = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY),
        launch=GridLaunch(server_lifecycle=lifecycle, lifecycle_boot_only=True),
        extra_envs=extra_envs,
    )

    assert row.status == "succeeded", row
    assert row.note == "server_lifecycle_boot_only"
    (config,) = (call.config for call in calls)
    assert (config.lifecycle.mode, config.lifecycle.pid_dir) == (LifecycleMode.BOOT, str(pid_dir))
    assert config.workload is WorkloadKind.NONE
    assert config.fixed is None
    assert not config.eval.enabled
    assert config.server.port == 30001
    assert pid_dir not in suite.stopped


@pytest.mark.parametrize(("cleanup", "stops_caller"), [(False, False), (True, True)])
def test_a_callers_reuse_round_reuses_its_server_and_the_last_one_takes_it_down(
    monkeypatch, suite, tmp_path, cleanup, stops_caller
):
    pid_dir = tmp_path / "arm"
    lifecycle = {"cleanup": cleanup, "pid_dir": str(pid_dir), "port": 30001}
    row, calls, _ = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_HOT),
        launch=GridLaunch(server_lifecycle=lifecycle, server_already_ready=True),
    )

    assert row.status == "succeeded", row
    (config,) = (call.config for call in calls)
    assert (config.lifecycle.mode, config.lifecycle.pid_dir) == (LifecycleMode.REUSE, str(pid_dir))
    assert (pid_dir in suite.stopped) is stops_caller


def test_a_round_stopped_at_a_step_fails_the_variant_under_that_steps_class(monkeypatch, suite, unheld):
    row, _, slot = _play(monkeypatch, suite, LaunchAttempt(outcome=STAGE_FAILED, stage="argv_parse"))

    assert row.status == "failed"
    assert row.error_class == ARGV_INVALID
    assert row.workspace == str(slot / f"{RUN_DIR_PREFIX}0")


def test_a_server_that_exits_on_a_capacity_error_fails_the_variant_as_engine_capacity(monkeypatch, suite, unheld):
    row, _, _ = _play(
        monkeypatch, suite, LaunchAttempt(outcome=STAGE_FAILED, stage="engine_init", server_log=_CAPACITY_SERVER_LOG)
    )

    assert row.status == "failed"
    assert row.error_class == ENGINE_CAPACITY


@pytest.mark.parametrize("warms", [False, True], ids=["lone-round", "warmup"])
def test_a_cancelled_round_is_the_runs_own_stop(monkeypatch, suite, request, warms):
    """Whichever round of a variant the cancel reaches first: the variant is skipped and its server stopped once."""
    if not warms:
        request.getfixturevalue("unheld")
    monkeypatch.setattr(suite_rounds, "cancel_scope_listener", _cancelled_scope)
    row, calls, slot = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT))

    assert row.status == "skipped"
    assert row.error_class == ORCHESTRATOR_CANCELLED_CLASS
    assert len(calls) == 1
    assert suite.stopped == [slot]


def test_a_round_refused_before_it_ran_fails_as_staging_missing_and_says_why_in_the_log(
    monkeypatch, suite, tmp_path, caplog
):
    monkeypatch.setenv(TOOL_CACHE_ENV, str(tmp_path / "unstaged"))

    with caplog.at_level("WARNING", logger=_grid_runner.log.name):
        row, calls, _ = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT))

    assert row.status == "failed"
    assert row.error_class == FailedStep.STAGING_MISSING.value
    assert not calls
    (refusal,) = [r for r in caplog.records if "refused before it ran" in r.getMessage()]
    assert row.error in refusal.getMessage() and refusal.exc_info is not None


def test_a_variants_unset_envs_are_not_inherited_by_its_rounds(monkeypatch, suite):
    _, calls, _ = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT),
        unset_envs=[_SCRUBBED_ENV],
    )

    warmup, measured = (call.config for call in calls)
    assert _SCRUBBED_ENV not in warmup.server.env_inherit and _SCRUBBED_ENV not in measured.server.env_inherit


def test_the_servers_a_round_starts_are_pinned_to_the_cpus_the_host_names(monkeypatch, suite):
    cpus = tuple(sorted(os.sched_getaffinity(0))[:2])
    monkeypatch.setenv("HYPERLOOM_SERVER_CPUS", ",".join(str(cpu) for cpu in cpus))
    _, calls, _ = _play(
        monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_COLD), LaunchAttempt(outcome=READY, metrics=_HOT)
    )

    assert [call.config.server.cpus for call in calls] == [cpus, cpus]


def test_server_cpus_that_name_no_cpu_list_refuse_the_round_as_an_invalid_config(monkeypatch, suite):
    monkeypatch.setenv("HYPERLOOM_SERVER_CPUS", "all")
    row, calls, _ = _play(monkeypatch, suite, LaunchAttempt(outcome=READY))

    assert row.status == "failed" and row.error_class == failed_step_error_class(FailedStep.CONFIG_INVALID), row
    assert "HYPERLOOM_SERVER_CPUS" in row.error
    assert not calls


def test_a_point_without_a_frozen_base_is_refused_as_an_invalid_config_and_runs_nothing(monkeypatch, suite, tmp_path):
    monkeypatch.setattr(_grid_runner, "_resolve_session_dir", lambda: tmp_path / "no_base")
    row, calls, _ = _play(monkeypatch, suite, LaunchAttempt(outcome=READY))

    assert row.status == "failed" and row.error_class == failed_step_error_class(FailedStep.CONFIG_INVALID), row
    assert not calls


class _Scope:
    reason = "operator cancelled"


class _cancelled_scope:
    def __enter__(self):
        return _Scope()

    def __exit__(self, *exc):
        return False


def test_a_multi_node_variant_restarts_the_cluster_and_measures_it_externally_after_a_discarded_pass(
    monkeypatch, suite
):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: "http://10.0.0.2:30000")
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setattr(_multi_node_env, "mn_bench_warmup_enabled", lambda: True)
    restarts: list[str] = []

    async def restart(run, at, rounds):
        restarts.append(at.variant.name)

    monkeypatch.setattr(_grid_runner, "_restart_multi_node_server", restart)

    row, calls, slot = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD, eval_scores=_SCORE),
        LaunchAttempt(outcome=READY, metrics=_HOT, eval_scores=_SCORE),
        extra_envs=_EVALUATING,
    )

    assert row.status == "succeeded", row
    assert row.output_throughput == pytest.approx(_HOT["output_throughput"])
    assert restarts == ["vA"]
    # Both passes run the variant's own point, eval included: a multi-node variant has no pair of its own.
    assert _sequence(calls) == [
        (LifecycleMode.EXTERNAL, WARMUP_SEED, True),
        (LifecycleMode.EXTERNAL, MEASURED_SEED, True),
    ]
    warm, measured = (call.config for call in calls)
    assert (measured.server.host, measured.server.port) == ("10.0.0.2", 30000)
    assert Path(warm.paths.run_dir).parent == slot / "mn_warmup"
    # The variant holds no test of its own: nothing but the round's own teardown touches its slot.
    assert suite.stopped == [slot]
    assert "run_grid_warmup_discarded_first" not in row.nonfatal_warnings


_CALLER_HELD = GridLaunch(server_lifecycle={"cleanup": True, "pid_dir": "/held", "port": 30001})


@pytest.mark.parametrize(
    ("agentx", "launch", "rounds"),
    [(False, GridLaunch(), 2), (True, GridLaunch(), 1), (False, _CALLER_HELD, 1)],
    ids=["inferencex", "agentx", "caller-held"],
)
def test_a_grid_left_its_servers_admits_a_variant_for_a_warmup_unless_its_point_is_measured_once(
    monkeypatch, suite, agentx, launch, rounds
):
    """An AgentX replay is long and stable and warms its own client inside the round, so it is measured once."""
    document = yaml.safe_load(suite.point.read_text())
    document["benchmark"]["agentx"] = {"enabled": agentx}
    suite.point.write_text(yaml.safe_dump(document), encoding="utf-8")
    planned = []

    async def measure_variant(run, at):
        planned.append(run.rounds_per_variant)
        return SimpleNamespace(status="succeeded")

    monkeypatch.setattr(_grid_runner, "_measure_variant", measure_variant)
    monkeypatch.setattr(_grid_runner, "_attach_grid_launch_evidence", lambda *args, **kwargs: None)

    asyncio.run(
        run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point), launch=launch)
    )

    assert planned == [rounds]


def _agentx(point: Path) -> None:
    document = yaml.safe_load(point.read_text())
    launch = {"replace_args": True, "append_args": list(_base("sglang").args)}
    document["benchmark"]["agentx"] = {"enabled": True, "launch_overrides": launch}
    document["benchmark"]["envs"]["WEKA_LOADER_OVERRIDE"] = "/traces"
    point.write_text(yaml.safe_dump(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("ready", "mode"), [(False, LifecycleMode.BOOT), (True, LifecycleMode.REUSE)], ids=["boots", "reuses"]
)
def test_an_agentx_round_holds_the_server_its_caller_asked_for(monkeypatch, suite, tmp_path, ready, mode):
    """A held AgentX round boots the server the rounds after it reuse; one that owned its own would boot again."""
    _agentx(suite.point)
    played = []

    def run_round(config, *, cancel=None):  # ScriptedRounds plays no AgentX load: the round's shape is the contract
        played.append(config)
        raise ScenarioError("the round is not played")

    monkeypatch.setattr(suite_rounds, "run_round", run_round)
    held = {"cleanup": False, "pid_dir": str(tmp_path / "held"), "port": 30001}
    launch = GridLaunch(server_lifecycle=held, server_already_ready=ready)

    with contextlib.suppress(ScenarioError):
        asyncio.run(
            run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point), launch=launch)
        )

    assert [(c.workload, c.lifecycle.mode, c.lifecycle.pid_dir) for c in played] == [
        (WorkloadKind.AGENTX, mode, held["pid_dir"])
    ]


@pytest.mark.parametrize(
    ("left_sec", "expected_sec", "measured"),
    [(-1.0, None, 0), (5.0, 10.0, 0), (40.0, 30.0, 0), (300.0, 30.0, 2)],
    ids=["spent", "a-variant-does-not-fit", "the-warmup-does-not-fit", "fits-with-its-warmup"],
)
def test_a_grid_starts_no_variant_the_session_budget_cannot_admit(monkeypatch, suite, left_sec, expected_sec, measured):
    """Admission prices each variant at its expected rounds, the discarded warmup included, not at the backstop."""
    attempts = tuple(LaunchAttempt(outcome=READY, metrics=_HOT) for _ in range(4))
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    clock = rounds.clock
    stop = GridStop(session_deadline_sec=clock.monotonic() + left_sec, variant_expected_sec=expected_sec)

    with installed_clock(clock):
        results = asyncio.run(
            run_grid(
                grid=[GridVariant("v0"), GridVariant("v1")],
                output_root=suite.out,
                recipe=GridRecipe(suite.point),
                stop=stop,
            )
        )

    assert sum(row.status == "succeeded" for row in results) == measured, results
    skipped = [row for row in results if row.status == "skipped"]
    assert len(skipped) == 2 - measured
    assert {row.error_class for row in skipped} <= {SESSION_TIME_EXHAUSTED_CLASS}
    assert len(rounds.calls) == measured * 2


def test_a_variant_whose_warmup_left_no_time_for_its_measured_round_is_skipped_and_its_server_stopped(
    monkeypatch, suite
):
    attempts = (LaunchAttempt(outcome=READY, metrics=_COLD, duration_sec=45.0),)
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    clock = rounds.clock
    stop = GridStop(session_deadline_sec=clock.monotonic() + 70.0, variant_expected_sec=30.0)

    with installed_clock(clock):
        results = asyncio.run(
            run_grid(
                grid=[GridVariant("v0"), GridVariant("v1")],
                output_root=suite.out,
                recipe=GridRecipe(suite.point),
                stop=stop,
            )
        )

    assert [(row.status, row.error_class) for row in results] == [("skipped", SESSION_TIME_EXHAUSTED_CLASS)] * 2
    assert [call.config.lifecycle.mode for call in rounds.calls] == [LifecycleMode.BOOT]
    assert suite.stopped == [suite.out / "variant_00_v0"]


@pytest.mark.parametrize(("left_sec", "passes"), [(250.0, 0), (350.0, 2)], ids=["two-rounds-fit", "three-rounds-fit"])
def test_a_multi_node_variant_is_admitted_for_a_warmup_round_beside_its_two_passes(
    monkeypatch, suite, left_sec, passes
):
    """The grid admits a variant before its config says the host holds no server: room for one round it never runs."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    async def restart(run, at, rounds):
        return None

    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: "http://10.0.0.2:30000")
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setattr(_multi_node_env, "mn_bench_warmup_enabled", lambda: True)
    monkeypatch.setattr(_grid_runner, "_restart_multi_node_server", restart)
    attempts = (LaunchAttempt(outcome=READY, metrics=_COLD), LaunchAttempt(outcome=READY, metrics=_HOT))
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    clock = rounds.clock
    stop = GridStop(session_deadline_sec=clock.monotonic() + left_sec, variant_expected_sec=100.0)

    with installed_clock(clock):
        [row] = asyncio.run(
            run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point), stop=stop)
        )

    assert row.status == ("succeeded" if passes else "skipped"), row
    assert [call.config.lifecycle.mode for call in rounds.calls] == [LifecycleMode.EXTERNAL] * passes
