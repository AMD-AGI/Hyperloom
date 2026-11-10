# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A grid whose session froze a base runs its rounds on inference_suite, played here by ``ScriptedRounds``.

The contract: the warmup, the measured, a caller's boot and a caller's reuse round are each one round with its own
run_dir; the server a variant booted for its own rounds is reaped with ``stop`` on its slot, a caller's last reuse
round takes the caller's server down, a failed step and a refusal map to their error classes, and a session that froze
no base runs nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.state.tests._session_base import freeze_base
from hyperloom.orchestrator.actions.executors import _grid_base, _grid_runner
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._grid_runner import GridLaunch, GridRecipe, GridStop, GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import run_grid
from hyperloom.orchestrator.actions.executors._suite.config import MEASURED_SEED
from hyperloom.orchestrator.actions.executors._suite.rounds import RUN_DIR_PREFIX, TOOL_CACHE_ENV
from hyperloom.orchestrator.actions.executors._suite.config import WARMUP_SEED
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _MODEL, _base
from hyperloom.orchestrator.rehearsal import READY, STAGE_FAILED, LaunchAttempt, LaunchScenario, ScriptedRounds
from hyperloom.orchestrator.rehearsal import VirtualClock, installed_clock
from hyperloom.orchestrator.rehearsal.scenario import ScenarioError
from hyperloom.orchestrator.state.failure_evidence import ARGV_INVALID
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS
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
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(tmp_path / "leaks"))
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
    return SimpleNamespace(point=_point(tmp_path / "point.yaml"), out=tmp_path / "out", stopped=stopped)


def _play(monkeypatch, suite: SimpleNamespace, *attempts: LaunchAttempt, launch: GridLaunch, **variant):
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
        launch=GridLaunch(warmup_before_measure=True),
    )

    assert row.status == "succeeded", row
    assert row.output_throughput == pytest.approx(_HOT["output_throughput"])
    assert "run_grid_warmup_discarded_first" in row.nonfatal_warnings
    warmup, measured = (call.config for call in calls)
    assert (warmup.lifecycle.mode, warmup.lifecycle.pid_dir) == (LifecycleMode.BOOT, str(slot))
    assert (measured.lifecycle.mode, measured.lifecycle.pid_dir) == (LifecycleMode.REUSE, str(slot))
    assert (warmup.fixed.seed, measured.fixed.seed) == (WARMUP_SEED, MEASURED_SEED)
    assert row.workspace == str(slot / f"{RUN_DIR_PREFIX}0")
    assert RunLayout(slot / "warmup_round" / f"{RUN_DIR_PREFIX}0").report.is_file()
    assert set(suite.stopped) == {slot}


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
    launch = GridLaunch(warmup_before_measure=True)

    with installed_clock(clock):
        [row] = asyncio.run(
            run_grid(
                grid=[GridVariant("vA")],
                output_root=suite.out,
                recipe=GridRecipe(suite.point),
                launch=launch,
                stop=stop,
            )
        )

    assert row.status == "succeeded", row
    warmup, measured = (call.config.deadlines.deadline_unix for call in rounds.calls)
    assert measured == int(started + session_sec)
    assert warmup == int(started + (session_sec if phase_sec is None else phase_sec))


def test_a_lone_round_owns_its_server_and_its_run_dir_is_the_variant_workspace(monkeypatch, suite):
    row, calls, slot = _play(
        monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT), launch=GridLaunch(warmup_before_measure=False)
    )

    assert row.status == "succeeded", row
    (config,) = (call.config for call in calls)
    assert config.lifecycle.mode is LifecycleMode.OWN
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
        launch=GridLaunch(server_lifecycle=lifecycle, lifecycle_boot_only=True, warmup_before_measure=False),
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
        launch=GridLaunch(server_lifecycle=lifecycle, server_already_ready=True, warmup_before_measure=False),
    )

    assert row.status == "succeeded", row
    (config,) = (call.config for call in calls)
    assert (config.lifecycle.mode, config.lifecycle.pid_dir) == (LifecycleMode.REUSE, str(pid_dir))
    assert (pid_dir in suite.stopped) is stops_caller


def test_a_round_stopped_at_a_step_fails_the_variant_under_that_steps_class(monkeypatch, suite):
    row, _, slot = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=STAGE_FAILED, stage="argv_parse"),
        launch=GridLaunch(warmup_before_measure=False),
    )

    assert row.status == "failed"
    assert row.error_class == ARGV_INVALID
    assert row.workspace == str(slot / f"{RUN_DIR_PREFIX}0")


def test_a_cancelled_round_is_the_runs_own_stop(monkeypatch, suite):
    monkeypatch.setattr(suite_rounds, "cancel_scope_listener", _cancelled_scope)
    row, _, _ = _play(
        monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT), launch=GridLaunch(warmup_before_measure=False)
    )

    assert row.status == "skipped"
    assert row.error_class == ORCHESTRATOR_CANCELLED_CLASS


def test_a_round_refused_before_it_ran_fails_as_staging_missing_and_says_why_in_the_log(
    monkeypatch, suite, tmp_path, caplog
):
    monkeypatch.setenv(TOOL_CACHE_ENV, str(tmp_path / "unstaged"))

    with caplog.at_level("WARNING", logger=_grid_runner.log.name):
        row, calls, _ = _play(
            monkeypatch,
            suite,
            LaunchAttempt(outcome=READY, metrics=_HOT),
            launch=GridLaunch(warmup_before_measure=False),
        )

    assert row.status == "failed"
    assert row.error_class == FailedStep.STAGING_MISSING.value
    assert not calls
    (refusal,) = [r for r in caplog.records if "refused before it ran" in r.getMessage()]
    assert row.error in refusal.getMessage() and refusal.exc_info is not None


def test_a_variants_unset_envs_are_not_inherited_by_its_round(monkeypatch, suite):
    _, calls, _ = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_HOT),
        launch=GridLaunch(warmup_before_measure=False),
        unset_envs=[_SCRUBBED_ENV],
    )

    (config,) = (call.config for call in calls)
    assert _SCRUBBED_ENV not in config.server.env_inherit


def test_a_point_without_a_frozen_base_is_refused_as_an_invalid_config_and_runs_nothing(monkeypatch, suite, tmp_path):
    monkeypatch.setattr(_grid_runner, "_resolve_session_dir", lambda: tmp_path / "no_base")
    row, calls, _ = _play(
        monkeypatch, suite, LaunchAttempt(outcome=READY), launch=GridLaunch(warmup_before_measure=False)
    )

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
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT),
        launch=GridLaunch(warmup_before_measure=False),
    )

    assert row.status == "succeeded", row
    assert row.output_throughput == pytest.approx(_HOT["output_throughput"])
    assert restarts == ["vA"]
    warm, measured = (call.config for call in calls)
    assert {warm.lifecycle.mode, measured.lifecycle.mode} == {LifecycleMode.EXTERNAL}
    assert (measured.server.host, measured.server.port) == ("10.0.0.2", 30000)
    assert (warm.fixed.seed, measured.fixed.seed) == (WARMUP_SEED, MEASURED_SEED)
    assert Path(warm.paths.run_dir).parent == slot / "mn_warmup"


@pytest.mark.parametrize(("agentx", "rounds"), [(False, 2), (True, 1)], ids=["inferencex", "agentx"])
def test_a_grid_asked_to_warm_up_plans_a_warmup_round_unless_its_point_is_measured_once(
    monkeypatch, suite, agentx, rounds
):
    """An AgentX replay is long and stable and warms its own client inside the round, so it is measured once."""
    document = yaml.safe_load(suite.point.read_text())
    document["benchmark"]["agentx"] = {"enabled": agentx}
    suite.point.write_text(yaml.safe_dump(document), encoding="utf-8")
    planned = []

    async def measure_variant(run, at):
        planned.append((run.auto_warmup_requested, run.rounds_per_variant))
        return SimpleNamespace(status="succeeded")

    monkeypatch.setattr(_grid_runner, "_measure_variant", measure_variant)
    monkeypatch.setattr(_grid_runner, "_attach_grid_launch_evidence", lambda *args, **kwargs: None)
    launch = GridLaunch(warmup_before_measure=True)

    asyncio.run(
        run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point), launch=launch)
    )

    assert planned == [(not agentx, rounds)]


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
    launch = GridLaunch(server_lifecycle=held, server_already_ready=ready, warmup_before_measure=False)

    with contextlib.suppress(ScenarioError):
        asyncio.run(
            run_grid(grid=[GridVariant("vA")], output_root=suite.out, recipe=GridRecipe(suite.point), launch=launch)
        )

    assert [(c.workload, c.lifecycle.mode, c.lifecycle.pid_dir) for c in played] == [
        (WorkloadKind.AGENTX, mode, held["pid_dir"])
    ]
