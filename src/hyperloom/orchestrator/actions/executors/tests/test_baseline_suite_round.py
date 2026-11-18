# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline whose session froze a base runs its rounds on inference_suite, played here by ``ScriptedRounds``.

The contract: each round writes its own run_dir with ``launch.json`` below the round's directory, the cold-start pair
boots one server the measured round reuses and the attempt reaps it with ``stop`` on its slot, a failed step maps
to its error class, and a round the budget cannot pay for -- or a PRELUDE retry past the optimization reserve -- boots
nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.state.tests._session_base import freeze_base
from hyperloom.orchestrator.actions.executors import baseline
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors._suite.config import MEASURED_SEED, WARMUP_SEED
from hyperloom.orchestrator.actions.executors._suite.rounds import RUN_DIR_PREFIX, TOOL_CACHE_ENV
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _MODEL, _base
from hyperloom.orchestrator.rehearsal import READY, STAGE_FAILED, LaunchAttempt, LaunchScenario, ScriptedRounds
from hyperloom.orchestrator.rehearsal import VirtualClock
from hyperloom.orchestrator.rehearsal.scenario import ScenarioError
from hyperloom.orchestrator.actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS
from hyperloom.orchestrator.phases.machine_state import RETRY_PAST_THE_RESERVE
from hyperloom.orchestrator.state.failure_evidence import ARGV_INVALID, ENGINE_CAPACITY, SERVER_INIT_DEAD
from inference_suite.runtime.process import Reaped
from inference_suite.contracts import FailedStep, LifecycleMode, RunLayout, WorkloadKind
from inference_suite.tests.test_compose import _launch

_COLD = {"output_throughput": 270.9, "completed_requests": 64}
_HOT = {"output_throughput": 4701.6, "completed_requests": 64}

# vLLM logs the start-failure marker first and the traceback, ending in the cause, below it.
_CAPACITY_SERVER_LOG = (
    "(EngineCore pid=7) ERROR core.py:1349] EngineCore failed to start.\n"
    "(EngineCore pid=7) Traceback (most recent call last):\n"
    + "".join(f"(EngineCore pid=7)     frame {i}\n" for i in range(55))
    + "(EngineCore pid=7) ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (237).\n"
    "(APIServer pid=1) RuntimeError: Engine core initialization failed. Failed core proc(s): {}\n"
)


def _point(path: Path) -> Path:
    envs = {"TP": 2, "CONC": 8, "ISL": 256, "OSL": 256, "NUM_PROMPTS": 64, "NUM_WARMUPS": 2, "RUN_EVAL": "false"}
    point = {
        "benchmark": {
            "framework": "sglang",
            "model": _MODEL,
            "precision": "bf16",
            "run_mode": "local",
            "envs": envs,
            "timeout_seconds": 600,
            "profiler": {"torch_profiler": {"enabled": False}},
            "gpu_selection": {"auto": False},
        }
    }
    path.write_text(yaml.safe_dump(point), encoding="utf-8")
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
    monkeypatch.setattr(suite_rounds, "_resolve_probe_python", lambda framework: sys.executable)
    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: None)

    stopped: list[Path] = []
    monkeypatch.setattr(baseline, "stop", lambda pid_dir: stopped.append(Path(pid_dir)) or Reaped((), ()))
    return SimpleNamespace(session=session, point=_point(tmp_path / "point.yaml"), ws=tmp_path / "ws", stopped=stopped)


class _ThroughputOnly(BaselineExecutor):
    """``ScriptedRounds`` plays no eval, so the rounds run their load alone."""

    runs_accuracy_eval = False


def _play(
    monkeypatch, suite: SimpleNamespace, *attempts: LaunchAttempt, state: object = None, **params
) -> tuple[dict, ScriptedRounds]:
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    executor = _ThroughputOnly(
        default_config_path=suite.point,
        session_dir=suite.session,
        shared_state=state or SimpleNamespace(session_budget_usable_sec=lambda: None),
    )
    task = SimpleNamespace(
        task_id="t-baseline", params={"output_dir": str(suite.ws), "timeout_sec": 10, "gpu_type": "mi300x", **params}
    )
    return asyncio.run(executor(SimpleNamespace(task=task, extra={}))), rounds


def test_the_cold_start_pair_boots_one_server_the_measured_round_reuses_and_the_slot_is_reaped(monkeypatch, suite):
    result, rounds = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, duration_sec=60.0, metrics=_COLD),
        LaunchAttempt(outcome=READY, duration_sec=30.0, metrics=_HOT),
    )

    assert result["status"] == "succeeded", result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    assert result["warmup_round_tput"] == pytest.approx(_COLD["output_throughput"])
    warmup, measured = (call.config for call in rounds.calls)
    assert (warmup.lifecycle.mode, warmup.lifecycle.pid_dir) == (LifecycleMode.BOOT, str(suite.ws))
    assert (measured.lifecycle.mode, measured.lifecycle.pid_dir) == (LifecycleMode.REUSE, str(suite.ws))
    assert (warmup.fixed.seed, measured.fixed.seed) == (WARMUP_SEED, MEASURED_SEED)
    assert suite.stopped == [suite.ws]


def test_each_round_leaves_its_run_dir_and_launch_below_its_round_directory(monkeypatch, suite):
    result, _ = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT),
    )

    run_dir = suite.ws / "measure_round" / f"{RUN_DIR_PREFIX}0"
    assert result["workspace"] == str(run_dir)
    assert result["result_dir"] == str(run_dir)
    assert result["report_path"] == str(RunLayout(run_dir).report)
    assert RunLayout(run_dir).launch.is_file()
    assert RunLayout(suite.ws / "warmup_round" / f"{RUN_DIR_PREFIX}0").report.is_file()


def test_a_lone_round_owns_its_server(monkeypatch, suite):
    result, rounds = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT), baseline_double_run=False)

    assert result["status"] == "succeeded", result
    (config,) = (call.config for call in rounds.calls)
    assert config.lifecycle.mode is LifecycleMode.OWN
    assert result["workspace"] == str(suite.ws / f"{RUN_DIR_PREFIX}0")


@pytest.mark.parametrize(("flag", "kv"), [(None, True), ("0", False)])
def test_a_suite_round_samples_kv_metrics_unless_the_operator_turns_them_off(monkeypatch, suite, flag, kv):
    if flag is not None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_KV_METRICS", flag)
    _, rounds = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT), baseline_double_run=False)

    (config,) = (call.config for call in rounds.calls)
    assert config.telemetry.kv is kv


@pytest.mark.parametrize(
    ("stage", "error_class"), [("argv_parse", ARGV_INVALID), ("weights_loading", SERVER_INIT_DEAD)]
)
def test_a_round_that_stopped_at_a_step_fails_under_that_steps_error_class(monkeypatch, suite, stage, error_class):
    result, _ = _play(monkeypatch, suite, LaunchAttempt(outcome=STAGE_FAILED, stage=stage), baseline_double_run=False)

    assert result["status"] == "failed"
    assert result["error_class"] == error_class
    assert result["run_status"] == "failed"
    assert Path(result["report_path"]).is_file()
    assert suite.stopped == [suite.ws]


def test_a_server_that_exits_on_a_capacity_error_fails_as_engine_capacity(monkeypatch, suite):
    attempt = LaunchAttempt(outcome=STAGE_FAILED, stage="engine_init", server_log=_CAPACITY_SERVER_LOG)

    result, _ = _play(monkeypatch, suite, attempt, baseline_double_run=False)

    assert result["status"] == "failed"
    assert result["error_class"] == ENGINE_CAPACITY
    assert "max_num_seqs (256) exceeds available Mamba cache blocks (237)" in result["error"]


def test_a_round_refused_before_it_ran_fails_as_staging_missing(monkeypatch, suite, tmp_path):
    monkeypatch.setenv(TOOL_CACHE_ENV, str(tmp_path / "unstaged"))

    result, rounds = _play(monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT), baseline_double_run=False)

    assert result["status"] == "failed"
    assert result["error_class"] == FailedStep.STAGING_MISSING.value
    assert not rounds.calls


def test_an_eval_failure_of_a_suite_round_is_eval_rooted() -> None:
    assert BaselineExecutor._is_eval_rooted_failure({"error_class": FailedStep.EVAL_FAILED.value})


def test_a_multi_node_round_restarts_the_cluster_and_measures_it_externally_after_a_discarded_pass(monkeypatch, suite):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: "http://10.0.0.2:30000")
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setattr(_multi_node_env, "mn_bench_warmup_enabled", lambda: True)
    restarts: list[str] = []

    async def restart(self, round_, *, server_args, resolved_model):
        restarts.append(str(round_.output_dir))

    monkeypatch.setattr(BaselineExecutor, "_restart_multi_node_server", restart)

    result, rounds = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_COLD),
        LaunchAttempt(outcome=READY, metrics=_HOT),
        baseline_double_run=False,
    )

    assert result["status"] == "succeeded", result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    assert restarts == [str(suite.ws)]
    warm, measured = (call.config for call in rounds.calls)
    assert {warm.lifecycle.mode, measured.lifecycle.mode} == {LifecycleMode.EXTERNAL}
    assert (warm.server.host, warm.server.port) == ("10.0.0.2", 30000)
    assert (warm.fixed.seed, measured.fixed.seed) == (WARMUP_SEED, MEASURED_SEED)
    assert Path(warm.paths.run_dir).parent == suite.ws / "mn_warmup"
    assert not measured.telemetry.gpu.enabled


def test_an_agentx_baseline_is_measured_once_on_a_server_it_owns(monkeypatch, suite):
    """An AgentX replay is long and stable and warms its own client inside the round: no discarded round before it."""
    document = yaml.safe_load(suite.point.read_text())
    launch = {"replace_args": True, "append_args": list(_base("sglang").args)}
    document["benchmark"]["agentx"] = {"enabled": True, "launch_overrides": launch}
    document["benchmark"]["envs"]["WEKA_LOADER_OVERRIDE"] = "/traces"
    suite.point.write_text(yaml.safe_dump(document), encoding="utf-8")
    played = []

    def run_round(config, *, cancel=None):  # ScriptedRounds plays no AgentX load: the round's shape is the contract
        played.append(config)
        raise ScenarioError("the round is not played")

    monkeypatch.setattr(suite_rounds, "run_round", run_round)
    executor = _ThroughputOnly(
        default_config_path=suite.point,
        session_dir=suite.session,
        shared_state=SimpleNamespace(session_budget_usable_sec=lambda: None),
    )
    params = {"output_dir": str(suite.ws), "timeout_sec": 10, "gpu_type": "mi300x", "baseline_double_run": True}
    task = SimpleNamespace(task_id="t-baseline", params=params)

    with contextlib.suppress(ScenarioError):
        asyncio.run(executor(SimpleNamespace(task=task, extra={})))

    assert [(config.workload, config.lifecycle.mode) for config in played] == [(WorkloadKind.AGENTX, LifecycleMode.OWN)]
    assert not (suite.ws / "warmup_round").exists()


# A measured workload: a 900s cold round of which 550s was the benchmark, a 400s hot pass, so one more variant costs
# 350s of boot plus 400s.
_ROUND_SEC = 900.0
_ONE_MORE_SEC = 750.0


def _prelude(*, usable_sec: float, max_minutes: float = 0.0, leg_failures: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        phase="PRELUDE",
        max_minutes=max_minutes,
        baseline_failures_this_leg=leg_failures,
        baseline_runtime_sec=_ROUND_SEC,
        baseline_post_ready_runtime_sec=550.0,
        baseline_warm_runtime_sec=400.0,
        session_budget_usable_sec=lambda: usable_sec,
    )


@pytest.mark.parametrize("usable_sec", [_ROUND_SEC + _ONE_MORE_SEC - 1.0, _ROUND_SEC - 1.0])
def test_a_round_the_session_cannot_pay_for_with_a_use_after_it_boots_nothing(monkeypatch, suite, usable_sec):
    result, rounds = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_HOT),
        state=_prelude(usable_sec=usable_sec),
        baseline_double_run=False,
    )

    assert not rounds.calls, "GPU time was spent on a round the session cannot finish"
    assert result["status"] == "failed"
    assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
    shortfall = result["budget_shortfall"]
    assert shortfall["round_sec"] == pytest.approx(_ROUND_SEC)
    assert shortfall["one_more_measurement_sec"] == pytest.approx(_ONE_MORE_SEC)
    assert shortfall["expected_cost_sec"] == pytest.approx(_ROUND_SEC + _ONE_MORE_SEC)


def test_a_round_the_session_can_pay_for_with_a_use_after_it_is_measured(monkeypatch, suite):
    result, rounds = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=READY, metrics=_HOT),
        state=_prelude(usable_sec=_ROUND_SEC + _ONE_MORE_SEC + 60.0),
        baseline_double_run=False,
    )

    assert result["status"] == "succeeded", result
    assert len(rounds.calls) == 1


@pytest.mark.parametrize(("leg_failures", "refused"), [(1, True), (0, False)])
def test_a_prelude_retry_past_the_optimization_reserve_is_refused_before_ignition(
    monkeypatch, suite, leg_failures, refused
):
    """Twelve hours with 14718.5s left sits below the 21600s reserve: a retry is refused, a first round is not."""
    state = _prelude(usable_sec=14_718.5, max_minutes=720, leg_failures=leg_failures)

    result, rounds = _play(
        monkeypatch, suite, LaunchAttempt(outcome=READY, metrics=_HOT), state=state, baseline_double_run=False
    )

    if not refused:
        assert result["status"] == "succeeded", result
        return
    assert not rounds.calls
    assert result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
    shortfall = result["budget_shortfall"]
    assert shortfall["reason"] == RETRY_PAST_THE_RESERVE
    assert shortfall["baseline_failures_this_leg"] == 1
    assert shortfall["affordable_sec"] == 0.0
    assert shortfall["bound"] == "optimization_reserve"
