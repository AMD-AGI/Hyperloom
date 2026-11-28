# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grid's callers on a session that froze a base: their rounds run on inference_suite, played by ScriptedRounds.

The contract: a conc sweep arm boots one server on its first rung, measures that rung on it, reuses it for every rung
after, and every rung serves the same launch; the arm's server and an explore variant's server are reaped with
``stop`` on the directory their rounds shared it through, however the rounds ended; an explore variant whose server
can be held is tested on one server on a port of its own, a discarded bench that never evaluates and then the
measured bench with the eval the variant asks for, a variant whose discarded bench failed is recorded with the failed
step's class and not measured, and an AgentX variant is one round that boots in its slot, whose server is stopped
before the variant's accuracy round boots one; explore launches one server once per round, starts no variant its
session or phase budget cannot fit, and launches none that only restates the argv or environment of the session's
base; an explore round the ledger answers whole succeeds without a launch, and
one left with nothing to run fails and reports its grid exhausted; a specialist rebench is one round that owns its
server on the port it reports.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.conc_sweep import runner as conc_sweep
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.actions.executors.serving import _server_lifecycle
from hyperloom.orchestrator.actions.executors.grid._grid_base import TS_SKIPPED_DEDUP, VariantResult
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._suite import held_test
from hyperloom.orchestrator.actions.executors._suite.held_test import RoundPurpose
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import AccuracyRoundOutcome
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors._suite.config import MEASURED_SEED, WARMUP_SEED
from hyperloom.orchestrator.actions.executors import explore as explore_module
from hyperloom.orchestrator.actions.executors.explore import ALREADY_MEASURED, BASELINE_NOOP, ExploreExecutor
from hyperloom.orchestrator.actions.executors.tests.test_grid_suite_round import _COLD, _HOT, suite  # noqa: F401
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _base
from hyperloom.orchestrator.rehearsal import READY, STAGE_FAILED, LaunchAttempt, LaunchScenario, ScriptedRounds
from hyperloom.orchestrator.rehearsal import VirtualClock, installed_clock
from hyperloom.orchestrator.rehearsal.scenario import ScenarioError
from hyperloom.orchestrator.specialists.rebench import RebenchBenchmark, RebenchServer, run_specialist_rebench
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.tests._session_base import freeze_base, session_base
from inference_suite.contracts import FailedStep, LifecycleMode, Workload, WorkloadKind
from inference_suite.tests.test_compose import _launch

_PORT = 30123


@pytest.fixture
def eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every point lifecycle-eligible on ``_PORT``."""
    params = {"eligible": True, "framework": "sglang", "port": _PORT, "reason": ""}
    monkeypatch.setattr(_server_lifecycle, "resolve_lifecycle_params", lambda _path: params)
    monkeypatch.setattr(conc_sweep, "resolve_lifecycle_params", lambda _path: params)


def _rounds(monkeypatch: pytest.MonkeyPatch, count: int) -> ScriptedRounds:
    return _scripted(monkeypatch, *(LaunchAttempt(outcome=READY, metrics=_HOT) for _ in range(count)))


def _scripted(monkeypatch: pytest.MonkeyPatch, *attempts: LaunchAttempt) -> ScriptedRounds:
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    return rounds


def _session(suite: SimpleNamespace) -> Path:
    return suite.point.parent / "session"


def _sweep(monkeypatch, suite: SimpleNamespace, concs: list[int]) -> list:
    """Sweep both arms over ``concs`` and return the config of every round played."""
    rounds = _rounds(monkeypatch, 2 * len(concs))
    state = replace(
        SharedState.load_or_init(_session(suite)),
        framework="sglang",
        tp=2,
        conc=concs[0],
        isl=1024,
        osl=1024,
        baseline_config_path=str(suite.point),
        current_best={"extra_envs": {"SGLANG_X": "1"}},
        cumulative_gain_validated=5.0,
        conc_sweep_concs=list(concs),
    )
    state.save(_session(suite))

    def materialize(src: Path, out_dir: Path, **_kw: object) -> Path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(src, Path(out_dir) / "conc_sweep_base.with_envs.yaml"))

    monkeypatch.setattr(conc_sweep, "prepare_grid_benchmark_config", materialize)
    with installed_clock(rounds.clock):
        payload = asyncio.run(conc_sweep.run_conc_sweep(_session(suite)))
    assert payload["summary"]["successful_pairs"] == len(concs), payload
    return [call.config for call in rounds.calls]


def test_a_conc_sweep_arm_boots_on_its_first_rung_reuses_one_launch_and_is_reaped(
    monkeypatch,
    suite,
    eligible,
):
    configs = _sweep(monkeypatch, suite, [8, 16])

    arms: dict[str, list] = {}
    for config in configs:
        arms.setdefault(config.lifecycle.pid_dir, []).append(config)
    assert len(arms) == 2
    for pid_dir, (boot, reuse) in arms.items():
        assert Path(pid_dir).name.startswith("server_")
        assert (boot.lifecycle.mode, reuse.lifecycle.mode) == (LifecycleMode.BOOT, LifecycleMode.REUSE)
        assert boot.workload is WorkloadKind.FIXED and boot.fixed.concurrency == 8
        assert reuse.fixed.concurrency == 16
        assert boot.server == reuse.server
        assert Path(pid_dir) in suite.stopped


def test_a_conc_sweep_arm_whose_only_rung_booted_its_server_is_reaped_when_the_arm_ends(
    monkeypatch,
    suite,
    eligible,
):
    configs = _sweep(monkeypatch, suite, [8])

    assert [config.lifecycle.mode for config in configs] == [LifecycleMode.BOOT] * 2
    assert {Path(config.lifecycle.pid_dir) for config in configs} <= set(suite.stopped)


@pytest.mark.parametrize("outcome", ["measured", "raised"])
def test_an_explore_variants_server_is_reaped_through_its_slot(monkeypatch, suite, eligible, tmp_path, outcome):
    async def grid(**_kw):
        if outcome == "raised":
            raise RuntimeError("the grid broke")
        return []

    monkeypatch.setattr(explore_module, "run_grid", grid)
    state = SharedState()
    state.baseline_tput = 800.0
    task = SimpleNamespace(
        task_id="explore-1",
        params={
            "config_path": str(suite.point),
            "output_dir": str(tmp_path / "explore"),
            "base_tput": 800.0,
            "keep_threshold_pct": 1.0,
            "grid": [{"name": "v0", "extra_args": "--max-running-requests 64", "extra_envs": {}}],
        },
    )

    run = ExploreExecutor(session_dir=_session(suite))(SimpleNamespace(task=task, extra={"shared_state": state}))
    if outcome == "raised":
        with pytest.raises(RuntimeError, match="the grid broke"):
            asyncio.run(run)
    else:
        asyncio.run(run)

    slot = tmp_path / "explore" / "v00_v0"
    assert suite.stopped == [slot]


def test_a_specialist_rebench_runs_one_round_that_owns_its_server_on_the_port_it_reports(monkeypatch, suite, tmp_path):
    rounds = _rounds(monkeypatch, 1)

    result = asyncio.run(
        run_specialist_rebench(
            output_dir=tmp_path / "rebench",
            benchmark=RebenchBenchmark(config_path=suite.point),
            server=RebenchServer(extra_envs={"RUN_EVAL": "false"}, port=_PORT),
        )
    )

    assert result["ok"] is True, result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    (config,) = (call.config for call in rounds.calls)
    assert config.lifecycle.mode is LifecycleMode.OWN
    assert config.server.port == result["port"] == _PORT


def _explore(
    monkeypatch,
    suite: SimpleNamespace,
    tmp_path: Path,
    rows: list[dict],
    state: SharedState,
    *,
    rounds: ScriptedRounds | None = None,
    phase_deadline: Deadline | None = None,
    **params,
):
    """Run explore over ``rows`` on played rounds; return its result and the rounds launched.

    A row that names no environment switches its eval off; without ``rounds`` every row is played a warmup and a
    decision round that measure alike.
    """
    rounds = rounds or _rounds(monkeypatch, 2 * len(rows))
    task = SimpleNamespace(
        task_id="explore-1",
        params={
            "config_path": str(suite.point),
            "output_dir": str(tmp_path / "explore"),
            "base_tput": 800.0,
            "keep_threshold_pct": 1.0,
            "grid": [{"extra_envs": {"RUN_EVAL": "false"}, **row} for row in rows],
            **params,
        },
    )
    executor = ExploreExecutor(session_dir=_session(suite))
    with installed_clock(rounds.clock):
        extra = {"shared_state": state, "phase_deadline": phase_deadline}
        result = asyncio.run(executor(SimpleNamespace(task=task, extra=extra)))
    return result, rounds.calls


def test_explore_benches_two_rows_that_launch_one_server_once(monkeypatch, suite, tmp_path):
    """The second row spells the first row's change twice, and the server applies the last spelling."""
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = _HOT["output_throughput"]
    rows = [
        {"name": "v_a", "extra_args": "--max-running-requests 64"},
        {"name": "v_b", "extra_args": "--max-running-requests 32 --max-running-requests 64"},
    ]

    result, calls = _explore(monkeypatch, suite, tmp_path, rows, state, base_tput=_HOT["output_throughput"])

    assert [call.config.lifecycle.mode for call in calls] == [LifecycleMode.BOOT, LifecycleMode.REUSE], result
    assert [(row["name"], row["reason"]) for row in result["skipped_dup"]] == [("v_b", "round_dup")]


def _round_shape(config) -> tuple:
    return (config.lifecycle.mode, config.lifecycle.pid_dir, config.fixed.seed, config.eval.enabled)


@pytest.mark.parametrize("run_eval", ["true", "false"])
def test_an_explore_variant_discards_a_bench_then_measures_and_evaluates_as_it_asks_on_one_server(
    monkeypatch, suite, tmp_path, run_eval
):
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    measured = LaunchAttempt(outcome=READY, metrics=_HOT, eval_scores={"exact_match,strict-match": 0.9})
    rounds = _scripted(monkeypatch, LaunchAttempt(outcome=READY, metrics=_COLD), measured)
    rows = [{"name": "v_knob", "extra_envs": {"HL_KNOB": "1", "RUN_EVAL": run_eval}}]

    result, calls = _explore(monkeypatch, suite, tmp_path, rows, state, rounds=rounds)

    slot = tmp_path / "explore" / "v00_v_knob"
    warmup, decision = (call.config for call in calls)
    assert _round_shape(warmup) == (LifecycleMode.BOOT, str(slot), WARMUP_SEED, False)
    assert _round_shape(decision) == (LifecycleMode.REUSE, str(slot), MEASURED_SEED, run_eval == "true")
    assert warmup.server.port == decision.server.port
    assert Path(warmup.paths.run_dir).parent == slot / "warmup_round" / "variant_00_v_knob"
    assert Path(decision.paths.run_dir).parent == slot / "variant_00_v_knob"
    assert suite.stopped == [slot]
    (winner,) = result["winners"]
    assert winner["tput"] == pytest.approx(_HOT["output_throughput"])
    reuse = winner["launch_evidence"]["warm_reuse"]
    assert (reuse["reused_ready_server"], reuse["provenance"]) == (True, "caller_ready_server")
    for config_file in (slot / "warmup_round" / "variant_00_v_knob", slot / "variant_00_v_knob"):
        bench = yaml.safe_load((config_file / "config.yaml").read_text(encoding="utf-8"))["benchmark"]
        assert "server_lifecycle" not in bench and "PORT" not in bench["envs"]
        assert bench["envs"]["RUN_EVAL"] == run_eval


def test_each_explore_variant_is_tested_on_a_server_of_its_own_on_its_own_port(monkeypatch, suite, tmp_path):
    ports = iter(range(23001, 23010))
    monkeypatch.setattr(held_test, "pick_server_port", lambda: next(ports))
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    rows = [
        {"name": "v_a", "extra_args": "--max-running-requests 64"},
        {"name": "v_b", "extra_args": "--max-running-requests 32"},
    ]

    _, calls = _explore(monkeypatch, suite, tmp_path, rows, state)

    slots = [tmp_path / "explore" / "v00_v_a", tmp_path / "explore" / "v01_v_b"]
    assert [(call.config.lifecycle.mode, call.config.lifecycle.pid_dir, call.config.server.port) for call in calls] == [
        (LifecycleMode.BOOT, str(slots[0]), 23001),
        (LifecycleMode.REUSE, str(slots[0]), 23001),
        (LifecycleMode.BOOT, str(slots[1]), 23002),
        (LifecycleMode.REUSE, str(slots[1]), 23002),
    ]
    assert suite.stopped == slots


def test_an_explore_variant_whose_discarded_bench_failed_is_recorded_with_the_steps_class_and_not_measured(
    monkeypatch, suite, tmp_path
):
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    rounds = _scripted(monkeypatch, LaunchAttempt(outcome=STAGE_FAILED, stage="engine_init"))
    rows = [{"name": "v_dies", "extra_args": "--max-running-requests 64"}]

    result, calls = _explore(monkeypatch, suite, tmp_path, rows, state, rounds=rounds)

    slot = tmp_path / "explore" / "v00_v_dies"
    assert [call.config.lifecycle.mode for call in calls] == [LifecycleMode.BOOT]
    (row,) = result["explore_search_update"]["tested"].values()
    assert (row["outcome"], row["reason"], row["stage"]) == ("FAILED", "warmup_failed", "warmup")
    assert row["error_class"] == failed_step_error_class(FailedStep.SERVER_EXITED_BEFORE_READY)
    assert row["workspace"] == str(slot / "warmup_round" / "variant_00_v_dies" / "suite_run_0")
    assert suite.stopped == [slot]
    assert result["status"] == "failed"


def test_an_explore_variant_of_an_agentx_point_is_one_round_that_boots_in_its_slot(monkeypatch, suite, tmp_path):
    document = yaml.safe_load(suite.point.read_text(encoding="utf-8"))
    launch = {"replace_args": True, "append_args": list(_base("sglang").args)}
    document["benchmark"]["agentx"] = {"enabled": True, "launch_overrides": launch}
    document["benchmark"]["envs"]["WEKA_LOADER_OVERRIDE"] = "/traces"
    suite.point.write_text(yaml.safe_dump(document), encoding="utf-8")
    rounds = _rounds(monkeypatch, 1)
    played = []

    def run_round(config, *, cancel=None):  # ScriptedRounds plays no AgentX load: the round's shape is the contract
        played.append(config)
        raise ScenarioError("the round is not played")

    monkeypatch.setattr(suite_rounds, "run_round", run_round)
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    rows = [{"name": "v_knob", "extra_envs": {"HL_KNOB": "1"}}]

    with pytest.raises(ScenarioError):
        _explore(monkeypatch, suite, tmp_path, rows, state, rounds=rounds)

    slot = tmp_path / "explore" / "v00_v_knob"
    (config,) = played
    assert (config.workload, config.lifecycle.mode, config.lifecycle.pid_dir) == (
        WorkloadKind.AGENTX,
        LifecycleMode.BOOT,
        str(slot),
    )
    assert not config.eval.enabled
    assert suite.stopped == [slot]


@pytest.mark.parametrize(
    ("session_minutes", "phase_sec", "error_class"),
    [(3.0, None, "session_time_exhausted"), (600.0, 10.0, "phase_time_exhausted")],
    ids=["session", "phase"],
)
def test_explore_starts_no_variant_the_budget_cannot_fit(
    monkeypatch, suite, tmp_path, session_minutes, phase_sec, error_class
):
    """A variant whose two rounds are expected to take 600s each starts neither with about 60s of the session left nor
    with 10s of the phase."""
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    state.max_minutes = session_minutes
    rows = [{"name": "v_too_long", "extra_args": "--max-running-requests 64"}]
    rounds = _rounds(monkeypatch, 1)
    phase = None if phase_sec is None else Deadline.after(phase_sec, now=rounds.clock.monotonic())

    result, calls = _explore(
        monkeypatch, suite, tmp_path, rows, state, rounds=rounds, phase_deadline=phase, baseline_runtime_sec=600.0
    )

    assert calls == []
    assert result["error_class"] == error_class
    assert "1 variant(s) went unmeasured" in result["error"]
    assert result["explore_search_update"]["tested"] == {}


def test_explore_succeeds_without_a_launch_when_the_ledger_already_measured_its_only_variant(
    monkeypatch, suite, tmp_path
):
    """A variant that lost on this stack and is proposed again alone is answered by its verdict, not launched."""
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    slower = LaunchAttempt(outcome=READY, metrics={"output_throughput": 700.0, "completed_requests": 64})
    rounds = _scripted(monkeypatch, LaunchAttempt(outcome=READY, metrics=_COLD), slower)
    rows = [{"name": "v_slow", "extra_args": "--max-running-requests 64"}]
    first, _ = _explore(monkeypatch, suite, tmp_path, rows, state, rounds=rounds)
    assert [row["outcome"] for row in first["explore_search_update"]["tested"].values()] == ["REVERT"]
    state.apply_explore_search_update(first["explore_search_update"])

    result, calls = _explore(monkeypatch, suite, tmp_path, rows, state, rounds=rounds)

    assert len(calls) == 2, result
    assert [(row["name"], row["reason"]) for row in result["skipped_dup"]] == [("v_slow", ALREADY_MEASURED)]
    assert (result["status"], result["explore_grid_exhausted"]) == ("succeeded", False)


_RECIPE_ARGS = ("--mem-fraction-static", "0.8", "--chunked-prefill-size", "16384")
_RECIPE_ENV = {"SGLANG_USE_AITER": "1", "OMP_NUM_THREADS": "1"}


def _serve_the_recipe_base(suite: SimpleNamespace, monkeypatch, workload: Workload) -> None:
    """Freeze a base whose recipe sets ``_RECIPE_ARGS`` and ``_RECIPE_ENV``, and bind ``suite.point`` to it."""
    session = _session(suite)
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session))
    base = session_base(_RECIPE_ARGS, _RECIPE_ENV)
    if workload is Workload.AGENTX:
        base = replace(base, query=replace(base.query, workload=Workload.AGENTX, isl=None, osl=None))
        document = yaml.safe_load(suite.point.read_text(encoding="utf-8"))
        document["benchmark"]["agentx"] = {"enabled": True, "launch_overrides": {"version": 1}}
        suite.point.write_text(yaml.safe_dump(document), encoding="utf-8")
    freeze_base(session, base)


@pytest.mark.parametrize("workload", [Workload.FIXED, Workload.AGENTX], ids=["fixed", "agentx"])
def test_explore_launches_no_variant_that_only_restates_the_recipe_base(
    monkeypatch, suite, eligible, tmp_path, workload
):
    """The base's own argv and env are already in effect, so restating them launches the base again.

    A restatement is answered without a launch, ahead of the round or at its turn, and its ledger row says why;
    a variant that changes one of the recipe's values still runs.
    """
    _serve_the_recipe_base(suite, monkeypatch, workload)
    launched: list[str] = []

    async def run_grid(*, grid, **_kw):
        (variant,) = grid
        launched.append(variant.name)
        return [VariantResult(variant.name, variant.extra_server_args, dict(variant.extra_envs), status="failed")]

    monkeypatch.setattr(explore_module, "run_grid", run_grid)
    rows = [
        {"name": "restates-recipe-env", "extra_envs": dict(_RECIPE_ENV)},
        {"name": "changes-recipe-arg", "extra_args": "--chunked-prefill-size 8192", "extra_envs": {}},
        {"name": "restates-recipe-args", "extra_args": " ".join(_RECIPE_ARGS), "extra_envs": {}},
        {"name": "changes-recipe-env", "extra_envs": {"SGLANG_USE_AITER": "0"}},
    ]
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0

    result, _ = _explore(monkeypatch, suite, tmp_path, rows, state)

    assert launched == ["changes-recipe-arg", "changes-recipe-env"]
    assert [(row["name"], row["reason"]) for row in result["skipped_dup"]] == [
        ("restates-recipe-env", BASELINE_NOOP),
        ("restates-recipe-args", BASELINE_NOOP),
    ]
    ledger = {row["name"]: row for row in result["explore_search_update"]["tested"].values()}
    for name in ("restates-recipe-env", "restates-recipe-args"):
        assert (ledger[name]["outcome"], ledger[name]["reason"]) == (TS_SKIPPED_DEDUP, BASELINE_NOOP)


def test_an_agentx_variants_server_is_stopped_once_and_before_its_accuracy_round_boots_another(
    monkeypatch, suite, tmp_path
):
    _serve_the_recipe_base(suite, monkeypatch, Workload.AGENTX)
    launches = []
    stopped_before_the_round: list[list[Path]] = []

    async def run_grid(*, grid, output_root, launch, **_kw):
        (variant,) = grid
        launches.append(launch)
        workspace = output_root / "variant_00_x" / "suite_run_0"
        row = VariantResult(variant.name, variant.extra_server_args, dict(variant.extra_envs), status="succeeded")
        return [replace(row, output_throughput=900.0, workspace=str(workspace))]

    async def accuracy_round(slot, launch, state, *, session):
        stopped_before_the_round.append(list(suite.stopped))
        return AccuracyRoundOutcome(slot / "accuracy", "")

    monkeypatch.setattr(explore_module, "run_grid", run_grid)
    monkeypatch.setattr(explore_module, "accuracy_round_due", lambda state: True)
    monkeypatch.setattr(explore_module, "run_candidate_accuracy_round", accuracy_round)
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0
    rows = [{"name": "changes-recipe-arg", "extra_args": "--chunked-prefill-size 8192", "extra_envs": {}}]

    result, _ = _explore(monkeypatch, suite, tmp_path, rows, state)

    slot = tmp_path / "explore" / "v00_changes-recipe-arg"
    (launch,) = launches
    assert (launch.held.purpose, launch.held.test.slot, launch.server_already_ready) == (
        RoundPurpose.MEASURE,
        slot,
        False,
    )
    assert stopped_before_the_round == [[slot]]
    assert suite.stopped == [slot]
    assert [winner["name"] for winner in result["winners"]] == ["changes-recipe-arg"]


def test_an_explore_grid_of_nothing_but_restatements_of_the_base_fails_as_exhausted(monkeypatch, suite, tmp_path):
    """Every variant is dropped ahead of the round, which is left nothing to run and measures nothing."""
    _serve_the_recipe_base(suite, monkeypatch, Workload.FIXED)
    rows = [
        {"name": "restates-recipe-env", "extra_envs": dict(_RECIPE_ENV)},
        {"name": "restates-recipe-args", "extra_args": " ".join(_RECIPE_ARGS), "extra_envs": {}},
    ]
    state = SharedState.load_or_init(_session(suite))
    state.baseline_tput = 800.0

    result, calls = _explore(monkeypatch, suite, tmp_path, rows, state)

    assert calls == []
    assert [row["reason"] for row in result["skipped_dup"]] == [BASELINE_NOOP, BASELINE_NOOP]
    assert (result["status"], result["explore_grid_exhausted"]) == ("failed", True)
