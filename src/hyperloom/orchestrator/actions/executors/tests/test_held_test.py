# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A held test runs a point's rounds on one server, in one order, played here by ``ScriptedRounds``.

The contract: a point that warms up runs a discarded bench that boots the server in the test's slot (warm-up seed,
never an eval) and then, once, the measured bench that reuses it and evaluates as the point says or as the test
forces; a point that does not warm up runs its measured bench alone on a server the test boots; every round runs on
the test's one port; a round out of that order is refused before it runs; the server is stopped once however the test
ends; and a round the host refuses is refused as it would be outside a test.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _multi_node_env
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._suite import held_test
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors._suite.config import MEASURED_SEED, WARMUP_SEED
from hyperloom.orchestrator.actions.executors._suite.held_test import (
    HeldPass,
    HeldRound,
    HeldTest,
    HeldTestOrderError,
    RoundPurpose,
    can_hold,
    warms_up,
)
from hyperloom.orchestrator.actions.executors._suite.rounds import (
    RUN_DIR_PREFIX,
    TOOL_CACHE_ENV,
    SuiteRoundRefused,
    SuiteRoute,
    suite_route,
)
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _base, _point
from hyperloom.orchestrator.rehearsal import READY, STAGE_FAILED, LaunchAttempt, LaunchScenario, ScriptedRounds
from hyperloom.orchestrator.rehearsal import VirtualClock
from hyperloom.orchestrator.rehearsal.scenario import ScenarioError
from hyperloom.orchestrator.state.tests._session_base import freeze_base
from inference_suite.contracts import FailedStep, LifecycleMode, WorkloadKind
from inference_suite.runtime.process import Reaped

WARMUP, MEASURE = RoundPurpose.WARMUP, RoundPurpose.MEASURE
_SCORE = {"exact_match,strict-match": 0.9}
_BOOTED = LaunchAttempt(outcome=READY, duration_sec=60.0)


@pytest.fixture
def held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A session's route, the slot of one test, and the stops and free ports the test asked for."""
    session = tmp_path / "session"
    session.mkdir()
    freeze_base(session, _base("sglang"))
    cache = tmp_path / "tools"
    cache.mkdir()
    (cache / "staging_manifest.json").write_bytes(b'{"schema": "inference-suite-staging/1", "tools": {}}\n')
    monkeypatch.setenv(TOOL_CACHE_ENV, str(cache))
    monkeypatch.setattr(suite_rounds, "_resolve_probe_python", lambda framework: f"/venv/{framework}/python")
    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: None)
    monkeypatch.setattr(suite_rounds, "pick_server_port", lambda: pytest.fail("a round of a test picked its own port"))
    return _Held(suite_route(session), tmp_path / "slot", monkeypatch)


class _Held:
    def __init__(self, route: SuiteRoute, slot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.route, self.slot, self.monkeypatch = route, slot, monkeypatch
        self.stopped: list[Path] = []
        self.survivors: tuple[int, ...] = ()
        self.picked: list[int] = []
        monkeypatch.setattr(held_test, "stop", self._stop)
        monkeypatch.setattr(held_test, "pick_server_port", self._pick)

    def _stop(self, pid_dir: Path) -> Reaped:
        self.stopped.append(Path(pid_dir))
        return Reaped(self.survivors, self.survivors)

    def _pick(self) -> int:
        self.picked.append(23001 + len(self.picked))
        return self.picked[-1]

    def point(self, **envs: str) -> Path:
        path = self.slot / "point.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_point("sglang", **{"RUN_EVAL": "false", **envs})), encoding="utf-8")
        return path

    def agentx_point(self) -> Path:
        path = self.point()
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        launch = {"replace_args": True, "append_args": list(_base("sglang").args)}
        document["benchmark"]["agentx"] = {"enabled": True, "launch_overrides": launch}
        document["benchmark"]["envs"]["WEKA_LOADER_OVERRIDE"] = "/traces"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return path

    def scripted(self, *attempts: LaunchAttempt) -> ScriptedRounds:
        rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock())
        self.monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
        return rounds

    def run(self, test: HeldTest, purpose: RoundPurpose, point: Path):
        round_ = HeldRound(point, self.slot / f"{purpose.value}_round")
        return test.run(purpose, self.route, round_, on_poll=lambda: None)


def _sequence(rounds: ScriptedRounds) -> list[tuple]:
    return [
        (call.config.lifecycle.mode, call.config.lifecycle.pid_dir, call.config.fixed.seed, call.config.eval.enabled)
        for call in rounds.calls
    ]


def test_a_warming_point_boots_on_a_discarded_bench_then_measures_and_evaluates_on_that_server(held):
    point = held.point(RUN_EVAL="true")
    written = point.read_text(encoding="utf-8")
    rounds = held.scripted(_BOOTED, LaunchAttempt(outcome=READY, eval_scores=_SCORE))

    with HeldTest(held.slot) as test:
        warmup = held.run(test, WARMUP, point)
        assert held.stopped == []
        measured = held.run(test, MEASURE, point)
        assert held.stopped == []

    slot = str(held.slot)
    assert _sequence(rounds) == [
        (LifecycleMode.BOOT, slot, WARMUP_SEED, False),
        (LifecycleMode.REUSE, slot, MEASURED_SEED, True),
    ]
    assert warmup.report.eval is None and measured.report.eval is not None
    assert warmup.layout.root == held.slot / "warmup_round" / f"{RUN_DIR_PREFIX}0"
    assert measured.layout.root == held.slot / "measure_round" / f"{RUN_DIR_PREFIX}0"
    assert held.stopped == [held.slot]
    assert point.read_text(encoding="utf-8") == written, "what a round is goes into the round, not the point"


@pytest.mark.parametrize(
    ("run_eval", "must_evaluate", "told", "evaluated"),
    [
        ("true", False, None, True),
        ("false", False, None, False),
        ("false", True, True, True),
        ("true", True, True, True),
    ],
)
def test_the_measured_round_evaluates_as_its_point_says_unless_the_test_forces_it(
    held, run_eval, must_evaluate, told, evaluated
):
    point = held.point(RUN_EVAL=run_eval)
    rounds = held.scripted(_BOOTED, LaunchAttempt(outcome=READY, eval_scores=_SCORE))

    with HeldTest(held.slot, must_evaluate=must_evaluate) as test:
        assert (HeldPass(test, WARMUP).evaluates, HeldPass(test, MEASURE).evaluates) == (False, told)
        held.run(test, WARMUP, point)
        held.run(test, MEASURE, point)

    assert [call.config.eval.enabled for call in rounds.calls] == [False, evaluated]


@pytest.mark.parametrize("named", [{"PORT": "8123"}, {}], ids=["point_names_a_port", "point_names_none"])
@pytest.mark.parametrize("given", [23777, None], ids=["caller_names_the_port", "test_takes_a_free_port"])
def test_every_round_of_a_test_runs_on_its_one_port_whatever_the_point_names(held, named, given):
    point = held.point(**named)
    if not named:
        document = yaml.safe_load(point.read_text(encoding="utf-8"))
        del document["benchmark"]["envs"]["PORT"]
        point.write_text(yaml.safe_dump(document), encoding="utf-8")
    rounds = held.scripted(_BOOTED, LaunchAttempt(outcome=READY))

    with HeldTest(held.slot, port=given) as test:
        held.run(test, WARMUP, point)
        held.run(test, MEASURE, point)

    assert held.picked == ([] if given else [23001])
    assert [call.config.server.port for call in rounds.calls] == [given or 23001] * 2


def test_a_pass_runs_its_round_of_the_test(held):
    point = held.point()
    rounds = held.scripted(_BOOTED, LaunchAttempt(outcome=READY))

    with HeldTest(held.slot) as test:
        for purpose in (WARMUP, MEASURE):
            round_ = HeldRound(point, held.slot / purpose.value)
            HeldPass(test, purpose).run(held.route, round_, on_poll=lambda: None)

    assert [call.config.lifecycle.mode for call in rounds.calls] == [LifecycleMode.BOOT, LifecycleMode.REUSE]


@pytest.mark.parametrize(
    ("asked", "why"),
    [
        ([MEASURE], "follows its warm-up"),
        ([WARMUP, WARMUP], "has run its warmup round"),
        ([WARMUP, MEASURE, MEASURE], "has run its measure round"),
        ([WARMUP, MEASURE, WARMUP], "has run its warmup round"),
    ],
)
def test_a_round_out_of_order_is_refused_before_it_runs(held, asked, why):
    point = held.point()
    rounds = held.scripted(*(LaunchAttempt(outcome=READY) for _ in asked))

    with HeldTest(held.slot) as test:
        for purpose in asked[:-1]:
            held.run(test, purpose, point)
        with pytest.raises(HeldTestOrderError, match=why):
            held.run(test, asked[-1], point)

    assert len(rounds.calls) == len(asked) - 1
    assert held.stopped == [held.slot]


def test_the_measured_round_is_refused_after_a_warmup_that_failed(held):
    point = held.point()
    rounds = held.scripted(LaunchAttempt(outcome=STAGE_FAILED, stage="weights_loading"), LaunchAttempt(outcome=READY))

    with HeldTest(held.slot) as test:
        warmup = held.run(test, WARMUP, point)
        assert warmup.report.failed_step is not None
        with pytest.raises(HeldTestOrderError, match=f"failed at {warmup.report.failed_step.value}"):
            held.run(test, MEASURE, point)

    assert len(rounds.calls) == 1


def test_a_test_that_has_stopped_its_server_runs_no_further_round(held):
    point = held.point()
    rounds = held.scripted(_BOOTED, LaunchAttempt(outcome=READY))
    held.survivors = (4242,)

    with HeldTest(held.slot) as test:
        held.run(test, WARMUP, point)
        assert test.stop() == (4242,)
        with pytest.raises(HeldTestOrderError, match="has stopped its server"):
            held.run(test, MEASURE, point)
        assert test.stop() == (4242,)

    assert len(rounds.calls) == 1
    assert held.stopped == [held.slot], "the server is stopped once, whoever asks again"


@pytest.mark.parametrize(("run_eval", "evaluated"), [("true", True), ("false", False)])
def test_a_point_that_does_not_warm_up_is_measured_once_on_a_server_the_test_boots(
    held, monkeypatch, run_eval, evaluated
):
    """Its one round evaluates as the point says: the eval a test can force is the one after a warmed-up bench."""
    monkeypatch.setattr(held_test, "warms_up", lambda config_path: False)
    point = held.point(RUN_EVAL=run_eval)
    rounds = held.scripted(LaunchAttempt(outcome=READY, eval_scores=_SCORE), LaunchAttempt(outcome=READY))

    with HeldTest(held.slot, must_evaluate=True) as test:
        with pytest.raises(HeldTestOrderError, match="measured without a warm-up"):
            held.run(test, WARMUP, point)
        held.run(test, MEASURE, point)
        with pytest.raises(HeldTestOrderError, match="has run its measure round"):
            held.run(test, MEASURE, point)

    assert _sequence(rounds) == [(LifecycleMode.BOOT, str(held.slot), MEASURED_SEED, evaluated)]
    assert held.stopped == [held.slot]


def test_a_warmup_is_refused_once_the_test_has_measured(held, monkeypatch):
    point = held.point()
    rounds = held.scripted(LaunchAttempt(outcome=READY), LaunchAttempt(outcome=READY))

    with HeldTest(held.slot) as test:
        monkeypatch.setattr(held_test, "warms_up", lambda config_path: False)
        held.run(test, MEASURE, point)
        monkeypatch.setattr(held_test, "warms_up", lambda config_path: True)
        with pytest.raises(HeldTestOrderError, match="the warm-up is the first round"):
            held.run(test, WARMUP, point)

    assert len(rounds.calls) == 1


def test_an_agentx_point_boots_its_one_round_in_the_slot_and_is_stopped_when_the_round_raises(held, monkeypatch):
    point = held.agentx_point()
    played = []

    def run_round(config, *, cancel=None):  # ScriptedRounds plays no AgentX load: the round's shape is the contract
        played.append(config)
        raise ScenarioError("the round is not played")

    monkeypatch.setattr(suite_rounds, "run_round", run_round)

    with pytest.raises(ScenarioError), HeldTest(held.slot, must_evaluate=True) as test:
        held.run(test, MEASURE, point)

    (config,) = played
    assert (config.workload, config.lifecycle.mode, config.lifecycle.pid_dir) == (
        WorkloadKind.AGENTX,
        LifecycleMode.BOOT,
        str(held.slot),
    )
    assert not config.eval.enabled
    assert config.server.port == 23001
    assert held.stopped == [held.slot]


def test_leaving_the_test_on_an_exception_stops_the_server_and_names_what_outlived_it(held, caplog):
    point = held.point()
    held.scripted(_BOOTED)
    held.survivors = (4242, 4243)

    with caplog.at_level(logging.ERROR, logger=held_test.__name__), pytest.raises(RuntimeError, match="between rounds"):
        with HeldTest(held.slot) as test:
            held.run(test, WARMUP, point)
            raise RuntimeError("between rounds")

    assert held.stopped == [held.slot]
    assert "(4242, 4243)" in caplog.text and str(held.slot) in caplog.text


def test_a_test_that_ran_no_round_still_stops_its_slot(held):
    with HeldTest(held.slot):
        pass

    assert held.stopped == [held.slot]
    assert held.picked == []


def test_a_round_the_host_refuses_passes_through_and_leaves_the_order_where_it_was(held):
    point = held.point()
    rounds = held.scripted(_BOOTED)
    unstaged = replace(held.route, manifest_sha256=None)

    with HeldTest(held.slot) as test:
        with pytest.raises(SuiteRoundRefused) as refused:
            test.run(WARMUP, unstaged, HeldRound(point, held.slot / "warmup_round"), on_poll=lambda: None)
        assert refused.value.error_class == failed_step_error_class(FailedStep.STAGING_MISSING)
        assert rounds.calls == []
        held.run(test, WARMUP, point)

    assert [call.config.lifecycle.mode for call in rounds.calls] == [LifecycleMode.BOOT]


def test_a_served_unprofiled_fixed_point_on_one_node_warms_up_and_no_other_does(held, monkeypatch):
    """An AgentX point's server can be held too, for its one round; no other point's can."""
    fixed = held.point()
    assert (can_hold(fixed), warms_up(fixed)) == (True, True)

    assert (can_hold(held.agentx_point()), warms_up(held.agentx_point())) == (True, False)

    profiled = yaml.safe_load(fixed.read_text(encoding="utf-8"))
    profiled["benchmark"].pop("agentx")
    profiled["benchmark"]["profiler"] = {"torch_profiler": {"enabled": True}}
    fixed.write_text(yaml.safe_dump(profiled), encoding="utf-8")
    assert (can_hold(fixed), warms_up(fixed)) == (False, False)

    scriptable = held.slot / "scriptable.yaml"
    scriptable.write_text(yaml.safe_dump({"benchmark": {"framework": "xdit"}}), encoding="utf-8")
    assert (can_hold(scriptable), warms_up(scriptable)) == (False, False)

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    assert (can_hold(held.point()), warms_up(held.point())) == (False, False)
