# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy score a cold-start pair's result carries: that of its measured round, which runs the eval after its
bench. The discarded warm-up never evaluates, so a result standing on the warm-up's figure carries none."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors._suite.held_test import HeldPass, HeldTest, RoundPurpose
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.benchmark_run.records import (
    ACCURACY_STAGE_KEY,
    EVAL_SCORE_KEYS,
    MEASURE_ROUND_DROPPED_WARNING,
    BenchmarkAttempt,
    _BenchmarkRound,
)
from hyperloom.orchestrator.actions.executors.benchmark_run.rounds import _BenchmarkRoundsMixin
from hyperloom.orchestrator.actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS
from hyperloom.orchestrator.state.shared_state import SharedState

MEASURED_SCORE = {
    "accuracy": 0.96,
    "accuracy_task": "gsm8k",
    "accuracy_metric": "exact_match,strict-match",
    "accuracy_source": "measure_round/eval/results.json",
    "accuracy_definition_digest": "sha256:measured",
    "accuracy_eval_kind": "",
}
MEASURED_FIRST = "baseline_double_run_discarded_first"
DRIFT_ANCHOR = "baseline_downward_drift_anchor"
EVAL_FAILED = "post_measure_accuracy_failed"


class _StubbedPair(_BenchmarkRoundsMixin):
    """A pair whose rounds return prepared results, in the order the sequencer asks for them."""

    shared_state = None

    def __init__(self, results: list[dict[str, Any]], *, hot_pass_affordable: bool = True):
        self.results = results
        self.hot_pass_affordable = hot_pass_affordable
        self.asked: list[tuple[str, bool | None]] = []

    async def _run_guarded_round(self, attempt, label, round_dir, held):
        self.asked.append((label, held.evaluates))
        return self.results[len(self.asked) - 1]

    def _measure_round_affordable(self, warmup_result, *, ctx_extra=None):
        return self.hot_pass_affordable, {"reason": "priced by the test"}


def _round(round_dir: Path, throughput: float, **extra: Any) -> dict[str, Any]:
    """The result of a round that measured in ``round_dir``; the directory itself is never created."""
    return {
        "status": "succeeded",
        "valid_measurement": True,
        "output_throughput": throughput,
        "output_dir": str(round_dir),
        "workspace": str(round_dir / "run"),
        "subprocess_runtime_sec": 300.0,
        **extra,
    }


def _attempt(tmp_path: Path, *, kind: str = "baseline", params: dict | None = None, run_eval_disabled: bool = False):
    return BenchmarkAttempt(
        timeout_sec=600,
        resolved_model="model",
        materialized_config_path=tmp_path / "materialized.yaml",
        effective_extra_server_args="",
        params=params or {},
        ctx=SimpleNamespace(task=SimpleNamespace(kind=kind), extra={}),
        run_eval_disabled=run_eval_disabled,
    )


async def _rounds(pair: _BenchmarkRoundsMixin, attempt: BenchmarkAttempt, slot: Path):
    return await pair._run_cold_start_guarded_rounds(attempt, slot, [])


def _play(pair: _BenchmarkRoundsMixin, attempt: BenchmarkAttempt, slot: Path):
    return asyncio.run(_rounds(pair, attempt, slot))


def _score(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in EVAL_SCORE_KEYS if key in result}


_EXITS = {
    # exit: (task kind, params, the measured round's result, hot pass affordable, returned round, its marker, scored)
    "measured": ("baseline", {}, {}, True, "measure_round", MEASURED_FIRST, True),
    "measured_exempt": ("baseline", {"quality_ref_exempt": True}, {}, True, "measure_round", MEASURED_FIRST, True),
    "measured_replay": ("replay_warm_recipe", {}, {}, True, "measure_round", MEASURED_FIRST, True),
    "downward_drift_anchor": ("baseline", {}, {"output_throughput": 300.0}, True, "warmup_round", DRIFT_ANCHOR, True),
    "cold_anchor_unaffordable": ("baseline", {}, {}, False, "warmup_round", MEASURE_ROUND_DROPPED_WARNING, False),
    "cold_anchor_reaped": (
        "baseline",
        {},
        {"status": "failed", "error_class": SESSION_TIME_EXHAUSTED_CLASS, "error": "reaped"},
        True,
        "warmup_round",
        MEASURE_ROUND_DROPPED_WARNING,
        False,
    ),
}


@pytest.mark.parametrize("exit_", list(_EXITS))
@pytest.mark.parametrize("evaluated", [MEASURED_SCORE, {}], ids=["measure_scored", "nothing_scored"])
def test_every_succeeded_exit_carries_the_measured_rounds_score_and_a_cold_anchor_none(tmp_path, exit_, evaluated):
    kind, params, measured, affordable, returned_round, marker, scored = _EXITS[exit_]
    slot = tmp_path / "slot"
    pair = _StubbedPair(
        [_round(slot / "warmup_round", 544.0), _round(slot / "measure_round", 540.0, **{**evaluated, **measured})],
        hot_pass_affordable=affordable,
    )

    result = _play(pair, _attempt(tmp_path, kind=kind, params=params), slot)

    assert (result["status"], result["output_dir"]) == ("succeeded", str(slot / returned_round))
    assert marker in result["nonfatal_warnings"]
    # A cold anchor stands on the warm-up, which never evaluated: a measured round that did not run to its end
    # hands it nothing.
    assert _score(result) == (evaluated if scored else {})
    assert bool(result.get("measure_round_dropped")) is not scored
    # The score came from the round's result: the rounds left nothing on disk to find one in.
    assert not slot.exists()


def test_the_warmup_never_evaluates_and_a_replay_forces_its_measured_round_to(tmp_path):
    slot = tmp_path / "slot"
    asked = {}
    for kind in ("replay_warm_recipe", "baseline"):
        pair = _StubbedPair(
            [_round(slot / "warmup_round", 544.0), _round(slot / "measure_round", 540.0, **MEASURED_SCORE)]
        )
        _play(pair, _attempt(tmp_path, kind=kind), slot)
        asked[kind] = pair.asked

    assert asked["replay_warm_recipe"] == [("warmup", False), ("measure", True)]
    assert asked["baseline"] == [("warmup", False), ("measure", None)]


def test_a_measured_round_that_failed_returns_its_failure_without_a_score(tmp_path):
    slot = tmp_path / "slot"
    failed = {"status": "failed", "error_class": "server_died", "error": "engine died"}
    measured = _round(slot / "measure_round", 0.0, **failed)
    pair = _StubbedPair([_round(slot / "warmup_round", 544.0), measured])

    result = _play(pair, _attempt(tmp_path), slot)

    assert result is measured
    assert (result["status"], result["error_class"]) == ("failed", "server_died")
    assert _score(result) == {}


def test_a_genuine_pair_is_ruled_on_the_score_its_measured_round_handed_over(tmp_path):
    """No eval output exists anywhere and the session holds an older reference: the result's score alone rules."""
    slot = tmp_path / "slot"
    pair = _StubbedPair([_round(slot / "warmup_round", 544.0), _round(slot / "measure_round", 540.0, **MEASURED_SCORE)])
    state = SharedState(enablement_mode="off", baseline_accuracy=0.81)
    task = SimpleNamespace(task_id="t-pair", kind="baseline", params={"framework": "vllm"})
    executor = BaselineExecutor()

    async def _played(_ctx, *, recorder=None):
        return await _rounds(pair, _attempt(tmp_path, params=task.params), slot)

    executor._run_pass = _played
    result = asyncio.run(executor._run_with_accuracy(SimpleNamespace(task=task, extra={"shared_state": state})))

    assert _score(result) == MEASURED_SCORE
    assert result["nonfatal_warnings"] == [MEASURED_FIRST]
    assert state.stop_reason == ""
    # Promoting the baseline moves the session's reference; measuring it does not.
    assert state.baseline_accuracy == 0.81


_FAILED_STAGE = {"status": "failed", "error_class": "eval_failed", "error": "the eval exited with code 1"}


def test_a_drift_anchor_takes_the_failed_eval_of_the_measured_round_it_replaces(tmp_path):
    """The anchor is the warm-up's figure, which never evaluated: what the attempt's eval left is the measured round's."""
    slot = tmp_path / "slot"
    measured = _round(
        slot / "measure_round",
        300.0,
        run_eval_disabled=False,
        nonfatal_warnings=[EVAL_FAILED],
        **{ACCURACY_STAGE_KEY: _FAILED_STAGE},
    )
    pair = _StubbedPair([_round(slot / "warmup_round", 544.0, run_eval_disabled=True), measured])

    result = _play(pair, _attempt(tmp_path), slot)

    assert result["output_dir"] == str(slot / "warmup_round")
    assert result[ACCURACY_STAGE_KEY] == _FAILED_STAGE
    assert result["run_eval_disabled"] is False
    assert result["nonfatal_warnings"] == [DRIFT_ANCHOR, EVAL_FAILED]
    assert _score(result) == {}


class _RecordedRound(_BenchmarkRoundsMixin):
    """A guarded round that records what it was run with instead of launching."""

    async def _run_reported_round(self, attempt, **round_):
        self.ran_with = attempt
        self.ran_as = round_["held"]
        return {"status": "succeeded"}


@pytest.mark.parametrize(
    ("attempt_disabled", "purpose", "must_evaluate", "round_disabled"),
    [
        # A measured round the test forces to evaluate is read as evaluating although the attempt's config has the
        # eval off.
        (True, RoundPurpose.MEASURE, True, False),
        (False, RoundPurpose.MEASURE, True, False),
        # The warm-up never evaluates, and is read as a round that did not although the attempt's config has the
        # eval on.
        (True, RoundPurpose.WARMUP, False, True),
        (False, RoundPurpose.WARMUP, False, True),
        # A measured round left to the config keeps the attempt's own flag.
        (True, RoundPurpose.MEASURE, False, True),
        (False, RoundPurpose.MEASURE, False, False),
    ],
)
def test_a_round_is_read_as_evaluating_when_its_test_says_so(
    tmp_path, attempt_disabled, purpose, must_evaluate, round_disabled
):
    guarded = _RecordedRound()
    attempt = _attempt(tmp_path, run_eval_disabled=attempt_disabled)
    held = HeldPass(HeldTest(tmp_path, must_evaluate=must_evaluate), purpose)

    asyncio.run(guarded._run_guarded_round(attempt, purpose.value, tmp_path / "round", held))

    assert guarded.ran_as is held
    assert guarded.ran_with.run_eval_disabled is round_disabled


class _EvaluatingRounds(_BenchmarkRoundsMixin):
    """A pair played down to the benchmark, each round's result stamped from its own run directory as a real one is.

    A round its test tells to evaluate leaves a results file there.
    """

    shared_state = None

    def __init__(self, score: float):
        self.score = score
        self.evaluates: dict[Path, bool | None] = {}

    async def _run_single_benchmark(self, attempt, *, output_dir, held):
        self.evaluates[output_dir] = held.evaluates
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        if self.evaluates[output_dir]:
            results = {"results": {"gsm8k": {"exact_match,strict-match": self.score}}}
            (run_dir / "results_2026-08-19T00-00-00.json").write_text(json.dumps(results), encoding="utf-8")
        result = _round(output_dir, 544.0, e2el_mean_ms=12.0)
        round_ = _BenchmarkRound(
            config_path=attempt.materialized_config_path,
            output_dir=output_dir,
            ctx_extra={},
            framework="vllm",
            watchdog_server_log=None,
            capture_meta={},
            timeout_sec=attempt.timeout_sec,
            session_deadline_sec=None,
            held=held,
            run_eval_disabled=attempt.run_eval_disabled,
        )
        self._stamp_round_quality(round_, result, {"framework": "vllm"}, [])
        return result

    def _measure_round_affordable(self, warmup_result, *, ctx_extra=None):
        return True, {}


def test_a_replay_whose_recipe_turns_the_eval_off_still_returns_its_forced_measured_rounds_score(tmp_path):
    slot = tmp_path / "slot"
    pair = _EvaluatingRounds(score=0.89)
    attempt = _attempt(tmp_path, kind="replay_warm_recipe", run_eval_disabled=True)

    result = _play(pair, attempt, slot)

    assert pair.evaluates == {slot / "warmup_round": False, slot / "measure_round": True}
    assert Path(result["output_dir"]).name == "measure_round"
    assert result["accuracy"] == pytest.approx(0.89)
    assert result["accuracy_source"].startswith(str(slot / "measure_round"))
    assert not list((slot / "warmup_round").rglob("results*.json"))
