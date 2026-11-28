# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy score a cold-start pair's result carries: that of the round of its attempt that ran the eval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.benchmark_run.records import (
    EVAL_SCORE_KEYS,
    MEASURE_ROUND_DROPPED_WARNING,
    BenchmarkAttempt,
    _BenchmarkRound,
)
from hyperloom.orchestrator.actions.executors.benchmark_run.rounds import _BenchmarkRoundsMixin
from hyperloom.orchestrator.actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS
from hyperloom.orchestrator.state.shared_state import SharedState

WARMUP_SCORE = {
    "accuracy": 0.96,
    "accuracy_task": "gsm8k",
    "accuracy_metric": "exact_match,strict-match",
    "accuracy_source": "warmup_round/eval/results.json",
    "accuracy_definition_digest": "sha256:warmup",
    "accuracy_eval_kind": "",
}
ACCURACY_ROUND_SCORE = {
    **WARMUP_SCORE,
    "accuracy": 0.91,
    "accuracy_source": "accuracy_round/eval/results.json",
    "accuracy_definition_digest": "sha256:accuracy",
}
MEASURED_FIRST = "baseline_double_run_discarded_first"
DRIFT_ANCHOR = "baseline_downward_drift_anchor"


class _StubbedPair(_BenchmarkRoundsMixin):
    """A pair whose rounds return prepared results, in the order the sequencer asks for them."""

    shared_state = None

    def __init__(self, results: list[dict[str, Any]], *, hot_pass_affordable: bool = True):
        self.results = results
        self.hot_pass_affordable = hot_pass_affordable
        self.asked: list[tuple[str, bool | None]] = []

    async def _run_guarded_round(self, attempt, label, round_dir, **config):
        self.asked.append((label, config["run_eval"]))
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


async def _rounds(pair: _BenchmarkRoundsMixin, attempt: BenchmarkAttempt, slot: Path, *, defer_accuracy: bool = False):
    return await pair._run_cold_start_guarded_rounds(attempt, slot, {"port": 8888}, [], defer_accuracy=defer_accuracy)


def _play(pair: _BenchmarkRoundsMixin, attempt: BenchmarkAttempt, slot: Path, *, defer_accuracy: bool = False):
    return asyncio.run(_rounds(pair, attempt, slot, defer_accuracy=defer_accuracy))


def _score(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in EVAL_SCORE_KEYS if key in result}


_EXITS = {
    # exit: (task kind, params, the measured round's result, hot pass affordable, returned round, its marker)
    "measured": ("baseline", {}, {}, True, "measure_round", MEASURED_FIRST),
    "measured_exempt": ("baseline", {"quality_ref_exempt": True}, {}, True, "measure_round", MEASURED_FIRST),
    "measured_replay": ("replay_warm_recipe", {}, {}, True, "measure_round", MEASURED_FIRST),
    "downward_drift_anchor": ("baseline", {}, {"output_throughput": 300.0}, True, "warmup_round", DRIFT_ANCHOR),
    "cold_anchor_unaffordable": ("baseline", {}, {}, False, "warmup_round", MEASURE_ROUND_DROPPED_WARNING),
    "cold_anchor_reaped": (
        "baseline",
        {},
        {"status": "failed", "error_class": SESSION_TIME_EXHAUSTED_CLASS, "error": "reaped"},
        True,
        "warmup_round",
        MEASURE_ROUND_DROPPED_WARNING,
    ),
}


@pytest.mark.parametrize("exit_", list(_EXITS))
@pytest.mark.parametrize("evaluated", [WARMUP_SCORE, {}], ids=["warmup_scored", "nothing_scored"])
def test_every_succeeded_exit_carries_the_score_of_the_round_that_evaluated(tmp_path, exit_, evaluated):
    kind, params, measured, affordable, returned_round, marker = _EXITS[exit_]
    slot = tmp_path / "slot"
    pair = _StubbedPair(
        [_round(slot / "warmup_round", 544.0, **evaluated), _round(slot / "measure_round", 540.0, **measured)],
        hot_pass_affordable=affordable,
    )

    result = _play(pair, _attempt(tmp_path, kind=kind, params=params), slot)

    assert (result["status"], result["output_dir"]) == ("succeeded", str(slot / returned_round))
    assert marker in result["nonfatal_warnings"]
    assert _score(result) == evaluated
    # The score came from the round's result: the rounds left nothing on disk to find one in.
    assert not slot.exists()


def test_a_replay_forces_its_warmup_to_evaluate_and_a_stock_baseline_leaves_it_to_the_config(tmp_path):
    slot = tmp_path / "slot"
    asked = {}
    for kind in ("replay_warm_recipe", "baseline"):
        pair = _StubbedPair(
            [_round(slot / "warmup_round", 544.0, **WARMUP_SCORE), _round(slot / "measure_round", 540.0)]
        )
        _play(pair, _attempt(tmp_path, kind=kind), slot)
        asked[kind] = pair.asked

    assert asked["replay_warm_recipe"] == [("warmup", True), ("measure", False)]
    assert asked["baseline"] == [("warmup", None), ("measure", False)]


def test_a_measured_round_that_failed_returns_its_failure_without_a_score(tmp_path):
    slot = tmp_path / "slot"
    failed = {"status": "failed", "error_class": "server_died", "error": "engine died"}
    measured = _round(slot / "measure_round", 0.0, **failed)
    pair = _StubbedPair([_round(slot / "warmup_round", 544.0, **WARMUP_SCORE), measured])

    result = _play(pair, _attempt(tmp_path), slot)

    assert result is measured
    assert (result["status"], result["error_class"]) == ("failed", "server_died")
    assert _score(result) == {}


def test_a_genuine_pair_is_ruled_on_the_score_its_own_rounds_handed_over(tmp_path):
    """No eval output exists anywhere and the session holds an older reference: the result's score alone rules."""
    slot = tmp_path / "slot"
    pair = _StubbedPair([_round(slot / "warmup_round", 544.0, **WARMUP_SCORE), _round(slot / "measure_round", 540.0)])
    state = SharedState(enablement_mode="off", baseline_accuracy=0.81)
    task = SimpleNamespace(task_id="t-pair", kind="baseline", params={"framework": "vllm"})
    executor = BaselineExecutor()

    async def _played(_ctx, *, recorder=None):
        return await _rounds(pair, _attempt(tmp_path, params=task.params), slot)

    executor._run_pass = _played
    result = asyncio.run(executor._run_with_accuracy(SimpleNamespace(task=task, extra={"shared_state": state})))

    assert _score(result) == WARMUP_SCORE
    assert result["nonfatal_warnings"] == [MEASURED_FIRST]
    assert state.stop_reason == ""
    # Promoting the baseline moves the session's reference; measuring it does not.
    assert state.baseline_accuracy == 0.81


def _deferred(tmp_path: Path, *, min_tput: float, accuracy_round: dict[str, Any]):
    """Play a pair whose eval waits for the accuracy round; the warm-up stub holds a score no deferred warm-up takes."""
    slot = tmp_path / "slot"
    pair = _StubbedPair(
        [
            _round(slot / "warmup_round", 544.0, **WARMUP_SCORE),
            _round(slot / "measure_round", 540.0),
            _round(slot / "accuracy_round", 0.0, **accuracy_round),
        ]
    )
    params = {"quality_ref_exempt": True, "post_measure_accuracy_min_tput": min_tput}
    attempt = _attempt(tmp_path, params=params, run_eval_disabled=True)
    return _play(pair, attempt, slot, defer_accuracy=True), pair


def test_a_deferred_eval_hands_over_the_accuracy_rounds_score_not_the_warmups(tmp_path):
    result, pair = _deferred(tmp_path, min_tput=100.0, accuracy_round=ACCURACY_ROUND_SCORE)

    assert pair.asked == [("warmup", None), ("measure", False), ("accuracy", True)]
    assert Path(result["output_dir"]).name == "measure_round"
    assert result["accuracy_stage"]["status"] == "succeeded"
    assert _score(result) == ACCURACY_ROUND_SCORE


def test_a_deferred_eval_that_is_skipped_leaves_no_score(tmp_path):
    result, pair = _deferred(tmp_path, min_tput=9999.0, accuracy_round=ACCURACY_ROUND_SCORE)

    assert [label for label, _run_eval in pair.asked] == ["warmup", "measure"]
    assert result["accuracy_stage"]["status"] == "skipped"
    assert _score(result) == {}


def test_a_deferred_accuracy_round_that_failed_leaves_no_score(tmp_path):
    failed = {**ACCURACY_ROUND_SCORE, "status": "failed", "error_class": "server_died"}
    result, _pair = _deferred(tmp_path, min_tput=100.0, accuracy_round=failed)

    assert result["accuracy_stage"]["error_class"] == "server_died"
    assert "post_measure_accuracy_failed" in result["nonfatal_warnings"]
    assert _score(result) == {}


class _RecordedRound(_BenchmarkRoundsMixin):
    """A guarded round that records what it was configured and run with instead of launching."""

    def _write_lifecycle_config(self, base_config_path, dest_dir, **lifecycle):
        self.configured_run_eval = lifecycle["run_eval"]
        return dest_dir / "baseline_lifecycle.yaml"

    async def _run_reported_round(self, attempt, **round_):
        self.ran_with = attempt
        return {"status": "succeeded"}


@pytest.mark.parametrize(
    ("attempt_disabled", "run_eval", "round_disabled"),
    [
        # A round the sequencer tells to evaluate is read as evaluating although the attempt's config has the eval off.
        (True, True, False),
        (False, True, False),
        # A round left to the config, or told not to evaluate, keeps the attempt's own flag.
        (True, None, True),
        (True, False, True),
        (False, None, False),
        (False, False, False),
    ],
)
def test_a_round_told_to_evaluate_runs_as_one_whose_eval_is_on(tmp_path, attempt_disabled, run_eval, round_disabled):
    guarded = _RecordedRound()
    attempt = _attempt(tmp_path, run_eval_disabled=attempt_disabled)

    asyncio.run(
        guarded._run_guarded_round(
            attempt, "warmup", tmp_path / "warmup_round", port=8888, cleanup=False, run_eval=run_eval
        )
    )

    assert guarded.configured_run_eval is run_eval
    assert guarded.ran_with.run_eval_disabled is round_disabled


class _EvaluatingRounds(_BenchmarkRoundsMixin):
    """A pair played down to the benchmark, each round's result stamped from its own run directory as a real one is.

    A round whose lifecycle config turns the eval on leaves a results file there.
    """

    shared_state = None

    def __init__(self, score: float):
        self.score = score
        self.evaluates: dict[Path, bool | None] = {}

    def _write_lifecycle_config(self, base_config_path, dest_dir, **lifecycle):
        self.evaluates[dest_dir] = lifecycle["run_eval"]
        return dest_dir / "baseline_lifecycle.yaml"

    async def _run_single_benchmark(self, attempt, *, config_path, output_dir, **_reuse):
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        if self.evaluates[output_dir]:
            results = {"results": {"gsm8k": {"exact_match,strict-match": self.score}}}
            (run_dir / "results_2026-08-19T00-00-00.json").write_text(json.dumps(results), encoding="utf-8")
        result = _round(output_dir, 544.0, e2el_mean_ms=12.0)
        round_ = _BenchmarkRound(
            config_path=config_path,
            materialized_config_path=attempt.materialized_config_path,
            output_dir=output_dir,
            ctx_extra={},
            framework="vllm",
            watchdog_server_log=None,
            capture_meta={},
            timeout_sec=attempt.timeout_sec,
            session_deadline_sec=None,
            server_already_ready=False,
            run_eval_disabled=attempt.run_eval_disabled,
        )
        self._stamp_round_quality(round_, result, {"framework": "vllm"}, [])
        return result

    def _measure_round_affordable(self, warmup_result, *, ctx_extra=None):
        return True, {}


def test_a_replay_whose_recipe_turns_the_eval_off_still_returns_its_forced_warmups_score(tmp_path):
    slot = tmp_path / "slot"
    pair = _EvaluatingRounds(score=0.89)
    attempt = _attempt(tmp_path, kind="replay_warm_recipe", run_eval_disabled=True)

    result = _play(pair, attempt, slot)

    assert pair.evaluates == {slot / "warmup_round": True, slot / "measure_round": False}
    assert Path(result["output_dir"]).name == "measure_round"
    assert result["accuracy"] == pytest.approx(0.89)
    assert result["accuracy_source"].startswith(str(slot / "warmup_round"))
    assert not list((slot / "measure_round").rglob("results*.json"))
