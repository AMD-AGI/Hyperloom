# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A score carries the eval definition digest it was taken under, and the baseline keeps it beside its score."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import ACCURACY_ROUND_KEY, _BaselineAccuracyMixin
from hyperloom.orchestrator.actions.executors.benchmark_run.records import record_eval_score
from hyperloom.orchestrator.loop._baseline_promotion import BaselinePromotionMixin, _adopt_eval_trigger
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite.contracts import RunLayout
from inference_suite.contracts.base_launch import DIGEST_PREFIX
from inference_suite.contracts.launch import EvalInvocation
from inference_suite.tests.test_compose import _launch

BASELINE_DIGEST = DIGEST_PREFIX + "a" * 64
OTHER_DIGEST = DIGEST_PREFIX + "b" * 64


def suite_eval(run_dir: Path, score: float, digest: str) -> Path:
    """Leave the launch and lm-eval result an inference_suite round with an eval under ``digest`` writes."""
    layout = RunLayout(run_dir)
    results = layout.eval_dir / "model"
    results.mkdir(parents=True)
    (results / "results_2026.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}), encoding="utf-8"
    )
    launch = dataclasses.replace(_launch(), eval=EvalInvocation(("python",), digest))
    layout.launch.write_text(json.dumps(launch.to_document()), encoding="utf-8")
    return run_dir


@pytest.fixture(autouse=True)
def _native_backend(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)


def test_an_agentx_score_carries_the_digest_of_its_accuracy_round(tmp_path):
    suite_eval(tmp_path / ag.ACCURACY_DIR, 0.945, BASELINE_DIGEST)
    out = ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")
    assert (out["accuracy"], out[ag.EVAL_DEFINITION_DIGEST_KEY]) == (0.945, BASELINE_DIGEST)


def test_a_score_no_inference_suite_round_took_carries_no_digest(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}), encoding="utf-8"
    )
    out = ag.parse_eval_results(tmp_path, framework="vllm")
    assert (out["accuracy"], out[ag.EVAL_DEFINITION_DIGEST_KEY]) == (0.9, "")


def test_a_round_records_the_digest_beside_its_score(tmp_path):
    suite_eval(tmp_path / "run", 0.9, BASELINE_DIGEST)
    result: dict = {}
    assert record_eval_score(result, ag.parse_eval_results(tmp_path, framework="vllm"))
    assert (result["accuracy"], result[ag.ACCURACY_DEFINITION_DIGEST_KEY]) == (0.9, BASELINE_DIGEST)


def test_the_baseline_promotes_its_digest_with_its_score():
    state = SharedState()
    promoter = SimpleNamespace(shared_state=state)
    result = {"accuracy": 0.945, ag.ACCURACY_DEFINITION_DIGEST_KEY: BASELINE_DIGEST}
    BaselinePromotionMixin._promote_anchor_run_facts(promoter, result, {})
    assert (state.baseline_accuracy, state.baseline_accuracy_definition_digest) == (0.945, BASELINE_DIGEST)


def test_a_salvaged_sibling_score_publishes_its_digest_with_it(tmp_path):
    state = SharedState()
    salvaged = ag.parse_eval_results(suite_eval(tmp_path / "sibling", 0.93, BASELINE_DIGEST), framework="vllm")
    result: dict = {}
    _BaselineAccuracyMixin._apply_salvaged_accuracy(SimpleNamespace(), result, salvaged, state)
    assert result[ag.ACCURACY_DEFINITION_DIGEST_KEY] == BASELINE_DIGEST
    assert (state.baseline_accuracy, state.baseline_accuracy_definition_digest) == (0.93, BASELINE_DIGEST)


def test_the_digest_survives_a_resume():
    state = SharedState(baseline_accuracy=0.945, baseline_accuracy_definition_digest=BASELINE_DIGEST)
    assert SharedState.from_dict(state.to_dict()).baseline_accuracy_definition_digest == BASELINE_DIGEST


@pytest.mark.parametrize(
    ("digest", "expected"),
    [(BASELINE_DIGEST, False), (OTHER_DIGEST, None)],
    ids=["same_definition", "other_definition"],
)
def test_a_candidate_is_graded_only_under_the_baselines_definition(tmp_path, digest, expected):
    """A drop to 0.0 under the baseline's definition is a regression; under another it is no score at all."""
    suite_eval(tmp_path / ag.ACCURACY_DIR, 0.0, digest)
    reference = ag.AccuracyReference(0.945, BASELINE_DIGEST)
    assert ag.grade_accuracy(str(tmp_path), reference, framework="vllm", benchmark_mode="agentx") is expected


def test_a_reference_no_inference_suite_round_took_does_not_compare_with_one_that_did(tmp_path):
    suite_eval(tmp_path / ag.ACCURACY_DIR, 0.94, BASELINE_DIGEST)
    results = ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")
    assert ag.AccuracyReference(0.945, "").comparable(results) is None
    assert ag.AccuracyReference(0.945, BASELINE_DIGEST).comparable(results) == 0.94


def test_the_baseline_reference_is_the_states_score_and_digest():
    state = SharedState(baseline_accuracy=0.945, baseline_accuracy_definition_digest=BASELINE_DIGEST)
    assert ag.AccuracyReference.of_baseline(state) == ag.AccuracyReference(0.945, BASELINE_DIGEST)


@pytest.mark.parametrize(
    ("digest", "outcome", "reason"),
    [(BASELINE_DIGEST, "KEEP", ""), (OTHER_DIGEST, "REVERT", "accuracy_unavailable")],
    ids=["same_definition", "other_definition"],
)
def test_an_explore_slot_under_another_definition_is_reverted(tmp_path, digest, outcome, reason):
    from hyperloom.orchestrator.actions.executors.explore import _ExploreRound

    slot = tmp_path / "variant_00"
    suite_eval(slot / ag.ACCURACY_DIR, 0.94, digest)
    explore = object.__new__(_ExploreRound)
    explore.framework = "sglang"
    explore.ss = SharedState(
        benchmark_mode="agentx", baseline_accuracy=0.945, baseline_accuracy_definition_digest=BASELINE_DIGEST
    )
    explore.accuracy_reference = ag.AccuracyReference.of_baseline(explore.ss)
    row: dict = {"gates": [], "reason": ""}
    explore._apply_accuracy_gate(row, slot)
    assert (row["outcome"], row["gates"][0]["reason"]) == (outcome, reason)


@pytest.mark.parametrize(
    ("digest", "expected"), [(BASELINE_DIGEST, False), (OTHER_DIGEST, None)], ids=["same_definition", "other"]
)
def test_an_integrate_bench_is_graded_only_under_the_baselines_definition(tmp_path, digest, expected):
    from hyperloom.orchestrator.actions.executors._integrate_bench import _grade_bench_accuracy
    from hyperloom.orchestrator.actions.executors._integrate_results import _PatchBench

    slot = tmp_path / "variant_00"
    suite_eval(slot / ag.ACCURACY_DIR, 0.0, digest)
    bench_leg = _PatchBench(
        params={"accuracy_baseline": 0.945, "framework": "vllm"},
        output_root=tmp_path,
        specialist_task_id="t",
        benchmark_mode="agentx",
        accuracy_definition_digest=BASELINE_DIGEST,
    )
    gate_evidence: dict = {}
    _grade_bench_accuracy({"status": "succeeded"}, bench_leg, str(slot), gate_evidence)
    assert gate_evidence["accuracy_pass"] is expected


@pytest.mark.parametrize(
    ("digest", "accuracy_pass"), [(BASELINE_DIGEST, False), (OTHER_DIGEST, None)], ids=["same_definition", "other"]
)
def test_a_kernel_rebaseline_is_graded_only_under_the_baselines_definition(tmp_path, digest, accuracy_pass):
    from hyperloom.orchestrator.kernel.integrate_accuracy import grade_integrate_accuracy

    bench = {"accuracy": 0.0, ag.ACCURACY_DEFINITION_DIGEST_KEY: digest}
    reference = ag.AccuracyReference(0.945, BASELINE_DIGEST)
    gate = grade_integrate_accuracy(bench, reference=reference, workspace=tmp_path, strict=True)
    assert (gate["accuracy_pass"], gate["blocked"]) == (accuracy_pass, True)


def test_a_kernel_rebaseline_reads_the_digest_of_the_score_its_workspace_holds(tmp_path):
    from hyperloom.orchestrator.kernel.integrate_accuracy import grade_integrate_accuracy

    suite_eval(tmp_path / ag.ACCURACY_DIR, 0.94, BASELINE_DIGEST)
    reference = ag.AccuracyReference(0.945, BASELINE_DIGEST)
    gate = grade_integrate_accuracy({}, reference=reference, workspace=tmp_path)
    assert (gate["accuracy_pass"], gate[ag.EVAL_DEFINITION_DIGEST_KEY]) == (True, BASELINE_DIGEST)


def _stamp_failure(result: dict) -> dict:
    return _BaselineAccuracyMixin._stamp_eval_failure_contract(
        result, kind=ag.EVAL_KIND_ACCURACY_UNAVAILABLE, observed_accuracy=None, evidence=""
    )


def test_an_unscored_eval_failure_records_the_definition_its_accuracy_round_planned(tmp_path):
    run_dir = suite_eval(tmp_path / ag.ACCURACY_DIR, 0.0, BASELINE_DIGEST)
    result = _stamp_failure({ACCURACY_ROUND_KEY: {"run_dir": str(run_dir), "error_class": "eval_failed"}})
    assert result[ag.ACCURACY_DEFINITION_DIGEST_KEY] == BASELINE_DIGEST


def test_a_scored_eval_failure_keeps_the_digest_of_its_score(tmp_path):
    run_dir = suite_eval(tmp_path / ag.ACCURACY_DIR, 0.0, OTHER_DIGEST)
    result = _stamp_failure(
        {
            ACCURACY_ROUND_KEY: {"run_dir": str(run_dir), "error_class": ""},
            ag.ACCURACY_DEFINITION_DIGEST_KEY: BASELINE_DIGEST,
        }
    )
    assert result[ag.ACCURACY_DEFINITION_DIGEST_KEY] == BASELINE_DIGEST


def test_an_eval_failure_outside_an_accuracy_round_records_no_digest():
    assert ag.ACCURACY_DEFINITION_DIGEST_KEY not in _stamp_failure({})


def test_the_enablement_trigger_keeps_the_digest_its_eval_failure_ran_under():
    state = SharedState()
    _adopt_eval_trigger(state, {ag.ACCURACY_DEFINITION_DIGEST_KEY: BASELINE_DIGEST})
    assert state.enablement.eval_definition_digest == BASELINE_DIGEST
    assert SharedState.from_dict(state.to_dict()).enablement.eval_definition_digest == BASELINE_DIGEST
