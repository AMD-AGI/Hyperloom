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
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import _BaselineAccuracyMixin
from hyperloom.orchestrator.actions.executors.benchmark_run.records import record_eval_score
from hyperloom.orchestrator.loop._baseline_promotion import BaselinePromotionMixin
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
