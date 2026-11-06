# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the eval definition: what its digest covers, where the served context comes from, and the refusal."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from inference_suite.compose import RoundFacts, compose
from inference_suite.contracts import EvalDefinition, EvalProbe, FailedStep, LaunchDelta, RunLayout, SuiteConfig
from inference_suite.contracts.launch import Launch, VersionCheck
from inference_suite.eval.definition import (
    PROMPT_ROOM_TOKENS,
    EvalPlan,
    EvalRefusal,
    StagedEval,
    definition_digest,
    plan_eval,
)
from inference_suite.eval.run import EVAL_MODULE, EvalRequest
from inference_suite.frameworks import Framework, framework_facts
from inference_suite.tests.test_base_launch_contract import _base
from inference_suite.tests.test_config_contract import _accuracy_round, _config, _eval

_FACTS = RoundFacts(config_sha256="d" * 64, installed_version="0.5.23", version_check=VersionCheck.PASSED)


def _round(config: SuiteConfig, **server_changes) -> tuple[SuiteConfig, Launch]:
    base = _base()
    server = replace(config.server, base=replace(config.server.base, digest=base.digest()), **server_changes)
    config = replace(config, server=server)
    launch = compose(config, base, _FACTS)
    assert isinstance(launch, Launch), launch
    return config, launch


@pytest.fixture
def staged(tmp_path: Path) -> StagedEval:
    task = tmp_path / "gsm8k.yaml"
    task.write_text("task: gsm8k\n", encoding="utf-8")
    return StagedEval(task_yaml=task, lm_eval_version="0.4.9.2")


def _plan(config: SuiteConfig, staged: StagedEval, tmp_path: Path, **server_changes) -> EvalPlan | EvalRefusal:
    config, launch = _round(config, **server_changes)
    return plan_eval(config, launch, staged, RunLayout(tmp_path / "run"))


def _accepted(config: SuiteConfig, staged: StagedEval, tmp_path: Path, **server_changes) -> EvalPlan:
    plan = _plan(config, staged, tmp_path, **server_changes)
    assert isinstance(plan, EvalPlan), plan
    return plan


def test_the_probe_thresholds_come_from_the_config_into_the_definition_and_the_request(staged, tmp_path):
    probe = EvalProbe(enabled=True, min_samples=48, length_ratio=0.6)
    plan = _accepted(replace(_accuracy_round(), eval=_eval(probe=probe)), staged, tmp_path)

    assert plan.definition.probe == probe
    request = plan.request
    assert (request.probe_enabled, request.probe_min_samples, request.probe_length_ratio) == (True, 48, 0.6)
    other = _accepted(replace(_accuracy_round(), eval=_eval(probe=replace(probe, length_ratio=0.7))), staged, tmp_path)
    assert other.digest != plan.digest


def test_the_definition_records_what_the_score_depends_on(staged, tmp_path):
    plan = _accepted(_accuracy_round(), staged, tmp_path)
    definition = plan.definition

    assert definition.max_length == 65536
    assert (definition.concurrency, definition.limit, definition.num_fewshot) == (32, 200, 5)
    assert definition.max_tokens_cap == plan.request.max_tokens == 4096
    assert definition.lm_eval_version == "0.4.9.2"
    assert definition.golden_applied is False
    assert definition.chat_template_sha256 is None
    assert plan.digest == definition_digest(EvalDefinition.from_document(definition.to_document()))
    assert plan.digest.startswith("sha256:")


def test_a_changed_task_file_is_another_definition(staged, tmp_path):
    before = _accepted(_accuracy_round(), staged, tmp_path).digest
    staged.task_yaml.write_text("task: gsm8k\nnum_fewshot: 8\n", encoding="utf-8")

    assert _accepted(_accuracy_round(), staged, tmp_path).digest != before


def test_a_round_that_applies_golden_says_so(staged, tmp_path):
    assert _accepted(_config(eval=_eval()), staged, tmp_path).definition.golden_applied is True


def test_the_last_context_flag_of_the_final_argv_is_the_served_context(staged, tmp_path):
    candidate = LaunchDelta(args_add=("--context-length", "16384"))

    assert _accepted(_accuracy_round(), staged, tmp_path, candidate=candidate).definition.max_length == 16384


@pytest.mark.parametrize("served", [PROMPT_ROOM_TOKENS, PROMPT_ROOM_TOKENS + 4095])
def test_a_served_context_without_room_for_prompt_and_completion_is_refused(staged, tmp_path, served):
    candidate = LaunchDelta(args_add=("--context-length", str(served)))

    refusal = _plan(_accuracy_round(), staged, tmp_path, candidate=candidate)

    assert isinstance(refusal, EvalRefusal)
    assert refusal.step is FailedStep.EVAL_CONTEXT_TOO_SHORT
    assert f"a context of {served} tokens" in refusal.detail
    assert "max_tokens_cap 4096" in refusal.detail


def test_a_smaller_completion_cap_fits_a_smaller_context(staged, tmp_path):
    candidate = LaunchDelta(args_add=("--context-length", "6144"))

    plan = _accepted(replace(_accuracy_round(), eval=_eval(max_tokens_cap=2048)), staged, tmp_path, candidate=candidate)

    assert (plan.definition.max_length, plan.request.max_tokens) == (6144, 2048)


def _model_dir(tmp_path: Path, config: dict[str, object]) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model


def test_without_a_context_flag_the_model_native_context_is_served(staged, tmp_path):
    model = _model_dir(tmp_path, {"text_config": {"max_position_embeddings": 131072}})
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")

    plan = _accepted(_accuracy_round(), staged, tmp_path, model_path=str(model), user=LaunchDelta())

    assert plan.definition.max_length == 131072
    assert plan.definition.chat_template_sha256 is not None
    assert plan.request.model_dir == str(model)


def test_an_unknown_served_context_is_refused(staged, tmp_path):
    model = _model_dir(tmp_path, {"architectures": ["X"]})

    refusal = _plan(_accuracy_round(), staged, tmp_path, model_path=str(model), user=LaunchDelta())

    assert isinstance(refusal, EvalRefusal)
    assert refusal.step is FailedStep.EVAL_CONTEXT_TOO_SHORT
    assert "an unknown context" in refusal.detail


def test_the_plan_writes_its_definition_and_request_and_names_its_process(staged, tmp_path):
    plan = _accepted(_accuracy_round(), staged, tmp_path)
    layout = RunLayout(tmp_path / "run")
    request_file = tmp_path / "eval_request.json"

    plan.write(layout, request_file)
    invocation = plan.invocation("/staged/lm-eval/bin/python", request_file)

    written = EvalDefinition.from_document(json.loads(layout.eval_definition.read_text(encoding="utf-8")))
    assert definition_digest(written) == plan.digest == invocation.definition_digest
    assert EvalRequest.from_document(json.loads(request_file.read_text(encoding="utf-8"))) == plan.request
    assert invocation.argv == ("/staged/lm-eval/bin/python", "-I", "-m", EVAL_MODULE, str(request_file))
    assert (plan.request.output_dir, plan.request.probe_file) == (str(layout.eval_dir), str(layout.eval_probe))


@pytest.mark.parametrize(
    "framework, flag",
    [(Framework.SGLANG, "--context-length"), (Framework.VLLM, "--max-model-len"), (Framework.ATOM, "--max-model-len")],
)
def test_each_serving_framework_names_its_context_flag(framework, flag):
    assert framework_facts(framework).server_flags.context_length == flag
