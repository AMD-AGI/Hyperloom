# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the eval process's request and the lm-eval arguments it runs with."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

import inference_suite.eval.run as eval_run
from inference_suite.eval.run import EVAL_REQUEST_SCHEMA, MODEL_NAME, EvalRequest, EvalRequestError, lm_eval_argv

_REQUEST = EvalRequest(
    task_yaml="/tools/infx/evals/gsm8k.yaml",
    base_url="http://127.0.0.1:30000/v1/chat/completions",
    served_model_name="glm",
    output_dir="/s/runs/r1/eval",
    probe_file="/s/runs/r1/eval/hyperloom_eval_probe.json",
    concurrency=32,
    max_length=65536,
    max_tokens=4096,
    num_fewshot=5,
    probe_enabled=True,
    probe_min_samples=48,
    probe_length_ratio=0.6,
    limit=200,
    model_dir="/models/GLM-5.3",
)
_REQUEST_FILE = "/s/runs/r1/eval/request.json"


def test_round_trip_through_json():
    document = json.loads(json.dumps(_REQUEST.to_document()))

    assert document["schema"] == EVAL_REQUEST_SCHEMA
    assert EvalRequest.from_document(document) == _REQUEST


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema": "other/1"}, "schema"),
        ({"surprise": 1}, "unknown eval request keys: surprise"),
        ({"base_url": None, "drop": "base_url"}, "incomplete"),
    ],
)
def test_a_malformed_request_is_refused(change, message):
    document = {**_REQUEST.to_document(), **change}
    if "drop" in change:
        document.pop(document.pop("drop"))

    with pytest.raises(EvalRequestError, match=message):
        EvalRequest.from_document(document)


def _option(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _model_args(argv: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in _option(argv, "--model_args").split(","))


def test_lm_eval_runs_greedy_chat_completions_with_the_request_budget():
    argv = lm_eval_argv(_REQUEST, _REQUEST_FILE)

    assert argv[0] == "run"
    assert _option(argv, "--model") == MODEL_NAME
    assert "--apply_chat_template" in argv and "--log_samples" in argv
    assert _option(argv, "--tasks") == _REQUEST.task_yaml
    assert _option(argv, "--output_path") == _REQUEST.output_dir
    assert (_option(argv, "--num_fewshot"), _option(argv, "--limit")) == ("5", "200")
    assert _option(argv, "--gen_kwargs") == "max_tokens=4096,temperature=0,top_p=1"


def test_the_model_args_carry_the_served_context_and_the_request_file():
    args = _model_args(lm_eval_argv(_REQUEST, _REQUEST_FILE))

    assert args["max_length"] == "65536"
    assert args["num_concurrent"] == "32"
    assert args["tokenized_requests"] == "False"
    assert args["request"] == _REQUEST_FILE
    assert not {"probe_enabled", "probe_file", "model_dir"} & set(args)


def test_the_request_file_reads_back_as_the_request(tmp_path: Path):
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(_REQUEST.to_document()))

    assert eval_run.read_request(str(request_file)) == _REQUEST


def test_a_request_without_a_limit_leaves_it_out():
    argv = lm_eval_argv(replace(_REQUEST, limit=None), _REQUEST_FILE)

    assert "--limit" not in argv


def test_a_value_lm_eval_would_split_is_refused():
    with pytest.raises(EvalRequestError, match="request"):
        lm_eval_argv(_REQUEST, "/s/runs/a,b/request.json")


def test_main_takes_exactly_one_request_file():
    with pytest.raises(SystemExit, match="usage"):
        eval_run.main([])


def test_the_process_module_imports_only_the_standard_library_at_import():
    tree = ast.parse(Path(eval_run.__file__).read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = {alias.name for node in top_level if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in top_level if isinstance(node, ast.ImportFrom)}

    assert all(
        module.split(".")[0] in {"__future__", "json", "sys", "collections", "dataclasses", "pathlib"}
        for module in modules
    )
