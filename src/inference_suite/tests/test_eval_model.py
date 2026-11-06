# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The eval model class: InferenceX's response parsing, the model's terminators, the truncation probe, #3293."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("lm_eval", reason="the eval model runs only where lm-eval is staged")

from lm_eval.api.registry import get_model, model_registry
from lm_eval.models.openai_completions import LocalChatCompletion

from inference_suite.eval.model import (
    MAX_STOP_STRINGS,
    SuiteChatCompletion,
    TruncationProbe,
    bounded_payload,
    generations,
    model_terminators,
)
from inference_suite.eval.run import MODEL_CLASS, MODEL_NAME


def _response(*finishes: str, content: str | None = "x") -> dict[str, object]:
    choices = [
        {"index": i, "finish_reason": finish, "message": {"content": content, "reasoning_content": "r"}}
        for i, finish in enumerate(finishes)
    ]
    return {"choices": choices}


def test_generations_fall_back_to_reasoning_content_and_keep_choice_order():
    response = {
        "choices": [
            {"index": 1, "message": {"content": "second"}},
            {"index": 0, "message": {"content": "", "reasoning_content": "first"}},
        ]
    }

    assert generations(response) == ["first", "second"]
    assert generations({"error": "x"}) == [""]
    assert generations({"choices": [{"index": 3, "message": {}}]}) == [""]


def test_terminators_come_from_every_eos_id_the_model_declares(tmp_path: Path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [151645, 151643, True]}))
    decoder = {"151645": {"content": "<|im_end|>"}, "151643": {"content": "<|endoftext|>"}}
    tokenizer = {"added_tokens_decoder": decoder, "eos_token": "<|im_end|>"}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer))

    assert model_terminators(str(tmp_path)) == (["<|im_end|>", "<|endoftext|>"], [151645, 151643])
    assert model_terminators(None) == ([], [])
    assert model_terminators(str(tmp_path / "missing")) == ([], [])


def test_the_probe_trips_only_after_its_minimum_samples_at_its_ratio():
    probe = TruncationProbe(enabled=True, min_samples=4, length_ratio=0.75)

    assert probe.observe([_response("length", "length", "length")]) is False
    assert probe.observe([_response("stop")]) is True
    assert probe.record() == {"tripped": True, "samples_seen": 4, "length_ratio_observed": 0.75}
    assert probe.observe([_response("length")]) is False


def test_a_disabled_probe_counts_and_never_trips():
    probe = TruncationProbe(enabled=False, min_samples=1, length_ratio=0.1)

    assert probe.observe([_response("length", "length")]) is False
    assert probe.record() == {"tripped": False, "samples_seen": 2, "length_ratio_observed": 1.0}


def test_the_task_stops_keep_their_place_ahead_of_the_model_terminators():
    payload = {"stop": ["</s>", "<|im_end|>"], "stop_token_ids": [7]}

    bounded = bounded_payload(payload, ["<|im_end|>", "<|endoftext|>", "a", "b"], [1, 7])

    assert bounded["stop"] == ["</s>", "<|im_end|>", "<|endoftext|>", "a"]
    assert len(bounded["stop"]) == MAX_STOP_STRINGS
    assert bounded["stop_token_ids"] == [1, 7]


def _model(tmp_path: Path, **changes) -> SuiteChatCompletion:
    settings = {
        "base_url": "http://127.0.0.1:1/v1/chat/completions",
        "model": "glm",
        "num_concurrent": 2,
        "tokenized_requests": False,
        "max_length": 65536,
        "probe_enabled": True,
        "probe_min_samples": 2,
        "probe_length_ratio": 0.5,
        "probe_file": str(tmp_path / "eval" / "hyperloom_eval_probe.json"),
        **changes,
    }
    return SuiteChatCompletion(**settings)


def test_the_name_the_process_registers_resolves_to_the_model_class():
    model_registry.register(MODEL_NAME, target=MODEL_CLASS)

    assert get_model(MODEL_NAME) is SuiteChatCompletion


def test_a_tripped_probe_is_recorded_and_answers_the_remaining_generations_empty(tmp_path: Path, capsys):
    model = _model(tmp_path)

    assert model.parse_generations(outputs=[_response("length", "length")]) == ["x", "x"]
    record = json.loads((tmp_path / "eval" / "hyperloom_eval_probe.json").read_text())
    answers = asyncio.run(model.amodel_call(None, None, [["m"]], generate=True, cache_keys=None))

    assert record == {"tripped": True, "samples_seen": 2, "length_ratio_observed": 1.0}
    assert "INFERENCE_SUITE_EVAL_PROBE_TRIPPED" in capsys.readouterr().err
    assert answers == [""]


def test_a_tripped_probe_answers_the_sequential_path_empty_without_a_request(tmp_path: Path, monkeypatch):
    def unreachable(self, *args, **kwargs):
        raise AssertionError("a request was sent after the probe tripped")

    monkeypatch.setattr(LocalChatCompletion, "model_call", unreachable)
    model = _model(tmp_path, num_concurrent=1)
    model.parse_generations(outputs=[_response("length", "length")])

    assert model.parse_generations(outputs=model.model_call([["m"]], generate=True, gen_kwargs={})) == [""]


def test_chat_history_is_sent_without_a_type_field(tmp_path: Path):
    prompt = _model(tmp_path).apply_chat_template([{"role": "user", "content": "q"}])

    assert json.loads(prompt.prompt) == [{"role": "user", "content": "q"}]


def test_a_request_that_failed_before_any_response_names_the_cause(tmp_path: Path, monkeypatch):
    async def unbound(self, *args, **kwargs):
        raise UnboundLocalError("outputs")

    monkeypatch.setattr(LocalChatCompletion, "amodel_call", unbound)

    with pytest.raises(RuntimeError, match="refused connection"):
        asyncio.run(_model(tmp_path).amodel_call(None, None, [["m"]], generate=True))


def test_generation_payloads_stop_on_the_model_terminators(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {"2": "</s>"}}))
    model = _model(tmp_path, model_dir=str(model_dir))
    messages = [{"role": "user", "content": "q"}]

    generate = model._create_payload(messages, generate=True, gen_kwargs={"until": ["Question:"], "max_tokens": 9})

    assert (generate["stop"], generate["stop_token_ids"], generate["max_tokens"]) == (["Question:", "</s>"], [2], 9)
