# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The greedy decode pass: the prompt set and its identity, the requests, the record and reading it back."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import MeasurementError, TestResult, Validity
from hyperloom.measure.decode import (
    OUTPUTS_PATH,
    DecodePassError,
    parse_outputs,
    plan_pass,
    read_decode_outputs,
    read_decode_pass,
    run_pass,
    vocab_size_of,
    write_outputs,
)
from hyperloom.measure.tests.decode_fixtures import expected_tokens, fake_server

PLAN = replace(plan_pass(isl=16, osl=1024, vocab_size=151936), prompts=6)


def test_the_cap_is_the_output_length_up_to_256_and_the_prompts_stay_below_every_vocabulary():
    assert plan_pass(16, 1024, 151936).max_tokens == 256 and plan_pass(16, 64, 151936).max_tokens == 64
    prompts = PLAN.prompt_ids()
    assert len(prompts) == 6 and {len(p) for p in prompts} == {16} and len({tuple(p) for p in prompts}) == 6
    assert all(PLAN.token_low <= t < PLAN.token_high for p in prompts for t in p)
    assert prompts == replace(PLAN).prompt_ids(), "the same plan always draws the same prompts"


@pytest.mark.parametrize("change", [{"seed": 1}, {"prompts": 5}, {"isl": 17}, {"max_tokens": 255}, {"token_high": 999}])
def test_everything_that_decides_the_prompt_set_changes_the_identity(change):
    assert replace(PLAN, **change).identity != PLAN.identity
    assert PLAN.identity.endswith("|temperature=0|top_k=1")


def test_the_pass_sends_greedy_requests_at_the_graded_point_and_returns_each_prompts_tokens_in_order():
    with fake_server(delay_sec=0.05) as server:
        outputs = run_pass(
            PLAN, base_url=f"http://127.0.0.1:{server.port}", model="m", concurrency=2, request_timeout_sec=10
        )

    prompts = PLAN.prompt_ids()
    assert outputs == tuple(tuple(expected_tokens(p, 256)) for p in prompts)
    assert {r["path"] for r in server.requests} == {"/v1/completions"}
    for sent in server.requests:
        assert (sent["temperature"], sent["top_k"], sent["max_tokens"], sent["model"]) == (0, 1, 256, "m")
        assert len(sent["prompt"]) == 16 and sent["return_token_ids"] is True
    assert server.peak == 2, "at most the graded concurrency in flight, and it is used"


@pytest.mark.parametrize(("mode", "reason"), [("error", "http_500"), ("no_ids", "no_token_ids_in_response")])
def test_a_pass_that_gets_no_answer_it_can_use_fails_with_its_reason_and_returns_nothing(mode, reason):
    with fake_server(mode) as server, pytest.raises(DecodePassError) as caught:
        run_pass(PLAN, base_url=f"http://127.0.0.1:{server.port}", model="m", concurrency=4, request_timeout_sec=10)
    assert str(caught.value) == reason


def test_a_server_that_is_not_there_fails_the_pass():
    with pytest.raises(DecodePassError, match="request_failed_"):
        run_pass(PLAN, base_url="http://127.0.0.1:1", model="m", concurrency=1, request_timeout_sec=2)


def test_the_record_has_the_identity_header_then_one_line_per_prompt_and_parses_back(tmp_path: Path):
    outputs = ((5, 6), (), (7,))
    path = write_outputs(tmp_path, replace(PLAN, prompts=3), outputs)

    assert path == tmp_path / OUTPUTS_PATH
    first = json.loads(path.read_text().splitlines()[0])["decode_pass"]
    assert first["identity"] == replace(PLAN, prompts=3).identity and first["isl"] == 16 and first["prompts"] == 3
    record = parse_outputs(path.read_text())
    assert record.identity == first["identity"] and record.outputs == outputs


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json\n",
        '{"decode_pass": {"identity": "x", "prompts": 1}}\n',  # header promises a line that is not there
        '{"decode_pass": {"identity": "x", "prompts": 1}}\n{"prompt": 1, "tokens": [1]}\n',  # out of order
        '{"decode_pass": {"identity": "x", "prompts": 1}}\n{"prompt": 0, "tokens": [1.5]}\n',
        '{"decode_pass": {"identity": "x", "prompts": 1}}\n{"prompt": 0, "tokens": [true]}\n',
    ],
)
def test_a_record_that_is_not_a_complete_pass_does_not_parse(text):
    with pytest.raises(DecodePassError, match="outputs.jsonl"):
        parse_outputs(text)


def _result(**fields) -> TestResult:
    base = {"request_id": "r1", "validity": Validity.SUCCEEDED, "output_dir": "tests/r1/attempt-1"}
    return TestResult(**{**base, **fields})


def test_the_gate_reads_the_outputs_and_the_identity_of_a_result_from_its_landed_bundle(tmp_path: Path):
    plan = replace(PLAN, prompts=2)
    write_outputs(tmp_path / "tests/r1/attempt-1", plan, ((1, 2), (3,)))
    result = _result(decode_outputs=OUTPUTS_PATH, decode_identity=plan.identity)

    assert read_decode_outputs(result, tmp_path) == ((1, 2), (3,))
    assert read_decode_pass(result, tmp_path).identity == plan.identity


def test_a_result_whose_recorded_identity_is_not_the_files_is_not_read(tmp_path: Path):
    plan = replace(PLAN, prompts=2)
    write_outputs(tmp_path / "tests/r1/attempt-1", plan, ((1, 2), (3,)))
    other = _result(decode_outputs=OUTPUTS_PATH, decode_identity=replace(plan, isl=17).identity)

    with pytest.raises(DecodePassError, match="holds identity"):
        read_decode_outputs(other, tmp_path)


def test_a_result_with_no_pass_or_a_path_out_of_the_session_is_not_read(tmp_path: Path):
    with pytest.raises(DecodePassError, match="records no decode pass"):
        read_decode_outputs(_result(), tmp_path)
    escaping = _result(output_dir="a/b", decode_outputs="x/y", decode_identity="i")
    with pytest.raises(DecodePassError, match="cannot be read"):
        read_decode_outputs(escaping, tmp_path)
    with pytest.raises(MeasurementError, match="relative path"):
        _result(decode_outputs="../../etc/passwd", decode_identity="i")


def test_the_result_codec_carries_the_decode_fields_and_older_documents_still_load():
    result = _result(decode_outputs=OUTPUTS_PATH, decode_identity="sha256:abc|temperature=0|top_k=1")

    assert TestResult.from_document(result.to_document()) == result
    older = {k: v for k, v in result.to_document().items() if not k.startswith("decode_")}
    loaded = TestResult.from_document(older)
    assert (loaded.decode_outputs, loaded.decode_identity) == ("", "")


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"decode_outputs": OUTPUTS_PATH}, "both set or both empty"),
        ({"decode_identity": "i"}, "both set or both empty"),
        ({"decode_outputs": OUTPUTS_PATH, "decode_identity": "i", "validity": Validity.CANDIDATE_FAILED}, "succeeded"),
    ],
)
def test_the_decode_fields_are_set_together_and_only_on_a_succeeded_test(fields, match):
    with pytest.raises(MeasurementError, match=match):
        _result(**fields)


def test_the_vocabulary_comes_from_the_model_config(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"text_config": {"vocab_size": 32000}}')
    assert vocab_size_of(tmp_path) == 32000
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(DecodePassError, match="model_config_has_no_vocab_size"):
        vocab_size_of(tmp_path)
    with pytest.raises(DecodePassError, match="model_config_unreadable"):
        vocab_size_of(tmp_path / "missing")
