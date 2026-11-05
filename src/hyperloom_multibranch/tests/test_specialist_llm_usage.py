# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.specialist.llm_usage import UNNAMED, launch_cost, launch_usage

OPUS, HAIKU = "claude-opus-5-5", "claude-haiku-5-5"


def _log(tmp_path, *rows):
    path = tmp_path / "process.log"
    path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n")
    return path


def _message(message_id, model, **usage):
    return {"type": "assistant", "message": {"id": message_id, "model": model, "usage": usage}}


def test_each_claude_message_counts_once_by_its_model_even_without_a_result_event(tmp_path):
    first = {
        "input_tokens": 10,
        "cache_read_input_tokens": 3000,
        "cache_creation_input_tokens": 400,
        "output_tokens": 90,
    }
    path = _log(
        tmp_path,
        {"type": "system", "subtype": "init", "session_id": "sess-1"},
        _message("m1", OPUS, **first),
        _message("m1", OPUS, **first),
        "not json, a stray line the CLI printed",
        _message("m2", HAIKU, input_tokens=5, output_tokens=7),
        _message("m3", OPUS, input_tokens=1, cache_read_input_tokens=50, output_tokens=2),
    )

    used = launch_usage(path)

    assert used.session_id == "sess-1"
    assert used.tokens == {OPUS: TokenCounts(11, 92, 3050, 400), HAIKU: TokenCounts(5, 7)}
    assert used.total.prompt == 11 + 3050 + 400 + 5


def test_a_split_cache_write_is_billed_by_its_lifetime(tmp_path):
    split = {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 100}
    path = _log(tmp_path, _message("m1", OPUS, cache_creation_input_tokens=400, cache_creation=split))

    assert launch_usage(path).tokens == {OPUS: TokenCounts(cache_write=300, cache_write_1h=100)}


def test_codex_turns_add_up_with_their_cached_input_split_off(tmp_path):
    path = _log(
        tmp_path,
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50}},
        {"type": "turn.completed", "usage": {"input_tokens": 200, "cached_input_tokens": 0, "output_tokens": 5}},
    )

    used = launch_usage(path)

    assert (used.session_id, used.tokens) == ("thread-1", {UNNAMED: TokenCounts(400, 55, 800)})


def test_a_message_the_cli_made_up_itself_is_no_model_call(tmp_path):
    path = _log(
        tmp_path,
        _message("m1", OPUS, input_tokens=1000, output_tokens=500),
        _message("m2", "<synthetic>", input_tokens=0, output_tokens=0),
    )

    used = launch_usage(path)

    assert used.tokens == {OPUS: TokenCounts(1000, 500)}
    assert launch_cost(used.tokens, OPUS) is not None


def _result(**usage):
    return {"type": "result", "usage": usage}


def test_a_gateway_that_reports_usage_only_on_the_result_rows_is_read_from_them(tmp_path):
    path = _log(
        tmp_path,
        _message("m1", "glm-5", input_tokens=0, output_tokens=0),
        _message("m2", "glm-5", input_tokens=0, output_tokens=0),
        _result(input_tokens=5000, output_tokens=700),
        _message("m3", "glm-5", input_tokens=0, output_tokens=0),
        _result(input_tokens=100, output_tokens=20),
    )

    assert launch_usage(path).tokens == {"glm-5": TokenCounts(5100, 720)}


def test_a_finished_query_counts_by_its_result_row_and_a_killed_one_by_its_messages(tmp_path):
    path = _log(
        tmp_path,
        _message("m1", OPUS, input_tokens=10, cache_read_input_tokens=3000, output_tokens=9),
        _result(input_tokens=10, cache_read_input_tokens=3000, output_tokens=500),
        _message("m2", OPUS, input_tokens=4, output_tokens=30),
    )

    assert launch_usage(path).tokens == {OPUS: TokenCounts(14, 530, 3000)}


def test_messages_without_ids_are_read_from_the_result_rows_since_their_repeats_cannot_be_told_apart(tmp_path):
    repeated = {"type": "assistant", "message": {"model": OPUS, "usage": {"input_tokens": 10, "output_tokens": 4}}}
    path = _log(tmp_path, repeated, repeated, _result(input_tokens=10, output_tokens=4))

    assert launch_usage(path).tokens == {OPUS: TokenCounts(10, 4)}


def test_a_missing_log_or_one_with_no_call_used_nothing(tmp_path):
    assert launch_usage(tmp_path / "missing.log").tokens == {}
    assert launch_usage(_log(tmp_path, {"type": "system", "subtype": "init"})).tokens == {}
