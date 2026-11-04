# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.specialist.llm_usage import UNNAMED, launch_usage

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


def test_a_missing_log_or_one_with_no_call_used_nothing(tmp_path):
    assert launch_usage(tmp_path / "missing.log").tokens == {}
    assert launch_usage(_log(tmp_path, {"type": "system", "subtype": "init"})).tokens == {}
