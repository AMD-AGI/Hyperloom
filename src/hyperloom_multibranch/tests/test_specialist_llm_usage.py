# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

from hyperloom_multibranch.specialist.llm_usage import LaunchCost, launch_cost

USAGE = {"input_tokens": 12, "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 400, "output_tokens": 900}


def _log(tmp_path, *rows):
    path = tmp_path / "process.log"
    path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n")
    return path


def test_the_last_result_event_gives_the_cost_every_prompt_token_and_the_session(tmp_path):
    path = _log(
        tmp_path,
        {"type": "system", "subtype": "init", "session_id": "sess-1"},
        {"type": "assistant", "message": {"usage": USAGE}},
        "not json, a stray line the CLI printed",
        {"type": "result", "subtype": "success", "total_cost_usd": 0.5, "usage": {}, "session_id": "sess-0"},
        {"type": "result", "subtype": "success", "total_cost_usd": 1.25, "usage": USAGE, "session_id": "sess-1"},
    )

    assert launch_cost(path) == LaunchCost(1.25, 3412, 900, "sess-1")


def test_a_launch_without_a_result_event_or_a_log_has_no_known_cost(tmp_path):
    killed = _log(tmp_path, {"type": "assistant", "message": {"usage": USAGE}}, '{"type": "result", "subtype"')

    assert launch_cost(killed) is None
    assert launch_cost(tmp_path / "missing.log") is None


def test_a_result_event_without_a_cost_is_not_one(tmp_path):
    codex_like = _log(tmp_path, {"type": "result", "usage": USAGE}, {"type": "result", "total_cost_usd": True})

    assert launch_cost(codex_like) is None
