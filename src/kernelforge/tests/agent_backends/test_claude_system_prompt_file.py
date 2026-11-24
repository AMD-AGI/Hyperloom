# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Claude CLI is handed its system prompt in a private file, so the prompt is not on the command line."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from kernelforge.agent_backends import claude as claude_mod
from kernelforge.tests.agent_backends._claude_fakes import _backend, _result_message, _spec


def test_a_query_starts_from_a_prompt_file_only_its_owner_can_read_and_removes_it_afterwards():
    captured: dict = {}
    seen: dict = {}
    backend = _backend([_result_message(session_id="sess-1")], captured)
    query = backend._query

    async def inspecting_query(prompt, options):
        handed = options.kwargs["system_prompt"]
        seen["form"] = handed["type"]
        seen["path"] = Path(handed["path"])
        seen["mode"] = stat.S_IMODE(seen["path"].stat().st_mode)
        async for message in query(prompt, options):
            yield message

    backend._query = inspecting_query

    asyncio.run(backend.run(_spec(system_prompt="measure with forge_driver.py and rocprofv3")))

    assert seen["form"] == "file"
    assert seen["mode"] == 0o600
    assert captured["system_prompt"] == "measure with forge_driver.py and rocprofv3"
    assert not seen["path"].exists()


def test_a_resumed_turn_is_handed_its_prompt_the_same_way():
    captured: dict = {}
    backend = _backend([_result_message(session_id="sess-1")], captured)

    asyncio.run(backend.resume(_spec(system_prompt="summarizer role"), "sess-1", "record your lesson"))

    assert captured["options"].kwargs["system_prompt"]["type"] == "file"
    assert captured["system_prompt"] == "summarizer role"


def test_an_sdk_that_cannot_read_a_prompt_file_is_given_the_prompt_inline(monkeypatch):
    monkeypatch.setattr(claude_mod, "_sdk_reads_prompt_files", lambda: False)
    captured: dict = {}
    backend = _backend([_result_message(session_id="sess-1")], captured)

    asyncio.run(backend.run(_spec(system_prompt="implementer system prompt")))

    assert captured["options"].kwargs["system_prompt"] == "implementer system prompt"


def test_a_caller_that_sets_the_prompt_option_itself_keeps_it(monkeypatch):
    captured: dict = {}
    backend = _backend([_result_message(session_id="sess-1")], captured)
    preset = {"type": "preset", "preset": "claude_code", "append": "extra"}

    asyncio.run(backend.run(_spec(provider_options={"system_prompt": preset})))

    assert captured["options"].kwargs["system_prompt"] == preset
