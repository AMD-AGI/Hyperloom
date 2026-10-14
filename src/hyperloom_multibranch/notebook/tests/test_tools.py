# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook tools a specialist's model calls: compact text back, errors marked, the caller bound by the handle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import StartupBrief
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import scripted_session
from hyperloom_multibranch.notebook.tools import ERROR, MAX_LINES, NotebookTool, lines, notebook_tools

GEMM_ON_C0 = {"parent_commit": "c0", "target": "moe gemm", "mechanism": "tuned kernel", "layer": "kernel"}
FLASH_ON_C1 = {"parent_commit": "c1", "target": "attention", "mechanism": "flash decode", "layer": "source"}
TUNED = {"config": {"extra_server_args": "--max-num-seqs 256"}}

Tools = Callable[..., dict[str, NotebookTool]]


@pytest.fixture
async def tools_of(tmp_path: Path) -> Tools:
    clock = FakeClock()
    root = tmp_path / "notebook"
    await scripted_session(root, clock)

    def bind(branch: str, briefing: object = None) -> dict[str, NotebookTool]:
        handle = LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, branch, branch), clock=clock)
        return {tool.name: tool for tool in notebook_tools(handle, briefing, branch=branch, ttl_seconds=60.0)}

    return bind


async def _call(tools: dict[str, NotebookTool], name: str, **arguments: object) -> str:
    return await tools[f"notebook_{name}"].handler(arguments)


async def test_every_tool_is_named_for_the_notebook_and_takes_only_its_declared_arguments(tools_of: Tools) -> None:
    tools = tools_of("s1")

    assert sorted(tools) == sorted(
        f"notebook_{name}"
        for name in ("lookup", "claim", "release", "search", "record", "branch_log", "append_branch_log", "brief")
    )
    assert all(tool.input_schema["additionalProperties"] is False for tool in tools.values())
    assert tools["notebook_claim"].input_schema["required"] == ["parent_commit", "target", "mechanism", "layer"]
    assert tools["notebook_lookup"].input_schema["properties"]["layer"]["enum"] == ["config", "source", "kernel"]


async def test_lookup_says_what_was_tried_on_the_state_and_elsewhere(tools_of: Tools) -> None:
    tools = tools_of("s1")

    tested = await _call(tools, "lookup", **GEMM_ON_C0, change=TUNED)
    free = await _call(tools, "lookup", **FLASH_ON_C1)
    elsewhere = await _call(tools, "lookup", **{**GEMM_ON_C0, "parent_commit": "i1"}, change=TUNED)

    assert tested.splitlines()[:2] == ["tested on c0", "- r1: improved +4.00%, committed as c1"]
    assert "- m1: improved +4.10%, merged as i1" in tested
    assert free == "free on c1"
    assert "- elsewhere, r1 on c0: improved +4.00%, committed as c1" in elsewhere


async def test_a_claim_is_the_handles_and_a_taken_one_is_refused_with_who_holds_it(tools_of: Tools) -> None:
    s1, s2 = tools_of("s1"), tools_of("s2")

    claimed = await _call(s1, "claim", **FLASH_ON_C1)
    refused = await _call(s2, "claim", **FLASH_ON_C1)
    released = await _call(s1, "release", **FLASH_ON_C1)
    again = await _call(s1, "release", **FLASH_ON_C1)

    assert claimed == "claimed attention / flash decode on c1 for s1, expires in 60s"
    assert refused.startswith(f"{ERROR} refused, in_progress on c1, claimed by s1")
    assert released == "released the claim on c1"
    assert again.startswith(f"{ERROR} ClaimLost:")


async def test_search_and_record_answer_in_a_few_lines(tools_of: Tools) -> None:
    tools = tools_of("s1")

    hits = await _call(tools, "search", query="split kv", limit=50)
    failed = await _call(tools, "record", request_id="r3")
    kept = await _call(tools, "record", request_id="r1")

    assert hits.startswith("- [") and len(hits.splitlines()) <= MAX_LINES
    assert await _call(tools, "search", query="no such thing") == "nothing matched"
    assert failed.splitlines()[1] == "outcome: the candidate failed (server exited during warmup)"
    assert "threshold 1.00%, pooled r1" in kept
    assert (await _call(tools, "record", request_id="r404")).startswith(ERROR)


async def test_the_branch_log_is_this_branchs_to_read_and_to_add_to(tools_of: Tools) -> None:
    tools = tools_of("s1")

    logged = await _call(tools, "append_branch_log", kind="plan", content="profile decode next")
    plans = await _call(tools, "branch_log", kinds=["plan"])
    everything = await _call(tools, "branch_log")

    assert logged.startswith("logged ")
    assert plans.endswith("[plan] profile decode next") and len(plans.splitlines()) == 1
    assert "[hypothesis/confirmed] the MoE GEMM is bound by memory bandwidth" in everything
    assert (await _call(tools, "branch_log", kinds=["diary"])).startswith(f"{ERROR} ValueError")
    status = await _call(tools, "append_branch_log", kind="hypothesis", content="x", hypothesis_status="open")
    assert status.startswith("logged ")
    assert await _call(tools_of("s3"), "branch_log") == "no entries"


async def test_the_brief_comes_from_the_briefing_the_tools_are_given(tools_of: Tools) -> None:
    class _Briefing:
        async def startup_brief(self, start_commit: str) -> StartupBrief:
            lessons = tuple(f"lesson {index}" for index in range(MAX_LINES + 5))
            return StartupBrief(start_commit, (), leads=("recipe: try expert parallel",), lessons=lessons)

    local = await _call(tools_of("s1"), "brief", start_commit="c1")
    briefed = await _call(tools_of("s1", _Briefing()), "brief", start_commit="c1")

    assert local.splitlines()[:3] == [
        "Startup brief from c1",
        "Tried:",
        "- moe gemm / tuned kernel on c0: committed +4.00% (add args --max-num-seqs 256)",
    ]
    assert "- scheduler / chunked prefill on c0: not judged" in local
    assert "Lessons:\n- lesson 0" in briefed and "... 5 more\nLeads:\n- recipe: try expert parallel" in briefed


async def test_bad_arguments_come_back_as_marked_errors(tools_of: Tools) -> None:
    tools = tools_of("s1")

    unknown_mode = await _call(tools, "lookup", **GEMM_ON_C0, change={"config": {"args_mode": "sideways"}})
    missing = await _call(tools, "search")
    bad_layer = await _call(tools, "claim", **{**FLASH_ON_C1, "layer": "firmware"})

    assert unknown_mode.startswith(f"{ERROR} RecipeError")
    assert missing == f"{ERROR} KeyError: 'query'"
    assert bad_layer.startswith(f"{ERROR} ValueError")


def test_long_lists_stop_at_the_cap_and_say_how_many_more() -> None:
    shown = lines([f"item {index}" for index in range(MAX_LINES + 3)] + ["x" * 500])

    assert len(shown) == MAX_LINES + 1 and shown[-1] == "... 4 more"
    assert lines(["x" * 500])[0].endswith("...") and len(lines(["x" * 500])[0]) == 160
