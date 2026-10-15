# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook tools a specialist's model calls: compact text back, errors marked, the caller bound by the handle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import Validity
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, StartupBrief, TestRecord
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock, intent, request, result
from hyperloom_multibranch.notebook.tests.sessions import scripted_session
from hyperloom_multibranch.notebook.tools import ERROR, MAX_LINES, NotebookTool, lines, notebook_tools, outcome

GEMM_ON_C0 = {"parent_commit": "c0", "target": "moe gemm", "mechanism": "tuned kernel", "layer": "kernel"}
FLASH_ON_C1 = {"parent_commit": "c1", "target": "attention", "mechanism": "flash decode", "layer": "source"}
TUNED = {"config": {"extra_server_args": "--max-num-seqs 256"}}

Tools = Callable[..., dict[str, NotebookTool]]


@pytest.fixture
async def tools_of(tmp_path: Path) -> Tools:
    clock = FakeClock()
    root = tmp_path / "notebook"
    await scripted_session(root, clock)

    def bind(branch: str, briefing: object = None, learnings: object = None) -> dict[str, NotebookTool]:
        handle = LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, branch, branch), clock=clock)
        bound = notebook_tools(handle, briefing, branch=branch, ttl_seconds=60.0, learnings=learnings)
        return {tool.name: tool for tool in bound}

    return bind


async def _call(tools: dict[str, NotebookTool], name: str, **arguments: object) -> str:
    return await tools[f"notebook_{name}"].handler(arguments)


async def test_every_tool_is_named_for_the_notebook_and_takes_only_its_declared_arguments(tools_of: Tools) -> None:
    tools = tools_of("s1")

    assert sorted(tools) == sorted(
        f"notebook_{name}"
        for name in (
            "lookup",
            "claim",
            "release",
            "search",
            "record",
            "branch_log",
            "append_branch_log",
            "learn",
            "learnings",
            "brief",
        )
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


async def test_a_learning_is_written_for_this_branch_and_a_vague_one_comes_back_as_an_error(tools_of: Tools) -> None:
    tools = tools_of("s1")

    learnt = await _call(
        tools,
        "learn",
        kind="pitfall",
        content="tuning the moe gemm again on c1 gave nothing over the first tuning",
        target="moe gemm",
        mechanism="tuned kernel",
        layer="kernel",
        request_ids=["r4"],
    )
    vague = await _call(tools, "learn", kind="insight", content="optimize")
    unknown = await _call(tools, "learn", kind="rumour", content="the scheduler favours short branches")

    assert learnt.startswith("learnt ")
    assert vague.startswith("error: ValueError") and "at least 20 characters" in vague
    assert unknown.startswith("error: ValueError")
    assert tools["notebook_learn"].input_schema["required"] == ["kind", "content"]
    assert tools["notebook_learn"].input_schema["properties"]["content"]["maxLength"] == 600


async def test_a_granted_claim_lists_this_sessions_learnings_about_its_direction(tools_of: Tools) -> None:
    s1, s2 = tools_of("s1"), tools_of("s2")
    pitfall = "flash decode needs block size 16 on this image"
    await _call(s2, "learn", kind="pitfall", content=pitfall, target="flash decode", mechanism="block size")
    await _call(s2, "learn", kind="tooling", content="the profiler needs --record-shapes to name the gemm shapes")

    claimed = (await _call(s1, "claim", **FLASH_ON_C1)).splitlines()
    plain = await _call(
        s1,
        "claim",
        parent_commit="c1",
        target="sampler",
        mechanism="top-k fusion",
        layer="config",
        change={"config": {"extra_server_args": "--enable-sampler-fusion"}},
    )

    assert len(claimed) == 3 and claimed[1] == "This session's learnings about it:"
    assert claimed[2].startswith(f"- [s2, pitfall about flash decode / block size] {pitfall} [")
    assert len(plain.splitlines()) == 1 and plain.startswith("claimed sampler / top-k fusion on c1")


async def test_the_learnings_tool_finds_this_sessions_and_what_its_finder_adds(tools_of: Tools) -> None:
    class _Past:
        async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int) -> list[str]:
            return ["Past sessions' learnings:", f"- [sess-0, same conditions, measured] {query} paid off [exp-1]"]

    s1 = tools_of("s1")
    await _call(s1, "learn", kind="insight", content="attention dominates decode at conc 64 on this model")

    found = (await _call(s1, "learnings", query="attention decode")).splitlines()
    none = await _call(s1, "learnings", query="speculative draft tokens")
    past = await _call(tools_of("s1", learnings=_Past()), "learnings", query="fp8 kv cache")

    assert found[0] == "This session's learnings:" and "attention dominates decode at conc 64" in found[1]
    assert none == "nothing matched"
    assert past.splitlines() == [
        "Past sessions' learnings:",
        "- [sess-0, same conditions, measured] fp8 kv cache paid off [exp-1]",
    ]


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


def test_a_withdrawn_or_unmeasured_test_says_so_and_why() -> None:
    asked = request("r1", intent("s1", "c0"))
    withdrawn = TestRecord(asked, withdrawn_at=1.0, withdrawal_reason="branch s1 stopped")
    lost = TestRecord(asked, result("r1", validity=Validity.MEASUREMENT_FAILED, failure_reason="no healthy node"))

    assert outcome(withdrawn) == "withdrawn (branch s1 stopped)"
    assert outcome(lost) == "not measured (no healthy node)"
