# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook tools a specialist's model calls: compact text back, errors marked, the caller bound by the handle."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer, intent_fingerprint
from hyperloom_multibranch.contracts.notebook.peers import Validity, Verdict
from hyperloom_multibranch.contracts.notebook.records import (
    Claim,
    ClaimKey,
    DedupAnswer,
    NotebookSnapshot,
    OpShares,
    StartupBrief,
    TestRecord,
    TriedDirection,
)
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, Conclusion, DedupStatus
from hyperloom_multibranch.notebook.changes import NO_CHANGE
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.contracts.notebook.records import REPLAY_TOOL, PeerReplay, WarmReplay
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog, SessionLearnings
from hyperloom_multibranch.notebook.tests.factories import FakeClock, intent, request, result
from hyperloom_multibranch.notebook.tests.sessions import scripted_session
from hyperloom_multibranch.notebook.tools import (
    CEILING_OPS,
    ERROR,
    MAX_LINES,
    NotebookTool,
    lines,
    notebook_tools,
    outcome,
    render_answer,
    render_brief,
    render_ceiling,
)

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


async def test_the_replay_tool_names_the_intent_to_claim_and_the_change_on_the_branch_head(tools_of: Tools) -> None:
    delta = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))
    on_head = WarmReplay("inference:older", "c3", delta, IntentLayer.CONFIG)

    async def warm_replay(canonical_id: str) -> WarmReplay | None:
        return on_head if canonical_id == on_head.canonical_id else None

    tools = tools_of("s1", SimpleNamespace(warm_replay=warm_replay))

    text = await _call(tools, "replay", canonical_id="inference:older")

    _, claim, change = text.splitlines()
    assert 'target "warm start", mechanism "inference:older" and layer "config"' in claim
    assert 'parent_commit "c3"' in claim and RecipeDelta.from_document(json.loads(change)) == delta
    assert REPLAY_TOOL in tools
    assert (await _call(tools, "replay", canonical_id="inference:none")).startswith(ERROR)
    assert (await _call(tools_of("s1"), "replay", canonical_id="inference:older")).startswith(ERROR)


async def test_the_replay_tool_replays_another_branchs_test_by_its_request_id_on_the_branch_head(
    tools_of: Tools,
) -> None:
    tested = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))
    on_head = PeerReplay("r1", "s1", "c3", tested, IntentLayer.CONFIG)

    async def peer_replay(request_id: str) -> PeerReplay:
        if request_id != on_head.request_id:
            raise ValueError(f"{request_id} was measured without a change")
        return on_head

    tools = tools_of("s3", SimpleNamespace(peer_replay=peer_replay))

    first, claim, change = (await _call(tools, "replay", request_id="r1")).splitlines()

    assert first == "replay of r1, the change s1 tested, on your head c3"
    assert claim == (
        'claim with parent_commit "c3", target "warm start", mechanism "r1" and layer "config", then submit this '
        "change on that commit:"
    )
    assert RecipeDelta.from_document(json.loads(change)) == tested
    assert await _call(tools, "replay", request_id="r9") == f"{ERROR} ValueError: r9 was measured without a change"
    for both_or_neither in ({"request_id": "r1", "canonical_id": "inference:older"}, {}):
        assert (await _call(tools, "replay", **both_or_neither)).startswith(f"{ERROR} name a champion's canonical_id")
    assert (await _call(tools_of("s3"), "replay", request_id="r1")).startswith(f"{ERROR} no replay of r1")


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
            "replay",
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

    assert claimed.splitlines() == [
        "claimed attention / flash decode on c1 for s1, expires in 60s",
        f"intent fingerprint (submit tests with it): {intent_fingerprint('attention', 'flash decode', IntentLayer.SOURCE)}",
    ]
    assert refused.startswith(f"{ERROR} refused, in_progress on c1, claimed by s1")
    assert released == "released the claim on c1"
    assert again.startswith(f"{ERROR} ClaimLost:")


async def test_a_claim_another_branch_is_testing_on_its_head_is_refused_saying_who_and_what_comes_next(
    tools_of: Tools,
) -> None:
    await _call(tools_of("s1"), "claim", **FLASH_ON_C1)

    refused = await _call(tools_of("s3"), "claim", **{**FLASH_ON_C1, "parent_commit": "c7"})

    assert refused == (
        f"{ERROR} refused, testing_elsewhere: s1 is testing this intent on c1; its outcome comes to your inbox when "
        "that test ends, and the intent is free again then"
    )


def test_an_answer_for_an_intent_tested_elsewhere_lists_the_tests_in_flight_with_their_branch() -> None:
    tried = intent("s1", "c1")
    held = Claim(ClaimKey("c1", tried.fingerprint), "s1", tried, "a", 60.0, 0.0, 0.0, 60.0, ClaimStatus.HELD, "r9")
    running = TestRecord(request("r9", tried), requested_at=1.0)

    answer = DedupAnswer(ClaimKey("c7", tried.fingerprint), "b", DedupStatus.TESTING_ELSEWHERE, held, (running,))

    assert render_answer(answer).splitlines()[1:] == ["- r9 on s1 from c1: in flight"]


async def test_a_claim_whose_change_misspells_a_key_is_refused_and_holds_nothing(tools_of: Tools) -> None:
    tools = tools_of("s1")

    misspelt = await _call(tools, "claim", **FLASH_ON_C1, change={"config": {"server_args": "--max-num-seqs 256"}})
    claimed = await _call(tools, "claim", **FLASH_ON_C1, change=TUNED)

    assert misspelt.startswith(f"{ERROR} RecipeError: unknown key 'server_args' in a config delta")
    assert claimed.startswith("claimed attention / flash decode on c1 for s1")


async def test_a_claim_states_its_ceiling_and_one_that_cannot_reach_the_bar_is_refused(tools_of: Tools) -> None:
    tools = tools_of("s1")

    claimed = (await _call(tools, "claim", **FLASH_ON_C1, target_share_pct=30)).splitlines()
    norm_on_c1 = {**FLASH_ON_C1, "target": "rms norm", "mechanism": "fused kernel", "change": TUNED}
    refused = await _call(tools, "claim", **norm_on_c1, target_share_pct=1.0)
    wrong = await _call(tools, "claim", **norm_on_c1, target_share_pct="most")

    assert claimed[2] == (
        "Amdahl ceiling +42.86% for a target worth 30.00% of GPU time; "
        "unverified, c1 has no op shares (start_analysis profiles it)"
    )
    assert refused.startswith(f"{ERROR} refused, a target worth 1.00% of GPU time on c1 can gain at most 1.01%")
    assert "below the bar of 1.60%" in refused
    assert wrong.startswith(f"{ERROR} ValueError: target_share_pct is a number")
    assert tools["notebook_claim"].input_schema["properties"]["target_share_pct"]["maximum"] == 100


def test_the_ceiling_line_names_the_parents_largest_ops() -> None:
    shares = OpShares("c1", "job-7", "p1", {"attention": 22.0, "moe_gemm": 41.5, "norm": 1.0})
    stated = Intent("s1", "c1", "moe gemm", "tuned kernel", IntentLayer.KERNEL, 41.5)
    unstated = replace(stated, target_share_pct=None)

    assert render_ceiling(stated, shares).splitlines() == [
        "Amdahl ceiling +70.94% for a target worth 41.50% of GPU time",
        (
            "largest ops on c1 (job-7): moe_gemm 41.5% (ceiling +70.9%), attention 22.0% (ceiling +28.2%), "
            "norm 1.0% (ceiling +1.0%)"
        ),
    ]
    assert render_ceiling(unstated, shares).startswith("largest ops on c1")
    assert render_ceiling(replace(unstated, layer=IntentLayer.CONFIG), shares) == ""
    assert render_ceiling(unstated, None) == ""


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

    assert len(claimed) == 4 and claimed[2] == "This session's learnings about it:"
    assert claimed[3].startswith(f"- [s2, pitfall about flash decode / block size] {pitfall} [")
    assert len(plain.splitlines()) == 2 and plain.startswith("claimed sampler / top-k fusion on c1")


async def test_the_learnings_tool_finds_this_sessions_and_what_its_finder_adds(tools_of: Tools) -> None:
    class _Past:
        def __init__(self) -> None:
            self.asked: list[str] = []

        async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
            self.asked.append(branch)
            return ["Past sessions' learnings:", f"- [sess-0, same conditions, measured] {query} paid off [exp-1]"]

    s1 = tools_of("s1")
    await _call(s1, "learn", kind="insight", content="attention dominates decode at conc 64 on this model")
    finder = _Past()

    found = (await _call(s1, "learnings", query="attention decode")).splitlines()
    none = await _call(s1, "learnings", query="speculative draft tokens")
    past = await _call(tools_of("s1", learnings=finder), "learnings", query="fp8 kv cache")

    assert found[0] == "This session's learnings:" and "attention dominates decode at conc 64" in found[1]
    assert none == "nothing matched"
    assert past.splitlines() == [
        "Past sessions' learnings:",
        "- [sess-0, same conditions, measured] fp8 kv cache paid off [exp-1]",
    ]
    assert finder.asked == ["s1"]


async def test_a_claim_and_the_learnings_tool_log_what_they_showed_the_branch(tools_of: Tools, tmp_path: Path) -> None:
    path = tmp_path / "reports" / LEARNINGS_LOG
    s1 = tools_of("s1", learnings=SessionLearnings(RetrievalLog(path)))
    pitfall = "flash decode stalls unless the block size is 16"
    await _call(tools_of("s2"), "learn", kind="pitfall", content=pitfall, target="flash decode", mechanism="block size")

    await _call(s1, "claim", **FLASH_ON_C1)
    await _call(s1, "learnings", query="block size")
    logged = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [(line["branch"], line["source"], line["query"], len(line["here"])) for line in logged] == [
        ("s1", "claim", "attention flash decode", 1),
        ("s1", "tool", "block size", 1),
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


def test_a_test_tried_without_a_change_says_so_rather_than_reading_as_no_gain() -> None:
    tried = (
        TriedDirection(
            "c0",
            "r1",
            "torch compile",
            "inductor fusion",
            IntentLayer.CONFIG,
            Verdict.FLAT,
            Conclusion.RECORDED_ONLY,
            0.22,
            NO_CHANGE,
        ),
        TriedDirection(
            "c0",
            "r2",
            "decode batching",
            "raise max-num-seqs",
            IntentLayer.CONFIG,
            Verdict.FLAT,
            Conclusion.RECORDED_ONLY,
            0.1,
            "add args --max-num-seqs 256",
        ),
    )

    lines = render_brief(StartupBrief("c1", tried)).splitlines()

    assert lines[2:4] == [
        "- torch compile / inductor fusion on c0: recorded_only +0.22% (measured without a change: its delta was empty)",
        "- decode batching / raise max-num-seqs on c0: recorded_only +0.10% (add args --max-num-seqs 256)",
    ]


def test_the_brief_lists_the_start_commits_largest_ops_with_their_ceilings_before_what_was_tried() -> None:
    shares = OpShares("c1", "job-1", "", {f"op{index}": float(index) for index in range(1, 8)} | {"gemm": 50.0})
    profiled = render_brief(StartupBrief("c1", (), op_shares=shares)).splitlines()

    assert profiled[1:4] == [
        "Ceilings (largest ops; claim with target_share_pct):",
        "- gemm: 50.0% of GPU time, at most +100.0%",
        "- op7: 7.0% of GPU time, at most +7.5%",
    ]
    assert len(profiled) == 2 + CEILING_OPS
    assert "Ceilings" not in render_brief(StartupBrief("c1", ()))


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
