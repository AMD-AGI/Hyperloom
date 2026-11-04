# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Past sessions' learnings after this session's: marked by the conditions they differ in, each query read once."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hyperloom_kb import ReadStatus, RemoteReadResult

from hyperloom_multibranch.contracts.blackboard.records import Learning, LearningRecord, BlackboardSnapshot
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind
from hyperloom_multibranch.blackboard.knowledge.learning_recall import (
    AGENTX_SESSION,
    INFERENCEX_SESSION,
    LEARNING_DECISION,
    LearningRecall,
    read_past_learnings,
)
from hyperloom_multibranch.blackboard.knowledge.recall import IdentityIndex
from hyperloom_multibranch.blackboard.knowledge.tests.fakes import FakeExperienceReader, FakeExport, completed_read
from hyperloom_multibranch.blackboard.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.blackboard.tests.factories import profile

SAME = {
    "model": "deepseek-ai/DeepSeek-R1",
    "gpu": "mi355x",
    "framework": "vllm",
    "tp": 8,
    "conc": 64,
    "isl": 1024,
    "osl": 1024,
}


def _past(
    experience_id: str, identity: dict[str, object], *, run: str = "sess-0", decision: str = "measured"
) -> dict[str, object]:
    return {
        "experience_id": experience_id,
        "source_run_id": run,
        "identity": identity,
        "change_summary": f"what {experience_id} learnt about the kv cache",
        "decision": decision,
        "baseline_value": None,
        "outcome_value": None,
    }


def test_past_learnings_say_which_conditions_differ_and_leave_this_session_out() -> None:
    reader = FakeExperienceReader(
        completed_read(
            _past("exp-same", SAME),
            _past("exp-tp4", {**SAME, "tp": 4}, decision="unmeasured"),
            _past("exp-mine", SAME, run="sess-1"),
        )
    )

    found = read_past_learnings(reader, profile(), "kv cache", limit=5)

    assert found.available and [item.experience_id for item in found.items] == ["exp-same", "exp-tp4"]
    assert [item.render() for item in found.items] == [
        "- [sess-0, same conditions, measured] what exp-same learnt about the kv cache [exp-same]",
        "- [sess-0, tp differ, unmeasured] what exp-tp4 learnt about the kv cache [exp-tp4]",
    ]
    ((decision, context, limit),) = reader.calls
    assert (decision, limit, context["objective"], context["observations"]) == (
        LEARNING_DECISION,
        5,
        {"id": "specialist_learning@v1"},
        {"query": "kv cache"},
    )


def test_the_read_context_holds_the_declared_identity_under_identity_and_the_workload_under_workload() -> None:
    reader = FakeExperienceReader(completed_read())

    read_past_learnings(reader, profile(), "kv cache", limit=5)

    ((_, context, _),) = reader.calls
    sizes = {"tp": 8, "conc": 64, "isl": 1024, "osl": 1024}
    assert sizes.items() <= context["identity"].items()
    assert sizes.items() <= context["workload"].items()


def test_a_learning_of_the_other_kind_of_session_says_which_kind() -> None:
    agentx_conditions = {key: value for key, value in SAME.items() if key not in ("isl", "osl")}
    reader = FakeExperienceReader(completed_read(_past("exp-inferencex", SAME), _past("exp-agentx", agentx_conditions)))

    in_agentx = read_past_learnings(reader, profile(isl=None, osl=None), "kv cache", limit=5)
    in_inferencex = read_past_learnings(reader, profile(), "kv cache", limit=5)

    assert [item.render() for item in in_agentx.items] == [
        f"- [sess-0, learnt in an {INFERENCEX_SESSION} session, measured] what exp-inferencex learnt about the kv"
        " cache [exp-inferencex]",
        "- [sess-0, same conditions, measured] what exp-agentx learnt about the kv cache [exp-agentx]",
    ]
    assert [item.render() for item in in_inferencex.items][1] == (
        f"- [sess-0, isl, osl differ, learnt in an {AGENTX_SESSION} session, measured] what exp-agentx learnt about"
        " the kv cache [exp-agentx]"
    )


def test_without_the_identity_in_the_summary_the_export_supplies_it() -> None:
    bare = {key: value for key, value in _past("exp-same", SAME).items() if key != "identity"}
    index = IdentityIndex(FakeExport([{"id": "exp-same", "identity": SAME}]))

    (item,) = read_past_learnings(
        FakeExperienceReader(completed_read(bare)), profile(), "kv", limit=5, identities=index
    ).items

    assert item.differing == ()


def _learnt(snapshot: BlackboardSnapshot) -> BlackboardSnapshot:
    smaller = Learning("s1", LearningKind.INSIGHT, "a smaller kv cache block helps decode")
    return dataclasses.replace(snapshot, learnings=(LearningRecord("s1:1", 1.0, smaller),))


async def test_this_sessions_learnings_come_first_and_each_query_is_read_once(snapshot: BlackboardSnapshot) -> None:
    reader = FakeExperienceReader(completed_read(_past("exp-same", SAME)))
    recall = LearningRecall(reader)
    learnt = _learnt(snapshot)

    first = await recall.lines(learnt, "kv cache block", limit=5)
    again = await recall.lines(learnt, "block cache kv", limit=5)

    assert first == [
        "This session's learnings:",
        "- [s1, insight] a smaller kv cache block helps decode [s1:1]",
        "Past sessions' learnings:",
        "- [sess-0, same conditions, measured] what exp-same learnt about the kv cache [exp-same]",
    ]
    assert again == first and len(reader.calls) == 1


async def test_an_unavailable_kb_says_so_and_is_read_again_next_time(snapshot: BlackboardSnapshot) -> None:
    reader = FakeExperienceReader(RemoteReadResult("", "unavailable", "", (), ("unregistered schema",)))
    recall = LearningRecall(reader)

    lines = await recall.lines(snapshot, "kv cache", limit=5)
    await recall.lines(snapshot, "kv cache", limit=5)

    assert lines == ["Past sessions' learnings:", "- unavailable (unregistered schema)"]
    assert len(reader.calls) == 2
    assert await LearningRecall().lines(snapshot, "kv cache", limit=5) == []


async def test_one_retrieval_logs_this_sessions_and_past_sessions_learnings_on_one_line(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    path = tmp_path / LEARNINGS_LOG
    reader = FakeExperienceReader(completed_read(_past("exp-same", SAME)))
    recall = LearningRecall(reader, log=RetrievalLog(path))

    await recall.lines(_learnt(snapshot), "kv cache block", limit=5, branch="s2")
    await LearningRecall(log=RetrievalLog(path)).lines(snapshot, "kv cache", limit=5, branch="s3")
    logged = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [(line["branch"], line["source"], line["here"], line["past"], line["past_status"]) for line in logged] == [
        ("s2", "tool", ["s1:1"], ["exp-same"], ReadStatus.COMPLETED.value),
        ("s3", "tool", [], [], ""),
    ]
