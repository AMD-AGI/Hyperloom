# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each learning a specialist did not supersede becomes one Experience of the learning schema, with its evidence."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from hyperloom_kb import RemoteExperienceKB
from hyperloom_kb.collect import load_mapping

from hyperloom_multibranch.contracts.blackboard import NotRecorded
from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import Learning, LearningRecord, BlackboardSnapshot
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind
from hyperloom_multibranch.blackboard.knowledge.experience import MAPPING_PATH as EXPERIENCE_MAPPING
from hyperloom_multibranch.blackboard.knowledge.learnings import (
    EXPORT_FORMAT,
    MAPPING_PATH,
    export_document,
    publish_learnings,
)
from hyperloom_multibranch.blackboard.knowledge.tests.kb_service import experience_service
from hyperloom_multibranch.blackboard.learning_search import current_learnings
from hyperloom_multibranch.blackboard.tests.factories import START

GEMM = Learning(
    "s1",
    LearningKind.INSIGHT,
    "the MoE GEMM is bound by memory bandwidth at conc 64",
    "moe gemm",
    "tuned kernel",
    IntentLayer.KERNEL,
    ("r1",),
)
WARMUP = Learning("s2", LearningKind.PITFALL, "chunked prefill crashes the server during warmup", request_ids=("r3",))
FLAG = Learning("s2", LearningKind.TOOLING, "vLLM 0.11 ignores VLLM_USE_V1=0 on this image")
OLD = Learning("s1", LearningKind.INSIGHT, "attention dominates decode at this concurrency")


def _with_learnings(snapshot: BlackboardSnapshot) -> BlackboardSnapshot:
    records = (
        LearningRecord("s1:1", START + 10, OLD),
        LearningRecord("s1:2", START + 20, GEMM),
        LearningRecord("s2:1", START + 30, WARMUP),
        LearningRecord("s2:2", START + 40, FLAG),
        LearningRecord(
            "s1:3",
            START + 50,
            dataclasses.replace(OLD, content="gemm, not attention, dominates decode", supersedes="s1:1"),
        ),
    )
    return dataclasses.replace(snapshot, learnings=records)


def test_a_superseded_learning_is_not_current(snapshot: BlackboardSnapshot) -> None:
    assert [record.entry_id for record in current_learnings(_with_learnings(snapshot))] == [
        "s1:2",
        "s2:1",
        "s2:2",
        "s1:3",
    ]


def test_each_current_learning_is_one_row_with_the_sessions_conditions_and_its_tests(
    snapshot: BlackboardSnapshot,
) -> None:
    document = export_document(_with_learnings(snapshot))
    rows = {row["entry_id"]: row for row in document["learnings"]}

    assert (document["format"], document["metadata"]) == (EXPORT_FORMAT, {"session_id": "sess-1"})
    gemm = rows["s1:2"]
    assert (gemm["kind"], gemm["target"], gemm["mechanism"], gemm["measured"]) == (
        "insight",
        "moe gemm",
        "tuned kernel",
        True,
    )
    assert gemm["evidence"] == ["r1 on c0: improved +4.00%, committed"]
    assert gemm["identity"]["gpu"] == "MI355X" and gemm["identity"]["tp"] == 8 and gemm["identity"]["layer"] == "kernel"
    assert rows["s2:1"]["measured"] is True
    assert rows["s2:1"]["evidence"] == ["r3 on c0: the candidate failed (server exited during warmup)"]
    assert (rows["s2:2"]["measured"], rows["s2:2"]["evidence"], "layer" in rows["s2:2"]["identity"]) == (
        False,
        [],
        False,
    )


def test_the_learnings_become_experiences_of_their_own_schema(snapshot: BlackboardSnapshot) -> None:
    report = publish_learnings(_with_learnings(snapshot), dry_run=True)
    collected = {item.unit_id: item.experience for item in report.collected}

    assert not report.skipped
    assert sorted(collected) == ["s1:2", "s1:3", "s2:1", "s2:2"]
    gemm = collected["s1:2"]
    assert gemm.schema_ref == load_mapping(MAPPING_PATH).declaration.schema_ref
    assert (gemm.objective, gemm.change["summary"], gemm.change["kind"]) == (
        "specialist_learning@v1",
        "the MoE GEMM is bound by memory bandwidth at conc 64",
        "insight",
    )
    assert (gemm.outcome["decision"], list(gemm.outcome["evidence"])) == (
        "measured",
        ["r1 on c0: improved +4.00%, committed"],
    )
    assert (gemm.identity["model"], gemm.identity["layer"]) == ("deepseek-ai/DeepSeek-R1", "kernel")
    assert gemm.provenance.source_ref == "multibranch:sess-1:s1:2"
    assert collected["s2:2"].outcome["decision"] == "unmeasured"
    assert publish_learnings(_with_learnings(snapshot), dry_run=True).collected == report.collected


def test_without_a_profile_there_is_nothing_to_export(snapshot: BlackboardSnapshot) -> None:
    with pytest.raises(NotRecorded):
        export_document(dataclasses.replace(_with_learnings(snapshot), profile=None))


def test_a_service_started_on_the_recipe_schema_registers_the_learning_schema_on_the_first_write(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    recipes = load_mapping(EXPERIENCE_MAPPING).declaration
    learnings = load_mapping(MAPPING_PATH).declaration
    with experience_service(tmp_path, recipes) as client:
        first = publish_learnings(_with_learnings(snapshot), kb=RemoteExperienceKB(client, learnings))
        again = publish_learnings(_with_learnings(snapshot), kb=RemoteExperienceKB(client, learnings))
        listed = client.list_experiences(schema_ref=learnings.schema_ref)
        recipe_listed = client.list_experiences(schema_ref=recipes.schema_ref)

    assert [row.status for row in first.collected] == ["created"] * 4
    assert [row.status for row in again.collected] == ["unchanged"] * 4
    assert {item["experience_id"] for item in listed.items} == {row.experience_id for row in first.collected}
    assert recipe_listed.items == ()
