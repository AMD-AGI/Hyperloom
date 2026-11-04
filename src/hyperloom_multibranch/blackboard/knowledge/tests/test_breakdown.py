# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-branch ``session_breakdown.json``: every blackboard fact, and the write-back outcome."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import BuildCharge, BuildPayment
from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom.orchestrator.knowledge.remote_recipe.models import RemoteWriteResult
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import (
    CardHold,
    LeaseStart,
    LlmUsage,
    LlmUsageRecord,
    BlackboardSnapshot,
    TestRecord,
)
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseEndReason, LeaseKind, UsageSource
from hyperloom_multibranch.blackboard.knowledge.breakdown import (
    EXPORTER_VERSION,
    FILENAME,
    SCHEMA_VERSION,
    CloseOutcome,
    SessionPlace,
    build_breakdown,
    write_breakdown,
)
from hyperloom_multibranch.blackboard.knowledge.recipe_writeback import FinalRecipe

FINAL_RECIPE = Recipe("rocm/vllm@sha256:" + "a" * 64, LaunchConfig(server_args="--tensor-parallel-size 8"))
NO_TOKENS = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cache_write_1h": 0}


def test_the_breakdown_lays_out_the_session_tree_tests_and_measurement(snapshot: BlackboardSnapshot) -> None:
    written = RemoteWriteResult("written", "", "inference:x", "sess-1", "optimized_throughput", 1041.0)

    breakdown = build_breakdown(snapshot, CloseOutcome(kb_write_back=written))

    assert (breakdown["schema_version"], breakdown["exporter_version"]) == (SCHEMA_VERSION, EXPORTER_VERSION)
    assert breakdown["metadata"]["session_profile"]["session_id"] == "sess-1"
    assert [row["activity"] for row in breakdown["metadata"]["manifest_fingerprints"]] == ["act-1"]
    assert breakdown["baseline"]["values"] == {"output_tps": 1000.0}
    assert {node["commit_id"] for node in breakdown["lineage"]["commits"]} == {"c0", "c1", "i1"}
    assert {row["branch"]: row["llm_usd"] for row in breakdown["branches"]} == {
        INTEGRATION_BRANCH: 0.0,
        "s1": 1.25,
        "s2": 0.5,
    }
    assert breakdown["spend"] == {
        "gpu_hours": 1.0,
        "held_card_hours": 0.0,
        "llm_usd": 2.0,
        "llm_unpriced_tokens": NO_TOKENS,
    }
    assert breakdown["card_holds"] == {"leases": [], "card_hours_by_branch": {}, "card_hours_by_kind": {}}
    assert breakdown["environment_changes"] == []
    assert [row["request"]["request_id"] for row in breakdown["tests"]] == ["r1", "r2", "r3", "r5", "r4", "m1"]
    assert [(row["node"], row["tests"], row["gpu_seconds"]) for row in breakdown["measurement_nodes"]] == [
        ("node-1", 5, 3000.0),
        ("node-2", 1, 600.0),
    ]
    assert breakdown["close"]["kb_write_back"]["status"] == "written"
    assert build_breakdown(snapshot)["close"] == {"final": None, "selection": {}, "kb_write_back": None}
    assert breakdown["builds"] == {"by_branch": [], "total": {"paid": 0, "build_seconds": 0, "reused": 0}}


def test_the_breakdown_adds_up_the_native_builds_by_branch_and_component(snapshot: BlackboardSnapshot) -> None:
    aiter = BuildCharge("k1", "aiter", 120.0, BuildPayment.PAID)
    reused = BuildCharge("k1", "aiter", 0.0, BuildPayment.REUSED)
    sglang = BuildCharge("k2", "sglang", 30.0, BuildPayment.PAID)
    charged = {"r1": (aiter, sglang), "r4": (reused,), "r2": (reused,)}

    def built(record: TestRecord) -> TestRecord:
        builds = charged.get(record.request_id)
        if builds is None or record.result is None:
            return record
        return dataclasses.replace(record, result=dataclasses.replace(record.result, builds=builds))

    found = build_breakdown(dataclasses.replace(snapshot, records=tuple(map(built, snapshot.records))))["builds"]

    assert found == {
        "by_branch": [
            {"branch": "s1", "component": "aiter", "paid": 1, "build_seconds": 120.0, "reused": 1},
            {"branch": "s1", "component": "sglang", "paid": 1, "build_seconds": 30.0, "reused": 0},
            {"branch": "s2", "component": "aiter", "paid": 0, "build_seconds": 0.0, "reused": 1},
        ],
        "total": {"paid": 2, "build_seconds": 150.0, "reused": 2},
    }


def test_the_close_section_holds_the_merge_statistics_the_scheduler_passes(snapshot: BlackboardSnapshot) -> None:
    merges = {"conflicts": 3, "rebased": 1, "resolved": 1, "lost": {"unresolved patch conflict": 1}}

    close = build_breakdown(snapshot, CloseOutcome(merges=merges))["close"]

    assert close["merges"] == merges and "merges" not in build_breakdown(snapshot)["close"]


def test_a_hosted_run_adds_the_session_block_and_its_records_under_the_names_the_legacy_flow_uses(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    hosted = {
        "CLAW_SESSION_ID": "claw-7",
        "HYPERLOOM_CODE_REVISION": "not-this-one",
        "HYPERLOOM_IMAGE": "rocm/sglang:x",
    }
    sandbox = {"request_id": "r1", "sandbox_id": "sb-1", "node_name": "node-1", "same_node": True}
    plugs = {"profile": "default", "fingerprint": "sha256:p", "sockets": {"measure": {"impl": "hosted"}}}
    (tmp_path / "hosted").mkdir()
    (tmp_path / "hosted" / "measurement_sandboxes.jsonl").write_text(
        json.dumps(sandbox) + '\n{"request_id": "r', "utf-8"
    )
    (tmp_path / "plugs.resolved.json").write_text(json.dumps(plugs), "utf-8")

    metadata = build_breakdown(snapshot, place=SessionPlace(tmp_path, environ=hosted, code_revision="38724291b"))[
        "metadata"
    ]
    local = build_breakdown(snapshot, place=SessionPlace(environ={}))["metadata"]

    assert metadata["session"] == {
        "session_id": "sess-1",
        "code_revision": "38724291b",
        "claw_session_id": "claw-7",
        "image": "rocm/sglang:x",
        "node_name": None,
    }
    assert (metadata["measurement_sandboxes"], metadata["plugs"]) == ([sandbox], plugs)
    assert not {"specialist_resources", "container_resources", "hosted_measurement"} & set(metadata)
    assert local["session"]["claw_session_id"] is None and "measurement_sandboxes" not in local


def test_the_agent_runners_samples_give_each_specialists_record_and_the_containers_last_reading(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    first, second = ({"kind": "specialist_resources", "task_id": task, "cpu_seconds": 9.5} for task in ("b01", "b02"))
    early, late = ({"kind": "container_resources", "memory_peak_bytes": peak, "oom_kill": 0} for peak in (512, 2048))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "resources.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (first, early, second, late)) + '{"kind": "spec', "utf-8"
    )

    metadata = build_breakdown(snapshot, place=SessionPlace(runner_dir=tmp_path, environ={}))["metadata"]
    unsampled = build_breakdown(snapshot, place=SessionPlace(runner_dir=tmp_path / "elsewhere", environ={}))["metadata"]

    assert (metadata["specialist_resources"], metadata["container_resources"]) == ([first, second], late)
    assert not {"specialist_resources", "container_resources"} & set(unsampled)


def test_the_spend_counts_the_tokens_of_unpriced_llm_calls_beside_the_dollars(snapshot: BlackboardSnapshot) -> None:
    tokens = TokenCounts(input=1_000, output=500, cache_read=20_000, cache_write_1h=300)
    unpriced = LlmUsageRecord("u1", snapshot.taken_at, LlmUsage("", UsageSource.JUDGE, None, tokens))

    spend = build_breakdown(dataclasses.replace(snapshot, llm_usage=(*snapshot.llm_usage, unpriced)))["spend"]

    assert spend["llm_usd"] == 2.0
    assert spend["llm_unpriced_tokens"] == {
        **NO_TOKENS,
        "input": 1_000,
        "output": 500,
        "cache_read": 20_000,
        "cache_write_1h": 300,
    }


def test_the_card_holds_count_each_lease_to_the_snapshot_and_total_them_by_branch_and_kind(
    snapshot: BlackboardSnapshot,
) -> None:
    start = snapshot.taken_at - 3600.0
    arbor = LeaseStart("lease-1", "s1", "s1", LeaseKind.AUTONOMOUS, 2, start, gpu_ids=(4, 5))
    forge = LeaseStart("lease-2", "tool:job-7", "s1", LeaseKind.TOOL_JOB, 1, start, tool="forge", sandbox_id="sb-1")
    # Card 4 went back after half an hour and card 5 is still out; the Forge job's one card went back after 15 min.
    holds = (CardHold(arbor, {4: start + 1800.0}), CardHold(forge, {0: start + 900.0}, LeaseEndReason.RELEASED))

    breakdown = build_breakdown(dataclasses.replace(snapshot, card_holds=holds))

    section = breakdown["card_holds"]
    assert [lease["card_hours"] for lease in section["leases"]] == [pytest.approx(1.5), pytest.approx(0.25)]
    assert section["leases"][0]["returned"] == {"4": start + 1800.0} and section["leases"][0]["end_reason"] is None
    assert section["leases"][1]["lease"]["tool"] == "forge" and section["leases"][1]["end_reason"] == "released"
    assert section["card_hours_by_branch"] == {"s1": pytest.approx(1.75)}
    assert section["card_hours_by_kind"] == {"autonomous": pytest.approx(1.5), "tool_job": pytest.approx(0.25)}
    assert breakdown["spend"]["held_card_hours"] == pytest.approx(1.75)
    assert json.loads(json.dumps(breakdown))["card_holds"]["leases"][1]["lease"]["kind"] == "tool_job"


def test_the_close_section_names_the_commit_final_selection_chose_and_how(snapshot: BlackboardSnapshot) -> None:
    final = FinalRecipe("i1", FINAL_RECIPE, Path("files"), "output_tps")
    selection = {"candidates": ["i1", "c1"], "chosen": "i1", "reason": "the integration head held its gain"}

    close = build_breakdown(snapshot, CloseOutcome(final=final, selection=selection))["close"]

    assert close["final"] == {
        "commit_id": "i1",
        "recipe_digest": FINAL_RECIPE.digest(),
        "recipe": FINAL_RECIPE.to_document(),
    }
    assert close["selection"] == selection


def test_the_breakdown_is_written_as_sorted_indented_json(snapshot: BlackboardSnapshot, tmp_path: Path) -> None:
    breakdown = build_breakdown(snapshot)

    path = write_breakdown(tmp_path / "export", breakdown)

    assert path == tmp_path / "export" / FILENAME
    assert json.loads(path.read_text(encoding="utf-8")) == breakdown
    assert path.read_text(encoding="utf-8").startswith('{\n  "agent_manifests"')
