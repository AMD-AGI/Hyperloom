# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each finished specialist test becomes one Experience, written once, with its conditions and its evidence."""

from __future__ import annotations

import dataclasses
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from hyperloom_kb import (
    Experience,
    ExperienceHTTPService,
    HTTPServiceConfig,
    LLMQueryPlanner,
    PlannerConfiguration,
    RemoteClient,
    RemoteConfig,
    RemoteExperienceKB,
    create_http_server,
)
from hyperloom_kb.collect import collect, load_mapping
from hyperloom_kb.database import open_database

from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    PatchRef,
    RecipeDelta,
)
from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion
from hyperloom_multibranch.notebook.knowledge.experience import (
    EXPORT_FORMAT,
    MAPPING_PATH,
    candidate_row,
    change_document,
    export_document,
    publish_experiences,
)
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

NO_REASONING = "decision reasoning is missing or non-specific"
TOKEN = "multibranch-test-token"


def _rows(snapshot: NotebookSnapshot) -> dict[str, dict[str, Any]]:
    document = export_document(snapshot, METRIC)
    assert document["format"] == EXPORT_FORMAT
    return {row["request_id"]: row for row in document["candidates"]}


def _collected(snapshot: NotebookSnapshot) -> dict[str, Any]:
    return {item.unit_id: item.experience for item in publish_experiences(snapshot, METRIC, dry_run=True).collected}


def _change_group(experience: Experience) -> dict[str, Any]:
    return {name: experience.change.get(name) for name in ("change_family", "change_fingerprint")}


def test_every_finished_specialist_test_is_one_row_with_its_derived_decision(snapshot: NotebookSnapshot) -> None:
    rows = _rows(snapshot)

    assert sorted(rows) == ["r1", "r2", "r3", "r4", "r5"]
    kept = rows["r1"]
    assert (kept["decision"], kept["baseline_value"], kept["outcome_value"], kept["commit_id"]) == (
        "keep",
        1000.0,
        1040.0,
        "c1",
    )
    assert kept["identity"]["gpu"] == "MI355X"
    assert (kept["identity"]["bottleneck"], kept["identity"]["mechanism"], kept["identity"]["layer"]) == (
        "moe gemm",
        "tuned kernel",
        "kernel",
    )
    assert kept["reasoning"] == "the MoE GEMM is bound by memory bandwidth"
    assert kept["learn"] == "confirmed: the MoE GEMM is bound by memory bandwidth"
    assert kept["interval"] == {"mean": 4.0, "low": 3.0, "high": 5.0}
    assert rows["r4"]["baseline_value"] == 1040.0
    assert (rows["r3"]["decision"], rows["r3"]["outcome_value"]) == ("failed", None)
    assert rows["r3"]["error_class"] == "server exited during warmup"
    assert rows["r5"]["accuracy"] == {"required": True, "passed": False, "value": None}
    assert rows["r2"]["decision"] == "revert"


def test_the_rows_with_reasoning_become_experiences_and_the_rest_say_why_not(snapshot: NotebookSnapshot) -> None:
    report = publish_experiences(snapshot, METRIC, dry_run=True)
    collected = {item.unit_id: item.experience for item in report.collected}

    assert {item.unit_id: item.reason for item in report.skipped} == {
        "r3": NO_REASONING,
        "r4": NO_REASONING,
        "r5": NO_REASONING,
    }
    kept = collected["r1"]
    assert (kept.outcome["decision"], kept.baseline["value"], kept.outcome["value"]) == ("keep", 1000.0, 1040.0)
    assert kept.rationale["reasoning"] == "the MoE GEMM is bound by memory bandwidth"
    assert kept.provenance.producer == "hyperloom-multibranch"
    assert kept.provenance.source_ref == "multibranch:sess-1:r1"
    assert kept.provenance.extra["tests"] == ["r1"]
    assert kept.reflection["text"].startswith("confirmed: the MoE GEMM is bound by memory bandwidth (measured +4.00%")
    assert collected["r2"].outcome["decision"] == "revert"


def test_a_candidate_that_broke_the_server_is_a_failed_experience(snapshot: NotebookSnapshot) -> None:
    why = BranchLogEntry("s2", BranchLogType.ATTEMPT, "chunked prefill should overlap prefill with decode", ("r3",))
    logs = {**snapshot.branch_logs, "s2": (*snapshot.branch_logs["s2"], BranchLogRecord("e9", 0.0, why))}

    failed = _collected(dataclasses.replace(snapshot, branch_logs=logs))["r3"]

    assert (failed.outcome["decision"], failed.outcome.get("value")) == ("failed", None)
    assert failed.outcome["error_class"] == "server_exited_during_warmup"
    assert failed.baseline["value"] == 1000.0


def test_publishing_again_yields_the_same_experiences(snapshot: NotebookSnapshot) -> None:
    assert _collected(snapshot) == _collected(snapshot)


def test_a_config_change_gets_the_same_fingerprint_from_either_flow(snapshot: NotebookSnapshot) -> None:
    attempt = {
        "attempt_id": "a-1",
        "arm": "config",
        "outcome": "KEEP",
        "ts": "2027-01-15T08:10:00Z",
        "measurement": {"before_tput": 1000.0, "after_tput": 1040.0, "gain_pct": 4.0},
        "measured_against": {"extra_server_args": "", "extra_envs": {}, "args_mode": "append"},
        "config_delta": {"extra_server_args": "--max-num-seqs 256", "extra_envs": {}, "args_mode": "append"},
        "reasoning": "the MoE GEMM is bound by memory bandwidth at this concurrency",
        "reasoning_origin": "action_payload.reasoning",
    }
    task = {
        "model_name": "deepseek-ai/DeepSeek-R1",
        "gpu_type": "MI355X",
        "framework_name": "vllm",
        "architecture": {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]},
        "framework_version": "0.11.0",
        "precision": "fp8",
    }
    sbd = {
        "metadata": {
            "grading": {"objective": "output_throughput"},
            "session": {"session_id": "sbd"},
            "task_config": task,
        },
        "timeline": [{"type": "framework_agent", "id": "ev-1", "ext": {"proposals": [], "attempts": [attempt]}}],
    }

    (from_sbd,) = collect("hyperloom-sbd-v6", sbd, dry_run=True).collected
    ours = _collected(snapshot)["r1"]

    assert _change_group(ours) == _change_group(from_sbd.experience)
    assert ours.schema_ref == from_sbd.experience.schema_ref


def test_a_delta_is_a_config_change_a_source_patch_or_a_kernel() -> None:
    patch = RecipeDelta(
        components=(ComponentChange("vllm", add_patches=(PatchRef("patch/vllm/0001-moe.patch", "c" * 64),)),)
    )
    kernel = RecipeDelta(add_kernels=(KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/moe.json", "d" * 64),))
    config = RecipeDelta(
        config=ConfigDelta(server_args="--a 1", remove_args=("--b",), envs={"X": "1"}, unset_envs=("Y",))
    )

    assert (change_document(patch)["family"], change_document(patch)["summary"]) == (
        "patch",
        "patch/vllm/0001-moe.patch",
    )
    assert change_document(kernel)["family"] == "kernel"
    assert change_document(config)["summary"] == "--a 1 X=1 drop --b unset Y"


def test_a_judgement_waiting_on_its_split_waits_to_be_published(snapshot: NotebookSnapshot) -> None:
    (record,) = [record for record in snapshot.records if record.request_id == "r1"]
    pending = dataclasses.replace(record, conclusion=Conclusion.SPLIT_REQUESTED, commit_id=None)

    assert snapshot.profile is not None
    assert candidate_row(pending, snapshot, snapshot.profile, METRIC) is None


def test_without_a_profile_there_is_nothing_to_export_and_agentx_publishes_nothing(snapshot: NotebookSnapshot) -> None:
    agentx = publish_experiences(
        dataclasses.replace(snapshot, profile=profile(isl=None, osl=None)), METRIC, dry_run=True
    )

    with pytest.raises(NotRecorded):
        export_document(dataclasses.replace(snapshot, profile=None), METRIC)
    assert agentx.collected == ()
    assert {item.reason for item in agentx.skipped} == {"session_not_measured_on_output_throughput"}


class _UnusedPlanner:
    model = "unused-planner"

    def complete(self, **_kwargs: Any) -> str:
        raise AssertionError("publishing never plans a read")


@pytest.mark.skipif(sys.platform == "win32", reason="the Experience KB's local store locks files with fcntl")
def test_the_experience_service_stores_each_experience_once_with_its_identity(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    declaration = load_mapping(MAPPING_PATH).declaration
    planner = LLMQueryPlanner(_UnusedPlanner(), PlannerConfiguration.create("unused-planner"))
    home = tmp_path / "service"
    server = create_http_server(
        ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), declaration, planner, database=open_database(home)),
        "127.0.0.1",
        0,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = RemoteClient(RemoteConfig(f"http://{host!s}:{port}", TOKEN, spool_root=tmp_path / "spool"))
        kb = RemoteExperienceKB(client, declaration)
        first = publish_experiences(snapshot, METRIC, kb=kb, receipt=tmp_path / "receipt.json")
        again = publish_experiences(snapshot, METRIC, kb=kb)
        listed = client.list_experiences()
        exported = client.export_page()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert [row.status for row in first.collected] == ["created", "created"]
    assert [row.status for row in again.collected] == ["unchanged", "unchanged"]
    assert {item["experience_id"] for item in listed.items} == {row.experience_id for row in first.collected}
    identities = {item["experience"]["id"]: item["experience"]["identity"] for item in exported.items}
    assert identities[first.collected[0].experience_id]["bottleneck"] == "moe gemm"
    assert (tmp_path / "receipt.json").is_file()
