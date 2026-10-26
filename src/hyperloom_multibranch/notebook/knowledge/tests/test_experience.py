# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each finished specialist test becomes one Experience, written once, with its conditions and its evidence."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest
from hyperloom_kb import Experience, RemoteExperienceKB
from hyperloom_kb.collect import collect, load_mapping

from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    PatchRef,
    RecipeDelta,
)
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict
from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import BranchLogEntry, BranchLogType, TestKind, Validity
from hyperloom_multibranch.contracts.notebook.records import (
    BranchLogRecord,
    Learning,
    LearningRecord,
    NotebookSnapshot,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LearningKind
from hyperloom_multibranch.notebook.knowledge.experience import (
    EXPORT_FORMAT,
    MAPPING_PATH,
    MULTIBRANCH_MAPPING_PATH,
    candidate_row,
    change_document,
    export_document,
    publish_experiences,
    publish_multibranch_experiences,
)
from hyperloom_multibranch.notebook.knowledge.tests.kb_service import experience_service
from hyperloom_multibranch.notebook.tests.factories import profile, result
from hyperloom_multibranch.notebook.tests.sessions import METRIC

NO_REASONING = "decision reasoning is missing or non-specific"


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


def test_the_measured_baseline_is_the_parent_recipe_once_its_digest_is_recorded(snapshot: NotebookSnapshot) -> None:
    digest = "sha256:" + "1" * 64
    commits = tuple(
        dataclasses.replace(commit, recipe_digest=digest) if commit.commit_id == "c0" else commit
        for commit in snapshot.lineage.commits
    )
    named = dataclasses.replace(snapshot, lineage=dataclasses.replace(snapshot.lineage, commits=commits))

    assert (_rows(snapshot)["r1"]["parent_state"], _rows(named)["r1"]["parent_state"]) == ("c0", digest)
    assert _rows(named)["r4"]["parent_state"] == "c1"


def test_a_candidate_that_broke_the_server_is_a_failed_experience(snapshot: NotebookSnapshot) -> None:
    why = BranchLogEntry("s2", BranchLogType.ATTEMPT, "chunked prefill should overlap prefill with decode", ("r3",))
    logs = {**snapshot.branch_logs, "s2": (*snapshot.branch_logs["s2"], BranchLogRecord("e9", 0.0, why))}

    failed = _collected(dataclasses.replace(snapshot, branch_logs=logs))["r3"]

    assert (failed.outcome["decision"], failed.outcome.get("value")) == ("failed", None)
    assert failed.outcome["error_class"] == "server_exited_during_warmup"
    assert failed.baseline["value"] == 1000.0


def test_a_learning_citing_the_test_is_its_reflection_unless_a_later_one_superseded_it(
    snapshot: NotebookSnapshot,
) -> None:
    first = Learning("s1", LearningKind.INSIGHT, "the MoE GEMM tuning pays at conc 64, not at 256", request_ids=("r1",))
    later = dataclasses.replace(first, content="the MoE GEMM tuning pays at every concurrency tried", supersedes="s1:1")
    other_branch = Learning(
        "s2", LearningKind.INSIGHT, "split kv lost 2% when the cache was not the bottleneck", request_ids=("r1",)
    )
    learnt = dataclasses.replace(
        snapshot,
        learnings=(
            LearningRecord("s1:1", 1.0, first),
            LearningRecord("s1:2", 2.0, later),
            LearningRecord("s2:1", 3.0, other_branch),
        ),
    )

    row = _rows(learnt)["r1"]
    kept = _collected(learnt)["r1"]

    assert (row["learn"], row["learn_source"]) == (later.content, "specialist_learning")
    assert _rows(snapshot)["r1"]["learn_source"] == "specialist_branch_log"
    assert kept.reflection["text"].startswith("the MoE GEMM tuning pays at every concurrency tried (measured +4.00%")
    assert kept.provenance.extra["reflection_source"] == "specialist_learning"


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
    assert change_document(kernel)["content"]["add_kernels"][0]["ref"] == "kernel/gemm/moe.json"
    assert change_document(config)["summary"] == "--a 1 X=1 drop --b unset Y"


def _with_delta(snapshot: NotebookSnapshot, request_id: str, delta: RecipeDelta) -> NotebookSnapshot:
    records = tuple(
        dataclasses.replace(record, request=dataclasses.replace(record.request, delta=delta))
        if record.request_id == request_id
        else record
        for record in snapshot.records
    )
    return dataclasses.replace(snapshot, records=records)


def test_a_kernel_change_goes_to_the_notebooks_own_declaration_and_the_rest_stays_shared(
    snapshot: NotebookSnapshot,
) -> None:
    artifact = KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/moe.json", "d" * 64)
    kernel = _with_delta(snapshot, "r1", RecipeDelta(add_kernels=(artifact,)))

    shared = publish_experiences(kernel, METRIC, dry_run=True)
    own = publish_multibranch_experiences(kernel, METRIC, dry_run=True)

    (recorded,) = own.collected
    assert recorded.unit_id == "r1"
    assert (recorded.experience.objective, recorded.experience.change["change_family"]) == (
        "e2e_throughput@v1",
        "kernel_artifact",
    )
    assert "kernel/gemm/moe.json" in recorded.experience.change["content"]
    assert recorded.experience.schema_ref == load_mapping(MULTIBRANCH_MAPPING_PATH).declaration.schema_ref
    assert {item.unit_id: item.reason for item in shared.skipped}["r1"] == (
        "change family is not one inference-recipe-v1 declares"
    )
    assert {item.unit_id: item.reason for item in own.skipped}[
        "r2"
    ] == "the shared inference-recipe-v1 holds this candidate"


def test_an_agentx_session_records_every_change_on_interactivity_in_the_same_repeat_groups(
    snapshot: NotebookSnapshot,
) -> None:
    agentx = dataclasses.replace(snapshot, profile=profile(isl=None, osl=None))
    shared = {item.unit_id: item.experience for item in publish_experiences(snapshot, METRIC, dry_run=True).collected}

    own = {
        item.unit_id: item.experience
        for item in publish_multibranch_experiences(agentx, METRIC, dry_run=True).collected
    }

    assert sorted(own) == sorted(shared) == ["r1", "r2"]
    assert {experience.objective for experience in own.values()} == {"e2e_interactivity@v1"}
    assert [_change_group(own[unit]) for unit in ("r1", "r2")] == [_change_group(shared[unit]) for unit in ("r1", "r2")]


def test_a_judgement_waiting_on_its_split_or_its_diagnosis_waits_to_be_published(snapshot: NotebookSnapshot) -> None:
    (record,) = [record for record in snapshot.records if record.request_id == "r1"]
    pending = [
        dataclasses.replace(record, conclusion=waiting, commit_id=None)
        for waiting in (Conclusion.SPLIT_REQUESTED, Conclusion.DIAGNOSIS_PENDING)
    ]

    assert snapshot.profile is not None
    assert [candidate_row(item, snapshot, snapshot.profile, METRIC) for item in pending] == [None, None]


def test_a_test_whose_measurement_failed_or_did_not_re_verify_is_never_published(snapshot: NotebookSnapshot) -> None:
    (record,) = [record for record in snapshot.records if record.request_id == "r1"]
    unmeasured = dataclasses.replace(record, result=result("r1", validity=Validity.MEASUREMENT_FAILED))
    unverified = dataclasses.replace(record, conclusion=Conclusion.UNVERIFIED, commit_id=None)

    assert snapshot.profile is not None
    assert candidate_row(unmeasured, snapshot, snapshot.profile, METRIC) is None
    assert candidate_row(unverified, snapshot, snapshot.profile, METRIC) is None


@pytest.mark.parametrize(
    "examined",
    [
        {"kind": TestKind.PROFILE},
        {"kind": TestKind.SWEEP},
        {"kind": TestKind.ACCURACY, "accuracy": True},
        {"kind": TestKind.REPRO, "attachments": (AttachmentRef("repro.sh", "file", "c" * 64),)},
    ],
)
def test_only_a_test_of_kind_test_is_published_as_the_other_kinds_examine_a_state(
    snapshot: NotebookSnapshot, examined: dict[str, object]
) -> None:
    (record,) = [record for record in snapshot.records if record.request_id == "r3"]
    other = dataclasses.replace(record, request=dataclasses.replace(record.request, **examined))

    assert snapshot.profile is not None
    assert candidate_row(record, snapshot, snapshot.profile, METRIC) is not None
    assert candidate_row(other, snapshot, snapshot.profile, METRIC) is None


def test_without_a_profile_there_is_nothing_to_export_and_agentx_publishes_nothing(snapshot: NotebookSnapshot) -> None:
    agentx = publish_experiences(
        dataclasses.replace(snapshot, profile=profile(isl=None, osl=None)), METRIC, dry_run=True
    )

    with pytest.raises(NotRecorded):
        export_document(dataclasses.replace(snapshot, profile=None), METRIC)
    assert agentx.collected == ()
    assert {item.reason for item in agentx.skipped} == {"session_not_measured_on_output_throughput"}


@pytest.mark.skipif(sys.platform == "win32", reason="the Experience KB's local store locks files with fcntl")
def test_the_experience_service_stores_each_experience_once_with_its_identity(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    declaration = load_mapping(MAPPING_PATH).declaration
    with experience_service(tmp_path, declaration) as client:
        kb = RemoteExperienceKB(client, declaration)
        first = publish_experiences(snapshot, METRIC, kb=kb, receipt=tmp_path / "receipt.json")
        again = publish_experiences(snapshot, METRIC, kb=kb)
        listed = client.list_experiences()
        exported = client.export_page()

    assert [row.status for row in first.collected] == ["created", "created"]
    assert [row.status for row in again.collected] == ["unchanged", "unchanged"]
    assert {item["experience_id"] for item in listed.items} == {row.experience_id for row in first.collected}
    identities = {item["experience"]["id"]: item["experience"]["identity"] for item in exported.items}
    assert identities[first.collected[0].experience_id]["bottleneck"] == "moe gemm"
    assert (tmp_path / "receipt.json").is_file()


def test_a_test_whose_eval_did_not_run_exports_its_accuracy_as_not_passed(snapshot: NotebookSnapshot) -> None:
    def unmeasured(record: TestRecord) -> TestRecord:
        if record.request.request_id != "r1":
            return record
        judged = dataclasses.replace(record.judgement, accuracy=AccuracyVerdict.NOT_MEASURED)
        return dataclasses.replace(record, request=dataclasses.replace(record.request, accuracy=True), judgement=judged)

    rows = _rows(dataclasses.replace(snapshot, records=tuple(map(unmeasured, snapshot.records))))

    assert rows["r1"]["accuracy"] == {"required": True, "passed": False, "value": None}


def test_a_blocked_gain_sent_for_repair_exports_the_accuracy_it_failed() -> None:
    import dataclasses

    from hyperloom_multibranch.contracts.notebook.peers import AccuracyValue, Verdict
    from hyperloom_multibranch.contracts.notebook.vocab import Conclusion
    from hyperloom_multibranch.notebook.knowledge import experience
    from hyperloom_multibranch.notebook.tests.factories import intent, judgement, request, result

    blocked = dataclasses.replace(
        judgement(5.0, ("r9", "r9b"), verdict=Verdict.BLOCKED), accuracy=AccuracyVerdict.FAILED
    )
    measured = dataclasses.replace(result("r9"), accuracy=(AccuracyValue("gsm8k", "exact_match", 0.61),))
    record = TestRecord(request("r9", intent("s1", "c0")), measured, blocked, Conclusion.REPAIR_REQUESTED)

    assert experience._accuracy(record) == {"required": True, "passed": False, "value": 0.61}
