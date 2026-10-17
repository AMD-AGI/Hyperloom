# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from hyperloom.contracts.recipe import ComponentChange, ConfigDelta, PatchRef, RecipeDelta
from hyperloom_multibranch.contracts.notebook import peers, records
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, Conclusion, LineageKind
from hyperloom_multibranch.notebook import codec


def _record() -> records.TestRecord:
    delta = RecipeDelta(
        config=ConfigDelta(server_args="--max-num-seqs 256", envs={"VLLM_ROCM_USE_AITER": "1"}),
        components=(ComponentChange("vllm", add_patches=(PatchRef("patch/vllm/0001.patch", "a" * 64),)),),
    )
    request = peers.TestRequest(
        kind=peers.TestKind.TEST,
        purpose=peers.Purpose.EXPLORE,
        branch_id="s1",
        parent_commit="c0",
        delta=delta,
        intent_fingerprint="i" * 64,
        avoid_nodes=("node-3",),
        accuracy=True,
        request_id="r1",
    )
    lost = peers.SandboxFacts("sandbox-0", node_name="node-0", gpu_ids=(0, 1))
    result = peers.TestResult(
        request_id="r1",
        validity=peers.Validity.SUCCEEDED,
        rounds=(peers.RoundValue("measure", concurrency=64, metrics={"output_tps": 1010.5}),),
        accuracy=(peers.AccuracyValue("gsm8k", "exact_match", 0.91, version="lm_eval=0.4.13;gsm8k=3"),),
        sandbox=peers.SandboxFacts("sandbox-1", node_name="node-1", gpu_ids=(0, 1), power_cap_w=750.0, backend="local"),
        attempts=(peers.Attempt(lost, peers.FailureClass.INFRA, "node lost", 10.0, 20.0),),
        manifest_sha256="a" * 64,
        gpu_seconds=1200.0,
    )
    judgement = peers.Judgement(peers.Verdict.IMPROVED, 4.5, 1.1, ("r1", "r2"), {"output_tps": 1.045}, True)
    return records.TestRecord(request, result, judgement, Conclusion.COMMITTED, "c1")


def test_a_test_record_survives_json_with_its_recipe_delta_enums_and_mappings() -> None:
    record = _record()

    stored = json.loads(json.dumps(codec.encode(record)))

    assert codec.decode(records.TestRecord, stored) == record
    assert stored["request"]["delta"]["config"]["extra_server_args"] == "--max-num-seqs 256"
    assert stored["result"]["rounds"][0]["phase"] == "measure"


def test_numbers_come_back_as_the_type_their_field_declares() -> None:
    intent = peers.Intent("s1", "c0", "moe gemm", "tuned kernel", peers.IntentLayer.KERNEL)
    key = records.ClaimKey("c0", intent.fingerprint)
    claim = records.Claim(key, "s1", intent, "a" * 64, 60, 1, 2, 62, ClaimStatus.HELD)

    decoded = codec.decode(records.Claim, json.loads(json.dumps(codec.encode(claim))))

    assert decoded == claim
    assert all(isinstance(value, float) for value in (decoded.ttl_seconds, decoded.acquired_at, decoded.expires_at))


def test_a_field_an_older_row_lacks_takes_its_default() -> None:
    stored = codec.encode(records.LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    del stored["reason"]

    assert codec.decode(records.LineageEvent, stored) == records.LineageEvent(LineageKind.OPEN, "s1", from_commit="c0")


def test_claims_and_lineage_views_round_trip() -> None:
    intent = peers.Intent("s1", "c0", "moe gemm", "tuned kernel", peers.IntentLayer.KERNEL)
    claim = records.Claim(
        records.ClaimKey("c0", intent.fingerprint), "s1", intent, "a" * 64, 60.0, 1.0, 2.0, 62.0, ClaimStatus.HELD
    )
    view = records.LineageView(
        commits=(records.CommitNode("c1", "c0", "s1", records.CommitOrigin.COMMIT, "r1"),),
        branches=(records.BranchNode("s1", LineageKind.OPEN, "c0", "c1"),),
    )

    assert codec.decode(records.Claim, codec.encode(claim)) == claim
    assert codec.decode(records.LineageView, codec.encode(view)) == view


def test_a_tool_result_pointer_keeps_its_op_shares_and_an_older_one_loads_without_them() -> None:
    pointer = records.ToolResultPointer(
        "s1", "job-7", "/artifacts/analysis.json", "trace analysed", "c1", "r1", {"moe_gemm": 41.5, "attention": 22.0}
    )
    older = {"branch": "s1", "job_id": "job-3", "location": "/artifacts/build.log", "summary": "build done"}

    assert codec.decode(records.ToolResultPointer, json.loads(json.dumps(codec.encode(pointer)))) == pointer
    assert codec.decode(records.ToolResultPointer, older) == records.ToolResultPointer(
        "s1", "job-3", "/artifacts/build.log", "build done"
    )


@pytest.mark.parametrize(
    ("commit_id", "shares"),
    [("c1", {f"op{index}": 1.0 for index in range(records.MAX_OP_SHARES + 1)}), ("", {"moe_gemm": 41.5})],
)
def test_op_shares_are_bounded_and_need_their_commit(commit_id: str, shares: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        records.ToolResultPointer("s1", "job-7", "/artifacts/analysis.json", "trace analysed", commit_id, "", shares)


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        (records.LineageEvent, {"kind": "branch", "branch": "s1"}),
        (records.LineageEvent, {"kind": "open", "branch": 7}),
        (records.QuarantineEvent, {"node": "n", "action": "lift", "reason": "r", "gpu_ids": "0,1"}),
        (records.TestRecord, ["not", "an", "object"]),
    ],
)
def test_data_of_another_shape_is_refused(kind: type, data: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        codec.decode(kind, data)
