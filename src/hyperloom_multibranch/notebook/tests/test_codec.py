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
        request_id="r1",
        kind=peers.TestKind.TEST,
        branch="s1",
        parent_commit="c0",
        intent_fingerprint="i" * 64,
        delta=delta,
        avoid_nodes=("node-3",),
        accuracy_rounds=True,
    )
    result = peers.TestResult(
        request_id="r1",
        validity=peers.Validity.SUCCESS,
        sandbox=peers.SandboxFacts(node="node-1", gpu_ids=(0, 1), power_cap_watts=750.0, backend="local"),
        manifest_sha256="m" * 64,
        rounds=(peers.RoundValues(peers.RoundKind.FORMAL, "output_tps", (1010.5, 1003)),),
        accuracy={"gsm8k": 0.91},
        attempts=(peers.Attempt("node-0", (0, 1), peers.FailureClass.INFRA, 10.0, 20.0),),
        gpu_seconds=1200,
    )
    judgement = peers.Judgement(peers.Verdict.IMPROVED, 4.5, 1.1, True, ("r1", "r2"), {"output_tps": 1.045})
    return records.TestRecord(request, result, judgement, Conclusion.COMMITTED, "c1")


def test_a_test_record_survives_json_with_its_recipe_delta_enums_and_mappings() -> None:
    record = _record()

    stored = json.loads(json.dumps(codec.encode(record)))

    assert codec.decode(records.TestRecord, stored) == record
    assert stored["request"]["delta"]["config"]["extra_server_args"] == "--max-num-seqs 256"
    assert stored["result"]["rounds"][0]["kind"] == "formal"


def test_numbers_come_back_as_the_type_their_field_declares() -> None:
    result = codec.decode(records.TestRecord, json.loads(json.dumps(codec.encode(_record())))).result

    assert result is not None
    assert result.rounds[0].values == (1010.5, 1003.0)
    assert isinstance(result.gpu_seconds, float)


def test_a_field_an_older_row_lacks_takes_its_default() -> None:
    stored = codec.encode(records.LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    del stored["reason"]

    assert codec.decode(records.LineageEvent, stored) == records.LineageEvent(LineageKind.OPEN, "s1", from_commit="c0")


def test_claims_and_lineage_views_round_trip() -> None:
    intent = peers.Intent("s1", "c0", "moe gemm", "tuned kernel", peers.IntentLayer.KERNEL, "f" * 64)
    claim = records.Claim(
        records.ClaimKey("c0", intent.fingerprint), "s1", intent, "a" * 64, 60.0, 1.0, 2.0, 62.0, ClaimStatus.HELD
    )
    view = records.LineageView(
        commits=(records.CommitNode("c1", "c0", "s1", records.CommitOrigin.COMMIT, "r1"),),
        branches=(records.BranchNode("s1", LineageKind.OPEN, "c0", "c1"),),
    )

    assert codec.decode(records.Claim, codec.encode(claim)) == claim
    assert codec.decode(records.LineageView, codec.encode(view)) == view


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
