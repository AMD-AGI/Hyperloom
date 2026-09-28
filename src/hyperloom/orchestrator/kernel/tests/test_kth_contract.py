# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom's KTH digests must equal KTH's own, byte for byte.

The golden values were produced by KTH at
49154fb80508bb37d008f87f8bfae8759209742f (``KernelCandidateEnvelope.digest``,
``qualification_contract.subject_digest`` and ``ExactBinding.subject_digest``)
from the inputs below.
"""

from __future__ import annotations

import base64
import json

import pytest

from hyperloom.orchestrator.kernel import kth_contract as contract

_PATCH = b"diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"
_PLAN = json.loads(
    '{"allow_fake_collectives": false, "allowed_paths": [], "bind_source": false, "budget": 2,'
    ' "candidate_registry_key": "AttentionControl", "expected_base": "", "kernel_path": "kernels/attention.py",'
    ' "mandatory_oracle_ids": ["SHAPE", "FINITE", "REQUIRED_OUTPUT", "OUTPUT_WRITTEN", "OUTPUT_GUARDS",'
    ' "INPUT_IMMUTABLE", "PROVENANCE", "REF", "NUMERICAL_POLICY", "ATTESTATION"], "operation_spec": "attention_mla",'
    ' "operation_spec_version": "2.0-poc", "oracle_versions": ["1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0"],'
    ' "plan_id": "fixture/cpu-attention-control-v1", "plan_version": "1", "reference_registry_key": "TrustedReference"}'
)
#: Every key KTH's ``KernelCandidateEnvelope.to_dict`` emits.
_KTH_NORMALIZED_KEYS = {
    "schema_version",
    "candidate_id",
    "attempt_id",
    "base_repository",
    "base_commit",
    "patch_digest",
    "patch_path",
    "changed_paths",
    "changed_symbols",
    "build",
    "artifact",
    "environment",
    "operation_schema",
    "tensors",
    "dispatch_paths",
    "call_sites",
    "tensor_lineage",
    "state_flow",
    "graph",
    "implementation",
    "tracelens",
    "geak_harness",
    "hyperloom_context",
    "fallback",
    "model_config",
    "field_provenance",
}


def _envelope() -> dict:
    artifact = contract.CandidateArtifact(
        candidate_id="kernel:forge-loop:k:standalone:unknown:triton:mi355x",
        attempt_id="hyperloom-0123",
        base_repository="/repo",
        base_commit="c" * 40,
        patch=_PATCH,
        changed_paths=("k.py",),
    )
    return contract.build_candidate_envelope(
        artifact,
        environment={
            "hardware": "mi355x",
            "software": "standalone==unknown triton",
            "producer": "hyperloom_context",
            "trust_level": "hypothesized",
            "provenance": "controller_publication",
        },
        geak_harness={
            "harness_id": "forge-loop",
            "observations": {"micro_validated": True, "mean_case_speedup": 1.25},
            "producer": "geak_harness",
            "trust_level": "hypothesized",
            "provenance": "m",
        },
        field_provenance={
            "patch": {
                "producer": "hyperloom_integration",
                "trust_level": "verified",
                "provenance": "git",
                "digest": artifact.patch_digest,
            }
        },
    )


def _binding(envelope: dict) -> dict:
    return {
        "envelope_digest": contract.envelope_digest(envelope),
        "patch_digest": envelope["patch_digest"],
        "base_commit": "c" * 40,
        "binary_digest": "",
        "module_digest": "",
        "loaded_identity": "",
        "build_digest": "",
        "compiler": "",
        "autospec_digest": "sha256:" + "1" * 64,
        "resolved_digest": "sha256:" + "2" * 64,
        "plan_digest": "sha256:" + "3" * 64,
        "kth_revision": "4" * 40,
        "environment_identity": contract.environment_identity(envelope["environment"]),
    }


def test_envelope_digest_matches_kth() -> None:
    assert (
        contract.envelope_digest(_envelope())
        == "sha256:3d3015ba0258592ccf1c2822065343ae63537523ae004863bc5c2d56be825e38"
    )


def test_reviewed_subject_digest_matches_kth() -> None:
    digest = contract.reviewed_subject_digest(
        base_commit="d" * 40, patch=_PATCH, kernel_path="kernels/attention.py", qualification_plan=_PLAN
    )
    assert digest == "sha256:65030c1538770b129d44e2e7b02b9b2fb92fbf806862d59ee459e074068bf089"


def test_adaptive_subject_digest_matches_kth() -> None:
    assert (
        contract.adaptive_subject_digest(_binding(_envelope()))
        == "sha256:9c63cbc733c56338520f977da7d21c97535bae0b21aad03fd932525076dd0b4c"
    )


def test_envelope_is_already_in_kth_normalized_form() -> None:
    envelope = _envelope()
    assert set(envelope) - {"patch_base64"} == _KTH_NORMALIZED_KEYS
    assert base64.b64decode(envelope["patch_base64"]) == _PATCH
    assert envelope["patch_digest"] == contract.sha256_digest(_PATCH)
    round_tripped = json.loads(json.dumps(envelope))
    assert contract.envelope_digest(round_tripped) == contract.envelope_digest(envelope)


def test_envelope_digest_ignores_key_order_but_not_content() -> None:
    envelope = _envelope()
    reordered = dict(reversed(list(envelope.items())))
    assert contract.envelope_digest(reordered) == contract.envelope_digest(envelope)
    tampered = {**envelope, "changed_paths": ["other.py"]}
    assert contract.envelope_digest(tampered) != contract.envelope_digest(envelope)


def test_one_patch_byte_changes_every_subject_digest() -> None:
    tampered = _PATCH.replace(b"X = 2", b"X = 3")
    original = contract.reviewed_subject_digest(
        base_commit="d" * 40, patch=_PATCH, kernel_path="kernels/attention.py", qualification_plan=_PLAN
    )
    assert original != contract.reviewed_subject_digest(
        base_commit="d" * 40, patch=tampered, kernel_path="kernels/attention.py", qualification_plan=_PLAN
    )
    assert original != contract.reviewed_subject_digest(
        base_commit="d" * 40,
        patch=_PATCH,
        kernel_path="kernels/attention.py",
        qualification_plan={**_PLAN, "budget": 1},
    )


@pytest.mark.parametrize("field", contract.ADAPTIVE_BINDING_KEYS)
def test_every_adaptive_binding_field_is_bound(field: str) -> None:
    binding = _binding(_envelope())
    assert contract.adaptive_subject_digest({**binding, field: "sha256:" + "f" * 64}) != (
        contract.adaptive_subject_digest(binding)
    )


def test_candidate_control_fields_are_found_at_any_depth() -> None:
    publication = {
        "operator_id": "kernel:x",
        "manifest": {"changed_files": ["k.py"], "notes": [{"verdict": "pass"}], "RTOL": 0.5},
        "kth_qualification": {"plan_id": "p"},
    }
    assert sorted(contract.candidate_control_fields(publication)) == [
        "kth_qualification",
        "kth_qualification.plan_id",
        "manifest.RTOL",
        "manifest.notes[0].verdict",
    ]
    assert not list(contract.candidate_control_fields({"manifest": {"mean_case_speedup": 1.2}}))
