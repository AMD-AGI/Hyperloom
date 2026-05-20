"""Envelope schema validation tests (P2 PR-D)."""

from __future__ import annotations

import pytest

from framework_agent.agent.envelope import (
    ENVELOPE_SCHEMAS,
    EnvelopeValidationError,
    validate_envelope,
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
def test_optimize_success_minimum_required_fields():
    envelope = {
        "payload_kind": "OptimizeSuccess",
        "patch_path": "/sess/runs/framework/fw-001/proposal.diff",
        "predicted_gain_pct": 5.0,
        "rationale": "block_manager refactor",
        "stage_a_elapsed_ms": 1234,
    }
    assert validate_envelope(envelope) == "OptimizeSuccess"


def test_optimize_success_with_discovered_flags_and_framework():
    envelope = {
        "payload_kind": "OptimizeSuccess",
        "patch_path": "",
        "predicted_gain_pct": 0.0,
        "rationale": "flag-discovery only",
        "discovered_flags": {"sglang": ["--max-running-requests"]},
        "target_framework": "sglang",
        "stage_a_elapsed_ms": 500,
    }
    assert validate_envelope(envelope) == "OptimizeSuccess"


def test_optimize_failure():
    envelope = {
        "payload_kind": "OptimizeFailure",
        "reason": "source_not_found",
        "stage_a_elapsed_ms": 30,
    }
    assert validate_envelope(envelope) == "OptimizeFailure"


@pytest.mark.parametrize("verdict", ["KEEP", "REVERT", "NEEDS_REVIEW"])
def test_integrate_success_all_verdicts(verdict):
    envelope = {
        "payload_kind": "IntegrateSuccess",
        "verdict": verdict,
        "patch_id": "fw-20260520-deadbeef",
        "tput_before": 5000.0,
        "tput_after": 5400.0,
        "accuracy_before": 0.9,
        "accuracy_after": 0.89,
        "accuracy_drop": 0.01,
        "stage_b_elapsed_ms": 720000,
    }
    assert validate_envelope(envelope) == "IntegrateSuccess"


def test_integrate_failure():
    envelope = {
        "payload_kind": "IntegrateFailure",
        "reason": "patch_apply_failed",
        "patch_id": "fw-20260520-deadbeef",
        "stage_b_elapsed_ms": 100,
    }
    assert validate_envelope(envelope) == "IntegrateFailure"


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------
def test_unknown_payload_kind_rejected():
    with pytest.raises(EnvelopeValidationError, match="unknown envelope"):
        validate_envelope({"payload_kind": "MysterySuccess"})


def test_missing_payload_kind_rejected():
    with pytest.raises(EnvelopeValidationError, match="payload_kind is required"):
        validate_envelope({"patch_path": "/x", "predicted_gain_pct": 1.0})


def test_non_dict_envelope_rejected():
    with pytest.raises(EnvelopeValidationError, match="must be a dict"):
        validate_envelope(["not", "a", "dict"])  # type: ignore[arg-type]


def test_optimize_success_missing_required_field():
    with pytest.raises(EnvelopeValidationError, match="failed schema"):
        validate_envelope({
            "payload_kind": "OptimizeSuccess",
            # missing patch_path / predicted_gain_pct / rationale / stage_a_elapsed_ms
        })


def test_integrate_success_bad_verdict():
    with pytest.raises(EnvelopeValidationError, match="failed schema"):
        validate_envelope({
            "payload_kind": "IntegrateSuccess",
            "verdict": "MAYBE",
            "patch_id": "fw-x",
            "stage_b_elapsed_ms": 0,
        })


def test_optimize_failure_empty_reason():
    with pytest.raises(EnvelopeValidationError, match="failed schema"):
        validate_envelope({
            "payload_kind": "OptimizeFailure",
            "reason": "",
            "stage_a_elapsed_ms": 0,
        })


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------
def test_envelope_schemas_cover_all_four_kinds():
    assert set(ENVELOPE_SCHEMAS) == {
        "OptimizeSuccess", "OptimizeFailure",
        "IntegrateSuccess", "IntegrateFailure",
    }
