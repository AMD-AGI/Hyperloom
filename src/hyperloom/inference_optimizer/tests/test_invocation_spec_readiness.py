# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A partial invocation spec must be declined, not burned through."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common import invocation_spec_readiness as isr


def _spec(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "invocation_spec.json"
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")
    return path


def test_a_complete_spec_is_ready(tmp_path: Path) -> None:
    verdict = isr.evaluate_spec_readiness(_spec(tmp_path, {"status": "complete", "missing_fields": []}))
    assert verdict.ready is True
    assert verdict.reason == ""
    assert verdict.missing_fields == ()


def test_a_partial_spec_is_declined_with_its_missing_fields(tmp_path: Path) -> None:
    """The whole point: this used to pass and then burn the entire budget."""
    path = _spec(tmp_path, {"status": "partial", "missing_fields": ["public_callable", "inputs"]})
    verdict = isr.evaluate_spec_readiness(path)
    assert verdict.ready is False
    assert verdict.reason == isr.REASON_PARTIAL
    assert verdict.missing_fields == ("public_callable", "inputs")
    assert "public_callable" in verdict.detail


def test_a_partial_spec_without_a_list_still_reports_a_reason(tmp_path: Path) -> None:
    verdict = isr.evaluate_spec_readiness(_spec(tmp_path, {"status": "partial"}))
    assert verdict.ready is False
    assert verdict.reason == isr.REASON_PARTIAL
    assert "unspecified" in verdict.detail


def test_an_absent_status_is_treated_as_complete(tmp_path: Path) -> None:
    """Older specs predate the field; only an explicit `partial` rejects."""
    assert isr.evaluate_spec_readiness(_spec(tmp_path, {"logical_operator": "gemm"})).ready is True


@pytest.mark.parametrize("status", ["complete", "COMPLETE", " Complete "])
def test_complete_is_matched_case_insensitively(tmp_path: Path, status: str) -> None:
    assert isr.evaluate_spec_readiness(_spec(tmp_path, {"status": status})).ready is True


@pytest.mark.parametrize("status", ["partial", "PARTIAL", " Partial "])
def test_partial_is_matched_case_insensitively(tmp_path: Path, status: str) -> None:
    assert isr.evaluate_spec_readiness(_spec(tmp_path, {"status": status})).reason == isr.REASON_PARTIAL


@pytest.mark.parametrize("value", ["", "   ", None])
def test_no_path_reports_missing(value: Any) -> None:
    verdict = isr.evaluate_spec_readiness(value)
    assert verdict.ready is False
    assert verdict.reason == isr.REASON_MISSING


def test_absent_file_reports_missing(tmp_path: Path) -> None:
    verdict = isr.evaluate_spec_readiness(tmp_path / "absent.json")
    assert verdict.reason == isr.REASON_MISSING


def test_unreadable_spec_is_distinguished_from_missing(tmp_path: Path) -> None:
    """An environment problem and absent upstream evidence are not the same fault."""
    verdict = isr.evaluate_spec_readiness(_spec(tmp_path, "not json"))
    assert verdict.ready is False
    assert verdict.reason == isr.REASON_UNREADABLE


def test_non_object_spec_is_unreadable(tmp_path: Path) -> None:
    assert isr.evaluate_spec_readiness(_spec(tmp_path, [1, 2])).reason == isr.REASON_UNREADABLE


def test_missing_fields_survives_a_ready_verdict(tmp_path: Path) -> None:
    """A complete spec may still list advisory gaps; they are reported, not fatal."""
    verdict = isr.evaluate_spec_readiness(_spec(tmp_path, {"status": "complete", "missing_fields": ["outputs"]}))
    assert verdict.ready is True
    assert verdict.missing_fields == ("outputs",)


def test_non_list_missing_fields_is_ignored(tmp_path: Path) -> None:
    verdict = isr.evaluate_spec_readiness(_spec(tmp_path, {"status": "complete", "missing_fields": "outputs"}))
    assert verdict.missing_fields == ()
