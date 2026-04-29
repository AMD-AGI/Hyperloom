#!/usr/bin/env python3
"""Validate Critic skill test fixtures and optional generated outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PATCH_OBJECTION_TYPES = {
    "benchmark_missing",
    "benchmark_invalid",
    "accuracy_missing",
    "accuracy_failed",
    "patch_scope_mismatch",
    "active_path_unproven",
    "micro_only_evidence",
    "cache_or_rebuild_risk",
    "rollback_missing",
    "triage_conflict",
    "cross_layer_conflict",
    "regression_risk",
    "insufficient_context",
}

KB_CATEGORIES = {
    "backend_exploration",
    "kernel_optimization",
    "call_stack_optimization",
    "server_params",
    "pitfall",
    "benchmark_methodology",
    "architecture_constraint",
    "target_comparison",
    "framework_comparison",
    "lesson",
    "crash_recovery",
    "dream_consolidation",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_case_file(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        fail(f"{path}: top-level value must be a list")

    seen: set[str] = set()
    for index, case in enumerate(data):
        prefix = f"{path.name}[{index}]"
        if not isinstance(case, dict):
            fail(f"{prefix}: case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            fail(f"{prefix}: missing id")
        if case_id in seen:
            fail(f"{prefix}: duplicate id {case_id}")
        seen.add(case_id)

        if not isinstance(case.get("description"), str) or not case["description"]:
            fail(f"{case_id}: missing description")
        if not isinstance(case.get("input_packet"), dict):
            fail(f"{case_id}: missing input_packet")
        if not isinstance(case.get("expected"), dict):
            fail(f"{case_id}: missing expected")

        request_type = case["input_packet"].get("request_type")
        expected_kind = case["expected"].get("kind")
        if request_type == "patch_vote":
            if expected_kind != "patch_vote":
                fail(f"{case_id}: expected.kind must be patch_vote")
            validate_patch_expected(case_id, case["expected"])
        elif request_type == "kb_draft":
            if expected_kind != "kb_draft":
                fail(f"{case_id}: expected.kind must be kb_draft")
            validate_kb_expected(case_id, case["expected"])
        else:
            fail(f"{case_id}: unsupported request_type {request_type!r}")

    return data


def validate_patch_expected(case_id: str, expected: dict[str, Any]) -> None:
    if "approval" not in expected:
        fail(f"{case_id}: patch expected must include approval")
    if not isinstance(expected["approval"], bool):
        fail(f"{case_id}: approval must be boolean")

    for key in ("objection_types", "objection_types_any_of"):
        for objection_type in as_list(expected.get(key)):
            if objection_type not in PATCH_OBJECTION_TYPES:
                fail(f"{case_id}: unknown objection type {objection_type}")


def validate_kb_expected(case_id: str, expected: dict[str, Any]) -> None:
    for key in ("categories_include", "categories_include_any_of"):
        for category in as_list(expected.get(key)):
            if category not in KB_CATEGORIES:
                fail(f"{case_id}: unknown KB category {category}")

    min_confidence = expected.get("min_confidence")
    if min_confidence is not None and not 0 <= float(min_confidence) <= 1:
        fail(f"{case_id}: min_confidence must be in [0, 1]")


def validate_outputs(cases: list[dict[str, Any]], outputs_path: Path) -> None:
    outputs = load_json(outputs_path)
    if not isinstance(outputs, dict):
        fail(f"{outputs_path}: outputs must be an object keyed by case id")

    by_id = {case["id"]: case for case in cases}
    missing = sorted(set(by_id) - set(outputs))
    if missing:
        fail(f"{outputs_path}: missing outputs for cases: {', '.join(missing)}")

    for case_id, case in by_id.items():
        output = outputs[case_id]
        if not isinstance(output, dict):
            fail(f"{case_id}: output must be an object")
        if case["expected"]["kind"] == "patch_vote":
            assert_patch_output(case_id, case["expected"], output)
        else:
            assert_kb_output(case_id, case["expected"], output)


def assert_patch_output(case_id: str, expected: dict[str, Any], output: dict[str, Any]) -> None:
    if output.get("kind") != "patch_vote":
        fail(f"{case_id}: output.kind must be patch_vote")
    if output.get("approval") is not expected["approval"]:
        fail(f"{case_id}: approval mismatch")

    objection_types = [obj.get("type") for obj in as_list(output.get("objections"))]
    if "objection_types" in expected and objection_types != expected["objection_types"]:
        fail(f"{case_id}: objection_types mismatch: {objection_types}")

    any_of = set(as_list(expected.get("objection_types_any_of")))
    if any_of and not any_of.intersection(objection_types):
        fail(f"{case_id}: expected one objection type from {sorted(any_of)}")

    required = set(as_list(output.get("required_evidence")))
    for evidence in as_list(expected.get("required_evidence_includes")):
        if evidence not in required:
            fail(f"{case_id}: missing required evidence {evidence}")

    any_required = set(as_list(expected.get("required_evidence_includes_any_of")))
    if any_required and not any_required.intersection(required):
        fail(f"{case_id}: expected one required evidence from {sorted(any_required)}")

    if expected.get("required_evidence_absent") and required:
        fail(f"{case_id}: required_evidence should be empty")

    confidence_any_of = set(as_list(expected.get("confidence_any_of")))
    if confidence_any_of and output.get("confidence") not in confidence_any_of:
        fail(f"{case_id}: confidence must be one of {sorted(confidence_any_of)}")

    min_blocker_count = int(expected.get("min_blocker_count", 0))
    blocker_count = sum(
        1 for objection in as_list(output.get("objections"))
        if objection.get("severity") == "blocker"
    )
    if blocker_count < min_blocker_count:
        fail(f"{case_id}: expected at least {min_blocker_count} blocker objections")


def assert_kb_output(case_id: str, expected: dict[str, Any], output: dict[str, Any]) -> None:
    if output.get("kind") != "kb_draft":
        fail(f"{case_id}: output.kind must be kb_draft")

    drafts = as_list(output.get("kb_drafts"))
    rejected = as_list(output.get("rejected_candidates"))

    if len(drafts) < int(expected.get("min_kb_drafts", 0)):
        fail(f"{case_id}: too few KB drafts")
    if "max_kb_drafts" in expected and len(drafts) > int(expected["max_kb_drafts"]):
        fail(f"{case_id}: too many KB drafts")
    if len(rejected) < int(expected.get("min_rejected_candidates", 0)):
        fail(f"{case_id}: too few rejected candidates")

    categories = [draft.get("category") for draft in drafts]
    for category in as_list(expected.get("categories_include")):
        if category not in categories:
            fail(f"{case_id}: missing KB category {category}")

    any_category = set(as_list(expected.get("categories_include_any_of")))
    if any_category and not any_category.intersection(categories):
        fail(f"{case_id}: expected one category from {sorted(any_category)}")

    required_fields = as_list(expected.get("drafts_must_include_fields"))
    for draft in drafts:
        for field in required_fields:
            if field not in draft:
                fail(f"{case_id}: draft missing field {field}")

    status_any_of = set(as_list(expected.get("result_status_any_of")))
    if status_any_of:
        statuses = {draft.get("result", {}).get("status") for draft in drafts}
        if not status_any_of.intersection(statuses):
            fail(f"{case_id}: expected result status from {sorted(status_any_of)}")

    min_confidence = expected.get("min_confidence")
    if min_confidence is not None:
        for draft in drafts:
            if float(draft.get("confidence", 0.0)) < float(min_confidence):
                fail(f"{case_id}: draft confidence below {min_confidence}")

    reason_terms = as_list(expected.get("rejection_reason_contains_any_of"))
    if reason_terms:
        reasons = " ".join(str(item.get("reason", "")) for item in rejected).lower()
        if not any(term.lower() in reasons for term in reason_terms):
            fail(f"{case_id}: rejected candidate reason did not match expected terms")

    if expected.get("must_not_create_unqualified_duplicate"):
        has_supersedes = any(draft.get("supersedes") for draft in drafts)
        has_rejection = bool(rejected)
        if not has_supersedes and not has_rejection:
            fail(f"{case_id}: duplicate must be rejected or explicitly supersede existing KB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["patch_vote_cases.json", "kb_draft_cases.json"],
        help="Case files to validate.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        help="Optional output JSON keyed by case id for assertion checks.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cases: list[dict[str, Any]] = []
    for case_name in args.cases:
        case_path = Path(case_name)
        if not case_path.is_absolute():
            case_path = base_dir / case_path
        cases.extend(validate_case_file(case_path))

    if args.outputs:
        validate_outputs(cases, args.outputs)

    print(f"OK: validated {len(cases)} Critic test cases")


if __name__ == "__main__":
    main()
