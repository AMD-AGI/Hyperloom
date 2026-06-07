#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Validate Critic skill fixtures and optional generated outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RISK_TYPES = {
    "benchmark_missing",
    "benchmark_invalid",
    "accuracy_missing",
    "accuracy_failed",
    "patch_scope_mismatch",
    "active_path_unproven",
    "micro_only_evidence",
    "cache_or_rebuild_risk",
    "rollback_missing",
    "robustness_conflict",
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

VERDICTS = {"approve", "reject", "redirect", "advise", "needs_review"}
DECISION_VERDICTS = {"adopt", "reject", "revise", "needs_info"}
DECISION_BASIS = {"kb", "llm", "mixed", "session", "insufficient_context"}
SOURCES = {"critic", "mock", "timeout", "critic_unavailable"}
CONFIDENCE = {"high", "medium", "low"}
SEVERITIES = {"blocker", "major", "minor"}
COORDINATOR_INTENT_TYPES = {
    "review_verdict",
    "send_message",
    "ask_question",
    "answer",
    "alert",
    "update_persona",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def require_type(case_id: str, data: dict[str, Any], key: str, expected_type: type) -> None:
    if key not in data:
        fail(f"{case_id}: missing required key {key}")
    if not isinstance(data[key], expected_type):
        fail(f"{case_id}: {key} must be {expected_type.__name__}")


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

        require_type(case_id, case, "description", str)
        require_type(case_id, case, "input_packet", dict)
        require_type(case_id, case, "expected", dict)

        request_type = case["input_packet"].get("request_type")
        expected_kind = case["expected"].get("kind")
        if request_type == "review_verdict":
            if expected_kind != "review_verdict":
                fail(f"{case_id}: expected.kind must be review_verdict")
            validate_review_expected(case_id, case["expected"])
        elif request_type == "kb_draft":
            if expected_kind != "kb_draft":
                fail(f"{case_id}: expected.kind must be kb_draft")
            validate_kb_expected(case_id, case["expected"])
        elif request_type == "critic_decision_request":
            if expected_kind != "critic_decision_review":
                fail(f"{case_id}: expected.kind must be critic_decision_review")
            validate_decision_expected(case_id, case["expected"])
        elif request_type == "coordinator_inbox":
            if expected_kind != "intent_envelope":
                fail(f"{case_id}: expected.kind must be intent_envelope")
            validate_coordinator_expected(case_id, case["expected"])
        else:
            fail(f"{case_id}: unsupported request_type {request_type!r}")

    return data


def validate_review_expected(case_id: str, expected: dict[str, Any]) -> None:
    require_type(case_id, expected, "verdict", str)
    if expected["verdict"] not in VERDICTS:
        fail(f"{case_id}: invalid verdict {expected['verdict']}")
    for key in ("risk_types", "risk_types_any_of"):
        for risk_type in as_list(expected.get(key)):
            if risk_type not in RISK_TYPES:
                fail(f"{case_id}: unknown risk type {risk_type}")


def validate_decision_expected(case_id: str, expected: dict[str, Any]) -> None:
    require_type(case_id, expected, "verdict", str)
    if expected["verdict"] not in DECISION_VERDICTS:
        fail(f"{case_id}: invalid decision verdict {expected['verdict']}")
    basis = expected.get("basis")
    if basis is not None and basis not in DECISION_BASIS:
        fail(f"{case_id}: invalid basis {basis}")


def validate_coordinator_expected(case_id: str, expected: dict[str, Any]) -> None:
    intent_types_any_of = as_list(expected.get("intent_types_any_of"))
    for intent_type in intent_types_any_of:
        if intent_type not in COORDINATOR_INTENT_TYPES:
            fail(f"{case_id}: unknown intent_type {intent_type}")
    expected_count = expected.get("intent_count")
    if expected_count is not None:
        try:
            int(expected_count)
        except (TypeError, ValueError):
            fail(f"{case_id}: intent_count must be int")
    proposals = as_list(expected.get("verdicts_per_proposal"))
    for entry in proposals:
        if not isinstance(entry, dict):
            fail(f"{case_id}: verdicts_per_proposal items must be dicts")
        if entry.get("verdict") not in VERDICTS:
            fail(f"{case_id}: bad verdict in verdicts_per_proposal")


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
        kind = case["expected"]["kind"]
        if kind == "review_verdict":
            assert_review_output(case_id, case["expected"], output)
        elif kind == "kb_draft":
            assert_kb_output(case_id, case["expected"], output)
        elif kind == "critic_decision_review":
            assert_decision_output(case_id, case["expected"], output)
        elif kind == "intent_envelope":
            assert_intent_envelope_output(case_id, case["expected"], output)
        else:
            fail(f"{case_id}: unexpected expected.kind {kind!r}")


def assert_review_output(case_id: str, expected: dict[str, Any], output: dict[str, Any]) -> None:
    if output.get("kind") != "review_verdict":
        fail(f"{case_id}: output.kind must be review_verdict")
    for key, typ in (
        ("target_proposal_msg_id", str),
        ("verdict", str),
        ("source", str),
        ("confidence", str),
        ("reasoning", str),
        ("kb_evidence", list),
        ("packet_evidence", list),
        ("risks", list),
        ("required_evidence", list),
        ("notes", list),
    ):
        require_type(case_id, output, key, typ)

    if output["verdict"] not in VERDICTS:
        fail(f"{case_id}: invalid verdict {output['verdict']}")
    if output["source"] not in SOURCES:
        fail(f"{case_id}: invalid source {output['source']}")
    if output["confidence"] not in CONFIDENCE:
        fail(f"{case_id}: invalid confidence {output['confidence']}")
    if output["verdict"] != expected["verdict"]:
        fail(f"{case_id}: verdict mismatch")
    if expected.get("target_proposal_msg_id") and (
        output["target_proposal_msg_id"] != expected["target_proposal_msg_id"]
    ):
        fail(f"{case_id}: target_proposal_msg_id mismatch")
    if expected.get("source") and output["source"] != expected["source"]:
        fail(f"{case_id}: source mismatch")

    if output.get("predicted_gain_pct") is not None:
        if not isinstance(output["predicted_gain_pct"], (int, float)):
            fail(f"{case_id}: predicted_gain_pct must be numeric or null")
    if output["verdict"] in {"approve", "redirect"} and output.get("predicted_gain_pct") is None:
        fail(f"{case_id}: approve/redirect must include predicted_gain_pct")
    if output["verdict"] == "redirect" and not isinstance(output.get("alternative_action"), dict):
        fail(f"{case_id}: redirect must include alternative_action")
    if output["verdict"] == "advise" and not output.get("advice_text"):
        fail(f"{case_id}: advise must include advice_text")
    if output["verdict"] in {"reject", "redirect"} and not output["kb_evidence"]:
        fail(f"{case_id}: reject/redirect must include kb_evidence")

    for field in ("kb_evidence", "packet_evidence", "required_evidence", "notes"):
        for item in output[field]:
            if not isinstance(item, str):
                fail(f"{case_id}: {field} items must be strings")
    for risk in output["risks"]:
        validate_risk(case_id, risk)

    risk_types = [risk.get("type") for risk in output["risks"]]
    if "risk_types" in expected and risk_types != expected["risk_types"]:
        fail(f"{case_id}: risk_types mismatch: {risk_types}")
    any_risk = set(as_list(expected.get("risk_types_any_of")))
    if any_risk and not any_risk.intersection(risk_types):
        fail(f"{case_id}: expected one risk type from {sorted(any_risk)}")

    required = set(output["required_evidence"])
    for evidence in as_list(expected.get("required_evidence_includes")):
        if evidence not in required:
            fail(f"{case_id}: missing required evidence {evidence}")
    if expected.get("required_evidence_absent") and required:
        fail(f"{case_id}: required_evidence should be empty")

    confidence_any_of = set(as_list(expected.get("confidence_any_of")))
    if confidence_any_of and output["confidence"] not in confidence_any_of:
        fail(f"{case_id}: confidence must be one of {sorted(confidence_any_of)}")

    min_blocker_count = int(expected.get("min_blocker_count", 0))
    blocker_count = sum(1 for risk in output["risks"] if risk.get("severity") == "blocker")
    if blocker_count < min_blocker_count:
        fail(f"{case_id}: expected at least {min_blocker_count} blocker risks")

    if len(output["kb_evidence"]) < int(expected.get("min_kb_evidence", 0)):
        fail(f"{case_id}: too little kb_evidence")
    if expected.get("alternative_action_required") and not output.get("alternative_action"):
        fail(f"{case_id}: missing alternative_action")
    if expected.get("advice_text_required") and not output.get("advice_text"):
        fail(f"{case_id}: missing advice_text")
    if "min_predicted_gain_pct" in expected:
        if float(output.get("predicted_gain_pct") or 0) < float(expected["min_predicted_gain_pct"]):
            fail(f"{case_id}: predicted_gain_pct below minimum")


def validate_risk(case_id: str, risk: Any) -> None:
    if not isinstance(risk, dict):
        fail(f"{case_id}: risks entries must be objects")
    for key in ("type", "severity", "reason", "required_fix", "evidence_ref"):
        require_type(case_id, risk, key, str)
        if not risk[key]:
            fail(f"{case_id}: risk field {key} must be non-empty")
    if risk["type"] not in RISK_TYPES:
        fail(f"{case_id}: unknown risk type {risk['type']}")
    if risk["severity"] not in SEVERITIES:
        fail(f"{case_id}: invalid severity {risk['severity']}")


def assert_kb_output(case_id: str, expected: dict[str, Any], output: dict[str, Any]) -> None:
    if output.get("kind") != "kb_draft":
        fail(f"{case_id}: output.kind must be kb_draft")
    require_type(case_id, output, "kb_drafts", list)
    require_type(case_id, output, "rejected_candidates", list)
    require_type(case_id, output, "notes", list)

    drafts = as_list(output.get("kb_drafts"))
    rejected = as_list(output.get("rejected_candidates"))
    for draft in drafts:
        validate_kb_draft(case_id, draft)
    for rejected_candidate in rejected:
        validate_rejected_candidate(case_id, rejected_candidate)
    for note in output["notes"]:
        if not isinstance(note, str):
            fail(f"{case_id}: notes items must be strings")

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


def validate_kb_draft(case_id: str, draft: Any) -> None:
    if not isinstance(draft, dict):
        fail(f"{case_id}: kb_drafts entries must be objects")
    for key in ("model_family", "model", "category", "action", "lesson"):
        require_type(case_id, draft, key, str)
        if not draft[key]:
            fail(f"{case_id}: KB draft field {key} must be non-empty")
    if draft["category"] not in KB_CATEGORIES:
        fail(f"{case_id}: unknown KB category {draft['category']}")
    if "tags" in draft and not isinstance(draft["tags"], list):
        fail(f"{case_id}: KB draft tags must be a list")
    if "result" in draft and not isinstance(draft["result"], dict):
        fail(f"{case_id}: KB draft result must be an object")
    if "confidence" in draft:
        confidence = float(draft["confidence"])
        if not 0.0 <= confidence <= 1.0:
            fail(f"{case_id}: KB draft confidence must be in [0, 1]")


def assert_decision_output(case_id: str, expected: dict[str, Any], output: dict[str, Any]) -> None:
    if output.get("kind") != "critic_decision_review":
        fail(f"{case_id}: output.kind must be critic_decision_review")
    require_type(case_id, output, "verdict", str)
    if output["verdict"] not in DECISION_VERDICTS:
        fail(f"{case_id}: invalid decision verdict {output['verdict']}")
    if output["verdict"] != expected["verdict"]:
        fail(f"{case_id}: decision verdict mismatch")
    basis = output.get("basis")
    if basis and basis not in DECISION_BASIS:
        fail(f"{case_id}: invalid basis {basis}")
    if expected.get("basis") and basis != expected["basis"]:
        fail(f"{case_id}: basis mismatch")
    required_keys = ("reason",)
    if output["verdict"] in ("adopt", "reject", "revise"):
        required_keys += ("session_evidence",)
    for key in required_keys:
        if key not in output:
            fail(f"{case_id}: missing required key {key}")
    if output["verdict"] == "needs_info":
        if not output.get("required_context"):
            fail(f"{case_id}: needs_info requires non-empty required_context")


def assert_intent_envelope_output(case_id: str, expected: dict[str, Any], output: dict[str, Any]) -> None:
    if output.get("kind") != "intent_envelope":
        fail(f"{case_id}: output.kind must be intent_envelope")
    require_type(case_id, output, "intents", list)
    intents = output["intents"]
    if not intents:
        fail(f"{case_id}: intent envelope must be non-empty")
    types = [item.get("intent_type") for item in intents]
    for intent_type in types:
        if intent_type not in COORDINATOR_INTENT_TYPES:
            fail(f"{case_id}: invalid intent_type {intent_type!r}")
    expected_count = expected.get("intent_count")
    if expected_count is not None and len(intents) != int(expected_count):
        fail(f"{case_id}: intent count mismatch — got {len(intents)}, expected {expected_count}")
    for entry in as_list(expected.get("verdicts_per_proposal")):
        target = entry.get("target_proposal_msg_id")
        verdict = entry.get("verdict")
        match = next(
            (
                i for i in intents
                if i.get("intent_type") == "review_verdict"
                and i.get("payload", {}).get("target_proposal_msg_id") == target
            ),
            None,
        )
        if match is None:
            fail(f"{case_id}: missing review_verdict for proposal {target}")
        if match["payload"].get("verdict") != verdict:
            fail(f"{case_id}: verdict for {target} expected {verdict}, got {match['payload'].get('verdict')}")
    intent_any = set(as_list(expected.get("intent_types_any_of")))
    if intent_any and not intent_any.intersection(types):
        fail(f"{case_id}: expected at least one intent_type from {sorted(intent_any)}")


def validate_rejected_candidate(case_id: str, rejected_candidate: Any) -> None:
    if not isinstance(rejected_candidate, dict):
        fail(f"{case_id}: rejected_candidates entries must be objects")
    require_type(case_id, rejected_candidate, "reason", str)
    if not rejected_candidate["reason"]:
        fail(f"{case_id}: rejected candidate reason must be non-empty")
    if "source_section" in rejected_candidate and not isinstance(
        rejected_candidate["source_section"], str
    ):
        fail(f"{case_id}: rejected candidate source_section must be a string")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "review_verdict_cases.json",
            "kb_draft_cases.json",
            "decision_review_cases.json",
            "coordinator_inbox_cases.json",
        ],
        help="Case files to validate.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        help="Output JSON keyed by case id for assertion checks. Defaults to expected_outputs.json if present.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cases: list[dict[str, Any]] = []
    for case_name in args.cases:
        case_path = Path(case_name)
        if not case_path.is_absolute():
            case_path = base_dir / case_path
        cases.extend(validate_case_file(case_path))

    outputs_path = args.outputs
    default_outputs = base_dir / "expected_outputs.json"
    if outputs_path is None and default_outputs.exists():
        outputs_path = default_outputs
    if outputs_path:
        validate_outputs(cases, outputs_path)

    print(f"OK: validated {len(cases)} Critic test cases")


if __name__ == "__main__":
    main()
