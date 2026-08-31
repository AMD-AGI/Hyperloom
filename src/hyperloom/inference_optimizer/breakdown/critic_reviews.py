"""Normalize durable Framework Agent critic-review evidence."""

from __future__ import annotations

from typing import Any

from .collectors._common import (
    _AUTHORING_TASK_KINDS,
    _FRAMEWORK_PHASES,
    _dict_rows,
    _first,
    _mapping,
    _optional_bool,
    _safe_get as _nested,
    _string_list,
    _to_int,
)


FRAMEWORK_REVIEW_FIELDS = (
    "proposal_msg_id",
    "candidate_id",
    "variant_name",
    "arm",
    "target_action",
    "source",
    "verdict",
    "effective_verdict",
    "reasoning",
    "confidence",
    "failure_reason_code",
    "required_evidence",
    "risks",
    "advice_text",
    "alternative_action",
    "followup_task_ids",
    "ts",
    "review_path",
)


def _candidate_id(value: Any) -> str:
    candidate = _mapping(value)
    return str(
        _first(
            candidate.get("candidate_id"),
            candidate.get("pr_url"),
            candidate.get("url"),
            candidate.get("ref"),
            candidate.get("head_sha"),
        )
        or ""
    )


def _has_patch_refs(params: dict[str, Any]) -> bool:
    return any(isinstance(params.get(key), list) and bool(params.get(key)) for key in ("patch_refs", "patches_written"))


def normalize_framework_reviews(
    *,
    request: dict[str, Any] | None,
    judge_bundle: dict[str, Any] | None,
    review: dict[str, Any] | None,
    emit: dict[str, Any] | None,
    review_path: str | None,
) -> list[dict[str, Any]]:
    """Return compact V6 Framework review rows from one Critic iteration."""
    request = _mapping(request)
    judge = _mapping(judge_bundle)
    review = _mapping(review)
    emit = _mapping(emit)
    normalized_review_path = str(review_path or "").replace("\\", "/") or None
    review_phase = (
        str(
            _first(
                judge.get("phase"),
                _nested(request, "context", "phase"),
                _nested(judge, "merged_context", "phase"),
            )
            or ""
        )
        .strip()
        .upper()
    )
    if review_phase and review_phase not in _FRAMEWORK_PHASES:
        return []

    proposals = {
        str(proposal.get("msg_id") or ""): proposal
        for proposal in _dict_rows(judge.get("proposals"))
        if proposal.get("msg_id")
    }
    effective: dict[str, dict[str, Any]] = {}
    for intent in _dict_rows(_nested(emit, "intent_envelope", "intents")):
        if str(intent.get("intent_type") or "") != "review_verdict":
            continue
        payload = _mapping(intent.get("payload"))
        target = str(payload.get("target_proposal_msg_id") or "")
        if target:
            effective[target] = payload

    rows: list[dict[str, Any]] = []
    for verdict_row in _dict_rows(review.get("review_verdicts")):
        proposal_id = str(verdict_row.get("target_proposal_msg_id") or "")
        proposal = proposals.get(proposal_id, {})
        payload = _mapping(proposal.get("payload"))
        params = _mapping(payload.get("params"))
        candidate = _mapping(_first(payload.get("candidate"), params.get("candidate")))
        action = str(_first(proposal.get("action_name"), payload.get("action_name"), payload.get("kind")) or "").lower()
        candidate_id = str(
            _first(
                payload.get("framework_agent_candidate_id"),
                params.get("framework_agent_candidate_id"),
                _candidate_id(candidate),
            )
            or ""
        )
        if action in {"params", "backends", "explore"}:
            arm = "config"
            target_action = "explore"
        elif action in {"framework_agent", "integrate", "integrate_patch"}:
            arm = "source"
            target_action = "integrate_patch"
        elif action == "specialist":
            task_kind = str(params.get("task_kind") or "").strip().lower()
            source_marker = bool(
                candidate_id
                or _optional_bool(params.get("framework_agent_authoring")) is True
                or _optional_bool(params.get("candidate_discovery")) is True
                or task_kind in _AUTHORING_TASK_KINDS
                or _has_patch_refs(params)
            )
            arm = "source" if source_marker else "config"
            target_action = "specialist"
        else:
            continue

        effective_row = effective.get(proposal_id, {})
        rows.append(
            {
                "proposal_msg_id": proposal_id,
                "candidate_id": candidate_id or None,
                "variant_name": _first(payload.get("variant_name"), params.get("variant_name"), None),
                "arm": arm,
                "target_action": target_action,
                "source": "critic" if str(verdict_row.get("source") or "critic") == "critic" else "critic_unavailable",
                "verdict": str(verdict_row.get("verdict") or ""),
                "effective_verdict": str(
                    _first(
                        effective_row.get("verdict"),
                        verdict_row.get("effective_verdict"),
                        verdict_row.get("verdict"),
                    )
                    or ""
                ),
                "reasoning": str(verdict_row.get("reasoning") or ""),
                "confidence": _first(verdict_row.get("confidence"), None),
                "failure_reason_code": _first(verdict_row.get("failure_reason_code"), None),
                "required_evidence": _string_list(verdict_row.get("required_evidence")),
                "risks": [
                    {
                        "severity": str(risk.get("severity") or ""),
                        "risk": str(_first(risk.get("risk"), risk.get("summary"), risk.get("reason")) or ""),
                    }
                    for risk in _dict_rows(verdict_row.get("risks"))
                ],
                "advice_text": _first(verdict_row.get("advice_text"), effective_row.get("advice_text"), None),
                "alternative_action": _first(verdict_row.get("alternative_action"), None),
                "followup_task_ids": _string_list(
                    _first(effective_row.get("followup_task_ids"), verdict_row.get("followup_task_ids"), [])
                ),
                "ts": str(_first(verdict_row.get("ts"), emit.get("ts"), review.get("ts")) or ""),
                "review_path": normalized_review_path,
                "phase": review_phase,
                "macro_cycle": _to_int(
                    _first(
                        payload.get("cycle"),
                        payload.get("macro_cycle"),
                        params.get("cycle"),
                        params.get("macro_cycle"),
                        proposal.get("cycle"),
                        proposal.get("macro_cycle"),
                        _nested(request, "context", "macro_cycle"),
                        _nested(request, "context", "cycle"),
                        _nested(judge, "merged_context", "macro_cycle"),
                        _nested(judge, "merged_context", "cycle"),
                    )
                ),
            }
        )
    return rows


__all__ = ["FRAMEWORK_REVIEW_FIELDS", "normalize_framework_reviews"]
