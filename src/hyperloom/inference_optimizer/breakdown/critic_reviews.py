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


#: The arm and the gated action of each reviewed action whose arm the action
#: alone decides. A specialist's arm depends on what it was dispatched to do.
_ACTION_ROUTES = {
    **dict.fromkeys(("params", "backends", "explore"), ("config", "explore")),
    **dict.fromkeys(("framework_agent", "integrate", "integrate_patch"), ("source", "integrate_patch")),
}


def _candidate_id(payload: dict[str, Any], params: dict[str, Any]) -> str:
    """The candidate a proposal names: its Framework Agent id, else the candidate's own reference."""
    candidate = _mapping(_first(payload.get("candidate"), params.get("candidate")))
    return str(
        _first(
            payload.get("framework_agent_candidate_id"),
            params.get("framework_agent_candidate_id"),
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


def _specialist_authors_source(params: dict[str, Any], candidate_id: str) -> bool:
    """Whether a specialist dispatch authors source rather than tunes configuration."""
    task_kind = str(params.get("task_kind") or "").strip().lower()
    return bool(
        candidate_id
        or _optional_bool(params.get("framework_agent_authoring")) is True
        or _optional_bool(params.get("candidate_discovery")) is True
        or task_kind in _AUTHORING_TASK_KINDS
        or _has_patch_refs(params)
    )


def _ruling(verdict: dict[str, Any], effective: dict[str, Any]) -> dict[str, Any]:
    """The critic's ruling on one proposal, beside what the emitted envelope made effective."""
    return {
        "source": "critic" if str(verdict.get("source") or "critic") == "critic" else "critic_unavailable",
        "verdict": str(verdict.get("verdict") or ""),
        "effective_verdict": str(
            _first(effective.get("verdict"), verdict.get("effective_verdict"), verdict.get("verdict")) or ""
        ),
        "reasoning": str(verdict.get("reasoning") or ""),
        "confidence": _first(verdict.get("confidence")),
        "failure_reason_code": _first(verdict.get("failure_reason_code")),
        "required_evidence": _string_list(verdict.get("required_evidence")),
        "risks": [
            {
                "severity": str(risk.get("severity") or ""),
                "risk": str(_first(risk.get("risk"), risk.get("summary"), risk.get("reason")) or ""),
            }
            for risk in _dict_rows(verdict.get("risks"))
        ],
        "advice_text": _first(verdict.get("advice_text"), effective.get("advice_text")),
        "alternative_action": _first(verdict.get("alternative_action")),
        "followup_task_ids": _string_list(_first(effective.get("followup_task_ids"), verdict.get("followup_task_ids"))),
    }


def normalize_framework_reviews(
    *,
    request: dict[str, Any] | None,
    judge_bundle: dict[str, Any] | None,
    review: dict[str, Any],
    emit: dict[str, Any],
    review_path: str | None,
) -> list[dict[str, Any]]:
    """Return compact V6 Framework review rows from one Critic iteration."""
    judge = _mapping(judge_bundle)
    normalized_review_path = str(review_path or "").replace("\\", "/") or None
    phase = _first(judge.get("phase"), _nested(request, "context", "phase"), _nested(judge, "merged_context", "phase"))
    review_phase = str(phase or "").strip().upper()
    if review_phase and review_phase not in _FRAMEWORK_PHASES:
        return []

    proposals = {str(row["msg_id"]): row for row in _dict_rows(judge.get("proposals")) if row.get("msg_id")}
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
        action = str(_first(proposal.get("action_name"), payload.get("action_name"), payload.get("kind")) or "").lower()
        candidate_id = _candidate_id(payload, params)
        if action == "specialist":
            arm = "source" if _specialist_authors_source(params, candidate_id) else "config"
            target_action = "specialist"
        elif action in _ACTION_ROUTES:
            arm, target_action = _ACTION_ROUTES[action]
        else:
            continue

        rows.append(
            {
                "proposal_msg_id": proposal_id,
                "candidate_id": candidate_id or None,
                "variant_name": _first(payload.get("variant_name"), params.get("variant_name")),
                "arm": arm,
                "target_action": target_action,
                **_ruling(verdict_row, effective.get(proposal_id, {})),
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


__all__ = ["normalize_framework_reviews"]
