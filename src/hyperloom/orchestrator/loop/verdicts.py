# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verdict helpers: collapsing, holding, and serialising Critic review results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..specialists.patch_safety import (
    ADVISE_VERDICT,
    advisory_only_reason_codes,
    advisory_rules_govern,
)

# Advisory fields carried on a Critic ``review_verdict`` payload beyond the bare verdict/reasoning.
_VERDICT_ADVISORY_LIST_KEYS: tuple[str, ...] = (
    "required_evidence",
    "risks",
    "notes",
    "kb_evidence",
    "packet_evidence",
)
# The verdict that ends a proposal's life; its counterpart ``ADVISE_VERDICT`` lets the proposal through.
_REJECT_VERDICT: str = "reject"

_VERDICT_ADVISORY_TEXT_KEYS: tuple[str, ...] = (
    "advice_text",
    "alternative_action",
)


def serialize_verdict_advisory(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the advisory field set from a ``review_verdict`` payload."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _VERDICT_ADVISORY_LIST_KEYS:
        raw = payload.get(key)
        if isinstance(raw, (list, tuple)):
            items = [item for item in raw if item not in (None, "")]
        elif raw in (None, ""):
            items = []
        else:
            items = [raw]
        if items:
            out[key] = list(items)
    for key in _VERDICT_ADVISORY_TEXT_KEYS:
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            out[key] = raw
    return out


# The fields a Critic states its grounds in: ``reasoning`` on a single verdict, ``rationale`` on one ``verdict_map``
# entry — the per-variant shape PolicyGate documents and every fixture uses.
_VERDICT_PROSE_KEYS: tuple[str, ...] = ("reasoning", "rationale")

# What a citation looks like: the code opens the verdict's grounds and a colon introduces the finding, the shape the
# field verdict used -- ``"specialist_quantitative_claim_violation: the proposal payload carries the forbidden
# predicted_gain_pct field."`` Nothing may precede the code but whitespace or a backtick, and only the opening line of
# each prose field is read.
_CITATION_OPENER: str = r"[ \t]*`?"


def _opening_prose_lines(entry: dict[str, Any]) -> list[str]:
    """Return the opening line of each prose field ``entry`` states grounds in."""
    openings: list[str] = []
    for key in _VERDICT_PROSE_KEYS:
        for line in str(entry.get(key) or "").splitlines():
            if line.strip():
                openings.append(line)
                break
    return openings


def cited_advisory_reason_code(entry: dict[str, Any]) -> str:
    """Return the advisory-only rule ``entry`` cites, from the field or its prose."""
    advisory = advisory_only_reason_codes()
    explicit = str(entry.get("failure_reason_code") or "").strip()
    if explicit:
        return explicit if explicit in advisory else ""
    # At most one code can open one line, so the sort only fixes the order the candidates are tried in.
    for opening in _opening_prose_lines(entry):
        for code in sorted(advisory):
            if re.match(rf"{_CITATION_OPENER}{re.escape(code)}`?[ \t]*:", opening):
                return code
    return ""


# Priority a batch of per-variant verdicts collapses by: one approved variant carries the proposal, otherwise one
# reject sinks it, and advice outranks a request for more review. :func:`collapse_verdict_map` applies this to the
# proceedable subset first so a genuine reject cannot sink siblings that may still run.
_VERDICT_COLLAPSE_ORDER: tuple[str, ...] = ("approve", _REJECT_VERDICT, ADVISE_VERDICT, "needs_review")
_PROCEEDABLE_VERDICTS: frozenset[str] = frozenset({"approve", ADVISE_VERDICT})


def collapse_verdicts(verdicts: Iterable[str]) -> str:
    """Collapse per-variant verdicts into the one the proposal is decided on."""
    present = set(verdicts)
    for candidate in _VERDICT_COLLAPSE_ORDER:
        if candidate in present:
            return candidate
    return "needs_review"


def proceedable_variant_names(held_by_name: Mapping[str, str]) -> set[str]:
    """Return variant names whose held verdict lets them reach a benchmark."""
    return {name for name, verdict in held_by_name.items() if verdict in _PROCEEDABLE_VERDICTS and str(name).strip()}


def collapse_verdict_map(held_by_name: Mapping[str, str]) -> tuple[str, set[str] | None]:
    """Collapse a held ``verdict_map`` and name the variants that may run."""
    proceedable = proceedable_variant_names(held_by_name)
    if proceedable:
        return collapse_verdicts(held_by_name[name] for name in proceedable), proceedable
    return collapse_verdicts(held_by_name.values()), None


def _states_findings(value: Any) -> bool:
    """Return whether a findings field states anything at all."""
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def verdict_rests_on_one_ground(entry: dict[str, Any]) -> bool:
    """Return whether ``entry`` refuses for a single reason."""
    if _states_findings(entry.get("required_evidence")):
        return False
    risks = entry.get("risks")
    if not isinstance(risks, (list, tuple)):
        return not _states_findings(risks)
    return len([risk for risk in risks if risk]) <= 1


# The findings a review lists outside its prose: the evidence it still wants and the risks it names.
_VERDICT_FINDING_KEYS: tuple[str, ...] = ("required_evidence", "risks")


def _batch_states_findings(payload: dict[str, Any]) -> bool:
    """Return whether a batch review states a finding of its own."""
    if not isinstance(payload, dict):
        return False
    return any(_states_findings(payload.get(key)) for key in _VERDICT_FINDING_KEYS)


def _inheritable_reason_code(payload: dict[str, Any]) -> str:
    """Return the payload's declared code, when it cannot soften a variant's reject."""
    code = str(payload.get("failure_reason_code") or "").strip()
    return "" if code in advisory_only_reason_codes() else code


def verdict_map_entry_grounds(entry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Return the grounds one ``verdict_map`` entry rests on."""
    if not isinstance(entry, dict):
        return {}
    grounds = dict(entry)
    if not isinstance(payload, dict):
        return grounds
    if not grounds.get("failure_reason_code"):
        code = _inheritable_reason_code(payload)
        if code:
            grounds["failure_reason_code"] = code
    return grounds


def _stated_verdict(entry: dict[str, Any]) -> str:
    """Return the verdict ``entry`` states, whatever a hold makes of it."""
    return str(entry.get("verdict") or "").strip()


def verdict_held_to_its_rule(entry: dict[str, Any], *, action_name: str) -> tuple[str, str]:
    """Return the verdict a ``review_verdict`` entry carries, and why it moved."""
    if not isinstance(entry, dict):
        return "", ""
    verdict = _stated_verdict(entry)
    if verdict != _REJECT_VERDICT:
        return verdict, ""
    if not advisory_rules_govern(action_name):
        return verdict, ""
    if not verdict_rests_on_one_ground(entry):
        return verdict, ""
    reason_code = cited_advisory_reason_code(entry)
    if reason_code:
        return ADVISE_VERDICT, reason_code
    return verdict, ""


def verdict_map_entry_held_to_its_rule(
    entry: dict[str, Any],
    payload: dict[str, Any],
    *,
    action_name: str,
) -> tuple[str, str]:
    """Return the verdict one ``verdict_map`` entry carries, and why it moved."""
    grounds = verdict_map_entry_grounds(entry, payload)
    if _batch_states_findings(payload):
        return _stated_verdict(grounds), ""
    return verdict_held_to_its_rule(grounds, action_name=action_name)
