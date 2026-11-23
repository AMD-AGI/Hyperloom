# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verdict helpers: collapsing and serialising Critic review results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# Advisory fields carried on a Critic ``review_verdict`` payload beyond the bare verdict/reasoning.
_VERDICT_ADVISORY_LIST_KEYS: tuple[str, ...] = (
    "required_evidence",
    "risks",
    "notes",
    "kb_evidence",
    "packet_evidence",
)
# The verdict that ends a proposal's life, and the one that lets it through carrying the Critic's notes.
_REJECT_VERDICT: str = "reject"
_ADVISE_VERDICT: str = "advise"

_VERDICT_ADVISORY_TEXT_KEYS: tuple[str, ...] = (
    "advice_text",
    "alternative_action",
)


@dataclass(frozen=True)
class CriticRuling:
    """The Critic's collapsed ruling on one proposal, with what it wrote alongside."""

    verdict: str
    reasoning: str
    advisory: dict[str, Any] | None = None
    approved_variant_names: set[str] | None = None
    payload: Mapping[str, Any] | None = None
    variant_verdicts: list[dict[str, Any]] | None = None


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


# Priority a batch of per-variant verdicts collapses by: one approved variant carries the proposal, otherwise one
# reject sinks it, and advice outranks a request for more review. :func:`collapse_verdict_map` applies this to the
# proceedable subset first so a genuine reject cannot sink siblings that may still run.
_VERDICT_COLLAPSE_ORDER: tuple[str, ...] = ("approve", _REJECT_VERDICT, _ADVISE_VERDICT, "needs_review")
_PROCEEDABLE_VERDICTS: frozenset[str] = frozenset({"approve", _ADVISE_VERDICT})


def collapse_verdicts(verdicts: Iterable[str]) -> str:
    """Collapse per-variant verdicts into the one the proposal is decided on."""
    present = set(verdicts)
    for candidate in _VERDICT_COLLAPSE_ORDER:
        if candidate in present:
            return candidate
    return "needs_review"


def proceedable_variant_names(verdict_by_name: Mapping[str, str]) -> set[str]:
    """Return variant names whose verdict lets them reach a benchmark."""
    return {name for name, verdict in verdict_by_name.items() if verdict in _PROCEEDABLE_VERDICTS and str(name).strip()}


def collapse_verdict_map(verdict_by_name: Mapping[str, str]) -> tuple[str, set[str] | None]:
    """Collapse a ``verdict_map`` and name the variants that may run."""
    proceedable = proceedable_variant_names(verdict_by_name)
    if proceedable:
        return collapse_verdicts(verdict_by_name[name] for name in proceedable), proceedable
    return collapse_verdicts(verdict_by_name.values()), None
