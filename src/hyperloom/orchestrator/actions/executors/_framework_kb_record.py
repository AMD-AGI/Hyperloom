# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The knowledge-base lesson an integrated framework patch leaves behind."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KbLesson:
    """The outcome and measured deltas one framework KB record stores."""

    outcome: str
    tps_delta_pct: float
    accuracy_delta_pct: float | None = None
    config_fingerprint: str = ""


class _Environment(NamedTuple):
    """The session the lesson was learned in, as the ledger indexes it."""

    session_id: str
    model_class: str
    gpu_type: str
    precision: str


@dataclass(frozen=True)
class _Attribution:
    """What the patch claims about itself: the proposal that carried it, backed by the specialist payload."""

    proposal: Mapping[str, Any]
    done_payload: Mapping[str, Any]

    def text(self, key: str) -> str:
        """The proposal's value for ``key``, or the payload's where the proposal left it empty."""
        return str(self.proposal.get(key) or self.done_payload.get(key) or "").strip()

    def name(self, key: str) -> str:
        """``text`` for a key whose vocabulary is case-insensitive."""
        return self.text(key).lower()

    def entries(self, key: str) -> list[str]:
        """The non-empty list under ``key``, which a writer may deliver as a single string."""
        value = self.proposal.get(key) or self.done_payload.get(key) or []
        items = [value] if isinstance(value, str) else value
        return [str(item).strip() for item in items if str(item).strip()]

    def dedup_keys(self) -> tuple[str, str]:
        """The upstream PR url and head sha the ledger dedups on."""
        return self.text("fa_pr_url"), self.text("fa_pr_sha")

    def accuracy_delta_pct(self, measured: float | None) -> float:
        """The measured accuracy delta, or the one the proposal declared when nothing was measured."""
        if measured is not None:
            return measured
        try:
            return float(self.proposal.get("accuracy_delta_pct") or self.done_payload.get("accuracy_delta_pct") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def first_patch(self) -> str:
        """The patch file the proposal delivered, named so the record points at the diff it graded."""
        written = self.proposal.get("patches_written") or []
        return str(written[0]) if isinstance(written, list) and written else ""


def _environment_of(shared_state: Any) -> _Environment:
    """The hardware and model identity every record in this session is indexed under."""
    if shared_state is None:
        return _Environment("", "", "", "")
    return _Environment(
        session_id=str(getattr(shared_state, "recipe_kb_session_id", "") or ""),
        model_class=str(getattr(shared_state, "model_class", "") or "").strip(),
        gpu_type=str(getattr(shared_state, "gpu_type", "") or "").strip(),
        precision=str(getattr(shared_state, "precision", "") or "").strip(),
    )


async def write_framework_lesson(
    lesson: KbLesson,
    proposal: Mapping[str, Any],
    *,
    done_payload: Mapping[str, Any] | None,
    shared_state: Any,
) -> None:
    """Append this attempt's outcome to the framework lesson ledger.

    Discovery dedups on the ledger, so a lane that writes nothing is re-benched
    forever; a proposal carrying neither dedup key cannot be deduped and is
    skipped instead. A write that fails is logged and swallowed -- the lesson is
    an optimisation for later rounds, not part of this round's verdict.
    """
    attribution = _Attribution(proposal, done_payload or {})
    pr_url, pr_sha = attribution.dedup_keys()
    if not pr_url and not pr_sha:
        log.warning("integrate_patch: framework proposal lacks both fa_pr_url and fa_pr_sha; KB writeback skipped")
        return
    environment = _environment_of(shared_state)
    try:
        from ...knowledge.kb_writeback import FrameworkOutcome, write_framework_record

        written = await write_framework_record(
            FrameworkOutcome(
                pr_url=pr_url,
                pr_sha=pr_sha,
                patch_path=attribution.first_patch(),
                outcome=lesson.outcome,
                tps_delta_pct=float(lesson.tps_delta_pct),
                session_id=environment.session_id,
                framework=attribution.name("framework"),
                gap_canonical_id=attribution.text("gap_canonical_id"),
                gap_keywords=[keyword.lower() for keyword in attribution.entries("gap_keywords")],
                model_class=environment.model_class,
                gpu_type=environment.gpu_type,
                precision=environment.precision,
                applicability=lesson.config_fingerprint or attribution.text("applicability"),
                provenance=attribution.text("provenance"),
                accuracy_delta_pct=attribution.accuracy_delta_pct(lesson.accuracy_delta_pct),
                changed_files=attribution.entries("changed_files"),
                source_framework=attribution.name("source_framework"),
                target_framework=attribution.name("target_framework"),
            )
        )
    except Exception as exc:  # noqa: BLE001 — KB write is best-effort
        log.warning("integrate_patch: framework KB writeback failed: %r", exc)
        return
    log.info(
        "integrate_patch: wrote framework KB record to %s (outcome=%s pr_url=%s tps_delta=%+.2f%%)",
        written,
        lesson.outcome,
        pr_url,
        float(lesson.tps_delta_pct),
    )
