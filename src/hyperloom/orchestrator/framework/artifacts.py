# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""FRAMEWORK candidate-level artifacts + outcome classification (Step 1).

Deterministic, LLM-free observability helpers for the FRAMEWORK_AGENT phase:

- :func:`write_decision_json` drops a uniform ``decision.json`` under
  ``runs/framework_agent/<slug>/`` for every candidate terminal event
  (critic-denied, executor KEEP/REVERT/apply_failed/..., authored-patch
  KEEP/REVERT). This is the single per-candidate fate record an operator
  or downstream tool can read without parsing the whole event log.
- :func:`summarize_candidate_outcomes` classifies a batch's progress rows
  into ``empty_discovery`` / ``tested_no_keep`` / ``tested_with_keep`` so the
  phase-done summary, report, and robustness advisory can tell "discovered
  nothing" apart from "tested candidates but none cleared the gate".

Both are pure / best-effort: a write failure never raises into the pump.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session_paths import runs_dir


log = logging.getLogger(__name__)


# Per-candidate terminal statuses that mean the candidate actually reached the
# apply/bench stage (as opposed to being filtered before any source change).
_TESTED_STATUSES: frozenset[str] = frozenset(
    {"kept", "reverted", "applied_no_bench", "apply_failed", "bench_reverted"}
)


def candidate_key(row: dict[str, Any] | None) -> str:
    """Canonical dedup/progress key for a FRAMEWORK candidate or progress row.

    The single source of truth for "which candidate is this" across the whole
    FRAMEWORK_AGENT pump: candidate selection, dedup, progress-row keying, task
    idempotency, and the known-id set all derive from this so a candidate that
    carries only a ``pr_url`` (no ``candidate_id``) can never dedup against a
    progress row keyed on its ``candidate_id`` (and vice-versa).

    Precedence mirrors the historical inline ``candidate_id or pr_url or ref``
    fallback. Progress rows persist this value in their ``candidate_id`` field,
    so passing a progress row back through here is idempotent.

    Args:
        row: A candidate dict or ``framework_agent_phase_progress`` row (or
            ``None``).

    Returns:
        The candidate key, or ``""`` when none of the identity fields are set.
    """
    if not isinstance(row, dict):
        return ""
    return str(row.get("candidate_id") or row.get("pr_url") or row.get("ref") or "")


def candidate_slug(candidate_id: str) -> str:
    """Filesystem-safe slug for a candidate id (PR url / ref / synthetic id).

    Args:
        candidate_id: The candidate identifier (may contain ``/``, ``:`` …).

    Returns:
        A lowercased slug with non-``[a-z0-9._-]`` runs collapsed to ``-``,
        capped at 96 chars, defaulting to ``"candidate"`` when empty.
    """
    out: list[str] = []
    for ch in str(candidate_id).lower():
        out.append(ch if (ch.isalnum() or ch in ".-_") else "-")
    slug = "".join(out).strip("-")
    return (slug or "candidate")[:96]


def write_decision_json(
    session_dir: Path | str,
    *,
    candidate_id: str,
    batch_id: str = "",
    status: str,
    kept: bool = False,
    reason: str = "",
    provenance: str = "",
    gain_pct: float | None = None,
    accuracy_pass: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Write ``runs/framework_agent/<slug>/decision.json`` for one candidate.

    Best-effort: returns the written path, or ``None`` on any failure (never
    raises — observability must not wedge the pump).

    Args:
        session_dir: The session root directory.
        candidate_id: The candidate identifier (used for the slug + payload).
        batch_id: The discovery batch this candidate belonged to.
        status: The terminal status (e.g. ``kept`` / ``reverted`` /
            ``critic_denied`` / ``apply_failed`` / ``already_present``).
        kept: Whether the candidate was promoted into the stack.
        reason: Human-readable rationale (critic rationale, failure text, …).
        provenance: ``raw_diff`` / ``authored`` / ``critic`` / ``audit`` …
        gain_pct: Measured throughput delta vs baseline, when benched.
        accuracy_pass: Accuracy-gate verdict, when evaluated.
        extra: Optional additional fields merged into the payload.

    Returns:
        The absolute path to the written ``decision.json``, or ``None``.
    """
    try:
        slug = candidate_slug(candidate_id)
        out_dir = runs_dir(Path(session_dir), "framework_agent", slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "candidate_id": str(candidate_id),
            "batch_id": str(batch_id or ""),
            "status": str(status or ""),
            "kept": bool(kept),
            "provenance": str(provenance or ""),
            "reason": str(reason or ""),
            "gain_pct": (float(gain_pct) if isinstance(gain_pct, (int, float)) else None),
            "accuracy_pass": (bool(accuracy_pass) if isinstance(accuracy_pass, bool) else None),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(extra, dict):
            for k, v in extra.items():
                payload.setdefault(str(k), v)
        dest = out_dir / "decision.json"
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(dest)
    except Exception:  # noqa: BLE001 — observability is best-effort
        log.debug("framework_agent_artifacts: write_decision_json failed", exc_info=True)
        return None


def write_semantic_audit(
    session_dir: Path | str,
    *,
    candidate_id: str,
    verdict: dict[str, Any],
) -> str | None:
    """Persist a candidate's semantic-audit verdict next to its decision.json.

    Writes ``semantic_audit.json`` + a readable ``semantic_audit.md`` under
    ``runs/framework_agent/<slug>/`` (G9 — co-located with decision.json).
    Best-effort: returns the JSON path, or ``None`` on failure.

    Args:
        session_dir: The session root directory.
        candidate_id: The candidate identifier (slug source).
        verdict: The ``fa phase-audit`` verdict dict.

    Returns:
        The absolute path to ``semantic_audit.json``, or ``None``.
    """
    if not isinstance(verdict, dict) or not verdict:
        return None
    try:
        slug = candidate_slug(candidate_id)
        out_dir = runs_dir(Path(session_dir), "framework_agent", slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "semantic_audit.json"
        json_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
        lines = [
            f"# Semantic audit — {candidate_id}",
            "",
            f"- semantic_status: {verdict.get('semantic_status')}",
            f"- applicability: {verdict.get('applicability')}",
            f"- recommended_next_step: {verdict.get('recommended_next_step')}",
            f"- confidence: {verdict.get('confidence')}",
            f"- layer: {verdict.get('layer')}",
            "",
            "## Evidence",
        ]
        for ev in verdict.get("evidence") or []:
            if isinstance(ev, dict):
                lines.append(
                    f"- {ev.get('local_file') or '(file?)'}"
                    + (f" [{ev.get('symbol')}]" if ev.get("symbol") else "")
                    + (f": {ev.get('reason')}" if ev.get("reason") else "")
                )
        risks = verdict.get("risks") or []
        if risks:
            lines += ["", "## Risks", *[f"- {r}" for r in risks]]
        (out_dir / "semantic_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(json_path)
    except Exception:  # noqa: BLE001 — observability is best-effort
        log.debug("framework_agent_artifacts: write_semantic_audit failed", exc_info=True)
        return None


def summarize_candidate_outcomes(
    progress: list[dict[str, Any]] | None,
    *,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Classify FRAMEWORK progress rows into a phase-outcome summary.

    Args:
        progress: ``framework_agent_phase_progress`` rows (each a dict carrying
            ``status`` / ``kept`` / ``batch_id``).
        batch_id: When set, only rows for this batch are counted; otherwise all
            rows are counted.

    Returns:
        ``{"total", "keeps", "tested", "by_status", "outcome_class"}`` where
        ``outcome_class`` is one of ``empty_discovery`` (no rows),
        ``tested_with_keep`` (>=1 KEEP), or ``tested_no_keep``.
    """
    rows = [r for r in (progress or []) if isinstance(r, dict)]
    if batch_id is not None:
        rows = [r for r in rows if str(r.get("batch_id") or "") == str(batch_id)]
    by_status: dict[str, int] = {}
    keeps = 0
    tested = 0
    for r in rows:
        st = str(r.get("status") or "")
        by_status[st] = by_status.get(st, 0) + 1
        if bool(r.get("kept")) or st == "kept":
            keeps += 1
        if st in _TESTED_STATUSES:
            tested += 1
    if not rows:
        outcome_class = "empty_discovery"
    elif keeps > 0:
        outcome_class = "tested_with_keep"
    else:
        outcome_class = "tested_no_keep"
    return {
        "total": len(rows),
        "keeps": keeps,
        "tested": tested,
        "by_status": by_status,
        "outcome_class": outcome_class,
    }


__all__ = [
    "candidate_key",
    "candidate_slug",
    "summarize_candidate_outcomes",
    "write_decision_json",
    "write_semantic_audit",
]
