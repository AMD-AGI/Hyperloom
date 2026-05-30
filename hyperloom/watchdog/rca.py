"""RCA — Tier 2 root-cause analysis.

For events that Tier 0 (triage) cannot classify deterministically,
this module prepares context and builds a prompt for an LLM-based RCA agent.

The RCA agent analyzes:
  - Event timeline reconstruction
  - Config diff analysis
  - Log snippet extraction
  - Hypothesis generation and ranking
  - Actionable recommendation output

RCA findings are persisted to <session_dir>/rca_reports/<event_id>.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.watchdog.event_log import read_events


@dataclass
class RCARequest:
    """Context bundle for an RCA analysis."""

    event_id: str
    event: dict[str, Any]
    session_dir: str
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)
    config_diff: str = ""


@dataclass
class RCAFinding:
    """Result of an RCA analysis."""

    event_id: str
    root_cause: str
    confidence: str  # "high" | "medium" | "low"
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""
    action: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def prepare_rca_context(
    event: dict[str, Any],
    session_dir: str,
    session_state: dict[str, Any] | None = None,
    window: int = 20,
) -> RCARequest:
    """Gather context for an RCA analysis."""
    recent = read_events(session_dir, limit=window)

    config_diff = ""
    state_file = Path(session_dir) / "state.json"
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text())
            actions = state_data.get("actions", [])
            if actions:
                last_action = actions[-1]
                config_diff = json.dumps(last_action, indent=2)[:2000]
        except (json.JSONDecodeError, KeyError):
            pass

    return RCARequest(
        event_id=event.get("event_id", "unknown"),
        event=event,
        session_dir=session_dir,
        recent_events=recent,
        session_state=session_state or {},
        config_diff=config_diff,
    )


def build_rca_prompt(request: RCARequest) -> str:
    """Build the prompt for an RCA agent."""
    timeline_str = ""
    for e in request.recent_events[-10:]:
        ts = e.get("timestamp", "?")
        etype = e.get("type", "?")
        sev = e.get("severity", "info")
        details = str(e.get("details", {}))[:200]
        timeline_str += f"  [{ts}] {sev.upper()} {etype}: {details}\n"

    return (
        "## Root Cause Analysis Request\n\n"
        f"**Triggering Event:**\n"
        f"  Type: {request.event.get('type', '?')}\n"
        f"  Severity: {request.event.get('severity', '?')}\n"
        f"  Details: {json.dumps(request.event.get('details', {}), indent=2)[:500]}\n\n"
        f"**Recent Event Timeline:**\n{timeline_str}\n"
        f"**Last Config Change:**\n```\n{request.config_diff[:1000]}\n```\n\n"
        "**Your Task:**\n"
        "1. Identify the root cause of the triggering event\n"
        "2. Rate your confidence (high/medium/low)\n"
        "3. List evidence supporting your conclusion\n"
        "4. Recommend ONE specific action (not 'investigate further')\n\n"
        "Write your finding as JSON with keys: "
        "root_cause, confidence, evidence (list), recommendation, action\n"
    )


def save_rca_finding(session_dir: str, finding: RCAFinding) -> Path:
    """Persist an RCA finding to disk."""
    rca_dir = Path(session_dir) / "rca_reports"
    rca_dir.mkdir(parents=True, exist_ok=True)
    path = rca_dir / f"{finding.event_id}.json"
    path.write_text(json.dumps(asdict(finding), indent=2, default=str) + "\n")
    return path


def load_rca_findings(session_dir: str) -> list[RCAFinding]:
    """Load all RCA findings for the session."""
    rca_dir = Path(session_dir) / "rca_reports"
    if not rca_dir.exists():
        return []
    findings = []
    for path in sorted(rca_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            findings.append(RCAFinding(**{
                k: v for k, v in data.items() if k in RCAFinding.__dataclass_fields__
            }))
        except (json.JSONDecodeError, TypeError):
            continue
    return findings
