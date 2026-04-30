"""SharedState — DESIGN §6.3 / §15.

Single struct held by the Conductor that every reactor reads (after policy
gate filtering). Persistence: ``state.json`` snapshotted at 30 min cadence
and after every KEEP — but it is NOT the source of truth; SQLite is
(see DESIGN §13.2 SoT table).

STATUS:
    Minimum-viable implementation. RCA / transition-application paths
    accept calls but are no-op for the dry-run; the full versions land
    in IMPLEMENTATION-CHECKLIST Phase 3 §3.4 / §3.6.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .execution_mode import ExecutionMode


@dataclass
class SharedState:
    """Read-mostly state struct injected into prompts (DESIGN §6.3)."""

    session_id: str = ""
    model_path: str = ""
    model_name: str = ""
    model_class: str = ""
    cwd: str = ""

    start_ts: float = field(default_factory=time.time)
    max_minutes: float = 0.0
    elapsed_minutes: float = 0.0

    baseline_tput: float = 0.0
    current_tput: float = 0.0
    cumulative_gain: float = 0.0  # percentage points
    baseline_accuracy: float | None = None

    crash_count: int = 0
    current_action: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.QUICK_PARAM_SWEEP

    stop_reason: str | None = None

    decisions: list[dict[str, Any]] = field(default_factory=list)
    rca_findings: list[dict[str, Any]] = field(default_factory=list)

    family_failure_streak: dict[str, int] = field(default_factory=dict)
    pruned_families: set[str] = field(default_factory=set)

    # ----------------------------------------------------------------------
    # derived properties
    # ----------------------------------------------------------------------
    @property
    def time_left_minutes(self) -> float:
        return max(0.0, self.max_minutes - self.elapsed_minutes)

    def refresh_elapsed(self) -> None:
        """Update ``elapsed_minutes`` from ``start_ts``. Called by clock."""
        self.elapsed_minutes = (time.time() - self.start_ts) / 60.0

    # ----------------------------------------------------------------------
    # mutation API
    # ----------------------------------------------------------------------
    def attach_rca(self, finding: dict[str, Any]) -> None:
        """Append an RCA finding. Full integration with PolicyGate landing
        with IMPL-CHECKLIST §3.4 — for the dry-run this just records."""
        self.rca_findings.append(dict(finding))

    # ----------------------------------------------------------------------
    # family-level dead-end pruning (DESIGN follow-up: scheduling problem #6)
    # ----------------------------------------------------------------------
    FAMILY_FAILURE_PRUNE_THRESHOLD: int = 3

    def record_action_outcome(self, family: str | None, succeeded: bool) -> None:
        """Track per-family failure streaks. After
        ``FAMILY_FAILURE_PRUNE_THRESHOLD`` consecutive failures the family
        is added to ``pruned_families`` so the scheduler can skip it.
        A success resets the streak and lifts any prune for that family.
        """
        if not family:
            return
        if succeeded:
            self.family_failure_streak[family] = 0
            self.pruned_families.discard(family)
            return
        cur = self.family_failure_streak.get(family, 0) + 1
        self.family_failure_streak[family] = cur
        if cur >= self.FAMILY_FAILURE_PRUNE_THRESHOLD:
            self.pruned_families.add(family)

    def is_family_pruned(self, family: str | None) -> bool:
        return bool(family) and family in self.pruned_families

    def unprune_family(self, family: str) -> None:
        """Manual override (used by triage's ``prune_branch`` un-prune
        flow if it ever needs to revive a family)."""
        self.pruned_families.discard(family)
        self.family_failure_streak.pop(family, None)

    def last_decisions(self, n: int = 5) -> list[dict[str, Any]]:
        return list(self.decisions[-n:])

    def append_decision(self, decision: dict[str, Any]) -> None:
        self.decisions.append(dict(decision))

    def apply_validated_transition(
        self, from_agent: str, changes: dict[str, Any]
    ) -> None:
        """Apply a state transition that has already passed PolicyGate.

        Dry-run path: only ``current_action``, ``current_tput``,
        ``cumulative_gain``, ``crash_count`` and ``baseline_*`` are accepted.
        Full schema enforcement lands with IMPL-CHECKLIST §3.6.
        """
        allowed = {
            "current_action", "current_tput", "cumulative_gain",
            "crash_count", "baseline_tput", "baseline_accuracy",
        }
        for k, v in changes.items():
            if k not in allowed:
                continue
            setattr(self, k, v)
        self.append_decision({
            "from": from_agent,
            "ts": time.time(),
            "changes": dict(changes),
        })

    def set_stopping(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def should_stop(self) -> bool:
        return self.stop_reason is not None

    # ----------------------------------------------------------------------
    # persistence (NOT SoT — SQLite is)
    # ----------------------------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "model_class": self.model_class,
            "cwd": self.cwd,
            "start_ts": self.start_ts,
            "max_minutes": self.max_minutes,
            "elapsed_minutes": self.elapsed_minutes,
            "baseline_tput": self.baseline_tput,
            "current_tput": self.current_tput,
            "cumulative_gain": self.cumulative_gain,
            "baseline_accuracy": self.baseline_accuracy,
            "crash_count": self.crash_count,
            "current_action": self.current_action,
            "execution_mode": self.execution_mode.value,
            "stop_reason": self.stop_reason,
            "decisions_tail": self.last_decisions(20),
            "rca_findings_tail": list(self.rca_findings[-20:]),
            "family_failure_streak": dict(self.family_failure_streak),
            "pruned_families": sorted(self.pruned_families),
        }

    def write_snapshot(self, session_dir: Path) -> None:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        snap = session_dir / "state.json"
        tmp = session_dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")
        tmp.replace(snap)

    # ----------------------------------------------------------------------
    # prompt rendering
    # ----------------------------------------------------------------------
    def summary(self) -> str:
        time_left = self.time_left_minutes
        gain = self.cumulative_gain
        last = self.last_decisions(3)
        last_md = "\n".join(
            f"  - {d.get('ts', '?')}: {d.get('from', '?')} → "
            f"{d.get('changes', {})}"
            for d in last
        ) or "  (none yet)"
        return (
            f"## Shared state (session={self.session_id})\n"
            f"- model: `{self.model_name or self.model_path}` "
            f"(class={self.model_class or 'unknown'})\n"
            f"- mode: {self.execution_mode.value}\n"
            f"- time: elapsed={self.elapsed_minutes:.2f}m, "
            f"left={time_left:.2f}m, max={self.max_minutes:.2f}m\n"
            f"- throughput: baseline={self.baseline_tput:.2f}, "
            f"current={self.current_tput:.2f}, "
            f"cumulative_gain={gain:.2f}%\n"
            f"- crashes: {self.crash_count}\n"
            f"- current_action: {self.current_action or '(idle)'}\n"
            f"- stop_reason: {self.stop_reason or '(running)'}\n"
            f"- last decisions:\n{last_md}\n"
        )
