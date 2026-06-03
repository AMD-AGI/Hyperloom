"""Event-driven check — reacts to specific event patterns from Conductor."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from ..config import Config
from ..models import Alert, ConductorEvent, Severity

log = logging.getLogger(__name__)


class EventCheck:
    """Analyze Conductor events to detect error patterns and cascading failures."""

    def __init__(self, config: Config):
        """Initialise the event check with empty per-source counters.

        Args:
            config (Config): Agent configuration (retained for future
                tuning of the event thresholds).
        """
        self._config = config
        self._error_counts: dict[str, int] = defaultdict(int)
        self._family_fail_counts: dict[str, int] = defaultdict(int)
        self._keep_revert_tracker: list[tuple[float, str, str]] = []

    def process_events(self, events: list[ConductorEvent]) -> list[Alert]:
        """Process a batch of conductor events and collect alerts.

        Args:
            events (list[ConductorEvent]): Events to analyse this cycle.

        Returns:
            list[Alert]: Alerts raised across all processed events.
        """
        alerts: list[Alert] = []
        for event in events:
            alerts.extend(self._check_event(event))
        return alerts

    def _check_event(self, event: ConductorEvent) -> list[Alert]:
        """Inspect a single event for error/failure/bouncing patterns.

        Tracks repeated high-severity agent alerts, repeated action
        family failures, and keep/revert bouncing.

        Args:
            event (ConductorEvent): The event to inspect.

        Returns:
            list[Alert]: Any alerts triggered by this event.
        """
        alerts: list[Alert] = []

        if event.intent_type == "alert" and event.agent != "robustness":
            severity_str = event.payload.get("severity", "low")
            if severity_str in ("high", "critical"):
                self._error_counts[event.agent] += 1
                if self._error_counts[event.agent] >= 3:
                    alerts.append(Alert(
                        check_name="repeated_agent_errors",
                        severity=Severity.CRITICAL,
                        summary=f"Agent '{event.agent}' has reported "
                                f"{self._error_counts[event.agent]} high-severity alerts",
                        evidence={
                            "agent": event.agent,
                            "count": self._error_counts[event.agent],
                            "latest": event.payload.get("summary", ""),
                        },
                        timestamp=time.time(),
                    ))

        if event.intent_type == "delegate":
            task_status = event.payload.get("status", "")
            family = event.payload.get("family", "")
            if task_status in ("failed", "error", "cancelled"):
                if family:
                    self._family_fail_counts[family] += 1
                    if self._family_fail_counts[family] >= 3:
                        alerts.append(Alert(
                            check_name="family_repeated_failure",
                            severity=Severity.CRITICAL,
                            summary=f"Action family '{family}' has failed "
                                    f"{self._family_fail_counts[family]} times — consider pruning",
                            evidence={
                                "family": family,
                                "fail_count": self._family_fail_counts[family],
                            },
                            timestamp=time.time(),
                        ))

        if event.intent_type == "update_state":
            decision = event.payload.get("decision", "")
            action = event.payload.get("action_name", "")
            if decision in ("keep", "revert"):
                self._keep_revert_tracker.append((event.timestamp, action, decision))
                self._prune_old_decisions(event.timestamp)
                alerts.extend(self._detect_bouncing())

        return alerts

    def _prune_old_decisions(self, now: float) -> None:
        """Drop keep/revert records older than the 30-minute window.

        Args:
            now (float): Current event timestamp used as the window
                anchor.
        """
        cutoff = now - 1800
        self._keep_revert_tracker = [
            (ts, a, d) for ts, a, d in self._keep_revert_tracker if ts > cutoff
        ]

    def _detect_bouncing(self) -> list[Alert]:
        """Detect actions oscillating between keep and revert decisions.

        Returns:
            list[Alert]: A critical alert for each action with at least
            two keeps and two reverts inside the tracking window.
        """
        action_decisions: dict[str, list[str]] = defaultdict(list)
        for _, action, decision in self._keep_revert_tracker:
            action_decisions[action].append(decision)

        alerts: list[Alert] = []
        for action, decisions in action_decisions.items():
            if len(decisions) < 4:
                continue
            keeps = sum(1 for d in decisions if d == "keep")
            reverts = sum(1 for d in decisions if d == "revert")
            if keeps >= 2 and reverts >= 2:
                alerts.append(Alert(
                    check_name="keep_revert_bouncing",
                    severity=Severity.CRITICAL,
                    summary=f"Action '{action}' is bouncing: {keeps} KEEPs and {reverts} REVERTs "
                            f"in last 30min",
                    evidence={
                        "action": action,
                        "keeps": keeps,
                        "reverts": reverts,
                        "decisions": decisions[-6:],
                    },
                    timestamp=time.time(),
                ))
        return alerts
