"""Watchdog scanner — polls the event log and dispatches triage/RCA workflows.

Runs as a background thread during an optimization session. Reads new events,
runs Tier 0 triage, optionally Tier 1 bench integrity, and escalates to
Tier 2 RCA when needed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hyperloom.watchdog.event_log import read_new_events, append_event
from hyperloom.watchdog.triage import triage_event, TriageResult
from hyperloom.watchdog.bench_integrity import BenchIntegrityChecker, IntegrityVerdict
from hyperloom.watchdog.rca import prepare_rca_context, build_rca_prompt, RCAFinding

logger = logging.getLogger("hyperloom.watchdog")

ActionCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ScannerStats:
    events_processed: int = 0
    triage_known: int = 0
    triage_needs_rca: int = 0
    triage_info: int = 0
    bench_integrity_fails: int = 0
    rca_dispatched: int = 0
    actions_emitted: int = 0


@dataclass
class WatchdogScanner:
    """Main watchdog scanner. Call start() to run in background thread,
    or scan_once() for a single pass.

    Parameters:
        session_dir:     path to the active session directory
        action_callback: called when the watchdog determines an action is needed
        rca_callback:    optional; called to dispatch an LLM RCA agent
        poll_interval:   seconds between polling passes (default 5)
    """

    session_dir: str
    action_callback: ActionCallback | None = None
    rca_callback: Callable[[str, str], None] | None = None
    poll_interval: float = 5.0
    stats: ScannerStats = field(default_factory=ScannerStats)

    _offset: int = 0
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    _bench_checker: BenchIntegrityChecker = field(
        default_factory=BenchIntegrityChecker, repr=False
    )

    def start(self) -> None:
        """Start the scanner in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="watchdog-scanner"
        )
        self._thread.start()
        logger.info("Watchdog scanner started (poll=%ss)", self.poll_interval)

    def stop(self) -> None:
        """Stop the background scanner."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval * 2)
            self._thread = None
        logger.info("Watchdog scanner stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def scan_once(self) -> list[TriageResult]:
        """Run a single scan pass. Returns triage results for new events."""
        events, new_offset = read_new_events(self.session_dir, self._offset)
        self._offset = new_offset

        results: list[TriageResult] = []
        for event in events:
            result = self._process_event(event)
            if result:
                results.append(result)

        return results

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.scan_once()
            except Exception:
                logger.exception("Watchdog scan error")
            self._stop_event.wait(self.poll_interval)

    def _process_event(self, event: dict[str, Any]) -> TriageResult | None:
        self.stats.events_processed += 1

        result = triage_event(event)

        if result.classification == "known_pattern":
            self.stats.triage_known += 1
            self._emit_action(result.action, event)
            return result

        if result.classification == "bench_integrity":
            details = event.get("details", {})
            if details.get("output_throughput") is not None:
                verdict = self._bench_checker.check(details)
                if not verdict.passed:
                    self.stats.bench_integrity_fails += 1
                    append_event(
                        self.session_dir,
                        source="watchdog",
                        event_type="bench_integrity_warning",
                        severity="warning",
                        details={
                            "errors": verdict.errors,
                            "warnings": verdict.warnings,
                            "details": verdict.details,
                        },
                    )
                    self._emit_action("bench_integrity_failed", {
                        "event": event,
                        "verdict_errors": verdict.errors,
                    })
            return result

        if result.classification == "needs_rca":
            self.stats.triage_needs_rca += 1
            self._dispatch_rca(event)
            return result

        self.stats.triage_info += 1
        return result

    def _emit_action(self, action: str, context: dict[str, Any]) -> None:
        self.stats.actions_emitted += 1
        if self.action_callback:
            try:
                self.action_callback(action, context)
            except Exception:
                logger.exception("Action callback error for %s", action)

    def _dispatch_rca(self, event: dict[str, Any]) -> None:
        if not self.rca_callback:
            logger.warning(
                "RCA needed for event %s but no rca_callback configured",
                event.get("event_id"),
            )
            return

        self.stats.rca_dispatched += 1
        rca_req = prepare_rca_context(event, self.session_dir)
        prompt = build_rca_prompt(rca_req)

        try:
            self.rca_callback(rca_req.event_id, prompt)
        except Exception:
            logger.exception("RCA dispatch failed for event %s", rca_req.event_id)
