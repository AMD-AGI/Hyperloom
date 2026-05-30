"""Watchdog — multi-tier event monitoring and automated triage.

Architecture:
    event_log.jsonl (append-only event stream)
        |
        v
    WatchdogScanner (background poll loop)
        |
        +---> Tier 0: triage.py (deterministic pattern matching, zero LLM tokens)
        |       +---> known_pattern -> action callback
        |       +---> needs_rca    -> Tier 2
        |       +---> info_only    -> log and skip
        |
        +---> Tier 1: bench_integrity.py (statistical sanity checks on benchmarks)
        |       +---> integrity errors -> action callback + warning event
        |
        +---> Tier 2: rca.py (LLM-based root cause analysis via dispatch)
                +---> RCA finding -> action callback
"""

from hyperloom.watchdog.event_log import append_event, read_events, read_new_events
from hyperloom.watchdog.scanner import WatchdogScanner
from hyperloom.watchdog.triage import triage_event, TriageResult
from hyperloom.watchdog.bench_integrity import BenchIntegrityChecker, IntegrityVerdict

__all__ = [
    "append_event",
    "read_events",
    "read_new_events",
    "WatchdogScanner",
    "triage_event",
    "TriageResult",
    "BenchIntegrityChecker",
    "IntegrityVerdict",
]
