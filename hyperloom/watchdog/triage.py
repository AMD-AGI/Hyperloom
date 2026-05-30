"""Triage — Tier 0 deterministic event classifier.

Classifies events by pattern-matching known error signatures.
Zero LLM tokens. This is the first line of defense.

Returns a TriageResult with:
  - classification:  "known_pattern" | "bench_integrity" | "needs_rca" | "info_only"
  - action:          what the orchestrator should do about it
  - rca_requested:   whether to escalate to Tier 2 LLM-based RCA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TriageResult:
    classification: str  # "known_pattern" | "bench_integrity" | "needs_rca" | "info_only"
    action: str
    description: str
    rca_requested: bool = False
    confidence: str = "high"  # "high" | "medium" | "low"


_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "oom_kill",
        "match": lambda e: (
            "oom" in str(e.get("details", {})).lower()
            or "out of memory" in str(e.get("details", {})).lower()
            or e.get("type") == "oom"
        ),
        "action": "reduce_batch_size_or_tp",
        "description": "GPU OOM — reduce batch size, increase TP, or revert config",
    },
    {
        "name": "server_crash",
        "match": lambda e: e.get("type") in ("server_crash", "server_died", "process_exit"),
        "action": "restart_server",
        "description": "Inference server crashed — restart with last-known-good config",
    },
    {
        "name": "bench_zero_completions",
        "match": lambda e: (
            e.get("type") == "benchmark"
            and e.get("details", {}).get("completed", -1) == 0
        ),
        "action": "check_server_health",
        "description": "Benchmark completed 0 requests — server may be hung or unreachable",
    },
    {
        "name": "timeout",
        "match": lambda e: (
            e.get("type") in ("timeout", "benchmark_timeout")
            or "timed out" in str(e.get("details", {})).lower()
        ),
        "action": "extend_timeout_or_check_server",
        "description": "Operation timed out — server overloaded or config too aggressive",
    },
    {
        "name": "accuracy_regression",
        "match": lambda e: (
            e.get("type") == "accuracy_gate_failed"
            or (e.get("type") == "gate_result"
                and not e.get("details", {}).get("accuracy_passed", True))
        ),
        "action": "revert_last_change",
        "description": "Accuracy regressed beyond tolerance — revert",
    },
    {
        "name": "throughput_regression",
        "match": lambda e: (
            e.get("type") == "gate_result"
            and not e.get("details", {}).get("throughput_passed", True)
        ),
        "action": "revert_last_change",
        "description": "Throughput regressed — revert last config change",
    },
    {
        "name": "agent_failure",
        "match": lambda e: e.get("type") in ("agent_crash", "agent_error", "agent_timeout"),
        "action": "classify_and_retry_agent",
        "description": "Specialist agent failed — classify failure and retry with escalation",
    },
    {
        "name": "gpu_thermal",
        "match": lambda e: (
            "thermal" in str(e.get("details", {})).lower()
            or "throttl" in str(e.get("details", {})).lower()
            or e.get("type") == "gpu_thermal"
        ),
        "action": "reduce_load_or_wait",
        "description": "GPU thermal throttling detected — reduce load or pause",
    },
    {
        "name": "disk_full",
        "match": lambda e: (
            "no space" in str(e.get("details", {})).lower()
            or "disk full" in str(e.get("details", {})).lower()
            or e.get("type") == "disk_full"
        ),
        "action": "cleanup_disk",
        "description": "Disk full — clean up profiles, logs, or checkpoints",
    },
    {
        "name": "nccl_rccl_error",
        "match": lambda e: (
            "nccl" in str(e.get("details", {})).lower()
            or "rccl" in str(e.get("details", {})).lower()
        ),
        "action": "restart_server_fresh",
        "description": "NCCL/RCCL collective error — fresh server restart required",
    },
]


def triage_event(event: dict[str, Any]) -> TriageResult:
    """Classify an event using deterministic pattern matching.

    Returns a TriageResult. If no pattern matches and severity >= warning,
    escalates to Tier 2 RCA.
    """
    for pattern in _PATTERNS:
        try:
            if pattern["match"](event):
                return TriageResult(
                    classification="known_pattern",
                    action=pattern["action"],
                    description=pattern["description"],
                    rca_requested=False,
                )
        except Exception:
            continue

    if event.get("type") == "benchmark":
        return TriageResult(
            classification="bench_integrity",
            action="validate_benchmark",
            description="Benchmark event — run integrity checks",
        )

    severity = event.get("severity", "info")
    if severity in ("error", "critical"):
        return TriageResult(
            classification="needs_rca",
            action="dispatch_rca",
            description=f"Unclassified {severity} event — escalate to RCA",
            rca_requested=True,
            confidence="low",
        )

    return TriageResult(
        classification="info_only",
        action="none",
        description="Informational event — no action needed",
    )
