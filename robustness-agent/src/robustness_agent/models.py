# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared data models used across providers, monitors, and checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Metrics snapshots
# ---------------------------------------------------------------------------

@dataclass
class GpuSnapshot:
    """One-time GPU metrics snapshot collected by monitors."""
    gpu_id: int
    utilization: float
    vram_used_mb: float
    vram_total_mb: float
    temperature_c: float
    power_watts: float
    ecc_errors: int = 0
    timestamp: float = 0.0

    @property
    def vram_used_pct(self) -> float:
        """VRAM utilization percentage (0 if total is zero).

        Returns:
            float: Used VRAM as a percentage of total, or ``0.0`` when
            the total is non-positive.
        """
        if self.vram_total_mb <= 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100.0


@dataclass
class ProcessInfo:
    """Minimal process info surfaced by process monitors."""
    pid: int
    state: str
    cmd: str
    rss_mb: float = 0.0
    gpu_id: Optional[int] = None


@dataclass
class DiskSnapshot:
    """Disk usage snapshot for a single mount."""
    mount: str
    total_gb: float
    used_gb: float
    available_gb: float

    @property
    def used_pct(self) -> float:
        """Disk utilization percentage (0 if total is zero).

        Returns:
            float: Used space as a percentage of total, or ``0.0`` when
            the total is non-positive.
        """
        if self.total_gb <= 0:
            return 0.0
        return (self.used_gb / self.total_gb) * 100.0


@dataclass
class ServerHealthStatus:
    """HTTP health probe result for a service endpoint."""
    url: str
    reachable: bool
    response_time_ms: float = 0.0
    status_code: int = 0
    error: str = ""


@dataclass
class FaultEvent:
    """Raw fault event emitted by monitors for downstream checks."""
    monitor_id: str
    category: str
    severity: str
    message: str
    timestamp: float = 0.0
    node: str = ""


# ---------------------------------------------------------------------------
# Alert / Finding
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Severity level attached to an alert.

    Attributes:
        INFO (str): Informational, no action required.
        WARNING (str): Degraded condition worth attention.
        CRITICAL (str): Severe condition requiring intervention.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CheckResult(str, Enum):
    """Outcome classification for a completed check.

    Attributes:
        OK (str): The check passed with no issues.
        WARN (str): The check found a non-critical issue.
        CRITICAL (str): The check found a critical issue.
    """

    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Structured alert emitted by checks and forwarded to the Coordinator."""
    check_name: str
    severity: Severity
    summary: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class RcaFinding:
    """RCA output capturing root cause, recommended action, and evidence."""
    trigger_alerts: list[Alert]
    root_cause: str
    suggested_action: str
    action_type: str = ""  # kill_task / prune_branch / escalate_strategy_change / recover
    action_payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_summary: str = ""


# ---------------------------------------------------------------------------
# Conductor events (subset we consume)
# ---------------------------------------------------------------------------

@dataclass
class ConductorEvent:
    """Subset of Conductor events consumed by the robustness agent."""
    event_id: int
    agent: str
    intent_type: str
    payload: dict[str, Any]
    timestamp: float
    topic: str = ""
