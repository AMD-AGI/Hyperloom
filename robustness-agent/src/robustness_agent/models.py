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
        if self.vram_total_mb <= 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100.0


@dataclass
class ProcessInfo:
    pid: int
    state: str
    cmd: str
    rss_mb: float = 0.0
    gpu_id: Optional[int] = None


@dataclass
class DiskSnapshot:
    mount: str
    total_gb: float
    used_gb: float
    available_gb: float

    @property
    def used_pct(self) -> float:
        if self.total_gb <= 0:
            return 0.0
        return (self.used_gb / self.total_gb) * 100.0


@dataclass
class ServerHealthStatus:
    url: str
    reachable: bool
    response_time_ms: float = 0.0
    status_code: int = 0
    error: str = ""


@dataclass
class FaultEvent:
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
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CheckResult(str, Enum):
    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Alert:
    check_name: str
    severity: Severity
    summary: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class RcaFinding:
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
    event_id: int
    agent: str
    intent_type: str
    payload: dict[str, Any]
    timestamp: float
    topic: str = ""
