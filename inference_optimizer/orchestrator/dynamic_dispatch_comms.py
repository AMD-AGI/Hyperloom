# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""File-based communication protocol for dynamically dispatched specialists.

Each agent gets a workspace at <session_dir>/agents/<agent_id>/ with:
  manifest.json   - task manifest (written by orchestrator at dispatch)
  heartbeat.json  - periodic liveness signal (written by agent)
  results.jsonl   - incremental results (appended by agent)
  done.json       - completion report (written by agent on exit)
  patches/        - generated patches (written by agent)
  new_knowledge.md - lessons learned (optional, written by agent)

The orchestrator polls heartbeat.json and done.json to track lifecycle.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskManifest:
    """Task manifest written by orchestrator at dispatch time."""

    agent_id: str
    task_description: str
    session_dir: str = ""
    dispatched_at: str = ""
    timeout_minutes: int = 120
    metadata: dict[str, Any] = field(default_factory=dict)
    # PID of the spawned claude subprocess, persisted so the Coordinator's
    # per-tick reaper can SIGTERM/SIGKILL the process group of an overdue
    # or stale agent even across resumes / different event-loop ticks.
    pid: int | None = None


@dataclass
class AgentHeartbeat:
    """Periodic liveness signal from agent."""

    agent_id: str
    timestamp: float
    status: str = "running"
    progress_note: str = ""
    iteration: int = 0


@dataclass
class CompletionReport:
    """Final completion report written by agent."""

    agent_id: str
    status: str  # "success" | "failed" | "timeout"
    summary: str = ""
    error: str = ""
    completed_at: str = ""
    config_changes: dict[str, Any] | None = None
    patches_written: list[str] | None = None
    new_knowledge: str = ""
    metrics: dict[str, Any] | None = None


@dataclass
class IncrementalResult:
    """A single incremental result from an agent."""

    agent_id: str
    category: str  # "config_change", "code_patch", "finding", "benchmark"
    description: str
    impact: str = "unknown"  # "high", "medium", "low", "unknown"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


# ─── Write functions ───────────────────────────────────────────────────────────


def write_task_manifest(session_dir: str, manifest: TaskManifest) -> Path:
    """Write task manifest at dispatch time."""
    agent_dir = Path(session_dir) / "agents" / manifest.agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "manifest.json"
    _atomic_write(path, asdict(manifest))
    return path


def write_heartbeat(session_dir: str, heartbeat: AgentHeartbeat) -> None:
    """Write heartbeat (called by agent periodically)."""
    agent_dir = Path(session_dir) / "agents" / heartbeat.agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "heartbeat.json"
    _atomic_write(path, asdict(heartbeat))


def write_completion(session_dir: str, report: CompletionReport) -> None:
    """Write completion report (called by agent on exit)."""
    agent_dir = Path(session_dir) / "agents" / report.agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "done.json"
    _atomic_write(path, asdict(report))


def append_result(session_dir: str, result: IncrementalResult) -> None:
    """Append an incremental result to results.jsonl."""
    agent_dir = Path(session_dir) / "agents" / result.agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "results.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result), default=str) + "\n")


# ─── Read functions ────────────────────────────────────────────────────────────


def read_manifest(session_dir: str, agent_id: str) -> TaskManifest | None:
    """Read the task manifest for an agent (None if missing / unreadable)."""
    path = Path(session_dir) / "agents" / agent_id / "manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return TaskManifest(**{k: v for k, v in data.items()
                               if k in TaskManifest.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError, OSError):
        return None


def read_heartbeat(session_dir: str, agent_id: str) -> AgentHeartbeat | None:
    """Read latest heartbeat for an agent."""
    path = Path(session_dir) / "agents" / agent_id / "heartbeat.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return AgentHeartbeat(**{k: v for k, v in data.items()
                                 if k in AgentHeartbeat.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError):
        return None


def read_completion(session_dir: str, agent_id: str) -> CompletionReport | None:
    """Read completion report for an agent (None if not yet done)."""
    path = Path(session_dir) / "agents" / agent_id / "done.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CompletionReport(**{k: v for k, v in data.items()
                                   if k in CompletionReport.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError):
        return None


def read_agent_results(session_dir: str, agent_id: str) -> list[IncrementalResult]:
    """Read all incremental results from an agent."""
    path = Path(session_dir) / "agents" / agent_id / "results.jsonl"
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                results.append(IncrementalResult(
                    **{k: v for k, v in data.items()
                       if k in IncrementalResult.__dataclass_fields__}
                ))
            except (json.JSONDecodeError, TypeError):
                continue
    return results


def is_agent_alive(session_dir: str, agent_id: str, timeout_s: float = 600.0) -> bool:
    """Check if agent is alive based on heartbeat or process log activity."""
    if read_completion(session_dir, agent_id) is not None:
        return False
    hb = read_heartbeat(session_dir, agent_id)
    if hb is not None:
        return (time.time() - hb.timestamp) < timeout_s
    agent_dir = Path(session_dir) / "agents" / agent_id
    log_path = agent_dir / "process.log"
    if log_path.exists():
        mtime = os.path.getmtime(log_path)
        if (time.time() - mtime) < timeout_s:
            return True
    return False


def get_agent_status_summary(session_dir: str) -> dict[str, list[str]]:
    """Get summary of all agents: active, completed, dead."""
    agents_dir = Path(session_dir) / "agents"
    if not agents_dir.exists():
        return {"active": [], "completed": [], "dead": []}

    active: list[str] = []
    completed: list[str] = []
    dead: list[str] = []

    for agent_dir in agents_dir.iterdir():
        if not agent_dir.is_dir():
            continue
        agent_id = agent_dir.name
        report = read_completion(session_dir, agent_id)
        if report:
            completed.append(agent_id)
        elif is_agent_alive(session_dir, agent_id):
            active.append(agent_id)
        else:
            hb = read_heartbeat(session_dir, agent_id)
            if hb:
                dead.append(agent_id)
            elif (agent_dir / "manifest.json").exists():
                dead.append(agent_id)

    return {"active": active, "completed": completed, "dead": dead}


def collect_patches(session_dir: str, agent_id: str) -> list[Path]:
    """Collect all patch files written by an agent."""
    patches_dir = Path(session_dir) / "agents" / agent_id / "patches"
    if not patches_dir.is_dir():
        return []
    return sorted(patches_dir.glob("*.patch")) + sorted(patches_dir.glob("*.diff"))


# ─── Internal ──────────────────────────────────────────────────────────────────


def _atomic_write(path: Path, data: Any) -> None:
    """Atomically write JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)
