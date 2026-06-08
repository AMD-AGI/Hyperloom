"""Dynamic specialist dispatch — CPU-only subprocess lifecycle management.

Provides a free-form dispatch path where the orchestration agent can spawn
specialist sub-agents with any natural-language task description, without
being locked to the predefined specialist domain catalogue.

All dynamic specialists are CPU-only (research, code analysis, KB search,
patch generation). They launch immediately with no GPU allocation or
queuing. GPU-bound work (profiling, kernel compilation, microbenchmarks)
should use the existing structured SpecialistRunner path.

Dispatch backend: Claude CLI.
  - Full tool access: Bash, Read, Write, Edit, Grep, Glob, Task, WebSearch
  - Stream-json output for structured logging
  - --add-dir for scoped codebase access
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import choices
from string import ascii_lowercase
from typing import Any

from .dynamic_dispatch_comms import (
    TaskManifest,
    write_task_manifest,
    read_completion,
    read_heartbeat,
    is_agent_alive,
    collect_patches,
)

log = logging.getLogger(__name__)


# ─── Enums ─────────────────────────────────────────────────────────────────────


class FailureType(enum.Enum):
    SUCCESS = "success"
    CRASH = "crash"
    TIMEOUT = "timeout"
    NO_OUTPUT = "no_output"
    IMPORT_ERROR = "import_error"
    UNKNOWN = "unknown"


class TaskPriority(enum.Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ─── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class AgentHandle:
    """Handle to a dispatched specialist with lifecycle tracking."""

    agent_id: str
    pid: int | None = None
    process: subprocess.Popen | None = None
    log_path: Path = field(default_factory=lambda: Path("/dev/null"))
    task_summary: str = ""
    role: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    dispatched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_process_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def runtime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.dispatched_at).total_seconds()


@dataclass
class TaskSpec:
    """Specification for a dynamic specialist task (CPU-only)."""

    prompt: str
    task_summary: str = ""
    role: str = "specialist"
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_minutes: int = 120
    metadata: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2


@dataclass
class FailureInfo:
    """Structured failure diagnosis for a specialist."""

    failure_type: FailureType
    detail: str
    retry_eligible: bool
    escalation_hint: str
    log_snippet: str = ""

    @property
    def failed(self) -> bool:
        return self.failure_type != FailureType.SUCCESS


@dataclass
class DispatchResult:
    """Result of a batch dispatch operation."""

    launched: list[AgentHandle]
    errors: list[str]


# ─── Agent ID generation ───────────────────────────────────────────────────────


def generate_agent_id(role: str = "agent") -> str:
    ts = int(time.time())
    suffix = "".join(choices(ascii_lowercase, k=4))
    return f"{role}-{ts}-{suffix}"


# ─── Core dispatch ─────────────────────────────────────────────────────────────


def dispatch_agent(
    prompt: str,
    session_dir: str,
    task_summary: str = "",
    role: str = "specialist",
    model: str = "claude-sonnet-4-6",
    timeout_minutes: int = 120,
    priority: TaskPriority = TaskPriority.NORMAL,
    metadata: dict | None = None,
    attempt: int = 1,
) -> AgentHandle:
    """Dispatch a single specialist agent via Claude CLI (CPU-only)."""
    agent_id = generate_agent_id(role)

    manifest = TaskManifest(
        agent_id=agent_id,
        task_description=task_summary or prompt[:200],
        session_dir=session_dir,
        dispatched_at=datetime.now(timezone.utc).isoformat(),
        timeout_minutes=timeout_minutes,
        metadata=metadata or {},
    )
    write_task_manifest(session_dir, manifest)

    handle = _dispatch_via_cli(prompt, session_dir, agent_id, model, timeout_minutes)
    handle.task_summary = task_summary
    handle.role = role
    handle.priority = priority
    handle.attempt = attempt
    handle.metadata = metadata or {}

    log.info("Dispatched %s (role=%s, attempt=%d)", agent_id, role, attempt)
    return handle


def dispatch_batch(
    tasks: list[TaskSpec],
    session_dir: str,
    model: str = "claude-sonnet-4-6",
) -> DispatchResult:
    """Dispatch multiple specialists. All launch immediately (CPU-only)."""
    launched: list[AgentHandle] = []
    errors: list[str] = []

    for task in sorted(tasks, key=lambda t: t.priority.value):
        try:
            handle = dispatch_agent(
                prompt=task.prompt,
                session_dir=session_dir,
                task_summary=task.task_summary,
                role=task.role,
                model=model,
                timeout_minutes=task.timeout_minutes,
                priority=task.priority,
                metadata=task.metadata,
            )
            launched.append(handle)
        except Exception as e:
            errors.append(f"Task '{task.task_summary}' failed to dispatch: {e}")

    return DispatchResult(launched=launched, errors=errors)


# ─── Lifecycle management ──────────────────────────────────────────────────────


def reap_completed(
    handles: list[AgentHandle],
    session_dir: str,
    heartbeat_timeout: float = 600.0,
) -> tuple[list[AgentHandle], list[AgentHandle]]:
    """Separate completed agents from still-running ones.

    Returns (still_running, newly_completed).
    """
    still_running: list[AgentHandle] = []
    newly_completed: list[AgentHandle] = []

    for h in handles:
        report = read_completion(session_dir, h.agent_id)
        process_dead = not h.is_process_alive()

        if report is not None or process_dead:
            newly_completed.append(h)
        else:
            hb = read_heartbeat(session_dir, h.agent_id)
            if hb and (time.time() - hb.timestamp) > heartbeat_timeout:
                newly_completed.append(h)
            else:
                still_running.append(h)

    return still_running, newly_completed


# ─── Failure classification ────────────────────────────────────────────────────


_CRASH_MARKERS = [
    "ImportError", "ModuleNotFoundError", "RuntimeError",
    "Traceback", "Killed", "Segmentation fault",
    "signal", "SIGKILL",
]


def classify_failure(
    session_dir: str,
    handle: AgentHandle,
) -> FailureInfo:
    """Diagnose why a specialist failed."""
    report = read_completion(session_dir, handle.agent_id)
    agent_dir = Path(session_dir) / "agents" / handle.agent_id
    log_path = agent_dir / "process.log"

    if report and report.status == "success":
        return FailureInfo(
            failure_type=FailureType.SUCCESS,
            detail="",
            retry_eligible=False,
            escalation_hint="",
        )

    log_tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(errors="replace")
            log_tail = text[-4000:]
        except Exception:
            pass

    if report and report.status == "failed":
        detail = report.error or report.summary or ""
        detail_lower = detail.lower()

        if "timeout" in detail_lower or "timed out" in detail_lower:
            return FailureInfo(
                failure_type=FailureType.TIMEOUT,
                detail=detail[:500],
                retry_eligible=True,
                escalation_hint=(
                    "Break the task into smaller sub-tasks. Focus on the "
                    "single highest-impact item. Reduce scope aggressively."
                ),
                log_snippet=log_tail[-1000:],
            )

        if "import" in detail_lower or "module" in detail_lower:
            return FailureInfo(
                failure_type=FailureType.IMPORT_ERROR,
                detail=detail[:500],
                retry_eligible=True,
                escalation_hint=(
                    "Environment issue. Focus on code analysis and patch "
                    "generation without importing GPU libraries."
                ),
                log_snippet=log_tail[-1000:],
            )

        return FailureInfo(
            failure_type=FailureType.CRASH,
            detail=detail[:500],
            retry_eligible=True,
            escalation_hint=(
                "Simplify the task. Try a different approach to the same "
                "optimization goal."
            ),
            log_snippet=log_tail[-1000:],
        )

    if not report and not handle.is_process_alive():
        for marker in _CRASH_MARKERS:
            idx = log_tail.find(marker)
            if idx >= 0:
                snippet = log_tail[max(0, idx - 50):idx + 300].strip()
                return FailureInfo(
                    failure_type=FailureType.CRASH,
                    detail=snippet[:500],
                    retry_eligible=True,
                    escalation_hint=(
                        "Specialist crashed. Re-dispatch with simpler task scope."
                    ),
                    log_snippet=snippet,
                )

        return FailureInfo(
            failure_type=FailureType.CRASH,
            detail=log_tail[-500:] if log_tail else "No log output found",
            retry_eligible=True,
            escalation_hint=(
                "Agent died with no clear error. Re-dispatch with "
                "narrower task and explicit output instructions."
            ),
            log_snippet=log_tail[-1000:],
        )

    return FailureInfo(
        failure_type=FailureType.NO_OUTPUT,
        detail="Agent completed but produced no done.json or results",
        retry_eligible=True,
        escalation_hint=(
            "Re-dispatch with clearer output instructions and narrower task."
        ),
    )


# ─── Retry with escalation ────────────────────────────────────────────────────


def build_retry_prompt(
    original_task: TaskSpec,
    failure_info: FailureInfo,
    attempt: int,
) -> str:
    """Build an escalated prompt for retrying a failed specialist."""
    escalation_levels = {
        2: "Try a different angle from the first attempt.",
        3: "Use a completely different strategy.",
    }
    level_hint = escalation_levels.get(attempt, "Last resort — minimal scope only.")

    return (
        f"## RETRY (attempt {attempt}) — Previous specialist failed\n\n"
        f"**Original task:** {original_task.task_summary}\n\n"
        f"**Previous failure ({failure_info.failure_type.value}):**\n"
        f"```\n{failure_info.detail[:400]}\n```\n\n"
        f"**Strategy change required:**\n"
        f"{failure_info.escalation_hint}\n\n"
        f"**Escalation level {attempt}:** {level_hint}\n\n"
        f"**CRITICAL:** Do NOT repeat the exact same approach that failed.\n\n"
        f"---\n\n"
        f"{original_task.prompt}"
    )


def should_retry(task: TaskSpec, failure_info: FailureInfo, attempt: int) -> bool:
    """Determine if a failed task should be retried."""
    if not failure_info.retry_eligible:
        return False
    if attempt > task.max_retries:
        return False
    return True


def build_retry_task(
    original_task: TaskSpec,
    failure_info: FailureInfo,
    attempt: int,
) -> TaskSpec:
    """Build a new TaskSpec for retrying a failed task."""
    retry_prompt = build_retry_prompt(original_task, failure_info, attempt)
    return TaskSpec(
        prompt=retry_prompt,
        task_summary=f"[retry-{attempt}] {original_task.task_summary}",
        role=original_task.role,
        priority=TaskPriority.HIGH,
        timeout_minutes=original_task.timeout_minutes,
        metadata={**original_task.metadata, "retry_attempt": attempt,
                  "original_failure": failure_info.failure_type.value},
        max_retries=0,
    )


# ─── CLI dispatch ─────────────────────────────────────────────────────────────


def _dispatch_via_cli(
    prompt: str,
    session_dir: str,
    agent_id: str,
    model: str,
    timeout_minutes: int,
) -> AgentHandle:
    """Spawn a claude CLI subprocess for a specialist task."""
    agent_dir = (Path(session_dir) / "agents" / agent_id).resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = agent_dir / "prompt.md"
    prompt_file.write_text(prompt)
    log_path = agent_dir / "process.log"

    allowed_tools = [
        "Bash", "Read", "Write", "Edit", "MultiEdit",
        "Grep", "Glob", "Task", "WebSearch", "WebFetch", "TodoWrite",
    ]

    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("claude CLI not found on PATH")

    cmd = [
        claude_bin,
        "--print",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--permission-mode", "auto",
        "--allowedTools", ",".join(allowed_tools),
        "--system-prompt-file", str(prompt_file),
        "-p", "Execute the task in your system prompt. Work autonomously.",
    ]

    add_dirs = [session_dir]
    for var in ("BASE_DIR", "INFERENCEX_PATH"):
        val = os.environ.get(var)
        if val and Path(val).is_dir():
            add_dirs.append(val)
    for d in add_dirs:
        if Path(d).is_dir():
            cmd.extend(["--add-dir", d])

    env = os.environ.copy()
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("ROCR_VISIBLE_DEVICES", None)

    repo_root = str(Path(session_dir).resolve().parent)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=repo_root,
        )

    return AgentHandle(
        agent_id=agent_id,
        pid=proc.pid,
        process=proc,
        log_path=log_path,
    )
