"""Dynamic agent dispatch — lifecycle management for specialist agents.

This module is the single entry point for:
  1. Dispatching specialist agents via Claude CLI
  2. Batch dispatch with GPU-aware queuing and CPU/GPU task splitting
  3. Tracking lifecycle (dispatched -> running -> completed/failed/timeout)
  4. Reaping completed agents and releasing GPU resources
  5. Failure classification and retry with escalation
  6. Remote SSH dispatch for multi-node setups (falls back to remote_agent.py
     if claude CLI is not available on the remote node)
  7. Kernel-agent specialized dispatch with MCP tools and git worktrees

Dispatch backend: Claude CLI exclusively.
  - Full tool access: Bash, Read, Write, Edit, Grep, Glob, Task, MCP, WebSearch
  - Stream-json output for structured logging
  - --add-dir for codebase access
  - --mcp-config for kernel tools

GPU allocation strategy:
  - CPU-only agents launch immediately with no GPU limit
  - GPU agents are allocated from the dynamic pool
  - If pool is exhausted, GPU tasks are deferred (not blocked)
  - Deferred tasks auto-dispatch when GPUs free up during reap cycles
  - Priority-based dispatch: higher priority tasks get GPUs first
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import choices
from string import ascii_lowercase
from typing import Any, Callable

from hyperloom.comms import (
    TaskManifest,
    write_task_manifest,
    read_completion,
    read_agent_results,
    read_heartbeat,
    is_agent_alive,
    get_agent_status_summary,
    collect_patches,
)
from hyperloom.gpu_pool import GPUPool

log = logging.getLogger(__name__)


# ─── Enums and types ───────────────────────────────────────────────────────────


class FailureType(enum.Enum):
    SUCCESS = "success"
    CRASH = "crash"
    TIMEOUT = "timeout"
    GPU_OOM = "gpu_oom"
    NO_OUTPUT = "no_output"
    IMPORT_ERROR = "import_error"
    UNKNOWN = "unknown"


class TaskPriority(enum.Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


DispatchEventCallback = Callable[[str, dict[str, Any]], None]


# ─── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class AgentHandle:
    """Handle to a dispatched agent with full lifecycle tracking."""

    agent_id: str
    pid: int | None = None
    process: subprocess.Popen | None = None
    log_path: Path = field(default_factory=lambda: Path("/dev/null"))
    gpu_ids: list[int] | None = None
    needs_gpu: bool = False
    task_summary: str = ""
    role: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    dispatched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def holder_name(self) -> str:
        return f"agent-{self.agent_id}"

    def is_process_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def runtime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.dispatched_at).total_seconds()


@dataclass
class TaskSpec:
    """Specification for a task to be dispatched."""

    prompt: str
    task_summary: str = ""
    needs_gpu: bool = False
    gpu_count: int = 1
    role: str = "specialist"
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_minutes: int = 120
    kb_domains: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2


@dataclass
class FailureInfo:
    """Structured failure diagnosis for an agent."""

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
    deferred: list[TaskSpec]
    errors: list[str]


# ─── Agent ID generation ───────────────────────────────────────────────────────


def generate_agent_id(role: str = "agent") -> str:
    ts = int(time.time())
    suffix = "".join(choices(ascii_lowercase, k=4))
    return f"{role}-{ts}-{suffix}"


# ─── Remote SSH support ────────────────────────────────────────────────────────


def parse_agent_node(agent_node: str | None = None) -> tuple[str, int] | None:
    """Parse HYPERLOOM_AGENT_NODE into (host, port). Returns None if not set."""
    node = (
        agent_node
        or os.environ.get("HYPERLOOM_AGENT_NODE")
        or os.environ.get("ARBOR_AGENT_NODE")
    )
    if not node:
        return None
    if ":" in node:
        host, port_str = node.rsplit(":", 1)
        return host, int(port_str)
    return node, 22


def _ssh_prefix(host: str, port: int) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(port),
        host,
    ]


def _build_remote_env(
    gpu_ids: list[str] | None,
    cwd: str,
) -> str:
    """Build env export commands for remote SSH execution."""
    exports = [f"cd {shlex.quote(cwd)}"]
    if gpu_ids:
        gpu_str = ",".join(gpu_ids)
        exports.append(f"export ROCR_VISIBLE_DEVICES={gpu_str}")
        exports.append(f"export HIP_VISIBLE_DEVICES={gpu_str}")
        exports.append(f"export CUDA_VISIBLE_DEVICES={gpu_str}")

    for var in ("ANTHROPIC_API_KEY", "SESSION_DIR", "MODEL_NAME",
                "FRAMEWORK", "BASE_DIR", "INFERENCEX_PATH"):
        val = os.environ.get(var, "")
        if val:
            exports.append(f"export {var}={shlex.quote(val)}")

    return " && ".join(exports) + " &&"


def _check_remote_claude(host: str, port: int) -> bool:
    """Check if claude CLI is available on the remote node."""
    try:
        result = subprocess.run(
            _ssh_prefix(host, port) + ["which claude"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# ─── MCP config ───────────────────────────────────────────────────────────────


def _load_mcp_servers(mcp_config_path: str | None) -> dict | None:
    if not mcp_config_path or not Path(mcp_config_path).exists():
        return None
    try:
        data = json.loads(Path(mcp_config_path).read_text())
        return data.get("mcpServers", {})
    except Exception:
        return None


def _mcp_tool_names(mcp_config_path: str | None) -> list[str]:
    servers = _load_mcp_servers(mcp_config_path)
    if not servers:
        return []
    return [f"mcp__{name}" for name in servers]


# ─── Core dispatch functions ───────────────────────────────────────────────────


def dispatch_agent(
    prompt: str,
    session_dir: str,
    task_summary: str = "",
    gpu_ids: list[int] | None = None,
    needs_gpu: bool = False,
    role: str = "specialist",
    model: str = "claude-sonnet-4-6",
    timeout_minutes: int = 120,
    mcp_config_path: str | None = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    metadata: dict | None = None,
    attempt: int = 1,
    event_callback: DispatchEventCallback | None = None,
) -> AgentHandle:
    """Dispatch a single specialist agent via Claude CLI."""
    agent_id = generate_agent_id(role)
    return _dispatch_with_id(
        agent_id=agent_id,
        prompt=prompt,
        session_dir=session_dir,
        task_summary=task_summary,
        gpu_ids=gpu_ids,
        needs_gpu=needs_gpu,
        role=role,
        model=model,
        timeout_minutes=timeout_minutes,
        mcp_config_path=mcp_config_path,
        priority=priority,
        metadata=metadata,
        attempt=attempt,
        event_callback=event_callback,
    )


def _dispatch_with_id(
    agent_id: str,
    prompt: str,
    session_dir: str,
    task_summary: str = "",
    gpu_ids: list[int] | None = None,
    needs_gpu: bool = False,
    role: str = "specialist",
    model: str = "claude-sonnet-4-6",
    timeout_minutes: int = 120,
    mcp_config_path: str | None = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    metadata: dict | None = None,
    attempt: int = 1,
    event_callback: DispatchEventCallback | None = None,
) -> AgentHandle:
    """Internal dispatch with a pre-generated agent_id."""
    manifest = TaskManifest(
        agent_id=agent_id,
        task_description=task_summary or prompt[:200],
        needs_gpu=needs_gpu,
        gpu_ids=gpu_ids,
        session_dir=session_dir,
        dispatched_at=datetime.now(timezone.utc).isoformat(),
        timeout_minutes=timeout_minutes,
        metadata=metadata or {},
    )
    write_task_manifest(session_dir, manifest)

    gpu_str_ids = [str(g) for g in gpu_ids] if gpu_ids else None

    handle = _dispatch_via_cli(
        prompt, session_dir, agent_id, model,
        timeout_minutes, gpu_str_ids, mcp_config_path,
    )

    handle.needs_gpu = needs_gpu
    handle.gpu_ids = gpu_ids if gpu_ids else []
    handle.task_summary = task_summary
    handle.role = role
    handle.priority = priority
    handle.attempt = attempt

    if event_callback:
        event_callback("agent_dispatched", {
            "agent_id": agent_id,
            "role": role,
            "needs_gpu": needs_gpu,
            "gpu_ids": gpu_ids,
            "attempt": attempt,
        })

    log.info(
        "Dispatched %s (role=%s, gpu=%s, attempt=%d)",
        agent_id, role, gpu_ids, attempt,
    )
    return handle


# ─── Batch dispatch with GPU-aware queuing ─────────────────────────────────────


def dispatch_batch(
    tasks: list[TaskSpec],
    session_dir: str,
    gpu_pool: GPUPool | None = None,
    model: str = "claude-sonnet-4-6",
    mcp_config_path: str | None = None,
    max_concurrent_gpu: int | None = None,
    current_gpu_agents: int = 0,
    event_callback: DispatchEventCallback | None = None,
) -> DispatchResult:
    """Dispatch multiple agents, respecting GPU capacity and priority.

    CPU-only agents always launch immediately (no limit).
    GPU agents launch until the pool is exhausted or max_concurrent_gpu reached.
    Deferred tasks are sorted by priority for later retry.

    Returns DispatchResult with launched, deferred, and any errors.
    """
    launched: list[AgentHandle] = []
    deferred: list[TaskSpec] = []
    errors: list[str] = []

    cpu_tasks = [t for t in tasks if not t.needs_gpu]
    gpu_tasks = sorted(
        [t for t in tasks if t.needs_gpu],
        key=lambda t: t.priority.value,
    )

    for task in cpu_tasks:
        try:
            handle = dispatch_agent(
                prompt=task.prompt,
                session_dir=session_dir,
                task_summary=task.task_summary,
                needs_gpu=False,
                role=task.role,
                model=model,
                timeout_minutes=task.timeout_minutes,
                mcp_config_path=mcp_config_path,
                priority=task.priority,
                metadata=task.metadata,
                event_callback=event_callback,
            )
            launched.append(handle)
        except Exception as e:
            errors.append(f"CPU task '{task.task_summary}' failed to dispatch: {e}")

    active_gpu_count = current_gpu_agents
    for task in gpu_tasks:
        if max_concurrent_gpu and active_gpu_count >= max_concurrent_gpu:
            deferred.append(task)
            continue

        if not gpu_pool:
            deferred.append(task)
            continue

        agent_id = generate_agent_id(task.role)
        holder = f"agent-{agent_id}"

        acquired = gpu_pool.acquire(task.gpu_count, holder=holder)
        if acquired is None:
            deferred.append(task)
            continue

        try:
            handle = _dispatch_with_id(
                agent_id=agent_id,
                prompt=task.prompt,
                session_dir=session_dir,
                task_summary=task.task_summary,
                gpu_ids=acquired,
                needs_gpu=True,
                role=task.role,
                model=model,
                timeout_minutes=task.timeout_minutes,
                mcp_config_path=mcp_config_path,
                priority=task.priority,
                metadata=task.metadata,
                event_callback=event_callback,
            )
            launched.append(handle)
            active_gpu_count += 1
        except Exception as e:
            gpu_pool.release(acquired)
            errors.append(f"GPU task '{task.task_summary}' failed to dispatch: {e}")

    if event_callback:
        event_callback("batch_dispatched", {
            "launched": len(launched),
            "deferred": len(deferred),
            "errors": len(errors),
        })

    return DispatchResult(launched=launched, deferred=deferred, errors=errors)


def dispatch_deferred(
    deferred: list[TaskSpec],
    session_dir: str,
    gpu_pool: GPUPool,
    model: str = "claude-sonnet-4-6",
    mcp_config_path: str | None = None,
    event_callback: DispatchEventCallback | None = None,
) -> DispatchResult:
    """Retry previously deferred GPU tasks now that GPUs may be free."""
    return dispatch_batch(
        deferred, session_dir, gpu_pool, model, mcp_config_path,
        event_callback=event_callback,
    )


# ─── Lifecycle management ──────────────────────────────────────────────────────


def reap_completed(
    handles: list[AgentHandle],
    session_dir: str,
    gpu_pool: GPUPool | None = None,
    heartbeat_timeout: float = 600.0,
    event_callback: DispatchEventCallback | None = None,
) -> tuple[list[AgentHandle], list[AgentHandle]]:
    """Check handles and separate completed from still-running.

    For completed agents, releases their GPUs automatically.
    Agents with no heartbeat for > heartbeat_timeout are marked dead.

    Returns (still_running, newly_completed).
    """
    still_running: list[AgentHandle] = []
    newly_completed: list[AgentHandle] = []

    for h in handles:
        report = read_completion(session_dir, h.agent_id)
        process_dead = not h.is_process_alive()

        if report is not None or process_dead:
            if gpu_pool and h.gpu_ids:
                release_agent_gpus(h, gpu_pool)
            newly_completed.append(h)

            if event_callback:
                status = "completed" if report else "crashed"
                event_callback("agent_finished", {
                    "agent_id": h.agent_id,
                    "role": h.role,
                    "status": status,
                    "runtime_s": h.runtime_seconds(),
                    "gpu_ids": h.gpu_ids,
                })
        else:
            hb = read_heartbeat(session_dir, h.agent_id)
            if hb and (time.time() - hb.timestamp) > heartbeat_timeout:
                if gpu_pool and h.gpu_ids:
                    release_agent_gpus(h, gpu_pool)
                newly_completed.append(h)
                if event_callback:
                    event_callback("agent_timeout", {
                        "agent_id": h.agent_id,
                        "role": h.role,
                        "last_heartbeat_age_s": time.time() - hb.timestamp,
                    })
            else:
                still_running.append(h)

    return still_running, newly_completed


def release_agent_gpus(handle: AgentHandle, gpu_pool: GPUPool) -> list[int]:
    """Release GPUs when an agent finishes. Returns freed GPU IDs."""
    if handle.gpu_ids and gpu_pool:
        gpu_pool.release(handle.gpu_ids)
        freed = list(handle.gpu_ids)
        handle.gpu_ids = []
        return freed
    return []


# ─── Failure classification ────────────────────────────────────────────────────


_CRASH_MARKERS = [
    "ImportError", "ModuleNotFoundError", "RuntimeError",
    "CUDA error", "HIP error", "Traceback",
    "Killed", "Segmentation fault", "signal", "SIGKILL",
    "OOM", "out of memory", "Cannot allocate memory",
]


def classify_agent_failure(
    session_dir: str,
    handle: AgentHandle,
) -> FailureInfo:
    """Diagnose why a specialist failed and suggest retry strategy."""
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

        if any(kw in detail_lower for kw in ("oom", "out of memory", "memory", "gpu")):
            return FailureInfo(
                failure_type=FailureType.GPU_OOM,
                detail=detail[:500],
                retry_eligible=True,
                escalation_hint=(
                    "Reduce batch size or model size for profiling. "
                    "Use CPU-only analysis instead of GPU profiling. "
                    "Consider splitting across more GPUs."
                ),
                log_snippet=log_tail[-1000:],
            )

        if "import" in detail_lower or "module" in detail_lower:
            return FailureInfo(
                failure_type=FailureType.IMPORT_ERROR,
                detail=detail[:500],
                retry_eligible=True,
                escalation_hint=(
                    "Environment issue. Dispatch as CPU-only research agent "
                    "that does not import GPU libraries directly."
                ),
                log_snippet=log_tail[-1000:],
            )

        return FailureInfo(
            failure_type=FailureType.CRASH,
            detail=detail[:500],
            retry_eligible=True,
            escalation_hint=(
                "Simplify the task. Try a different approach to the same "
                "optimization goal. Avoid the specific operation that crashed."
            ),
            log_snippet=log_tail[-1000:],
        )

    if not report and not handle.is_process_alive():
        for marker in _CRASH_MARKERS:
            idx = log_tail.find(marker)
            if idx >= 0:
                snippet = log_tail[max(0, idx - 50):idx + 300].strip()

                if marker in ("OOM", "out of memory", "Cannot allocate memory"):
                    return FailureInfo(
                        failure_type=FailureType.GPU_OOM,
                        detail=snippet[:500],
                        retry_eligible=True,
                        escalation_hint=(
                            "GPU OOM crash. Reduce batch/model size, or dispatch "
                            "as CPU-only research agent."
                        ),
                        log_snippet=snippet,
                    )

                if marker in ("ImportError", "ModuleNotFoundError"):
                    return FailureInfo(
                        failure_type=FailureType.IMPORT_ERROR,
                        detail=snippet[:500],
                        retry_eligible=True,
                        escalation_hint=(
                            "Import/environment error. Dispatch CPU-only "
                            "research agent instead of GPU agent."
                        ),
                        log_snippet=snippet,
                    )

                return FailureInfo(
                    failure_type=FailureType.CRASH,
                    detail=snippet[:500],
                    retry_eligible=True,
                    escalation_hint=(
                        "Specialist crashed. Re-dispatch with simpler task scope. "
                        "If crash is environment-related, use CPU-only agent."
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
    """Build an escalated prompt for retrying a failed specialist task."""
    escalation_levels = {
        2: "Try a different angle from the first attempt.",
        3: "Use a completely different strategy. If GPU failed, try CPU-only analysis.",
        4: "Focus only on the single simplest possible improvement.",
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
        f"**CRITICAL:** Do NOT repeat the exact same approach that failed. "
        f"Every retry must use a fundamentally different strategy:\n"
        f"- If GPU profiling failed -> try static code analysis\n"
        f"- If complex patch failed -> try simple config change\n"
        f"- If task was too broad -> focus on single highest-impact item\n"
        f"- If environment issue -> work around it, don't fight it\n\n"
        f"---\n\n"
        f"{original_task.prompt}"
    )


def should_retry(task: TaskSpec, failure_info: FailureInfo, attempt: int) -> bool:
    """Determine if a failed task should be retried."""
    if not failure_info.retry_eligible:
        return False
    if attempt > task.max_retries:
        return False
    if failure_info.failure_type == FailureType.IMPORT_ERROR and attempt > 1:
        return False
    return True


def build_retry_task(
    original_task: TaskSpec,
    failure_info: FailureInfo,
    attempt: int,
) -> TaskSpec:
    """Build a new TaskSpec for retrying a failed task with adjusted parameters."""
    retry_prompt = build_retry_prompt(original_task, failure_info, attempt)

    needs_gpu = original_task.needs_gpu
    if failure_info.failure_type == FailureType.GPU_OOM:
        needs_gpu = False

    return TaskSpec(
        prompt=retry_prompt,
        task_summary=f"[retry-{attempt}] {original_task.task_summary}",
        needs_gpu=needs_gpu,
        gpu_count=original_task.gpu_count,
        role=original_task.role,
        priority=TaskPriority.HIGH,
        timeout_minutes=original_task.timeout_minutes,
        kb_domains=original_task.kb_domains,
        metadata={**original_task.metadata, "retry_attempt": attempt,
                  "original_failure": failure_info.failure_type.value},
        max_retries=0,
    )


# ─── Kernel-agent dispatch ─────────────────────────────────────────────────────


def _build_kernel_mcp_config(kernel_agents_root: str | Path) -> str | None:
    """Build MCP config pointing to kernel-agents MCP server."""
    ka_root = Path(kernel_agents_root)
    server_py = ka_root / "src" / "kernel_agents" / "mcp_server" / "server.py"
    if not server_py.exists():
        return None

    config = {
        "mcpServers": {
            "kernel-tools": {
                "command": "python3",
                "args": [str(server_py)],
                "cwd": str(ka_root),
            }
        }
    }

    config_path = Path(tempfile.mkdtemp(prefix="hyperloom-mcp-")) / "kernel_mcp.json"
    config_path.write_text(json.dumps(config, indent=2))
    return str(config_path)


def _setup_kernel_worktree(session_dir: str, agent_id: str) -> str | None:
    """Create isolated git worktree for kernel agent's patch work."""
    worktree_dir = Path(session_dir) / "kernel_phase" / "worktrees" / agent_id
    branch_name = f"kernel-agent-{agent_id}"

    try:
        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(worktree_dir)],
            capture_output=True, text=True, check=True,
        )
        return str(worktree_dir)
    except subprocess.CalledProcessError:
        worktree_dir.mkdir(parents=True, exist_ok=True)
        return str(worktree_dir)


def dispatch_kernel_agent(
    prompt: str,
    session_dir: str,
    task_summary: str,
    gpu_ids: list[int],
    model: str = "claude-sonnet-4-6",
    timeout_minutes: int = 90,
    kernel_agents_root: str | Path | None = None,
    mcp_config_path: str | None = None,
    event_callback: DispatchEventCallback | None = None,
) -> AgentHandle:
    """Dispatch a kernel-optimization agent with GPU tools and isolated worktree.

    Key differences from regular dispatch_agent:
      - MCP config includes kernel-agents GPU tools (build/test/bench/pmc/registers)
      - Agent gets --add-dir to session and kernel-agents repo
      - Each agent gets a git worktree for isolated patch development
      - Single GPU per agent via ROCR_VISIBLE_DEVICES
    """
    env_ka = kernel_agents_root or os.environ.get("KERNEL_AGENTS_PATH")
    if not env_ka:
        raise RuntimeError(
            "kernel_agents_root not provided and KERNEL_AGENTS_PATH not set"
        )
    ka_root = Path(env_ka)

    agent_id = generate_agent_id("kernel")

    effective_mcp = mcp_config_path
    if not effective_mcp:
        effective_mcp = _build_kernel_mcp_config(ka_root)

    worktree_path = _setup_kernel_worktree(session_dir, agent_id)

    manifest = TaskManifest(
        agent_id=agent_id,
        task_description=task_summary,
        needs_gpu=True,
        gpu_ids=gpu_ids,
        session_dir=session_dir,
        dispatched_at=datetime.now(timezone.utc).isoformat(),
        timeout_minutes=timeout_minutes,
        metadata={
            "phase": "kernel",
            "worktree": worktree_path,
            "kernel_agents_root": str(ka_root),
        },
    )
    write_task_manifest(session_dir, manifest)

    gpu_str_ids = [str(g) for g in gpu_ids]

    handle = _dispatch_kernel_via_cli(
        prompt=prompt,
        session_dir=session_dir,
        agent_id=agent_id,
        model=model,
        timeout_minutes=timeout_minutes,
        gpu_ids=gpu_str_ids,
        mcp_config_path=effective_mcp,
        kernel_agents_root=str(ka_root),
        worktree_path=worktree_path,
    )

    handle.needs_gpu = True
    handle.gpu_ids = gpu_ids
    handle.task_summary = task_summary
    handle.role = "kernel"

    if event_callback:
        event_callback("kernel_agent_dispatched", {
            "agent_id": agent_id,
            "gpu_ids": gpu_ids,
            "worktree": worktree_path,
        })

    return handle


# ─── Claude CLI dispatch (sole backend) ───────────────────────────────────────


def _dispatch_via_cli(
    prompt: str,
    session_dir: str,
    agent_id: str,
    model: str,
    timeout_minutes: int,
    gpu_ids: list[str] | None,
    mcp_config_path: str | None,
) -> AgentHandle:
    """Dispatch via Claude CLI.

    If HYPERLOOM_AGENT_NODE is set, dispatches via SSH to the remote node.
    If claude is not available on the remote node, falls back to remote_agent.py.
    """
    agent_dir = (Path(session_dir) / "agents" / agent_id).resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = agent_dir / "prompt.md"
    prompt_file.write_text(prompt)
    log_path = agent_dir / "process.log"

    allowed_tools = [
        "Bash", "Read", "Write", "Edit", "MultiEdit",
        "Grep", "Glob", "Task", "WebSearch", "WebFetch", "TodoWrite",
    ]
    allowed_tools.extend(_mcp_tool_names(mcp_config_path))

    cmd = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--permission-mode", "auto",
        "--allowedTools", ",".join(allowed_tools),
        "--system-prompt-file", str(prompt_file),
        "-p", "Execute the task in your system prompt. Work autonomously.",
    ]

    if mcp_config_path:
        cmd.extend(["--mcp-config", mcp_config_path])

    add_dirs = [session_dir]
    for var in ("BASE_DIR", "INFERENCEX_PATH"):
        val = os.environ.get(var)
        if val and Path(val).is_dir():
            add_dirs.append(val)
    for d in add_dirs:
        if Path(d).is_dir():
            cmd.extend(["--add-dir", d])

    env = os.environ.copy()
    if gpu_ids:
        gpu_str = ",".join(gpu_ids)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["ROCR_VISIBLE_DEVICES"] = gpu_str

    remote = parse_agent_node()
    if remote:
        host, port = remote
        if _check_remote_claude(host, port):
            env_exports = _build_remote_env(gpu_ids, session_dir)
            shell_cmd = env_exports + " " + " ".join(shlex.quote(c) for c in cmd)
            full_cmd = _ssh_prefix(host, port) + [shell_cmd]
        else:
            full_cmd = _build_remote_agent_fallback(
                host, port, prompt_file, model, session_dir, agent_id, gpu_ids
            )
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(full_cmd, stdout=log_f, stderr=subprocess.STDOUT)
    else:
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


def _build_remote_agent_fallback(
    host: str,
    port: int,
    prompt_file: Path,
    model: str,
    session_dir: str,
    agent_id: str,
    gpu_ids: list[str] | None,
) -> list[str]:
    """Build SSH command to run remote_agent.py when claude CLI is unavailable."""
    env_exports = _build_remote_env(gpu_ids, session_dir)
    remote_cmd = (
        f"python3 -m hyperloom.remote_agent"
        f" --prompt-file {shlex.quote(str(prompt_file))}"
        f" --model {shlex.quote(model)}"
        f" --session-dir {shlex.quote(session_dir)}"
        f" --agent-id {shlex.quote(agent_id)}"
    )
    shell_cmd = env_exports + " " + remote_cmd
    return _ssh_prefix(host, port) + [shell_cmd]


def _dispatch_kernel_via_cli(
    prompt: str,
    session_dir: str,
    agent_id: str,
    model: str,
    timeout_minutes: int,
    gpu_ids: list[str] | None,
    mcp_config_path: str | None,
    kernel_agents_root: str,
    worktree_path: str | None,
) -> AgentHandle:
    """CLI dispatch specialized for kernel agents."""
    agent_dir = (Path(session_dir) / "agents" / agent_id).resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = agent_dir / "prompt.md"
    prompt_file.write_text(prompt)
    log_path = agent_dir / "process.log"

    allowed_tools = [
        "Bash", "Read", "Write", "Edit", "MultiEdit",
        "Grep", "Glob", "Task", "WebSearch", "WebFetch", "TodoWrite",
    ]
    allowed_tools.extend(_mcp_tool_names(mcp_config_path))

    cmd = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--permission-mode", "auto",
        "--allowedTools", ",".join(allowed_tools),
        "--system-prompt-file", str(prompt_file),
        "-p", "Execute the kernel optimization task in your system prompt. Work autonomously.",
    ]

    if mcp_config_path:
        cmd.extend(["--mcp-config", mcp_config_path])

    add_dirs = [session_dir, kernel_agents_root]
    if worktree_path:
        add_dirs.append(worktree_path)
    for var in ("BASE_DIR", "INFERENCEX_PATH"):
        val = os.environ.get(var)
        if val and Path(val).is_dir():
            add_dirs.append(val)
    for d in add_dirs:
        if Path(d).is_dir():
            cmd.extend(["--add-dir", d])

    env = os.environ.copy()
    if gpu_ids:
        gpu_str = ",".join(gpu_ids)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["ROCR_VISIBLE_DEVICES"] = gpu_str

    cwd = worktree_path if worktree_path and Path(worktree_path).is_dir() else session_dir

    remote = parse_agent_node()
    if remote:
        host, port = remote
        env_exports = _build_remote_env(gpu_ids, cwd)
        shell_cmd = env_exports + " " + " ".join(shlex.quote(c) for c in cmd)
        full_cmd = _ssh_prefix(host, port) + [shell_cmd]
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(full_cmd, stdout=log_f, stderr=subprocess.STDOUT)
    else:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=cwd,
            )

    return AgentHandle(
        agent_id=agent_id,
        pid=proc.pid,
        process=proc,
        log_path=log_path,
    )


# ─── Convenience: wait for all agents ─────────────────────────────────────────


def wait_for_agents(
    handles: list[AgentHandle],
    session_dir: str,
    gpu_pool: GPUPool | None = None,
    poll_interval: float = 10.0,
    timeout: float = 7200.0,
    event_callback: DispatchEventCallback | None = None,
) -> list[AgentHandle]:
    """Block until all agents complete or timeout.

    Returns the list of all handles (completed or timed out).
    Frees GPUs as agents complete, enabling deferred tasks if desired.
    """
    start = time.time()
    running = list(handles)
    all_completed: list[AgentHandle] = []

    while running and (time.time() - start) < timeout:
        time.sleep(poll_interval)
        running, newly_done = reap_completed(
            running, session_dir, gpu_pool, event_callback=event_callback,
        )
        all_completed.extend(newly_done)

    all_completed.extend(running)
    return all_completed


# ─── Status summary ───────────────────────────────────────────────────────────


def get_dispatch_summary(
    handles: list[AgentHandle],
    session_dir: str,
) -> dict[str, Any]:
    """Get a structured summary of all dispatched agents."""
    active = []
    completed = []
    failed = []

    for h in handles:
        report = read_completion(session_dir, h.agent_id)
        if report:
            if report.status == "success":
                completed.append(h)
            else:
                failed.append(h)
        elif h.is_process_alive():
            active.append(h)
        else:
            failed.append(h)

    return {
        "total": len(handles),
        "active": len(active),
        "completed": len(completed),
        "failed": len(failed),
        "active_agents": [
            {"id": h.agent_id, "role": h.role, "runtime_s": h.runtime_seconds()}
            for h in active
        ],
        "completed_agents": [
            {"id": h.agent_id, "role": h.role}
            for h in completed
        ],
        "failed_agents": [
            {"id": h.agent_id, "role": h.role}
            for h in failed
        ],
    }
