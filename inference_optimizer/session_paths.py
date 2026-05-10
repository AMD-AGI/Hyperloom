"""Per-session path helpers (DESIGN v0.6.1 §23).

Single source of truth for every path *inside* a session directory. The
skeleton itself is created by :func:`paths.make_session_dir`; this module
only computes (and lazily mkdirs) the per-task / per-kernel sub-paths
that executors and kernel handlers fill in.

Why this lives apart from ``paths.py``:

* ``paths.py`` is concerned with *where* the session lives (resolution +
  skeleton mkdir). It never knows about task_ids or kernel_ids.
* ``session_paths.py`` is concerned with *what's inside* a session. It
  takes a ``session_dir: Path`` argument explicitly so all callers
  (executors, kernel handlers, prompt injection) share the same
  derivation rules.

Hard rule: executor / handler / cli code MUST go through this module
for any sub-path under ``session_dir``. No string concatenation like
``session_dir / "runs" / kind / task_id`` is allowed elsewhere.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Top-level files
# ---------------------------------------------------------------------------
def manifest_path(session_dir: Path) -> Path:
    """Absolute path to ``manifest.json`` (Python-written session resume tag)."""
    return Path(session_dir) / "manifest.json"


def state_path(session_dir: Path) -> Path:
    """Absolute path to ``state.json`` (SharedState — Coordinator-written)."""
    return Path(session_dir) / "state.json"


# ---------------------------------------------------------------------------
# Per-task workspaces under runs/<action>/<task_id>/
# ---------------------------------------------------------------------------
# Action names that get a stable home under ``runs/``. Listed here so a
# typo in a caller's `kind` argument fails loudly via _validate_action()
# rather than silently mkdir-ing ``runs/<typo>/``.
_RUNS_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "backends", "params", "sweep",
    "integrate", "kernel_opt",
})


def _validate_action(action: str) -> str:
    a = str(action or "").strip()
    if a not in _RUNS_ACTIONS:
        raise ValueError(
            f"runs_dir: unknown action {action!r}; expected one of "
            f"{sorted(_RUNS_ACTIONS)!r}"
        )
    return a


def runs_root(session_dir: Path) -> Path:
    """``<sd>/runs/`` — parent of all per-action subtrees."""
    return Path(session_dir) / "runs"


def runs_dir(session_dir: Path, action: str, task_id: str) -> Path:
    """``<sd>/runs/<action>/<task_id>/`` — per-task data-plane workspace.

    Caller is expected to ``mkdir(parents=True, exist_ok=True)`` before
    writing files into the returned path; SubAgentRunner pre-creates this
    in normal coordinator-managed runs, so executors typically just read
    ``ctx.extra["workspace"]``.
    """
    a = _validate_action(action)
    tid = str(task_id or "").strip() or "unknown"
    return runs_root(session_dir) / a / tid


# ---------------------------------------------------------------------------
# Kernel agent long-lived workspaces (cross-task, keyed by kernel_id)
# ---------------------------------------------------------------------------
def kernel_workspace(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/kernel-agent-workspace/<kernel_id>/``.

    Holds the extracted source, accumulated GEAK/OOB candidates and the
    currently chosen patch for one kernel. Survives across tasks so
    re-issuing run_optimization on the same kernel can reuse prior
    artefacts.
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "kernel-agent-workspace" / kid


def patches_dir(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/patches/<kernel_id>/`` — KEEP-promoted real on-disk changes.

    Each kernel that ever passed the integrate gate gets a directory
    with the original source backup and the applied patch; REVERT
    restores from the backup before re-baselining.
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "patches" / kid


# ---------------------------------------------------------------------------
# Reports / logs
# ---------------------------------------------------------------------------
def reports_dir(session_dir: Path) -> Path:
    return Path(session_dir) / "reports"


def report_file(session_dir: Path, ts: str, suffix: str = "md") -> Path:
    """``<sd>/reports/<ts>_final.{md,json}``."""
    return reports_dir(session_dir) / f"{ts}_final.{suffix}"


def logs_dir(session_dir: Path) -> Path:
    return Path(session_dir) / "logs"


def agent_log(session_dir: Path, role: str) -> Path:
    """``<sd>/logs/<role>.log`` — one append-only log per persistent agent."""
    return logs_dir(session_dir) / f"{role}.log"


# ---------------------------------------------------------------------------
# Per-agent inbox/outbox + system prompt snapshot
# ---------------------------------------------------------------------------
def agent_dir(session_dir: Path, role: str) -> Path:
    return Path(session_dir) / "agents" / role


def agent_inbox(session_dir: Path, role: str) -> Path:
    return agent_dir(session_dir, role) / "inbox.jsonl"


def agent_outbox(session_dir: Path, role: str) -> Path:
    return agent_dir(session_dir, role) / "outbox.jsonl"


def agent_persona(session_dir: Path, role: str) -> Path:
    return agent_dir(session_dir, role) / "persona.md"


def agent_prompt_snapshot(session_dir: Path, role: str) -> Path:
    """``<sd>/agents/<role>/system_prompt.snapshot.md`` — written once at
    Coordinator start, then read for resume / drift inspection."""
    return agent_dir(session_dir, role) / "system_prompt.snapshot.md"


__all__ = [
    "agent_dir",
    "agent_inbox",
    "agent_log",
    "agent_outbox",
    "agent_persona",
    "agent_prompt_snapshot",
    "kernel_workspace",
    "logs_dir",
    "manifest_path",
    "patches_dir",
    "report_file",
    "reports_dir",
    "runs_dir",
    "runs_root",
    "state_path",
]
