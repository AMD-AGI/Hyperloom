# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-session path helpers — single source of truth for every path *inside*
a session directory (``paths.py`` owns *where* the session lives).

Hard rule: all code MUST derive sub-paths under ``session_dir`` through this
module; no ad-hoc string concatenation elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


# Top-level files
def manifest_path(session_dir: Path) -> Path:
    """Absolute path to ``manifest.json`` (Python-written session resume tag)."""
    return Path(session_dir) / "manifest.json"


def state_path(session_dir: Path) -> Path:
    """Absolute path to ``state.json`` (SharedState — Coordinator-written)."""
    return Path(session_dir) / "state.json"


# Per-task workspaces under runs/<action>/<task_id>/.
# Which actions own a runs/ workspace is derived from the ActionRegistry's
# ``pipeline_phase`` field; these are the phases whose executors write there.
_RUNS_WORKSPACE_PHASES: frozenset[str] = frozenset({
    "measure", "analysis", "explore", "deep", "validate", "support",
})

# Fallback used only when ActionRegistry can't load (broken yaml / early
# bootstrap). Must stay in sync with the _RUNS_WORKSPACE_PHASES union;
# tests/test_action_catalogue.py enforces alignment.
_RUNS_ACTIONS_FALLBACK: frozenset[str] = frozenset({
    "baseline",
    "replay_warm_recipe",
    "roofline", "profile",
    "sweep",
    "conc_sweep",
    "explore",
    "specialist",
    "integrate_patch",
    "framework_pr",
    "integrate", "kernel_opt", "deep_kernel_analysis", "gemm_tuning",
    "operator_tuning", "vendor_kernel_config",
    "recover",
})


@lru_cache(maxsize=1)
def _runs_actions() -> frozenset[str]:
    """Action names that own a ``runs/<kind>/<task_id>/`` workspace, from
    action metadata. Lazy + cached; falls back to ``_RUNS_ACTIONS_FALLBACK``
    when the registry can't load.
    """
    try:
        from .orchestrator.action_registry import ActionRegistry  # local: avoid import-time cycle
        registry = ActionRegistry().load()
    except Exception:
        return _RUNS_ACTIONS_FALLBACK
    return frozenset(
        a.name for a in registry.all()
        if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
    )


def _validate_action(action: str) -> str:
    a = str(action or "").strip()
    valid = _runs_actions()
    if a not in valid:
        raise ValueError(
            f"runs_dir: unknown action {action!r}; expected one of "
            f"{sorted(valid)!r}"
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


# Kernel agent long-lived workspaces (cross-task, keyed by kernel_id)
def kernel_workspace(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/kernel-agent-workspace/<kernel_id>/`` — extracted source,
    GEAK/OOB candidates, and the chosen patch for one kernel. Keyed by
    ``kernel_id`` and survives across tasks (vs the per-invocation
    :func:`kernel_agent_runs_dir`).
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "kernel-agent-workspace" / kid


def kernel_agent_runs_dir(session_dir: Path, session_id: str) -> Path:
    """``<sd>/kernel-agent/runs/<session_id>/`` — per-tool-invocation
    kernel-agent output (logs, status JSON, optimization_attempts.jsonl,
    TraceLens analysis). Keyed by tool-invocation session id (vs the
    kernel_id-keyed :func:`kernel_workspace`).
    """
    sid = str(session_id or "").strip() or "unknown"
    return Path(session_dir) / "kernel-agent" / "runs" / sid


def patches_dir(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/patches/<kernel_id>/`` — KEEP-promoted on-disk changes: the
    original source backup + applied patch (REVERT restores from backup).
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "patches" / kid


# Session-breakdown record fragments (recorder write-side spool).
def breakdown_parts_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/breakdown/parts/`` — per-producer breakdown record
    fragments. Each owner writes its own files here (atomic + uniquely named);
    the exporter assembles them into ``session_breakdown.json``. Single-owner
    per section, so there is no cross-producer write contention.
    """
    return Path(session_dir) / "runtime" / "breakdown" / "parts"


# Reports / logs
def reports_dir(session_dir: Path) -> Path:
    return Path(session_dir) / "reports"


def report_file(session_dir: Path, ts: str, suffix: str = "md") -> Path:
    """``<sd>/reports/<ts>_final.{md,json}``."""
    return reports_dir(session_dir) / f"{ts}_final.{suffix}"


def research_hints_md(session_dir: Path) -> Path:
    """``<sd>/research_hints.md`` — human-readable proven-prior hints
    collected by the research scout."""
    return Path(session_dir) / "research_hints.md"


def research_hints_json(session_dir: Path) -> Path:
    """``<sd>/research_hints.json`` — structured mirror of the research
    hints (machine-readable; advisory gap-scoring reads this)."""
    return Path(session_dir) / "research_hints.json"


def competitor_target_json(session_dir: Path) -> Path:
    """``<sd>/competitor_target.json`` — LLM-authored competitor target
    numbers (each per-concurrency entry carries its own source)."""
    return Path(session_dir) / "competitor_target.json"


def logs_dir(session_dir: Path) -> Path:
    return Path(session_dir) / "logs"


def agent_log(session_dir: Path, role: str) -> Path:
    """``<sd>/logs/<role>.log`` — one append-only log per persistent agent."""
    return logs_dir(session_dir) / f"{role}.log"


# Launcher-side artefacts (stdout, PID file, robustness monitor logs)
def optimizer_runs_dir(session_dir: Path) -> Path:
    """``<sd>/optimizer_runs/`` — launcher artefacts (run_<tag>.log / .pid /
    robustness_monitor_*.log). Under $USER_DATA_PATH so one override moves
    the whole run tail.
    """
    return Path(session_dir) / "optimizer_runs"


def optimizer_run_log(session_dir: Path, run_tag: str) -> Path:
    """``<sd>/optimizer_runs/run_<tag>.log``."""
    tag = str(run_tag or "").strip() or "unknown"
    return optimizer_runs_dir(session_dir) / f"run_{tag}.log"


def optimizer_run_pidfile(session_dir: Path, run_tag: str) -> Path:
    """``<sd>/optimizer_runs/run_<tag>.pid``."""
    tag = str(run_tag or "").strip() or "unknown"
    return optimizer_runs_dir(session_dir) / f"run_{tag}.pid"


# Per-agent inbox/outbox + system prompt snapshot
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


# External baseline comparison artefacts. Dedicated top-level subdir (not
# runs/) because target_analysis is a prep-phase action and prep is not in
# _RUNS_WORKSPACE_PHASES.
def target_analysis_dir(session_dir: Path) -> Path:
    """``<sd>/target_analysis/`` — external baseline artefacts. Owner:
    TargetAnalysisExecutor; reader: ReportExecutor.
    """
    return Path(session_dir) / "target_analysis"


def target_baseline_json(session_dir: Path) -> Path:
    """``<sd>/target_analysis/target_baseline.json`` — machine-readable
    ``BaselineSummary`` written by ``target_analysis`` and read by
    ``report`` to render an advisory section in ``final.md``."""
    return target_analysis_dir(session_dir) / "target_baseline.json"


def target_analysis_report_md(session_dir: Path) -> Path:
    """``<sd>/target_analysis/target_analysis_report.md`` — short human
    note suitable for inclusion / linking from the final report."""
    return target_analysis_dir(session_dir) / "target_analysis_report.md"


# Cortex KB integration paths — single source of truth for every file under
# ``<sd>/runtime/cortex/``. Callers MUST go through these helpers so the NDJSON
# protocol stays homogeneous across producers/consumers.
def cortex_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/`` — Cortex KB per-session bookkeeping root."""
    return Path(session_dir) / "runtime" / "cortex"


def cortex_sid_file(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_sid`` — Cortex session id from T0 ``session
    begin`` (resume reuses it). Absent => --degraded-kb or T0 not yet run.
    """
    return cortex_dir(session_dir) / ".kb_sid"


def cortex_warm_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_warm.json`` — T0 snapshot of
    ``find-recipe`` output. Read by §3.5 specialist assembly (M5).
    """
    return cortex_dir(session_dir) / ".kb_warm.json"


def cortex_pitfalls_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_pitfalls.json`` — T0 snapshot of
    ``traps`` output. Read by §3.5 specialist assembly (M5).
    """
    return cortex_dir(session_dir) / ".kb_pitfalls.json"


def cortex_pending_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_pending.ndjson`` — append-only async write
    queue for T2/T3 ops. Consumed by the cortex_kb_flusher daemon; drained at
    T4 before ``session commit``.
    """
    return cortex_dir(session_dir) / ".kb_pending.ndjson"


def cortex_flushed_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_flushed.ndjson`` — successfully-POSTed
    rows, kept around for offline audit / breakdown collection.
    """
    return cortex_dir(session_dir) / ".kb_flushed.ndjson"


def cortex_dead_letter_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_dead_letter.ndjson`` — rows that failed
    permanently (HTTP 4xx business-logic rejects); robustness HIGH alert.
    """
    return cortex_dir(session_dir) / ".kb_dead_letter.ndjson"


def cortex_audit_jsonl(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_audit.jsonl`` — append-only audit of every
    direct Cortex CLI invocation. Source of truth for breakdown.kb_provenance.
    """
    return cortex_dir(session_dir) / ".kb_audit.jsonl"


# recipe-snapshot v2 per-session bookkeeping. Separate
# ``runtime/recipe_snapshot/`` subtree (not runtime/cortex/) to stay decoupled
# from the legacy /v1/points client. Writes are local-only, so only the
# read-side audit log survives here.
def recipe_snapshot_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_snapshot/`` — dispatcher / remote-client
    per-session bookkeeping root.
    """
    return Path(session_dir) / "runtime" / "recipe_snapshot"


def recipe_snapshot_audit_jsonl(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_snapshot/.audit.jsonl`` — append-only audit of
    every recipe-snapshot remote READ call (writes are local-only and skip it).
    """
    return recipe_snapshot_dir(session_dir) / ".audit.jsonl"


def pr_monitor_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.pr_monitor_status.json`` — boot-time PR Monitor
    reachability snapshot; breakdown reads it for pr_monitor:* warnings.

    Schema: ``{enabled, url, reachable, mcp_url, window_days, status_text}``.
    """
    return cortex_dir(session_dir) / ".pr_monitor_status.json"


def cortex_flusher_pid(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_flusher.pid`` — flusher daemon pid file
    (one line). Robustness reads this to detect a dead flusher.
    """
    return cortex_dir(session_dir) / ".kb_flusher.pid"


def cortex_flusher_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_flusher_status.json`` — boot-time flusher
    spawn decision; merged with the live pid check for
    kb_provenance.flusher_status.

    Schema: ``{enabled, spawned, pid, cmd, cortex_kb_url, interval_sec,
    batch_size, reason, ts}``.
    """
    return cortex_dir(session_dir) / ".kb_flusher_status.json"


__all__ = [
    "agent_dir",
    "agent_inbox",
    "agent_log",
    "agent_outbox",
    "agent_persona",
    "agent_prompt_snapshot",
    "breakdown_parts_dir",
    "competitor_target_json",
    "cortex_audit_jsonl",
    "cortex_dead_letter_ndjson",
    "cortex_dir",
    "cortex_flushed_ndjson",
    "cortex_flusher_pid",
    "cortex_flusher_status_json",
    "cortex_pending_ndjson",
    "cortex_pitfalls_json",
    "cortex_sid_file",
    "cortex_warm_json",
    "kernel_agent_runs_dir",
    "kernel_workspace",
    "logs_dir",
    "manifest_path",
    "optimizer_run_log",
    "optimizer_run_pidfile",
    "optimizer_runs_dir",
    "patches_dir",
    "report_file",
    "reports_dir",
    "research_hints_json",
    "research_hints_md",
    "runs_dir",
    "runs_root",
    "state_path",
    "target_analysis_dir",
    "target_analysis_report_md",
    "target_baseline_json",
]
