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
    """Compute the path to ``manifest.json`` (the Python-written resume tag).

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/manifest.json``.
    """
    return Path(session_dir) / "manifest.json"


def state_path(session_dir: Path) -> Path:
    """Compute the path to ``state.json`` (the Coordinator-written SharedState).

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/state.json``.
    """
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
    """Normalise and validate an action name against the runs-workspace set.

    Args:
        action (str): The candidate action name (whitespace-stripped before
            comparison).

    Returns:
        str: The stripped action name when it is recognised.

    Raises:
        ValueError: If the action is not one of the names returned by
            :func:`_runs_actions`.
    """
    a = str(action or "").strip()
    valid = _runs_actions()
    if a not in valid:
        raise ValueError(
            f"runs_dir: unknown action {action!r}; expected one of "
            f"{sorted(valid)!r}"
        )
    return a


def runs_root(session_dir: Path) -> Path:
    """Compute ``<sd>/runs/``, the parent of all per-action subtrees.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runs``.
    """
    return Path(session_dir) / "runs"


def runs_dir(session_dir: Path, action: str, task_id: str) -> Path:
    """Compute ``<sd>/runs/<action>/<task_id>/``, a per-task data-plane workspace.

    Caller is expected to ``mkdir(parents=True, exist_ok=True)`` before
    writing files into the returned path; SubAgentRunner pre-creates this
    in normal coordinator-managed runs, so executors typically just read
    ``ctx.extra["workspace"]``.

    Args:
        session_dir (Path): The session root directory.
        action (str): The owning action name; validated against the
            runs-workspace action set.
        task_id (str): The task identifier; blank/empty falls back to
            ``"unknown"``.

    Returns:
        Path: The absolute path to ``<session_dir>/runs/<action>/<task_id>``.

    Raises:
        ValueError: If ``action`` is not a recognised runs-workspace action.
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
    """Compute ``<sd>/reports/``, the host dir for generated report files.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/reports``.
    """
    return Path(session_dir) / "reports"


def report_file(session_dir: Path, ts: str, suffix: str = "md") -> Path:
    """Compute the path to a final report file ``<sd>/reports/<ts>_final.<suffix>``.

    Args:
        session_dir (Path): The session root directory.
        ts (str): The timestamp tag used as the filename prefix.
        suffix (str): The file extension without the dot (e.g. ``"md"`` or
            ``"json"``). Defaults to ``"md"``.

    Returns:
        Path: The absolute path to ``<session_dir>/reports/<ts>_final.<suffix>``.
    """
    return reports_dir(session_dir) / f"{ts}_final.{suffix}"


# ---------------------------------------------------------------------------
# Full-trace artefacts (token + decision timeline) under reports/trace/
# ---------------------------------------------------------------------------
# Layout (see FULL_TRACE_DESIGN §3.3):
#
#   <sd>/reports/trace/
#     llm_calls.jsonl              # in-process components append directly
#     ext/<component>-<pid>.jsonl  # each out-of-process child writes its own
#     decision_trace.jsonl         # collector join product (token+decision)
#
# All trace writers are best-effort and swallow OSError; these helpers only
# compute paths (callers mkdir the parent before writing).
def trace_dir(session_dir: Path) -> Path:
    """``<sd>/reports/trace/`` — root of the unified token+decision trace."""
    return reports_dir(session_dir) / "trace"


def llm_calls_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/llm_calls.jsonl`` — append-only ledger of every
    in-process LLM call (orchestration / kernel / specialist
    in-process fallback / codex / critic / proposal_scorer).

    Out-of-process children write to :func:`ext_trace_path` instead; the
    collector merges both streams.
    """
    return trace_dir(session_dir) / "llm_calls.jsonl"


def trace_ext_dir(session_dir: Path) -> Path:
    """``<sd>/reports/trace/ext/`` — parent of every out-of-process child's
    own ``<component>-<pid>.jsonl`` shard."""
    return trace_dir(session_dir) / "ext"


def ext_trace_path(session_dir: Path, component: str, pid: int) -> Path:
    """``<sd>/reports/trace/ext/<component>-<pid>.jsonl``.

    Each independent agent process (geak / oob / robustness / critic-agent
    CLI / tracelens) writes its own shard so concurrent children never
    contend on a shared file; the collector globs ``ext/*.jsonl`` and merges.
    The ``pid`` keeps shards disjoint across re-spawns of the same component.
    """
    comp = str(component or "").strip() or "unknown"
    return trace_ext_dir(session_dir) / f"{comp}-{int(pid)}.jsonl"


def decision_trace_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/decision_trace.jsonl`` — collector output joining
    every decision to its LLM token spend along the phase→tick timeline."""
    return trace_dir(session_dir) / "decision_trace.jsonl"


def conversations_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/conversations.jsonl`` — append-only record of the
    full prompt + completion text for every in-process LLM call.

    Sibling of :func:`llm_calls_path`: that ledger holds the *token* account
    (kept small, no prompt text — see FULL_TRACE_DESIGN §9), while this file
    carries the *conversation* (redacted full prompt/response) so a session
    can be replayed or exported (e.g. to Langfuse) after the fact. Both share
    the same ``session_id`` / ``component`` / ``tick`` / ``phase`` join keys
    so the two streams line up against ``decision_trace``.
    """
    return trace_dir(session_dir) / "conversations.jsonl"


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
    """Compute ``<sd>/logs/``, the host dir for per-agent log files.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/logs``.
    """
    return Path(session_dir) / "logs"


def agent_log(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/logs/<role>.log``, one append-only log per persistent agent.

    Args:
        session_dir (Path): The session root directory.
        role (str): The persistent agent role name.

    Returns:
        Path: The absolute path to ``<session_dir>/logs/<role>.log``.
    """
    return logs_dir(session_dir) / f"{role}.log"


# Launcher-side artefacts (stdout, PID file, robustness monitor logs)
def optimizer_runs_dir(session_dir: Path) -> Path:
    """``<sd>/optimizer_runs/`` — launcher artefacts (run_<tag>.log / .pid /
    robustness_monitor_*.log). Under $USER_DATA_PATH so one override moves
    the whole run tail.
    """
    return Path(session_dir) / "optimizer_runs"


def optimizer_run_log(session_dir: Path, run_tag: str) -> Path:
    """Compute the launcher stdout log path ``<sd>/optimizer_runs/run_<tag>.log``.

    Args:
        session_dir (Path): The session root directory.
        run_tag (str): The launcher run tag; blank/empty falls back to
            ``"unknown"``.

    Returns:
        Path: The absolute path to
            ``<session_dir>/optimizer_runs/run_<tag>.log``.
    """
    tag = str(run_tag or "").strip() or "unknown"
    return optimizer_runs_dir(session_dir) / f"run_{tag}.log"


def optimizer_run_pidfile(session_dir: Path, run_tag: str) -> Path:
    """Compute the launcher pid-file path ``<sd>/optimizer_runs/run_<tag>.pid``.

    Args:
        session_dir (Path): The session root directory.
        run_tag (str): The launcher run tag; blank/empty falls back to
            ``"unknown"``.

    Returns:
        Path: The absolute path to
            ``<session_dir>/optimizer_runs/run_<tag>.pid``.
    """
    tag = str(run_tag or "").strip() or "unknown"
    return optimizer_runs_dir(session_dir) / f"run_{tag}.pid"


# Per-agent inbox/outbox + system prompt snapshot
def agent_dir(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/agents/<role>/``, the per-agent artefact root.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.

    Returns:
        Path: The absolute path to ``<session_dir>/agents/<role>``.
    """
    return Path(session_dir) / "agents" / role


def agent_inbox(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/agents/<role>/inbox.jsonl``, the agent's inbound message log.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.

    Returns:
        Path: The absolute path to ``<session_dir>/agents/<role>/inbox.jsonl``.
    """
    return agent_dir(session_dir, role) / "inbox.jsonl"


def agent_outbox(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/agents/<role>/outbox.jsonl``, the agent's outbound message log.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.

    Returns:
        Path: The absolute path to ``<session_dir>/agents/<role>/outbox.jsonl``.
    """
    return agent_dir(session_dir, role) / "outbox.jsonl"


def agent_persona(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/agents/<role>/persona.md``, the agent's persona document.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.

    Returns:
        Path: The absolute path to ``<session_dir>/agents/<role>/persona.md``.
    """
    return agent_dir(session_dir, role) / "persona.md"


def agent_prompt_snapshot(session_dir: Path, role: str) -> Path:
    """Compute the path to the per-agent system-prompt snapshot.

    Written once at Coordinator start, then read for resume / drift
    inspection.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.

    Returns:
        Path: The absolute path to
            ``<session_dir>/agents/<role>/system_prompt.snapshot.md``.
    """
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
    """Compute the path to the machine-readable target ``BaselineSummary``.

    Written by ``target_analysis`` and read by ``report`` to render an
    advisory section in ``final.md``.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/target_analysis/target_baseline.json``.
    """
    return target_analysis_dir(session_dir) / "target_baseline.json"


def target_analysis_report_md(session_dir: Path) -> Path:
    """Compute the path to the short human-readable target-analysis note.

    The note is suitable for inclusion / linking from the final report.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/target_analysis/target_analysis_report.md``.
    """
    return target_analysis_dir(session_dir) / "target_analysis_report.md"


# Cortex KB integration paths — single source of truth for every file under
# ``<sd>/runtime/cortex/``. Callers MUST go through these helpers so the NDJSON
# protocol stays homogeneous across producers/consumers.
def cortex_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/cortex/``, the Cortex KB per-session bookkeeping root.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/cortex``.
    """
    return Path(session_dir) / "runtime" / "cortex"


def cortex_sid_file(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_sid`` — Cortex session id from T0 ``session
    begin`` (resume reuses it). Absent => --degraded-kb or T0 not yet run.
    """
    return cortex_dir(session_dir) / ".kb_sid"


def cortex_warm_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_warm.json``, the T0 ``find-recipe`` snapshot.

    Read by §3.5 specialist assembly (M5).

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_warm.json``.
    """
    return cortex_dir(session_dir) / ".kb_warm.json"


def cortex_pitfalls_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_pitfalls.json``, the T0 ``traps`` snapshot.

    Read by §3.5 specialist assembly (M5).

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_pitfalls.json``.
    """
    return cortex_dir(session_dir) / ".kb_pitfalls.json"


def cortex_pending_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_pending.ndjson`` — append-only async write
    queue for T2/T3 ops. Consumed by the cortex_kb_flusher daemon; drained at
    T4 before ``session commit``.
    """
    return cortex_dir(session_dir) / ".kb_pending.ndjson"


def cortex_flushed_ndjson(session_dir: Path) -> Path:
    """Compute the path to ``.kb_flushed.ndjson``, the successfully-POSTed rows.

    Kept around for offline audit / breakdown collection.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_flushed.ndjson``.
    """
    return cortex_dir(session_dir) / ".kb_flushed.ndjson"


def cortex_dead_letter_ndjson(session_dir: Path) -> Path:
    """Compute the path to ``.kb_dead_letter.ndjson``, the permanent-failure rows.

    Holds rows that failed permanently (HTTP 4xx business-logic rejects);
    raises a robustness HIGH alert.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_dead_letter.ndjson``.
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
    """Compute ``<sd>/runtime/recipe_snapshot/``, the dispatcher bookkeeping root.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_snapshot``.
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
    """Compute the path to ``.kb_flusher.pid``, the flusher daemon pid file.

    The one-line file is read by robustness checks to detect a dead flusher.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_flusher.pid``.
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
    "conversations_path",
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
    "decision_trace_path",
    "ext_trace_path",
    "kernel_agent_runs_dir",
    "kernel_workspace",
    "llm_calls_path",
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
    "trace_dir",
    "trace_ext_dir",
]
