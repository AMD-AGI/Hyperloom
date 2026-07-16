# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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


def optimizer_lock_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/optimizer.lock`` — the single-optimizer session lock.

    A live ``optimize`` process holds an exclusive advisory lock on this file
    for its whole lifetime and writes its owner metadata (pid / host /
    heartbeat) into it. A second optimizer attaching to the same session must
    fail fast instead of clobbering ``state.json`` / ``coordinator.db`` leases.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/optimizer.lock``.
    """
    return Path(session_dir) / "runtime" / "optimizer.lock"


# Phases (from the ActionRegistry ``pipeline_phase`` field) whose executors own
# a per-task ``runs/<action>/<task_id>/`` workspace.
_RUNS_WORKSPACE_PHASES: frozenset[str] = frozenset(
    {
        "measure",
        "analysis",
        "explore",
        "deep",
        "validate",
        "support",
    }
)

# Fallback used only when ActionRegistry can't load. Must stay in sync with the
# _RUNS_WORKSPACE_PHASES union; tests/test_action_catalogue.py enforces this.
_RUNS_ACTIONS_FALLBACK: frozenset[str] = frozenset(
    {
        "baseline",
        "replay_warm_recipe",
        "roofline",
        "profile",
        "sweep",
        "conc_sweep",
        "explore",
        "specialist",
        "integrate_patch",
        "framework_agent",
        "integrate",
        "kernel_opt",
        "deep_kernel_analysis",
        "gemm_tuning",
        "operator_tuning",
        "vendor_kernel_config",
        "recover",
    }
)


@lru_cache(maxsize=1)
def _runs_actions() -> frozenset[str]:
    """Action names that own a ``runs/<kind>/<task_id>/`` workspace, from
    action metadata. Lazy + cached; falls back to ``_RUNS_ACTIONS_FALLBACK``
    when the registry can't load.

    Returns:
        The set of action names that own a runs-workspace.
    """
    try:
        from hyperloom.orchestrator.actions.registry import ActionRegistry  # local: avoid import cycle

        registry = ActionRegistry().load()
    except Exception:
        return _RUNS_ACTIONS_FALLBACK
    return frozenset(a.name for a in registry.all() if a.pipeline_phase in _RUNS_WORKSPACE_PHASES)


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
        raise ValueError(f"runs_dir: unknown action {action!r}; expected one of {sorted(valid)!r}")
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


def kernel_agent_runs_root(session_dir: Path) -> Path:
    """``<sd>/kernel-agent/runs/`` — the parent of all per-tool-invocation
    kernel-agent run dirs (keyed by tool-invocation session id beneath it).

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/kernel-agent/runs``.
    """
    return Path(session_dir) / "kernel-agent" / "runs"


def kernel_agent_runs_dir(session_dir: Path, session_id: str) -> Path:
    """``<sd>/kernel-agent/runs/<session_id>/`` — per-tool-invocation
    kernel-agent output (logs, status JSON, optimization_attempts.jsonl,
    TraceLens analysis). Keyed by tool-invocation session id.

    Args:
        session_dir: The session root directory.
        session_id: Tool-invocation session id; blank falls back to
            ``"unknown"``.

    Returns:
        ``<session_dir>/kernel-agent/runs/<session_id>``.
    """
    sid = str(session_id or "").strip() or "unknown"
    return kernel_agent_runs_root(session_dir) / sid


def patches_dir(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/patches/<kernel_id>/`` — KEEP-promoted on-disk changes: the
    original source backup + applied patch (REVERT restores from backup).

    Args:
        session_dir: The session root directory.
        kernel_id: Kernel id keying the patch dir; blank falls back to
            ``"unknown"``.

    Returns:
        ``<session_dir>/patches/<kernel_id>``.
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "patches" / kid


# Session-breakdown record fragments (recorder write-side spool).
def breakdown_parts_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/breakdown/parts/`` — per-producer breakdown record
    fragments. Each owner writes its own files here (atomic + uniquely named);
    the exporter assembles them into ``session_breakdown.json``. Single-owner
    per section, so there is no cross-producer write contention.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/breakdown/parts``.
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


# Full-trace artefacts (token + decision timeline) under reports/trace/.
# Layout:
#
#   <sd>/reports/trace/
#     llm_calls.jsonl              # every in-process LLM call's token row
#     ext/<component>-<pid>.jsonl  # out-of-process child shards (compat path)
#     decision_trace.jsonl         # collector join product (token+decision)
#
# All trace writers are best-effort and swallow OSError; these helpers only
# compute paths. The parent process is the sole writer of llm_calls.jsonl.
# Out-of-process children write their own ext/*.jsonl shard under
# ``trace_ext_dir`` which the collector and the Langfuse emitter backfill at
# read time. The ext shards are a child-compatibility path: new producers
# should run in-process and parent-append into llm_calls.jsonl.
def trace_dir(session_dir: Path) -> Path:
    """``<sd>/reports/trace/`` — root of the unified token+decision trace.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/reports/trace``.
    """
    return reports_dir(session_dir) / "trace"


def llm_calls_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/llm_calls.jsonl`` — append-only ledger of every
    in-process LLM call (orchestration / kernel / specialist
    in-process fallback / codex / critic / proposal_scorer).

    Out-of-process child shards live under :func:`trace_ext_dir`; the collector
    merges both streams.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/reports/trace/llm_calls.jsonl``.
    """
    return trace_dir(session_dir) / "llm_calls.jsonl"


def trace_ext_dir(session_dir: Path) -> Path:
    """``<sd>/reports/trace/ext/`` — parent of every out-of-process child's
    own ``<component>-<pid>.jsonl`` shard.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/reports/trace/ext``.
    """
    return trace_dir(session_dir) / "ext"


def decision_trace_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/decision_trace.jsonl`` — collector output joining
    every decision to its LLM token spend along the phase→tick timeline.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/reports/trace/decision_trace.jsonl``.
    """
    return trace_dir(session_dir) / "decision_trace.jsonl"


def proposal_task_map_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/proposal_task_map.jsonl`` — append-only map of
    ``{proposal_msg_id -> task_id}`` stamped when an approved proposal is
    materialized into a task. Lets the decision-trace collector attribute a
    Critic review call (which only knows the proposal ``msg_id`` at review
    time) to the decision the proposal eventually became."""
    return trace_dir(session_dir) / "proposal_task_map.jsonl"


def forge_steps_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/forge_steps.jsonl`` — append-only audit of the
    Kernel-Forge autonomous loop's key steps (per-iteration rationale /
    validation / bench / keep-revert + a run summary), recovered from the forge
    kernel-backend stdout. Backfilled into the trace as ``forge:iter:<n>`` /
    ``forge:summary`` spans so a trace shows forge's decision process, not just
    its token total."""
    return trace_dir(session_dir) / "forge_steps.jsonl"


def gemm_tuning_steps_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/gemm_tuning.jsonl`` — append-only audit of each
    GEMM-tuning run (forge / geak), one row per dispatched run carrying the
    tuning ``engine``, micro-decision, best speedup and per-tuner summary.
    Backfilled into the trace as ``gemm_tuning:<engine>`` spans so a trace
    attributes the deterministic GEMM tuner as its own source, not just folds
    its gain into the kernel total."""
    return trace_dir(session_dir) / "gemm_tuning.jsonl"


def specialist_intel_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/specialist_intel.jsonl`` — append-only audit of the
    intel/tool calls each specialist made (WebSearch / WebFetch / pr_monitor /
    cortex_kb / Read / Grep / ...), recovered from the subprocess stream-json
    log. Backfilled into the trace as per-call ``intel:<tool>`` spans so a
    trace shows what a specialist *read*, not just its token total."""
    return trace_dir(session_dir) / "specialist_intel.jsonl"


def conversations_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/conversations.jsonl`` — append-only record of the
    full prompt + completion text for every in-process LLM call.

    Sibling of :func:`llm_calls_path`: that ledger holds the *token* account
    (kept small, no prompt text), while this file carries the *conversation*
    (redacted full prompt/response) so a session can be replayed or exported
    after the fact. Both share the same ``session_id`` / ``component`` /
    ``tick`` / ``phase`` join keys so the two streams line up against
    ``decision_trace``.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/reports/trace/conversations.jsonl``.
    """
    return trace_dir(session_dir) / "conversations.jsonl"


def research_hints_md(session_dir: Path) -> Path:
    """``<sd>/research_hints.md`` — human-readable proven-prior hints
    collected by the research scout.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/research_hints.md``.
    """
    return Path(session_dir) / "research_hints.md"


def research_hints_json(session_dir: Path) -> Path:
    """``<sd>/research_hints.json`` — structured mirror of the research
    hints (machine-readable; advisory gap-scoring reads this).

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/research_hints.json``.
    """
    return Path(session_dir) / "research_hints.json"


def competitor_target_json(session_dir: Path) -> Path:
    """``<sd>/competitor_target.json`` — LLM-authored competitor target
    numbers (each per-concurrency entry carries its own source).

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/competitor_target.json``.
    """
    return Path(session_dir) / "competitor_target.json"


# Per-agent artefacts
def agent_dir(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/agents/<role>/``, the per-agent artefact root.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.

    Returns:
        Path: The absolute path to ``<session_dir>/agents/<role>``.
    """
    return Path(session_dir) / "agents" / role


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
# runs/) because target_analysis is a prep-phase action.
def target_analysis_dir(session_dir: Path) -> Path:
    """``<sd>/target_analysis/`` — external baseline artefacts. Owner:
    TargetAnalysisExecutor; reader: ReportExecutor.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/target_analysis``.
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


def cortex_warm_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_warm.json``, the T0 ``find-recipe`` snapshot.

    Read by specialist assembly.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_warm.json``.
    """
    return cortex_dir(session_dir) / ".kb_warm.json"


def cortex_pitfalls_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_pitfalls.json``, the T0 ``traps`` snapshot.

    Read by specialist assembly.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_pitfalls.json``.
    """
    return cortex_dir(session_dir) / ".kb_pitfalls.json"


def cortex_lessons_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_lessons.json``, the T0 ``lessons`` snapshot.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_lessons.json``.
    """
    return cortex_dir(session_dir) / ".kb_lessons.json"


def cortex_pending_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_pending.ndjson`` — append-only async write
    queue for T2/T3 ops. Consumed by the cortex_kb_flusher daemon; drained at
    T4 before ``session commit``.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/cortex/.kb_pending.ndjson``.
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

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/cortex/.kb_audit.jsonl``.
    """
    return cortex_dir(session_dir) / ".kb_audit.jsonl"


# recipe-snapshot per-session bookkeeping. Separate ``runtime/recipe_snapshot/``
# subtree (not runtime/cortex/). Writes are local-only, so only the read-side
# audit log survives here.
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

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/recipe_snapshot/.audit.jsonl``.
    """
    return recipe_snapshot_dir(session_dir) / ".audit.jsonl"


def pr_monitor_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.pr_monitor_status.json`` — boot-time PR Monitor
    reachability snapshot; breakdown reads it for pr_monitor:* warnings.

    Schema: ``{enabled, url, reachable, mcp_url, window_days, status_text}``.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/cortex/.pr_monitor_status.json``.
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

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/cortex/.kb_flusher_status.json``.
    """
    return cortex_dir(session_dir) / ".kb_flusher_status.json"


def _prune_old_workdirs(root: Path, *, keep: int) -> None:
    """Delete all but the newest ``keep`` per-turn workdirs under *root*.

    Entries are sorted by name (zero-padded turn index, so lexical == chrono).
    All filesystem errors are swallowed best-effort — pruning must never break
    the caller.
    """
    try:
        entries = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return
    if len(entries) <= keep:
        return
    for stale in entries[: len(entries) - keep]:
        try:
            for child in stale.rglob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            for child in sorted(stale.rglob("*"), key=lambda p: -len(p.parts)):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        # Best-effort cleanup; the outer rmdir / next sweep retries.
                        pass
            stale.rmdir()
        except OSError:
            continue


def allocate_turn_workdir(session_dir: Path, subdir: str, turn_idx: int, *, keep: int) -> Path:
    """Allocate (and create) ``<sd>/<subdir>/<turn_idx:06d>/`` for a subprocess
    agent's per-turn scratch, pruning stale turn dirs down to the newest *keep*.

    Args:
        session_dir: The session root directory.
        subdir: The agent's workdir name under the session dir (e.g.
            ``"critic-workdir"`` / ``"robustness-workdir"``).
        turn_idx: The current turn index; rendered zero-padded to 6 digits.
        keep: How many of the most-recent turn dirs to retain.

    Returns:
        The created per-turn workdir path.
    """
    root = Path(session_dir) / subdir
    root.mkdir(parents=True, exist_ok=True)
    _prune_old_workdirs(root, keep=keep)
    wd = root / f"{turn_idx:06d}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


__all__ = [
    "allocate_turn_workdir",
    "agent_dir",
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
    "cortex_lessons_json",
    "cortex_pending_ndjson",
    "cortex_pitfalls_json",
    "cortex_warm_json",
    "decision_trace_path",
    "proposal_task_map_path",
    "forge_steps_path",
    "gemm_tuning_steps_path",
    "kernel_agent_runs_dir",
    "kernel_agent_runs_root",
    "llm_calls_path",
    "manifest_path",
    "patches_dir",
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
]
