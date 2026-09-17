# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session path helpers — single source of truth for every path *inside* a session directory (``paths.py`` owns *where* the session lives)."""

from __future__ import annotations

import re
from pathlib import Path

from ..protocol.action_surfaces import ACTION_CATALOGUE

# Named because session_package.py needs them as glob strings, where a Path helper
# does not fit.
BRINGUP_SEGMENT: str = "bringup"
ENABLEMENT_SEGMENT: str = "enablement"


# Top-level files
def manifest_path(session_dir: Path) -> Path:
    """Compute the path to ``manifest.json`` (the Python-written resume tag)."""
    return Path(session_dir) / "manifest.json"


def state_path(session_dir: Path) -> Path:
    """Compute the path to ``state.json`` (the Coordinator-written SharedState)."""
    return Path(session_dir) / "state.json"


def optimizer_lock_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/optimizer.lock`` — the single-optimizer session lock."""
    return Path(session_dir) / "runtime" / "optimizer.lock"


def supervisor_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/supervisor`` — the out-of-band supervisor's own store.

    Everything the supervisor writes lives here: it must not be a second
    writer to ``coordinator.db``, which may sit on a network filesystem.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/supervisor``.
    """
    return Path(session_dir) / "runtime" / "supervisor"


def coordinator_tick_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/supervisor/coordinator_tick.json``.

    Stamped by the coordinator at the top of every tick and read by the
    supervisor, which uses it to tell a wedged loop from a busy one.

    Schema: ``{pid, hostname, tick, stamped_unix}``.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to the tick stamp.
    """
    return supervisor_dir(session_dir) / "coordinator_tick.json"


def supervisor_status_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/supervisor/status.json``.

    The supervisor's own view of the session, rewritten every poll: what it
    observed, what it would do, and what it actually did.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to the status snapshot.
    """
    return supervisor_dir(session_dir) / "status.json"


def supervisor_log_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/supervisor/supervisor.log``.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: Where the supervisor process's own output is redirected.
    """
    return supervisor_dir(session_dir) / "supervisor.log"


def pod_history_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/pod_history.jsonl`` — the optimizer-owner ledger."""
    return Path(session_dir) / "runtime" / "pod_history.jsonl"


# Phases (from the catalogue ``pipeline_phase`` field) whose executors own a per-task ``runs/<action>/<task_id>/``
# workspace.
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


# Action names that own a ``runs/<kind>/<task_id>/`` workspace.
_RUNS_ACTIONS: frozenset[str] = frozenset(
    a.name for a in ACTION_CATALOGUE.values() if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
)


def _validate_action(action: str) -> str:
    """Normalise and validate an action name against the runs-workspace set."""
    a = str(action or "").strip()
    if a not in _RUNS_ACTIONS:
        raise ValueError(f"runs_dir: unknown action {action!r}; expected one of {sorted(_RUNS_ACTIONS)!r}")
    return a


def _validate_id_component(value: str, *, field: str) -> str:
    """Reject blank ids and path-traversal in an LLM-controlled single-segment id."""
    v = str(value or "").strip()
    if not v or v == "." or "/" in v or "\\" in v or ".." in Path(v).parts or Path(v).is_absolute():
        raise ValueError(f"{field}: unsafe path component {value!r}")
    return v


def fs_safe_id(value: str, *, fallback: str = "anon") -> str:
    """Fold an identifier into a single path segment every filesystem accepts."""
    folded = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")
    return folded or fallback


def runs_root(session_dir: Path) -> Path:
    """Compute ``<sd>/runs/``, the parent of all per-action subtrees."""
    return Path(session_dir) / "runs"


def runs_dir(session_dir: Path, action: str, task_id: str) -> Path:
    """Compute ``<sd>/runs/<action>/<task_id>/``, a per-task data-plane workspace."""
    a = _validate_action(action)
    tid = _validate_id_component(task_id, field="runs_dir.task_id")
    return runs_root(session_dir) / a / tid


# Suffix probes before unique_runs_dir gives up.
_MAX_RUNS_DIR_ATTEMPTS: int = 200


def unique_runs_dir(session_dir: Path, action: str, task_id: str) -> Path:
    """Create a fresh :func:`runs_dir` workspace, suffixing ``-2``, ``-3``, … when earlier attempts already claimed the name."""
    base = runs_dir(session_dir, action, task_id)
    for suffix in range(1, _MAX_RUNS_DIR_ATTEMPTS + 1):
        candidate = base if suffix == 1 else base.with_name(f"{base.name}-{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"unique_runs_dir: {base} still taken after {_MAX_RUNS_DIR_ATTEMPTS} suffixes")


def kernel_agent_runs_root(session_dir: Path) -> Path:
    """``<sd>/kernel-agent/runs/`` — the parent of all per-tool-invocation kernel-agent run dirs (keyed by tool-invocation session id beneath it)."""
    return Path(session_dir) / "kernel-agent" / "runs"


def kernel_agent_runs_dir(session_dir: Path, session_id: str) -> Path:
    """``<sd>/kernel-agent/runs/<session_id>/`` — per-tool-invocation kernel-agent output (logs, status JSON, optimization_attempts.jsonl, TraceLens analysis)."""
    sid = _validate_id_component(session_id, field="kernel_agent_runs_dir.session_id")
    return kernel_agent_runs_root(session_dir) / sid


def patches_dir(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/patches/<kernel_id>/`` — KEEP-promoted on-disk changes: the original source backup + applied patch (REVERT restores from backup)."""
    kid = _validate_id_component(kernel_id, field="patches_dir.kernel_id")
    return Path(session_dir) / "patches" / kid


# Session-breakdown record fragments (recorder write-side spool).
def breakdown_parts_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/breakdown/parts/`` — per-producer breakdown record fragments."""
    return Path(session_dir) / "runtime" / "breakdown" / "parts"


# Reports / logs
def reports_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/reports/``, the host dir for generated report files."""
    return Path(session_dir) / "reports"


def sbd_v6_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/reports/sbd_v6/`` for V6 timeline source events."""
    return reports_dir(session_dir) / "sbd_v6"


def sbd_v6_timeline_dir(session_dir: Path) -> Path:
    """Compute the append-only V6 timeline event directory."""
    return sbd_v6_dir(session_dir) / "timeline"


def sbd_v6_timeline_event_path(session_dir: Path, sequence: int, event_type: str) -> Path:
    """Compute one ordered V6 timeline event path."""
    return sbd_v6_timeline_dir(session_dir) / f"{int(sequence):06d}-{event_type}.json"


def sbd_v6_write_warnings_path(session_dir: Path) -> Path:
    """Compute the durable V6 write-warning ledger path."""
    return sbd_v6_dir(session_dir) / "write_warnings.jsonl"


def enablement_dir(session_dir: Path) -> Path:
    """``<sd>/reports/enablement/`` — enablement round artifacts."""
    return reports_dir(session_dir) / ENABLEMENT_SEGMENT


def enablement_round_dir(session_dir: Path, task_id: str) -> Path:
    """``<sd>/reports/enablement/<task_id>/`` — one directory per round."""
    tid = _validate_id_component(task_id, field="enablement_round_dir.task_id")
    return enablement_dir(session_dir) / tid


def bringup_dir(session_dir: Path) -> Path:
    """``<sd>/reports/bringup/`` — bring-up observation artifacts."""
    return reports_dir(session_dir) / BRINGUP_SEGMENT


def enablement_builds_dir(session_dir: Path, task_id: str) -> Path:
    """``<sd>/enablement/builds/<task_id>/`` — targeted-build workspace for one task."""
    tid = _validate_id_component(task_id, field="enablement_builds_dir.task_id")
    return Path(session_dir) / ENABLEMENT_SEGMENT / "builds" / tid


def enablement_stacks_dir(session_dir: Path) -> Path:
    """``<sd>/enablement/stacks/`` — venv roots for enablement launch attempts."""
    return Path(session_dir) / ENABLEMENT_SEGMENT / "stacks"


# Full-trace artefacts (token + decision timeline) under reports/trace/.
def trace_dir(session_dir: Path) -> Path:
    """``<sd>/reports/trace/`` — root of the unified token+decision trace."""
    return reports_dir(session_dir) / "trace"


def llm_calls_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/llm_calls.jsonl`` — append-only ledger of every in-process LLM call; the ``component`` label is drawn from the closed set :data:`hyperloom.orchestrator.trace.llm_trace.VALID_COMPONENTS` (e.g. orchestration / kernel_agent / specialist / critic)."""
    return trace_dir(session_dir) / "llm_calls.jsonl"


def trace_ext_dir(session_dir: Path) -> Path:
    """``<sd>/reports/trace/ext/`` — parent of every out-of-process child's own ``<component>-<pid>.jsonl`` shard."""
    return trace_dir(session_dir) / "ext"


def decision_trace_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/decision_trace.jsonl`` — collector output joining every decision to its LLM token spend along the phase→tick timeline."""
    return trace_dir(session_dir) / "decision_trace.jsonl"


def proposal_task_map_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/proposal_task_map.jsonl`` — append-only map of ``{proposal_msg_id -> task_id}`` stamped when an approved proposal is materialized into a task."""
    return trace_dir(session_dir) / "proposal_task_map.jsonl"


def forge_steps_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/forge_steps.jsonl`` — append-only audit of the Kernel-Forge autonomous loop's key steps (per-iteration rationale / validation / bench / keep-revert + a run summary), recovered from the forge kernel-backend stdout."""
    return trace_dir(session_dir) / "forge_steps.jsonl"


def gemm_tuning_steps_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/gemm_tuning.jsonl`` — append-only audit of each GEMM-tuning run (forge / geak), one row per dispatched run carrying the tuning ``engine``, micro-decision, best speedup and per-tuner summary."""
    return trace_dir(session_dir) / "gemm_tuning.jsonl"


def specialist_intel_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/specialist_intel.jsonl`` — append-only audit of the intel/tool calls each specialist made (WebSearch / WebFetch / pr_monitor / recipe_kb / Read / Grep / ...), recovered from the subprocess stream-json log."""
    return trace_dir(session_dir) / "specialist_intel.jsonl"


def conversations_path(session_dir: Path) -> Path:
    """``<sd>/reports/trace/conversations.jsonl`` — append-only record of the full prompt + completion text for every in-process LLM call."""
    return trace_dir(session_dir) / "conversations.jsonl"


def research_hints_md(session_dir: Path) -> Path:
    """``<sd>/research_hints.md`` — human-readable proven-prior hints collected by the research scout."""
    return Path(session_dir) / "research_hints.md"


def research_hints_json(session_dir: Path) -> Path:
    """``<sd>/research_hints.json`` — structured mirror of the research hints (machine-readable; advisory gap-scoring reads this)."""
    return Path(session_dir) / "research_hints.json"


def forge_cycle_dir(session_dir: Path, macro_cycle: int) -> Path:
    """Return the output root for one Forge macro cycle."""
    cycle = max(0, int(macro_cycle))
    return Path(session_dir) / "kernel-agent" / "forge" / f"cycle-{cycle}"


def forge_attempt_dir(session_dir: Path, macro_cycle: int, attempt: int) -> Path:
    """Return the output root for one Forge controller attempt inside a cycle."""
    return forge_cycle_dir(session_dir, macro_cycle) / f"attempt-{max(0, int(attempt))}"


def next_forge_attempt_dir(session_dir: Path, macro_cycle: int) -> Path:
    """Return a fresh attempt directory for this macro cycle."""
    cycle_root = forge_cycle_dir(session_dir, macro_cycle)
    highest = -1
    if cycle_root.is_dir():
        for entry in cycle_root.iterdir():
            if not entry.is_dir() or not entry.name.startswith("attempt-"):
                continue
            try:
                highest = max(highest, int(entry.name[len("attempt-") :]))
            except ValueError:
                continue
    return forge_attempt_dir(session_dir, macro_cycle, highest + 1)


def forge_handoff_dir(session_dir: Path, macro_cycle: int) -> Path:
    """Return the handoff directory for one Forge macro cycle."""
    return forge_cycle_dir(session_dir, macro_cycle) / "handoff"


def competitor_target_json(session_dir: Path) -> Path:
    """``<sd>/competitor_target.json`` — LLM-authored competitor target numbers (each per-concurrency entry carries its own source)."""
    return Path(session_dir) / "competitor_target.json"


# Per-agent artefacts
def agent_dir(session_dir: Path, role: str) -> Path:
    """Compute ``<sd>/agents/<role>/``, the per-agent artefact root."""
    return Path(session_dir) / "agents" / role


def agent_prompt_snapshot(session_dir: Path, role: str, *, phase: str = "") -> Path:
    """Compute the path to the per-agent system-prompt snapshot."""
    stem = f"system_prompt.{phase.strip().upper()}" if phase.strip() else "system_prompt"
    return agent_dir(session_dir, role) / f"{stem}.snapshot.md"


def agent_mcp_setup_path(session_dir: Path, role: str) -> Path:
    """Compute the per-agent MCP setup snapshot path."""
    return agent_dir(session_dir, role) / "mcp_setup.json"


# External baseline comparison artefacts.
def target_analysis_dir(session_dir: Path) -> Path:
    """``<sd>/target_analysis/`` — external baseline artefacts."""
    return Path(session_dir) / "target_analysis"


def target_baseline_json(session_dir: Path) -> Path:
    """Compute the path to the machine-readable target ``BaselineSummary``."""
    return target_analysis_dir(session_dir) / "target_baseline.json"


def target_analysis_report_md(session_dir: Path) -> Path:
    """Compute the path to the short human-readable target-analysis note."""
    return target_analysis_dir(session_dir) / "target_analysis_report.md"


# Recipe KB integration paths — single source of truth for every file under ``<sd>/runtime/recipe_kb/``.


def recipe_kb_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/recipe_kb/``, the Recipe KB per-session bookkeeping root.

    This directory holds only *derived* bookkeeping — the authoritative recipe
    store is the local KB root (``$HYPERLOOM_LOCAL_KB_ROOT`` / ``workspace_root()/kb``,
    mirrored to gbrain), which lives outside the session tree. The snapshots
    here (``.kb_pitfalls.json`` / ``.kb_lessons.json``) are rewritten by every
    T0 anchor, so a session that predates the ``runtime/cortex`` ->
    ``runtime/recipe_kb`` rename simply regenerates them; no migration is needed.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/recipe_kb``.
    """
    return Path(session_dir) / "runtime" / "recipe_kb"


def recipe_kb_pitfalls_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_pitfalls.json``, the T0 ``traps`` snapshot."""
    return recipe_kb_dir(session_dir) / ".kb_pitfalls.json"


def recipe_kb_lessons_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_lessons.json``, the T0 ``lessons`` snapshot."""
    return recipe_kb_dir(session_dir) / ".kb_lessons.json"


def recipe_kb_pending_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.kb_pending.ndjson`` — legacy async KB write queue."""
    return recipe_kb_dir(session_dir) / ".kb_pending.ndjson"


def recipe_kb_flushed_ndjson(session_dir: Path) -> Path:
    """Compute the path to ``.kb_flushed.ndjson``, the successfully-POSTed rows."""
    return recipe_kb_dir(session_dir) / ".kb_flushed.ndjson"


def recipe_kb_dead_letter_ndjson(session_dir: Path) -> Path:
    """Compute the path to ``.kb_dead_letter.ndjson``, the permanent-failure rows."""
    return recipe_kb_dir(session_dir) / ".kb_dead_letter.ndjson"


def recipe_kb_audit_jsonl(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.kb_audit.jsonl`` — reserved append-only audit slot for Recipe KB CLI invocations; no producer writes it today and no breakdown section reads it (the former ``kb_provenance`` audit counts were dropped in the V5→V6 migration)."""
    return recipe_kb_dir(session_dir) / ".kb_audit.jsonl"


# recipe-snapshot per-session bookkeeping.
def recipe_snapshot_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/recipe_snapshot/``, the dispatcher bookkeeping root."""
    return Path(session_dir) / "runtime" / "recipe_snapshot"


def recipe_snapshot_audit_jsonl(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_snapshot/.audit.jsonl`` — append-only audit of local Recipe operations and remote KB Store publish attempts."""
    return recipe_snapshot_dir(session_dir) / ".audit.jsonl"


def pr_monitor_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.pr_monitor_status.json`` — boot-time PR Monitor reachability snapshot."""
    return recipe_kb_dir(session_dir) / ".pr_monitor_status.json"


def recipe_kb_flusher_pid(session_dir: Path) -> Path:
    """Compute the path to ``.kb_flusher.pid``, the flusher daemon pid file."""
    return recipe_kb_dir(session_dir) / ".kb_flusher.pid"


def recipe_kb_flusher_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.kb_flusher_status.json`` — boot-time flusher spawn decision."""
    return recipe_kb_dir(session_dir) / ".kb_flusher_status.json"


def _prune_old_workdirs(root: Path, *, keep: int) -> None:
    """Delete all but the newest ``keep`` per-turn workdirs under *root*."""
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
    """Allocate (and create) ``<sd>/<subdir>/<turn_idx:06d>/`` for a subprocess agent's per-turn scratch, pruning stale turn dirs down to the newest *keep*."""
    root = Path(session_dir) / subdir
    root.mkdir(parents=True, exist_ok=True)
    _prune_old_workdirs(root, keep=keep)
    wd = root / f"{turn_idx:06d}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def session_failures_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/reports/failures/`` — durable failure evidence store."""
    return reports_dir(session_dir) / "failures"


def failure_evidence_path(session_dir: Path, failure_id: str) -> Path:
    """Compute the path for one failure evidence JSON file."""
    return session_failures_dir(session_dir) / f"{failure_id}.json"


__all__ = [
    "allocate_turn_workdir",
    "agent_dir",
    "agent_mcp_setup_path",
    "agent_prompt_snapshot",
    "breakdown_parts_dir",
    "competitor_target_json",
    "conversations_path",
    "recipe_kb_audit_jsonl",
    "recipe_kb_dead_letter_ndjson",
    "recipe_kb_dir",
    "recipe_kb_flushed_ndjson",
    "recipe_kb_flusher_pid",
    "recipe_kb_flusher_status_json",
    "recipe_kb_lessons_json",
    "recipe_kb_pending_ndjson",
    "recipe_kb_pitfalls_json",
    "decision_trace_path",
    "proposal_task_map_path",
    "forge_steps_path",
    "gemm_tuning_steps_path",
    "kernel_agent_runs_dir",
    "kernel_agent_runs_root",
    "llm_calls_path",
    "manifest_path",
    "patches_dir",
    "failure_evidence_path",
    "forge_cycle_dir",
    "forge_handoff_dir",
    "BRINGUP_SEGMENT",
    "ENABLEMENT_SEGMENT",
    "bringup_dir",
    "enablement_builds_dir",
    "enablement_dir",
    "enablement_round_dir",
    "enablement_stacks_dir",
    "reports_dir",
    "research_hints_json",
    "session_failures_dir",
    "research_hints_md",
    "runs_dir",
    "runs_root",
    "sbd_v6_dir",
    "sbd_v6_timeline_dir",
    "sbd_v6_timeline_event_path",
    "sbd_v6_write_warnings_path",
    "state_path",
    "target_analysis_dir",
    "target_analysis_report_md",
    "target_baseline_json",
    "trace_dir",
]
