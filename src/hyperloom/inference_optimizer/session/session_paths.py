# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session path helpers — single source of truth for every path *inside*
a session directory (``paths.py`` owns *where* the session lives).

Hard rule: all code MUST derive sub-paths under ``session_dir`` through this
module; no ad-hoc string concatenation elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from ..protocol.action_surfaces import ACTION_CATALOGUE


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


def pod_history_path(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/pod_history.jsonl`` — the optimizer-owner ledger.

    ``optimizer.lock`` holds only the *current* owner: each acquirer truncates
    and rewrites it, so a session whose sandbox was rebuilt mid-run keeps no
    record of the pods that came before. ``manifest.json`` pins the first owner
    and the lock pins the last, which makes a multi-rebuild session read like a
    single-pod one in post-mortem. This append-only ledger records one line per
    acquisition so the whole ownership chain survives.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/pod_history.jsonl``.
    """
    return Path(session_dir) / "runtime" / "pod_history.jsonl"


# Phases (from the catalogue ``pipeline_phase`` field) whose executors own
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


# Action names that own a ``runs/<kind>/<task_id>/`` workspace.
_RUNS_ACTIONS: frozenset[str] = frozenset(
    a.name for a in ACTION_CATALOGUE.values() if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
)


def _validate_action(action: str) -> str:
    """Normalise and validate an action name against the runs-workspace set.

    Args:
        action (str): The candidate action name (whitespace-stripped before
            comparison).

    Returns:
        str: The stripped action name when it is recognised.

    Raises:
        ValueError: If the action does not own a runs-workspace.
    """
    a = str(action or "").strip()
    if a not in _RUNS_ACTIONS:
        raise ValueError(f"runs_dir: unknown action {action!r}; expected one of {sorted(_RUNS_ACTIONS)!r}")
    return a


def _validate_id_component(value: str, *, field: str) -> str:
    """Reject blank ids and path-traversal in an LLM-controlled single-segment id.

    Legitimate ids (uuid hex, ``k001``) are never blank and never contain a
    separator or ``..``; either would relocate a per-task sandbox.
    """
    v = str(value or "").strip()
    if not v or v == "." or "/" in v or "\\" in v or ".." in Path(v).parts or Path(v).is_absolute():
        raise ValueError(f"{field}: unsafe path component {value!r}")
    return v


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
        task_id (str): The task identifier; must be non-blank.

    Returns:
        Path: The absolute path to ``<session_dir>/runs/<action>/<task_id>``.

    Raises:
        ValueError: If ``action`` is not a recognised runs-workspace action, or
            ``task_id`` is blank or path-like.
    """
    a = _validate_action(action)
    tid = _validate_id_component(task_id, field="runs_dir.task_id")
    return runs_root(session_dir) / a / tid


# Suffix probes before unique_runs_dir gives up.
_MAX_RUNS_DIR_ATTEMPTS: int = 200


def unique_runs_dir(session_dir: Path, action: str, task_id: str) -> Path:
    """Create a fresh :func:`runs_dir` workspace, suffixing ``-2``, ``-3``, …
    when earlier attempts already claimed the name. ``mkdir(exist_ok=False)``
    makes the claim atomic against concurrent callers.

    Args:
        session_dir (Path): The session root directory.
        action (str): The owning action name; validated against the
            runs-workspace action set.
        task_id (str): The task identifier; must be non-blank.

    Returns:
        Path: The newly created workspace directory.

    Raises:
        ValueError: If ``action`` is not a recognised runs-workspace action, or
            ``task_id`` is blank or path-like.
        RuntimeError: If every suffix up to ``_MAX_RUNS_DIR_ATTEMPTS`` is taken.
    """
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
        session_id: Tool-invocation session id; must be non-blank.

    Returns:
        ``<session_dir>/kernel-agent/runs/<session_id>``.
    """
    sid = _validate_id_component(session_id, field="kernel_agent_runs_dir.session_id")
    return kernel_agent_runs_root(session_dir) / sid


def patches_dir(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/patches/<kernel_id>/`` — KEEP-promoted on-disk changes: the
    original source backup + applied patch (REVERT restores from backup).

    Args:
        session_dir: The session root directory.
        kernel_id: Kernel id keying the patch dir; must be non-blank.

    Returns:
        ``<session_dir>/patches/<kernel_id>``.
    """
    kid = _validate_id_component(kernel_id, field="patches_dir.kernel_id")
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
    """``<sd>/reports/enablement/`` — enablement round artifacts.

    Lives under ``reports/`` because the archive collector drops ``runs/``
    wholesale but retains this subtree.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/reports/enablement``.
    """
    return reports_dir(session_dir) / "enablement"


def enablement_round_dir(session_dir: Path, task_id: str) -> Path:
    """``<sd>/reports/enablement/<task_id>/`` — one directory per round.

    Args:
        session_dir: The session root directory.
        task_id: The specialist task id that drove the round.

    Returns:
        ``<session_dir>/reports/enablement/<task_id>``.

    Raises:
        ValueError: If ``task_id`` is not a safe single path component.
    """
    tid = _validate_id_component(task_id, field="enablement_round_dir.task_id")
    return enablement_dir(session_dir) / tid


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
    in-process LLM call; the ``component`` label is drawn from the closed set
    :data:`hyperloom.orchestrator.trace.llm_trace.VALID_COMPONENTS`
    (e.g. orchestration / kernel_agent / specialist / critic).

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
    recipe_kb / Read / Grep / ...), recovered from the subprocess stream-json
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


def agent_prompt_snapshot(session_dir: Path, role: str, *, phase: str = "") -> Path:
    """Compute the path to the per-agent system-prompt snapshot.

    The unsuffixed name holds the boot prompt; a phase suffix names one scope
    the Coordinator later re-scoped orchestration into.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role name.
        phase (str): Pipeline phase this prompt was scoped to; ``""`` selects
            the unsuffixed boot snapshot.

    Returns:
        Path: The absolute path to
            ``<session_dir>/agents/<role>/system_prompt[.<PHASE>].snapshot.md``.
    """
    stem = f"system_prompt.{phase.strip().upper()}" if phase.strip() else "system_prompt"
    return agent_dir(session_dir, role) / f"{stem}.snapshot.md"


def agent_mcp_setup_path(session_dir: Path, role: str) -> Path:
    """Compute the per-agent MCP setup snapshot path."""
    return agent_dir(session_dir, role) / "mcp_setup.json"


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


# Recipe KB integration paths — single source of truth for every file under
# ``<sd>/runtime/recipe_kb/``. Callers MUST go through these helpers so the NDJSON
# protocol stays homogeneous across producers/consumers.


def recipe_kb_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/runtime/recipe_kb/``, the Recipe KB per-session bookkeeping root.

    This directory holds only *derived* bookkeeping — the authoritative recipe
    store is the local KB root (``$HYPERLOOM_LOCAL_KB_ROOT`` / ``workspace_root()/kb``,
    mirrored to gbrain), which lives outside the session tree. The snapshots
    here (``.kb_warm.json`` / ``.kb_pitfalls.json`` / ``.kb_lessons.json``) are
    rewritten by every T0 anchor and duplicate ``shared_state.warm_start_recipe``,
    so a session that predates the ``runtime/cortex`` -> ``runtime/recipe_kb``
    rename simply regenerates them on resume; no migration is needed.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/recipe_kb``.
    """
    return Path(session_dir) / "runtime" / "recipe_kb"


def recipe_kb_warm_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_warm.json``, the T0 warm-start recipe snapshot.

    Written by ``recipe_kb_t0.run_t0_anchor`` from the
    ``_cascade_warm_start_search`` result. Write-only debug snapshot;
    specialist assembly reads ``shared_state.warm_start_recipe`` instead.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_kb/.kb_warm.json``.
    """
    return recipe_kb_dir(session_dir) / ".kb_warm.json"


def recipe_kb_pitfalls_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_pitfalls.json``, the T0 ``traps`` snapshot.

    Write-only debug snapshot; specialist assembly reads
    ``shared_state.warm_start_pitfalls`` instead.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_kb/.kb_pitfalls.json``.
    """
    return recipe_kb_dir(session_dir) / ".kb_pitfalls.json"


def recipe_kb_lessons_json(session_dir: Path) -> Path:
    """Compute the path to ``.kb_lessons.json``, the T0 ``lessons`` snapshot.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_kb/.kb_lessons.json``.
    """
    return recipe_kb_dir(session_dir) / ".kb_lessons.json"


def recipe_kb_pending_ndjson(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.kb_pending.ndjson`` — legacy async KB write
    queue. The flusher daemon that drained it was retired and nothing writes
    rows today; ``cli/preflight`` and the breakdown telemetry collector only
    report its (normally zero) depth.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/recipe_kb/.kb_pending.ndjson``.
    """
    return recipe_kb_dir(session_dir) / ".kb_pending.ndjson"


def recipe_kb_flushed_ndjson(session_dir: Path) -> Path:
    """Compute the path to ``.kb_flushed.ndjson``, the successfully-POSTed rows.

    Kept around for offline audit / breakdown collection.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_kb/.kb_flushed.ndjson``.
    """
    return recipe_kb_dir(session_dir) / ".kb_flushed.ndjson"


def recipe_kb_dead_letter_ndjson(session_dir: Path) -> Path:
    """Compute the path to ``.kb_dead_letter.ndjson``, the permanent-failure rows.

    Holds rows that failed permanently (HTTP 4xx business-logic rejects).
    Surfaced in preflight queue output and breakdown ``kb_provenance`` line
    counts; no in-repo robustness alert is wired to it.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_kb/.kb_dead_letter.ndjson``.
    """
    return recipe_kb_dir(session_dir) / ".kb_dead_letter.ndjson"


def recipe_kb_audit_jsonl(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.kb_audit.jsonl`` — reserved append-only audit
    slot for Recipe KB CLI invocations; no producer writes it today, so
    ``breakdown.kb_provenance.audit_tail_count`` / ``audit_status_counts`` stay
    empty. The live KB audit channel is :func:`recipe_snapshot_audit_jsonl`,
    written via ``RecipeKB.audit_hook`` in ``cli/kb.py``.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/recipe_kb/.kb_audit.jsonl``.
    """
    return recipe_kb_dir(session_dir) / ".kb_audit.jsonl"


# recipe-snapshot per-session bookkeeping. Separate ``runtime/recipe_snapshot/``
# subtree (not runtime/recipe_kb/). Writes are local-only, so only the read-side
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
    local Recipe operations and remote KB Store publish attempts.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/recipe_snapshot/.audit.jsonl``.
    """
    return recipe_snapshot_dir(session_dir) / ".audit.jsonl"


def pr_monitor_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.pr_monitor_status.json`` — boot-time PR Monitor
    reachability snapshot; breakdown reads it for pr_monitor:* warnings.

    Schema: ``{enabled, reachable, mcp_url, status_text}``.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/recipe_kb/.pr_monitor_status.json``.
    """
    return recipe_kb_dir(session_dir) / ".pr_monitor_status.json"


def recipe_kb_flusher_pid(session_dir: Path) -> Path:
    """Compute the path to ``.kb_flusher.pid``, the flusher daemon pid file.

    Probed by the breakdown telemetry collector (``_collect_flusher_status``),
    which reads the first line and ``os.kill(pid, 0)``s it to set
    ``kb_provenance.flusher_status.alive``. No in-repo component writes it.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_kb/.kb_flusher.pid``.
    """
    return recipe_kb_dir(session_dir) / ".kb_flusher.pid"


def recipe_kb_flusher_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_kb/.kb_flusher_status.json`` — boot-time flusher
    spawn decision; merged with the live pid check for
    kb_provenance.flusher_status.

    Schema: ``{enabled, spawned, pid, interval_sec, batch_size, reason, ts}``.

    Args:
        session_dir: The session root directory.

    Returns:
        ``<session_dir>/runtime/recipe_kb/.kb_flusher_status.json``.
    """
    return recipe_kb_dir(session_dir) / ".kb_flusher_status.json"


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


def session_failures_dir(session_dir: Path) -> Path:
    """Compute ``<sd>/reports/failures/`` — durable failure evidence store.

    Args:
        session_dir: The session root directory.

    Returns:
        Path: ``<session_dir>/reports/failures``.
    """
    return reports_dir(session_dir) / "failures"


def failure_evidence_path(session_dir: Path, failure_id: str) -> Path:
    """Compute the path for one failure evidence JSON file.

    Args:
        session_dir: The session root directory.
        failure_id: The failure's stable id (from ``make_failure_id``).

    Returns:
        Path: ``<session_dir>/reports/failures/<failure_id>.json``.
    """
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
    "recipe_kb_warm_json",
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
    "enablement_dir",
    "enablement_round_dir",
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
