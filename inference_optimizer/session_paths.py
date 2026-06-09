# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-session path helpers.

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

from functools import lru_cache
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
# Single source of truth for "which actions own a runs/<kind>/<task_id>/
# workspace": derived from the ActionRegistry via the ``pipeline_phase``
# yaml field. Phases listed here are exactly the ones whose executors
# write per-task artefacts under ``runs/``.
#
# The run-workspace set is derived from action metadata so adding a new
# executor-backed action does not require updating this file by hand.
_RUNS_WORKSPACE_PHASES: frozenset[str] = frozenset({
    "measure", "analysis", "explore", "deep", "validate", "support",
})

# Hardcoded fallback used only when ActionRegistry can't be loaded
# (broken yaml / partial install / very early bootstrap). MUST stay in
# sync with the union of action names whose ``pipeline_phase`` is in
# ``_RUNS_WORKSPACE_PHASES``; the regression test in
# ``tests/test_p1_2_full_action_catalogue.py`` enforces alignment.
# The fallback is intentionally explicit because it is used only when
# action metadata cannot be loaded during bootstrap.
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
    "assess_remaining_gaps",
    "dynamic_action",
    "dynamic_specialist",
    "dynamic_specialist_check",
    "dynamic_specialist_collect",
    "integrate", "kernel_opt", "deep_kernel_analysis", "gemm_tuning",
    "operator_tuning", "vendor_kernel_config",
    "recover",
})


@lru_cache(maxsize=1)
def _runs_actions() -> frozenset[str]:
    """Return the set of action names that own a ``runs/<kind>/<task_id>/``
    workspace, derived from action metadata.

    Lazy + cached so importing :mod:`session_paths` stays cheap (no
    PyYAML load at import time) and so repeated calls in hot paths
    (per-task ``_pre_mkdir_workspace`` / ``runs_dir``) are O(1).

    Falls back to ``_RUNS_ACTIONS_FALLBACK`` if the action registry
    can't be loaded — preferable to crashing at first use, since the
    fallback covers every action that production code currently dispatches.
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


# ---------------------------------------------------------------------------
# Kernel agent long-lived workspaces (cross-task, keyed by kernel_id)
# ---------------------------------------------------------------------------
def kernel_workspace(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/kernel-agent-workspace/<kernel_id>/``.

    Holds the extracted source, accumulated GEAK/OOB candidates and the
    currently chosen patch for one kernel. Survives across tasks so
    re-issuing run_optimization on the same kernel can reuse prior
    artefacts.

    Sibling of :func:`kernel_agent_runs_dir` — the two have intentionally
    disjoint scopes: this dir is keyed by ``kernel_id`` and survives
    across tool invocations, while ``kernel-agent/runs/<session_id>/``
    is keyed by a tool-invocation session id and holds per-call logs /
    status / TraceLens output.
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "kernel-agent-workspace" / kid


def kernel_agent_runs_dir(session_dir: Path, session_id: str) -> Path:
    """``<sd>/kernel-agent/runs/<session_id>/`` — kernel-agent tool output root.

    Distinct from :func:`kernel_workspace` (which is keyed by
    ``kernel_id`` and survives across tool calls). This path holds the
    per-tool-invocation artefacts produced by
    ``kernel-agent/tools/{tracelens_analysis,kernel_optimization,
    parallel_e2e_runner}.py``: per-run logs / status JSON,
    ``optimization_attempts.jsonl``, TraceLens ``standalone_analysis.md``,
    verification JSON, etc. — see ``kernel-agent/SKILL.md`` "Artifacts".

    The tools default ``--workspace-path`` to the session root and then
    write under this subdirectory; callers (Coordinator
    ``kernel_request_handlers``) pass ``--workspace-path=<sd>``.
    """
    sid = str(session_id or "").strip() or "unknown"
    return Path(session_dir) / "kernel-agent" / "runs" / sid


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
    in-process LLM call (orchestration / kernel / dynamic_action / specialist
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
    return Path(session_dir) / "logs"


def agent_log(session_dir: Path, role: str) -> Path:
    """``<sd>/logs/<role>.log`` — one append-only log per persistent agent."""
    return logs_dir(session_dir) / f"{role}.log"


# ---------------------------------------------------------------------------
# Launcher-side artefacts (stdout, PID file, robustness monitor logs)
# ---------------------------------------------------------------------------
def optimizer_runs_dir(session_dir: Path) -> Path:
    """``<sd>/optimizer_runs/`` — host of every ``setsid nohup`` launcher
    artefact (``run_<tag>.log`` / ``run_<tag>.pid`` /
    ``robustness_monitor_*.log``).

    The legacy SKILL template wrote these under ``$REPO_ROOT/optimizer_runs/``,
    which broke cross-shell tailing and made monitor scripts depend on the
    git checkout location. Sitting under ``$USER_DATA_PATH`` instead means
    a single ``USER_DATA_PATH`` override moves the whole run tail.
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


# ---------------------------------------------------------------------------
# dynamic_action artefact paths.
# Layout: ``<session_dir>/agents/orchestration/dynamic_actions/<dyn_id>/``
# holds ``spec.json``, ``seed_kit.json``, ``sub_agent_journal.md``,
# ``proposal_set.json``, ``critic_verdict.json``,
# ``dispatch_history.jsonl``, and ``telemetry.json``.
# ---------------------------------------------------------------------------
def dynamic_actions_root(session_dir: Path) -> Path:
    """Parent dir of every per-``dyn_id`` artefact dir."""
    return agent_dir(session_dir, "orchestration") / "dynamic_actions"


def dynamic_action_artifact_dir(session_dir: Path, dyn_id: str) -> Path:
    """Per-``dyn_id`` artefact root. Caller mkdir's before writing."""
    did = str(dyn_id or "").strip() or "unknown"
    return dynamic_actions_root(session_dir) / did


def dynamic_action_spec_path(session_dir: Path, dyn_id: str) -> Path:
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "spec.json"


def dynamic_action_seed_kit_path(session_dir: Path, dyn_id: str) -> Path:
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "seed_kit.json"


def dynamic_action_dispatch_history_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "dispatch_history.jsonl"


def dynamic_action_proposal_set_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "proposal_set.json"


def dynamic_action_critic_verdict_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    """Verdict envelope written by the Critic."""
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "critic_verdict.json"


def dynamic_action_telemetry_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    """Per-``dyn_id`` terminal-state rollup written on lifecycle exit."""
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "telemetry.json"


# ---------------------------------------------------------------------------
# External baseline comparison artefacts (DESIGN: target_analysis is report-only)
# ---------------------------------------------------------------------------
# These paths sit under a dedicated top-level subdir rather than
# ``runs/target_analysis/<task_id>/`` because ``target_analysis`` is a
# ``prep``-phase action and ``prep`` is NOT in ``_RUNS_WORKSPACE_PHASES``.
# Putting the artefacts under ``runs/`` would trip ``_validate_action``
# and reduce coupling clarity — the comparison data is intentionally
# decoupled from any per-task data plane.
def target_analysis_dir(session_dir: Path) -> Path:
    """``<sd>/target_analysis/`` — host dir for external baseline artefacts.

    Owner: :class:`inference_optimizer.orchestrator.action_executors.TargetAnalysisExecutor`.
    Reader: :class:`inference_optimizer.orchestrator.action_executors.ReportExecutor`.
    Nothing else under ``inference_optimizer/`` should reach into this dir.
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


# ---------------------------------------------------------------------------
# Cortex KB integration paths
# ---------------------------------------------------------------------------
# Single source of truth for every file under ``<sd>/runtime/cortex/``. The
# directory itself is created by :func:`paths.make_session_dir`; the helpers
# below only compute the well-known file names.  Callers MUST go through
# these helpers (no ad-hoc string concatenation) so the legacy NDJSON
# protocol stays homogeneous across producers / consumers (CortexKBClient,
# flusher daemon, breakdown collector, robustness monitor).
def cortex_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/`` — Cortex KB per-session bookkeeping root."""
    return Path(session_dir) / "runtime" / "cortex"


def cortex_sid_file(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_sid`` — single-line file holding the Cortex
    session id returned by T0 ``session begin``.

    Used by resume to skip re-begin and continue draining
    :func:`cortex_pending` / committing the existing session. Absent file
    means either ``--degraded-kb`` was selected or T0 has not yet run.
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
    queue for T2 / T3 operations.

    Producers: CortexKBClient enqueue on synchronous CLI failure (or for
    always-async ops). Consumer: ``cortex_kb_flusher`` daemon (5s / 50 line
    batch). Drained synchronously at T4 before ``session commit``.
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
    """``<sd>/runtime/cortex/.kb_audit.jsonl`` — append-only synchronous
    audit of every Cortex CLI invocation (success or failure) the
    Coordinator made directly, independent of NDJSON fan-out. Source of
    truth for ``breakdown.kb_provenance``.
    """
    return cortex_dir(session_dir) / ".kb_audit.jsonl"


# ---------------------------------------------------------------------------
# recipe-snapshot v2 — per-session bookkeeping.
#
# Lives under a separate ``runtime/recipe_snapshot/`` subtree (NOT
# ``runtime/cortex/``) so the v2 dispatcher can stay decoupled from
# the legacy ``/v1/points`` client during the gradual cutover.
#
# History: under the original Phase 1 design this directory also held
# ``.pending.ndjson`` / ``.flushed.ndjson`` / ``.dead_letter.ndjson``
# queues for failed central-server writes. Those have been retired —
# under the local-write design (commit "feat(recipe_kb): local-only
# recipe-snapshot store with history archival") writes never go to
# the central server, so the failed-write fan-out has nothing to
# queue. Only the read-side audit log (``.audit.jsonl``) and the
# directory itself survive; both are kept for the dispatcher's
# remote-failure logging path.
# ---------------------------------------------------------------------------
def recipe_snapshot_dir(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_snapshot/`` — dispatcher / remote-client
    per-session bookkeeping root.
    """
    return Path(session_dir) / "runtime" / "recipe_snapshot"


def recipe_snapshot_audit_jsonl(session_dir: Path) -> Path:
    """``<sd>/runtime/recipe_snapshot/.audit.jsonl`` — append-only
    synchronous audit of every recipe-snapshot remote READ call
    (success or failure) the dispatcher made directly. Writes are
    local-only and don't traverse this audit (the local store has
    its own atomic write contract).
    """
    return recipe_snapshot_dir(session_dir) / ".audit.jsonl"


def pr_monitor_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.pr_monitor_status.json`` — one-shot marker
    written by ``cli._bootstrap_knowledge_plane`` with
    the boot-time PR Monitor reachability snapshot. Breakdown collector
    reads it to emit ``warnings`` entries like ``pr_monitor:disabled``
    or ``pr_monitor:unreachable`` so dashboards can light up on
    --degraded-pr / cross-cluster failures without scraping logs.

    Schema (JSON):

    ``{enabled: bool, url: str | None, reachable: bool, mcp_url: str,
       window_days: int, status_text: str}``
    """
    return cortex_dir(session_dir) / ".pr_monitor_status.json"


def cortex_flusher_pid(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_flusher.pid`` — flusher daemon pid file
    (one line). Robustness reads this to detect a dead flusher.
    """
    return cortex_dir(session_dir) / ".kb_flusher.pid"


def cortex_flusher_status_json(session_dir: Path) -> Path:
    """``<sd>/runtime/cortex/.kb_flusher_status.json`` — one-shot marker
    written by ``cli._maybe_spawn_kb_flusher`` with the
    boot-time flusher spawn decision. Breakdown collector merges this
    with the live pid-file check to populate
    ``kb_provenance.flusher_status``.

    Schema (JSON):

    ``{enabled: bool, spawned: bool, pid: int | None, cmd: list[str],
       cortex_kb_url: str | None, interval_sec: float, batch_size: int,
       reason: str, ts: str}``
    """
    return cortex_dir(session_dir) / ".kb_flusher_status.json"


__all__ = [
    "agent_dir",
    "agent_inbox",
    "agent_log",
    "agent_outbox",
    "agent_persona",
    "agent_prompt_snapshot",
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
    "dynamic_action_artifact_dir",
    "dynamic_action_critic_verdict_path",
    "dynamic_action_dispatch_history_path",
    "dynamic_action_proposal_set_path",
    "dynamic_action_seed_kit_path",
    "dynamic_action_spec_path",
    "dynamic_actions_root",
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
