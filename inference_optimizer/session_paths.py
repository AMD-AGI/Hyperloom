"""Per-session path helpers ().

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


# ---------------------------------------------------------------------------
# Per-task workspaces under runs/<action>/<task_id>/
# ---------------------------------------------------------------------------
# Single source of truth for "which actions own a runs/<kind>/<task_id>/
# workspace": derived from the ActionRegistry via the ``pipeline_phase``
# yaml field. Phases listed here are exactly the ones whose executors
# write per-task artefacts under ``runs/``.
#
# History: this used to be a hand-maintained ``_RUNS_ACTIONS`` frozenset;
# adding a new action (e.g. ``validate_stack``) required updating four
# independent locations (yaml, cli register_executor, prompt builder
# enabled-set, and this whitelist). Forgetting the last one made the
# orchestrator loop forever proposing the action — every dispatch raised
# ``ValueError: runs_dir: unknown action ...`` from inside the executor,
# but mission TODOs never cleared. Driving the set from ``pipeline_phase``
# keeps the four sources aligned automatically.
_RUNS_WORKSPACE_PHASES: frozenset[str] = frozenset({
    "measure", "analysis", "explore", "deep", "validate", "support",
})

# Hardcoded fallback used only when ActionRegistry can't be loaded
# (broken yaml / partial install / very early bootstrap). MUST stay in
# sync with the union of action names whose ``pipeline_phase`` is in
# ``_RUNS_WORKSPACE_PHASES``; the regression test in
# ``tests/test_p1_2_full_action_catalogue.py`` enforces alignment.
# ``specialist`` is yaml-less (v0.8 M5, parameterised by
# ``params.domain``) so it is added explicitly.
# ``support`` was added in 2026-05 alongside the real ``recover``
# executor (Change C of the gpu-leak-robustness-fix plan); ``recover``
# is the only fallback ``support`` entry.
_RUNS_ACTIONS_FALLBACK: frozenset[str] = frozenset({
    "baseline",
    # GAP 1 — Coordinator-internal warm-recipe replay. Same workspace
    # shape as ``baseline`` (under ``runs/replay_warm_recipe/<task_id>/``);
    # included so the registry-loader-failure path still pre-mkdirs the
    # workspace for the replay task that the PRELUDE hook will enqueue.
    "replay_warm_recipe",
    # Coordinator-internal analysis actions. Which one runs is chosen
    # by ``shared_state.enable_roofline`` (``--enable-roofline`` /
    # ``--no-enable-roofline``, default on): ``roofline`` is the
    # composite action (profile + trace_analyze + analysis.md
    # snapshot); ``profile`` is the lighter trace-only fallback. Both
    # land under ``runs/<kind>/<task_id>/`` so both names need a
    # fallback entry for the loader-failure path. LLM proposals of
    # either name are denied by PolicyGate
    # (``analysis_action_not_llm_proposable``).
    "roofline", "profile",
    "sweep",
    "explore",
    "specialist",
    # PR-A1 (Arbor-into-Hyperloom): ``integrate_patch`` is an
    # EXPLORE-phase deterministic Python executor that consumes
    # ``runs/specialist/<task_id>/worktree/`` patches; pipeline_phase
    # ``explore`` already includes it in the registry-derived set, the
    # fallback only matters when the yaml can't be loaded.
    "integrate_patch",
    # FRAMEWORK_PR phase: per-candidate Coordinator-internal executor
    # mirroring integrate_patch (applies an upstream PR + benches +
    # KEEP/REVERT). pipeline_phase=explore puts it in the registry-derived
    # runs/<kind>/ set; the fallback only matters on registry load failure.
    "framework_pr",
    # IR-7 (Saturday May 2026): ``assess_remaining_gaps`` is a thin
    # wrapper that dispatches the ``session_steward_specialist``
    # domain. Pipeline_phase ``explore`` includes it in the
    # registry-derived set; fallback covers the rare bootstrap miss.
    "assess_remaining_gaps",
    # Cross-domain ReAct sub-agent; owns
    # ``agents/orchestration/dynamic_actions/<dyn_id>/``.
    "dynamic_action",
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

    Returns:
        frozenset[str]: The action names whose ``pipeline_phase`` is in
            ``_RUNS_WORKSPACE_PHASES`` (plus the synthetic ``specialist``),
            or ``_RUNS_ACTIONS_FALLBACK`` when the registry cannot load.
    """
    try:
        from .orchestrator.action_registry import ActionRegistry  # local: avoid import-time cycle
        registry = ActionRegistry().load()
    except Exception:
        return _RUNS_ACTIONS_FALLBACK
    # ``specialist`` is a synthetic action
    # with no yaml meta (parameterised by ``params.domain``); the
    # registry-derived path can't see it, so we always add it explicitly.
    return frozenset(
        a.name for a in registry.all()
        if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
    ) | frozenset({"specialist"})


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

    Args:
        session_dir (Path): The session root directory.
        kernel_id (str): The kernel identifier; blank/empty falls back to
            ``"unknown"``.

    Returns:
        Path: The absolute path to
            ``<session_dir>/kernel-agent-workspace/<kernel_id>``.
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

    Args:
        session_dir (Path): The session root directory.
        session_id (str): The kernel-agent tool-invocation session id;
            blank/empty falls back to ``"unknown"``.

    Returns:
        Path: The absolute path to
            ``<session_dir>/kernel-agent/runs/<session_id>``.
    """
    sid = str(session_id or "").strip() or "unknown"
    return Path(session_dir) / "kernel-agent" / "runs" / sid


def patches_dir(session_dir: Path, kernel_id: str) -> Path:
    """``<sd>/patches/<kernel_id>/`` — KEEP-promoted real on-disk changes.

    Each kernel that ever passed the integrate gate gets a directory
    with the original source backup and the applied patch; REVERT
    restores from the backup before re-baselining.

    Args:
        session_dir (Path): The session root directory.
        kernel_id (str): The kernel identifier; blank/empty falls back to
            ``"unknown"``.

    Returns:
        Path: The absolute path to ``<session_dir>/patches/<kernel_id>``.
    """
    kid = str(kernel_id or "").strip() or "unknown"
    return Path(session_dir) / "patches" / kid


# ---------------------------------------------------------------------------
# Reports / logs
# ---------------------------------------------------------------------------
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

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/optimizer_runs``.
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


# ---------------------------------------------------------------------------
# Per-agent inbox/outbox + system prompt snapshot
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# dynamic_action artefact paths.
# Layout: ``<session_dir>/agents/orchestration/dynamic_actions/<dyn_id>/``
# holds ``spec.json``, ``seed_kit.json``, ``sub_agent_journal.md``,
# ``proposal_set.json``, ``critic_verdict.json``,
# ``dispatch_history.jsonl``, and ``telemetry.json``.
# ---------------------------------------------------------------------------
def dynamic_actions_root(session_dir: Path) -> Path:
    """Compute the parent dir of every per-``dyn_id`` artefact dir.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/agents/orchestration/dynamic_actions``.
    """
    return agent_dir(session_dir, "orchestration") / "dynamic_actions"


def dynamic_action_artifact_dir(session_dir: Path, dyn_id: str) -> Path:
    """Compute the per-``dyn_id`` artefact root. Caller mkdir's before writing.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier; blank/empty falls back
            to ``"unknown"``.

    Returns:
        Path: The absolute path to the ``<dyn_id>`` artefact directory under
            :func:`dynamic_actions_root`.
    """
    did = str(dyn_id or "").strip() or "unknown"
    return dynamic_actions_root(session_dir) / did


def dynamic_action_spec_path(session_dir: Path, dyn_id: str) -> Path:
    """Compute the path to a dynamic action's ``spec.json``.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier.

    Returns:
        Path: The absolute path to ``<dyn_id artefact dir>/spec.json``.
    """
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "spec.json"


def dynamic_action_seed_kit_path(session_dir: Path, dyn_id: str) -> Path:
    """Compute the path to a dynamic action's ``seed_kit.json``.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier.

    Returns:
        Path: The absolute path to ``<dyn_id artefact dir>/seed_kit.json``.
    """
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "seed_kit.json"


def dynamic_action_dispatch_history_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    """Compute the path to a dynamic action's ``dispatch_history.jsonl``.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier.

    Returns:
        Path: The absolute path to
            ``<dyn_id artefact dir>/dispatch_history.jsonl``.
    """
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "dispatch_history.jsonl"


def dynamic_action_proposal_set_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    """Compute the path to a dynamic action's ``proposal_set.json``.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier.

    Returns:
        Path: The absolute path to ``<dyn_id artefact dir>/proposal_set.json``.
    """
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "proposal_set.json"


def dynamic_action_critic_verdict_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    """Compute the path to the Critic-written verdict envelope ``critic_verdict.json``.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier.

    Returns:
        Path: The absolute path to ``<dyn_id artefact dir>/critic_verdict.json``.
    """
    return dynamic_action_artifact_dir(session_dir, dyn_id) / "critic_verdict.json"


def dynamic_action_telemetry_path(
    session_dir: Path, dyn_id: str,
) -> Path:
    """Compute the path to a dynamic action's terminal-state telemetry rollup.

    The rollup (``telemetry.json``) is written on lifecycle exit.

    Args:
        session_dir (Path): The session root directory.
        dyn_id (str): The dynamic-action identifier.

    Returns:
        Path: The absolute path to ``<dyn_id artefact dir>/telemetry.json``.
    """
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

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/target_analysis``.
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
    """Compute ``<sd>/runtime/cortex/``, the Cortex KB per-session bookkeeping root.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/cortex``.
    """
    return Path(session_dir) / "runtime" / "cortex"


def cortex_sid_file(session_dir: Path) -> Path:
    """Compute the path to ``.kb_sid``, the single-line Cortex session id file.

    Holds the Cortex session id returned by T0 ``session begin``. Used by
    resume to skip re-begin and continue draining :func:`cortex_pending` /
    committing the existing session. Absent file means either
    ``--degraded-kb`` was selected or T0 has not yet run.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to ``<session_dir>/runtime/cortex/.kb_sid``.
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
    """Compute the path to ``.kb_pending.ndjson``, the async write queue.

    Append-only queue for T2 / T3 operations. Producers: CortexKBClient
    enqueue on synchronous CLI failure (or for always-async ops). Consumer:
    ``cortex_kb_flusher`` daemon (5s / 50 line batch). Drained synchronously
    at T4 before ``session commit``.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
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
    """Compute the path to ``.kb_audit.jsonl``, the synchronous CLI audit log.

    Append-only audit of every Cortex CLI invocation (success or failure)
    the Coordinator made directly, independent of NDJSON fan-out. Source of
    truth for ``breakdown.kb_provenance``.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_audit.jsonl``.
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
    """Compute ``<sd>/runtime/recipe_snapshot/``, the dispatcher bookkeeping root.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_snapshot``.
    """
    return Path(session_dir) / "runtime" / "recipe_snapshot"


def recipe_snapshot_audit_jsonl(session_dir: Path) -> Path:
    """Compute the path to the recipe-snapshot remote-READ audit log.

    Append-only synchronous audit (``.audit.jsonl``) of every
    recipe-snapshot remote READ call (success or failure) the dispatcher
    made directly. Writes are local-only and don't traverse this audit (the
    local store has its own atomic write contract).

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/recipe_snapshot/.audit.jsonl``.
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

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
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
    """``<sd>/runtime/cortex/.kb_flusher_status.json`` — one-shot marker
    written by ``cli._maybe_spawn_kb_flusher`` with the
    boot-time flusher spawn decision. Breakdown collector merges this
    with the live pid-file check to populate
    ``kb_provenance.flusher_status``.

    Schema (JSON):

    ``{enabled: bool, spawned: bool, pid: int | None, cmd: list[str],
       cortex_kb_url: str | None, interval_sec: float, batch_size: int,
       reason: str, ts: str}``

    Args:
        session_dir (Path): The session root directory.

    Returns:
        Path: The absolute path to
            ``<session_dir>/runtime/cortex/.kb_flusher_status.json``.
    """
    return cortex_dir(session_dir) / ".kb_flusher_status.json"


__all__ = [
    "agent_dir",
    "agent_inbox",
    "agent_log",
    "agent_outbox",
    "agent_persona",
    "agent_prompt_snapshot",
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
    "dynamic_action_artifact_dir",
    "dynamic_action_critic_verdict_path",
    "dynamic_action_dispatch_history_path",
    "dynamic_action_proposal_set_path",
    "dynamic_action_seed_kit_path",
    "dynamic_action_spec_path",
    "dynamic_actions_root",
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
    "runs_dir",
    "runs_root",
    "state_path",
    "target_analysis_dir",
    "target_analysis_report_md",
    "target_baseline_json",
]
