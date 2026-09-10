# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``report`` ActionRunner."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common import io as _common_io
from hyperloom.common.platform_probe import platform_fingerprint

from ...bus.message_bus import MessageBus
from ...bus.storage.connection import SqliteConnection
from ...phases.machine_state import AGENTX_PREFLIGHT_STOP_REASON
from ...state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import db_path_for


log = logging.getLogger(__name__)


def _count_server_boot_failures(session_dir: Path | None) -> int:
    """Count ``warmup_failed`` variants (server boot failures) from the journal."""
    if session_dir is None:
        return 0
    path = Path(session_dir) / "reports" / "optimization_journal.json"
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    entries = blob.get("entries") if isinstance(blob, dict) else None
    if not isinstance(entries, list):
        return 0
    return sum(1 for e in entries if isinstance(e, dict) and str(e.get("reason") or "").strip() == "warmup_failed")


def _safe_call(state: Any, method: str, default: Any) -> Any:
    """Call a zero-arg SharedState helper, returning ``default`` when absent or raising."""
    fn = getattr(state, method, None)
    if not callable(fn):
        return default
    try:
        return fn()
    except Exception:  # noqa: BLE001 — report must never crash on annotations
        return default


# Benign upstream WARN fragments that must never be promoted as the ``baseline_failed`` headline; the full text still
# appears in the per-attempt logs.
_BENIGN_FAILURE_PATTERNS: tuple[str, ...] = ("modeling_cohere2.py",)


def _is_benign_failure_text(text: str) -> bool:
    """Return True when ``text`` matches a known-benign upstream WARN pattern."""
    blob = str(text or "")
    return any(pat in blob for pat in _BENIGN_FAILURE_PATTERNS)


def _highlight_is_benign(highlight: dict[str, Any]) -> bool:
    """Return True when a highlight's *headline* is only a benign upstream WARN."""
    return _is_benign_failure_text(str(highlight.get("summary", "")))


def _partition_benign_lines(text: str) -> tuple[list[str], list[str]]:
    """Split an error blob into ``(kept_lines, suppressed_benign_lines)``."""
    kept: list[str] = []
    suppressed: list[str] = []
    for line in str(text or "").splitlines():
        if _is_benign_failure_text(line):
            stripped = line.strip()
            if stripped:
                suppressed.append(stripped[:200])
        else:
            kept.append(line)
    return kept, suppressed


def _classify_root_cause_type(error_class: str, error_text: str) -> str:
    """Map a baseline attempt's ``error_class`` + message to a coarse enum."""
    from .baseline import _KV_CACHE_OOM_MARKERS

    blob = f"{error_class} {error_text}".lower()
    if error_class == "kv_cache_oom" or any(m in blob for m in _KV_CACHE_OOM_MARKERS):
        return "kv_cache_oom"
    if "out of memory" in blob or "hip oom" in blob:
        return "oom"
    if error_class == "timeout" or "benchmark exceeded" in blob:
        return "benchmark_timeout"
    if (
        error_class == "server_init_dead"
        or "engine core" in blob
        or "enginecore" in blob
        or "workerproc" in blob
        or "engine process failed" in blob
    ):
        return "engine_core_init"
    if error_class == "subprocess_nonzero" or "nccl" in blob or "worker" in blob:
        return "worker_crash"
    return "unknown"


def _pick_failure_headline(text: str) -> str:
    """Pick the most informative single line out of a server.log excerpt."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    priority = (
        "out of memory",
        "fatal",
        "runtimeerror",
        "nccl",
        "engine core",
        "enginecore",
        "workerproc",
        "error:",
    )
    for keyword in priority:
        for ln in lines:
            if keyword in ln.lower():
                return ln
    return lines[-1]


def _last_failed_baseline_attempt(state: SharedState) -> dict[str, Any] | None:
    """Return the most recent *failed* baseline attempt record, or ``None``."""
    attempts = getattr(state, "baseline_attempts", None) or []
    failed = [a for a in attempts if isinstance(a, dict) and str(a.get("status")) == "failed"]
    if failed:
        return failed[-1]
    failures = getattr(state, "last_action_failures", None) or []
    baseline_failures = [f for f in failures if isinstance(f, dict) and str(f.get("action")) == "baseline"]
    return baseline_failures[-1] if baseline_failures else None


def _resolve_attempt_server_log(attempt: dict[str, Any]) -> Path | None:
    """Best-effort path to a baseline attempt's ``server.log``."""
    candidates: list[Path] = []
    workspace = attempt.get("workspace")
    if workspace:
        ws = Path(str(workspace))
        candidates.append(ws.parent / "server.log")
        candidates.append(ws / "server.log")
    stderr_log = attempt.get("stderr_log_path")
    if stderr_log:
        candidates.append(Path(str(stderr_log)))
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _build_failure_summary(
    state: SharedState,
    session_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Surface the real terminal error on ``baseline_failed``."""
    if str(getattr(state, "stop_reason", "") or "") != "baseline_failed":
        return None
    try:
        attempt = _last_failed_baseline_attempt(state)
        if attempt is None:
            return None

        error_class = str(attempt.get("error_class") or "unknown")
        raw_error = str(attempt.get("error_excerpt") or "")
        kept_lines, suppressed = _partition_benign_lines(raw_error)
        error_text = "\n".join(kept_lines).strip()

        server_log_abs = _resolve_attempt_server_log(attempt)
        # Only the benign WARN (or nothing) survived: dig the real terminal marker out of server.log when available.
        if not error_text and server_log_abs is not None:
            try:
                from ._subprocess_kill import server_log_death_excerpt

                excerpt = server_log_death_excerpt(str(server_log_abs))
            except Exception:  # noqa: BLE001 — excerpt enrichment is best-effort
                log.debug("server_log_death_excerpt failed", exc_info=True)
                excerpt = None
            if excerpt:
                error_text = excerpt.strip()
                if error_class in ("", "unknown"):
                    error_class = "server_init_dead"
        if not error_text:
            error_text = "(no terminal error captured; see logs)"

        root_cause = _pick_failure_headline(error_text)
        server_log_rel: str | None = None
        if server_log_abs is not None:
            if session_dir is not None:
                try:
                    server_log_rel = server_log_abs.relative_to(session_dir).as_posix()
                except ValueError:
                    server_log_rel = str(server_log_abs)
            else:
                server_log_rel = str(server_log_abs)

        summary: dict[str, Any] = {
            "root_cause": root_cause[:500],
            "root_cause_type": _classify_root_cause_type(error_class, error_text),
            "error_class": error_class,
            "last_attempt_id": str(attempt.get("task_id") or ""),
            "server_log": server_log_rel,
        }
        if suppressed:
            summary["suppressed_benign"] = suppressed[:5]
        return summary
    except Exception:  # noqa: BLE001 — report must never crash on the summary
        log.warning(
            "report_executor: failed to build failure_summary",
            exc_info=True,
        )
        return None


_STOP_REASON_EXPLANATIONS: dict[str, str] = {
    # Terminal reasons the coordinator loop sets directly.
    "target_reached": "Target reached: a requested --target-gain / --target-tput / --target-roofline was met.",
    "time_exhausted": "Wall-clock budget (--max-hours) was exhausted; the best validated result was kept.",
    "max_ticks": "The coordinator hit its max-ticks safety cap; the best validated result was kept.",
    "signal": "Stopped on an OS stop signal (e.g. SIGINT / SIGTERM).",
    "emergency": (
        "Emergency stop: recoverable crashes spiked past the safety threshold inside the crash "
        "window; the best validated result was preserved before exit."
    ),
    "coordinator_exception": "The coordinator loop raised an unhandled exception; the last validated result was preserved.",
    "custom": "A caller-supplied stop condition (stop_when) fired.",
    "unknown": "No specific stop reason was recorded (e.g. a terminal session was resumed); treat as unclassified.",
    # Policy / robustness governor.
    "policy_loop": "The policy gate detected a decision loop and stopped to avoid spinning on the same transition.",
    "crash_threshold_exceeded": "Too many recoverable crashes accumulated; the run stopped to preserve the validated result.",
    "robustness_escalated": (
        "Robustness escalated: the run closed early with budget still on the clock (not a target hit). "
        "Common triggers are a validated-gain plateau, rising crash_count, or a stale aiter JIT build. "
        "A close driven by the remaining budget reports time_exhausted instead. "
        "The best validated result was locked in before exit."
    ),
    "user_stop_requested": "Stopped on an explicit operator request.",
    "baseline_failed": "Baseline never produced a valid measurement; see the failure summary / server log.",
    "server_argv_invalid": (
        "The installed framework's own argument parser refused the composed server argv before any "
        "server was launched. Nothing in the framework source is at fault: correct the offending flag "
        "in --server-args / the recipe for the installed version, then re-run."
    ),
    "environment_fault": (
        "This host cannot run the combination as installed - the framework is missing from the serving "
        "interpreter, a compiled extension has no build for this platform, the checkpoint path resolves "
        "to nothing, or the port is already held. No patch to the model or the framework source repairs "
        "any of those, so the run stopped instead of spending rounds authoring against the machine."
    ),
    # PRELUDE-phase early exits (before optimization begins).
    "prelude_baseline_failed": "PRELUDE baseline failed before optimization could start; see the baseline failure summary.",
    "prelude_policy_loop": "The policy gate detected a decision loop during PRELUDE and stopped.",
    "time_exhausted_during_prelude": (
        "The session's wall-clock budget ran out during preparation, before optimization began. Whatever PRELUDE "
        "was doing when the clock reached zero — measuring the baseline, bringing up the framework agent, taking "
        "the roofline — is where the time went; the phase record shows which arms ran and what each cost."
    ),
    "prelude_cold_anchor_low_budget": (
        "The baseline's hot pass was skipped because the clock could not cover it together with one variant to "
        "measure against it, so the only figure available is the cold pass's — depressed by the server boot, the "
        "first request's kernel compile and the graph capture. Optimizing against it would report every variant as "
        "an improvement over a baseline that was never the baseline, so the run stopped with the figure kept and "
        "marked. Resume with more budget to measure a comparable baseline."
    ),
    # Recipe KB knowledge-plane bootstrap failures.
    "recipe_kb_t0_failed": "Recipe KB knowledge-plane bootstrap (t0) failed; the run stopped early.",
    "recipe_kb_drain_failed": "Recipe KB knowledge-plane drain failed; the run stopped early.",
    "recipe_kb_commit_failed": "Recipe KB knowledge-plane commit failed; the run stopped early.",
    "warm_replay_rollback_failed": (
        "Warm replay rollback could not restore every Recipe/Kernel mutation; "
        "the run stopped to avoid continuing from an uncertain code state."
    ),
    "active_inferencex_checkout_missing": (
        "The session-authoritative InferenceX checkout is missing during "
        "resume; the run stopped instead of falling back to a different tree."
    ),
    # Search / phase plateaus and completions.
    "plateau_kernel": "KERNEL_AGENT plateaued: no further validated kernel win was found.",
    "no_kernel_skipped": "No kernel candidates were available, so the kernel phase was skipped and the run closed.",
    "sweep_done": "SWEEP finished the concurrency ladder.",
    "sweep_failed": "The concurrency sweep reached a failed terminal result.",
    "optimize_no_more_leverage": (
        "OPTIMIZE exhausted both levers: neither configuration search nor source/upstream landing had leverage left."
    ),
    "optimize_phase_budget_exhausted": "OPTIMIZE spent its phase budget.",
    "optimize_budget_cap": "OPTIMIZE reached the absolute per-phase wall-clock cap.",
    # Retired reason names, kept so a report over an archived session still explains what it is reading.
    "plateau_explore": "The configuration search plateaued: no new leverage was found in the search space.",
    "framework_agent_phase_done": "The framework-enablement agent completed its phase.",
    "framework_agent_plateau": "The framework-enablement agent plateaued with no further progress.",
    "global_converged": "Cyclic phases converged: repeated macro-cycles stopped yielding new validated gain.",
    # Pre-flight gates (fail fast before booting a server).
    "model_context_window_too_small": "Preflight gate: the model's max context window cannot hold the requested ISL + OSL.",
    "unsupported_model_arch": "Preflight gate: the model architecture (e.g. multimodal / vision) is unsupported.",
    "model_config_incompatible": (
        "Preflight gate: config.json is corrupt or declares RoPE scaling without a max-position field, "
        "which would crash engine init."
    ),
    "baseline_arg_error": "Two or more baseline attempts fast-exited on a bad CLI arg (deterministic), so the slow-baseline retry budget was not burned.",
    "enablement_stalled": "The enablement loop stopped without a baseline that boots: a revalidation the round depended on never promoted.",
    "enablement_attempts_exhausted": "The enablement loop stopped after too many consecutive rounds bought no ground. A bring-up that is still clearing new boot failures is bounded by the run's wall clock instead.",
    "baseline_accuracy_failed": "The baseline produced no accuracy result even though the accuracy test was expected to run (broken eval or missing quality gate). The run stopped rather than optimize against an unvalidated baseline.",
    AGENTX_PREFLIGHT_STOP_REASON: (
        "HYPERLOOM_AGENTX is on but its benchmark client (aiperf) is missing or is not the pinned "
        "AIPERF_REF build, and the automatic install did not resolve it. This is an environment gap, "
        "not something a framework patch can close, so the run stopped on the first occurrence rather "
        "than spending its budget in the enablement lane. Fix: run "
        "src/hyperloom/inference_optimizer/assets/install.sh --only-aiperf (the failure it prints is "
        "the real cause), or point AIPERF_BIN at an existing pinned build."
    ),
    # Host-level terminals: something outside the model ended the run.
    "supervisor_coordinator_died": "The out-of-band supervisor found the coordinator's process gone; this record was written by the supervisor because there was no coordinator left to write one.",
    "supervisor_tick_stalled": "The out-of-band supervisor found the coordinator's tick not advancing inside its stall window and asked the session to end.",
}


def _explain_stop_reason(stop_reason, state=None):
    """Return a human-readable explanation for a terminal ``stop_reason``."""
    reason = str(stop_reason or "").strip()
    text = _STOP_REASON_EXPLANATIONS.get(reason, "")
    if reason == "sweep_done" and text:
        return _explain_conc_sweep_skip(state) or text
    return text


def _explain_conc_sweep_skip(state) -> str:
    """Name a skipped concurrency sweep, or ``\"\"`` when one ran to a result."""
    last = getattr(state, "last_conc_sweep", None)
    if not isinstance(last, dict) or not last.get("was_skipped"):
        return ""
    # Imported here, not at module scope: ``kernel.conc_sweep`` imports the grid runner in this same package, so a
    # top-level import is the edge CodeQL reports as a cycle.
    from ...kernel.conc_sweep import conc_sweep_declined_to_run  # noqa: PLC0415

    detail = str(last.get("skip_reason") or "").strip() or "no reason recorded"
    if conc_sweep_declined_to_run(last):
        return f"Post-sweep concurrency sweep did not run ({detail}); the phase settled and the run closed."
    return (
        f"Post-sweep concurrency sweep exhausted its budget without a comparable "
        f"baseline/optimized pair ({detail}); the phase settled and the run closed."
    )


def _platform_fingerprint(gpu_type: str | None = None) -> dict[str, Any]:
    """Platform record for the run report, scoped to this session's node count."""
    try:  # local import keeps this free of executor import order
        from ._multi_node_env import is_multi_node

        multi_node: bool | None = bool(is_multi_node())
    except Exception:  # noqa: BLE001 - scope marker only; unknown is not False
        multi_node = None
    return platform_fingerprint(gpu_type, multi_node=multi_node)


def _append_composite_perf_section(lines: list[str], summary: dict[str, Any]) -> None:
    """Render recorded grading; summaries predating the snapshot keep their legacy layout."""
    comparison = summary.get("performance_comparison")
    if comparison is not None:
        from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT, INTVTY_V1

        # The objective is the interactivity axis; total throughput is the guard the verdict also consulted, so it is
        # rendered as a second axis rather than as the figure the session was scored on.
        graded_on_intvty = comparison["objective"] == GRADED_INTVTY
        lines.extend(["## Performance comparison", ""])
        lines.append(f"- objective           : `{comparison['objective']}`")
        lines.append(f"- reference           : `{comparison['reference']:.1f}`")
        lines.append(f"- candidate           : `{comparison['candidate']:.1f}`")
        gain = comparison["gain_pct"]
        gain_text = f"{gain:+.2f}%" if gain is not None else "unavailable"
        lines.append(f"- comparison gain     : `{gain_text}` (diagnostic; not a cumulative validation)")
        lines.append(f"- comparable          : `{str(comparison['comparable']).lower()}`")
        if comparison["degrade_reason"]:
            lines.append(f"- degrade reason      : `{comparison['degrade_reason']}`")
        lines.append(f"- verdict             : `{comparison['verdict']}`")
        lines.append(f"- grading mode        : `{INTVTY_V1 if graded_on_intvty else GRADED_OUTPUT}`")
        if graded_on_intvty:
            lines.append(f"- reference tput      : `{comparison['tput_reference']:.1f}` tok/s (guard axis)")
            lines.append(f"- candidate tput      : `{comparison['tput_candidate']:.1f}` tok/s (guard axis)")
        return

    from hyperloom.common.gain_math import gain_pct
    from hyperloom.common.perf_metric import (
        INTVTY_V1,
        intvty_grading_enabled,
        intvty_of,
        parse_intvty_noise_pct,
        perf_snapshot_from_mapping,
        total_tput_of,
    )

    baseline = perf_snapshot_from_mapping(summary.get("baseline_perf"))
    if not baseline:
        return
    cb = summary.get("current_best") or {}
    cb_snap = perf_snapshot_from_mapping(cb) if isinstance(cb, dict) else None
    lines.append("## AgentX perf (interactivity objective, per-chip tput guard)")
    lines.append("")
    lines.append(f"- baseline intvty P90 : `{intvty_of(baseline):.1f}` tok/s/user (slow tail)")
    lines.append(f"- baseline total tput : `{total_tput_of(baseline):.1f}` tok/s")
    if cb_snap:
        lines.append(f"- current_best intvty : `{intvty_of(cb_snap):.1f}` tok/s/user")
        lines.append(f"- current_best total  : `{total_tput_of(cb_snap):.1f}` tok/s")
        gain = gain_pct(intvty_of(cb_snap), intvty_of(baseline))
        if gain is not None:
            lines.append(f"- intvty gain (graded): `{gain:+.2f}%`")
        tput_gain = gain_pct(total_tput_of(cb_snap), total_tput_of(baseline))
        if tput_gain is not None:
            lines.append(f"- total tput change   : `{tput_gain:+.2f}%` (guard axis, not the objective)")
    if intvty_grading_enabled(benchmark_mode=str(summary.get("benchmark_mode") or "")):
        lines.append(f"- grading mode        : `{INTVTY_V1}` (noise band `{parse_intvty_noise_pct():.1f}%`)")
    else:
        lines.append("- grading mode        : `output_throughput` (AgentX grading not in effect)")


def _cumulative_validation_status(summary: dict[str, Any]) -> str:
    """Classify a stored validation stamp without turning a diagnostic into a validation."""
    if not (
        summary["cumulative_gain_validated_ts"]
        or summary["cumulative_gain_validated_stack_len"]
        or summary["cumulative_gain_validated"]
    ):
        return "unavailable"
    if summary["optimization_stack_len"] != summary["cumulative_gain_validated_stack_len"]:
        return "stale"
    from hyperloom.common.perf_metric import VERDICT_KEEP

    comparison = summary["performance_comparison"]
    if not comparison["comparable"] or comparison["gain_pct"] is None:
        return "unavailable"
    # Anything short of KEEP -- a REVERT, or a RECORDED point the frontier neither promotes nor discards -- disagrees
    # with a stamp claiming the stack's gain was validated.
    if comparison["verdict"] != VERDICT_KEEP or not math.isclose(
        summary["cumulative_gain_validated"], comparison["gain_pct"], abs_tol=1e-9
    ):
        return "inconsistent"
    return "current"


def _build_summary_dict(
    state: SharedState,
    ev_counts: dict[str, int],
    highlights: list[dict],
    *,
    external_baseline: dict[str, Any] | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable session summary dict."""
    from hyperloom.common.gain_math import gain_pct

    from ...state.shared_state import resolve_graded_comparison

    graded = resolve_graded_comparison(state, state.current_best, against_baseline=True)
    # The wind-down report is rendered inside closing_phase, before the loop assigns the terminal stop_reason;
    # closing_phase is only entered on the wall-clock deadline, so fall back to time_exhausted rather than blank.
    stop_reason = str(getattr(state, "stop_reason", "") or "").strip()
    if not stop_reason and getattr(state, "closing_phase", False):
        stop_reason = "time_exhausted"
    summary: dict[str, Any] = {
        "session_id": state.session_id,
        "model_name": state.model_name,
        "model_path": state.model_path,
        "model_class": state.model_class,
        "framework": getattr(state, "framework", "") or "",
        "stop_reason": stop_reason,
        "stop_reason_explanation": _explain_stop_reason(stop_reason, state),
        "baseline_tput": state.baseline_tput,
        "baseline_perf": dict(getattr(state, "baseline_perf", None) or {}),
        # Read back by the graded-axes section: the persisted AgentX marker outlives the shell, so a report rendered
        # from a resumed session still names the mode the run was graded under.
        "benchmark_mode": str(getattr(state, "benchmark_mode", "") or ""),
        "baseline_accuracy": state.baseline_accuracy,
        "current_best": state.current_best,
        "performance_comparison": {
            "objective": graded.objective,
            "reference": graded.reference,
            "candidate": graded.candidate,
            "gain_pct": gain_pct(graded.candidate, graded.reference),
            "comparable": graded.comparable,
            "degrade_reason": graded.degrade_reason,
            "verdict": graded.verdict,
            # The guard axis is snapshotted alongside the objective so a re-rendered report can say what the 2-D
            # verdict weighed, instead of re-deriving it from a ``current_best`` that has since moved on.
            "tput_reference": graded.tput_reference,
            "tput_candidate": graded.tput_candidate,
        },
        # Preserve the historical validation stamp, even when it disagrees with today's diagnostic.
        "cumulative_gain_validated": state.cumulative_gain_validated,
        "cumulative_gain_validated_ts": state.cumulative_gain_validated_ts,
        "cumulative_gain_validated_stack_len": state.cumulative_gain_validated_stack_len,
        "optimization_stack_len": len(state.optimization_stack or []),
        # Honesty annotations: surface unfinished/unvalidated work; read defensively for partial-state stubs.
        "has_unvalidated_keeps": _safe_call(state, "optimization_stack_has_unvalidated_keeps", False),
        "untried_hot_reusable_kernels": list(_safe_call(state, "untried_hot_reusable_kernels", []) or []),
        "pending_keep_kernels": list(_safe_call(state, "pending_keep_kernel_ids", []) or []),
        "crash_count": state.crash_count,
        "server_boot_failures": _count_server_boot_failures(session_dir),
        "pruned_families": state.pruned_families,
        "platform": _platform_fingerprint(getattr(state, "gpu_type", None)),
        "max_minutes": state.max_minutes,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "event_counts_by_topic": ev_counts,
        "highlights": highlights,
        # Degraded-mode advisory: benchmark numbers reflect the text path only.
        "degraded_mode": bool(getattr(state, "degraded_mode", False)),
        "model_warnings": list(getattr(state, "model_warnings", None) or []),
        # The card's partition shape these numbers were measured in.
        "compute_partition": dict(getattr(state, "compute_partition", None) or {}),
    }
    if external_baseline:
        summary["external_baseline"] = external_baseline
    # Roofline comparison: emit only when at least one snapshot exists.
    from ...kernel.roofline_snapshot import build_roofline_comparison_from_history

    cmp = build_roofline_comparison_from_history(getattr(state, "roofline_snapshots", None))
    if cmp:
        summary["roofline_comparison"] = cmp
    # Real terminal root cause on baseline_failed: promote the last failed baseline attempt's engine/worker fault over
    # benign upstream WARNs.
    failure_summary = _build_failure_summary(state, session_dir)
    if failure_summary:
        summary["failure_summary"] = failure_summary
    summary["cumulative_validation_status"] = _cumulative_validation_status(summary)
    return deepcopy(summary)


def _format_md(summary: dict[str, Any]) -> str:
    """Render the human-readable Markdown report from a summary dict."""
    cb = summary.get("current_best") or {}
    cb_tput = cb.get("tput") if isinstance(cb, dict) else None
    lines: list[str] = []
    lines.append(f"# Inference Optimizer Report — {summary['session_id']}")
    lines.append("")
    lines.append(f"- **Model**: {summary['model_name']}  (`{summary['model_path']}`)")
    lines.append(f"- **Stop reason**: `{summary['stop_reason']}`")
    stop_expl = str(summary.get("stop_reason_explanation") or "").strip()
    if stop_expl:
        lines.append(f"- **Why it stopped**: {stop_expl}")
    stop_detail = str(summary.get("stop_detail") or "").strip()
    if stop_detail:
        lines.append(f"- **Stop detail**: {stop_detail}")
    failure_summary = summary.get("failure_summary")
    if isinstance(failure_summary, dict) and failure_summary.get("root_cause"):
        lines.append(
            f"- **Root cause**: "
            f"`{failure_summary.get('root_cause_type', 'unknown')}` — "
            f"{failure_summary.get('root_cause')}"
        )
        if failure_summary.get("server_log"):
            lines.append(f"- **Server log**: `{failure_summary.get('server_log')}`")
    if summary.get("degraded_mode"):
        lines.append(
            "- **⚠ Degraded mode**: ran on the TEXT path only (multimodal inputs ignored) — see 'Degraded mode' below"
        )
    lines.append(f"- **Budget**: {summary['max_minutes']} minutes")
    lines.append(f"- **Generated**: {summary['report_generated_at']}")
    lines.append("")
    # Per-framework primary metric: serving reports throughput (tok/s/GPU), scriptable xDiT reports per-image latency
    # (e2el_mean_ms).
    from hyperloom.inference_optimizer import framework_registry

    _fw = summary.get("framework")
    lines.append("## Throughput")
    lines.append("")
    lines.append(f"- baseline            : `{framework_registry.format_primary_metric(_fw, summary['baseline_tput'])}`")
    if cb_tput is not None:
        lines.append(
            f"- current_best        : `{framework_registry.format_primary_metric(_fw, cb_tput)}` "
            f"(action=`{cb.get('action', '?')}`)"
        )
    # Printed even when never validated, so the absence is stated, not implied.
    val_gain = summary.get("cumulative_gain_validated", 0.0) or 0.0
    val_ts = summary.get("cumulative_gain_validated_ts") or ""
    val_len = summary.get("cumulative_gain_validated_stack_len", 0) or 0
    stack_len = summary.get("optimization_stack_len", 0) or 0
    if val_ts:
        stale = " ⚠ stack changed since validation" if stack_len > val_len else ""
        lines.append(
            f"- cumulative_gain_val : `{val_gain:.2f}%` (validated_at_stack_len={val_len}, ts={val_ts}){stale}"
        )
    elif val_gain or val_len:
        stale = " ⚠ stack changed since validation" if stack_len > val_len else ""
        lines.append(
            f"- cumulative_gain_val : `{val_gain:.2f}%` (validated_at_stack_len={val_len}, ts=<missing>){stale}"
        )
    else:
        lines.append("- cumulative_gain_val : `0.00%` ⚠ never validated — nothing has promoted in this session")
    validation_status = summary.get("cumulative_validation_status")
    if validation_status:
        explanations = {
            "current": "stored gain agrees with the current measured comparison",
            "stale": "stack changed since validation; stored gain does not validate the current stack",
            "unavailable": "no current comparable validation evidence; comparison is diagnostic only",
            "inconsistent": "stored gain disagrees with the current grading evidence; retained as historical only",
        }
        lines.append(f"- validation status   : `{validation_status}` ({explanations[validation_status]})")
    if cb.get("ttft_mean_ms") is not None:
        lines.append(f"- ttft_mean      : `{cb.get('ttft_mean_ms'):.1f}` ms")
    if cb.get("e2el_mean_ms") is not None:
        lines.append(f"- e2el_mean      : `{cb.get('e2el_mean_ms'):.1f}` ms")
    _append_composite_perf_section(lines, summary)
    lines.append("")
    lines.extend(_format_completeness_annotations(summary))
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- crash_count    : {summary['crash_count']}")
    boot_fail = summary.get("server_boot_failures") or 0
    if boot_fail:
        lines.append(f"- server_boot_failures : {boot_fail}  *(warmup_failed variants; excluded from crash_count)*")
    lines.append(f"- pruned_families: {summary['pruned_families'] or '(none)'}")
    plat = summary.get("platform") or {}
    if plat.get("status") != "ok":
        # Always emit the line, with the reason when there is one: a silent absence is indistinguishable from a host
        # that was never checked, and the no-reason case is the one that needs saying -- it is a summary written
        # before this field existed, not a probe that failed.
        why = str(plat.get("reason") or "").strip()
        lines.append(f"- platform       : not recorded{f' — {why}' if why else ''}")
    else:
        if plat.get("multi_node_session"):
            lines.append(
                f"- platform scope : ⚠ multi-node session — sampled on `{plat.get('host')}` "
                f"(the orchestrator), which may not be the benchmark node"
            )
        else:
            lines.append(f"- host           : {plat.get('host', 'unknown')}")

        def _q(value: Any) -> str:
            """Render an unresolved field as '?' rather than the word None."""
            return "?" if value in (None, "") else str(value)

        lines.append(
            f"- platform       : {_q(plat.get('cpu'))} — SMT {_q(plat.get('smt'))}, "
            f"{_q(plat.get('nps'))} ({_q(plat.get('numa_nodes'))} NUMA nodes / "
            f"{_q(plat.get('sockets'))} sockets), governor {_q(plat.get('governor'))}, "
            f"boost {_q(plat.get('boost'))}"
        )
        gpu, stack = plat.get("gpu") or {}, plat.get("stack") or {}
        lines.append(
            f"- accelerators   : {gpu.get('host_count') or '?'}× {gpu.get('gfx_arch', 'unknown')} "
            f"on the host, amdgpu {gpu.get('amdgpu_driver', 'unknown')}"
        )
        lines.append(
            "- stack          : "
            + ", ".join(f"{k} {v}" for k, v in sorted(stack.items()))
            + f", kernel {plat.get('kernel', 'unknown')}"
        )
    lines.append("")
    lines.append("## Event counts")
    lines.append("")
    if not summary.get("event_counts_by_topic"):
        lines.append("- (no events recorded)")
    else:
        for topic, n in sorted(summary["event_counts_by_topic"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{topic}`: {n}")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    if not summary["highlights"]:
        lines.append("(no highlight events captured)")
    else:
        for h in summary["highlights"][:50]:
            lines.append(f"- `{h.get('topic', '?')}` from `{h.get('from_agent', '?')}`: {h.get('summary', '')}")
    lines.append("")

    lines.extend(_format_degraded_mode_section(summary))
    lines.extend(_format_compute_partition_section(summary))

    roofline_cmp = summary.get("roofline_comparison")
    if roofline_cmp:
        lines.extend(_format_roofline_comparison_section(roofline_cmp))

    ext = summary.get("external_baseline")
    if ext:
        lines.extend(_format_external_baseline_section(ext))

    lines.extend(_format_conc_sweep_curve_section(summary))

    return "\n".join(lines)


def _format_degraded_mode_section(summary: dict[str, Any]) -> list[str]:
    """Render the degraded-mode section (multimodal models run on the text path)."""
    warnings = summary.get("model_warnings") or []
    if not summary.get("degraded_mode") and not warnings:
        return []
    lines = ["## Degraded mode", ""]
    lines.append(
        "This run executed on the **text path only**. Multimodal (image/audio) "
        "inputs were ignored, so throughput/accuracy reflect the text decoder "
        "alone. Pass `--no-allow-mm-text-fallback` to fail-fast instead."
    )
    lines.append("")
    for w in warnings:
        if not isinstance(w, dict):
            continue
        name = w.get("model_name") or "?"
        arch = w.get("architecture") or "?"
        signal = w.get("signal") or w.get("kind") or "multimodal signal"
        lines.append(f"- `{name}` (arch `{arch}`): {signal}")
    lines.append("")
    return lines


def _format_compute_partition_section(summary: dict[str, Any]) -> list[str]:
    """State the compute-partition shape these numbers were measured in."""
    shape = summary.get("compute_partition") or {}
    partitions = int(shape.get("partitions") or 0)
    if not shape.get("mode") or partitions <= 1:
        return []
    streams = int(shape.get("streams_per_partition") or 0)
    lines = ["## Compute partitioning", ""]
    lines.append(
        f"This card was split into {partitions} partitions (`{shape['mode']}`), so these "
        f"numbers are not comparable with an unpartitioned run of the same configuration."
    )
    lines.append("")
    lines.append(f"- mode              : `{shape['mode']}` ({partitions} partitions)")
    if shape.get("cu_per_partition"):
        # Absent is its own answer.
        probed = shape.get("cu_probed")
        origin = "" if probed is None else (" (from the device)" if probed else " (derived from the board table)")
        lines.append(f"- CU per partition  : {shape['cu_per_partition']}{origin}")
    if shape.get("gib_per_partition"):
        lines.append(f"- HBM per partition : `{float(shape['gib_per_partition']):.1f}` GiB")
    # Omitted where nothing fans out: the number would describe a placement that never happened, directly above a
    # paragraph saying it did not.
    if streams and shape.get("fanout_expected") is not False:
        lines.append(f"- streams/partition : {streams} ({streams * partitions} concurrent streams total)")
    lines.append("")
    if shape.get("fanout_expected") is False:
        lines.append(
            f"**This session's benchmark does not place work on individual partitions.** The "
            f"throughput is therefore one device's, not the total across all {partitions}, and "
            f"which device it was is not recorded here -- whole cards enumerate before "
            f"partitions, so it may be a whole card or a single partition."
        )
    else:
        lines.append(
            f"Whether the throughput is one partition's or the total across all {partitions} "
            f"depends on the benchmark placing work on each of them. The shape above is read "
            f"from the card, but the fan-out is not this process's to do, and it cannot be "
            f"verified from here -- so do not read the figure as an aggregate unless the "
            f"benchmark entrypoint is known to fan out."
        )
    lines.append("")
    lines.append(
        "Partitioning only ever gives a single stream fewer CUs, so per-request latency "
        "is worse here than on the whole card by construction."
    )
    lines.append("")
    return lines


def _format_completeness_annotations(summary: dict[str, Any]) -> list[str]:
    """Render honesty annotations for work left unfinished (unvalidated KEEPs, untried hot kernels, KEEPs awaiting integrate)."""
    unvalidated = bool(summary.get("has_unvalidated_keeps"))
    untried = list(summary.get("untried_hot_reusable_kernels") or [])
    pending_keeps = list(summary.get("pending_keep_kernels") or [])
    if not (unvalidated or untried or pending_keeps):
        return []
    lines: list[str] = ["## Completeness annotations", ""]
    if unvalidated:
        lines.append(
            "- ⚠ `optimization_stack` has KEEPs landed since the last "
            "validated measurement — `cumulative_gain_validated` does not "
            "yet reflect them (unvalidated)."
        )
    if pending_keeps:
        lines.append(f"- ⚠ kernel_opt KEEPs awaiting integrate: {', '.join(pending_keeps)}.")
    if untried:
        lines.append(f"- ⚠ reusable hot kernels with no kernel_opt attempt: {', '.join(untried)}.")
    lines.append("")
    return lines


def _extract_executive_summary(analysis_md_path: str) -> str:
    """Extract the ``## Executive Summary`` block (up to the next level-2 heading) from analysis.md."""
    if not analysis_md_path:
        return "(no analysis.md path recorded)"
    try:
        text = Path(analysis_md_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read {analysis_md_path}: {exc})"
    # Strip base64 image data URLs so the report stays compact.
    import re

    text = re.sub(
        r"!\[[^\]]*\]\(data:image/[^)]+\)",
        "[image stripped]",
        text,
    )
    lines = text.splitlines()
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start is None and stripped.startswith("## Executive Summary"):
            start = i
            continue
        if start is not None and stripped.startswith("## ") and i > start:
            end = i
            break
    if start is None:
        return "(analysis.md does not contain a `## Executive Summary` block)"
    block = "\n".join(lines[start:end]).strip()
    # Cap at ~2KB.
    if len(block) > 2048:
        block = block[:2045] + "..."
    return block


def _format_roofline_comparison_section(cmp: dict[str, Any]) -> list[str]:
    """Render the ``## Roofline Comparison`` section from ``cmp`` (built by :func:`roofline_snapshot.build_roofline_comparison_from_history`)."""
    from ...kernel.roofline_snapshot import format_roofline_metrics_table

    lines: list[str] = ["## Roofline Comparison", ""]
    baseline = cmp.get("baseline") or {}
    latest = cmp.get("latest") or {}
    base_id = baseline.get("snapshot_id")
    latest_id = latest.get("snapshot_id")
    mode = cmp.get("mode") or ("single_snapshot" if (base_id is not None and base_id == latest_id) else "before_after")

    if not baseline.get("analysis_md_path"):
        lines.append(
            "_No roofline snapshot was captured during this session — "
            "the `roofline` composite action never completed successfully._"
        )
        lines.append("")
        return lines

    if mode == "single_snapshot":
        lines.append(
            f"_Only one roofline snapshot was captured this session "
            f"(snapshot #{base_id}). PR #321 retired the legacy "
            "close-phase auto-roofline; refreshes are now driven by a "
            "10% gain watermark over `last_roofline_tput` (see "
            "`Coordinator._maybe_enqueue_watermark_roofline`). The "
            "watermark did not cross during this session, so the "
            "PRELUDE bootstrap snapshot is the only datapoint available "
            "for the report._"
        )
        lines.append("")
        lines.append(
            "_The **Theoretical peak** below is the decode "
            "memory-roofline ceiling derived from the GPU's HBM "
            "bandwidth and the model's weight + KV-cache traffic per "
            "token (see `roofline_ceiling.compute_peak_from_state`). "
            "It is an upper bound: real `output_throughput` always "
            "stays under it because of comm overhead, kernel "
            "efficiency < 100%, and KV-cache fragmentation. **Within "
            "roofline %** = measured / peak; **Gap to roofline %** = "
            "100 − Within._"
        )
        lines.append("")
        lines.extend(format_roofline_metrics_table(cmp))
        lines.append(f"### Snapshot #{base_id} — Executive Summary")
        lines.append("")
        lines.append(f"`{baseline.get('analysis_md_path')}`")
        if baseline.get("ts"):
            lines.append(f"_captured: {baseline.get('ts')}_")
        lines.append("")
        lines.append(_extract_executive_summary(str(baseline.get("analysis_md_path") or "")))
        lines.append("")
        return lines

    lines.append(
        "Before/after comparison of TraceLens Executive Summaries. "
        "The baseline snapshot was captured at PRELUDE; the latest "
        "snapshot was captured after a +10% gain watermark refresh "
        "(see `Coordinator._maybe_enqueue_watermark_roofline`)."
    )
    lines.append("")
    # The ceiling is normally a session constant, but a runtime dtype / quantization change moves it — and then the
    # two sides' Within % have different denominators, which the caveat has to say out loud.
    anchor_note = (
        "The ceiling is a session-level constant "
        "(hardware + model + isl/osl don't change), so baseline and "
        "latest are compared against the same anchor."
        if cmp.get("ceilings_comparable", True)
        else "The two snapshots model different ceilings (the runtime dtype / "
        "quantization changed), so each Within % is measured against its own "
        "anchor and the Δ column is withheld."
    )
    lines.append(
        "_The **Theoretical peak** below is the decode "
        "memory-roofline ceiling derived from the GPU's HBM "
        "bandwidth and the model's weight + KV-cache traffic per "
        "token (see `roofline_ceiling.compute_peak_from_state`). "
        "It is an upper bound: real `output_throughput` always "
        "stays under it because of comm overhead, kernel "
        "efficiency < 100%, and KV-cache fragmentation. **Within "
        "roofline %** = measured / peak; **Gap to roofline %** = "
        f"100 − Within. {anchor_note}_"
    )
    lines.append("")
    lines.extend(format_roofline_metrics_table(cmp))
    lines.append(f"### Baseline snapshot #{base_id}")
    lines.append("")
    lines.append(f"`{baseline.get('analysis_md_path')}`")
    if baseline.get("ts"):
        lines.append(f"_captured: {baseline.get('ts')}_")
    lines.append("")
    lines.append(_extract_executive_summary(str(baseline.get("analysis_md_path") or "")))
    lines.append("")
    lines.append(f"### Post-optimization snapshot #{latest_id}")
    lines.append("")
    lines.append(f"`{latest.get('analysis_md_path')}`")
    if latest.get("ts"):
        lines.append(f"_captured: {latest.get('ts')}_")
    lines.append("")
    lines.append(_extract_executive_summary(str(latest.get("analysis_md_path") or "")))
    lines.append("")
    return lines


def _format_external_baseline_section(ext: dict[str, Any]) -> list[str]:
    """Render the advisory external-baseline section (report-only)."""
    lines: list[str] = []
    status = str(ext.get("status") or "unknown")
    reason = str(ext.get("reason") or "").strip()
    if reason == "no_target_gpu_configured":
        lines.append("## External baseline (not requested)")
    else:
        lines.append("## External baseline (competitor target, advisory)")
    lines.append("")
    if reason == "no_target_gpu_configured":
        lines.append(
            "- No `--compare-against-gpu` was specified; only a marker JSON "
            "was written. Re-run with `--compare-against-gpu <gpu>` (e.g. "
            "`b300` / `mi355x` / `h200`) to fetch the matching InferenceX "
            "reference data point."
        )
        lines.append(f"- Fetched at: {ext.get('fetched_at') or '(unknown)'}")
        lines.append(f"- Status: `{status}` reason=`{reason}` (rows matched: {ext.get('row_count', 0)})")
        lines.append("")
        lines.append(
            "> Advisory only. This block does not feed Objective, scoring, or "
            "any agent prompt; it is shown here purely for post-mortem "
            "comparison."
        )
        lines.append("")
        return lines

    q = ext.get("query") or {}
    lines.append(
        "- Query: "
        f"model=`{q.get('model') or '(unset)'}`  "
        f"gpu=`{q.get('gpu') or '(unset)'}`  "
        f"framework=`{q.get('framework') or '(any)'}`  "
        f"precision=`{q.get('precision') or '(any)'}`  "
        f"ISL/OSL=`{q.get('isl') or '(any)'}/{q.get('osl') or '(any)'}`"
    )
    lines.append(f"- Fetched at: {ext.get('fetched_at') or '(unknown)'}")
    reason_suffix = f" reason=`{reason}`" if reason else ""
    lines.append(f"- Status: `{status}`{reason_suffix} (rows matched: {ext.get('row_count', 0)})")
    warning = ext.get("warning") or ""
    if warning:
        lines.append(f"- Warning: {warning}")

    best = ext.get("best")
    if status == "ok" and isinstance(best, dict):
        lines.append("")
        lines.append(
            f"- Reference best per-GPU throughput: "
            f"**{float(best.get('tput_per_gpu', 0.0)):.1f}** tok/s/GPU "
            f"at concurrency {best.get('conc')}, decode TP {best.get('decode_tp')}"
        )
        ttft = float(best.get("mean_ttft_ms") or 0.0)
        tpot = float(best.get("mean_tpot_ms") or 0.0)
        e2el = float(best.get("mean_e2el_ms") or 0.0)
        if ttft:
            lines.append(f"- Reference mean TTFT: {ttft:.1f} ms")
        if tpot:
            lines.append(f"- Reference mean TPOT: {tpot:.3f} ms")
        if e2el:
            lines.append(f"- Reference mean E2E latency: {e2el:.1f} ms")
        if best.get("date"):
            lines.append(f"- Reference run date: {best.get('date')}")
    else:
        lines.append("")
        lines.append("- No reference best available — orchestrator was not affected by this section.")

    lines.append("")
    lines.append(
        "> Advisory only. This block does not feed Objective, scoring, or any "
        "agent prompt; it is shown here purely for post-mortem comparison."
    )
    lines.append("")
    return lines


def _format_conc_sweep_curve_section(summary: dict[str, Any]) -> list[str]:
    """Render the concurrency-sweep curve section in the Markdown report."""
    png_rel = summary.get("conc_sweep_curve_png")
    if not png_rel:
        return []
    # final.md and the PNG both live in reports_dir, so the embed must be relative to final.md's own directory (its
    # basename), not the session-root-relative path stored in final.json.
    png_md_rel = Path(str(png_rel)).name
    lines: list[str] = []
    lines.append("## Concurrency Sweep — Throughput vs Interactivity")
    lines.append("")
    # The PNG draws its own axis labels, and they differ by workload; naming them here too drifts from the chart.
    lines.append("The post-optimization concurrency ladder.  Red = baseline, orange = optimized.")
    lines.append("")
    lines.append(f"![Concurrency sweep curve]({png_md_rel})")
    lines.append("")
    return lines


def _render_conc_sweep_curve_for_report(
    session_dir: Path,
    output_dir: Path,
    state: SharedState,
) -> Path | None:
    """Render the concurrency-sweep curve PNG into the reports directory."""
    from hyperloom.inference_optimizer.session.session_paths import reports_dir as _reports_dir
    from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve

    json_path = _reports_dir(session_dir) / "conc_sweep_summary.json"
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("report_executor: cannot load conc_sweep_summary.json for plot: %s", exc)
        return None

    png_path = output_dir / "conc_sweep_curve.png"
    tp = int(payload.get("tp") or getattr(state, "tp", 0) or 1)

    model_label = str(getattr(state, "model_name", "") or "")
    gpu_label = str(getattr(state, "gpu_type", "") or "").upper()
    isl = int(payload.get("isl") or 0)
    osl = int(payload.get("osl") or 0)

    return render_conc_sweep_curve(
        payload,
        png_path,
        model_label=model_label,
        gpu_label=gpu_label,
        tp=tp,
        isl=isl,
        osl=osl,
        draw_ceiling=False,
    )


def _load_external_baseline(session_dir: Path) -> dict[str, Any] | None:
    """Best-effort load of ``target_analysis/target_baseline.json``; ``None`` when missing / unreadable (errors swallowed so a corrupt JSON never breaks report generation)."""
    try:
        from hyperloom.inference_optimizer.session.session_paths import target_baseline_json

        path = target_baseline_json(session_dir)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "report_executor: failed to load external baseline from %s: %s",
            session_dir,
            exc,
        )
        return None


def _write_kernel_opt_summary(
    state: SharedState,
    session_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Build + write ``reports/kernel_optimization_summary.json``."""
    try:
        from ...kernel.attempt_summary import build_kernel_optimization_summary

        summary = build_kernel_optimization_summary(state, session_dir)
        out_path = output_dir / "kernel_optimization_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        return out_path
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "report_executor: failed to write kernel_optimization_summary.json: %s",
            exc,
        )
        return None


def _read_conc_sweep_pointer(session_dir: Path) -> dict[str, Any] | None:
    """Build the small ``conc_sweep_summary`` pointer for ``final.json`` (report_path + status + summary); ``None`` when conc_sweep wrote no summary."""
    from hyperloom.inference_optimizer.session.session_paths import reports_dir as _reports_dir

    json_path = _reports_dir(session_dir) / "conc_sweep_summary.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "report_executor: cannot read conc_sweep_summary.json: %s",
            exc,
        )
        return None
    try:
        rel = json_path.relative_to(session_dir).as_posix()
    except ValueError:
        rel = json_path.as_posix()
    return {
        "report_path": rel,
        "status": data.get("status"),
        "summary": data.get("summary", {}),
        "budget_exhausted": bool(data.get("budget_exhausted", False)),
        "total_budget_sec": data.get("total_budget_sec"),
    }


def _read_ko_summary_totals(path: Path) -> dict[str, int]:
    """Re-read totals so the final.json pointer doesn't drift from disk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        totals = data.get("totals") or {}
        return {k: int(v) for k, v in totals.items() if isinstance(v, (int, float))}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _highlight(payload: dict, topic: str, from_agent: str) -> dict[str, Any]:
    """Pick the most useful 1-line summary out of an event's payload."""
    summary = ""
    if topic == "proposal":
        summary = f"action_name={payload.get('action_name')}"
    elif topic == "review_verdict":
        summary = f"verdict={payload.get('verdict')} reason={(payload.get('reasoning') or '')[:60]}"
    elif topic == "decision":
        summary = (
            f"kind={payload.get('kind')} action={payload.get('action_name')} task={(payload.get('task_id') or '')[:8]}"
        )
    elif topic == "delegated_result":
        out_tput = (payload.get("result") or {}).get("output_throughput")
        decision = (payload.get("result") or {}).get("decision")
        summary = f"kind={payload.get('kind')} state={payload.get('state')} tput={out_tput} decision={decision}"
    elif topic == "response":
        summary = f"kind={payload.get('kind')} status={payload.get('status')}"
    elif topic == "alert":
        summary = f"sev={payload.get('severity')} {payload.get('summary', '')}"
    else:
        summary = json.dumps({k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))})[:80]
    return {"topic": topic, "from_agent": from_agent, "summary": summary, "payload": payload}


# ---------------------------------------------------------------------------
class ReportExecutor:
    """ActionRunner for the ``report`` action."""

    DEFAULT_HIGHLIGHT_TOPICS = (
        "proposal",
        "review_verdict",
        "decision",
        "delegated_result",
        "response",
        "alert",
    )

    def __init__(self, *, max_highlights: int = 50):
        """Initialize the report executor."""
        self.max_highlights = int(max_highlights)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the report-generation action for the given context."""
        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            return {"status": "failed", "error": "report_executor: could not resolve session_dir"}

        params = ctx.task.params or {}
        from hyperloom.inference_optimizer.session.session_paths import reports_dir

        output_dir = Path(params.get("output_dir") or reports_dir(session_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        max_highlights = int(params.get("max_highlights", self.max_highlights))
        highlight_topics = params.get("highlight_topics") or self.DEFAULT_HIGHLIGHT_TOPICS

        state = SharedState.load_or_init(session_dir)
        # Only demote benign upstream WARNs from highlights on a baseline failure.
        suppress_benign_highlights = str(getattr(state, "stop_reason", "") or "") == "baseline_failed"

        # Pull bus stats over a fresh connection.
        db = SqliteConnection(db_path_for(session_dir))
        try:
            bus = MessageBus(db)
            ev_rows = await bus.tail(n=10_000)
            ev_counts = Counter(m.topic for m in ev_rows)
            highlights: list[dict] = []
            for m in ev_rows:
                if m.topic in highlight_topics:
                    h = _highlight(m.payload or {}, m.topic, m.from_agent)
                    # On baseline_failed, suppress benign upstream WARN headlines so they never become the top-level
                    # highlight.
                    if suppress_benign_highlights and _highlight_is_benign(h):
                        continue
                    highlights.append(h)
        finally:
            db.close()

        highlights = highlights[:max_highlights]
        external_baseline = _load_external_baseline(session_dir)
        summary = _build_summary_dict(
            state,
            dict(ev_counts),
            highlights,
            external_baseline=external_baseline,
            session_dir=session_dir,
        )

        # Kernel-optimization forensic summary in a separate file (pointer added to final.json).
        ko_summary_path = _write_kernel_opt_summary(state, session_dir, output_dir)
        if ko_summary_path is not None:
            try:
                rel = ko_summary_path.relative_to(session_dir)
                summary["kernel_optimization_summary"] = {
                    "report_path": str(rel),
                    "totals": _read_ko_summary_totals(ko_summary_path),
                }
            except ValueError:
                summary["kernel_optimization_summary"] = {
                    "report_path": str(ko_summary_path),
                    "totals": _read_ko_summary_totals(ko_summary_path),
                }

        # Post-sweep concurrency comparison pointer.
        cs_pointer = _read_conc_sweep_pointer(session_dir)
        if cs_pointer is not None:
            summary["conc_sweep_summary"] = cs_pointer

        # Best-effort InferenceX-style concurrency sweep plot.
        conc_sweep_curve_png: Path | None = None
        try:
            conc_sweep_curve_png = _render_conc_sweep_curve_for_report(
                session_dir=session_dir,
                output_dir=output_dir,
                state=state,
            )
        except Exception:  # noqa: BLE001
            log.debug("report_executor: conc_sweep curve render failed", exc_info=True)
        if conc_sweep_curve_png is not None:
            try:
                summary["conc_sweep_curve_png"] = conc_sweep_curve_png.relative_to(session_dir).as_posix()
            except ValueError:
                summary["conc_sweep_curve_png"] = conc_sweep_curve_png.as_posix()

        json_path = output_dir / "final.json"
        md_path = output_dir / "final.md"
        # Atomic write: a kill mid-flush must never leave a non-empty but invalid final.json on disk (issue #464 —
        # downstream keys off it, and the crash-safe fallback would otherwise see garbled JSON).
        _common_io.atomic_write_text(json_path, json.dumps(summary, indent=2, sort_keys=True))
        md_path.write_text(_format_md(summary), encoding="utf-8")

        log.info(
            "report_executor: wrote %s and %s (cumulative_gain_validated=%.2f%%)",
            md_path,
            json_path,
            state.cumulative_gain_validated,
        )
        publish_result = self._maybe_publish_results(session_dir, state)
        return {
            "status": "succeeded",
            "session_id": state.session_id,
            "json_path": str(json_path),
            "md_path": str(md_path),
            "summary": summary,
            "publish_result": publish_result,
        }

    def _resolve_session_dir(self, ctx) -> Path | None:
        """Best-effort session_dir resolution."""
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("session_dir"):
            return Path(extra["session_dir"])
        params = ctx.task.params or {}
        if params.get("session_dir"):
            return Path(params["session_dir"])
        from hyperloom.inference_optimizer.session.paths import session_dir as _sd

        candidate = _sd()
        if candidate.exists() and (candidate / "state.json").exists():
            return candidate
        return None

    def _maybe_publish_results(self, session_dir: Path, state: SharedState) -> dict[str, Any]:
        """Best-effort publish hook for code-driven optimizer runs (opt-in unless the results service URL is configured)."""
        service_url = os.environ.get("HYPERLOOM_RESULTS_SERVICE_URL", "")
        auto_publish = os.environ.get("HYPERLOOM_RESULTS_AUTO_PUBLISH", "").lower()
        if not service_url and auto_publish not in {"1", "true", "yes"}:
            return {"enabled": False, "reason": "HYPERLOOM_RESULTS_SERVICE_URL not set"}

        repo_root = Path(__file__).resolve().parents[3]
        helper = repo_root / "ci" / "publish_artifacts.py"
        if not helper.exists():
            return {"enabled": False, "reason": f"{helper} not found"}

        cmd = [
            "python3",
            str(helper),
            "--task-dir",
            str(session_dir),
            "--out-dir",
            str(session_dir / "normalized"),
            "--model",
            state.model_name or "unknown",
            "--display-name",
            state.session_id or "hyperloom-report",
        ]
        if service_url:
            cmd.extend(["--url", service_url])

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            return {
                "enabled": True,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        except Exception as e:
            log.warning("report_executor: result publish failed: %s", e)
            return {"enabled": True, "error": str(e)}


report_executor = ReportExecutor()


__all__ = ["ReportExecutor", "report_executor"]
