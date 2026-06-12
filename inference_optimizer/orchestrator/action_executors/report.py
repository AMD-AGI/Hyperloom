# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Real ``report`` ActionRunner.

Reads SharedState + the bus event log and writes
``$SESSION_DIR/reports/final.json`` (machine-readable, dashboard shape) and
``final.md`` (human-readable). The returned dict surfaces both paths. Files
are compact: stop_reason, baseline + best, cumulative gain, event counts,
and a top-N highlight list.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..message_bus import MessageBus
from ..shared_state import SharedState
from ...paths import db_path_for
from ...storage.connection import SqliteConnection


log = logging.getLogger(__name__)


def _safe_call(state: Any, method: str, default: Any) -> Any:
    """Call a zero-arg SharedState helper, returning ``default`` when absent
    or raising."""
    fn = getattr(state, method, None)
    if not callable(fn):
        return default
    try:
        return fn()
    except Exception:  # noqa: BLE001 — report must never crash on annotations
        return default


def _build_summary_dict(
    state: SharedState,
    ev_counts: dict[str, int],
    highlights: list[dict],
    *,
    external_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable session summary dict.

    Args:
        state (SharedState): The session's shared state.
        ev_counts (dict[str, int]): Event counts keyed by bus topic.
        highlights (list[dict]): Top-N highlighted decisions/verdicts.
        external_baseline (dict[str, Any] | None): Optional external
            baseline comparison block to embed.

    Returns:
        dict[str, Any]: The summary payload written to ``final.json``,
        including an optional roofline-comparison block.
    """
    summary: dict[str, Any] = {
        "session_id":       state.session_id,
        "model_name":       state.model_name,
        "model_path":       state.model_path,
        "model_class":      state.model_class,
        "stop_reason":      state.stop_reason,
        "baseline_tput":    state.baseline_tput,
        "baseline_accuracy": state.baseline_accuracy,
        # steward verdict + history (report §9.1 reads the rationale verbatim).
        "remaining_gaps_assessment": dict(
            state.last_remaining_gaps_assessment or {}
        ),
        "remaining_gaps_assessments_history": list(
            state.remaining_gaps_assessments or []
        ),
        "current_best":     state.current_best,
        "cumulative_gain":  state.cumulative_gain,
        # Per-round-sum gain (back-compat) vs the validated gain (what the
        # run actually delivered).
        "cumulative_gain_validated":          state.cumulative_gain_validated,
        "cumulative_gain_validated_ts":       state.cumulative_gain_validated_ts,
        "cumulative_gain_validated_stack_len": state.cumulative_gain_validated_stack_len,
        "optimization_stack_len":             len(state.optimization_stack or []),
        # Honesty annotations: surface unfinished/unvalidated work instead of
        # blocking the report; read defensively for partial-state stubs.
        "has_unvalidated_keeps":              _safe_call(state, "optimization_stack_has_unvalidated_keeps", False),
        "untried_hot_reusable_kernels":       list(_safe_call(state, "untried_hot_reusable_kernels", []) or []),
        "pending_keep_kernels":               list(_safe_call(state, "pending_keep_kernel_ids", []) or []),
        "crash_count":      state.crash_count,
        "pruned_families":  state.pruned_families,
        "max_minutes":      state.max_minutes,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "event_counts_by_topic": ev_counts,
        "highlights": highlights,
        # Degraded-mode advisory: surfaces a multimodal-text-fallback run so the
        # reader knows benchmark numbers reflect the text path only.
        "degraded_mode": bool(getattr(state, "degraded_mode", False)),
        "model_warnings": list(getattr(state, "model_warnings", None) or []),
    }
    if external_baseline:
        summary["external_baseline"] = external_baseline
    # Roofline Comparison (PR #321): walk the append-only
    # ``state.roofline_snapshots`` (entry[0] baseline, entry[-1] latest);
    # emit only when at least one snapshot exists.
    from ..roofline_snapshot import build_roofline_comparison_from_history
    cmp = build_roofline_comparison_from_history(
        getattr(state, "roofline_snapshots", None)
    )
    if cmp:
        summary["roofline_comparison"] = cmp
    return summary


def _format_md(summary: dict[str, Any]) -> str:
    """Render the human-readable Markdown report from a summary dict.

    Args:
        summary (dict[str, Any]): The summary payload built by
            :func:`_build_summary_dict`.

    Returns:
        str: The full Markdown report body.
    """
    cb = summary.get("current_best") or {}
    cb_tput = cb.get("tput") if isinstance(cb, dict) else None
    lines: list[str] = []
    lines.append(f"# Inference Optimizer Report — {summary['session_id']}")
    lines.append("")
    lines.append(f"- **Model**: {summary['model_name']}  (`{summary['model_path']}`)")
    lines.append(f"- **Stop reason**: `{summary['stop_reason']}`")
    stop_detail = str(summary.get("stop_detail") or "").strip()
    if stop_detail:
        lines.append(f"- **Stop detail**: {stop_detail}")
    if summary.get("degraded_mode"):
        lines.append(
            "- **⚠ Degraded mode**: ran on the TEXT path only "
            "(multimodal inputs ignored) — see 'Degraded mode' below"
        )
    lines.append(f"- **Budget**: {summary['max_minutes']} minutes")
    lines.append(f"- **Generated**: {summary['report_generated_at']}")
    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.append(f"- baseline_tput        : `{summary['baseline_tput']:.1f}` tok/s/GPU")
    if cb_tput is not None:
        lines.append(f"- current_best        : `{cb_tput:.1f}` tok/s/GPU "
                      f"(action=`{cb.get('action','?')}`)")
    # Per-round sum — informational, not end-to-end deliverable.
    lines.append(
        f"- cumulative_gain     : `{summary['cumulative_gain']:.2f}%`"
        f"  *(per-round sum — informational only)*"
    )
    # Validated gain — always printed so the report never quotes only the
    # (often inflated) raw sum.
    val_gain = summary.get("cumulative_gain_validated", 0.0) or 0.0
    val_ts = summary.get("cumulative_gain_validated_ts") or ""
    val_len = summary.get("cumulative_gain_validated_stack_len", 0) or 0
    stack_len = summary.get("optimization_stack_len", 0) or 0
    if val_ts:
        stale = " ⚠ stack changed since validation" if stack_len > val_len else ""
        lines.append(
            f"- cumulative_gain_val : `{val_gain:.2f}%` "
            f"(validated_at_stack_len={val_len}, ts={val_ts}){stale}"
        )
    else:
        lines.append(
            f"- cumulative_gain_val : `0.00%` "
            f"⚠ never validated — no full-stack rebench ran in this session"
        )
    if cb.get("ttft_mean_ms") is not None:
        lines.append(f"- ttft_mean      : `{cb.get('ttft_mean_ms'):.1f}` ms")
    if cb.get("e2el_mean_ms") is not None:
        lines.append(f"- e2el_mean      : `{cb.get('e2el_mean_ms'):.1f}` ms")
    lines.append("")
    lines.extend(_format_completeness_annotations(summary))
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- crash_count    : {summary['crash_count']}")
    lines.append(f"- pruned_families: {summary['pruned_families'] or '(none)'}")
    lines.append("")
    lines.append("## Event counts")
    lines.append("")
    if not summary.get("event_counts_by_topic"):
        lines.append("- (no events recorded)")
    else:
        for topic, n in sorted(summary["event_counts_by_topic"].items(),
                                key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{topic}`: {n}")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    if not summary["highlights"]:
        lines.append("(no highlight events captured)")
    else:
        for h in summary["highlights"][:50]:
            lines.append(
                f"- `{h.get('topic','?')}` from `{h.get('from_agent','?')}`: "
                f"{h.get('summary','')}"
            )
    lines.append("")

    lines.extend(_format_degraded_mode_section(summary))

    # steward verdict transcript.
    lines.extend(_format_steward_section(summary))

    roofline_cmp = summary.get("roofline_comparison")
    if roofline_cmp:
        lines.extend(_format_roofline_comparison_section(roofline_cmp))

    ext = summary.get("external_baseline")
    if ext:
        lines.extend(_format_external_baseline_section(ext))

    return "\n".join(lines)


def _format_degraded_mode_section(summary: dict[str, Any]) -> list[str]:
    """Render the degraded-mode section (multimodal models run on the text path).

    Empty when the run was not degraded. Lists each recorded model warning so
    the reader knows benchmark numbers reflect the text decoder alone.
    """
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


def _format_completeness_annotations(summary: dict[str, Any]) -> list[str]:
    """Render honesty annotations for work left unfinished (unvalidated
    KEEPs, untried hot kernels, KEEPs awaiting integrate)."""
    unvalidated = bool(summary.get("has_unvalidated_keeps"))
    untried = list(summary.get("untried_hot_reusable_kernels") or [])
    pending_keeps = list(summary.get("pending_keep_kernels") or [])
    if not (unvalidated or untried or pending_keeps):
        return []
    lines: list[str] = ["## Completeness annotations", ""]
    if unvalidated:
        lines.append(
            "- ⚠ `optimization_stack` has KEEPs landed since the last "
            "full-stack rebench — `cumulative_gain_validated` does not "
            "yet reflect them (unvalidated)."
        )
    if pending_keeps:
        lines.append(
            f"- ⚠ kernel_opt KEEPs awaiting integrate: "
            f"{', '.join(pending_keeps)}."
        )
    if untried:
        lines.append(
            f"- ⚠ reusable hot kernels with no kernel_opt attempt: "
            f"{', '.join(untried)}."
        )
    lines.append("")
    return lines


def _format_steward_section(summary: dict[str, Any]) -> list[str]:
    """Render any legacy session_steward verdict + history.

    Steward was retired in P3_17; kept for back-compat with older state.json
    carrying a populated ``last_remaining_gaps_assessment``.
    """
    assessment = summary.get("remaining_gaps_assessment") or {}
    history = summary.get("remaining_gaps_assessments_history") or []
    if not assessment and not history:
        return []
    lines: list[str] = ["## Remaining gaps (steward assessment)", ""]
    if assessment:
        rec = assessment.get("recommendation", "")
        ts = assessment.get("ts", "")
        potential = assessment.get(
            "remaining_potential_pct_estimate", 0.0
        ) or 0.0
        rationale = (
            assessment.get("rationale", "") or ""
        ).strip()
        next_gap = assessment.get("next_gap_canonical_id", "")
        lines.append(f"- final verdict: `{rec}` at `{ts}`")
        lines.append(
            f"- remaining_potential_pct_estimate: `{potential:.2f}%`"
        )
        if next_gap:
            lines.append(f"- next_gap_canonical_id: `{next_gap}`")
        if rationale:
            lines.append("")
            lines.append("> " + rationale.replace("\n", "\n> "))
            lines.append("")
    if len(history) > 1:
        lines.append(f"- prior assessments: {len(history) - 1}")
    lines.append("")
    return lines


def _extract_executive_summary(analysis_md_path: str) -> str:
    """Extract the ``## Executive Summary`` block (up to the next level-2
    heading) from analysis.md. Best-effort: returns a marker string when the
    file is missing / unparseable rather than crashing the report.
    """
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
    # Cap at ~2KB (full analysis.md remains on disk via ``analysis_md_path``).
    if len(block) > 2048:
        block = block[:2045] + "..."
    return block


def _format_roofline_comparison_section(cmp: dict[str, Any]) -> list[str]:
    """Render the ``## Roofline Comparison`` section from ``cmp`` (built by
    :func:`roofline_snapshot.build_roofline_comparison_from_history`).

    Two modes: ``single_snapshot`` (only the PRELUDE bootstrap ran; one
    Executive Summary + Base metric table) and ``before_after`` (a watermark
    refresh produced a distinct snapshot; two summaries + Base/Opt/Δ table).
    """
    from ..roofline_snapshot import format_roofline_metrics_table

    lines: list[str] = ["## Roofline Comparison", ""]
    baseline = cmp.get("baseline") or {}
    latest = cmp.get("latest") or {}
    base_id = baseline.get("snapshot_id")
    latest_id = latest.get("snapshot_id")
    mode = cmp.get("mode") or (
        "single_snapshot"
        if (base_id is not None and base_id == latest_id)
        else "before_after"
    )

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
        lines.append(_extract_executive_summary(
            str(baseline.get("analysis_md_path") or "")
        ))
        lines.append("")
        return lines

    lines.append(
        "Before/after comparison of TraceLens Executive Summaries. "
        "The baseline snapshot was captured at PRELUDE; the latest "
        "snapshot was captured after a +10% gain watermark refresh "
        "(see `Coordinator._maybe_enqueue_watermark_roofline`)."
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
        "100 − Within. The ceiling is a session-level constant "
        "(hardware + model + isl/osl don't change), so baseline and "
        "latest are compared against the same anchor._"
    )
    lines.append("")
    lines.extend(format_roofline_metrics_table(cmp))
    lines.append(f"### Baseline snapshot #{base_id}")
    lines.append("")
    lines.append(f"`{baseline.get('analysis_md_path')}`")
    if baseline.get("ts"):
        lines.append(f"_captured: {baseline.get('ts')}_")
    lines.append("")
    lines.append(_extract_executive_summary(
        str(baseline.get("analysis_md_path") or "")
    ))
    lines.append("")
    lines.append(f"### Post-optimization snapshot #{latest_id}")
    lines.append("")
    lines.append(f"`{latest.get('analysis_md_path')}`")
    if latest.get("ts"):
        lines.append(f"_captured: {latest.get('ts')}_")
    lines.append("")
    lines.append(_extract_executive_summary(
        str(latest.get("analysis_md_path") or "")
    ))
    lines.append("")
    return lines


def _format_external_baseline_section(ext: dict[str, Any]) -> list[str]:
    """Render the advisory external-baseline section (report-only).

    ``ext`` comes from :func:`_load_external_baseline`. Facts only (no derived
    gap %, no "should reach" wording) so it never reads as an implicit KPI.
    Heading varies by ``ext['reason']``: ``ok`` (full reference-best),
    ``no_target_gpu_configured`` ("(not requested)"), else "(advisory)".
    """
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
        lines.append(
            f"- Status: `{status}` reason=`{reason}` "
            f"(rows matched: {ext.get('row_count', 0)})"
        )
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
    lines.append(
        f"- Status: `{status}`{reason_suffix} "
        f"(rows matched: {ext.get('row_count', 0)})"
    )
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
        lines.append(
            "- No reference best available — orchestrator was not affected by "
            "this section."
        )

    lines.append("")
    lines.append(
        "> Advisory only. This block does not feed Objective, scoring, or any "
        "agent prompt; it is shown here purely for post-mortem comparison."
    )
    lines.append("")
    return lines


def _load_external_baseline(session_dir: Path) -> dict[str, Any] | None:
    """Best-effort load of ``target_analysis/target_baseline.json``; ``None``
    when missing / unreadable (errors swallowed so a corrupt JSON never
    breaks report generation)."""
    try:
        from ...session_paths import target_baseline_json
        path = target_baseline_json(session_dir)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "report_executor: failed to load external baseline from %s: %s",
            session_dir, exc,
        )
        return None


def _write_kernel_opt_summary(
    state: SharedState,
    session_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Build + write ``reports/kernel_optimization_summary.json``.

    Best-effort (failure logged, returns ``None`` so the final.json write
    still happens). Aggregates ``kernel_opt_attempts`` with per-kernel
    ``results/<kid>.json`` for the "why no optimized kernel?" view.
    """
    try:
        from ..kernel_attempt_summary import build_kernel_optimization_summary
        summary = build_kernel_optimization_summary(state, session_dir)
        out_path = output_dir / "kernel_optimization_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        # Author-time breakdown capture: mirror the summary into the recorder.
        try:
            from ...breakdown.recorder import instrument
            instrument.record_singleton_section(
                session_dir, "kernel_optimization_summary", summary,
                producer="coordinator",
            )
        except Exception:  # noqa: BLE001
            pass
        return out_path
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "report_executor: failed to write kernel_optimization_summary.json: %s",
            exc,
        )
        return None


def _read_conc_sweep_pointer(session_dir: Path) -> dict[str, Any] | None:
    """Build the small ``conc_sweep_summary`` pointer for ``final.json``
    (report_path + status + summary); ``None`` when conc_sweep wrote no
    summary."""
    from ...session_paths import reports_dir as _reports_dir
    json_path = _reports_dir(session_dir) / "conc_sweep_summary.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "report_executor: cannot read conc_sweep_summary.json: %s", exc,
        )
        return None
    try:
        rel = json_path.relative_to(session_dir).as_posix()
    except ValueError:
        rel = json_path.as_posix()
    return {
        "report_path":      rel,
        "status":           data.get("status"),
        "summary":          data.get("summary", {}),
        "budget_exhausted": bool(data.get("budget_exhausted", False)),
        "total_budget_sec": data.get("total_budget_sec"),
    }


def _read_ko_summary_totals(path: Path) -> dict[str, int]:
    """Re-read totals so the final.json pointer doesn't drift from disk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        totals = data.get("totals") or {}
        return {
            k: int(v)
            for k, v in totals.items()
            if isinstance(v, (int, float))
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _highlight(payload: dict, topic: str, from_agent: str) -> dict[str, Any]:
    """Pick the most useful 1-line summary out of an event's payload.

    Args:
        payload (dict): The bus event payload.
        topic (str): The event topic, which selects the summary format.
        from_agent (str): The agent that emitted the event.

    Returns:
        dict[str, Any]: A highlight record with ``topic``, ``from_agent``,
        a 1-line ``summary``, and the original ``payload``.
    """
    summary = ""
    if topic == "proposal":
        summary = f"action_name={payload.get('action_name')}"
    elif topic == "review_verdict":
        summary = (f"verdict={payload.get('verdict')} "
                   f"reason={(payload.get('reasoning') or '')[:60]}")
    elif topic == "decision":
        summary = (f"kind={payload.get('kind')} "
                   f"action={payload.get('action_name')} "
                   f"task={(payload.get('task_id') or '')[:8]}")
    elif topic == "delegated_result":
        out_tput = (payload.get("result") or {}).get("output_throughput")
        decision = (payload.get("result") or {}).get("decision")
        summary = (f"kind={payload.get('kind')} state={payload.get('state')} "
                   f"tput={out_tput} decision={decision}")
    elif topic == "response":
        summary = (f"kind={payload.get('kind')} "
                   f"status={payload.get('status')}")
    elif topic == "alert":
        summary = (f"sev={payload.get('severity')} {payload.get('summary','')}")
    else:
        summary = json.dumps({k: v for k, v in payload.items()
                                if isinstance(v, (str, int, float, bool))})[:80]
    return {"topic": topic, "from_agent": from_agent, "summary": summary,
            "payload": payload}


# ---------------------------------------------------------------------------
class ReportExecutor:
    """ActionRunner for the ``report`` action.

    Honours ``ctx.task.params``::

        output_dir:        write final.{md,json} here (default
                           ``$SESSION_DIR/reports``)
        highlight_topics:  list of topics to surface in ``highlights``
                           (default: proposal / review_verdict / decision /
                            delegated_result / response / alert)
        max_highlights:    cap the highlights list (default 50)
    """

    DEFAULT_HIGHLIGHT_TOPICS = (
        "proposal", "review_verdict", "decision",
        "delegated_result", "response", "alert",
    )

    def __init__(self, *, max_highlights: int = 50):
        """Initialize the report executor.

        Args:
            max_highlights (int): Maximum number of highlight events to
                include in the report. Defaults to ``50``.
        """
        self.max_highlights = int(max_highlights)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the report-generation action for the given context.

        Args:
            ctx: Action context; used to resolve the session directory
                and report parameters.

        Returns:
            A result dict with a ``status`` field, failing when the
            session directory cannot be resolved.
        """
        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            return {"status": "failed",
                    "error": "report_executor: could not resolve session_dir"}

        params = ctx.task.params or {}
        from ...session_paths import reports_dir
        output_dir = Path(params.get("output_dir") or reports_dir(session_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        max_highlights = int(params.get("max_highlights", self.max_highlights))
        highlight_topics = (
            params.get("highlight_topics") or self.DEFAULT_HIGHLIGHT_TOPICS
        )

        state = SharedState.load_or_init(session_dir)

        # Pull bus stats over a fresh connection (SQLite WAL shares cleanly).
        db = SqliteConnection(db_path_for(session_dir))
        try:
            bus = MessageBus(db)
            ev_rows = await bus.tail(n=10_000)
            ev_counts = Counter(m.topic for m in ev_rows)
            highlights: list[dict] = []
            for m in ev_rows:
                if m.topic in highlight_topics:
                    highlights.append(_highlight(m.payload or {}, m.topic, m.from_agent))
        finally:
            db.close()

        highlights = highlights[:max_highlights]
        external_baseline = _load_external_baseline(session_dir)
        summary = _build_summary_dict(
            state,
            dict(ev_counts),
            highlights,
            external_baseline=external_baseline,
        )

        # Kernel-optimization forensic summary in a separate file (pointer
        # added to final.json for discoverability).
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

        # Post-sweep concurrency comparison pointer (conc_sweep writes its
        # own JSON/CSV; surface a compact summary for discoverability).
        cs_pointer = _read_conc_sweep_pointer(session_dir)
        if cs_pointer is not None:
            summary["conc_sweep_summary"] = cs_pointer

        json_path = output_dir / "final.json"
        md_path = output_dir / "final.md"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        md_path.write_text(_format_md(summary), encoding="utf-8")

        log.info(
            "report_executor: wrote %s and %s "
            "(cumulative_gain=%.2f%% per_round_sum / %.2f%% validated)",
            md_path, json_path,
            state.cumulative_gain, state.cumulative_gain_validated,
        )
        publish_result = self._maybe_publish_results(session_dir, state)
        return {
            "status":      "succeeded",
            "session_id":  state.session_id,
            "json_path":   str(json_path),
            "md_path":     str(md_path),
            "summary":     summary,
            "publish_result": publish_result,
        }

    def _resolve_session_dir(self, ctx) -> Path | None:
        """Best-effort session_dir resolution.

        Order: ``ctx.extra['session_dir']`` → ``task.params['session_dir']``
        → :func:`paths.session_dir` (only if it exists with ``state.json``)
        → None (runner returns failed).
        """
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("session_dir"):
            return Path(extra["session_dir"])
        params = ctx.task.params or {}
        if params.get("session_dir"):
            return Path(params["session_dir"])
        from ...paths import session_dir as _sd
        candidate = _sd()
        if candidate.exists() and (candidate / "state.json").exists():
            return candidate
        return None

    def _maybe_publish_results(self, session_dir: Path, state: SharedState) -> dict[str, Any]:
        """Best-effort publish hook for code-driven optimizer runs (opt-in
        unless the results service URL is configured)."""
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
