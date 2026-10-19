# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Markdown rendering of the final optimization report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_composite_perf_section(summary: dict[str, Any]) -> list[str]:
    """Render recorded grading; summaries predating the snapshot keep their legacy layout."""
    lines: list[str] = []
    comparison = summary.get("performance_comparison")
    if comparison is not None:
        from hyperloom.common.perf_metric import GRADED_OUTPUT, INTVTY_OBJECTIVES, INTVTY_V1

        # Read the family, not one percentile: the graded axis is the median while the session marker still names
        # the tail, and pinning either one here prints the wrong mode for the other.
        graded_on_intvty = comparison["objective"] in INTVTY_OBJECTIVES
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
            lines.append(f"- reference tput      : `{comparison['tput_reference']:.1f}` tok/s (total, diagnostic)")
            lines.append(f"- candidate tput      : `{comparison['tput_candidate']:.1f}` tok/s (total, diagnostic)")
        return lines

    from hyperloom.common.gain_math import gain_pct
    from hyperloom.common.perf_metric import (
        GRADED_INTVTY,
        INTVTY_V1,
        intvty_grading_enabled,
        intvty_of,
        parse_intvty_noise_pct,
        perf_snapshot_from_mapping,
        total_tput_of,
    )

    baseline = perf_snapshot_from_mapping(summary.get("baseline_perf"))
    if not baseline:
        return lines
    cb = summary.get("current_best") or {}
    cb_snap = perf_snapshot_from_mapping(cb) if isinstance(cb, dict) else None
    lines.extend(["## AgentX perf (interactivity objective, per-chip tput guard)", ""])
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
            lines.append(f"- total tput change   : `{tput_gain:+.2f}%` (diagnostic, neither objective nor guard)")
    grading = summary.get("grading") if isinstance(summary.get("grading"), dict) else {}
    objective = str(grading.get("objective") or "").strip()
    if objective == GRADED_INTVTY:
        noise_pct = grading.get("noise_pct")
        band = float(noise_pct) if isinstance(noise_pct, (int, float)) else parse_intvty_noise_pct()
        lines.append(f"- grading mode        : `{INTVTY_V1}` (noise band `{band:.1f}%`)")
    elif objective:
        lines.append(f"- grading mode        : `{objective}`")
    else:
        if intvty_grading_enabled(benchmark_mode=str(summary.get("benchmark_mode") or "")):
            lines.append(f"- grading mode        : `{INTVTY_V1}` (noise band `{parse_intvty_noise_pct():.1f}%`)")
        else:
            lines.append("- grading mode        : `output_throughput` (AgentX grading not in effect)")
    return lines


def _format_report_header(summary: dict[str, Any]) -> list[str]:
    """The title block: what ran, why it stopped, and what the run was allowed to spend."""
    lines = [
        f"# Inference Optimizer Report — {summary['session_id']}",
        "",
        f"- **Model**: {summary['model_name']}  (`{summary['model_path']}`)",
        f"- **Stop reason**: `{summary['stop_reason']}`",
    ]
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
    return lines


def _format_cumulative_gain_lines(summary: dict[str, Any]) -> list[str]:
    """What the session's cumulative gain was last validated against, printed even when it never was.

    Stated rather than implied: an absent line is indistinguishable from a session that never promoted.
    """
    val_gain = summary.get("cumulative_gain_validated", 0.0) or 0.0
    val_ts = summary.get("cumulative_gain_validated_ts") or ""
    val_len = summary.get("cumulative_gain_validated_stack_len", 0) or 0
    stack_len = summary.get("optimization_stack_len", 0) or 0
    changed_since_validation = stack_len > val_len or bool(summary.get("has_unvalidated_keeps"))
    stale = " ⚠ stack changed since validation" if changed_since_validation else ""
    if val_ts:
        lines = [f"- cumulative_gain_val : `{val_gain:.2f}%` (validated_at_stack_len={val_len}, ts={val_ts}){stale}"]
    elif val_gain or val_len:
        lines = [f"- cumulative_gain_val : `{val_gain:.2f}%` (validated_at_stack_len={val_len}, ts=<missing>){stale}"]
    else:
        lines = ["- cumulative_gain_val : `0.00%` ⚠ never validated — nothing has promoted in this session"]
    validation_status = summary.get("cumulative_validation_status")
    if validation_status:
        explanations = {
            "current": "stored gain agrees with the current measured comparison",
            "stale": "stack changed since validation; stored gain does not validate the current stack",
            "unavailable": "no current comparable validation evidence; comparison is diagnostic only",
            "inconsistent": "stored gain disagrees with the current grading evidence; retained as historical only",
        }
        lines.append(f"- validation status   : `{validation_status}` ({explanations[validation_status]})")
    return lines


def _format_throughput_section(summary: dict[str, Any]) -> list[str]:
    """The primary metric of the framework this session ran, baseline against current best.

    Serving frameworks report throughput (tok/s); scriptable xDiT reports per-image latency (e2el_mean_ms).
    """
    from hyperloom.inference_optimizer import framework_registry

    framework = summary.get("framework")
    cb = summary.get("current_best") or {}
    cb_tput = cb.get("tput") if isinstance(cb, dict) else None
    lines = [
        "## Throughput",
        "",
        f"- baseline            : `{framework_registry.format_primary_metric(framework, summary['baseline_tput'])}`",
    ]
    if cb_tput is not None:
        lines.append(
            f"- current_best        : `{framework_registry.format_primary_metric(framework, cb_tput)}` "
            f"(action=`{cb.get('action', '?')}`)"
        )
    lines.extend(_format_cumulative_gain_lines(summary))
    if cb.get("ttft_mean_ms") is not None:
        lines.append(f"- ttft_mean      : `{cb.get('ttft_mean_ms'):.1f}` ms")
    if cb.get("e2el_mean_ms") is not None:
        lines.append(f"- e2el_mean      : `{cb.get('e2el_mean_ms'):.1f}` ms")
    return lines


def _format_platform_lines(plat: dict[str, Any]) -> list[str]:
    """The host these numbers were measured on, down to the CPU, accelerators and software stack."""
    if plat.get("status") != "ok":
        # Always emit the line, with the reason when there is one: a silent absence is indistinguishable from a host
        # that was never checked, and the no-reason case is the one that needs saying -- it is a summary written
        # before this field existed, not a probe that failed.
        why = str(plat.get("reason") or "").strip()
        return [f"- platform       : not recorded{f' — {why}' if why else ''}"]

    def _q(value: Any) -> str:
        """Render an unresolved field as '?' rather than the word None."""
        return "?" if value in (None, "") else str(value)

    if plat.get("multi_node_session"):
        lines = [
            f"- platform scope : ⚠ multi-node session — sampled on `{plat.get('host')}` "
            f"(the orchestrator), which may not be the benchmark node"
        ]
    else:
        lines = [f"- host           : {plat.get('host', 'unknown')}"]
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
    return lines


def _format_run_summary_section(summary: dict[str, Any]) -> list[str]:
    """What the session cost in crashes and pruned families, and the host it all ran on."""
    lines = ["## Run summary", "", f"- crash_count    : {summary['crash_count']}"]
    boot_fail = summary.get("server_boot_failures") or 0
    if boot_fail:
        lines.append(f"- server_boot_failures : {boot_fail}  *(warmup_failed variants; excluded from crash_count)*")
    lines.append(f"- pruned_families: {summary['pruned_families'] or '(none)'}")
    lines.extend(_format_platform_lines(summary.get("platform") or {}))
    lines.append("")
    return lines


def _format_event_counts_section(summary: dict[str, Any]) -> list[str]:
    """How many events each bus topic carried, busiest first."""
    lines = ["## Event counts", ""]
    counts = summary.get("event_counts_by_topic")
    if not counts:
        lines.append("- (no events recorded)")
    else:
        for topic, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{topic}`: {n}")
    lines.append("")
    return lines


def _format_highlights_section(summary: dict[str, Any]) -> list[str]:
    """The highlight events the session published, capped at the first fifty."""
    lines = ["## Highlights", ""]
    if not summary["highlights"]:
        lines.append("(no highlight events captured)")
    else:
        for h in summary["highlights"][:50]:
            lines.append(f"- `{h.get('topic', '?')}` from `{h.get('from_agent', '?')}`: {h.get('summary', '')}")
    lines.append("")
    return lines


def _format_md(summary: dict[str, Any]) -> str:
    """Render the human-readable Markdown report from a summary dict."""
    lines = [
        *_format_report_header(summary),
        *_format_throughput_section(summary),
        *_format_composite_perf_section(summary),
        "",
        *_format_completeness_annotations(summary),
        *_format_run_summary_section(summary),
        *_format_event_counts_section(summary),
        *_format_highlights_section(summary),
        *_format_degraded_mode_section(summary),
        *_format_compute_partition_section(summary),
    ]
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
    from hyperloom.inference_optimizer.roofline_snapshot import format_roofline_metrics_table

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
            "`KernelPhase.maybe_enqueue_watermark_roofline`). The "
            "watermark did not cross during this session, so the "
            "PRELUDE bootstrap snapshot is the only datapoint available "
            "for the report._"
        )
        lines.append("")
        lines.append(
            "_The **Theoretical peak** below is the decode "
            "memory-roofline ceiling derived from the GPU's HBM "
            "bandwidth and the model's weight + KV-cache traffic per "
            "token (see `roofline_ceiling.compute_roofline_breakdown_from_state`). "
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
        "(see `KernelPhase.maybe_enqueue_watermark_roofline`)."
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
        "token (see `roofline_ceiling.compute_roofline_breakdown_from_state`). "
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


_EXTERNAL_BASELINE_ADVISORY = (
    "> Advisory only. This block does not feed Objective, scoring, or any "
    "agent prompt; it is shown here purely for post-mortem comparison."
)


def _format_unrequested_external_baseline(ext: dict[str, Any]) -> list[str]:
    """Say that no competitor target was asked for, and how to ask for one."""
    return [
        "## External baseline (not requested)",
        "",
        "- No `--compare-against-gpu` was specified; only a marker JSON "
        "was written. Re-run with `--compare-against-gpu <gpu>` (e.g. "
        "`b300` / `mi355x` / `h200`) to fetch the matching InferenceX "
        "reference data point.",
        f"- Fetched at: {ext.get('fetched_at') or '(unknown)'}",
        f"- Status: `{str(ext.get('status') or 'unknown')}` "
        f"reason=`{str(ext.get('reason') or '').strip()}` (rows matched: {ext.get('row_count', 0)})",
        "",
        _EXTERNAL_BASELINE_ADVISORY,
        "",
    ]


def _format_external_baseline_query(ext: dict[str, Any]) -> list[str]:
    """The heading, the data point that was asked for, and how the fetch went."""
    q = ext.get("query") or {}
    reason = str(ext.get("reason") or "").strip()
    reason_suffix = f" reason=`{reason}`" if reason else ""
    lines = [
        "## External baseline (competitor target, advisory)",
        "",
        "- Query: "
        f"model=`{q.get('model') or '(unset)'}`  "
        f"gpu=`{q.get('gpu') or '(unset)'}`  "
        f"framework=`{q.get('framework') or '(any)'}`  "
        f"precision=`{q.get('precision') or '(any)'}`  "
        f"ISL/OSL=`{q.get('isl') or '(any)'}/{q.get('osl') or '(any)'}`",
        f"- Fetched at: {ext.get('fetched_at') or '(unknown)'}",
        f"- Status: `{str(ext.get('status') or 'unknown')}`{reason_suffix} (rows matched: {ext.get('row_count', 0)})",
    ]
    warning = ext.get("warning") or ""
    if warning:
        lines.append(f"- Warning: {warning}")
    return lines


def _format_external_reference_best(ext: dict[str, Any]) -> list[str]:
    """The reference data point the fetch matched, or the note that there is none."""
    best = ext.get("best")
    if str(ext.get("status") or "unknown") != "ok" or not isinstance(best, dict):
        return ["", "- No reference best available — orchestrator was not affected by this section."]
    lines = [
        "",
        f"- Reference best per-GPU throughput: "
        f"**{float(best.get('tput_per_gpu', 0.0)):.1f}** tok/s/GPU "
        f"at concurrency {best.get('conc')}, decode TP {best.get('decode_tp')}",
    ]
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
    return lines


def _format_external_baseline_section(ext: dict[str, Any]) -> list[str]:
    """Render the advisory external-baseline section (report-only)."""
    if str(ext.get("reason") or "").strip() == "no_target_gpu_configured":
        return _format_unrequested_external_baseline(ext)
    lines = _format_external_baseline_query(ext)
    comparison = ext.get("comparison")
    if isinstance(comparison, dict) and comparison.get("benchmark_mode") == "agentx":
        from hyperloom.inference_optimizer.baseline_comparison.research_hints import full_gap_summary

        return [*lines, "", full_gap_summary(comparison), ""]
    return [*lines, *_format_external_reference_best(ext), "", _EXTERNAL_BASELINE_ADVISORY, ""]


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
