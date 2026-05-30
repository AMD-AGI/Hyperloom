"""Session report renderers.

Takes a session_breakdown.json dict and produces a human-readable markdown
report with grouped sections.

Entry point: :func:`render_session_report`
"""

from __future__ import annotations

from typing import Any


SECTION_GROUPS: list[tuple[str, list[str]]] = [
    ("Session & Workload", ["session", "workload"]),
    ("Performance Results", ["baseline", "final", "attribution"]),
    ("Agent Dispatch", ["agent_timeline", "capability_summary"]),
    ("Kernel Optimization", ["kernel_lifecycle", "profiling",
                             "geak_invocations", "oob_invocations"]),
    ("Benchmarking", ["sweep"]),
    ("Monitoring", ["watchdog_events", "telemetry"]),
    ("Source Artifacts", ["source_files"]),
]


def render_session_report(breakdown: dict[str, Any]) -> str:
    """Render a full session report from breakdown JSON.

    Returns a markdown string with grouped sections.
    """
    lines: list[str] = []
    warnings = breakdown.get("warnings", [])

    lines.append("# Hyperloom Session Report")
    lines.append("")
    lines.append(_render_executive_summary(breakdown))
    lines.append("")

    for group_title, section_ids in SECTION_GROUPS:
        section_lines = []
        for sid in section_ids:
            renderer = _RENDERERS.get(sid)
            if renderer:
                content = renderer(breakdown.get(sid, {}), breakdown)
                if content:
                    section_lines.append(content)

        if section_lines:
            lines.append(f"## {group_title}")
            lines.append("")
            lines.extend(section_lines)
            lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _render_executive_summary(breakdown: dict[str, Any]) -> str:
    """Render a 3-5 line executive summary."""
    session = breakdown.get("session", {})
    baseline = breakdown.get("baseline", {})
    final = breakdown.get("final", {})
    workload = breakdown.get("workload", {})

    baseline_tput = baseline.get("throughput_tok_s", 0)
    final_tput = final.get("throughput_tok_s")
    gain = final.get("cumulative_gain_pct", 0)
    elapsed = session.get("elapsed_minutes", 0)
    agent_count = session.get("agent_count", 0)
    stop = session.get("stop_reason", "unknown")

    parts = [
        f"**Model:** {workload.get('model_path', 'unknown')}",
        f"**Framework:** {workload.get('framework', 'unknown')}",
        f"**GPU:** {workload.get('gpu_type', 'unknown')}",
        f"**Baseline:** {baseline_tput:.1f} tok/s" if baseline_tput else "**Baseline:** N/A",
        f"**Final:** {final_tput:.1f} tok/s (+{gain:.1f}%)" if final_tput else f"**Gain:** +{gain:.1f}%",
        f"**Runtime:** {elapsed:.0f} min | **Agents:** {agent_count} | **Stop:** {stop}",
    ]
    return " | ".join(parts[:3]) + "\n" + " | ".join(parts[3:])


def _render_session(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Session Metadata",
        "",
        f"- **ID:** `{data.get('session_id', '-')}`",
        f"- **Started:** {data.get('created_at_utc', '-')}",
        f"- **Ended:** {data.get('ended_at_utc', '-')}",
        f"- **Duration:** {data.get('elapsed_minutes', 0):.1f} min",
        f"- **Stop reason:** {data.get('stop_reason', '-')}",
        f"- **Agents dispatched:** {data.get('agent_count', 0)}",
        "",
    ]
    return "\n".join(lines)


def _render_workload(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    obj = data.get("objective", {})
    lines = [
        "### Workload Configuration",
        "",
        f"- **Framework:** {data.get('framework', '-')}",
        f"- **Model:** `{data.get('model_path', '-')}`",
        f"- **GPU type:** {data.get('gpu_type', '-')}",
        f"- **TP:** {data.get('tp', 'auto')}",
        f"- **Target:** {obj.get('kind', '-')} = {obj.get('value', '-')}",
        "",
    ]
    return "\n".join(lines)


def _render_baseline(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Baseline",
        "",
        f"- **Throughput:** {data.get('throughput_tok_s', 0):.1f} tok/s",
    ]
    if data.get("latency_mean_ms"):
        lines.append(f"- **Latency (mean):** {data['latency_mean_ms']:.1f} ms")
    if data.get("accuracy"):
        lines.append(f"- **Accuracy:** {data['accuracy']:.4f}")
    lines.append("")
    return "\n".join(lines)


def _render_final(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    tput = data.get("throughput_tok_s")
    lines = [
        "### Final Results",
        "",
        f"- **Throughput:** {tput:.1f} tok/s" if tput else "- **Throughput:** N/A",
        f"- **Cumulative gain:** +{data.get('cumulative_gain_pct', 0):.1f}%",
        f"- **Validated:** {'Yes' if data.get('validated') else 'No'}",
    ]
    action_path = data.get("action_path", [])
    if action_path:
        lines.append(f"- **Optimization stack:** {' -> '.join(action_path[:10])}")
    lines.append("")
    return "\n".join(lines)


def _render_agent_timeline(data: Any, _bd: dict) -> str:
    if not data or not isinstance(data, list):
        return ""
    lines = [
        "### Agent Dispatch Timeline",
        "",
        f"| Agent | Role | Status | Runtime | GPU |",
        f"|-------|------|--------|---------|-----|",
    ]
    for event in data[:30]:
        gpu = ",".join(str(g) for g in event.get("gpu_ids", [])) or "-"
        lines.append(
            f"| `{event.get('agent_id', '-')[:20]}` "
            f"| {event.get('role', '-')} "
            f"| {event.get('status', '-')} "
            f"| {event.get('runtime_s', 0):.0f}s "
            f"| {gpu} |"
        )
    if len(data) > 30:
        lines.append(f"| ... | ({len(data) - 30} more agents) | | | |")
    lines.append("")
    return "\n".join(lines)


def _render_capability_summary(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Capability Summary",
        "",
        "| Capability | Status | Attempts | Keeps |",
        "|------------|--------|----------|-------|",
    ]
    for name in ("geak", "oob", "tracelens", "magpie", "specialist"):
        cap = data.get(name, {})
        lines.append(
            f"| {name.upper()} | {cap.get('status', '-')} "
            f"| {cap.get('attempts', 0)} | {cap.get('keeps', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_kernel_lifecycle(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    detected = data.get("detected", [])
    optimized = data.get("optimized", [])
    adopted = data.get("adopted", [])
    rejected = data.get("rejected", [])

    lines = [
        "### Kernel Lifecycle",
        "",
        f"- **Detected:** {len(detected)}",
        f"- **Optimized:** {len(optimized)}",
        f"- **Adopted:** {len(adopted)}",
        f"- **Rejected:** {len(rejected)}",
        "",
    ]

    if adopted:
        lines.append("**Adopted kernels:**")
        lines.append("")
        for k in adopted:
            gain = k.get("e2e_gain_pct")
            gain_s = f"+{gain:.1f}%" if gain else "N/A"
            lines.append(f"- `{k.get('kernel_id', '-')}`: {gain_s}")
        lines.append("")

    return "\n".join(lines)


def _render_profiling(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Profiling Summary",
        "",
        f"- **Total GPU time:** {data.get('total_gpu_time_us', 0) / 1000:.1f} ms",
        f"- **Hot kernels (>5%):** {data.get('hot_kernel_count', 0)}",
    ]
    top = data.get("top_kernels", [])
    if top:
        lines.append("")
        lines.append("| Kernel | % GPU | Count |")
        lines.append("|--------|-------|-------|")
        for k in top[:10]:
            pct = k.get("percentage", 0) * 100
            lines.append(f"| `{k.get('name', '-')[:40]}` | {pct:.1f}% | {k.get('count', 0)} |")
    lines.append("")
    return "\n".join(lines)


def _render_invocations(data: Any, bd: dict) -> str:
    """Render GEAK + OOB invocations combined."""
    geak = bd.get("geak_invocations", [])
    oob = bd.get("oob_invocations", [])
    if not geak and not oob:
        return ""

    lines = ["### Kernel Optimization Invocations", ""]
    if geak:
        lines.append(f"**GEAK invocations:** {len(geak)}")
        lines.append("")
        for inv in geak[:10]:
            decision = inv.get("decision", "-")
            speedup = inv.get("micro_speedup")
            speedup_s = f"{speedup:.2f}x" if speedup else "-"
            lines.append(f"- `{inv.get('kernel_id', '-')}` [{inv.get('backend', '-')}]: "
                        f"{decision} (speedup: {speedup_s})")
        lines.append("")

    if oob:
        lines.append(f"**OOB invocations:** {len(oob)}")
        lines.append("")
        for inv in oob[:10]:
            decision = inv.get("decision", "-")
            lines.append(f"- `{inv.get('kernel_id', '-')}`: {decision}")
        lines.append("")

    return "\n".join(lines)


def _render_sweep(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Benchmark Sweep",
        "",
        f"- **Grid size:** {data.get('grid_size', 0)}",
    ]
    best = data.get("best_overall", {})
    if best:
        lines.append(f"- **Best variant:** {best.get('variant_name', '-')} "
                    f"@ {best.get('throughput_tok_s', 0):.1f} tok/s")
    lines.append("")
    return "\n".join(lines)


def _render_watchdog_events(data: Any, _bd: dict) -> str:
    if not data or not isinstance(data, list):
        return ""
    lines = [
        "### Watchdog Events",
        "",
        f"Total events: {len(data)}",
        "",
    ]
    critical = [e for e in data if e.get("severity") in ("error", "critical")]
    if critical:
        lines.append("**Critical/Error events:**")
        lines.append("")
        for ev in critical[:10]:
            lines.append(f"- [{ev.get('ts', '-')}] {ev.get('category', '')}: {ev.get('summary', '')}")
        lines.append("")
    return "\n".join(lines)


def _render_attribution(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    entries = data.get("gain_per_entry", [])
    if not entries:
        return ""

    lines = [
        "### Gain Attribution",
        "",
        f"- **Method:** {data.get('method', '-')}",
        "",
        "| Action | Delta | Cumulative |",
        "|--------|-------|------------|",
    ]
    for entry in entries[:20]:
        lines.append(
            f"| {entry.get('action', '-')} "
            f"| +{entry.get('delta_pct', 0):.1f}% "
            f"| {entry.get('cum_gain_after', 0):.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_telemetry(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Telemetry",
        "",
        f"- **GPU count:** {data.get('gpu_count', '-')}",
        f"- **GPU type:** {data.get('gpu_type', '-')}",
        f"- **Traces:** {len(data.get('trace_paths', []))}",
        "",
    ]
    return "\n".join(lines)


def _render_source_files(data: dict[str, Any], _bd: dict) -> str:
    if not data:
        return ""
    lines = [
        "### Source Files",
        "",
        f"- **Session dir:** `{data.get('session_dir', '-')}`",
        f"- **Agent manifests:** {len(data.get('manifests', []))}",
        f"- **Agent logs:** {len(data.get('agent_logs', []))}",
        f"- **Patches:** {len(data.get('patches', []))}",
        f"- **Benchmark reports:** {len(data.get('benchmark_reports', []))}",
        "",
    ]
    return "\n".join(lines)


_RENDERERS: dict[str, Any] = {
    "session": _render_session,
    "workload": _render_workload,
    "baseline": _render_baseline,
    "final": _render_final,
    "agent_timeline": _render_agent_timeline,
    "capability_summary": _render_capability_summary,
    "kernel_lifecycle": _render_kernel_lifecycle,
    "profiling": _render_profiling,
    "geak_invocations": _render_invocations,
    "oob_invocations": lambda data, bd: "",  # handled by geak_invocations renderer
    "sweep": _render_sweep,
    "watchdog_events": _render_watchdog_events,
    "attribution": _render_attribution,
    "telemetry": _render_telemetry,
    "source_files": _render_source_files,
}


__all__ = ["render_session_report", "SECTION_GROUPS"]
