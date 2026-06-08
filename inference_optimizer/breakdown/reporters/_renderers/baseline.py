# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Baseline measurement renderer."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, md_kv_list, md_table, register_renderer
from ._invocation import render_invocation_block


@register_renderer("baseline")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the baseline-measurement section.

    Surfaces baseline throughput, accuracy and latency, the attempts
    history, and the launch invocation block, emitting data-quality
    warnings (e.g. missing throughput or TTFT). Skipped when neither
    throughput nor attempts were recorded.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered baseline section.
    """
    b = breakdown.get("baseline") or {}
    session = breakdown.get("session") or {}
    tput = b.get("throughput_tok_s_per_gpu")
    acc = b.get("accuracy")
    ttft = b.get("ttft_mean_ms")
    e2el = b.get("e2el_mean_ms")
    ttft_source = str(b.get("ttft_e2el_source") or "")
    fail_streak = int(b.get("failure_streak") or 0)
    attempts = b.get("attempts_history") or []

    facts: list[str] = []
    warnings: list[str] = []
    decisions: list[Decision] = []

    if tput:
        facts.append(f"Baseline throughput: {float(tput):.2f} tok/s/gpu.")
        decisions.append(Decision(
            kind="attempted",
            subject="baseline",
            metric_pct=None,
            rationale=f"baseline_tput={float(tput):.2f}",
        ))
    else:
        warnings.append("No baseline_tput recorded — every subsequent gain is uncomputable.")
        decisions.append(Decision(
            kind="not_attempted",
            subject="baseline",
            rationale="no throughput captured",
        ))
    if acc:
        facts.append(f"Baseline accuracy: {float(acc):.4g}.")
    if ttft is not None:
        facts.append(f"Baseline TTFT mean: {float(ttft):.1f} ms.")
    else:
        warnings.append(
            "ttft_mean_ms is null — baseline benchmark_report.json was unreachable "
            "from the exporter's vantage point (often: ``last_baseline.workspace`` "
            "is a container path that no longer resolves on wekafs)."
        )
    if e2el is not None:
        facts.append(f"Baseline e2el mean: {float(e2el):.1f} ms.")
    if fail_streak:
        warnings.append(f"baseline_failure_streak={fail_streak} — baseline retried after failure(s).")
    if attempts:
        facts.append(f"Baseline attempts recorded: {len(attempts)}.")

    # Annotate the ttft row inline when it was reconstructed via the
    # runs/baseline/ disk walk fallback — readers should see at a
    # glance that the latency didn't come from state.last_baseline.
    ttft_display: Any = ttft
    if ttft is not None and ttft_source == "runs_baseline_disk":
        ttft_display = f"{float(ttft):.1f} (reconstructed from runs/baseline/ disk walk)"
    md_parts: list[str] = []
    md_parts.append(md_kv_list([
        ("throughput_tok_s_per_gpu", tput),
        ("accuracy",                 acc),
        ("ttft_mean_ms",             ttft_display),
        ("e2el_mean_ms",             e2el),
        ("ttft_e2el_source",         ttft_source or None),
        ("config_path",              b.get("config_path")),
        ("benchmark_report_path",    b.get("benchmark_report_path")),
        ("failure_streak",           fail_streak or None),
    ]))
    if attempts:
        rows = [
            [a.get("ts"), a.get("status"), a.get("decision"),
             a.get("key_metric"), a.get("error_class")]
            for a in attempts[:10]
        ]
        md_parts.append("")
        md_parts.append("**Baseline attempts** (last 10):")
        md_parts.append(md_table(
            ["ts", "status", "decision", "key_metric", "error_class"], rows,
        ))

    inv_md = render_invocation_block(b.get("invocation"), session.get("image"))
    if inv_md:
        md_parts.append("")
        md_parts.append(inv_md)

    return RenderedSection(
        section_id="baseline",
        title="Baseline",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        decisions=decisions,
        warnings=warnings,
        skipped=not (tput or attempts),
    )
