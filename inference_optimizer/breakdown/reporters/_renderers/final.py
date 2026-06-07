# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Final / validated result renderer."""

from __future__ import annotations

from typing import Any

from ..base import (
    Decision,
    RenderedSection,
    fmt_pct,
    md_kv_list,
    register_renderer,
)
from ._invocation import render_invocation_block


@register_renderer("final")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    f = breakdown.get("final") or {}
    b = breakdown.get("baseline") or {}
    session = breakdown.get("session") or {}
    final_tput = f.get("throughput_tok_s_per_gpu")
    base_tput = b.get("throughput_tok_s_per_gpu")
    gain_v = f.get("cumulative_gain_pct_validated")
    gain_round = f.get("cumulative_gain_pct_per_round_sum")
    val_stack_len = f.get("validated_at_stack_len")
    val_ts = f.get("validated_ts")
    stack_changed = bool(f.get("stack_changed_after_validation"))
    extra_args = f.get("extra_server_args") or ""
    action_path = f.get("action_path") or []

    facts: list[str] = []
    warnings: list[str] = []
    decisions: list[Decision] = []

    if final_tput:
        facts.append(f"Final throughput: {float(final_tput):.2f} tok/s/gpu.")
    if base_tput and final_tput:
        delta = float(final_tput) - float(base_tput)
        facts.append(f"Delta vs baseline: {delta:+.2f} tok/s/gpu.")
    if gain_v is not None:
        facts.append(f"Validated cumulative gain: {fmt_pct(gain_v, plus=True)}.")
        decisions.append(Decision(
            kind="kept" if (gain_v or 0) > 0 else "attempted",
            subject="final",
            metric_pct=float(gain_v),
            rationale=f"validated at stack_len={val_stack_len} ts={val_ts}",
        ))
    if gain_round is not None:
        facts.append(
            f"Per-round summed gain: {fmt_pct(gain_round)} "
            "(non-additive, do not present as the user-visible number)."
        )
    if action_path:
        facts.append(
            "Final stack: " + " → ".join(f"`{p}`" for p in action_path)
        )
    if extra_args:
        facts.append(f"Final extra_server_args: `{extra_args}`.")
    if stack_changed:
        warnings.append(
            "stack_changed_after_validation = true — optimization_stack grew "
            "after the last successful validate_stack run, so the reported "
            "cumulative_gain_pct_validated may be stale relative to the "
            "current best."
        )
    if gain_v is None and (final_tput or base_tput):
        warnings.append(
            "cumulative_gain_pct_validated is null while baseline/final "
            "throughput are set — validate_stack never ran or the snapshot "
            "predates V2."
        )

    md_kv = md_kv_list([
        ("final_throughput_tok_s_per_gpu", final_tput),
        ("cumulative_gain_pct_validated",  gain_v),
        ("cumulative_gain_pct_per_round_sum", gain_round),
        ("validated_at_stack_len",         val_stack_len),
        ("validated_ts",                   val_ts),
        ("stack_changed_after_validation", stack_changed),
        ("extra_server_args",              extra_args or None),
        ("action_path",                    action_path or None),
        ("ttft_mean_ms",                   f.get("ttft_mean_ms")),
        ("e2el_mean_ms",                   f.get("e2el_mean_ms")),
        ("ttft_e2el_source",               f.get("ttft_e2el_source") or None),
        ("extra_envs",                     f.get("extra_envs") or None),
    ])

    md_parts = [md_kv]
    inv_md = render_invocation_block(f.get("invocation"), session.get("image"))
    if inv_md:
        md_parts.append("")
        md_parts.append(inv_md)

    return RenderedSection(
        section_id="final",
        title="Final Result",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        decisions=decisions,
        warnings=warnings,
        skipped=not (final_tput or gain_v),
    )
