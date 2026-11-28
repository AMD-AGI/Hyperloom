# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist prompt sections on the dispatch's situation: hardware, budget, gap, KB context, recipe, lessons, pitfalls."""

from __future__ import annotations

import json
from typing import Any

from hyperloom.common.prompt_safety import defang_prompt_structure
from .agentx_context import corpus_lines, grading_lines
from .specialist_focus import _is_agentx
from ..specialists.prompt_inputs import SpecialistPromptInputs


_NONE_PLACEHOLDER = "(none)"


# Section 2 — Hardware context
def _section_hardware(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 2 (hardware + workload context) of the prompt.

    Emits GPU type, TP, HBM, peak TFLOPs, and any populated workload
    fields (precision, concurrency, ISL/OSL, max_model_len, arch notes).

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the hardware-context section.
    """
    rows: list[str] = ["## 2. HARDWARE CONTEXT", ""]
    if inp.gpu_type:
        rows.append(f"- gpu_type: {inp.gpu_type}")
    else:
        rows.append(f"- gpu_type: {_NONE_PLACEHOLDER}")
    if inp.allocated_gpu_ids:
        rows.append("- allocated specialist GPU ids: " + ", ".join(str(g) for g in inp.allocated_gpu_ids))
    if inp.tp > 0:
        rows.append(f"- TP: {inp.tp}")
    else:
        rows.append(f"- TP: {_NONE_PLACEHOLDER}")
    # Workload context.
    workload_rows: list[str] = []
    if inp.precision:
        workload_rows.append(f"- precision: {inp.precision}")
    if inp.conc > 0:
        workload_rows.append(f"- concurrency: {inp.conc}")
    if _is_agentx(inp):
        # The corpus fixes the request shape, so ISL/OSL carry no information.
        workload_rows += corpus_lines(inp.agentx_corpus_shape)
        workload_rows += grading_lines(inp.agentx_grading, inp.agentx_backend)
    else:
        if inp.isl > 0:
            workload_rows.append(f"- ISL (input seq len): {inp.isl}")
        if inp.osl > 0:
            workload_rows.append(f"- OSL (output seq len): {inp.osl}")
    if inp.max_model_len > 0:
        workload_rows.append(f"- max_model_len: {inp.max_model_len}")
    if workload_rows:
        rows.append("")
        rows.append("Workload:")
        rows.extend(workload_rows)
    if inp.arch_notes:
        rows.append("")
        rows.append(f"Model architecture (advisory): {inp.arch_notes}")
    if inp.target_gap_notes:
        rows.append("")
        rows.append(inp.target_gap_notes)
    return rows


# Section 2a — Execution budget (wall-clock)
def _section_execution_budget(inp: SpecialistPromptInputs) -> list[str]:
    """Render the wall-clock budget block so the specialist can self-throttle.

    Renders the time left on the dispatch deadline (seconds + minutes) and the
    dispatch start timestamp. Returns ``[]`` when no budget was supplied, so the
    section is omitted entirely rather than emitting a placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``wall_budget_sec`` /
            ``started_at_iso``).

    Returns:
        The rendered execution-budget section lines, or ``[]`` when no budget
        is set.
    """
    if inp.wall_budget_sec <= 0:
        return []
    mins = inp.wall_budget_sec / 60.0
    rows = [
        "## 2a. EXECUTION BUDGET (wall-clock)",
        "",
        f"- Hard wall-clock budget for this entire dispatch: **{inp.wall_budget_sec:.0f}s (~{mins:.0f} min)**.",
    ]
    if inp.started_at_iso:
        rows.append(f"- Dispatch started at: {inp.started_at_iso} (UTC).")
    rows.extend(
        [
            "- The Coordinator hard-kills your subprocess when this budget is "
            + "exhausted — turns are NOT the stop signal. Scope your work to "
            + "reach a deliverable conclusion inside the budget.",
            "- Self-throttle: check elapsed wall-clock with Bash "
            + "(``date -u +%s`` vs the start above), keep your "
            + "``specialist_done.partial.json`` checkpoint current, and write "
            + "the final ``specialist_done.json`` before the budget runs out so "
            + "your best work is never lost to a kill.",
        ]
    )
    return rows


# Section 3 — Gap statement
def _section_gap(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 3 (gap statement) of the specialist prompt.

    Emits the canonical gap id, layer, symptom, and most-recent evidence
    JSON, or a ``(none)`` placeholder when no gap is set.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the gap-statement section.
    """
    rows = ["## 3. GAP STATEMENT", ""]
    if not inp.gap_canonical_id:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append(f"- gap_canonical_id: `{inp.gap_canonical_id}`")
    if inp.gap_layer:
        rows.append(f"- layer: {inp.gap_layer}")
    if inp.gap_symptom:
        rows.append(f"- symptom: {inp.gap_symptom}")
    if inp.gap_evidence:
        rows.append("")
        rows.append("Most recent evidence:")
        rows.append("```json")
        rows.append(json.dumps(inp.gap_evidence, sort_keys=True, indent=2))
        rows.append("```")
    return rows


# Section 4 — optional KB context
def _section_kb_context(inp: SpecialistPromptInputs) -> list[str]:
    """Build the advisory KB-context section of the specialist prompt.

    Renders the research hints, or the none placeholder.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines for the KB-context section.
    """
    rows = ["## 4. KB CONTEXT (optional, advisory)", ""]
    if inp.research_hints:
        rows.extend(
            [
                "Structured KB context is empty for this (model, hardware, domain), but "
                + "the research scout collected source-backed priors this "
                + "session. Treat these as your advisory prior (co-equal with "
                + "RecipeKB priors; the benchmark still decides the final answer):",
                "",
                inp.research_hints,
                "",
                "Anchor proposals on these hints where they fit the gap " + "(Section 3) and hardware (Section 2).",
            ]
        )
        return rows
    rows.extend(
        [
            _NONE_PLACEHOLDER,
            "",
            "(No structured KB context supplied. Work from the gap evidence "
            + "(Section 3), the recipe (Section 5), the PR tools (Section 6) "
            + "and source inspection; record missing RecipeKB / "
            + "research / PR questions in ``residual_questions`` so a "
            + "future round can warm richer advisory context.)",
        ]
    )
    return rows


def _section_experience_kb(inp: SpecialistPromptInputs) -> list[str]:
    """Render the Experience KB section; omitted when this dispatch rendered no Experience.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines carrying the Experience service block, or ``[]``.
    """
    if not inp.experience_kb_block:
        return []
    return [
        "## 4b. EXPERIENCE KB (measured outcomes from earlier sessions)",
        "",
        "Compare each Experience's identity and baseline configuration with Sections 2 and 3 "
        + "before relying on it. When one shaped a proposal, cite it in that proposal's "
        + "``experience_citations``, or for a patch you wrote in the payload's top-level "
        + "``experience_citations``: ``{id, stance, claim}`` with ``stance`` one of ``adopt`` "
        + "(you did its change), ``adapt`` (you did it modified), ``avoid`` (you left it out "
        + "because of its outcome), or ``contrast`` (you chose a different change designed "
        + "against it), and ``claim`` one sentence on why. Only ids shown below are kept.",
        "",
        inp.experience_kb_block,
    ]


def _section_runtime_findings(inp: SpecialistPromptInputs) -> list[str]:
    """Render the scanned runtime findings; omitted when none were injected."""
    if not inp.runtime_findings:
        return []
    return [
        "## 4c. RUNTIME FINDINGS (current best server.log, already scanned)",
        "",
        "Do not re-read the server log for these. A disabled or falling-back hot "
        "path is restored first; do not optimize the fallback implementation.",
        "",
        "```text",
        inp.runtime_findings,
        "```",
    ]


# Section 5 — Recipe summary
def _section_recipe(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 5 (warm-start recipe summary) of the prompt.

    Dumps the T0 warm-start recipe as JSON, or a ``(none)`` placeholder
    when no warm-start recipe was supplied.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the recipe-summary section.
    """
    rows = ["## 5. WARM-START RECIPE SUMMARY", ""]
    if not inp.warm_start_recipe:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append("**find-recipe result:**")
    rows.append("```json")
    rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, separators=(",", ":")))
    rows.append("```")
    return rows


# Section 5b — Related lessons (positive priors from prior KEEPs)
def _section_lessons(inp: SpecialistPromptInputs) -> list[str]:
    """Render the recipe row's ``lessons`` from prior KEEPs, compactly
    (statement + measured_impact).

    Rows are the flat shape ``Recipe.to_dict`` writes —
    ``{statement, measured_impact, ...}`` — normalised by ``recipe_kb_t0``
    before they land on ``warm_start_lessons``.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_lessons``).

    Returns:
        The rendered related-lessons section lines.
    """
    rows = ["## 5b. RELATED LESSONS (prior KEEPs on this model+hw)", ""]
    for point in inp.warm_start_lessons:
        line = _kb_point_line(inp, point, text_key="statement", count_label="validated")
        if not line:
            continue
        rows.append(line)
        impact_str = _render_measured_impact(point.get("measured_impact")) if isinstance(point, dict) else ""
        if impact_str:
            rows.append(f"    impact: {impact_str}")
    if len(rows) == 2:  # only the header + blank line: no lessons, or all filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


def _kb_point_line(
    inp: SpecialistPromptInputs,
    point: Any,
    *,
    text_key: str,
    count_label: str,
    lead_meta_key: str = "",
) -> str:
    """Render one RecipeKB lesson / pitfall row as a prompt bullet.

    Args:
        inp: The specialist prompt inputs (reads the framework version).
        point: The row: a ``{text_key, confidence, validated_count, ...}``
            dict, or a bare string.
        text_key: The dict key holding the row's text.
        count_label: The label ``validated_count`` renders under.
        lead_meta_key: A dict key rendered first in the metadata, if any.

    Returns:
        The bullet line, or "" when the row carries no text.
    """
    # External warm-start data may arrive as a plain string rather than a
    # dict "point"; render the bare text to tolerate shape drift.
    if isinstance(point, str):
        text = point.strip()
        return f"- **{defang_prompt_structure(text)}**" if text else ""
    if not isinstance(point, dict):
        return ""
    text = str(point.get(text_key) or "").strip()
    if not text:
        return ""
    meta_bits: list[str] = []
    lead = str(point.get(lead_meta_key) or "").strip() if lead_meta_key else ""
    if lead:
        meta_bits.append(f"{lead_meta_key}={lead}")
    conf = point.get("confidence")
    if isinstance(conf, (int, float)) and conf > 0:
        meta_bits.append(f"conf={float(conf):.2f}")
    # validated_count is the strongest cross-session signal; fall back to source_session_id.
    vc = point.get("validated_count")
    if isinstance(vc, int) and vc > 1:
        meta_bits.append(f"{count_label}={vc}")
    recent_ids = point.get("source_session_ids")
    if isinstance(recent_ids, list) and recent_ids:
        meta_bits.append(f"recent={recent_ids[-1]}")
    else:
        src_sid = str(point.get("source_session_id") or "").strip()
        if src_sid:
            meta_bits.append(f"src={src_sid}")
    meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
    # Version-mismatch annotation; the LLM gets the final call.
    return f"- **{defang_prompt_structure(text)}**{meta}{_format_version_note(inp, point)}"


def _format_version_note(
    inp: SpecialistPromptInputs,
    row: dict[str, Any],
) -> str:
    """Render a ``[from sglang@X.Y, you're on A.B]`` annotation when
    the row's framework_version differs; empty when either side is
    unknown or they match.

    Args:
        inp: The specialist prompt inputs (reads ``framework`` /
            ``framework_version``).
        row: The lesson or pitfall row (reads ``framework_version``).

    Returns:
        The version-mismatch annotation, or "" when unknown or matching.
    """
    lesson_fv = str(row.get("framework_version") or "").strip()
    current_fv = (inp.framework_version or "").strip()
    if not lesson_fv or not current_fv:
        return ""
    if lesson_fv == current_fv:
        return ""
    framework_label = (inp.framework or "framework").strip() or "framework"
    return f" [from {framework_label}@{lesson_fv}, you're on {current_fv}]"


def _render_measured_impact(raw: Any) -> str:
    """Back-compat renderer for a row's ``measured_impact`` (dict, legacy
    string, or other).

    Args:
        raw: The ``measured_impact`` value (dict, string, None, or other).

    Returns:
        A compact human-readable impact string ("" when ``raw`` is None).
    """
    if isinstance(raw, dict):
        parts: list[str] = []
        gain = raw.get("gain_pct")
        if isinstance(gain, (int, float)):
            parts.append(f"+{float(gain):.2f}%")
        tput = raw.get("throughput_after")
        if isinstance(tput, (int, float)):
            parts.append(f"tput={float(tput):.1f}")
        depth = raw.get("stack_depth_at_apply")
        if isinstance(depth, int):
            parts.append(f"depth={depth}")
        when = str(raw.get("measured_at") or "").strip()
        if when:
            parts.append(when[:10])  # keep yyyy-mm-dd for compactness
        return ", ".join(parts)
    if isinstance(raw, str):
        return raw.strip()
    if raw is None:
        return ""
    return str(raw).strip()


# Section 5c — Known pitfalls (anti-priors from prior REVERTs)
def _section_pitfalls(inp: SpecialistPromptInputs) -> list[str]:
    """Render the recipe row's ``pitfalls`` from prior REVERTs (description +
    severity); framed as forbidden paths, not suggestions.

    Rows are the flat ``{description, severity, ...}`` shape ``Recipe.to_dict``
    writes, normalised by ``recipe_kb_t0``.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_pitfalls``).

    Returns:
        The rendered known-pitfalls section lines.
    """
    rows = ["## 5c. KNOWN PITFALLS (do NOT repeat — prior REVERTs)", ""]
    for point in inp.warm_start_pitfalls:
        line = _kb_point_line(inp, point, text_key="description", count_label="observed", lead_meta_key="severity")
        if line:
            rows.append(line)
    if len(rows) == 2:  # only the header + blank line: no pitfalls, or all filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows
