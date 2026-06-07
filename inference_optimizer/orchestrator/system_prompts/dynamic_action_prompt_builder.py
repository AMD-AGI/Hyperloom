# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-turn prompt assembler for the ``dynamic_action`` sub-agent.

Every turn's prompt is composed deterministically from the immutable
seed kit + the running journal of prior turns + tool results. When
the journal exceeds ``JOURNAL_TRUNCATE_RATIO * INPUT_TOKEN_CAP`` the
builder clips the middle (keep first N + last N) — no LLM summary,
no new tool call. Public surface is :func:`build_turn_prompt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..dynamic_action_proposal import (
    ALLOWED_PROPOSAL_FIELDS,
    EXPECTED_PROVENANCE,
    FORBIDDEN_PROPOSAL_FIELDS,
)
from ..dynamic_action_tools import (
    ALL_DYNAMIC_TOOLS,
    BENCH_REGISTRY,
    MAX_BENCH_WALL_CLOCK_SEC,
    MAX_READ_SOURCE_CHARS,
)


# Single-turn token budget.
INPUT_TOKEN_CAP: int = 32_000
OUTPUT_TOKEN_CAP: int = 4_000

# Journal section is clipped when it would exceed this fraction of
# the input cap.
JOURNAL_TRUNCATE_RATIO: float = 0.70

# Earliest + latest turns kept when truncating.
JOURNAL_KEEP_HEAD: int = 2
JOURNAL_KEEP_TAIL: int = 4

_CHARS_PER_TOKEN: float = 4.0


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


@dataclass
class JournalTurn:
    """One journal row: LLM text + parsed action + the result the
    runner attached (tool_result for resource tools,
    proposal_validation for ``emit_proposal``)."""

    turn: int
    llm_text: str = ""
    parsed_action: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] | None = None
    proposal_validation: dict[str, Any] | None = None


@dataclass
class PromptInputs:
    """Everything :func:`build_turn_prompt` needs."""

    dyn_id: str
    seed_kit: dict[str, Any]
    spec_payload: dict[str, Any]
    journal: list[JournalTurn]
    turn_cap: int


def _render_seed_kit_section(seed_kit: dict[str, Any]) -> str:
    motivation = seed_kit.get("motivation_gap_text") or ""
    roofline = seed_kit.get("roofline_summary") or ""
    profile = seed_kit.get("profile_keyslices") or []
    kept = seed_kit.get("kept_patches") or []
    reverted = seed_kit.get("reverted_patches") or []
    pitfalls = seed_kit.get("kb_pitfalls") or []
    sources = seed_kit.get("source_root_hints") or []
    lines: list[str] = ["## Seed kit (orchestration-curated, read-only)"]
    lines.append(f"### motivation_gap_text\n{motivation or '(empty)'}")
    lines.append(f"### roofline_summary\n{roofline or '(empty)'}")
    lines.append("### profile_keyslices")
    if profile:
        for row in profile:
            lines.append(
                f"- {row.get('name')!r} gpu_pct={row.get('gpu_pct')} "
                f"bottleneck={row.get('bottleneck')!r}"
            )
    else:
        lines.append("(none)")
    lines.append("### kept_patches")
    if kept:
        for row in kept:
            lines.append(
                f"- {row.get('name')!r} action={row.get('action')!r} "
                f"gain_pct={row.get('gain_pct')}"
            )
    else:
        lines.append("(none)")
    lines.append("### reverted_patches")
    if reverted:
        for row in reverted:
            lines.append(
                f"- {row.get('name')!r} reason={row.get('reason')!r}"
            )
    else:
        lines.append("(none)")
    lines.append("### kb_pitfalls")
    if pitfalls:
        for row in pitfalls:
            text = (row.get("text") or "").replace("\n", " ")
            lines.append(f"- {text}")
    else:
        lines.append("(none)")
    lines.append("### source_root_hints")
    if sources:
        for root in sources:
            lines.append(f"- {root}")
    else:
        lines.append("(none)")
    return "\n".join(lines)


def _render_journal_section(journal: list[JournalTurn]) -> str:
    if not journal:
        return "## Journal\n(no prior turns)"
    rows: list[str] = ["## Journal"]
    for entry in journal:
        rows.append(f"### turn {entry.turn}")
        if entry.llm_text:
            rows.append(f"LLM emitted:\n{entry.llm_text}")
        if entry.parsed_action:
            rows.append(f"parsed_action: {entry.parsed_action}")
        if entry.tool_result is not None:
            rows.append(f"tool_result: {entry.tool_result}")
        if entry.proposal_validation is not None:
            rows.append(
                f"proposal_validation: {entry.proposal_validation}"
            )
    return "\n".join(rows)


def _truncate_journal(
    journal: list[JournalTurn],
) -> list[JournalTurn]:
    """Drop middle journal turns when the section would dominate the
    prompt; keep head + tail and insert an elision marker."""
    if len(journal) <= JOURNAL_KEEP_HEAD + JOURNAL_KEEP_TAIL:
        return list(journal)
    head = journal[:JOURNAL_KEEP_HEAD]
    tail = journal[-JOURNAL_KEEP_TAIL:]
    dropped = len(journal) - len(head) - len(tail)
    marker = JournalTurn(
        turn=-1,
        llm_text=(
            f"[--- {dropped} turn(s) elided by mechanical truncation; "
            f"sub-agent journal is the full record on disk ---]"
        ),
    )
    return [*head, marker, *tail]


def _maybe_truncate(journal: list[JournalTurn], rendered_so_far_tokens: int) -> tuple[list[JournalTurn], bool]:
    """Return ``(possibly_truncated_journal, did_truncate)`` based on
    the running prompt token estimate."""
    rendered = _render_journal_section(journal)
    if _estimate_tokens(rendered) + rendered_so_far_tokens <= int(
        INPUT_TOKEN_CAP * JOURNAL_TRUNCATE_RATIO,
    ):
        return journal, False
    return _truncate_journal(journal), True


def build_system_prompt(turn_cap: int) -> str:
    """Identity + iron rules + tool catalogue. Stable across turns."""
    forbidden = ", ".join(sorted(FORBIDDEN_PROPOSAL_FIELDS))
    tools = ", ".join(sorted(ALL_DYNAMIC_TOOLS))
    bench_line = (
        f"- run_bench: bench_id ∈ {{{', '.join(sorted(BENCH_REGISTRY))}}}; "
        f"each call is hard-killed at {MAX_BENCH_WALL_CLOCK_SEC:.0f}s "
        f"wall-clock.\n"
        if BENCH_REGISTRY else
        "- run_bench is DISABLED; rely on read_source +\n"
        "  read_session_artifact for hypothesis exploration.\n"
    )
    micro_bench_line = (
        "- micro-bench results are for your internal reasoning only;\n"
        "  the system does NOT promote them and does NOT accept any\n"
        "  numeric speedup claim derived from them.\n"
        if BENCH_REGISTRY else ""
    )
    return (
        "You are a dynamic_action sub-agent: a multi-turn ReAct LLM\n"
        "exploring **one** cross-domain patch combination that no\n"
        "single specialist could surface on its own.\n\n"
        "## Iron rules (mechanical denials happen on violation)\n"
        f"- You have at most {turn_cap} turns. Each turn you must\n"
        "  either call exactly one tool OR emit_proposal.\n"
        f"- Available tools: {tools}.\n"
        f"{bench_line}"
        f"- read_source / read_session_artifact return ≤ "
        f"{MAX_READ_SOURCE_CHARS} chars; use a more specific path if\n"
        "  truncated.\n"
        f"- emit_proposal payload MUST equal the closed schema:\n"
        f"  {sorted(ALLOWED_PROPOSAL_FIELDS)}; provenance must be\n"
        f"  '{EXPECTED_PROVENANCE}'.\n"
        f"- FORBIDDEN payload fields (any reject): {forbidden}.\n"
        "- expected_qualitative_argument SHOULD stay qualitative;\n"
        "  numeric speedup claims (e.g. '20%', '1.5x') are flagged for\n"
        "  the Critic, not auto-rejected.\n"
        "- patch_text must be a valid unified diff; scope_domains\n"
        "  must be a subset of the dispatch spec's scope_domains.\n"
        f"{micro_bench_line}"
        "- emit_proposal with patch_text='' signals 'no feasible\n"
        "  cross-domain combo'; this is a legitimate terminal state.\n"
    )


def build_turn_prompt(inputs: PromptInputs) -> tuple[str, bool]:
    """Render one turn's user prompt + whether the journal was clipped.

    The runner sends the returned string as the user-side message and
    pairs it with the static system prompt from :func:`build_system_prompt`.
    """
    spec = inputs.spec_payload
    header_lines = [
        f"# Dispatch {inputs.dyn_id}",
        f"scope_domains: {spec.get('scope_domains')}",
        f"side_effects_declared: {spec.get('side_effects_declared')}",
        f"budget_hint: {spec.get('budget_hint')}",
        f"motivation_gap_text:\n{spec.get('motivation_gap_text')}",
        "",
    ]
    seed_section = _render_seed_kit_section(inputs.seed_kit)
    header = "\n".join(header_lines)
    fixed_tokens = _estimate_tokens(header) + _estimate_tokens(seed_section)
    journal, did_truncate = _maybe_truncate(inputs.journal, fixed_tokens)
    journal_section = _render_journal_section(journal)
    next_turn = (
        inputs.journal[-1].turn + 1 if inputs.journal else 1
    )
    remaining = max(0, inputs.turn_cap - (next_turn - 1))
    footer = (
        f"\n## Next action\n"
        f"You are on turn {next_turn}/{inputs.turn_cap} "
        f"(remaining={remaining}). Emit exactly one tool call OR an\n"
        "emit_proposal payload. Do not include any other prose; the\n"
        "runner only parses the structured action.\n"
    )
    body = "\n\n".join([header, seed_section, journal_section, footer])
    return body, did_truncate


__all__ = [
    "INPUT_TOKEN_CAP",
    "JOURNAL_KEEP_HEAD",
    "JOURNAL_KEEP_TAIL",
    "JOURNAL_TRUNCATE_RATIO",
    "JournalTurn",
    "OUTPUT_TOKEN_CAP",
    "PromptInputs",
    "build_system_prompt",
    "build_turn_prompt",
]
