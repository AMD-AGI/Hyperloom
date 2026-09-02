# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM prompt builders for the narrative pass.

The LLM writes only an executive summary plus one paragraph per
non-skipped section; deterministic ``markdown_block``s are stitched in
verbatim. Guard rails asked for in the system prompt: no numbers outside
``key_facts``/``decisions``/``global_facts``, honest capability status,
JSON output keyed by section so stitching stays clean.

A prompt is a request, not a constraint, so :func:`parse_llm_response` also
enforces the two guard rails that can corrupt the document rather than merely
misinform: prose that would restructure the report, and prose that ignored the
length brief. Claims about numbers remain the system prompt's problem.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import RenderedSection
from .cross_section import GlobalFacts

__all__ = [
    "SYSTEM_PROMPT",
    "build_user_prompt",
]


SYSTEM_PROMPT = """\
You are writing the narrative portions of a Hyperloom session
performance report. The numerical facts (throughputs, gains, kernel
counts, paths, decisions) have already been computed and rendered as
markdown blocks that will be stitched into the final document AS-IS.

You may NOT:
- Invent numbers, percentages, kernel names, paths, or GPU types. Every
  numeric or named entity you write MUST appear verbatim in one of:
  ``global_facts``, ``key_facts``, or ``decisions``.
- Describe a capability (for example ``explore`` / ``conc_sweep`` /
  ``specialist`` / ``geak`` / ``forge`` / ``kernel_opt``) as "ran" /
  "contributed" / "applied" unless its decision is one of: ``kept`` /
  ``attempted`` / ``reverted`` / ``rejected`` / ``partial``. Legacy
  aliases such as ``backends`` / ``params`` / ``validate_stack`` must be
  described as archived compatibility rows unless a decision says they
  actually ran. Capabilities listed in ``capabilities_not_attempted``
  MUST be described as "never ran" / "not attempted" / "not invoked".
- Write a paragraph for any section whose ``skipped`` flag is true.
- Rephrase or "summarize" the deterministic markdown block.

You MUST:
- Output strictly valid JSON, no leading or trailing prose, with this
  shape:
    {
      "executive_summary":   "<3-5 sentences>",
      "section_narratives":  {"<section_id>": "<1 short paragraph>", ...}
    }
- For each non-skipped section in the input, include one entry in
  ``section_narratives`` (omit skipped sections entirely).
- Surface every entry in ``global_facts.data_quality_flags`` in the
  executive summary (concisely; users have been bitten by silent data
  issues like all-zero GPU monitoring readings).
- Reflect the ``attribution_method`` honestly: any value other than
  ``"validated"`` (for example ``"unattributed"`` or ``"missing"``)
  MUST be described in plain language as not validated.
"""


def _section_input(rendered: RenderedSection) -> dict[str, Any]:
    """Project a rendered section into the JSON shape the LLM receives.

    Args:
        rendered (RenderedSection): The section to project.

    Returns:
        dict[str, Any]: A JSON-friendly dict with the section id, title,
            skipped flag, key facts, decisions and warnings (the markdown
            block is deliberately excluded so the LLM cannot rewrite it).
    """
    return {
        "section_id": rendered.section_id,
        "title": rendered.title,
        "skipped": rendered.skipped,
        "key_facts": list(rendered.key_facts),
        "decisions": [
            {
                "kind": d.kind,
                "subject": d.subject,
                "metric_pct": d.metric_pct,
                "rationale": d.rationale,
            }
            for d in rendered.decisions
        ],
        "warnings": list(rendered.warnings),
    }


def build_user_prompt(
    rendered: list[RenderedSection],
    global_facts: GlobalFacts,
) -> str:
    """Build the user-message JSON the LLM sees (string so the exact bytes are log-inspectable).

    Args:
        rendered: Rendered sections to include; skipped sections are withheld.
        global_facts: Global facts block prepended to the prompt payload.

    Returns:
        A pretty-printed JSON string representing the user message.
    """
    payload = {
        "global_facts": global_facts.as_prompt_dict(),
        # Skipped sections are withheld.
        "sections": [_section_input(s) for s in rendered if not s.skipped],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# The system prompt asks for "3-5 sentences" and "1 short paragraph"
# (see SYSTEM_PROMPT). These ceilings are those shapes with generous room, so
# they only catch output that ignored the brief outright.
_MAX_EXEC_SUMMARY_CHARS = 1500
_MAX_NARRATIVE_CHARS = 800

# Narratives are pasted into a slot the composer owns -- the summary under an
# H2, each section paragraph under an H3. Markdown that *opens a block* escapes
# that slot and re-parents every deterministic block after it: an unterminated
# HTML comment comments the rest of the report out, an odd code fence swallows
# it. CommonMark has many ways to open a block, and a blacklist that misses one
# fails silently and totally -- so match the act of opening a block, and treat
# it as evidence the model ignored the brief.
_BLOCK_OPENER = re.compile(
    r"""
    ^\s{0,3}(
        \#{1,6}(\s|$)                      # ATX heading
      | (```|~~~)                          # fenced code
      | (={3,}|-{3,}|\*{3,}|_{3,})\s*$     # setext underline / thematic break
      | <                                  # any HTML block: tag, comment, PI, CDATA
    )
    """,
    re.VERBOSE,
)


def _sanitize(text: str, *, max_chars: int) -> str:
    """Return model prose only when it is safe to paste into the report.

    Prose that opens a markdown block, or that ran past the length the prompt
    asked for, is discarded whole rather than repaired: a partially stripped
    paragraph is prose the model did not write, and half a truncated sentence
    reads worse than the deterministic text. ``""`` means "nothing usable",
    which every caller already treats as "fall back".

    Note the anchor: only a line *starting* a block is rejected, so ordinary
    prose containing ``<`` (``latency < 5ms``) passes untouched.

    Args:
        text (str): Raw narrative as the model wrote it.
        max_chars (int): Length ceiling for the whole narrative.

    Returns:
        str: The narrative, or ``""`` when it must not be used.
    """
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) > max_chars:
        return ""
    if any(_BLOCK_OPENER.match(line) for line in cleaned.splitlines()):
        return ""
    return cleaned


def parse_llm_response(raw: str) -> dict[str, Any]:
    """Best-effort parse of the LLM's JSON output.

    Tolerates a code fence; on any failure returns empty fields so the
    deterministic-only output path stays usable. Surviving values are passed
    through :func:`_sanitize`, so a field may come back empty even when the
    model filled it -- the composer already treats empty as "fall back".

    Args:
        raw: Raw text returned by the LLM, optionally wrapped in a code fence.

    Returns:
        A dict with ``executive_summary`` and ``section_narratives`` keys;
        both empty when parsing fails.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"executive_summary": "", "section_narratives": {}}
    if not isinstance(data, dict):
        return {"executive_summary": "", "section_narratives": {}}
    return {
        "executive_summary": _sanitize(str(data.get("executive_summary") or ""), max_chars=_MAX_EXEC_SUMMARY_CHARS),
        "section_narratives": {
            str(k): _sanitize(str(v), max_chars=_MAX_NARRATIVE_CHARS)
            for k, v in (data.get("section_narratives") or {}).items()
        },
    }
