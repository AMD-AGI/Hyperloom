# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM prompt builders for the narrative pass."""

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
    """Project a rendered section into the JSON shape the LLM receives."""
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
    """Build the user-message JSON the LLM sees (string so the exact bytes are log-inspectable)."""
    payload = {
        "global_facts": global_facts.as_prompt_dict(),
        # Skipped sections are withheld.
        "sections": [_section_input(s) for s in rendered if not s.skipped],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# The system prompt asks for "3-5 sentences" and "1 short paragraph" (see SYSTEM_PROMPT).
_MAX_EXEC_SUMMARY_CHARS = 1500
_MAX_NARRATIVE_CHARS = 800

# Narratives are pasted into a slot the composer owns -- the summary under an H2, each section paragraph under an H3.
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
    """Return model prose only when it is safe to paste into the report."""
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) > max_chars:
        return ""
    if any(_BLOCK_OPENER.match(line) for line in cleaned.splitlines()):
        return ""
    return cleaned


def parse_llm_response(raw: str) -> dict[str, Any]:
    """Best-effort parse of the LLM's JSON output."""
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
