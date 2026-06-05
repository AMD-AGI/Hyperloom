"""LLM prompt builders for the narrative pass.

The LLM is allowed to do exactly two things:

1. Write a 3-5 sentence executive summary that paraphrases (not
   re-states) the ``GlobalFacts`` headline + attribution + funnel +
   data-quality flags.
2. Write ONE paragraph of narrative for each non-skipped section that
   connects its ``key_facts`` to the surrounding context. The
   deterministic ``markdown_block`` of that section is rendered
   verbatim immediately after the LLM's paragraph; the LLM never sees
   or rewrites it.

Hard guard rails in the system prompt:

* No number that does not appear in ``key_facts``, ``decisions``, or
  ``global_facts`` may appear in the LLM output.
* No capability may be referenced as "used" / "ran" / "contributed"
  unless its decision is ``kept`` / ``attempted`` / ``reverted`` /
  ``rejected`` / ``partial`` — never the bare action name. Sections
  with ``not_attempted`` must be described as such.
* Sections with ``skipped=True`` must not be mentioned.
* Output must be valid JSON with the shape
  ``{"executive_summary": str, "section_narratives": {<section_id>: str}}``.

Why JSON output: stitching the LLM prose into the deterministic
markdown skeleton needs section-keyed access, so a single text blob is
brittle. JSON keeps the boundary clean.
"""

from __future__ import annotations

import json
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
- Describe a capability (for example ``explore`` / ``sweep`` /
  ``specialist`` / ``geak`` / ``oob`` / ``kernel_opt``) as "ran" /
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
- Reflect the ``attribution_method`` honestly: if it is
  ``"best-effort reconstructed"`` or ``"missing"``, say so in plain
  language — do not present a reconstructed attribution as validated.
"""


def _section_input(rendered: RenderedSection) -> dict[str, Any]:
    return {
        "section_id":  rendered.section_id,
        "title":       rendered.title,
        "skipped":     rendered.skipped,
        "key_facts":   list(rendered.key_facts),
        "decisions":   [
            {
                "kind":       d.kind,
                "subject":    d.subject,
                "metric_pct": d.metric_pct,
                "rationale":  d.rationale,
            }
            for d in rendered.decisions
        ],
        "warnings":    list(rendered.warnings),
    }


def build_user_prompt(
    rendered: list[RenderedSection],
    global_facts: GlobalFacts,
) -> str:
    """Build the user-message JSON the LLM sees.

    Returns a JSON string (not a dict) because the LLM's chat input is
    a string and we want the exact bytes the model receives to be
    inspectable in logs.
    """
    payload = {
        "global_facts": global_facts.as_prompt_dict(),
        # Don't show the LLM skipped sections — keeps the prompt smaller
        # and prevents the model from being tempted to "explain" a
        # phantom section.
        "sections":     [_section_input(s) for s in rendered if not s.skipped],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def parse_llm_response(raw: str) -> dict[str, Any]:
    """Best-effort parse of the LLM's JSON output.

    Tolerates a leading ```json fence and surrounding whitespace because
    chat models sometimes hedge with code fences despite the system
    prompt forbidding them. On any failure returns
    ``{"executive_summary": "", "section_narratives": {}}`` so the
    deterministic-only output path is still usable.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip optional ```json ... ``` fence.
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
        "executive_summary": str(data.get("executive_summary") or "").strip(),
        "section_narratives": {
            str(k): str(v).strip()
            for k, v in (data.get("section_narratives") or {}).items()
        },
    }
