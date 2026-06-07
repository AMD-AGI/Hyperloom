# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared TraceLens markdown sanitizers for LLM prompt injection.

Roofline-v2 N11: TraceLens ``analysis.md`` often embeds charts as
``![alt](data:image/png;base64,...)`` data URLs. The payload is opaque
noise to text LLMs (GEAK, Claude Code, Codex, Orchestrator) and can
bloat prompts to hundreds of KB. Strip in-memory before injection; the
on-disk ``analysis.md`` stays intact for operators.
"""

from __future__ import annotations

import re

_BASE64_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(data:image/[^;]+;base64,[^)]+\)"
)


def strip_base64_data_urls(text: str | None) -> str:
    """Replace markdown ``data:image/...;base64,...`` images with placeholders.

    Preserves alt-text in a short marker so downstream agents know a chart
    was present without receiving the binary payload.
    """
    if not text:
        return text or ""
    if "data:image/" not in text:
        return text

    def _sub(match: re.Match[str]) -> str:
        alt = match.group("alt") or "image"
        return f"![{alt}](<<stripped: base64 image — {alt}>>)"

    return _BASE64_IMAGE_PATTERN.sub(_sub, text)
