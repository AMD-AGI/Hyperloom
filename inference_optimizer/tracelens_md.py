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

    Args:
        text (str | None): Markdown text to sanitize; ``None`` is treated as
            empty.

    Returns:
        str: Markdown with base64 image data URLs replaced by alt-text
        placeholders, or the original text when no such images are present.
    """
    if not text:
        return text or ""
    if "data:image/" not in text:
        return text

    def _sub(match: re.Match[str]) -> str:
        """Build the placeholder replacement for one matched data-URL image.

        Args:
            match (re.Match[str]): Regex match with a named ``alt`` group.

        Returns:
            str: Markdown image whose URL is a stripped-base64 marker carrying
            the original alt-text.
        """
        alt = match.group("alt") or "image"
        return f"![{alt}](<<stripped: base64 image — {alt}>>)"

    return _BASE64_IMAGE_PATTERN.sub(_sub, text)
