# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared TraceLens markdown sanitizers for LLM prompt injection."""

from __future__ import annotations

import re

_BASE64_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\(data:image/[^;]+;base64,[^)]+\)")


def strip_base64_data_urls(text: str | None) -> str:
    """Replace markdown ``data:image/...;base64,...`` images with placeholders."""
    if not text:
        return text or ""
    if "data:image/" not in text:
        return text

    def _sub(match: re.Match[str]) -> str:
        """Build the placeholder replacement for one matched data-URL image."""
        alt = match.group("alt") or "image"
        return f"![{alt}](<<stripped: base64 image — {alt}>>)"

    return _BASE64_IMAGE_PATTERN.sub(_sub, text)
