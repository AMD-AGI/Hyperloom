# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Model names as the Inspector compares them: a provider route, region, version or date suffix aside."""

from __future__ import annotations

import re
from typing import Any


def normalize(model: Any) -> str:
    """A comparable model id: lower case without provider prefix, region, version or date suffix, and dashes for
    dots, underscores and spaces. ``Claude-Opus-4.6``, ``anthropic/claude-opus-4-6``,
    ``us.anthropic.claude-opus-4-6-v1:0`` and ``claude-opus-4-6[1m]`` all become ``claude-opus-4-6``. A normalized
    id normalizes to itself (``x-20260101-latest`` loses both suffixes)."""
    m, last = str(model or ""), None
    for _ in range(4):
        if m == last:
            break
        last, m = m, _normalize_once(m)
    return m


def _normalize_once(m: str) -> str:
    m = m.strip().lower().rsplit("/", 1)[-1]
    m = re.sub(r"\[[^\]]*\]$", "", m).split("@", 1)[0]
    m = re.sub(r"^(?:[a-z]{2,6}\.)?anthropic\.", "", m)
    m = re.sub(r"-v\d+:\d+$", "", m)  # Bedrock's -v1:0; a bare -v3 is part of the name (deepseek-v3)
    m = re.sub(r"[._\s]+", "-", m)
    return re.sub(r"-(?:20\d{6}|20\d\d-\d\d-\d\d|latest)$", "", m)
