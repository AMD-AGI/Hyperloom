# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic slug generation per ``kb-critic-integration-contract`` Appendix D."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from .errors import SlugifyError


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LEADING_TRAILING_DASH_RE = re.compile(r"^-+|-+$")

_ASCII_RE = re.compile(r"^[\x00-\x7f]*$")

_MIN_LEN = 8
_MAX_LEN = 80
_TRUNC_LEN = 72


def _ascii_only(text: str) -> bool:
    """Report whether a string contains only ASCII characters."""
    return bool(_ASCII_RE.match(text))


def slugify(topic: str) -> str:
    """ASCII-only deterministic slug."""
    if not isinstance(topic, str):
        raise SlugifyError(f"topic must be str, got {type(topic).__name__}")
    if not topic.strip():
        raise SlugifyError("empty: topic is empty or whitespace-only")
    normalised = unicodedata.normalize("NFKC", topic)
    if not _ascii_only(normalised):
        # First non-ASCII offset (rough, character-based).
        offset = next(
            (i for i, ch in enumerate(normalised) if ord(ch) > 127),
            -1,
        )
        raise SlugifyError(f"non_ascii: topic contains non-ASCII characters (offset={offset})")
    lowered = normalised.lower()
    replaced = _NON_ALNUM_RE.sub("-", lowered)
    trimmed = _LEADING_TRAILING_DASH_RE.sub("", replaced)
    if not trimmed:
        raise SlugifyError("empty: slug collapsed to empty after normalisation")
    if len(trimmed) > _MAX_LEN:
        digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:7]
        return f"{trimmed[:_TRUNC_LEN]}-{digest}"
    if len(trimmed) < _MIN_LEN:
        raise SlugifyError(f"too_short: slug={trimmed!r} length={len(trimmed)} < {_MIN_LEN}")
    return trimmed


def slugify_safe(topic: str) -> str:
    """Non-ASCII safe wrapper (contract §7.2 / G-6)."""
    if not isinstance(topic, str) or not topic.strip():
        raise SlugifyError("empty: topic is empty or whitespace-only")
    normalised = unicodedata.normalize("NFKC", topic)
    if _ascii_only(normalised):
        return slugify(topic)
    digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:8]
    return f"auto-{digest}"


__all__ = ["slugify", "slugify_safe"]
