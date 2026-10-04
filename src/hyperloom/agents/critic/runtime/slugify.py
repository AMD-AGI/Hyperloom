# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic slug generation per ``kb-critic-integration-contract`` Appendix D."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Callable

from .errors import SlugifyError


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LEADING_TRAILING_DASH_RE = re.compile(r"^-+|-+$")
_REPEATED_DASH_RE = re.compile(r"-+")

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
    folded = _REPEATED_DASH_RE.sub("-", trimmed)
    if not folded:
        raise SlugifyError("empty: slug collapsed to empty after normalisation")
    if len(folded) > _MAX_LEN:
        digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:7]
        return f"{folded[:_TRUNC_LEN]}-{digest}"
    if len(folded) < _MIN_LEN:
        raise SlugifyError(f"too_short: slug={folded!r} length={len(folded)} < {_MIN_LEN}")
    return folded


def slugify_safe(
    topic: str,
    translate_fn: Callable[[str], str] | None = None,
    *,
    fallback_prefix: str = "auto",
) -> str:
    """Non-ASCII safe wrapper (contract §7.2 / G-6)."""
    if not isinstance(topic, str) or not topic.strip():
        raise SlugifyError("empty: topic is empty or whitespace-only")
    normalised = unicodedata.normalize("NFKC", topic)
    if _ascii_only(normalised):
        return slugify(topic)
    if translate_fn is not None:
        try:
            translated = translate_fn(topic)
        except Exception:  # noqa: BLE001 — fall back per contract §7.2
            translated = None
        if translated:
            try:
                return slugify(translated)
            except SlugifyError:
                pass
    digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:8]
    return f"{fallback_prefix}-{digest}"


__all__ = ["slugify", "slugify_safe"]
