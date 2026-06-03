"""Deterministic slug generation per ``kb-critic-integration-contract`` Appendix D.

Two writers — Critic and Alchemist — share this algorithm so the same fact
collapses onto the same ``(scope, kind, slug)`` tuple. Any change here is a
breaking SDK bump (contract D.5).

Public API:

* :func:`slugify(topic)` — ASCII-only deterministic slug, raises
  :class:`SlugifyError` for ``empty`` / ``non_ascii`` / ``too_short``.
* :func:`slugify_safe(topic, translate_fn=None, fallback_prefix='auto')` —
  Non-ASCII safe wrapper. Pure-ASCII input falls through to ``slugify``;
  otherwise the caller can inject a translation function (e.g. an LLM
  call) and we slugify the result. If translation fails or is absent, we
  fall back to ``<fallback_prefix>-<sha256(topic)[:8]>``.
"""

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
    """Report whether a string contains only ASCII characters.

    Args:
        text (str): The string to test.

    Returns:
        bool: True when every character is in the ASCII range.
    """
    return bool(_ASCII_RE.match(text))


def slugify(topic: str) -> str:
    """ASCII-only deterministic slug.

    See contract Appendix D.2 for the step-by-step spec.

    Args:
        topic (str): The topic string to slugify.

    Returns:
        str: The deterministic slug (hash-suffixed when over the max length).

    Raises:
        SlugifyError: If ``topic`` is not a string, is empty/whitespace,
            contains non-ASCII characters, collapses to empty, or is shorter
            than the minimum length.
    """
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
        raise SlugifyError(
            f"non_ascii: topic contains non-ASCII characters (offset={offset})"
        )
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
        raise SlugifyError(
            f"too_short: slug={folded!r} length={len(folded)} < {_MIN_LEN}"
        )
    return folded


def slugify_safe(
    topic: str,
    translate_fn: Callable[[str], str] | None = None,
    *,
    fallback_prefix: str = "auto",
) -> str:
    """Non-ASCII safe wrapper (contract §7.2 / G-6).

    * Pure ASCII → :func:`slugify`.
    * Non-ASCII + ``translate_fn`` provided → ``slugify(translate_fn(topic))``.
    * Non-ASCII without translate_fn (or translate_fn raises) → the
      deterministic fallback ``<prefix>-<sha8>`` so writes remain idempotent.

    Args:
        topic (str): The topic string to slugify.
        translate_fn (Callable[[str], str] | None): Optional translator used
            to romanise non-ASCII input before slugifying.
        fallback_prefix (str): Prefix for the deterministic hash fallback.

    Returns:
        str: A slug — from :func:`slugify`, from the translated text, or the
        ``<prefix>-<sha8>`` fallback.

    Raises:
        SlugifyError: If ``topic`` is not a non-empty string.
    """
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
