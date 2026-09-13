# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build a KB scope dict from explicit context + session memory."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .errors import ScopeError


ORG_DEFAULT = "hyperloom"

# 6 required dimensions
CRITICAL_SCOPE_KEYS: tuple[str, ...] = (
    "org",
    "framework",
    "model",
    "model_family",
    "workload",
    "precision",
)

# Optional dimensions — never write "unknown"; either set the value or omit the key
OPTIONAL_SCOPE_KEYS: tuple[str, ...] = ("scale", "objective")


# Crude model-family heuristic.
_MODEL_FAMILY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"^qwen", re.IGNORECASE), "qwen"),
    (re.compile(r"^llama", re.IGNORECASE), "llama"),
    (re.compile(r"^glm", re.IGNORECASE), "glm"),
    (re.compile(r"^mistral", re.IGNORECASE), "mistral"),
    (re.compile(r"^kimi", re.IGNORECASE), "kimi"),
    (re.compile(r"^gemma", re.IGNORECASE), "gemma"),
)


_UNKNOWN_VALUES: frozenset[str] = frozenset({"", "unknown", "null", "none"})


def _normalise(value: Any) -> str:
    """Apply contract G-3 normalisation: trim + lowercase, stringified."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    return text


def _is_present(value: Any) -> bool:
    """Report whether a value is a meaningful (non-unknown) scope value."""
    text = _normalise(value)
    return text not in _UNKNOWN_VALUES


def _pick(
    key: str,
    *sources: Mapping[str, Any] | None,
) -> str:
    """Return the first present, normalised value for ``key`` across sources."""
    for src in sources:
        if not src:
            continue
        if key in src and _is_present(src[key]):
            return _normalise(src[key])
    return ""


def derive_model_family(model: str) -> str:
    """Best-effort family extractor for ``model`` strings."""
    text = _normalise(model)
    if not text:
        return ""
    for pat, family in _MODEL_FAMILY_RULES:
        if pat.match(text):
            return family
    return ""


def build_scope(
    packet_context: Mapping[str, Any] | None = None,
    *,
    session_context: Mapping[str, Any] | None = None,
    require_critical: bool = True,
) -> dict[str, Any]:
    """Construct a KB scope dict from explicit + session contexts."""
    pc = dict(packet_context or {})
    sc = dict(session_context or {})

    scope: dict[str, Any] = {}

    for key in CRITICAL_SCOPE_KEYS:
        if key == "org":
            value = _pick(key, pc, sc) or ORG_DEFAULT
            scope[key] = value
            continue
        value = _pick(key, pc, sc)
        if not value and key == "model_family":
            model_value = _pick("model", pc, sc)
            value = derive_model_family(model_value) or ""
        scope[key] = value or "unknown"

    for key in OPTIONAL_SCOPE_KEYS:
        if _is_present(pc.get(key)):
            scope[key] = _normalise(pc[key])
        elif _is_present(sc.get(key)):
            scope[key] = _normalise(sc[key])

    if require_critical:
        missing_critical = [k for k in ("model", "framework") if scope[k] == "unknown"]
        if missing_critical:
            raise ScopeError(
                f"cannot build KB scope: missing critical keys "
                f"{missing_critical!r}; explicit context keys="
                f"{sorted(pc.keys())!r}; session context keys="
                f"{sorted(sc.keys())!r}"
            )

    return scope


def scope_cache_key(
    scope: Mapping[str, Any],
    *,
    topic: str | None = None,
    kind: str | None = None,
    limit: int = 10,
) -> str:
    """Stable, hashable representation of a ``list_priors`` query."""
    parts = [f"{k}={scope[k]}" for k in sorted(scope.keys())]
    if topic:
        parts.append(f"topic={_normalise(topic)}")
    if kind:
        parts.append(f"kind={_normalise(kind)}")
    parts.append(f"limit={limit}")
    return "|".join(parts)


__all__ = [
    "CRITICAL_SCOPE_KEYS",
    "OPTIONAL_SCOPE_KEYS",
    "ORG_DEFAULT",
    "build_scope",
    "derive_model_family",
    "scope_cache_key",
]
