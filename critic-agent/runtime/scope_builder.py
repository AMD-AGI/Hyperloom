# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Build a KB scope dict from explicit context + session memory.

The KB service requires the 6 mandatory scope dimensions
``{org, framework, model, model_family, workload, precision}`` to be present
on every write (contract §2.1). Two optional dimensions ``{scale, objective}``
may be added when known. ``org`` is fixed at ``"hyperloom"`` in v1.

Inputs to :func:`build_scope`:

* ``packet_context``: explicit context coming from the current Coordinator
  packet or decision request.
* ``session_context``: context recovered from session memory (already
  merged with previous turns).

Order of precedence: ``packet_context > session_context > "unknown"``. The
service normalises values via ``trim().lowercase()`` (G-3); we do the same
client-side so list / metadata filters round-trip without surprises.

If :data:`CRITICAL_SCOPE_KEYS` cannot be filled by either input we raise
:class:`ScopeError`. The caller (typically ``decision_reviewer``) treats
this as a hard signal to skip KB reads / writes and downgrade the verdict
to ``needs_review``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .errors import ScopeError


ORG_DEFAULT = "hyperloom"

# 6 必填维度
CRITICAL_SCOPE_KEYS: tuple[str, ...] = (
    "org",
    "framework",
    "model",
    "model_family",
    "workload",
    "precision",
)

# 可选维度 — 不写 "unknown"，要么填要么不填 key
OPTIONAL_SCOPE_KEYS: tuple[str, ...] = ("scale", "objective")


# Crude model-family heuristic (extend as new model families enter Hyperloom).
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
    text = _normalise(value)
    return text not in _UNKNOWN_VALUES


def _pick(
    key: str,
    *sources: Mapping[str, Any] | None,
) -> str:
    for src in sources:
        if not src:
            continue
        if key in src and _is_present(src[key]):
            return _normalise(src[key])
    return ""


def derive_model_family(model: str) -> str:
    """Best-effort family extractor for ``model`` strings.

    Returns ``""`` when the model is unknown/empty so callers can decide
    whether to default to ``"unknown"`` (and emit a warning) or to refuse
    the write entirely.
    """
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
    """Construct a KB scope dict from explicit + session contexts.

    Args:
        packet_context: Context from the current request. Wins on conflict.
        session_context: Context recovered from :mod:`session_memory`.
        require_critical: If True (default), raise :class:`ScopeError` when
            ``model`` or ``framework`` cannot be filled. Set to False for
            dry-run / preview callers that want to surface missing keys
            non-fatally.

    Returns:
        A dict with all 6 critical keys (``"unknown"`` filling otherwise),
        and any optional keys present in either input. ``org`` defaults to
        ``"hyperloom"``.
    """
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
            # Derive from model when family is missing.
            model_value = _pick("model", pc, sc)
            value = derive_model_family(model_value) or ""
        scope[key] = value or "unknown"

    for key in OPTIONAL_SCOPE_KEYS:
        if _is_present(pc.get(key)):
            scope[key] = _normalise(pc[key])
        elif _is_present(sc.get(key)):
            scope[key] = _normalise(sc[key])

    if require_critical:
        missing_critical = [
            k for k in ("model", "framework") if scope[k] == "unknown"
        ]
        if missing_critical:
            raise ScopeError(
                f"cannot build KB scope: missing critical keys "
                f"{missing_critical!r}; explicit context keys="
                f"{sorted(pc.keys())!r}; session context keys="
                f"{sorted(sc.keys())!r}"
            )

    return scope


def scope_cache_key(scope: Mapping[str, Any], *, topic: str | None = None) -> str:
    """Stable, hashable representation of (scope + optional topic).

    Used by ``session_memory.SessionMemory.get_cached_priors``.
    """
    parts = [f"{k}={scope[k]}" for k in sorted(scope.keys())]
    if topic:
        parts.append(f"topic={_normalise(topic)}")
    return "|".join(parts)


__all__ = [
    "CRITICAL_SCOPE_KEYS",
    "OPTIONAL_SCOPE_KEYS",
    "ORG_DEFAULT",
    "build_scope",
    "derive_model_family",
    "scope_cache_key",
]
