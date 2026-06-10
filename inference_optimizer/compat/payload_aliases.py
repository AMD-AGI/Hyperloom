# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-only deprecation aliases for renamed payload fields.

The payload-surface field ``extra_sglang_args`` (sglang-era name) is
a read-only legacy alias for the framework-neutral
``extra_server_args``. The field carries arbitrary server-launch
flags that get routed into ``EXTRA_SGLANG_ARGS`` / ``EXTRA_VLLM_ARGS``
/ ``EXTRA_ATOM_ARGS`` by the per-framework Magpie wrapper, so the
sglang-specific name is a lie now that vllm + atom support exists.

This module provides the single, well-tested reader helper that every
in-process payload reader in the Hyperloom orchestrator funnels
through. The contract is intentionally narrow:

* **Read-only on legacy**. Writers always emit the canonical name
  directly; the alias only flows in the read direction so the
  in-flight surface drifts cleanly toward the new key.
* **Single ``DeprecationWarning`` per call**. Fires only when the
  legacy key is actually used (i.e. the canonical key is absent
  AND the legacy key is present). Reported filename is the caller
  via ``stacklevel=3`` so the warning points at the migration target.
* **No value transformation**. The helper returns the raw string
  (coerced from whatever JSON/YAML loaded — Python ``None`` → ``""``
  and non-str → ``str``). The reader site is responsible for any
  ``.strip()`` / shape normalisation, matching the per-site
  conventions already in the codebase.

The static guard ``test_no_legacy_writer_sites.py`` tracks the
allowlist of remaining legacy references.
"""

from __future__ import annotations

import warnings
from typing import Any


# Canonical and legacy key names. Kept as module-level constants so
# downstream code can grep them deterministically.
CANONICAL_KEY: str = "extra_server_args"
LEGACY_KEY: str = "extra_sglang_args"


_DEPRECATION_MESSAGE: str = (
    f"payload field {LEGACY_KEY!r} is a deprecation alias for "
    f"{CANONICAL_KEY!r}. The legacy name carries the same value but "
    f"will be removed in the next Hyperloom release — switch the "
    f"writer site (or the operator script emitting this payload) to "
    f"the canonical name."
)


def _coerce_str(value: Any) -> str:
    """Coerce a payload value into a string the same way every reader
    site previously did inline (``str(payload.get(...) or "")``).

    ``None`` collapses to the empty string so callers that immediately
    ``.strip()`` get the same result. Non-string, non-None values fall
    through ``str()``.

    Args:
        value (Any): Raw payload value loaded from JSON/YAML.

    Returns:
        str: ``""`` for ``None``, the value unchanged when already a string,
        otherwise ``str(value)``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # The LLM occasionally emits server flags as a JSON list
    # (``["--flag", "value"]``); space-join into shell tokens rather than
    # emitting a Python repr that the Magpie wrapper would splice verbatim
    # into ``vllm/sglang serve`` (rejected as "unrecognized arguments").
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


def read_extra_server_args(payload: dict, *, default: str = "") -> str:
    """Read the ``extra_server_args`` value from a payload dict, with
    a one-release read-only fallback to the legacy ``extra_sglang_args``
    key.

    Resolution order:

    1. If ``payload[CANONICAL_KEY]`` is present (any value, including
       empty string), return it coerced via :func:`_coerce_str`. No
       warning is emitted — the canonical key being present means the
       writer has already migrated.
    2. Else if ``payload[LEGACY_KEY]`` is present, emit a single
       ``DeprecationWarning`` (``stacklevel=3`` so the report points
       at the caller's caller — usually the executor / coordinator
       site that needs migrating) and return the coerced value.
    3. Else return ``default`` (empty string by default).

    The check is ``in``, not truthiness, so a deliberately empty
    string value is distinguished from a missing key. This matches
    the legacy reader convention which used
    ``str(payload.get("extra_sglang_args") or "")`` and treated both
    missing-key and empty-value the same — the helper preserves the
    cumulative behaviour while pinning the canonical-vs-legacy split.

    Args:
        payload (dict): Dict-like payload (typically ``Intent.payload`` /
            ``Task.params`` / a JSON envelope body / a SharedState entry).
            Other Mapping shapes (e.g. ``MappingProxy``) work as long as
            ``__contains__`` and ``__getitem__`` are implemented.
        default (str): Returned when neither key is present. Defaults to the
            empty string so the helper drops in for the legacy
            ``str(payload.get(...) or "")`` idiom.

    Returns:
        str: The coerced canonical value, the coerced legacy value (with a
        ``DeprecationWarning``), or ``default`` when neither key is present.
    """
    if CANONICAL_KEY in payload:
        return _coerce_str(payload[CANONICAL_KEY])
    if LEGACY_KEY in payload:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        return _coerce_str(payload[LEGACY_KEY])
    return default


def read_extra_server_args_from_envs(envs: dict, *, default: str = "") -> str:
    """Same contract as :func:`read_extra_server_args` but operates on
    the ``envs`` dict shape carried by the Magpie YAML materializer.

    The per-framework env names (``EXTRA_SGLANG_ARGS``,
    ``EXTRA_VLLM_ARGS``, ``EXTRA_ATOM_ARGS``) are **not** renamed —
    those are the canonical per-framework slots that the Magpie
    wrapper looks up by name. This helper covers the *payload-surface*
    field that some materializer call-sites expose alongside the env
    map (the framework-neutral pre-routing slot).

    Args:
        envs (dict): Magpie ``envs`` mapping that may carry the canonical or
            legacy payload-surface key.
        default (str): Returned when neither key is present (default ``""``).

    Returns:
        str: The coerced canonical value, the coerced legacy value (with a
        ``DeprecationWarning``), or ``default`` when neither key is present.
    """
    if CANONICAL_KEY in envs:
        return _coerce_str(envs[CANONICAL_KEY])
    if LEGACY_KEY in envs:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        return _coerce_str(envs[LEGACY_KEY])
    return default


def migrate_legacy_key_in_place(payload: dict) -> bool:
    """One-shot transform helper for persisted payloads (state.json
    rows, KB JSONL records, audit DB rows). Used by the SharedState
    loader on first read of a legacy state.json file so subsequent
    saves emit the canonical name only.

    Behaviour:

    * If ``LEGACY_KEY`` is present and ``CANONICAL_KEY`` is NOT
      present, copy the value over to ``CANONICAL_KEY``, delete
      ``LEGACY_KEY``, and return ``True``.
    * If both keys are present, leave both alone and return ``False``
      (the canonical wins on read; the caller can choose to drop
      ``LEGACY_KEY`` separately if it wants).
    * Otherwise return ``False`` (no work).

    Does NOT emit a warning — this is the *persistence-side* migration
    path, run once at load. The read-side ``DeprecationWarning`` is
    the audit channel for in-flight payloads.

    Args:
        payload (dict): Mutable payload mapping to migrate in place.

    Returns:
        bool: ``True`` when the legacy key was copied to the canonical key and
        removed; ``False`` otherwise (both present, or no legacy key).
    """
    if LEGACY_KEY in payload and CANONICAL_KEY not in payload:
        payload[CANONICAL_KEY] = payload.pop(LEGACY_KEY)
        return True
    return False


__all__ = [
    "CANONICAL_KEY",
    "LEGACY_KEY",
    "read_extra_server_args",
    "read_extra_server_args_from_envs",
    "migrate_legacy_key_in_place",
]
