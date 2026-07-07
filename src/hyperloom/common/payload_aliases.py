# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-only deprecation aliases for renamed payload fields (tree-reform.MD §7).

Canonical home for the ``extra_sglang_args`` → ``extra_server_args`` compat
shim that was independently copied into the kernel-agent and robustness-agent
sub-packages. The legacy sglang-era name is a read-only alias for the
framework-neutral canonical key.

Contract (identical to the historical ``hyperloom.inference_optimizer.compat.payload_aliases``,
now re-exported from ``hyperloom.inference_optimizer.compat`` after the
tree-reform.MD P2.4 ``compat/`` -> ``compat.py`` flattening):

* **Read-only on legacy**. Writers always emit the canonical name; the alias
  only flows in the read direction.
* **Single ``DeprecationWarning`` per call**, only when the legacy key is used
  (canonical absent AND legacy present). ``stacklevel=3`` points the report at
  the caller's caller (the migration target).
* **No value transformation** beyond coercion to ``str`` (``None`` → ``""``,
  list/tuple → space-joined shell tokens, else ``str(value)``).

Stdlib-only (no first-party imports) so any package may depend on it.
"""

from __future__ import annotations

import warnings
from typing import Any


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
    """Read ``extra_server_args`` from a payload dict, with a one-release
    read-only fallback to the legacy ``extra_sglang_args`` key.

    Resolution order:

    1. If ``payload[CANONICAL_KEY]`` is present (any value, including empty
       string), return it coerced via :func:`_coerce_str`; no warning.
    2. Else if ``payload[LEGACY_KEY]`` is present, emit a single
       ``DeprecationWarning`` (``stacklevel=3``) and return the coerced value.
    3. Else return ``default``.

    The check is ``in``, not truthiness, so a deliberately empty string value
    is distinguished from a missing key.

    Args:
        payload (dict): Dict-like payload (``Intent.payload`` / ``Task.params`` /
            a JSON envelope body / a SharedState entry).
        default (str): Returned when neither key is present.

    Returns:
        str: The coerced canonical value, the coerced legacy value (with a
        ``DeprecationWarning``), or ``default``.
    """
    if CANONICAL_KEY in payload:
        return _coerce_str(payload[CANONICAL_KEY])
    if LEGACY_KEY in payload:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        return _coerce_str(payload[LEGACY_KEY])
    return default


def migrate_legacy_key_in_place(payload: dict) -> bool:
    """One-shot transform for persisted payloads (state.json rows, KB JSONL
    records, audit DB rows), run once at load so subsequent saves emit the
    canonical name only.

    * If ``LEGACY_KEY`` is present and ``CANONICAL_KEY`` is NOT present, copy
      the value to ``CANONICAL_KEY``, delete ``LEGACY_KEY``, return ``True``.
    * If both keys are present, leave both and return ``False``.
    * Otherwise return ``False``.

    Silent by design (the read-side warning is the audit channel).

    Args:
        payload (dict): Mutable payload mapping to migrate in place.

    Returns:
        bool: ``True`` when the legacy key was migrated; ``False`` otherwise.
    """
    if LEGACY_KEY in payload and CANONICAL_KEY not in payload:
        payload[CANONICAL_KEY] = payload.pop(LEGACY_KEY)
        return True
    return False


__all__ = [
    "CANONICAL_KEY",
    "LEGACY_KEY",
    "read_extra_server_args",
    "migrate_legacy_key_in_place",
]
