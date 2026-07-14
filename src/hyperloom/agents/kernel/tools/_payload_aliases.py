# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Stdlib-only payload-args helper for standalone kernel-agent tools.

Kept as an independent, stdlib-only module because ``tools/`` scripts must run
standalone on remote nodes
without a ``hyperloom`` import: they are invoked as bare
``python3 <root>/tools/<tool>.py --args`` subprocesses (see
``HYPERLOOM_KERNEL_AGENT_ROOT`` in
``hyperloom.orchestrator.kernel.request_handlers``), imported via the bare
module name ``from _payload_aliases import read_extra_server_args`` (not a
package-relative ``from ._payload_aliases import``) by
``kernel_optimization.py``, and some of their code paths execute inside Ray
workers (``tools/backends/``) that do not inherit the driver's ``sys.path``.
This mirrors the same, deliberate exception already made for ``_paths.py``
(see tree-reform-lessons.MD §13) — do not "finish" this extraction by
importing ``hyperloom.common`` here. Behaviour is pinned by
``test_payload_aliases_shim.py``.
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


__all__ = [
    "CANONICAL_KEY",
    "LEGACY_KEY",
    "read_extra_server_args",
]
