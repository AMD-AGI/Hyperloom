# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``*_CUSTOM_HEADERS`` wire format: one parser, one renderer, one env-ref rule.

Both protocol lines and both packages configure gateway headers through the
same two spellings -- a JSON object, or newline-delimited ``Name: value`` -- so
the format has exactly one reader here. A second copy is a second answer to
"did this header survive", and the symptom of the two disagreeing is a 401 that
names a variable which is set correctly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping

log = logging.getLogger(__name__)

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# A value carrying ", Some-Name:" almost certainly meant to be two headers.
_PACKED_PAIR_RE = re.compile(r",\s*[A-Za-z0-9][A-Za-z0-9_-]*\s*:")

__all__ = [
    "expand_env_refs",
    "format_custom_headers",
    "parse_custom_headers",
]


def expand_env_refs(raw: str, env: Mapping[str, str] | None = None) -> str:
    """Substitute shell-style ``${VAR}`` references, from *env* or the process."""
    source = env if env is not None else os.environ
    return _ENV_REF_RE.sub(lambda match: str(source.get(match.group(1), "")), raw)


def _from_json(text: str) -> dict[str, str] | None:
    """Headers from a JSON document, or ``None`` when it is not JSON at all."""
    if not text.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        # A list is well-formed JSON that cannot be a header map; parsing it as
        # ``Name: value`` lines would invent headers out of its punctuation.
        return {}
    return {str(k).strip(): str(v).strip() for k, v in parsed.items() if str(k).strip()}


def _from_lines(text: str) -> dict[str, str]:
    """Headers from the newline-delimited ``Name: value`` form."""
    headers: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip():
            headers[name.strip()] = value.strip()
    dropped = sum(1 for line in text.splitlines() if line.strip() and ":" not in line)
    if dropped:
        log.warning("ignored %d custom header line(s) without a 'Name: value' colon", dropped)
    return headers


def parse_custom_headers(raw: str | None, *, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse custom LLM headers: a JSON object, or ``Name: value`` lines.

    Args:
        raw: The variable's value, before ``${VAR}`` expansion.
        env: Mapping the ``${VAR}`` references resolve against; the process
            environment when omitted.

    Returns:
        Header name to value, both stripped. Empty when *raw* is empty or names
        no header.
    """
    if not raw:
        return {}
    expanded = expand_env_refs(raw, env)
    text = expanded.strip()
    if not text:
        return {}
    headers = _from_json(text)
    if headers is None:
        headers = _from_lines(expanded)
    for name, value in headers.items():
        # An empty value usually means an unresolved ${VAR}; a blank subscription key still 401s at the gateway, so
        # surface it rather than fail silently.
        if not value:
            log.warning("custom header %r has an empty value (unresolved ${VAR}?)", name)
        # Comma-separated pairs on one line are not supported: a header value may legitimately contain commas, so
        # splitting on them would corrupt real values.
        elif _PACKED_PAIR_RE.search(value):
            log.warning(
                "custom header %r value %r looks like it packs more headers on one "
                "line; put each on its own line (comma-separated is not split)",
                name,
                value,
            )
    return headers


def format_custom_headers(headers: Mapping[str, str]) -> str:
    """Render headers as the newline-delimited form both SDKs understand."""
    return "\n".join(f"{name}: {value}" for name, value in headers.items())
