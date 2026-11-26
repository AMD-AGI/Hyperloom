# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Secret masking applied to every value the Inspector serves.

The session directory holds prompts, server command lines and env snapshots;
any of them may carry a credential. Masking is structural (dict keys that
name a secret) and textual (token shapes and ``KEY=value`` pairs in strings).

It is a superset of what Hyperloom masks before it writes
(:mod:`hyperloom.common.env_safety`): every shape :func:`redact_secret_values`
masks, every name :func:`is_credential_name` calls a credential (as a key of an
environment mapping, as ``NAME=value`` and as a ``--flag value``), plus the
Inspector's own (URL credentials, ``x-api-key``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    is_credential_name,
    is_secret_shaped_env_name,
    redact_secret_values,
)

MASK = "<redacted>"
# The marker redact_secret_values writes in place of a secret; the Inspector shows its own.
_HYPERLOOM_MASK = "[REDACTED]"

# Matched against the *end* of a dict key, so ``HF_TOKEN`` / ``api_key`` mask
# while counters such as ``input_tokens`` stay visible.
_SECRET_KEY = re.compile(
    r"(?:^|[_\-.])(?:api[_\-]?key|token|secret|password|passwd|credentials?|authorization|custom[_\-]?headers)$",
    re.IGNORECASE,
)

_TEXT_PATTERNS = [
    # Anthropic / OpenAI style keys.
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),
    # HuggingFace tokens.
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    # GitHub tokens.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
]
# ``Bearer xyz`` / ``x-api-key: xyz``: keep the label, mask the value.
_LABELLED = re.compile(r"(?i)(bearer\s+|x-api-key[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9._\-~+/=]{8,})")
# ``FOO_TOKEN=value`` / ``"api_key": "value"`` inside free text.
_ASSIGN = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD)[\"']?\s*[:=]\s*[\"']?)"
    r"((?=[^\s\"',;]*[A-Za-z])[^\s\"',;]{4,})"
)
# ``NAME=value`` / ``"NAME": "value"`` for an upper-case environment name, masked when Hyperloom calls the name a
# credential (``AWS_ACCESS_KEY_ID``, ``HYPERLOOM_SANDBOX_KEY_DEV``, ``LLM_GATEWAY_KEY``).
_ENV_ASSIGN = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9_]*)([\"']?\s*[:=]\s*[\"']?)([^\s\"',;]+)")
# ``--api-key value`` / ``--hf-token=value``, masked when Hyperloom calls the flag a credential.
_FLAG = re.compile(r"(?<!\S)(--?[A-Za-z][A-Za-z0-9_\-]*)(=|[ \t]+)(?!-)([^\s\"',;]+)")
# Credentials embedded in URLs: scheme://user:pass@host
_URL_CRED = re.compile(r"([a-z][a-z0-9+.\-]*://)([^/\s:@]+):([^/\s@]+)@", re.IGNORECASE)


def _mask_named(match: re.Match) -> str:
    """The ``name``, separator and value of *match*, the value masked when the name is a credential's."""
    name, sep, value = match.groups()
    return name + sep + (MASK if is_credential_name(name) else value)


def redact_text(text: str) -> str:
    """Mask secret-shaped substrings in *text*."""
    if not text:
        return text
    for pat in _TEXT_PATTERNS:
        text = pat.sub(MASK, text)
    text = _LABELLED.sub(lambda m: m.group(1) + MASK, text)
    text = _ASSIGN.sub(lambda m: m.group(1) + MASK, text)
    # After the Inspector's own, which mask a whole key (``sk-...``) where Hyperloom keeps its prefix.
    text = redact_secret_values(text).replace(_HYPERLOOM_MASK, MASK)
    text = _ENV_ASSIGN.sub(_mask_named, text)
    text = _FLAG.sub(_mask_named, text)
    return _URL_CRED.sub(lambda m: f"{m.group(1)}{m.group(2)}:{MASK}@", text)


def is_secret_key(key: str) -> bool:
    """Whether a dict key names a secret anywhere in a document: a credential-shaped key or one of Hyperloom's own
    secret environment names."""
    return bool(_SECRET_KEY.search(key)) or key.upper() in BENCHMARK_SECRET_ENV_NAMES


def is_env_field(key: object) -> bool:
    """Whether a document field named *key* holds environment variables (``env``, ``extra_envs``, ``launch_env``), as
    Hyperloom decides it (``env_safety._is_env_field``, private there; a contract test holds the two equal)."""
    name = str(key).lower()
    return name in ("environment", "environ") or name.rstrip("s").endswith("env")


def _is_secret_env_name(key: str) -> bool:
    """Whether a key of an environment mapping (``env``, ``extra_envs``, ``launch_env``) names a secret: also every
    name Hyperloom masks in one."""
    return is_secret_key(key) or is_credential_name(key) or is_secret_shaped_env_name(key)


def redact(value: Any) -> Any:
    """Return a copy of *value* with secrets masked, recursing into dicts and lists."""
    return _redact(value, is_secret_key)


def _redact(value: Any, names_secret: Callable[[str], bool]) -> Any:
    """:func:`redact`, with *names_secret* telling which keys of a dict here hold a secret."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and names_secret(k) and isinstance(v, str) and v:
                out[k] = MASK
            else:
                out[k] = _redact(v, _is_secret_env_name if is_env_field(k) else is_secret_key)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(v, names_secret) for v in value]
    return value
