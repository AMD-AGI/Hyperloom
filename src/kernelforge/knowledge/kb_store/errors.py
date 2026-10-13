# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bound and redact KB Store failures before they cross a runtime boundary."""

from __future__ import annotations

import re
from typing import Any

from kernelforge.knowledge.kb_store.config import knowledge_config_from_runtime

_MAX_ERROR_LENGTH = 240
_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[^\s,;}\]]+")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(token|password|secret|credential|authorization|api[_-]?key)"
    r"(\s*[:=]\s*)[^\s,;}\]]+"
)
_URL_CREDENTIAL_RE = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


def sanitize_read_error(exc: Exception, *, secrets: tuple[str, ...] = ()) -> str:
    """Return a bounded exception summary with credential-like values redacted."""
    message = f"{type(exc).__name__}: {exc}"
    message = _BEARER_SECRET_RE.sub("Bearer [REDACTED]", message)
    message = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        message,
    )
    message = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", message)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:_MAX_ERROR_LENGTH]


def kb_store_secrets(config: Any) -> tuple[str, ...]:
    """Return credentials that must be removed from a store failure."""
    knowledge = knowledge_config_from_runtime(config)
    return tuple(value for value in (knowledge.kb_store_token,) if value)


__all__ = ["kb_store_secrets", "sanitize_read_error"]
