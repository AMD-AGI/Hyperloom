# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mask credentials in launch records that are written to a group-readable file."""

from __future__ import annotations

import re
from typing import Any

from hyperloom.common.env_safety import redact_secret_env_values, redact_secret_values

_SECRET_FLAG_VALUE_RE = re.compile(
    r"(?i)(--[a-z0-9_-]*(?:api[-_]?key|token|secret|password|passwd|credential)[a-z0-9_-]*)(=|\s+)(\S+)"
)
_ARGS_KEYS = frozenset({"requested_server_args", "operator_server_args", "server_args"})
_ENV_KEYS = frozenset({"requested_server_env"})


def mask_server_args(text: Any) -> Any:
    """``server_args`` with the value of any credential-named flag and any recognizable credential masked."""
    if not isinstance(text, str):
        return text
    return redact_secret_values(_SECRET_FLAG_VALUE_RE.sub(r"\1\2[REDACTED]", text))


def mask_launch_secrets(node: Any) -> Any:
    """A copy of ``node`` with every recorded server-args string and requested server env masked, at any depth."""
    if isinstance(node, dict):
        return {
            key: mask_server_args(value)
            if key in _ARGS_KEYS and isinstance(value, str)
            else redact_secret_env_values(value)
            if key in _ENV_KEYS and isinstance(value, dict)
            else mask_launch_secrets(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [mask_launch_secrets(item) for item in node]
    return node
