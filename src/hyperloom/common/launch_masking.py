# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mask credentials in server-args text that is written to a group-readable file."""

from __future__ import annotations

import re
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

# A credential flag is matched by its whole name: it ends in api-key / token / secret / password, so
# ``--max-total-tokens`` and ``--tokenizer-path`` are not credentials. A value that is itself a flag is not consumed.
_SECRET_FLAG_RE = re.compile(
    r"(?i)(?P<flag>--(?:[a-z0-9]+[-_])*(?:api[-_]?key|token|secret|password|passwd|credential)(?![a-z0-9_-]))"
    r"(?:=\S*|(?P<gap>\s+)(?!--)\S+)"
)


def _mask_flag(match: re.Match[str]) -> str:
    gap = match.group("gap")
    return f"{match.group('flag')}{gap or '='}[REDACTED]"


def mask_server_args(text: Any) -> Any:
    """``server_args`` with the value of any credential-named flag and any recognizable credential masked."""
    if not isinstance(text, str):
        return text
    return redact_secret_values(_SECRET_FLAG_RE.sub(_mask_flag, text))
