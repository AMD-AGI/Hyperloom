# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which environment-variable names hold a credential, so a record that leaves the process keeps none in the clear."""

from __future__ import annotations

import re

__all__ = ["SENSITIVE_ENV_NAME_MARKERS", "is_credential_env_name", "is_credential_flag"]

# Env-var name fragments whose value is masked before an env snapshot leaves this process. Shared with the Langfuse
# session_start snapshot, which matches every marker as a substring.
SENSITIVE_ENV_NAME_MARKERS: tuple[str, ...] = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "SECRET_KEY",
    "AUTH",
    "SIGNATURE",
    "HEADERS",
    "CUSTOM_HEADERS",
)

# As a substring TOKEN also hits tuning knobs (``MAX_NUM_BATCHED_TOKENS``, ``TOKENIZERS_PARALLELISM``), so a record of
# launch knobs treats it as a credential only as a whole name segment (``HF_TOKEN``, ``HF_TOKEN_2``,
# ``GITHUB_TOKEN_BACKUP``), and never in ``PER_TOKEN`` (``SGLANG_USE_AITER_FP8_PER_TOKEN``).
_SUBSTRING_MARKERS_EXCEPT_TOKEN: tuple[str, ...] = tuple(m for m in SENSITIVE_ENV_NAME_MARKERS if m != "TOKEN")
_TOKEN_SEGMENT_RE = re.compile(r"(?:^|_)TOKEN(?=_|$)")
_PER_TOKEN_SEGMENT_RE = re.compile(r"(?:^|_)PER_TOKEN(?=_|$)")


def is_credential_env_name(key: object) -> bool:
    """Whether a variable named ``key`` holds a credential rather than a launch knob."""
    upper = str(key or "").strip().upper()
    if any(marker in upper for marker in _SUBSTRING_MARKERS_EXCEPT_TOKEN):
        return True
    return bool(_TOKEN_SEGMENT_RE.search(_PER_TOKEN_SEGMENT_RE.sub("", upper)))


def is_credential_flag(flag: str) -> bool:
    """Whether the command-line option ``flag`` (``--api-key``, ``--hf-token=...``) is named like a credential."""
    return is_credential_env_name(flag.lstrip("-").partition("=")[0].replace("-", "_"))
