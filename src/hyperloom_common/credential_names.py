# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which environment-variable names hold a credential, so a record that leaves the process keeps none in the clear."""

from __future__ import annotations

import re

__all__ = ["SENSITIVE_ENV_NAME_MARKERS", "env_name_spelled_by", "is_credential_env_name"]

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

# Knobs whose names carry a credential marker in a phrase that can also name one (an auth token's cache, a header that
# carries Authorization), matched whole in the form :func:`env_name_spelled_by` gives.
_CREDENTIAL_MARKED_KNOBS: frozenset[str] = frozenset(
    {"SGLANG_CHAT_SEGMENT_TOKEN_CACHE", "SGLANG_CHAT_SEGMENT_TOKEN_CACHE_MAX_TOKENS", "ALLOWED_HEADERS"}
)
# As a substring TOKEN also hits counts and tokenizers (``MAX_NUM_BATCHED_TOKENS``, ``TOKENIZERS_PARALLELISM``), so it
# names a credential only where a word ends in it (``HF_TOKEN``, ``HFTOKEN``, ``GITHUB_TOKEN_BACKUP``), and never in a
# phrase that sizes, selects or balances tokens (``SGLANG_USE_AITER_FP8_PER_TOKEN``, ``--long-prefill-token-threshold``,
# ``SGLANG_SIMULATE_ACC_TOKEN_MODE``, ``--prefill-delayer-token-usage-low-watermark``, ``DEEPEP_COMBINE_TOKEN_BALANCED``).
_SUBSTRING_MARKERS_EXCEPT_TOKEN: tuple[str, ...] = tuple(m for m in SENSITIVE_ENV_NAME_MARKERS if m != "TOKEN")
_TOKEN_WORD_END_RE = re.compile(r"TOKEN(?=_|$)")
_TOKEN_KNOB_PHRASE_RE = re.compile(
    r"(?:^|_)(?:PER_TOKEN|TOKEN_(?:THRESHOLD|NUM|BUDGETS?|PROB|IDS|MAP|MODE|REUSE|USAGE|BALANCED))(?=_|$)"
)
_KEY_LAST_SEGMENT_RE = re.compile(r"(?:^|_)KEY$")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def env_name_spelled_by(name: object) -> str:
    """The env name an env name or a flag spells: ``--hf-auth`` and ``--hfAuth`` as ``HF_AUTH``."""
    text = str(name or "").strip().lstrip("-")
    return _CAMEL_BOUNDARY_RE.sub("_", text).replace("-", "_").upper()


def is_credential_env_name(name: object) -> bool:
    """Whether an env name, or the flag spelling one, holds a credential rather than a launch knob.

    A flag is compared as the env name it spells, so an env and a flag of the same name are never classified apart.
    """
    upper = env_name_spelled_by(name)
    if upper in _CREDENTIAL_MARKED_KNOBS:
        return False
    if any(marker in upper for marker in _SUBSTRING_MARKERS_EXCEPT_TOKEN) or _KEY_LAST_SEGMENT_RE.search(upper):
        return True
    return bool(_TOKEN_WORD_END_RE.search(_TOKEN_KNOB_PHRASE_RE.sub("", upper)))
