# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which EXPLORE launches re-measure a config on purpose, so no filter may drop them."""

from __future__ import annotations

from typing import Any

STACK_REVALIDATE_SOURCE: str = "stack_revalidate"

_CONFIG_REPLAY_PROVENANCE = frozenset({"geak_revalidate"})


def is_stack_revalidation(params: dict | None) -> bool:
    return str((params or {}).get("source") or "") == STACK_REVALIDATE_SOURCE


def _is_config_replay_variant(variant: Any) -> bool:
    """Whether a variant replays an already-validated config verbatim."""
    return str(getattr(variant, "provenance", "") or "").strip() in _CONFIG_REPLAY_PROVENANCE
