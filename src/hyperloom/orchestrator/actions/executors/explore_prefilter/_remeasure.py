# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which EXPLORE launches re-measure a config on purpose, so no filter may drop them."""

from __future__ import annotations

from typing import Any

STACK_REVALIDATE_SOURCE: str = "stack_revalidate"

_CONFIG_REPLAY_PROVENANCE = frozenset({"geak_revalidate"})

# Separate from _CONFIG_REPLAY_PROVENANCE (shared with filter_operator_pinned_envs):
# a revalidation re-measures its own config, so it must always run.
_NOOP_FILTER_ALWAYS_RUNS = frozenset({"resume_stack_revalidate"})


def is_stack_revalidation(params: dict | None) -> bool:
    return str((params or {}).get("source") or "") == STACK_REVALIDATE_SOURCE


def _provenance(variant: Any) -> str:
    return str(getattr(variant, "provenance", "") or "").strip()


def _is_config_replay_variant(variant: Any) -> bool:
    """Whether a variant replays an already-validated config verbatim."""
    return _provenance(variant) in _CONFIG_REPLAY_PROVENANCE


def _is_noop_filter_exempt(variant: Any) -> bool:
    """Whether ``variant`` must skip the baseline-noop probe outright and always run."""
    return _is_config_replay_variant(variant) or _provenance(variant) in _NOOP_FILTER_ALWAYS_RUNS
