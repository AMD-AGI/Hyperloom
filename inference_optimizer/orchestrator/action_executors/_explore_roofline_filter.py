# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline-aware advisory annotator for the EXPLORE grid.

Advisory only (no longer a hard filter — the LLM / budget / overtime kill
decide what runs): :func:`categorize_variant` maps a variant's flags + envs
to the roofline directions it targets, and :func:`compute_saturation_advisory`
returns annotations the executor surfaces (``roofline_advisory``); nothing is
dropped. Unrecognised flags land in the empty-set bucket (no advisory).
"""

from __future__ import annotations

import re
from typing import Any, Iterable


# Direction names match ``roofline_snapshot._SATURATION_LABEL_MAP``:
# ``compute`` / ``memory`` / ``host_overhead`` / ``comm``.
_FLAG_TO_DIRECTIONS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (re.compile(r"(?<!\S)--num-continuous-decode-steps(?!\S)"),
        frozenset({"host_overhead"})),
    (re.compile(r"(?<!\S)--scheduler-recv-interval(?!\S)"),
        frozenset({"host_overhead"})),
    (re.compile(r"(?<!\S)--cuda-graph-max-bs(?!\S)"),
        frozenset({"host_overhead"})),
    (re.compile(r"(?<!\S)--enable-cuda-graph(?!\S)"),
        frozenset({"host_overhead"})),
    (re.compile(r"(?<!\S)--tokenizer-worker-num(?!\S)"),
        frozenset({"host_overhead"})),
    (re.compile(r"(?<!\S)--stream-interval(?!\S)"),
        frozenset({"host_overhead"})),
    (re.compile(r"(?<!\S)--enable-torch-compile(?!\S)"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--torch-compile-max-bs(?!\S)"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--enable-spec-v2(?!\S)"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--speculative-"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--disable-radix-cache(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--mem-fraction-static(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--max-running-requests(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--kv-cache-dtype(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--attention-backend(?!\S)"),
        frozenset({"compute", "memory"})),
)

_ENV_PREFIX_TO_DIRECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("AITER_", frozenset({"compute"})),
    ("TRITON_", frozenset({"compute"})),
    ("HIPBLASLT_", frozenset({"compute"})),
    ("PYTORCH_TUNABLEOP_", frozenset({"compute"})),
)

_ENV_NAME_TO_DIRECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("SGLANG_OPT_USE_MULTI_STREAM_OVERLAP", frozenset({"host_overhead"})),
    ("SGLANG_OPT_USE_TILELANG_INDEXER", frozenset({"compute"})),
    ("SGLANG_HACK_FLASHMLA_BACKEND", frozenset({"compute"})),
    ("SGLANG_ENABLE_SPEC_V2", frozenset({"compute"})),
)


DEFAULT_SATURATION_THRESHOLD_PCT = 80.0


def categorize_variant(
    extra_args: str | None,
    extra_envs: dict[str, str] | None,
) -> frozenset[str]:
    """Return the set of roofline directions a variant's flags target.

    An empty set means the variant is uncategorized (no advisory).

    Args:
        extra_args: Extra server-arg string for the variant, or None.
        extra_envs: Mapping of extra environment variables, or None.

    Returns:
        Frozenset of roofline direction names the variant targets.
    """
    cats: set[str] = set()
    args = (extra_args or "").strip()
    if args:
        for pat, dirs in _FLAG_TO_DIRECTIONS:
            if pat.search(args):
                cats |= dirs
    if extra_envs:
        for raw_key in extra_envs.keys():
            key = str(raw_key).upper()
            for env_name, dirs in _ENV_NAME_TO_DIRECTIONS:
                if key == env_name:
                    cats |= dirs
                    break
            else:
                for prefix, dirs in _ENV_PREFIX_TO_DIRECTIONS:
                    if key.startswith(prefix):
                        cats |= dirs
                        break
    return frozenset(cats)


def compute_saturation_advisory(
    grid: Iterable[Any],
    saturation_snapshot: dict[str, float] | None,
    *,
    threshold_pct: float = DEFAULT_SATURATION_THRESHOLD_PCT,
) -> list[dict[str, Any]]:
    """Return advisory annotations for variants likely to be saturated.

    Args:
        grid: Iterable of ``GridVariant``-shaped objects (must expose
            ``name``, ``extra_server_args``, ``extra_envs``).
        saturation_snapshot: Mapping ``direction -> percent`` from the
            most recent roofline run.
        threshold_pct: Saturation cutoff.

    Returns:
        Per-variant advisory dicts ``{name, extra_server_args, categories,
        saturated_directions, reason}``. Empty when no snapshot, no direction
        crosses the threshold, or no categorized variant targets a saturated
        direction. Advisory only — the caller never drops variants on this.
    """
    if not saturation_snapshot:
        return []
    saturated: set[str] = {
        direction
        for direction, pct in saturation_snapshot.items()
        if isinstance(pct, (int, float)) and float(pct) >= threshold_pct
    }
    if not saturated:
        return []
    advisory: list[dict[str, Any]] = []
    for gv in grid:
        variant_args = getattr(gv, "extra_server_args", None)
        if variant_args is None:
            variant_args = getattr(gv, "extra_sglang_args", "")
        cats = categorize_variant(
            variant_args,
            getattr(gv, "extra_envs", None),
        )
        if not cats:
            continue
        if cats <= saturated:
            advisory.append({
                "name": getattr(gv, "name", "?"),
                "extra_server_args": variant_args or "",
                "categories": sorted(cats),
                "saturated_directions": sorted(saturated),
                "reason": "likely_saturated",
            })
    return advisory


__all__ = (
    "DEFAULT_SATURATION_THRESHOLD_PCT",
    "categorize_variant",
    "compute_saturation_advisory",
)
