"""Roofline-aware variant filter (opt-in, off by default).

Today the EXPLORE phase fans every specialist / default-grid variant out
to a full Magpie benchmark, even when the most recent roofline snapshot
already shows the variant's target direction is saturated above the
diminishing-returns threshold. On slow workloads (e.g. Qwen3-32B TP=1
BF16, where decode-side throughput is ~1.5 % of HBM-bound peak and the
model is *bandwidth-bound*, not host-bound) this means burning ~70 min
per variant on host-overhead reducers that physically cannot help.

This module categorizes variants by the roofline direction(s) their flags
target and provides a filter that drops a variant only when ALL its
target directions are above the saturation threshold. Conservatively:

* Uncategorized variants (flags this module doesn't recognise) are kept —
  we can't say they're wasted.
* A variant that targets multiple directions is kept if **any** of them
  is below the threshold (it might still help via the non-saturated one).
* When no direction in the snapshot crosses the threshold, no filtering
  happens.

The filter is opt-in via ``--explore-roofline-hard-gate`` /
``SharedState.explore_roofline_hard_gate``. The existing
``roofline_saturation_advisory`` (soft prompt hint) is untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Categorization table — variant flag/env → roofline direction(s) it targets.
# ---------------------------------------------------------------------------
# Direction names match ``roofline_snapshot._SATURATION_LABEL_MAP``:
# ``compute`` / ``memory`` / ``host_overhead`` / ``comm``.
#
# Conservative defaults:
# * Each entry is a regex against the variant's ``extra_sglang_args`` string.
# * Single-direction entries are high-confidence (e.g. host-only knobs).
# * Multi-direction entries cover knobs that legitimately affect more than
#   one direction (e.g. ``--attention-backend`` swaps the kernel **and**
#   its memory access pattern). The filter keeps the variant unless EVERY
#   listed direction is saturated, so multi-direction tags are
#   intentionally lenient.
# * Anything not matched here stays in the "uncategorized" bucket and the
#   filter never drops it.
#
# Ownership / update protocol: any new sglang flag that targets a known
# roofline direction MUST be added here, otherwise it falls into the
# "uncategorized -> keep" bucket above and the hard-gate filter is a no-op
# for it. Flag rot drifts toward "keep" (safe, but defeats the gate).
_FLAG_TO_DIRECTIONS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    # Host-overhead reducers (CPU scheduling / kernel launch overhead /
    # CUDA graph replay).
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
    # Compute / fusion (torch.compile-style, speculative decoding).
    (re.compile(r"(?<!\S)--enable-torch-compile(?!\S)"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--torch-compile-max-bs(?!\S)"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--enable-spec-v2(?!\S)"),
        frozenset({"compute"})),
    (re.compile(r"(?<!\S)--speculative-"),
        frozenset({"compute"})),
    # Memory / KV cache.
    (re.compile(r"(?<!\S)--disable-radix-cache(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--mem-fraction-static(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--max-running-requests(?!\S)"),
        frozenset({"memory"})),
    (re.compile(r"(?<!\S)--kv-cache-dtype(?!\S)"),
        frozenset({"memory"})),
    # Attention backend swaps both kernel and memory access pattern;
    # treat as multi-direction (lenient).
    (re.compile(r"(?<!\S)--attention-backend(?!\S)"),
        frozenset({"compute", "memory"})),
)

# Env-var prefix → directions. Used in addition to flag matching.
_ENV_PREFIX_TO_DIRECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    # Tile-Lang / fused-MoE / GEMM-tuning envs all target the compute
    # roofline. SGLANG_HACK_FLASHMLA_BACKEND, SGLANG_OPT_USE_TILELANG_*,
    # AITER_USE_*, TRITON_*, HIPBLASLT_*, PYTORCH_TUNABLEOP_* etc.
    ("AITER_", frozenset({"compute"})),
    ("TRITON_", frozenset({"compute"})),
    ("HIPBLASLT_", frozenset({"compute"})),
    ("PYTORCH_TUNABLEOP_", frozenset({"compute"})),
)

# Specific env-var → directions. Pattern-match the env *value* too when
# we need to disambiguate (e.g. SGLANG_OPT_USE_MULTI_STREAM_OVERLAP toggles
# host_overhead, not compute, even though it's a SGLANG_ env).
_ENV_NAME_TO_DIRECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("SGLANG_OPT_USE_MULTI_STREAM_OVERLAP", frozenset({"host_overhead"})),
    ("SGLANG_OPT_USE_TILELANG_INDEXER", frozenset({"compute"})),
    ("SGLANG_HACK_FLASHMLA_BACKEND", frozenset({"compute"})),
    ("SGLANG_ENABLE_SPEC_V2", frozenset({"compute"})),
)


_DEFAULT_SATURATION_THRESHOLD_PCT = 80.0


def categorize_variant(
    extra_args: str | None,
    extra_envs: dict[str, str] | None,
) -> frozenset[str]:
    """Return the set of roofline directions a variant's flags target.

    An empty set means the variant is uncategorized; the caller (the
    filter) treats those as "potentially useful, keep". Knowing more than
    one direction is normal — a variant can stack flags from multiple
    categories.
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


@dataclass(frozen=True)
class _DroppedVariant:
    name: str
    extra_sglang_args: str
    categories: tuple[str, ...]
    saturated_directions: tuple[str, ...]
    reason: str


def filter_variants_by_roofline(
    grid: Iterable[Any],
    saturation_snapshot: dict[str, float] | None,
    *,
    threshold_pct: float = _DEFAULT_SATURATION_THRESHOLD_PCT,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Apply the opt-in saturation filter to an explore grid.

    Args:
        grid: Iterable of ``GridVariant``-shaped objects (anything with
            ``.name``, ``.extra_sglang_args``, ``.extra_envs`` attributes).
            We don't import ``GridVariant`` here to keep this module
            decoupled from ``_grid_runner`` and trivially unit-testable.
        saturation_snapshot: Mapping ``direction -> percent`` from the
            most recent roofline run (``SharedState.roofline_saturation_history
            [-1]``). When ``None`` / empty / no direction crosses the
            threshold, the filter is a no-op.
        threshold_pct: Saturation cutoff. Defaults to the same 80 % the
            soft advisory uses (single source of truth).

    Returns:
        ``(kept, dropped)`` — ``kept`` preserves the input order; ``dropped``
        is a list of JSON-friendly dicts the caller can surface in
        ``state.json`` / the LLM prompt for traceability.
    """
    grid_list = list(grid)
    if not saturation_snapshot:
        return grid_list, []
    saturated: set[str] = {
        direction
        for direction, pct in saturation_snapshot.items()
        if isinstance(pct, (int, float)) and float(pct) >= threshold_pct
    }
    if not saturated:
        return grid_list, []

    kept: list[Any] = []
    dropped: list[dict[str, Any]] = []
    for gv in grid_list:
        cats = categorize_variant(
            getattr(gv, "extra_sglang_args", ""),
            getattr(gv, "extra_envs", None),
        )
        # Uncategorized → keep (unknown intent is unknown reward).
        if not cats:
            kept.append(gv)
            continue
        # All target directions saturated → drop.
        if cats <= saturated:
            dropped.append({
                "name": getattr(gv, "name", "?"),
                "extra_sglang_args": getattr(gv, "extra_sglang_args", ""),
                "categories": sorted(cats),
                "saturated_directions": sorted(saturated),
                "reason": "all_target_directions_saturated",
            })
        else:
            kept.append(gv)
    return kept, dropped


__all__ = (
    "categorize_variant",
    "filter_variants_by_roofline",
)
