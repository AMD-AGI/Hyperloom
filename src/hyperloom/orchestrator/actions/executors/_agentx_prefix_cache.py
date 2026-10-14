# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Refuse an AgentX baseline whose server ran a prefix-heavy replay with the prefix cache off."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

#: ``theoretical_prefix_cache_hit`` (percent) at or above which a replay is prefix-heavy. AgentX's canonical
#: corpora sit near 97; the synthetic random-token workload reports nothing at all.
PREFIX_HEAVY_HIT_PCT = 50.0

#: The ``error_class`` of a baseline refused for running its replay with the prefix cache off.
PREFIX_CACHE_OFF_ERROR_CLASS = "agentx_baseline_prefix_cache_off"

_CACHED_TOKEN_RE = re.compile(rb"#cached-token:\s*(\d+)")


def _as_pct(value: Any) -> float | None:
    """A theoretical prefix hit in percent, from either unit.

    aiperf reports a percentage (96.8 on the 062126 corpus) while InferenceX's aggregate and Hyperloom's canonical
    corpus shape carry the fraction (0.975), so a value of at most 1 is read as a fraction; no AgentX corpus sits below
    a 1% hit.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) * 100.0 if value <= 1.0 else float(value)


def _theoretical_prefix_hit_pct(result: Mapping[str, Any]) -> float | None:
    """The replay's theoretical prefix hit in percent, wherever the launch path recorded it.

    The measurement's own field comes first; otherwise the raw result file, read at its top level (aiperf's mapped
    export) or under ``request_metrics.cache`` (InferenceX's aggregate).
    """
    pct = _as_pct(result.get("theoretical_prefix_cache_hit"))
    if pct is not None:
        return pct
    raw_path = str(result.get("raw_result_path") or "")
    if not raw_path:
        return None
    try:
        raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    pct = _as_pct(raw.get("theoretical_prefix_cache_hit"))
    if pct is not None:
        return pct
    metrics = raw.get("request_metrics")
    cache = metrics.get("cache") if isinstance(metrics, dict) else None
    return _as_pct(cache.get("theoretical_cache_hit_rate")) if isinstance(cache, dict) else None


def _server_reused_cached_tokens(server_log_path: str) -> bool | None:
    """Whether any SGLang prefill batch reused a cached token; ``None`` when the log names no prefill batch."""
    seen = False
    try:
        with Path(server_log_path).open("rb") as handle:
            for line in handle:
                for match in _CACHED_TOKEN_RE.finditer(line):
                    if int(match.group(1)) > 0:
                        return True
                    seen = True
    except OSError:
        return None
    return False if seen else None


def prefix_cache_off_evidence(result: Mapping[str, Any]) -> str:
    """Why this measurement ran a prefix-heavy replay with the prefix cache off, or ``""``.

    Two observations each settle it: the server's resolved ``disable_radix_cache``, and a server log whose prefill
    batches never reused a cached token. Both are read from what the server did, not from how it was launched, so the
    check holds whichever launcher started it. A run neither can be read from is left alone.
    """
    hit = _theoretical_prefix_hit_pct(result)
    if hit is None or hit < PREFIX_HEAVY_HIT_PCT:
        return ""
    evidence = result.get("launch_evidence") if isinstance(result.get("launch_evidence"), Mapping) else {}
    server_config = evidence.get("observed_server_config") if isinstance(evidence, Mapping) else None
    if isinstance(server_config, Mapping) and server_config.get("disable_radix_cache") is True:
        return f"the server ran with disable_radix_cache=True on a replay with {hit:.1f}% theoretical prefix hit"
    server_log = str(result.get("server_log_path") or "")
    if server_log and _server_reused_cached_tokens(server_log) is False:
        return f"no prefill batch reused a cached token on a replay with {hit:.1f}% theoretical prefix hit"
    return ""


def refuse_cacheless_agentx_baseline(ctx: Any, result: dict[str, Any]) -> None:
    """Fail a succeeded baseline that replayed a prefix-heavy corpus with the prefix cache off, and stop the run.

    The caller runs it only for a genuine baseline, the measurement every later candidate is graded against; a
    candidate validated through a ``baseline``-kind task (``quality_ref_exempt``) is measured as it ran. It stops on the
    first one rather than retrying, because the next attempt would launch the same server.
    """
    if result.get("status") != "succeeded":
        return
    evidence = prefix_cache_off_evidence(result)
    if not evidence:
        return
    result["status"] = "failed"
    result["error_class"] = PREFIX_CACHE_OFF_ERROR_CLASS
    result["error"] = (
        f"AgentX baseline refused: {evidence}. Every later candidate is graded against this baseline, so any "
        "change that turns the cache back on would be credited with the cache itself as a gain. Keep the radix "
        "cache on for agentic replays (no --disable-radix-cache in the server args), as upstream's agentic "
        "recipes do."
    )
    log.error("baseline_executor: %s", result["error"])
    shared_state = (getattr(ctx, "extra", None) or {}).get("shared_state")
    if shared_state is not None:
        shared_state.set_stop_reason("baseline_failed")
