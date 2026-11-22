# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens analysis settings of a PD-disaggregated profile's prefill and decode traces."""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.trace_analysis import DECODE_ONLY_OR_GENERIC_MODE, UNSPLIT_MODE

# The TraceLens input each PD role's trace yields: a decode engine's trace holds decode steps only, so its steady-state
# chunk is taken with or without the decode-only label; a prefill engine's few steps are analysed whole.
ROLE_STEADY_STATE_MODE = {"prefill": UNSPLIT_MODE, "decode": DECODE_ONLY_OR_GENERIC_MODE}


def pd_multi_trace_input(profile_result: dict[str, Any]) -> dict[str, Any] | None:
    """The multi-trace analysis input of a PD profile with decode nodes and two or more nodes in all, else ``None``."""
    node_traces = profile_result.get("role_node_traces")
    if not isinstance(node_traces, dict) or not node_traces.get("decode"):
        return None
    if sum(len(paths or []) for paths in node_traces.values()) < 2:
        return None
    return {"role_node_traces": node_traces}
