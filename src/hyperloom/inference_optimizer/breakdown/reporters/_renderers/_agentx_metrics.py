# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared rendering for the AgentX graded axes.

One copy, used by every section that shows a measured round, so the axes a report shows are the axes a verdict
reads: a session graded on median interactivity reports that axis, not only the output-throughput figure it is not
ranked on.
"""

from __future__ import annotations

import math
from typing import Any

from ..base import as_dict, md_kv_list

#: ``perf`` key -> label, ordered as a reader needs them rather than alphabetically: the objective, then the two
#: guards that can veto it, then the comparability inputs a pair is refused on, then the latency detail, then what
#: is carried for continuity alone.
#:
#: Labels repeat the key instead of prettifying it. The key is what the operator greps for in the JSON and what the
#: policy names, so a prettier label would mean the report and the artifact disagree about what a number is called.
_GRADED_ROWS: tuple[tuple[str, str], ...] = (
    ("e2e_norm_intvty_p50", "e2e_norm_intvty_p50 (objective)"),
    ("e2e_norm_intvty_p90", "e2e_norm_intvty_p90 (guard, frontier x)"),
    ("output_tput_per_gpu", "output_tput_per_gpu (guard, frontier y)"),
    ("ttft_p50_ms", "ttft_p50_ms"),
    ("ttft_p90_ms", "ttft_p90_ms"),
    ("tpot_p50_ms", "tpot_p50_ms"),
    ("tpot_p90_ms", "tpot_p90_ms"),
    ("duration_seconds", "duration_seconds (comparability)"),
    ("request_error_rate", "request_error_rate (comparability)"),
    ("total_throughput", "total_throughput (reported, not graded)"),
    ("input_throughput", "input_throughput (reported, not graded)"),
)

#: The three the frontier is read on. Promoted to key facts because a reader skimming the section should not have to
#: open the table to learn where the round sits.
_HEADLINE = ("e2e_norm_intvty_p50", "e2e_norm_intvty_p90", "output_tput_per_gpu")

#: What makes a round AgentX-graded, and it is only these two. Every other row in the table is filled for an
#: ordinary measurement as well: ``benchmark_result._merge_raw_result`` fills ``duration_seconds`` from raw
#: ``duration``, the four latency percentiles from ``median_ttft_ms`` / ``p90_ttft_ms`` / ``median_tpot_ms`` /
#: ``p90_tpot_ms``, and ``request_error_rate`` whenever the raw result carries it, none of them gated on the
#: workload; ``output_tput_per_gpu`` is stamped unconditionally by ``writeback._promote_baseline``; and total and
#: input throughput are on every report. The interactivity pair is different because its only producer is
#: ``agentx/mapping.py``, so a value there means the agentic mapper ran.
#:
#: Testing the wider set headed a section "AgentX graded axes" over a synthetic SGLang session that merely reported
#: a duration -- while the change that added the section claimed it appears on no synthetic session.
_DECIDING_AXES = ("e2e_norm_intvty_p50", "e2e_norm_intvty_p90")


def has_graded_axes(perf: Any) -> bool:
    """Whether *perf* carries an axis that exists only when the AgentX verdict ran.

    A synthetic round carries the graded axes as nulls, and rendering the block for it would claim the session was
    graded on axes it never had; emptiness is how a non-AgentX round says so. Deliberately narrower than the table
    it gates: a row being *displayed* beside the objective does not make it evidence the objective was measured.
    """
    axes = as_dict(perf)
    return any(axes.get(key) is not None for key in _DECIDING_AXES)


def graded_axes_facts(perf: Any, *, label: str) -> list[str]:
    """One fact per frontier axis the round measured, prefixed by *label*."""
    axes = as_dict(perf)
    facts: list[str] = []
    for key in _HEADLINE:
        value = axes.get(key)
        # ``isfinite`` for the same reason ``_md_cell`` guards it: a hand-edited artifact can carry NaN, and a fact
        # line reading "nan" beside a table cell reading "—" would be two answers to one question.
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            facts.append(f"{label} {key}: {float(value):.4g}.")
    return facts


def render_graded_axes(perf: Any) -> str:
    """The graded axes as a KV block; empty when the round measured none of them."""
    if not has_graded_axes(perf):
        return ""
    axes = as_dict(perf)
    return md_kv_list([(label, axes.get(key)) for key, label in _GRADED_ROWS])
