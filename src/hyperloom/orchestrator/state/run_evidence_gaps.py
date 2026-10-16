# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gaps shaped like a cause, derived from what a run was measured doing.

The rest of the gap vocabulary is shaped like an outcome -- short of target, crashed, no promotion for N rounds --
which tells a session it is stuck without telling it what it is stuck on. The censuses beside this module measure
causes: a kernel recompiled once per call shape, a launch that could fold into its neighbour, a step class that
never reached a graph, a prompt tokenised again on every turn. On GLM-5.3 those four carried the hand-tuned
stack's largest single steps, and not one of them has an outcome a session would have noticed.

Each gap names what was measured and what it costs, so a specialist is pointed at a thing to change rather than at
a number to improve.
"""

from __future__ import annotations

import logging
from hashlib import sha1
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.session_paths import jit_cache_dir
from hyperloom.orchestrator.measurement.host_input_cost import input_cost
from hyperloom.orchestrator.measurement.jit_variants import variant_counts
from hyperloom.orchestrator.state._shared_state.phase_state import GAP_SEVERITY_HIGH
from kernelforge.foldable_launches import foldable_launches
from kernelforge.graph_coverage import graph_coverage
from kernelforge.trace_events import resolve_capture

log = logging.getLogger(__name__)

#: The specialist domains these censuses hand their findings to. A gap names a
#: domain the dispatcher knows, or nothing claims it.
KERNEL_DOMAIN: str = "kernel_switch_specialist"
SERVING_DOMAIN: str = "serving_specialist"

#: The lane each domain's work belongs to, which is what a phase reads to decide it has something to do.
_LANE_OF_DOMAIN: dict[str, str] = {KERNEL_DOMAIN: "kernel_agent", SERVING_DOMAIN: "framework"}

#: Gaps one census may raise, so one pathological run cannot bury every other signal.
_PER_CENSUS_CAP: int = 3

#: Launch cost per step, in milliseconds, below which folding a kernel away is not worth a specialist's round.
_FOLD_MIN_MS_PER_STEP: float = 0.2


def _gap(kind: str, key: str, symptom: str, *, severity: str, domain_hint: str) -> dict[str, Any]:
    """One cause-shaped gap row, keyed so repeated refreshes upsert rather than multiply.

    ``layer`` names the lane that can act, which is what the phase prompts read
    when they ask whether a lane has anything to do; a census that measured
    kernel work has to say ``kernel_agent`` there or the phase sees no lever and
    closes. Where the evidence came from stays on ``source`` / ``provenance``.
    """
    return {
        "canonical_id": f"gap.run_evidence.{kind}.{sha1(key.encode()).hexdigest()[:16]}",
        "symptom": symptom,
        "layer": _LANE_OF_DOMAIN[domain_hint],
        "severity": severity,
        "domain_hint": domain_hint,
        "source": "run_evidence",
        "provenance": kind,
    }


def _recompile_gaps(session_dir: Any) -> list[dict[str, Any]]:
    """Kernels compiled once per call shape, which no kernel rewrite can recover."""
    out: list[dict[str, Any]] = []
    for row in variant_counts(jit_cache_dir(Path(str(session_dir or ".")))):
        if not row.get("single_constant_leak"):
            continue
        widest = (row.get("varying") or [{}])[0]
        out.append(
            _gap(
                "recompile",
                row["kernel"],
                f"{row['kernel']} was compiled into {row['variants']} variants, "
                f"{widest.get('distinct_values', '?')} of them differing only after `_{widest.get('after', '?')}_`: "
                "a constant is tracking a runtime quantity, and every new call shape stalls the engine to compile. "
                "Making that argument runtime rather than constexpr removes the recompiles without changing what "
                "the kernel computes.",
                severity=GAP_SEVERITY_HIGH,
                domain_hint=KERNEL_DOMAIN,
            )
        )
        if len(out) >= _PER_CENSUS_CAP:
            break
    return out


def _foldable_gaps(trace_path: Any) -> list[dict[str, Any]]:
    """Launches that cost more than the work they do, with the neighbour each could fold into."""
    out: list[dict[str, Any]] = []
    for window_class, rows in foldable_launches(trace_path).items():
        for row in rows:
            neighbour = row.get("after") or row.get("before")
            if not neighbour or row["ms_per_step"] < _FOLD_MIN_MS_PER_STEP:
                continue
            out.append(
                _gap(
                    "foldable_launch",
                    f"{window_class}::{row['kernel']}",
                    f"In {window_class}, {row['kernel']} runs {row['launches_per_step']:.0f} times per step at "
                    f"{row['us_per_call']:.1f} us a call ({row['ms_per_step']:.2f} ms per step), always beside "
                    f"{neighbour['name']} ({neighbour['share']:.0%}). Folding it into that neighbour removes a "
                    "launch per call without changing what is computed.",
                    severity="medium",
                    domain_hint=KERNEL_DOMAIN,
                )
            )
            if len(out) >= _PER_CENSUS_CAP:
                return out
    return out


def _graph_gaps(trace_path: Any) -> list[dict[str, Any]]:
    """Step classes paying a launch for every kernel they run."""
    out: list[dict[str, Any]] = []
    for window_class, row in graph_coverage(trace_path).items():
        if not row.get("ungraphed"):
            continue
        out.append(
            _gap(
                "ungraphed_step",
                window_class,
                f"{window_class} runs {row['eager_time_pct']:.0f}% of its kernel time eagerly, "
                f"{row['eager_kernels_per_step']:.0f} launches per step, while other classes in the same run "
                "replay a captured graph. Establish whether the backend refuses to capture this class on this "
                "platform or whether the module carries nothing for the capture to walk; the two have different "
                "fixes.",
                severity=GAP_SEVERITY_HIGH,
                domain_hint=SERVING_DOMAIN,
            )
        )
        if len(out) >= _PER_CENSUS_CAP:
            break
    return out


def _input_cost_gaps(workspace: Any) -> list[dict[str, Any]]:
    """Time paid per input token in front of the engine, where a cached prefix should cost nothing."""
    cost = input_cost(workspace)
    if not cost or not cost.get("flagged"):
        return []
    return [
        _gap(
            "front_end_input_cost",
            "ttft_per_input_token",
            f"Time-to-first-token grows {cost['ms_per_100k_tokens']:.0f} ms per 100k input tokens "
            f"({cost['low_ttft_ms']:.0f} ms at {cost['low_isl'] // 1000}k, {cost['high_ttft_ms']:.0f} ms at "
            f"{cost['high_isl'] // 1000}k). On a cached prefix the engine does little work for those tokens, so "
            "the cost is in front of it: rendering the template, tokenising the whole prompt again, pickling it "
            "to the scheduler.",
            severity=GAP_SEVERITY_HIGH,
            domain_hint=SERVING_DOMAIN,
        )
    ]


def run_evidence_gaps(*, session_dir: Any, trace_path: Any, workspace: Any) -> list[dict[str, Any]]:
    """Every cause-shaped gap this run's own evidence supports.

    Each census reads its own artifact and is kept to it: a run whose JIT cache
    a later process left unreadable still has a trace to count launches in, and
    the caller catches ``OSError`` around the whole call, so letting one census
    raise would quietly cost the other three.
    """
    capture = resolve_capture(trace_path)
    gaps: list[dict[str, Any]] = []
    for name, census in (
        ("recompile", lambda: _recompile_gaps(session_dir)),
        ("graph_coverage", lambda: _graph_gaps(capture)),
        ("front_end_input_cost", lambda: _input_cost_gaps(workspace)),
        ("foldable_launch", lambda: _foldable_gaps(capture)),
    ):
        try:
            gaps.extend(census())
        except OSError:
            log.warning("run evidence: the %s census could not read its artifact", name, exc_info=True)
    return gaps


__all__ = ["run_evidence_gaps"]
