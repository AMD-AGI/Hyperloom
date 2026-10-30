# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase chain: its identifiers and order, the actions each phase admits, and the rows that record a move."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloom.inference_optimizer.protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
)


# Phase identifiers + ordering (monotonic chain)
PHASE_PRELUDE = "PRELUDE"
PHASE_ENABLEMENT = "ENABLEMENT"
PHASE_FRAMEWORK_AGENT = "FRAMEWORK_AGENT"
PHASE_KERNEL_AGENT = "KERNEL_AGENT"
PHASE_SWEEP = "SWEEP"
PHASE_CLOSE = "CLOSE"

PHASE_NAMES: tuple[str, ...] = (
    PHASE_PRELUDE,
    PHASE_ENABLEMENT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_SWEEP,
    PHASE_CLOSE,
)
PHASE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES)}


def phase_index(phase: str) -> int:
    """Return monotonic index of ``phase`` (Inv-2.1 check); unknown → -1."""
    return PHASE_INDEX.get((phase or "").strip().upper(), -1)


# What ENABLEMENT runs only for its lane: the round's specialist and integrate, and the KEEP's revalidation.
_ENABLEMENT_LANE_ACTIONS = frozenset({"baseline", "specialist", "integrate_patch"})

# Phase transition survival filter: actions that survive across phase boundaries.
# LLM-proposable set is allowed_actions_for(phase); Coordinator-auto actions
# stay out of PROPOSABLE so the prompt does not invite LLM proposals for them.
PHASE_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    PHASE_PRELUDE: frozenset(
        {
            "target_analysis",
            "baseline",
            "roofline",
            "profile",
            # Coordinator-internal: replay a warm recipe from a prior session at PRELUDE entry.
            "replay_warm_recipe",
        }
    ),
    PHASE_ENABLEMENT: frozenset({"target_analysis", "roofline", "profile", "targeted_build"})
    | _ENABLEMENT_LANE_ACTIONS,
    # Three levers: configuration grids (``explore``), investigation and authoring (``specialist``), and landing a
    # patch from any source (``integrate_patch``).
    PHASE_FRAMEWORK_AGENT: frozenset(
        {
            "explore",
            "specialist",
            "integrate_patch",
            # roofline/profile auto-enqueued on the cumulative-gain watermark.
            "roofline",
            "profile",
        }
    ),
    # No specialist: KERNEL is a single-pipeline phase; a specialist dispatched here would occupy gpu_research_lane
    # that kernel_agent holds via benchmark_lane's expansion, and its authoring output has no path to integration.
    PHASE_KERNEL_AGENT: frozenset(
        {
            "integrate",
            "roofline",
            "profile",
            # Coordinator-internal: the phase's whole pipeline, enqueued once at entry.
            "kernel_agent",
        }
    ),
    # No specialist below: SWEEP is the validation window and CLOSE only reports.
    PHASE_SWEEP: frozenset(
        {
            # conc_sweep: Coordinator-internal CONC-ladder benchmark.
            "conc_sweep",
        }
    ),
    PHASE_CLOSE: frozenset(
        {
            "report",
            "session_breakdown",
        }
    ),
}


# Actions a single phase reserves for the Coordinator. COORDINATOR_INTERNAL_ACTIONS
# cannot say this: other phases let the agent propose them.
PHASE_COORDINATOR_RESERVED: dict[str, frozenset[str]] = {
    PHASE_ENABLEMENT: _ENABLEMENT_LANE_ACTIONS,
}


def allowed_actions_for(phase: str) -> tuple[str, ...]:
    """Return the phase's LLM-proposable actions as a sorted tuple (deterministic)."""
    key = (phase or "").strip().upper()
    actions = PHASE_ALLOWED_ACTIONS.get(key, frozenset())
    return tuple(sorted(actions - COORDINATOR_INTERNAL_ACTIONS - PHASE_COORDINATOR_RESERVED.get(key, frozenset())))


def render_phase_action_bullets(
    *,
    disabled_suffix: dict[str, str] | None = None,
) -> list[str]:
    """Render per-phase action bullets for the prompt; admission refuses anything outside them."""
    suffix = disabled_suffix or {}
    out: list[str] = []
    for phase in PHASE_NAMES:
        actions = allowed_actions_for(phase)
        flag = suffix.get(phase)
        if flag:
            out.append(f"- **{phase}**: {', '.join(actions)} (DISABLED: {flag} — phase skipped)")
        else:
            out.append(f"- **{phase}**: {', '.join(actions)}")
    return out


# phase_history cap (record_phase_transition, append_phase_history_event).
_PHASE_HISTORY_CAP = 100


@dataclass(frozen=True)
class HistoryEntry:
    """The fields of one phase_history row before they are normalized."""

    from_phase: str
    to_phase: str
    reason: str
    evidence: dict[str, Any] | None
    ts: str
    ts_unix: float
    cycle: int = 0


def make_history_row(entry: HistoryEntry) -> dict[str, Any]:
    """Construct a canonical phase_history row; ``reason`` unvalidated for resume tools."""
    return {
        "from_phase": (entry.from_phase or "").strip().upper(),
        "to_phase": (entry.to_phase or "").strip().upper(),
        "reason": (entry.reason or "").strip(),
        "evidence": dict(entry.evidence or {}),
        "ts": entry.ts,
        "ts_unix": float(entry.ts_unix or 0.0),
        "cycle": int(entry.cycle or 0),
    }


# Lifecycle events — operator-facing phase/step boundary log.
# Phase-boundary marker: a point-in-time "entered <phase>" mark with no matching END (unlike START, which pairs with a
# later END for the same step).
LIFECYCLE_STATUS_ENTER = "ENTER"

# Human-friendly labels for the coordinator phases.
PHASE_HUMAN_LABELS: dict[str, str] = {
    PHASE_PRELUDE: "Prelude (baseline + roofline)",
    PHASE_ENABLEMENT: "Enablement (make the combo runnable)",
    PHASE_FRAMEWORK_AGENT: "Optimize (config / source / upstream)",
    PHASE_KERNEL_AGENT: "Kernel optimization",
    PHASE_SWEEP: "Concurrency sweep",
    PHASE_CLOSE: "Close (report)",
}

# Human-friendly labels for the lifecycle steps surfaced to operators.
LIFECYCLE_STEP_LABELS: dict[str, str] = {
    "roofline": "TraceLens",
    "trace_analyze": "TraceLens",
    "run_gemm_tuning": "GEMM tuning",
    "run_optimization": "GEAK",
    "integrate": "Integrate",
    "apply_patch": "Integrate",
    "explore": "Validate (bench on the stack)",
    "sweep": "Concurrency sweep",
    "report": "Report",
    "session_breakdown": "Report (session breakdown)",
}


def lifecycle_label(name: str) -> str:
    """Resolve a human-friendly label for a step or phase name."""
    key = (name or "").strip()
    if key in LIFECYCLE_STEP_LABELS:
        return LIFECYCLE_STEP_LABELS[key]
    upper = key.upper()
    if upper in PHASE_HUMAN_LABELS:
        return PHASE_HUMAN_LABELS[upper]
    return key


@dataclass(frozen=True)
class LifecycleMark:
    """The fields of one lifecycle boundary before they are normalized into an event row."""

    step: str
    status: str
    phase: str | None = None
    label: str | None = None
    artifacts: dict[str, str] | None = None
    detail: str = ""
    duration_s: float | None = None


def make_lifecycle_event(mark: LifecycleMark, *, seq: int, ts: str) -> dict[str, Any]:
    """Construct a canonical lifecycle event row."""
    event: dict[str, Any] = {
        "seq": int(seq),
        "ts": ts,
        "phase": (mark.phase or "").strip().upper(),
        "step": (mark.step or "").strip(),
        "label": (mark.label or lifecycle_label(mark.step)),
        "status": (mark.status or "").strip().upper(),
        "detail": (mark.detail or "").strip(),
        "artifacts": {str(k): str(v) for k, v in (mark.artifacts or {}).items() if v not in (None, "")},
    }
    if mark.duration_s is not None:
        try:
            event["duration_s"] = round(float(mark.duration_s), 3)
        except (TypeError, ValueError):
            # A malformed duration_s is omitted rather than failing creation.
            pass
    return event
