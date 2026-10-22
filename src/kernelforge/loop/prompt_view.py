# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compact, state-driven prompt view for the long-horizon forge-loop."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from kernelforge.loop.run_state import MAX_PINNED_ITERATIONS, LoopStateStore, RunState

if TYPE_CHECKING:
    from kernelforge.loop.handoffs import HandoffStore
    from kernelforge.loop.runner import IterationResult

# Canonical on-disk locations, shown to the agent so it can Read detail on demand.
_ARCHIVE_REL = "forge_experiments/candidates"
_HANDOFFS_REL = "forge_experiments/handoffs"
_STATE_REL = "forge_experiments/run_state.json"
_EVENTS_REL = "forge_experiments/events.jsonl"

# How many pins the retrieval map names.
_MAX_RENDERED_PINS = 6


def _fmt_ms(value: float | None) -> str:
    try:
        return f"{float(value):.4f} ms"
    except (TypeError, ValueError):
        return "?"


def _render_pins(state: RunState, result_events: list[dict]) -> list[str]:
    """Render the pinned iterations the map points at, best lineage first."""
    pinned = list(state.pinned_iterations)
    if not pinned:
        return []

    best = state.best.iteration
    holds_best = best in pinned
    head = [best] if holds_best else []
    others = [iteration for iteration in pinned if iteration != best]
    # Guarded rather than sliced directly: ``others[-0:]`` is the whole list, so a budget of zero would render every
    # pin instead of none.
    recent_budget = max(0, _MAX_RENDERED_PINS - len(head))
    selected = head + (others[-recent_budget:] if recent_budget else [])

    speedups = {
        int(event["iter"]): event["mean_case_speedup"]
        for event in result_events
        if event.get("mean_case_speedup") is not None
    }
    if holds_best and state.best.mean_case_speedup is not None:
        # The best record carries the authoritative post-decision score, which outlives the bounded event window the
        # other pins are scored from.
        speedups[best] = state.best.mean_case_speedup

    rendered: list[str] = []
    for iteration in selected:
        parts = [str(iteration)]
        if holds_best and iteration == best:
            parts.append("best")
        speedup = speedups.get(iteration)
        if speedup is not None:
            parts.append(f"{float(speedup):.6f}x")
        rendered.append(" ".join(parts))
    return rendered


def _recent_line(event: dict) -> str:
    """One compact fact line for a recent iteration-result event."""
    it = event.get("iter", "?")
    decision = str(event.get("decision") or "").strip()
    plan = str(event.get("plan") or "").replace("\n", " ").strip()[:60]
    parts = [f"iter {it} {decision}".rstrip()]
    if plan:
        parts.append(plan)
    mean_case_speedup = event.get("mean_case_speedup")
    if mean_case_speedup is not None:
        parts.append(f"mean case speedup={float(mean_case_speedup):.6f}x")
    wall = event.get("wall_ms")
    if wall is not None:
        parts.append(f"wall={_fmt_ms(wall)}")
    err = str(event.get("error_sig") or "").replace("\n", " ").strip()[:80]
    if err:
        parts.append(f"error: {err}")
    return " | ".join(parts)


# How many recent attempt lines the header renders.
MAX_RECENT_ATTEMPT_LINES = 6

# How many recent iteration outcomes the long-horizon prompt header is built from.
LONG_HORIZON_OUTCOME_WINDOW = max(MAX_PINNED_ITERATIONS, MAX_RECENT_ATTEMPT_LINES)


def render_long_horizon_header(
    state: RunState,
    recent_events: list[dict],
    *,
    max_recent: int = MAX_RECENT_ATTEMPT_LINES,
    max_chars: int = 4000,
    include_handoffs: bool = False,
) -> str:
    """Render the compact long-horizon memory header, or \"\" when state is empty."""
    # Render nothing until there is substantive history, so the loop's cold-start prompt (iteration 1, before any
    # result) is unchanged.
    has_history = (
        state.best.iteration > 0
        or bool(state.best.commit_hash)
        or state.stall.unresolved_stall_iters > 0
        or any(e.get("type") == "iteration_result" for e in recent_events)
    )
    if not has_history:
        return ""

    # Control-state lead: the compact overview the implementer acts on.
    lead: list[str] = [
        "## Long-Horizon Memory (state-driven; full detail on disk)",
        f"Phase: {state.phase}",
    ]

    # Only claim a "best" once a real KEEP exists; before that, surface the baseline so the agent still knows the bar
    # to beat.
    if state.best.iteration > 0 or state.best.commit_hash:
        label = "validated KB warm-start" if state.best.source == "warm_start" else f"iter {state.best.iteration}"
        speedup_text = f"{state.best.mean_case_speedup:.6f}x" if state.best.mean_case_speedup is not None else "?"
        best_line = f"Current best: {label}, mean case speedup {speedup_text}, raw mean {_fmt_ms(state.best.wall_ms)}"
        if state.baseline_wall_ms is not None:
            best_line += f" (baseline {_fmt_ms(state.baseline_wall_ms)})"
        if state.best.plan:
            best_line += f' — plan: "{state.best.plan}"'
        lead.append(best_line)
    elif state.baseline_wall_ms is not None:
        lead.append(
            "Baseline: mean case speedup 1.000000x, raw mean "
            f"{_fmt_ms(state.baseline_wall_ms)} (no kept improvement yet)"
        )

    if state.stall.unresolved_stall_iters > 0:
        lead.append(f"Stall: {state.stall.unresolved_stall_iters} iteration(s) without improvement")

    # Iteration outcomes feed both the pin hint below and the recent attempts.
    result_events = [e for e in recent_events if e.get("type") == "iteration_result"]

    # Retrieval map — always kept, so the agent always knows where the full detail lives even if the recent list is
    # trimmed for budget.
    pins = _render_pins(state, result_events)
    pin_hint = f" (pinned: {', '.join(pins)})" if pins else ""
    retrieval: list[str] = [
        "Full detail lives on disk. Read on demand instead of guessing:",
        f"- {_STATE_REL} — current control state",
        f"- {_EVENTS_REL} — append-only event history",
        f"- {_ARCHIVE_REL}/index.jsonl — one summary row per attempt",
        f"- {_ARCHIVE_REL}/iter_NNN/"
        + "{kernel.py,change.diff,validation.txt,meta.json}"
        + f" — full attempt detail{pin_hint}",
        "- analysis/<commit>/ — evidence",
    ]
    if include_handoffs:
        retrieval.append(f"- {_HANDOFFS_REL}/iter_NNN.json — structured iteration handoffs")

    # Recent factual attempts — summaries only, never full logs.
    recent_lines = [f"- {_recent_line(e)}" for e in result_events[-max_recent:]]

    def _assemble(recent: list[str]) -> str:
        parts = list(lead)
        if recent:
            parts.append("")
            parts.append("Recent attempts (facts; read files for detail):")
            parts.extend(recent)
        parts.append("")
        parts.extend(retrieval)
        return "\n".join(parts).strip()

    # Enforce the ceiling by dropping the OLDEST recent line first; the lead + retrieval map are always kept (they are
    # what make the loop resumable).
    header = _assemble(recent_lines)
    while recent_lines and len(header) > max_chars:
        recent_lines.pop(0)
        header = _assemble(recent_lines)

    # The fixed control state + retrieval map is the minimum useful view.
    return header


def long_horizon_header(
    state: RunState,
    store: LoopStateStore,
    handoff_store: HandoffStore | None = None,
    *,
    scope: Collection[int] | None = None,
) -> str:
    """The compact long-horizon header for the Implementer prompt, from the ``scope`` iterations when given, or \"\"."""
    outcomes = [
        event
        for event in store.recent_results(LONG_HORIZON_OUTCOME_WINDOW)
        if scope is None or event.get("iter") in scope
    ]
    return render_long_horizon_header(
        state,
        outcomes,
        include_handoffs=bool(handoff_store and handoff_store.latest()),
    )


def compact_history_entry(r: IterationResult) -> str:
    """One-line history entry for the agent prompt — keeps tokens bounded."""
    rat = (r.agent_rationale or "").replace("\n", " ").strip()[:80]
    if not r.validation_passed:
        last = ""
        if r.validation_summary:
            lines = [ln for ln in r.validation_summary.splitlines() if ln.strip()]
            if lines:
                last = lines[-1][:60]
        return f"iter {r.iteration} REVERT(validation) last='{last}' rat='{rat}'"
    parts = [f"iter {r.iteration}", "KEEP" if r.kept else "REVERT(perf)"]
    if r.mean_case_speedup is not None:
        parts.append(f"mean_case_speedup={r.mean_case_speedup:.4f}x")
    if r.wall_ms is not None:
        parts.append(f"wall={r.wall_ms:.3f}ms")
    if r.snr_db is not None:
        parts.append(f"snr={r.snr_db:.1f}dB")
    if r.vgpr:
        parts.append(f"vgpr={r.vgpr}")
    if r.pmc_diagnosis:
        parts.append(f"pmc='{r.pmc_diagnosis[:40]}'")
    if rat:
        parts.append(f"rat='{rat}'")
    return " ".join(parts)
